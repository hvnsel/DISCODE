"""
discover_policy.py
==================

The **term-structured policy** ("terms in a bag") — one transformer that writes
every DOF's expression as a bag of short, independently decoded TERMS, and lets
each DOF condition on the expressions already committed for the other DOFs.

Why terms
---------
Every ground truth this code searches for is a SUM of forces, and a sum has no
order.  A flat sequence policy has to pick one anyway, and the root ``add`` it
must write first costs it a whole nesting level.  Here the root ``add`` is
never a token the model emits: each term is generated on its own, with its own
depth and length budget (``max_terms`` slots of ``max_term_len`` tokens), and
the terms are assembled under one ``add`` afterwards
(:func:`discover_core.assemble_terms`).  That gives each term the full nesting
budget back, and it is what lets the ordering of the terms carry no
information.

Layout
------
``x`` is ``(B, N, K, Lt)`` — batch, DOF *slot*, term index, intra-term
position.  The ``N`` slots hold the DOFs in a slice order ``pi`` (a
permutation, varied across the sampled batch); the DOF occupying slot ``s`` is
``pi[s]``.  Each term gets ``Pt = Lt + 1`` input positions: a learned BOS, the
``Lt`` shifted tokens, and a trailing readout position that carries the term's
last token to later terms and slices.  Flattened slot-major, then term, then
position::

    flat(s, j, p) = (s*K + j) * Pt + p

Attention from query ``(s, j, p)`` to key ``(s', j', p')`` is allowed iff::

    s' <  s                          (earlier DOF; gated by cross_slice_attention)
    s' == s and j' <  j              (an earlier term in my bag — ALL its positions)
    s' == s and j' == j and p' <= p  (causal within my own term)

so slot ``s`` sees exactly the slots before it under ``pi`` — a finite, known
set that is stored with the buffer entry as its **context**, which is what
keeps entries reproducible across epochs.

Two invariants hold this together, and both are load-bearing:

* **Permutation happens at slice granularity only, never inside a term.**
  Every term stays a left-to-right prefix, so ``_parse_stack`` /
  ``get_valid_tokens`` / ``is_complete`` — all of which require a prefix —
  work untouched.

* **The teacher-forcing shift is per term, not over the flattened sequence.**
  Each term starts from its own BOS.  Shifting the flattened sequence instead
  would feed one term's last token into the next term's first input, so
  information would cross term and slice boundaries through the *input*
  rather than through attention, and ``cross_slice_attention=False`` would not
  actually isolate the slices.  The readout position exists for the same
  reason: without it a term's final token would never be fed in anywhere and
  would be invisible to every later term and slice.

Knobs
-----
* ``term_position_encoding=True`` — the model knows which term came first (a
  term-index embedding band).  ``False`` drops that one band and nothing else,
  so terms ``1..j-1`` are indistinguishable to term ``j`` and the bag becomes a
  genuine bag.  Mask, sampler and update are byte-identical between the two.
* ``cross_slice_attention=False`` confines attention to the token's own slice,
  which turns the model into N independent term-bag policies that share
  weights.  An internal coupling force appears in two equations at once with
  opposite sign; with cross-slice attention the second DOF can copy the
  subtree the first one found (the sign costs nothing — VARPRO fits the
  leading coefficient), and switching it off is the only way to tell whether
  that copying is doing anything.

STOP
----
A STOP action closes the bag.  It is an extra output logit (index
``N_TOKENS``), legal only at intra-term position 0 and never at term 0.  It is
deliberately NOT added to ``dc.ALL_TOKENS``: that table drives ``tok2idx``,
``ARITY``, ``MASK_TOKEN = N_TOKENS`` and the critic's vocabulary.  A STOPped
term writes nothing into ``x``.

Buffer entries store the flat ASSEMBLED tau, so the critic, hall of fame,
dedup, ``denormalize_expr`` and the scoring worker never see terms.  The update
rebuilds its teacher-forcing input by :func:`discover_core.split_terms` on that
tau, which is deterministic — the sampled term order is preserved exactly, so
term ``j`` is replayed against precisely the prefix it saw when it was drawn.

Nothing here touches the reward, the constant fitting, or the grammar; this
module is the policy and its sampling / update only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import discover_core as dc


# Share of the sampled batch that gets an independent random slice order when
# ``slice_order='random'``.  The rest keeps the epoch's base order (see
# :func:`make_batch_orders` for why it is a mixture and not one or the other).
PERMUTE_FRACTION = 0.5

TERM_GRAMMARS = ('free', 'varpro')


# ── Positional encoding ─────────────────────────────────────────────────────
def sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    """Standard ``(max_len, d_model)`` sinusoid, for position WITHIN a term.

    Position is one of several additive conditioning bands (token, intra-term
    position, DOF identity, slot, term index), so it lives in ``d_model``.
    """
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


# ── Slice ordering ──────────────────────────────────────────────────────────
def choose_order(best_r, mode='reward', rng=None):
    """Pick the epoch's base slice order.

    ``'reward'`` puts the most-converged DOF first so the others condition on
    it — that is where the copying is supposed to pay.  Early on every
    ``best_r`` is ``-inf`` and there is nothing to order by, so it falls back
    to a random permutation.
    """
    rng = np.random.default_rng() if rng is None else rng
    n   = len(best_r)
    if n <= 1 or mode == 'fixed':
        return tuple(range(n))
    if mode == 'random' or not any(np.isfinite(r) for r in best_r):
        return tuple(int(d) for d in rng.permutation(n))
    return tuple(sorted(range(n), key=lambda d: -best_r[d]))


def make_batch_orders(base_order, batch_size, slice_order='random', rng=None):
    """``(B, N)`` slice orders for one sampled batch.

    ``slice_order='random'`` gives a *mixture*: ``PERMUTE_FRACTION`` of the rows
    draw an independent permutation and the rest keep ``base_order``.  Both
    halves are needed.  Without permutation the model only ever learns one
    direction — with a fixed order, DOF 0 never sees DOF 1 and can never copy
    from it.  Without the base rows, the reward-driven "most-converged DOF
    first" ordering never gets exercised at the batch size that matters.
    """
    rng   = np.random.default_rng() if rng is None else rng
    base  = np.asarray(base_order, dtype=np.int64)
    n_dof = len(base)
    orders = np.tile(base, (batch_size, 1))
    if n_dof <= 1 or slice_order != 'random':
        return orders
    n_perm = int(round(batch_size * PERMUTE_FRACTION))
    if n_perm > 0:
        rows = rng.choice(batch_size, size=min(n_perm, batch_size), replace=False)
        for r in rows:
            orders[r] = rng.permutation(n_dof)
    return orders


def make_context(order_row, slot, slice_taus):
    """Build the buffer context for the slice in ``slot``.

    ``prefix_slices`` holds **only** the slices that preceded this one under
    ``pi`` — under the attention mask those are exactly the tokens this slice
    could see.  Slices that came after are irrelevant and must not be stored,
    or the update would condition on tokens the sample never saw.

    ``slice_taus`` is indexed by *slot*, not by DOF.
    """
    return {
        'dof_order': tuple(int(d) for d in order_row),
        'slot': int(slot),
        'prefix_slices': {int(order_row[s]): list(slice_taus[s])
                          for s in range(slot)},
    }


# ── Term grammar ────────────────────────────────────────────────────────────
def term_valid_mask(term_prefix, position, max_term_len, term_grammar='free'):
    """Validity over ``dc.ALL_TOKENS`` for the next token of ONE term.

    Term mode differs from the flat grammar in exactly the ways
    :func:`discover_core.get_valid_tokens` documents: the term sits one level
    below an implicit root ``add`` (``depth_offset=1``), it may be a bare
    variable, it must not itself be ``add``-rooted, and it has a per-term
    length floor of 1.  ``'varpro'`` additionally restricts a power op's child
    to a bare variable — the constant-fitter's closed-form domain.
    """
    return torch.as_tensor(dc.get_valid_tokens(
        term_prefix, position, max_term_len,
        depth_offset=1, min_len=1, allow_leaf_at_zero=True,
        forbid_at_root=('add',),
        power_child_vars_only=(term_grammar == 'varpro')), dtype=torch.float32)


def stop_allowed(term_index, bag):
    """May the bag be closed before term ``term_index`` is generated?

    Never at term 0 (a bag needs a term).  A one-term bag whose only term is a
    bare leaf is also refused: it would assemble to ``['x1']``, which the flat
    grammar never produces (no leaf at the root).  Every other one-term bag is
    op-rooted and is a legal flat expression on its own.
    """
    if term_index == 0:
        return False
    if term_index == 1 and len(bag[0]) == 1:
        return False
    return True


# ── Attention mask ──────────────────────────────────────────────────────────
_TERM_MASK_CACHE = {}


def build_term_attn_mask(n_dof, n_terms, term_len_in, cross_slice_attention,
                         device=None, dtype=torch.float32):
    """``(N*K*Pt, N*K*Pt)`` additive float mask, ``-inf`` where attention is
    forbidden.

    ``term_len_in`` is ``Pt = max_term_len + 1`` input positions per term.  See
    the block predicate in the module docstring.  The diagonal is always
    allowed, so no row is fully masked and softmax never sees an all-``-inf``
    row.  The tensor is cached and returned by reference — never write into it.
    """
    key = (n_dof, n_terms, term_len_in, bool(cross_slice_attention),
           str(device), dtype)
    if key in _TERM_MASK_CACHE:
        return _TERM_MASK_CACHE[key]

    K, Pt = n_terms, term_len_in
    n   = n_dof * K * Pt
    idx = torch.arange(n)
    s = idx // (K * Pt)
    j = (idx // Pt) % K
    p = idx % Pt
    sq, sk = s.unsqueeze(1), s.unsqueeze(0)      # [query, key]
    jq, jk = j.unsqueeze(1), j.unsqueeze(0)
    pq, pk = p.unsqueeze(1), p.unsqueeze(0)

    same_slice = sq == sk
    allowed = (same_slice & (jk < jq)) | (same_slice & (jk == jq) & (pk <= pq))
    if cross_slice_attention:
        allowed = allowed | (sk < sq)

    mask = torch.zeros(n, n, dtype=dtype)
    mask.masked_fill_(~allowed, float('-inf'))
    if device is not None:
        mask = mask.to(device)
    _TERM_MASK_CACHE[key] = mask
    return mask


# ── The policy ──────────────────────────────────────────────────────────────
class TermBagPolicy(nn.Module):
    """Causal transformer over ``n_dof`` slices x ``max_terms`` terms x
    ``max_term_len`` tokens.  See the module docstring for the layout, the
    mask, and what the two knobs isolate.

    Conditioning is five additive bands: token, intra-TERM position, DOF
    identity, slot, and — only when ``term_position_encoding`` — term index.
    ``dof_emb`` says which DOF a slice *is* — necessary because its physical
    position moves with ``pi`` — and ``slot_emb`` says where in ``pi`` it
    sits, which is how the model tells "I am first, there is no context" from
    "I am third, two expressions precede me".  ``out_proj`` has
    ``n_tokens + 1`` outputs; the last is STOP.
    """

    def __init__(self, n_tokens=None, max_terms=8, max_term_len=8, n_dof=2,
                 d_model=128, n_heads=4, ff_dim=512, n_layers=4,
                 cross_slice_attention=True, term_position_encoding=True):
        super().__init__()
        if int(max_terms) < 2:
            # With one term slot the bag is always "full" at m=1 and could be a
            # bare leaf — assembling to ['x1'], outside the flat grammar.
            raise ValueError("max_terms must be >= 2")
        self.n_tokens     = dc.N_TOKENS if n_tokens is None else int(n_tokens)
        self.vocab_size   = self.n_tokens + 1          # input side: + MASK_TOKEN
        self.stop_idx     = self.n_tokens              # output side: STOP logit
        self.max_terms    = int(max_terms)
        self.max_term_len = int(max_term_len)
        self.n_dof        = int(n_dof)
        self.d_model      = int(d_model)
        self.cross_slice_attention  = bool(cross_slice_attention)
        self.term_position_encoding = bool(term_position_encoding)

        self.tok_emb  = nn.Embedding(self.vocab_size, d_model)
        self.dof_emb  = nn.Embedding(self.n_dof, d_model)
        self.slot_emb = nn.Embedding(self.n_dof, d_model)
        # term_position_encoding=False removes this band and nothing else.
        self.term_emb = (nn.Embedding(self.max_terms, d_model)
                         if self.term_position_encoding else None)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.term_len_in = self.max_term_len + 1       # BOS + Lt shifted + readout
        self.register_buffer('pos_pe', sinusoidal(self.term_len_in, d_model),
                             persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.out_proj = nn.Linear(d_model, self.n_tokens + 1)

    def forward(self, x, dof_order, n_slots=None):
        """
        x         : (B, N, K, Lt) long — token indices; ``MASK_TOKEN`` where
                    nothing has been generated (incl. unused term slots).
        dof_order : (B, N) long — ``dof_order[b, s]`` is the DOF in slot ``s``.
        n_slots   : compute only the first ``n_slots`` slots (exact under
                    causality; used by the sampler, where it roughly halves
                    the average sequence length).

        returns   : (B, S, K, Lt, n_tokens + 1) logits.  ``[b, s, j, p]``
                    predicts the token at ``[b, s, j, p]``; index ``n_tokens``
                    is STOP.
        """
        B, N, K, Lt = x.shape
        if K != self.max_terms or Lt != self.max_term_len or N > self.n_dof:
            raise ValueError(f"x is (B, {N}, {K}, {Lt}); this policy was built "
                             f"for n_dof={self.n_dof}, max_terms="
                             f"{self.max_terms}, max_term_len={self.max_term_len}")
        S = N if n_slots is None else int(n_slots)
        x = x[:, :S]
        dof_order = dof_order[:, :S]
        Pt = Lt + 1

        bos = self.bos.view(1, 1, 1, 1, -1).expand(B, S, K, 1, self.d_model)
        e   = torch.cat([bos, self.tok_emb(x)], dim=3)             # (B,S,K,Pt,D)

        slots = torch.arange(S, device=x.device)
        h = (e
             + self.pos_pe[:Pt].view(1, 1, 1, Pt, -1)
             + self.dof_emb(dof_order).view(B, S, 1, 1, -1)
             + self.slot_emb(slots).view(1, S, 1, 1, -1))
        if self.term_emb is not None:
            terms = torch.arange(K, device=x.device)
            h = h + self.term_emb(terms).view(1, 1, K, 1, -1)

        mask = build_term_attn_mask(S, K, Pt, self.cross_slice_attention,
                                    device=h.device, dtype=h.dtype)
        out = self.encoder(h.reshape(B, S * K * Pt, self.d_model), mask=mask)
        out = out.view(B, S, K, Pt, self.d_model)[:, :, :, :Lt, :]  # drop readout
        return self.out_proj(out)


# ── Sampling ────────────────────────────────────────────────────────────────
def sample_term_batch(policy, batch_size, n_dof, dof_order, term_grammar='free',
                      fixed_slices=None, device=None):
    """Sample ``batch_size`` joint sequences, one term at a time.

    ``dof_order``    : ``(n_dof,)`` — one order for the whole batch — or
                       ``(B, n_dof)`` for a per-row order.
    ``fixed_slices`` : optional ``{dof: tau}`` pinned (split into terms and
                       written into ``x``) instead of sampled.

    returns a list of ``B`` dicts ``{dof: tau}``, where each ``tau`` is the
    ASSEMBLED sum of the terms drawn for that slice, in the order they were
    drawn.

    Per term, generation is left to right under :func:`term_valid_mask` with
    the term's own length budget, and a term is closed by ``is_complete``.
    Before each new term the model may choose STOP (see :func:`stop_allowed`),
    which closes the bag.  A bag that fills all ``max_terms`` slots made no
    STOP decision.

    Two details that are easy to get wrong:

    * Validity is evaluated on the **intra-term** prefix at the **intra-term**
      position.  ``get_valid_tokens`` computes ``remaining = max_len -
      position``, so passing a flattened index would silently corrupt the
      length budget.

    * Everything not generated — the tail of a term after ``is_complete``
      fires, every unused term slot, every slice not yet reached — stays
      ``MASK_TOKEN``.  :func:`build_term_inputs` reproduces exactly this
      padding during the update; a mismatch there is invisible and degrades
      the policy silently.
    """
    if term_grammar not in TERM_GRAMMARS:
        raise ValueError(f"term_grammar must be one of {TERM_GRAMMARS}, "
                         f"got {term_grammar!r}")
    policy.eval()
    device = next(policy.parameters()).device if device is None else device
    K, Lt, STOP = policy.max_terms, policy.max_term_len, policy.stop_idx

    order = np.asarray(dof_order, dtype=np.int64)
    if order.ndim == 1:
        order = np.tile(order, (batch_size, 1))
    if order.shape != (batch_size, n_dof):
        raise ValueError(f"dof_order must be (n_dof,) or (B, n_dof); "
                         f"got {order.shape} for B={batch_size}, N={n_dof}")
    order_t = torch.as_tensor(order, dtype=torch.long, device=device)

    x    = torch.full((batch_size, n_dof, K, Lt), dc.MASK_TOKEN,
                      dtype=torch.long, device=device)
    bags = [[[] for _ in range(n_dof)] for _ in range(batch_size)]   # [b][slot] -> [term, ...]
    closed = [[False] * n_dof for _ in range(batch_size)]

    fixed = dict(fixed_slices or {})
    for b in range(batch_size):
        for s in range(n_dof):
            d = int(order[b, s])
            if d not in fixed:
                continue
            terms = dc.split_terms(fixed[d])[:K]
            bags[b][s] = [list(t)[:Lt] for t in terms]
            for j, t in enumerate(bags[b][s]):
                for p, tk in enumerate(t):
                    x[b, s, j, p] = dc.tok2idx(tk)
            closed[b][s] = True

    for slot in range(n_dof):
        for j in range(K):
            if all(closed[b][slot] for b in range(batch_size)):
                break
            cur   = [[] for _ in range(batch_size)]          # this term's prefix
            tdone = [closed[b][slot] for b in range(batch_size)]
            for p in range(Lt):
                if all(tdone):
                    break
                with torch.no_grad():
                    logits = policy(x, order_t, n_slots=slot + 1)[:, slot, j, p, :]
                prob = torch.softmax(logits.float(), dim=-1).cpu()
                for b in range(batch_size):
                    if tdone[b]:
                        continue
                    r = term_valid_mask(cur[b], p, Lt, term_grammar)
                    can_stop = p == 0 and stop_allowed(j, bags[b][slot])
                    r_full = torch.cat([r, torch.tensor([1.0 if can_stop else 0.0])])
                    p_b = prob[b] * r_full
                    if p_b.sum() == 0:
                        p_b = r_full.clone()     # fall back to the uniform-valid draw
                    if p_b.sum() == 0:
                        tdone[b] = True; closed[b][slot] = True   # genuine dead end
                        continue
                    a = int(Categorical(p_b / p_b.sum()).sample().item())
                    if a == STOP:
                        tdone[b] = True; closed[b][slot] = True
                        continue
                    tok = dc.idx2tok(a)
                    cur[b].append(tok)
                    x[b, slot, j, p] = a
                    if dc.is_complete(cur[b]):
                        tdone[b] = True
                        bags[b][slot].append(cur[b])
            # A term that ran out of budget without completing cannot enter the
            # bag (the assembled tau would be malformed).  Erase its tokens so
            # ``x`` matches what ``build_term_inputs`` will rebuild, and close.
            for b in range(batch_size):
                if not closed[b][slot] and cur[b] and not dc.is_complete(cur[b]):
                    x[b, slot, j, :] = dc.MASK_TOKEN
                    closed[b][slot] = True

    return [{int(order[b, s]): dc.assemble_terms(bags[b][s]) for s in range(n_dof)}
            for b in range(batch_size)]


# ── Teacher-forcing inputs ──────────────────────────────────────────────────
def build_term_inputs(entries, n_dof, max_terms, max_term_len, device):
    """Teacher-forcing inputs for buffer entries ``(r, tau, consts, context)``.

    Each stored flat tau — the entry's own, and the context's
    ``prefix_slices`` — is split back into terms with
    :func:`discover_core.split_terms` and written term-by-term into its slot;
    everything else — the tail of every term, every unused term slot, every
    not-yet-generated slice — stays ``MASK_TOKEN``, which is exactly the
    padding :func:`sample_term_batch` leaves behind.

    Entries in one call may carry different ``dof_order``s; the model takes the
    order per batch element, so no grouping is needed.

    returns ``(x, dof_order, slots)`` with ``x`` of shape ``(B, N, K, Lt)``.
    """
    B = len(entries)
    K, Lt = int(max_terms), int(max_term_len)
    x     = torch.full((B, n_dof, K, Lt), dc.MASK_TOKEN, dtype=torch.long)
    order = torch.zeros(B, n_dof, dtype=torch.long)
    slots = []
    for bi, entry in enumerate(entries):
        tau, ctx = entry[1], entry[3]
        dord, slot = ctx['dof_order'], int(ctx['slot'])
        order[bi] = torch.as_tensor(dord, dtype=torch.long)
        slots.append(slot)
        for s in range(n_dof):
            if s == slot:
                t = tau
            elif s < slot:
                t = ctx['prefix_slices'].get(int(dord[s]), [])
            else:
                continue                     # not generated yet: stays MASK
            for j, term in enumerate(dc.split_terms(t)[:K]):
                for p, tk in enumerate(term[:Lt]):
                    x[bi, s, j, p] = dc.tok2idx(tk)
    return x.to(device), order.to(device), slots


# ── Beam search over one term ───────────────────────────────────────────────
def beam_search_term(policy, x, dof_order, slot, term_index, beam_width,
                     term_grammar='free', n_return=10):
    """Beam search **within one term** at ``(slot, term_index)``.

    ``x`` is ``(1, N, K, Lt)`` with every earlier slice and every earlier term
    of this slice committed; positions of the target term are ``MASK_TOKEN``.
    A joint beam over every position is not worth its branching factor, so
    everything before the target term is simply taken as given.  The
    distribution is recomputed after every committed token — under
    autoregression there is no static marginal matrix to read rows out of.
    STOP is excluded — the point of the beam is to propose a *term*.
    """
    policy.eval()
    device = next(policy.parameters()).device
    Lt, nt = policy.max_term_len, policy.n_tokens
    order  = np.asarray(dof_order, dtype=np.int64).reshape(-1)
    order_t = torch.as_tensor(order, dtype=torch.long, device=device).unsqueeze(0)

    beams = [(0.0, [])]
    for pos in range(Lt):
        live = [(lp, t) for lp, t in beams if not dc.is_complete(t)]
        fin  = [(lp, t) for lp, t in beams if dc.is_complete(t)]
        if not live:
            break

        xb = x.repeat(len(live), 1, 1, 1)
        for i, (_lp, t) in enumerate(live):
            for k, tk in enumerate(t[:Lt]):
                xb[i, slot, term_index, k] = dc.tok2idx(tk)
        with torch.no_grad():
            logits = policy(xb, order_t.expand(len(live), -1),
                            n_slots=slot + 1)[:, slot, term_index, pos, :]
        p = torch.softmax(logits.float(), dim=-1).cpu()[:, :nt]   # drop STOP

        new_beams = list(fin)
        for i, (lp, t) in enumerate(live):
            mask  = term_valid_mask(t, pos, Lt, term_grammar)
            p_pos = p[i] * mask
            if p_pos.sum() == 0:
                continue
            p_pos = p_pos / p_pos.sum()
            log_p = torch.log(p_pos + 1e-10)
            k     = min(beam_width, int(mask.sum().item()))
            topk_vals, topk_idx = log_p.topk(k)
            for v, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                if mask[idx] == 0:
                    continue
                new_beams.append((lp + v, t + [dc.idx2tok(idx)]))

        complete   = [(lp, t) for lp, t in new_beams if dc.is_complete(t)]
        incomplete = [(lp, t) for lp, t in new_beams if not dc.is_complete(t)]
        incomplete.sort(key=lambda z: z[0], reverse=True)
        beams = complete + incomplete[:beam_width]
        if len(complete) >= n_return:
            break

    return [t for _lp, t in beams if dc.is_complete(t)][:n_return]


def beam_term_candidates(policy, dof_order, n_dof, beam_width, n_return=10,
                         term_grammar='free', device=None):
    """One beam-searched term per DOF, **appended to a sampled bag**.

    A joint prefix is sampled exactly as in :func:`sample_term_batch`, so the
    contexts these candidates carry are built the same way as the sampled
    ones; then for each slot the beam proposes one more term on top of that
    slot's own sampled bag (or, if the bag is already full, re-proposes its
    last term).  That is the incremental-refinement move this policy is built
    to make cheap: "the bag is good, add one more term".

    Returns ``[(dof, assembled_tau, context), ...]``.
    """
    device = next(policy.parameters()).device if device is None else device
    K, Lt  = policy.max_terms, policy.max_term_len
    order  = np.asarray(dof_order, dtype=np.int64).reshape(-1)

    prefix = sample_term_batch(policy, 1, n_dof, order, term_grammar,
                               device=device)[0]
    out = []
    for slot in range(n_dof):
        d = int(order[slot])
        x = torch.full((1, n_dof, K, Lt), dc.MASK_TOKEN,
                       dtype=torch.long, device=device)
        slice_taus = []
        for s in range(n_dof):
            tau_s = prefix[int(order[s])] if s < slot else []
            slice_taus.append(tau_s)
            for j, term in enumerate(dc.split_terms(tau_s)[:K]):
                for p, tk in enumerate(term[:Lt]):
                    x[0, s, j, p] = dc.tok2idx(tk)
        bag = [list(t)[:Lt] for t in dc.split_terms(prefix[d])[:K]]
        if len(bag) < K:
            j, base = len(bag), bag
        else:
            j, base = K - 1, bag[:-1]
        for jj, term in enumerate(base):
            for p, tk in enumerate(term):
                x[0, slot, jj, p] = dc.tok2idx(tk)
        ctx = make_context(order, slot, slice_taus)
        for term in beam_search_term(policy, x, order, slot, j, beam_width,
                                     term_grammar, n_return=n_return):
            if not base and len(term) == 1:
                continue                 # a lone bare leaf is not a valid bag
            out.append((d, dc.assemble_terms(base + [term]), dict(ctx)))
    return out


# ── J-GRPO for the term layout ──────────────────────────────────────────────
def jgrpo_terms(policy, policy_old, policy_ref, S_alpha, R_alpha, dof, n_dof,
                eps, beta, critic=None, critic_max_len=None):
    """J-GRPO objective for one DOF against the shared term-structured policy.

    The input is the teacher-forced joint tensor from :func:`build_term_inputs`
    — the attention mask does the hiding — and the positions scored are

    * every token of every term of the entry's own slot, at ``[slot, j, p]``;
    * plus the STOP decision at ``[slot, m, 0]`` when the bag has ``m <
      max_terms`` terms.  A full bag made no STOP decision and scores none.

    Per scored position the objective is the PPO-clipped ratio against
    ``policy_old`` times the group advantage, minus ``beta`` times the KL to
    ``policy_ref``; the entropy is returned alongside for the caller's bonus.
    Both are per-position means.  Validity masks are applied at sampling only.

    Advantages are group-relative **within this DOF's batch** and in
    log-residual units (:func:`discover_core.log_residual_score`), never
    pooled across DOFs: reward scales differ wildly between them, and pooling
    would let the DOF with the wider spread dominate the shared trunk.  Above
    ``dc.MAX_GRPO_BATCH`` entries the batch is subsampled, prioritised in the
    same log-residual units so near-top refinements are not crowded out by the
    compressed r-scale.

    Returns ``(objective, entropy, contributed)``.  ``contributed=False`` means
    this DOF has nothing to say this step — nothing above ``R_alpha``, a
    degenerate advantage spread, or no scorable position.  Under a *shared*
    policy that must zero out this DOF alone and leave the others' gradients
    intact, which is why the degenerate cases return a flag instead of
    aborting the step.
    """
    device = next(policy.parameters()).device
    zero   = torch.zeros((), device=device)
    K, Lt, STOP = policy.max_terms, policy.max_term_len, policy.stop_idx

    valid = [e for e in S_alpha if e[0] - R_alpha > 0]
    if not valid:
        return zero, zero, False

    if len(valid) > dc.MAX_GRPO_BATCH:
        scores  = dc.log_residual_score(np.array([e[0] for e in valid]))
        advs    = scores - float(dc.log_residual_score(R_alpha))
        probs   = advs - advs.min() + 1e-6
        probs  /= probs.sum()
        indices = np.random.choice(len(valid), dc.MAX_GRPO_BATCH,
                                   replace=False, p=probs)
        valid   = [valid[i] for i in indices]

    for e in valid:
        if int(e[3]['dof_order'][int(e[3]['slot'])]) != int(dof):
            raise ValueError(
                f"buffer entry context is for DOF "
                f"{e[3]['dof_order'][e[3]['slot']]}, not DOF {dof}")

    x, order, slots = build_term_inputs(valid, n_dof, K, Lt, device)

    logits_cur = policy(x, order)
    with torch.no_grad():
        logits_old = policy_old(x, order)
        logits_ref = policy_ref(x, order)

    batch_rewards = dc.log_residual_score(
        np.array([e[0] for e in valid], dtype=np.float64))
    adv_mean = float(batch_rewards.mean())
    adv_std  = float(batch_rewards.std())
    if adv_std < 1e-3:
        # Degenerate spread: (s - mean)/(std + eps) would just amplify noise
        # into +-1 advantages.  Skip this DOF, keep the rest of the step.
        return zero, zero, False
    group_adv = (batch_rewards - adv_mean) / (adv_std + 1e-6)

    # NOTE: critic baseline kept for reference but intentionally NOT used for
    # the advantage.  Left here so the critic pathway stays available.
    if critic is not None and critic_max_len is not None:
        with torch.no_grad():
            crit_in = []
            for e in valid:
                tokens = ([dc.tok2idx(tk) for tk in e[1]]
                          + [dc.MASK_TOKEN] * (critic_max_len - len(e[1])))
                crit_in.append(tokens[:critic_max_len])
            _critic_baselines = critic(
                torch.tensor(crit_in, dtype=torch.long, device=device)
            ).cpu().numpy()

    log_p_cur = F.log_softmax(logits_cur, dim=-1)
    total_obj = torch.zeros((), device=device)
    total_ent = torch.zeros((), device=device)
    n_scored  = 0                                   # token positions (incl. STOP)

    for bi, entry in enumerate(valid):
        tau  = entry[1]
        slot = slots[bi]
        A_i  = float(group_adv[bi])
        terms = dc.split_terms(tau)[:K]

        positions = [(j, p, dc.tok2idx(tok))
                     for j, term in enumerate(terms)
                     for p, tok in enumerate(term[:Lt])]
        if len(terms) < K:
            positions.append((len(terms), 0, STOP))

        for j, p, ti in positions:
            lp_cur_k = log_p_cur[bi, slot, j, p, ti]
            lp_old_k = F.log_softmax(logits_old[bi, slot, j, p], dim=-1)[ti].detach()
            h        = torch.exp(lp_cur_k - lp_old_k)
            # PPO clip: min(h*A, clip(h)*A).  Multiplying by A *after* the min
            # is only equivalent for A >= 0; for A < 0 it leaves the ratio
            # unclipped below 1-eps, giving unbounded down-weighting gradients
            # on tokens shared with good expressions.
            obj_k    = torch.min(h * A_i, torch.clamp(h, 1 - eps, 1 + eps) * A_i)
            p_ref    = torch.softmax(logits_ref[bi, slot, j, p], dim=-1).detach()
            kl_k     = F.kl_div(log_p_cur[bi, slot, j, p], p_ref,
                                reduction='sum', log_target=False)
            total_obj = total_obj + obj_k - beta * kl_k
            total_ent = total_ent + Categorical(
                logits=logits_cur[bi, slot, j, p]).entropy()
            n_scored += 1

    if n_scored == 0:
        return zero, zero, False
    return total_obj / n_scored, total_ent / n_scored, True
