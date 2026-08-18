"""
discode_policy.py
=================

The **autoregressive joint policy** — one transformer that writes every DOF's
token sequence as a single flattened sequence, so that a DOF's expression can
condition on the expressions already committed for the other DOFs.

It replaces two things at once:

1. *Position factorisation.*  :func:`discode_core.sample_batch` calls the
   diffusion model **once** on an all-``MASK_TOKEN`` input and draws every
   token from the resulting position-wise marginals.  That network can express
   "position 4 is often ``intpower``" but never "*given* position 3 is
   ``intpower``, position 4 should be ``x1``".  All structural coherence in a
   sampled expression comes from :func:`discode_core.get_valid_tokens` and from
   whatever VARPRO manages to fit.  Autoregression fixes this, and the fix
   applies to a single-DOF run too.

2. *Cross-DOF structure.*  An internal coupling force appears in two equations
   at once with opposite sign (Newton's third law).  Under N independent
   policies each DOF must discover the shared subtree separately; under a joint
   model the second DOF can copy it, and the sign costs nothing because VARPRO
   fits the leading coefficient — only the structure has to be copied.

Set ``cross_slice_attention=False`` to keep (1) and drop (2): that is N
independent AR models sharing one set of weights, and it is the only way to
attribute a result to one defect or the other.

Layout
------
The flattened sequence is ``n_dof`` *slices* of ``max_len`` tokens::

        slice order pi (a permutation, varied across the sampled batch)
    +------------------- flattened sequence, length N*L ----------------+
    |  slot 0: tau_pi(0)      |  slot 1: tau_pi(1)      |  ...          |
    +-------------------------------------------------------------------+
                          causal attention mask

A *slot* is a position in the flattened sequence; the DOF that occupies it is
``pi[slot]``.  Under the causal mask, slot ``s`` sees exactly the slots before
it under ``pi`` — a finite, known set that is stored with the buffer entry as
its **context**, which is what keeps entries reproducible across epochs.

Two invariants hold this together, and both are load-bearing:

* **Permutation happens at slice granularity only, never inside a slice.**
  Every slice stays a left-to-right prefix, so ``_parse_stack`` /
  ``get_valid_tokens`` / ``is_complete`` — all of which require a prefix —
  work untouched.

* **The teacher-forcing shift is per slice, not over the flattened sequence.**
  Each slice gets its own learned BOS at its intra-slice position 0.  Shifting
  the flattened sequence instead would feed slice ``s-1``'s last token into
  slice ``s``'s first input, so information would cross the slice boundary
  through the *input* rather than through attention, and
  ``cross_slice_attention=False`` would not actually isolate the slices.  Per
  slice, the two ablation configs differ by the attention mask and nothing
  else.  Each slice therefore occupies ``L + 1`` input positions: a BOS, the
  ``L`` shifted tokens, and a trailing readout position that carries the
  slice's final token so later slices can attend to it.  Its own logits are
  discarded.  Without it, a token at ``l = L - 1`` would never be fed in
  anywhere, and a full-length expression's last token would be invisible to
  every later slice — precisely the conditioning this model exists to provide.

Nothing here touches the reward, the constant fitting, or the grammar; this is
a change to the policy and to sampling only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import discode_core as dc


# Share of the sampled batch that gets an independent random slice order when
# ``slice_order='random'``.  The rest keeps the epoch's base order (see
# :func:`make_batch_orders` for why it is a mixture and not one or the other).
PERMUTE_FRACTION = 0.5


# ── Positional encoding ─────────────────────────────────────────────────────
def sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    """Standard ``(max_len, d_model)`` sinusoid, for position WITHIN a slice.

    The diffusion model built its sinusoid in *vocab* space and added it to the
    one-hot before the input projection, which for a 2-DOF grammar meant
    ``D = VOCAB_SIZE = 10`` and ``D // 4 = 2`` frequency pairs to encode both
    position and diffusion step.  There is no room for a third band there, and
    we need three (position, DOF identity, slot), so the encodings move into
    ``d_model``.
    """
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


# ── Attention masks ─────────────────────────────────────────────────────────
_MASK_CACHE = {}


def build_attn_mask(n_dof: int, slice_len: int, cross_slice_attention: bool,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """``(N*P, N*P)`` additive float mask, ``-inf`` where attention is forbidden.

    ``slice_len`` is the number of *input* positions per slice (``max_len + 1``
    for :class:`ARJointPolicy`, which appends a readout position), not the
    number of tokens.

    Always causal.  With ``cross_slice_attention=False``, attention is
    additionally confined to the token's own slice, which turns the model into
    N independent AR models that share weights — the V0+AR ablation — out of
    the same code path.

    The diagonal is always allowed, so no row is fully masked and softmax never
    sees an all-``-inf`` row.
    """
    key = (n_dof, slice_len, bool(cross_slice_attention), str(device), dtype)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    n = n_dof * slice_len
    idx = torch.arange(n)
    allowed = idx.unsqueeze(0) <= idx.unsqueeze(1)          # [q, k] causal
    if not cross_slice_attention:
        same_slice = (idx.unsqueeze(0) // slice_len) == (idx.unsqueeze(1) // slice_len)
        allowed = allowed & same_slice

    mask = torch.zeros(n, n, dtype=dtype)
    mask.masked_fill_(~allowed, float('-inf'))
    if device is not None:
        mask = mask.to(device)
    _MASK_CACHE[key] = mask
    return mask


# ── The policy ──────────────────────────────────────────────────────────────
class ARJointPolicy(nn.Module):
    """Causal transformer over ``n_dof`` slices of ``max_len`` tokens.

    Replaces :class:`discode_core.DiffusionModel`.  Three departures from it:

    * **One causal stack, not encoder-then-decoder.**  The old model ran
      ``TransformerEncoder(h)`` and then ``TransformerDecoder(h, enc_out)`` on
      the *same* input, which is a 4-layer self-attention stack with extra
      parameters and no cross-attention to anything external.  ``n_layers``
      defaults to 4 = the old ``n_enc + n_dec``, so this is comparable in size
      to *one* ``DiffusionModel``.  Note that V0 runs N of those and this runs
      one, so total parameter counts still differ by roughly N; the training
      loop prints both, and ``n_layers``/``d_model`` are exposed so a
      capacity-matched ablation can be configured explicitly.

    * **Conditioning lives in ``d_model``** (see :func:`sinusoidal`), as four
      additive terms: token, intra-slice position, DOF identity, slot.
      ``dof_emb`` says which DOF a slice *is* — necessary because its physical
      position moves with ``pi`` — and ``slot_emb`` says where in ``pi`` it
      sits, which is how the model tells "I am first, there is no context"
      from "I am third, two expressions precede me".

    * **No diffusion step ``t``.**  It is meaningless under autoregression, so
      the ``t``-keyed positional-encoding cache goes away with it.
    """

    def __init__(self, n_tokens=None, max_len=32, n_dof=2, d_model=128,
                 n_heads=4, ff_dim=512, n_layers=4, cross_slice_attention=True):
        super().__init__()
        self.n_tokens   = dc.N_TOKENS if n_tokens is None else int(n_tokens)
        self.vocab_size = self.n_tokens + 1          # + MASK_TOKEN
        self.max_len    = int(max_len)
        self.n_dof      = int(n_dof)
        self.d_model    = int(d_model)
        self.cross_slice_attention = bool(cross_slice_attention)

        self.tok_emb  = nn.Embedding(self.vocab_size, d_model)
        self.dof_emb  = nn.Embedding(self.n_dof, d_model)
        self.slot_emb = nn.Embedding(self.n_dof, d_model)
        # Per-slice BOS: the input at intra-slice position 0.  Zeros is fine —
        # pos/dof/slot still identify the position uniquely.
        self.bos = nn.Parameter(torch.zeros(d_model))
        # max_len + 1 input positions per slice: BOS, the shifted tokens, and
        # the trailing readout position (see the module docstring).
        self.slice_len = self.max_len + 1
        self.register_buffer('pos_pe', sinusoidal(self.slice_len, d_model),
                             persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.out_proj = nn.Linear(d_model, self.n_tokens)

    def forward(self, x, dof_order, n_slots=None):
        """
        x         : (B, N, L) long — token indices in **slot** order.
                    ``MASK_TOKEN`` for not-yet-generated positions and for the
                    padding after a slice completes.
        dof_order : (B, N) long — ``dof_order[b, s]`` is the DOF in slot ``s``.
        n_slots   : optional int — compute only the first ``n_slots`` slots.
                    Causality makes this exact (later slots cannot influence
                    earlier ones), and during sampling it roughly halves the
                    average sequence length.

        returns   : (B, S, L, n_tokens) logits, S = ``n_slots`` or N.  Logits at
                    ``[b, s, l]`` predict the token **at** ``[b, s, l]``.
        """
        B, N, L = x.shape
        if L > self.max_len or N > self.n_dof:
            raise ValueError(f"x is (B, {N}, {L}); this policy was built for "
                             f"n_dof={self.n_dof}, max_len={self.max_len}")
        S = N if n_slots is None else int(n_slots)
        x = x[:, :S]
        dof_order = dof_order[:, :S]
        P = L + 1                                # input positions per slice

        # Per-slice right shift: each slice starts from its own BOS, and its
        # last token lands on the trailing readout position.
        bos = self.bos.view(1, 1, 1, -1).expand(B, S, 1, self.d_model)
        e   = torch.cat([bos, self.tok_emb(x)], dim=2)          # (B,S,P,D)

        slots = torch.arange(S, device=x.device)
        h = (e
             + self.pos_pe[:P].view(1, 1, P, -1)
             + self.dof_emb(dof_order).unsqueeze(2)
             + self.slot_emb(slots).view(1, S, 1, -1))

        mask = build_attn_mask(S, P, self.cross_slice_attention,
                               device=h.device, dtype=h.dtype)
        out = self.encoder(h.reshape(B, S * P, self.d_model), mask=mask)
        out = out.view(B, S, P, self.d_model)[:, :, :L, :]      # drop readout
        return self.out_proj(out)


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
    ``pi`` — under causal masking those are exactly the tokens this slice could
    see.  Slices that came after are irrelevant and must not be stored, or the
    update would condition on tokens the sample never saw.

    ``slice_taus`` is indexed by *slot*, not by DOF.
    """
    return {
        'dof_order': tuple(int(d) for d in order_row),
        'slot': int(slot),
        'prefix_slices': {int(order_row[s]): list(slice_taus[s])
                          for s in range(slot)},
    }


# ── Sampling ────────────────────────────────────────────────────────────────
def sample_joint_batch(policy, batch_size, max_len, n_dof, dof_order,
                       fixed_slices=None, device=None):
    """Sample ``batch_size`` joint token sequences, strictly left to right.

    ``dof_order``    : ``(n_dof,)`` — one order for the whole batch — or
                       ``(B, n_dof)`` for a per-row order.
    ``fixed_slices`` : optional ``{dof: tau}`` pinned instead of sampled (used
                       to condition on an elite, and by the beam path).

    returns a list of ``B`` dicts ``{dof: tau}``.

    Two details that are easy to get wrong:

    * Validity is evaluated on the **intra-slice** prefix at the **intra-slice**
      position.  ``get_valid_tokens`` computes ``remaining = max_len -
      position``, so passing a flattened index would silently corrupt the
      length budget.

    * When ``is_complete`` fires at ``l = k``, positions ``k+1 .. L-1`` of that
      slice stay ``MASK_TOKEN``.  :func:`build_ar_inputs` reproduces exactly
      this padding during the update; a mismatch there is invisible and
      degrades the policy silently.
    """
    policy.eval()
    device = next(policy.parameters()).device if device is None else device

    order = np.asarray(dof_order, dtype=np.int64)
    if order.ndim == 1:
        order = np.tile(order, (batch_size, 1))
    if order.shape != (batch_size, n_dof):
        raise ValueError(f"dof_order must be (n_dof,) or (B, n_dof); "
                         f"got {order.shape} for B={batch_size}, N={n_dof}")
    order_t = torch.as_tensor(order, dtype=torch.long, device=device)

    x    = torch.full((batch_size, n_dof, max_len), dc.MASK_TOKEN,
                      dtype=torch.long, device=device)
    taus = [[[] for _ in range(n_dof)] for _ in range(batch_size)]  # [b][slot]
    done = [[False] * n_dof for _ in range(batch_size)]

    fixed = dict(fixed_slices or {})
    for b in range(batch_size):
        for s in range(n_dof):
            d = int(order[b, s])
            if d not in fixed:
                continue
            tau = list(fixed[d])[:max_len]
            taus[b][s] = tau
            for k, tk in enumerate(tau):
                x[b, s, k] = dc.tok2idx(tk)
            done[b][s] = True

    for slot in range(n_dof):
        for l in range(max_len):
            if all(done[b][slot] for b in range(batch_size)):
                break
            with torch.no_grad():
                logits = policy(x, order_t, n_slots=slot + 1)[:, slot, l, :]
            p = torch.softmax(logits.float(), dim=-1).cpu()
            for b in range(batch_size):
                if done[b][slot]:
                    continue
                tau = taus[b][slot]
                r   = get_valid_mask(tau, len(tau), max_len)
                p_b = p[b] * r
                if p_b.sum() == 0:
                    p_b = r.clone()          # fall back to the uniform-valid draw
                if p_b.sum() == 0:
                    done[b][slot] = True     # genuine dead end
                    continue
                tok = dc.idx2tok(int(Categorical(p_b / p_b.sum()).sample().item()))
                tau.append(tok)
                x[b, slot, l] = dc.tok2idx(tok)
                if dc.is_complete(tau):
                    done[b][slot] = True

    return [{int(order[b, s]): taus[b][s] for s in range(n_dof)}
            for b in range(batch_size)]


def get_valid_mask(tau, position, max_len):
    """``get_valid_tokens`` as a float tensor (the callers all want one)."""
    return torch.as_tensor(dc.get_valid_tokens(tau, position, max_len),
                           dtype=torch.float32)


# ── Beam search ─────────────────────────────────────────────────────────────
def beam_search_slice(policy, x, dof_order, slot, beam_width, max_len,
                      n_return=10):
    """Beam search **within one slice**, with the preceding slices committed.

    ``x`` is ``(1, N, L)`` carrying those committed slices, with ``slot``'s own
    positions left at ``MASK_TOKEN``.  A joint beam over all ``N*L`` positions
    is not worth its branching factor, so the slices before this one are simply
    taken as given.

    Unlike :func:`discode_core.beam_search_expressions`, the distribution is
    recomputed after every committed token — under autoregression there is no
    single static marginal matrix to read rows out of.
    """
    policy.eval()
    device = next(policy.parameters()).device
    order  = np.asarray(dof_order, dtype=np.int64).reshape(-1)
    order_t = torch.as_tensor(order, dtype=torch.long, device=device).unsqueeze(0)

    beams = [(0.0, [])]
    for pos in range(max_len):
        live = [(lp, t) for lp, t in beams if not dc.is_complete(t)]
        fin  = [(lp, t) for lp, t in beams if dc.is_complete(t)]
        if not live:
            break

        xb = x.repeat(len(live), 1, 1)
        for i, (_lp, t) in enumerate(live):
            for k, tk in enumerate(t[:max_len]):
                xb[i, slot, k] = dc.tok2idx(tk)
        with torch.no_grad():
            logits = policy(xb, order_t.expand(len(live), -1),
                            n_slots=slot + 1)[:, slot, pos, :]
        p = torch.softmax(logits.float(), dim=-1).cpu()

        new_beams = list(fin)
        for i, (lp, t) in enumerate(live):
            mask  = get_valid_mask(t, pos, max_len)
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


def beam_candidates(policy, dof_order, max_len, n_dof, beam_width,
                    n_return=10, device=None):
    """One beam-searched slice per DOF, each conditioned on a sampled prefix.

    Returns ``[(dof, tau, context), ...]``.  The prefix slices are sampled from
    the policy exactly as in :func:`sample_joint_batch`, so the contexts these
    candidates carry are built the same way as the sampled ones.
    """
    device = next(policy.parameters()).device if device is None else device
    order  = np.asarray(dof_order, dtype=np.int64).reshape(-1)

    prefix = sample_joint_batch(policy, 1, max_len, n_dof, order, device=device)[0]
    out = []
    for slot in range(n_dof):
        d = int(order[slot])
        x = torch.full((1, n_dof, max_len), dc.MASK_TOKEN,
                       dtype=torch.long, device=device)
        slice_taus = []
        for s in range(n_dof):
            tau_s = prefix[int(order[s])] if s < slot else []
            slice_taus.append(tau_s)
            for k, tk in enumerate(tau_s[:max_len]):
                x[0, s, k] = dc.tok2idx(tk)
        ctx = make_context(order, slot, slice_taus)
        for tau in beam_search_slice(policy, x, order, slot, beam_width,
                                     max_len, n_return=n_return):
            out.append((d, tau, dict(ctx)))
    return out


# ── J-GRPO under autoregression ─────────────────────────────────────────────
def build_ar_inputs(entries, n_dof, max_len, device):
    """Teacher-forcing inputs for buffer entries ``(r, tau, consts, context)``.

    For each entry: the context's ``prefix_slices`` go into their slots, the
    entry's own ``tau`` into ``context['slot']``, and everything else stays
    ``MASK_TOKEN`` — including the tail of every slice after its last token,
    which is exactly the padding :func:`sample_joint_batch` leaves behind.

    Entries in one call may carry different ``dof_order``s; the model takes the
    order per batch element, so no grouping is needed.

    returns ``(x, dof_order, slots)``.
    """
    B = len(entries)
    x     = torch.full((B, n_dof, max_len), dc.MASK_TOKEN, dtype=torch.long)
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
            for k, tk in enumerate(t[:max_len]):
                x[bi, s, k] = dc.tok2idx(tk)
    return x.to(device), order.to(device), slots


def jgrpo_ar(policy, policy_old, policy_ref, S_alpha, R_alpha, dof,
             n_dof, max_len, eps, beta, critic=None):
    """J-GRPO for one DOF against the shared autoregressive policy.

    Differs from :func:`discode_core.compute_jgrpo` only in how the inputs are
    built and which positions are scored:

    * No ``forward_diffusion`` — the input is the teacher-forced joint
      sequence, and the causal mask does the hiding that masking used to do.
    * Only the positions of the entry's own slot contribute to the objective.

    The PPO clip, the KL-to-reference term, the ``MAX_GRPO_BATCH`` prioritised
    subsampling in log-residual units, and ``log_residual_score`` advantages are
    unchanged; they were tuned and this change does not touch them.

    Returns ``(objective, entropy, contributed)``.  ``contributed=False`` means
    this DOF has nothing to say this step — under a *shared* policy that must
    zero out this DOF alone and leave the others' gradients intact, which is
    why the three degenerate cases return a flag instead of aborting the step
    the way the per-DOF version could afford to.
    """
    device = next(policy.parameters()).device
    zero   = torch.zeros((), device=device)

    valid = [e for e in S_alpha if e[0] - R_alpha > 0]
    if not valid:
        return zero, zero, False

    if len(valid) > dc.MAX_GRPO_BATCH:
        # Prioritise in log-residual units so near-top refinements are not
        # crowded out by the compressed r-scale.
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

    x, order, slots = build_ar_inputs(valid, n_dof, max_len, device)

    logits_cur = policy(x, order)
    with torch.no_grad():
        logits_old = policy_old(x, order)
        logits_ref = policy_ref(x, order)

    # ── Group-relative advantages, within THIS DOF's batch ──────────────────
    # Never pooled across DOFs: reward scales differ wildly between them, and
    # pooling would let the DOF with the wider spread dominate the shared
    # trunk.  Advantages are in log-residual units for the reason given in
    # discode_core.compute_jgrpo.
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
    if critic is not None:
        with torch.no_grad():
            crit_in = []
            for e in valid:
                tokens = ([dc.tok2idx(tk) for tk in e[1]]
                          + [dc.MASK_TOKEN] * (max_len - len(e[1])))
                crit_in.append(tokens[:max_len])
            _critic_baselines = critic(
                torch.tensor(crit_in, dtype=torch.long, device=device)
            ).cpu().numpy()

    log_p_cur = F.log_softmax(logits_cur, dim=-1)
    total_obj = torch.zeros((), device=device)
    total_ent = torch.zeros((), device=device)
    n_terms   = 0

    for bi, entry in enumerate(valid):
        tau  = entry[1]
        slot = slots[bi]
        A_i  = float(group_adv[bi])
        for k, tok in enumerate(tau):
            if k >= max_len:
                break
            ti       = dc.tok2idx(tok)
            lp_cur_k = log_p_cur[bi, slot, k, ti]
            lp_old_k = F.log_softmax(logits_old[bi, slot, k], dim=-1)[ti].detach()
            h        = torch.exp(lp_cur_k - lp_old_k)
            # PPO clip: min(h*A, clip(h)*A).  Multiplying by A *after* the min
            # is only equivalent for A >= 0; for A < 0 it leaves the ratio
            # unclipped below 1-eps, giving unbounded down-weighting gradients
            # on tokens shared with good expressions.
            obj_k    = torch.min(h * A_i, torch.clamp(h, 1 - eps, 1 + eps) * A_i)
            p_ref    = torch.softmax(logits_ref[bi, slot, k], dim=-1).detach()
            kl_k     = F.kl_div(log_p_cur[bi, slot, k], p_ref,
                                reduction='sum', log_target=False)
            total_obj = total_obj + obj_k - beta * kl_k
            total_ent = total_ent + Categorical(
                logits=logits_cur[bi, slot, k]).entropy()
            n_terms  += 1

    if n_terms == 0:
        return zero, zero, False
    return total_obj / n_terms, total_ent / n_terms, True
