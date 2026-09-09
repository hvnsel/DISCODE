"""
discover_policy_test.py
=======================

Correctness invariants for :mod:`discover_policy`.  Run this before anything
long::

    python discover_policy_test.py

Each check here has a specific failure it exists to catch, and every one of
them is silent — a model with a mis-aligned teacher-forcing shift, or with
information leaking backwards through the attention mask, trains perfectly
happily towards the wrong target.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

import discover_core as dc
import discover_policy as dp


N_DOF    = 2
N_TERMS  = 4
TERM_LEN = 6

_PASS, _FAIL = [], []


def check(name, ok, detail=''):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ''))
    return ok


def make_term_policy(cross=True, seed=0, term_pe=True):
    torch.manual_seed(seed)
    return dp.TermBagPolicy(n_tokens=dc.N_TOKENS, max_terms=N_TERMS,
                            max_term_len=TERM_LEN, n_dof=N_DOF,
                            d_model=64, n_heads=4, ff_dim=128, n_layers=2,
                            cross_slice_attention=cross,
                            term_position_encoding=term_pe).eval()


def random_term_x(policy, batch=2, seed=1):
    """A batch of syntactically arbitrary but in-range token tensors."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, policy.vocab_size,
                         (batch, policy.n_dof, policy.max_terms, policy.max_term_len),
                         generator=g)


def _flat_reachable(tau, budget=80):
    """Is ``tau`` a sequence the FLAT grammar would itself generate?"""
    return all(dc.get_valid_tokens(tau[:i], i, budget)[dc.tok2idx(tk)]
               for i, tk in enumerate(tau))


# ── slice ordering ──────────────────────────────────────────────────────────
def test_ordering():
    print("\nslice ordering")
    rng = np.random.default_rng(0)
    check('choose_order puts the most-converged DOF first',
          dp.choose_order([-0.5, 0.9, 0.2], 'reward', rng) == (1, 2, 0))
    check('choose_order falls back to random when nothing has converged',
          len(set(dp.choose_order([-np.inf] * 3, 'reward', rng))) == 3)
    check('choose_order fixed is the identity',
          dp.choose_order([-np.inf, 1.0], 'fixed', rng) == (0, 1))

    orders = dp.make_batch_orders((0, 1), 200, 'random', rng)
    frac = float((orders[:, 0] == 1).mean())
    check('make_batch_orders mixes permuted and base rows', 0.1 < frac < 0.6,
          f"{frac:.2f} of rows lead with DOF 1")
    fixed = dp.make_batch_orders((1, 0), 20, 'fixed', rng)
    check('make_batch_orders fixed keeps the base order everywhere',
          bool((fixed == np.array([1, 0])).all()))


# ── T.1 Term-level causality ────────────────────────────────────────────────
def test_t1_term_causal():
    """Perturbing a token in term j must move no logit in an earlier term, in
    an earlier slice, at or before its own position, or in another batch row.
    An off-by-one in the 3-level flatten would leak here and nowhere else."""
    print("\nT.1  term-level causality")
    policy = make_term_policy()
    x = random_term_x(policy)
    order = torch.tensor([[0, 1], [1, 0]])
    s, j, p = 1, 1, 2
    x2 = x.clone()
    x2[0, s, j, p] = (x2[0, s, j, p] + 3) % policy.vocab_size
    with torch.no_grad():
        base, pert = policy(x, order), policy(x2, order)

    check('T.1 earlier slices bitwise unchanged', torch.equal(base[0, :s], pert[0, :s]))
    check('T.1 earlier terms in the same slice bitwise unchanged',
          torch.equal(base[0, s, :j], pert[0, s, :j]))
    check('T.1 positions <= p in the same term bitwise unchanged',
          torch.equal(base[0, s, j, :p + 1], pert[0, s, j, :p + 1]))
    check('T.1 other batch rows unaffected', torch.equal(base[1], pert[1]))
    check('T.1 later positions in the same term DO change (not vacuous)',
          not torch.equal(base[0, s, j, p + 1:], pert[0, s, j, p + 1:]))
    check('T.1 later terms in the same slice DO change (not vacuous)',
          not torch.equal(base[0, s, j + 1:], pert[0, s, j + 1:]))


# ── T.2 Readout position ────────────────────────────────────────────────────
def test_t2_term_readout():
    """A term's LAST token lives only on its trailing readout position.  If the
    readout were dropped from the input, the next term could never see it —
    the model would train fine and silently ignore every term's final token.
    Tested in isolation (no early break)."""
    print("\nT.2  readout: a term's last token reaches the next term")
    policy = make_term_policy()
    x = random_term_x(policy)
    order = torch.tensor([[0, 1], [1, 0]])
    x2 = x.clone()
    x2[0, 0, 0, TERM_LEN - 1] = (x2[0, 0, 0, TERM_LEN - 1] + 5) % policy.vocab_size
    with torch.no_grad():
        base, pert = policy(x, order), policy(x2, order)
    check('T.2 last token of term 0 moves term 1\'s logits',
          not torch.equal(base[0, 0, 1], pert[0, 0, 1]))
    check('T.2 ...and the next slice\'s logits (cross-slice on)',
          not torch.equal(base[0, 1], pert[0, 1]))


# ── T.3 Slice isolation ─────────────────────────────────────────────────────
def test_t3_term_slice_isolation():
    """cross_slice_attention=False must make the slices truly independent.

    This is the check that catches a flattened (rather than per-term) BOS
    shift: with a flattened shift, one slice's last token becomes the next
    slice's first input and leaks across the boundary through the input,
    where no attention mask can stop it."""
    print("\nT.3  slice isolation (cross_slice_attention=False)")
    order = torch.tensor([[0, 1], [1, 0]])
    for cross, expect_leak in ((False, False), (True, True)):
        policy = make_term_policy(cross=cross)
        x = random_term_x(policy)
        moved = False
        for j in range(N_TERMS):
            for l in range(TERM_LEN):
                x2 = x.clone()
                x2[0, 0, j, l] = (x2[0, 0, j, l] + 5) % policy.vocab_size
                with torch.no_grad():
                    if not torch.equal(policy(x, order)[0, 1], policy(x2, order)[0, 1]):
                        moved = True
                        break
            if moved:
                break
        if expect_leak:
            check('T.3 with cross-attention ON, slice 0 reaches slice 1', moved)
        else:
            check('T.3 with cross-attention OFF, slice 1 is untouched by slice 0', not moved)


# ── T.4 n_slots truncation (used by the sampler) ────────────────────────────
def test_t4_term_n_slots():
    """Truncating the forward to the first S slots must be exactly equivalent.

    The sampler relies on this to avoid running the full sequence for every
    token; causality guarantees it, but a mask built for the wrong length
    would break it quietly."""
    print("\nT.4  n_slots equivalence")
    policy = make_term_policy()
    x = random_term_x(policy)
    order = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        full, trunc = policy(x, order), policy(x, order, n_slots=1)
    check('T.4 n_slots=1 matches the first slot of the full forward',
          torch.allclose(full[:, :1], trunc, atol=0, rtol=0))


# ── T.5 Grammar validity ────────────────────────────────────────────────────
def test_t5_term_grammar():
    """Every assembled tau must be something the FLAT grammar could itself have
    written.  An add-rooted term, a bare-leaf singleton, or a term over the
    depth budget would each pass is_complete and score — and be a candidate
    no other part of the system can represent."""
    print("\nT.5  grammar validity of sampled bags")
    policy = make_term_policy()
    order = dp.make_batch_orders((0, 1), 64, slice_order='random',
                                 rng=np.random.default_rng(0))
    samples = dp.sample_term_batch(policy, 64, N_DOF, order)
    taus = [tau for s in samples for tau in s.values()]
    complete = [t for t in taus if dc.is_complete(t)]
    reach = [t for t in complete if _flat_reachable(t)]
    bags = [dc.split_terms(t) for t in complete]
    uniq = {tuple(t) for t in complete}
    per_dof = [len(s) == N_DOF and set(s) == set(range(N_DOF)) for s in samples]

    check('T.5 every sampled bag assembles to a complete tau',
          len(complete) == len(taus), f"{len(complete)}/{len(taus)}")
    check('T.5 every assembled tau is reachable under the flat grammar',
          len(reach) == len(complete), f"{len(reach)}/{len(complete)}")
    check('T.5 samples are keyed by DOF, one slice each', all(per_dof))
    check('T.5 no bag exceeds max_terms', all(len(b) <= N_TERMS for b in bags))
    check('T.5 no term exceeds max_term_len',
          all(len(t) <= TERM_LEN for b in bags for t in b))
    check('T.5 no term is add-rooted', all(t[0] != 'add' for b in bags for t in b))
    check('T.5 samples are diverse', len(uniq) > 0.5 * max(len(complete), 1),
          f"{len(uniq)} unique of {len(complete)}")


# ── T.6 split / assemble round trip ─────────────────────────────────────────
def test_t6_split_assemble():
    """split_terms must invert assemble_terms exactly, including m == 1, or the
    update replays a different prefix than the sampler saw."""
    print("\nT.6  split_terms / assemble_terms round trip")
    rng = np.random.default_rng(0)
    V = ['x1', 'x2', 'x3', 'x4']

    def node(d=0):
        r = rng.random()
        if d >= 2 or r < .4:
            return [str(rng.choice(V))]
        if r < .6:
            return ['intpower', str(rng.choice(V))]
        ch = [node(d + 1) for _ in range(rng.integers(2, 4))]
        return ['mul'] + [t for c in ch for t in c] + ['end']

    ok = True
    for _ in range(200):
        m = int(rng.integers(1, 6))
        terms = [node() for _ in range(m)]
        if m == 1 and len(terms[0]) == 1:
            terms[0] = ['intpower', terms[0][0]]        # the sampler forbids this bag
        tau = dc.assemble_terms(terms)
        ok &= dc.split_terms(tau) == terms and dc.is_complete(tau)
    check('T.6 split(assemble(terms)) == terms, incl. m == 1', bool(ok))
    check('T.6 a non-add tau is a single term',
          dc.split_terms(['intpower', 'x1']) == [['intpower', 'x1']])
    # The failure mode the term-root rule exists for:
    bad = dc.assemble_terms([['add', 'x1', 'x2', 'end'], ['x3']])
    check('T.6 an add-rooted term assembles to something the flat grammar rejects',
          dc.is_complete(bad) and not _flat_reachable(bad),
          'is_complete accepts it; get_valid_tokens does not')


# ── T.7 Padding convention ──────────────────────────────────────────────────
def test_t7_term_padding():
    """build_term_inputs must reproduce the sampler's tensor byte for byte:
    MASK after each term's last token, MASK in every unused term slot, MASK in
    every not-yet-generated slice.  A mismatch trains against inputs the sample
    never saw and degrades silently."""
    print("\nT.7  update inputs reproduce the sampler's padding")
    policy = make_term_policy()
    order = np.array([1, 0])
    sample = dp.sample_term_batch(policy, 1, N_DOF, order)[0]
    slice_taus = [sample[int(order[s])] for s in range(N_DOF)]

    ok = True
    for slot in range(N_DOF):
        ctx = dp.make_context(order, slot, slice_taus)
        entry = (1.0, slice_taus[slot], [], ctx)
        x, ord_t, slots = dp.build_term_inputs([entry], N_DOF, N_TERMS, TERM_LEN,
                                               torch.device('cpu'))
        expect = torch.full((N_DOF, N_TERMS, TERM_LEN), dc.MASK_TOKEN, dtype=torch.long)
        for s in range(slot + 1):
            for j, term in enumerate(dc.split_terms(slice_taus[s])[:N_TERMS]):
                for p, tk in enumerate(term[:TERM_LEN]):
                    expect[s, j, p] = dc.tok2idx(tk)
        ok &= torch.equal(x[0], expect)
        ok &= slots[0] == slot
        ok &= torch.equal(ord_t[0], torch.as_tensor(order))
    check('T.7 update inputs reproduce the sampler\'s padding exactly', bool(ok))

    ctx = dp.make_context(order, 1, slice_taus)
    check('T.7 context stores only the slices that preceded it',
          set(ctx['prefix_slices']) == {int(order[0])})


# ── T.8 Context reproducibility ─────────────────────────────────────────────
def test_t8_term_context_reproducible():
    """A stored entry must give identical logits whenever it is replayed,
    regardless of what else shares the batch.  This is what keeps buffer
    entries comparable across epochs — if batch composition moved an entry's
    logits, ``logits_cur`` / ``logits_old`` / ``logits_ref`` could disagree
    about what the entry even means."""
    print("\nT.8  context reproducibility")
    policy = make_term_policy()
    tau_a = ['add', 'x1', 'x2', 'intpower', 'x1', 'end']
    tau_b = ['add', 'x3', 'mul', 'x1', 'x3', 'end', 'end']
    entry_a = (0.9, tau_b, [], {'dof_order': (0, 1), 'slot': 1, 'prefix_slices': {0: tau_a}})
    entry_b = (0.5, tau_a, [], {'dof_order': (1, 0), 'slot': 1, 'prefix_slices': {1: tau_b}})

    def logits_for(entries, which):
        x, order, slots = dp.build_term_inputs(entries, N_DOF, N_TERMS, TERM_LEN,
                                               torch.device('cpu'))
        with torch.no_grad():
            return policy(x, order)[which, slots[which]]

    alone, shared, again = (logits_for([entry_a], 0), logits_for([entry_a, entry_b], 0),
                            logits_for([entry_a], 0))
    check('T.8 replaying the same entry is bitwise identical', torch.equal(alone, again))
    check('T.8 batch composition does not move an entry\'s logits', torch.equal(alone, shared))


# ── jgrpo_terms plumbing ────────────────────────────────────────────────────
def test_jgrpo_terms():
    """The update runs, produces gradients, and degrades gracefully.  Beyond
    those smoke checks, this pins one thing numerically: with policy_old a
    deepcopy, every scored ratio h must be EXACTLY 1.  A term-axis indexing
    slip (scoring [j, p] for the token at [j, p+1], or the wrong term) breaks
    that and nothing else."""
    print("\njgrpo_terms")
    policy = make_term_policy()
    old = copy.deepcopy(policy).eval(); ref = copy.deepcopy(policy).eval()
    order = np.array([0, 1])
    samples = dp.sample_term_batch(policy, 8, N_DOF, order)
    buf = []
    for i, s in enumerate(samples):
        st = [s[0], s[1]]
        buf.append((0.3 + 0.08 * i, s[1], [], dp.make_context(order, 1, st)))

    policy.train()
    obj, ent, ok = dp.jgrpo_terms(policy, old, ref, buf, R_alpha=0.2, dof=1,
                                  n_dof=N_DOF, eps=0.2, beta=0.01)
    check('jgrpo_terms contributes on a normal buffer', ok)
    if ok:
        (-(obj + 0.02 * ent)).backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        check('jgrpo_terms produces gradients',
              bool(grads) and any(g.abs().sum() > 0 for g in grads))
    policy.eval()

    # Degenerate spread must zero out THIS DOF only, not abort the step.
    flat = [(0.5, e[1], [], e[3]) for e in buf]
    _o, _e, ok2 = dp.jgrpo_terms(policy, old, ref, flat, 0.4, dof=1, n_dof=N_DOF,
                                 eps=0.2, beta=0.01)
    check('jgrpo_terms reports no contribution on a degenerate spread', not ok2)

    # A context whose slot does not belong to the DOF being trained is a bug.
    caught = False
    try:
        dp.jgrpo_terms(policy, old, ref, buf, 0.2, dof=0, n_dof=N_DOF, eps=0.2, beta=0.01)
    except ValueError:
        caught = True
    check('jgrpo_terms rejects a context belonging to another DOF', caught)

    # h == 1 exactly, at every scored position, when old is a deepcopy.
    x, ord_t, slots = dp.build_term_inputs(buf, N_DOF, N_TERMS, TERM_LEN, torch.device('cpu'))
    with torch.no_grad():
        lc = torch.log_softmax(policy(x, ord_t), -1)
        lo = torch.log_softmax(old(x, ord_t), -1)
    max_dev, n_stop, n_full = 0.0, 0, 0
    for bi, e in enumerate(buf):
        terms = dc.split_terms(e[1])[:N_TERMS]
        pos = [(j, p, dc.tok2idx(t)) for j, tm in enumerate(terms) for p, t in enumerate(tm[:TERM_LEN])]
        if len(terms) < N_TERMS:
            pos.append((len(terms), 0, policy.stop_idx)); n_stop += 1
        else:
            n_full += 1
        for j, p, ti in pos:
            h = float(torch.exp(lc[bi, slots[bi], j, p, ti] - lo[bi, slots[bi], j, p, ti]))
            max_dev = max(max_dev, abs(h - 1.0))
    check('jgrpo_terms: h == 1 at every scored position with a deepcopy old',
          max_dev == 0.0, f"max |h-1| = {max_dev:.2e}")
    check('jgrpo_terms: STOP scored iff the bag is not full',
          n_stop + n_full == len(buf), f"{n_stop} partial, {n_full} full bags")


# ── term_position_encoding is one band ──────────────────────────────────────
def test_term_pe_band():
    """term_position_encoding=False must remove exactly one embedding band and
    nothing else — the mask, sampler and update are shared.  If anything else
    differed between the two configs a comparison between them would not be
    attributable."""
    print("\nterm_position_encoding = one embedding band")
    a = make_term_policy(term_pe=True); b = make_term_policy(term_pe=False)
    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    check('order-blind policy has no term_emb', b.term_emb is None)
    check('order-blind policy removes exactly max_terms * d_model parameters',
          n_a - n_b == N_TERMS * a.d_model, f"{n_a - n_b} vs {N_TERMS * a.d_model}")
    x = random_term_x(b); order = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        out = b(x, order)
    check('order-blind forward has the same output shape',
          tuple(out.shape) == (2, N_DOF, N_TERMS, TERM_LEN, dc.N_TOKENS + 1))


def main():
    dc.configure_grammar(N_DOF)
    print(f"grammar: {dc.ALL_TOKENS}  (N_TOKENS={dc.N_TOKENS})")
    test_ordering()
    test_t1_term_causal()
    test_t2_term_readout()
    test_t3_term_slice_isolation()
    test_t4_term_n_slots()
    test_t5_term_grammar()
    test_t6_split_assemble()
    test_t7_term_padding()
    test_t8_term_context_reproducible()
    test_jgrpo_terms()
    test_term_pe_band()

    print(f"\n{'=' * 60}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    return 1 if _FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
