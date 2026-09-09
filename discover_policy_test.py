"""
discover_policy_test.py
======================

Correctness invariants for :mod:`discover_policy`.  Run this before anything
long::

    python discover_policy_test.py

Each check here has a specific failure it exists to catch, and every one of
them is silent — a model with a mis-aligned teacher-forcing shift, or with
information leaking backwards through the attention mask, trains perfectly
happily towards the wrong target.

The numbering follows §7 of the implementation plan.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import discover_core as dc
import discover_policy as dp


N_DOF   = 2
MAX_LEN = 12
TAU_A   = ['add', 'x1', 'x2', 'x3', 'intpower', 'x1', 'end']
TAU_B   = ['add', 'x1', 'x3', 'x4', 'intpower', 'x3', 'end']

_PASS, _FAIL = [], []


def check(name, ok, detail=''):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ''))
    return ok


def make_policy(cross=True, seed=0, n_dof=N_DOF, max_len=MAX_LEN):
    torch.manual_seed(seed)
    return dp.ARJointPolicy(n_tokens=dc.N_TOKENS, max_len=max_len, n_dof=n_dof,
                            d_model=64, n_heads=4, ff_dim=128, n_layers=2,
                            cross_slice_attention=cross).eval()


def write(x, b, slot, tau):
    for k, tk in enumerate(tau[:x.shape[2]]):
        x[b, slot, k] = dc.tok2idx(tk)
    return x


def random_x(policy, batch=2, seed=1):
    """A batch of syntactically arbitrary but in-range token tensors."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, policy.vocab_size, (batch, policy.n_dof, policy.max_len),
                         generator=g)


# ── 7.1 Teacher-forcing alignment ───────────────────────────────────────────
def test_7_1_teacher_forcing():
    """Overfit one example; the argmax must land on tau[k], not tau[k+1].

    An off-by-one in the shift is the single most common way this change
    breaks: a model trained one position out still trains, just to the wrong
    target, and every downstream number looks plausible.
    """
    print("\n7.1  teacher-forcing alignment")
    policy = make_policy()
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    slot = 1
    order = torch.tensor([[1, 0]], dtype=torch.long)
    x = torch.full((1, N_DOF, MAX_LEN), dc.MASK_TOKEN, dtype=torch.long)
    write(x, 0, 0, TAU_A)
    write(x, 0, slot, TAU_B)
    target = torch.tensor([dc.tok2idx(t) for t in TAU_B], dtype=torch.long)

    for _ in range(400):
        logits = policy(x, order)[0, slot, :len(TAU_B), :]
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    with torch.no_grad():
        pred = policy(x, order)[0, slot, :len(TAU_B), :].argmax(-1)
    got = [dc.idx2tok(int(i)) for i in pred]
    ok = got == TAU_B
    check('7.1 argmax reproduces tau[k] (not tau[k+1])', ok,
          f"loss={loss.item():.2e} got={got}")

    # A shifted model would instead reproduce tau[k+1]; name that explicitly.
    shifted = got[:-1] == TAU_B[1:]
    check('7.1 model is NOT predicting tau[k+1]', not shifted or ok)


# ── 7.2 Causal masking ──────────────────────────────────────────────────────
def test_7_2_causal():
    """Perturbing token (s, l) must not move any logit at or before it.

    Token (s, l) enters the input at position (s, l+1) under the per-slice
    shift, so logits at every flattened input position <= s*P + l are
    untouched.  If they move, information is flowing backwards and the AR
    factorisation is invalid.
    """
    print("\n7.2  causal masking")
    policy = make_policy()
    x = random_x(policy)
    with torch.no_grad():
        base = policy(x, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

    s, l = 1, 4
    x2 = x.clone()
    x2[0, s, l] = (x2[0, s, l] + 3) % policy.vocab_size
    with torch.no_grad():
        pert = policy(x2, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

    earlier_slices = torch.equal(base[0, :s], pert[0, :s])
    same_slice     = torch.equal(base[0, s, :l + 1], pert[0, s, :l + 1])
    other_row      = torch.equal(base[1], pert[1])
    later          = not torch.equal(base[0, s, l + 1:], pert[0, s, l + 1:])

    check('7.2 logits in earlier slices bitwise unchanged', earlier_slices)
    check('7.2 logits at <= (s,l) in the same slice bitwise unchanged', same_slice)
    check('7.2 other batch rows unaffected', other_row)
    check('7.2 logits after (s,l) DO change (test is not vacuous)', later)


# ── 7.3 Slice isolation ─────────────────────────────────────────────────────
def test_7_3_slice_isolation():
    """cross_slice_attention=False must make the slices truly independent.

    This is the check that catches a flattened (rather than per-slice) BOS
    shift: with a flattened shift, slice 0's last token becomes slice 1's first
    input and leaks across the boundary through the input, where no attention
    mask can stop it.
    """
    print("\n7.3  slice isolation (cross_slice_attention=False)")
    order = torch.tensor([[0, 1]], dtype=torch.long)

    for cross, expect_leak in ((False, False), (True, True)):
        policy = make_policy(cross=cross)
        x = random_x(policy, batch=1, seed=7)
        with torch.no_grad():
            base = policy(x, order)

        moved_any = False
        for l in range(policy.max_len):          # every position, incl. L-1
            x2 = x.clone()
            x2[0, 0, l] = (x2[0, 0, l] + 5) % policy.vocab_size
            with torch.no_grad():
                pert = policy(x2, order)
            if not torch.equal(base[0, 1], pert[0, 1]):
                moved_any = True
                break

        if expect_leak:
            check('7.3 with cross-attention ON, slice 0 DOES reach slice 1',
                  moved_any, 'incl. the last token, via the readout position')
        else:
            check('7.3 with cross-attention OFF, slice 1 is untouched by slice 0',
                  not moved_any)


# ── n_slots truncation (used by the sampler) ────────────────────────────────
def test_n_slots_equivalence():
    """Truncating the forward to the first S slots must be exactly equivalent.

    The sampler relies on this to avoid running the full N*L sequence for every
    token; causality guarantees it, but a mask built for the wrong length would
    break it quietly.
    """
    print("\nextra  n_slots truncation equivalence")
    policy = make_policy()
    x = random_x(policy, batch=3, seed=11)
    order = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.long)
    with torch.no_grad():
        full = policy(x, order)
        trunc = policy(x, order, n_slots=1)
    check('n_slots=1 matches the first slot of the full forward',
          torch.allclose(full[:, :1], trunc, atol=0, rtol=0))


# ── 7.4 Grammar validity ────────────────────────────────────────────────────
def test_7_4_grammar():
    """Every sampled tau is a valid prefix; most are complete expressions.

    Incomplete taus are legitimate (a genuine dead end, or the length budget
    running out) and are discarded downstream by ``energy_worker``; what must
    never happen is a tau that violates the grammar outright.
    """
    print("\n7.4  grammar validity of sampled expressions")
    policy = make_policy()
    order = dp.make_batch_orders((0, 1), 64, slice_order='random',
                                 rng=np.random.default_rng(0))
    samples = dp.sample_joint_batch(policy, 64, MAX_LEN, N_DOF, order)

    taus = [tau for s in samples for tau in s.values()]
    bad  = [t for t in taus if dc._parse_stack(t)[0] is None]
    complete = [t for t in taus if dc.is_complete(t)]
    uniq = {tuple(t) for t in complete}
    per_dof = [len(s) == N_DOF and set(s) == set(range(N_DOF)) for s in samples]

    check('7.4 no sampled tau violates the grammar', not bad,
          f"{len(bad)} bad of {len(taus)}")
    check('7.4 samples are keyed by DOF, one slice each', all(per_dof))
    check('7.4 an untrained policy produces complete expressions',
          len(complete) > 0.5 * len(taus),
          f"{len(complete)}/{len(taus)} complete")
    check('7.4 samples are diverse', len(uniq) > 0.5 * max(len(complete), 1),
          f"{len(uniq)} unique of {len(complete)} complete")
    check('7.4 no tau exceeds max_len', all(len(t) <= MAX_LEN for t in taus))


# ── 7.5 Context reproducibility ─────────────────────────────────────────────
def test_7_5_context_reproducible():
    """A stored entry must give identical logits whenever it is replayed.

    This is what keeps buffer entries comparable across epochs.  It has to hold
    regardless of what else shares the batch — if batch composition moved an
    entry's logits, ``logits_cur`` / ``logits_old`` / ``logits_ref`` could
    disagree about what the entry even means.
    """
    print("\n7.5  context reproducibility")
    policy = make_policy()
    entry_a = (0.9, TAU_B, [], {'dof_order': (0, 1), 'slot': 1,
                                'prefix_slices': {0: TAU_A}})
    entry_b = (0.8, TAU_A, [], {'dof_order': (1, 0), 'slot': 1,
                                'prefix_slices': {1: TAU_B}})

    def logits_for(entries, which):
        x, order, slots = dp.build_ar_inputs(entries, N_DOF, MAX_LEN,
                                             torch.device('cpu'))
        with torch.no_grad():
            out = policy(x, order)
        return out[which, slots[which]]

    alone  = logits_for([entry_a], 0)
    shared = logits_for([entry_a, entry_b], 0)
    again  = logits_for([entry_a], 0)

    check('7.5 replaying the same entry is bitwise identical',
          torch.equal(alone, again))
    check('7.5 batch composition does not move an entry\'s logits',
          torch.equal(alone, shared))


# ── 7.6 Padding convention ──────────────────────────────────────────────────
def test_7_6_padding():
    """The tensor the update builds must equal the one sampling produced.

    Sampling leaves MASK_TOKEN after ``is_complete`` fires and in every slice
    that has not been reached yet.  ``build_ar_inputs`` has to reproduce both,
    byte for byte, or train and sample distributions drift apart invisibly.
    """
    print("\n7.6  padding convention (sample vs. update)")
    policy = make_policy()
    order = np.array([1, 0])
    sample = dp.sample_joint_batch(policy, 1, MAX_LEN, N_DOF, order)[0]
    slice_taus = [sample[int(order[s])] for s in range(N_DOF)]

    ok = True
    for slot in range(N_DOF):
        ctx = dp.make_context(order, slot, slice_taus)
        entry = (1.0, slice_taus[slot], [], ctx)
        x, ord_t, slots = dp.build_ar_inputs([entry], N_DOF, MAX_LEN,
                                             torch.device('cpu'))
        # What the sampler had in hand at the moment it generated this slot.
        expect = torch.full((N_DOF, MAX_LEN), dc.MASK_TOKEN, dtype=torch.long)
        for s in range(slot + 1):
            for k, tk in enumerate(slice_taus[s][:MAX_LEN]):
                expect[s, k] = dc.tok2idx(tk)
        ok &= torch.equal(x[0], expect)
        ok &= slots[0] == slot
        ok &= torch.equal(ord_t[0], torch.as_tensor(order))

    check('7.6 update inputs reproduce the sampler\'s padding exactly', bool(ok))

    # The context must carry the preceding slices and nothing else.
    ctx = dp.make_context(order, 1, slice_taus)
    check('7.6 context stores only the slices that preceded it',
          set(ctx['prefix_slices']) == {int(order[0])})


# ── jgrpo_ar plumbing ───────────────────────────────────────────────────────
def test_jgrpo_ar():
    """The update runs, produces gradients, and degrades gracefully."""
    print("\nextra  jgrpo_ar")
    import copy
    policy = make_policy()
    policy.train()
    old = copy.deepcopy(policy).eval()
    ref = copy.deepcopy(policy).eval()

    ctx0 = {'dof_order': (0, 1), 'slot': 1, 'prefix_slices': {0: TAU_A}}
    buf = [(0.9, TAU_B, [], ctx0), (0.6, TAU_A, [], ctx0), (0.3, TAU_B[:5] + ['end'], [], ctx0)]
    R = min(e[0] for e in buf)

    obj, ent, ok = dp.jgrpo_ar(policy, old, ref, buf, R, dof=1, n_dof=N_DOF,
                               max_len=MAX_LEN, eps=0.2, beta=0.01)
    check('jgrpo_ar contributes on a normal buffer', ok)
    if ok:
        (-(obj + 0.02 * ent)).backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        check('jgrpo_ar produces gradients',
              bool(grads) and any(g.abs().sum() > 0 for g in grads))

    # Degenerate spread must zero out THIS DOF only, not abort the step.
    flat = [(0.5, TAU_B, [], ctx0), (0.5 + 1e-9, TAU_A, [], ctx0)]
    _o, _e, ok2 = dp.jgrpo_ar(policy, old, ref, flat, 0.4, dof=1, n_dof=N_DOF,
                              max_len=MAX_LEN, eps=0.2, beta=0.01)
    check('jgrpo_ar reports no contribution on a degenerate spread', not ok2)

    # A context whose slot does not belong to the DOF being trained is a bug.
    try:
        dp.jgrpo_ar(policy, old, ref, buf, R, dof=0, n_dof=N_DOF,
                    max_len=MAX_LEN, eps=0.2, beta=0.01)
        caught = False
    except ValueError:
        caught = True
    check('jgrpo_ar rejects a context belonging to another DOF', caught)


# ── slice ordering ──────────────────────────────────────────────────────────
def test_ordering():
    print("\nextra  slice ordering")
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


# ═══════════════════════════════════════════════════════════════════════════
# Term-structured policy (TermBagPolicy) — same invariants, one level down
# ═══════════════════════════════════════════════════════════════════════════
N_TERMS  = 4
TERM_LEN = 6


def make_term_policy(cross=True, seed=0, term_pe=True):
    torch.manual_seed(seed)
    return dp.TermBagPolicy(n_tokens=dc.N_TOKENS, max_terms=N_TERMS,
                            max_term_len=TERM_LEN, n_dof=N_DOF,
                            d_model=64, n_heads=4, ff_dim=128, n_layers=2,
                            cross_slice_attention=cross,
                            term_position_encoding=term_pe).eval()


def random_term_x(policy, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, policy.vocab_size,
                         (batch, policy.n_dof, policy.max_terms, policy.max_term_len),
                         generator=g)


def _flat_reachable(tau, budget=80):
    """Is ``tau`` a sequence the FLAT grammar would itself generate?"""
    return all(dc.get_valid_tokens(tau[:i], i, budget)[dc.tok2idx(tk)]
               for i, tk in enumerate(tau))


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


def test_t2_term_readout():
    """A term's LAST token lives only on its trailing readout position.  If the
    readout were dropped from the input, the next term could never see it —
    the model would train fine and silently ignore every term's final token.
    Tested in isolation (no early break), unlike 7.3."""
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


def test_t3_term_slice_isolation():
    """cross_slice_attention=False must isolate slices in the term layout too."""
    print("\nT.3  slice isolation under the term layout")
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


def test_t4_term_n_slots():
    """n_slots truncation must be exact — the sampler relies on it."""
    print("\nT.4  n_slots equivalence (term layout)")
    policy = make_term_policy()
    x = random_term_x(policy)
    order = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        full, trunc = policy(x, order), policy(x, order, n_slots=1)
    check('T.4 n_slots=1 matches the first slot of the full forward',
          torch.allclose(full[:, :1], trunc, atol=0, rtol=0))


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

    check('T.5 every sampled bag assembles to a complete tau',
          len(complete) == len(taus), f"{len(complete)}/{len(taus)}")
    check('T.5 every assembled tau is reachable under the flat grammar',
          len(reach) == len(complete), f"{len(reach)}/{len(complete)}")
    check('T.5 no bag exceeds max_terms', all(len(b) <= N_TERMS for b in bags))
    check('T.5 no term exceeds max_term_len',
          all(len(t) <= TERM_LEN for b in bags for t in b))
    check('T.5 no term is add-rooted', all(t[0] != 'add' for b in bags for t in b))
    check('T.5 samples are diverse', len(uniq) > 0.5 * max(len(complete), 1),
          f"{len(uniq)} unique of {len(complete)}")


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


def test_t7_term_padding():
    """build_term_inputs must reproduce the sampler's tensor byte for byte:
    MASK after each term's last token, MASK in every unused term slot, MASK in
    every not-yet-generated slice.  A mismatch trains against inputs the sample
    never saw and degrades silently."""
    print("\nT.7  update inputs reproduce the sampler's padding (term layout)")
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


def test_t8_term_context_reproducible():
    """Replaying an entry must be bitwise reproducible and independent of what
    else is in the batch."""
    print("\nT.8  context reproducibility (term layout)")
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


def test_jgrpo_terms():
    """Beyond the jgrpo_ar smoke checks, this pins the one thing the flat
    version never tested numerically: with policy_old a deepcopy, every scored
    ratio h must be EXACTLY 1.  A term-axis indexing slip (scoring [j, p] for
    the token at [j, p+1], or the wrong term) breaks that and nothing else."""
    print("\nextra  jgrpo_terms")
    import copy
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

    flat = [(0.5, e[1], [], e[3]) for e in buf]
    _o, _e, ok2 = dp.jgrpo_terms(policy, old, ref, flat, 0.4, dof=1, n_dof=N_DOF,
                                 eps=0.2, beta=0.01)
    check('jgrpo_terms reports no contribution on a degenerate spread', not ok2)

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


def test_change2_band():
    """Change 2 must remove exactly one embedding band and nothing else — the
    mask, sampler and update are shared.  If anything else differed between
    the two configs the ablation would not be attributable."""
    print("\nextra  Change 2 = one embedding band")
    a = make_term_policy(term_pe=True); b = make_term_policy(term_pe=False)
    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    check('Change 2 policy has no term_emb', b.term_emb is None)
    check('Change 2 removes exactly max_terms * d_model parameters',
          n_a - n_b == N_TERMS * a.d_model, f"{n_a - n_b} vs {N_TERMS * a.d_model}")
    x = random_term_x(b); order = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        out = b(x, order)
    check('Change 2 forward has the same output shape',
          tuple(out.shape) == (2, N_DOF, N_TERMS, TERM_LEN, dc.N_TOKENS + 1))


def main():
    dc.configure_grammar(N_DOF)
    print(f"grammar: {dc.ALL_TOKENS}  (N_TOKENS={dc.N_TOKENS})")
    test_7_1_teacher_forcing()
    test_7_2_causal()
    test_7_3_slice_isolation()
    test_n_slots_equivalence()
    test_7_4_grammar()
    test_7_5_context_reproducible()
    test_7_6_padding()
    test_jgrpo_ar()
    test_ordering()
    # term-structured policy
    test_t1_term_causal()
    test_t2_term_readout()
    test_t3_term_slice_isolation()
    test_t4_term_n_slots()
    test_t5_term_grammar()
    test_t6_split_assemble()
    test_t7_term_padding()
    test_t8_term_context_reproducible()
    test_jgrpo_terms()
    test_change2_band()

    print(f"\n{'=' * 60}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    return 1 if _FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
