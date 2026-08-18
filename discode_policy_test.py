"""
discode_policy_test.py
======================

Correctness invariants for :mod:`discode_policy`.  Run this before anything
long::

    python discode_policy_test.py

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

import discode_core as dc
import discode_policy as dp


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

    print(f"\n{'=' * 60}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    return 1 if _FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
