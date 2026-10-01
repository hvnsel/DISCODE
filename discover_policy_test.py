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
            return ['power', str(rng.choice(V))]
        ch = [node(d + 1) for _ in range(rng.integers(2, 4))]
        return ['mul'] + [t for c in ch for t in c] + ['end']

    ok = True
    for _ in range(200):
        m = int(rng.integers(1, 6))
        terms = [node() for _ in range(m)]
        if m == 1 and len(terms[0]) == 1:
            terms[0] = ['power', terms[0][0]]        # the sampler forbids this bag
        tau = dc.assemble_terms(terms)
        ok &= dc.split_terms(tau) == terms and dc.is_complete(tau)
    check('T.6 split(assemble(terms)) == terms, incl. m == 1', bool(ok))
    check('T.6 a non-add tau is a single term',
          dc.split_terms(['power', 'x1']) == [['power', 'x1']])
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
    tau_a = ['add', 'x1', 'x2', 'power', 'x1', 'end']
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


# ── T.9 Beam path ───────────────────────────────────────────────────────────
def test_t9_beam_terms():
    """The beam proposes one extra term per DOF on top of a sampled bag.  Every
    candidate must be a complete, flat-reachable tau within the bag limits,
    never a lone bare leaf, and carry a context that names its own DOF in the
    requested slice order — and it must replay through the update path."""
    print("\nT.9  beam candidates")
    policy = make_term_policy()
    torch.manual_seed(3)
    order = (1, 0)
    cands = dp.beam_term_candidates(policy, order, N_DOF, beam_width=4, n_return=4)
    check('T.9 the beam returns candidates', len(cands) > 0, f"{len(cands)} candidates")
    taus = [tau for _d, tau, _c in cands]
    check('T.9 every candidate is complete', all(dc.is_complete(t) for t in taus))
    check('T.9 every candidate is reachable under the flat grammar',
          all(_flat_reachable(t) for t in taus))
    check('T.9 no candidate exceeds max_terms / max_term_len',
          all(len(dc.split_terms(t)) <= N_TERMS for t in taus)
          and all(len(term) <= TERM_LEN for t in taus for term in dc.split_terms(t)))
    check('T.9 no candidate is a lone bare leaf', all(len(t) > 1 for t in taus))
    check('T.9 each context names the candidate\'s own DOF in the requested order',
          all(c['dof_order'][c['slot']] == d and tuple(c['dof_order']) == order
              for d, _t, c in cands))
    check('T.9 candidates for slot 1 carry exactly one prefix slice',
          all(set(c['prefix_slices']) == ({order[0]} if c['slot'] == 1 else set())
              for _d, _t, c in cands))

    # A beam candidate is a buffer entry like any other: the update must accept
    # it for its own DOF and reject it for the other one.
    d0 = cands[0][0]
    same = [c for c in cands if c[0] == d0]
    if len(same) >= 2:
        buf = [(0.3 + 0.1 * i, tau, [], ctx) for i, (_d, tau, ctx) in enumerate(same)]
        old = copy.deepcopy(policy).eval()
        policy.train()
        _o, _e, ok = dp.jgrpo_terms(policy, old, old, buf, 0.2, dof=d0, n_dof=N_DOF,
                                    eps=0.2, beta=0.01)
        policy.eval()
        check('T.9 beam candidates replay through jgrpo_terms', ok)
        caught = False
        try:
            dp.jgrpo_terms(policy, old, old, buf, 0.2, dof=1 - d0, n_dof=N_DOF,
                           eps=0.2, beta=0.01)
        except ValueError:
            caught = True
        check('T.9 ...and are rejected for the other DOF', caught)


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


def test_directional_leaves():
    """``dleaf`` / ``vleaf``: every tau-walker must agree on their const slots
    and their value, and the default table must be untouched.

    The failure this exists to catch is silent and total.  Each directional
    leaf owns ``N_DOF`` const slots, and *seven* separate functions walk a tau
    handing out those slots in pre-order (``count_total_consts``,
    ``_parse_tree``, ``_build_param_code``, ``evaluate``, ``compile_to_numpy``,
    ``expr_to_sympy_str``, ``expr_to_str``).  If any one of them disagrees by a
    single slot, every constant after that point shifts by one and the
    candidate is evaluated as a different expression than the one that was
    fitted — with no error raised anywhere.
    """
    print("\ndirectional leaves (dleaf / vleaf)")
    try:
        # default table first: enabling the feature must not change it
        dc.configure_grammar(N_DOF)
        base_tokens = list(dc.ALL_TOKENS)
        check('default grammar has no directional leaves',
              dc.DIR_LEAF_SET == set() and 'dleaf' not in dc.ALL_TOKENS)

        dc.configure_grammar(N_DOF, directional_leaves=True)
        check('enabling adds exactly two tokens',
              dc.N_TOKENS == len(base_tokens) + 2
              and set(dc.ALL_TOKENS) - set(base_tokens) == {'dleaf', 'vleaf'},
              f"{dc.ALL_TOKENS}")
        check('both are leaves',
              dc.ARITY['dleaf'] == 0 and dc.ARITY['vleaf'] == 0)
        check('dleaf spans the displacement columns, vleaf the velocities',
              dc.LEAF_CHANNELS['dleaf'] == [2 * d for d in range(N_DOF)]
              and dc.LEAF_CHANNELS['vleaf'] == [2 * d + 1 for d in range(N_DOF)])
        check('each owns N_DOF const slots',
              dc.CONST_SLOTS['dleaf'] == N_DOF == dc.CONST_SLOTS['vleaf'])

        taus = {
            'dleaf':                    ['dleaf'],
            'add(dleaf, vleaf)':        ['add', 'dleaf', 'vleaf', 'end'],
            'power(dleaf)':             ['power', 'dleaf'],
            'add(x1, power(dleaf))':    ['add', 'x1', 'power', 'dleaf', 'end'],
            'mul(dleaf, vleaf)':        ['mul', 'dleaf', 'vleaf', 'end'],
        }
        agree = []
        for name, tau in taus.items():
            n_count = dc.count_total_consts(tau)
            n_parse = dc._parse_tree(tau)[1]
            n_code  = dc._build_param_code(tau)[1]
            agree.append(n_count == n_parse == n_code)
        check('count_total_consts / _parse_tree / _build_param_code agree',
              all(agree), f"{len(agree)} taus")

        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 2 * N_DOF))
        devs = []
        for name, tau in taus.items():
            consts = list(rng.normal(size=dc.count_total_consts(tau)))
            fn = dc.compile_to_numpy(tau, consts)
            a = np.asarray(fn(*[X[:, k] for k in range(2 * N_DOF)]), float)
            b = dc.evaluate(tau, torch.tensor(X, dtype=torch.float64),
                            consts).numpy()
            devs.append(float(np.abs(a - b).max()))
        check('compile_to_numpy and evaluate agree numerically',
              max(devs) < 1e-10, f"max deviation {max(devs):.2e}")

        fn = dc.compile_to_numpy(['dleaf'], [1.0, -1.0])
        v = np.asarray(fn(*[X[:, k] for k in range(2 * N_DOF)]), float)
        check('dleaf(1, -1) is the relative coordinate x1 - x3',
              np.allclose(v, X[:, 0] - X[:, 2]))
        fn = dc.compile_to_numpy(['vleaf'], [1.0, -1.0])
        v = np.asarray(fn(*[X[:, k] for k in range(2 * N_DOF)]), float)
        check('vleaf(1, -1) is the relative velocity x2 - x4',
              np.allclose(v, X[:, 1] - X[:, 3]))

        # VARPRO boundary: bare leaf stays closed-form, powered leaf does not
        m1 = dc._expand_monomials(dc._parse_tree(['dleaf'])[0])
        check('a bare dleaf keeps the closed form (N_DOF monomials)',
              m1 is not None and len(m1) == N_DOF)
        m2 = dc._expand_monomials(dc._parse_tree(['power', 'dleaf'])[0])
        check('power(dleaf) leaves the closed form (nonlinear in the weights)',
              m2 is None)

        # reachability: the varpro term grammar must admit them under a power,
        # which is the entire point of the token
        mask = dp.term_valid_mask(['power'], 1, TERM_LEN,
                                  term_grammar='varpro').numpy()
        allowed = {t for t, m in zip(dc.ALL_TOKENS, mask) if m > 0}
        check("varpro lets a power op take a directional leaf",
              {'dleaf', 'vleaf'} <= allowed, f"allowed: {sorted(allowed)}")
        check('power(dleaf) is a complete term',
              dc.is_complete(['power', 'dleaf']))

        # the policy's output layer must widen with the table
        pol = dp.TermBagPolicy(n_tokens=dc.N_TOKENS, max_terms=N_TERMS,
                               max_term_len=TERM_LEN, n_dof=N_DOF,
                               d_model=32, n_heads=4, ff_dim=64, n_layers=1)
        check('policy vocabulary tracks the widened table',
              pol.out_proj.out_features == dc.N_TOKENS + 1
              and pol.stop_idx == dc.N_TOKENS)
    finally:
        dc.configure_grammar(N_DOF)     # restore for anything downstream


def test_blend():
    """``blend(u) = e^(au)(c1 cos bu + c2 sin bu)`` -- the one transcendental.

    Two failures this exists to catch, both of which happened while it was
    written and both of which are silent.  (1) The polish step's bounds were
    shared across every exponent-like slot, which clamped a blend's DECAY RATE
    to be non-negative and so forbade every decaying oscillation.  (2) The
    integrated-column cache rounds exponent values to 6 decimals, so
    least_squares' default finite-difference step of ~1.5e-8 hit the same cache
    key, the residual came back bitwise identical, the gradient was exactly
    zero, and the polish silently terminated at its starting grid point.
    """
    print("\nblend (the transcendental operator)")
    try:
        dc.configure_grammar(N_DOF)
        base = list(dc.ALL_TOKENS)
        check('blend is absent by default', 'blend' not in base)

        dc.configure_grammar(N_DOF, transcendental=True)
        check('enabling adds exactly one token',
              set(dc.ALL_TOKENS) - set(base) == {'blend'}, f"{dc.ALL_TOKENS}")
        check('blend is unary and owns 4 const slots',
              dc.ARITY['blend'] == 1 and dc.CONST_SLOTS['blend'] == 4)

        taus = {'blend(x1)':          ['blend', 'x1'],
                'add(x1,blend(x2))':  ['add', 'x1', 'blend', 'x2', 'end'],
                'mul(blend(x1),x2)':  ['mul', 'blend', 'x1', 'x2', 'end']}
        ok = all(dc.count_total_consts(t) == dc._parse_tree(t)[1]
                 == dc._build_param_code(t)[1] for t in taus.values())
        check('every walker agrees on blend const slots', ok)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 2 * N_DOF)) * 0.8
        cases = {'exp':  ([1.0, 0.0, 1.0, 0.0], lambda u: np.exp(u)),
                 'cos':  ([1.0, 0.0, 0.0, 1.0], lambda u: np.cos(u)),
                 'sin':  ([0.0, 1.0, 0.0, 1.0], lambda u: np.sin(u)),
                 'damped': ([1.0, 0.0, -0.4, 2.0],
                            lambda u: np.exp(-0.4 * u) * np.cos(2 * u))}
        devs = []
        for _n, (cs, ref) in cases.items():
            consts = cs + [1.0]
            fn = dc.compile_to_numpy(['blend', 'x1'], consts)
            a = np.asarray(fn(*[X[:, k] for k in range(2 * N_DOF)]), float)
            b = dc.evaluate(['blend', 'x1'],
                            torch.tensor(X, dtype=torch.float64), consts).numpy()
            devs.append(max(float(np.abs(a - ref(X[:, 0])).max()),
                            float(np.abs(a - b).max())))
        check('blend reproduces exp, cos, sin and a damped oscillation exactly',
              max(devs) < 1e-12, f"max deviation {max(devs):.2e}")

        m = dc._expand_monomials(dc._parse_tree(['blend', 'x1'])[0])
        check('blend of a variable keeps the closed form (2 monomials, 2 slots)',
              m is not None and len(m) == 2
              and all(len(f['exps']) == 2 for f in m))
        kinds = sorted(f['factors'][0][0] for f in m)
        check('its factors are the cos and sin branches',
              kinds == ['blendcos', 'blendsin'], f"{kinds}")

        # the two bugs, as direct assertions on the helpers
        check("a blend's decay rate is allowed to be negative",
              dc.exp_slot_bound('blend_a')[0] < 0.0,
              f"blend_a bound {dc.exp_slot_bound('blend_a')}")
        check("a power's exponent is NOT allowed negative (singular at 0)",
              dc.exp_slot_bound('power')[0] >= 0.0,
              f"power bound {dc.exp_slot_bound('power')}")
        check('the a-grid is denser near zero than at its ends',
              min(abs(b - a) for a, b in zip(dc.BLEND_A_GRID, dc.BLEND_A_GRID[1:])
                  if abs(a) < 0.6 and abs(b) < 0.6)
              < abs(dc.BLEND_A_GRID[1] - dc.BLEND_A_GRID[0]),
              f"{dc.BLEND_A_GRID}")

        dc.configure_grammar(N_DOF, directional_leaves=True, transcendental=True)
        check('blend of a directional leaf leaves the closed form',
              dc._expand_monomials(dc._parse_tree(['blend', 'dleaf'])[0]) is None)
        mask = dp.term_valid_mask(['blend'], 1, TERM_LEN,
                                  term_grammar='varpro').numpy()
        allowed = {t for t, mv in zip(dc.ALL_TOKENS, mask) if mv > 0}
        check('varpro lets blend take a variable or a directional leaf',
              {'dleaf', 'vleaf', 'x1'} <= allowed, f"{sorted(allowed)}")
        check('a blend of a blend is forbidden',
              'blend' not in allowed)
    finally:
        dc.configure_grammar(N_DOF)


def test_intpower():
    """``intpower(u) = c * u^n`` -- the standard integer power, next to ``power``.

    ``power`` contains u^n only as a special case the fit never lands on: its
    polish moves the exponent off the integer and its wrong-parity amplitude
    comes back small but nonzero, so a Duffing truth printed as
    ``-0.5001*Abs(x)**3.0*sign(x) + 9.1e-6*Abs(x)**3.0``.  ``intpower`` exists
    to give ``-0.5*x**3``, which only holds if its exponent stays an EXACT
    integer through both fit paths: the closed form must not polish it, and the
    nonlinear fallback must pin and snap it.  Both failures are silent -- the
    reward barely moves -- so they are asserted here through the real fitter.
    """
    import contextlib
    import io

    import sympy as sp

    from discover_data import build_truth_system, generate_dataset
    from discover_mdof_sim import get_mdof_system
    from discover_sdof_sim import get_sdof_system

    def load(spec, directional_leaves=False, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            system = build_truth_system(spec, **kw)
        dc.configure_grammar(system.n_dof, system.var_names, directional_leaves)
        _X, _y, raw, ns = generate_dataset(system, device=None)
        dc.set_problem_data(ns, raw, True, None, 0.5)
        return ns

    print("\nintpower (the standard integer power)")
    saved = (dc.NORM_STATS, dc.RAW_TRAJECTORIES, dc.ENERGY_NORMALIZE,
             dc.MAX_TRAJ, dc.W_ACC)
    try:
        dc.configure_grammar(N_DOF)
        check('intpower is in the default table next to power',
              {'intpower', 'power'} <= set(dc.ALL_TOKENS), f"{dc.ALL_TOKENS}")
        check('intpower is unary and owns 2 const slots',
              dc.ARITY['intpower'] == 1 and dc.CONST_SLOTS['intpower'] == 2)

        taus = {'intpower(x1)':           ['intpower', 'x1'],
                'add(x1,intpower(x2))':   ['add', 'x1', 'intpower', 'x2', 'end'],
                'mul(intpower(x1),x3)':   ['mul', 'intpower', 'x1', 'x3', 'end'],
                'add(intpower,power)':    ['add', 'intpower', 'x1',
                                           'power', 'x3', 'end']}
        ok = all(dc.count_total_consts(t) == dc._parse_tree(t)[1]
                 == dc._build_param_code(t)[1] for t in taus.values())
        check('every walker agrees on intpower const slots', ok)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(7, 2 * N_DOF))
        X[0, 0] = -1.5                        # make sure a negative base is seen
        cols = [X[:, k] for k in range(2 * N_DOF)]
        devs = []
        for n in (1, 2, 3):
            consts = [-0.7, float(n), 1.0]
            a = np.asarray(dc.compile_to_numpy(['intpower', 'x1'], consts)(*cols),
                           float)
            b = dc.evaluate(['intpower', 'x1'],
                            torch.tensor(X, dtype=torch.float64), consts).numpy()
            devs.append(max(float(np.abs(a - (-0.7) * X[:, 0] ** n).max()),
                            float(np.abs(a - b).max())))
        check('intpower is c * u**n exactly, odd powers keeping the sign',
              max(devs) < 1e-12, f"max deviation {max(devs):.2e}")

        # A slot value that drifted in an optimiser must still evaluate as an
        # integer power in EVERY walker, or the fit and the score disagree.
        drift = [1.0, 2.7, 1.0]
        a = np.asarray(dc.compile_to_numpy(['intpower', 'x1'], drift)(*cols), float)
        code = dc._build_param_code(['intpower', 'x1'])[0]
        f_par = eval(f"lambda c,{','.join(dc.VARIABLES)}: {code}",
                     {'np': np, '_itp': dc._intpower_np})
        p = np.asarray(f_par(np.array(drift), *cols), float)
        check('a drifted exponent (2.7) evaluates as u**3 in every walker',
              np.allclose(a, X[:, 0] ** 3) and np.allclose(p, X[:, 0] ** 3)
              and dc.expr_to_str(['intpower', 'x1'], drift).endswith('^3'))

        m = dc._expand_monomials(dc._parse_tree(['intpower', 'x1'])[0])
        check('intpower of a variable is ONE closed-form monomial (no parity twin)',
              m is not None and len(m) == 1
              and m[0]['factors'][0][0] == 'intpower')
        check('its grid is the positive integers; power keeps its own',
              dc.exp_slot_grid('intpower') == [1.0, 2.0, 3.0]
              and dc.exp_slot_grid('abspower') == list(dc.POWER_EXP_GRID))

        mask = dp.term_valid_mask(['intpower'], 1, TERM_LEN,
                                  term_grammar='varpro').numpy()
        allowed = {t for t, mv in zip(dc.ALL_TOKENS, mask) if mv > 0}
        check('varpro restricts an intpower child to bare variables',
              allowed == set(dc.VARIABLES), f"{sorted(allowed)}")
        under_int = dp.term_valid_mask(['intpower'], 1, TERM_LEN).numpy()
        under_pow = dp.term_valid_mask(['power'], 1, TERM_LEN).numpy()
        check('no power op may sit directly under another (nesting is degenerate)',
              all(under_int[dc.tok2idx(t)] == 0 and under_pow[dc.tok2idx(t)] == 0
                  for t in ('intpower', 'power')))

        # sympy round trip: plain integer powers come back as intpower, the
        # Abs / sign forms as power, and the value is unchanged
        x1, x2 = dc.SYMS['x1'], dc.SYMS['x2']
        expr = -0.5 * x1 ** 3 + x1 ** 2 * x2 + 2.0 * sp.Abs(x1) ** 1.5
        rt = dc.sympy_to_tau(expr)
        ok = rt is not None
        if ok:
            tau, consts = rt
            ref = sp.lambdify([dc.SYMS[v] for v in dc.VARIABLES], expr,
                              'numpy')(*cols)
            got = dc.compile_to_numpy(tau, consts)(*cols)
            ok = (tau.count('intpower') == 2 and tau.count('power') == 1
                  and np.allclose(got, ref))
        check('sympy_to_tau maps x**n to intpower and Abs(x)**p to power',
              ok, f"{rt[0] if rt else rt}")

        # The promise itself, through the real closed-form fitter.
        load(get_sdof_system('duffing'), n_traj=2, n_pts=600, t_end=12.0)
        tau = ['add', 'x1', 'x2', 'intpower', 'x1', 'end']
        c = dc.optimise_consts_energy(tau, 0)
        r = dc.energy_reward([(tau, c)])
        phys = dc.denormalize_expr(tau, c, 0)
        check('the closed form fits Duffing with the exponent EXACTLY 3',
              c[3] == 3.0 and r > 0.999, f"n = {c[3]!r}, r = {r:.6f}")
        check('...and it prints as a standard power, no Abs / sign',
              'x**3' in phys and 'Abs' not in phys and 'sign' not in phys, phys)

        # ...and through the nonlinear fallback, which intpower(dleaf) takes.
        ns = load(get_mdof_system('cubic_coupled'), directional_leaves=True,
                  n_traj=2, n_pts=1000, t_end=20.0)
        tau = ['add', 'x1', 'x2', 'intpower', 'dleaf', 'end']
        check('intpower(dleaf) leaves the closed form',
              dc._expand_monomials(dc._parse_tree(tau)[0]) is None)
        c = dc.optimise_consts_energy(tau, 0)
        r = dc.energy_reward([(tau, c), None])
        X_std = ns[1]
        ratio = (c[5] / X_std[2]) / (c[4] / X_std[0])  # physical q2 : q1 weight
        check('the nonlinear fit lands on exponent EXACTLY 3 and on q1 - q2',
              c[3] == 3.0 and abs(ratio + 1.0) < 0.01 and r > 0.999,
              f"n = {c[3]!r}, q2/q1 = {ratio:+.4f}, r = {r:.6f}")
    finally:
        dc.configure_grammar(N_DOF)
        (dc.NORM_STATS, dc.RAW_TRAJECTORIES, dc.ENERGY_NORMALIZE,
         dc.MAX_TRAJ, dc.W_ACC) = saved


def test_simulation_reward():
    """``reward='simulation'``: integrate the candidate forward, score its
    displacement against the record.

    Every failure here is silent.  A phase error in the integrator reads as a
    worse equation; a DOF simulated against the wrong partner motion loses its
    coupling term; a NaN from a diverging candidate poisons the batch ranking;
    and a pool that scores by a different reward than the parent configured
    trains on a number nobody sees.  Each is asserted through the real path.
    """
    import contextlib
    import io

    import discover_rollout as ro
    from discover_analysis import simulation_scores
    from discover_data import build_truth_system, generate_dataset
    from discover_mdof_sim import get_mdof_system
    from discover_sdof_sim import get_sdof_system

    def load(spec, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            system = build_truth_system(spec, **kw)
        dc.configure_grammar(system.n_dof, system.var_names)
        _X, _y, raw, _ns = generate_dataset(system, device=None)
        dc.set_problem_data(_ns, raw, True, None, 0.5)
        t = raw[0][0]
        return (system, t, np.stack([r[2] for r in raw]),
                np.stack([r[3] for r in raw]))

    print("\nsimulation reward")
    saved = (dc.NORM_STATS, dc.RAW_TRAJECTORIES, dc.ENERGY_NORMALIZE,
             dc.MAX_TRAJ, dc.W_ACC, dc.REWARD_MODE, dc.SIM_WINDOW,
             dc.SIM_WEIGHTS)
    try:
        # The integrator alone, on the exact truth function.
        spec = get_sdof_system('duffing')
        system, t, states, accs = load(spec, n_traj=2, n_pts=600, t_end=12.0)
        stride, sub = ro.auto_steps(t, states, accs)
        res_auto = ro.rollout_residuals(spec.accel_fns[0], t, states, 0,
                                        stride=stride, substeps=sub)
        res_each = ro.rollout_residuals(spec.accel_fns[0], t, states, 0)
        check('RK4 reproduces the record from the exact truth, auto and '
              'per-sample steps', max(res_auto.max(), res_each.max()) < 1e-4,
              f"stride={stride} substeps={sub}  residual "
              f"{res_auto.max():.1e} / {res_each.max():.1e}")
        res_v = ro.rollout_residuals(spec.accel_fns[0], t, states, 0,
                                     stride=stride, substeps=sub,
                                     weights=(0, 1, 0))
        res_a = ro.rollout_residuals(spec.accel_fns[0], t, states, 0,
                                     stride=stride, substeps=sub,
                                     weights=(0, 0, 1), accs=accs)
        check('...its simulated velocity and acceleration match the record '
              'too', max(res_v.max(), res_a.max()) < 1e-4,
              f"velocity {res_v.max():.1e}  acceleration {res_a.max():.1e}")

        # The weights: three, non-negative, summing to 1 -- and the residual
        # is exactly their weighted sum of the per-channel NRMSEs.
        bad_ok = True
        for bad in ((0.5, 0.5, 0.5), (1.2, -0.2, 0.0), (1.0, 0.0), (0.5, 0.5, None)):
            try:
                dc.set_reward('simulation', sim_weights=bad)
                bad_ok = False
            except ValueError:
                pass
        check('sim_weights that are not three non-negatives summing to 1 '
              'are refused', bad_ok)
        def lin(S):                                    # the linear model
            return -1.0 * S[0] - 0.3 * S[1]
        parts = [ro.rollout_residuals(lin, t, states, 0, stride=stride,
                                      substeps=sub, weights=w, accs=accs)
                 for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        mix = ro.rollout_residuals(lin, t, states, 0, stride=stride,
                                   substeps=sub, weights=(0.5, 0.3, 0.2),
                                   accs=accs)
        check('the residual is the weighted sum of the channel NRMSEs',
              np.allclose(mix, 0.5 * parts[0] + 0.3 * parts[1] + 0.2 * parts[2],
                          rtol=0, atol=1e-12),
              f"disp {parts[0].mean():.3f}  vel {parts[1].mean():.3f}  "
              f"acc {parts[2].mean():.3f}")

        # Through the engine: the fitted truth against a missing cubic.
        truth, wrong = (['add', 'x1', 'x2', 'intpower', 'x1', 'end'],
                        ['add', 'x1', 'x2', 'end'])
        ct, cw = dc.optimise_consts_energy(truth, 0), dc.optimise_consts_energy(wrong, 0)
        dc.set_reward('simulation')
        r_t = dc.candidate_reward([(truth, ct)])
        r_w = dc.candidate_reward([(wrong, cw)])
        check('the fitted Duffing truth simulates to r ~ 1, the linear model '
              'clearly below', r_t > 0.999 and r_w < r_t - 0.05,
              f"truth {r_t:.6f}  linear {r_w:.4f}")
        dc.set_reward('simulation', sim_window=2.0)
        r_tw = dc.candidate_reward([(truth, ct)])
        r_ww = dc.candidate_reward([(wrong, cw)])
        check('2 s windows keep the truth at ~1 and forgive part of the '
              'linear model\'s phase drift', r_tw > 0.999 and r_w < r_ww < r_tw,
              f"truth {r_tw:.6f}  linear {r_ww:.4f}")

        # The mode switch reaches the worker's scoring path.
        dc.set_reward('energy')
        ok_e = abs(dc.candidate_reward([(truth, ct)])
                   - dc.energy_reward([(truth, ct)])) < 1e-12
        dc.set_reward('simulation')
        out = dc.energy_worker((0, truth, None, None))
        ok_s = (out is not None and
                abs(out[1] - dc.simulation_reward([(truth, out[3])])) < 1e-12)
        check('candidate_reward and the pool worker follow set_reward',
              ok_e and ok_s)
        dc.set_reward('simulation', sim_weights=(0.4, 0.3, 0.3))
        r_tm = dc.candidate_reward([(truth, ct)])
        r_wm = dc.candidate_reward([(wrong, cw)])
        check('with all three channels weighted the truth stays at ~1',
              r_tm > 0.999 and r_wm < r_tm - 0.05,
              f"truth {r_tm:.6f}  linear {r_wm:.4f}")
        r_an3, _rows = simulation_scores(system, [dc.denormalize_expr(truth, ct, 0)],
                                         weights=(0.4, 0.3, 0.3))
        check('...and discover_score reproduces it', abs(r_an3 - r_tm) < 1e-3,
              f"engine {r_tm:.6f}  printed {r_an3:.6f}")
        dc.set_reward('simulation')                    # back to displacement
        try:
            dc.set_reward('bogus')
            ok = False
        except ValueError:
            ok = True
        check('an unknown reward name is refused', ok)

        # A diverging equation: a large finite residual, not a NaN.
        r_bad = dc.candidate_reward([(wrong, [3.0, 2.0])])
        check('a diverging equation scores small but finite',
              np.isfinite(r_bad) and 0.0 < r_bad < 0.3, f"r = {r_bad:.4f}")

        # The scoring script reproduces the engine from the printed equation.
        phys = dc.denormalize_expr(truth, ct, 0)
        r_an, _rows = simulation_scores(system, [phys])
        check('discover_score reproduces the trained simulation reward',
              abs(r_an - r_t) < 1e-3, f"engine {r_t:.6f}  printed {r_an:.6f}")

        # Coupled: each DOF against the partner's MEASURED motion.
        spec = get_mdof_system('coupled_duffing')
        system, t, states, accs = load(spec, n_traj=2, n_pts=1000, t_end=10.0)
        stride, sub = ro.auto_steps(t, states, accs)
        worst = max(ro.rollout_residuals(spec.accel_fns[d], t, states, d,
                                         stride=stride, substeps=sub).max()
                    for d in range(2))
        check('each DOF of coupled Duffing reproduces the record against the '
              'measured partner', worst < 1e-3, f"worst residual {worst:.1e}")
        tau = system.truth_taus[0]
        c = dc.optimise_consts_energy(tau, 0)
        r_c = dc.candidate_reward([(tau, c), None])
        # the coupling term is the bare x3 (= q2) leaf; flip its coefficient
        slot = sum(dc.CONST_SLOTS.get(tk, 0) for tk in tau[:tau.index('x3')])
        flipped = list(c)
        flipped[slot] = -flipped[slot]
        r_f = dc.candidate_reward([(tau, flipped), None])
        check('...and a flipped coupling sign is punished',
              r_c > 0.999 and r_f < r_c - 0.05,
              f"truth {r_c:.6f}  flipped {r_f:.4f}")

        # A record trimmed at the front starts at t > 0; the full-record
        # horizon (its duration) must still keep every sample.
        tt = 0.5 + 0.01 * np.arange(100)
        traj = (tt, np.zeros(2), np.zeros((2, 100)))
        n_full = len(dc._slice_traj(traj, tt[-1] - tt[0])[0])
        n_half = len(dc._slice_traj(traj, 0.5)[0])
        check('the horizon is measured from the record start, not from t = 0',
              n_full == 100 and n_half == 51, f"{n_full}, {n_half}")
    finally:
        dc.configure_grammar(N_DOF)
        (dc.NORM_STATS, dc.RAW_TRAJECTORIES, dc.ENERGY_NORMALIZE,
         dc.MAX_TRAJ, dc.W_ACC) = saved[:5]
        dc.set_reward(*saved[5:])
        dc._SIM_DATA.clear()


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
    test_t9_beam_terms()
    test_jgrpo_terms()
    test_term_pe_band()
    test_directional_leaves()
    test_blend()
    test_intpower()
    test_simulation_reward()

    print(f"\n{'=' * 60}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    return 1 if _FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
