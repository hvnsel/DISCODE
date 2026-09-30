"""
discover_edit_test.py
=====================

PROTOTYPE ADD-ON.  Correctness invariants for the move-based PPO agent
(:mod:`discover_edit_env`, :mod:`discover_edit_policy`,
:mod:`discover_edit_train`).  Run before anything long::

    python discover_edit_test.py

As in :mod:`discover_policy_test`, every check guards a failure that would be
silent: a mask that lets an illegal move through only shows up as a crash deep
in a rollout, a reward that does not telescope trains towards the wrong
objective, and a slot-order leak in the policy trains happily on noise.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

import discover_core as dc
import discover_edit_env as ee
import discover_edit_train as et
from discover_data import build_truth_system, generate_dataset
from discover_edit_policy import EditPolicy, masked_dist
from discover_mdof_sim import get_mdof_system
from discover_sdof_sim import get_sdof_system


_PASS, _FAIL = [], []


def check(name, ok, detail=''):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ''))
    return ok


def load(spec_fn, key, **kw):
    """Simulate a registry system (quietly) and register it as the problem."""
    with contextlib.redirect_stdout(io.StringIO()):
        system = build_truth_system(spec_fn(key), **kw)
    dc.configure_grammar(system.n_dof, system.var_names)
    _X, _y, raw, ns = generate_dataset(system, device=None)
    dc.set_problem_data(ns, raw, True, None, 0.5)
    return system


def _flat_reachable(tau, budget=80):
    return all(dc.get_valid_tokens(tau[:i], i, budget)[dc.tok2idx(tk)]
               for i, tk in enumerate(tau))


def act(spec, *a):
    return spec.actions.index(tuple(a))


# ── moves and masks ─────────────────────────────────────────────────────────
def test_moves():
    print("\nmoves and legality")
    dc.configure_grammar(2)
    spec = ee.EditSpec()
    rng = np.random.default_rng(0)
    agree = valid = closed = flat = True
    n_states = 0
    for _ep in range(60):
        terms = ()
        for _t in range(spec.max_steps):
            mask = spec.legal_mask(terms)
            for a in range(spec.n_actions):
                new = spec.apply(terms, a)
                agree &= bool(mask[a]) == (new is not None)
                if new is None or a == spec.STOP:
                    continue
                valid &= (len(new) <= spec.max_terms
                          and all(spec.term_ok(t) for t in new)
                          and len({ee._signature(t) for t in new}) == len(new))
                closed &= all(ee._separable(t) for t in new)
                tau = dc.assemble_terms([list(t) for t in new])
                if len(new) > 1 or (new and new[0][0] in dc._OP_TOKENS):
                    flat &= dc.is_complete(tau) and _flat_reachable(tau)
                n_states += 1
            a = int(rng.choice(np.flatnonzero(mask)))
            if a == spec.STOP:
                break
            terms = spec.apply(terms, a)
    check('the mask is exactly "apply() succeeds"', agree, f"{n_states} successor states")
    check('every legal successor is within budget, valid and duplicate-free', valid)
    check('closed-form grammar: every term takes the fast fit', closed)
    check('every successor is an expression the flat grammar could write', flat)

    s = spec
    x = ('x1',)
    check('STOP is illegal on the empty equation', not s.legal_mask(())[s.STOP])
    check('ADD(op, v) appends op(v)',
          s.apply((x,), act(s, 'addop', 'intpower', 'x3')) == (x, ('intpower', 'x3')))
    check('WRAP powers the LAST factor of a product',
          s.apply((('mul', 'x1', 'x3', 'end'),), act(s, 'wrap', 0, 'intpower'))
          == (('mul', 'x1', 'intpower', 'x3', 'end'),))
    check('MUL flattens into an existing product (no mul under mul)',
          s.apply((('mul', 'x1', 'x3', 'end'),), act(s, 'mul', 0, 'x2'))
          == (('mul', 'x1', 'x3', 'x2', 'end'),))
    check('no power of a power',
          s.apply((('intpower', 'x1'),), act(s, 'wrap', 0, 'power')) is None)
    check('a const is never wrapped or multiplied',
          s.apply((('const',),), act(s, 'wrap', 0, 'intpower')) is None
          and s.apply((('const',),), act(s, 'mul', 0, 'x1')) is None)
    check('x*y is a duplicate of y*x',
          s.apply((('mul', 'x1', 'x3', 'end'), ('x3',)), act(s, 'mul', 1, 'x1')) is None)
    check('a product of two two-amplitude powers is refused (nonlinear fit)',
          s.apply((('mul', 'power', 'x1', 'x3', 'end'),), act(s, 'wrap', 0, 'power')) is None)
    full = tuple((v,) for v in ('x1', 'x2', 'x3', 'x4', 'const')) + (('intpower', 'x1'),)
    check('nothing can be added to a full bag',
          not any(s.legal_mask(full)[1:1 + s.n_global]))

    dc.configure_grammar(2)
    op = ee.EditSpec(closed_form=False)
    wa = op.apply((('x1',), ('x3',)), op.actions.index(('wrapall', 'intpower')))
    check('open grammar: WRAP_ALL wraps the whole equation',
          wa == (('intpower', 'add', 'x1', 'x3', 'end'),), f"{wa}")
    check('closed grammar has no WRAP_ALL',
          not any(a[0] == 'wrapall' for a in spec.actions))


def test_canonical():
    print("\ncanonical bags")
    dc.configure_grammar(2)
    bag = (('x1',), ('intpower', 'x3'), ('mul', 'x1', 'x2', 'end'))
    check('canonical() ignores slot order',
          ee.canonical(bag) == ee.canonical(bag[::-1]) == ee.canonical(bag[1:] + bag[:1]))
    check('bag_from_tau inverts assemble_terms',
          ee.bag_from_tau(dc.assemble_terms([list(t) for t in bag])) == bag)


# ── scoring and reward ──────────────────────────────────────────────────────
def test_scoring():
    print("\nscoring and reward")
    system = load(get_sdof_system, 'duffing', n_traj=2, n_pts=600, t_end=12.0)
    spec = ee.EditSpec()
    sc = ee.Scorer(spec)

    rng = np.random.default_rng(1)
    ok = True
    for _ep in range(8):
        terms, total = (), 0.0
        start = sc.value(0, terms)
        for _t in range(spec.max_steps):
            a = int(rng.choice(np.flatnonzero(spec.legal_mask(terms))))
            if a == spec.STOP:
                break
            new = spec.apply(terms, a)
            total += sc.value(0, new) - sc.value(0, terms)
            terms = new
        ok &= abs(total - (sc.value(0, terms) - start)) < 1e-9
    check('rewards telescope: return == final score - start score', ok)

    truth = ee.bag_from_tau(system.truth_taus[0])
    rivals = {'power instead of intpower': (('x1',), ('x2',), ('power', 'x1')),
              '+ intpower(xdot)': truth + (('intpower', 'x2'),),
              '+ power(xdot)': truth + (('power', 'x2'),),
              '+ const': truth + (('const',),),
              'linear only': (('x1',), ('x2',))}
    t_score = sc.value(0, truth)
    worst = {k: t_score - sc.value(0, b) for k, b in rivals.items()}
    check('at the default lam the truth outscores its variants and junk',
          all(v > 0 for v in worst.values()),
          ', '.join(f"{k} {v:+.2f}" for k, v in worst.items()))

    n = sc.n_fits
    sc.get(0, truth[::-1])
    check('the cache is keyed on the bag, not the slot order', sc.n_fits == n)

    f_truth = sc.get(0, truth).feats
    f_lin = sc.get(0, (('x1',), ('x2',))).feats
    check('residual features vanish at the truth', float(np.max(f_truth)) < 0.01,
          f"max {np.max(f_truth):.4f}")
    # order: x, xdot, x^2, xdot^2, x^3, xdot^3, x*xdot, x^2*xdot, xdot^2*x
    check('...and point at the missing x^3 on the linear model',
          int(np.argmax(f_lin)) == 4, f"{np.round(f_lin, 3)}")


def test_truth_reachable():
    print("\ntruths reachable by the moves")
    dc.configure_grammar(1)
    spec = ee.EditSpec()
    seq = [('add', 'x1'), ('add', 'x2'), ('addop', 'intpower', 'x1'), ('mul', 2, 'x2')]
    terms = ()
    for a in seq:
        terms = spec.apply(terms, spec.actions.index(a))
    vdp = ee.bag_from_tau(get_sdof_system('vanderpol').truth_taus[0])
    check("van der Pol's truth is 4 moves from the empty equation",
          ee.canonical(terms) == ee.canonical(vdp), f"{terms}")
    ok, info = True, []
    for fn, key in ((get_sdof_system, 'duffing'), (get_sdof_system, 'vanderpol'),
                    (get_mdof_system, 'coupled_duffing'),
                    (get_mdof_system, 'duffing_chain3'),
                    (get_mdof_system, 'cubic_coupled'),
                    (get_mdof_system, 'coupled_beats')):
        sysd = fn(key)
        dc.configure_grammar(len(sysd.accel_fns), sysd.var_names)
        sp = ee.EditSpec()
        for tau in sysd.truth_taus:
            r = sp.reachable(ee.bag_from_tau(tau))
            ok &= r
            info.append(f"{key}:{'y' if r else 'N'}")
    check('every registry truth is a bag the moves can build', ok, ' '.join(info))


# ── policy ──────────────────────────────────────────────────────────────────
def _policy_and_obs():
    dc.configure_grammar(2)
    spec = ee.EditSpec()
    torch.manual_seed(0)
    pol = EditPolicy.for_spec(spec, dc.N_TOKENS, 9, d_model=32, n_heads=4,
                              n_layers=2)
    B, K, L = 3, spec.max_terms, spec.max_term_len
    EMPTY, PAD = dc.N_TOKENS, dc.N_TOKENS + 1
    tokens = torch.full((B, K, L), PAD, dtype=torch.long)
    tokens[:, :, 0] = EMPTY
    mask = torch.zeros(B, K, dtype=torch.bool)
    bags = [(('x1',), ('intpower', 'x3'), ('mul', 'x1', 'x2', 'end')),
            (('x2',),), ()]
    for b, bag in enumerate(bags):
        for j, t in enumerate(bag):
            tokens[b, j, :len(t)] = torch.tensor([dc.tok2idx(x) for x in t])
            mask[b, j] = True
    glob = torch.randn(B, 9)
    dof = torch.tensor([0, 1, 0])
    return spec, pol, (tokens, mask, glob, dof), bags


def test_policy():
    print("\npolicy")
    spec, pol, obs, bags = _policy_and_obs()
    logits, value = pol(*obs)
    check('one logit per move, one value per state',
          tuple(logits.shape) == (3, spec.n_actions) and tuple(value.shape) == (3,))

    # permutation equivariance over term slots
    tokens, mask, glob, dof = obs
    perm = torch.tensor([2, 0, 1, 3, 4, 5])
    logits2, value2 = pol(tokens[:, perm], mask[:, perm], glob, dof)
    G, P, K = 1 + spec.n_global, spec.per_term, spec.max_terms
    t1 = logits[:, G:].reshape(3, K, P)
    t2 = logits2[:, G:].reshape(3, K, P)
    check('permuting term slots permutes their move logits',
          torch.allclose(t1[:, perm], t2, atol=1e-5))
    check('...and leaves the global moves and the value unchanged',
          torch.allclose(logits[:, :G], logits2[:, :G], atol=1e-5)
          and torch.allclose(value, value2, atol=1e-5))

    # an EMPTY slot is invisible: garbage in it moves nothing that matters
    tok3 = tokens.clone()
    tok3[0, 5, :3] = torch.tensor([dc.tok2idx('mul'), dc.tok2idx('x1'),
                                   dc.tok2idx('x2')])
    logits3, value3 = pol(tok3, mask, glob, dof)
    check('an empty slot does not leak into the other logits or the value',
          torch.allclose(logits[:, :G + 5 * P], logits3[:, :G + 5 * P], atol=1e-5)
          and torch.allclose(value, value3, atol=1e-5))

    legal = torch.zeros(3, spec.n_actions, dtype=torch.bool)
    for b, bag in enumerate(bags):
        legal[b] = torch.as_tensor(spec.legal_mask(bag))
    d = masked_dist(logits, legal)
    draws = d.sample((500,))
    check('masked sampling never draws an illegal move',
          bool(legal.gather(1, draws.T).all()))
    check('illegal moves carry no probability',
          float(d.probs.detach()[~legal].max()) < 1e-12
          and bool(torch.isfinite(d.entropy().detach()).all()))


def test_gae():
    print("\nadvantages")
    adv, ret = et.gae([1.0, 2.0], [0.5, 0.25], lam=0.5)
    check('GAE matches a hand computation',
          np.allclose(adv, [1.625, 1.75]) and np.allclose(ret, [2.125, 2.0]))
    _a, ret1 = et.gae([1.0, 2.0], [0.5, 0.25], lam=1.0)
    check('lambda = 1 gives the Monte-Carlo return-to-go', np.allclose(ret1, [3.0, 2.0]))


def test_per_dof_advantages():
    print("\nper-DOF advantage normalisation")
    adv = torch.tensor([6.0, 7.0, 8.0, 0.01, 0.02, 0.03])
    dof = torch.tensor([1, 1, 1, 0, 0, 0])
    out = et.normalise_per_dof(adv, dof)
    ok = all(abs(float(out[dof == d].mean())) < 1e-6
             and abs(float(out[dof == d].std()) - 1.0) < 1e-4 for d in (0, 1))
    check("each DOF's advantages are standardised on their own", ok,
          f"{out.numpy().round(3)}")
    check("...so the unsolved DOF keeps a full-size gradient",
          abs(float(out[3]) - float(out[0])) < 1e-4)


def test_restarts():
    print("\nrestarts")
    load(get_sdof_system, 'duffing', n_traj=2, n_pts=600, t_end=12.0)
    spec = ee.EditSpec()
    sc = ee.Scorer(spec)
    torch.manual_seed(0)
    pol = EditPolicy.for_spec(spec, dc.N_TOKENS, ee.n_global_features(True),
                              d_model=32, n_layers=1)
    start = (('x1',), ('x2',))
    eps, finals, _v = et.rollout(pol, spec, sc, [0, 0], starts=[start, ()])
    first = [e[0]['obs'][1].sum() for e in eps]
    check('an episode can start from a given equation, the others from empty',
          int(first[0]) == len(start) and int(first[1]) == 0, f"{first}")
    check('...and every episode still ends on a legal equation',
          all(spec.reachable(f) for f in finals))


def test_elite_buffer():
    print("\nself-imitation buffer")

    class _Scores:
        def __init__(self, s): self.s = s
        def value(self, d, bag): return self.s[ee.canonical(bag)]

    def ep(rewards):
        return [{'obs': tuple(np.zeros(2) for _ in range(5)), 'action': k,
                 'reward': r} for k, r in enumerate(rewards)]

    A, C = (('x1',),), (('x1',), ('x2',))
    sc = _Scores({ee.canonical(A): 5.0, ee.canonical(C): 9.0})
    eb = et.EliteBuffer(per_dof=2)
    eb.add([ep([1, 2]), ep([0.5, 0.5, 2]), ep([3, 3, 3]), ep([1, -1])],
           [A, A, C, ()], [0, 0, 0, 0], sc)
    kept = {k[1]: v for k, v in eb.items.items()}
    check('the same equation keeps its shorter path; an empty ending is dropped',
          set(kept) == {ee.canonical(A), ee.canonical(C)}
          and len(kept[ee.canonical(A)][1]) == 2)
    check('each move stores its return-to-go',
          [t[2] for t in kept[ee.canonical(C)][1]] == [9.0, 6.0, 3.0])
    eb1 = et.EliteBuffer(per_dof=1)
    eb1.add([ep([1, 2]), ep([3, 3, 3])], [A, C], [0, 0], sc)
    check('only the best episodes per DOF are kept',
          [k[1] for k in eb1.items] == [ee.canonical(C)])


def test_smoke_training():
    print("\ntraining smoke test")
    with contextlib.redirect_stdout(io.StringIO()):
        system = build_truth_system(get_sdof_system('duffing'), n_traj=2,
                                    n_pts=600, t_end=12.0)
        out = et.DISCOVER_EDIT_TRAIN(system, n_iters=2, n_envs=8, use_pool=False,
                                     d_model=32, n_layers=1)
    check('two PPO iterations run end to end', len(out['history']) == 2)
    check('...and report a best and a greedy equation for the DOF',
          out['best'][0] is not None and out['greedy'][0]['expr'] is not None,
          out['best'][0]['expr'] if out['best'][0] else '')


def main():
    test_moves()
    test_canonical()
    test_scoring()
    test_truth_reachable()
    test_policy()
    test_gae()
    test_elite_buffer()
    test_per_dof_advantages()
    test_restarts()
    test_smoke_training()
    print(f"\n{'=' * 60}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    return 1 if _FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
