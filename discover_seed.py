"""
discover_seed.py
================
Sparse regression over every monomial of the states up to a given degree, and
the warm start it gives the trainer.

With integer powers up to 3, every equation the search can write without
directional leaves is a weighted sum of a fixed list of monomials -- 34 for two
DOFs, plus a constant.  Sequentially thresholded least squares on that list
(SINDy-style: fit, drop the small terms, refit) finds the best equation of
every size directly.  :mod:`discover_diagnose` uses it to tell the data, the
setup and the search apart; :func:`seed_entries` uses it to start training
from those equations instead of from nothing (``DISCOVER_TRAIN(seed_list=
True)``): each is written as a term bag, scored with the run's own reward --
its constants fitted exactly as every sampled candidate's are -- and put in
its DOF's replay buffer before the first epoch, so the policy learns from
them from the first update on.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

SEED_DEGREE = 3           # monomials up to this total degree


def monomials(n_vars, max_degree):
    """Exponent tuples of every monomial up to ``max_degree``, constant first."""
    out = [(0,) * n_vars]
    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), deg):
            e = [0] * n_vars
            for i in combo:
                e[i] += 1
            out.append(tuple(e))
    return out


def theta(Z, exps):
    """Library matrix: one column per monomial of the columns of ``Z``."""
    return np.stack([np.prod(Z ** np.array(e)[None, :], axis=1) for e in exps],
                    axis=1)


def stlsq_path(Th, y, max_terms, n_lam=60):
    """Sequentially thresholded least squares over a sweep of thresholds:
    ``{n_terms: coef}``, the best fit found for each equation size, plus
    ``'dense'``.  Columns are scaled to unit RMS so one threshold fits all."""
    norm = np.sqrt(np.mean(Th ** 2, axis=0)) + 1e-300
    A = Th / norm
    dense = np.linalg.lstsq(A, y, rcond=None)[0]
    best = {'dense': (dense / norm, float(np.sum((A @ dense - y) ** 2)))}
    for lam in np.geomspace(1e-4, 1.0, n_lam) * float(np.std(y)):
        xi = dense.copy()
        for _ in range(20):
            small = np.abs(xi) < lam
            xi[small] = 0.0
            big = ~small
            if not big.any():
                break
            new = np.zeros_like(xi)
            new[big] = np.linalg.lstsq(A[:, big], y, rcond=None)[0]
            if np.array_equal(np.abs(new) < lam, small):
                xi = new
                break
            xi = new
        k = int(np.count_nonzero(xi))
        if 1 <= k <= max_terms:
            sse = float(np.sum((A @ xi - y) ** 2))
            if k not in best or sse < best[k][1]:
                best[k] = (xi / norm, sse)
    return best


def monomial_term(e):
    """The grammar's token sequence for the monomial with exponents ``e`` over
    ``x1 .. x{2N}``: a bare variable, ``intpower`` of one (its exponent is
    fitted on the integer grid), or their ``mul``; ``const`` for the constant."""
    factors = [[f'x{i + 1}'] if p == 1 else ['intpower', f'x{i + 1}']
               for i, p in enumerate(e) if p]
    if not factors:
        return ['const']
    if len(factors) == 1:
        return factors[0]
    return ['mul'] + [tok for f in factors for tok in f] + ['end']


def _bag_ok(terms, max_terms, max_term_len, term_grammar):
    """Could the term-bag policy have written these terms?  Every token must be
    legal where it stands (:func:`discover_policy.term_valid_mask`), and a
    one-term bag may not be a bare leaf (:func:`discover_policy.stop_allowed`)."""
    import discover_core as dc
    import discover_policy as dp
    if not terms or len(terms) > max_terms:
        return False
    if len(terms) == 1 and len(terms[0]) == 1:
        return False
    for term in terms:
        if len(term) > max_term_len:
            return False
        for k, tok in enumerate(term):
            if tok not in dc.ALL_TOKENS:
                return False
            mask = dp.term_valid_mask(term[:k], k, max_term_len, term_grammar)
            if float(mask[dc.tok2idx(tok)]) <= 0.0:
                return False
    return True


def seed_bags(system, max_terms, max_term_len, term_grammar='free',
              degree=SEED_DEGREE):
    """Per DOF, the distinct term bags of the best sparse fit of each size, as
    flat token sequences: ``[[tau, ...] per DOF]``.  Fitted to the
    acceleration ``system`` carries, on its displacement and velocity."""
    import discover_core as dc
    N, P = system.n_dof, system.n_trials
    X = np.concatenate([np.stack([c for d in range(N)
                                  for c in (system.disp[:, d, p],
                                            system.vel[:, d, p])], axis=1)
                        for p in range(P)])                       # (rows, 2N)
    exps = monomials(2 * N, degree)
    scale = X.std(axis=0) + 1e-300
    Th = theta(X / scale, exps)
    out = []
    for d in range(N):
        y = np.concatenate([system.acc[:, d, p] for p in range(P)])
        ys = float(np.std(y)) + 1e-300
        seen, taus = set(), []
        for k, (xi, _sse) in sorted(stlsq_path(Th, y / ys, max_terms).items(),
                                    key=lambda z: (z[0] == 'dense', z[0])):
            if k == 'dense':
                continue
            terms = [monomial_term(exps[j]) for j in np.flatnonzero(xi)]
            key = tuple(sorted(tuple(t) for t in terms))
            if key in seen or not _bag_ok(terms, max_terms, max_term_len,
                                          term_grammar):
                continue
            seen.add(key)
            taus.append(dc.assemble_terms(terms))
        out.append(taus)
    return out


def seed_entries(system, max_terms, max_term_len, term_grammar, horizon,
                 pool=None):
    """Buffer entries ``(reward, tau, consts, context)`` per DOF for
    :func:`seed_bags`' equations, scored by the configured reward exactly as
    sampled candidates are (:func:`discover_core.energy_worker`: constants
    fitted to the DOF's own balance, then the reward), each as if sampled
    first in its row.  Sorted best first."""
    import discover_core as dc
    import discover_policy as dp
    N = system.n_dof
    bags = seed_bags(system, max_terms, max_term_len, term_grammar)
    tasks = [(d, tau, None, horizon) for d in range(N) for tau in bags[d]]
    results = (list(pool.map(dc.energy_worker, tasks)) if pool is not None
               else [dc.energy_worker(t) for t in tasks])
    out = [[] for _ in range(N)]
    for res in results:
        if res is None:
            continue
        d, r, tau, consts = res[0], res[1], res[2], res[3]
        if not np.isfinite(r) or r <= 0.0:
            continue
        order = [d] + [j for j in range(N) if j != d]
        ctx = dp.make_context(order, 0, [tau])
        ctx['scored_with'] = {'modes': {'alone': float(r)},
                              'partners': {'alone': {}}, 'tuned': False}
        out[d].append((float(r), list(tau), list(consts), ctx))
    for entries in out:
        entries.sort(key=lambda e: e[0], reverse=True)
    return out
