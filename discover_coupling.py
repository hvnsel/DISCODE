"""
discover_coupling.py
====================

Reciprocity and Newton's third law as training options:
``DISCOVER_TRAIN(coupling='reciprocal' | 'newton3')``.  The reasoning is in
``notes_coupling_constraints.md``.

Under a coupling level the policy writes one slice per DOF and one more.  A
DOF's slice holds its LOCAL terms, in its own q and qd only; the last slice
holds the coupling as an ENERGY (:func:`discover_core.coupling_mask`).  An
energy term with no velocity in it is part of a potential V, one with a
velocity part of a dissipation function R, and the coupling force on DOF r is

    m_r a_r = -m_0 (dV/dq_r + dR/dqdot_r)        (V, R per unit mass of DOF 0)

Every coupling force is then a derivative of one function, for any number of
DOFs, so the mass-weighted coupling Jacobians are symmetric: m_i da_i/dq_j =
m_j da_j/dq_i is the symmetry of d2V/dq_i dq_j, and the same for R in the
velocities.  Under 'newton3' the energy depends on the differences q_a - q_b
alone (the relative leaves of ``dc.configure_grammar``), so a common shift of
every DOF leaves it unchanged and the coupling forces sum to zero.

For each sampled row :func:`system_worker` keeps the terms the closed-form fit
can complete, fits every constant of the system at once -- the local terms',
the energy's and the mass ratios m_r/m_0 -- writes each DOF's equation out
(its local terms, then the forces the energy exerts on it) and scores the
DOFs together.

An energy term's exponents reach one above the force caps, since a
differentiated factor's exponent drops by one, and are limited per monomial
so that every force stays in the grammar: a monomial's only differentiated
factor (a q factor of V, a qd factor of R) has exponent >= 1; with two or
more, each is in [1, 3]; an R monomial's q factors are not differentiated and
keep exponents <= 3.  An R monomial with a q factor must not push at rest, or
the force it modulates would have an asymmetric stiffness there, so its only
qd factor has exponent >= 2, and a term with such a monomial whose qd factor
is a bare variable, or that has a ``blend`` qd factor, is dropped.

Everything is in the normalised units the rest of DISCOVER uses: the energy
is written in the normalised leaves z, and a force carries dz/dq of each leaf
it is differentiated in and the ratio of the equations' output scales.
"""

from __future__ import annotations

import numpy as np

import discover_core as dc

LEVELS = ('free', 'reciprocal', 'newton3')

# Ridge of the joint fit (``dc._varpro``): each term's contribution costs
# RIDGE times its squared size, relative to the fit's rows.  Two near-identical
# terms -- the cube of (q1 - q2) and its ``power`` with the exponent tuned to
# 2.9999 -- otherwise take enormous opposite coefficients that cancel, which
# no rounded copy of the equation survives; a term of normal size pays ~1e-6
# of the fit.
RIDGE = 1e-6

_POWER = (dc.INTPOWER_OP, 'abspower', 'sgnpower')    # factors with an exponent
_EKIND = {dc.INTPOWER_OP: 'eintpower', 'abspower': 'epower',
          'sgnpower': 'epower'}                      # their energy slot kinds


def _dz(c, r):
    """dz_c/dx_r: how normalised leaf ``c`` moves with DOF ``r``'s own state
    of the same kind (q for a q leaf, qd for a qd leaf)."""
    N = dc.N_DOF
    if c < 2 * N:
        return 1.0 / float(dc.NORM_STATS[1][c]) if c // 2 == r else 0.0
    a, b = dc.PAIRS[(c - 2 * N) // 2]
    s = 1.0 / float(dc.PAIR_STD[c - 2 * N])
    return s if r == a else -s if r == b else 0.0


def _dofs(c):
    """The DOFs normalised leaf ``c`` depends on."""
    N = dc.N_DOF
    return {c // 2} if c < 2 * N else set(dc.PAIRS[(c - 2 * N) // 2])


def energy_term(term):
    """``(plan, vel, limits, dofs)`` for one energy term: its closed-form plan
    (``dc._linear_plan``), whether it is an R term, the exponent limits of its
    slots and the DOFs it acts on; or None when it is dropped (see the module
    docstring) or exerts no force."""
    plan = dc._linear_plan(term)
    if plan is None:
        return None
    vel = int(any(f[1] % 2 for m in plan[0] for f in m['factors']))
    limits, dofs = {}, set()
    for m in plan[0]:
        diff = [f for f in m['factors'] if f[1] % 2 == vel]
        mod = len(diff) < len(m['factors'])
        if mod and ((len(diff) == 1 and diff[0][0] == 'var')
                    or any(f[0] not in _POWER + ('var',) for f in diff)):
            return None
        for f in m['factors']:
            d = f[1] % 2 == vel
            if d:
                dofs |= _dofs(f[1])
            if f[0] not in _POWER:
                continue
            cap = dc.INTPOWER_MAX if f[0] == dc.INTPOWER_OP else dc.MAX_EXP
            lo, hi = ((2.0 if mod else 1.0, np.inf) if d and len(diff) == 1
                      else (1.0, cap) if d else (-np.inf, cap))
            a, z = limits.get(f[2], (-np.inf, np.inf))
            limits[f[2]] = (max(a, lo), min(z, hi))
    return (plan, vel, limits, dofs) if dofs else None


def _factor(f, ev, deriv):
    """Monomial factor ``f`` -- or, with ``deriv``, its derivative in its own
    leaf -- as ``(tokens, consts, scale)``: a grammar factor (no tokens for a
    constant) times ``scale``.  Exponents are read with the energy caps."""
    v = dc.VARIABLES[f[1]]
    if f[0] == 'var':
        return ([], [], 1.0) if deriv else ([v], [1.0], 1.0)
    if f[0] == dc.INTPOWER_OP:
        n = dc._snap_int_power(ev[f[2]], dc.INTPOWER_MAX + 1)
        if not deriv:
            return [dc.INTPOWER_OP, v], [1.0, float(n), 1.0], 1.0
        if n == 1:
            return [], [], 1.0
        return [dc.INTPOWER_OP, v], [1.0, float(n - 1), 1.0], float(n)
    if f[0] in _POWER:
        # d/du |u|^p = p sgn(u)|u|^(p-1), d/du sgn(u)|u|^p = p |u|^(p-1):
        # the derivative flips the parity
        p = dc._clip_exp(ev[f[2]], dc.MAX_EXP + 1.0)
        odd = (f[0] == 'sgnpower') != deriv
        return (['power', v],
                ([0.0, 1.0] if odd else [1.0, 0.0]) + [p - 1.0 if deriv else p,
                                                       1.0],
                p if deriv else 1.0)
    # blendcos / blendsin: e^(a u) cos(b u), e^(a u) sin(b u); derivatives
    # e^(a u) (a cos(b u) - b sin(b u)), e^(a u) (b cos(b u) + a sin(b u))
    a = float(np.clip(ev[f[2]], *dc.BLEND_A_BOUND))
    b = float(np.clip(ev[f[3]], *dc.BLEND_B_BOUND))
    amps = ({'blendcos': (a, -b), 'blendsin': (b, a)} if deriv
            else {'blendcos': (1.0, 0.0), 'blendsin': (0.0, 1.0)})[f[0]]
    return [dc.BLEND_OP, v], [*amps, a, b, 1.0], 1.0


def _pieces(m, vel, ev, r):
    """The force energy monomial ``m`` (unit coefficient) exerts on DOF
    ``r``, times m_r/m_0, in physical units: ``[(factors, scale), ...]``, a
    sum of products of grammar factors, one per factor it is differentiated
    in that moves with DOF ``r``."""
    out = []
    for i, f in enumerate(m['factors']):
        k = -_dz(f[1], r) if f[1] % 2 == vel else 0.0
        if k:
            parts = [_factor(h, ev, j == i) for j, h in enumerate(m['factors'])]
            out.append(([p[:2] for p in parts], k * parts[i][2]))
    return out


def _value(tokens, consts, cols):
    """A grammar factor from :func:`_factor` on the normalised leaves
    ``cols``, as ``dc.compile_to_numpy`` evaluates it."""
    z, c = cols[dc.COL_MAP[tokens[-1]]], consts
    if len(tokens) == 1:
        return c[0] * z
    if tokens[0] == dc.INTPOWER_OP:
        return c[0] * dc._intpower_np(z, c[1])
    if tokens[0] == 'power':
        return np.abs(z) ** dc._clip_exp(c[2]) * (c[0] + c[1] * np.sign(z))
    return np.exp(c[2] * z) * (c[0] * np.cos(c[3] * z) + c[1] * np.sin(c[3] * z))


def _force(m, vel, ev, r, cols):
    """:func:`_pieces` evaluated on ``cols``, or None where ``m`` exerts no
    force on ``r``."""
    pieces = _pieces(m, vel, ev, r)
    if not pieces:
        return None
    out = np.zeros(len(cols[0]))
    for parts, scale in pieces:
        g = np.full(len(cols[0]), scale)
        for t, c in parts:
            if t:
                g = g * _value(t, c, cols)
        out = out + g
    return np.where(np.isfinite(out), out, 0.0)


def _product(parts, scale):
    """``scale`` times the product of grammar factors ``parts`` as one term
    ``(tau, consts)``; the scale goes into the first factor's amplitudes."""
    parts = [(t, list(c)) for t, c in parts if t]
    if not parts:
        return ['const'], [scale]
    t0, c0 = parts[0]
    for a in ((0, 1) if t0[0] in ('power', dc.BLEND_OP) else (0,)):
        c0[a] *= scale
    tau = [tok for t, _c in parts for tok in t]
    return ((['mul'] + tau + ['end'] if len(parts) > 1 else tau),
            [x for _t, c in parts for x in c])


def complete(taus):
    """The terms of a sampled row the joint fit can complete.

    ``taus`` are the row's N_DOF local taus, then its energy tau.  Returns
    ``(kept, dofs)``: per slice its kept terms, and per sampled energy term
    the DOFs it acts on (none if dropped)."""
    N = dc.N_DOF
    kept = [[t for t in dc.split_terms(tau) if dc._linear_plan(t) is not None]
            for tau in taus[:N]]
    terms = dc.split_terms(taus[N])
    plans = [energy_term(t) for t in terms]
    kept.append([t for t, e in zip(terms, plans) if e is not None])
    return kept, [sorted(e[3]) if e else [] for e in plans]


def fit_system(kept, masses=None, horizon=None):
    """Fit a sampled row's constants jointly.

    ``kept`` holds per slice its kept terms (N_DOF local slices, then the
    energy).  Each DOF's rows are its own work-energy and acceleration rows,
    exactly as in the per-DOF fit.  A local monomial's column is its basis in
    its own DOF's rows; an energy monomial's column holds, in the rows of
    each DOF r it acts on, the force it exerts on r per unit coefficient
    (:func:`_pieces`, over m_r/m_0).  The exponents and the mass ratios
    m_r/m_0 (``masses=None``; else fixed by ``masses``) are searched as in
    the per-DOF fit (``dc._varpro``).  Returns ``(exprs, energy, mu)`` --
    each DOF's equation ``(tau, consts)``, its local terms then the energy's
    forces on it; the energy ``(tau, consts)``; the mass ratios -- or None.
    """
    N = dc.N_DOF
    data = [dc._energy_contexts(d, None, horizon) for d in range(N)]
    if not data[0][0]:
        return None
    setup = dc._fit_setup(data[0][0])
    plans = [[dc._linear_plan(t) for t in kept[d]] for d in range(N)]
    eplans = [energy_term(t) for t in kept[N]]
    if None in sum(plans, []) + eplans:
        return None
    plan = lambda sl, k: plans[sl][k] if sl < N else eplans[k][0]

    idx = [(sl, k, j) for sl in range(N + 1)
           for k in range(len(kept[sl])) for j in range(len(plan(sl, k)[0]))]
    exp_slots, kinds, limits, acts = [], {}, {}, set()
    for sl in range(N + 1):
        for k in range(len(kept[sl])):
            for s in plan(sl, k)[2]:
                kind = plan(sl, k)[3][s]
                exp_slots.append((sl, k, s))
                kinds[(sl, k, s)] = _EKIND.get(kind, kind) if sl == N else kind
    for k, (_p, _vel, lim, dofs) in enumerate(eplans):
        limits.update({(N, k, s): v for s, v in lim.items()})
        acts |= dofs
    if masses is None:
        exp_slots += [(-1, r) for r in sorted(acts - {0})]
        kinds.update({(-1, r): 'mass' for r in acts - {0}})
        mu_of = lambda ev: [ev.get((-1, r), 1.0) for r in range(N)]
    else:
        fixed = [float(m) / float(masses[0]) for m in masses]
        mu_of = lambda ev: fixed
    exp_slots = sorted(exp_slots)

    cache = {}

    def rows(sl, k, j, r, ci, ev):
        """Monomial ``j`` of term ``k`` of slice ``sl`` in DOF ``r``'s
        context ``ci`` (an energy's times m_r/m_0), or None where it is not
        in ``r``'s equation."""
        if sl < N and sl != r:
            return None
        m = plan(sl, k)[0][j]
        evk = {s: ev[(sl, k, s)] for s in m['exps']}
        key = (sl, k, j, r, ci,
               tuple(sorted((s, round(float(v), 6)) for s, v in evk.items())))
        if key not in cache:
            ctx = data[r][0][ci]
            phi = (dc._eval_monomial(m, ctx[1], evk, len(ctx[2])) if sl < N
                   else _force(m, eplans[k][1], evk, r, ctx[1]))
            cache[key] = (None if phi is None else
                          dc._basis_rows(ctx, phi, data[r][2] if sl < N
                                         else 1.0, setup))
        return cache[key]

    blocks = [(r, ci, dc._target_rows(c, data[r][1], setup))
              for r in range(N) for ci, c in enumerate(data[r][0])]
    b = np.concatenate([t for _r, _ci, t in blocks])

    def design(ev):
        mu = mu_of(ev)
        cols = []
        for sl, k, j in idx:
            col = []
            for r, ci, t in blocks:
                x = rows(sl, k, j, r, ci, ev)
                col.append(np.zeros(len(t)) if x is None
                           else x if sl < N else x / mu[r])
            cols.append(np.concatenate(col))
        return np.column_stack(cols) if cols else np.zeros((len(b), 0))

    coef, ev = dc._varpro(design, b, exp_slots, kinds, limits, RIDGE)
    if coef is None or not np.all(np.isfinite(coef)):
        return None
    at = {key: i for i, key in enumerate(idx)}
    mu = mu_of(ev)

    def consts(sl, k):
        monos, matched, slots, _kd = plan(sl, k)
        c = [1.0] * dc.count_total_consts(kept[sl][k])
        for j in range(len(monos)):
            c[matched[j]] = float(coef[at[(sl, k, j)]])
        for s in slots:
            c[s] = float(ev[(sl, k, s)])
        return c

    exprs = []
    for r in range(N):
        terms = [(t, consts(r, k)) for k, t in enumerate(kept[r])]
        for k, ((monos, _mt, slots, _kd), vel, _l, _d) in enumerate(eplans):
            evk = {s: ev[(N, k, s)] for s in slots}
            for j, m in enumerate(monos):
                a = float(coef[at[(N, k, j)]]) / (mu[r] * data[r][2])
                terms += [_product(parts, scale * a)
                          for parts, scale in _pieces(m, vel, evk, r)]
        if not terms:
            return None
        exprs.append((dc.assemble_terms([t for t, _c in terms]),
                      [x for _t, c in terms for x in c]))
    energy = (dc.assemble_terms(kept[N]),
              [x for k in range(len(kept[N])) for x in consts(N, k)])
    return exprs, energy, mu


def energy_lines(tau, consts, digits=4):
    """The energy ``(tau, consts)`` as ``V = ...`` and ``R = ...`` in the
    physical states, per unit mass of DOF 0."""
    groups, o = {'V': ([], []), 'R': ([], [])}, 0
    for t in dc.split_terms(tau):
        n = dc.count_total_consts(t)
        g = groups['R' if any(x in dc.VAR_SET and dc.COL_MAP[x] % 2
                              for x in t) else 'V']
        g[0].append(t)
        g[1].extend(consts[o:o + n])
        o += n
    return [dc.denormalize_expr(dc.assemble_terms(ts), cs, None, digits, name)
            for name, (ts, cs) in groups.items() if ts]


def system_worker(args):
    """Fit and score one sampled row as a system (``coupling`` != 'free').

    args = (taus, masses, horizon): the row's N_DOF local taus, then its
    energy tau.  Returns, aligned with those slices, ``(slice, reward, tau,
    consts, {'peer': reward}, {'peer': {dof: (tau, consts)}}, extras)``, or
    None for a slice scoring 0.  A DOF's tau is its whole equation and its
    reward its own; the energy's tau is its kept terms and its reward the
    DOFs' mean.  ``extras = {'own': the sampled tau, 'masses': m_d/m_0}``,
    for the energy also ``'term_dofs'``: per sampled term the DOFs it acts
    on.  Under the simulation reward the DOFs are integrated together; under
    the energy reward each is scored on its own balance.
    """
    taus, masses, horizon = args
    N = dc.N_DOF
    kept, dofs = complete(taus)
    fit = fit_system(kept, masses, horizon)
    if fit is None:
        return [None] * (N + 1)
    exprs, energy, mu = fit
    if dc.REWARD_MODE == 'simulation':
        rs = dc.coupled_simulation_rewards(exprs, range(N), horizon=horizon)
    else:
        rs = {d: dc.energy_reward([e if k == d else None
                                   for k, e in enumerate(exprs)],
                                  horizon=horizon) for d in range(N)}
    rs[N] = float(np.mean([rs[d] for d in range(N)]))
    out = []
    for s, (tau, consts) in enumerate(exprs + [energy]):
        extras = {'own': list(taus[s]), 'masses': mu}
        if s == N:
            extras['term_dofs'] = dofs
        out.append((s, rs[s], tau, consts, {'peer': rs[s]},
                    {'peer': {j: exprs[j] for j in range(N) if j != s}},
                    extras) if tau and rs[s] > 1e-6 else None)
    return out
