"""
discover_coupling.py
====================

Reciprocity and Newton's third law as training options:
``DISCOVER_TRAIN(coupling='reciprocal' | 'newton3')``.  The plan and the
reasoning are in ``notes_coupling_constraints.md``.

The policy still writes one equation per DOF.  For each sampled row
:func:`system_worker` keeps the terms the closed-form fit can complete, adds
every coupling term's PARTNERS to the equations it reaches, fits all the
constants of the system at once (a term and its partners share theirs) and
scores the DOFs together.

Partner rule.  A term t of DOF d reaches DOF r when it depends on r's
variable: r's position, or r's velocity if t contains any velocity.  Its
partner in r's equation is d/dq_r of the integral of t over d's own variable,
times the mass ratio m_d/m_r, which makes m_d*da_d/dq_r = m_r*da_r/dq_d.

* ``'reciprocal'``: terms over the raw leaves.  A monomial's partner is worked
  out one variable at a time with d/dx |x|^e sgn(x)^o = e |x|^(e-1) sgn(x)^(1-o)
  and its inverse for the integral, so a partner is again a product of powers
  of single variables.  A coupling monomial whose partner would leave the
  grammar (a ``blend`` factor, a variable repeated in the product) drops its
  term, and the exponents of coupling factors are searched only where the
  partner's exponents stay in [0, MAX_EXP].
* ``'newton3'``: every term is local (its own DOF's raw leaves) or relative
  (the leaves of one difference q_a - q_b, see ``dc.configure_grammar``), and a
  relative term's partner is its mirror, -m_d/m_r times itself: the forces
  cancel.

Everything is in the normalised units the rest of DISCOVER uses, so a partner
also carries the ratio of the two equations' output scales y_std and, under
``'reciprocal'``, of the two variables' scales X_std.
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


def _reach(mono, d, level):
    """The DOFs other than ``d`` whose equations monomial ``mono`` of DOF
    ``d`` reaches, or None when its partner cannot be written."""
    cols = [f[1] for f in mono['factors']]
    if level == 'newton3':
        pairs = [dc.PAIRS[(c - 2 * dc.N_DOF) // 2] for c in cols
                 if c >= 2 * dc.N_DOF]
        return {b if a == d else a for a, b in pairs}
    vel = any(c % 2 for c in cols)
    reach = {c // 2 for c in cols if c % 2 == vel and c // 2 != d}
    if reach and (len(set(cols)) < len(cols)
                  or any(f[0].startswith('blend') for f in mono['factors'])):
        return None
    return reach


def _pw(f, ev, d):
    """``(exponent, odd)`` of a power-family factor of DOF ``d``."""
    if f[0] == 'var':
        return 1.0, True
    if f[0] == dc.INTPOWER_OP:
        n = dc._snap_int_power(ev[(d, f[2])])
        return float(n), n % 2 == 1
    return dc._clip_exp(ev[(d, f[2])]), f[0] == 'sgnpower'


def _partner(mono, d, r, ev):
    """``(coef, {col: (exponent, odd)})``: the 'reciprocal' partner, in DOF
    ``r``'s equation, of monomial ``mono`` of DOF ``d``."""
    vel = any(f[1] % 2 for f in mono['factors'])
    own, oc = 2 * d + vel, 2 * r + vel
    pw = {f[1]: _pw(f, ev, d) for f in mono['factors']}
    e, odd = pw[oc]
    pw[oc] = (e - 1.0, not odd)          # the derivative ...
    f, odd = pw.get(own, (0.0, False))
    pw[own] = (f + 1.0, not odd)         # ... of the integral over x_own
    return e / (f + 1.0), {c: p for c, p in pw.items() if p != (0.0, False)}


def _value(pw, cols):
    v = 1.0
    for c, (e, odd) in pw.items():
        t = np.abs(cols[c]) ** e
        v = v * (np.sign(cols[c]) * t if odd else t)
    return np.where(np.isfinite(v), v, 0.0)


def _power_term(a, pw):
    """``a * prod |x_c|^e sgn(x_c)^o`` as grammar tokens: the bare variable
    for x_c itself, else a ``power`` (consts: even amplitude, odd amplitude,
    exponent, leaf)."""
    tau, consts = [], []
    for c, (e, odd) in sorted(pw.items()):
        amp = a if not tau else 1.0
        if (e, odd) == (1.0, True):
            tau += [dc.VARIABLES[c]]
            consts += [amp]
        else:
            tau += ['power', dc.VARIABLES[c]]
            consts += [0.0, amp, e, 1.0] if odd else [amp, 0.0, e, 1.0]
    return (['mul'] + tau + ['end'] if len(pw) > 1 else tau), consts


def complete(taus, level):
    """Split each DOF's sampled tau into the terms the joint fit can complete.

    Returns ``(kept, term_dofs)``: per DOF its kept terms, and per sampled
    term the DOFs whose equations it ends up in (its own alone if dropped).
    """
    kept, term_dofs = [], []
    for d, tau in enumerate(taus):
        keep, dofs = [], []
        for term in dc.split_terms(tau):
            plan = dc._linear_plan(term)
            reach = (None if plan is None
                     else [_reach(m, d, level) for m in plan[0]])
            if reach is not None and None not in reach:
                keep.append(term)
                dofs.append(sorted({d}.union(*reach)))
            else:
                dofs.append([d])
        kept.append(keep)
        term_dofs.append(dofs)
    return kept, term_dofs


def fit_system(terms, level, masses=None, horizon=None):
    """Fit a system's constants jointly, partners included.

    ``terms[d]`` are DOF d's kept terms.  Each DOF's rows are its own
    work-energy and acceleration rows, exactly as in the per-DOF fit; a
    monomial's column carries its own basis in its DOF's rows and its
    partner's in every DOF it reaches, and the exponents and mass ratios are
    searched as in the per-DOF fit (``dc._varpro``).  ``masses=None`` fits
    the mass ratios m_d/m_0.  Returns ``(exprs, mu)`` -- each DOF's full
    equation ``(tau, consts)``, its own terms then the partners it receives,
    and the mass ratios -- or None.
    """
    N = dc.N_DOF
    taus = [dc.assemble_terms(t) for t in terms]
    plans = [dc._linear_plan(t) if t else ([], [], [], {}) for t in taus]
    if any(p is None for p in plans):
        return None
    data = [dc._energy_contexts(d, None, horizon) for d in range(N)]
    if not data[0][0]:
        return None
    setup = dc._fit_setup(data[0][0])
    X_std = dc.NORM_STATS[1]

    idx = [(d, j) for d in range(N) for j in range(len(plans[d][0]))]
    offs = [np.cumsum([0] + [dc.count_total_consts(t) for t in terms[d]])
            for d in range(N)]
    term_of = {(d, j): int(np.searchsorted(offs[d], plans[d][1][j], 'right'))
               - 1 for d, j in idx}
    reach = {(d, j): _reach(plans[d][0][j], d, level) for d, j in idx}
    if level == 'newton3':        # a relative term reaches as a whole
        for d, j in idx:
            reach[(d, j)] = set().union(*(reach[x] for x in idx
                                          if term_of[x] == term_of[(d, j)]
                                          and x[0] == d))
    exp_slots = [(d, s) for d in range(N) for s in plans[d][2]]
    kinds = {(d, s): k for d in range(N) for s, k in plans[d][3].items()}
    limits = {}
    if level == 'reciprocal':
        for (d, j), R in reach.items():
            m = plans[d][0][j]
            vel = any(f[1] % 2 for f in m['factors'])
            for f in m['factors']:
                if R and len(f) == 3:
                    if f[1] == 2 * d + vel:        # integrated: e + 1 <= MAX
                        limits[(d, f[2])] = (-np.inf, dc.MAX_EXP - 1.0)
                    elif f[1] % 2 == vel and f[1] // 2 in R:
                        limits[(d, f[2])] = (1.0, np.inf)   # differentiated
    if masses is None:
        fit_mu = sorted({x for (d, j), R in reach.items() if R
                         for x in R | {d}} - {0})
        exp_slots += [(-1, d) for d in fit_mu]
        kinds.update({(-1, d): 'mass' for d in fit_mu})
        mu_of = lambda ev: [ev.get((-1, d), 1.0) for d in range(N)]
    else:
        fixed = [float(m) / float(masses[0]) for m in masses]
        mu_of = lambda ev: fixed
    exp_slots = sorted(exp_slots)

    def kappa(d, j, r, ev):
        mu = mu_of(ev)
        k = mu[d] / mu[r] * data[d][2] / data[r][2]
        if level == 'newton3':
            return -k
        vel = any(f[1] % 2 for f in plans[d][0][j]['factors'])
        return k * X_std[2 * d + vel] / X_std[2 * r + vel]

    def basis(d, j, r, cols, ev):
        m = plans[d][0][j]
        if r == d or level == 'newton3':
            return dc._eval_monomial(m, cols, {s: ev[(d, s)] for s in m['exps']},
                                     len(cols[0]))
        coef, pw = _partner(m, d, r, ev)
        return coef * _value(pw, cols)

    cache = {}

    def rows(r, ci, d, j, ev):
        m = plans[d][0][j]
        key = (r, ci, d, j, tuple(sorted((s, round(float(ev[(d, s)]), 6))
                                         for s in m['exps'])))
        if key not in cache:
            ctx = data[r][0][ci]
            cache[key] = dc._basis_rows(ctx, basis(d, j, r, ctx[1], ev),
                                        data[r][2], setup)
        return cache[key]

    blocks = [(r, ci, dc._target_rows(c, data[r][1], setup))
              for r in range(N) for ci, c in enumerate(data[r][0])]
    b = np.concatenate([t for _r, _ci, t in blocks])

    def design(ev):
        cols = []
        for d, j in idx:
            col = []
            for r, ci, t in blocks:
                if r == d:
                    col.append(rows(r, ci, d, j, ev))
                elif r in reach[(d, j)]:
                    col.append(kappa(d, j, r, ev) * rows(r, ci, d, j, ev))
                else:
                    col.append(np.zeros(len(t)))
            cols.append(np.concatenate(col))
        return np.column_stack(cols) if cols else np.zeros((len(b), 0))

    coef, ev = dc._varpro(design, b, exp_slots, kinds, limits, RIDGE)
    if coef is None or not np.all(np.isfinite(coef)):
        return None
    col_of = {k: i for i, k in enumerate(idx)}

    consts = []
    for d, (monos, matched, slots, _k) in enumerate(plans):
        c = [1.0] * dc.count_total_consts(taus[d])
        for j in range(len(monos)):
            c[matched[j]] = float(coef[col_of[(d, j)]])
        for s in slots:
            c[s] = float(ev[(d, s)])
        consts.append(c)

    recv = [[] for _ in range(N)]
    for d in range(N):
        monos, matched, o = plans[d][0], plans[d][1], offs[d]
        for k, term in enumerate(terms[d]):
            js = [j for j in range(len(monos)) if term_of[(d, j)] == k]
            if level == 'newton3':
                for r in sorted(reach[(d, js[0])]):
                    cs = list(consts[d][o[k]:o[k + 1]])
                    for j in js:
                        cs[matched[j] - o[k]] *= kappa(d, j, r, ev)
                    recv[r].append((term, cs))
                continue
            for j in js:
                for r in sorted(reach[(d, j)]):
                    a, pw = _partner(monos[j], d, r, ev)
                    recv[r].append(_power_term(
                        kappa(d, j, r, ev) * consts[d][matched[j]] * a, pw))

    exprs = []
    for r in range(N):
        tl = list(terms[r]) + [t for t, _c in recv[r]]
        if not tl:
            return None
        exprs.append((dc.assemble_terms(tl),
                      consts[r] + [x for _t, c in recv[r] for x in c]))
    return exprs, mu_of(ev)


def system_worker(args):
    """Fit and score one sampled row as a system (``coupling`` != 'free').

    args = (taus, level, masses, horizon): one sampled tau per DOF.  Returns,
    aligned with the DOFs, ``(dof, reward, tau, consts, {'peer': reward},
    {'peer': {other dof: (tau, consts)}}, extras)`` with the full equations
    (own terms then partners) and ``extras = {'own': the sampled tau,
    'term_dofs': per sampled term the DOFs it reaches, 'masses': m_d/m_0}``,
    or None for a DOF scoring 0.  Under the simulation reward the DOFs are
    integrated together; under the energy reward each is scored on its own
    balance, partners included.
    """
    taus, level, masses, horizon = args
    N = dc.N_DOF
    kept, term_dofs = complete(taus, level)
    fit = fit_system(kept, level, masses, horizon)
    if fit is None:
        return [None] * N
    exprs, mu = fit
    if dc.REWARD_MODE == 'simulation':
        rs = dc.coupled_simulation_rewards(exprs, range(N), horizon=horizon)
    else:
        rs = {d: dc.energy_reward([e if k == d else None
                                   for k, e in enumerate(exprs)],
                                  horizon=horizon) for d in range(N)}
    return [(d, rs[d], *exprs[d], {'peer': rs[d]},
             {'peer': {j: exprs[j] for j in range(N) if j != d}},
             {'own': list(taus[d]), 'term_dofs': term_dofs[d], 'masses': mu})
            if rs[d] > 1e-6 else None for d in range(N)]
