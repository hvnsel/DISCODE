"""
discover_core.py
===============

DISCOVER — coupled ODE discovery from response data.

The engine behind the whole DISCOVER algorithm: degree-of-freedom-agnostic
transformer-policy deep symbolic regression (after Bastiani et al., 2025) driven
by a **work-energy (power balance) reward** instead of a point-wise
acceleration NRMSE or a forward-simulation NRMSE.

Nothing in this module knows where the data came from.  It is imported by
every DISCOVER driver as::

    import discover_core as dc

and is configured through exactly two calls: :func:`configure_grammar` (once
the number of DOFs is known) and :func:`set_problem_data` (once trajectories
exist).  Data loading lives in :mod:`discover_data`, the policy in
:mod:`discover_policy`, the training loop in :mod:`discover_train`.

Motivation
----------
For a mechanical system the work-energy theorem states that the power injected
into each mass integrates to its change in kinetic energy::

    d/dt (1/2 * m_i * xdot_i^2) = xdot_i * f_i = xdot_i * m_i * xddot_i
    =>   integral_0^t  xdot_i * xddot_i dt  =  1/2 (xdot_i(t)^2 - xdot_i(0)^2)

The mass cancels because the discovered expression predicts the *acceleration*
``a_i = xddot_i^pred`` directly.

The balance is applied **per DOF, not summed**.  The relation above is an exact
kinematic identity for each DOF on its own: whatever coupling forces the other
DOFs exert are already contained in ``a_i``, so DOF ``i``'s balance closes
without any knowledge of DOF ``j``.  Each DOF's running residual is therefore
formed and normalised independently (by ``std(dKE_i)``, so a low-velocity DOF
is not swamped by a high-velocity one) and only then averaged::

    E_i(t) = | trapz(xdot_i * a_i, 0 -> t)                    (predicted work in)
              - 1/2 [ xdot_i(t)^2 - xdot_i(0)^2 ] | / s_i     (measured d KE)

    r = 1 / ( 1 + mean_i mean_t E_i(t) )

Consequences worth keeping in mind: constants for DOF ``i`` are fit against
DOF ``i``'s balance alone (see :func:`_energy_contexts`), and candidate
scoring likewise evaluates one DOF at a time (see :func:`energy_worker`).
The work-energy reward never co-simulates.  The simulation reward does where
the trainer's ``sim_coupling`` asks for it: those candidates go to
:func:`coupled_worker` and are integrated together with other DOFs'
equations (:func:`coupled_simulation_rewards`).

``r = 1`` is a perfect energy match.  The reward needs **no forward ODE
integration**: it uses the *measured* velocities together with the predicted
acceleration evaluated on the *measured* states.  Constants are fitted
energetically (least-squares on the running residual), so no point-wise
acceleration target is ever used.

Two reward schemes
------------------
:func:`set_reward` chooses what a candidate is SCORED by; the constant fit is
the same closed-form work-energy / acceleration fit under both:

    'energy'      the work-energy residual above, blended with acceleration
                  NRMSE by ``W_ACC`` (the default)
    'simulation'  the candidate is integrated forward from the measured
                  initial state and scored by a weighted sum of the NRMSEs
                  of its simulated displacement, velocity and acceleration
                  and of a phase-blind time-frequency term against the
                  record (``SIM_WEIGHTS``, summing to 1) -- see
                  :func:`simulation_reward` and :mod:`discover_rollout`

Both are per DOF and both map a residual ``e`` to ``r = 1 / (1 + e)``, so
everything downstream of the score is shared.

Grammar
-------
Leaves are bare state variables ``x1..x{2N}`` (each carrying an **implicit
leading coefficient**) and free constants.  With
``configure_grammar(..., directional_leaves=True)`` there are two more leaves::

    dleaf = a_1 q_1  + a_2 q_2  + ... + a_N q_N       N_DOF fitted weights
    vleaf = a_1 qd_1 + a_2 qd_2 + ... + a_N qd_N      N_DOF fitted weights

A directional leaf is a fitted DIRECTION in one block of the state rather than
a choice of one channel, which is what lets ``intpower(dleaf)`` express
``(q_1 - q_2)^3`` as a single term instead of the four monomials its expansion
needs.  Measured on ``cubic_coupled``: 3 terms and 6 constants against 6 terms
and 16 constants for the same reward.  A bare directional leaf is still linear
in its own weights, so it keeps the closed-form fit; under a power op the
weights are nonlinear and the fit falls back to the optimiser, where the
integer exponent is gridded rather than optimised through.

Powers are two unary operators::

    intpower(u) = c * u^n                       n on INTPOWER_GRID = (1, 2, 3)
    power(u)    = |u|^p * (c1 + c2 * sign(u))   p searched on POWER_EXP_GRID

``intpower`` is the STANDARD integer power: one amplitude and an exact integer
exponent, searched on the integer grid and never polished, so the fitted term
is ``c * x**3`` and nothing else.  Polynomial truths -- Duffing's ``x^3``, van
der Pol's ``x^2 * xdot`` -- are written with it.

``power`` is the general family.  ``|u|^p`` is the EVEN part and
``sign(u)|u|^p`` the ODD part, so one operator covers it all: ``c2 = 0`` is a
pure ``|u|^p``, ``c1 = 0`` is a pure ``sign(u)|u|^p`` (and at ``p = 1`` that is
just ``u``, at ``p = 0`` it is ``sign(u)``), and both amplitudes nonzero gives
an ASYMMETRIC response -- stiffness differing in tension and compression, which
no single-parity operator can express at one exponent.  An integer ``p`` is
only a special case of it, and one the fit never lands on exactly: the polish
moves the exponent off the integer (Duffing's cube fits as 2.9998) and the
wrong-parity amplitude comes back small but nonzero.  That is why ``intpower``
is its own token.

``c1`` and ``c2`` are fitted LINEARLY, so parity is decided by the fitter and is
not a token the search has to guess.  ``p`` is searched on a grid and then
polished within [MIN_EXP, MAX_EXP]; the bound is what stops the runaway a free
exponent shows under noise (measured 13.4 on duffing and 14.7 on cubic_coupled
at 5% NSR).

``intpower`` consumes **two** constant slots (c, n) and ``power`` **three**
(c1, c2, p), in pre-order, *before* the child's constants.

State layout for an N-DOF system::

    state = [q_1, qd_1, q_2, qd_2, ..., q_N, qd_N]     (length 2N)
    x{2d+1} = q_{d+1}      x{2d+2} = qd_{d+1}          (0-indexed d)

Call :func:`configure_grammar` once (with the number of DOFs) before building
any models, and :func:`set_problem_data` once the training data exist.
"""

from __future__ import annotations

import math
import multiprocessing as _mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product as _iproduct

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import discover_rollout as _ro


DEVICE = torch.device('cpu')

# ── Static grammar constants ────────────────────────────────────────────────
# The STANDARD integer power::
#
#     intpower(u) = c * u^n                   2 const slots: (c, n)
#
# n is searched on INTPOWER_GRID and never polished, so a fitted intpower is
# exactly ``c * x**3``: an integer exponent, one amplitude, no parity twin.
# ``power`` below contains u^n only as a special case it never fits exactly --
# measured on the registry truths its exponents come back as 2.9998 / 1.9997
# with a 1e-6 wrong-parity amplitude, so duffing's ``-0.5*x^3`` printed as
# ``-0.5001*Abs(x)**3.0*sign(x) + 9.1e-6*Abs(x)**3.0``.
#
# Positive exponents only, on BOTH fit paths: u^-n is singular at the origin,
# which is where an oscillator spends its time.  INTPOWER_MAX = 3 also keeps a
# four-cube expansion at 3^4 = 81 grid combos, exhaustive and exact.
INTPOWER_OP   = 'intpower'
INTPOWER_MAX  = 3
INTPOWER_GRID = tuple(float(n) for n in range(1, INTPOWER_MAX + 1))
# The general power, which replaced the old abspower / sgnpower pair::
#
#     power(u) = |u|^p * (c1 + c2*sgn(u))     3 const slots: (c1, c2, p)
#
# It subsumes both exactly.  |u|^p is the EVEN part and sgn(u)|u|^p the ODD
# part, so c2 = 0 is the old abspower and c1 = 0 is the old sgnpower.
#
# Parity is not a dial and cannot be one: every function decomposes uniquely
# into even + odd, so any "blend" of the two is a linear combination, which the
# root ``add`` already gives for free.  Instead c1 and c2 are fitted LINEARLY,
# which deletes the parity decision from the search rather than smoothing it.
# The pay-off beyond the old trio is asymmetric response -- stiffness that
# differs in tension and compression (contact, one-sided springs, backlash) --
# which needs BOTH parities at the SAME exponent and so was unreachable before.
#
# NOTE: a 1-element tuple needs the trailing comma — ('power') is the *string*
# 'power' and silently breaks every `tok in POWER_OPS` check.
POWER_OPS    = ('power',)
# Directional leaves.  ``dleaf`` evaluates to a fitted linear combination of the
# DISPLACEMENT channels, ``vleaf`` the same over the VELOCITY channels -- the
# "a*q1 + b*q2 + ..." leaf.  Off by default; enable per-run with
# ``configure_grammar(n_dof, directional_leaves=True)``.  They are ADDITIVE:
# the bare variables stay in the table, so nothing that worked before changes.
DIR_LEAVES   = ('dleaf', 'vleaf')
BINARY_OPS   = ['add', 'mul']            # ['add', 'sub', 'mul']
UNARY_OPS    = [INTPOWER_OP] + list(POWER_OPS)     # every power operator
N_ARY_OPS    = ['add', 'mul']
CONSTANTS    = ['const']
# Depth 5 (not 4): the root ``add`` of a sum-of-terms costs a whole level, and
# van der Pol's ``mu*(1 - x^2)*xdot`` needs four levels *below* it.  At 4 the
# search provably could not write that truth.  MIN_EXPR_LEN 2 (not 6): at 6 the
# linear oscillator ``add(x1, x2, end)`` (4 tokens) was unreachable too.  The
# term-structured policy applies its own per-term floor via ``min_len``.
MAX_TREE_DEPTH = 5
MIN_EXPR_LEN   = 2

# ``power`` exponents are searched on this GRID, never freely optimised.  That
# is deliberate: a free exponent runs away under noise (measured on duffing at
# 5% NSR it reached 13.4, and on cubic_coupled 14.7, because higher powers
# absorb accumulated noise).  A bounded grid cannot do that, so the grid IS the
# noise protection.  The points cover the characters that matter --
#     0.0 sgn   1.0 linear   1.5 Hertzian contact   2.0/3.0 quadratic, cubic
#
# FIVE points, not seven, and the count is load-bearing.  The grid is searched
# EXHAUSTIVELY only while ``len(grid) ** n_exponent_slots <= MAX_GRID_COMBOS``;
# above that the fitter falls back to cyclic coordinate descent, which is not
# globally optimal.  cubic_coupled's expanded-cube truth has FOUR exponent
# slots: at 5 points that is 625 combos (exhaustive, and it recovers 3,2,2,3),
# at 7 points it is 2401 (coordinate descent, which stalls at 1.5 on one slot
# and drops the truth reward from 0.99994 to 0.96617).  The dropped 0.5 and 2.5
# are still reachable -- the continuous polish after the grid search is clipped
# only to [MIN_EXP, MAX_EXP], not to grid points.
POWER_EXP_GRID = (0.0, 1.0, 1.5, 2.0, 3.0)

# The single transcendental operator, opt-in via
# ``configure_grammar(..., transcendental=True)``::
#
#     blend(u) = e^(a*u) * (c1*cos(b*u) + c2*sin(b*u))    4 slots: c1, c2, a, b
#
# This is the general solution of y'' - 2a y' + (a^2+b^2) y = 0, i.e. the whole
# family generated by a complex exponent lambda = a + ib.  One token therefore
# covers exp, sin, cos, sinh, cosh, linear-in-the-limit AND every damped or
# growing oscillation, with (a, b) fitted rather than chosen:
#
#     (a, b) = (1, 0)   ->  c1 * e^u             pure exponential
#     (a, b) = (0, 1)   ->  c1 cos u + c2 sin u  pure oscillation
#     (a, b) = (r cos t, r sin t)                r is scale, t is CHARACTER
#
# Writing t as the angle of lambda makes "how exp-like versus how sin-like" a
# continuous coordinate instead of a choice between three tokens.  The family is
# closed under differentiation, which is what makes it the right one for an ODE:
# a family that leaves itself under d/du leaves the grammar too.
#
# NOTE the endpoint degeneracy: at b = 0 the sin branch is identically zero, so
# that column is rank-deficient and its amplitude is unidentifiable.  lstsq
# handles it, and b = 0 is kept in the grid because it is the pure-exponential
# endpoint.
BLEND_OP      = 'blend'
# The a-grid is DENSER NEAR ZERO on purpose.  Physical damping rates are small,
# so the interesting part of this axis is |a| < 0.5, and a uniform grid straddles
# it badly: measured on a -6 e^(-0.4q) sin(1.5q) truth (which wants a = -0.218 in
# normalised units) the neighbouring uniform points -0.5 and 0.0 score 0.52 and
# 0.50 while a = -0.2 scores 0.92, so the grid search picked from two equally bad
# options and the polish did not recover.
BLEND_A_GRID  = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
BLEND_B_GRID  = (0.0, 0.5, 1.0, 1.5, 2.0)       # frequency
BLEND_A_BOUND = (-2.0, 2.0)                     # polish bound for a
BLEND_B_BOUND = (0.0, 3.0)                      # polish bound for b

# The mass ratios m_d/m_0 a coupled system fit searches when masses are not
# given (discover_coupling): geometric, since only ratios matter.
MASS_GRID  = (0.25, 0.5, 1.0, 2.0, 4.0)
MASS_BOUND = (0.01, 100.0)

# MIN_EXP is 0.0, not 0.5, deliberately: sgnpower's exponent is a CHARACTER
# dial.  sgn(u)*|u|^p is u at p = 1 and sgn(u) at p = 0, continuously in
# between, so the range [0, 1] interpolates viscous damping into Coulomb
# friction.  Measured span-escape against {1, u, u^3, sgn, |u|} peaks at
# p ~ 0.25 -- i.e. the richest part of the dial is exactly what a floor of 0.5
# used to exclude.  It must NOT go below 0: |u|^p is singular at the origin,
# which for an oscillator is where the data spends its time.
MIN_EXP      = 0.0
MAX_EXP      = 3.0

# Per-operator const-block sizes.  ``intpower`` takes TWO slots (amplitude,
# exponent) and ``power`` THREE (even amplitude, odd amplitude, exponent), so
# read the count from ``CONST_SLOTS`` and never assume either.
INTPOWER_CONST_SLOTS = 2
POWER_CONST_SLOTS = 3
BLEND_CONST_SLOTS = 4        # c1, c2, a, b
# token -> number of const slots it consumes, in pre-order.  Rebuilt by
# ``configure_grammar``; every tau-walker reads this instead of special-casing.
CONST_SLOTS       = {}

_OP_TOKENS         = set(BINARY_OPS + UNARY_OPS) | {BLEND_OP}
_FORBIDDEN_NESTING = {
    'add': {'add'}, 'mul': {'mul'},
    # nested fitted powers are degenerate ((|u|^p)^q == |u|^(p*q), and the same
    # for u^n under either op) — forbid any power op directly under another
    **{p: set(UNARY_OPS) for p in UNARY_OPS},
    # a blend of a blend is not degenerate but is uninterpretable and blows the
    # exponent-slot budget, so forbid it too
    BLEND_OP: {BLEND_OP},
}

# ── Configurable grammar state (set by configure_grammar) ───────────────────
N_DOF      = 2
N_VARS     = 4
VARIABLES  = ['x1', 'x2', 'x3', 'x4']    # physical state-column names
VAR_SET    = set(VARIABLES)
COL_MAP    = {v: i for i, v in enumerate(VARIABLES)}
DIR_LEAF_SET  = set()                    # enabled directional-leaf tokens
TRANSCENDENTAL_OPS = ()                  # ('blend',) when enabled
LEAF_CHANNELS = {}                       # 'dleaf' -> [col, ...] it spans
ALL_TOKENS = BINARY_OPS + UNARY_OPS + VARIABLES + CONSTANTS + ['end']
N_TOKENS   = len(ALL_TOKENS)
MASK_TOKEN = N_TOKENS
VOCAB_SIZE = N_TOKENS + 1
ARITY      = {}
SYMS       = {}                       # sympy symbols x1..x{2N}
PAIRS      = []                       # DOF pairs behind the relative leaves
PAIR_STD   = None                     # their normalising stds (q, qd per pair)
VAR_NAMES  = ['x', 'y']               # per-DOF display names (positions)

# ── Problem-data state (set by set_problem_data) ────────────────────────────
# NORM_STATS      : (X_mean[2N], X_std[2N], y_mean[N], y_std[N])
# RAW_TRAJECTORIES: list of (t_eval[Npts], ic[2N], state_arr[2N, Npts][, acc[N, Npts]])
NORM_STATS       = None
RAW_TRAJECTORIES = []
ENERGY_NORMALIZE = True   # divide the running residual by std(dKE) per traj

# Number of trajectories (trials) used by BOTH the constant fit and the reward.
# Every call site resolves ``max_traj=None`` to this, so scoring and fitting can
# never silently disagree about how much data they saw.
MAX_TRAJ = 5

# Blend weight between the two objectives, in [0, 1]:
#     residual = (1 - W_ACC) * energy_residual  +  W_ACC * accel_NRMSE
# W_ACC = 0 is the pure work-energy reward (the default; nothing changes).
# W_ACC = 1 is pure pointwise acceleration matching.
#
# Why blend at all.  The two objectives are sensitive to DIFFERENT terms, and
# measurably so.  Dropping a term from the LO-NO truth and re-fitting costs:
#
#     dropped term      d(accel reward)     d(energy reward)
#     damping  b*v          +0.025              +0.322
#     stiffness k*q         +0.475              +0.207
#
# Damping is ~3% of |a|, so acceleration matching penalises omitting it by ~3%
# -- far below the spread between candidates, which is why a small dissipative
# term is effectively never selected under pure NRMSE.  Over a cycle, however,
# the conservative terms do zero NET work while damping is the only term that
# accumulates, so the cumulative energy balance re-weights by DISSIPATED WORK
# and the ordering inverts.  Conversely the energy objective is comparatively
# weak on stiffness, where acceleration matching is strongest.
#
# Both objectives are LINEAR in the coefficients given the exponents, so the
# blend is a single stacked least-squares solve -- the VARPRO fast path is
# preserved, and the acceleration block needs no cumtrapz.
W_ACC = 0.0

# What a candidate is SCORED by (set by set_reward).  W_ACC above keeps
# steering the constant fit under both schemes; under 'simulation' it no
# longer enters the score.
#   SIM_WINDOW   None = one free run per trial from its first sample; a
#                number = restart from the measured state every SIM_WINDOW s;
#                'auto' = every 3 periods of the simulated DOF's own fastest
#                motion (a set integrated together: its fastest member's).
#                Windows apply to the pointwise terms; the time-frequency and
#                frequency-content terms always get a free run of their own
#   SIM_WEIGHTS  (displacement, velocity, acceleration, time-frequency,
#                frequency content) weights, non-negative and summing to 1
#   TRIAL_DECAY  how a candidate's trials are combined, under both schemes:
#                ranked from worst fit to best, the k-th worst weighted
#                TRIAL_DECAY**(k-1); 1 = plain mean, 0 = worst trial only
#                (discover_rollout.combine_trials)
# Each trial's number in the record, for reports, when the run uses only some
# of its trials (discover_data.select_trials); None numbers them 0, 1, ...
TRIAL_IDS = None
REWARD_MODES = ('energy', 'simulation')
REWARD_MODE  = 'energy'
SIM_WINDOW   = None
SIM_WEIGHTS  = (1.0, 0.0, 0.0, 0.0, 0.0)
TRIAL_DECAY  = 1.0
_SIM_DATA    = {}          # (max_traj, horizon) -> stacked record, per process
_TF_DATA     = {}          # (max_traj, horizon, dof) -> tf_setup, per process
_SPEC_DATA   = {}          # (max_traj, horizon, dof) -> spec_setup, per process


# ── Grammar configuration ───────────────────────────────────────────────────
def configure_grammar(n_dof: int, var_names=None, directional_leaves=False,
                      transcendental=False, relative_leaves=False):
    """Initialise the global token table for an ``n_dof`` system (2*n_dof vars).

    Leaf tokens are the bare state variables (each with an implicit leading
    coefficient) plus ``const``.  The power operators are ``intpower``,
    consuming (amplitude, integer exponent), and ``power``, consuming (even
    amplitude, odd amplitude, exponent).

    ``directional_leaves`` adds ``dleaf`` and ``vleaf``, each a fitted linear
    combination over one block of state channels::

        dleaf = a_1 q_1 + a_2 q_2 + ... + a_N q_N        (N_DOF const slots)
        vleaf = a_1 qd_1 + a_2 qd_2 + ... + a_N qd_N     (N_DOF const slots)

    They are leaves (arity 0) and are legal anywhere a bare variable is,
    including as a power op's child -- which is the point, since
    ``intpower(dleaf)`` is how ``(q_1 - q_2)^3`` becomes ONE term instead of
    the four monomials its expansion needs.  Bare variables are kept, so the
    default table is unchanged when this is off.

    ``relative_leaves`` (``coupling='newton3'``) appends, for every pair of
    DOFs a < b, two more variables after the states: q_a - q_b and
    qd_a - qd_b, each normalised by its own std (``PAIR_STD``).  They are
    variables in every respect (``x{2N+1}``, ``x{2N+2}``, ...), evaluated from
    the states by :func:`_feats`; odd columns are still velocities.
    """
    global N_DOF, N_VARS, VARIABLES, VAR_SET, COL_MAP, PAIRS
    global ALL_TOKENS, N_TOKENS, MASK_TOKEN, VOCAB_SIZE, ARITY, SYMS, VAR_NAMES
    global DIR_LEAF_SET, LEAF_CHANNELS, CONST_SLOTS, TRANSCENDENTAL_OPS

    N_DOF  = int(n_dof)
    PAIRS  = ([(a, b) for a in range(N_DOF) for b in range(a + 1, N_DOF)]
              if relative_leaves else [])
    N_VARS = 2 * N_DOF + 2 * len(PAIRS)
    VARIABLES = [f'x{i + 1}' for i in range(N_VARS)]
    VAR_SET   = set(VARIABLES)
    COL_MAP   = {v: i for i, v in enumerate(VARIABLES)}

    if directional_leaves:
        DIR_LEAF_SET  = set(DIR_LEAVES)
        LEAF_CHANNELS = {'dleaf': [2 * d for d in range(N_DOF)],
                         'vleaf': [2 * d + 1 for d in range(N_DOF)]}
    else:
        DIR_LEAF_SET  = set()
        LEAF_CHANNELS = {}
    dir_toks = [t for t in DIR_LEAVES if t in DIR_LEAF_SET]
    TRANSCENDENTAL_OPS = (BLEND_OP,) if transcendental else ()

    ALL_TOKENS = (BINARY_OPS + UNARY_OPS + list(TRANSCENDENTAL_OPS)
                  + VARIABLES + dir_toks + CONSTANTS + ['end'])
    N_TOKENS   = len(ALL_TOKENS)

    CONST_SLOTS = {}
    for t in VARIABLES: CONST_SLOTS[t] = 1
    for t in CONSTANTS: CONST_SLOTS[t] = 1
    CONST_SLOTS[INTPOWER_OP] = INTPOWER_CONST_SLOTS
    for t in POWER_OPS: CONST_SLOTS[t] = POWER_CONST_SLOTS
    for t in TRANSCENDENTAL_OPS: CONST_SLOTS[t] = BLEND_CONST_SLOTS
    for t in dir_toks: CONST_SLOTS[t] = N_DOF
    MASK_TOKEN = N_TOKENS
    VOCAB_SIZE = N_TOKENS + 1

    ARITY = {}
    for t in BINARY_OPS: ARITY[t] = 2
    for t in UNARY_OPS:  ARITY[t] = 1
    for t in TRANSCENDENTAL_OPS: ARITY[t] = 1
    for t in VARIABLES:  ARITY[t] = 0
    for t in DIR_LEAF_SET: ARITY[t] = 0
    for t in CONSTANTS:  ARITY[t] = 0
    ARITY['end'] = 0

    SYMS = {v: sp.Symbol(v) for v in VARIABLES}

    if var_names is not None:
        VAR_NAMES = list(var_names)
    elif N_DOF == 2:
        VAR_NAMES = ['x', 'y']
    else:
        VAR_NAMES = [f'q{d + 1}' for d in range(N_DOF)]


def set_problem_data(norm_stats, raw_trajectories, energy_normalize=True,
                     max_traj=5, w_acc=None):
    """Register the training data + energy-reward options (also used in workers).

    ``max_traj=None`` means "use every registered trajectory".
    """
    global NORM_STATS, RAW_TRAJECTORIES, ENERGY_NORMALIZE, MAX_TRAJ, W_ACC
    global PAIR_STD
    NORM_STATS       = norm_stats
    PAIR_STD = (np.array([np.concatenate([tr[2][2 * a + v] - tr[2][2 * b + v]
                                          for tr in raw_trajectories]).std()
                          + 1e-8 for a, b in PAIRS for v in (0, 1)])
                if PAIRS else None)
    RAW_TRAJECTORIES = raw_trajectories
    ENERGY_NORMALIZE = bool(energy_normalize)
    MAX_TRAJ = len(raw_trajectories) if max_traj is None else int(max_traj)
    if w_acc is not None:
        W_ACC = float(np.clip(w_acc, 0.0, 1.0))
    _SIM_DATA.clear()
    _TF_DATA.clear()
    _SPEC_DATA.clear()


def set_reward(mode='energy', sim_window=None,
               sim_weights=(1.0, 0.0, 0.0, 0.0), trial_decay=1.0):
    """Choose what candidates are scored by (also called in the workers).

    ``'energy'`` is the work-energy / acceleration blend of
    :func:`energy_reward`; ``'simulation'`` is the forward-simulation NRMSE of
    :func:`simulation_reward`.  ``sim_window`` and ``sim_weights`` -- the
    (displacement, velocity, acceleration[, time-frequency[, frequency
    content]]) weights, which must be non-negative and sum to 1 -- only matter
    for the latter.  ``sim_window`` is None (free runs), a restart interval in
    seconds, or ``'auto'`` (:func:`discover_rollout.auto_window`, per DOF); it
    applies to the pointwise terms, the time-frequency and frequency-content
    terms always running free.
    ``trial_decay`` in [0, 1] sets how the trials are combined under either
    scheme (see ``TRIAL_DECAY``); it never touches the constant fit.
    """
    global REWARD_MODE, SIM_WINDOW, SIM_WEIGHTS, TRIAL_DECAY
    if mode not in REWARD_MODES:
        raise ValueError(f"reward must be one of {REWARD_MODES}, got {mode!r}")
    try:
        decay = float(trial_decay)
    except (TypeError, ValueError):
        decay = float('nan')
    if not 0.0 <= decay <= 1.0:
        raise ValueError("trial_decay must be in [0, 1] (1 = plain mean over "
                         f"trials, 0 = worst trial only), got {trial_decay!r}")
    if sim_window is not None and sim_window != 'auto':
        try:
            ok = float(sim_window) > 0.0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError("sim_window must be None, 'auto' or > 0 s, "
                             f"got {sim_window!r}")
    weights = _ro.check_weights(sim_weights)
    REWARD_MODE = mode
    SIM_WINDOW  = (sim_window if sim_window in (None, 'auto')
                   else float(sim_window))
    SIM_WEIGHTS = weights
    TRIAL_DECAY = decay


def reward_label():
    """Short name of the configured reward, for log lines."""
    return 'sim' if REWARD_MODE == 'simulation' else 'energy'


def describe_reward():
    """One line saying what candidates are scored by and what the constants
    are fitted to."""
    if REWARD_MODE == 'simulation':
        names = ('displacement NRMSE', 'velocity NRMSE', 'acceleration NRMSE',
                 'time-frequency term', 'frequency-content term')
        parts = [(w, nm) for w, nm in zip(SIM_WEIGHTS, names) if w > 0]
        what = (parts[0][1] if len(parts) == 1 else
                ' + '.join(f'{w:.2f} {nm}' for w, nm in parts))
        if SIM_WINDOW is None:
            how = 'one free run per trial'
        else:
            every = ("3 periods of each DOF's own fastest motion"
                     if SIM_WINDOW == 'auto' else f'{SIM_WINDOW:g} s')
            how = f'restarted from the record every {every}'
            free = [nm for w, nm in zip(SIM_WEIGHTS[3:],
                                        ('time-frequency', 'frequency content'))
                    if w > 0.0]
            if free:
                how += f"; {' and '.join(free)} on a free run"
        text = (f"forward simulation, {what} ({how}); constants fitted "
                f"to {1 - W_ACC:.2f} work-energy + {W_ACC:.2f} accel-NRMSE")
    else:
        text = (f"work-energy, blend w_acc={W_ACC:.2f} "
                f"({1 - W_ACC:.2f} work-energy + {W_ACC:.2f} accel-NRMSE)")
    if TRIAL_DECAY != 1.0:
        text += (f"; trials ranked worst-fit first, weight x{TRIAL_DECAY:g} "
                 f"per rank")
    return text


def _resolve_max_traj(max_traj):
    return MAX_TRAJ if max_traj is None else int(max_traj)


def tok2idx(t): return ALL_TOKENS.index(t)
def idx2tok(i): return ALL_TOKENS[i]


def _var_display_map():
    """x{2d+1} -> position name, x{2d+2} -> velocity name, and a relative
    leaf -> its difference, e.g. ``(x-y)``."""
    disp = {}
    for d in range(N_DOF):
        disp[f'x{2 * d + 1}'] = VAR_NAMES[d]
        disp[f'x{2 * d + 2}'] = VAR_NAMES[d] + 'dot'
    for k, (a, b) in enumerate(PAIRS):
        for v, dot in enumerate(('', 'dot')):
            disp[VARIABLES[2 * (N_DOF + k) + v]] = \
                f'({VAR_NAMES[a]}{dot}-{VAR_NAMES[b]}{dot})'
    return disp


def _feats(S):
    """Normalised leaf columns of physical states ``S`` (rows q1, qd1, q2,
    ...): the states, then each relative leaf's difference."""
    X_mean, X_std = NORM_STATS[0], NORM_STATS[1]
    f = (S - X_mean[:, None]) / X_std[:, None]
    if not PAIRS:
        return f
    rel = np.stack([S[2 * a + v] - S[2 * b + v]
                    for a, b in PAIRS for v in (0, 1)])
    return np.concatenate([f, rel / PAIR_STD[:, None]])


# ── Numeric power helpers ───────────────────────────────────────────────────
def _snap_int_power(p):
    """``intpower``'s exponent as the integer it evaluates at: rounded and
    clipped to [1, INTPOWER_MAX].  Every walker snaps through here, so a slot
    value that drifted in an optimiser can never evaluate as a fractional
    power."""
    p = float(p)
    if not np.isfinite(p):
        return 1
    return int(round(float(np.clip(p, 1, INTPOWER_MAX))))


def _intpower_np(a, n):
    """``a ** n`` at the snapped integer exponent -- the plain power, so odd
    ``n`` keeps the sign of ``a`` and even ``n`` drops it."""
    return a ** _snap_int_power(n)


def _clip_exp(p):
    return float(np.clip(float(p), MIN_EXP, MAX_EXP))


def exp_slot_grid(kind):
    """Search grid for an exponent-like slot, by its factor kind."""
    if kind == 'blend_a': return list(BLEND_A_GRID)
    if kind == 'blend_b': return list(BLEND_B_GRID)
    if kind == INTPOWER_OP: return list(INTPOWER_GRID)
    if kind == 'mass': return list(MASS_GRID)
    return list(POWER_EXP_GRID)


def exp_slot_bound(kind):
    """Polish bound for an exponent-like slot.  ``blend``'s decay rate must be
    allowed NEGATIVE (that is what decay means), where a power's exponent must
    not go below zero at all -- |u|^p is singular at the origin.  An
    ``intpower`` exponent is never polished; its bound only has to contain the
    integer grid."""
    if kind == 'blend_a': return BLEND_A_BOUND
    if kind == 'blend_b': return BLEND_B_BOUND
    if kind == INTPOWER_OP: return (1.0, float(INTPOWER_MAX))
    if kind == 'mass': return MASS_BOUND
    return (MIN_EXP, MAX_EXP)


def dir_leaf_terms(tok):
    """``[(weight_offset, variable_name), ...]`` for a directional leaf token.

    ``weight_offset`` indexes within the leaf's own block of ``N_DOF`` const
    slots; ``variable_name`` is the state column that weight multiplies.
    """
    return [(k, VARIABLES[col]) for k, col in enumerate(LEAF_CHANNELS[tok])]


def count_total_consts(tau):
    """Constant slots in pre-order, from ``CONST_SLOTS``: variables and
    ``const`` take 1, ``intpower`` 2 (amplitude, exponent), ``power`` 3 (even
    amplitude, odd amplitude, exponent), ``blend`` 4, and a directional leaf
    ``N_DOF`` (one weight per channel it spans)."""
    return sum(CONST_SLOTS.get(t, 0) for t in tau)


def _const_reader(consts):
    """``_next_const(default=1.0)`` handing out ``consts`` in pre-order as
    floats, and ``float(default)`` once they run short."""
    const_list = list(consts) if consts else []
    const_idx  = [0]

    def _next_const(default=1.0):
        ci = const_idx[0]; const_idx[0] += 1
        return float(const_list[ci]) if ci < len(const_list) else float(default)
    return _next_const


# ── tau -> sympy conversion ─────────────────────────────────────────────────
def expr_to_sympy_str(tau: list, consts: list) -> str:
    _next_const = _const_reader(consts)
    pos         = [0]

    def node_to_str():
        if pos[0] >= len(tau): return '0'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in DIR_LEAF_SET:
            parts = [f"(({_next_const(1.0 if k == 0 else 0.0)})*{v})"
                     for k, v in dir_leaf_terms(tok)]
            return '(' + ' + '.join(parts) + ')'
        if tok in VAR_SET:
            v = _next_const()
            return f"(({v})*{tok})"
        if tok == 'const':
            v = _next_const()
            return f"({v})"
        if tok == INTPOWER_OP:
            c = _next_const()
            n = _snap_int_power(_next_const(2.0))
            child = node_to_str()
            return f"(({c})*(({child})**({n})))"
        if tok in POWER_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            pe = _clip_exp(_next_const(2.0))
            child = node_to_str()
            return (f"(((Abs({child}))**({pe}))*(({c1}) + ({c2})*sign({child})))")
        if tok in TRANSCENDENTAL_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            av = float(np.clip(_next_const(0.0), *BLEND_A_BOUND))
            bv = float(np.clip(_next_const(1.0), *BLEND_B_BOUND))
            child = node_to_str()
            return (f"(exp(({av})*({child}))*(({c1})*cos(({bv})*({child}))"
                    f" + ({c2})*sin(({bv})*({child}))))")
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                children.append(node_to_str())
            if pos[0] < len(tau): pos[0] += 1
            if tok == 'add': return '(' + ' + '.join(children) + ')'
            if tok == 'mul': return '(' + ' * '.join(children) + ')'
        return tok

    return node_to_str()


def _parse_local_dict():
    d = dict(SYMS)
    d.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
              'Abs': sp.Abs, 'sign': sp.sign})
    return d


# ── Critic ─────────────────────────────────────────────────────────────────
class ExprCritic(nn.Module):
    def __init__(self, vocab_size=None, max_len=32, d=64):
        super().__init__()
        vocab_size   = VOCAB_SIZE if vocab_size is None else vocab_size
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d)
        self.net = nn.Sequential(
            nn.Linear(max_len * d, 256), nn.ReLU(),
            nn.Linear(256, 64),          nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(self.emb(x).flatten(1)).squeeze(-1)


# ── Symbolic tree validity helpers ──────────────────────────────────────────
def _parse_stack(tau):
    stack = [('fixed', 1, 1, None)]
    for tok in tau:
        if not stack: break
        kind, val, d, op = stack[-1]
        if tok == 'end':
            if kind != 'nary': return None, 0, 0, 0, None
            stack.pop()
            if stack:
                pk, pv, pd, pop = stack[-1]
                if pk == 'fixed':
                    stack[-1] = ('fixed', pv - 1, pd, pop)
                    if pv - 1 == 0: stack.pop()
        elif tok in N_ARY_OPS:
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1, d, op)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1, d, op)
            stack.append(('nary', 0, d + 1, tok))
        else:
            a = ARITY[tok]
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1, d, op)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1, d, op)
            if a > 0: stack.append(('fixed', a, d + 1, tok))
    open_slots = 0; nary_depth = 0
    for kind, val, d, op in stack:
        if kind == 'fixed':
            open_slots += val
        elif kind == 'nary':
            # an open n-ary node still needs max(2 - val, 0) children + 'end'
            open_slots += max(2 - val, 0) + 1
            nary_depth += 1
    current_depth = stack[-1][2] if stack else 0
    parent_op     = stack[-1][3] if stack else None
    return stack, open_slots, nary_depth, current_depth, parent_op


def get_valid_tokens(tau, position, max_len, max_consts=20,
                     depth_offset=0, min_len=None, allow_leaf_at_zero=False,
                     forbid_at_root=(), power_child_vars_only=False):
    """Validity mask over ``ALL_TOKENS`` for the next token of prefix ``tau``.

    The keyword arguments exist for **term-mode** generation, where ``tau`` is
    one term of a sum that will later be assembled under a root ``add`` (see
    :func:`assemble_terms` and :func:`discover_policy.term_valid_mask`).  At
    their defaults the mask describes the flat whole-expression grammar, which
    is the reference every assembled bag has to be reachable under — the
    policy test checks exactly that.

    depth_offset          : depth the term's root will sit at in the assembled
                            tree, minus one.  A term under a root ``add`` passes
                            1, so its nesting budget matches what the flat
                            grammar would allow for the same assembled tau.
    min_len               : per-term length floor; ``None`` -> ``MIN_EXPR_LEN``.
    allow_leaf_at_zero    : let a term be a bare variable / const.
    forbid_at_root        : tokens not allowed at position 0.  Term mode passes
                            ``('add',)``: an ``add``-rooted term would assemble to
                            ``add(add(...), ...)``, which ``is_complete`` accepts
                            but ``_FORBIDDEN_NESTING`` forbids — a candidate the
                            rest of the system would never generate.
    power_child_vars_only : restrict a power op's child to a bare variable or a
                            directional leaf.  A bare variable keeps the fit in
                            the closed form; a directional leaf deliberately
                            leaves it (``(a q1 + b q2)^n`` is nonlinear in the
                            weights) because that is the only way to write a
                            power of a relative coordinate as one term.
    """
    remaining = max_len - position
    stack, open_slots, nary_depth, current_depth, parent_op = _parse_stack(tau)
    if stack is None: return np.zeros(N_TOKENS, dtype=np.float32)
    min_len = MIN_EXPR_LEN if min_len is None else int(min_len)
    depth   = current_depth + int(depth_offset)

    n_consts = tau.count('const')
    top_kind = stack[-1][0] if stack else None
    top_val  = stack[-1][1] if stack else 0
    forbidden_child_ops = _FORBIDDEN_NESTING.get(parent_op, set())

    mask = np.zeros(N_TOKENS, dtype=np.float32)
    for idx, tok in enumerate(ALL_TOKENS):
        if tok == 'end':
            if top_kind == 'nary' and top_val >= 2:
                new_open = open_slots - 1
                if new_open <= remaining - 1:
                    if not (new_open == 0 and nary_depth == 1 and position + 1 < min_len):
                        mask[idx] = 1.0
            continue

        a = ARITY[tok]
        # A token consumes an open slot when the top frame is a fixed-arity
        # parent, or when it is an n-ary node still short of 2 children (it
        # pays off one owed child).  A child of an already-satisfied n-ary
        # node consumes nothing: the pending 'end' remains owed.
        if top_kind == 'fixed':
            consume = 1
        elif top_kind == 'nary':
            consume = 1 if top_val < 2 else 0
        else:
            consume = 1
        if tok in N_ARY_OPS:
            new_open = open_slots - consume + 3   # >= 2 children + 'end'
            if new_open > remaining - 1: continue
        else:
            new_open = open_slots - consume + a
            if new_open < 0 or new_open > remaining - 1: continue

        if position == 0 and a == 0 and not allow_leaf_at_zero: continue
        if position == 0 and tok in forbid_at_root: continue
        if (a == 0 and open_slots == 1 and nary_depth == 0
                and position + 1 < min_len): continue
        if tok == 'const' and n_consts >= max_consts: continue
        if tok in _OP_TOKENS and depth >= MAX_TREE_DEPTH: continue
        if tok in forbidden_child_ops: continue
        if (power_child_vars_only
                and (parent_op in UNARY_OPS or parent_op in TRANSCENDENTAL_OPS)
                and tok not in VAR_SET and tok not in DIR_LEAF_SET): continue

        if tok in _OP_TOKENS:
            child_pos   = position + 1
            child_depth = depth + 1
            if tok in N_ARY_OPS:
                child_leaf_valid = True
            elif top_kind == 'nary':
                child_leaf_valid = True
            else:
                child_would_complete = (open_slots == 1 and nary_depth == 0)
                child_leaf_valid = (not child_would_complete) or (child_pos + 1 >= min_len)
            if not child_leaf_valid and not (child_depth < MAX_TREE_DEPTH): continue

        mask[idx] = 1.0

    return mask


def newton3_mask(dof, term_prefix):
    """``'newton3'`` leaf mask over ``ALL_TOKENS`` for one term of DOF
    ``dof``'s slice: the term is LOCAL (its own q and qd only) or RELATIVE
    (the two leaves of one pair that contains ``dof``), never both, and never
    uses another DOF's own variables or a directional leaf."""
    own = set(VARIABLES[2 * dof:2 * dof + 2])
    pairs = [set(VARIABLES[2 * (N_DOF + k):2 * (N_DOF + k) + 2])
             for k, p in enumerate(PAIRS) if dof in p]
    used = [t for t in term_prefix if t in VAR_SET]
    ok = (own if used and used[0] in own
          else next(p for p in pairs if used[0] in p) if used
          else own.union(*pairs))
    return np.array([0.0 if (t in VAR_SET or t in DIR_LEAF_SET)
                     and t not in ok else 1.0 for t in ALL_TOKENS],
                    dtype=np.float32)


# ── Sum-of-terms helpers (term-structured policy) ────────────────────────────
def _subtree_end(tau, start):
    """Index one past the complete subtree that begins at ``tau[start]``.

    Same stack walk as :func:`is_complete`, started mid-sequence.  If the
    subtree is not closed by the end of ``tau`` (a partial term), returns
    ``len(tau)``.
    """
    stack = [('fixed', 1)]
    pos = start
    while pos < len(tau) and stack:
        tok = tau[pos]; pos += 1
        kind, val = stack[-1]
        if tok == 'end':
            if kind != 'nary': return pos
            stack.pop()
            if stack:
                pk, pv = stack[-1]
                if pk == 'fixed':
                    stack[-1] = ('fixed', pv - 1)
                    if pv - 1 == 0: stack.pop()
        elif tok in N_ARY_OPS:
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1)
            stack.append(('nary', 0))
        else:
            a = ARITY[tok]
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1)
            if a > 0: stack.append(('fixed', a))
    return pos


def split_terms(tau):
    """Inverse of :func:`assemble_terms`: the top-level summands of ``tau``.

    An ``add``-rooted tau yields the token span of each child, in the order
    they appear (a trailing partial child is returned as-is).  Anything else is
    a single term.  Deterministic, so the term-structured policy can rebuild
    its teacher-forcing input from the flat tau a buffer entry stores.
    """
    tau = list(tau)
    if not tau:
        return []
    if tau[0] != 'add':
        return [tau]
    terms, pos = [], 1
    while pos < len(tau) and tau[pos] != 'end':
        end = _subtree_end(tau, pos)
        terms.append(tau[pos:end])
        pos = end
    return terms


def assemble_terms(terms):
    """``['add'] + t1 + ... + tm + ['end']``, or the bare term when ``m == 1``.

    ``add`` requires >= 2 children, so a single term is emitted unwrapped;
    that is still a valid expression because the sampler never lets a
    one-term bag be a bare leaf (see the STOP rule in
    :func:`discover_policy.sample_term_batch`).
    """
    terms = [list(t) for t in terms if t]
    if not terms:
        return []
    if len(terms) == 1:
        return terms[0]
    return ['add'] + [tok for t in terms for tok in t] + ['end']


def is_complete(tau):
    if not tau: return False
    stack = [('fixed', 1)]
    for tok in tau:
        if not stack: return False
        kind, val = stack[-1]
        if tok == 'end':
            if kind != 'nary': return False
            if val < 2: return False
            stack.pop()
            if stack:
                pk, pv = stack[-1]
                if pk == 'fixed':
                    stack[-1] = ('fixed', pv - 1)
                    if pv - 1 == 0: stack.pop()
        elif tok in N_ARY_OPS:
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1)
            stack.append(('nary', 0))
        else:
            a = ARITY[tok]
            if kind == 'fixed':
                stack[-1] = ('fixed', val - 1)
                if val - 1 == 0: stack.pop()
            elif kind == 'nary':
                stack[-1] = ('nary', val + 1)
            if a > 0: stack.append(('fixed', a))
    return len(stack) == 0


# ── Parametric numpy builder (nonlinear const fit, amplitude slots) ─────────
def _build_param_code(tau):
    """Return (code_str, n_consts, grid_exp_indices, grid_exp_kinds).

    ``code_str`` is a numpy expression in ``c`` (constant vector) and the state
    variables ``x1..x{2N}`` implementing the candidate's *normalised* output;
    it calls ``_itp`` (:func:`_intpower_np`), so eval it with that in scope.
    ``grid_exp_indices`` holds the exponent-like const slots and
    ``grid_exp_kinds`` which grid each is searched on (:func:`exp_slot_grid`)
    rather than optimised through, so the caller pins each one and fits the
    rest (see ``optimise_consts_energy``).  An ``intpower`` exponent is also
    snapped inside the code, so even an unpinned one evaluates as an integer.
    """
    const_indices = []; grid_exp_indices = []; grid_exp_kinds = []
    pos = [0]

    def build():
        if pos[0] >= len(tau): return '0'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in DIR_LEAF_SET:
            parts = []
            for _, v in dir_leaf_terms(tok):
                ci = len(const_indices); const_indices.append(ci)
                parts.append(f'c[{ci}]*{v}')
            return '(' + '+'.join(parts) + ')'
        if tok in VAR_SET:
            ci = len(const_indices); const_indices.append(ci)
            return f'c[{ci}]*{tok}'
        if tok == 'const':
            ci = len(const_indices); const_indices.append(ci)
            return f'c[{ci}]'
        if tok == INTPOWER_OP:
            ci = len(const_indices); const_indices.append(ci)
            pi = len(const_indices); const_indices.append(pi)
            grid_exp_indices.append(pi); grid_exp_kinds.append(INTPOWER_OP)
            child = build()
            return f'c[{ci}]*np.clip(_itp({child},c[{pi}]),-1e15,1e15)'
        if tok in POWER_OPS:
            c1 = len(const_indices); const_indices.append(c1)
            c2 = len(const_indices); const_indices.append(c2)
            pi = len(const_indices); const_indices.append(pi)
            grid_exp_indices.append(pi); grid_exp_kinds.append('power')
            child = build()
            return (f'np.clip((np.abs({child})**np.clip(c[{pi}],'
                    f'{MIN_EXP!r},{MAX_EXP!r}))'
                    f'*(c[{c1}]+c[{c2}]*np.sign({child})),-1e15,1e15)')
        if tok in TRANSCENDENTAL_OPS:
            c1 = len(const_indices); const_indices.append(c1)
            c2 = len(const_indices); const_indices.append(c2)
            ai = len(const_indices); const_indices.append(ai)
            bi = len(const_indices); const_indices.append(bi)
            grid_exp_indices.append(ai); grid_exp_kinds.append('blend_a')
            grid_exp_indices.append(bi); grid_exp_kinds.append('blend_b')
            child = build()
            al, ah = BLEND_A_BOUND; bl, bh = BLEND_B_BOUND
            ex = f'np.clip(np.exp(np.clip(c[{ai}],{al!r},{ah!r})*{child}),-1e15,1e15)'
            bb = f'np.clip(c[{bi}],{bl!r},{bh!r})'
            return (f'np.clip({ex}*(c[{c1}]*np.cos({bb}*{child})'
                    f'+c[{c2}]*np.sin({bb}*{child})),-1e15,1e15)')
        if tok in N_ARY_OPS:
            parts = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                parts.append(build())
            if pos[0] < len(tau): pos[0] += 1
            if not parts: return '0'
            if tok == 'add': return '(' + '+'.join(parts) + ')'
            if tok == 'mul': return '(' + '*'.join(parts) + ')'
        return '0'

    code = build()
    return code, len(const_indices), grid_exp_indices, grid_exp_kinds


# ── Pretty printing ─────────────────────────────────────────────────────────
def expr_to_str(tau, consts=None):
    if not tau: return '?'
    disp        = _var_display_map()
    _next_const = _const_reader(consts)
    pos         = [0]

    def n2s():
        if pos[0] >= len(tau): return '?'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in DIR_LEAF_SET:
            parts = []
            for k, v in dir_leaf_terms(tok):
                c = _next_const(1.0 if k == 0 else 0.0)
                parts.append(f"{c:+.4g}*{disp.get(v, v)}")
            return '(' + ' '.join(parts) + ')'
        if tok in VAR_SET:
            v = _next_const()
            return f"{v:.4g}*{disp.get(tok, tok)}"
        if tok == 'const':
            v = _next_const()
            return f"{v:.4f}"
        if tok == INTPOWER_OP:
            c = _next_const()
            n = _snap_int_power(_next_const(2.0))
            child = n2s()
            return f"{c:.4g}*({child})^{n}"
        if tok in POWER_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            pe = _clip_exp(_next_const(2.0))
            child = n2s()
            # print the readable special cases rather than the general form
            if abs(c2) < 1e-9:
                return f"{c1:.4g}*|{child}|^{pe:.4g}"
            if abs(c1) < 1e-9:
                if abs(pe) < 1e-9:
                    return f"{c2:.4g}*sgn({child})"
                if abs(pe - round(pe)) < 1e-9:
                    return f"{c2:.4g}*({child})^{int(round(pe))}"
                return f"{c2:.4g}*sgn({child})|{child}|^{pe:.4g}"
            return f"|{child}|^{pe:.4g}*({c1:+.4g}{c2:+.4g}*sgn({child}))"
        if tok in TRANSCENDENTAL_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            av = float(np.clip(_next_const(0.0), *BLEND_A_BOUND))
            bv = float(np.clip(_next_const(1.0), *BLEND_B_BOUND))
            child = n2s()
            # print the recognisable special cases
            if abs(bv) < 1e-9:
                return f"{c1:.4g}*exp({av:.4g}*{child})"
            osc = (f"{c1:.4g}*cos({bv:.4g}*{child})" if abs(c2) < 1e-9 else
                   f"{c2:.4g}*sin({bv:.4g}*{child})" if abs(c1) < 1e-9 else
                   f"({c1:+.4g}*cos({bv:.4g}*{child}){c2:+.4g}*sin({bv:.4g}*{child}))")
            if abs(av) < 1e-9:
                return osc
            return f"exp({av:.4g}*{child})*{osc}"
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end': children.append(n2s())
            if pos[0] < len(tau): pos[0] += 1
            if tok == 'add': return '(' + ' + '.join(children) + ')'
            if tok == 'mul': return '(' + ' * '.join(children) + ')'
        return tok

    try: return n2s()
    except Exception: return str(tau)


# ── NumPy compilation ───────────────────────────────────────────────────────
def compile_to_numpy(tau, consts):
    """Return ``fn(x1, ..., x{2N})`` evaluating the expression in normalised space."""
    if not tau: return None
    _next_const = _const_reader(consts)
    pos         = [0]

    def build():
        if pos[0] >= len(tau): return None
        tok = tau[pos[0]]; pos[0] += 1
        if tok in DIR_LEAF_SET:
            parts = [f'({_next_const(1.0 if k == 0 else 0.0)!r}*{v})'
                     for k, v in dir_leaf_terms(tok)]
            return '(' + '+'.join(parts) + ')'
        if tok in VAR_SET:
            c = _next_const()
            return f'({c!r}*{tok})'
        if tok == 'const':
            v = _next_const()
            return repr(v)
        if tok == INTPOWER_OP:
            c = _next_const()
            n = _snap_int_power(_next_const(2.0))
            child = build()
            if child is None: return None
            return f'({c!r}*np.clip(({child})**{n},-1e15,1e15))'
        if tok in POWER_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            pe = _clip_exp(_next_const(2.0))
            child = build()
            if child is None: return None
            return (f'(np.clip((np.abs({child})**{pe!r})'
                    f'*({c1!r}+{c2!r}*np.sign({child})),-1e15,1e15))')
        if tok in TRANSCENDENTAL_OPS:
            c1 = _next_const()
            c2 = _next_const(0.0)
            av = float(np.clip(_next_const(0.0), *BLEND_A_BOUND))
            bv = float(np.clip(_next_const(1.0), *BLEND_B_BOUND))
            child = build()
            if child is None: return None
            return (f'(np.clip(np.clip(np.exp({av!r}*{child}),-1e15,1e15)'
                    f'*({c1!r}*np.cos({bv!r}*{child})'
                    f'+{c2!r}*np.sin({bv!r}*{child})),-1e15,1e15))')
        if tok in N_ARY_OPS:
            parts = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end': parts.append(build())
            if pos[0] < len(tau): pos[0] += 1
            if not parts: return '0'
            if tok == 'add': return '(' + '+'.join(parts) + ')'
            if tok == 'mul': return '(' + '*'.join(parts) + ')'
        return '0'

    sig = ','.join(VARIABLES)
    try:
        code   = build()
        fn_raw = eval(f'lambda {sig}: {code}', {'np': np})

        def fn(*args, _f=fn_raw):
            with np.errstate(invalid='ignore', over='ignore'):
                return _f(*args)

        fn(*([0.0] * N_VARS))   # smoke test
        return fn
    except Exception:
        return None


# ── Work-energy (power balance) reward ──────────────────────────────────────
def _cumtrapz(y, t):
    """Cumulative trapezoidal integral with a leading 0 (same length as y)."""
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def _horizon_mask(t_eval, horizon):
    """Samples within ``horizon`` seconds of the record's START, or None when
    the horizon covers the whole record.

    Measured from ``t_eval[0]``, not from 0: an experimental record trimmed at
    the front starts at a nonzero time while ``t_end`` (the horizon every
    epoch uses) is its duration, and comparing the two directly used to drop
    the last ``t_eval[0]`` seconds of the record from every fit and score."""
    if horizon is None:
        return None
    rel = t_eval - t_eval[0]
    if horizon >= rel[-1] * (1.0 - 1e-12):
        return None
    return rel <= horizon


def _slice_traj(traj, horizon):
    """Return (t, state_arr) sliced to ``horizon`` (or full)."""
    t_eval, ic, state_arr = traj[:3]
    mh = _horizon_mask(t_eval, horizon)
    if mh is not None:
        if int(mh.sum()) < 3:
            return None, None
        return t_eval[mh], state_arr[:, mh]
    return t_eval, state_arr


def _slice_accs(traj, horizon):
    """Measured accelerations (N_DOF, n) sliced to ``horizon``, or None.

    Only needed when W_ACC > 0.  Returns None if the trajectory tuple carries
    no acceleration channel, in which case the caller falls back to pure
    energy for that trajectory."""
    if len(traj) < 4 or traj[3] is None:
        return None
    t_eval = traj[0]
    accs = np.asarray(traj[3], dtype=float)
    mh = _horizon_mask(t_eval, horizon)
    if mh is not None:
        if int(mh.sum()) < 3:
            return None
        return accs[:, mh]
    return accs


def _predicted_accel(fn, feats, y_mean_d, y_std_d):
    """Physical predicted acceleration for one DOF over a trajectory slice."""
    try:
        pred_n = fn(*[feats[k, :] for k in range(N_VARS)])
        pred_n = np.asarray(pred_n, dtype=float)
        if pred_n.ndim == 0:
            pred_n = np.full(feats.shape[1], float(pred_n))
    except Exception:
        return None
    pred_n = np.where(np.isfinite(pred_n), pred_n, 0.0)
    return y_mean_d + y_std_d * pred_n


def energy_reward(exprs, max_traj=None, horizon=None):
    """Global work-energy reward for a full set of per-DOF expressions.

    ``exprs`` is a length-``N_DOF`` list whose entry ``d`` is ``(tau, consts)``
    for DOF ``d`` or ``None`` if that DOF has no expression yet.  Only the DOFs
    that have one are scored, each on its OWN balance: per trajectory, its
    running residual ``|cum_power_d(t) - dKE_d(t)|`` (divided by
    ``std(dKE_d)`` when ``ENERGY_NORMALIZE``) is averaged over time and
    blended with its acceleration NRMSE by ``W_ACC``, and the DOFs are then
    averaged.  The first ``max_traj`` trajectories (``None`` = the configured
    ``MAX_TRAJ``) are combined by :func:`discover_rollout.combine_trials`
    (``TRIAL_DECAY``) into ``e``.

    Returns ``r = 1 / (1 + e)``; 0 if nothing is evaluable.
    """
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return 0.0
    y_mean, y_std = NORM_STATS[2], NORM_STATS[3]

    present = [d for d in range(N_DOF) if exprs[d] is not None]
    if not present:
        return 0.0
    fns = {}
    for d in present:
        f = compile_to_numpy(*exprs[d])
        if f is None:
            return 0.0
        fns[d] = f

    w = W_ACC
    residual_means = []
    for traj in RAW_TRAJECTORIES[:_resolve_max_traj(max_traj)]:
        t_sl, st_sl = _slice_traj(traj, horizon)
        if t_sl is None:
            continue
        acc_meas = _slice_accs(traj, horizon) if w > 0.0 else None
        feats = _feats(st_sl)
        # Compute each DOF's residual independently and normalise by its own
        # KE scale so that a low-velocity DOF is not swamped by a high-velocity
        # DOF when both are included in the same energy balance.
        per_dof = []
        ok = True
        for d in present:
            vel_d = st_sl[2 * d + 1, :]
            acc_d = _predicted_accel(fns[d], feats, y_mean[d], y_std[d])
            if acc_d is None:
                ok = False
                break
            cp_d  = _cumtrapz(vel_d * acc_d, t_sl)
            dke_d = 0.5 * (vel_d ** 2 - vel_d[0] ** 2)
            res_d = np.abs(cp_d - dke_d)
            if ENERGY_NORMALIZE:
                res_d = res_d / _ro._scale(dke_d)
            e_res = float(np.mean(res_d))
            # Blend in pointwise acceleration NRMSE.  Both terms are already
            # dimensionless (energy by std(dKE), accel by std(a_meas)), so the
            # weighted sum is meaningful without further rescaling.
            if w > 0.0 and acc_meas is not None:
                a_meas = acc_meas[d, :]
                a_res = (float(np.sqrt(np.mean((acc_d - a_meas) ** 2)))
                         / _ro._scale(a_meas))
                per_dof.append((1.0 - w) * e_res + w * a_res)
            else:
                per_dof.append(e_res)
        if not ok:
            continue
        residual_means.append(float(np.mean(per_dof)))

    if not residual_means:
        return 0.0
    return 1.0 / (1.0 + _ro.combine_trials(residual_means, TRIAL_DECAY))


# ── Forward-simulation reward ───────────────────────────────────────────────
def _sim_data(max_traj, horizon):
    """``(t, states, accs, stride, substeps)`` for the simulation reward: the
    first ``max_traj`` trials stacked on their shared time grid, and the RK4
    step :func:`discover_rollout.auto_steps` sizes from them.  Cached per
    process until :func:`set_problem_data` replaces the data."""
    key = (_resolve_max_traj(max_traj), horizon)
    if key not in _SIM_DATA:
        ts, sts, acs = [], [], []
        for traj in RAW_TRAJECTORIES[:key[0]]:
            t_sl, st_sl = _slice_traj(traj, horizon)
            if t_sl is None:
                continue
            ts.append(np.asarray(t_sl, dtype=float))
            sts.append(np.asarray(st_sl, dtype=float))
            acs.append(_slice_accs(traj, horizon))
        out = None
        if ts:
            if any(len(x) != len(ts[0]) or not np.allclose(x, ts[0]) for x in ts):
                raise ValueError("the simulation reward needs every trial on "
                                 "one shared time grid")
            states = np.stack(sts)
            accs = (np.stack(acs) if all(a is not None for a in acs) else None)
            stride, substeps = _ro.auto_steps(ts[0], states, accs)
            out = (ts[0], states, accs, stride, substeps)
        _SIM_DATA[key] = out
    return _SIM_DATA[key]


def _window(data, dofs=None):
    """``SIM_WINDOW`` in seconds for this record, ``'auto'`` resolved for the
    DOFs ``dofs`` being simulated (default: all) -- one DOF alone gets its own
    3 periods, a set integrated together restarts at its fastest member's."""
    if SIM_WINDOW == 'auto':
        t, states, accs, _stride, _substeps = data
        return _ro.auto_window(t, states, accs, dofs=dofs)
    return SIM_WINDOW


def _tf_setup(max_traj, horizon, dof, data):
    """The measured side of DOF ``dof``'s time-frequency term
    (:func:`discover_rollout.tf_setup`) on ``_sim_data``'s record and stride,
    cached per process like the record itself."""
    key = (_resolve_max_traj(max_traj), horizon, dof)
    if key not in _TF_DATA:
        t, states, _accs, stride, _substeps = data
        _TF_DATA[key] = _ro.tf_setup(t, states[:, 2 * dof, :], stride)
    return _TF_DATA[key]


def _spec_setup(max_traj, horizon, dof, data):
    """The measured side of DOF ``dof``'s frequency-content term
    (:func:`discover_rollout.spec_setup`), cached like :func:`_tf_setup`."""
    key = (_resolve_max_traj(max_traj), horizon, dof)
    if key not in _SPEC_DATA:
        t, states, _accs, stride, _substeps = data
        _SPEC_DATA[key] = _ro.spec_setup(t, states[:, 2 * dof + 1, :], stride)
    return _SPEC_DATA[key]


def _accel_fn(d, expr):
    """DOF ``d``'s ``(tau, consts)`` as the rollout's ``accel(S)``: physical
    states in, physical acceleration out (the expression itself works in
    normalised units).  None if it does not compile."""
    fn = compile_to_numpy(*expr)
    if fn is None:
        return None
    y_mean, y_std = NORM_STATS[2], NORM_STATS[3]
    ym, ys = float(y_mean[d]), float(y_std[d])

    def accel(S):
        return ym + ys * fn(*_feats(S))
    return accel


def print_record_summary(max_traj=None, horizon=None):
    """Print, per DOF and trial, what the simulation reward has to work with:
    the two strongest spectral peaks of the displacement, the envelope swing
    the time-frequency term sees (the largest std of a band's detrended
    log-amplitude, nepers: ~0 for a steady or decaying oscillation, ~0.3 and
    up for clear beats), and the beat period read off the envelope's own
    spectrum.  A trial whose swing stands out is the one carrying the beats;
    one whose peaks fall outside the bands is invisible to the term.  Also
    printed: how many samples per cycle each DOF's fastest motion gets (under
    ~10 the record is too coarse to compare that motion point by point, or
    to fit its constants well), the windows ``sim_window='auto'`` picked, and
    the range the frequency-content term compares when it is weighted."""
    data = _sim_data(max_traj, horizon)
    if data is None:
        return
    t, states, accs, stride, _sub = data
    T = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t)))
    D = dt * stride
    print(f"[record] per trial, what the simulation reward compares "
          f"({T:.4g} s per trial, {1.0 / D:.4g} Hz simulation grid):")
    fast = []
    for d in range(N_DOF):
        w = _ro.fastest_omega(states, accs, d)
        if w > 0.0:
            fast.append((d, w / (2.0 * np.pi), 2.0 * np.pi / (w * dt)))
    if fast:
        print("  fastest motion: " + ", ".join(
            f"DOF {d} ~{f:.3g} Hz ({n:.3g} samples/cycle)" for d, f, n in fast)
              + ("" if min(n for _d, _f, n in fast) >= 10 else
                 "  -- under ~10 samples a cycle is coarse for the pointwise "
                 "terms and the constant fit: raise desired_timesteps"))
    if SIM_WINDOW == 'auto':
        wins = [_window(data, d) for d in range(N_DOF)]
        if any(w for w in wins):
            parts = [f"{w:.3g} s (DOF {d})" if w else f"free (DOF {d})"
                     for d, w in enumerate(wins)]
            together = _window(data) if N_DOF > 1 else None
            print("  sim_window='auto': pointwise terms restart every "
                  + ", ".join(parts)
                  + (f"; DOFs simulated together every {together:.3g} s"
                     if together else ""))
        else:
            print("  sim_window='auto': no motion to time -- free runs")
    for d in range(N_DOF):
        if SIM_WEIGHTS[4] > 0.0:
            sp = _spec_setup(max_traj, horizon, d, data)
            print(f"  DOF {d}: frequency content compared over "
                  + (f"{sp['f_lo']:.3g}-{sp['f_hi']:.3g} Hz"
                     if sp['band'] is not None else
                     "nothing (record too short or motionless)"))
        tf = _tf_setup(max_traj, horizon, d, data)
        if tf['G'] is None:
            print(f"  DOF {d}: no oscillation found -- the time-frequency term "
                  f"reduces to its trend")
            continue
        freqs, rel = _ro.spectral_peaks(t, states[:, 2 * d, :], stride,
                                        tf['f_lo'], tf['f_hi'])
        print(f"  DOF {d}: time-frequency bands {tf['f_lo']:.3g}-{tf['f_hi']:.3g} Hz"
              + ("" if tf['rate'] is not None else
                 "; record too short for the beat-rate term"))
        print("     trial   peak Hz   2nd peak Hz (rel)   envelope swing   "
              "beat period s")
        swing = tf['swing']
        strongest = int(np.argmax(swing)) if len(swing) > 1 else -1
        for p in range(len(swing)):
            f1, f2, r2 = freqs[p, 0], freqs[p, 1], rel[p, 1]
            second = (f"{f2:9.4g} ({r2:4.2f})" if np.isfinite(f2) else
                      f"{'-':>16s}")
            beat = (f"{1.0 / tf['rate']['nu_peak'][p]:13.4g}"
                    if tf['rate'] is not None and swing[p] >= 0.1
                    else f"{'-':>13s}")
            mark = "   <- strongest" if p == strongest else ""
            label = TRIAL_IDS[p] if TRIAL_IDS and p < len(TRIAL_IDS) else p
            print(f"     {label:5d} {f1:9.4g}   {second}   {swing[p]:14.3f}   "
                  f"{beat}{mark}")


def simulation_reward(exprs, max_traj=None, horizon=None):
    """Forward-simulation reward for a set of per-DOF expressions.

    Same argument convention as :func:`energy_reward`.  Each present DOF's
    equation is integrated from the measured initial state of every trial --
    or of every ``SIM_WINDOW``-second window, ``'auto'`` resolved for that DOF
    -- with the OTHER DOFs' states read off the record, and its residual is
    the ``SIM_WEIGHTS``-weighted sum of the NRMSEs of its simulated
    displacement, velocity and acceleration against the measured ones and of
    the time-frequency and frequency-content terms; see
    :mod:`discover_rollout`.  Aggregated like the energy reward: mean
    over DOFs per trial, the trials combined by
    :func:`discover_rollout.combine_trials` (``TRIAL_DECAY``) into ``e``,
    then ``r = 1 / (1 + e)``.  0 if nothing is evaluable.

    A DOF is simulated alone, so its score never depends on the other DOFs'
    current expressions: the same isolation :func:`energy_worker` relies on.
    """
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return 0.0
    present = [d for d in range(N_DOF) if exprs[d] is not None]
    if not present:
        return 0.0
    data = _sim_data(max_traj, horizon)
    if data is None:
        return 0.0
    if SIM_WEIGHTS[2] > 0.0 and data[2] is None:
        return 0.0                  # nothing measured to compare against

    per_dof = []
    for d in present:
        res = _alone_residuals(d, exprs[d], data, max_traj, horizon)
        if res is None or not np.all(np.isfinite(res)):
            return 0.0
        per_dof.append(res)
    return 1.0 / (1.0 + _ro.combine_trials(per_dof, TRIAL_DECAY))


def _alone_residuals(d, expr, data, max_traj, horizon):
    """Per-trial simulation residuals of DOF ``d``'s ``(tau, consts)``
    simulated alone (the other DOFs read off the record); None if it does not
    compile."""
    t, states, accs, stride, substeps = data
    accel = _accel_fn(d, expr)
    if accel is None:
        return None
    tf = (_tf_setup(max_traj, horizon, d, data)
          if SIM_WEIGHTS[3] > 0.0 else None)
    spec = (_spec_setup(max_traj, horizon, d, data)
            if SIM_WEIGHTS[4] > 0.0 else None)
    return _ro.rollout_residuals(accel, t, states, d,
                                 window=_window(data, d), stride=stride,
                                 substeps=substeps, weights=SIM_WEIGHTS,
                                 accs=accs, tf=tf, spec=spec)


def simulation_trial_rewards(expr, dof, max_traj=None, horizon=None):
    """DOF ``dof``'s ``(tau, consts)`` simulated alone, scored trial by trial
    as :func:`simulation_reward` scores it before it combines the trials:
    ``1 / (1 + residual)`` per trial, or None if it cannot be scored.
    Combining the residuals with :func:`discover_rollout.combine_trials`
    gives :func:`simulation_reward`'s value back."""
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return None
    data = _sim_data(max_traj, horizon)
    if data is None or (SIM_WEIGHTS[2] > 0.0 and data[2] is None):
        return None
    res = _alone_residuals(dof, expr, data, max_traj, horizon)
    if res is None:
        return None
    res = np.asarray(res, dtype=float)
    return np.where(np.isfinite(res), 1.0 / (1.0 + res), 0.0)


def coupled_simulation_rewards(exprs, score, max_traj=None, horizon=None):
    """Simulation rewards when DOFs are integrated TOGETHER: ``{dof: r}`` for
    every DOF in ``score``.

    Every DOF with an expression in ``exprs`` (a length-N list of
    ``(tau, consts)`` or None) is integrated as one coupled system, each
    equation driven by the others' SIMULATED states; a DOF without one is read
    off the record.  Each scored DOF gets ``r = 1 / (1 + e)``, ``e`` its own
    channel's residual per trial, weighted as in :func:`simulation_reward`,
    with the trials combined by :func:`discover_rollout.combine_trials`
    (``TRIAL_DECAY``) -- except that a trial where the set blew up scores
    every DOF in it 0 (:func:`discover_rollout.rollout_set_residuals`).  A
    scored DOF whose expression does not compile gets 0 and the rest are
    integrated without it.
    """
    out = {int(d): 0.0 for d in score}
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return out
    data = _sim_data(max_traj, horizon)
    if data is None:
        return out
    t, states, accs, stride, substeps = data
    if SIM_WEIGHTS[2] > 0.0 and accs is None:
        return out
    accels = {}
    for d in range(N_DOF):
        if exprs[d] is not None:
            accel = _accel_fn(d, exprs[d])
            if accel is not None:
                accels[d] = accel
    scored = [d for d in out if d in accels]
    if not scored:
        return out
    tfs = ({d: _tf_setup(max_traj, horizon, d, data) for d in accels}
           if SIM_WEIGHTS[3] > 0.0 else None)
    specs = ({d: _spec_setup(max_traj, horizon, d, data) for d in accels}
             if SIM_WEIGHTS[4] > 0.0 else None)
    res = _ro.rollout_set_residuals(accels, t, states,
                                    window=_window(data, sorted(accels)),
                                    stride=stride, substeps=substeps,
                                    weights=SIM_WEIGHTS, accs=accs, tfs=tfs,
                                    score=scored, specs=specs)
    for d in scored:
        if np.all(np.isfinite(res[d])):
            out[d] = 1.0 / (1.0 + _ro.combine_trials(res[d], TRIAL_DECAY))
    return out


def candidate_reward(exprs, max_traj=None, horizon=None):
    """The configured reward (:func:`set_reward`) of a set of per-DOF
    expressions: :func:`energy_reward` or :func:`simulation_reward`."""
    if REWARD_MODE == 'simulation':
        return simulation_reward(exprs, max_traj=max_traj, horizon=horizon)
    return energy_reward(exprs, max_traj=max_traj, horizon=horizon)


def log_residual_score(r):
    """Map the stored reward ``r = 1/(1 + e)`` back to ``-log(e) = logit(r)``.

    This is the scale-invariant view of the energy residual: a k-fold residual
    improvement is worth ``log(k)`` *everywhere* on the reward axis, instead of
    being compressed to nothing as ``r -> 1`` (e.g. residual 0.040 -> 0.015 is
    r 0.961 -> 0.985, a 0.024 gap, but a full +1.0 in log units).  Monotone in
    ``r``, so it never changes rankings — only gradient magnitudes.  Used for
    GRPO advantages; buffer contents, gating, and reporting stay in r-units.
    """
    r = np.clip(np.asarray(r, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(r) - np.log1p(-r)


# ── Fast constant fitting via variable projection (VARPRO) ──────────────────
MAX_MONOMIALS = 40
# Row budget for the least-squares fits (linear fast path AND nonlinear
# fallback).  Integration always runs on the full time grid — this only
# strides the rows handed to the optimiser, which is what actually scales
# with desired_timesteps.
MAX_FIT_ROWS  = 8000

# Exponent-search budget for the exact (VARPRO) path.
#   MAX_GRID_COMBOS : use the exhaustive product grid up to this many combos
#                     (globally optimal on the grid).
#   MAX_EXP_SLOTS   : above MAX_GRID_COMBOS, fall back to cyclic coordinate
#                     descent, but refuse expressions with more exponent slots
#                     than this (cost grows as slots * grid * passes).
#   MAX_CD_PASSES   : coordinate-descent sweeps before giving up on further
#                     improvement.
# Same idea for the NONLINEAR fallback: pin each intpower exponent to an
# integer and fit the rest, rather than letting least_squares chase a discrete
# parameter through a continuous slot.  Without this a power over a directional
# leaf fits its exponent to the local slope and lands on 1.
MAX_NL_GRID     = 36
MAX_GRID_COMBOS = 1600
MAX_EXP_SLOTS   = 40
MAX_CD_PASSES   = 3


def _energy_contexts(target_dof, max_traj, horizon):
    """Precompute the per-trajectory energy context shared by both the fast
    (linear/VARPRO) and the slow (nonlinear) constant fitters.

    The target DOF is fit to its **own** work-energy balance only::

        integral(vel_t * a_target) ~= 1/2 (vel_t^2 - vel_t(0)^2)

    which is an exact per-DOF kinematic identity (coupling forces are already
    contained in ``a_target``).  The other DOFs are therefore *not* mixed into
    the balance: doing so would let their prediction errors contaminate this
    DOF's constants and would let a high-KE DOF swamp a low-KE one.

    Returns ``(contexts, ym_t, ys_t)`` where each context is
    ``(t_sl, cols, vel_t, dke, scale, acc_t, acc_scale)``.  ``acc_t`` is the
    MEASURED acceleration of the target DOF (or ``None`` if unavailable) and
    is used only when ``W_ACC > 0``.
    """
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return None, 0.0, 0.0
    y_mean, y_std = NORM_STATS[2], NORM_STATS[3]

    contexts = []
    for traj in RAW_TRAJECTORIES[:_resolve_max_traj(max_traj)]:
        t_sl, st_sl = _slice_traj(traj, horizon)
        if t_sl is None:
            continue
        feats = _feats(st_sl)
        vel_t = st_sl[2 * target_dof + 1, :]
        # Per-DOF balance: no contribution from the other DOFs.
        dke   = 0.5 * (vel_t ** 2 - vel_t[0] ** 2)
        # Normalise by this DOF's own KE so a low-velocity DOF is weighted the
        # same as a high-velocity one.
        scale = _ro._scale(dke) if ENERGY_NORMALIZE else 1.0
        cols = [feats[k, :] for k in range(N_VARS)]
        accs_sl = _slice_accs(traj, horizon)
        if accs_sl is not None:
            acc_t = accs_sl[target_dof, :]
            acc_scale = _ro._scale(acc_t)
        else:
            acc_t, acc_scale = None, 1.0
        contexts.append((t_sl, cols, vel_t, dke, scale, acc_t, acc_scale))

    ym_t, ys_t = float(y_mean[target_dof]), float(y_std[target_dof])
    return contexts, ym_t, ys_t


def _parse_tree(tau):
    """Parse ``tau`` into a nested node tree, assigning const slots in the same
    pre-order the evaluator/compiler consume them (power ops consume their
    amplitudes then exponent *before* their child)."""
    pos = [0]; slot = [0]

    def parse():
        if pos[0] >= len(tau):
            return {'op': 'const_one'}
        tok = tau[pos[0]]; pos[0] += 1
        if tok in DIR_LEAF_SET:
            cols  = list(LEAF_CHANNELS[tok])
            slots = list(range(slot[0], slot[0] + len(cols)))
            slot[0] += len(cols)
            return {'op': 'dirleaf', 'tok': tok, 'cols': cols, 'coeffs': slots}
        if tok in VAR_SET:
            s = slot[0]; slot[0] += 1
            return {'op': 'var', 'col': COL_MAP[tok], 'coeff': s}
        if tok == 'const':
            s = slot[0]; slot[0] += 1
            return {'op': 'const', 'val': s}
        if tok == INTPOWER_OP:
            s0 = slot[0]; slot[0] += 1     # amplitude       c
            s1 = slot[0]; slot[0] += 1     # exponent        n
            child = parse()
            return {'op': tok, 'coeff': s0, 'exp': s1, 'child': child}
        if tok in POWER_OPS:
            s0 = slot[0]; slot[0] += 1     # even amplitude  c1
            s1 = slot[0]; slot[0] += 1     # odd amplitude   c2
            s2 = slot[0]; slot[0] += 1     # exponent        p
            child = parse()
            return {'op': tok, 'c1': s0, 'c2': s1, 'exp': s2, 'child': child}
        if tok in TRANSCENDENTAL_OPS:
            s0 = slot[0]; slot[0] += 1     # cos amplitude  c1
            s1 = slot[0]; slot[0] += 1     # sin amplitude  c2
            s2 = slot[0]; slot[0] += 1     # decay rate     a
            s3 = slot[0]; slot[0] += 1     # frequency      b
            child = parse()
            return {'op': 'blend', 'c1': s0, 'c2': s1,
                    'a': s2, 'b': s3, 'child': child}
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                children.append(parse())
            if pos[0] < len(tau):
                pos[0] += 1
            return {'op': tok, 'children': children}
        return {'op': 'const_one'}

    root = parse()
    return root, slot[0]


def _expand_monomials(node):
    """Distribute products over sums into a flat list of monomials, or ``None``
    if the structure isn't linear-in-coefficients friendly.

    Supported factors: bare variables and power ops whose child is a **bare
    variable** (the documented VARPRO applicability boundary).  Anything else
    — ``sub``, powers of sums/products — returns ``None`` so the caller falls
    back to the nonlinear optimiser instead of silently mis-fitting.
    """
    op = node['op']
    if op == 'var':
        return [{'factors': [('var', node['col'])],
                 'amp': {node['coeff']}, 'exps': set()}]
    if op == 'dirleaf':
        # A directional leaf is a SUM of singly-weighted channels, so at degree
        # one it is linear in its own weights and the closed form still applies
        # -- exactly as if the term had been written ``add(x1, x3, ...)``.  Under
        # a power op the weights appear nonlinearly; the power-op branches below
        # reject any non-``var`` child, which routes those to the optimiser.
        return [{'factors': [('var', c)], 'amp': {sl}, 'exps': set()}
                for c, sl in zip(node['cols'], node['coeffs'])]
    if op == 'const':
        return [{'factors': [], 'amp': {node['val']}, 'exps': set()}]
    if op == INTPOWER_OP or op in POWER_OPS or op == 'blend':
        ch = node['child']
        if ch.get('op') != 'var':
            return None          # op of a non-variable subtree -> nonlinear
    if op == INTPOWER_OP:
        # ONE monomial, the plain u^n: a single amplitude, so unlike ``power``
        # there is no other-parity twin for the fit to hand a nonzero value.
        return [{'factors': [(INTPOWER_OP, ch['col'], node['exp'])],
                 'amp': {node['coeff']}, 'exps': {node['exp']}}]
    if op in POWER_OPS:
        # TWO monomials sharing ONE exponent slot: the even part |u|^p with
        # amplitude c1, and the odd part sgn(u)|u|^p with amplitude c2.  Both
        # are existing factor kinds, so ``_eval_monomial`` needs no change --
        # 'abspower'/'sgnpower' are now internal names for even/odd, not tokens.
        col, e = ch['col'], node['exp']
        return [{'factors': [('abspower', col, e)],
                 'amp': {node['c1']}, 'exps': {e}},
                {'factors': [('sgnpower', col, e)],
                 'amp': {node['c2']}, 'exps': {e}}]
    if op == 'blend':
        # Two monomials sharing BOTH nonlinear slots: e^(a u) cos(b u) with
        # amplitude c1 and e^(a u) sin(b u) with amplitude c2.  Given (a, b) the
        # term is linear in the amplitudes, so the closed form still applies --
        # which is why blend costs like an exponent rather than like a fit.
        col, sa, sb = ch['col'], node['a'], node['b']
        return [{'factors': [('blendcos', col, sa, sb)],
                 'amp': {node['c1']}, 'exps': {sa, sb}},
                {'factors': [('blendsin', col, sa, sb)],
                 'amp': {node['c2']}, 'exps': {sa, sb}}]
    if op == 'add':
        out = []
        for ch in node['children']:
            sub = _expand_monomials(ch)
            if sub is None:
                return None
            out.extend(sub)
            if len(out) > MAX_MONOMIALS:
                return None
        return out
    if op == 'mul':
        out = [{'factors': [], 'amp': set(), 'exps': set()}]
        for ch in node['children']:
            sub = _expand_monomials(ch)
            if sub is None:
                return None
            merged = []
            for a in out:
                for b in sub:
                    merged.append({'factors': a['factors'] + b['factors'],
                                   'amp': a['amp'] | b['amp'],
                                   'exps': a['exps'] | b['exps']})
                    if len(merged) > MAX_MONOMIALS:
                        return None
            out = merged
        return out
    # sub / unknown -> not supported
    return None


def _eval_monomial(mono, cols, exp_val, n):
    """Basis value of a monomial with its single free amplitude set to 1."""
    phi = np.ones(n)
    for f in mono['factors']:
        if f[0] == 'var':
            phi = phi * cols[f[1]]
        elif f[0] in ('blendcos', 'blendsin'):
            # ('blendcos'|'blendsin', col, a_slot, b_slot) -- the cos and sin
            # branches of one ``blend`` op, sharing both nonlinear slots.
            kind, col, sa, sb = f
            u  = cols[col]
            av = float(np.clip(exp_val[sa], *BLEND_A_BOUND))
            bv = float(np.clip(exp_val[sb], *BLEND_B_BOUND))
            e  = np.clip(np.exp(av * u), -1e15, 1e15)
            phi = phi * (e * (np.sin(bv * u) if kind == 'blendsin'
                              else np.cos(bv * u)))
        elif f[0] == INTPOWER_OP:
            # (INTPOWER_OP, col, exp_slot) -- the plain u^n, integer exponent.
            _kind, col, slot = f
            phi = phi * np.clip(_intpower_np(cols[col], exp_val[slot]),
                                -1e15, 1e15)
        else:
            # ('abspower'|'sgnpower', col, exp_slot) -- the EVEN and ODD parts of
            # one ``power`` op.  These are internal factor names now, not tokens.
            kind, col, slot = f
            a = cols[col]
            p = _clip_exp(exp_val[slot])
            base = np.clip(np.abs(a) ** p, -1e15, 1e15)
            phi = phi * (np.sign(a) * base if kind == 'sgnpower' else base)
    return np.where(np.isfinite(phi), phi, 0.0)


def _linear_plan(tau):
    """``(monos, matched, exp_slots, slot_kind)`` for the closed-form fit of
    ``tau``: its monomials, the amplitude slot each one owns alone, the sorted
    exponent-like slots and the factor kind of each; or None when ``tau`` is
    not linear in its amplitudes given those slots."""
    try:
        root, _nc = _parse_tree(tau)
        monos = _expand_monomials(root)
    except Exception:
        return None
    if not monos or len(monos) > MAX_MONOMIALS:
        return None

    # Each monomial needs a coefficient slot unique to it (so freezing the rest
    # to 1 reproduces independent amplitudes); otherwise it isn't separable.
    amp_sets = [m['amp'] for m in monos]
    matched = []
    for j, amp in enumerate(amp_sets):
        excl = amp.difference(*amp_sets[:j], *amp_sets[j + 1:])
        if not excl:
            return None
        matched.append(min(excl))

    exp_slots = sorted(set().union(*[m['exps'] for m in monos]))
    # which power op owns each exponent slot (drives the grid type)
    slot_kind = {}
    for m in monos:
        for f in m['factors']:
            if f[0] == 'var':
                continue
            if len(f) == 4:                      # blend: two nonlinear slots
                slot_kind[f[2]] = 'blend_a'
                slot_kind[f[3]] = 'blend_b'
            else:
                slot_kind[f[2]] = f[0]
    return monos, matched, exp_slots, slot_kind


# ── Fit rows ────────────────────────────────────────────────────────────────
# Quadrature accuracy requires INTEGRATING on the fine time grid, but the
# least-squares fit of ~M coefficients does not need every integrated row.
# All cumtrapz run on the full grid; only the rows handed to lstsq are strided
# down to <= MAX_FIT_ROWS.
#
# BLEND: the fit objective must match the SCORING objective, or the search
# ranks candidates by a criterion their constants were never optimised for.
# Both objectives are linear in the coefficients given the exponents:
#
#   energy :  (dKE - int(v*ym)) / s_e  ~=  sum_j c_j * ys*int(v*phi_j)/s_e
#   accel  :  (a_meas - ym)     / s_a  ~=  sum_j c_j * ys*phi_j       /s_a
#
# so the blend is one stacked least-squares problem with the energy rows
# scaled by sqrt(1-w) and the acceleration rows by sqrt(w).  Both targets are
# pre-normalised to unit-ish variance (by std(dKE) and std(a_meas)
# respectively), which is what makes w meaningful as a weight rather than an
# arbitrary units conversion.  The acceleration block needs no cumtrapz, so
# w=1 is strictly cheaper than w=0.
def _fit_setup(contexts):
    """``(use_e, use_a, sq_e, sq_a, stride)``: which row blocks the fit
    stacks, the sqrt weight of each and the row stride."""
    w = W_ACC
    use_e = w < 1.0
    use_a = w > 0.0 and all(c[5] is not None for c in contexts)
    if not use_a:
        w, use_e, use_a = 0.0, True, False
    n_pts = sum(len(c[2]) for c in contexts)
    return (use_e, use_a, np.sqrt(1.0 - w), np.sqrt(w),
            max(1, int(np.ceil(n_pts / MAX_FIT_ROWS))))


def _target_rows(ctx, ym, setup):
    """One context's least-squares target: energy block, then acceleration."""
    t_sl, _cols, vel_t, dke, scale, acc_t, acc_scale = ctx
    use_e, use_a, sq_e, sq_a, stride = setup
    parts = []
    if use_e:
        parts.append(sq_e * ((dke - _cumtrapz(vel_t * ym, t_sl))
                             / scale)[::stride])
    if use_a:
        parts.append(sq_a * ((acc_t - ym) / acc_scale)[::stride])
    return np.concatenate(parts)


def _basis_rows(ctx, phi, ys, setup):
    """Basis ``phi``'s design column in one context.

    It must be stacked EXACTLY as :func:`_target_rows`: energy block then
    acceleration block, each carrying the sqrt weight applied to its half of
    the target.  Weighting only the target (and not the design matrix) does not
    weight the problem at all -- it just rescales the residual.  a_phys =
    ym + ys*pred, so the acceleration column is simply ys*phi / std(a_meas):
    no integration, which is why w=1 is cheaper per candidate than w=0.
    """
    t_sl, _cols, vel_t, _dke, scale, _acc_t, acc_scale = ctx
    use_e, use_a, sq_e, sq_a, stride = setup
    parts = []
    if use_e:
        parts.append(sq_e * (ys * _cumtrapz(vel_t * phi, t_sl)
                             / scale)[::stride])
    if use_a:
        parts.append(sq_a * (ys * phi / acc_scale)[::stride])
    return np.concatenate(parts)


def _varpro(design, b, exp_slots, slot_kind, limits=None):
    """Variable projection: given the exponent-like slot values ``ev`` the fit
    is the linear least-squares ``design(ev) @ coef ~= b``.  The slots are
    searched on their grids -- exhaustively up to ``MAX_GRID_COMBOS``
    combinations, by coordinate descent beyond that, up to ``MAX_EXP_SLOTS``
    slots -- and then polished.  ``slot_kind[s]`` picks slot ``s``'s grid and
    bound, which ``limits[s] = (lo, hi)`` narrows.  Returns ``(coef, ev)``, or
    ``(None, None)``.
    """
    limits = limits or {}

    def _solve(ev):
        A = design(ev)
        if not np.all(np.isfinite(A)):
            return None, None, np.inf
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = A @ coef - b
        return coef, res, float(res @ res)

    # ── no exponents: a single exact least-squares solve ────────────────────
    if not exp_slots:
        return _solve({})[0], {}

    if len(exp_slots) > MAX_EXP_SLOTS:
        return None, None

    # ── exponent search, then polish continuous slots (VARPRO) ──────────────
    # Each slot is searched on its own kind's grid (``exp_slot_grid``):
    # ``intpower`` on the exact integer grid (the fit is then *exact* per
    # combination, and it is never polished -- see below), ``power`` on
    # POWER_EXP_GRID (its factor kinds are the internal 'abspower'/'sgnpower'
    # names), ``blend`` on its (a, b) grids.  The NONLINEAR fallback in
    # ``optimise_consts_energy`` reads the same grids, so the two paths agree.
    # Negative exponents are reachable from neither path by design: u^-n and
    # |u|^-p are singular at the origin.
    def _slot_grid(s):
        lo, hi = limits.get(s, (-np.inf, np.inf))
        return [v for v in exp_slot_grid(slot_kind.get(s)) if lo <= v <= hi]

    def _bound(s):
        lo, hi = exp_slot_bound(slot_kind.get(s))
        a, z = limits.get(s, (lo, hi))
        return max(lo, a), min(hi, z)

    grids = [_slot_grid(s) for s in exp_slots]
    if not all(grids):
        return None, None
    n_combos = math.prod(len(g) for g in grids)

    if n_combos <= MAX_GRID_COMBOS:
        # Exhaustive over the product grid: globally optimal on the grid.
        best_ev, best_cost = None, np.inf
        for combo in _iproduct(*grids):
            ev = {s: float(combo[i]) for i, s in enumerate(exp_slots)}
            _c, _r, cost = _solve(ev)
            if cost < best_cost:
                best_cost, best_ev = cost, dict(ev)
    else:
        # ── Cyclic coordinate descent over exponent slots ───────────────────
        # The product grid explodes (5 power slots = 5^5 = 3125 combos),
        # but the fit is LINEAR given the exponents, so sweeping one slot at a
        # time costs sum(len(grid)) per pass instead of prod(len(grid)).  This
        # matters more than it looks: the expanded-cube truth structure
        # (x1^3, x3^3, x1^2*x3, x1*x3^2) has FOUR intpower slots, so under the
        # old `len(exp_slots) > 2 -> return None` rule it was pushed onto the
        # multi-start nonlinear optimiser and systematically under-fit — the
        # exact same structure scored 0.925 fitted that way and 0.999 fitted
        # exactly.  Coordinate descent is not guaranteed globally optimal on
        # the grid, but it is enormously better than handing the whole problem
        # to a nonlinear solver.
        best_ev = {}
        for s in exp_slots:
            g = _slot_grid(s)
            best_ev[s] = float(np.median(g))  # 1.5 on POWER_EXP_GRID, 2 on INTPOWER_GRID
        _c, _r, best_cost = _solve(best_ev)
        for _pass in range(MAX_CD_PASSES):
            improved = False
            for s in exp_slots:
                cur = best_ev[s]
                for v in _slot_grid(s):
                    if float(v) == cur:
                        continue
                    trial = dict(best_ev); trial[s] = float(v)
                    _c, _r, cost = _solve(trial)
                    if cost < best_cost - 1e-15:
                        best_cost, best_ev, improved = cost, trial, True
            if not improved:
                break
    if best_ev is None:
        return None, None

    # Every ``power`` exponent gets a least-squares polish after the grid search,
    # so a fitted exponent is NOT restricted to grid points -- it can land on
    # 1.3 or 2.7, which the grid alone cannot reach.  What keeps that safe is the
    # BOUND, not the grid: the polish is clipped to [MIN_EXP, MAX_EXP] = [0, 3],
    # and the runaway measured under noise (13.4 on duffing, 14.7 on
    # cubic_coupled at 5% NSR) needed room above 3 to happen.
    #
    # ``intpower`` exponents are NOT polished.  Their grid is exhaustive, and
    # the polish is exactly what moves an integer exponent off the integer --
    # duffing's cube polishes from 3 to 2.9998 -- which is the whole difference
    # between ``intpower`` and ``power``.
    cont_slots = [s for s in exp_slots if slot_kind.get(s) != INTPOWER_OP]
    if cont_slots:
        # PER-KIND bounds.  A shared [MIN_EXP, MAX_EXP] silently clamps a
        # blend's decay rate to be non-negative, which forbids every decaying
        # oscillation -- the damped-oscillator truth came back with a = 0.000
        # pinned against the lower bound.
        bnds = [_bound(s) for s in cont_slots]

        def _resid(p):
            ev = dict(best_ev)
            for i, s in enumerate(cont_slots):
                ev[s] = float(np.clip(p[i], *bnds[i]))
            _c, r, _cost = _solve(ev)
            return r if r is not None else np.full(len(b), 1e6)

        try:
            # diff_step MUST exceed the column caches' key granularity.  They
            # round exponent values to 6 decimals, so least_squares' default
            # step of ~1.5e-8 lands on the SAME key, the perturbed solve reuses
            # the cached column, the residual comes back bitwise identical, the
            # numerical gradient is exactly zero, and the polish terminates at
            # its starting point.
            rr = least_squares(_resid,
                               np.array([best_ev[s] for s in cont_slots]),
                               bounds=([b_[0] for b_ in bnds],
                                       [b_[1] for b_ in bnds]),
                               diff_step=1e-3, max_nfev=40)
            for i, s in enumerate(cont_slots):
                best_ev[s] = float(np.clip(rr.x[i], *bnds[i]))
        except Exception:
            pass

    return _solve(best_ev)[0], best_ev


def _fit_consts_linear(tau, contexts, ym_t, ys_t, n_consts):
    """Closed-form / VARPRO (:func:`_varpro`) constant fit of ``tau`` to one
    DOF's ``contexts``.

    Returns a consts list of length ``n_consts`` on success, or ``None`` to
    signal the caller should fall back to the nonlinear optimiser.
    """
    plan = _linear_plan(tau)
    if plan is None:
        return None
    monos, matched, exp_slots, slot_kind = plan
    setup = _fit_setup(contexts)
    b_full = np.concatenate([_target_rows(c, ym_t, setup) for c in contexts])

    # ── Integrated-column cache ─────────────────────────────────────────────
    # A monomial's column depends only on the exponent slots IT contains
    # (usually one), so cache per (context, monomial, relevant slot values):
    # a two-slot 5x5 POWER_EXP_GRID search then costs 5 integrations per
    # single-slot monomial instead of 25.
    col_cache = {}

    def _column(ci, j, exp_val):
        m = monos[j]
        key = (ci, j, tuple(sorted((s, round(float(exp_val[s]), 6))
                                   for s in m['exps'])))
        if key not in col_cache:
            ctx = contexts[ci]
            col_cache[key] = _basis_rows(
                ctx, _eval_monomial(m, ctx[1], exp_val, len(ctx[2])),
                ys_t, setup)
        return col_cache[key]

    def _design(exp_val):
        return np.vstack([
            np.column_stack([_column(ci, j, exp_val)
                             for j in range(len(monos))])
            for ci in range(len(contexts))])

    coef, ev = _varpro(_design, b_full, exp_slots, slot_kind)
    if coef is None or not np.all(np.isfinite(coef)):
        return None
    consts = [1.0] * n_consts
    for j, s in enumerate(matched):
        consts[s] = float(coef[j])
    for s, v in ev.items():
        consts[s] = float(v)
    return consts


def optimise_consts_energy(tau, target_dof, max_traj=None, horizon=None,
                           max_nfev=150):
    """Fit ``tau``'s constants to minimise the target DOF's **own** running
    energy residual ``| integral(vel_t * a_target) - 1/2 (vel_t^2 - vel_t0^2) |``.

    The per-DOF work-energy relation is an exact kinematic identity, so each DOF
    is fit independently: the other DOFs are not mixed into this DOF's
    balance, so their errors cannot contaminate its constants.
    ``max_traj=None`` uses the configured ``MAX_TRAJ`` — the same trajectories
    the reward scores on.

    Fast path: for expressions that are linear in their coefficients given the
    exponents, the fit is a closed-form least-squares / variable projection;
    ``intpower`` exponents are searched on the exact integer grid and never
    polished, ``power`` exponents are searched on ``POWER_EXP_GRID`` and then
    polished within [MIN_EXP, MAX_EXP].
    Anything the fast path can't handle — powers of non-variable subtrees,
    over-large expansions — falls back to the general multi-start nonlinear
    optimiser below.
    """
    n_consts = count_total_consts(tau)
    if n_consts == 0:
        return []
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return [1.0] * n_consts

    contexts, ym_t, ys_t = _energy_contexts(target_dof, max_traj, horizon)
    if not contexts:
        return [1.0] * n_consts

    # Fast path: closed-form / VARPRO.
    fast = _fit_consts_linear(tau, contexts, ym_t, ys_t, n_consts)
    if fast is not None:
        return fast

    # Fallback: general nonlinear least-squares (multi-start).
    code, _n_c, grid_exp_indices, grid_exp_kinds = _build_param_code(tau)
    sig = ','.join(VARIABLES)
    try:
        fn_param = eval(f'lambda c,{sig}: {code}',
                        {'np': np, '_itp': _intpower_np})
    except Exception:
        return [1.0] * n_consts

    # Same rows and blend as the fast path.  If the fallback minimised the
    # pure energy residual while the reward scored a blend, constants would be
    # optimised for a different objective than the one they are judged by —
    # the exact fit/score mismatch that makes candidates look worse than they
    # are.
    _use_e, _use_a, _sq_e, _sq_a, _stride = _fit_setup(contexts)

    def residuals(c):
        parts = []
        for (t_sl, cols, vel_t, dke, scale, acc_t, acc_scale) in contexts:
            n_bad = len(vel_t[::_stride]) * (_use_e + _use_a)
            try:
                with np.errstate(invalid='ignore', over='ignore'):
                    pred_n = fn_param(c, *cols)
                pred_n = np.asarray(pred_n, dtype=np.float64)
                if pred_n.ndim == 0:
                    pred_n = np.full(len(vel_t), float(pred_n))
                if not np.all(np.isfinite(pred_n)):
                    parts.append(np.ones(n_bad) * 1e6)
                    continue
            except Exception:
                parts.append(np.ones(n_bad) * 1e6)
                continue
            acc  = ym_t + ys_t * pred_n
            if _use_e:
                # ``+ 0.0`` turns a -0.0 into +0.0, as adding the old
                # all-zero ``cum_other`` did: the residuals stay bitwise equal
                cum = _cumtrapz(vel_t * acc, t_sl) + 0.0
                parts.append(_sq_e * ((cum - dke) / scale)[::_stride])
            if _use_a:
                parts.append(_sq_a * ((acc - acc_t) / acc_scale)[::_stride])
        return np.concatenate(parts)

    # Per-KIND bounds.  A power's exponent must not go below 0 (|u|^p is
    # singular at the origin) while a blend's decay rate must be allowed
    # negative -- that is what decay is.  A single shared bound would silently
    # forbid every decaying oscillation.
    if grid_exp_indices:
        lb = np.full(n_consts, -np.inf); ub = np.full(n_consts, np.inf)
        for pi, kd in zip(grid_exp_indices, grid_exp_kinds):
            lb[pi], ub[pi] = exp_slot_bound(kd)
        opt_method = 'trf'; opt_bounds = (lb, ub)
    else:
        opt_method = 'lm'; opt_bounds = (-np.inf, np.inf)

    # scale of the energy target sets the coefficient scale
    e_scale = 1.0
    for (_t, _cols, vel_t, dke, _s, _at, _as) in contexts:
        e_scale = max(e_scale, float(np.std(dke)) + 1e-9)

    def _make_smart(cs, pv, direction='unit'):
        """Initial const vector in pre-order.

        coefficient=cs for variables and consts; (amplitude=cs, exponent=pv)
        for ``intpower`` and (0, cs, pv) for ``power``; a pure oscillation for
        ``blend``; and for a directional leaf, a whole ``N_DOF``-long weight
        block seeded by ``direction``:

            'unit'   (1, 0, 0, ...)   the leaf starts as a bare variable
            'sum'    (1, 1, 1, ...)   every channel equally
            'diff'   (1, -1, 1, ...)  the relative coordinate

        Those three are the canonical directions, and 'diff' is the one a
        coupled system's ``(q_1 - q_2)^n`` needs.  This function MUST cover
        every token that owns const slots -- a length mismatch silently drops
        the whole initialiser in the loop below.
        """
        init = []
        for t in tau:
            if t in DIR_LEAF_SET:
                k = CONST_SLOTS[t]
                if direction == 'sum':
                    init.extend([1.0] * k)
                elif direction == 'diff':
                    init.extend([1.0 if i % 2 == 0 else -1.0 for i in range(k)])
                else:
                    init.extend([1.0] + [0.0] * (k - 1))
            elif t == INTPOWER_OP:
                init.extend([cs, pv])
            elif t in POWER_OPS:
                # (even amplitude, odd amplitude, exponent).  The odd part
                # carries the amplitude by default because an odd restoring
                # force is the common case; the even part starts at zero.
                init.extend([0.0, cs, pv])
            elif t in TRANSCENDENTAL_OPS:
                # (cos amplitude, sin amplitude, decay rate, frequency).  Start
                # as a pure undamped oscillation, the most common transcendental
                # in a mechanical restoring force, and let the grid move it.
                init.extend([cs, 0.0, 0.0, 1.0])
            elif t in CONST_SLOTS:
                init.extend([cs] * CONST_SLOTS[t])
        return np.array(init, dtype=float)

    # Three starts (the first alone per pinned grid point below), seeding the
    # other two canonical directions first when there is a directional leaf.
    if any(t in DIR_LEAF_SET for t in tau):
        inits = [_make_smart( e_scale, 3.0, 'diff'),
                 _make_smart(-e_scale, 3.0, 'diff'),
                 _make_smart( e_scale, 3.0, 'sum')]
    else:
        inits = [_make_smart(e_scale, 1.0), _make_smart(e_scale, 2.0),
                 _make_smart(e_scale, 3.0)]
    for init in inits:
        # per-kind, like the bounds: an init outside its slot's bound makes
        # least_squares reject the start as infeasible
        for pi, kd in zip(grid_exp_indices, grid_exp_kinds):
            init[pi] = float(np.clip(init[pi], *exp_slot_bound(kd)))

    # Grid the exponent slots instead of optimising through them: pin each to a
    # grid point of its kind with a hairline bound and fit everything else.
    # This is the same exhaustive-grid guarantee the closed-form path gives, and
    # it is what makes ``intpower(dleaf)`` land on exponent 3 rather than on the
    # local slope.
    per_slot = [exp_slot_grid(kd) for kd in grid_exp_kinds]
    n_combos = 1
    for g in per_slot:
        n_combos *= len(g)
    combos = [None]
    if grid_exp_indices and n_combos <= MAX_NL_GRID:
        combos = list(_iproduct(*per_slot))

    best_cost, best_c = np.inf, None
    for combo in combos:
        if combo is None:
            lb_c, ub_c, method_c = opt_bounds[0], opt_bounds[1], opt_method
        else:
            lb_c = np.array(opt_bounds[0], dtype=float)
            ub_c = np.array(opt_bounds[1], dtype=float)
            for pi, n in zip(grid_exp_indices, combo):
                lb_c[pi], ub_c[pi] = n - 1e-6, n + 1e-6
            method_c = 'trf'
        # With the exponent pinned there is far less to search, so one
        # initialiser per grid point and a fraction of the evaluation budget.
        # Measured on ``intpower(dleaf)`` over ``cubic_coupled``: the answer
        # (exponent 3, direction -(q1-q2) to 4 decimals) is fully recovered at
        # 30 evaluations per grid point and degrades below 20, while the
        # default 150 costs 2.3x the time for an identical result.
        use  = inits[:1] if combo is not None else inits
        nfev = max(30, int(max_nfev) // 4) if combo is not None else max_nfev
        for init in use:
            init = np.asarray(init, dtype=float).copy()
            if combo is not None:
                for pi, n in zip(grid_exp_indices, combo):
                    init[pi] = float(n)
            try:
                res = least_squares(residuals, init, method=method_c,
                                    bounds=(lb_c, ub_c), max_nfev=nfev)
                if res.cost < best_cost:
                    best_cost, best_c = res.cost, res.x.tolist()
            except Exception:
                pass
    if best_c is None:
        return [1.0] * n_consts
    # A pinned slot comes back within its 1e-6 hairline (3.000000000000105), and
    # an unpinned one wherever it started.  ``_itp`` already evaluated both as
    # the snapped integer, so storing that integer changes no value -- it only
    # makes the constant say what the fit actually used.
    for pi, kd in zip(grid_exp_indices, grid_exp_kinds):
        if kd == INTPOWER_OP:
            best_c[pi] = float(_snap_int_power(best_c[pi]))
    return best_c


def get_traj_horizon(epoch, n_epochs, t_end):
    """Time horizon the constant fit and the reward integrate over at ``epoch``.

    A growing-horizon curriculum (30% of the record at epoch 0, reaching 100%
    after the first fifth of training) is deliberately switched off: every
    epoch scores the full record, so rewards are comparable across epochs.
    """
    return t_end


def structural_novelty(tau, buffer, n=3):
    if not buffer or len(tau) < n: return 1.0
    tau_ngrams = set(tuple(tau[i:i + n]) for i in range(len(tau) - n + 1))
    if not tau_ngrams: return 1.0
    max_overlap = 0.0
    for entry in buffer:
        buf_tau = entry[1]
        if len(buf_tau) < n: continue
        buf_ngrams = set(tuple(buf_tau[i:i + n]) for i in range(len(buf_tau) - n + 1))
        overlap = len(tau_ngrams & buf_ngrams) / max(len(tau_ngrams), 1)
        if overlap > max_overlap: max_overlap = overlap
    return 1.0 - max_overlap


# ── Energy worker (ProcessPool) ─────────────────────────────────────────────
def init_energy_worker(n_dof, var_names, norm_stats, raw_trajs,
                       energy_normalize=True, max_traj=5, w_acc=None,
                       directional_leaves=False, transcendental=False,
                       reward='energy', sim_window=None,
                       sim_weights=(1.0, 0.0, 0.0, 0.0), trial_decay=1.0,
                       relative_leaves=False):
    # Every grammar flag MUST match the parent's: a worker with a different
    # token table computes a different ``count_total_consts`` for the same tau
    # and silently mis-fits every candidate it is handed.  The reward settings
    # likewise, or the pool scores by a different reward than the parent logs.
    configure_grammar(n_dof, var_names, directional_leaves, transcendental,
                      relative_leaves)
    set_problem_data(norm_stats, raw_trajs, energy_normalize, max_traj, w_acc)
    set_reward(reward, sim_window, sim_weights, trial_decay)


def energy_worker(args):
    """
    args = (target_dof, cand_tau, elite_exprs, horizon)
      elite_exprs : length-N list of (tau, consts) or None.  ACCEPTED FOR API
                    COMPATIBILITY AND IGNORED — see below.

    Fits the candidate's constants to the target DOF's own work-energy balance,
    then scores that DOF **in isolation** with the configured reward
    (:func:`candidate_reward` -- work-energy, or forward simulation).  Returns
    (target_dof, reward, cand_tau, cand_consts) or None.

    Why the partner elites are not mixed in
    ---------------------------------------
    ``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an exact per-DOF
    kinematic identity: coupling forces are already inside ``a_d``, so DOF d's
    balance is complete without any knowledge of the other DOFs.  The constant
    fit already exploited this (``_energy_contexts`` fits each DOF alone).
    Averaging the partner's residual into the score therefore adds no
    within-epoch information — it is a monotone shift shared by every candidate
    of this DOF, so rankings are unchanged — while making scores incomparable
    ACROSS epochs, because the partner elite changes underneath them.  That
    single defect was the source of three separate symptoms: the epoch-0
    asymmetry (a candidate scored without a partner keeps the elite slot
    forever), stale buffer entries driving unfair pruning and R_alpha drift,
    and ``best_r`` being dethroned purely because the partner improved.
    Scoring in isolation removes all three at once.  :func:`coupled_worker`
    is the deliberate exception, for the share of a batch that the trainer's
    ``sim_coupling`` sends to be simulated with other DOFs' equations.
    """
    target_dof, cand_tau, _elite_exprs, horizon = args
    consts = optimise_consts_energy(cand_tau, target_dof, horizon=horizon)
    exprs  = [None] * N_DOF
    exprs[target_dof] = (cand_tau, consts)
    r = candidate_reward(exprs, horizon=horizon)
    if r <= 1e-6:
        return None
    return (target_dof, r, cand_tau, consts)


def coupled_worker(args):
    """Score candidates simulated with other DOFs' equations
    (``reward='simulation'`` only).

    args = (members, tops, horizon, weights)
      members : list of ``(dof, tau)`` -- the candidates to fit and score, at
                most one per DOF: a row of one joint sample, or one equation
      tops    : ``{dof: (tau, consts)}`` -- each DOF's top equation (its best
                so far); a member's own DOF's entry is ignored
      horizon : as in :func:`energy_worker`
      weights : ``{mode: weight}``, the ways to simulate each member --
                ``'alone'`` on its own, the other DOFs read off the record
                (:func:`simulation_reward`); ``'top'`` together with every
                other DOF's top equation; ``'peer'`` together with the other
                members.  A DOF with nothing to simulate is read off the
                record, so with no tops ``'top'`` is ``'alone'``, and so is
                ``'peer'`` for a lone member.

    Each member's constants are fitted exactly as in :func:`energy_worker` --
    to its own DOF's balance on the measured states -- so only the scoring
    differs.  In a coupled mode each member is scored on its own channel, and
    a set that blows up scores every equation in it 0
    (:func:`coupled_simulation_rewards`).  Returns a list aligned with
    ``members`` of ``(dof, reward, tau, consts, {mode: reward},
    {mode: {dof: (tau, consts)}})`` -- the reward the weighted mean over
    modes, the last the partner equations each mode simulated it with (empty
    for a DOF read off the record) -- or None.

    This gives up the isolation :func:`energy_worker` argues for on purpose:
    a member's score now depends on the equations it is simulated with.  An
    equation that fits against the measured partner but derails the coupled
    set is caught here and nowhere else.
    """
    members, tops, horizon, weights = args
    weights = {m: float(w) for m, w in weights.items() if w > 0.0}
    fitted = [(d, tau, optimise_consts_energy(tau, d, horizon=horizon))
              for d, tau in members]
    alone = {}

    def r_alone(d, tau, consts):
        if d not in alone:
            exprs = [None] * N_DOF
            exprs[d] = (tau, consts)
            alone[d] = simulation_reward(exprs, horizon=horizon)
        return alone[d]

    modes = [{} for _ in fitted]
    if 'peer' in weights and len(fitted) > 1:
        exprs = [None] * N_DOF
        for d, tau, consts in fitted:
            exprs[d] = (tau, consts)
        rs = coupled_simulation_rewards(exprs, [d for d, _t, _c in fitted],
                                        horizon=horizon)
        for k, (d, _t, _c) in enumerate(fitted):
            modes[k]['peer'] = rs[d]
    for k, (d, tau, consts) in enumerate(fitted):
        if 'alone' in weights:
            modes[k]['alone'] = r_alone(d, tau, consts)
        if 'peer' in weights and len(fitted) == 1:
            modes[k]['peer'] = r_alone(d, tau, consts)
        if 'top' in weights:
            exprs = [None] * N_DOF
            for j, top in (tops or {}).items():
                if j != d:
                    exprs[j] = top
            if any(e is not None for e in exprs):
                exprs[d] = (tau, consts)
                modes[k]['top'] = coupled_simulation_rewards(
                    exprs, [d], horizon=horizon)[d]
            else:
                modes[k]['top'] = r_alone(d, tau, consts)
    total = sum(weights.values())
    out = []
    for (d, tau, consts), md in zip(fitted, modes):
        r = sum(w * md[m] for m, w in weights.items()) / total
        used = {}
        for m in weights:
            if m == 'top':
                used[m] = {j: e for j, e in (tops or {}).items() if j != d}
            elif m == 'peer':
                used[m] = {j: (t2, c2) for j, t2, c2 in fitted if j != d}
            else:
                used[m] = {}
        out.append(None if r <= 1e-6 else (d, r, tau, consts, md, used))
    return out


def score_member(d, tau, consts, weights, tops=None, peers=None, horizon=None):
    """DOF ``d``'s equation scored the ways ``weights`` names, its partners
    held fixed: ``'alone'`` against the record, ``'top'`` with ``tops``,
    ``'peer'`` with ``peers`` (each ``{dof: (tau, consts)}``; a DOF's own
    entry is ignored, and a mode with no partner falls back to ``'alone'``).
    The same rewards :func:`coupled_worker` gives, one member at a time.
    Returns ``(reward, {mode: reward})``."""
    weights = {m: float(w) for m, w in weights.items() if w > 0.0}
    md, alone = {}, []

    def r_alone():
        if not alone:
            exprs = [None] * N_DOF
            exprs[d] = (tau, consts)
            alone.append(simulation_reward(exprs, horizon=horizon))
        return alone[0]

    for m in weights:
        if m == 'alone':
            md[m] = r_alone()
            continue
        exprs = [None] * N_DOF
        for j, e in ((tops if m == 'top' else peers) or {}).items():
            if j != d:
                exprs[j] = e
        if any(e is not None for e in exprs):
            exprs[d] = (tau, consts)
            md[m] = coupled_simulation_rewards(exprs, [d], horizon=horizon)[d]
        else:
            md[m] = r_alone()
    total = sum(weights.values())
    return sum(w * md[m] for m, w in weights.items()) / total, md


REFINE_SPAN = 0.5      # simulation tuning moves each amplitude by at most +-50%


def _amplitude_slots(tau):
    """Per constant slot of ``tau`` (pre-order): True if it scales a term -- a
    coefficient, a ``const``, an amplitude -- False if it shapes one (an
    ``intpower`` or ``power`` exponent, a ``blend`` rate), which simulation
    tuning leaves alone so the structure it was found with stays put."""
    _code, n_c, grid_exp_indices, _kinds = _build_param_code(tau)
    mask = np.ones(n_c, dtype=bool)
    mask[grid_exp_indices] = False
    return mask


def refine_worker(args):
    """Tune an equation's amplitudes on the simulation itself
    (``reward='simulation'``).

    args = (dof, tau, consts, weights, tops, peers, horizon, max_evals)

    The constants come from the closed-form fit to the equation error, which
    is cheap but blind to what a free run magnifies: a beat period is a
    DIFFERENCE of two close frequencies, so a 1% coupling error can move it by
    tens of percent.  Starting from the fitted constants, Powell's method
    moves the amplitude slots (:func:`_amplitude_slots`; exponents and rates
    stay) by up to ``REFINE_SPAN`` of their value to maximise the equation's
    own reward -- scored exactly as it was (:func:`score_member`, same modes,
    partners held fixed) -- within ``max_evals`` simulations.  Returns
    ``(dof, reward, tau, consts, {mode: reward}, n_evals)``: the best
    constants seen, which are the given ones unless something scored higher.
    """
    from scipy.optimize import minimize

    d, tau, consts, weights, tops, peers, horizon, max_evals = args
    c0 = np.asarray(consts, dtype=float)
    r0, md0 = score_member(d, tau, list(c0), weights, tops, peers, horizon)
    amp = _amplitude_slots(tau)
    free = np.flatnonzero(amp)
    if (free.size == 0 or len(c0) != len(amp)
            or max_evals <= 0 or r0 <= 1e-6):
        return (d, r0, tau, list(c0), md0, 0)
    base = c0[free]
    scale = np.where(np.abs(base) > 1e-12, base, 1.0)
    best = {'r': r0, 'c': c0, 'md': md0, 'n': 0}

    def residual(x):
        c = c0.copy()
        c[free] = base + scale * x
        r, md = score_member(d, tau, list(c), weights, tops, peers, horizon)
        best['n'] += 1
        if r > best['r']:
            best.update(r=r, c=c, md=md)
        return 1.0 / max(r, 1e-12) - 1.0

    minimize(residual, np.zeros(free.size), method='Powell',
             bounds=[(-REFINE_SPAN, REFINE_SPAN)] * free.size,
             options={'maxfev': int(max_evals), 'xtol': 1e-3, 'ftol': 1e-4})
    return (d, best['r'], tau, [float(v) for v in best['c']], best['md'],
            best['n'])


_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS')


def make_pool(initargs, n_workers):
    """A scoring pool of SINGLE-THREADED workers running
    :func:`init_energy_worker` with ``initargs``; returns ``(pool, saved_env)``
    -- hand both to :func:`close_pool`.

    Left alone, every worker starts numpy's and scipy's OpenBLAS (and torch's
    OpenMP) with one thread per core, and each BLAS thread commits its own
    work buffer.  On Windows, which does not overcommit, a run with three
    such workers died on its first fits with "OpenBLAS error: Memory
    allocation still failed after 10 retries".  They are also slower:
    measured on 90 duffing bags, three workers at the default took 34.5 s,
    three times SLOWER than fitting the same bags serially (11.0 s), because
    3 x 4 BLAS threads fight over 4 cores on small least-squares solves; with
    one BLAS thread each the pool takes 4.4 s.

    The thread count is read when numpy loads, so the workers are SPAWNED --
    fresh interpreters that see the setting -- rather than forked, which would
    inherit this process's already-initialised thread pool.  And the setting
    must hold for the pool's whole life, not just its creation: a spawn pool
    starts workers on demand, and one started after the variables were
    restored came up with 7 threads.  This process's own torch thread count
    is pinned first, so the variables cannot reach its policy updates.
    """
    torch.set_num_threads(torch.get_num_threads())
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: '1' for k in _THREAD_VARS})
    pool = ProcessPoolExecutor(max_workers=n_workers,
                               mp_context=_mp.get_context('spawn'),
                               initializer=init_energy_worker,
                               initargs=initargs)
    return pool, saved


def close_pool(pool, saved):
    """Shut the pool down, then restore the thread variables it needed."""
    pool.shutdown(wait=True, cancel_futures=True)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── J-GRPO ──────────────────────────────────────────────────────────────────
# The objective itself lives in :func:`discover_policy.jgrpo_terms`.  Above this
# many buffer entries a GRPO step subsamples, prioritised in log-residual units
# (see :func:`log_residual_score`) so near-top refinements are not crowded out
# by the compressed r-scale.
MAX_GRPO_BATCH = 32


# ── Denormalisation to physical units ───────────────────────────────────────
PASTE_DIGITS = 12          # significant figures in the paste-ready block
def denormalize_expr(tau, consts, dof, digits=4):
    """Return ``"<accel name> = <physical expression>"`` for DOF ``dof``, its
    constants to ``digits`` significant figures.  Four read well; an equation
    meant to be pasted and simulated wants :data:`PASTE_DIGITS`, since a
    sensitive one drifts visibly within a fraction of a second on constants
    rounded to four."""
    if NORM_STATS is None: return None
    X_mean, X_std, y_mean, y_std = NORM_STATS
    try:
        expr_str = expr_to_sympy_str(tau, consts)
        sym_expr = parse_expr(expr_str, local_dict=_parse_local_dict())

        disp = _var_display_map()
        subs = {}
        for k, v in enumerate(VARIABLES):
            if k < 2 * N_DOF:
                subs[SYMS[v]] = ((sp.Symbol(disp[v]) - float(X_mean[k]))
                                 / float(X_std[k]))
            else:                                   # a relative leaf
                a, b = PAIRS[(k - 2 * N_DOF) // 2]
                subs[SYMS[v]] = ((sp.Symbol(disp[VARIABLES[2 * a + k % 2]])
                                  - sp.Symbol(disp[VARIABLES[2 * b + k % 2]]))
                                 / float(PAIR_STD[k - 2 * N_DOF]))
        sym_expr = sym_expr.subs(subs)
        sym_expr = sp.expand(float(y_mean[dof]) + float(y_std[dof]) * sym_expr)
        accel_name = VAR_NAMES[dof] + 'ddot'
        return f"{accel_name} = {sp.N(sym_expr, digits)}"
    except Exception as e:
        return f"(denorm failed: {e})"