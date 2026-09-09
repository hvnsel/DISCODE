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
DOF ``i``'s balance alone (``_energy_contexts`` sets ``cum_other = 0``), and
candidate scoring likewise evaluates one DOF at a time (see
:func:`energy_worker`).  There is no co-simulation anywhere in the reward path.

``r = 1`` is a perfect energy match.  The reward needs **no forward ODE
integration**: it uses the *measured* velocities together with the predicted
acceleration evaluated on the *measured* states.  Constants are fitted
energetically (least-squares on the running residual), so no point-wise
acceleration target is ever used.

Grammar
-------
Leaves are bare state variables ``x1..x{2N}`` (each carrying an **implicit
leading coefficient**) and free constants.  Powers are the unary operators::

    abspower(u) = c * |u|^p            p in [MIN_EXP, MAX_EXP]  (fitted)
    sgnpower(u) = c * sign(u) * |u|^p  p in [MIN_EXP, MAX_EXP]  (fitted)
    intpower(u) = c * u^n              n integer, |n| <= INTPOWER_MAX (fitted, snapped)

Each power operator consumes **two** constant slots (coefficient, exponent) in
pre-order, *before* its child's constants.

State layout for an N-DOF system::

    state = [q_1, qd_1, q_2, qd_2, ..., q_N, qd_N]     (length 2N)
    x{2d+1} = q_{d+1}      x{2d+2} = qd_{d+1}          (0-indexed d)

Call :func:`configure_grammar` once (with the number of DOFs) before building
any models, and :func:`set_problem_data` once the training data exist.
"""

from __future__ import annotations

from itertools import product as _iproduct

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


DEVICE = torch.device('cpu')

# ── Static grammar constants ────────────────────────────────────────────────
# Power operators in the grammar.  To re-enable the continuous-exponent ops,
# set POWER_OPS = ('abspower', 'sgnpower', 'intpower').  NOTE: a 1-element
# tuple needs the trailing comma — ('intpower') is the *string* 'intpower'
# and silently breaks every `tok in DOUBLE_CONST_TOKS` check.
POWER_OPS    = ('intpower',)
BINARY_OPS   = ['add', 'mul']            # ['add', 'sub', 'mul']
UNARY_OPS    = list(POWER_OPS)
N_ARY_OPS    = ['add', 'sub', 'mul']
CONSTANTS    = ['const']
# Depth 5 (not 4): the root ``add`` of a sum-of-terms costs a whole level, and
# van der Pol's ``mu*(1 - x^2)*xdot`` needs four levels *below* it.  At 4 the
# search provably could not write that truth.  MIN_EXPR_LEN 2 (not 6): at 6 the
# linear oscillator ``add(x1, x2, end)`` (4 tokens) was unreachable too.  The
# term-structured policy applies its own per-term floor via ``min_len``.
MAX_TREE_DEPTH = 5
MIN_EXPR_LEN   = 2

# Exponent limits: continuous exponents (abspower/sgnpower) live in
# [MIN_EXP, MAX_EXP]; intpower exponents are snapped to a nonzero integer in
# [-INTPOWER_MAX, INTPOWER_MAX].
INTPOWER_MAX = 3
MIN_EXP      = 0.5
MAX_EXP      = 7.0

# Tokens that carry an implicit leading coefficient / extra constants.
# COEFF_TOKENS is rebuilt in configure_grammar (it is the variable set).
COEFF_TOKENS      = set()
DOUBLE_CONST_TOKS = set(POWER_OPS)       # coefficient + exponent

_OP_TOKENS         = set(BINARY_OPS + UNARY_OPS)
_FORBIDDEN_NESTING = {
    'add': {'add'}, 'sub': {'add'}, 'mul': {'mul'},
    # nested fitted powers are degenerate ((|u|^p)^q == |u|^(p*q)) — forbid
    **{p: set(POWER_OPS) for p in POWER_OPS},
}

# ── Configurable grammar state (set by configure_grammar) ───────────────────
N_DOF      = 2
N_VARS     = 4
VARIABLES  = ['x1', 'x2', 'x3', 'x4']    # physical state-column names
VAR_SET    = set(VARIABLES)
COL_MAP    = {v: i for i, v in enumerate(VARIABLES)}
ALL_TOKENS = BINARY_OPS + UNARY_OPS + VARIABLES + CONSTANTS + ['end']
N_TOKENS   = len(ALL_TOKENS)
MASK_TOKEN = N_TOKENS
VOCAB_SIZE = N_TOKENS + 1
ARITY      = {}
SYMS       = {}                       # sympy symbols x1..x{2N}
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


# ── Grammar configuration ───────────────────────────────────────────────────
def configure_grammar(n_dof: int, var_names=None):
    """Initialise the global token table for an ``n_dof`` system (2*n_dof vars).

    Leaf tokens are the bare state variables (each with an implicit leading
    coefficient) plus ``const``.  Powers are the unary ops ``abspower`` /
    ``sgnpower`` / ``intpower``, each consuming (coefficient, exponent).
    """
    global N_DOF, N_VARS, VARIABLES, VAR_SET, COL_MAP, COEFF_TOKENS
    global ALL_TOKENS, N_TOKENS, MASK_TOKEN, VOCAB_SIZE, ARITY, SYMS, VAR_NAMES

    N_DOF  = int(n_dof)
    N_VARS = 2 * N_DOF
    VARIABLES = [f'x{i + 1}' for i in range(N_VARS)]
    VAR_SET   = set(VARIABLES)
    COL_MAP   = {v: i for i, v in enumerate(VARIABLES)}
    COEFF_TOKENS = set(VARIABLES)

    ALL_TOKENS = BINARY_OPS + UNARY_OPS + VARIABLES + CONSTANTS + ['end']
    N_TOKENS   = len(ALL_TOKENS)
    MASK_TOKEN = N_TOKENS
    VOCAB_SIZE = N_TOKENS + 1

    ARITY = {}
    for t in BINARY_OPS: ARITY[t] = 2
    for t in UNARY_OPS:  ARITY[t] = 1
    for t in VARIABLES:  ARITY[t] = 0
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
    NORM_STATS       = norm_stats
    RAW_TRAJECTORIES = raw_trajectories
    ENERGY_NORMALIZE = bool(energy_normalize)
    MAX_TRAJ = len(raw_trajectories) if max_traj is None else int(max_traj)
    if w_acc is not None:
        W_ACC = float(np.clip(w_acc, 0.0, 1.0))


def set_energy_options(energy_normalize=True):
    global ENERGY_NORMALIZE
    ENERGY_NORMALIZE = bool(energy_normalize)


def _resolve_max_traj(max_traj):
    return MAX_TRAJ if max_traj is None else int(max_traj)


def set_blend_weight(w_acc):
    """Set the acceleration/energy blend weight (see W_ACC)."""
    global W_ACC
    W_ACC = float(np.clip(w_acc, 0.0, 1.0))


def _resolve_w_acc(w_acc):
    return W_ACC if w_acc is None else float(np.clip(w_acc, 0.0, 1.0))


def tok2idx(t): return ALL_TOKENS.index(t)
def idx2tok(i): return ALL_TOKENS[i]


def _var_display_map():
    """x{2d+1} -> position name, x{2d+2} -> velocity name."""
    disp = {}
    for d in range(N_DOF):
        disp[f'x{2 * d + 1}'] = VAR_NAMES[d]
        disp[f'x{2 * d + 2}'] = VAR_NAMES[d] + 'dot'
    return disp


# ── Numeric power helpers ───────────────────────────────────────────────────
def _snap_int_power(p):
    n = int(round(float(np.clip(p, -INTPOWER_MAX, INTPOWER_MAX))))
    return n if n != 0 else 1


def _intpower_np(a, n):
    n     = _snap_int_power(n)
    n_abs = abs(n)
    sf    = np.sign(a) ** (n_abs % 2)
    return sf * np.abs(a) ** n if n > 0 else sf * (np.abs(a) + 1e-8) ** n


def _clip_exp(p):
    return float(np.clip(float(p), MIN_EXP, MAX_EXP))


def count_total_consts(tau):
    """Constant slots in pre-order: variables and ``const`` take 1 slot,
    power operators take 2 (coefficient, exponent)."""
    n = 0
    for t in tau:
        if t == 'const' or t in VAR_SET:
            n += 1
        elif t in DOUBLE_CONST_TOKS:
            n += 2
    return n


# ── sympy <-> tau conversion ────────────────────────────────────────────────
def expr_to_sympy_str(tau: list, consts: list) -> str:
    const_list = list(consts) if consts else []
    const_idx  = [0]
    pos        = [0]

    def _next_const(default=1.0):
        ci = const_idx[0]; const_idx[0] += 1
        return float(const_list[ci]) if ci < len(const_list) else float(default)

    def node_to_str():
        if pos[0] >= len(tau): return '0'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in VAR_SET:
            v = _next_const()
            return f"(({v})*{tok})"
        if tok == 'const':
            v = _next_const()
            return f"({v})"
        if tok in DOUBLE_CONST_TOKS:
            c = _next_const()
            p = _next_const(2.0)
            child = node_to_str()
            if tok == 'abspower':
                return f"(({c})*((Abs({child}))**({_clip_exp(p)})))"
            if tok == 'sgnpower':
                return f"(({c})*(sign({child})*(Abs({child}))**({_clip_exp(p)})))"
            # intpower
            n = _snap_int_power(p)
            return f"(({c})*(({child})**({n})))"
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                children.append(node_to_str())
            if pos[0] < len(tau): pos[0] += 1
            if tok == 'add': return '(' + ' + '.join(children) + ')'
            if tok == 'sub': return '(' + ' - '.join(children) + ')'
            if tok == 'mul': return '(' + ' * '.join(children) + ')'
        return tok

    return node_to_str()


def sympy_to_tau(expr):
    """Best-effort conversion of a sympy expression into ``(tau, consts)``.

    Handles numbers, bare symbols (with optional numeric coefficient), integer
    powers of a symbol (-> ``intpower``), ``Abs(sym)**p`` (-> ``abspower``),
    ``sign(sym)*Abs(sym)**p`` (-> ``sgnpower``), and Add/Mul of the above.
    Returns ``None`` on anything unsupported.
    """
    tau = []; consts = []

    def _emit_var(sym, coeff):
        name = str(sym)
        if name not in VAR_SET:
            raise ValueError(f"Unknown symbol: {name}")
        tau.append(name); consts.append(float(coeff))

    def _emit_power(op, sym, p, coeff):
        name = str(sym)
        if name not in VAR_SET:
            raise ValueError(f"Unknown symbol: {name}")
        tau.append(op); consts.append(float(coeff)); consts.append(float(p))
        tau.append(name); consts.append(1.0)     # inner variable coefficient

    def _as_signed_abs_power(e):
        """Return ('sgnpower', sym, p) for sign(s)*Abs(s)**p patterns."""
        if isinstance(e, sp.Mul) and len(e.args) == 2:
            a, b = e.args
            if isinstance(a, sp.sign) and isinstance(b, sp.Pow):
                base, ex = b.args
                if (isinstance(base, sp.Abs) and isinstance(ex, sp.Number)
                        and a.args[0] == base.args[0]
                        and isinstance(base.args[0], sp.Symbol)):
                    return ('sgnpower', base.args[0], float(ex))
        return None

    def _as_power(e):
        sap = _as_signed_abs_power(e)
        if sap is not None:
            return sap
        if isinstance(e, sp.Pow):
            base, ex = e.args
            if isinstance(ex, sp.Integer) and isinstance(base, sp.Symbol):
                return ('intpower', base, float(ex))
            if isinstance(ex, sp.Number):
                if isinstance(base, sp.Symbol):
                    return ('intpower', base, float(ex))
                if isinstance(base, sp.Abs) and isinstance(base.args[0], sp.Symbol):
                    return ('abspower', base.args[0], float(ex))
        return None

    def traverse(e, coeff=1.0):
        if isinstance(e, sp.Symbol):
            _emit_var(e, coeff); return
        pw = _as_power(e)
        if pw is not None:
            _emit_power(pw[0], pw[1], pw[2], coeff); return
        if isinstance(e, (sp.Integer, sp.Float, sp.Rational,
                          sp.core.numbers.NegativeOne, sp.core.numbers.Half)):
            tau.append('const'); consts.append(float(e) * coeff); return
        if isinstance(e, sp.Add):
            if coeff != 1.0:
                raise ValueError("coefficient on Add unsupported")
            tau.append('add')
            for a in list(e.args): traverse(a)
            tau.append('end'); return
        if isinstance(e, sp.Mul):
            args    = list(e.args)
            nums    = [a for a in args if isinstance(a, sp.Number)]
            nonnums = [a for a in args if not isinstance(a, sp.Number)]
            cnum    = coeff * float(np.prod([float(a) for a in nums])) if nums else coeff
            if len(nonnums) == 1:
                traverse(nonnums[0], cnum); return
            sap = _as_signed_abs_power(e)
            if sap is not None:
                _emit_power(sap[0], sap[1], sap[2], coeff); return
            if cnum != 1.0:
                raise ValueError("coefficient on multi-factor Mul unsupported")
            tau.append('mul')
            for a in nonnums: traverse(a)
            tau.append('end'); return
        raise ValueError(f"Unsupported: {type(e)} — {e}")

    try:
        traverse(expr)
    except ValueError:
        return None
    return tau, consts


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
    power_child_vars_only : restrict a power op's child to a bare variable, which
                            is exactly the closed-form/VARPRO applicability
                            boundary (``_expand_monomials`` returns ``None`` for
                            a power of anything else).
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
        if (power_child_vars_only and parent_op in POWER_OPS
                and tok not in VAR_SET): continue

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


# ── Expression evaluation (torch, normalised features) ──────────────────────
def evaluate(tau, x, consts):
    """``x`` is (N, 2*N_DOF): normalised state-variable columns."""
    if not tau: return None
    const_list = list(consts) if consts else []
    const_idx  = [0]
    pos        = [0]

    def _next_const(default=1.0):
        ci = const_idx[0]; const_idx[0] += 1
        return float(const_list[ci]) if ci < len(const_list) else float(default)

    def parse_node():
        if pos[0] >= len(tau): return None
        tok = tau[pos[0]]; pos[0] += 1

        if tok in VAR_SET:
            c = _next_const()
            r = c * x[:, COL_MAP[tok]]
            return None if torch.any(~torch.isfinite(r)) else r
        if tok == 'const':
            v = _next_const()
            return torch.full((x.shape[0],), v, dtype=x.dtype, device=x.device)
        if tok in DOUBLE_CONST_TOKS:
            c = _next_const()
            p = _next_const(2.0)
            child = parse_node()
            if child is None: return None
            if tok == 'intpower':
                n     = _snap_int_power(p)
                n_abs = abs(n)
                sf    = torch.sign(child) if (n_abs % 2 == 1) else torch.ones_like(child)
                if n > 0:
                    r = c * sf * torch.pow(torch.abs(child), float(n_abs))
                else:
                    r = c * sf * torch.pow(torch.abs(child) + 1e-8, float(n))
            else:
                pe   = _clip_exp(p)
                base = torch.pow(torch.abs(child) + 1e-8, pe)
                if tok == 'sgnpower':
                    r = c * torch.sign(child) * base
                else:                                   # abspower
                    r = c * base
            return None if torch.any(~torch.isfinite(r)) else r
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                child = parse_node()
                if child is None: return None
                children.append(child)
            if pos[0] < len(tau): pos[0] += 1
            if len(children) < 2: return None
            if tok == 'add':
                r = children[0]
                for c in children[1:]: r = r + c
                return r
            if tok == 'sub':
                r = children[0]
                for c in children[1:]: r = r - c
                return r
            if tok == 'mul':
                r = children[0]
                for c in children[1:]:
                    r = r * c
                    if torch.any(~torch.isfinite(r)): return None
                return r
        return None

    try:
        result = parse_node()
    except Exception:
        return None
    if result is None: return None
    if torch.any(~torch.isfinite(result)): return None
    return result


# ── Parametric numpy builder (shared by const-opt + compilation) ────────────
def _build_param_code(tau):
    """Return (code_str, n_consts, power_indices, int_power_indices).

    ``code_str`` is a numpy expression in ``c`` (constant vector) and the state
    variables ``x1..x{2N}`` implementing the candidate's *normalised* output.
    ``power_indices`` holds const slots that are continuous exponents
    (abspower/sgnpower); ``int_power_indices`` holds intpower exponent slots.
    """
    const_indices = []; power_indices = []; int_power_indices = []
    pos = [0]

    def build():
        if pos[0] >= len(tau): return '0'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in VAR_SET:
            ci = len(const_indices); const_indices.append(ci)
            return f'c[{ci}]*{tok}'
        if tok == 'const':
            ci = len(const_indices); const_indices.append(ci)
            return f'c[{ci}]'
        if tok in DOUBLE_CONST_TOKS:
            ci = len(const_indices); const_indices.append(ci)
            pi = len(const_indices); const_indices.append(pi)
            child = build()
            if tok == 'intpower':
                int_power_indices.append(pi)
                return f'c[{ci}]*np.clip(_itp({child},c[{pi}]),-1e15,1e15)'
            power_indices.append(pi)
            if tok == 'sgnpower':
                return (f'c[{ci}]*np.clip(np.sign({child})*(np.abs({child})+1e-8)'
                        f'**np.clip(c[{pi}],{MIN_EXP!r},{MAX_EXP!r}),-1e15,1e15)')
            return (f'c[{ci}]*np.clip((np.abs({child})+1e-8)'
                    f'**np.clip(c[{pi}],{MIN_EXP!r},{MAX_EXP!r}),-1e15,1e15)')
        if tok in N_ARY_OPS:
            parts = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end':
                parts.append(build())
            if pos[0] < len(tau): pos[0] += 1
            if not parts: return '0'
            if tok == 'add': return '(' + '+'.join(parts) + ')'
            if tok == 'sub': return '(' + '-'.join(parts) + ')'
            if tok == 'mul': return '(' + '*'.join(parts) + ')'
        return '0'

    code = build()
    return code, len(const_indices), power_indices, int_power_indices


# ── Pretty printing ─────────────────────────────────────────────────────────
def expr_to_str(tau, consts=None):
    if not tau: return '?'
    disp       = _var_display_map()
    const_list = list(consts) if consts else []
    const_idx  = [0]; pos = [0]

    def _next_const(default=1.0):
        ci = const_idx[0]; const_idx[0] += 1
        return float(const_list[ci]) if ci < len(const_list) else float(default)

    def n2s():
        if pos[0] >= len(tau): return '?'
        tok = tau[pos[0]]; pos[0] += 1
        if tok in VAR_SET:
            v = _next_const()
            return f"{v:.4g}*{disp.get(tok, tok)}"
        if tok == 'const':
            v = _next_const()
            return f"{v:.4f}"
        if tok in DOUBLE_CONST_TOKS:
            c = _next_const()
            p = _next_const(2.0)
            child = n2s()
            if tok == 'abspower':
                return f"{c:.4g}*|{child}|^{_clip_exp(p):.4g}"
            if tok == 'sgnpower':
                return f"{c:.4g}*sgn({child})|{child}|^{_clip_exp(p):.4g}"
            return f"{c:.4g}*({child})^{_snap_int_power(p)}"
        if tok in N_ARY_OPS:
            children = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end': children.append(n2s())
            if pos[0] < len(tau): pos[0] += 1
            if tok == 'add': return '(' + ' + '.join(children) + ')'
            if tok == 'sub': return '(' + ' - '.join(children) + ')'
            if tok == 'mul': return '(' + ' * '.join(children) + ')'
        return tok

    try: return n2s()
    except Exception: return str(tau)


# ── NumPy compilation ───────────────────────────────────────────────────────
def compile_to_numpy(tau, consts):
    """Return ``fn(x1, ..., x{2N})`` evaluating the expression in normalised space."""
    if not tau: return None
    const_list = list(consts) if consts else []
    const_idx  = [0]; pos = [0]

    def _next_const(default=1.0):
        ci = const_idx[0]; const_idx[0] += 1
        return float(const_list[ci]) if ci < len(const_list) else float(default)

    def build():
        if pos[0] >= len(tau): return None
        tok = tau[pos[0]]; pos[0] += 1
        if tok in VAR_SET:
            c = _next_const()
            return f'({c!r}*{tok})'
        if tok == 'const':
            v = _next_const()
            return repr(v)
        if tok in DOUBLE_CONST_TOKS:
            c = _next_const()
            p = _next_const(2.0)
            child = build()
            if child is None: return None
            if tok == 'intpower':
                n = _snap_int_power(p)
                return f'({c!r}*np.clip(_itp({child},{n}),-1e15,1e15))'
            pe = _clip_exp(p)
            if tok == 'sgnpower':
                return (f'({c!r}*np.clip(np.sign({child})*(np.abs({child})+1e-8)'
                        f'**{pe!r},-1e15,1e15))')
            return (f'({c!r}*np.clip((np.abs({child})+1e-8)'
                    f'**{pe!r},-1e15,1e15))')
        if tok in N_ARY_OPS:
            parts = []
            while pos[0] < len(tau) and tau[pos[0]] != 'end': parts.append(build())
            if pos[0] < len(tau): pos[0] += 1
            if not parts: return '0'
            if tok == 'add': return '(' + '+'.join(parts) + ')'
            if tok == 'sub': return '(' + '-'.join(parts) + ')'
            if tok == 'mul': return '(' + '*'.join(parts) + ')'
        return '0'

    sig = ','.join(VARIABLES)
    try:
        code   = build()
        fn_raw = eval(f'lambda {sig}: {code}', {'np': np, '_itp': _intpower_np})

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


def _slice_traj(traj, horizon):
    """Return (t, state_arr) sliced to ``horizon`` (or full)."""
    t_eval, ic, state_arr = traj[:3]
    if horizon is not None and horizon < t_eval[-1]:
        mh = t_eval <= horizon
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
    if horizon is not None and horizon < t_eval[-1]:
        mh = t_eval <= horizon
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


def energy_reward(exprs, max_traj=None, horizon=None, w_acc=None):
    """Global work-energy reward for a full set of per-DOF expressions.

    ``exprs`` is a length-``N_DOF`` list whose entry ``d`` is ``(tau, consts)``
    for DOF ``d`` or ``None`` if that DOF has no expression yet.  DOFs that are
    ``None`` are omitted from *both* the predicted-power sum and the measured
    kinetic-energy sum so the balance stays consistent.

    Returns ``r = 1 / (1 + mean_t |cum_power(t) - dKE(t)|)`` averaged over the
    first ``max_traj`` trajectories (``None`` = the configured ``MAX_TRAJ``);
    0 if nothing is evaluable.
    """
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return 0.0
    X_mean, X_std, y_mean, y_std = NORM_STATS

    present = [d for d in range(N_DOF) if exprs[d] is not None]
    if not present:
        return 0.0
    fns = {}
    for d in present:
        f = compile_to_numpy(*exprs[d])
        if f is None:
            return 0.0
        fns[d] = f

    w = _resolve_w_acc(w_acc)
    residual_means = []
    for traj in RAW_TRAJECTORIES[:_resolve_max_traj(max_traj)]:
        t_sl, st_sl = _slice_traj(traj, horizon)
        if t_sl is None:
            continue
        acc_meas = _slice_accs(traj, horizon) if w > 0.0 else None
        feats = (st_sl - X_mean[:, None]) / X_std[:, None]
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
                scale = float(np.std(dke_d))
                if scale < 1e-12:
                    scale = float(np.mean(np.abs(dke_d))) + 1e-12
                res_d = res_d / scale
            e_res = float(np.mean(res_d))
            # Blend in pointwise acceleration NRMSE.  Both terms are already
            # dimensionless (energy by std(dKE), accel by std(a_meas)), so the
            # weighted sum is meaningful without further rescaling.
            if w > 0.0 and acc_meas is not None:
                a_meas = acc_meas[d, :]
                a_sc   = float(np.std(a_meas))
                if a_sc < 1e-12:
                    a_sc = float(np.mean(np.abs(a_meas))) + 1e-12
                a_res = float(np.sqrt(np.mean((acc_d - a_meas) ** 2))) / a_sc
                per_dof.append((1.0 - w) * e_res + w * a_res)
            else:
                per_dof.append(e_res)
        if not ok:
            continue
        residual_means.append(float(np.mean(per_dof)))

    if not residual_means:
        return 0.0
    return 1.0 / (1.0 + float(np.mean(residual_means)))


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
MAX_GRID_COMBOS = 1600
MAX_EXP_SLOTS   = 40
MAX_CD_PASSES   = 3


def _energy_contexts(target_dof, other_exprs, max_traj, horizon):
    """Precompute the per-trajectory energy context shared by both the fast
    (linear/VARPRO) and the slow (nonlinear) constant fitters.

    The target DOF is fit to its **own** work-energy balance only::

        integral(vel_t * a_target) ~= 1/2 (vel_t^2 - vel_t(0)^2)

    which is an exact per-DOF kinematic identity (coupling forces are already
    contained in ``a_target``).  The other DOFs are therefore *not* mixed into
    the balance: doing so would let their prediction errors contaminate this
    DOF's constants and would let a high-KE DOF swamp a low-KE one.
    ``other_exprs`` is accepted for API compatibility but no longer used.

    Returns ``(contexts, ym_t, ys_t)`` where each context is
    ``(t_sl, cols, vel_t, cum_other, dke, scale, acc_t, acc_scale)`` with
    ``cum_other == 0``.  ``acc_t`` is the MEASURED acceleration of the target
    DOF (or ``None`` if unavailable) and is used only when ``W_ACC > 0``.
    """
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return None, 0.0, 0.0
    X_mean, X_std, y_mean, y_std = NORM_STATS

    contexts = []
    for traj in RAW_TRAJECTORIES[:_resolve_max_traj(max_traj)]:
        t_sl, st_sl = _slice_traj(traj, horizon)
        if t_sl is None:
            continue
        n     = st_sl.shape[1]
        feats = (st_sl - X_mean[:, None]) / X_std[:, None]
        vel_t = st_sl[2 * target_dof + 1, :]
        # Per-DOF balance: no contribution from the other DOFs.
        cum_other = np.zeros(n)
        dke       = 0.5 * (vel_t ** 2 - vel_t[0] ** 2)
        # Normalise by this DOF's own KE so a low-velocity DOF is weighted the
        # same as a high-velocity one.
        scale = 1.0
        if ENERGY_NORMALIZE:
            scale = float(np.std(dke))
            if scale < 1e-12:
                scale = float(np.mean(np.abs(dke))) + 1e-12
        cols = [feats[k, :] for k in range(N_VARS)]
        accs_sl = _slice_accs(traj, horizon)
        if accs_sl is not None:
            acc_t = accs_sl[target_dof, :]
            acc_scale = float(np.std(acc_t))
            if acc_scale < 1e-12:
                acc_scale = float(np.mean(np.abs(acc_t))) + 1e-12
        else:
            acc_t, acc_scale = None, 1.0
        contexts.append((t_sl, cols, vel_t, cum_other, dke, scale,
                         acc_t, acc_scale))

    ym_t, ys_t = float(y_mean[target_dof]), float(y_std[target_dof])
    return contexts, ym_t, ys_t


def _parse_tree(tau):
    """Parse ``tau`` into a nested node tree, assigning const slots in the same
    pre-order the evaluator/compiler consume them (power ops consume
    coefficient then exponent *before* their child)."""
    pos = [0]; slot = [0]

    def parse():
        if pos[0] >= len(tau):
            return {'op': 'const_one'}
        tok = tau[pos[0]]; pos[0] += 1
        if tok in VAR_SET:
            s = slot[0]; slot[0] += 1
            return {'op': 'var', 'col': COL_MAP[tok], 'coeff': s}
        if tok == 'const':
            s = slot[0]; slot[0] += 1
            return {'op': 'const', 'val': s}
        if tok in DOUBLE_CONST_TOKS:
            s0 = slot[0]; slot[0] += 1     # coefficient
            s1 = slot[0]; slot[0] += 1     # exponent
            child = parse()
            return {'op': tok, 'coeff': s0, 'exp': s1, 'child': child}
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
    if op == 'const':
        return [{'factors': [], 'amp': {node['val']}, 'exps': set()}]
    if op in POWER_OPS:
        ch = node['child']
        if ch.get('op') != 'var':
            return None          # power of a non-variable subtree -> nonlinear
        return [{'factors': [(op, ch['col'], node['exp'])],
                 'amp': {node['coeff']}, 'exps': {node['exp']}}]
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
        else:                                    # (power_op, col, exp_slot)
            kind, col, slot = f
            a = cols[col]
            if kind == 'intpower':
                phi = phi * np.clip(_intpower_np(a, exp_val[slot]), -1e15, 1e15)
            else:
                p = _clip_exp(exp_val[slot])
                base = np.clip((np.abs(a) + 1e-8) ** p, -1e15, 1e15)
                phi = phi * (np.sign(a) * base if kind == 'sgnpower' else base)
    return np.where(np.isfinite(phi), phi, 0.0)


def _fit_consts_linear(tau, contexts, ym_t, ys_t, n_consts):
    """Closed-form (0 exponents) / VARPRO (1-2 exponents) constant fit.

    Returns a consts list of length ``n_consts`` on success, or ``None`` to
    signal the caller should fall back to the nonlinear optimiser.
    """
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
    for j in range(len(monos)):
        others = set()
        for k in range(len(monos)):
            if k != j:
                others |= amp_sets[k]
        excl = amp_sets[j] - others
        if not excl:
            return None
        matched.append(min(excl))

    exp_slots = sorted(set().union(*[m['exps'] for m in monos]))
    # which power op owns each exponent slot (drives the grid type below)
    slot_kind = {}
    for m in monos:
        for f in m['factors']:
            if f[0] != 'var':
                slot_kind[f[2]] = f[0]
    M = len(monos)

    # ── Fit-row subsampling ─────────────────────────────────────────────────
    # Quadrature accuracy requires INTEGRATING on the fine time grid, but the
    # least-squares fit of ~M coefficients does not need every integrated row.
    # All cumtrapz below run on the full grid; only the rows handed to lstsq
    # are strided down to <= MAX_FIT_ROWS.
    #
    # BLEND: the fit objective must match the SCORING objective, or the search
    # ranks candidates by a criterion their constants were never optimised for.
    # Both objectives are linear in the coefficients given the exponents:
    #
    #   energy :  (dKE - int(v*ym)) / s_e  ~=  sum_j c_j * ys*int(v*phi_j)/s_e
    #   accel  :  (a_meas - ym)     / s_a  ~=  sum_j c_j * ys*phi_j       /s_a
    #
    # so the blend is one stacked least-squares problem with the energy rows
    # scaled by sqrt(1-w) and the acceleration rows by sqrt(w).  Both targets
    # are pre-normalised to unit-ish variance (by std(dKE) and std(a_meas)
    # respectively), which is what makes w meaningful as a weight rather than
    # an arbitrary units conversion.  The acceleration block needs no cumtrapz,
    # so w=1 is strictly cheaper than w=0.
    w = _resolve_w_acc(None)
    use_e = w < 1.0
    use_a = w > 0.0 and all(c[6] is not None for c in contexts)
    if not use_a:
        w, use_e, use_a = 0.0, True, False
    sq_e, sq_a = np.sqrt(1.0 - w), np.sqrt(w)

    n_pts = sum(len(c[2]) for c in contexts)
    stride = max(1, int(np.ceil(n_pts / MAX_FIT_ROWS)))

    ctx_b = []
    for (t_sl, cols, vel_t, cum_other, dke, scale, acc_t, acc_scale) in contexts:
        parts = []
        if use_e:
            off = _cumtrapz(vel_t * ym_t, t_sl)
            parts.append(sq_e * ((dke - cum_other - off) / scale)[::stride])
        if use_a:
            parts.append(sq_a * ((acc_t - ym_t) / acc_scale)[::stride])
        ctx_b.append(np.concatenate(parts))
    b_full  = np.concatenate(ctx_b)
    total_n = len(b_full)

    # ── Integrated-column cache ─────────────────────────────────────────────
    # A monomial's column depends only on the exponent slots IT contains
    # (usually one), so cache per (context, monomial, relevant slot values):
    # an 18x18 grid then costs 18 integrations per single-slot monomial
    # instead of 324.
    col_cache = {}

    def _column(ci, j, exp_val):
        m = monos[j]
        key = (ci, j, tuple(sorted((s, round(float(exp_val[s]), 6))
                                   for s in m['exps'])))
        col = col_cache.get(key)
        if col is None:
            (t_sl, cols, vel_t, cum_other, dke, scale, _at, _as) = contexts[ci]
            phi = _eval_monomial(m, cols, exp_val, len(vel_t))
            # The column must be stacked EXACTLY as b_full was: energy block
            # then acceleration block, each carrying the same sqrt weight that
            # was applied to its half of the target.  Weighting only the target
            # (and not the design matrix) does not weight the problem at all —
            # it just rescales the residual — so both must be done together.
            parts = []
            if use_e:
                parts.append(sq_e * (ys_t * _cumtrapz(vel_t * phi, t_sl)
                                     / scale)[::stride])
            if use_a:
                # a_phys = ym + ys*pred, so the acceleration column is simply
                # ys*phi / std(a_meas) — no integration, which is why w=1 is
                # cheaper per candidate than w=0.
                parts.append(sq_a * (ys_t * phi / _as)[::stride])
            col = np.concatenate(parts)
            col_cache[key] = col
        return col

    def _solve(exp_val):
        A = np.vstack([
            np.column_stack([_column(ci, j, exp_val) for j in range(M)])
            for ci in range(len(contexts))])
        if not np.all(np.isfinite(A)):
            return None, None, np.inf
        coef, *_ = np.linalg.lstsq(A, b_full, rcond=None)
        res = A @ coef - b_full
        return coef, res, float(res @ res)

    def _finish(coef, exp_val):
        if coef is None or not np.all(np.isfinite(coef)):
            return None
        consts = [1.0] * n_consts
        for j in range(M):
            consts[matched[j]] = float(coef[j])
        for s, v in exp_val.items():
            consts[s] = float(v)
        return consts

    # ── no exponents: a single exact least-squares solve ────────────────────
    if not exp_slots:
        coef, _res, _cost = _solve({})
        return _finish(coef, {})

    if len(exp_slots) > MAX_EXP_SLOTS:
        return None

    # ── exponent search, then polish continuous slots (VARPRO) ──────────────
    # intpower slots get the exact integer grid (the fit is then *exact* per
    # combination — no polish needed); abspower/sgnpower slots get the coarse
    # float grid and a least-squares polish afterwards.
    int_grid = [float(n) for n in range(1, INTPOWER_MAX + 1) if n != 0]
    cont_step = 0.5 if len(exp_slots) == 1 else 1.0
    cont_grid = list(np.arange(MIN_EXP, MAX_EXP + 1e-9, cont_step))

    def _slot_grid(s):
        return int_grid if slot_kind.get(s) == 'intpower' else cont_grid

    grids = [_slot_grid(s) for s in exp_slots]
    n_combos = 1
    for g in grids:
        n_combos *= len(g)

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
        # The product grid explodes (4 intpower slots = 18^4 = 104976 combos),
        # but the fit is LINEAR given the exponents, so sweeping one slot at a
        # time costs sum(len(grid)) per pass instead of prod(len(grid)).  This
        # matters more than it looks: the expanded-cube truth structure
        # (x1^3, x3^3, x1^2*x3, x1*x3^2) has FOUR intpower slots, so under the
        # old `len(exp_slots) > 2 -> return None` rule it was pushed onto the
        # multi-start nonlinear optimiser and systematically under-fit — the
        # exact same structure scored 0.925 fitted that way and 0.999 fitted
        # exactly.  Structurally-wrong 2-slot candidates got the exact path and
        # therefore beat the truth on reward.  Coordinate descent is not
        # guaranteed globally optimal on the grid, but it is enormously better
        # than handing the whole problem to a nonlinear solver.
        best_ev = {}
        for s in exp_slots:
            g = _slot_grid(s)
            best_ev[s] = 2.0 if slot_kind.get(s) == 'intpower' else float(np.median(g))
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
        return None

    cont_slots = [s for s in exp_slots if slot_kind.get(s) != 'intpower']
    if cont_slots:
        def _resid(p):
            ev = dict(best_ev)
            for i, s in enumerate(cont_slots):
                ev[s] = float(np.clip(p[i], MIN_EXP, MAX_EXP))
            _c, r, _cost = _solve(ev)
            return r if r is not None else np.full(total_n, 1e6)

        try:
            rr = least_squares(_resid,
                               np.array([best_ev[s] for s in cont_slots]),
                               bounds=([MIN_EXP] * len(cont_slots),
                                       [MAX_EXP] * len(cont_slots)),
                               max_nfev=40)
            for i, s in enumerate(cont_slots):
                best_ev[s] = float(np.clip(rr.x[i], MIN_EXP, MAX_EXP))
        except Exception:
            pass

    coef, _res, _cost = _solve(best_ev)
    return _finish(coef, best_ev)


def optimise_consts_energy(tau, target_dof, other_exprs=None,
                           max_traj=None, horizon=None,
                           n_inits=3, max_nfev=150):
    """Fit ``tau``'s constants to minimise the target DOF's **own** running
    energy residual ``| integral(vel_t * a_target) - 1/2 (vel_t^2 - vel_t0^2) |``.

    The per-DOF work-energy relation is an exact kinematic identity, so each DOF
    is fit independently.  ``other_exprs`` is accepted for API compatibility but
    is no longer used (the other DOFs are not mixed into this DOF's balance, so
    their errors cannot contaminate its constants).  ``max_traj=None`` uses the
    configured ``MAX_TRAJ`` — the same trajectories the reward scores on.

    Fast path: for expressions that are linear in their coefficients (with 0-2
    power exponents) the fit is a closed-form least-squares / variable
    projection; intpower exponents are searched on the exact integer grid.
    Anything the fast path can't handle — powers of non-variable subtrees,
    over-large expansions — falls back to the general multi-start nonlinear
    optimiser below.
    """
    n_consts = count_total_consts(tau)
    if n_consts == 0:
        return []
    if NORM_STATS is None or not RAW_TRAJECTORIES:
        return [1.0] * n_consts

    _X_mean, _X_std, y_mean, y_std = NORM_STATS

    contexts, ym_t, ys_t = _energy_contexts(target_dof, other_exprs, max_traj, horizon)
    if not contexts:
        return [1.0] * n_consts

    # Fast path: closed-form / VARPRO.
    fast = _fit_consts_linear(tau, contexts, ym_t, ys_t, n_consts)
    if fast is not None:
        return fast

    # Fallback: general nonlinear least-squares (multi-start).
    code, n_c, power_indices, int_power_indices = _build_param_code(tau)
    if n_c != n_consts:
        n_consts = n_c
    sig = ','.join(VARIABLES)
    try:
        fn_param = eval(f'lambda c,{sig}: {code}', {'np': np, '_itp': _intpower_np})
    except Exception:
        return [1.0] * n_consts

    _total_rows = sum(len(ctx[2]) for ctx in contexts)
    _stride = max(1, int(np.ceil(_total_rows / MAX_FIT_ROWS)))

    # Same blend as the fast path.  If the fallback minimised the pure energy
    # residual while the reward scored a blend, constants would be optimised
    # for a different objective than the one they are judged by — the exact
    # fit/score mismatch that makes candidates look worse than they are.
    _w     = _resolve_w_acc(None)
    _use_a = _w > 0.0 and all(c[6] is not None for c in contexts)
    if not _use_a:
        _w = 0.0
    _sq_e, _sq_a = np.sqrt(1.0 - _w), np.sqrt(_w)

    def residuals(c):
        parts = []
        for (t_sl, cols, vel_t, cum_other, dke, scale,
             acc_t, acc_scale) in contexts:
            n_bad = len(vel_t[::_stride]) * (2 if _use_a else 1)
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
            if _w < 1.0:
                cum = _cumtrapz(vel_t * acc, t_sl) + cum_other
                parts.append(_sq_e * ((cum - dke) / scale)[::_stride])
            if _use_a:
                parts.append(_sq_a * ((acc - acc_t) / acc_scale)[::_stride])
        return np.concatenate(parts)

    # bounds for power exponents
    if power_indices or int_power_indices:
        lb = np.full(n_consts, -np.inf); ub = np.full(n_consts, np.inf)
        for pi in power_indices:     lb[pi] = MIN_EXP; ub[pi] = MAX_EXP
        for pi in int_power_indices: lb[pi] = 0.5; ub[pi] = INTPOWER_MAX + 0.5
        opt_method = 'trf'; opt_bounds = (lb, ub)
    else:
        opt_method = 'lm'; opt_bounds = (-np.inf, np.inf)

    rng     = np.random.default_rng()
    # scale of the energy target sets the coefficient scale
    e_scale = 1.0
    for (_t, _cols, vel_t, _co, dke, _s, _at, _as) in contexts:
        e_scale = max(e_scale, float(np.std(dke)) + 1e-9)

    def _make_smart(cs, pv):
        """Initial const vector in pre-order: coefficient=cs for variables and
        consts; (coefficient=cs, exponent=pv) for power operators."""
        init = []
        for t in tau:
            if t in VAR_SET or t == 'const':
                init.append(cs)
            elif t in DOUBLE_CONST_TOKS:
                init.append(cs); init.append(pv)
        return np.array(init, dtype=float)

    inits = [
        _make_smart( e_scale, 1.0), _make_smart( e_scale, 2.0),
        _make_smart( e_scale, 3.0), _make_smart(-e_scale, 1.0),
        _make_smart(-e_scale, 2.0), _make_smart(-e_scale, 3.0),
        _make_smart( e_scale * 0.1, 3.0), _make_smart(-e_scale * 0.1, 3.0),
        np.ones(n_consts), np.zeros(n_consts),
        rng.uniform(-e_scale * 2, e_scale * 2, n_consts),
        rng.uniform(-1, 1, n_consts),
    ]
    for init in inits:
        if len(init) != n_consts:
            continue
        for pi in power_indices:
            init[pi] = float(np.clip(init[pi], MIN_EXP, MAX_EXP))
        for pi in int_power_indices:
            init[pi] = float(np.clip(init[pi], 1.0, INTPOWER_MAX))
    inits = [i for i in inits if len(i) == n_consts][:n_inits]

    best_cost, best_c = np.inf, None
    for init in inits:
        try:
            res = least_squares(residuals, init, method=opt_method,
                                bounds=opt_bounds, max_nfev=max_nfev)
            if res.cost < best_cost:
                best_cost, best_c = res.cost, res.x.tolist()
        except Exception:
            pass
    return best_c if best_c is not None else [1.0] * n_consts


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
                       energy_normalize=True, max_traj=5, w_acc=None):
    configure_grammar(n_dof, var_names)
    set_problem_data(norm_stats, raw_trajs, energy_normalize, max_traj, w_acc)


def energy_worker(args):
    """
    args = (target_dof, cand_tau, elite_exprs, horizon)
      elite_exprs : length-N list of (tau, consts) or None.  ACCEPTED FOR API
                    COMPATIBILITY AND IGNORED — see below.

    Fits the candidate's constants to the target DOF's own work-energy balance,
    then scores that DOF **in isolation**.  Returns
    (target_dof, energy_reward, cand_tau, cand_consts) or None.

    Why the partner elites are not mixed in
    ---------------------------------------
    ``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an exact per-DOF
    kinematic identity: coupling forces are already inside ``a_d``, so DOF d's
    balance is complete without any knowledge of the other DOFs.  The constant
    fit already exploited this (``_energy_contexts`` sets ``cum_other = 0``).
    Averaging the partner's residual into the score therefore adds no
    within-epoch information — it is a monotone shift shared by every candidate
    of this DOF, so rankings are unchanged — while making scores incomparable
    ACROSS epochs, because the partner elite changes underneath them.  That
    single defect was the source of three separate symptoms: the epoch-0
    asymmetry (a candidate scored without a partner keeps the elite slot
    forever), stale buffer entries driving unfair pruning and R_alpha drift,
    and ``best_r`` being dethroned purely because the partner improved.
    Scoring in isolation removes all three at once.
    """
    target_dof, cand_tau, _elite_exprs, horizon = args
    consts = optimise_consts_energy(cand_tau, target_dof, None, horizon=horizon)
    exprs  = [None] * N_DOF
    exprs[target_dof] = (cand_tau, consts)
    r = energy_reward(exprs, horizon=horizon)
    if r <= 1e-6:
        return None
    return (target_dof, r, cand_tau, consts)


# ── J-GRPO ──────────────────────────────────────────────────────────────────
# The objective itself lives in :func:`discover_policy.jgrpo_terms`.  Above this
# many buffer entries a GRPO step subsamples, prioritised in log-residual units
# (see :func:`log_residual_score`) so near-top refinements are not crowded out
# by the compressed r-scale.
MAX_GRPO_BATCH = 32


# ── Denormalisation to physical units ───────────────────────────────────────
def denormalize_expr(tau, consts, dof):
    """Return ``"<accel name> = <physical expression>"`` for DOF ``dof``."""
    if NORM_STATS is None: return None
    X_mean, X_std, y_mean, y_std = NORM_STATS
    try:
        expr_str = expr_to_sympy_str(tau, consts)
        sym_expr = parse_expr(expr_str, local_dict=_parse_local_dict())

        phys = []
        subs = {}
        for d in range(N_DOF):
            q  = sp.Symbol(VAR_NAMES[d])
            qd = sp.Symbol(VAR_NAMES[d] + 'dot')
            phys.extend([q, qd])
        for k in range(N_VARS):
            subs[SYMS[f'x{k + 1}']] = (phys[k] - float(X_mean[k])) / float(X_std[k])
        sym_expr = sym_expr.subs(subs)
        sym_expr = sp.expand(float(y_mean[dof]) + float(y_std[dof]) * sym_expr)
        accel_name = VAR_NAMES[dof] + 'ddot'
        return f"{accel_name} = {sp.N(sym_expr, 4)}"
    except Exception as e:
        return f"(denorm failed: {e})"