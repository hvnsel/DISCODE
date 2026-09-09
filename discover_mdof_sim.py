"""
discover_mdof_sim.py
===================

DISCOVER on a **simulated multi-DOF** system.  Same idea as
:mod:`discover_sdof_sim`, with coupled DOFs: the states come from integrating a
known set of coupled equations, so the work-energy identity closes exactly and
the reward ceiling is ~1.0.  The only question is whether the search recovers
the equations.

The library here is deliberately generic — coupled Duffing oscillators, not any
particular rig.  Each DOF's truth is a sum of terms that the grammar can write
exactly, which matters: if the truth is not representable, the best achievable
reward is unknown and a shortfall can't be attributed to the search.

Add a system by writing one ``TruthSystem`` and registering it in ``_REGISTRY``.

Grammar note for ``truth_taus``
-------------------------------
For an N-DOF system the state layout is ``x1 = q_1``, ``x2 = qd_1``,
``x3 = q_2``, ``x4 = qd_2``, ...  Every variable leaf carries an implicit
fitted coefficient, so ``['add', 'x1', 'x2', 'x3', 'end']`` is
``c1*q1 + c2*qd1 + c3*q2``.  ``intpower`` supplies (coefficient, integer
exponent), so ``q1^3`` is ``['intpower', 'x1']``.

Cubic COUPLING: ``(q1 - q2)^3`` expands into four monomials (``q1^3,
q1^2*q2, q1*q2^2, q2^3``), so its truth tau is long and carries four fitted
exponent slots.  That used to fall off the closed-form path; it no longer does
— :func:`discover_core._fit_consts_linear` sends >``MAX_GRID_COMBOS`` cases to
cyclic coordinate descent, which was added for exactly this structure and fits
it essentially exactly.  ``coupled_duffing`` and ``duffing_chain3`` still keep
the coupling linear and the cubic on the grounded springs, which keeps their
truths short; ``cubic_coupled`` is the deliberate opposite (see its docstring).
"""

from __future__ import annotations

from discover_data import TruthSystem, build_truth_system, print_truth_rewards
from discover_train import DISCOVER_TRAIN


# ── N-DOF system library ────────────────────────────────────────────────────
def _coupled_duffing():
    """Two Duffing oscillators, linearly coupled (unit masses)::

        q1'' = -c1*q1' - k1*q1 - a1*q1^3 - kc*(q1 - q2)
        q2'' = -c2*q2' - k2*q2 - a2*q2^3 - kc*(q2 - q1)

    Linear modes at ~0.40 Hz and ~0.28 Hz; at the ic_scale below the cubic
    terms carry ~30% of the restoring force, so they are well identified.
    """
    c1, c2 = 0.15, 0.10
    k1, k2 = 4.00, 2.50
    kc     = 1.50
    a1, a2 = 0.80, 0.50

    def a0(s):
        q1, v1, q2, _v2 = s
        return -c1 * v1 - k1 * q1 - a1 * q1 ** 3 - kc * (q1 - q2)

    def a1_(s):
        q1, _v1, q2, v2 = s
        return -c2 * v2 - k2 * q2 - a2 * q2 ** 3 - kc * (q2 - q1)

    return TruthSystem(
        name='mdof_coupled_duffing',
        accel_fns=[a0, a1_],
        truth_strs=[
            f"xddot = {-(k1+kc):.4g}*x {-c1:+.4g}*xdot {+kc:+.4g}*y {-a1:+.4g}*x^3",
            f"yddot = {-(k2+kc):.4g}*y {-c2:+.4g}*ydot {+kc:+.4g}*x {-a2:+.4g}*y^3",
        ],
        truth_taus=[
            ['add', 'x1', 'x2', 'x3', 'intpower', 'x1', 'end'],   # q1, qd1, q2, q1^3
            ['add', 'x1', 'x3', 'x4', 'intpower', 'x3', 'end'],   # q1, q2, qd2, q2^3
        ],
        var_names=['x', 'y'],
        t_end=25.0, n_pts=2500, n_traj=4,
        ic_scale=[1.5, 1.5, 1.5, 1.5],
        seed=0,
    )


def _duffing_chain3():
    """Three-mass chain, cubic springs at the two ends (unit masses)::

        q1'' = -c*q1' - k*q1 - a*q1^3 - k*(q1 - q2)
        q2'' = -c*q2' - k*(q2 - q1) - k*(q2 - q3)
        q3'' = -c*q3' - k*q3 - a*q3^3 - k*(q3 - q2)

    Included to exercise the N-DOF path beyond two policies.
    """
    c, k, a = 0.12, 3.0, 0.60

    def a0(s):
        q1, v1, q2 = s[0], s[1], s[2]
        return -c * v1 - k * q1 - a * q1 ** 3 - k * (q1 - q2)

    def a1_(s):
        q1, q2, v2, q3 = s[0], s[2], s[3], s[4]
        return -c * v2 - k * (q2 - q1) - k * (q2 - q3)

    def a2(s):
        q2, q3, v3 = s[2], s[4], s[5]
        return -c * v3 - k * q3 - a * q3 ** 3 - k * (q3 - q2)

    return TruthSystem(
        name='mdof_duffing_chain3',
        accel_fns=[a0, a1_, a2],
        truth_strs=[
            f"q1ddot = {-2*k:.4g}*q1 {-c:+.4g}*q1dot {+k:+.4g}*q2 {-a:+.4g}*q1^3",
            f"q2ddot = {+k:.4g}*q1 {-2*k:+.4g}*q2 {-c:+.4g}*q2dot {+k:+.4g}*q3",
            f"q3ddot = {+k:.4g}*q2 {-2*k:+.4g}*q3 {-c:+.4g}*q3dot {-a:+.4g}*q3^3",
        ],
        truth_taus=[
            ['add', 'x1', 'x2', 'x3', 'intpower', 'x1', 'end'],
            ['add', 'x1', 'x3', 'x4', 'x5', 'end'],
            ['add', 'x3', 'x5', 'x6', 'intpower', 'x5', 'end'],
        ],
        var_names=['q1', 'q2', 'q3'],
        t_end=20.0, n_pts=2500, n_traj=4,
        ic_scale=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        seed=0,
    )


def _cubic_coupled():
    """Two oscillators joined by a **cubic** coupling spring (unit masses)::

        q1'' = -c1*q1' - k1*q1 - a*(q1 - q2)^3
        q2'' = -c2*q2' - k2*q2 + a*(q1 - q2)^3      (equal and opposite)

    This system exists to make cross-DOF structure sharing visible.  A *linear*
    coupling is a single leaf token (``x3`` in DOF 0, ``x1`` in DOF 1) — found
    immediately, and far too small to be worth copying, so a joint policy can
    show no advantage on it.  The cubic coupling expands to four monomials
    (``q1^3, q1^2*q2, q1*q2^2, q2^3``) that appear in BOTH equations, so the
    shared subtree is genuinely multi-token.

    The sign flip costs a joint policy nothing: VARPRO fits each monomial's
    leading coefficient, so only the *structure* has to cross between DOFs.
    """
    c1, c2 = 0.12, 0.10
    k1, k2 = 3.00, 2.00
    a      = 1.20

    def a0(s):
        q1, v1, q2, _v2 = s
        return -c1 * v1 - k1 * q1 - a * (q1 - q2) ** 3

    def a1_(s):
        q1, _v1, q2, v2 = s
        return -c2 * v2 - k2 * q2 + a * (q1 - q2) ** 3

    # (q1 - q2)^3 = q1^3 - 3 q1^2 q2 + 3 q1 q2^2 - q2^3.  Every leaf carries an
    # implicit fitted coefficient and every intpower fits its own exponent, so
    # the four monomials are written structurally and the signs come out of the
    # fit.
    _cube = ['intpower', 'x1',                    # q1^3
             'mul', 'intpower', 'x1', 'x3', 'end',   # q1^2 * q2
             'mul', 'x1', 'intpower', 'x3', 'end',   # q1 * q2^2
             'intpower', 'x3']                    # q2^3

    return TruthSystem(
        name='mdof_cubic_coupled',
        accel_fns=[a0, a1_],
        truth_strs=[
            f"xddot = {-k1:.4g}*x {-c1:+.4g}*xdot {-a:+.4g}*(x - y)^3",
            f"yddot = {-k2:.4g}*y {-c2:+.4g}*ydot {+a:+.4g}*(x - y)^3",
        ],
        truth_taus=[
            ['add', 'x1', 'x2'] + _cube + ['end'],   # q1, qd1, + the cube
            ['add', 'x3', 'x4'] + _cube + ['end'],   # q2, qd2, + the same cube
        ],
        var_names=['x', 'y'],
        t_end=25.0, n_pts=2500, n_traj=4,
        ic_scale=[1.2, 1.2, 1.2, 1.2],
        seed=0,
    )


_REGISTRY = {
    'coupled_duffing': _coupled_duffing,
    'duffing_chain3':  _duffing_chain3,
    'cubic_coupled':   _cubic_coupled,
}


def get_mdof_system(key):
    if key not in _REGISTRY:
        raise ValueError(f"Unknown MDOF system '{key}'. "
                         f"Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[key]()


# ── Entry point ─────────────────────────────────────────────────────────────
def DISCOVER_MDOF_SIM(
    system_key       = 'coupled_duffing',
    # data generation
    n_traj           = None,     # None = use the spec's value
    t_end            = None,
    n_pts            = None,
    seed             = None,
    plot_data        = False,
    show_truth       = True,
    # search
    n_epochs         = 300,
    batch_size       = 200,
    max_len          = 24,
    lr               = 1e-4,
    alpha            = 0.20,
    C                = 5,
    G                = 5,
    lam_start        = 0.02,
    lam_end          = 0.001,
    eps              = 0.2,
    beta             = 0.01,
    max_buffer       = 500,
    beam_interval    = 10,
    beam_width       = 20,
    novelty_weight   = 0.15,
    energy_normalize = True,
    max_traj         = None,
    w_acc            = 0.5,
    use_pool         = True,
    # policy (see DISCOVER_TRAIN)
    cross_slice_attention = True,
    n_layers              = 4,
    d_model               = 128,
    slice_order           = 'random',
    sample_order          = 'reward',
    max_terms             = 8,         # term slots per DOF (bag capacity)
    max_term_len          = 8,         # token budget per term
    term_grammar          = 'free',    # 'free' | 'varpro'
    term_position_encoding = True,     # False -> order-blind bag
    seed_policy           = None,
):
    """Generate an N-DOF system from the library and try to rediscover it."""
    spec = get_mdof_system(system_key)
    system = build_truth_system(spec, n_traj=n_traj, t_end=t_end,
                                n_pts=n_pts, seed=seed)

    if plot_data:
        from discover_data import plot_system_data
        plot_system_data(system)

    if show_truth:
        print_truth_rewards(system, max_traj=max_traj, w_acc=w_acc,
                            energy_normalize=energy_normalize)

    return DISCOVER_TRAIN(
        system_data      = system,
        n_epochs         = n_epochs,
        batch_size       = batch_size,
        max_len          = max_len,
        lr               = lr,
        alpha            = alpha,
        C                = C,
        G                = G,
        lam_start        = lam_start,
        lam_end          = lam_end,
        eps              = eps,
        beta             = beta,
        max_buffer       = max_buffer,
        beam_interval    = beam_interval,
        beam_width       = beam_width,
        novelty_weight   = novelty_weight,
        energy_normalize = energy_normalize,
        max_traj         = max_traj,
        w_acc            = w_acc,
        use_pool         = use_pool,
        cross_slice_attention = cross_slice_attention,
        n_layers              = n_layers,
        d_model               = d_model,
        slice_order           = slice_order,
        sample_order          = sample_order,
        max_terms             = max_terms,
        max_term_len          = max_term_len,
        term_grammar          = term_grammar,
        term_position_encoding = term_position_encoding,
        seed                  = seed_policy,
    )


if __name__ == '__main__':
    DISCOVER_MDOF_SIM(
        system_key = 'coupled_duffing',
        n_epochs   = 300,
        batch_size = 200,
        max_len    = 24,
        plot_data  = True,
    )
