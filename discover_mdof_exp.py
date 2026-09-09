"""
discover_mdof_exp.py
===================

DISCOVER on **multi-DOF experimental data** — the real target.  Every channel of
the MATLAB record becomes a grammar variable, and the policy searches for every
DOF's acceleration at once, so coupling terms are expressible.

Read the ceiling before reading the reward
------------------------------------------
On processed experimental data the channels rarely satisfy ``a = dv/dt`` and
``v = dq/dt`` exactly.  Anything applied per integration stage — a high-pass to
kill drift, a band-pass on the accelerometer — breaks the identity the reward
is built on, and the MEASURED acceleration then scores well below 1.0 on its
own energy balance.  That number is the ceiling: no expression can beat it, and
the ranking among expressions can invert near it.  ``show_ceiling`` prints it
per trial and per DOF, and it should be read first, every run.

If the ceiling is low, the fix is in the data conditioning, not the search:
every channel must be filtered with the SAME number of passes, or the
derivative relations between them no longer hold.  Use
:mod:`discover_mdof_sim` to confirm the pipeline itself is healthy before
spending epochs on a record with a bad ceiling.

``w_acc``
---------
The pure work-energy reward is weak on stiffness and strong on damping; pure
acceleration NRMSE is the reverse — dropping a small damping term costs only a
few percent of acceleration error but a large share of the energy residual.
The blend ``residual = (1-w_acc)*energy + w_acc*accel_NRMSE`` is a single
stacked least-squares solve, so it keeps the closed-form constant fit.
"""

from __future__ import annotations

from discover_data import load_mat_data, identity_ceiling, plot_system_data
from discover_train import DISCOVER_TRAIN


def DISCOVER_MDOF_EXP(
    mat_path         = "AllData_ProcessedNOhit.mat",
    var_names        = None,     # None -> ['q1', 'q2', ...]
    # data conditioning
    trim_timesteps_front = 1000,
    trim_timesteps_back  = 70_000,
    desired_timesteps    = 600,
    plot_data        = False,
    show_ceiling     = True,
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
    max_traj         = 10,
    w_acc            = 0.9,
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
    center_features  = False,
    system_data      = None,     # pass a preloaded SystemData to skip loading
):
    """Identify every DOF of an experimental record with the DISCOVER pipeline."""
    system = system_data
    if system is None:
        system = load_mat_data(mat_path,
                               trim_timesteps_front=trim_timesteps_front,
                               trim_timesteps_back=trim_timesteps_back,
                               desired_timesteps=desired_timesteps,
                               var_names=var_names,
                               plot_data=False)
    print(f"[data] {system!r}")

    if plot_data:
        plot_system_data(system)

    if show_ceiling:
        worst, _ = identity_ceiling(system, verbose=True)
        print(f"[ceiling] worst per-DOF/per-trial ceiling: {worst:.4f} "
              f"— no expression can score above this\n", flush=True)

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
        center_features  = center_features,
    )


if __name__ == '__main__':
    DISCOVER_MDOF_EXP(
        mat_path             = "AllData_ProcessedNOhit.mat",
        var_names            = ['q1', 'q2'],
        n_epochs             = 500,
        batch_size           = 150,
        max_len              = 40,
        trim_timesteps_front = 1000,
        trim_timesteps_back  = 125000,
        desired_timesteps    = 500,
        plot_data            = True,
    )

    # 1000, 125000 for NOhit
    # 1000, 128000 for LOhit
    # 0, 0 for LONO combined
