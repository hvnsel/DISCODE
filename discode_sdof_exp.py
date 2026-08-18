"""
discode_sdof_exp.py
===================

DISCODE on a **single DOF of experimental data**.  One channel is sliced out of
a MATLAB record and identified on its own.

This is legitimate rather than a shortcut: the work-energy relation
``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an exact per-DOF kinematic
identity, so DOF ``d``'s balance closes without any knowledge of the other
DOFs — whatever coupling forces they exert are already contained in ``a_d``.
The discovered expression may of course reference only the states of this DOF,
so any true coupling shows up as unexplained residual.  If you want the
coupling terms, use :mod:`discode_mdof_exp`, which keeps every channel as a
grammar variable.

Read the ceiling before reading the reward
------------------------------------------
On real data the measured acceleration does not close its own energy balance
perfectly — filtering, drift and integration artefacts all leak in — so the
best score ANY expression can reach is below 1.0.  ``show_ceiling`` prints that
number.  A reward at the ceiling means the search has converged and the
remaining error is in the data; a reward *above* it means the data channels are
mutually inconsistent, which is a warning, not a success.
"""

from __future__ import annotations

from discode_data import load_mat_data, select_dof, identity_ceiling, plot_system_data
from discode_train import DISCODE_TRAIN


def DISCODE_SDOF_EXP(
    mat_path         = "SN_data.mat",
    dof_index        = 0,        # which channel of the record to identify
    var_name         = 'x',
    # data conditioning
    trim_timesteps_front = 1000,
    trim_timesteps_back  = 70_000,
    desired_timesteps    = 600,
    plot_data        = False,
    show_ceiling     = True,
    # search
    n_epochs         = 300,
    batch_size       = 200,
    max_len          = 20,
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
    # policy architecture (see DISCODE_TRAIN)
    architecture          = 'joint',
    cross_slice_attention = True,
    n_layers              = 4,
    d_model               = 128,
    slice_order           = 'random',
    sample_order          = 'reward',
    seed_policy           = None,
    center_features  = False,
    system_data      = None,     # pass a preloaded SystemData to skip loading
):
    """Identify one experimental channel with the DISCODE pipeline."""
    full = system_data
    if full is None:
        full = load_mat_data(mat_path,
                             trim_timesteps_front=trim_timesteps_front,
                             trim_timesteps_back=trim_timesteps_back,
                             desired_timesteps=desired_timesteps,
                             plot_data=False)

    system = select_dof(full, dof_index, var_name=var_name)
    print(f"[data] {system!r}")

    if plot_data:
        plot_system_data(system)

    if show_ceiling:
        worst, _ = identity_ceiling(system, verbose=True)
        print(f"[ceiling] worst per-DOF/per-trial ceiling: {worst:.4f} "
              f"— no expression can score above this\n", flush=True)

    return DISCODE_TRAIN(
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
        architecture          = architecture,
        cross_slice_attention = cross_slice_attention,
        n_layers              = n_layers,
        d_model               = d_model,
        slice_order           = slice_order,
        sample_order          = sample_order,
        seed                  = seed_policy,
        center_features  = center_features,
    )


if __name__ == '__main__':
    DISCODE_SDOF_EXP(
        mat_path             = "SH_data.mat",
        dof_index            = 0,
        n_epochs             = 300,
        batch_size           = 150,
        max_len              = 40,
        trim_timesteps_front = 600,
        trim_timesteps_back  = 120_000,
        desired_timesteps    = 2000,
        plot_data            = True,
    )
