"""
discover_edit_sim.py
====================

PROTOTYPE ADD-ON.  Runs the move-based PPO agent (:mod:`discover_edit_train`)
on a simulated system from the EXISTING libraries -- any key of
:mod:`discover_sdof_sim`'s or :mod:`discover_mdof_sim`'s registry (``duffing``,
``vanderpol``, ``coupled_duffing``, ``coupled_beats``, ...) -- so its results
line up one-for-one with the term-bag trainer's on the same data.

    python discover_edit_sim.py

prints the standard ``[truth]`` reward, the truth's score under the edit
objective (the number to beat), per-iteration progress, and ends with the same
paste-ready ``DISCOVERED_EXPRS`` block as the other drivers.
"""

from __future__ import annotations

import discover_mdof_sim
import discover_sdof_sim
from discover_data import build_truth_system, print_truth_rewards
from discover_edit_train import DISCOVER_EDIT_TRAIN


def get_system(key):
    """A system spec from either registry, so new systems need no change here."""
    for lib in (discover_sdof_sim, discover_mdof_sim):
        if key in lib._REGISTRY:
            return lib._REGISTRY[key]()
    raise ValueError(f"Unknown system '{key}'. Choose from "
                     f"{sorted(discover_sdof_sim._REGISTRY) + sorted(discover_mdof_sim._REGISTRY)}.")


def DISCOVER_EDIT_SIM(
    system_key  = 'duffing',
    # data generation (None = the registry's value)
    n_traj      = None,
    t_end       = None,
    n_pts       = None,
    seed        = None,       # the DATA seed, as in the other drivers
    seed_policy = 0,          # the training seed (policy init, sampling)
    show_truth  = True,
    # everything else goes straight to DISCOVER_EDIT_TRAIN
    **train_kwargs,
):
    """Simulate ``system_key`` and train the edit agent on it."""
    spec = get_system(system_key)
    system = build_truth_system(spec, n_traj=n_traj, t_end=t_end,
                                n_pts=n_pts, seed=seed)
    if show_truth:
        print_truth_rewards(system, max_traj=train_kwargs.get('max_traj'),
                            w_acc=train_kwargs.get('w_acc', 0.5),
                            energy_normalize=train_kwargs.get('energy_normalize', True),
                            directional_leaves=train_kwargs.get('directional_leaves', False),
                            transcendental=train_kwargs.get('transcendental', False),
                            reward=train_kwargs.get('reward', 'energy'),
                            sim_window=train_kwargs.get('sim_window'),
                            sim_w_vel=train_kwargs.get('sim_w_vel', 0.0))
    return DISCOVER_EDIT_TRAIN(system, seed=seed_policy, **train_kwargs)


if __name__ == '__main__':
    DISCOVER_EDIT_SIM(
        system_key = 'duffing',
        n_iters    = 100,
        n_envs     = 64,
        max_steps  = 10,
    )
