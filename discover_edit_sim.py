"""
discover_edit_sim.py
====================

PROTOTYPE ADD-ON.  Runs the move-based PPO agent (:mod:`discover_edit_train`)
on a simulated system from the EXISTING libraries -- ``duffing``, ``linear``
and ``vanderpol`` from :mod:`discover_sdof_sim`, ``coupled_duffing``,
``duffing_chain3`` and ``cubic_coupled`` from :mod:`discover_mdof_sim` -- so
its results line up one-for-one with the term-bag trainer's on the same data.

    python discover_edit_sim.py

prints the standard ``[truth]`` reward, the truth's score under the edit
objective (the number to beat), per-iteration progress, and ends with the same
paste-ready ``DISCOVERED_EXPRS`` block as the other drivers.
"""

from __future__ import annotations

from discover_data import build_truth_system, print_truth_rewards
from discover_edit_train import DISCOVER_EDIT_TRAIN
from discover_mdof_sim import get_mdof_system
from discover_sdof_sim import get_sdof_system

SDOF_KEYS = ('duffing', 'linear', 'vanderpol')
MDOF_KEYS = ('coupled_duffing', 'duffing_chain3', 'cubic_coupled')


def get_system(key):
    if key in SDOF_KEYS:
        return get_sdof_system(key)
    if key in MDOF_KEYS:
        return get_mdof_system(key)
    raise ValueError(f"Unknown system '{key}'. Choose from "
                     f"{sorted(SDOF_KEYS + MDOF_KEYS)}.")


def DISCOVER_EDIT_SIM(
    system_key  = 'duffing',
    # data generation (None = the registry's value)
    n_traj      = None,
    t_end       = None,
    n_pts       = None,
    seed        = None,
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
                            transcendental=train_kwargs.get('transcendental', False))
    return DISCOVER_EDIT_TRAIN(system, **train_kwargs)


if __name__ == '__main__':
    DISCOVER_EDIT_SIM(
        system_key = 'duffing',
        n_iters    = 100,
        n_envs     = 64,
        max_steps  = 10,
    )
