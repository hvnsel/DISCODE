"""
discode_plot_exp.py
===================
Simulate discovered equations forward from an EXPERIMENTAL initial condition
and compare against the record.

Works for any number of DOFs — set ``DOF_INDEX`` to an integer to look at a
single channel (matching a :mod:`discode_sdof_exp` run), or leave it ``None``
for the full multi-DOF record.

USAGE
-----
1. Paste one discovered physical expression per DOF into ``DISCOVERED_EXPRS``.
   The trainer prints a paste-ready block when it finishes.
2. Point the config at the same data the run used — the trim and downsample
   settings matter, since they change the time grid the equations were fitted
   on.
3. ``python discode_plot_exp.py``

Variable names in expressions follow ``VAR_NAMES``::

    VAR_NAMES = ['x', 'y']   ->  x, xdot, y, ydot
    VAR_NAMES = ['q1','q2']  ->  q1, q1dot, q2, q2dot

Supported syntax:  ``+ - * / ** ^``, ``sin cos exp sqrt abs Abs sign sgn``.

Reading the output
------------------
Two figures are produced and they answer different questions.  The forward
simulation compounds error over time, so a small frequency error diverges
visibly even when the structure is right; the energy residual is evaluated on
the MEASURED states and is exactly what the search optimised.  Trust the
residual for "is this equation right", the simulation for "is this equation
usable as a model".
"""

from __future__ import annotations

import sys

sys.path.insert(0, '.')

from discode_analysis import plot_discovered
from discode_data import load_mat_data, select_dof

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

MAT_PATH  = "AllData_ProcessedNOhit.mat"
VAR_NAMES = ['q1', 'q2']    # one name per DOF, in data-column order
DOF_INDEX = None            # None = all channels; an int = single-DOF slice
TRIAL     = 0               # which trial supplies the IC and the comparison

# One discovered expression per DOF (RHS only, or with "<var>ddot = " prefix).
DISCOVERED_EXPRS = [
]

# Data loading — must match the training run.
TRIM_FRONT        = 600
TRIM_BACK         = 120_000
DESIRED_TIMESTEPS = 2000

# Simulation
T_END         = None        # override end time (None = use the record's)
SOLVER        = 'LSODA'     # 'RK45' | 'LSODA' | 'Radau'
MAX_STEP      = None        # override max step (None = 20 * dt)
BLOW_UP_LIMIT = 1e4         # stop when any |state| exceeds this (None = off)

# Energy residual
PLOT_ENERGY      = True
ENERGY_NORMALIZE = True

# ═══════════════════════════════════════════════════════════════════════════


def main():
    if not DISCOVERED_EXPRS or not any(str(e).strip() for e in DISCOVERED_EXPRS):
        print("Nothing to plot — paste one expression per DOF into "
              "DISCOVERED_EXPRS.")
        return

    system = load_mat_data(MAT_PATH,
                           trim_timesteps_front=TRIM_FRONT,
                           trim_timesteps_back=TRIM_BACK,
                           desired_timesteps=DESIRED_TIMESTEPS,
                           var_names=VAR_NAMES if DOF_INDEX is None else None,
                           plot_data=False)
    if DOF_INDEX is not None:
        system = select_dof(system, DOF_INDEX, var_name=VAR_NAMES[0])
    print(f"[data] {system!r}")

    return plot_discovered(
        system, DISCOVERED_EXPRS, trial=TRIAL, t_end=T_END,
        solver=SOLVER, max_step=MAX_STEP, blow_up_limit=BLOW_UP_LIMIT,
        plot_energy=PLOT_ENERGY, energy_normalize=ENERGY_NORMALIZE,
        out_prefix=f"{system.name}_trial{TRIAL}",
        ref_label='Experimental',
    )


if __name__ == '__main__':
    main()
