"""
discode_plot_sim.py
===================
Simulate discovered equations forward from a SIMULATED system's initial
condition and compare against the known truth.

Works for any number of DOFs: ``SIM_KEY`` may name any entry in the
:mod:`discode_sdof_sim` or :mod:`discode_mdof_sim` libraries.

USAGE
-----
1. Paste one discovered physical expression per DOF into ``DISCOVERED_EXPRS``.
   The trainer prints a paste-ready block when it finishes.
2. Set ``SIM_KEY`` (and ``SIM_OVERRIDES`` if the run overrode ``n_traj`` /
   ``t_end`` / ``n_pts`` / ``seed``) to reproduce the exact dataset that was
   trained on — the ICs are drawn from a seeded generator, so the same spec
   always gives the same trials.
3. ``python discode_plot_sim.py``

Variable names in expressions follow the system's ``var_names`` (printed on
load).  Supported syntax: ``+ - * / ** ^``, ``sin cos exp sqrt abs Abs sign
sgn``.

Reading the output
------------------
On simulated data the reference IS the truth, so any visible gap is the
discovered equation's error and nothing else — unlike the experimental plotter,
where measurement conditioning contributes too.  The truth expressions are
printed alongside for direct comparison of the coefficients.
"""

from __future__ import annotations

import sys

sys.path.insert(0, '.')

from discode_analysis import plot_discovered
from discode_data import build_truth_system

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

SIM_KEY       = 'coupled_duffing'   # any key from the sdof/mdof sim libraries
SIM_OVERRIDES = {}                  # e.g. {'n_traj': 2, 'n_pts': 4000, 'seed': 1}
TRIAL         = 0                   # which trial supplies the IC + comparison

# One discovered expression per DOF (RHS only, or with "<var>ddot = " prefix).
DISCOVERED_EXPRS = [
    "xddot = -5.874*x - 5.068e-6*xdot**7 - 0.1496*xdot + 1.504*y + 0.007891*ydot - 0.006921",
    "yddot = -6.287e-5*x*y + 1.5*x + 4.478e-5*xdot*y - 0.5002*y**3 + 0.0001182*y**2 - 4.0*y - 0.1*ydot - 3.128e-6"
]

# Simulation
T_END         = None        # override end time (None = use the full record)
SOLVER        = 'LSODA'     # 'RK45' | 'LSODA' | 'Radau'
MAX_STEP      = None        # override max step (None = 20 * dt)
BLOW_UP_LIMIT = 1e4         # stop when any |state| exceeds this (None = off)

# Energy residual
PLOT_ENERGY      = True
ENERGY_NORMALIZE = True

# ═══════════════════════════════════════════════════════════════════════════


def get_sim_spec(key):
    """Look ``key`` up in the SDOF and MDOF simulated libraries."""
    from discode_sdof_sim import _REGISTRY as SDOF_REG
    from discode_mdof_sim import _REGISTRY as MDOF_REG
    reg = {**SDOF_REG, **MDOF_REG}
    if key not in reg:
        raise ValueError(f"Unknown SIM_KEY '{key}'. Choose from {sorted(reg)}.")
    return reg[key]()


def main():
    if not DISCOVERED_EXPRS or not any(str(e).strip() for e in DISCOVERED_EXPRS):
        print("Nothing to plot — paste one expression per DOF into "
              "DISCOVERED_EXPRS.")
        return

    system = build_truth_system(get_sim_spec(SIM_KEY), **SIM_OVERRIDES)
    print(f"[data] {system!r}")
    print(f"[data] variables: "
          f"{', '.join(f'{v}, {v}dot' for v in system.var_names)}")
    for d, s in enumerate(system.truth_strs):
        print(f"[truth DOF {d}]  {s}")

    return plot_discovered(
        system, DISCOVERED_EXPRS, trial=TRIAL, t_end=T_END,
        solver=SOLVER, max_step=MAX_STEP, blow_up_limit=BLOW_UP_LIMIT,
        plot_energy=PLOT_ENERGY, energy_normalize=ENERGY_NORMALIZE,
        out_prefix=f"{system.name}_trial{TRIAL}",
        ref_label='Truth',
    )


if __name__ == '__main__':
    main()
