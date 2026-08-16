"""
discode_score.py
================
Score discovered equations with the SAME reward the training loop uses.
No simulation, no plots — paste expressions in, get the number out.

Paste one physical (denormalised) expression per DOF into ``EXPRS`` — the
trainer prints a paste-ready ``DISCOVERED_EXPRS`` block when it finishes — point
the config at the same data the run used, and execute the file.

Expressions use the ``VAR_NAMES`` convention::

    VAR_NAMES = ['q1','q2']  ->  q1, q1dot, q2, q2dot
    VAR_NAMES = ['x','y']    ->  x, xdot, y, ydot

Syntax accepted:  ``+ - * / ** ^``, ``sin cos exp sqrt abs Abs sign sgn``.

What it reports
---------------
  r_energy   the training reward, reproduced exactly:
             mean over time -> mean over DOFs -> mean over trials -> 1/(1+res)
  ceiling    the same quantity computed with the MEASURED acceleration.
             This is the best score ANY expression can achieve on this data.
             If r_energy is at the ceiling, the search has converged and the
             remaining error is in the data, not the equation.
  headroom   ceiling - r_energy.  A NEGATIVE headroom is a data-integrity
             warning, not a sign of success: it means the expression closes the
             energy balance better than the measured acceleration does, which
             can only happen when the disp/vel/acc channels are mutually
             inconsistent (e.g. filtered a different number of times each).

Note the reward convention: ``1/(1 + mean_d res_d)``, the reward of the mean
residual, NOT ``mean_d [ 1/(1+res_d) ]``.  The second is always the larger of
the two.  The per-DOF columns below are printed in the second form for
diagnosis; the headline number uses the first, so it matches training.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')

from discode_analysis import score_system

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

SOURCE = 'experimental'          # 'experimental' | 'simulated'

# ── experimental ───────────────────────────────────────────────────────────
MAT_PATH          = "AllData_ProcessedNOhit.mat"
VAR_NAMES         = ['q1', 'q2']
DOF_INDEX         = None         # None = every channel; an int = SDOF slice
# Must match the training run you want to compare against.
TRIM_FRONT        = 600
TRIM_BACK         = 120_000
DESIRED_TIMESTEPS = 2000

# ── simulated ──────────────────────────────────────────────────────────────
SIM_KEY = 'coupled_duffing'      # a key from discode_mdof_sim / discode_sdof_sim
SIM_OVERRIDES = {}               # e.g. {'n_traj': 2, 'n_pts': 4000}

# ── common ─────────────────────────────────────────────────────────────────
TRIALS           = None          # None = every trial; or e.g. [1, 2]
ENERGY_NORMALIZE = True

# One physical expression per DOF.
EXPRS = [
]

# ═══════════════════════════════════════════════════════════════════════════


def build_system():
    """Rebuild the dataset the run was trained on."""
    if SOURCE == 'experimental':
        from discode_data import load_mat_data, select_dof
        sysd = load_mat_data(MAT_PATH,
                             trim_timesteps_front=TRIM_FRONT,
                             trim_timesteps_back=TRIM_BACK,
                             desired_timesteps=DESIRED_TIMESTEPS,
                             var_names=VAR_NAMES if DOF_INDEX is None else None,
                             plot_data=False)
        if DOF_INDEX is not None:
            sysd = select_dof(sysd, DOF_INDEX, var_name=VAR_NAMES[0])
        return sysd

    if SOURCE == 'simulated':
        # Lazy: the sim libraries live in the drivers, which import torch.
        from discode_data import build_truth_system
        from discode_sdof_sim import _REGISTRY as SDOF_REG
        from discode_mdof_sim import _REGISTRY as MDOF_REG
        reg = {**SDOF_REG, **MDOF_REG}
        if SIM_KEY not in reg:
            raise ValueError(f"Unknown SIM_KEY '{SIM_KEY}'. "
                             f"Choose from {sorted(reg)}.")
        return build_truth_system(reg[SIM_KEY](), verbose=False, **SIM_OVERRIDES)

    raise ValueError(f"SOURCE must be 'experimental' or 'simulated', "
                     f"got {SOURCE!r}")


def score(exprs=None, system=None, trials=TRIALS, normalize=ENERGY_NORMALIZE,
          verbose=True):
    exprs = EXPRS if exprs is None else exprs
    if not exprs or not any(str(e).strip() for e in exprs):
        print(__doc__.split('What it reports')[0])
        print("Nothing to score — paste one expression per DOF into EXPRS.")
        return None, None

    system = build_system() if system is None else system
    N = system.n_dof
    if len(exprs) != N:
        raise ValueError(f"{len(exprs)} expressions but the data has {N} DOF")

    t = np.asarray(system.time, dtype=float)
    keep = list(range(system.n_trials)) if trials is None else list(trials)

    if verbose:
        fs = (len(t) - 1) / (t[-1] - t[0])
        print(f"\n{'=' * 72}")
        print(f"{system.name}   {len(t)} pts   {t[-1]-t[0]:.3f} s   "
              f"{fs:.0f} Hz eff   trials {keep}")
        print(f"{'=' * 72}")
        for d, e in enumerate(exprs):
            s = str(e).strip()
            s = s if len(s) <= 88 else s[:88] + ' ...'
            print(f"  DOF{d} ({system.var_names[d]}): {s}")

    r_pred, r_ceil, rows = score_system(system, exprs, normalize=normalize,
                                        trials=keep, verbose=verbose)

    if verbose:
        print(f"\n{'trial':>6}"
              + ''.join(f"{'r DOF'+str(d):>12}" for d in range(N))
              + f"{'r trial':>11}{'ceiling':>11}")
        for tr, rp, rc in rows:
            print(f"{tr:>6}"
                  + ''.join(f"{1/(1+rp[d]):>12.4f}" for d in range(N))
                  + f"{1/(1+np.mean(rp)):>11.4f}{1/(1+np.mean(rc)):>11.4f}")
        print(f"\n  r_energy = {r_pred:.4f}      ceiling = {r_ceil:.4f}"
              f"      headroom = {r_ceil - r_pred:+.4f}")
        if r_ceil - r_pred < -0.01:
            print("  -> NEGATIVE headroom: the expression closes the balance "
                  "better than the measured\n     acceleration does. The "
                  "channels are mutually inconsistent — check the data "
                  "conditioning,\n     not the equation.")
        elif r_ceil - r_pred < 0.01:
            print("  -> at the ceiling: the remaining error is in the DATA, "
                  "not the equation")
        print()
    return r_pred, r_ceil


if __name__ == '__main__':
    score()
