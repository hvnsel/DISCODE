# DISCODE — DIScover Coupled Ordinary Differential Equations

Diffusion-model deep symbolic regression (GRPO-trained, after Bastiani et al. 2025)
that recovers equations of motion from measured or simulated response data, using a
**work-energy reward** instead of pointwise acceleration NRMSE or forward-simulation
error.

## File map

**Engine** — shared, not edited day to day:

| file | what it is |
|---|---|
| `discode_core.py` | the engine: grammar, expression evaluation, VARPRO constant fitting, the reward, J-GRPO, the scoring worker |
| `discode_data.py` | everything that produces a `SystemData`: `.mat` loading, truth simulation, dataset prep, ceiling + truth-reward diagnostics |
| `discode_train.py` | `DISCODE_TRAIN(system_data, ...)` — the one N-DOF training loop |
| `discode_analysis.py` | post-hoc: expression parsing, forward simulation, energy residual, comparison plots |

**Drivers** — what you open and run:

| file | what it does |
|---|---|
| `discode_sdof_sim.py` | 1-DOF demo on a synthetic system (Duffing / linear / Van der Pol) |
| `discode_mdof_sim.py` | N-DOF demo on a synthetic system (coupled Duffing, 3-mass chain) |
| `discode_sdof_exp.py` | one channel of an experimental `.mat` record |
| `discode_mdof_exp.py` | the full experimental record — the real target |

**Analysis** — paste discovered equations in, run:

| file | what it does |
|---|---|
| `discode_score.py` | reproduces the training reward + the data's ceiling. No plots |
| `discode_plot_sim.py` | forward-simulates discovered equations vs simulated truth |
| `discode_plot_exp.py` | forward-simulates discovered equations vs the experimental record |

All three handle any number of DOFs.

## Quickstart

```bash
python discode_sdof_sim.py     # does the pipeline work?  (recovers Duffing in a few epochs)
python discode_mdof_sim.py     # does it work with coupled DOFs?
python discode_mdof_exp.py     # the real problem
```

Each run ends with a paste-ready block:

```
DISCOVERED_EXPRS = [
    "xddot = -0.5001*x**3 - 1.0*x - 0.3*xdot",
]
```

Paste it into `discode_score.py` or a plotter, point the config at the same data
(the trim/downsample settings change the time grid, so they must match), and run.

## Two numbers to read before the reward

**The ceiling.** The reward the *measured* acceleration scores on its own energy
balance. No expression can beat it. On simulated data it must be ~1.0 — if it
isn't, the trapezoid rule is under-resolving the `v*a` integrand and every reward
is capped; raise `n_pts`. On experimental data it is typically well below 1.0
because filtering breaks `a = dv/dt` and `v = dq/dt`, and near the ceiling the
ranking between expressions can invert. Printed by every driver.

**Negative headroom** (`ceiling - r_energy < 0`) means an expression closes the
balance better than the measured acceleration does. That is a data-integrity
warning, not a success: the disp/vel/acc channels are mutually inconsistent,
usually because they were filtered a different number of times each.

## `w_acc`

```
residual = (1 - w_acc) * energy_residual  +  w_acc * accel_NRMSE
```

The two objectives are sensitive to different terms. Dropping a damping term costs
~2.5% acceleration NRMSE but ~32% energy reward; dropping a stiffness term is the
reverse. Pure energy therefore cannot rank stiffness well, and pure NRMSE
effectively never selects a small dissipative term. Both are linear in the
coefficients, so the blend is one stacked least-squares solve and the closed-form
constant fit is preserved. Default `0.5`.

## DOFs are scored independently

`∫ v_d·a_d dt = ½(v_d² − v_d(0)²)` is an exact per-DOF kinematic identity — coupling
forces are already inside `a_d` — so each DOF's balance closes on its own. Mixing a
partner's residual into the score adds no within-epoch information (it is a shift
shared by every candidate of that DOF) while making rewards incomparable *across*
epochs as the partner's elite changes. There is no co-simulation in the reward path.

Coupling terms are still discoverable: in an MDOF run every DOF's states are grammar
variables, so DOF 0's expression may reference `q2`, `q2dot`, etc.

## Adding a system

Write one `TruthSystem` and register it in the driver's `_REGISTRY`:

```python
TruthSystem(
    name='my_system',
    accel_fns=[a0, a1],          # a_d(state), state = [q1, qd1, q2, qd2, ...]
    truth_strs=[...],            # human-readable, printed for comparison
    truth_taus=[...],            # optional: enables the [truth] reward printout
    var_names=['x', 'y'],
    t_end=25.0, n_pts=2500, n_traj=4,
    ic_scale=[1.5, 1.5, 1.5, 1.5],
    seed=0,
)
```

`ic_scale` sets the amplitude trials are drawn at — put it where the nonlinearity
actually lives, or a cubic term is unidentifiable. `n_pts` needs >~40 points per
cycle of the fastest mode; `identity_ceiling` checks and warns.

For `truth_taus`, the state layout is `x1 = q1`, `x2 = qd1`, `x3 = q2`, ... Every
variable leaf carries an implicit fitted coefficient, so `['add','x1','x2','end']`
is `c1*q1 + c2*qd1`. Powers come from `intpower` (coefficient + integer exponent):
`q1**3` is `['intpower','x1']`.
