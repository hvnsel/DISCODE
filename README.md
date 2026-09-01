# DISCODE — DIScover Coupled Ordinary Differential Equations

Deep symbolic regression (GRPO-trained, after Bastiani et al. 2025) that recovers
equations of motion from measured or simulated response data, using a
**work-energy reward** instead of pointwise acceleration NRMSE or forward-simulation
error.

The default policy is a single **autoregressive transformer** that writes every
DOF's expression as one sequence, so each DOF conditions on the ones already
written. The original per-DOF masked-diffusion policies are still available as
`architecture='independent'`. See [Policy architecture](#policy-architecture).

## File map

**Engine** — shared, not edited day to day:

| file | what it is |
|---|---|
| `discode_core.py` | the engine: grammar, expression evaluation, VARPRO constant fitting, the reward, J-GRPO, the scoring worker, the masked-diffusion policy |
| `discode_policy.py` | the joint autoregressive policy and the terms-in-a-bag policy: models, attention masks, sampling, beam search, `jgrpo_ar` / `jgrpo_terms` |
| `discode_policy_test.py` | correctness invariants for the above — run it before anything long |
| `discode_data.py` | everything that produces a `SystemData`: `.mat` loading, truth simulation, dataset prep, ceiling + truth-reward diagnostics |
| `discode_train.py` | `DISCODE_TRAIN(system_data, ...)` — the one N-DOF training loop |
| `discode_analysis.py` | post-hoc: expression parsing, forward simulation, energy residual, comparison plots |

**Drivers** — what you open and run:

| file | what it does |
|---|---|
| `discode_sdof_sim.py` | 1-DOF demo on a synthetic system (Duffing / linear / Van der Pol) |
| `discode_mdof_sim.py` | N-DOF demo on a synthetic system (coupled Duffing, 3-mass chain, cubic-coupled) |
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

## Policy architecture

Every driver takes `architecture=`, plus the knobs below. Defaults are the joint
autoregressive policy.

| config | flags | what it isolates |
|---|---|---|
| **V0** | `architecture='independent'` | the original: N one-shot masked-diffusion policies |
| **V0+AR** | `architecture='joint', cross_slice_attention=False` | autoregression alone |
| **B** | `architecture='joint', cross_slice_attention=True` | + cross-DOF structure sharing |
| **C1** | `architecture='terms'` | terms in a bag: each DOF's expression is generated as short independent *terms* summed at assembly; the model still knows term order |
| **C2** | `architecture='terms', term_position_encoding=False` | + drop the one embedding band that carries term order — the bag becomes a bag |

**Terms in a bag (C1 / C2).** Every ground truth here is a sum of forces, and a
sum has no order — yet the flat policy has to pick one, and the root `add` it
must write first costs a whole nesting level. The `terms` architecture generates
each term as its own short subexpression (with its own depth and length budget,
under `max_terms` / `max_term_len`) and staples them under an `add` that is never
a token the model emits. A STOP action closes the bag. C1 and C2 share the mask,
sampler and update byte-for-byte; C2 only removes the term-index embedding.
`term_grammar='varpro'` additionally restricts every term to the constant-fitter's
closed-form domain (powers only over bare variables), which keeps every candidate
off the slow nonlinear fit. The grammar limits also moved with this change —
`MAX_TREE_DEPTH` 4→5 and `MIN_EXPR_LEN` 6→2 — because at the old values the
linear-oscillator and van der Pol truths in `discode_sdof_sim.py` were
unreachable for *any* policy. That shifts every pre-existing baseline once.

Two separate defects motivate this, and they are worth tracking separately.

**Position factorisation.** The masked-diffusion policy is called *once* on an
all-`MASK` input, and every token is drawn from the resulting position-wise
marginals. It can learn "position 4 is often `intpower`" but never "*given*
position 3 is `intpower`, position 4 should be `x1`" — all structural coherence
comes from the grammar mask and from whatever VARPRO fits. Autoregression fixes
this, and the fix applies to a 1-DOF run too, which is the cleanest place to
measure it (`discode_sdof_sim.py`, no jointness to confound it).

**Cross-DOF structure.** An internal coupling force appears in two equations at
once with opposite sign. Independent policies must discover the shared subtree
twice; a joint policy can copy it, and the sign is free because VARPRO fits the
leading coefficient. This only shows up when the shared subtree is more than one
token — use `system_key='cubic_coupled'`, whose `(q1-q2)^3` expands to four
monomials in both equations. Linear coupling is a single leaf and proves nothing.

If **B** beats V0 but **V0+AR** does not, the gain is cross-DOF sharing. If V0+AR
already captures it, the joint trunk is unnecessary and this should revert to N
independent AR models.

V0 has N policies and the joint variants have one, so parameter counts do not
match by construction; both are printed at startup, and `n_layers` / `d_model`
are exposed so a capacity-matched run can be configured rather than assumed.

Other knobs: `slice_order` (`'random'` mixes permuted and base slice orders
across the batch, so the model learns to condition in either direction — with a
fixed order DOF 0 would never see DOF 1) and `sample_order` (`'reward'` puts the
most-converged DOF first so the others condition on it).

Buffer entries are `(reward, tau, consts, context)`. The context records the
slice order, the slice's slot, and the slices that preceded it — exactly what the
sample could see under the causal mask — so an entry stays reproducible in
isolation and comparable across epochs. It is `None` under `'independent'`.

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
