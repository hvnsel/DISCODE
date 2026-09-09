# DISCOVER — coupled ODE discovery from response data

Deep symbolic regression (GRPO-trained, after Bastiani et al. 2025) that recovers
equations of motion from measured or simulated response data, using a
**work-energy reward** instead of pointwise acceleration NRMSE or forward-simulation
error.

The policy is a single **term-structured transformer** ("terms in a bag"): each
DOF's equation is generated as a bag of short terms summed under a root `add` the
model never has to write, and every DOF can condition on the terms already
committed for the others. See [Policy architecture](#policy-architecture).

## File map

**Engine** — shared, not edited day to day:

| file | what it is |
|---|---|
| `discover_core.py` | the engine: grammar, expression evaluation, VARPRO constant fitting, the reward, the scoring worker, the critic |
| `discover_policy.py` | the terms-in-a-bag policy: model, attention mask, sampling, beam search, `jgrpo_terms` |
| `discover_policy_test.py` | correctness invariants for the above — run it before anything long |
| `discover_data.py` | everything that produces a `SystemData`: `.mat` loading, truth simulation, dataset prep, ceiling + truth-reward diagnostics |
| `discover_train.py` | `DISCOVER_TRAIN(system_data, ...)` — the one N-DOF training loop |
| `discover_analysis.py` | post-hoc: expression parsing, forward simulation, energy residual, comparison plots |

**Drivers** — what you open and run:

| file | what it does |
|---|---|
| `discover_sdof_sim.py` | 1-DOF demo on a synthetic system (Duffing / linear / Van der Pol) |
| `discover_mdof_sim.py` | N-DOF demo on a synthetic system (coupled Duffing, 3-mass chain, cubic-coupled) |
| `discover_sdof_exp.py` | one channel of an experimental `.mat` record |
| `discover_mdof_exp.py` | the full experimental record — the real target |

**Analysis** — paste discovered equations in, run:

| file | what it does |
|---|---|
| `discover_score.py` | reproduces the training reward + the data's ceiling. No plots |
| `discover_plot_sim.py` | forward-simulates discovered equations vs simulated truth |
| `discover_plot_exp.py` | forward-simulates discovered equations vs the experimental record |

All three handle any number of DOFs.

## Quickstart

```bash
pip install -r requirements.txt
python discover_sdof_sim.py     # does the pipeline work?  (recovers Duffing in a few epochs)
python discover_mdof_sim.py     # does it work with coupled DOFs?
python discover_mdof_exp.py     # the real problem
```

Each run ends with a paste-ready block:

```
DISCOVERED_EXPRS = [
    "xddot = -0.5001*x**3 - 1.0*x - 0.3*xdot",
]
```

Paste it into `discover_score.py` or a plotter, point the config at the same data
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

There is one policy: a causal transformer that writes each DOF's equation as a
**bag of terms**. Every ground truth here is a sum of forces, and a sum has no
order — yet a flat sequence policy has to pick one, and the root `add` it must
write first costs a whole nesting level. So each term is generated as its own
short subexpression, with its own depth and length budget (`max_terms` slots per
DOF, `max_term_len` tokens per term), and the terms are stapled under an `add`
that is never a token the model emits. A STOP action closes the bag.

All DOFs are written by the same network, one *slice* per DOF, in a slice order
that varies across the batch, so a DOF can attend to the terms already committed
for the DOFs before it. An internal coupling force appears in two equations at
once with opposite sign; the second DOF can copy the subtree, and the sign is
free because VARPRO fits the leading coefficient.

Every driver exposes these knobs and passes them straight through to
`DISCOVER_TRAIN`:

| knob | default | what it does |
|---|---|---|
| `term_position_encoding` | `True` | the model knows which term came first. `False` drops that one embedding band and nothing else — terms `1..j-1` become indistinguishable to term `j`, and the bag is a genuine bag. Mask, sampler and update are byte-identical between the two |
| `cross_slice_attention` | `True` | `False` confines attention to a DOF's own slice: N independent term-bag policies sharing weights. This is how to tell whether cross-DOF structure sharing is doing anything. Use `system_key='cubic_coupled'`, whose `(q1-q2)^3` expands to four monomials in both equations — linear coupling is a single leaf and proves nothing |
| `term_grammar` | `'free'` | `'varpro'` restricts every term to the constant-fitter's closed-form domain (powers only over bare variables), which keeps every candidate off the slow nonlinear fit |
| `max_terms`, `max_term_len` | `8`, `8` | bag capacity per DOF and token budget per term |
| `n_layers`, `d_model` | `4`, `128` | transformer depth and width; the parameter count is printed at startup |
| `slice_order` | `'random'` | mixes permuted and base slice orders across the batch, so the model learns to condition in either direction — with a fixed order DOF 0 would never see DOF 1 |
| `sample_order` | `'reward'` | the epoch's base order: most-converged DOF first, so the others condition on it |

`max_len` is the token window of the per-DOF critic; the policy's own length
budget is `max_terms × max_term_len`.

Buffer entries are `(reward, tau, consts, context)`. `tau` is the flat assembled
sum, so the critic, hall of fame, dedup and scoring never see terms; the update
splits it back into terms deterministically, so each term is replayed against
exactly the prefix it saw when it was drawn. The context records the slice
order, the slice's slot, and the slices that preceded it — exactly what the
sample could see under the attention mask — so an entry stays reproducible in
isolation and comparable across epochs. Two bags with the same terms in a
different order are the same expression, so the buffer dedups on the sorted
term multiset.

The grammar limits are `MAX_TREE_DEPTH = 5` and `MIN_EXPR_LEN = 2`; at the older
4 / 6 the linear-oscillator and van der Pol truths in `discover_sdof_sim.py` were
unreachable for any policy.

Run `python discover_policy_test.py` before anything long. It checks the
term-level causality of the mask, that the update's teacher-forcing tensor
reproduces the sampler's padding byte for byte, and that every sampled bag
assembles to something the flat grammar could itself have written.

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
