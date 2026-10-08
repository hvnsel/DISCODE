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
| `discover_core.py` | the engine: grammar, expression evaluation, VARPRO constant fitting, the rewards, the scoring worker, the critic |
| `discover_rollout.py` | the forward-simulation reward's integrator: one DOF against the measured record, or several integrated together (torch-free, shared with `discover_score.py`) |
| `discover_policy.py` | the terms-in-a-bag policy: model, attention mask, sampling, beam search, `jgrpo_terms` |
| `discover_policy_test.py` | correctness invariants for the above — run it before anything long |
| `discover_data.py` | everything that produces a `SystemData`: `.mat` loading (velocity and acceleration derived from the displacement by default), truth simulation, dataset prep, ceiling + truth-reward diagnostics |
| `discover_train.py` | `DISCOVER_TRAIN(system_data, ...)` — the one N-DOF training loop |
| `discover_analysis.py` | post-hoc: expression parsing, forward simulation, energy residual, comparison plots |

**Drivers** — what you open and run:

| file | what it does |
|---|---|
| `discover_sdof_sim.py` | 1-DOF demo on a synthetic system (Duffing / linear / Van der Pol) |
| `discover_mdof_sim.py` | N-DOF demo on a synthetic system (coupled Duffing, 3-mass chain, cubic-coupled, coupled beats) |
| `discover_sdof_exp.py` | one channel of an experimental `.mat` record |
| `discover_mdof_exp.py` | the full experimental record — the real target |

**Analysis** — paste discovered equations in, run:

| file | what it does |
|---|---|
| `discover_score.py` | reproduces the training reward + the data's ceiling. No plots |
| `discover_plot_sim.py` | forward-simulates discovered equations vs simulated truth |
| `discover_plot_exp.py` | forward-simulates discovered equations vs the experimental record |
| `discover_inspect.py` | one trial as the simulation reward sees it: per DOF the wavelet map and velocity spectrum, measured and (with equations) simulated, plus the sampling, windows and per-term residuals |
| `discover_diagnose.py` | is it the data, the setup or the search? Checks that the channels agree and how finely each DOF is sampled, finds the best equations the search space holds by sparse regression, scores them with the run's reward, and puts your run's equations next to them ([below](#data-setup-or-search-discover_diagnosepy)) |

All five handle any number of DOFs.

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
balance. No expression can beat it (under the energy reward — the simulation
reward below has no ceiling, and the experimental drivers skip it there). On simulated data it must be ~1.0 — if it
isn't, the trapezoid rule is under-resolving the `v*a` integrand and every reward
is capped; raise `n_pts`. With the file's own channels (`channels='measured'`,
[below](#velocity-and-acceleration-come-from-the-displacement)) it is typically
well below 1.0 on experimental data, because filtering breaks `a = dv/dt` and
`v = dq/dt`, and near the ceiling the ranking between expressions can invert;
with the default derived channels it is ~1.0. Printed by every driver.

**Negative headroom** (`ceiling - r_energy < 0`) means an expression closes the
balance better than the measured acceleration does. That is a data-integrity
warning, not a success: the disp/vel/acc channels are mutually inconsistent,
usually because they were filtered a different number of times each. The
default `channels='disp'` removes that by deriving velocity and acceleration
from the displacement.

## Velocity and acceleration come from the displacement

`load_mat_data` takes `channels`:

| `channels` | velocity and acceleration are |
|---|---|
| `'disp'` (default) | the first and second derivatives of the displacement on the trimmed, downsampled grid (fourth-order differences, error ~1e-5 at 50 samples per cycle) |
| `'measured'` | the file's own `Vel` and `Acc` |

An equation of motion can only match a record whose three channels describe one
motion, and the NOhit record's do not: at its 6.19 Hz mode the velocity is 10%
larger than the derivative of the displacement and the acceleration 10% larger
than the derivative of the velocity, with no phase shift; at 19.7 Hz both are
1.2% (`discover_diagnose.py`, below). That is the signature of velocity and
displacement integrated from the acceleration with a zero-phase high-pass near
2 Hz after each step, whose gain gives `1 + (2/f)^2`: 1.10 at 6.19 Hz, 1.01 at
19.7 Hz. A filter that reaches all three channels alike leaves a linear
equation of motion exactly satisfied, because it only rescales each mode. The
displacement carries the high-pass twice; its derivatives carry the same two,
so all three channels agree again.

Measured on a stand-in built that way (the diagnostic's NOhit equations plus a
little DOF 1 damping as the truth, the acceleration integrated twice with a
zero-phase 2 Hz high-pass after each step, five trials of 2.5 s at 1024 Hz),
whose channel check reads 1.13 at its 5.6 Hz mode and 1.01 at 19.8 Hz:

| | the file's channels | derived from the displacement |
|---|---|---|
| the list's best equations, alone | 0.89, 0.77 | 0.99, 0.99 |
| simulated together | 0.78, 0.66 | 0.99, 0.99 |
| DOF 1 recovered (truth `1996*q1 - 1398*q2 - 0.85*q2dot - 2.07e7*q2**3`) | `2990*q1 - 1495*q2 - 0.777*q2dot - 3.46e7*q2**3` | `2001*q1 - 1399*q2 - 0.850*q2dot - 3.44e7*q2**3 + 4576*q2*q2dot**2` |

The linear terms (stiffness, coupling, damping) come back within 1% on both
DOFs. The cubic ones
come back about 1.6× too large: the processing shrank the low mode's
displacement (to 0.78 here), and the cubic coefficients describe the
displacement as recorded. They simulate the record correctly; keep that in mind
when reading them as physics.

The loader prints, per DOF, the derived channels' RMS over the file's (below 1
where the high-pass took low-frequency motion out of the file's displacement)
and warns when the acceleration's is above 1.5, which would be noise in the
displacement amplified by differentiating twice. `channels='measured'` in the
driver's `load_mat_data` call goes back to the file's channels.

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

## Two reward schemes: `reward='energy'` or `reward='simulation'`

Every driver and `DISCOVER_TRAIN` take `reward`:

| `reward` | a candidate is scored by |
|---|---|
| `'energy'` (default) | the work-energy residual blended with acceleration NRMSE by `w_acc`, on the measured states (above) |
| `'simulation'` | integrating it forward from the measured initial state and comparing the result with the record: weighted NRMSEs of displacement, velocity and acceleration, plus an optional phase-blind time-frequency term (displacement NRMSE only by default) |

Both map a residual `e` to `r = 1/(1+e)` and both are per DOF, so everything
after the score — buffers, GRPO, the hall of fame — is shared. The constants are
fitted the same way under both: the closed-form work-energy / acceleration fit,
steered by `w_acc`. Under `'simulation'`, `w_acc` therefore only shapes the fit;
the simulation decides the ranking.

**Combining the trials (`trial_decay`).** Both rewards average a candidate's
residual over the trials by default. `trial_decay` ranks the trials by how badly
that candidate fits them and weights the k-th worst `trial_decay**(k-1)`: `1` is
the plain mean, `0.5` halves the weight at each rank (52, 26, 13, 6 and 3% over
five trials), `0` keeps only the worst trial. A behaviour only one trial shows —
beats in one of five experimental records, say — then counts most until it is
captured, without anyone choosing that trial: the ranking is per candidate, so
it follows whichever trial that equation fits worst. The constant fit is
untouched. On four `coupled_duffing` trials the truth still scores above 0.99
at `0.5`, while a wrong linear equation drops from 0.363 to 0.342 under the
simulation reward. The catch is that a trial that is simply bad data dominates
too. Set `TRIAL_DECAY` in `discover_score.py` to match the run.

**One DOF at a time, by default.** In a multi-DOF run, DOF d's equation is
integrated on its own, and the other DOFs' states are read off the measured
record at every step. A coupling term is judged against the partner's true
motion, and DOF d's score never depends on the partner's current expression —
the same isolation the energy reward has. The full coupled simulation of a
finished set of equations is what `discover_plot_sim.py` / `discover_plot_exp.py`
do, and an equation that works against the measured partner can still blow up
there.

**Coupled scoring (`sim_coupling`).** `sim_coupling = (f_top, f_peer)` sends
part of every DOF's batch to be simulated together with other equations, each
driven by the others' *simulated* states:

- **with top** (`f_top` of each DOF's equations): integrated together with every
  other DOF's top equation, its best so far;
- **with peers** (`f_peer` of the batch's rows): the policy samples all DOFs'
  equations jointly, one row at a time, and a peer row is integrated whole —
  each DOF's equation with the other DOFs' equations from the same sample;
- the rest are simulated alone, as above.

Constants are fitted the same way in every mode. Each equation is scored on its
own channel, but a coupled run that blows up charges every equation in it the
worst residual in the set, because once the set diverges none of them works.
Each epoch prints the best reward per mode, and the end of the run prints the
best equations simulated together. The true `coupled_duffing` equations score
above 0.999 in all three modes.

`sim_coupling_mode` says how the fractions apply:

| `sim_coupling_mode` | each equation is | its reward |
|---|---|---|
| `'split'` (default) | scored ONE way, drawn with those fractions | that way's score |
| `'blend'` | scored all three ways | `(1 - f_top - f_peer)*r_alone + f_top*r_top + f_peer*r_peer` |

Measured on `coupled_beats` (30 rows from an untrained policy, each DOF's best
alone equation as its top), an equation's score with the top tracked its alone
score: rank correlation 0.96 and 0.85, and the best scored 0.9985 both ways.
Its score with peers mostly reflected its row-mate: an equation at 0.9985 alone
scored 0.40 next to a poor partner. So under `'split'` a peer-mode equation
reaches the buffer only when its whole row is good, and an equation that works
only against the measured partner still wins whenever it is drawn alone.
Under `'blend'` that equation pays every time, but the peer lottery enters
every reward. To penalise equations that fail when coupled without that noise,
use `sim_coupling=(0.5, 0.0), sim_coupling_mode='blend'`. `'blend'` costs one
alone and one with-top simulation per equation, plus one joint simulation per
row.

Six more knobs, used only by `'simulation'`:

| knob | default | what it does |
|---|---|---|
| `sim_window` | `None` | `None` is one free run per trial, from its first sample to its last. A number restarts the simulation from the measured state every that many seconds (multiple shooting); `'auto'` every 3 periods of the fastest motion of the DOF being simulated (DOFs integrated together: of the fastest among them). Windows apply to the pointwise terms: a small frequency error then costs a bounded phase error per window instead of saturating the NRMSE near 1.4 over a long free run, where it stops ranking anything. Per DOF because on a record with a fast and a slow DOF the slow one would otherwise get windows of about one of its own cycles, where its frequency error cannot show (measured: a 2% error on a 1 Hz DOF next to a 4 Hz one, displacement NRMSE 0.054 on the 4 Hz DOF's windows, 0.193 on its own). The time-frequency and frequency-content terms always get a free run of their own, at half the RK4 substeps (`FREE_STEP_FRACTION`): both are phase-blind, and that run is ~97% of a candidate's simulation time. Measured on a 20 Hz + 6.5 Hz record, this moves `R_tf` by under 1e-3 at 10–20 samples per cycle and makes a candidate's simulation a third cheaper: at 2000 samples per trial, 0.78 s instead of 1.16 s, what 500 samples used to cost |
| `sim_weights` | `(1.0, 0.0, 0.0, 0.0)` | the weights of (displacement, velocity, acceleration, time-frequency[, frequency content]): `w_q*NRMSE(q) + w_v*NRMSE(qdot) + w_a*NRMSE(qddot) + w_tf*R_tf + w_f*R_f`. Three to five non-negative numbers summing to 1, or the run refuses to start; the ones left out are 0. The acceleration is the equation's own along its simulated trajectory, its right-hand side at the simulated state, compared with the measured acceleration. `R_tf` and `R_f` are below |
| `sim_coupling` | `(0.0, 0.0)` | the fractions of each DOF's equations simulated with the other DOFs' top equations and with the other DOFs' equations from the same sample (above). Non-negative, summing to at most 1; the rest are simulated alone. Needs two DOFs or more |
| `sim_coupling_mode` | `'split'` | `'split'` scores each equation one of those ways; `'blend'` scores it all three ways and weights the rewards by the fractions (above) |
| `sim_refine` | `0` | per DOF per epoch, the best this many equations get their amplitudes tuned on the simulation itself (below) |
| `sim_refine_evals` | `60` | simulations per tuned equation |

`sim_weights` takes a fifth weight, for the frequency-content term below; e.g.
`(0.15, 0.15, 0.15, 0.35, 0.2)` keeps the pointwise and envelope terms in the
same proportions as `(0.2, 0.2, 0.2, 0.4)`, roughly.

The integrator is RK4 on the record's own samples. The step is sized from the
data: at least 50 steps per cycle of the fastest measured motion, so an
oversampled simulated record is stepped every few samples and a coarse one gets
substeps. An equation that diverges is frozen at 10× the largest measured
amplitude and scores a large but finite residual.

**The time-frequency term (`w_tf`)** is for beats and anything else that lives in
how an oscillation's amplitude evolves. Point-by-point NRMSE is phase-sensitive:
over a long free run a small frequency error slides the simulation out of phase,
NRMSE climbs to ~1, and every slightly-wrong equation scores the same. Beats make
that worse, because the beat frequency is a *difference* of two close frequencies,
so it amplifies coefficient errors by f/Δf. `R_tf` instead compares the local
amplitude in each frequency band over time (an analytic Morlet filter bank on the
displacement, bands chosen from the record), plus a signed slow trend:

- **Phase:** a band's amplitude ignores phase. A quarter-period shift of a
  decaying tone costs 0.006, where NRMSE reads 1.42.
- **Beats:** two close frequencies in one band make its amplitude rise and fall.
  The timing of that rise and fall is compared directly, and each band's level
  and modulation depth regardless of timing.
- **Beat rate:** timing alone grades a beat period only while the simulated beat
  slips by less than about half a beat over the record; past that every wrong
  rate scored the same, so the search had no direction toward the right
  coupling. Each band's detrended log-amplitude therefore also has its power
  spectrum compared — a beat is a peak at its rate — by the earth mover's
  distance along log-frequency, which grows with the rate error at any size (a
  beat twice too fast costs one octave). On two tones 0.05 Hz apart, beats off
  by 1 / 3 / 5 mHz cost 0.20 / 0.49 / 0.68, beat rates 1.2× / 1.4× / 2× / 2.6×
  cost 0.80 / 1.01 / 1.60 / 1.87, rates 0.8× / 0.6× / 0.4× cost 0.90 / 1.36 /
  2.12, and no beat at all 2.42. A trial whose measured envelope barely moves
  has no rate to compare and counts in proportion to its envelope swing.
- **Moving centre:** a slower mode that the motion rides on sits in its own band
  and cannot fake a modulation.
- **No oscillation:** creep, drift or a decaying offset finds no bands and lands
  in the trend, so the term reduces to a smoothed NRMSE.
- **Graded on `coupled_beats`:** with mass 1 simulated against the measured mass
  2, a 0.25 / 1 / 2 / 5 % stiffness error costs 0.07 / 0.27 / 0.57 / 1.24, and
  dropping the coupling (no beats) costs 2.05.

Amplitudes discard the phase between DOFs, so keep some displacement weight, e.g.
`sim_weights=(0.5, 0, 0, 0.5)`. It costs ~3 ms per candidate on a 1300-sample
grid with 4 trials. Its constants live in `discover_rollout.py` (`TF_*`).

**The frequency-content term (`w_f`, the fifth weight)** asks where in
frequency the motion's energy sits, not when: the earth mover's distance between
the simulated and the measured velocity power spectra of each trial, measured in
cycles over the record (the distance in Hz times the record length) and taken as
`R_f = ln(1 + cycles)`. A spectrum moved by `df` is `T*df` cycles — exactly the
phase a frequency error of `df` piles up over the record — so:

- **Frequency errors at any size, however far the phase has slipped.** On a
  6.5 Hz mode over 5 s, 1% costs 0.28 and 10% costs 1.45 (`ln(1 + T*df)` to
  within 2%). A windowed NRMSE barely sees a 1% error (each window is too short
  for the phase to slip), and the envelope term, phase-blind with bands ~f/6
  wide, sees it only indirectly: where a measured partner drives the DOF at its
  true frequency, a detuned DOF beats against that forcing.
- **Energy in the wrong mode** costs its share times the distance it has to
  move. On a 6.5 + 20.3 Hz record, a 1% error on both modes costs 0.60, the
  20 Hz mode dying four times too fast 3.26, losing it entirely 3.93.
- **Beats** come out as two close peaks; having both, in the right places and
  proportion, is what makes the beat right.

Velocity rather than displacement, so a higher mode that is small in
displacement still counts; amplitude is normalised out. The logarithm keeps the
term on the scale of the others: moving energy between distant modes is tens
of cycles. Measurement noise of 1–3% of the signal leaves the truth at
0.004–0.03. On `coupled_beats` with `sim_weights=(0.2,)*5` the truth scores
0.996 and a 3% stiffer equation 0.725. It shares the time-frequency term's free
run and adds ~1 ms per candidate (5 trials of 2500 samples). Its constants are
`SPEC_*` in `discover_rollout.py`.

Measured per candidate on the simulated registry (4 trials, 1500–2500 samples),
a free run costs 25–150 ms and a 2 s window 3–15 ms, against ~2 ms for the energy
reward. Both rewards share the constant fit, which usually costs more: on an
untrained policy's free-grammar candidates its median was 0.3 s.

**Tuning the constants on the simulation (`sim_refine`).** Constants come from
the closed-form fit to the equation error: cheap, but blind to what a free run
magnifies, and biased whenever the channels were processed differently — with
`w_acc` near 1 the fit leans on the acceleration channel, so a filter on that
channel moves every constant. A beat period is a difference of two close
frequencies, so a coupling constant a few percent off puts the beats in the
wrong place, and the true structure then scores no better than wrong ones. Each
epoch, `sim_refine` takes each DOF's best few equations and moves their
amplitudes (coefficients, never exponents) by up to ±50% with Powell's method to
maximise their own reward, scored exactly as before, within `sim_refine_evals`
simulations. A tuned equation replaces its untuned twin in the buffer, and a
structure that keeps winning resumes from its tuned constants, so tuning
accumulates over epochs.

Measured on a proxy of a beats-in-one-trial record — `coupled_beats` with five
trials, four started near a single mode and the fifth with one mass displaced,
the acceleration channel low-passed — DOF 0's candidates, fitted then tuned (80
simulations each):

| acceleration filter | true structure | + spurious cubic | no coupling | wrong coupling (ẏ) |
|---|---|---|---|---|
| strong (2nd order, 0.6 Hz): fitted | 0.286 | 0.286 | 0.269 | 0.268 |
| strong: tuned | **0.628** | 0.621 | 0.395 | 0.410 |
| mild (1st order, 2 Hz): fitted | 0.776 | 0.755 | 0.371 | 0.382 |
| mild: tuned | **0.968** | 0.915 | 0.527 | 0.532 |

Fitted, the strongly filtered record cannot tell the true structure from wrong
ones; tuned, the coupled structures pull clear, and on the mild filter the true
equation comes back as `-4.401*x - 0.02999*xdot + 0.4011*y` (truth `-4.4*x -
0.03*xdot + 0.4*y`). A tuning simulation costs what a scoring one does — about
0.3 s on five 660-sample trials with `sim_window='auto'` — so `sim_refine=2,
sim_refine_evals=40` adds 160 per epoch, spread over the pool.

**What gets printed.** With `reward='simulation'` a run starts with each DOF's
fastest motion and how many samples a cycle of it gets — under ~10 the record
is too coarse to compare that motion point by point or to fit its constants
well, so raise `desired_timesteps` — the windows `sim_window='auto'` picked per
DOF, the range the frequency-content term compares (when weighted), and a
report per DOF and trial: the two strongest
spectral peaks, the envelope swing the
time-frequency term sees (the largest std of a band's detrended log-amplitude;
~0 for a steady or decaying oscillation, 0.3 and up for clear beats) and the
beat period read off the envelope's own spectrum. The trial whose swing stands
out is the one carrying the beats; a peak outside the bands is invisible to the
term. `discover_inspect.py` prints the same for any record and plots one trial:
per DOF the wavelet map (a beat is a band that swells and fades, a mode that
dies a band that goes dark) and the velocity spectrum, measured and — given
equations — simulated, with every residual term per trial. Every best equation
printed each epoch, and in the final summary, comes
with the equations it was simulated with to earn its score — per mode, each
other DOF's equation in physical units, or "the measured record" — even when
that is the other DOF's own best.

To score pasted equations with it afterwards, set `REWARD = 'simulation'` (and
the same `SIM_WINDOW` / `SIM_WEIGHTS`) in `discover_score.py`; `SIM_COUPLED = True`
scores them integrated together instead.

## Data, setup or search? (`discover_diagnose.py`)

When a run stalls, the record, the reward settings and the search itself can
all be at fault. With integer powers up to 3, every candidate the search can
write is a weighted sum of a fixed list of monomials (34 for two DOFs, plus a
constant), so the best it could find can be computed without training:

1. **The data.** Per DOF, the samples per cycle of its fastest motion, and at
   its main spectral peaks the gain and phase of velocity against d/dt
   displacement and of acceleration against d/dt velocity (ideal 1.00 and
   0°), on the file's own channels whatever `CHANNELS` the run uses. These
   come from cross-spectra, so no numerical derivative enters. Beyond 5% or
   5°, or under 10 samples per cycle, it is flagged.
2. **The search space's best.** Sparse regression on the monomials
   (SINDy-style thresholded least squares) gives the best equation of each
   size up to `MAX_TERMS` (the run's `max_terms`). Each is simulated and
   scored with the run's `SIM_WINDOW`, `SIM_WEIGHTS` and `TRIAL_DECAY`, and
   the best is tuned on the simulation as `sim_refine` does. With
   `CHANNELS = 'measured'` it is fitted twice: to the file's acceleration
   (what the training's constant fit leans on through `w_acc`) and to the
   derivative of its velocity. With the default `'disp'` the two are the same.
3. **Your equations** (`EXPRS`, the run's best per DOF), scored the same way.
4. **Trial by trial** (`PER_TRIAL`). The reward combines the trials, so it
   cannot say whether one equation could serve them all. Each trial is scored
   on its own: the all-trials best, the same terms with constants fitted to
   that trial alone (and a table of those constants, trial by trial), and the
   best of the list fitted and tuned to that trial alone
   (`TRIAL_TUNE_EVALS`), with your equations alongside. It adds about 1.5
   minutes on a NOhit-sized record.

| it prints | which means |
|---|---|
| the list's best simulates clearly better than your run's (by 0.1+) | the space holds a better equation and the search does not find it: settings or method |
| the list's best is poor too (under 0.5) | no equation of this form simulates the record: the data, or physics the measured states do not hold (forcing, friction, an unmeasured mode) |
| (`CHANNELS='measured'`) fitted to d/dt velocity the list simulates clearly better than fitted to the acceleration | the acceleration channel is biased, and the constant fit leans on it, so the constants come out wrong before the search starts |
| a channel or sampling flag | fix the data first |
| a trial fits clearly better on its own than with the all-trials equation (by 0.2+) | the trials need different constants or terms: the dynamics change between trials (amplitude-dependent behaviour outside the list, or trials run under different conditions); the constants table shows which |
| a trial fits poorly even on its own (under 0.6) | something in that trial lies outside the list (friction, contact, forcing, an unmeasured mode), or its data is bad |

Checked on four stand-ins, 15–60 s each:

- **Filtered acceleration** (`coupled_beats`, five trials, the acceleration
  low-passed as in the `sim_refine` table above): acceleration flagged at
  45–50° lag and gain 0.96–0.98. Fitted to the acceleration, every equation in the
  list simulates at 0.04–0.06; fitted to d/dt velocity, it recovers `-4.402*x -
  0.02999*xdot + 0.4000*y` (truth `-4.4*x - 0.03*xdot + 0.4*y`) at 0.85.
- **An unmeasured third mass** driving q2: q1 recovered exactly at 0.999; q2's
  best is 0.45, reported as nothing in the space doing well.
- **20 Hz and 6.5 Hz modes sampled at 102 Hz** (noise-free): flagged at 5
  samples per cycle, and still recovered within 7% on every coefficient, 0.99
  for each DOF simulated together.
- **Velocity and displacement integrated with a 2 Hz high-pass** (the stand-in
  [above](#velocity-and-acceleration-come-from-the-displacement)): flagged at
  1.13 on both pairs at 5.6 Hz; the list's best 0.89 and 0.77 on the file's
  channels, 0.99 and 0.99 on the derived ones.

On NOhit it read 1.10 at 6.19 Hz and 1.01 at 19.7 Hz on both pairs and both
DOFs, and the list's best was 0.59 on the file's channels and 0.59 / 0.60 on
the derived ones: the channels disagree, but that is not what caps the fit.

The trial-by-trial check on 1-DOF stand-ins: three trials with stiffness 40,
46 and 52 fit alone at 1.00 and with the all-trials equation at 0.41–0.69, and
the constants table reads 40, 46, 52; three trials of one equation score 1.00
both ways. On the high-pass stand-in the trials agree, their linear constants
within 1% of each other.

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
`q1**3` is `['intpower','x1']`. Fractional, signed or asymmetric powers come from
`power` (even amplitude, odd amplitude, exponent); its exponent is polished
continuously, so it is not the way to write a clean integer power.
