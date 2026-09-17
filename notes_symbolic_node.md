# What to take from Symbolic Neural ODEs

Status: **investigation, nothing implemented**. Boddupalli & Moehlis,
arXiv:2608.22112. Every number below the first section was measured against
the code on this branch, on `sdof_sim duffing`
(`qddot = -0.3*qdot - 1.0*q - 0.5*q^3`, 4 trials x 2000 pts, dt = 0.01),
with the paper's state-wise noise model (their eq. 4.1: per-state RMS times a
normal draw, reported as % NSR).

## The short version

Their architecture is not the transferable part. Their **objective** is.

They replace the one-step derivative-matching loss that SINDy and everything
like it uses with an H-step RK4 rollout loss, and every headline result in the
paper is downstream of that one choice. Our work-energy reward is on the wrong
side of it: it integrates the *error* along the record but never feeds the
model's own state back, so it sits in the H=1 regime that their Figure 12 shows
getting stuck.

Measured on our own duffing, fitting the true structure's three coefficients:

| NSR | our energy fit (c, k, a) | rel. err | rollout fit H=200 | rel. err |
|---|---|---|---|---|
| 0% | 0.3000, 1.0000, 0.5001 | 0.0000 | 0.3000, 1.0000, 0.5000 | 0.0000 |
| 5% | 0.3287, 0.9839, **-0.1562** | 0.475 | 0.2969, 0.9965, 0.5178 | **0.016** |
| 10% | 0.2763, 0.5631, **-0.1490** | 0.605 | 0.2998, 0.9950, 0.5345 | **0.025** |
| 20% | 0.1975, **-0.1602**, **-0.1502** | 0.934 | 0.3235, 0.9976, 0.5642 | **0.070** |

Truth is `(0.30, 1.0, 0.50)`. At 5% noise our fit already has the **wrong sign**
on the cubic; at 20% it has the wrong sign on the spring as well, i.e. it
reports an unstable system. The rollout fit keeps all three signs and lands
within 7% at 20% noise — a 13x accuracy gap, and a qualitative one.

## The thing I got wrong first

`notes_soft_leaves.md` lists "a rollout-consistency gate on elites" as a
defensible out-of-scope idea. That is the wrong half of the mechanism, and it
is measurably wrong.

A *gate* means: fit as we do now, then rerank survivors by their H-step rollout
error against the observed record. Five candidates on duffing, each given its
own best pointwise fit:

```
                                   reward       H=20       H=50      H=100      H=200
NSR 0%
truth        -cv -kq -aq^3       0.999972  5.350e-12  3.249e-11  8.414e-11  1.241e-10
truth + junk (+ b q v)           0.999978  8.797e-12  5.263e-11  1.308e-10  1.858e-10
  reward picks : truth + junk        <- wrong
  H=any  picks : truth               <- right

NSR 10%
truth        -cv -kq -aq^3       0.700784  4.610e-03  1.066e-02  3.165e-02  8.562e-02
linear       -cv -kq             0.676635  4.582e-03  1.053e-02  3.069e-02  7.346e-02
vdp-ish      -cv -kq -bq^2v      0.661537  4.268e-03  8.562e-03  2.269e-02  4.955e-02
truth + junk (+ b q v)           0.701287  4.554e-03  1.036e-02  3.088e-02  8.745e-02
  reward picks : truth + junk        <- wrong
  H=any  picks : vdp-ish             <- also wrong
```

On clean data the gate earns its keep: it reverses the parsimony inversion the
soft-leaf note flagged as unfixable by smoothing (reward prefers the junk term
by 6e-6; the rollout rejects it by 1.5-1.6x at every H). At any real noise level the
gate is useless, because it is reranking candidates whose coefficients are
already wrong. Refitting under the rollout loss fixes them; reranking after a
pointwise fit does not.

**The horizon has to be inside the fit.**

## Their Figure 12, reproduced on our system

The paper's most load-bearing empirical claim is that short horizons converge
to the wrong model while long ones converge to the right one *and get there in
fewer epochs*. Sweeping H in the coefficient fit:

```
rel. err of the fitted (c, k, a)
   H        5%      10%      20%
   0 (pw) 0.475    0.605    0.934      <- what we do now
   5      0.280    0.725    2.474
  10      0.161    0.434    1.233
  25      0.119    0.302    0.818
  50      0.101    0.188    0.492
 100      0.064    0.114    0.159
 200      0.016    0.025    0.070
```

Two things fall out:

1. **Monotone in H, not saturated at 200.** Whatever horizon we pick, longer is
   better on this system, and the benefit is still growing where I stopped.
2. **There is a threshold, and below it the horizon hurts.** At 20% NSR, H=5 is
   *worse* than our current pointwise fit (2.47 vs 0.93). This is their H=8 and
   H=16 curves failing to converge. A half-hearted horizon is worse than none.

It is not the initial guess doing the work. Starting from `(0.2, 1.2, 0.3)` and
from `(1.0, 0.1, 5.0)` gives identical answers to four decimals at every H and
every noise level — only the iteration count differs. For a fixed structure the
rollout landscape over coefficients is unimodal here.

## What it costs

Per candidate, on this problem:

```
our fit, closed-form linear path          0.97 ms
our fit, nonlinear path                   2.73 ms
our energy reward                         0.63 ms
                                        --------
current total per candidate              ~4 ms

one rollout forward, H=200, 36 segments  11.7 ms   (vectorised, lockstep RK4)
full rollout refit, H=200, 36 segments     343 ms  (7 residual evaluations)
full rollout refit, H=200, 144 segments    565 ms
```

The naive per-segment Python loop is 2.5 s; advancing every segment in lockstep
gets that to 0.34 s, so vectorising is worth an order of magnitude and is the
difference between "impossible" and "elite-only".

Budget arithmetic: a refit on the top 8 elites for 200 epochs is about **9
minutes** of added wall-clock. A refit on every sampled candidate is about
**4 hours**. So this is an elite-refinement stage, not a replacement reward.

## Proposal, in three tiers

Tier 1 and 2 are cheap and independent; tier 3 is the one that actually matters
and is the one to argue about.

1. **Rollout gate on elites, clean data only.** ~12 ms per elite. Fixes the
   junk-term inversion. Honest label required: it is a parsimony tiebreak on
   low-noise records, not a noise defence. On the experimental records this is
   near-worthless, per the table above.
2. **Report the fitted model's divergence.** Free, and we currently don't. At
   20% NSR the truth structure's pointwise fit blows up under its own dynamics
   (2.4e7 by H=1000) while the linear decoy stays bounded. A model that cannot
   integrate its own record is not a model, and we currently score it 0.563 and
   move on.
3. **Rollout refinement of elite coefficients.** Keep the pointwise energy fit
   as the cheap screen — it is 100x faster and it is what makes searching
   thousands of structures possible at all — then refit the promoted elites'
   coefficients under an H-step rollout loss and re-score. The *structure*
   search stays where it is; only the constants get the horizon treatment.
   This is the smallest change that buys the 13x.

Multiple shooting is part of why tier 3 works and should not be dropped: the
segments are re-initialised from observed states every `stride` samples rather
than integrated from one initial condition, which is what keeps the residual
bounded and the landscape unimodal. All the numbers above use 36–144 segments.

## What does not transfer

- **Their architecture.** Fixed K stacks x L operational layers with four
  hard-wired primitives (linear combination, product of powers, gated product,
  `w_out * [exp|sin|sgn](w_in * z)`). It is a supernet — every primitive is
  always present, and sparsity comes from an l1 or l1/2 penalty plus a
  post-training truncation sweep. This is the DARTS bargain the soft-leaf note
  argues against taking, and it shows in their own results: they say plainly
  that the method "doesn't always give models with the structure of the true
  model" and lean on Appendix A to argue the wrong-structure models are at
  least dynamically close. Our discrete search either finds the structure or
  does not; there is no truncation tolerance to tune.
- **Their gated product** `prod[sigma(w) z + (1 - sigma(w))]`, which softens
  "how many factors are in this monomial" by letting a gate drive a factor to
  the identity. We reach the same place for free: `(w . x)^0 = 1` turns a
  factor off, once the exponent range admits zero. Noted in the soft-leaf note;
  this paper is where the idea comes from.
- **mAIC's parameter count.** Their `k` counts surviving network weights after
  truncation. We have no analogue — VARPRO coefficients are not free parameters
  in the same sense, and our complexity is a token count. The *shape* of mAIC
  (goodness of fit over the horizon, plus a complexity term, plus the
  small-sample correction) is reusable; the specific `k` is not.
- **Their truncation sweep** over tolerances `[0.001, 0.01, 0.1, 1]` via
  sympy's `nsimplify`. This is a continuous method's way of getting back to a
  symbolic model. We start symbolic.
- **K-fold cross-validation.** They need it because they hit multiple local
  minima in a continuous parameter space. Our per-DOF, per-candidate fits are
  small and mostly convex; the local-minimum problem we have is in *structure*
  space, which K-fold does nothing for.

## What we do that they can't

Recorded so the comparison is not one-sided.

- **Per-DOF independence.** Our work-energy residual is an exact kinematic
  identity per DOF, so each DOF is fitted and scored in isolation with no
  co-simulation and no cross-DOF error contamination. Their rollout couples
  every state, so one bad DOF poisons the whole loss — and a rollout refinement
  would import exactly that coupling. This is the real cost of tier 3 and it is
  not priced in above.
- **No horizon hyperparameter.** Their H has to be tuned per system and they
  say so: H=1 for noise-free Rössler, 4 for the pendulum, 16 for Takens-Bogdanov
  and van der Pol, 32-256 for the chaotic examples. Get it wrong low and you
  converge to the wrong model; get it wrong high and you pay for it and can hit
  the stiffness problems they cite. We currently have no such knob, and tier 3
  introduces one.
- **Structure search, not weight pruning.** Their model class is fixed before
  training; ours is generated. Their expressivity is bounded by K and L.

## What their examples say about data, which is orthogonal to all of this

Their pendulum (4.4) and Hopf (4.3) sections make a point that has nothing to do
with the horizon: a single trajectory, or a trajectory confined to an attractor,
does not determine the vector field. Small-amplitude pendulum data is fit
perfectly well by a linear model; limit-cycle-only Hopf data is fit by any
system with a similar closed orbit. Their fix is multiple trajectories spanning
different amplitudes.

We already do this — `n_traj=4` with `ic_scale` spreading the initial
conditions, and `max_traj` (recently fixed to actually reach the worker) scoring
across all of them. Worth knowing that the mitigation is deliberate and that
dropping to one trajectory would reintroduce a documented failure mode.

## Open questions

- Every number here is one system, one noise seed, one structure. The
  coefficient-recovery gap should be re-measured on `cubic_coupled` (2 DOF,
  where the rollout coupling cost above actually bites) before anyone commits.
- Where is the threshold H as a function of dt and the system's fastest time
  scale? "H=5 is worse than nothing" needs to become a rule, not an anecdote.
- The rollout refit here was handed the true structure in physical
  coordinates. Doing it through the grammar and the normalisation means
  differentiating a compiled expression through an integrator; whether
  `least_squares` behaves as well on the general case is untested.
- Does refining elites' coefficients under a different objective than the one
  that promoted them destabilise J-GRPO? The advantage is z-scored per DOF over
  the sampled batch; injecting a differently-scored elite into that batch is
  not obviously safe.
