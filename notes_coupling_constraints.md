# Plan: reciprocity and Newton's third law as training options

Status: **NOT APPROVED. Nothing implemented.** Written 2026-10-17 against
`claude/discode-8token-grammar` at `63a251f`, for the user to learn from and
decide on. Explained, with the proposed code changes, in the artifact
"DISCOVER Coupling Constraints".

## 0. Summary

- One new training option, `coupling = 'free' | 'reciprocal' | 'newton3'`.
  `'free'` is the default and stays exactly today's behaviour.
- `'reciprocal'`: after the policy writes a system, every coupling term gets
  its partner terms in the other equations, worked out by one rule, sharing
  one fitted coefficient. Every system produced passes the reciprocity test.
- `'newton3'`: reciprocal, and coupling terms may only depend on differences
  between DOFs. Partners are then mirror copies and coupling forces cancel.
- Which level to use is prior knowledge about the system, the way units are.
  Neither property holds for every system.
- The policy and the sampler barely change: partners are added after
  sampling, so there are no forced tokens. What changes is that each sampled
  row is fitted and scored as one system, and credit follows each term.

## 1. Definitions

All statements are for the equations after each is multiplied by its DOF's
mass (with equal masses, the accelerations as written).

### 1.1 Reciprocity

The effect of DOF j on equation i equals the effect of DOF i on equation j.

- Terms without velocities pass if ∂fᵢ/∂qⱼ = ∂fⱼ/∂qᵢ for every pair.
- Terms with velocities pass if ∂fᵢ/∂q̇ⱼ = ∂fⱼ/∂q̇ᵢ for every pair. Another
  DOF's position inside a velocity term only modulates it and needs no
  partner (q̇₁·q₂ in q̈₁ is a damper on DOF 1 whose strength depends on where
  DOF 2 is).
- For a linear system this is the textbook split of
  M q̈ + (D + G) q̇ + (K + N) q = 0: symmetric D and K are reciprocal, the
  skew-symmetric gyroscopic G and circulatory N are not.

### 1.2 Newton's third law

Coupling forces come in sets that cancel. With reciprocity in place this holds
exactly when every coupling depends only on differences: a function of
differences doesn't change when every DOF shifts by the same amount, and that
is the condition for a term's forces to add up to zero. Physically, a
connection that touches only two bodies can't know where they are, only how
far apart they are.

"Differences only" includes any nonlinearity of a difference
((q₁ − q₂)³, sign(Δ)|Δ|^1.5), velocity differences, combinations whose weights
sum to zero (q₁ − 2q₂ + q₃), and products of different differences.
Differences only make sense between DOFs of the same kind, along one axis.

### 1.3 They are independent

| | reciprocal | not reciprocal |
|---|---|---|
| obeys Newton's third law | spring or damper between two masses | actuator between two masses, driven by a sensor on mass 1 |
| breaks it | two pistons on one sealed air volume | motor on mass 1, driven by a sensor on mass 2, pushing on the ground |

Newton's third law brings reciprocity with it only for passive connections.

### 1.4 When each holds (prior knowledge)

- Newton's third law always holds in mechanics when everything is included
  (it is momentum conservation). A model includes only some bodies, so it
  fails in the equations whenever a coupling also pushes on something outside
  the model: ground, a cylinder wall, a fluid, a motor's mount.
- Reciprocity is a property of passive systems whose forces come from stored
  energy and ordinary damping. It fails with spinning parts, magnetic fields,
  an outside power source (active control) and moving material (flow over a
  wing, a brake pad on a spinning disc).
- So: impose `'reciprocal'` when the system is passive with none of those;
  impose `'newton3'` only if, on top of that, every coupling is a connection
  between modeled bodies of the same kind along one axis, with nothing else
  taking part. When unsure, use `'free'`, or fit with and without and keep
  whichever predicts held-out trials better.

## 2. The option

| level | what the policy may write | what the code adds after sampling |
|---|---|---|
| `'free'` | anything the grammar allows | nothing (today) |
| `'reciprocal'` | anything the grammar allows | each coupling term's partners, tied to it |
| `'newton3'` | local terms, or terms built only from differences | each relative term's mirror, tied to it |

`'newton3'` sits on top of `'reciprocal'` rather than being a second switch:
Newton's third law without reciprocity in practice only happens with powered
connections,
where nothing would be constrained anyway, and for a term touching three or
more DOFs "equal and opposite" needs the reciprocity rule to say how the force
splits.

### LO–NO in free response

Both hold, so `'newton3'` applies: the connection's linear spring, damper and
nonlinear spring act through q₁ − q₂ and q̇₁ − q̇₂; the LO's spring and damper
to ground are local terms; nothing spins, flows or is actively controlled.
The published Wang & Moore model passes at this level. Check first:

- q₁ and q₂ are each mass's motion from a fixed frame (not the NO measured
  relative to the LO);
- the masses: fix Wang & Moore's equal masses or fit the ratio;
- nothing else couples the two (a flexible frame would be a hidden body).

## 3. The mechanism

### 3.1 Is a term a coupling term?

A term in equation i couples to DOF j exactly when its derivative with respect
to DOF j's variable is nonzero. Which variables appear is a quick first pass;
the derivative is the real test (q₂ − q₂ contains q₂ and doesn't couple). A
term made of a local part and a coupling part is never split by hand: the
local part has zero derivative with respect to DOF j and drops out by itself.

### 3.2 The partner rule

For a term t in equation i and another DOF j:

1. Differentiate t with respect to DOF j's variable (positions for terms
   without velocities, velocities for terms with them).
2. If the result is nonzero, integrate it once in DOF i's own variable. That
   is the partner, and it goes into equation j with t's coefficient times the
   mass ratio mᵢ/mⱼ.

All partners of one term come from one parent (the term integrated once in
its own variable), so for a term touching three or more DOFs the partners also
agree with each other.

| term in q̈₁ (equal masses) | partner in q̈₂ |
|---|---|
| c·sign(Δ)\|Δ\|^1.5, Δ = q₁ − q₂ | −c·sign(Δ)\|Δ\|^1.5 |
| c·sign(q₂)\|q₂\|^p | c·p·q₁·\|q₂\|^(p−1) |
| c·\|q₂\|^p | c·p·q₁·sign(q₂)\|q₂\|^(p−1) |
| c·q₁·\|q₂\|^p | c·(p/2)·q₁²·sign(q₂)\|q₂\|^(p−1) |
| q₁³·(q₁ − q₂) | −q₁⁴/4 |
| (q₁ + q₂)·q₁ | q₁²/2 (the q₁² part drops out) |
| h(a₁q₁ + a₂q₂) (a `dleaf`) | (a₂/a₁)·h (zero if the fit sets a₂ = 0) |
| c·q₂·sin(q₁²) | none: needs ∫sin(q₁²) dq₁, no closed form |

For the term shapes the grammar writes, partners stay in the grammar:
differentiating or integrating a power shifts its exponent by one and swaps
signed and absolute; `blend` is closed under both. The exception is a bare
sign(q), whose derivative is a spike.

### 3.3 Completing a sampled system

For each slice, in generation order, the policy writes its own terms; then,
for each term it wrote, the partner rule runs against every other DOF and the
partners are added to those equations. Rules that keep it correct:

- only differentiate terms a slice wrote itself, never ones it received;
- send all of a term's partners at once, from the same parent;
- integrate without expanding (by substitution in the combination): plain
  integration of the derivative of c(q₁ − q₂)³ returns the mirror plus an
  extra −c·q₂³, a spring from DOF 2 to ground the policy would have to
  cancel (checked with sympy);
- merge repeats: terms with the same parent become one coefficient;
- drop a term whose partner has no closed form (later: mask it during
  generation).

Because the policy has no choice over partners, they can be added after the
whole row is sampled. Later slices already see earlier slices' terms through
cross-slice attention, so they don't need the partners as tokens.

### 3.4 Worked example (3 DOFs, equal masses, order 1 → 2 → 3)

    slice 1 writes   −k₁q₁,  c(q₁−q₂)³,  d·q₂q₃
      → slice 2 gets −c(q₁−q₂)³,  d·q₁q₃       → slice 3 gets d·q₁q₂
    slice 2 writes   −e(q̇₂−q̇₃)
      → slice 3 gets  e(q̇₂−q̇₃)
    slice 3 writes   −k₃q₃                      (local: nothing to add)

    q̈₁ = −k₁q₁ + c(q₁−q₂)³ + d·q₂q₃
    q̈₂ = −c(q₁−q₂)³ + d·q₁q₃ − e(q̇₂−q̇₃)
    q̈₃ =  d·q₁q₂ + e(q̇₂−q̇₃) − k₃q₃

Under `'newton3'`, d·q₂q₃ could not have been written (it uses other DOFs'
raw positions), and the remaining couplings cancel pairwise.

### 3.5 The Newton's third law level

One extra mask on leaves: every term is either **local** (only its own DOF's
variables; unconstrained, no partner) or **relative** (only differences).
Raw variables of other DOFs, and a local factor times a difference
(q₁³·(q₁ − q₂), which passes reciprocity but whose forces don't cancel), are
masked. For a function of one difference the partner rule always returns the
mirror, so the implant is a copy with the opposite sign (times the mass
ratio); a combination over three DOFs gets copies scaled by its weights. A
term built from several differences is completed by integrating from another
participant's position up to its own DOF's position, which keeps the parent a
function of differences: h(q₂ − q₃) in q̈₁ gets the parent (q₁ − q₂)·h(q₂ − q₃),
and the three faces cancel.

## 4. Guarantees and limits

- `'reciprocal'`: every system passes the reciprocity test; a coupling term
  can never sit in one equation without its partners, and they share one
  coefficient.
- `'newton3'`: in addition, every coupling's forces cancel.
- Non-reciprocal systems (gyroscopic, follower loads, active control) become
  impossible to write; use `'free'` for them.
- Energy can still be pumped in: a damper with the wrong sign is reciprocal.
  That is a separate constraint (sign bounds on damping coefficients).
- Both hold for the whole structure. With only some points measured, the
  parts in between are hidden bodies and the measured equations need not
  show equal and opposite pairs.

## 5. Beyond lumped systems

The rule behind Newton's third law is that internal forces depend only on how
the structure deforms, not on how it moves as a rigid body.

| setting | what replaces q₁ − q₂ | reciprocity means |
|---|---|---|
| masses on one axis | differences | symmetric stiffness and damping |
| FEM | strain from each element's nodes (weights sum to zero); invariance to rigid rotation too, with rotational DOFs | symmetric stiffness matrices |
| PDE | spatial derivatives; ρu_tt = ∂σ/∂x | self-adjoint spatial operator |
| large motion in 2D/3D | lengths and angles between points | symmetric tangent stiffness |

Structural cases that break reciprocity: rotating shafts, axially moving
strings and belts, pipes conveying fluid, follower loads, active control.

## 6. Proposed implementation (not approved)

### 6.1 Option plumbing (`discover_train.py`)

- `DISCOVER_TRAIN(..., coupling='free', masses=None)`. `masses=None` fits the
  mass ratios with DOF 0 fixed at 1; a list fixes them. Defaults keep
  `'free'` byte-identical; the user's driver files are not edited.
- In the coupled levels `sim_coupling` / `sim_coupling_mode` are ignored with
  a printed note: every candidate is already a complete system.
- Phase 1 runs with `directional_leaves=False` and `transcendental=False`.

### 6.2 Relative leaves (`discover_core.configure_grammar`, `discover_data`)

- `'newton3'` adds 2N tokens `r1..rN` and `rv1..rvN`: in DOF i's slice, `rj`
  means qᵢ − qⱼ and `rvj` means q̇ᵢ − q̇ⱼ. 2N tokens rather than one per pair,
  and products of differences not involving DOF i are still writable
  (q₂ − q₃ = r3 − r2 in slice 1).
- Built from the raw positions in physical units, then normalized with their
  own scale (subtracting already-normalized columns is wrong: each column has
  its own mean and scale). In simulation they are rebuilt from the simulated
  states at every RK4 stage (`_accel_fn` extended).

### 6.3 Leaf masks (`discover_policy.term_valid_mask`, `dc.get_valid_tokens`)

- New arguments: the slice's DOF and the coupling level. Under `'newton3'`:
  other DOFs' raw leaves masked; `ri`/`rvi` masked in DOF i's own slice; once
  a term has a local leaf, relative leaves are masked, and the other way
  round.
- The beam path (`beam_term_candidates`) passes the same arguments.
- No forced tokens, so the training log-probs need no change; masks keep the
  existing convention (applied at sampling, `discover_policy.py:631`).

### 6.4 Completion step (new `discover_coupling.py`)

- `complete_system(row, order, level)`: for every slice's own terms, run the
  partner rule against every other DOF and return, per DOF, its own terms
  plus received partners. Each partner carries the source term's constant
  symbols (amplitudes, exponents), so the fit ties them.
- Works on the physical-units form of each term (the substitution
  `denormalize_expr` already does), because DISCOVER writes terms in scaled
  variables and scales each equation's output by y_std.
- Integration: substitution when the term is a function of one combination,
  otherwise separable (the derivative splits into a factor in the own
  variable times a factor free of it), otherwise the term is dropped and the
  reason recorded. Power terms use the grammar's own power rules: sympy's
  general integrator returns piecewise forms for symbolic exponents.
- Memoized by term structure (tokens with constants as symbols), lambdified
  with the constants as arguments, so sympy runs once per distinct shape.
- Merges terms with the same parent into one coefficient.

### 6.5 Joint constant fit (`discover_core.fit_system_consts`, new)

- Today each DOF is fitted alone (`optimise_consts_energy` →
  `_energy_contexts` → `_fit_consts_linear`).
- New: one fit for the whole system. Unknowns: each term group's amplitudes
  and exponents, plus mass ratios unless given. Objective: the sum over DOFs
  of each DOF's current residual (its own work-energy balance blended with
  acceleration NRMSE by `w_acc`, each scaled as today), with each DOF's
  prediction including the partners it received.
- Fast path: exponents and mass ratios fixed, amplitudes are linear: stack
  every DOF's rows; a group's column covers its own DOF's rows and each
  partner DOF's rows (times the mass ratio) → one least-squares solve.
  Exponents on the existing grids, the source's exponent only (a partner's is
  tied, e.g. p − 1). Mass ratios by an outer search (1-D for 2 DOFs).
  Fallback: system-wide nonlinear least squares, as today's fallback.

### 6.6 Scoring (`discover_core.system_worker`, new)

- One task per sampled row: completion → joint fit → every DOF simulated
  together (`rollout_set`, as `coupled_simulation_rewards` does) → per-DOF
  rewards; a set that blows up scores 0 for every DOF, as now. Under
  `reward='energy'`, each DOF's energy residual with its partners included.
- Beam extras have no row; in the coupled levels beam search is off at first.

### 6.7 Training and credit (`discover_train.py`, `discover_policy.jgrpo_terms`)

- Per-DOF buffers and per-DOF GRPO stay. Each entry also stores the row it
  came from, so a DOF's best equation comes with the partners it received.
- Credit per term: a term's tokens get the mean of the advantages of the DOFs
  it appears in (local: its own DOF; coupling: its own DOF and every DOF that
  received its partners). Each advantage is computed in its own DOF's
  log-residual units first, so rewards are never pooled raw.
- `jgrpo_terms` takes an advantage per term instead of one per entry.

### 6.8 Seeds and printing

- `seed_list` rows go through the same completion and joint fit before
  entering the buffers. A reciprocal STLSQ (columns tied across DOFs) later.
- Printing: each DOF's equation with received partners marked by their
  source DOF, shared coefficients and the mass ratio; plus the flat
  denormalized equations, paste-ready for the plot scripts as today.

### 6.9 Tests (no long training runs)

- Partner rule on every row of the table in 3.2, symbolically.
- Property test: random grammar terms → complete → numeric check at random
  states that mass-weighted position and velocity Jacobians are symmetric
  (`'reciprocal'`) and that each group's forces sum to zero (`'newton3'`).
- Normalization: partners agree in physical units with scale-only and
  centered features.
- Joint fit recovers coefficients and the mass ratio exactly on noise-free
  2- and 3-DOF reciprocal systems.
- Sampler: no masked leaves under `'newton3'`.
- `'free'` byte-identical: existing suites pass. One short seeded smoke run
  per level.

### 6.10 Order of work

1. Two DOFs (LO–NO size): plumbing, relative leaves and masks, completion
   for leaves, add/sub/mul, intpower and power, joint fit (masses fixed or
   ratio fitted), whole-system scoring, per-term credit, printing, tests.
2. N DOFs: couplings over three or more DOFs, merging across several slices,
   zero-sum `dleaf`, `blend` terms, reciprocal seeds.
3. A trained or exact expert that masks non-completable terms during
   generation instead of dropping them; sign bounds for passivity as their
   own option; FEM and PDE leaves (strains from a mesh, spatial derivatives).

## 7. Decisions still open (the user's)

1. One option with three levels (suggested) or two independent switches.
2. LO–NO masses: fixed equal (Wang & Moore) or fitted ratio.
3. Credit per term from the start (suggested) or per DOF first.
4. Non-completable terms: drop the term (suggested for Phase 1) or reject
   the whole system.
5. Directional leaves and `blend` off in the coupled levels for Phase 1?
6. Passivity (sign bounds) as a separate option, and when.

No novelty is claimed here: masks for constraints (Φ-SO, DSR) and
must-include constraints (NeuroLogic decoding) already exist.

## 8. Rules once the user approves

- Develop on `claude/discode-8token-grammar`, merge into
  `claude/discode-ppo-edit-agent`, run `discover_edit_test.py` and
  `discover_policy_test.py`, push. Commit trailers as for every commit here.
- `'free'` stays the default and byte-identical; every new option is off by
  default.
- Do not change the user's driver settings or weights; the user turns new
  options on.
- Long training runs are the user's; run unit tests and short seeded smoke
  tests only.
- One phase at a time, reporting at the end of each.
