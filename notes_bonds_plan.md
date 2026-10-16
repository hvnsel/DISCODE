# Plan: Newton's third law by construction ("bonds")

Status: **NOT APPROVED. Nothing implemented.** The user wants to be taught
every part of phases 0–4 first and will say when to implement. Do not start
coding before that. Written 2026-10-16 against `claude/discode-8token-grammar`
at `d161dd9`.

Companion notes: `notes_architecture.md`, `notes_soft_leaves.md` (directional
leaves), `notes_symattn.md`, `notes_symbolic_node.md`.

## 0. Summary

Today each DOF's equation is written, fitted and scored on its own, so a
coupling force can appear in one equation without its equal-and-opposite
partner in the other, and the runs show exactly that (section 1.2). The plan:
write a system as **ground terms per DOF** plus **one force law per connection
("bond")** in the relative variables Δᵢⱼ = qᵢ − qⱼ, and assemble the per-DOF
equations with opposite signs and mass weights. Every candidate then obeys
Newton's third law by construction, and the search also decides which DOFs are
connected and fits the mass ratios. The scoring (RK4 rollout, restart windows,
R_tf, R_f, trial weighting) is unchanged.

Masses and springs are only one instance. Section 2.0 states the idea for any
system whose parts exchange a conserved quantity (lumped ODEs, finite-volume
and FEM discretizations of PDEs). The method must not depend on any particular
dataset; the LO–NO rig below is an illustration only.

**Dropped 2026-10-16:** momentum balance and the test on measured data (old
Phase 0). The structure guarantees action–reaction, so the method never needs
to check it.

## 1. Why: what is wrong today

### 1.1 Action–reaction, stated pairwise

Every coupling term must appear in the equations of both entities it joins,
with opposite signs, scaled by one ratio for the whole pair (for masses,
c2 / c1 = −m1 / m2). Three ways to break it:

1. the term has the **same sign** in both equations (would need a negative ratio);
2. the term is in **only one** equation;
3. different coupling terms imply **different ratios**.

Illustration used below: the LO–NO rig (Wang & Moore 2021 Eq. 7; SIVA
parameters, Lopez & Moore approach III; already in `lono_comparison_plot.py`),
q1 = LO, q2 = NO, equal masses.

### 1.2 Our discovered equations break it

The pair the user asked LaTeX for on 2026-10-14:

    q̈1 = −4.585e8 q1³ − 1.512e4 q1 − 72.62 q̇1³ − 1.495 q̇1 + 2.593e7 q2³ + 1241 q2 + 132 q̇2³ − 2.989 q̇2
    q̈2 = 5.046e7 q1 q2² + 1314 q1 − 3.592e7 q2³ − 1182 q2 + 15.68 q̇2³ + 4.793 q̇2² − 0.5087 q̇2

Coupling terms, with coefficient c1 in q̈1 and c2 in q̈2:

| term | c1 | c2 | implied ratio −c2/c1 | verdict |
|---|---|---|---|---|
| q2 | +1241 | −1182 | 0.95 | fine |
| q2³ | +2.593e7 | −3.592e7 | 1.39 | ✗ rule 3 (disagrees with 0.95) |
| q1 q2² | 0 | +5.046e7 | none | ✗ rule 2 |
| q̇2 | −2.989 | −0.509 | negative | ✗ rule 1 |
| q̇2³ | +132 | +15.68 | negative | ✗ rule 1 |
| q̇2² | 0 | +4.793 | none | ✗ rule 2 |

The linear spring nearly pairs up; the cubic coupling and the damping do not,
and those are the parts with the least signal in the data. Two terms are
not physical at all: +15.68 q̇2³ in q̈2 pushes along the velocity (injects
energy), and 4.793 q̇2² has the same sign whichever way the mass moves.

### 1.3 Why the current design allows it

- `discover_policy.py` says so in its docstring: an internal coupling force
  "appears in two equations at once with opposite sign; with cross-slice
  attention the second DOF can copy the subtree the first one found". That
  copying is hoped for, never enforced.
- Each DOF is fitted on its own (`optimise_consts_energy`) and scored on its
  own (`energy_worker`, or `coupled_worker` against partners). Nothing ties the
  two equations' coupling terms together.
- RL keeps whatever fits best. On noisy data, breaking the law usually fits a
  little better, so it wins.

### 1.4 The published model in bond form needs 5 terms

With Δ = q1 − q2 and SIVA's numbers (β = 2.0012, so (q1 − q2)|q1 − q2|^β is the
cube to within the fit):

    G1(q1, q̇1) = −14 068 q1 − 2.803 q̇1            (LO to ground: k1/m, b1/m)
    G2(q2, q̇2) = 0                                 (NO is not grounded)
    F(Δ, Δ̇)   = −1 421.6 Δ − 0.5735 Δ̇ − 2.018e7 Δ³ (k2/m, b2/m, α/m)

    q̈1 = G1 + F        q̈2 = G2 − F                 (equal masses)

Expanding reproduces the published q̈1 and q̈2 exactly. Our runs use 15 terms
to approximate it.

## 2. The idea

### 2.0 The general form (not just masses and springs)

A system is a set of **entities** (a DOF, a mesh node, a finite-volume cell, a
finite element) that exchange some conserved quantity (momentum, energy, mass,
heat). Each entity's equation has two kinds of terms:

- **local terms**: depend only on that entity's own state (a support, a
  foundation, a body force, a source);
- **exchange terms**: one law per interaction, depending on the relative state
  of the entities it joins, computed **once** and added to one side and
  subtracted from the other.

Because every exchange term is added and subtracted, the exchanged quantity is
conserved by construction, whatever the policy writes. Instances:

| system | entity | exchanged quantity | exchange law |
|---|---|---|---|
| lumped ODE (masses) | mass / DOF | momentum | connection force |
| finite-volume PDE | cell | mass, momentum, energy | flux across a face |
| FEM | node (exchange via elements) | momentum | element force vector (sums to zero) |
| heat conduction | node / cell | heat | heat flux |

The method is the same in every case: the policy writes local laws and exchange
laws; the code assembles the equations.

### 2.1 Graph view

- **Nodes** are the masses, plus the ground.
- **Bonds** are the springs, dampers and contacts between masses.
- Each bond has one force law that depends only on how its two ends have moved
  relative to each other.
- Newton's third law: a bond pushes its two ends equally and oppositely.

The policy writes the pieces; the code assembles the equations and applies the
signs, so the policy can never get the signs wrong.

### 2.2 Equations for N DOFs

    mᵢ q̈ᵢ = Gᵢ(qᵢ, q̇ᵢ) + Σ_{j≠i} Fᵢⱼ(Δᵢⱼ, Δ̇ᵢⱼ),   Fⱼᵢ = −Fᵢⱼ,   Δᵢⱼ = qᵢ − qⱼ

Every Fᵢⱼ enters exactly twice with opposite signs, whatever the policy writes,
so the law holds by construction; nothing in the method needs to check it.

Three DOFs, every connection allowed:

    m1 q̈1 = G1 + F12 + F13
    m2 q̈2 = G2 − F12 + F23
    m3 q̈3 = G3 − F13 − F23

### 2.3 Search-space size (monomials up to cubic, the seed dictionary)

| DOF | free (every equation sees every variable) | bonds, all pairs | bonds, chain |
|---|---|---|---|
| 2 | 68 | 27 | 27 |
| 3 | 249 | 54 | 45 |
| 5 | 1 425 | 135 | 81 |
| 10 | 17 700 | 495 | 171 |

Free grows like N⁴; bonds like N² with all pairs allowed, N for a chain.
Counting: C(n+3, 3) − 1 monomials in n variables; a node or a bond has 2
variables, so 9 each.

### 2.4 Assumptions to state in the paper

- Every DOF is measured in absolute (inertial) coordinates.
- Connections act through the relative motion of their two ends (any spring,
  damper or contact between two masses does).
- Lumped translational DOFs with a diagonal mass matrix.
- External forcing enters as ground (node) terms.

## 3. Novelty

### 3.1 Gap table (from abstracts and quick searches; verify every cell)

| | writes equations | open-ended forms (no fixed library) | third law guaranteed | finds which DOFs connect | finds unknown masses |
|---|---|---|---|---|---|
| Graph neural ODEs (MCGNODE, BROGNET, Dynami-CAL, graph oscillators) | ✗ neural | n/a | ✓ | ✗ usually given | ? |
| Network symbolic regression (TPSINDy, ND²) | ✓ | partly | ✗ | ✗ given | n/a (first order) |
| EUCLID (FEM material laws) | ✓ | ✗ library | ✓ FEM force balance | ✗ mesh given | ✗ quasi-static |
| SINDy with tied coefficients | ✓ | ✗ library | ✓ if imposed | ✗ | ✗ must be known |
| RL symbolic regression (Φ-SO, DSR) | ✓ | ✓ | ✗ one equation at a time | ✗ | ✗ |
| Graph + coupling inference (NRI 2018; UDE structural inference, arXiv 2207.04962) | ✗ neural | n/a | ✗ | ✓ | ? |
| Our current DISCOVER | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Bonds (this plan)** | ✓ | ✓ | ✓ | ✓ | ✓ |

### 3.2 The claim

> The first symbolic-regression policy whose actions include the interaction
> graph itself and cross-equation bond laws assembled with action–reaction, so
> every sampled model obeys Newton's third law while the structure, laws and
> mass ratios are discovered together, with no fixed term library, known graph
> or known masses.

### 3.3 Which internals are novel

| piece | novel? | note |
|---|---|---|
| Bond slices: one law written once, assembled into two equations with opposite signs inside the generative policy | **yes, strongest** | in-generation constraints in RL symbolic regression (Φ-SO units, DSR priors) act within one equation; none tie equations together that we found |
| Unknown mass ratios fitted with the bonds | supports the claim | bilinear fit; identifiability statement |
| Topology as an action (STOP at term 0 on a bond slice = "no bond") | **partly** | graph inference from trajectories is known (NRI, UDE structural inference, network-kernel inference); doing it as discrete actions of an RL symbolic policy trained on the same reward is what we did not find |
| Edge embedding = dof_emb(i) + dof_emb(j) + bond-type | no | standard GNN practice; describe, don't claim |
| Graph-shaped attention mask | no | Graph Attention Networks (ICLR 2018), graph transformers; cite |

Optional strengthener for the topology piece: **per-bond credit** — score each
bond decision by the reward with versus without that bond. RL-specific, ours to
test, costs extra simulations.

### 3.4 Do not claim

- "First to enforce Newton's third law" (neural ODEs, FEM and constrained
  SINDy already do).
- "First to use relative coordinates" (textbook).

### 3.5 Why symmetry methods don't cover it

Yang et al. (NeurIPS 2024) impose symmetries of the whole vector field. On a
grounded system the ground breaks translation symmetry, so action–reaction is
not a symmetry of the equations; it is a structure (local terms plus paired
exchange terms). Only a structured generator can impose it.

### 3.6 Still to do before claiming anything

A proper literature review to fill the "?" cells and catch what quick searches
missed, then run the framing past the PI.

### 3.7 Sources found so far

- MCGNODE: https://arxiv.org/pdf/2209.10740v1
- BROGNET (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/file/57fabaa549352c52d5d312171b16970e-Paper-Conference.pdf
- Learning to simulate unseen physical systems (antisymmetric messages): https://arxiv.org/pdf/2201.11976
- Dynami-CAL GraphNet: https://arxiv.org/html/2501.07373v2
- Graph oscillators (MSSP 2024): https://www.sciencedirect.com/science/article/abs/pii/S088832702400195X
- ND² (Nature Computational Science 2025): https://fi.ee.tsinghua.edu.cn/~dingjingtao/papers/ND2-NCS-25.pdf
- Neural symbolic regression of network dynamics: https://arxiv.org/html/2410.11185v1
- Symmetry-informed governing equation discovery: https://mlanthology.org/neurips/2024/yang2024neurips-symmetryinformed
- Φ-SO: https://arxiv.org/abs/2303.03192
- Shared symbolic backbones (multi-output SR): https://arxiv.org/html/2607.26528
- Neural relational inference: https://proceedings.mlr.press/v80/kipf18a
- UDE structural inference: https://arxiv.org/pdf/2207.04962
- Joint network + interaction-kernel inference: https://www.sciencedirect.com/science/article/abs/pii/S1063520326000151
- Oscillator network structure inference: https://arxiv.org/pdf/2310.15990
- Restoring-force-surface background (Masri et al.; He, Xu & Masri 2012): https://www2.mdpi.com/2076-3417/9/2/306
- EUCLID: https://arxiv.org/abs/2010.13496v1 ; Bayesian-EUCLID: https://ar5iv.labs.arxiv.org/html/2203.07422
- Gonzalez 2026 (symmetry-informed NN-CC): https://arxiv.org/pdf/2601.21720

## 4. Baseline: how the code works today

- **Grammar** (`discover_core.configure_grammar`): leaves `x1..x{2N}` are the
  state columns (x1 = q1, x2 = q̇1, x3 = q2, ...), each with an implicit leading
  coefficient; optional directional leaves `dleaf`/`vleaf` (fitted linear
  combinations, the soft way to write (q1 − q2)³ as one term); ops `add`,
  `sub`, `mul`, `intpower` (amplitude, integer exponent), `power` (even and odd
  amplitude, exponent), optional `blend`.
- **Normalization**: expressions see normalized columns;
  `NORM_STATS = (X_mean[2N], X_std[2N], y_mean[N], y_std[N])` and the
  prediction is a_d = y_mean[d] + y_std[d]·f(normalized states).
- **Policy** (`discover_policy.TermBagPolicy`): input `(B, N, K, Lt)` — batch,
  DOF slice, term, token; slice order π permuted across the batch; additive
  embeddings: token, intra-term position (sinusoidal), `dof_emb`, `slot_emb`,
  `term_emb`; attention: earlier slices (if `cross_slice_attention`), earlier
  terms of the own slice, causal within a term; STOP only at a term's first
  position and never at term 0 (`stop_allowed`).
- **Sampling**: `sample_term_batch` returns `{dof: assembled tau}` per row, with
  the context each slice saw.
- **Fit and score**: `energy_worker` fits one DOF (`optimise_consts_energy`:
  closed form / VARPRO on (1 − W_ACC)·energy + W_ACC·acceleration NRMSE), then
  `candidate_reward` (energy or simulation). `coupled_worker` handles
  `sim_coupling` (alone / with tops / with peers, 'split' or 'blend');
  `refine_worker` does `sim_refine`; `refresh_partners` re-scores bests.
- **Trainer**: one buffer and `best_r` per DOF; `jgrpo_terms` per DOF, with
  group-relative advantages in log-residual units, never pooled across DOFs.
- **Seeds**: `discover_seed` STLSQ per DOF over monomials up to cubic (34 for 2
  DOF) plus single-variable power columns.
- **Simulation**: `rollout` (one DOF, others from the record), `rollout_set`
  (DOFs together), 'auto' windows, R_tf, R_f, `combine_trials`.

## 5. Design: bond mode

### 5.1 Slices and leaves

- **Node slice i**: allowed leaves {qᵢ, q̇ᵢ} only (x_{2i+1}, x_{2i+2}), plus
  constants; all ops as today. Gives Gᵢ.
- **Bond slice (i, j)**, i < j: allowed leaves {Δᵢⱼ, Δ̇ᵢⱼ} only. Gives Fᵢⱼ, the
  force on i from j; j gets −Fᵢⱼ.
- Enforced by a per-slice allowed-leaf set passed to `get_valid_tokens` /
  `term_valid_mask`.
- **Empty bags are legal**: an empty bond bag means "not connected"; an empty
  node bag means "not grounded" (the NO on the rig). Needs STOP at term 0 in
  bond mode and a representation for empty and single-leaf bags in buffers and
  teacher forcing (today a bag is stored as one flat tau; an empty bag has no
  tau, and a lone leaf is refused by `stop_allowed`).

### 5.2 Relative channels

- Build Δᵢⱼ = qᵢ − qⱼ and Δ̇ᵢⱼ = q̇ᵢ − q̇ⱼ **in physical units** from the raw
  trajectories, then normalize them with their own mean and std (new
  `NORM_STATS` entries). Subtracting two normalized columns is wrong: different
  means and scales, so it is not proportional to q1 − q2.
- In simulation the assembled equations compute Δ from the simulated states at
  every RK4 stage.

### 5.3 Assembly (`assemble_system`, new)

    aᵢ = wᵢ · [ Gᵢ + Σ_{bonds (i,j)} Fᵢⱼ − Σ_{bonds (j,i)} Fⱼᵢ ]

- wᵢ = 1/mᵢ with w₁ fixed: the overall mass scale cannot be identified from
  free response (scaling every force and mass together leaves accelerations
  unchanged).
- Output (a): **flat per-DOF equations** in physical units, the same strings as
  today ("q1ddot = ..."). Then `rollout_set`, R_tf/R_f, `combine_trials`, the
  plot scripts, `lono_comparison_plot.py` and `denormalize_expr` work unchanged.
- Output (b): a structured record (ground laws, bond laws, masses) for printing.

### 5.4 Joint constant fit (`fit_system_consts`, new)

- Unknowns: node amplitudes (one equation each), bond amplitudes (shared by two
  equations, with ± and the mass weights), exponents (VARPRO grids as today),
  mass weights.
- Given exponents and masses: **one stacked linear least-squares solve** over
  all DOFs' rows, each DOF's rows scaled as today (acceleration block by
  y_std, energy block by std(dKE)), blended by W_ACC.
- Masses known (μ = 1 on the rig): done. Unknown: outer search over log(wᵢ/w₁),
  i = 2..N (grid, then polish), inner linear solve (variable projection).
- Dissipative option: bounded least squares (`scipy.optimize.lsq_linear`), with
  odd velocity terms forced to oppose motion (node: amplitude ≤ 0 on q̇ᵢ terms;
  bond: ≤ 0 on Δ̇ terms of Fᵢⱼ) and even powers of velocity removed from the
  grammar.
- Exponent search becomes joint across slices; do it slice by slice (others
  held) or restrict the grids, to avoid a product of grids.

### 5.5 Policy

- Slices: N nodes + B bonds (B set by the topology option); `x` becomes
  `(batch, N + B, K, Lt)`; `slot_emb` sized N + B.
- Embeddings: node i = `dof_emb(i)` + kind(node); bond (i, j) = `dof_emb(i)` +
  `dof_emb(j)` + kind(bond), symmetric in i and j.
- **Topology as an action**: STOP allowed at term 0 of a bond slice; that first
  STOP logit is the "no bond" decision. (A dedicated per-pair decision head can
  come later.)
- Attention mask option: 'causal' (today's rule over all N + B slices) or
  'graph' (bond (i, j) sees nodes i, j and bonds sharing a node; node i sees its
  bonds), always restricted to earlier slots for causality.
- Slice order: the same base-order-plus-random-permutation mixture over all
  N + B slices.
- `build_term_inputs` must rebuild exactly the sampler's padding for N + B
  slices (the module's load-bearing invariant).
- Beam search off in bond mode at first.

### 5.6 Scoring

- A candidate is a whole system: `fit_system_consts` → `assemble_system` →
  `rollout_set` (all DOFs together) → existing per-DOF residual terms →
  `combine_trials` → r = 1 / (1 + residual).
- The alone / top / peer machinery (`sim_coupling`, `coupled_worker`,
  `refresh_partners`) is not needed in bond mode: every candidate is already
  complete and co-simulated.
- `sim_refine` adapted: tune the shared amplitudes and masses of the top
  systems on the simulation.
- Keep each DOF's channel residual alongside the total (for credit and reports).

### 5.7 Trainer and GRPO

- One buffer of systems: reward, per-slice terms, constants, masses, per-DOF
  residuals, context. Dedup key: tuple of each slice's sorted-term key.
- Advantage: group-relative over the batch of systems, in log-residual units.
- Per-slice credit option: node i uses DOF i's residual; bond (i, j) uses the
  mean of DOFs i and j. This answers the current "never pool across DOFs" rule,
  since a system residual mixes DOFs with different scales.
- Optional per-bond counterfactual credit (Phase 3): re-score without the bond;
  credit = r(with) − r(without).
- Critic off in bond mode at first.

### 5.8 Seeds

Structured STLSQ: columns per node (own monomials, 9 each) and per bond
(monomials of Δ, Δ̇, 9 each; each bond column enters DOF i with +wᵢ and DOF j
with −wⱼ), DOF rows stacked, masses fixed or scanned. This doubles as the
"SINDy with tied coefficients" baseline.

### 5.9 Options (all off by default; 'free' stays byte-identical to today)

- `structure = 'free' | 'bonds'`
- `topology = 'chain' | 'all' | [(i, j), ...]`
- `masses = None (fit ratios) | [m1, ..., mN]`
- `dissipative_only = False`
- `shared_bond_law = False` (one law on every bond, per-bond scale)
- `bond_attention = 'causal' | 'graph'`
- `slice_credit = 'system' | 'per_slice'`

### 5.10 Printing

- Structured: `ground 0: ...`, `bond 0–1: ...`, `m1/m0 = ...`.
- Flat per-DOF denormalized equations, paste-ready for the plot scripts.

## 6. Phases

### Phase 0 — dropped (2026-10-16)

The momentum-balance diagnostic and the test on measured data are out. The
method enforces action–reaction by construction, so it never needs to check
it, and a test tied to one dataset is not part of a general method. Counting
unpaired coupling terms (the rules in 1.1) survives only as an evaluation
metric in section 7.

### Phase 1 — bond mode, fixed topology (large)

- Sections 5.1–5.10 with the topology given (one bond for 2 DOF, 'chain' for
  more) and masses fixed or fitted.
- Done when: on simulated 2–3 DOF truths the exact structure is recovered; on
  the LO–NO record the 5-term structure (G1: q1, q̇1; G2 empty; F: Δ, Δ̇, Δ³)
  comes out with constants near SIVA's; free mode unchanged.
- Tests:
  - assembly property test: random bags → Σ mᵢ aᵢ contains no bond terms;
  - relative-channel normalization;
  - joint fit recovers known coefficients and masses exactly on noise-free
    synthetic data;
  - sampler respects per-slice leaves and the STOP rules;
  - teacher forcing reproduces the sampler's input for N + B slices;
  - free mode byte-identical (existing suites pass);
  - short seeded smoke run.

### Phase 2 — topology as an action (medium)

- 'all' topology; STOP at term 0 learned on bond slices.
- New simulated truths in `discover_data`: 3–8 DOF chains and stars, with and
  without cross bonds, unequal masses, bond laws outside any library (|Δ|^1.5,
  a contact gap, tanh friction).
- Done when: the true graph is recovered (topology F1 = 1 without noise;
  reported under noise) along with the laws.

### Phase 3 — learning improvements and ablations (medium)

- Per-slice credit; per-bond counterfactual credit (optional); shared bond law;
  graph attention.
- Ablations: each on/off; soft penalty vs hard structure; fixed vs fitted
  masses.
- Done when: the ablation table shows which pieces help.

### Phase 4 — beyond lumped ODEs (research, later)

- **k-node interactions**: k = 1 ground, k = 2 bond, k ≥ 3 element. Needed for
  beam bending (curvature uses wᵢ₋₁, wᵢ, wᵢ₊₁, so pairwise bonds only give
  string/wave equations) and for FEM elements.
- **Energy-based laws**: generate a potential and a dissipation function;
  derivatives give self-equilibrated forces for any k, and invariance to rigid
  rotation gives moment balance.
- **PDE limit**: a chain with one bond law, m üᵢ = F(uᵢ₊₁ − uᵢ) − F(uᵢ − uᵢ₋₁),
  becomes ρ u_tt = ∂σ/∂x as the spacing h → 0 (F(Δ) = A σ(Δ/h), m = ρAh). The
  relative variable becomes the strain, the bond law the constitutive law. Heat,
  diffusion and transport have the same shape: the bond is the flux across a
  face. A mesh fixes the connectivity (O(N) interactions) and one shared law is
  discovered however fine the mesh.
- **FEM test suited to the lab**: known linear FEM model plus unknown nonlinear
  joints or contacts. A joint is a bond across an interface; discover which
  joints are nonlinear and their laws.
- **Red flags** (raise with the PI):
  - modal or reduced coordinates destroy the structure (each mode mixes every
    node): discover in physical coordinates, project afterwards;
  - partial sensing: the law holds for the whole structure, not a measured
    subset; needs full-field data or a model of the unmeasured part;
  - non-reciprocal physics: gyroscopic effects, follower loads, an unmeasured
    fluid, active control, non-reciprocal metamaterials;
  - large rotations: need proper strain measures or vector relative positions
    with central forces.
- **Prior art there**: EUCLID (FEM force balance, fixed library, quasi-static);
  weak-form PDE discovery with flux-form libraries. Novelty must stay the policy:
  structure, laws and masses together, open-ended, dynamic data.

## 7. Experiments that prove the claim

1. Action–reaction: number of unpaired coupling terms (rules in 1.1), current
   DISCOVER vs bonds (zero by construction).
2. Topology: simulated 3–8 DOF with unknown graph; topology F1, spurious
   couplings.
3. Masses: unequal masses; mass-ratio error; vs tied-coefficient SINDy given true
   and wrong masses.
4. Open-ended laws: |Δ|^1.5, contact gap, tanh friction; library methods vs
   ours.
5. Interpretability: LO–NO constants vs SIVA (k1, b1, k2, b2, α).
6. Scaling: success rate vs N.

Baselines: free-mode DISCOVER; the structured seeds as tied-coefficient SINDy;
optionally PySR per equation; neural methods in the discussion only.
Metrics: exact structure recovery, coefficient error, simulation NRMSE on
held-out trials, violation, topology F1, mass-ratio error, evaluations to
solution.

## 8. Decisions still open (the user's)

1. Masses on the rig: fix μ = 1 (Wang & Moore equal masses) or fit? Suggest
   fitting for the paper, fixed as a check.
2. Dissipative-only rule on by default?
3. Topology default for N ≥ 3 ('all' with the topology action, or 'chain').
4. Is assuming "connections act through relative motion" acceptable for the
   claims?
5. Reward weights stay the user's own (the user reports NRMSE-only works well).
6. Scope and timeline against the PI's expectations.

## 9. Risks

- Joint exponent search cost; the bilinear fit with unknown masses has local
  minima (grid first).
- Credit-assignment noise from a single system reward (per-slice credit).
- Empty and single-leaf bags in buffers and teacher forcing.
- A DOF with no bonds has an unidentifiable mass (harmless, but report it).
- Small ground terms vs noise when deciding which DOFs are grounded.
- Measured coordinates must be absolute; check the sensors.
- Novelty: the literature review is not done.
- With no test on the data, the structure is an assumption: state it in the
  paper (section 2.4).

## 10. Teaching plan (check off as we go)

Restarted 2026-10-16 from the general form; the momentum-balance lessons were
dropped with Phase 0. Examples are illustrations; nothing depends on one
dataset.

- [ ] L1 The general form: entities, local terms, exchange terms, assembly; instances from ODEs to PDEs and FEM (section 2.0)
- [ ] L2 Exchange laws depend on relative state; relative variables and normalization (Phase 1)
- [ ] L3 The joint fit: shared coefficients, unknown scale factors (masses), variable projection (Phase 1)
- [ ] L4 Policy internals today (slices, terms, embeddings, attention mask, STOP) and what changes (Phases 1–2)
- [ ] L5 RL training: system-level reward, GRPO, buffers, per-slice credit (Phases 1–3)
- [ ] L6 Discovering which exchanges exist, as actions; empty bags; what NRI does differently (Phase 2)
- [ ] L7 Experiments, baselines, ablations and the novelty argument (Phases 2–3)
- [ ] L8 Beyond pairs: exchanges over k entities (elements), energies, the PDE limit, FEM, red flags (Phase 4)

## 11. Rules once the user approves

- Develop on `claude/discode-8token-grammar`, merge into
  `claude/discode-ppo-edit-agent`, run `discover_edit_test.py` and
  `discover_policy_test.py`, push. Commit trailers as for every commit here.
- 'free' stays the default and byte-identical; every new option is off by
  default.
- Do not change the user's driver settings or weights (`discover_mdof_exp.py`,
  `discover_score.py`, `discover_plot_*.py`, `discover_inspect.py` config
  lines); the user turns new options on.
- Long training runs are the user's; run unit tests and short seeded smoke tests
  only.
- One phase at a time, in order, reporting at the end of each.
