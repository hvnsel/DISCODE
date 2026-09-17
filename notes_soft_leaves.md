# Proposal: soft leaves

Status: **not implemented**. Design note plus the measurements that motivated it.
Everything below was measured against the code on this branch, mostly on
`sdof_sim duffing` and `mdof_sim cubic_coupled`.

## The headline

A leaf stops being a choice among state variables and becomes a **fitted
direction in state space**.

```
before   add, mul, intpower, x1, x2, x3, x4, const, end     9 tokens, grows with DOFs
after    add, mul, intpower, dleaf, vleaf, const, end       7 tokens, fixed forever
```

`dleaf` evaluates to `a*q1 + b*q2 + ...` over the displacement channels,
`vleaf` to `a*q1dot + b*q2dot + ...` over the velocity channels. For an N-DOF
system each carries N weights, but the token count stays at seven.

In one sentence: the policy proposes *"a cubic displacement term and a damped
velocity term"*, and the fitter decides which directions.

Three things that must be true for it to work:

1. The weights are **fitted, not emitted and not learned by the policy**. They
   ride the same `least_squares` path in `_fit_consts_linear` that already fits
   continuous exponents, solved fresh per candidate. This is what keeps it out
   of supernet territory: no shared weights, no co-adaptation, no
   discretisation gap, none of the DARTS pathologies.
2. Displacement and velocity are separate tokens. Not for dimensional reasons
   (see "corrections" below) but because a mixture of position and speed is not
   a mechanical element anyone can read.
3. The block is chosen **per leaf, never per term**. `vanderpol`'s truth is a
   single term containing `x**2 * xdot`. Force a whole term into one block and
   that truth becomes unreachable.

## Where the search space is actually jagged

Measured on `duffing`. Five things can vary; only four were ever discrete.

| axis | jagged? | soft leaf fixes it? | evidence |
|---|---|---|---|
| coefficient sign and size | **no** | n/a | smooth unimodal curve through zero |
| which variable a leaf is | **yes, the big one** | **yes** | 0.086 → 0.9999 cliff becomes a monotone ramp |
| exponent | yes, integer grid of 3 | no | real structure at p = 2.5, 3.5 that the grid cannot reach |
| which operator | yes | **yes, see below** | `mul` and `intpower` are two spellings of one monomial |
| number of terms | yes | no, and it does not matter | reward is *flat* there, not jagged |

### The sign flip is already smooth

`xddot = -0.3*xdot - 1.0*x + c*x**3`, sweeping `c`:

```
c = -1.00  0.626      c = +0.00  0.626
c = -0.50  0.9999     c = +0.10  0.583
c = -0.25  0.770      c = +0.50  0.456
```

VARPRO already bought this. `-x**3` and `+x**3` are the same structure.

### Which variable is the real cliff

```
-0.5*x**3      reward 0.999853
-0.5*xdot**3   reward 0.086389      gap 0.913, nothing representable between
```

Relax the leaf into a mixture and the gap becomes walkable:

```
-0.5*(u*x + (1-u)*xdot)**3
u = 0.00  0.086    u = 0.50  0.309    u = 0.90  0.554
u = 0.20  0.157    u = 0.70  0.385    u = 1.00  0.9999
```

Monotone, so there is a gradient to follow.

### The fitted direction is the physically correct answer

On `cubic_coupled`, whose truth is `-1.2*(q1 - q2)**3`:

```
c3*(q1 + b*q2)**3, sweeping b
b = -2.00  0.812     b = -0.80  0.900
b = -1.50  0.860     b = -0.50  0.764
b = -1.20  0.924     b =  0.00  0.521
b = -1.00  0.99989   b = +1.00  0.769
```

Sharp, unimodal, peaked exactly at the true relative coordinate. Parameter
counts at matched reward:

```
one-hot leaves, cube expanded to 4 monomials   6 amplitudes   reward 0.99996
soft leaf                                      3 amplitudes
                                               + 1 weight     reward 0.99989
```

Letting the direction range over the *whole* state space still recovers it.
Fitted direction, converted back to physical units by dividing out the column
standard deviations:

```
normalised    q1 +0.7385  q1dot -0.0000  q2 -0.6743  q2dot +0.0000
physical      q1 +0.9999  q1dot -0.0000  q2 -1.0000  q2dot +0.0000
```

Exactly `q1 - q2`, with 3.4e-05 of the weight landing on velocity. **The fitted
weights are only interpretable after undoing the per-column scaling.**

## Why the operator axis dissolves too

`mul` and `intpower` are not two functions:

```
x * x      vs   x**2      identical: True
x * x * x  vs   x**3      identical: True
```

Write a term as a product of powers and both operators disappear:

```
term  =  c · Π_k ( w_k · x )^{p_k}
```

`mul` becomes "k > 1". `intpower` becomes "p ≠ 1". Neither is a decision any
more. That form covers nearly everything the sampler actually produces:

```
product of powers                    92.4%
product of powers of soft leaves      4.7%
residue                               2.9%      (3012 sampled terms)
```

The residue is terms containing a sum of unlike things. Each is itself a sum,
so it splits across separate bag slots, which already exist. The examples that
came up were degenerate anyway.

What is left discrete under that form:

| quantity | status |
|---|---|
| amplitude `c` | continuous, fitted |
| each factor's direction `w_k` | continuous, fitted |
| each factor's exponent `p_k` | continuous, once re-enabled |
| number of factors `k` | **soft**: `(w·x)^0 = 1` turns a factor off |
| each factor's block | discrete, two choices |
| number of terms | discrete, and a flatness problem not a smoothness one |

The `k` row is the identity-element trick from the Symbolic Neural ODEs paper
arriving on its own. Caveat: the exponent range currently excludes zero
(`intpower` uses nonzero integers in ±3, continuous ops use 0.5 to 7.0), so
admitting it is a prerequisite.

## Scaling consequence

The token table stops growing with the system:

```
N_DOF    current tokens    soft-leaf tokens
    1            7                7
    2            9                7
    3           11                7
    5           15                7
    8           21                7
```

And the number of distinct degree-3 monomials that must be written as separate
terms:

```
N_DOF   state vars   one-hot leaves   soft leaves
    1            2                9             1
    2            4               34             1
    5           10              285             1
    8           16              968             1
```

"A cubic in some direction" is one template at every size.

## Corrections to earlier reasoning

Recorded because both were wrong and the wrong versions are tempting.

**Dimensional analysis prunes nothing here.** Every leaf and every term carries
a free fitted coefficient with unconstrained units, so `c1*q1 + c2*q1dot` is
dimensionally fine for suitable coefficients. There is no structure this
grammar can write that units rule out. The argument for splitting displacement
from velocity is interpretability and identifiability, not dimensional
validity. It is a modelling prior, the same species of assumption as capping
tree depth.

**Cross-block multiplication must stay legal.** Multiplying unlike quantities
is always dimensionally fine, and `vanderpol` needs `x**2 * xdot`.

**The term bag is not permutation invariant, even with
`term_position_encoding=False`.** The module docstring overclaims. Measured:

```
A alone at term 0   vs  A at term 0 with B after it    max|Δlogit| 0.000e+00
A at term 0         vs  A at term 1, after B           max|Δlogit| 2.268e-01
prefix [A,B] vs [B,A], logits for term 2               max|Δlogit| 1.754e-02
```

The causal mask over terms means term 0 is encoded never having seen term 1
while term 1 is encoded having seen term 0. Dropping the band removes the
explicit index, not the ordering.

And that conditioning is not earning its keep:

```
                      bags with a duplicate    wasted term slots
term order visible          73.9%                  27.5%
term order hidden           72.4%                  23.2%
```

Three quarters of bags repeat themselves either way.

## What this does not fix

The reward is **flat** in the add-a-junk-term direction:

```
truth structure         0.999950
truth + one junk term   0.999972
```

The fitter hands the junk a coefficient around 1e-5. Smoothing helps where
there is slope to follow and does nothing on a plateau. Parsimony needs an
explicit complexity penalty, which is a separate decision from any of this.

## Explicitly out of scope

These came up in the same discussion and are each defensible on their own, but
none is required for the leaf change and none should be bundled into it:

- re-enabling continuous exponents (machinery exists, switched off via `POWER_OPS`)
- rollout refinement of elite coefficients, from the Symbolic Neural ODEs
  paper (see `notes_symbolic_node.md` — the *gate* form of this idea, as
  written here originally, was measured and does not work)
- a complexity penalty for parsimony
- making the term bag genuinely order-blind rather than causally chained
- any form of pretraining or transfer (ruled out)

## Open questions

- Every measurement above is on a **static** landscape. The monotone path
  exists; nobody has run the search with soft leaves. That is evidence the
  gradient is there, not that the search exploits it. **And a relaxation
  has since been built and measured** (`notes_symattn.md`): on `vanderpol`
  it parks in a spurious local minimum in 16 of 16 runs, at a residual
  280x worse than what the discrete search reaches by sampling the same
  structure. Smoothing can create minima the jagged space does not have.
  Any soft-leaf work has to be checked against that case specifically.
- **Conditioning.** A direction can be ill-posed where a variable choice never
  is. If two channels are correlated in a record, the fit returns something
  arbitrary that still scores well. Report the conditioning of the fitted
  direction alongside the reward.
- **Cost.** 59% of candidates currently take the closed-form linear path. A
  soft leaf makes the fit nonlinear, so essentially none would. Mitigation is
  to use soft leaves only on promoted elites.
- **Dedup breaks.** Two candidates match today if their term multisets match.
  Every soft-leaf candidate has a slightly different fitted direction, so
  everything is unique and the buffer fills with near-duplicates. The key has
  to become the template, or the template plus a coarsened direction.
- `split_terms`, the dedup key and the flat-reachability test all assume a leaf
  is a token and would need teaching.

## Suggested first experiment

Run `cubic_coupled` with and without soft leaves and compare three things:
whether the true relative coordinate is recovered, the parameter count at
matched reward, and the epochs needed to find the coupling at all. Use 1-DOF
`duffing` as the control, since there is no coupling and therefore no relative
coordinate, so soft leaves should buy nothing. If they appear to help on the
control, the measurement is picking up overfitting rather than structure.
