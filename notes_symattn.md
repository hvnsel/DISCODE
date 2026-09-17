# SymNN's idea, wired with attention

Status: **prototype, measured, not wired into the pipeline.** Builds on
`notes_symbolic_node.md` (their objective) and `notes_architecture.md`
(attention over state channels). This note is about taking the part of
Symbolic Neural ODEs that is actually good — the forward pass *is* the
expression — and replacing the part that is weak.

## Where their architecture is weak

Their wiring is **dense**. Every operational layer reads the entire previous
vector `z ∈ R^{4L(k-1)+n+1}` through a learned dense weight, so nothing in the
architecture says which quantities feed which operation. Sparsity is imposed
afterwards, by an ℓ1 or ℓ1/2 penalty during training plus a post-hoc truncation
sweep over tolerances `[0.001, 0.01, 0.1, 1]` using sympy's `nsimplify`.

Selection is done by **penalty**, not by structure. Their own results show the
cost: they say plainly that the method "doesn't always give models with the
structure of the true model" and fall back on Appendix A to argue the
wrong-structure models are at least dynamically close.

That is exactly the gap attention fills — it is a selection mechanism, and they
use none of it.

## The machine

An append-only **memory** of computed quantities, starting as the state
channels. Each step appends one new quantity whose operands come from attention
over everything computed so far:

```
lin(M) = Σ a_i M_i                            one head
pow(M) = sgn(u)|u|^p,  u = Σ a_i M_i          one head + a free exponent
mul(M) = (Σ a_i M_i)(Σ b_j M_j)               two heads
```

with `a_i = proj(logits)_i * tanh(s_i)` — concentration from the projection,
sign from a gate. The sign gate is not optional: softmax weights are
non-negative and `cubic_coupled`'s truth needs `x - y`.

The readout is linear over the whole memory and is solved by **least squares at
every evaluation** — VARPRO, exactly as the existing fitter does. So gradient
descent only ever touches the wiring; it never sees a coefficient.

Three properties fall out of the memory being append-only and attendable:

- **It is a DAG, not a tree.** A shared subexpression is computed once and
  referenced by later steps. `cubic_coupled` wants `(x - y)` in more than one
  place; a tree grammar rediscovers it each time.
- **Nothing is indexed by DOF**, so it is permutation-equivariant over DOFs by
  construction (the property `notes_architecture.md` measures the current
  policy lacking by 129% of its logit scale).
- **The attention map is the wiring diagram.** No truncation sweep needed to
  read the model out.

## The result that matters: sparsemax closes the discretisation gap

First run used softmax with a temperature annealed 2.0 → 0.05. Classic
supernet failure:

```
softmax + annealing
  best soft residual during training   1.381e-09
  residual as trained                  1.104e-02
  discovered:  (y:+0.580  x:-0.013)^2.475
  physical direction:  x:-0.0222  y:+1.0000
```

Seven orders of magnitude paid at discretisation, and the recovered structure
is wrong — exponent 2.475 instead of 3, and the relative coordinate collapsed
to plain `y`.

The cause is specific. A **direction** like `x - y` needs *two* surviving
operands. Annealing softmax toward one-hot destroys precisely the thing we are
trying to find. What is needed is exact zeros with multi-element support, which
is what **sparsemax** gives — it projects onto the simplex, so weights hit
exactly zero while the survivors stay free. No annealing, no truncation sweep,
and no gap by construction:

```
sparsemax (no annealing)
  best soft residual during training   1.392e-09
  residual as trained                  1.398e-09
  residual after discretisation        1.398e-09      gap x1.0

  step 0 [lin] M4 = xdot:-0.479  x:+0.125  y:+0.003
  step 1 [pow] M5 = (y:+0.230  x:-0.228)^3.000
  step 2 [mul] M6 = (M5:+0.400 y:-0.257) * (M4:+0.241 y:-0.038)
  readout:  x:-1.073   xdot:-0.118   M4:-0.078   M5:+4.768

  pow direction in PHYSICAL units:  x:-1.0000  xdot:+0.0000
                                    y:+1.0000  ydot:-0.0000
```

Exponent **3.000**. Direction **exactly `-(x - y)`**, with the overall sign
absorbed by the readout's `+4.768` since `(-(x-y))^3 = -(x-y)^3`. Both velocity
channels get **exactly zero**, not a small number to be truncated later. The
readout reproduces `-3x - 0.12xdot - 1.2(x-y)^3` in normalised units.

This is the contribution over their method, and it is one line: **sparsemax
instead of softmax means there is no truncation tolerance to sweep and no gap
to pay.** It is also the answer to the DARTS objection in
`notes_soft_leaves.md` — the discretisation gap is the whole objection, and a
simplex projection removes it rather than managing it.

## Then it stops being a good story

Four schedules x four seeds, clean data. "hits" counts seeds reaching a
residual below 1e-6, i.e. actually finding the structure rather than a
plausible-looking near miss.

```
cubic_coupled    truth  -3x -0.12xdot -1.2(x-y)^3        want (x-y)^3
  [pow, lin]              best 2.31e-02   0/4   (x-0.48,y-1.00,ydot-0.24)^2.668
  [lin, pow]              best 4.37e-09   1/4   (x-1.00,y+1.00)^3.000
  [lin, pow, mul]         best 1.38e-09   3/4   (x-1.00,y+1.00)^3.000
  [lin, lin, pow, mul]    best 1.39e-09   3/4   (x+1.00,y-1.00)^3.000

duffing          truth  -0.3xdot -1x -0.5x^3             want x^3
  [pow, lin]              best 9.01e-11   3/4   (x+1.00)^3.001
  [lin, pow]              best 8.88e-11   4/4   (x+1.00)^3.001
  [lin, pow, mul]         best 5.97e-11   4/4   (x+1.00)^3.000
  [lin, lin, pow, mul]    best 7.45e-11   3/4   (x-1.00)^3.000

vanderpol        truth  0.6(1-x^2)xdot - x               want x^2 * xdot
  [pow, lin]              best 1.15e-02   0/4   (x-1.00,xdot+0.53)^1.764
  [lin, pow]              best 1.15e-02   0/4   (x+1.00,xdot-0.53)^1.764
  [lin, pow, mul]         best 1.14e-02   0/4   (x-1.00,xdot+0.53)^1.783
  [lin, lin, pow, mul]    best 1.13e-02   0/4   (x+1.00,xdot-0.44)^1.797
```

Three things to read out of that table.

**The op schedule matters more than anything else.** `cubic_coupled` goes
0/4 → 1/4 → 3/4 purely on the order and count of operations, with the *same*
data and the same objective. The schedule is hand-chosen in this prototype.
Making the op choice itself a sparsemax over `{lin, pow, mul}` is the obvious
next step and is untested, so every failure below is confounded with it.

**Extra capacity does not hurt.** Going from 2 to 4 operations does not degrade
the recovered structure, because the VARPRO readout simply gives the useless
ones near-zero coefficients. That is a real advantage of solving the readout in
closed form rather than learning it.

**vanderpol fails completely — 0/4 on every schedule, 16 runs.** It always
lands on `(x + 0.53 xdot)^1.78`, residual 1.14e-2, and will not move. This is
the sharpest result in the note and it is negative: the machine reliably finds
structures of the form `c (w·x)^p` and reliably fails to find a genuine cross
term `x^2 * xdot`. The `pow` op gets a decent fit on a mixed direction early,
and there is no downhill path from there to the product form — **the relaxation
created a smooth path to a wrong answer.**

That is worth sitting with, because it is the exact case `notes_soft_leaves.md`
already flagged for a different reason: vanderpol is the system cited there as
proof that a block must be chosen per leaf rather than per term. It is turning
out to be the discriminating example for more than one design decision.

The remaining failure mode is ordinary **local minima** — duffing misses on 1
of 4 seeds at two of the four schedules. SymNN has this too and runs K-fold
cross-validation precisely because of "the multiple local minima in parameter
space of the optimization landscape." Attention does not fix it; it inherits
it. Multi-start is cheap here (each fit is seconds) and has to be in the
protocol rather than an afterthought.

## Noise breaks the free exponent

At 5% NSR the continuous exponent runs away:

```
duffing        (xdot)^13.446   (x)^7.167   (x)^10.189   (xdot)^10.798
cubic_coupled  (xdot,ydot)^14.702          (x,xdot,y)^1.136
```

This is the same degeneracy `notes_symbolic_node.md` measures on the exponent
sweep — under noise the residual falls monotonically with exponent, because
higher powers absorb more. There the exponent was capped at 6 and it pinned to
6; here it is unbounded and it goes to 14.7.

Worth recording as a correction to `notes_soft_leaves.md`, which lists
re-enabling continuous exponents as a defensible improvement and treats the
integer cap at ±3 purely as a limitation ("real structure at p = 2.5, 3.5 that
the grid cannot reach"). **The cap is also a defence.** Any move to continuous
exponents needs a bound or a prior, not just the machinery switched back on.

## What is genuinely new here

Relative to SymNN: selection by attention rather than by penalty, exact zeros
rather than a truncation sweep, no discretisation gap, DAG reuse, and
DOF-equivariance. Relative to DARTS and the supernet literature: the simplex
projection removes the failure mode instead of mitigating it.

Relative to *us*: this is a different algorithm, not a modification. It
replaces token-sequence generation and J-GRPO with gradient descent on a
wiring. That is a serious thing to propose and the measurements above do not
yet justify it — 0/4 on vanderpol and a hand-chosen schedule are not a case for
replacing a working search.

## Honest ordering

1. The **sparsemax finding is portable on its own**. Wherever
   `notes_architecture.md` proposes attention over state channels, it should be
   sparsemax rather than softmax, for exactly the reason measured here. That
   applies whether or not the register machine is ever built.
2. Make the op choice a sparsemax too, and re-run the schedule table. The
   schedule is currently the single biggest lever (0/4 to 3/4 on
   `cubic_coupled`) and it is hand-set, so every other conclusion is
   confounded with it.
3. Multi-start is not optional given 3/4 and 0/4. Cheap here — each fit is
   seconds — but it belongs in the protocol.
4. Work out why vanderpol is unreachable. Until a genuine cross term can be
   found, this covers `c (w·x)^p` and nothing else — which is 92.4% of terms
   by the soft-leaf note's count, but the missing 7.6% includes the case that
   note treats as decisive.
5. Only then is there a case for comparing it against the existing search.

## Open questions

- Does the memory's DAG reuse actually pay? `cubic_coupled` is the one system
  where it should, and the recovered machine did reference `M4` and `M5` in
  later steps, but nothing isolates that benefit from the rest.
- The junk problem persists: `M4` is a meaningless linear op that still draws
  a readout coefficient of `-0.078`. Same flatness as everywhere else in this
  project; needs the same complexity penalty.
- Everything here is one trajectory of one record, fitted with the windowed
  work-energy objective from `notes_symbolic_node.md`. Multi-trajectory and
  multi-DOF are untested.
- Sparsemax is not the only choice — α-entmax interpolates between softmax
  (α=1) and sparsemax (α=2) and would make the sparsity itself a knob.

## Post-mortem on van der Pol: the relaxation is the problem

0/16 was not an op-set problem. Two further attempts and a diagnostic settled
it.

**Attempt 2 — give `mul` its exponents.** The soft-leaf note's canonical form is
`c · Π_k (w_k·x)^{p_k}`, so `mul` became `spow(u,p1) * spow(v,p2)`, making
`x^2 * xdot` reachable in one op with no dead intermediate. It **regressed
everything**:

```
                        old [lin,pow,mul]      new [lin,ppow]      [lin,ppow,ppow]
vanderpol               1.14e-02  0/4          -                   2.15e-05  0/4
duffing                 5.97e-11  4/4          3.01e-05  0/4       1.00e-08  2/4
cubic_coupled           1.38e-09  3/4          3.50e-02  0/4       crashed
```

`u^{p1} · v^{p2}` is **unidentifiable** when the two heads select overlapping
operands: `p1` and `p2` are then pinned only through their sum, so the
optimiser wanders along a flat direction. It reported exponents of 4.22 and
3.32 for a plain cubic. A product of powers needs one exponent *per slot*, not
one per head.

**The diagnostic that ended it.** Before trying a third op, check whether the
objective can even see the answer. Windowed work-energy residual on fixed
design matrices:

```
TRUTH   [xdot, x^2*xdot, x]                    1.890e-09
decoy   [xdot, (x+0.53xdot)^1.78, x]           1.677e-02
decoy   [xdot, (x+0.53xdot)^2, x]              1.711e-02
linear  [xdot, x]                              3.666e-01
```

The objective separates the truth by **seven orders of magnitude**. It is not
a scoring problem. And the decoy family has a genuine basin:

```
residual over (x + c*xdot)^p, with x and xdot also in the readout
      c       p=1.5       p=2.0       p=2.5       p=3.0
   0.00   3.618e-01   3.610e-01   3.602e-01   3.595e-01
   0.20   1.278e-01   1.156e-01   1.059e-01   9.832e-02
   0.53   1.708e-02   1.711e-02   1.919e-02   2.247e-02
   0.80   6.125e-02   6.234e-02   6.531e-02   6.928e-02
```

`c = 0.53` is a local minimum in `c`, and at the bottom of it the residual is
**flat in the exponent** (1.708e-02 → 1.711e-02 → 1.919e-02). That is exactly
where all 16 runs parked, and why they stopped moving.

**The discrete search does not have this problem.** Handed the same system, the
existing grammar and fitter:

```
xdot + x + x*x*xdot      reward 0.999940    residual 6.04e-05
xdot + x + x^3           reward 0.526705    residual 8.99e-01
xdot + x + x*xdot        reward 0.513295    residual 9.48e-01
xdot + x                 reward 0.513300    residual 9.48e-01
```

`0.506*xdot - 0.847*x - 0.976*x*x*xdot`, found by *sampling* the structure and
fitting, at 280x lower residual than the relaxation's local minimum.

### The conclusion, which cuts against this whole line of work

**Smoothing the search space created a local minimum that the discrete search
does not have.** The relaxation offers a continuous downhill path to a wrong
answer, and once in that basin the gradient with respect to the exponent
vanishes. A sampler has no such trap — it does not have to walk there, it
proposes the structure and scores it.

This is the honest answer to the "round the jaggedness" thread that started
`notes_soft_leaves.md`. Jaggedness is not purely a cost. The ability to *jump*
is what gets past a basin that gradient descent falls into, and the two systems
where the relaxation succeeded (duffing, cubic_coupled) are precisely the ones
whose target is `c(w·x)^p` — a form the relaxation can reach by descending.
`vanderpol`'s cross term is not, on this landscape, reachable that way.

What survives: **sparsemax removing the discretisation gap** is still real and
still portable to the state-channel attention in `notes_architecture.md`. That
finding is about the *discretisation* step and is independent of everything
above. What does not survive is the case for replacing the discrete search
with a relaxation.
