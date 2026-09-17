# Architecture: what a creative transformer would look like here

Status: **one module prototyped, nothing wired in.** Companion to
`notes_soft_leaves.md` (what a leaf should be) and `notes_symbolic_node.md`
(what the objective should be). This note is about the network.

## What was actually creative about SymNN

Not the primitives. The move is that **the network's forward pass is the
expression** — there is no model class and model, there is one object whose
weights are the coefficients and whose wiring is the structure. That
identification is what earns them an architecture paper rather than a loss
paper.

We currently do the opposite, and hard: the transformer is a language model
over token strings, the expression is a separate symbolic object, and the only
thing connecting them is a scalar reward. Concretely, what our policy sees:

```
forward(x, dof_order, n_slots)
  x          token indices
  dof_order  which DOF sits in which slot
```

That is the whole input. **The policy never sees the data.** A record of four
2000-point trajectories is compressed to one scalar per DOF per episode, and
that scalar is the only channel through which anything about the system reaches
the network. Against a search space of roughly 2^51 bags at 2 DOF (≤4 terms of
≤4 tokens over 9 symbols), that is the binding constraint on the whole method —
not the policy's capacity, not the grammar.

The interesting question is not "how do we make the transformer evaluate the
expression" — that road ends in their supernet, which `notes_soft_leaves.md`
argues against. It is: **attention is already a soft-selection mechanism, and
we are using a transformer as a sequence prior while throwing away the one
operation it is actually good at.**

## Three properties the current policy lacks, measured

```
1. DOF-equivariance.  Relabel DOF 0 <-> DOF 1 (dof_order band AND the
   variable tokens x1<->x3, x2<->x4) and the logits should permute.
      max |logit| in play                    3.7927
      slice0(a) vs slice1(b) under the swap  4.8917
      -> break of 129% of the logit scale.  No symmetry at all.

2. Vocabulary grows with the system.
      N_DOF   tokens   of which variables
          1        7         2   (29%)
          2        9         4   (44%)
          3       11         6   (55%)
          5       15        10   (67%)
   At 5 DOF two thirds of the vocabulary is variables, and every one of them
   has to be learned from scratch as an unrelated symbol.

3. The term bag is not permutation-equivariant (measured in the soft-leaf
   note: 2.268e-01 logit delta when a term moves slot). And the conditioning
   it buys is not working — 73.9% of sampled bags contain a duplicate term.
```

All three are the same root cause: **identity is in the vocabulary and in
learned index embeddings, where it should be in the attention.**

## A. Leaves as attention heads over the state channels (prototyped)

The state channels `{q1, q1dot, q2, q2dot, ...}` stop being vocabulary entries
and become a **set of key/value tokens**. A leaf token emits a query; the
attention distribution over channels *is* the leaf's direction `w`, and the
leaf evaluates to `w . x`.

The keys must not be learned per index, or identity comes straight back in.
They are built from role and data:

```
key_c = role_emb[disp | vel]  +  W_feat @ phi_c
```

where `phi_c` are cheap statistics of channel `c` against the target DOF's
measured acceleration. On `cubic_coupled`
(`xddot = -3x - 0.12xdot - 1.2(x-y)^3`):

```
channel    corr(u,a)  corr(u^3,a)  corr(|u|u,a)  log scale
x            -0.967      -0.770       -0.883       -0.443
xdot         -0.019      -0.005       -0.011       -0.174
y            -0.063      +0.113       +0.043       -0.439
ydot         -0.001      +0.086       +0.048       -0.238
```

What this buys, all four measured on the prototype:

**Exact DOF-equivariance.** `max |w(perm(phi)) - perm(w(phi))| = 5.96e-08`.
Permuting the DOFs permutes the attention and nothing else. Nothing in the
module is indexed by DOF number — `role_emb` is 2 x d_head at every system
size.

**Fixed vocabulary.** 7 tokens forever, the soft-leaf note's table, now with a
mechanism behind it. Adding a DOF adds two keys to a set, not two symbols to a
language.

**The policy finally sees the data.** With *untrained* weights the attention
already concentrates on `x` and `y` rather than the velocities, purely because
their `phi` vectors have large norm and the velocity channels' are near zero.
The signal is sitting in the keys before any learning happens. This is the
cheapest possible form of data conditioning and it is not pretraining —
`phi` is computed from the record in front of us.

**Temperature is the smoothing knob** the earlier thread was looking for, and
it falls out rather than being bolted on:

```
   tau         x      xdot         y      ydot   entropy
  0.05    0.9586    0.0000    0.0414    0.0000    0.172     <- a variable token
  0.20    0.6830    0.0030    0.3113    0.0027    0.657
  0.50    0.5117    0.0585    0.3737    0.0561    1.038
  1.00    0.3962    0.1340    0.3386    0.1312    1.269
  3.00    0.2996    0.2087    0.2843    0.2073    1.372
 10.00    0.2648    0.2376    0.2606    0.2371    1.385     <- uniform, ln 4 = 1.386
```

tau -> 0 *is* today's discrete variable choice. Large tau is a soft leaf. The
search space melts and refreezes on one scalar, and the whole "smooth the
jaggedness" thread becomes an annealing schedule.

**The dleaf / vleaf split is a mask on one module**, not two separate tokens:

```
dleaf  x=0.539  xdot=0.000  y=0.461  ydot=0.000
vleaf  x=0.000  xdot=0.505  y=0.000  ydot=0.495
```

### The one real problem, and two fixes

Softmax weights are non-negative. `cubic_coupled`'s truth needs `x - y`, so
plain attention **cannot represent the answer**. Both fixes were measured:

```
(i)  signed values, w_c = softmax(a)_c * tanh(s_c), normalised
     x=+1.0000  xdot=+0.0000  y=-1.0000  ydot=+0.0000     exact

(ii) attention selects the support, VARPRO fits the signed direction
     support {x, y}      physical  x=-1.0000  xdot=+0.0000  y=+0.9897  ydot=+0.0000
     support all four    physical  x=-1.0000  xdot=-0.0577  y=+0.9966  ydot=+0.0071
```

(ii) is the better split and is what the soft-leaf note already argues for:
attention decides *which channels participate*, `least_squares` decides the
signed weights. The overall sign is free — `(-(x-y))^3 = -(x-y)^3` is absorbed
by the term's fitted amplitude.

### It does not disturb J-GRPO

Sampling stays categorical, so the update is unchanged. Today the variable
choice comes from `out_proj` over the vocabulary; here it comes from attention
over channels. The factorisation is still autoregressive:

```
log p = log p(leaf token)  +  log p(channel | leaf)
```

Hard attention during the search is exactly today's behaviour, only
DOF-equivariant and data-conditioned. Soft attention is for the refinement
pass. Annealing tau moves between them.

## B. Residual cross-attention: make the bag a conversation

The measured defect: 73.9% of bags contain a duplicate term (72.4% with the
term index hidden). The causal chain over terms exists so that term j+1 knows
about terms 1..j, and it is not working — three quarters of bags repeat
themselves.

The reason is that term j+1 is conditioned on the *tokens* of the earlier
terms, which is the least useful thing to know about them. What it needs is
what they **failed to explain**. So: fit terms 1..j, compute the energy
residual, encode it as a set of tokens, and let term j+1's decoding
cross-attend to it.

This is the same move as A — put the data in the attention — applied one level
up, and it is the change with the largest information-theoretic argument
behind it: the data-to-policy bandwidth goes from one scalar per episode to an
attended representation per term.

Cost is the obstacle. A residual per term per sample means a fit per term per
sample. The closed-form linear path (`_fit_consts_linear`, 59% of candidates
today) is the way in: a partial bag's linear fit is one lstsq, ~1 ms. Terms
beyond the closed-form domain would need a proxy.

## C. Term slots as learned queries, decoded in parallel

The bag under an implicit `add` root is a **set**. The architecture should be
permutation-equivariant; measurement says it isn't, and the current mitigation
(a flag that drops the term-index band) removes the index while leaving the
causal ordering, which is why it barely moves the duplicate rate.

The honest fix is the DETR shape: K learned term queries, bidirectional
attention among them, all terms decoded in parallel, and an explicit "empty"
class per query replacing the STOP action. Permutation-equivariance by
construction, and the duplicate problem becomes a set-prediction problem with
known remedies.

This is the least novel of the three — DETR is 2020 and set prediction is well
trodden — but it is the right structure for what we are generating, and it
composes with A and B rather than competing.

## What I would not do

**Make the forward pass evaluate the expression.** That is the literal SymNN
analogue: every primitive always present, sparsity by penalty and a
post-training truncation sweep. It is the supernet, with the discretisation gap
and the collapse-toward-parameter-free-ops failure mode, and their own paper
reports the symptom — models that fit but do not have the true structure,
defended by an appendix arguing the wrong-structure models are at least
dynamically close. Our discrete search either finds the structure or does not.

**A hypernetwork that emits coefficients.** Replaces `least_squares` with
something worse and drags in pretraining, which is ruled out.

**Tree-structured attention biases** (parent/child/sibling as attention bias).
Defensible, cheap, and genuinely incremental — it is a better positional
encoding, not a different architecture.

## Order to do them in

A first. It is self-contained, it is the one with a working prototype, it
subsumes the soft-leaf proposal, it turns the smoothing thread into a
temperature schedule, and it leaves J-GRPO alone. C second, because it is
mechanical once A has removed the variable tokens. B last and only if A and C
land, because it is the one that changes the cost model.

## Open questions

- `phi_c` is computed once per DOF against the *measured* acceleration. Under
  the bag decomposition it should arguably be recomputed against the running
  residual, which makes A and B the same mechanism at different granularity.
  Worth checking whether the static version is enough.
- Equivariance is exact in the leaf module. It is **not** exact in the policy
  as a whole until `dof_emb` goes too, and `dof_emb` is currently load-bearing
  (it says which DOF a slice *is*, which matters because slot order permutes).
  Replacing it means a slice's identity has to come from its own channel
  statistics instead. Untested, and the likeliest place for this to fall over.
- The temperature sweep is on one random query of an untrained module. That it
  interpolates is arithmetic; that annealing it *helps the search* is not
  established, and the static-landscape caveat from the soft-leaf note applies
  with full force.
- Attention over channels gives one direction per leaf. `vanderpol` needs
  `x**2 * xdot`, which is two leaves in one term with different supports —
  fine, but it means the per-leaf query has to vary within a term, so the query
  comes from the leaf's position in the term, not from the term.
