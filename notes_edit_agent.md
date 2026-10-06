# Prototype: move-based search trained with PPO

Status: **prototype add-on**, branch `claude/discode-ppo-edit-agent`. It adds
five files and changes none: the grammar, the constant fitter and the reward
all come from `discover_core.py` exactly as they are.

| file | what it is |
|---|---|
| `discover_edit_env.py` | the MDP: states, moves, legality, scoring, what the agent sees |
| `discover_edit_policy.py` | the actor-critic |
| `discover_edit_train.py` | `DISCOVER_EDIT_TRAIN(system_data, ...)` — rollout, GAE, PPO |
| `discover_edit_sim.py` | driver over the existing SDOF/MDOF sim libraries |
| `discover_edit_test.py` | invariants — run before anything long |

```bash
python discover_edit_test.py      # 37 checks
python discover_edit_sim.py       # duffing, prints the same DISCOVERED_EXPRS block
```

## The idea

The term-bag policy writes each equation from scratch and learns from one
number at the end. Here an agent holds one DOF's current equation and edits it
a move at a time, and every move is scored:

| move | effect |
|---|---|
| `ADD(v)` / `ADD(op, v)` | append `v` or `op(v)` as a new term |
| `WRAP(j, op)` | apply `op` to the last factor of term `j` (`x*y → x*y^2`) |
| `MUL(j, v)` | multiply term `j` by `v` (`x^2 → x^2*xdot`) |
| `DELETE(j)` | remove term `j` |
| `WRAP_ALL(op)` | wrap the whole equation, `op(t1 + ... + tn)` — open grammar only |
| `STOP` | end the episode |

"Get rid of the move from n steps ago" is `DELETE(j)`: the state is the
equation, not the move history, which keeps the problem a proper MDP.

Three things it has that the from-scratch policy does not:

- **A reward for every move**, so credit assignment is local.
- **It sees the equation it is editing** — the bag is the observation.
- **It sees what the equation still misses.** `residual_features` is one step
  of matching pursuit: the acceleration residual and a set of candidate shapes
  (`u, u^2, u^3, u_i u_j, u_i^2 u_j` over every channel) are projected off the
  span of the current terms, and each feature is the fraction of the
  acceleration's RMS that shape could still remove. On van der Pol's linear
  model the largest is `x^2*xdot` (0.504), exactly the missing term; on the
  fitted truth every feature is 0.000.

## The objective

    score(s) = -log E(s) - lam * n_consts(s)          r = 1/(1+E)

Each move is paid the change in score. The rewards telescope, so with gamma = 1
an episode's return is its final score minus its starting score, and GAE with
gamma = 1 optimises exactly the score of the equation the agent stops on.

`-log E` rather than `r`, because every useful comparison lives in the last
decimals of `r`. Measured on duffing:

| equation | -log E | constants |
|---|---|---|
| empty (mean acceleration) | -0.85 | 0 |
| `x + xdot` | 1.33 | 2 |
| truth `x + xdot + intpower(x)` | 10.52 | 5 |
| same with `power(x)` | 10.66 | 6 |
| truth + `intpower(xdot)` | 10.69 | 8 |

**The penalty counts fitted constants, not tokens.** `power(x)` and
`intpower(x)` are both two tokens, but `power`'s extra amplitude and continuous
exponent buy 0.14 here, so a token count pays the agent to use the non-standard
power. **`lam = 0.5`**, not 0.25: on a 2-trial, 600-point record `power` buys
0.32 over `intpower`, and at 0.25 the power variant won (-0.07); at 0.5 the
truth wins by 0.18. Junk extensions buy 0.04–0.39 for 3–4 constants while real
terms buy 2–10, so every registry truth still wins by a clear margin.

## PPO alone is not enough: self-imitation

On van der Pol, once constants are charged, **no partial equation beats the
empty one**: `x` alone scores -0.35, `x + xdot` -0.84, the empty equation
+0.006, and only the complete truth (7.05) is better. Every path to the truth
is a string of losses until its last move. The untrained policy found the
truth's structure in its first iteration (6.08), but PPO climbs the *average*
return, that episode was outvoted by 63 that lost, and the policy converged on
adding and deleting one term for all ten moves — `STOP` is illegal on an empty
equation — ending on `xddot = 0`.

The fix is self-imitation learning (Oh et al., 2018), `EliteBuffer` in
`discover_edit_train.py`: keep the best episodes seen per DOF and, on every
PPO minibatch, add `-log pi(a|s) * max(R - V(s), 0)` over their moves, where
`R` is the move's return-to-go. It only pushes where the real return beat the
value estimate, so a path found once gets learned and the push fades once the
value function has caught up (the logged `sil` loss falls from ~10 to ~0.01).
It is the same idea as the term-bag trainer's risk-seeking top-quantile update,
in PPO form. `sil_coef = 0` gives plain PPO back.

## Found while building it

**Products of two `power` factors fall off the fast fit.** `power(x)*power(xdot)`
expands to four monomials over two pairs of amplitudes, which the closed-form
fitter cannot separate; it went to the nonlinear optimiser at **15–27 s per fit**
against 0.25 s for a separable five-term equation. A random walk hits such a
product about once per 25 states, and in lockstep one slow fit stalls every
episode. The closed-form grammar now requires every term to pass the fitter's
own separability test.

**The existing scoring pool is probably running slower than serial.** Three
workers left at OpenBLAS's default of one thread per core fitted 90 bags in
34.5 s — three times slower than fitting them serially (11.0 s) — because
3 × 4 BLAS threads fight over 4 cores on small least-squares solves. With one
BLAS thread per worker the same pool takes 3.6–4.4 s. Both trainers now spawn
single-threaded workers through `discover_core.make_pool`. `DISCOVER_TRAIN`'s
old default pool also ran a Windows machine out of memory ("OpenBLAS error:
Memory allocation still failed after 10 retries").

**Plain correlation is the wrong residual feature.** It is scale-free, so on
van der Pol's fitted truth it reported 0.995 against `x` — a shape the equation
already contains — from a misfit of 1e-4. The projected version fixes both.

## First results

Standard simulated records (4 trials, noise-free), `w_acc = 0.5`, 64 episodes
per iteration, at most 10 moves each, `lam = 0.5`, residual features and
self-imitation on, seed 0, three single-threaded workers on four cores. One
seed each — these show the method works, not how reliably.

| system | truth score | best equation first seen | greedy policy's answer | policy's mean final score | wall-clock |
|---|---|---|---|---|---|
| duffing, 60 it | 8.02 | iteration 0 | the truth | -2.62 → 7.71 (it 30) → 7.99 | 130 s |
| van der Pol, 80 it | 7.05 | iteration 5 (7.55, see below) | the truth | -3.48 → 6.69 (it 20) → ~7.45 | 171 s |
| van der Pol, 80 it, **PPO only** | 7.05 | iteration 0 (6.08) | `xddot = 0` | -3.48 → 0.01, stuck | 99 s |
| coupled Duffing (2 DOF), 100 it | 6.13 / 6.54 | between it 20 and 30, both DOFs | both truths | -4.2 / -6.8 → 6.13 / 6.53 | 162 s |

The greedy answers, as printed:

```
duffing        xddot = -0.5*x**3 - 1.0*x - 0.3*xdot
van der Pol    xddot = -0.6*x**2*xdot - 1.0*x + 0.6*xdot
coupled        xddot = -0.7997*x**3 - 5.501*x - 0.15*xdot + 1.5*y
               yddot = 1.5*x - 0.5006*y**3 - 4.0*y - 0.1*ydot
```

Duffing is too easy to show much — an untrained policy stumbles onto it in the
first iteration — so its signal is the mean final score climbing from -2.62 to
7.99 as episodes shrink from 8.6 moves to 4.0 (the truth takes three plus
STOP). Coupled Duffing is the real test: nothing found for 20 iterations, then
both DOFs' truths within ten more, then the policy converges on them.

Van der Pol's best (7.55) beats its truth (7.05) by writing `x^2*xdot` as
`x*x*xdot`: that is three constant slots against `intpower(x)*xdot`'s four,
because an `intpower` exponent counts as a slot. Both print as `x**2*xdot`.

Once the policy settles, an iteration costs under a second: every equation it
visits is already in the cache.

## Limitations and next steps

- Every episode starts empty and lasts at most `max_steps` moves, so an
  equation that needs more moves than that is unreachable in one episode:
  `cubic_coupled`'s expanded cube needs about twelve. Restarting some episodes
  from hall-of-fame equations would lift that.
- Paths whose first moves only pay later sit in a dip that the penalty deepens.
  Self-imitation gets through van der Pol's; a longer dip needs the path to be
  found by chance first.
- The penalty counts constant SLOTS, not degrees of freedom: a product of
  bare variables carries one slot per factor but one effective amplitude, and
  an `intpower` exponent is a discrete choice charged like a continuous one.
  Counting one per monomial amplitude plus one per continuous exponent would be
  the principled version.
- Tested on noise-free simulations only, one seed each, and not yet on the
  experimental record. Directional leaves, `blend` and the open grammar
  (`WRAP_ALL`) are covered by the tests but not by a training run; a power of a
  directional leaf is nonlinear by nature (~0.5 s per fit), so expect slower
  iterations with `directional_leaves=True`.
- No comparison yet against `DISCOVER_TRAIN` at equal compute. The runs above
  took 2–3 minutes each, but that is not a controlled comparison.
