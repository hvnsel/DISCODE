"""
discover_edit_train.py
======================

PROTOTYPE ADD-ON.  PPO for the move-based search.  ``DISCOVER_EDIT_TRAIN`` is
the edit agent's counterpart of :func:`discover_train.DISCOVER_TRAIN`; it takes
the same ``SystemData`` and leaves the existing trainer untouched.

One iteration
-------------
1. Rollout.  ``n_envs`` episodes in lockstep, DOFs dealt round-robin, most
   starting from the empty equation and ``restart_frac`` of them from one of
   the DOF's best equations so far.  At every step the policy picks a legal
   move for every live episode, the new equations are fitted and scored in one
   batch (cache first, then the worker pool), and each move is paid the change
   in score (:mod:`discover_edit_env`).  An episode ends on STOP or after
   ``max_steps`` moves.
2. Advantages.  GAE with gamma = 1: the rewards telescope to the final score,
   so undiscounted IS the objective, not an approximation to it.  The step
   budget is part of the observation, so running out of steps is a true
   terminal and nothing is bootstrapped past it.
3. PPO update.  Clipped surrogate, value loss and entropy bonus over
   ``ppo_epochs`` passes of shuffled minibatches -- plus, on every minibatch,
   a self-imitation loss on the best episodes seen so far
   (:class:`EliteBuffer`).  PPO alone climbs the average return and let the
   one episode that found van der Pol's structure be outvoted by 63 that did
   not; self-imitation is what makes a path found once get learned.

Every equation any episode reaches goes into a per-DOF hall of fame, so the
answer is the best equation seen during training, not only where the final
policy happens to stop.  The greedy policy's own answer is reported alongside.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import discover_core as dc
import discover_edit_env as ee
from discover_data import generate_dataset
from discover_edit_policy import EditPolicy, masked_dist


N_WORKERS = max(1, min(os.cpu_count() or 4, 3))
_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def make_pool(initargs, n_workers=N_WORKERS):
    """A scoring pool of SINGLE-THREADED workers; returns ``(pool, saved_env)``
    -- hand both to :func:`close_pool`.

    Measured on 90 duffing bags: three workers left at OpenBLAS's default of
    one thread per core took 34.5 s, three times SLOWER than fitting the same
    bags serially (11.0 s), because 3 x 4 BLAS threads fight over 4 cores on
    small least-squares solves.  With one BLAS thread each the pool takes
    4.4 s.  The thread count is read when numpy loads, so the workers are
    SPAWNED -- fresh interpreters that see the setting -- rather than forked,
    which would inherit this process's already-initialised thread pool.  And
    the setting must hold for the pool's whole life, not just its creation:
    a spawn pool starts workers on demand, and one started after the variables
    were restored came up with 7 threads.
    """
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: '1' for k in _THREAD_VARS})
    pool = ProcessPoolExecutor(max_workers=n_workers,
                               mp_context=mp.get_context('spawn'),
                               initializer=dc.init_energy_worker,
                               initargs=initargs)
    return pool, saved


def close_pool(pool, saved):
    """Shut the pool down, then restore the thread variables it needed."""
    pool.shutdown(wait=True, cancel_futures=True)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class HallOfFame:
    """The best distinct equations seen for one DOF, by score."""

    def __init__(self, size=10):
        self.size = int(size)
        self.best = {}                          # canonical bag -> score

    def add(self, bag, score):
        key = ee.canonical(bag)
        if not key:
            return
        if score > self.best.get(key, -np.inf):
            self.best[key] = float(score)
            if len(self.best) > 8 * self.size:
                self.best = dict(self.top(self.size))

    def top(self, k=None):
        items = sorted(self.best.items(), key=lambda kv: -kv[1])
        return items[:k if k is not None else self.size]


class EliteBuffer:
    """Self-imitation (Oh et al., 2018): the best episodes seen per DOF, ranked
    by the score of the equation they ended on, replayed with

        loss = -log pi(a|s) * max(R - V(s), 0)  +  c_v * max(R - V(s), 0)^2 / 2

    where ``R`` is the move's return-to-go.  Only moves whose real return beat
    the value estimate push, so a path found once is learned, and stops being
    pushed once the value function has caught up with it.

    Why PPO needs it here.  PPO climbs the AVERAGE return, and on a deceptive
    landscape one lucky episode in 64 is outvoted.  On van der Pol only the
    complete equation beats the empty one once constants are charged -- ``x``
    alone scores -0.35, ``x + xdot`` -0.84, the empty equation +0.006, the
    truth 7.05 -- so every partial path is a loss.  The first iteration found
    the truth's structure (6.08) and plain PPO still converged on adding and
    deleting one term for all ten moves, ending on ``xddot = 0``.
    """

    def __init__(self, per_dof=16):
        self.per_dof = int(per_dof)
        self.items = {}                 # (dof, final bag) -> (score, transitions)

    def add(self, episodes, finals, dofs, scorer):
        for ep, fin, d in zip(episodes, finals, dofs):
            if not ep or not fin:
                continue
            key = (d, ee.canonical(fin))
            old = self.items.get(key)
            if old is not None and len(old[1]) <= len(ep):
                continue                # same equation: keep the shorter path
            rtg = np.cumsum([s['reward'] for s in ep][::-1])[::-1]
            self.items[key] = (scorer.value(d, fin),
                               [(s['obs'], s['action'], float(g))
                                for s, g in zip(ep, rtg)])
        by_dof = {}
        for k, v in self.items.items():
            by_dof.setdefault(k[0], []).append((k, v))
        self.items = {k: v for lst in by_dof.values()
                      for k, v in sorted(lst, key=lambda kv: -kv[1][0])[:self.per_dof]}

    def sample(self, n, rng):
        """``(obs[5], actions, returns)`` for ``n`` stored moves, or None."""
        trans = [t for _s, ts in self.items.values() for t in ts]
        if not trans:
            return None
        pick = [trans[i] for i in rng.integers(0, len(trans), size=n)]
        obs = [torch.as_tensor(np.stack([p[0][i] for p in pick])) for i in range(5)]
        return (obs, torch.tensor([p[1] for p in pick], dtype=torch.long),
                torch.tensor([p[2] for p in pick], dtype=torch.float32))


# ── rollout ─────────────────────────────────────────────────────────────────
def rollout(policy, spec, scorer, dofs, greedy=False, starts=None):
    """Play one episode per entry of ``dofs`` (the DOF each episode edits),
    each from the empty equation or from its entry in ``starts``.

    Returns ``(episodes, finals, visited)``: per episode the list of its
    transitions ``{'obs', 'action', 'logp', 'value', 'reward'}``, the bag each
    episode ended on, and every ``(dof, bag)`` any move produced.
    """
    n = len(dofs)
    terms = [tuple(s) for s in starts] if starts is not None else [()] * n
    alive = [True] * n
    episodes = [[] for _ in range(n)]
    visited = []
    scorer.score_many([(dofs[b], terms[b]) for b in range(n)])
    for t in range(spec.max_steps):
        idx = [b for b in range(n) if alive[b]]
        if not idx:
            break
        obs = ee.encode(spec, scorer, [(dofs[b], terms[b], t) for b in idx])
        tok, tm, gl, df, am = (torch.as_tensor(x) for x in obs)
        with torch.no_grad():
            logits, values = policy(tok, tm, gl, df)
            dist = masked_dist(logits, am)
            acts = dist.probs.argmax(1) if greedy else dist.sample()
            logps = dist.log_prob(acts)
        acts = acts.tolist()

        new = []
        for k, b in enumerate(idx):
            nt = terms[b] if acts[k] == spec.STOP else spec.apply(terms[b], acts[k])
            if nt is None:                       # the mask should make this impossible
                raise RuntimeError(f"illegal move {spec.actions[acts[k]]} on {terms[b]}")
            new.append(nt)
        scorer.score_many([(dofs[b], new[k]) for k, b in enumerate(idx)])

        for k, b in enumerate(idx):
            stop = acts[k] == spec.STOP
            reward = 0.0 if stop else (scorer.value(dofs[b], new[k])
                                       - scorer.value(dofs[b], terms[b]))
            episodes[b].append({'obs': tuple(x[k] for x in obs),
                                'action': acts[k], 'logp': float(logps[k]),
                                'value': float(values[k]), 'reward': reward})
            if not stop:
                visited.append((dofs[b], new[k]))
            terms[b] = new[k]
            if stop or t == spec.max_steps - 1:
                alive[b] = False
    return episodes, terms, visited


def gae(rewards, values, lam, gamma=1.0):
    """Generalised advantage estimates for ONE episode that ends in a true
    terminal (value 0 after the last step).  Returns ``(adv, returns)``."""
    T = len(rewards)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nxt = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=float)


def make_batch(episodes, lam):
    """Flatten episodes into PPO tensors: ``(obs[5], actions, old_logp, adv,
    returns)``.  ``obs`` follows :func:`discover_edit_env.encode`'s order."""
    cols = [[] for _ in range(5)]
    acts, logps, advs, rets = [], [], [], []
    for ep in episodes:
        if not ep:
            continue
        adv, ret = gae([s['reward'] for s in ep], [s['value'] for s in ep], lam)
        for s, a, r in zip(ep, adv, ret):
            for i in range(5):
                cols[i].append(s['obs'][i])
            acts.append(s['action']); logps.append(s['logp'])
            advs.append(a); rets.append(r)
    obs = [torch.as_tensor(np.stack(c)) for c in cols]
    return (obs, torch.tensor(acts, dtype=torch.long),
            torch.tensor(logps, dtype=torch.float32),
            torch.tensor(advs, dtype=torch.float32),
            torch.tensor(rets, dtype=torch.float32))


def normalise_per_dof(adv, dof):
    """Standardise advantages within each DOF, never across DOFs -- the same
    rule DISCOVER_TRAIN follows.  Once one DOF finds its truth its returns are
    ~6 while an unsolved DOF's are ~0, and a pooled normalisation hands the
    solved DOF the whole gradient: coupled Duffing's DOF 0 stalled on
    ``xddot = -0.04*xdot`` for 80 iterations after DOF 1 converged."""
    adv = adv.clone()
    for d in torch.unique(dof):
        m = dof == d
        if int(m.sum()) > 1:
            adv[m] = (adv[m] - adv[m].mean()) / (adv[m].std() + 1e-8)
    return adv


def ppo_update(policy, opt, batch, epochs=4, minibatch_size=256, clip_eps=0.2,
               vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, elite=None,
               sil_coef=1.0, sil_value_coef=0.01, sil_batch=64, rng=None):
    """Clipped-surrogate PPO on one rollout batch, plus the self-imitation loss
    on ``elite`` (an :class:`EliteBuffer`) when given.  Returns the mean of
    ``(policy_loss, value_loss, entropy, approx_kl, clip_fraction, sil_loss)``."""
    obs, acts, old_logp, adv, ret = batch
    n = len(acts)
    adv = normalise_per_dof(adv, obs[3])
    rng = np.random.default_rng() if rng is None else rng
    stats = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, minibatch_size):
            i = perm[s:s + minibatch_size]
            logits, v = policy(*(o[i] for o in obs[:4]))
            dist = masked_dist(logits, obs[4][i])
            logp = dist.log_prob(acts[i])
            ratio = torch.exp(logp - old_logp[i])
            pg = -torch.min(ratio * adv[i],
                            torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]).mean()
            vl = 0.5 * ((v - ret[i]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * ent

            sil = torch.zeros(())
            sb = elite.sample(sil_batch, rng) if (elite is not None and sil_coef > 0) else None
            if sb is not None:
                s_obs, s_act, s_ret = sb
                s_logits, s_v = policy(*s_obs[:4])
                gap = (s_ret - s_v).clamp(min=0.0)
                s_logp = masked_dist(s_logits, s_obs[4]).log_prob(s_act)
                sil = (-(s_logp * gap.detach()).mean()
                       + 0.5 * sil_value_coef * (gap ** 2).mean())
                loss = loss + sil_coef * sil

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats.append((pg.item(), vl.item(), ent.item(),
                              (old_logp[i] - logp).mean().item(),
                              ((ratio - 1).abs() > clip_eps).float().mean().item(),
                              float(sil)))
    return np.mean(stats, axis=0)


# ── reporting ───────────────────────────────────────────────────────────────
def describe(scorer, spec, dof, bag):
    """``(score, r, normalised expr, physical expr)`` for a scored bag."""
    s = scorer.get(dof, bag)
    r = 1.0 / (1.0 + s.E) if np.isfinite(s.E) else 0.0
    return (spec.score(s), r, dc.expr_to_str(s.tau, s.consts),
            dc.denormalize_expr(s.tau, s.consts, dof))


def _print_truth(system, spec, scorer):
    taus = [t for t in (getattr(system, 'truth_taus', None) or []) if t is not None]
    if not taus:
        return
    print(f"[truth] known structure under the edit objective "
          f"(score = -log E - {spec.lam:g} * n_consts):")
    for d, tau in enumerate(system.truth_taus):
        if tau is None:
            continue
        bag = ee.bag_from_tau(tau)
        s = scorer.get(d, bag)
        note = '' if spec.reachable(bag) else '   [NOT reachable by the moves]'
        print(f"    DOF {d}: score {spec.score(s):7.3f}   -log E {s.neg_log_E:6.3f}"
              f"   {s.n_consts} consts{note}")
    print()


# ── entry point ─────────────────────────────────────────────────────────────
def DISCOVER_EDIT_TRAIN(
    system_data,
    n_iters            = 150,
    n_envs             = 64,      # episodes per iteration, DOFs dealt round-robin
    max_steps          = 10,      # moves per episode
    max_terms          = 6,       # terms per equation
    max_term_len       = 8,       # tokens per term
    lam                = 0.5,     # score cost per fitted constant
    closed_form        = True,    # False -> WRAP_ALL and powers of any subtree
    residual_features  = True,    # show the policy what the equation misses
    # PPO
    lr                 = 3e-4,
    ppo_epochs         = 4,
    minibatch_size     = 256,
    clip_eps           = 0.2,
    vf_coef            = 0.5,
    ent_coef           = 0.01,
    gae_lambda         = 0.95,
    max_grad_norm      = 0.5,
    # self-imitation (EliteBuffer); sil_coef = 0 is plain PPO
    sil_coef           = 1.0,
    sil_value_coef     = 0.01,
    sil_batch          = 64,
    sil_episodes       = 16,      # best episodes kept per DOF
    # restarts: this fraction of episodes starts from a hall-of-fame equation
    restart_frac       = 0.25,
    restart_top        = 8,       # ...drawn from the DOF's best this many
    # policy
    d_model            = 64,
    n_heads            = 4,
    n_layers           = 2,
    # reward / data -- same meaning as in DISCOVER_TRAIN
    energy_normalize   = True,
    max_traj           = None,
    w_acc              = 0.5,
    reward             = 'energy',  # 'energy' | 'simulation'
    sim_window         = None,      # simulation: None = one free run per trial
    sim_w_vel          = 0.0,       # simulation: velocity weight in the NRMSE
    center_features    = False,
    directional_leaves = False,
    transcendental     = False,
    use_pool           = True,
    seed               = 0,
    log_every          = 5,
    hof_size           = 10,
):
    """Train the edit agent on ``system_data`` and return a dict with, per DOF,
    the best equation seen (``best``), the greedy policy's equation
    (``greedy``), plus ``history``, ``policy``, ``spec`` and ``scorer``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    system = system_data
    N = system.n_dof

    dc.configure_grammar(N, system.var_names, directional_leaves, transcendental)
    _X, _y, raw_trajs, norm_stats = generate_dataset(system, None,
                                                     center=center_features)
    dc.set_problem_data(norm_stats, raw_trajs, energy_normalize, max_traj, w_acc)
    dc.set_reward(reward, sim_window, sim_w_vel)
    spec = ee.EditSpec(max_terms, max_term_len, max_steps, lam, closed_form)

    print(f"[edit] System='{system.name}'  N_DOF={N}  tokens={dc.ALL_TOKENS}")
    print(f"[edit] {spec.n_actions} moves ({spec.n_global} add-type, "
          f"{spec.per_term} per term x {spec.max_terms} slots)  "
          f"max_steps={spec.max_steps}  lam={spec.lam:g}  "
          f"grammar={'closed-form' if closed_form else 'open'}  "
          f"residual_features={residual_features}")
    print(f"[edit] reward: {dc.describe_reward()}; "
          f"{dc.MAX_TRAJ} of {len(raw_trajs)} trajectories\n")

    pool = saved_env = None
    if use_pool:
        pool, saved_env = make_pool((N, system.var_names, norm_stats, raw_trajs,
                                     energy_normalize, max_traj, w_acc,
                                     directional_leaves, transcendental,
                                     reward, sim_window, sim_w_vel))
    try:
        scorer = ee.Scorer(spec, pool, residual_features)
        _print_truth(system, spec, scorer)

        policy = EditPolicy.for_spec(spec, dc.N_TOKENS,
                                     ee.n_global_features(residual_features),
                                     d_model=d_model, n_heads=n_heads,
                                     n_layers=n_layers)
        policy.train()
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        print(f"[edit] policy: {sum(p.numel() for p in policy.parameters()):,} "
              f"parameters\n")

        hofs = [HallOfFame(hof_size) for _ in range(N)]
        elite = EliteBuffer(sil_episodes)
        rng = np.random.default_rng(seed)
        dofs = [b % N for b in range(n_envs)]
        history = []
        for it in range(n_iters):
            t0 = time.time()
            fits_before = scorer.n_fits
            # Restarts.  An equation already found stays in the hall of fame,
            # but nothing edits it again unless an episode starts there:
            # coupled Duffing's DOF 0 kept x + xdot + y plus junk as its best
            # for 80 iterations without ever cleaning it up.  Starting a share
            # of episodes from the best few equations refines them directly.
            starts = []
            for d in dofs:
                top = hofs[d].top(restart_top) if hofs[d].best else []
                starts.append(top[rng.integers(len(top))][0]
                              if top and rng.random() < restart_frac else ())
            episodes, finals, visited = rollout(policy, spec, scorer, dofs,
                                                starts=starts)
            for d, bag in visited:
                hofs[d].add(bag, scorer.value(d, bag))
            elite.add(episodes, finals, dofs, scorer)
            batch = make_batch(episodes, gae_lambda)
            pg, vl, ent, kl, cf, sil = ppo_update(
                policy, opt, batch, ppo_epochs, minibatch_size, clip_eps,
                vf_coef, ent_coef, max_grad_norm, elite, sil_coef,
                sil_value_coef, sil_batch, rng)

            ret = float(np.mean([sum(s['reward'] for s in ep) for ep in episodes]))
            # the policy's own level: episodes that started from scratch
            final = []
            for d in range(N):
                v = [scorer.value(d, f) for f, dd, s0 in zip(finals, dofs, starts)
                     if dd == d and not s0]
                final.append(float(np.mean(v)) if v else float('nan'))
            best = [hofs[d].top(1)[0][1] if hofs[d].best else -np.inf
                    for d in range(N)]
            steps = float(np.mean([len(ep) for ep in episodes]))
            history.append({'iter': it, 'return': ret, 'final': final,
                            'best': best, 'steps': steps, 'entropy': ent,
                            'kl': kl, 'clip_frac': cf, 'value_loss': vl,
                            'sil_loss': sil,
                            'fits': scorer.n_fits - fits_before,
                            'time': time.time() - t0})
            if it % log_every == 0 or it == n_iters - 1:
                fin = ' '.join(f"{x:6.2f}" for x in final)
                bst = ' '.join(f"{x:6.2f}" for x in best)
                print(f"it {it:4d} | return {ret:6.2f} | final {fin} | best {bst} "
                      f"| steps {steps:4.1f} | ent {ent:5.2f} | kl {kl:+.4f} "
                      f"| sil {sil:6.3f} "
                      f"| fits {history[-1]['fits']:4d} | "
                      f"{history[-1]['time']:5.1f}s", flush=True)
                for d in range(N):
                    if hofs[d].best:
                        print(f"          DOF {d} best: "
                              f"{describe(scorer, spec, d, hofs[d].top(1)[0][0])[2]}")

        # the greedy policy's own answer, one episode per DOF
        _eps, g_finals, _v = rollout(policy, spec, scorer, list(range(N)),
                                     greedy=True)

        print(f"\n{'=' * 64}\n=== DISCOVER-EDIT done :: {system.name} ===")
        best_out, greedy_out = [], []
        truth_strs = getattr(system, 'truth_strs', None) or ['unknown'] * N
        for d in range(N):
            print(f"\nDOF {d}")
            if truth_strs[d] != 'unknown':
                print(f"  Truth          : {truth_strs[d]}")
            if not hofs[d].best:
                print("  nothing found")
                best_out.append(None)
            else:
                bag = hofs[d].top(1)[0][0]
                sc, r, norm, phys = describe(scorer, spec, d, bag)
                best_out.append({'bag': bag, 'score': sc, 'r': r, 'expr': phys,
                                 'tau': scorer.get(d, bag).tau,
                                 'consts': scorer.get(d, bag).consts})
                print(f"  Best seen      : score {sc:.3f}  r {r:.6f}")
                print(f"    normalised   : {norm}")
                print(f"    physical     : {phys}")
            sc, r, _norm, phys = describe(scorer, spec, d, g_finals[d])
            greedy_out.append({'bag': g_finals[d], 'score': sc, 'r': r,
                               'expr': phys})
            print(f"  Greedy policy  : score {sc:.3f}  r {r:.6f}  {phys}")
            print(f"  --- Hall of Fame (top {min(5, len(hofs[d].best))}) ---")
            for rank, (bag, sc) in enumerate(hofs[d].top(5), 1):
                _s, r, _n, phys = describe(scorer, spec, d, bag)
                print(f"    #{rank}: score {sc:7.3f}  r {r:.6f}  {phys}")

        print(f"\n{'=' * 64}\nDISCOVERED_EXPRS = [")
        for d, b in enumerate(best_out):
            print(f'    "{b["expr"]}",' if b else f'    "",   # DOF {d}: nothing found')
        print("]\n", flush=True)
        return {'best': best_out, 'greedy': greedy_out, 'history': history,
                'policy': policy, 'spec': spec, 'scorer': scorer}
    finally:
        if pool is not None:
            close_pool(pool, saved_env)
