"""
discode_train.py
================

The DISCODE training loop.  One function, :func:`DISCODE_TRAIN`, which takes a
:class:`discode_data.SystemData` and returns the best expression found for each
DOF.

It knows nothing about MATLAB files, system libraries or ground truth — the
drivers (``discode_sdof_sim``, ``discode_sdof_exp``, ``discode_mdof_sim``,
``discode_mdof_exp``) build the ``SystemData`` and call this.  A single-DOF run
is just ``n_dof == 1``; there is no separate SDOF loop.

Per epoch, for each DOF independently:

  1. The DOF's diffusion policy samples a batch of candidate token sequences
     (plus a beam-search batch every ``beam_interval`` epochs).
  2. Every candidate has its constants fitted to that DOF's own work-energy
     balance and is scored by the reward (see :mod:`discode_core`).  The fit
     and the score are the same closed-form VARPRO path wherever possible.
  3. The top-``alpha`` fraction is promoted into the DOF's replay buffer,
     ranked by a novelty-scaled score but STORED as the raw reward.
  4. The critic and the J-GRPO objective are trained on that buffer.

DOFs do not interact.  ``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an
exact per-DOF kinematic identity — coupling forces are already inside ``a_d`` —
so each balance closes on its own, and mixing a partner's residual into the
score would only make rewards incomparable across epochs.  See the docstring of
:func:`discode_core.energy_worker`.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

import discode_core as dc
from discode_data import generate_dataset

N_WORKERS = max(1, min(os.cpu_count() or 4, 16))
HOF_SIZE = 10


# ── Per-DOF training state ──────────────────────────────────────────────────
class DOFState:
    def __init__(self, max_len, lr, n_epochs, device):
        self.model     = dc.DiffusionModel(n_tokens=dc.N_TOKENS, max_len=max_len).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-5)
        self.model_old = copy.deepcopy(self.model).to(device); self.model_old.eval()
        self.model_ref = copy.deepcopy(self.model).to(device); self.model_ref.eval()
        self.critic     = dc.ExprCritic(vocab_size=dc.VOCAB_SIZE, max_len=max_len, d=64).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # buffer entry: (energy_reward, tau, consts)
        self.buffer     = []
        self.R_alpha    = 0.0
        self.best_r     = -np.inf
        self.best_entry = None
        self.hof        = []


def _train_critic(st, max_len, device):
    if len(st.buffer) < 16:
        return None
    st.critic.train()
    n_cs = min(5, max(1, len(st.buffer) // 32))
    c_loss = None
    for _ in range(n_cs):
        cb = min(64, len(st.buffer))
        ci = np.random.choice(len(st.buffer), cb, replace=False)
        c_taus = []; c_targets = []
        for ix in ci:
            e   = st.buffer[ix]
            tks = ([dc.tok2idx(tk) for tk in e[1]] +
                   [dc.MASK_TOKEN] * (max_len - len(e[1])))
            c_taus.append(tks[:max_len]); c_targets.append(e[0])
        cx = torch.tensor(c_taus,    dtype=torch.long,    device=device)
        cy = torch.tensor(c_targets, dtype=torch.float32, device=device)
        c_loss = F.mse_loss(st.critic(cx), cy)
        st.critic_opt.zero_grad(); c_loss.backward(); st.critic_opt.step()
    st.critic.eval()
    return c_loss.item() if c_loss is not None else None


def _grpo_steps(st, C, max_len, eps, beta, lam_t, device):
    if not st.buffer:
        return
    S_grpo = [(e[0], e[1], e[2]) for e in st.buffer]
    R_grpo = min(r for r, _, _ in S_grpo)
    st.model.train()
    for _j in range(C):
        t_diff  = int(np.random.randint(1, max_len + 1))
        jg, ent = dc.compute_jgrpo(
            st.model, st.model_old, st.model_ref,
            S_grpo, R_grpo, t_diff, max_len, eps, beta,
            critic=st.critic if len(st.buffer) >= 32 else None)
        loss = -(jg + lam_t * ent)
        st.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(st.model.parameters(), 1.0)
        st.optimizer.step()
    st.scheduler.step()


# ── Main loop ───────────────────────────────────────────────────────────────
def DISCODE_TRAIN(
    system_data,               # a discode_data.SystemData
    n_epochs         = 300,
    batch_size       = 200,
    max_len          = 20,
    lr               = 1e-4,
    alpha            = 0.20,
    C                = 5,
    G                = 5,
    lam_start        = 0.02,
    lam_end          = 0.001,
    eps              = 0.2,
    beta             = 0.01,
    max_buffer       = 500,
    beam_interval    = 10,
    beam_width       = 20,
    novelty_weight   = 0.15,
    energy_normalize = True,
    max_traj         = None,   # None = use every trial in the record
    w_acc            = 0.5,    # 0 = pure work-energy
                               # 1 = pure acceleration NRMSE
                               # blend: residual = (1-w)*energy + w*accel_NRMSE
    use_pool         = True,
    center_features  = False,  # see generate_dataset(): scale-only by default
):
    """Search for one acceleration expression per DOF of ``system_data``.

    Returns a list of ``(reward, tau, consts)`` — the best entry per DOF, or
    ``None`` for a DOF where nothing survived.
    """
    device = dc.DEVICE
    system = system_data
    N = system.n_dof

    dc.configure_grammar(N, system.var_names)
    print(f"[vocab]  {N}-DOF tokens: {dc.ALL_TOKENS}")
    print(f"[config] System='{system.name}'  N_DOF={N}  reward=work-energy")
    print(f"[config] {len(system.time)} pts  {system.t_end:.4g} s  "
          f"{(len(system.time)-1)/system.t_end:.0f} Hz eff  "
          f"{system.n_trials} trials")

    X_torch, y_list, raw_trajs, norm_stats = generate_dataset(
        system, device, center=center_features)
    dc.set_problem_data(norm_stats, raw_trajs, energy_normalize, max_traj, w_acc)
    print(f"[config] reward blend: w_acc={dc.W_ACC:.2f} "
          f"({(1-dc.W_ACC):.2f} work-energy + {dc.W_ACC:.2f} accel-NRMSE)")
    print(f"[config] trajectories used per fit/score: {dc.MAX_TRAJ} "
          f"of {len(raw_trajs)} available\n")

    # Energy-scoring pool (static data injected once).
    pool = None
    if use_pool:
        pool = ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=dc.init_energy_worker,
            initargs=(N, system.var_names, norm_stats, raw_trajs,
                      energy_normalize, max_traj, w_acc),
        )
        print(f"Energy-scoring pool : {N_WORKERS} workers\n")

    states = [DOFState(max_len, lr, n_epochs, device) for _ in range(N)]

    for ep in range(n_epochs):
        t0_ep   = time.time()
        horizon = dc.get_traj_horizon(ep, n_epochs, system.t_end)
        lam_t   = lam_start * (lam_end / lam_start) ** (ep / max(n_epochs - 1, 1))

        print(f"\n{'=' * 64}")
        print(f"Ep {ep + 1}/{n_epochs}  |  horizon={horizon:.2f}s  |  lam={lam_t:.5f}")

        # refresh reference / old policies
        for st in states:
            if ep % G == 0:
                st.model_ref = copy.deepcopy(st.model).to(device); st.model_ref.eval()
            st.model_old = copy.deepcopy(st.model).to(device); st.model_old.eval()

        # ── Step 1: Sampling ────────────────────────────────────────────────
        t1 = time.time()
        all_exprs = []
        for i, st in enumerate(states):
            exprs = dc.sample_batch(st.model, batch_size, max_len)
            if ep > 0 and ep % beam_interval == 0:
                exprs.extend(dc.beam_search_expressions(
                    st.model, beam_width, max_len, n_return=beam_width))
            all_exprs.append(exprs)
        print(f"  [1-sample]   {batch_size} exprs/DOF  ({time.time()-t1:.2f}s)")
        sys.stdout.flush()

        # ── Step 2: Energy scoring (each DOF scored in isolation) ───────────
        t1 = time.time()
        tasks = []
        for i in range(N):
            for tau in all_exprs[i]:
                tasks.append((i, tau, None, horizon))
        n_tasks = len(tasks)
        print(f"  [2-energy]   {n_tasks} candidates queued "
              f"({'pool' if pool else 'serial'}) ...", flush=True)

        report_every = max(1, n_tasks // 20)  # ~5% granularity
        t_score0 = time.time()
        results = []
        iterator = (pool.map(dc.energy_worker, tasks) if pool is not None
                    else (dc.energy_worker(tk) for tk in tasks))
        for k, res in enumerate(iterator, start=1):
            results.append(res)
            if k % report_every == 0 or k == n_tasks:
                elapsed = time.time() - t_score0
                rate = k / elapsed if elapsed > 0 else 0.0
                eta = (n_tasks - k) / rate if rate > 0 else 0.0
                print(f"    [2-energy]   {k}/{n_tasks} scored "
                      f"({100.0*k/n_tasks:5.1f}%)  "
                      f"{rate:6.1f} eq/s  ETA {eta:6.1f}s", flush=True)

        per_dof = [[] for _ in range(N)]
        for res in results:
            if res is None:
                continue
            tgt, r, tau, c = res
            per_dof[tgt].append((r, tau, c))
        print(f"  [2-energy]   scored  ({time.time()-t1:.2f}s)", flush=True)

        # ── Step 3: promotion + buffer update + train ───────────────────────
        t1 = time.time()
        for i, st in enumerate(states):
            batch_r = per_dof[i]
            if not batch_r:
                print(f"  DOF {i}: no valid candidate this epoch"); continue

            # Novelty is a SELECTION heuristic, never a reward.  It depends on
            # the buffer contents at scoring time, so baking it into the stored
            # value would make entries from different epochs incomparable.
            # Rank by the scaled score, but gate on and store the raw reward.
            ranked = [(r, r, tau, c) for r, tau, c in batch_r]   # (r_sel, r_raw, ...)
            if st.buffer and novelty_weight > 0:
                ranked = [(r * (1 - novelty_weight
                                + novelty_weight * dc.structural_novelty(tau, st.buffer)),
                           r, tau, c)
                          for _rs, r, tau, c in ranked]
            ranked.sort(key=lambda z: z[0], reverse=True)
            batch_r = [(r_raw, tau, c) for _rs, r_raw, tau, c in ranked]

            k_top = max(1, int(len(batch_r) * alpha))
            if len(st.buffer) < max_buffer:
                promoted = batch_r[:k_top]
            else:
                promoted = [(r, t, c) for r, t, c in batch_r[:k_top] if r > st.R_alpha]
                if not promoted:
                    promoted = batch_r[:1]

            seen = set(tuple(e[1]) for e in st.buffer)
            n_added = 0
            for entry in promoted:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key); st.buffer.append(entry); n_added += 1
                if entry[0] > st.best_r:
                    st.best_r = entry[0]; st.best_entry = entry
                st.hof.append(entry)
                st.hof.sort(key=lambda z: z[0], reverse=True)
                st.hof[:] = st.hof[:HOF_SIZE]

            if not st.buffer:
                print(f"  DOF {i}: buffer empty, skip train"); continue

            st.R_alpha = float(np.percentile([e[0] for e in st.buffer], 10))

            c_loss = _train_critic(st, max_len, device)
            _grpo_steps(st, C, max_len, eps, beta, lam_t, device)

            st.buffer.sort(key=lambda z: z[0], reverse=True)
            if len(st.buffer) >= max_buffer:
                st.buffer = st.buffer[:max_buffer]
                n_rm = max(1, int(max_buffer * alpha))
                st.buffer = st.buffer[:-n_rm]
            if st.buffer:
                st.R_alpha = float(np.percentile([e[0] for e in st.buffer], 5))

            c_str = f'  critic_loss={c_loss:.4f}' if c_loss is not None else ''
            best_raw = max(z[0] for z in batch_r)
            print(f"  DOF {i}: best={best_raw:.4f}  +{n_added} buf={len(st.buffer)}{c_str}")
        print(f"  [3-train]    ({time.time()-t1:.2f}s)", flush=True)

        # ── epoch report ────────────────────────────────────────────────────
        print(f"\n  [EPOCH TOTAL]  {time.time() - t0_ep:.1f}s")
        for i, st in enumerate(states):
            if st.best_entry is None:
                print(f"   DOF {i}: (no energy survivor yet)"); continue
            be = st.best_entry
            print(f"   DOF {i}: best_energy={st.best_r:.4f}  {dc.expr_to_str(be[1], be[2])}")
            d = dc.denormalize_expr(be[1], be[2], i)
            if d:
                print(f"          denorm: {d}")

    # ── Final summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 64}\n=== DISCODE done :: {system.name} ===")
    best_per_dof = []
    for i, st in enumerate(states):
        if not st.buffer:
            print(f"\nDOF {i}: no expression found"); best_per_dof.append(None); continue
        st.buffer.sort(key=lambda z: z[0], reverse=True)
        best = st.buffer[0]
        best_per_dof.append(best)
        print(f"\nDOF {i}")
        print(f"  Expr (normalised) : {dc.expr_to_str(best[1], best[2])}")
        print(f"  energy_reward     : {best[0]:.6f}")
        if system.truth_strs[i] != 'unknown':
            print(f"  Truth             : {system.truth_strs[i]}")
        d = dc.denormalize_expr(best[1], best[2], i)
        if d:
            print(f"  Expr (physical)   : {d}")
        print(f"  --- Hall of Fame (top {len(st.hof)}) ---")
        for rank, hof in enumerate(st.hof, 1):
            dh = dc.denormalize_expr(hof[1], hof[2], i)
            print(f"    #{rank}: energy={hof[0]:.4f}  "
                  f"{dh if dh else dc.expr_to_str(hof[1], hof[2])}")

    # Paste-ready block for discode_score.py / discode_plot_*.py
    print(f"\n{'=' * 64}\nDISCOVERED_EXPRS = [")
    for i, best in enumerate(best_per_dof):
        if best is None:
            print(f'    "",   # DOF {i}: nothing found')
            continue
        phys = dc.denormalize_expr(best[1], best[2], i) or dc.expr_to_str(best[1], best[2])
        print(f'    "{phys}",')
    print("]\n", flush=True)

    if pool is not None:
        pool.shutdown(wait=False)
    return best_per_dof
