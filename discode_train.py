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

Per epoch:

  1. The policy samples a batch of candidate token sequences per DOF (plus a
     beam-search batch every ``beam_interval`` epochs).
  2. Every candidate has its constants fitted to that DOF's own work-energy
     balance and is scored by the reward (see :mod:`discode_core`).  The fit
     and the score are the same closed-form VARPRO path wherever possible.
  3. The top-``alpha`` fraction is promoted into the DOF's replay buffer,
     ranked by a novelty-scaled score but STORED as the raw reward.
  4. The critic and the J-GRPO objective are trained on that buffer.

Two architectures share this loop, selected by ``architecture``:

``'joint'`` (default)
    One autoregressive transformer writes every DOF's sequence as one
    flattened sequence, so each DOF conditions on the expressions already
    committed for the others (:mod:`discode_policy`).  Set
    ``cross_slice_attention=False`` to keep autoregression but forbid
    cross-DOF attention — N independent AR models sharing weights.

``'independent'``
    The original N one-shot masked-diffusion policies, one per DOF, unchanged.

The **reward and the buffers stay per-DOF under both**.
``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an exact per-DOF kinematic
identity — coupling forces are already inside ``a_d`` — so each balance closes
on its own, and mixing a partner's residual into the score would only make
rewards incomparable across epochs.  See the docstring of
:func:`discode_core.energy_worker`.  Only the *policy* is joint; advantages are
still z-scored within a single DOF's batch.

Buffer entries are ``(energy_reward, tau, consts, context)``.  Under
``'joint'`` the context records the slice order, this slice's slot, and the
slices that preceded it — everything the sample could see — so the entry stays
reproducible in isolation and comparable across epochs.  Under
``'independent'`` it is ``None``.
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
import discode_policy as dp
from discode_data import generate_dataset

N_WORKERS = max(1, min(os.cpu_count() or 4, 3))
HOF_SIZE = 10


# ── Per-DOF training state ──────────────────────────────────────────────────
class DOFState:
    """Everything that stays per-DOF: the replay buffer, its gating thresholds,
    the hall of fame, and the critic.

    The critic predicts *this* DOF's reward and is cheap, so it stays here even
    under a shared policy.  (Its baseline is deliberately computed and not used
    for the advantage — see :func:`discode_core.compute_jgrpo`.)
    """

    def __init__(self, max_len, device):
        self.critic     = dc.ExprCritic(vocab_size=dc.VOCAB_SIZE, max_len=max_len, d=64).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # buffer entry: (energy_reward, tau, consts, context)
        self.buffer     = []
        self.R_alpha    = 0.0
        self.best_r     = -np.inf
        self.best_entry = None
        self.hof        = []


# ── Policy state (one joint policy, or N independent ones) ──────────────────
class PolicyState:
    """Owns the policy weights, optimiser, schedule, and the old/ref copies.

    Both architectures expose the same three operations — :meth:`refresh`,
    :meth:`sample`, :meth:`update` — so the training loop below has a single
    code path and ``architecture='independent'`` keeps running exactly as it
    did before this change.
    """

    def __init__(self, n_dof, max_len, lr, n_epochs, device,
                 architecture='joint', cross_slice_attention=True,
                 n_layers=4, d_model=128, slice_order='random',
                 sample_order='reward', rng=None,
                 max_terms=8, max_term_len=8, term_grammar='free',
                 term_position_encoding=True):
        if architecture not in ('joint', 'independent', 'terms'):
            raise ValueError(f"architecture must be 'joint', 'independent' or "
                             f"'terms', got {architecture!r}")
        self.architecture = architecture
        self.n_dof        = n_dof
        self.device       = device
        self.slice_order  = slice_order
        self.sample_order = sample_order
        self.term_grammar = term_grammar
        self.rng          = np.random.default_rng() if rng is None else rng

        if architecture in ('joint', 'terms'):
            if architecture == 'joint':
                self.policy = dp.ARJointPolicy(
                    n_tokens=dc.N_TOKENS, max_len=max_len, n_dof=n_dof,
                    d_model=d_model, n_layers=n_layers,
                    cross_slice_attention=cross_slice_attention).to(device)
            else:
                self.policy = dp.TermBagPolicy(
                    n_tokens=dc.N_TOKENS, max_terms=max_terms,
                    max_term_len=max_term_len, n_dof=n_dof,
                    d_model=d_model, n_layers=n_layers,
                    cross_slice_attention=cross_slice_attention,
                    term_position_encoding=term_position_encoding).to(device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-5)
            self.policy_old = copy.deepcopy(self.policy).to(device); self.policy_old.eval()
            self.policy_ref = copy.deepcopy(self.policy).to(device); self.policy_ref.eval()
            self.nets = [self.policy]
        else:
            self.models = [dc.DiffusionModel(n_tokens=dc.N_TOKENS, max_len=max_len).to(device)
                           for _ in range(n_dof)]
            self.optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
            self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
                o, T_max=n_epochs, eta_min=1e-5) for o in self.optimizers]
            self.models_old = [copy.deepcopy(m).to(device).eval() for m in self.models]
            self.models_ref = [copy.deepcopy(m).to(device).eval() for m in self.models]
            self.nets = self.models

    def n_params(self):
        return sum(p.numel() for net in self.nets for p in net.parameters()
                   if p.requires_grad)

    def dedup_key(self, tau):
        """Buffer-dedup key for a token sequence.

        Under ``'terms'`` two bags with the same terms in a different order are
        the same expression (the reward is order-blind), so the key is the
        sorted term multiset.  The other architectures keep the raw sequence so
        their behaviour — and any baseline numbers — are unchanged.
        """
        if self.architecture == 'terms':
            return tuple(sorted(tuple(t) for t in dc.split_terms(tau)))
        return tuple(tau)

    def refresh(self, ep, G):
        """Refresh the reference policy every ``G`` epochs, the old one every epoch."""
        if self.architecture in ('joint', 'terms'):
            if ep % G == 0:
                self.policy_ref = copy.deepcopy(self.policy).to(self.device); self.policy_ref.eval()
            self.policy_old = copy.deepcopy(self.policy).to(self.device); self.policy_old.eval()
        else:
            if ep % G == 0:
                self.models_ref = [copy.deepcopy(m).to(self.device).eval() for m in self.models]
            self.models_old = [copy.deepcopy(m).to(self.device).eval() for m in self.models]

    # ── sampling ────────────────────────────────────────────────────────────
    def sample(self, batch_size, max_len, best_r, do_beam, beam_width):
        """Return, per DOF, a list of ``(tau, context)``.

        Under ``'joint'`` the slice order is chosen once per epoch by
        ``sample_order`` (``'reward'`` puts the most-converged DOF first, so the
        others condition on it) and then varied across the batch by
        ``slice_order`` — see :func:`discode_policy.make_batch_orders` for why
        both matter.
        """
        n = self.n_dof
        if self.architecture == 'independent':
            out = []
            for i, m in enumerate(self.models):
                exprs = dc.sample_batch(m, batch_size, max_len)
                if do_beam:
                    exprs.extend(dc.beam_search_expressions(
                        m, beam_width, max_len, n_return=beam_width))
                out.append([(tau, None) for tau in exprs])
            return out

        base   = dp.choose_order(best_r, self.sample_order, self.rng)
        orders = dp.make_batch_orders(base, batch_size, self.slice_order, self.rng)
        if self.architecture == 'terms':
            samples = dp.sample_term_batch(self.policy, batch_size, n, orders,
                                           term_grammar=self.term_grammar)
        else:
            samples = dp.sample_joint_batch(self.policy, batch_size, max_len, n, orders)

        out = [[] for _ in range(n)]
        for b, sample in enumerate(samples):
            row = orders[b]
            slice_taus = [sample[int(row[s])] for s in range(n)]
            for slot in range(n):
                out[int(row[slot])].append(
                    (slice_taus[slot], dp.make_context(row, slot, slice_taus)))

        if do_beam:
            if self.architecture == 'terms':
                cands = dp.beam_term_candidates(
                    self.policy, base, n, beam_width, n_return=beam_width,
                    term_grammar=self.term_grammar)
            else:
                cands = dp.beam_candidates(
                    self.policy, base, max_len, n, beam_width, n_return=beam_width)
            for d, tau, ctx in cands:
                out[d].append((tau, ctx))
        return out

    # ── policy update ───────────────────────────────────────────────────────
    def update(self, states, C, max_len, eps, beta, lam_t, trainable=None):
        """``trainable`` is the set of DOFs that promoted something this epoch.

        A DOF that produced no valid candidate sits the epoch out, exactly as
        it did before this change — under a shared policy that also stops a
        stale buffer from pushing the trunk on its own.
        """
        eligible = [i for i, st in enumerate(states)
                    if st.buffer and (trainable is None or i in trainable)]

        if self.architecture == 'independent':
            for i in eligible:
                _grpo_steps_independent(self, i, states[i], C, max_len, eps, beta, lam_t)
            return

        # One shared policy: one backward pass per GRPO step, but the advantage
        # stays per-DOF (never pooled — reward scales differ wildly between
        # DOFs, and pooling would let the wider-spread DOF dominate the trunk).
        self.policy.train()
        for _j in range(C):
            total_loss = None
            for d in eligible:
                st = states[d]
                critic = st.critic if len(st.buffer) >= 32 else None
                if self.architecture == 'terms':
                    jg, ent, ok = dp.jgrpo_terms(
                        self.policy, self.policy_old, self.policy_ref,
                        st.buffer, R_alpha=min(e[0] for e in st.buffer),
                        dof=d, n_dof=self.n_dof, eps=eps, beta=beta,
                        critic=critic, critic_max_len=max_len)
                else:
                    jg, ent, ok = dp.jgrpo_ar(
                        self.policy, self.policy_old, self.policy_ref,
                        st.buffer, R_alpha=min(e[0] for e in st.buffer),
                        dof=d, n_dof=self.n_dof, max_len=max_len, eps=eps, beta=beta,
                        critic=critic)
                if not ok:
                    # A degenerate spread (or nothing above R_alpha) zeroes out
                    # THIS DOF only.  Returning early here — as the per-DOF
                    # version could afford to — would throw away the other
                    # DOFs' gradients along with it.
                    continue
                term = -(jg + lam_t * ent)
                total_loss = term if total_loss is None else total_loss + term
            if total_loss is None:
                continue
            self.optimizer.zero_grad(); total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
        self.scheduler.step()


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


def _grpo_steps_independent(ps, i, st, C, max_len, eps, beta, lam_t):
    """The original per-DOF masked-diffusion update, unchanged."""
    model = ps.models[i]
    S_grpo = [(e[0], e[1], e[2]) for e in st.buffer]
    R_grpo = min(r for r, _, _ in S_grpo)
    model.train()
    for _j in range(C):
        t_diff  = int(np.random.randint(1, max_len + 1))
        jg, ent = dc.compute_jgrpo(
            model, ps.models_old[i], ps.models_ref[i],
            S_grpo, R_grpo, t_diff, max_len, eps, beta,
            critic=st.critic if len(st.buffer) >= 32 else None)
        loss = -(jg + lam_t * ent)
        ps.optimizers[i].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ps.optimizers[i].step()
    ps.schedulers[i].step()


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
    # ── policy architecture ────────────────────────────────────────────────
    architecture          = 'joint',   # 'joint' | 'independent'
    cross_slice_attention = True,      # False -> V0+AR: AR, but no cross-DOF
    n_layers              = 4,         # joint only (= old n_enc + n_dec)
    d_model               = 128,       # joint only
    slice_order           = 'random',  # batch slice orders: 'random' | 'fixed'
    sample_order          = 'reward',  # epoch base order: 'reward'|'random'|'fixed'
    seed                  = None,      # slice-order RNG
    # ── 'terms' architecture only ──────────────────────────────────────────
    max_terms             = 8,         # term slots per DOF (bag capacity)
    max_term_len          = 8,         # token budget per term
    term_grammar          = 'free',    # 'free' | 'varpro' (see discode_policy)
    term_position_encoding = True,     # False -> Change 2: order-blind bag
):
    """Search for one acceleration expression per DOF of ``system_data``.

    Returns a list of ``(reward, tau, consts, context)`` — the best entry per
    DOF, or ``None`` for a DOF where nothing survived.  ``context`` is ``None``
    under ``architecture='independent'``.

    The ablation matrix is five configs of this one function:

    ==========  ==========================================================
    V0          ``architecture='independent'``
    V0+AR       ``architecture='joint', cross_slice_attention=False``
    B           ``architecture='joint', cross_slice_attention=True``
    C1          ``architecture='terms'`` — terms in a bag, term order known
    C2          ``architecture='terms', term_position_encoding=False``
    ==========  ==========================================================

    C1 asks whether generating an expression as short independent TERMS
    summed at assembly helps at all; C2 drops the one embedding band that
    tells term k which earlier term came first, and asks whether permutation
    invariance helps on top.  Mask, sampler and update are identical between
    them.  See the section comment in ``discode_policy`` for the layout.

    V0 has N policies and the joint variants have one, so their parameter
    counts do not match by construction; both are printed at startup, and
    ``n_layers`` / ``d_model`` are exposed so a capacity-matched run can be
    configured explicitly rather than assumed.
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

    states = [DOFState(max_len, device) for _ in range(N)]
    ps = PolicyState(N, max_len, lr, n_epochs, device,
                     architecture=architecture,
                     cross_slice_attention=cross_slice_attention,
                     n_layers=n_layers, d_model=d_model,
                     slice_order=slice_order, sample_order=sample_order,
                     rng=np.random.default_rng(seed),
                     max_terms=max_terms, max_term_len=max_term_len,
                     term_grammar=term_grammar,
                     term_position_encoding=term_position_encoding)
    if architecture == 'joint':
        print(f"[policy] joint autoregressive  "
              f"cross_slice_attention={cross_slice_attention}  "
              f"n_layers={n_layers}  d_model={d_model}")
        print(f"[policy] slice_order={slice_order}  sample_order={sample_order}")
    elif architecture == 'terms':
        print(f"[policy] terms-in-a-bag  max_terms={max_terms}  "
              f"max_term_len={max_term_len}  term_grammar={term_grammar}  "
              f"term_position_encoding={term_position_encoding}")
        print(f"[policy] cross_slice_attention={cross_slice_attention}  "
              f"n_layers={n_layers}  d_model={d_model}  "
              f"slice_order={slice_order}  sample_order={sample_order}")
    else:
        print(f"[policy] {N} independent masked-diffusion policies")
    print(f"[policy] {ps.n_params():,} trainable parameters "
          f"({len(ps.nets)} network{'s' if len(ps.nets) > 1 else ''})\n")

    for ep in range(n_epochs):
        t0_ep   = time.time()
        horizon = dc.get_traj_horizon(ep, n_epochs, system.t_end)
        lam_t   = lam_start * (lam_end / lam_start) ** (ep / max(n_epochs - 1, 1))

        print(f"\n{'=' * 64}")
        print(f"Ep {ep + 1}/{n_epochs}  |  horizon={horizon:.2f}s  |  lam={lam_t:.5f}")

        # refresh reference / old policies
        ps.refresh(ep, G)

        # ── Step 1: Sampling ────────────────────────────────────────────────
        t1 = time.time()
        all_exprs = ps.sample(batch_size, max_len,
                              best_r=[st.best_r for st in states],
                              do_beam=(ep > 0 and ep % beam_interval == 0),
                              beam_width=beam_width)
        print(f"  [1-sample]   {batch_size} exprs/DOF  ({time.time()-t1:.2f}s)")
        sys.stdout.flush()

        # ── Step 2: Energy scoring (each DOF scored in isolation) ───────────
        # Contexts never go to the workers — they have no policy and no use for
        # them, and shipping them would inflate the pickling cost per
        # candidate.  ``pool.map`` preserves order, so they zip back by index.
        t1 = time.time()
        tasks, ctxs = [], []
        for i in range(N):
            for tau, ctx in all_exprs[i]:
                tasks.append((i, tau, None, horizon))
                ctxs.append(ctx)
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
        for k, res in enumerate(results):
            if res is None:
                continue
            tgt, r, tau, c = res
            per_dof[tgt].append((r, tau, c, ctxs[k]))
        print(f"  [2-energy]   scored  ({time.time()-t1:.2f}s)", flush=True)

        # ── Step 3: promotion + buffer update + train ───────────────────────
        # Promotion and the critic are per-DOF; the policy update is not (a
        # joint policy takes one step across every DOF's buffer), so it sits
        # between the two per-DOF passes rather than inside the loop.  For
        # `independent` this is a pure reordering: the buffers never interact,
        # so pruning DOF 0 before or after DOF 1's update makes no difference.
        t1 = time.time()
        report = {}
        for i, st in enumerate(states):
            batch_r = per_dof[i]
            if not batch_r:
                print(f"  DOF {i}: no valid candidate this epoch"); continue

            # Novelty is a SELECTION heuristic, never a reward.  It depends on
            # the buffer contents at scoring time, so baking it into the stored
            # value would make entries from different epochs incomparable.
            # Rank by the scaled score, but gate on and store the raw reward.
            ranked = [(e[0], e) for e in batch_r]        # (r_sel, entry)
            if st.buffer and novelty_weight > 0:
                ranked = [(e[0] * (1 - novelty_weight
                                   + novelty_weight * dc.structural_novelty(e[1], st.buffer)),
                           e)
                          for _rs, e in ranked]
            ranked.sort(key=lambda z: z[0], reverse=True)
            batch_r = [e for _rs, e in ranked]

            k_top = max(1, int(len(batch_r) * alpha))
            if len(st.buffer) < max_buffer:
                promoted = batch_r[:k_top]
            else:
                promoted = [e for e in batch_r[:k_top] if e[0] > st.R_alpha]
                if not promoted:
                    promoted = batch_r[:1]

            # Dedup on the token sequence alone, never on the context: the same
            # structure found under a different slice order is the same
            # expression, and the reward that gates it was computed in
            # isolation either way.
            seen = set(ps.dedup_key(e[1]) for e in st.buffer)
            n_added = 0
            for entry in promoted:
                key = ps.dedup_key(entry[1])
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
            report[i] = (max(e[0] for e in batch_r), n_added, c_loss)

        ps.update(states, C, max_len, eps, beta, lam_t, trainable=set(report))

        for i, st in enumerate(states):
            if i not in report:
                continue
            st.buffer.sort(key=lambda z: z[0], reverse=True)
            if len(st.buffer) >= max_buffer:
                st.buffer = st.buffer[:max_buffer]
                n_rm = max(1, int(max_buffer * alpha))
                st.buffer = st.buffer[:-n_rm]
            if st.buffer:
                st.R_alpha = float(np.percentile([e[0] for e in st.buffer], 5))

            best_raw, n_added, c_loss = report[i]
            c_str = f'  critic_loss={c_loss:.4f}' if c_loss is not None else ''
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
