"""
discover_train.py
=================

The DISCOVER training loop.  One function, :func:`DISCOVER_TRAIN`, which takes a
:class:`discover_data.SystemData` and returns the best expression found for each
DOF.

It knows nothing about MATLAB files, system libraries or ground truth — the
drivers (``discover_sdof_sim``, ``discover_sdof_exp``, ``discover_mdof_sim``,
``discover_mdof_exp``) build the ``SystemData`` and call this.  A single-DOF run
is just ``n_dof == 1``; there is no separate SDOF loop.

Per epoch:

  1. The policy samples a batch of candidate token sequences per DOF (plus a
     beam-search batch every ``beam_interval`` epochs).
  2. Every candidate has its constants fitted to that DOF's own work-energy
     balance and is scored by the chosen reward (``reward='energy'``, the
     work-energy / acceleration blend, or ``reward='simulation'``, a forward
     simulation against the record -- see :mod:`discover_core`).  The fit is
     the same closed-form VARPRO path wherever possible under both.
  3. The top-``alpha`` fraction is promoted into the DOF's replay buffer,
     ranked by a novelty-scaled score but STORED as the raw reward.
  4. The critic and the J-GRPO objective are trained on that buffer.

The policy is the term-structured transformer of :mod:`discover_policy`: one
network writes every DOF's expression as a bag of short terms summed under a
root ``add`` it never has to emit, and each DOF conditions on the bags already
committed for the DOFs before it in the slice order.  Set
``cross_slice_attention=False`` to forbid that cross-DOF attention (N
independent term-bag policies sharing weights) and
``term_position_encoding=False`` to make the bag order-blind.

The **reward and the buffers stay per-DOF**.
``integral(v_d * a_d) = 1/2 (v_d^2 - v_d(0)^2)`` is an exact per-DOF kinematic
identity — coupling forces are already inside ``a_d`` — so each balance closes
on its own, and mixing a partner's residual into the score would only make
rewards incomparable across epochs.  See the docstring of
:func:`discover_core.energy_worker`.  Only the *policy* is shared; advantages
are still z-scored within a single DOF's batch.

Buffer entries are ``(reward, tau, consts, context)``.  ``tau`` is the
flat assembled sum.  The context records the slice order, this slice's slot,
and the slices that preceded it — everything the sample could see — so the
entry stays reproducible in isolation and comparable across epochs.
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import discover_core as dc
import discover_policy as dp
from discover_data import generate_dataset

N_WORKERS = max(1, min(os.cpu_count() or 4, 3))
HOF_SIZE = 10


# ── Per-DOF training state ──────────────────────────────────────────────────
class DOFState:
    """Everything that stays per-DOF: the replay buffer, its gating thresholds,
    the hall of fame, and the critic.

    The critic predicts *this* DOF's reward and is cheap, so it stays here even
    under a shared policy.  (Its baseline is deliberately computed and not used
    for the advantage — see :func:`discover_policy.jgrpo_terms`.)  ``max_len``
    is the critic's token window; an assembled tau longer than that is
    truncated on the way in.
    """

    def __init__(self, max_len, device):
        self.critic     = dc.ExprCritic(vocab_size=dc.VOCAB_SIZE, max_len=max_len, d=64).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # buffer entry: (reward, tau, consts, context)
        self.buffer     = []
        self.R_alpha    = 0.0
        self.best_r     = -np.inf
        self.best_entry = None
        self.hof        = []


# ── Policy state (one shared term-bag policy) ───────────────────────────────
class PolicyState:
    """Owns the policy weights, optimiser, schedule, and the old/ref copies,
    and exposes the three operations the loop needs: :meth:`refresh`,
    :meth:`sample`, :meth:`update`.
    """

    def __init__(self, n_dof, lr, n_epochs, device,
                 cross_slice_attention=True, n_layers=4, d_model=128,
                 slice_order='random', sample_order='reward', rng=None,
                 max_terms=8, max_term_len=8, term_grammar='free',
                 term_position_encoding=True):
        self.n_dof        = n_dof
        self.device       = device
        self.slice_order  = slice_order
        self.sample_order = sample_order
        self.term_grammar = term_grammar
        self.rng          = np.random.default_rng() if rng is None else rng

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

    def n_params(self):
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    @staticmethod
    def dedup_key(tau):
        """Buffer-dedup key for a token sequence: the sorted term multiset.

        Two bags with the same terms in a different order are the same
        expression (the reward is order-blind), so they must not both enter
        the buffer.
        """
        return tuple(sorted(tuple(t) for t in dc.split_terms(tau)))

    def refresh(self, ep, G):
        """Refresh the reference policy every ``G`` epochs, the old one every epoch."""
        if ep % G == 0:
            self.policy_ref = copy.deepcopy(self.policy).to(self.device); self.policy_ref.eval()
        self.policy_old = copy.deepcopy(self.policy).to(self.device); self.policy_old.eval()

    # ── sampling ────────────────────────────────────────────────────────────
    def sample(self, batch_size, best_r, do_beam, beam_width):
        """Return, per DOF, a list of ``(tau, context)``.

        The slice order is chosen once per epoch by ``sample_order``
        (``'reward'`` puts the most-converged DOF first, so the others
        condition on it) and then varied across the batch by ``slice_order``
        — see :func:`discover_policy.make_batch_orders` for why both matter.
        """
        n = self.n_dof
        base   = dp.choose_order(best_r, self.sample_order, self.rng)
        orders = dp.make_batch_orders(base, batch_size, self.slice_order, self.rng)
        samples = dp.sample_term_batch(self.policy, batch_size, n, orders,
                                       term_grammar=self.term_grammar)

        out = [[] for _ in range(n)]
        for b, sample in enumerate(samples):
            row = orders[b]
            slice_taus = [sample[int(row[s])] for s in range(n)]
            for slot in range(n):
                out[int(row[slot])].append(
                    (slice_taus[slot], dp.make_context(row, slot, slice_taus)))

        if do_beam:
            cands = dp.beam_term_candidates(
                self.policy, base, n, beam_width, n_return=beam_width,
                term_grammar=self.term_grammar)
            for d, tau, ctx in cands:
                out[d].append((tau, ctx))
        return out

    # ── policy update ───────────────────────────────────────────────────────
    def update(self, states, C, critic_max_len, eps, beta, lam_t, trainable=None):
        """``trainable`` is the set of DOFs that promoted something this epoch.

        A DOF that produced no valid candidate sits the epoch out — under a
        shared policy that also stops a stale buffer from pushing the trunk on
        its own.
        """
        eligible = [i for i, st in enumerate(states)
                    if st.buffer and (trainable is None or i in trainable)]

        # One shared policy: one backward pass per GRPO step, but the advantage
        # stays per-DOF (never pooled — reward scales differ wildly between
        # DOFs, and pooling would let the wider-spread DOF dominate the trunk).
        self.policy.train()
        for _j in range(C):
            total_loss = None
            for d in eligible:
                st = states[d]
                critic = st.critic if len(st.buffer) >= 32 else None
                jg, ent, ok = dp.jgrpo_terms(
                    self.policy, self.policy_old, self.policy_ref,
                    st.buffer, R_alpha=min(e[0] for e in st.buffer),
                    dof=d, n_dof=self.n_dof, eps=eps, beta=beta,
                    critic=critic, critic_max_len=critic_max_len)
                if not ok:
                    # A degenerate spread (or nothing above R_alpha) zeroes out
                    # THIS DOF only.  Returning early here would throw away the
                    # other DOFs' gradients along with it.
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


# ── Coupled scoring (reward='simulation', sim_coupling) ─────────────────────
_MODE_LABEL = {'alone': 'alone', 'top': 'with top', 'peer': 'with peers'}


def _scored_with(entry, dof, n_dof, indent=10):
    """Lines naming what DOF ``dof``'s equation was simulated with to earn its
    score: per mode, every other DOF's equation in physical units -- printed
    even when it is that DOF's current best -- or the measured record."""
    sw = entry[3].get('scored_with') if isinstance(entry[3], dict) else None
    if not sw:
        return []
    pad, lines = ' ' * indent, []
    for m in ('alone', 'top', 'peer'):
        if m not in sw['modes']:
            continue
        partners = sw['partners'].get(m, {})
        lines.append(f"{pad}scored {_MODE_LABEL[m]} ({sw['modes'][m]:.4f}), "
                     f"simulated with:")
        for j in range(n_dof):
            if j == dof:
                continue
            e = partners.get(j)
            what = ('the measured record' if e is None else
                    dc.denormalize_expr(e[0], e[1], j)
                    or dc.expr_to_str(e[0], e[1]))
            lines.append(f"{pad}  DOF {j}: {what}")
    if sw.get('tuned'):
        lines.append(f"{pad}(amplitudes tuned on the simulation)")
    return lines


def check_coupling(sim_coupling, reward, n_dof, mode='split'):
    """``(f_top, f_peer)`` as floats, or a ValueError unless they are two
    non-negative fractions summing to at most 1 -- and, when either is
    nonzero, the reward is ``'simulation'`` and there are two DOFs or more --
    and ``mode`` is ``'split'`` or ``'blend'``."""
    if mode not in ('split', 'blend'):
        raise ValueError("sim_coupling_mode must be 'split' or 'blend', "
                         f"got {mode!r}")
    try:
        f = tuple(float(x) for x in sim_coupling)
    except (TypeError, ValueError):
        f = ()
    if (len(f) != 2 or not all(np.isfinite(x) and x >= 0.0 for x in f)
            or sum(f) > 1.0 + 1e-9):
        raise ValueError("sim_coupling must be two non-negative fractions "
                         "(with top, with peers) summing to at most 1, "
                         f"got {sim_coupling!r}")
    if sum(f) > 0.0 and reward != 'simulation':
        raise ValueError("sim_coupling needs reward='simulation'")
    if sum(f) > 0.0 and n_dof < 2:
        raise ValueError("sim_coupling needs at least two DOFs")
    return f


def coupling_plan(n_rows, n_cands, f_top, f_peer, rng):
    """Which candidates are simulated with other DOFs' equations this epoch.

    Index ``b < n_rows`` of every DOF's candidate list is row ``b`` of one
    joint sample; ``n_cands[d]`` is the length of DOF ``d``'s list (its rows
    plus any beam extras).  Returns ``(peer_rows, top)``: the
    ``round(f_peer * n_rows)`` rows whose equations are simulated together,
    and per DOF the set of ``round(f_top * n_cands[d])`` of its other
    candidates to simulate with the other DOFs' top equations.
    """
    n_peer = min(n_rows, int(round(f_peer * n_rows)))
    peer_rows = sorted(int(b) for b in rng.choice(n_rows, size=n_peer,
                                                  replace=False))
    taken = set(peer_rows)
    top = []
    for n_d in n_cands:
        free = [k for k in range(n_d) if k >= n_rows or k not in taken]
        n_top = min(len(free), int(round(f_top * n_d)))
        top.append(set(int(k) for k in rng.choice(free, size=n_top,
                                                  replace=False))
                   if n_top else set())
    return peer_rows, top


# ── Main loop ───────────────────────────────────────────────────────────────
def DISCOVER_TRAIN(
    system_data,               # a discover_data.SystemData
    n_epochs         = 300,
    batch_size       = 200,
    max_len          = 20,     # critic token window; the policy's own budget
                               # is max_terms * max_term_len
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
                               # (under reward='simulation' it only steers the
                               # constant fit)
    reward           = 'energy',   # 'energy' | 'simulation'
    trial_decay      = 1.0,    # how a candidate's trials are combined, either
                               # reward: ranked worst-fit first, the k-th worst
                               # weighted trial_decay**(k-1); 1 = plain mean,
                               # 0.5 = halve per rank, 0 = worst trial only
    sim_window       = None,   # simulation: None = one free run per trial,
                               # a number = restart from the record every
                               # sim_window seconds, 'auto' = every 3 periods
                               # of the fastest motion; windows apply to the
                               # pointwise terms, time-frequency runs free
    sim_weights      = (1.0, 0.0, 0.0, 0.0),  # simulation: weights of
                               # (displacement, velocity, acceleration,
                               # time-frequency), non-negative, summing to 1
    sim_coupling     = (0.0, 0.0),  # simulation, 2+ DOFs: the fractions of
                               # each DOF's equations simulated with the other
                               # DOFs' top equations, and with the other DOFs'
                               # equations from the same sample; the rest are
                               # simulated alone, partners read off the record
    sim_coupling_mode = 'split',  # 'split': each equation scored ONE of those
                               # ways, drawn with those fractions; 'blend':
                               # each scored all three ways, rewards weighted
                               # by them
    sim_refine       = 0,      # simulation: per DOF per epoch, the best this
                               # many equations get their amplitudes tuned on
                               # the simulation itself (0 = off)
    sim_refine_evals = 60,     # simulation: simulations per tuned equation
    use_pool         = True,
    center_features  = False,  # see generate_dataset(): scale-only by default
    # ── policy (see discover_policy) ───────────────────────────────────────
    cross_slice_attention = True,      # False -> no cross-DOF attention
    n_layers              = 4,         # transformer depth
    d_model               = 128,       # transformer width
    slice_order           = 'random',  # batch slice orders: 'random' | 'fixed'
    sample_order          = 'reward',  # epoch base order: 'reward'|'random'|'fixed'
    seed                  = None,      # slice-order RNG
    max_terms             = 8,         # term slots per DOF (bag capacity)
    max_term_len          = 8,         # token budget per term
    term_grammar          = 'free',    # 'free' | 'varpro' (see discover_policy)
    term_position_encoding = True,     # False -> order-blind bag
    directional_leaves     = False,    # True -> add the dleaf / vleaf tokens
    transcendental         = False,    # True -> add the blend token
):
    """Search for one acceleration expression per DOF of ``system_data``.

    Returns a list of ``(reward, tau, consts, context)`` — the best entry per
    DOF, or ``None`` for a DOF where nothing survived.

    ``reward`` picks what candidates are scored by.  ``'energy'`` is the
    work-energy residual blended with acceleration NRMSE by ``w_acc``.
    ``'simulation'`` integrates each candidate forward from the measured
    initial state -- one DOF at a time, the other DOFs' states read off the
    record -- and scores ``w_q*NRMSE(q) + w_v*NRMSE(qdot) + w_a*NRMSE(a)
    + w_tf*R_tf`` against the record, ``sim_weights = (w_q, w_v, w_a, w_tf)``
    summing to 1.  The acceleration is the equation's own along its simulated
    trajectory; ``R_tf`` compares the local amplitude in each frequency band
    over time, so it sees beats and is blind to phase drift (see
    :mod:`discover_rollout`).
    ``sim_window`` restarts the simulation from the record every that many
    seconds (``'auto'``: every 3 periods of the fastest motion) for the
    pointwise terms, which on one long free run saturate for any slightly-off
    frequency; the time-frequency term always gets a free run of its own.
    Constants are fitted the same way under both.

    ``trial_decay`` sets how each candidate's trials are combined into its
    reward, under either scheme.  The trials are ranked by how badly the
    candidate fits them and the k-th worst is weighted ``trial_decay**(k-1)``
    (normalised): 1 is the plain mean, 0.5 halves the weight at each rank (52,
    26, 13, 6, 3% over five trials), 0 keeps the worst trial alone.  A
    behaviour only one trial shows -- beats, say -- then counts most until it
    is captured, without anyone choosing that trial.  The constant fit is
    untouched (:func:`discover_rollout.combine_trials`).

    ``sim_coupling = (f_top, f_peer)`` lets the simulation reward see the
    coupled system.  Each epoch a fraction ``f_top`` of every DOF's equations
    is integrated together with the other DOFs' top equations (each one's best
    so far), and a fraction ``f_peer`` of the jointly sampled rows is
    integrated whole: each DOF's equation with the other DOFs' equations from
    the same sample.  Each equation is still scored on its own channel, but a
    set that blows up charges every equation in it.  The rest are simulated
    alone as before (:func:`discover_core.coupled_worker`).

    ``sim_refine`` tunes, each epoch, the amplitudes of each DOF's best few
    equations on the simulation itself, scored exactly as they were (same
    modes and partners), within ``sim_refine_evals`` simulations each --
    starting from the buffer's tuned copy of the same structure when it has
    one, so tuning accumulates over the epochs a structure keeps winning.  The
    closed-form fit targets the equation error, and a free run magnifies what
    that misses -- a beat period is a difference of two close frequencies,
    so a 1% coupling error can move it by tens of percent.  A tuned equation
    replaces its untuned twin in the buffer when it scores higher
    (:func:`discover_core.refine_worker`).

    ``sim_coupling_mode`` says how the fractions apply.  ``'split'`` scores
    each equation ONE way, drawn with those fractions, and its reward is
    whatever that way scored.  Promotion and the GRPO advantage compare raw
    rewards across the whole batch, so an equation that only works against
    the measured partner still wins whenever it is drawn alone, and since a
    peer score mostly reflects the row-mate, a peer-mode equation reaches the
    buffer only when its whole row is good.  ``'blend'`` scores every
    equation all three ways and takes ``(1 - f_top - f_peer) * r_alone
    + f_top * r_top + f_peer * r_peer``, so an equation that fails coupled
    pays every time -- but the peer lottery then enters every reward;
    ``sim_coupling=(f, 0)`` with ``'blend'`` penalises coupled failure
    without it.  ``'blend'`` costs one alone and one with-top simulation per
    equation plus one joint simulation per row.

    Two knobs isolate two hypotheses and are worth tracking separately:

    ``term_position_encoding``
        ``True``: the model knows which term came first.  ``False`` drops the
        one embedding band that tells term ``k`` which earlier term came
        first, and asks whether permutation invariance helps on top of term
        decomposition.  Mask, sampler and update are identical between them.

    ``cross_slice_attention``
        ``False`` forbids a DOF from attending to the other DOFs' bags, and
        asks whether cross-DOF structure sharing is doing anything.  The gain
        can only show when the shared subtree is more than one token — use
        ``system_key='cubic_coupled'`` in :mod:`discover_mdof_sim`.

    ``n_layers`` / ``d_model`` are exposed so a capacity-matched comparison can
    be configured explicitly; the parameter count is printed at startup.
    """
    device = dc.DEVICE
    system = system_data
    N = system.n_dof
    f_top, f_peer = check_coupling(sim_coupling, reward, N, sim_coupling_mode)
    coupled = f_top + f_peer > 0.0
    if int(sim_refine) < 0 or int(sim_refine_evals) < 1:
        raise ValueError("sim_refine must be >= 0 and sim_refine_evals >= 1")
    if sim_refine and reward != 'simulation':
        raise ValueError("sim_refine needs reward='simulation'")

    dc.configure_grammar(N, system.var_names, directional_leaves,
                         transcendental)
    print(f"[vocab]  {N}-DOF tokens: {dc.ALL_TOKENS}")
    if directional_leaves:
        print(f"[vocab]  dleaf spans {[dc.VARIABLES[c] for c in dc.LEAF_CHANNELS['dleaf']]}"
              f", vleaf spans {[dc.VARIABLES[c] for c in dc.LEAF_CHANNELS['vleaf']]}"
              f"  ({dc.N_DOF} fitted weights each)")
    if transcendental:
        print(f"[vocab]  blend(u) = e^(a u)(c1 cos(b u) + c2 sin(b u)): "
              f"a grid {dc.BLEND_A_GRID}, b grid {dc.BLEND_B_GRID}")
    print(f"[config] System='{system.name}'  N_DOF={N}  reward={reward}")
    print(f"[config] {len(system.time)} pts  {system.t_end:.4g} s  "
          f"{(len(system.time)-1)/system.t_end:.0f} Hz eff  "
          f"{system.n_trials} trials")

    X_torch, y_list, raw_trajs, norm_stats = generate_dataset(
        system, device, center=center_features)
    dc.set_problem_data(norm_stats, raw_trajs, energy_normalize, max_traj, w_acc)
    dc.set_reward(reward, sim_window, sim_weights, trial_decay)
    print(f"[config] reward: {dc.describe_reward()}")
    if coupled and sim_coupling_mode == 'split':
        print(f"[config] coupling (split): per DOF, {f_top:.0%} of equations "
              f"simulated with the other DOFs' top, {f_peer:.0%} with the other "
              f"DOFs' equations from the same sample, "
              f"{1.0 - f_top - f_peer:.0%} alone")
    elif coupled:
        ways = [(m_, w) for m_, w in (('alone', 1.0 - f_top - f_peer),
                                      ('top', f_top), ('peer', f_peer)) if w > 0]
        print("[config] coupling (blend): every equation scored "
              + ", ".join(_MODE_LABEL[m_] for m_, _w in ways) + "; reward = "
              + " + ".join(f"{w:.2f} {_MODE_LABEL[m_]}" for m_, w in ways))
    if sim_refine:
        print(f"[config] tuning: per DOF per epoch, the best {sim_refine} "
              f"equations' amplitudes tuned on the simulation "
              f"({sim_refine_evals} runs each)")
    print(f"[config] trajectories used per fit/score: {dc.MAX_TRAJ} "
          f"of {len(raw_trajs)} available\n")
    if reward == 'simulation':
        dc.print_record_summary(
            horizon=dc.get_traj_horizon(0, n_epochs, system.t_end))
        print()

    # Scoring pool (static data injected once).
    pool = saved_env = None
    if use_pool:
        pool, saved_env = dc.make_pool(
            (N, system.var_names, norm_stats, raw_trajs, energy_normalize,
             max_traj, w_acc, directional_leaves, transcendental, reward,
             sim_window, sim_weights, trial_decay), N_WORKERS)
        print(f"Scoring pool : {N_WORKERS} single-threaded workers\n")

    states = [DOFState(max_len, device) for _ in range(N)]
    ps = PolicyState(N, lr, n_epochs, device,
                     cross_slice_attention=cross_slice_attention,
                     n_layers=n_layers, d_model=d_model,
                     slice_order=slice_order, sample_order=sample_order,
                     rng=np.random.default_rng(seed),
                     max_terms=max_terms, max_term_len=max_term_len,
                     term_grammar=term_grammar,
                     term_position_encoding=term_position_encoding)
    print(f"[policy] terms-in-a-bag  max_terms={max_terms}  "
          f"max_term_len={max_term_len}  term_grammar={term_grammar}  "
          f"term_position_encoding={term_position_encoding}")
    print(f"[policy] cross_slice_attention={cross_slice_attention}  "
          f"n_layers={n_layers}  d_model={d_model}  "
          f"slice_order={slice_order}  sample_order={sample_order}")
    print(f"[policy] {ps.n_params():,} trainable parameters  "
          f"(critic max_len={max_len})\n")

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
        all_exprs = ps.sample(batch_size,
                              best_r=[st.best_r for st in states],
                              do_beam=(ep > 0 and ep % beam_interval == 0),
                              beam_width=beam_width)
        print(f"  [1-sample]   {batch_size} exprs/DOF  ({time.time()-t1:.2f}s)")
        sys.stdout.flush()

        # ── Step 2: Scoring ─────────────────────────────────────────────────
        # Each candidate is scored alone -- its DOF simulated or balanced in
        # isolation, the partners read off the record -- unless sim_coupling
        # sends it to be simulated with its row's other equations ('peer') or
        # with the other DOFs' top equations ('top'), or, under 'blend', all
        # three ways (one task per row: the row's equations are fitted once).
        # Row b of every DOF's list is one joint sample (PolicyState.sample);
        # beam extras come after the rows.  Contexts never go to the
        # workers -- they have no policy and no use for them, and shipping them
        # would inflate the pickling cost per candidate; every task carries
        # the (dof, index) slots its results belong to instead.
        t1 = time.time()
        mode, tops, top_sets, blend = {}, {}, [set() for _ in range(N)], None
        iso_tasks, iso_slots, cpl_tasks, cpl_slots = [], [], [], []
        if coupled:
            n_rows = min([batch_size] + [len(c) for c in all_exprs])
            tops = {j: (st.best_entry[1], st.best_entry[2])
                    for j, st in enumerate(states) if st.best_entry is not None}
            if sim_coupling_mode == 'blend':
                blend = {'alone': 1.0 - f_top - f_peer, 'top': f_top,
                         'peer': f_peer}
                for b in range(n_rows):
                    cpl_tasks.append(([(i, all_exprs[i][b][0])
                                       for i in range(N)], tops, horizon, blend))
                    cpl_slots.append([(i, b) for i in range(N)])
                for i in range(N):              # beam extras have no row
                    for k in range(n_rows, len(all_exprs[i])):
                        cpl_tasks.append(([(i, all_exprs[i][k][0])], tops,
                                          horizon, blend))
                        cpl_slots.append([(i, k)])
                mode.update({key: 'blend' for sl in cpl_slots for key in sl})
            else:
                peer_rows, top_sets = coupling_plan(
                    n_rows, [len(c) for c in all_exprs], f_top, f_peer, ps.rng)
                for b in peer_rows:
                    cpl_tasks.append(([(i, all_exprs[i][b][0])
                                       for i in range(N)], {}, horizon,
                                      {'peer': 1.0}))
                    cpl_slots.append([(i, b) for i in range(N)])
                    mode.update({(i, b): 'peer' for i in range(N)})
        for i in range(N):
            has_top = any(j != i for j in tops)
            for k, (tau, _ctx) in enumerate(all_exprs[i]):
                if (i, k) in mode:
                    continue
                if has_top and k in top_sets[i]:
                    cpl_tasks.append(([(i, tau)], tops, horizon, {'top': 1.0}))
                    cpl_slots.append([(i, k)])
                    mode[(i, k)] = 'top'
                else:
                    iso_tasks.append((i, tau, None, horizon))
                    iso_slots.append([(i, k)])
                    mode[(i, k)] = 'alone'
        n_tasks = len(mode)
        stage = f"2-{dc.reward_label()}"
        print(f"  [{stage}]   {n_tasks} candidates queued "
              f"({'pool' if pool else 'serial'}) ...", flush=True)

        report_every = max(1, n_tasks // 20)  # ~5% granularity
        t_score0 = time.time()
        if pool is not None:      # both submitted at once: no idle workers
            runs = [(pool.map(dc.energy_worker, iso_tasks), iso_slots, False),
                    (pool.map(dc.coupled_worker, cpl_tasks), cpl_slots, True)]
        else:
            runs = [(map(dc.energy_worker, iso_tasks), iso_slots, False),
                    (map(dc.coupled_worker, cpl_tasks), cpl_slots, True)]
        results, k = {}, 0
        for iterator, slots, many in runs:
            for slot, res in zip(slots, iterator):
                results.update(zip(slot, res if many else [res]))
                k_prev, k = k, k + len(slot)
                if k // report_every > k_prev // report_every or k == n_tasks:
                    elapsed = time.time() - t_score0
                    rate = k / elapsed if elapsed > 0 else 0.0
                    eta = (n_tasks - k) / rate if rate > 0 else 0.0
                    print(f"    [{stage}]   {k}/{n_tasks} scored "
                          f"({100.0*k/n_tasks:5.1f}%)  "
                          f"{rate:6.1f} eq/s  ETA {eta:6.1f}s", flush=True)

        # Every scored candidate as [reward, tau, consts, {mode: reward},
        # {mode: {dof: (tau, consts)}}, tuned]: the partner equations each mode
        # simulated it with, empty where a DOF was read off the record.
        scored = {}
        for key, res in results.items():
            if res is None:
                continue
            if len(res) > 5:
                scored[key] = [res[1], res[2], res[3], res[4], res[5], False]
            else:
                scored[key] = [res[1], res[2], res[3], {'alone': res[1]},
                               {'alone': {}}, False]
        print(f"  [{stage}]   scored  ({time.time()-t1:.2f}s)", flush=True)

        if sim_refine:
            # Tuning picks up where it left off: a structure the buffer already
            # holds tuned starts from those constants, not from a fresh fit.
            t2 = time.time()
            jobs, keys = [], []
            for i in range(N):
                tuned_before = {ps.dedup_key(e[1]): e for e in states[i].buffer
                                if (e[3].get('scored_with') or {}).get('tuned')}
                mine = sorted((v[0], k) for (d, k), v in scored.items() if d == i)
                for _r, k in mine[::-1][:sim_refine]:
                    r0, tau, c, _md, used, _t = scored[(i, k)]
                    twin = tuned_before.get(ps.dedup_key(tau))
                    if twin is not None and twin[0] > r0:
                        c = twin[2]
                    w = blend if mode[(i, k)] == 'blend' else {mode[(i, k)]: 1.0}
                    jobs.append((i, tau, c, w, used.get('top', {}),
                                 used.get('peer', {}), horizon, sim_refine_evals))
                    keys.append((i, k))
            outs = (pool.map(dc.refine_worker, jobs) if pool is not None
                    else map(dc.refine_worker, jobs))
            before, after, n_runs = {}, {}, 0
            for key, out in zip(keys, outs):
                v = scored[key]
                before[key[0]] = max(before.get(key[0], 0.0), v[0])
                n_runs += out[5]
                if out[1] > v[0]:
                    v[0], v[2], v[3], v[5] = out[1], out[3], out[4], True
                after[key[0]] = max(after.get(key[0], 0.0), v[0])
            print(f"  [2b-tune]  {len(jobs)} equations' amplitudes tuned on the "
                  f"simulation, {n_runs} runs ({time.time()-t2:.2f}s)", flush=True)
            for i in sorted(before):
                print(f"    DOF {i}  best {before[i]:.4f} -> {after[i]:.4f}")

        per_dof = [[] for _ in range(N)]
        for i in range(N):
            for k, (_tau, ctx) in enumerate(all_exprs[i]):
                v = scored.get((i, k))
                if v is None:
                    continue
                if reward == 'simulation':
                    ctx['scored_with'] = {'modes': v[3], 'partners': v[4],
                                          'tuned': v[5]}
                per_dof[i].append((v[0], v[1], v[2], ctx))
        if coupled and sim_coupling_mode == 'blend':
            for i in range(N):
                got = [v for (d, _k), v in scored.items() if d == i]
                if got:
                    best = max(got, key=lambda z: z[0])
                    print(f"    DOF {i}  best {best[0]:.4f} = "
                          + "  ".join(f"{_MODE_LABEL[m_]} {best[3][m_]:.4f}"
                                      for m_ in ('alone', 'top', 'peer')
                                      if m_ in best[3]))
        elif coupled:
            for i in range(N):
                parts = []
                for m_ in ('alone', 'top', 'peer'):
                    keys = [key for key, md in mode.items()
                            if key[0] == i and md == m_]
                    rs = [scored[key][0] for key in keys if key in scored]
                    parts.append(f"{_MODE_LABEL[m_]} {len(keys)}: "
                                 + (f"best {max(rs):.4f}" if rs else "-"))
                print(f"    DOF {i}  " + "  |  ".join(parts))

        # ── Step 3: promotion + buffer update + train ───────────────────────
        # Promotion and the critic are per-DOF; the policy update is not (the
        # shared policy takes one step across every DOF's buffer), so it sits
        # between the two per-DOF passes rather than inside the loop.
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
            # isolation either way.  The key is the sorted term multiset, so
            # the same terms drawn in a different order collapse too.
            seen = {ps.dedup_key(e[1]): n for n, e in enumerate(st.buffer)}
            n_added = n_tuned = 0
            for entry in promoted:
                key = ps.dedup_key(entry[1])
                if key in seen:
                    # A repeat is dropped -- unless its amplitudes were tuned
                    # on the simulation and it now beats the buffer's copy,
                    # which it then replaces (here and in the hall of fame).
                    sw = entry[3].get('scored_with') or {}
                    n_old = seen[key]
                    if not (sw.get('tuned') and entry[0] > st.buffer[n_old][0]):
                        continue
                    st.buffer[n_old] = entry
                    st.hof = [h for h in st.hof if ps.dedup_key(h[1]) != key]
                    n_tuned += 1
                else:
                    seen[key] = len(st.buffer); st.buffer.append(entry); n_added += 1
                if entry[0] > st.best_r:
                    st.best_r = entry[0]; st.best_entry = entry
                st.hof.append(entry)
                st.hof.sort(key=lambda z: z[0], reverse=True)
                st.hof[:] = st.hof[:HOF_SIZE]

            if not st.buffer:
                print(f"  DOF {i}: buffer empty, skip train"); continue

            st.R_alpha = float(np.percentile([e[0] for e in st.buffer], 10))
            c_loss = _train_critic(st, max_len, device)
            report[i] = (max(e[0] for e in batch_r), n_added, n_tuned, c_loss)

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

            best_raw, n_added, n_tuned, c_loss = report[i]
            c_str = f'  critic_loss={c_loss:.4f}' if c_loss is not None else ''
            t_str = f' ({n_tuned} replaced by tuned twins)' if n_tuned else ''
            print(f"  DOF {i}: best={best_raw:.4f}  +{n_added} "
                  f"buf={len(st.buffer)}{t_str}{c_str}")
        print(f"  [3-train]    ({time.time()-t1:.2f}s)", flush=True)

        # ── epoch report ────────────────────────────────────────────────────
        print(f"\n  [EPOCH TOTAL]  {time.time() - t0_ep:.1f}s")
        for i, st in enumerate(states):
            if st.best_entry is None:
                print(f"   DOF {i}: (no survivor yet)"); continue
            be = st.best_entry
            print(f"   DOF {i}: best_{dc.reward_label()}={st.best_r:.4f}  "
                  f"{dc.expr_to_str(be[1], be[2])}")
            d = dc.denormalize_expr(be[1], be[2], i)
            if d:
                print(f"          denorm: {d}")
            for line in _scored_with(be, i, N):
                print(line)

    # ── Final summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 64}\n=== DISCOVER done :: {system.name} ===")
    best_per_dof = []
    for i, st in enumerate(states):
        if not st.buffer:
            print(f"\nDOF {i}: no expression found"); best_per_dof.append(None); continue
        st.buffer.sort(key=lambda z: z[0], reverse=True)
        best = st.buffer[0]
        best_per_dof.append(best)
        print(f"\nDOF {i}")
        print(f"  Expr (normalised) : {dc.expr_to_str(best[1], best[2])}")
        print(f"  {dc.reward_label() + '_reward':<18}: {best[0]:.6f}")
        if system.truth_strs[i] != 'unknown':
            print(f"  Truth             : {system.truth_strs[i]}")
        d = dc.denormalize_expr(best[1], best[2], i)
        if d:
            print(f"  Expr (physical)   : {d}")
        for line in _scored_with(best, i, N, indent=2):
            print(line)
        print(f"  --- Hall of Fame (top {len(st.hof)}) ---")
        for rank, hof in enumerate(st.hof, 1):
            dh = dc.denormalize_expr(hof[1], hof[2], i)
            print(f"    #{rank}: {dc.reward_label()}={hof[0]:.4f}  "
                  f"{dh if dh else dc.expr_to_str(hof[1], hof[2])}")

    # The best equations integrated together -- what the plot scripts do.
    if reward == 'simulation' and sum(b is not None for b in best_per_dof) > 1:
        together = dc.coupled_simulation_rewards(
            [(b[1], b[2]) if b is not None else None for b in best_per_dof],
            [i for i, b in enumerate(best_per_dof) if b is not None],
            horizon=dc.get_traj_horizon(n_epochs - 1, n_epochs, system.t_end))
        print(f"\n{'=' * 64}\nBest equations simulated together "
              f"(reward as trained, partners simulated, not measured):")
        for i, r in together.items():
            print(f"  DOF {i}: {r:.6f}   (its training score "
                  f"{best_per_dof[i][0]:.6f})")

    # Paste-ready block for discover_score.py / discover_plot_*.py
    print(f"\n{'=' * 64}\nDISCOVERED_EXPRS = [")
    for i, best in enumerate(best_per_dof):
        if best is None:
            print(f'    "",   # DOF {i}: nothing found')
            continue
        phys = dc.denormalize_expr(best[1], best[2], i) or dc.expr_to_str(best[1], best[2])
        print(f'    "{phys}",')
    print("]\n", flush=True)

    if pool is not None:
        dc.close_pool(pool, saved_env)
    return best_per_dof
