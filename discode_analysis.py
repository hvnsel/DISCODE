"""
discode_analysis.py
===================

Shared machinery for looking at a discovered equation after the search: turning
a printed expression back into something evaluable, integrating it forward, and
computing the work-energy residual it leaves behind.

One implementation, used by :mod:`discode_score`, :mod:`discode_plot_sim` and
:mod:`discode_plot_exp`.  This matters more than it looks: the residual
definition here has to agree with :func:`discode_core.energy_reward` or the
number you read after training will not be the number you trained on.

Expression convention
---------------------
Expressions are the PHYSICAL (denormalised) strings the trainer prints, written
in terms of ``var_names``::

    var_names = ['x', 'y']    ->  x, xdot, y, ydot
    var_names = ['q1', 'q2']  ->  q1, q1dot, q2, q2dot

A leading ``<var>ddot =`` is optional and stripped.  Accepted syntax:
``+ - * / ** ^``, ``sin cos exp sqrt abs Abs sign sgn``, and ``|u|`` for
absolute value.

Reward convention
-----------------
:func:`trainer_reward` reproduces the training reward exactly:

    mean over time  ->  mean over DOFs  ->  mean over trials  ->  1/(1+res)

Note this is NOT ``mean_d [ 1/(1+res_d) ]``.  The trainer's form is always the
smaller of the two and is dominated by the worst DOF; :func:`per_dof_rewards`
gives the per-DOF breakdown for diagnosis, but the headline number should be
:func:`trainer_reward`.

No torch anywhere in this module — scoring and plotting run without it.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.integrate import solve_ivp

_NS_BASE = {'__builtins__': {}, 'np': np, 'pi': np.pi}

_FN_MAP = (('sgn', 'np.sign'), ('sign', 'np.sign'), ('Abs', 'np.abs'),
           ('abs', 'np.abs'), ('sin', 'np.sin'), ('cos', 'np.cos'),
           ('exp', 'np.exp'), ('sqrt', 'np.sqrt'), ('tanh', 'np.tanh'))


# ── Expression handling ─────────────────────────────────────────────────────
def clean_expr(expr: str) -> str:
    """Normalise a printed expression into a numpy-evaluable string."""
    if '=' in expr:
        expr = expr.split('=', 1)[1]
    expr = expr.strip()
    expr = re.sub(r'\^', '**', expr)
    for fn, rep in _FN_MAP:
        expr = re.sub(rf'(?<!np\.)\b{fn}\s*\(', rep + '(', expr)
    expr = re.sub(r'\|([^|]+)\|', r'np.abs(\1)', expr)
    return expr


def clean_exprs(exprs, var_names=None):
    """Clean a list of expressions, checking the count against ``var_names``."""
    if var_names is not None and len(exprs) != len(var_names):
        raise ValueError(f"{len(exprs)} expressions but {len(var_names)} "
                         f"var_names — one expression per DOF is required")
    return [clean_expr(e) for e in exprs]


def eval_accel(clean, var_names, disp, vel):
    """Evaluate one cleaned expression on measured states.

    ``disp`` / ``vel`` are (n_pts, n_dof).  Returns ``(a, n_bad)`` where ``a``
    is (n_pts,) with non-finite samples zeroed and ``n_bad`` counts them — a
    nonzero count means the expression has a singularity on this data.
    """
    ns = dict(_NS_BASE)
    for d, nm in enumerate(var_names):
        ns[nm] = disp[:, d]
        ns[nm + 'dot'] = vel[:, d]
    try:
        a = np.asarray(eval(clean, ns), dtype=float)          # noqa: S307
    except Exception as exc:                                   # noqa: BLE001
        print(f"    [warn] evaluation failed: {exc}")
        return np.zeros(disp.shape[0]), disp.shape[0]
    if a.ndim == 0:
        a = np.full(disp.shape[0], float(a))
    n_bad = int((~np.isfinite(a)).sum())
    return np.where(np.isfinite(a), a, 0.0), n_bad


def predict_accel(cleans, var_names, disp, vel):
    """Evaluate every DOF's expression on measured states.

    Returns ``(acc_list, n_bad_total)`` with one (n_pts,) array per DOF.
    """
    accs, bad = [], 0
    for c in cleans:
        a, nb = eval_accel(c, var_names, disp, vel)
        accs.append(a)
        bad += nb
    return accs, bad


# ── Forward simulation ──────────────────────────────────────────────────────
def build_rhs(cleans, var_names):
    """ODE right-hand side for the full N-DOF discovered system.

    State layout ``[q1, q1dot, q2, q2dot, ...]`` — the same as the grammar's.
    """
    def rhs(_t, state):
        ns = dict(_NS_BASE)
        for d, name in enumerate(var_names):
            ns[name]         = state[2 * d]
            ns[name + 'dot'] = state[2 * d + 1]
        derivs = []
        for d, expr in enumerate(cleans):
            qdot = state[2 * d + 1]
            try:
                qddot = float(eval(expr, ns))                 # noqa: S307
            except Exception:                                  # noqa: BLE001
                qddot = 0.0
            if not np.isfinite(qddot):
                qddot = 0.0
            derivs.extend([qdot, qddot])
        return derivs

    return rhs


def forward_simulate(cleans, var_names, t_eval, ic, solver='LSODA',
                     max_step=None, blow_up_limit=1e4, progress=True):
    """Integrate the discovered system from ``ic`` over ``t_eval``.

    A discovered equation is under no obligation to be stable, so integration
    is guarded: a terminal event stops the solver as soon as any state
    component exceeds ``blow_up_limit``, and the (possibly partial) solution is
    returned rather than raised.  Check ``sol.t[-1]`` against ``t_eval[-1]``
    before trusting a comparison.
    """
    rhs  = build_rhs(cleans, var_names)
    dt   = float(np.mean(np.diff(t_eval)))
    mxst = max_step if max_step is not None else dt * 20

    events = []
    if blow_up_limit is not None:
        def _blow_up(_t, state):
            return blow_up_limit - np.max(np.abs(state))
        _blow_up.terminal  = True
        _blow_up.direction = -1
        events.append(_blow_up)

    t0, t1 = float(t_eval[0]), float(t_eval[-1])
    integrand = rhs
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=t1 - t0, desc="Simulating", unit="s",
                        bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f} s"
                                   "  [{elapsed}<{remaining}, {rate_fmt}]")
            last_t = [t0]

            def integrand(t, state):                           # noqa: F811
                inc = t - last_t[0]
                if inc > 0:
                    pbar.update(inc)
                    last_t[0] = t
                return rhs(t, state)
        except ImportError:
            pbar = None

    try:
        sol = solve_ivp(integrand, (t0, t1), list(ic), t_eval=t_eval,
                        method=solver, rtol=1e-6, atol=1e-9,
                        max_step=mxst, dense_output=False, events=events)
    except ValueError:
        # A rapidly diverging RHS can jump past blow_up_limit within a single
        # step, so the event's root search brackets two same-sign values and
        # brentq raises.  Re-integrate without the event and truncate below.
        sol = solve_ivp(integrand, (t0, t1), list(ic), t_eval=t_eval,
                        method=solver, rtol=1e-6, atol=1e-9,
                        max_step=mxst, dense_output=False)
    if pbar is not None:
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    blow_t = None
    if blow_up_limit is not None and getattr(sol, 't_events', None) \
            and sol.t_events[0].size > 0:
        blow_t = float(sol.t_events[0][0])
    elif blow_up_limit is not None and sol.y.size:
        # Truncate at the first non-finite or over-limit sample (event fallback).
        mag = np.max(np.abs(sol.y), axis=0)
        over = ~np.isfinite(mag) | (mag > blow_up_limit)
        if over.any():
            cut = int(np.argmax(over))
            blow_t = float(sol.t[cut]) if cut < sol.t.size else float(sol.t[-1])
            sol.t = sol.t[:cut]
            sol.y = sol.y[:, :cut]

    blew_up = blow_t is not None
    if blew_up:
        print(f"[sim] BLOW-UP at t={blow_t:.4f}s  "
              f"(|state| > {blow_up_limit:.3g}) — integration stopped early.",
              flush=True)
    elif not sol.success:
        print(f"[sim] WARNING: {sol.message}", flush=True)
    if sol.t.size > 1:
        pct = 100.0 * (sol.t[-1] - sol.t[0]) / (t_eval[-1] - t_eval[0])
        status = "BLOW-UP" if blew_up else ("OK" if sol.success else "partial")
        print(f"[sim] {status}: t=[{sol.t[0]:.3f}, {sol.t[-1]:.3f}]s  "
              f"({pct:.0f}%)", flush=True)
    return sol


# ── Work-energy residual ────────────────────────────────────────────────────
def cumtrapz(y, t):
    """Cumulative trapezoidal integral with a leading 0 (same length as y)."""
    out = np.zeros_like(np.asarray(y, dtype=float))
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def energy_residual(t, vel, acc_pred, normalize=True):
    """Per-DOF work-energy residual curves for one trial.

    ``vel`` is (n_pts, n_dof); ``acc_pred`` is a list of n_dof arrays.  Returns
    ``(per_cum_power, per_dke, per_residual)``, each a list of n_dof (n_pts,)
    arrays.  Every DOF is normalised by its OWN kinetic-energy scale so a
    low-velocity DOF is not swamped by a high-velocity one — the same
    normalisation the reward uses.
    """
    _n_pts, N = vel.shape
    per_cum_power, per_dke, per_residual = [], [], []
    for d in range(N):
        cp  = cumtrapz(vel[:, d] * acc_pred[d], t)
        dke = 0.5 * (vel[:, d] ** 2 - vel[0, d] ** 2)
        res = np.abs(cp - dke)
        if normalize:
            scale = float(np.std(dke))
            if scale < 1e-12:
                scale = float(np.mean(np.abs(dke))) + 1e-12
            res = res / scale
        per_cum_power.append(cp)
        per_dke.append(dke)
        per_residual.append(res)
    return per_cum_power, per_dke, per_residual


def mean_residuals(t, vel, acc_pred, normalize=True):
    """Mean-over-time residual per DOF for one trial: a length-N list."""
    _cp, _dke, per_res = energy_residual(t, vel, acc_pred, normalize)
    return [float(np.mean(r)) for r in per_res]


def trainer_reward(res_per_trial):
    """The training reward from a list (per trial) of per-DOF mean residuals.

    ``1 / (1 + mean_trials mean_dofs res)`` — reward of the mean residual, not
    the mean of per-DOF rewards.
    """
    per_trial = [float(np.mean(r)) for r in res_per_trial]
    return 1.0 / (1.0 + float(np.mean(per_trial)))


def per_dof_rewards(res_per_trial):
    """Per-DOF rewards ``1/(1+res_d)``, averaged over trials.  Diagnostic only —
    these are each >= the headline :func:`trainer_reward`."""
    arr = np.asarray(res_per_trial, dtype=float)     # (n_trials, n_dof)
    return [1.0 / (1.0 + float(m)) for m in arr.mean(axis=0)]


def plot_discovered(system, exprs, trial=0, t_end=None, solver='LSODA',
                    max_step=None, blow_up_limit=1e4, plot_energy=True,
                    energy_normalize=True, out_prefix=None, show=True,
                    ref_label='Reference'):
    """Integrate the discovered system from one trial's IC and compare it with
    that trial's record.

    Produces two figures: a 3 x N grid of displacement / velocity /
    acceleration (reference vs simulated) and, if ``plot_energy``, a 2 x N grid
    of the work-energy balance (measured dKE vs predicted work in, and the
    residual).

    The two figures answer different questions and can disagree.  The forward
    simulation compounds error over time and so is unforgiving of anything that
    shifts the phase; the energy residual is evaluated on the MEASURED states
    and is exactly what the search optimised.  A good residual with a diverging
    simulation usually means a small frequency error, not a wrong structure.

    Returns a dict with the solution, the NRMSEs and the energy rewards.
    """
    import matplotlib.pyplot as plt

    N = system.n_dof
    var_names = system.var_names
    cleans = clean_exprs(exprs, var_names)
    if len(cleans) != N or len(var_names) != N:
        raise ValueError(
            f"System has {N} DOF but got {len(cleans)} expression(s) and "
            f"{len(var_names)} var_name(s) — one per DOF is required.  Either "
            f"provide an expression and var_name for every channel, or set "
            f"DOF_INDEX to analyse a single channel.")
    for d, (name, expr) in enumerate(zip(var_names, cleans)):
        print(f"[expr DOF {d}]  {name}ddot = {expr}", flush=True)

    t    = np.asarray(system.time, dtype=float)
    disp = system.disp[:, :, trial]
    vel  = system.vel[:, :, trial]
    acc  = system.acc[:, :, trial]

    if t_end is not None:
        mask = t <= float(t_end)
        t, disp, vel, acc = t[mask], disp[mask], vel[mask], acc[mask]

    ic = []
    for d in range(N):
        ic.extend([float(disp[0, d]), float(vel[0, d])])
    print(f"[IC]  {ic}", flush=True)

    sol = forward_simulate(cleans, var_names, t, ic, solver=solver,
                           max_step=max_step, blow_up_limit=blow_up_limit)
    has_sim = sol.t.size > 1
    sim_lbl = ('Simulated' if sol.success else
               f'Simulated (partial '
               f'{100*(sol.t[-1]-sol.t[0])/(t[-1]-t[0]):.0f}%)')

    sim_disp = [sol.y[2 * d]     for d in range(N)] if has_sim else [None] * N
    sim_vel  = [sol.y[2 * d + 1] for d in range(N)] if has_sim else [None] * N
    sim_acc  = ([np.gradient(sol.y[2 * d + 1], sol.t) for d in range(N)]
                if has_sim else [None] * N)

    # ── NRMSE ───────────────────────────────────────────────────────────────
    nrmse = {}
    if has_sim and sol.success:
        for d in range(N):
            name = var_names[d]
            for lbl, sv, ev_raw in [
                (f'{name}',      sim_disp[d], disp[:, d]),
                (f'{name}dot',   sim_vel[d],  vel[:, d]),
                (f'{name}ddot',  sim_acc[d],  acc[:, d]),
            ]:
                ev = np.interp(sol.t, t, ev_raw)
                s  = np.std(ev)
                val = float(np.sqrt(np.mean((sv - ev) ** 2)) / (s if s > 0 else 1.0))
                nrmse[lbl] = val
                print(f"  DOF {d} NRMSE {lbl:<10}: {val:.4f}", flush=True)

    # ── Trajectory figure ───────────────────────────────────────────────────
    prefix = out_prefix or system.name
    rows = 3
    fig, axes = plt.subplots(rows, N, figsize=(5 * N, 3.5 * rows),
                             sharex='col', squeeze=False)
    fig.suptitle(f"{system.name}  —  forward simulation  (trial {trial})",
                 fontsize=10)

    row_labels = ['{n}  (displacement)', '{n}dot  (velocity)',
                  '{n}ddot  (acceleration)']
    ref_arrs = [disp, vel, acc]
    sim_arrs = [sim_disp, sim_vel, sim_acc]

    for d in range(N):
        name  = var_names[d]
        short = str(exprs[d]).split('=')[-1].strip()
        short = short[:70] + ('...' if len(short) > 70 else '')
        axes[0][d].set_title(f"DOF {d} ({name})\n{name}ddot = {short}", fontsize=7)

        for row in range(rows):
            ax = axes[row][d]
            ax.plot(t, ref_arrs[row][:, d], color='#1f77b4', lw=0.9, alpha=0.75,
                    label=ref_label if d == 0 else '_')
            if sim_arrs[row][d] is not None:
                ax.plot(sol.t, sim_arrs[row][d], color='#d62728', lw=1.2, ls='--',
                        label=sim_lbl if d == 0 else '_')
            ax.set_ylabel(row_labels[row].format(n=name), fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[rows - 1][d].set_xlabel('Time  (s)', fontsize=8)

    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    out = f"{prefix}_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved  {out}", flush=True)

    # ── Energy figure ───────────────────────────────────────────────────────
    rewards = None
    if plot_energy:
        acc_pred, n_bad = predict_accel(cleans, var_names, disp, vel)
        if n_bad:
            print(f"[energy] {n_bad} non-finite predicted samples were zeroed",
                  flush=True)
        per_cp, per_dke, per_res = energy_residual(t, vel, acc_pred,
                                                   normalize=energy_normalize)
        res_means = [float(np.mean(r)) for r in per_res]
        rewards = [1.0 / (1.0 + m) for m in res_means]
        r_trial = 1.0 / (1.0 + float(np.mean(res_means)))
        for d, name in enumerate(var_names):
            print(f"[energy] DOF {d} ({name})  mean residual = {res_means[d]:.4e}"
                  f"   r = {rewards[d]:.4f}", flush=True)
        print(f"[energy] trial reward (trainer convention) = {r_trial:.4f}",
              flush=True)

        efig, eaxes = plt.subplots(2, N, figsize=(5 * N, 6),
                                   sharex='col', squeeze=False)
        efig.suptitle(f"{system.name}  —  per-DOF work-energy residual  "
                      f"(trial {trial})", fontsize=10)

        for d in range(N):
            name = var_names[d]
            ax0, ax1 = eaxes[0, d], eaxes[1, d]

            ax0.plot(t, per_dke[d], color='#1f77b4', lw=1.1,
                     label=r'measured $\Delta$KE')
            ax0.plot(t, per_cp[d], color='#d62728', lw=1.1, ls='--',
                     label='predicted work-in')
            ax0.set_title(f"DOF {d} ({name})", fontsize=9)
            ax0.set_ylabel('energy', fontsize=8)
            ax0.grid(True, alpha=0.3)
            ax0.legend(fontsize=7)

            ax1.plot(t, per_res[d], color='#2ca02c', lw=1.1)
            norm_lbl = '  (norm.)' if energy_normalize else ''
            ax1.set_ylabel(f'|residual|{norm_lbl}', fontsize=8)
            ax1.set_xlabel('Time  (s)', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"mean = {res_means[d]:.3e}   r = {rewards[d]:.4f}",
                          fontsize=8)

        efig.tight_layout()
        out_e = f"{prefix}_energy_residual.png"
        efig.savefig(out_e, dpi=150, bbox_inches='tight')
        print(f"Saved  {out_e}", flush=True)

    if show:
        plt.show()
    else:
        plt.close('all')

    return {'sol': sol, 'nrmse': nrmse, 'per_dof_rewards': rewards}


def score_system(system, exprs, normalize=True, trials=None, verbose=True):
    """Score a set of expressions against a ``SystemData`` with the training
    reward, alongside the ceiling set by the measured acceleration.

    Returns ``(r_pred, r_ceiling, rows)`` where ``rows`` is one
    ``(trial, res_pred_per_dof, res_ceil_per_dof)`` tuple per trial.
    """
    t = np.asarray(system.time, dtype=float)
    keep = (list(range(system.n_trials)) if trials is None else list(trials))
    cleans = clean_exprs(exprs, system.var_names)

    rows, res_pred, res_ceil, bad_total = [], [], [], 0
    for tr in keep:
        disp = system.disp[:, :, tr]
        vel  = system.vel[:, :, tr]
        acc  = system.acc[:, :, tr]

        a_pred, nb = predict_accel(cleans, system.var_names, disp, vel)
        bad_total += nb
        rp = mean_residuals(t, vel, a_pred, normalize)
        rc = mean_residuals(t, vel, [acc[:, d] for d in range(system.n_dof)],
                            normalize)
        rows.append((tr, rp, rc))
        res_pred.append(rp)
        res_ceil.append(rc)

    r_pred = trainer_reward(res_pred)
    r_ceil = trainer_reward(res_ceil)

    if verbose and bad_total:
        print(f"  [warn] {bad_total} non-finite predicted samples were zeroed "
              f"— the expression has a singularity on this data")
    return r_pred, r_ceil, rows
