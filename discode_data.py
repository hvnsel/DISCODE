"""
discode_data.py
===============

Everything that produces a :class:`SystemData` for DISCODE, and nothing else.

A ``SystemData`` is the single container the whole pipeline runs on.  It holds
three (m, n_dof, p_trials) arrays — displacement, velocity, acceleration — on a
shared (m,) time grid, plus the variable names used by the grammar.  There are
exactly two ways to make one:

  * :func:`load_mat_data`  — read an experimental MATLAB record from disk.
  * :func:`simulate_truth` — integrate a known set of accelerations from given
    initial conditions.

Both are N-DOF; a single-DOF system is just ``n_dof == 1``.  :func:`select_dof`
slices one channel out of a multi-DOF record so the SDOF drivers can reuse an
experimental file.

:func:`generate_dataset` converts a ``SystemData`` into the normalised stats and
raw trajectories that :mod:`discode_core` is configured with.

Two diagnostics live here because they are properties of the *data*, not of the
search:

  * :func:`identity_ceiling` — the reward the MEASURED acceleration itself
    scores.  This is the best value ANY expression can reach on this data.  On
    simulated data it must be ~1.0; if it is not, the trapezoid rule is
    under-resolving the ``v*a`` integrand and every reward is capped below the
    truth.  On experimental data it is typically well below 1.0, and a
    discovered expression scoring *above* it is a warning about the data, not a
    success.
  * :func:`print_truth_rewards` — the reward of a known truth structure with its
    constants fitted by the standard optimiser.  This is the number training
    should converge to on a simulated system.

``torch`` is imported lazily inside :func:`generate_dataset` so that the
plotting and scoring scripts can use this module without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ── Container ───────────────────────────────────────────────────────────────
class SystemData:
    """Displacement / velocity / acceleration on a common time grid.

    Parameters
    ----------
    acc, vel, disp : (m, n_dof, p_trials) arrays
    time           : (m,) array
    var_names      : one display name per DOF, e.g. ``['x', 'y']``.  The grammar
                     variable for DOF ``d`` is ``var_names[d]`` and its velocity
                     is ``var_names[d] + 'dot'``.
    truth_strs     : optional human-readable ground truth, one string per DOF.
    truth_taus     : optional ground-truth token sequences, one per DOF (or
                     ``None`` for a DOF whose truth is not representable in the
                     grammar).  Consumed by :func:`print_truth_rewards`.
    """

    def __init__(self, name, acc, vel, disp, time, var_names=None,
                 truth_strs=None, truth_taus=None):
        self.name = name
        self.acc  = acc     # (m, n_dof, p_trials)
        self.vel  = vel     # (m, n_dof, p_trials)
        self.disp = disp    # (m, n_dof, p_trials)
        self.time = time    # (m,)

        m, n_dof, p = acc.shape
        self.n_dof     = n_dof
        self.n_trials  = p
        self.t_end     = float(time[-1] - time[0])
        self.var_names = var_names or [f'q{d + 1}' for d in range(n_dof)]

        # No ground truth for experimental data.
        self.truth_strs = truth_strs or ['unknown'] * n_dof
        self.truth_taus = truth_taus or [None] * n_dof

    def __repr__(self):
        fs = (len(self.time) - 1) / self.t_end if self.t_end > 0 else 0.0
        return (f"<SystemData {self.name}: {self.n_dof} DOF, "
                f"{self.n_trials} trials, {len(self.time)} pts, "
                f"{self.t_end:.3f} s, {fs:.0f} Hz>")


# ── Experimental loading ────────────────────────────────────────────────────
def load_mat_data(filepath, trim_timesteps_front=1000, trim_timesteps_back=70000,
                  desired_timesteps=600, var_names=None, plot_data=False):
    """
    Load Acc, Vel, Disp, Time from a MATLAB .mat file.

    Supports both:
      - MATLAB v7.3 (HDF5-based, opened with h5py).  MATLAB stores arrays
        column-major so a MATLAB (m x n x p) array appears in h5py as
        (p, n, m); we transpose to recover (m, n_dof, p_trials).
      - MATLAB v5/v6 (loaded with scipy.io.loadmat).  scipy already returns
        arrays in the original (m, n_dof [, p_trials]) shape.

    The first ``trim_timesteps_front`` and last ``trim_timesteps_back`` rows are
    dropped to keep only the free response, then the record is downsampled to
    roughly ``desired_timesteps`` samples.
    """
    import h5py
    import scipy.io

    # ── Try HDF5 (v7.3) first, fall back to scipy for v5/v6 ─────────────────
    try:
        with h5py.File(filepath, 'r') as f:
            if 'Acc' in f:
                grp = f
            else:
                grp = next(v for v in f.values() if hasattr(v, 'keys') and 'Acc' in v)
            acc  = np.array(grp['Acc']).T
            vel  = np.array(grp['Vel']).T
            disp = np.array(grp['Disp']).T
            time = np.array(grp['Time']).flatten()
    except OSError:
        print("[load_mat_data] h5py failed — trying scipy.io.loadmat (v5/v6 format)",
              flush=True)
        mat = scipy.io.loadmat(filepath)
        # Find variables case-insensitively.
        keys = {k.lower(): k for k in mat if not k.startswith('_')}
        print(f"[load_mat_data] top-level variables: {list(keys.values())}", flush=True)

        def _extract_field(source, name):
            """Pull ``name`` out of a dict-like, structured numpy array, or
            squeeze-wrapped MATLAB struct (handles the common [0,0] wrapper)."""
            nl = name.lower()
            # dict / h5py-group style
            if hasattr(source, 'keys'):
                for k in source.keys():
                    if k.lower() == nl:
                        return np.array(source[k], dtype=float)
            # numpy structured array / void (MATLAB struct loaded by scipy)
            if hasattr(source, 'dtype') and source.dtype.names:
                for fn in source.dtype.names:
                    if fn.lower() == nl:
                        val = source[fn]
                        # unwrap (1,1) cell wrappers common in scipy structs
                        while hasattr(val, 'shape') and val.shape in ((1, 1), (1,)):
                            val = val.flat[0]
                        return np.array(val, dtype=float)
            raise KeyError(name)

        def _get(name):
            # 1. Try direct top-level key.
            top_key = keys.get(name.lower())
            if top_key is not None:
                return _extract_field(mat, top_key)
            # 2. Search inside every top-level struct variable.
            for raw_key in keys.values():
                var = mat[raw_key]
                # unwrap (1,1) outer wrapper
                inner = var.flat[0] if (hasattr(var, 'shape') and
                                        var.shape in ((1, 1), (1,))) else var
                try:
                    return _extract_field(inner, name)
                except (KeyError, AttributeError, TypeError):
                    continue
            raise KeyError(
                f"Field '{name}' not found at top level or inside any struct. "
                f"Top-level variables: {list(keys.values())}"
            )

        acc  = _get('Acc')
        vel  = _get('Vel')
        disp = _get('Disp')
        time = _get('Time').flatten()
        # Ensure shape is (m, n_dof, p_trials).
        if acc.ndim == 2:           # (m, n_dof) — single trial
            acc  = acc[:, :, np.newaxis]
            vel  = vel[:, :, np.newaxis]
            disp = disp[:, :, np.newaxis]
        elif acc.ndim == 3 and acc.shape[0] < acc.shape[2]:
            # Might be (n_dof, m, p) — transpose so time is axis 0
            acc  = acc.transpose(1, 0, 2)
            vel  = vel.transpose(1, 0, 2)
            disp = disp.transpose(1, 0, 2)
        print(f"[load_mat_data] loaded via scipy: shape acc={acc.shape}", flush=True)

    # ``-0`` is not "keep everything" in Python — acc[600:-0] is acc[600:0],
    # which is empty.  A pre-trimmed record therefore needs end=None.
    n_raw = acc.shape[0]
    end = -trim_timesteps_back if trim_timesteps_back else None
    if trim_timesteps_front + trim_timesteps_back >= n_raw:
        raise ValueError(
            f"trim removes everything: the record has {n_raw} samples but "
            f"trim_timesteps_front={trim_timesteps_front} + "
            f"trim_timesteps_back={trim_timesteps_back} = "
            f"{trim_timesteps_front + trim_timesteps_back}.\n"
            f"If this file was already trimmed and impact-aligned (e.g. by "
            f"data_combine.m), use trim_timesteps_front=0, "
            f"trim_timesteps_back=0.")

    acc  = acc[trim_timesteps_front: end]
    vel  = vel[trim_timesteps_front: end]
    disp = disp[trim_timesteps_front: end]
    time = time[trim_timesteps_front: end]

    down_sample_factor = max(1, acc.shape[0] // desired_timesteps)
    acc  = acc[::down_sample_factor, :, :]
    vel  = vel[::down_sample_factor, :, :]
    disp = disp[::down_sample_factor, :, :]
    time = time[::down_sample_factor]

    name = os.path.splitext(os.path.basename(filepath))[0]

    if plot_data:
        plot_system_data(SystemData(name, acc, vel, disp, time,
                                    var_names=var_names))

    return SystemData(name, acc, vel, disp, time, var_names=var_names)


def select_dof(system, dof_index, var_name='x'):
    """Return a new 1-DOF ``SystemData`` keeping only channel ``dof_index``."""
    if not (0 <= dof_index < system.n_dof):
        raise ValueError(f"dof_index={dof_index} out of range for "
                         f"{system.n_dof}-DOF data")
    sl = slice(dof_index, dof_index + 1)
    return SystemData(
        name=f"{system.name}_dof{dof_index + 1}",
        acc=system.acc[:, sl, :],
        vel=system.vel[:, sl, :],
        disp=system.disp[:, sl, :],
        time=system.time,
        var_names=[var_name],
        truth_strs=[system.truth_strs[dof_index]],
        truth_taus=[system.truth_taus[dof_index]],
    )


# ── Simulated truth ─────────────────────────────────────────────────────────
def simulate_truth(name, accel_fns, ics, t_eval, var_names=None,
                   truth_strs=None, truth_taus=None,
                   rtol=1e-10, atol=1e-12, method='RK45', verbose=True):
    """Integrate a known system and package the result as a ``SystemData``.

    Parameters
    ----------
    accel_fns : list of N callables ``a_d(state) -> float``, where
                ``state = [q_1, qd_1, ..., q_N, qd_N]`` — the same layout the
                grammar uses (``x1 = q_1``, ``x2 = qd_1``, ``x3 = q_2``, ...).
    ics       : (p_trials, 2N) array of initial states in that layout.
    t_eval    : (m,) time grid; the integration runs on ``[t[0], t[-1]]``.

    The resulting record satisfies ``a = dv/dt`` and ``v = dq/dt`` to solver
    tolerance, so the work-energy identity holds in the data by construction
    and the reward ceiling is ~1.0 (checked by :func:`identity_ceiling`).
    """
    ics = np.atleast_2d(np.asarray(ics, dtype=float))
    t   = np.asarray(t_eval, dtype=float)
    t   = t - t[0]                                  # sim clock starts at 0
    N   = len(accel_fns)
    if ics.shape[1] != 2 * N:
        raise ValueError(f"ics must have {2*N} columns for {N} DOF, "
                         f"got {ics.shape[1]}")

    def rhs(_t, s):
        out = np.empty_like(s)
        for d in range(N):
            out[2 * d]     = s[2 * d + 1]
            out[2 * d + 1] = accel_fns[d](s)
        return out

    m, p = len(t), ics.shape[0]
    acc  = np.empty((m, N, p))
    vel  = np.empty_like(acc)
    disp = np.empty_like(acc)

    for j in range(p):
        sol = solve_ivp(rhs, (0.0, t[-1]), ics[j], t_eval=t,
                        rtol=rtol, atol=atol, method=method)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed on trial {j}: {sol.message}")
        st = sol.y                                   # (2N, m)
        for d in range(N):
            disp[:, d, j] = st[2 * d]
            vel[:,  d, j] = st[2 * d + 1]
            acc[:,  d, j] = np.array([accel_fns[d](st[:, k])
                                      for k in range(st.shape[1])])
        if verbose:
            print(f"[simulate] trial {j}: ic={ics[j]}  "
                  f"max|q|={np.abs(st[0::2]).max():.4g}", flush=True)

    sim = SystemData(name, acc, vel, disp, t, var_names=var_names,
                     truth_strs=truth_strs, truth_taus=truth_taus)

    identity_ceiling(sim, verbose=verbose, warn=True)
    return sim


@dataclass
class TruthSystem:
    """Declarative spec for a simulated system, consumed by
    :func:`build_truth_system`.

    ``accel_fns[d](state)`` returns the acceleration of DOF ``d`` given
    ``state = [q_1, qd_1, ..., q_N, qd_N]``.

    ``ic_scale`` is a length-2N vector of per-component initial-condition
    amplitudes; trials are drawn uniformly from ``[-ic_scale, +ic_scale]`` with
    a fixed seed, so a given spec always produces the same dataset.  Set it to
    the amplitude the system's nonlinearity actually lives at — too small and a
    cubic term is unidentifiable, too large and the response leaves the regime
    of interest.

    ``n_pts`` must resolve the ``v*a`` integrand, which has content up to twice
    the highest natural frequency: aim for >~40 points per cycle of the FASTEST
    mode.  :func:`identity_ceiling` checks this and warns.
    """
    name:       str
    accel_fns:  list
    truth_strs: list
    truth_taus: list = None
    var_names:  list = None
    t_end:      float = 20.0
    n_pts:      int = 2000
    n_traj:     int = 4
    ic_scale:   list = None
    seed:       int = 0

    @property
    def n_dof(self):
        return len(self.accel_fns)


def build_truth_system(spec, n_traj=None, t_end=None, n_pts=None, seed=None,
                       verbose=True):
    """Draw initial conditions for ``spec`` and integrate it into a
    ``SystemData``.  Any of ``n_traj`` / ``t_end`` / ``n_pts`` / ``seed``
    overrides the value carried by the spec."""
    n_traj = spec.n_traj if n_traj is None else int(n_traj)
    t_end  = spec.t_end  if t_end  is None else float(t_end)
    n_pts  = spec.n_pts  if n_pts  is None else int(n_pts)
    seed   = spec.seed   if seed   is None else int(seed)

    N = spec.n_dof
    scale = (np.ones(2 * N) if spec.ic_scale is None
             else np.asarray(spec.ic_scale, dtype=float))
    if scale.shape != (2 * N,):
        raise ValueError(f"ic_scale must have {2*N} entries for {N} DOF")

    rng = np.random.default_rng(seed)
    ics = rng.uniform(-1.0, 1.0, size=(n_traj, 2 * N)) * scale
    t   = np.linspace(0.0, t_end, n_pts)

    if verbose:
        print(f"\n[simulate] {spec.name}: {N} DOF, {n_traj} trials, "
              f"{n_pts} pts over {t_end:g} s ({(n_pts-1)/t_end:.0f} Hz)")
        for d, s in enumerate(spec.truth_strs):
            print(f"    truth DOF {d}: {s}")

    return simulate_truth(spec.name, spec.accel_fns, ics, t,
                          var_names=spec.var_names,
                          truth_strs=spec.truth_strs,
                          truth_taus=spec.truth_taus,
                          verbose=verbose)


# ── Diagnostics ─────────────────────────────────────────────────────────────
def _cumtrapz(y, t):
    out = np.zeros_like(np.asarray(y, dtype=float))
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def identity_ceiling(system, verbose=True, warn=False):
    """Reward achieved by the MEASURED acceleration — the ceiling for any
    expression on this data.

    Returns ``(worst, per_trial)`` where ``per_trial[j][d]`` is the per-DOF
    reward for trial ``j``.  Set ``warn=True`` to print the under-resolution
    warning when the ceiling falls below 0.99 (only meaningful for simulated
    data, where the states are exact by construction and a low ceiling can
    ONLY be trapezoid under-resolution of the ``v*a`` integrand).
    """
    t = np.asarray(system.time, dtype=float)
    worst, per_trial = 1.0, []
    for j in range(system.n_trials):
        line = []
        for d in range(system.n_dof):
            v   = system.vel[:, d, j]
            lhs = _cumtrapz(v * system.acc[:, d, j], t)
            rhs = 0.5 * (v ** 2 - v[0] ** 2)
            sc  = np.std(rhs) + 1e-30
            r   = 1.0 / (1.0 + np.mean(np.abs(lhs - rhs)) / sc)
            line.append(r)
            worst = min(worst, r)
        per_trial.append(line)

    if verbose:
        print(f"\n[ceiling] r( v, a_measured ) per trial — the best any "
              f"expression can score:")
        for j, line in enumerate(per_trial):
            cells = '  '.join(f"DOF{d}={r:.4f}" for d, r in enumerate(line))
            print(f"    trial {j}: {cells}", flush=True)

    if warn and worst < 0.99:
        fs_eff = (len(t) - 1) / (t[-1] - t[0])
        print(f"\n[ceiling] *** WARNING: identity ceiling {worst:.4f} < 0.99 at "
              f"{fs_eff:.0f} Hz effective sampling. ***\n"
              f"[ceiling] On simulated data this is trapezoid under-resolution, "
              f"not physics —\n"
              f"[ceiling] raise n_pts until the ceiling clears 0.99.  The v*a "
              f"integrand has\n"
              f"[ceiling] content up to 2x the highest natural frequency, so it "
              f"needs >~40\n"
              f"[ceiling] points per cycle of the FASTEST mode.  Rewards computed "
              f"on this\n"
              f"[ceiling] dataset are capped below the truth.\n", flush=True)

    return worst, per_trial


def print_truth_rewards(system, max_traj=None, w_acc=None,
                        energy_normalize=True):
    """Score the known truth structure of each DOF with the standard reward.

    Requires ``system.truth_taus``.  Constants are fitted by the same
    :func:`discode_core.optimise_consts_energy` the trainer uses, so the number
    printed is exactly what the search would score if it proposed the truth
    structure — the target it should converge to.

    Caveat: the exponents in ``truth_taus`` are FITTED, not pinned.  At small
    amplitudes an exponent is weakly identifiable and the grid can settle away
    from the nominal value; the printed expression shows what it actually
    chose.
    """
    import discode_core as dc

    if all(t is None for t in system.truth_taus):
        return None

    max_traj = system.n_trials if max_traj is None else int(max_traj)
    dc.configure_grammar(system.n_dof, system.var_names)
    _X, _y, raw, ns = generate_dataset(system, device=None)
    # w_acc MUST match the training run or the printed target is not the number
    # the search is chasing.
    dc.set_problem_data(ns, raw, energy_normalize, max_traj, w_acc)

    print(f"\n[truth] reward of the known structure "
          f"(max_traj={max_traj}, w_acc={dc.W_ACC:.2f}):")
    out = []
    for d in range(system.n_dof):
        tau = system.truth_taus[d]
        if tau is None:
            print(f"    DOF {d}: (no tau supplied — truth not representable "
                  f"in the grammar)")
            out.append(None)
            continue
        consts = dc.optimise_consts_energy(tau, d, max_traj=max_traj)
        exprs  = [None] * system.n_dof
        exprs[d] = (tau, consts)
        r = dc.energy_reward(exprs, max_traj=max_traj, horizon=None)
        out.append(r)
        print(f"    DOF {d}: r_truth = {r:.4f}    "
              f"{dc.denormalize_expr(tau, consts, d)}", flush=True)
    print(flush=True)
    return out


# ── Dataset preparation ─────────────────────────────────────────────────────
def generate_dataset(system, device=None, center=False):
    """
    Convert a SystemData object into normalised stats and raw trajectories.

    The energy reward only uses ``raw_trajs`` (measured velocities and
    accelerations) and ``norm_stats`` (to denormalise the predicted
    acceleration), so ``X_torch`` / ``y_list`` are returned for
    parity/inspection but are not required by the energy pipeline.  Pass
    ``device=None`` to skip building them entirely (no torch import).

    Returns
    -------
    X_torch    : (m*p, 2*n_dof) normalised feature tensor, or None
    y_list     : list of N tensors, each (m*p,) normalised acceleration, or None
    raw_trajs  : list of (t_eval, ic[2N], state_arr[2N, m], acc_arr[N, m]) per trial
    norm_stats : (X_mean[2N], X_std[2N], y_mean[N], y_std[N])
    """
    acc  = system.acc   # (m, n_dof, p)
    vel  = system.vel
    disp = system.disp
    time = system.time
    m, n_dof, p_trials = acc.shape

    # Feature matrix: interleaved [disp_1, vel_1, disp_2, vel_2, ...] (trial-major)
    X_parts = []
    for d in range(n_dof):
        X_parts.append(disp[:, d, :].T.reshape(-1, 1))  # (m*p, 1)
        X_parts.append(vel[:,  d, :].T.reshape(-1, 1))
    X = np.hstack(X_parts)  # (m*p, 2*n_dof)

    # ── Centre or not ───────────────────────────────────────────────────────
    # Default is SCALE-ONLY (center=False).  Subtracting the mean is an affine
    # change of variable, and a polynomial that is simple in physical
    # coordinates is NOT simple in shifted ones: with x = X_mean + X_std*u,
    # the physical term alpha*(x1-x2)^3 expands into degree-3, degree-2,
    # degree-1 AND constant monomials in u.  The grammar then has to discover
    # those extra terms just to express the truth, which both lengthens the
    # target expression and hands a reward advantage to candidates that pad
    # themselves with spurious low-order terms.  Measured on a cubic-coupled
    # truth: the degree-{1,3} expanded cube scores 0.99916 centred vs 0.99998
    # uncentred — a ~40x residual penalty paid purely for the offset, which is
    # invisible in r-units but is exactly what the log-residual GRPO advantage
    # chases.  These are zero-mean free-decay signals about a homogeneous
    # equilibrium, so there is nothing to centre in the first place.
    X_mean = X.mean(0) if center else np.zeros(X.shape[1])
    X_std  = X.std(0) + 1e-8

    y_mean = np.empty(n_dof)
    y_std  = np.empty(n_dof)
    for d in range(n_dof):
        yd = acc[:, d, :].T.reshape(-1)  # (m*p,)
        y_mean[d] = yd.mean() if center else 0.0
        y_std[d]  = yd.std() + 1e-8

    X_torch, y_list = None, None
    if device is not None:
        import torch
        X_norm  = (X - X_mean) / X_std
        X_torch = torch.tensor(X_norm, dtype=torch.float32, device=device)
        y_list  = []
        for d in range(n_dof):
            yd = acc[:, d, :].T.reshape(-1)
            y_list.append(torch.tensor((yd - y_mean[d]) / y_std[d],
                                       dtype=torch.float32, device=device))

    norm_stats = (X_mean, X_std, y_mean, y_std)

    # Raw trajectories: one entry per trial
    raw_trajs = []
    for trial in range(p_trials):
        state_rows = []
        for d in range(n_dof):
            state_rows.append(disp[:, d, trial])  # q_{d+1}
            state_rows.append(vel[:,  d, trial])  # qd_{d+1}
        state_arr = np.vstack(state_rows)           # (2*n_dof, m)
        ic = state_arr[:, 0].copy()                 # (2*n_dof,)
        acc_arr = acc[:, :, trial].T                # (n_dof, m) physical acceleration
        raw_trajs.append((time, ic, state_arr, acc_arr))

    return X_torch, y_list, raw_trajs, norm_stats


# ── Data plotting ───────────────────────────────────────────────────────────
def plot_system_data(system, save=True, show=True):
    """One disp/vel/acc figure per DOF, all trials overlaid."""
    import matplotlib.pyplot as plt

    time = system.time
    trial_colors = plt.cm.tab10.colors

    for d in range(system.n_dof):
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        fig.suptitle(f"{system.name}  —  DOF {d + 1}  "
                     f"({len(time)} pts,  dt={float(time[1]-time[0]):.4g} s)",
                     fontsize=11)

        row_data   = [system.disp[:, d, :], system.vel[:, d, :], system.acc[:, d, :]]
        row_labels = ['Displacement', 'Velocity', 'Acceleration']
        row_units  = ['m (or counts)', 'm/s', 'm/s^2']

        for ax, data, label, unit in zip(axes, row_data, row_labels, row_units):
            for tr in range(system.n_trials):
                ax.plot(time, data[:, tr],
                        color=trial_colors[tr % len(trial_colors)],
                        lw=0.9, alpha=0.85,
                        label=f'trial {tr + 1}' if d == 0 else '_')
            ax.set_ylabel(f'{label}\n[{unit}]', fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time  (s)', fontsize=9)
        if system.n_trials > 1:
            axes[0].legend(fontsize=7, ncol=min(system.n_trials, 6),
                           loc='upper right')

        plt.tight_layout()
        if save:
            out_png = f"{system.name}_dof{d + 1}_data.png"
            plt.savefig(out_png, dpi=130, bbox_inches='tight')
            print(f"[plot_data] saved {out_png}", flush=True)
        if show:
            plt.show()
        else:
            plt.close(fig)