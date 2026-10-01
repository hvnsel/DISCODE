"""
discover_rollout.py
==================

Forward simulation of ONE DOF's equation against a measured record -- the
integrator behind the simulation reward (``reward='simulation'``)::

    residual_d = w_q * NRMSE(q_d_sim, q_d)
               + w_v * NRMSE(qd_d_sim, qd_d)
               + w_a * NRMSE(a_d_sim, a_d)

per trial, with ``weights = (w_q, w_v, w_a)`` non-negative and summing to 1.
NRMSE is the RMS error over the simulated samples divided by the measured
channel's standard deviation over that trial.  ``a_d_sim`` is the equation's
own acceleration along its simulated trajectory -- its right-hand side
evaluated at the simulated state -- not a finite difference of the simulated
velocity.

Torch-free, so the engine (:func:`discover_core.simulation_reward`, used in
training) and the scoring script (:mod:`discover_score`) run this one
implementation, and the number read after training is the number trained on.

Per DOF, like the energy reward
-------------------------------
Only DOF d's own state ``[q_d, qd_d]`` is integrated.  Every OTHER DOF's state
is read off the measured record, linearly interpolated between samples.  A
coupling term is therefore judged against the partner's true motion, and DOF
d's score never depends on what the search currently believes about the
partner -- the same isolation the energy reward has (see
:func:`discover_core.energy_worker`).  The full coupled simulation of a
finished set of equations is :func:`discover_analysis.forward_simulate`.

Windows
-------
``window=None`` is one free run per trial, from its first sample to its last.
A number is a restart interval in seconds: the record is cut into windows of
that length and each one is simulated from the measured state at its start
(multiple shooting).  Every window of every trial is integrated together as
one vectorised batch, so the sequential cost is the window length rather than
the record length, and a small frequency error costs a bounded phase error per
window instead of one that grows over the whole record.

Integration
-----------
Classical RK4 on the record's own samples.  :func:`auto_steps` sizes the step
from the data: at least ``SIM_STEPS_PER_CYCLE`` steps per cycle of the fastest
measured motion, which on an oversampled (simulated) record means stepping
every few samples and on a coarse one means substeps between samples.  The
other DOFs' states are interpolated linearly between samples at each stage.

A window whose state leaves ``blowup`` times the largest measured magnitude of
its own channel, or goes non-finite, is frozen at its last in-bound state for
the rest of the window.  A diverging equation therefore scores a large but
finite residual -- larger the earlier it diverges -- instead of a NaN.
"""

from __future__ import annotations

import numpy as np

SIM_BLOWUP = 10.0
# Automatic step size: at least this many RK4 steps per cycle of the fastest
# measured motion.  RK4's phase error on an oscillator is ~2*pi*(w h)^4/120 per
# cycle, so 50 steps cost ~1e-5 rad a cycle -- invisible next to any modelling
# error -- while 30 would already cost ~1e-4, i.e. a residual of ~1e-2 over a
# hundred cycles on a perfect equation.
SIM_STEPS_PER_CYCLE = 50
MAX_SUBSTEPS = 8


def _scale(x):
    """Normalising scale of one measured channel: its std, or mean |x| if flat."""
    s = float(np.std(x))
    if s < 1e-12:
        s = float(np.mean(np.abs(x))) + 1e-12
    return s


def check_weights(weights):
    """``(w_q, w_v, w_a)`` as floats -- the displacement, velocity and
    acceleration weights -- or a ValueError unless they are three
    non-negative numbers summing to 1."""
    try:
        w = tuple(float(x) for x in weights)
    except (TypeError, ValueError):
        w = ()
    if (len(w) != 3 or not all(np.isfinite(x) and x >= 0.0 for x in w)
            or abs(sum(w) - 1.0) > 1e-6):
        raise ValueError("sim_weights must be three non-negative weights "
                         "(displacement, velocity, acceleration) summing to 1, "
                         f"got {weights!r}")
    return w


def auto_steps(t, states, accs=None, per_cycle=SIM_STEPS_PER_CYCLE):
    """``(stride, substeps)`` giving ``per_cycle`` RK4 steps per cycle of the
    fastest measured motion.

    The frequency is estimated per DOF and trial as ``rms(a) / rms(qd)`` (exact
    for one harmonic, weighted towards the higher mode otherwise) and the
    largest one is used.  An oversampled record -- a simulated one usually is,
    at hundreds of samples per cycle -- is stepped every ``stride`` samples and
    compared there; a coarse one gets ``substeps`` RK4 steps per sample.
    ``accs`` is (P, N, m); without it ``rms(qd) / rms(q)`` is used.
    """
    t = np.asarray(t, dtype=float)
    states = np.asarray(states, dtype=float)
    if len(t) < 3:
        return 1, 1
    dt = float(np.median(np.diff(t)))
    omega = 0.0
    for p in range(states.shape[0]):
        for d in range(states.shape[1] // 2):
            rq = float(np.sqrt(np.mean(states[p, 2 * d] ** 2)))
            rv = float(np.sqrt(np.mean(states[p, 2 * d + 1] ** 2)))
            if accs is not None:
                ra = float(np.sqrt(np.mean(np.asarray(accs)[p, d] ** 2)))
                w = ra / rv if rv > 0 else 0.0
            else:
                w = rv / rq if rq > 0 else 0.0
            if np.isfinite(w):
                omega = max(omega, w)
    if omega <= 0.0 or dt <= 0.0:
        return 1, 1
    per_sample_cycle = 2.0 * np.pi / (omega * dt)       # samples per cycle
    if per_sample_cycle >= 2 * per_cycle:
        return int(per_sample_cycle // per_cycle), 1
    return 1, int(min(MAX_SUBSTEPS, max(1, np.ceil(per_cycle / per_sample_cycle))))


def rollout(accel, t, states, dof, window=None, stride=1, substeps=1,
            blowup=SIM_BLOWUP, with_acc=True):
    """Integrate DOF ``dof`` of every trial in ``states``.

    Parameters
    ----------
    accel    : ``accel(S) -> a``.  ``S`` is a (2N, W) array of PHYSICAL states
               in the grammar's row order ``[q1, qd1, q2, qd2, ...]``, one
               column per window; ``a`` is DOF ``dof``'s physical acceleration,
               (W,) or a scalar.
    t        : (m,) time grid shared by every trial.
    states   : (P, 2N, m) measured states, one block per trial.
    dof      : the DOF to integrate.  Every other DOF is read off ``states``.
    window   : None for one free run per trial, or the restart interval in
               seconds.
    stride   : samples per RK4 step (the solution is compared every
               ``stride`` samples).
    substeps : RK4 steps per ``stride`` samples.  :func:`auto_steps` picks
               both from the data.
    blowup   : the divergence limit, in multiples of each channel's largest
               measured magnitude.
    with_acc : also return the equation's acceleration along the simulated
               trajectory (one more evaluation per sample).

    Returns ``(q_sim, qd_sim, a_sim, simulated)``, each (P, m).  ``simulated``
    marks the samples that hold a prediction; at the others ``q_sim`` and
    ``qd_sim`` hold the measured state.  ``a_sim`` is the equation evaluated at
    ``(q_sim, qd_sim)`` and the measured partner states, at every predicted
    sample and at each trial's first one (NaN elsewhere, and everywhere when
    ``with_acc`` is False).
    """
    states = np.asarray(states, dtype=float)
    t = np.asarray(t, dtype=float)
    P, n_rows, m = states.shape
    iq, iv = 2 * dof, 2 * dof + 1
    partners = np.array([k for k in range(n_rows) if k not in (iq, iv)], dtype=int)
    stride, K = max(1, int(stride)), max(1, int(substeps))

    q_sim = states[:, iq, :].copy()
    v_sim = states[:, iv, :].copy()
    a_sim = np.full((P, m), np.nan)
    simulated = np.zeros((P, m), dtype=bool)

    if window is None:
        starts, n_rec = np.array([0]), (m - 1) // stride
    else:
        dt = float(np.median(np.diff(t))) if m > 1 else 1.0
        n = stride * max(1, int(round(float(window) / (dt * stride))))
        if n >= m - 1:
            starts, n_rec = np.array([0]), (m - 1) // stride
        else:
            starts, n_rec = np.arange(0, m - 1, n), n // stride
    if n_rec < 1:
        return q_sim, v_sim, a_sim, simulated

    trial = np.repeat(np.arange(P), len(starts))        # (W,) window -> trial
    start = np.tile(starts, P)                           # (W,) window -> sample
    W = trial.size

    q = states[trial, iq, start].copy()
    v = states[trial, iv, start].copy()
    q_lim = blowup * float(np.max(np.abs(states[:, iq, :]))) + 1e-12
    v_lim = blowup * float(np.max(np.abs(states[:, iv, :]))) + 1e-12
    alive = np.ones(W, dtype=bool)

    part = states[:, partners, :] if partners.size else None     # (P, C, m)
    # The 2K+1 distinct RK4 stage points of one stride, as (whole samples,
    # fraction) offsets from its first sample.
    offs = [i * stride / (2.0 * K) for i in range(2 * K + 1)]
    offs = [(int(np.floor(o)), o - np.floor(o)) for o in offs]

    S = np.empty((n_rows, W))
    a_out = np.empty(W)

    def f(qq, vv, pp):
        S[iq] = qq
        S[iv] = vv
        if pp is not None:
            S[partners] = pp
        try:
            # copied out: an expression that is a bare variable returns a view
            # of S, which the next stage overwrites
            a_out[:] = accel(S)
        except Exception:                                   # noqa: BLE001
            a_out[:] = np.nan
        return a_out.copy()

    with np.errstate(all='ignore'):
        if with_acc:
            a_sim[trial, start] = f(q, v, part[trial, :, start].T
                                    if part is not None else None)
        for j in range(n_rec):
            base = start + stride * j
            # windows running past the record end read clipped samples; their
            # predictions are never kept
            h = (t[np.minimum(base + stride, m - 1)]
                 - t[np.minimum(base, m - 1)]) / K
            pts = []
            if part is not None:
                for io, fo in offs:
                    i0 = np.minimum(base + io, m - 1)
                    p0 = part[trial, :, i0].T                        # (C, W)
                    if fo > 0.0:
                        p1 = part[trial, :, np.minimum(i0 + 1, m - 1)].T
                        p0 = p0 + fo * (p1 - p0)
                    pts.append(p0)
            else:
                pts = [None] * len(offs)
            for k in range(K):
                pa, pm, pb = pts[2 * k], pts[2 * k + 1], pts[2 * k + 2]
                k1q = v
                k1v = f(q, v, pa)
                k2q = v + 0.5 * h * k1v
                k2v = f(q + 0.5 * h * k1q, v + 0.5 * h * k1v, pm)
                k3q = v + 0.5 * h * k2v
                k3v = f(q + 0.5 * h * k2q, v + 0.5 * h * k2v, pm)
                k4q = v + h * k3v
                k4v = f(q + h * k3q, v + h * k3v, pb)
                qn = q + (h / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
                vn = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                # a NaN fails both comparisons, so this also catches non-finite
                alive = alive & (np.abs(qn) <= q_lim) & (np.abs(vn) <= v_lim)
                q = np.where(alive, qn, q)
                v = np.where(alive, vn, v)
            tgt = base + stride
            keep = tgt <= m - 1
            q_sim[trial[keep], tgt[keep]] = q[keep]
            v_sim[trial[keep], tgt[keep]] = v[keep]
            simulated[trial[keep], tgt[keep]] = True
            if with_acc:
                # pts[-1] is the partner state at this very sample
                a_now = f(q, v, pts[-1])
                a_sim[trial[keep], tgt[keep]] = a_now[keep]
    return q_sim, v_sim, a_sim, simulated


def rollout_residuals(accel, t, states, dof, window=None, stride=1, substeps=1,
                      weights=(1.0, 0.0, 0.0), accs=None, blowup=SIM_BLOWUP):
    """Simulation residual of DOF ``dof`` for every trial: a (P,) array.

    ``w_q * NRMSE(q) + w_v * NRMSE(qd) + w_a * NRMSE(a)`` with ``weights =
    (w_q, w_v, w_a)`` (see :func:`check_weights`), each NRMSE taken over the
    simulated samples and normalised by that trial's measured std of the
    channel.  ``accs`` is the measured acceleration, (P, N, m); it is needed
    only when ``w_a > 0``.  The simulated acceleration of a window that blew
    up is clipped at ``blowup`` times the largest measured one, so it too
    scores large but finite.  See :func:`rollout` for the other arguments.
    """
    w_q, w_v, w_a = check_weights(weights)
    if w_a > 0.0 and accs is None:
        raise ValueError("an acceleration weight needs the measured "
                         "acceleration (accs)")
    states = np.asarray(states, dtype=float)
    q_sim, v_sim, a_sim, sim = rollout(accel, t, states, dof, window, stride,
                                       substeps, blowup, with_acc=w_a > 0.0)
    iq, iv = 2 * dof, 2 * dof + 1
    if w_a > 0.0:
        a_meas = np.asarray(accs, dtype=float)[:, dof, :]
        a_lim = blowup * float(np.max(np.abs(a_meas))) + 1e-12
        a_sim = np.clip(np.where(np.isfinite(a_sim), a_sim, a_lim), -a_lim, a_lim)

    def nrmse(sim_x, meas_x, mk):
        return (np.sqrt(np.mean((sim_x[mk] - meas_x[mk]) ** 2))
                / _scale(meas_x))

    out = np.full(states.shape[0], np.nan)
    for p in range(states.shape[0]):
        mk = sim[p]
        if not mk.any():
            continue
        res = 0.0
        if w_q > 0.0:
            res += w_q * nrmse(q_sim[p], states[p, iq], mk)
        if w_v > 0.0:
            res += w_v * nrmse(v_sim[p], states[p, iv], mk)
        if w_a > 0.0:
            res += w_a * nrmse(a_sim[p], a_meas[p], mk)
        out[p] = res
    return out
