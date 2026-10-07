"""
discover_rollout.py
==================

Forward simulation of ONE DOF's equation against a measured record -- the
integrator behind the simulation reward (``reward='simulation'``)::

    residual_d = w_q * NRMSE(q_d_sim, q_d)
               + w_v * NRMSE(qd_d_sim, qd_d)
               + w_a * NRMSE(a_d_sim, a_d)
               + w_tf * R_tf(q_d_sim, q_d)
               + w_f * R_f(qd_d_sim, qd_d)

per trial, with ``weights = (w_q, w_v, w_a, w_tf, w_f)`` non-negative and
summing to 1 (a 3-tuple means ``w_tf = w_f = 0``, a 4-tuple ``w_f = 0``).
NRMSE is the RMS error over the simulated samples divided by the measured
channel's standard deviation over that trial.  ``a_d_sim`` is the equation's
own acceleration along its simulated trajectory -- its right-hand side
evaluated at the simulated state -- not a finite difference of the simulated
velocity.  ``R_tf`` is the time-frequency term and ``R_f`` the
frequency-content term, both below.

Time-frequency term
-------------------
Point-by-point NRMSE is phase-sensitive: over a long free run any small
frequency error slides the simulated oscillation out of phase, NRMSE climbs to
~1 and stays there, and every slightly-wrong equation scores the same.  Beats
-- slow amplitude modulation from two close frequencies -- are where that
hurts most, because the beat frequency is a DIFFERENCE of frequencies and so
amplifies every coefficient error by f / delta_f.

``R_tf`` compares the local amplitude in each frequency band over time
instead of the raw signal (an analytic Morlet filter bank on the displacement,
:func:`tf_setup` / :func:`tf_residuals`), plus a signed slow trend:

* the amplitude of ``a cos(2 pi f t + phi)`` in the band at ``f`` is ``a`` for
  any ``phi``, so a run that has slipped in phase but has the right amplitude
  history is not punished;
* two close frequencies inside one band make that band's amplitude rise and
  fall at ``delta_f`` -- a beat -- and an equation without the modulation pays
  its full depth;
* each band's amplitude level and modulation depth are also compared
  regardless of timing, so an equation that beats at the wrong rate still
  scores better than one that does not beat (:func:`tf_residuals`);
* motion about a centre that itself oscillates and decays (another, slower
  mode) has that centre in its own band, so it cannot fake a modulation;
* non-oscillatory motion -- drift, creep, a decaying offset -- has no band
  content and lands in the signed trend, where the term reduces to a smoothed
  NRMSE.

Amplitudes are compared in log, so a 2x error costs the same at high and low
amplitude and a low-amplitude regime late in a decay counts as much as the
start.  Every parameter (frequency range, bands, floors) comes from the
measured record.  Amplitudes discard phase relations between DOFs, so the
term complements the pointwise ones rather than replacing them.  It always
gets a free run: with a ``window`` the pointwise terms are taken on the
restarted windows and the envelope on a separate free run, since a windowed
signal jumps at every window boundary.

Frequency-content term
----------------------
``R_f`` asks WHERE IN FREQUENCY the motion's energy sits, not when: it is the
earth mover's (Wasserstein-1) distance between the simulated and the measured
velocity power spectra of each trial, measured in CYCLES OVER THE RECORD --
the distance in Hz times the record length -- and taken as
``R_f = ln(1 + cycles)`` (:func:`spec_setup` / :func:`spec_residuals`).
Moving a spectrum by ``df`` is ``T * df`` cycles, exactly the phase a frequency
error of ``df`` piles up over the record, so:

* a frequency error is graded at any size, however far the phase has slipped:
  1% on a 6.5 Hz mode over a 5 s record is 0.32 cycles, ``R_f`` 0.28 -- the
  slow drift a windowed NRMSE cannot see (each window is too short for the
  phase to slip) and the envelope term sees only indirectly (amplitude is
  phase-blind and its bands are ~f/6 wide; it catches a detuned DOF where a
  measured partner drives it at the true frequency and the two beat).  The
  cost keeps growing with the error instead of saturating as a free run's
  NRMSE does;
* energy in the wrong mode costs its share times the distance it has to move,
  so a simulation that lets a mode die out pays for moving that mode's energy
  to the ones it kept;
* beats come out as two close peaks: having both, at the right places and in
  the right proportion, is what makes the beat right.

The logarithm keeps the term on the scale of the others: moving energy
between distant modes costs tens of cycles, and taken linearly that would
drown every other term, while below a cycle or so ``R_f`` is still about the
cycles themselves.  Velocity rather than displacement, so the spectrum is the
kinetic energy's and a higher mode that is small in displacement still
counts.  Amplitude is normalised out -- the pointwise terms see it.  It shares
the time-frequency term's free run.

Torch-free, so the engine (:func:`discover_core.simulation_reward`, used in
training) and the scoring script (:mod:`discover_score`) run this one
implementation, and the number read after training is the number trained on.

Per DOF, like the energy reward -- or several together
------------------------------------------------------
:func:`rollout` integrates only DOF d's own state ``[q_d, qd_d]``.  Every
OTHER DOF's state is read off the measured record, linearly interpolated
between samples.  A coupling term is therefore judged against the partner's
true motion, and DOF d's score never depends on what the search currently
believes about the partner -- the same isolation the energy reward has (see
:func:`discover_core.energy_worker`).

:func:`rollout_set` integrates several DOFs together as one coupled system,
each driven by the others' simulated states -- the trainer's ``sim_coupling``
scores part of each batch this way, and :func:`rollout_set_residuals` charges
every DOF of a set that blows up.  With one DOF it is :func:`rollout`, to the
bit.  The full coupled simulation of a finished set of equations in the plot
scripts is :func:`discover_analysis.forward_simulate`.

Windows
-------
``window=None`` is one free run per trial, from its first sample to its last.
A number is a restart interval in seconds: the record is cut into windows of
that length and each one is simulated from the measured state at its start
(multiple shooting).  Every window of every trial is integrated together as
one vectorised batch, so the sequential cost is the window length rather than
the record length, and a small frequency error costs a bounded phase error per
window instead of one that grows over the whole record -- on a free run the
pointwise errors saturate near 1.4 for any slightly-off frequency and stop
ranking anything.  :func:`auto_window` picks a few periods of the fastest
motion -- of the DOF being simulated, not of the whole record: a slow DOF next
to a fast one would otherwise be cut into windows shorter than one of its own
cycles, where no frequency error shows.  The time-frequency and
frequency-content terms always get a free run of their own.

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
# sim_window='auto': restart every this many periods of the fastest motion
SIM_AUTO_PERIODS = 3.0

# Time-frequency term (see the module docstring).
#   TF_OMEGA0      Morlet centre frequency: a band at f spans f/6 in frequency
#                  and ~one period in time, so two lines closer than ~f/6 share
#                  a band and show up as its amplitude beating
#   TF_PER_OCTAVE  bands per octave
#   TF_LO_CYCLES   lowest band f_lo = TF_LO_CYCLES / record length; at 6 the
#                  lowest band keeps ~half its samples after the edge mask
#   TF_POWER_FRAC  highest band: the frequency below which this fraction of the
#                  measured VELOCITY power lies (velocity, not displacement,
#                  so a higher mode that is small in displacement is not cut)
#   TF_BAND_FLOOR  drop a band whose mean measured amplitude is below this
#                  fraction of the measured signal's std (noise-only bands)
#   TF_MAG_FLOOR   per band, amplitudes are floored at this fraction of the
#                  band's largest measured amplitude before the log (40 dB)
#   TF_TREND       include the signed slow trend
#   TF_RATE        include the beat-rate term (tf_residuals)
#   TF_RATE_LO_CYCLES  slowest modulation it compares: this many cycles over
#                  the record
#   TF_RATE_REF    envelope swing (std of a band's detrended log-amplitude,
#                  nepers) at which a trial's beat-rate term counts in full;
#                  a trial whose envelope barely moves has no beat rate to
#                  compare and is scaled down in proportion
#   TF_RATE_FLOOR  modulation spectra are floored at this fraction of the
#                  measured one's mean power, so a flat simulated envelope
#                  reads as "no rate" rather than as its rounding noise
#   TF_RATE_SHARE  the beat-rate term and the envelope swing use only bands
#                  whose mean measured amplitude is at least this fraction of
#                  the trial's strongest band: a band on a filter's tail sees
#                  the record's edges more than the signal
TF_OMEGA0 = 6.0
TF_PER_OCTAVE = 8
TF_LO_CYCLES = 6.0
TF_POWER_FRAC = 0.99
TF_BAND_FLOOR = 1e-2
TF_MAG_FLOOR = 1e-2
TF_TREND = True
TF_RATE = True
TF_RATE_LO_CYCLES = 1.0
TF_RATE_REF = 0.3
TF_RATE_FLOOR = 1e-2
TF_RATE_SHARE = 0.1

# Frequency-content term (see the module docstring).
#   SPEC_LO_CYCLES   lowest frequency compared: this many cycles over the record
#                    (below lie the mean and slow drift)
#   SPEC_POWER_FRAC  highest frequency compared: SPEC_MARGIN times the one
#   SPEC_MARGIN      below which this fraction of the measured velocity power
#                    lies, pooled over trials -- the margin keeps a simulated
#                    mode a little above the measured top one in view
#   SPEC_NYQUIST     ... and never above this fraction of the grid's Nyquist
#   SPEC_FLOOR       both spectra are floored at this fraction of the measured
#                    one's mean power before normalising, so a motionless
#                    simulation reads as energy spread evenly, not as nothing
SPEC_LO_CYCLES = 2.0
SPEC_POWER_FRAC = 0.995
SPEC_MARGIN = 1.5
SPEC_NYQUIST = 0.8
SPEC_FLOOR = 1e-3


def _scale(x):
    """Normalising scale of one measured channel: its std, or mean |x| if flat."""
    s = float(np.std(x))
    if s < 1e-12:
        s = float(np.mean(np.abs(x))) + 1e-12
    return s


def check_weights(weights):
    """``(w_q, w_v, w_a, w_tf, w_f)`` as floats -- the displacement, velocity,
    acceleration, time-frequency and frequency-content weights -- or a
    ValueError unless they are three to five non-negative numbers summing to
    1.  The weights left out are 0."""
    try:
        w = tuple(float(x) for x in weights)
    except (TypeError, ValueError):
        w = ()
    if (len(w) not in (3, 4, 5)
            or not all(np.isfinite(x) and x >= 0.0 for x in w)
            or abs(sum(w) - 1.0) > 1e-6):
        raise ValueError("sim_weights must be three to five non-negative "
                         "weights (displacement, velocity, acceleration"
                         "[, time-frequency[, frequency content]]) summing to "
                         f"1, got {weights!r}")
    return w + (0.0,) * (5 - len(w))


def combine_trials(res, decay=1.0):
    """One residual from per-trial residuals, trials on the last axis.

    The trials are ranked from worst fit to best and the k-th worst is
    weighted ``decay**(k-1)``, normalised: ``decay=1`` is the plain mean,
    ``0`` the worst trial alone, and ``0.5`` halves the weight at each rank --
    with five trials 52, 26, 13, 6 and 3%.  Each equation is ranked by its own
    residuals, so whichever trial it fits worst counts most: a phenomenon
    that only one trial shows (beats, say) cannot be averaged away by the
    trials that lack it.  A 2-D ``res`` (DOFs x trials) is averaged over DOFs
    first; at ``decay=1`` it is simply the mean of everything.  NaN or inf in
    gives NaN or inf out.
    """
    r = np.asarray(res, dtype=float)
    if decay == 1.0:
        return float(np.mean(r))
    if r.ndim > 1:
        r = r.reshape(-1, r.shape[-1]).mean(axis=0)
    w = float(decay) ** np.arange(r.size)
    return float(np.dot(w, np.sort(r)[::-1]) / w.sum())


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
    if len(t) < 3:
        return 1, 1
    dt = float(np.median(np.diff(t)))
    omega = fastest_omega(states, accs)
    if omega <= 0.0 or dt <= 0.0:
        return 1, 1
    per_sample_cycle = 2.0 * np.pi / (omega * dt)       # samples per cycle
    if per_sample_cycle >= 2 * per_cycle:
        return int(per_sample_cycle // per_cycle), 1
    return 1, int(min(MAX_SUBSTEPS, max(1, np.ceil(per_cycle / per_sample_cycle))))


def fastest_omega(states, accs=None, dofs=None):
    """The fastest measured motion of ``dofs`` (default: every DOF), in rad/s:
    the largest over trials of ``rms(a) / rms(qd)`` -- exact for one harmonic,
    weighted towards the higher mode otherwise -- or of ``rms(qd) / rms(q)``
    without ``accs`` (P, N, m).  0 if nothing moves."""
    states = np.asarray(states, dtype=float)
    accs = None if accs is None else np.asarray(accs, dtype=float)
    which = (range(states.shape[1] // 2) if dofs is None
             else [int(d) for d in np.atleast_1d(dofs)])
    omega = 0.0
    for p in range(states.shape[0]):
        for d in which:
            rq = float(np.sqrt(np.mean(states[p, 2 * d] ** 2)))
            rv = float(np.sqrt(np.mean(states[p, 2 * d + 1] ** 2)))
            if accs is not None:
                ra = float(np.sqrt(np.mean(accs[p, d] ** 2)))
                w = ra / rv if rv > 0 else 0.0
            else:
                w = rv / rq if rq > 0 else 0.0
            if np.isfinite(w):
                omega = max(omega, w)
    return omega


def auto_window(t, states, accs=None, periods=SIM_AUTO_PERIODS, dofs=None):
    """The ``sim_window='auto'`` restart interval, in seconds: ``periods``
    periods of the fastest measured motion of ``dofs`` (default: every DOF),
    estimated as in :func:`auto_steps`.  Short enough that a small frequency
    error costs a bounded phase error per window, long enough to show each
    window's frequency and damping.  A DOF simulated alone takes its own
    (``dofs=d``): on a record with a 20 Hz and a 6.5 Hz DOF, the slow one would
    otherwise get windows of about one of its own cycles, too short for its
    frequency error to show.  A set integrated together restarts as one, at
    its fastest member's.  None if the record has no motion to time."""
    omega = fastest_omega(states, accs, dofs)
    return periods * 2.0 * np.pi / omega if omega > 0.0 else None


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
    _dofs, q, v, a, sim, _blown = rollout_set({dof: accel}, t, states, window,
                                              stride, substeps, blowup, with_acc)
    return q[:, 0], v[:, 0], a[:, 0], sim


def rollout_set(accels, t, states, window=None, stride=1, substeps=1,
                blowup=SIM_BLOWUP, with_acc=True):
    """Integrate several DOFs of every trial TOGETHER, as one coupled system.

    ``accels`` maps each DOF to integrate to its ``accel(S) -> a`` (see
    :func:`rollout`); every DOF not in it is read off ``states``.  Each
    equation sees the other integrated DOFs' SIMULATED states, so a coupling
    term is driven by the partner's predicted motion, not the measured one.
    With one DOF this is :func:`rollout`, to the bit.  A window freezes as a
    whole when any of its DOFs leaves its bound: past that point the coupled
    state is meaningless for all of them.

    Returns ``(dofs, q_sim, qd_sim, a_sim, simulated, blown)``: ``dofs`` the
    integrated DOFs in ascending order; ``q_sim``, ``qd_sim`` and ``a_sim``
    (P, n, m) in that order; ``simulated`` (P, m); and ``blown`` (P,), True
    for a trial with a window that left its bounds.  The other arguments are
    as in :func:`rollout`.
    """
    states = np.asarray(states, dtype=float)
    t = np.asarray(t, dtype=float)
    P, n_rows, m = states.shape
    dofs = sorted(int(d) for d in accels)
    fns = [accels[d] for d in dofs]
    n = len(dofs)
    iq = np.array([2 * d for d in dofs], dtype=int)
    iv = iq + 1
    own = set(iq.tolist()) | set(iv.tolist())
    partners = np.array([k for k in range(n_rows) if k not in own], dtype=int)
    stride, K = max(1, int(stride)), max(1, int(substeps))

    q_sim = states[:, iq, :].copy()
    v_sim = states[:, iv, :].copy()
    a_sim = np.full((P, n, m), np.nan)
    simulated = np.zeros((P, m), dtype=bool)
    blown = np.zeros(P, dtype=bool)

    if window is None:
        starts, n_rec = np.array([0]), (m - 1) // stride
    else:
        dt = float(np.median(np.diff(t))) if m > 1 else 1.0
        n_win = stride * max(1, int(round(float(window) / (dt * stride))))
        if n_win >= m - 1:
            starts, n_rec = np.array([0]), (m - 1) // stride
        else:
            starts, n_rec = np.arange(0, m - 1, n_win), n_win // stride
    if n_rec < 1:
        return dofs, q_sim, v_sim, a_sim, simulated, blown

    trial = np.repeat(np.arange(P), len(starts))        # (W,) window -> trial
    start = np.tile(starts, P)                           # (W,) window -> sample
    W = trial.size

    q = states[trial, :, start][:, iq].T.copy()          # (n, W)
    v = states[trial, :, start][:, iv].T.copy()
    q_lim = blowup * np.max(np.abs(states[:, iq, :]), axis=(0, 2))[:, None] + 1e-12
    v_lim = blowup * np.max(np.abs(states[:, iv, :]), axis=(0, 2))[:, None] + 1e-12
    alive = np.ones(W, dtype=bool)

    part = states[:, partners, :] if partners.size else None     # (P, C, m)
    # The 2K+1 distinct RK4 stage points of one stride, as (whole samples,
    # fraction) offsets from its first sample.
    offs = [i * stride / (2.0 * K) for i in range(2 * K + 1)]
    offs = [(int(np.floor(o)), o - np.floor(o)) for o in offs]

    S = np.empty((n_rows, W))
    a_out = np.empty((n, W))

    def f(qq, vv, pp):
        S[iq] = qq
        S[iv] = vv
        if pp is not None:
            S[partners] = pp
        for k, fn in enumerate(fns):
            try:
                # copied out: an expression that is a bare variable returns a
                # view of S, which the next stage overwrites
                a_out[k] = fn(S)
            except Exception:                               # noqa: BLE001
                a_out[k] = np.nan
        return a_out.copy()

    with np.errstate(all='ignore'):
        if with_acc:
            a_sim[trial, :, start] = f(q, v, part[trial, :, start].T
                                       if part is not None else None).T
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
                ok = (np.abs(qn) <= q_lim) & (np.abs(vn) <= v_lim)
                alive = alive & (ok[0] if n == 1 else ok.all(axis=0))
                q = np.where(alive, qn, q)
                v = np.where(alive, vn, v)
            tgt = base + stride
            keep = tgt <= m - 1
            q_sim[trial[keep], :, tgt[keep]] = q[:, keep].T
            v_sim[trial[keep], :, tgt[keep]] = v[:, keep].T
            simulated[trial[keep], tgt[keep]] = True
            if with_acc:
                # pts[-1] is the partner state at this very sample
                a_now = f(q, v, pts[-1])
                a_sim[trial[keep], :, tgt[keep]] = a_now[:, keep].T
    blown[trial[~alive]] = True
    return dofs, q_sim, v_sim, a_sim, simulated, blown


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 2))))


def _band_stats(logA, valid):
    """Mean and std over each band's valid samples of ``logA`` (..., K, M)."""
    n = valid.sum(axis=-1)
    mu = (logA * valid).sum(axis=-1) / n
    sd = np.sqrt((((logA - mu[..., None]) ** 2) * valid).sum(axis=-1) / n)
    return mu, sd


def _mod_setup(valid, D, N, fk, T):
    """What the beat-rate term needs, per band: a Hann taper over the band's
    valid span and the centred sample index there (to remove a straight-line
    trend), plus the modulation frequencies it compares and their spacing in
    octaves."""
    K, M = valid.shape
    win, xc = np.zeros((K, M)), np.zeros((K, M))
    sxx = np.ones(K)
    for k in range(K):
        idx = np.flatnonzero(valid[k])
        if idx.size >= 8:
            win[k, idx] = np.hanning(idx.size)
            xc[k, idx] = np.arange(idx.size) - (idx.size - 1) / 2.0
            sxx[k] = float((xc[k, idx] ** 2).sum())
    nu = np.fft.rfftfreq(N, D)
    rng = ((nu >= TF_RATE_LO_CYCLES / T)
           & (nu <= min(2.0 * float(fk.max()) / TF_OMEGA0, 0.25 / D)))
    return {'win': win, 'xc': xc, 'sxx': sxx, 'n': np.maximum(valid.sum(axis=1), 1),
            'N': N, 'range': rng, 'du': np.diff(np.log2(nu[rng]))}


def _mod_spectrum(L, valid, mod, use):
    """One trial's modulation power spectrum over the compared range -- the
    log-amplitude of every band in ``use`` detrended (mean and slope) over its
    valid span, tapered, transformed and summed in power -- and the largest
    detrended std among those bands (the envelope swing, nepers)."""
    mu = (L * valid).sum(axis=1) / mod['n']
    slope = (L * mod['xc']).sum(axis=1) / mod['sxx']
    det = (L - mu[:, None] - slope[:, None] * mod['xc']) * valid * use[:, None]
    swing = float(np.sqrt((det ** 2).sum(axis=1) / mod['n']).max())
    S = (np.abs(np.fft.rfft(det * mod['win'], mod['N'], axis=1)) ** 2).sum(axis=0)
    return S[mod['range']], swing


def _peak_freq(S, nu):
    """Frequency of the largest value of ``S`` on the uniform grid ``nu``,
    refined between bins by a parabola through the peak and its neighbours."""
    i = int(np.argmax(S))
    if 0 < i < len(S) - 1:
        den = S[i - 1] - 2.0 * S[i] + S[i + 1]
        if den < 0.0:
            return float(nu[i] + 0.5 * (S[i - 1] - S[i + 1]) / den
                         * (nu[1] - nu[0]))
    return float(nu[i])


def _rate_distance(S_sim, cdf_meas, floor, du):
    """Wasserstein-1 distance, in octaves, between the simulated modulation
    spectrum and the measured one (given as its CDF over the same grid)."""
    S = S_sim + floor
    cdf = np.cumsum(S) / S.sum()
    return float(np.sum(np.abs(cdf[:-1] - cdf_meas[:-1]) * du))


def tf_setup(t, x_meas, stride):
    """The measured side of the time-frequency term for one DOF, computed once.

    ``x_meas`` is (P, m): the measured displacement of every trial on the full
    time grid ``t``.  The analysis grid is every ``stride``-th sample --
    exactly the samples a free-running :func:`rollout` predicts at that stride
    -- so the simulated side needs no resampling.

    Builds the frequency range and bands from the data, the analytic Morlet
    filter bank::

        G_k(f) = 2 exp(-(w0 (f - f_k) / f_k)^2 / 2)   for f > 0, else 0

    (amplitude-normalised: ``a cos(2 pi f_k t)`` gives a band amplitude of
    ``a``), the per-band edge masks (the wavelet's own span at each end), the
    floors, the measured log-amplitudes and the measured signed trend.  Pass
    the result to :func:`tf_residuals`.  A record too short to analyse, or one
    with no oscillation, simply yields no bands.
    """
    t = np.asarray(t, dtype=float)
    x = np.atleast_2d(np.asarray(x_meas, dtype=float))
    P, m = x.shape
    s = max(1, int(stride))
    n = (m - 1) // s
    idx = np.arange(n + 1) * s
    M = n + 1
    setup = {'stride': s, 'idx': idx, 'M': M, 'N': _next_pow2(2 * M),
             'bands': np.zeros(0), 'G': None, 'valid': None, 'eps': None,
             'logA': None, 'trend': None, 'rate': None, 'swing': None,
             'f_lo': np.nan, 'f_hi': np.nan}
    if M < 16:
        return setup
    tt = t[idx] - t[idx[0]]
    T = float(tt[-1])
    D = float(np.median(np.diff(tt)))
    N = setup['N']
    xg = x[:, idx]
    X = np.fft.fft(xg, N, axis=1)
    f = np.fft.fftfreq(N, D)
    std = np.array([_scale(row) for row in xg])
    setup['std'] = std

    # Frequency range.  The top comes from the VELOCITY power (displacement
    # power times f^2) of the mean-removed, Hann-tapered record: a higher mode
    # that is small in displacement must still get its bands, and the taper
    # stops the record's start from looking broadband.  Noise that this lets
    # in is removed by the band floor below.
    f_lo = TF_LO_CYCLES / T
    fr = np.fft.rfftfreq(N, D)
    xc = (xg - xg.mean(axis=1, keepdims=True)) * np.hanning(M)
    pw = (np.abs(np.fft.rfft(xc, N, axis=1)) ** 2).sum(axis=0) * fr ** 2
    f_hi = 0.0
    if pw.sum() > 0.0:
        cum = np.cumsum(pw)
        f_hi = float(fr[min(int(np.searchsorted(cum, TF_POWER_FRAC * cum[-1])),
                            len(fr) - 1)])
    f_hi = min(f_hi, 0.2 / D)                 # 0.4 x the grid's Nyquist
    setup['f_lo'], setup['f_hi'] = f_lo, f_hi

    if f_hi >= f_lo:
        n_bands = int(np.floor(TF_PER_OCTAVE * np.log2(f_hi / f_lo))) + 1
        fk = f_lo * 2.0 ** (np.arange(n_bands) / TF_PER_OCTAVE)
        G = (2.0 * np.exp(-0.5 * (TF_OMEGA0 * (f[None, :] - fk[:, None])
                                  / fk[:, None]) ** 2)
             * (f[None, :] > 0.0))
        # a band's own time span: sigma_t = w0 / (2 pi f_k); drop sqrt(2)
        # sigma_t at each end, where the transform sees the record's edges
        edge = np.sqrt(2.0) * TF_OMEGA0 / (2.0 * np.pi * fk)
        valid = (tt[None, :] >= edge[:, None]) & (tt[None, :] <= T - edge[:, None])
        A = np.stack([np.abs(np.fft.ifft(X[p][None, :] * G, axis=1)[:, :M])
                      for p in range(P)])                          # (P, K, M)
        n_valid = valid.sum(axis=1)
        band_mean = np.array([A[:, k, valid[k]].mean() if n_valid[k] else 0.0
                              for k in range(n_bands)])
        keep = (n_valid >= 4) & (band_mean >= TF_BAND_FLOOR * float(std.mean()))
        if keep.any():
            fk, G, valid, A = fk[keep], G[keep], valid[keep], A[:, keep]
            eps = TF_MAG_FLOOR * np.array([A[:, k, valid[k]].max()
                                           for k in range(len(fk))])
            logA = np.log(A + eps[None, :, None])
            mu, sd = _band_stats(logA, valid)
            setup.update(bands=fk, G=G, valid=valid, eps=eps, logA=logA,
                         logA_mean=mu, logA_std=sd)
            mod = _mod_setup(valid, D, N, fk, T)
            level = np.array([[A[p, k, valid[k]].mean() for k in range(len(fk))]
                              for p in range(P)])
            use = level >= TF_RATE_SHARE * level.max(axis=1, keepdims=True)
            spec = [_mod_spectrum(logA[p], valid, mod, use[p]) for p in range(P)]
            swing = np.array([w for _S, w in spec])
            setup['swing'] = swing
            if TF_RATE and mod['range'].sum() >= 4:
                floor = np.array([TF_RATE_FLOOR * S.mean() for S, _w in spec])
                cdf = np.stack([np.cumsum(S + fl) / (S + fl).sum()
                                if (S + fl).sum() > 0 else np.zeros_like(S)
                                for (S, _w), fl in zip(spec, floor)])
                nu = np.fft.rfftfreq(N, D)[mod['range']]
                setup['rate'] = {'mod': mod, 'cdf': cdf, 'floor': floor,
                                 'use': use,
                                 'weight': np.minimum(1.0, swing / TF_RATE_REF)
                                 * (floor > 0),
                                 'nu_peak': np.array([_peak_freq(S, nu)
                                                      for S, _w in spec])}

    if TF_TREND:
        f_tr = 0.5 * f_lo
        tau = 1.0 / (2.0 * np.pi * f_tr)
        tr_valid = (tt >= 2.0 * tau) & (tt <= T - 2.0 * tau)
        if tr_valid.sum() >= 4:
            gain = np.exp(-0.5 * (f / f_tr) ** 2)
            setup['trend'] = {
                'gain': gain, 'valid': tr_valid,
                'xbar': np.fft.ifft(X * gain[None, :], axis=1).real[:, :M]}
    return setup


def spectral_peaks(t, x_meas, stride, f_lo=0.0, f_hi=np.inf, n_peaks=2):
    """The ``n_peaks`` strongest local maxima of each trial's displacement
    spectrum (mean removed, Hann-tapered, on the ``stride`` grid) between
    ``f_lo`` and ``f_hi``: ``(freqs, rel)``, each (P, n_peaks), ``rel`` the
    amplitude over the strongest's; NaN where a trial has fewer peaks.  Peaks
    closer than the record can resolve (1/T) merge into one -- the envelope
    swing still shows their beat."""
    t = np.asarray(t, dtype=float)
    x = np.atleast_2d(np.asarray(x_meas, dtype=float))
    idx = np.arange((x.shape[1] - 1) // max(1, int(stride)) + 1) * max(1, int(stride))
    xg = x[:, idx]
    D = float(np.median(np.diff(t[idx])))
    n = _next_pow2(4 * len(idx))
    fr = np.fft.rfftfreq(n, D)
    amp = np.abs(np.fft.rfft((xg - xg.mean(axis=1, keepdims=True))
                             * np.hanning(len(idx)), n, axis=1))
    freqs = np.full((x.shape[0], n_peaks), np.nan)
    rel = np.full((x.shape[0], n_peaks), np.nan)
    band = (fr >= f_lo) & (fr <= f_hi)
    for p in range(x.shape[0]):
        a = amp[p]
        loc = np.flatnonzero(band[1:-1] & (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])) + 1
        if loc.size:
            top = loc[np.argsort(a[loc])[::-1][:n_peaks]]
            freqs[p, :top.size] = fr[top]
            rel[p, :top.size] = a[top] / a[top[0]]
    return freqs, rel


def tf_residuals(setup, q_sim):
    """The time-frequency term ``R_tf`` for every trial: a (P,) array.

    ``q_sim`` is the simulated displacement on the full time grid, as
    :func:`rollout` returns it.  With ``L = ln(A + eps_k)`` the log band
    amplitude, per trial::

        R_local = mean over bands of the mean over valid times of
                  | L_sim(t) - L_meas(t) |
        R_stat  = mean over bands of  | mean_t L_sim - mean_t L_meas |
                                    + | std_t  L_sim - std_t  L_meas |
        R_trend = RMS(trend_sim - trend_meas) / std(measured)    (sign kept)
        R_rate  = W1(modulation spectrum_sim, modulation spectrum_meas)
                  in octaves, x min(1, measured envelope swing / TF_RATE_REF)
        R_tf    = R_local + R_stat + R_trend + R_rate

    ``R_local`` sees WHEN the amplitude rises and falls, so it grades a beat
    period that is slightly off -- but only while the simulated beat has
    slipped by less than about half a beat over the record; past that, beats
    out of step cost more than no beats at all.  ``R_stat`` is blind to timing:
    it scores each band's amplitude level and modulation depth, so an equation
    that beats at the wrong rate still beats one that does not beat.  Measured
    on two tones 0.05 Hz apart over 100 s: with ``R_local`` alone a single
    unmodulated tone scored 0.39 against 0.49-0.53 for beats off by 0.005-0.05
    Hz; with ``R_stat`` added it scores 1.07 against 0.55-0.68.

    Neither grades a beat rate that is far off, and the beat period is what
    the coupling sets.  ``R_rate`` does: each band's log-amplitude, detrended
    (so a decay is not a "modulation"), has a power spectrum in which a beat
    is a peak at its rate; summed over bands, the simulated and measured
    spectra are compared by the earth mover's distance along log-frequency,
    which grows with the rate error at any size -- a beat twice too fast
    costs one octave.  A trial whose measured envelope barely moves has no
    rate to compare, so its term is scaled by its envelope swing.

    Each band counts equally whatever its valid length.  The trend is
    normalised by the whole channel's std, not the trend's own, so a near-zero
    trend cannot blow up.
    """
    x = np.atleast_2d(np.asarray(q_sim, dtype=float))[:, setup['idx']]
    P = x.shape[0]
    out = np.zeros(P)
    if setup['G'] is None and setup['trend'] is None:
        return out
    x = np.where(np.isfinite(x), x, 0.0)
    X = np.fft.fft(x, setup['N'], axis=1)
    M = setup['M']
    if setup['G'] is not None:
        valid = setup['valid']
        n_valid = valid.sum(axis=1)
        eps = setup['eps'][:, None]
        rate = setup.get('rate')
        for p in range(P):
            A = np.abs(np.fft.ifft(X[p][None, :] * setup['G'], axis=1)[:, :M])
            L = np.log(A + eps)
            d = np.abs(L - setup['logA'][p])
            mu, sd = _band_stats(L, valid)
            out[p] += float(np.mean((d * valid).sum(axis=1) / n_valid))
            out[p] += float(np.mean(np.abs(mu - setup['logA_mean'][p])
                                    + np.abs(sd - setup['logA_std'][p])))
            if rate is not None and rate['weight'][p] > 0.0:
                S, _swing = _mod_spectrum(L, valid, rate['mod'], rate['use'][p])
                out[p] += rate['weight'][p] * _rate_distance(
                    S, rate['cdf'][p], rate['floor'][p], rate['mod']['du'])
    tr = setup['trend']
    if tr is not None:
        xbar = np.fft.ifft(X * tr['gain'][None, :], axis=1).real[:, :M]
        err = (xbar - tr['xbar'])[:, tr['valid']]
        out += np.sqrt(np.mean(err ** 2, axis=1)) / setup['std']
    return out


def _velocity_power(v, N):
    """Power spectrum of each row of ``v`` (P, M), mean removed, untapered,
    zero-padded to ``N``: (P, N // 2 + 1)."""
    v = np.where(np.isfinite(v), v, 0.0)
    return np.abs(np.fft.rfft(v - v.mean(axis=1, keepdims=True), N, axis=1)) ** 2


def spec_setup(t, v_meas, stride):
    """The measured side of the frequency-content term for one DOF, computed
    once.

    ``v_meas`` is (P, m): the measured velocity of every trial on the full
    time grid ``t``.  The analysis grid is every ``stride``-th sample, as for
    :func:`tf_setup`.  Each trial's velocity is transformed with its mean
    removed and NO taper -- a decaying record keeps its early, strongest
    motion, which a taper would fade out -- zero-padded to a quarter of the
    record's own frequency resolution, so a shift smaller than one bin still
    moves the spectrum.  The compared range ``f_lo``-``f_hi`` comes from the
    record (``SPEC_*``); over it, each trial's power is the distribution of its
    kinetic energy over frequency, kept as a CDF for :func:`spec_residuals`.
    A record too short to analyse, or without motion, yields no range.
    """
    t = np.asarray(t, dtype=float)
    v = np.atleast_2d(np.asarray(v_meas, dtype=float))
    P, m = v.shape
    s = max(1, int(stride))
    idx = np.arange((m - 1) // s + 1) * s
    M = idx.size
    setup = {'stride': s, 'idx': idx, 'M': M, 'band': None,
             'f_lo': np.nan, 'f_hi': np.nan}
    if M < 16:
        return setup
    tt = t[idx] - t[idx[0]]
    T = float(tt[-1])
    D = float(np.median(np.diff(tt)))
    N = _next_pow2(4 * M)
    f = np.fft.rfftfreq(N, D)
    S = _velocity_power(v[:, idx], N)
    f_lo = SPEC_LO_CYCLES / T
    pooled = np.where(f >= f_lo, S.sum(axis=0), 0.0)
    if pooled.sum() <= 0.0:
        return setup
    cum = np.cumsum(pooled)
    f_pow = float(f[min(int(np.searchsorted(cum, SPEC_POWER_FRAC * cum[-1])),
                        len(f) - 1)])
    f_hi = min(SPEC_MARGIN * f_pow, SPEC_NYQUIST * 0.5 / D)
    band = (f >= f_lo) & (f <= f_hi)
    if band.sum() < 4:
        return setup
    Sm = S[:, band]
    floor = SPEC_FLOOR * Sm.mean(axis=1)                 # per trial
    weight = (floor > 0.0).astype(float)
    Sf = Sm + floor[:, None]
    cdf = np.cumsum(Sf, axis=1) / np.maximum(Sf.sum(axis=1, keepdims=True),
                                             1e-300)
    setup.update(band=band, N=N, T=T, df=float(f[1] - f[0]), f_lo=f_lo,
                 f_hi=f_hi, cdf=cdf, floor=floor, weight=weight)
    return setup


def spec_residuals(setup, v_sim):
    """The frequency-content term ``R_f`` for every trial: a (P,) array.

    ``v_sim`` is the simulated velocity on the full time grid, as
    :func:`rollout` returns it.  Per trial, the Wasserstein-1 (earth mover's)
    distance between the simulated and the measured velocity power spectra
    over the compared range, both normalised to unit energy, times the record
    length: how far, in Hz, the simulated energy has to move to sit where the
    measured energy does, expressed as the cycles of phase a frequency error
    of that size piles up over the record.  A pure shift of the spectrum by
    ``df`` is ``T * df`` cycles; energy in the wrong mode costs its share
    times the distance between the modes.  Returned as ``ln(1 + cycles)``
    (see the module docstring).  0 for a trial whose measured velocity has no
    energy in range.
    """
    v = np.atleast_2d(np.asarray(v_sim, dtype=float))[:, setup['idx']]
    out = np.zeros(v.shape[0])
    if setup.get('band') is None:
        return out
    S = _velocity_power(v, setup['N'])[:, setup['band']] + setup['floor'][:, None]
    tot = S.sum(axis=1, keepdims=True)
    ok = (setup['weight'] > 0.0) & (tot[:, 0] > 0.0)
    cdf = np.cumsum(S, axis=1) / np.where(tot > 0.0, tot, 1.0)
    w1 = np.abs(cdf[:, :-1] - setup['cdf'][:, :-1]).sum(axis=1) * setup['df']
    return np.where(ok, np.log1p(setup['T'] * w1), 0.0)


def rollout_residuals(accel, t, states, dof, window=None, stride=1, substeps=1,
                      weights=(1.0, 0.0, 0.0, 0.0), accs=None, tf=None,
                      blowup=SIM_BLOWUP, spec=None):
    """Simulation residual of DOF ``dof`` for every trial: a (P,) array.

    ``w_q * NRMSE(q) + w_v * NRMSE(qd) + w_a * NRMSE(a) + w_tf * R_tf
    + w_f * R_f`` with ``weights = (w_q, w_v, w_a[, w_tf[, w_f]])`` (see
    :func:`check_weights`), each NRMSE taken over the simulated samples and
    normalised by that trial's measured std of the channel.  ``accs`` is the
    measured acceleration, (P, N, m); it is needed only when ``w_a > 0``.
    ``tf`` is this DOF's :func:`tf_setup` and ``spec`` its :func:`spec_setup`,
    both at the same ``stride``; each is needed only when its weight is
    positive.  With a ``window`` the NRMSEs are taken over the restarted
    windows and ``R_tf`` and ``R_f`` over a free run of their own -- an
    envelope or a spectrum needs the whole record in one piece.  The
    simulated acceleration of a window that blew up is clipped at ``blowup``
    times the largest measured one, so it too scores large but finite.  See
    :func:`rollout` for the other arguments.
    """
    return rollout_set_residuals({dof: accel}, t, states, window, stride,
                                 substeps, weights, accs,
                                 None if tf is None else {dof: tf}, blowup,
                                 specs=None if spec is None else {dof: spec})[dof]


def rollout_set_residuals(accels, t, states, window=None, stride=1, substeps=1,
                          weights=(1.0, 0.0, 0.0, 0.0), accs=None, tfs=None,
                          blowup=SIM_BLOWUP, score=None, specs=None):
    """Simulation residuals of DOFs integrated together (:func:`rollout_set`):
    ``{dof: (P,) array}`` for every DOF in ``score`` (default: all of
    ``accels``).

    Each is the DOF's own channel residual, exactly as
    :func:`rollout_residuals` computes it -- except in a trial where the
    coupled run blew up.  There every DOF of the set is charged the largest
    residual among them: once the set diverges no equation in it can be said
    to work, and one whose own channel happened to be in bounds when the
    window froze would otherwise score as if it had been fine.  ``tfs`` and
    ``specs`` map every integrated DOF to its :func:`tf_setup` and
    :func:`spec_setup`; each is needed only when its weight is positive.  The
    other arguments are as in :func:`rollout_residuals`.
    """
    w_q, w_v, w_a, w_tf, w_f = check_weights(weights)
    dofs = sorted(int(d) for d in accels)
    score = dofs if score is None else sorted(int(d) for d in score)
    if not set(score) <= set(dofs):
        raise ValueError("only integrated DOFs can be scored")
    if w_a > 0.0 and accs is None:
        raise ValueError("an acceleration weight needs the measured "
                         "acceleration (accs)")
    for w, sets, name, fn in ((w_tf, tfs, 'time-frequency', 'tf_setup'),
                              (w_f, specs, 'frequency-content', 'spec_setup')):
        if w > 0.0 and any(su is None or su['stride'] != max(1, int(stride))
                           for su in ((sets or {}).get(d) for d in dofs)):
            raise ValueError(f"the {name} weight needs every integrated DOF's "
                             f"{fn} at the same stride")
    states = np.asarray(states, dtype=float)
    _dofs, q_sim, v_sim, a_sim, sim, blown = rollout_set(
        accels, t, states, window, stride, substeps, blowup,
        with_acc=w_a > 0.0)
    q_free, v_free = q_sim, v_sim
    if (w_tf > 0.0 or w_f > 0.0) and window is not None:
        # an envelope or a spectrum needs the whole record in one run: their
        # own free run
        _d, q_free, v_free, _a, _s, blown_free = rollout_set(
            accels, t, states, None, stride, substeps, blowup, with_acc=False)
        blown = blown | blown_free

    def nrmse(sim_x, meas_x, mk):
        return (np.sqrt(np.mean((sim_x[mk] - meas_x[mk]) ** 2))
                / _scale(meas_x))

    def channel(k, d):
        iq, iv = 2 * d, 2 * d + 1
        q_d = q_sim[:, k]
        if w_a > 0.0:
            a_meas = np.asarray(accs, dtype=float)[:, d, :]
            a_lim = blowup * float(np.max(np.abs(a_meas))) + 1e-12
            a_d = np.clip(np.where(np.isfinite(a_sim[:, k]), a_sim[:, k], a_lim),
                          -a_lim, a_lim)
        r_tf = tf_residuals(tfs[d], q_free[:, k]) if w_tf > 0.0 else None
        r_f = spec_residuals(specs[d], v_free[:, k]) if w_f > 0.0 else None
        out = np.full(states.shape[0], np.nan)
        for p in range(states.shape[0]):
            mk = sim[p]
            if not mk.any():
                continue
            res = 0.0
            if w_q > 0.0:
                res += w_q * nrmse(q_d[p], states[p, iq], mk)
            if w_v > 0.0:
                res += w_v * nrmse(v_sim[p, k], states[p, iv], mk)
            if w_a > 0.0:
                res += w_a * nrmse(a_d[p], a_meas[p], mk)
            if w_tf > 0.0:
                res += w_tf * r_tf[p]
            if w_f > 0.0:
                res += w_f * r_f[p]
            out[p] = res
        return out

    shared = len(dofs) > 1 and blown.any()
    # the unscored DOFs' channels matter only for charging a blow-up
    res = {d: channel(k, d) for k, d in enumerate(dofs)
           if shared or d in score}
    if shared:
        worst = np.max(np.stack([res[d] for d in dofs]), axis=0)
        return {d: np.where(blown, worst, res[d]) for d in score}
    return {d: res[d] for d in score}
