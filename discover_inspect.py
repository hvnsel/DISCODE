"""
discover_inspect.py
===================
Look at a record the way the simulation reward sees it -- before training, to
know what the search is up against, or after, to see where a discovered
equation goes wrong.

For one trial it plots a row per DOF:

1. the displacement over time, measured and -- with ``EXPRS`` -- simulated;
2. the measured wavelet map: the amplitude of each frequency over time, in dB
   (analytic Morlet, the time-frequency term's own transform).  A beat is a
   band that swells and fades, with dark notches where it cancels; a mode
   that dies out is a band that goes dark.  Dashed lines bound the band
   centres the time-frequency term compares; the thin white curves mark
   where the map starts to see the record's ends;
3. the same map of the simulation, on the same colour scale (with ``EXPRS``);
4. the velocity power spectrum -- where the kinetic energy sits -- measured
   and simulated, over the span the frequency-content term compares, each
   normalised to unit energy there (smoothed over one resolution cell for
   display; the term itself compares the raw spectra).

and prints, for every trial: each DOF's fastest motion and how many samples
a cycle of it gets, the windows ``sim_window='auto'`` picks, the strongest
spectral peaks, the envelope swing and the beat period -- the run's startup
report -- and, with ``EXPRS``, every term of the simulation residual.

USAGE
-----
1. Point the config at the record and at the trim / downsample settings of the
   run: they set the time grid everything is compared on.
2. Optionally paste one expression per DOF into ``EXPRS``.
3. ``python discover_inspect.py``
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')

import discover_rollout as ro
from discover_analysis import expr_accels, record_states

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

SOURCE = 'experimental'          # 'experimental' | 'simulated'

# ── experimental ───────────────────────────────────────────────────────────
MAT_PATH          = "AllData_ProcessedNOhit.mat"
VAR_NAMES         = ['q1', 'q2']
# Must match the training run you want to look at.
TRIM_FRONT        = 1000
TRIM_BACK         = 125_000
DESIRED_TIMESTEPS = 500

# ── simulated ──────────────────────────────────────────────────────────────
SIM_KEY = 'coupled_beats'        # a key from discover_mdof_sim / discover_sdof_sim
SIM_OVERRIDES = {}               # e.g. {'n_traj': 2, 'n_pts': 4000}

# ── common ─────────────────────────────────────────────────────────────────
TRIAL      = 4                   # the trial plotted (0-based, as the run prints)
SIM_WINDOW = 'auto'              # as in the run: None | seconds | 'auto'
COUPLED    = False               # False: each DOF against the record, as the
                                 # reward scores it; True: integrated together,
                                 # as the plot scripts run them
F_RANGE    = None                # (f_min, f_max) Hz for the maps and spectra;
                                 # None = 3 cycles per record .. 0.8 x Nyquist
SHOW       = True                # False: only save the figure
OUT        = None                # None = '<record>_trial<k>_inspect.png'

# One physical expression per DOF, or none to look at the record alone.
EXPRS = [
]

# ═══════════════════════════════════════════════════════════════════════════

MEAS_COLOR, SIM_COLOR = '#1f77b4', '#d62728'
DB_FLOOR = -40.0                 # the maps' colour range, dB below their peak
VOICES = 16                      # wavelet frequencies per octave


def build_system():
    """The record the run trained on."""
    if SOURCE == 'experimental':
        from discover_data import load_mat_data
        return load_mat_data(MAT_PATH, trim_timesteps_front=TRIM_FRONT,
                             trim_timesteps_back=TRIM_BACK,
                             desired_timesteps=DESIRED_TIMESTEPS,
                             var_names=VAR_NAMES, plot_data=False)
    if SOURCE == 'simulated':
        from discover_data import build_truth_system
        from discover_mdof_sim import _REGISTRY as MDOF_REG
        from discover_sdof_sim import _REGISTRY as SDOF_REG
        reg = {**SDOF_REG, **MDOF_REG}
        if SIM_KEY not in reg:
            raise ValueError(f"Unknown SIM_KEY '{SIM_KEY}'. "
                             f"Choose from {sorted(reg)}.")
        return build_truth_system(reg[SIM_KEY](), verbose=False, **SIM_OVERRIDES)
    raise ValueError(f"SOURCE must be 'experimental' or 'simulated', "
                     f"got {SOURCE!r}")


def wavelet_map(t, x, freqs):
    """Amplitude of each frequency in ``freqs`` over time, (K, m): the analytic
    Morlet bank of :func:`discover_rollout.tf_setup`, so a cosine of
    amplitude ``a`` at ``f_k`` reads ``a`` in band ``k``."""
    x = np.asarray(x, dtype=float)
    m = x.size
    D = float(np.median(np.diff(t)))
    N = 1 << int(np.ceil(np.log2(max(2 * m, 2))))
    X = np.fft.fft(np.where(np.isfinite(x), x - np.nanmean(x), 0.0), N)
    f = np.fft.fftfreq(N, D)
    fk = np.asarray(freqs, dtype=float)[:, None]
    G = 2.0 * np.exp(-0.5 * (ro.TF_OMEGA0 * (f[None, :] - fk) / fk) ** 2) * (f > 0)
    return np.abs(np.fft.ifft(X[None, :] * G, axis=1)[:, :m])


def velocity_spectrum(t, v):
    """``(f, power)`` of a velocity record as the frequency-content term
    takes it: mean removed, untapered, zero-padded."""
    D = float(np.median(np.diff(t)))
    N = 1 << int(np.ceil(np.log2(max(4 * len(v), 2))))
    v = np.where(np.isfinite(v), v, 0.0)
    return (np.fft.rfftfreq(N, D),
            np.abs(np.fft.rfft(v - v.mean(), N)) ** 2)


def report(system, t, states, accs, stride, tfs, specs):
    """The run's startup report, plus the sampling and the windows."""
    N, dt = system.n_dof, float(np.median(np.diff(t)))
    print(f"\n[record] {system.name}: {system.n_trials} trials of "
          f"{t[-1] - t[0]:.4g} s at {1.0 / dt:.4g} Hz "
          f"(compared every {stride} sample(s))")
    for d in range(N):
        w = ro.fastest_omega(states, accs, d)
        if w <= 0.0:
            print(f"  DOF {d} ({system.var_names[d]}): no motion")
            continue
        n = 2.0 * np.pi / (w * dt)
        win = ro.auto_window(t, states, accs, dofs=d)
        print(f"  DOF {d} ({system.var_names[d]}): fastest motion "
              f"~{w / (2 * np.pi):.3g} Hz, {n:.3g} samples per cycle"
              + ("  <- coarse: raise DESIRED_TIMESTEPS" if n < 10 else "")
              + f"; 'auto' window {win:.3g} s")
    if N > 1:
        both = ro.auto_window(t, states, accs)
        print(f"  DOFs simulated together restart every {both:.3g} s")
    for d in range(N):
        tf, sp = tfs[d], specs[d]
        print(f"\n  DOF {d}: time-frequency bands "
              + (f"{tf['f_lo']:.3g}-{tf['f_hi']:.3g} Hz" if tf['G'] is not None
                 else "none (no oscillation)")
              + "; frequency content "
              + (f"{sp['f_lo']:.3g}-{sp['f_hi']:.3g} Hz"
                 if sp['band'] is not None else "none"))
        if tf['G'] is None:
            continue
        freqs, rel = ro.spectral_peaks(t, states[:, 2 * d, :], stride,
                                       tf['f_lo'], tf['f_hi'])
        print("     trial   peak Hz   2nd peak Hz (rel)   envelope swing   "
              "beat period s")
        for p in range(states.shape[0]):
            f2 = freqs[p, 1]
            second = (f"{f2:9.4g} ({rel[p, 1]:4.2f})" if np.isfinite(f2)
                      else f"{'-':>16s}")
            beat = (f"{1.0 / tf['rate']['nu_peak'][p]:13.4g}"
                    if tf['rate'] is not None and tf['swing'][p] >= 0.1
                    else f"{'-':>13s}")
            mark = "   <- plotted" if p == TRIAL else ""
            print(f"     {p:5d} {freqs[p, 0]:9.4g}   {second}   "
                  f"{tf['swing'][p]:14.3f}   {beat}{mark}")


def simulate(accels, t, states, stride, substeps):
    """Free runs of every trial: ``(q, v)``, each (P, N, m), simulated
    together (``COUPLED``) or each DOF against the record."""
    N = len(accels)
    if COUPLED:
        _d, q, v, _a, _s, _b = ro.rollout_set(dict(enumerate(accels)), t,
                                              states, None, stride, substeps,
                                              with_acc=False)
        return q, v
    runs = [ro.rollout(accels[d], t, states, d, None, stride, substeps,
                       with_acc=False) for d in range(N)]
    return (np.stack([r[0] for r in runs], axis=1),
            np.stack([r[1] for r in runs], axis=1))


def residual_terms(accels, t, states, accs, stride, substeps, tfs, specs):
    """Each term of the simulation residual, per DOF and trial: ``{dof:
    (P, 5)}``, pointwise terms on ``SIM_WINDOW``, the envelope and spectrum on
    a free run, the partners measured or simulated as ``COUPLED`` says."""
    N, P = len(accels), states.shape[0]
    out = {d: np.full((P, 5), np.nan) for d in range(N)}
    for k in range(5):
        w = tuple(float(j == k) for j in range(5))
        if COUPLED:
            win = (ro.auto_window(t, states, accs) if SIM_WINDOW == 'auto'
                   else SIM_WINDOW)
            res = ro.rollout_set_residuals(dict(enumerate(accels)), t, states,
                                           window=win, stride=stride,
                                           substeps=substeps, weights=w,
                                           accs=accs, tfs=tfs, specs=specs)
        else:
            res = {}
            for d in range(N):
                win = (ro.auto_window(t, states, accs, dofs=d)
                       if SIM_WINDOW == 'auto' else SIM_WINDOW)
                res[d] = ro.rollout_residuals(accels[d], t, states, d,
                                              window=win, stride=stride,
                                              substeps=substeps, weights=w,
                                              accs=accs, tf=tfs[d],
                                              spec=specs[d])
        for d in range(N):
            out[d][:, k] = res[d]
    return out


def main():
    import matplotlib
    if not SHOW:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    system = build_system()
    N, P = system.n_dof, system.n_trials
    if not 0 <= TRIAL < P:
        raise ValueError(f"TRIAL={TRIAL} but the record has {P} trials "
                         f"(0..{P - 1})")
    t = np.asarray(system.time, dtype=float)
    states, accs = record_states(system, range(P))
    stride, substeps = ro.auto_steps(t, states, accs)
    tfs = {d: ro.tf_setup(t, states[:, 2 * d, :], stride) for d in range(N)}
    specs = {d: ro.spec_setup(t, states[:, 2 * d + 1, :], stride)
             for d in range(N)}
    report(system, t, states, accs, stride, tfs, specs)

    exprs = [e for e in EXPRS if str(e).strip()]
    q_sim = v_sim = None
    if exprs:
        accels = expr_accels(exprs, system.var_names)
        q_sim, v_sim = simulate(accels, t, states, stride, substeps)
        terms = residual_terms(accels, t, states, accs, stride, substeps,
                               tfs, specs)
        how = ('integrated together' if COUPLED
               else 'each against the measured partner')
        print(f"\n[simulation] residual terms per trial, {how} (pointwise on "
              f"sim_window={SIM_WINDOW!r}, envelope and spectrum on a free "
              f"run; lower is better):")
        for d in range(N):
            print(f"  DOF {d}   trial   disp NRMSE   vel NRMSE   acc NRMSE"
                  f"      R_tf       R_f")
            for p in range(P):
                row = "  ".join(f"{x:10.3f}" for x in terms[d][p])
                print(f"           {p:5d}  {row}"
                      + ("   <- plotted" if p == TRIAL else ""))

    # The grid everything is compared on.
    idx = np.arange(0, t.size, stride)
    tg = t[idx]
    T = float(tg[-1] - tg[0])
    fs = 1.0 / float(np.median(np.diff(tg)))
    f_min, f_max = F_RANGE if F_RANGE else (3.0 / T, 0.4 * fs)
    n_f = max(8, int(np.ceil(VOICES * np.log2(f_max / f_min))) + 1)
    freqs = f_min * (f_max / f_min) ** (np.arange(n_f) / (n_f - 1))
    edge = np.sqrt(2.0) * ro.TF_OMEGA0 / (2.0 * np.pi * freqs)

    n_col = 4 if q_sim is not None else 3
    fig, axes = plt.subplots(N, n_col, figsize=(4.6 * n_col, 3.3 * N),
                             squeeze=False)
    for d in range(N):
        name = system.var_names[d]
        ax_t, ax_m = axes[d, 0], axes[d, 1]
        ax_s = axes[d, 2] if n_col == 4 else None
        ax_f = axes[d, -1]
        x_meas = states[TRIAL, 2 * d, idx]

        ax_t.plot(tg, x_meas, color=MEAS_COLOR, lw=0.9, label='measured')
        if q_sim is not None:
            ax_t.plot(tg, q_sim[TRIAL, d, idx], color=SIM_COLOR, lw=0.9,
                      ls='--', label='simulated')
        ax_t.set_title(f"DOF {d} ({name}) -- trial {TRIAL}", fontsize=9)
        ax_t.set_xlabel('time (s)', fontsize=8)
        ax_t.set_ylabel(f'{name} (displacement)', fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(fontsize=7, loc='upper right')

        A_meas = wavelet_map(tg, x_meas, freqs)
        ref = float(A_meas.max()) if A_meas.max() > 0 else 1.0
        maps = [(ax_m, A_meas, 'measured')]
        if ax_s is not None:
            maps.append((ax_s, wavelet_map(tg, q_sim[TRIAL, d, idx], freqs),
                         'simulated'))
        for ax, A, label in maps:
            db = 20.0 * np.log10(np.maximum(A / ref, 10 ** (DB_FLOOR / 20)))
            im = ax.pcolormesh(tg, freqs, db, shading='auto', cmap='magma',
                               vmin=DB_FLOOR, vmax=0.0, rasterized=True)
            ax.set_yscale('log')
            ax.set_ylim(freqs[0], freqs[-1])
            ax.plot(tg[0] + edge, freqs, color='white', lw=0.6, alpha=0.7)
            ax.plot(tg[-1] - edge, freqs, color='white', lw=0.6, alpha=0.7)
            ax.set_xlim(tg[0], tg[-1])
            tf = tfs[d]
            if tf['G'] is not None:
                for fb in (tf['f_lo'], tf['f_hi']):
                    ax.axhline(fb, color='white', lw=0.8, ls='--', alpha=0.9)
            ax.set_title(f"{label}: amplitude by frequency over time",
                         fontsize=9)
            ax.set_xlabel('time (s)', fontsize=8)
            ax.set_ylabel('frequency (Hz)', fontsize=8)
            fig.colorbar(im, ax=ax, pad=0.01).set_label('dB re. measured peak',
                                                        fontsize=7)

        sp = specs[d]
        f_v, S_meas = velocity_spectrum(tg, states[TRIAL, 2 * d + 1, idx])
        lo, hi = ((sp['f_lo'], sp['f_hi']) if sp['band'] is not None
                  else (f_min, f_max))
        span = (f_v >= lo) & (f_v <= hi)
        # one resolution cell (1/T) of the zero-padded grid
        cell = max(1, int(round(1.0 / (T * (f_v[1] - f_v[0])))))

        def share(S):
            S = np.convolve(S, np.ones(cell) / cell, mode='same')
            tot = S[span].sum()
            return S / tot if tot > 0 else S
        ax_f.semilogy(f_v[span], share(S_meas)[span], color=MEAS_COLOR,
                      lw=0.9, label='measured')
        if v_sim is not None:
            _f, S_sim = velocity_spectrum(tg, v_sim[TRIAL, d, idx])
            ax_f.semilogy(f_v[span], share(S_sim)[span], color=SIM_COLOR,
                          lw=0.9, ls='--', label='simulated')
        top = share(S_meas)[span].max() if span.any() else 1.0
        ax_f.set_ylim(top * 1e-6, top * 3.0)
        ax_f.set_xlim(lo, hi)
        ax_f.set_title('velocity power over the compared span: where the '
                       'kinetic energy sits', fontsize=9)
        ax_f.set_xlabel('frequency (Hz)', fontsize=8)
        ax_f.set_ylabel('share of energy per bin', fontsize=8)
        ax_f.grid(True, alpha=0.3, which='both')
        ax_f.legend(fontsize=7, loc='upper right')
        for ax in axes[d]:
            ax.tick_params(labelsize=7)

    fig.suptitle(f"{system.name} -- trial {TRIAL}: what the simulation "
                 f"reward sees", fontsize=11)
    fig.tight_layout()
    out = OUT or f"{system.name}_trial{TRIAL}_inspect.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved  {out}")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == '__main__':
    main()
