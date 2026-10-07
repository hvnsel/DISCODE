"""
discover_diagnose.py
====================
Is it the data, the setup, or the search?  Three checks on a record, and your
run's equations scored alongside.

1. **The data.**  Do the channels agree with each other -- is the measured
   velocity the derivative of the displacement, and the acceleration that of
   the velocity -- and how many samples does each DOF's fastest motion get?
   Checked in the frequency domain at each DOF's main peaks, as a gain and a
   phase (ideal: 1.00 and 0 deg), so a filtered or delayed channel shows up
   directly.  Candidates are fitted to the acceleration and judged on the
   simulated displacement: if those channels disagree, no equation can
   satisfy both.

2. **What the search space can do.**  With integer powers up to 3, every
   candidate is a weighted sum of a fixed list of monomials -- 34 for two
   DOFs, plus a constant.  Sparse regression on that list (SINDy-style: least
   squares, drop the small terms, refit) finds the best equations of every
   size directly, with no search; each is then simulated and scored with the
   training reward, and the best is tuned on the simulation as the trainer
   tunes its own.

   - No equation from the list simulates well: the search cannot succeed
     whatever its settings -- the data, or physics these states do not hold
     (forcing, friction, a mode that is not measured).
   - Some do, and your run's best is far below them: the search is what
     fails -- its settings or the method.

   With ``CHANNELS = 'measured'`` it is fitted twice: to the file's
   acceleration (what the training's constant fit leans on) and to the
   derivative of the file's velocity.  If the second simulates clearly better,
   the acceleration channel is biased and that bias is what holds the
   constants back.  With ``'disp'`` (the loader's default) both are
   derivatives of the displacement and agree by construction; the data check
   above still reads the file's own channels.

3. **Your equations**, scored the same way, next to the list's best.

USAGE
-----
1. Point the config at the record, with the trim / downsample settings and the
   reward settings of the run.
2. Paste your run's best equation per DOF into ``EXPRS`` (optional).
3. ``python discover_diagnose.py``  -- a couple of minutes.
"""

from __future__ import annotations

import sys
from itertools import combinations_with_replacement

import numpy as np

sys.path.insert(0, '.')

import discover_rollout as ro
from discover_analysis import clean_expr, expr_accels, record_states
from discover_data import time_derivative

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

SOURCE = 'experimental'          # 'experimental' | 'simulated'

# ── experimental ───────────────────────────────────────────────────────────
MAT_PATH          = "AllData_ProcessedNOhit.mat"
VAR_NAMES         = ['q1', 'q2']
# Must match the training run.
TRIM_FRONT        = 1000
TRIM_BACK         = 125_000
DESIRED_TIMESTEPS = 2000

# Velocity and acceleration, as in the run: 'disp' (the displacement's
# derivatives, the loader's default) or 'measured' (the file's own).
CHANNELS = 'disp'

# ── simulated ──────────────────────────────────────────────────────────────
SIM_KEY = 'coupled_beats'        # a key from discover_mdof_sim / discover_sdof_sim
SIM_OVERRIDES = {}

# ── the run's reward settings ──────────────────────────────────────────────
SIM_WINDOW  = 'auto'
SIM_WEIGHTS = (0.2, 0.2, 0.2, 0.4)
TRIAL_DECAY = 0.5

# ── the list ───────────────────────────────────────────────────────────────
MAX_DEGREE  = 3                  # monomials up to this total degree
MAX_TERMS   = 8                  # largest sparse equation: the run's max_terms,
                                 #   so the list holds only what the search can write
TUNE_EVALS  = 60                 # simulations to tune each DOF's best (0 = off)

# Your run's best equation per DOF (optional).
EXPRS = [
]

# ═══════════════════════════════════════════════════════════════════════════


def build_system(channels=None):
    """The record the run trained on (``channels`` overrides ``CHANNELS``)."""
    if SOURCE == 'experimental':
        from discover_data import load_mat_data
        return load_mat_data(MAT_PATH, trim_timesteps_front=TRIM_FRONT,
                             trim_timesteps_back=TRIM_BACK,
                             desired_timesteps=DESIRED_TIMESTEPS,
                             var_names=VAR_NAMES, plot_data=False,
                             channels=CHANNELS if channels is None else channels)
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


# ── 1. the data ─────────────────────────────────────────────────────────────
def channel_agreement(t, x, y, f_peaks):
    """Gain and phase (deg) of channel ``y`` against d/dt of channel ``x`` at
    each frequency in ``f_peaks``, pooled over trials (rows): ideal 1, 0.
    Estimated from the cross-spectrum, so it needs no numerical derivative."""
    D = float(np.median(np.diff(t)))
    n = x.shape[1]
    N = 1 << int(np.ceil(np.log2(4 * n)))
    win = np.hanning(n)
    X = np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * win, N, axis=1)
    Y = np.fft.rfft((y - y.mean(axis=1, keepdims=True)) * win, N, axis=1)
    f = np.fft.rfftfreq(N, D)
    out = []
    for fp in f_peaks:
        k = int(np.argmin(np.abs(f - fp)))
        H = (Y[:, k] * np.conj(X[:, k])).sum() / ((np.abs(X[:, k]) ** 2).sum()
                                                 + 1e-300)
        ideal = 2j * np.pi * f[k]
        r = H / ideal
        out.append((float(f[k]), float(np.abs(r)),
                    float(np.degrees(np.angle(r)))))
    return out


def main_peaks(t, v, share=0.1, n_max=3):
    """Up to ``n_max`` frequencies of the pooled velocity spectrum's local
    maxima holding at least ``share`` of the strongest, strongest first, at
    least 5% apart -- where the channels carry enough motion to compare."""
    D = float(np.median(np.diff(t)))
    n = v.shape[1]
    N = 1 << int(np.ceil(np.log2(4 * n)))
    S = (np.abs(np.fft.rfft((v - v.mean(axis=1, keepdims=True)) * np.hanning(n),
                            N, axis=1)) ** 2).sum(axis=0)
    f = np.fft.rfftfreq(N, D)
    ok = f >= 2.0 / (t[-1] - t[0])
    loc = np.flatnonzero(ok[1:-1] & (S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])) + 1
    loc = loc[S[loc] >= share * S[loc].max()] if loc.size else loc
    out = []
    for k in loc[np.argsort(S[loc])[::-1]]:
        if all(abs(f[k] - g) > 0.05 * g for g in out):
            out.append(float(f[k]))
        if len(out) == n_max:
            break
    return out


def check_data(system, t, states, accs):
    """Print the sampling and the channel agreement; return the flags."""
    N, P = system.n_dof, system.n_trials
    dt = float(np.median(np.diff(t)))
    flags = []
    print(f"\n[1. data] {system.name}: {P} trials of {t[-1] - t[0]:.4g} s at "
          f"{1.0 / dt:.4g} Hz")
    for d in range(N):
        name = system.var_names[d]
        w = ro.fastest_omega(states, accs, d)
        n_cyc = 2.0 * np.pi / (w * dt) if w > 0 else np.inf
        coarse = n_cyc < 10
        print(f"  DOF {d} ({name}): fastest motion ~{w / (2 * np.pi):.3g} Hz, "
              f"{n_cyc:.3g} samples per cycle"
              + ("  <- coarse: raise DESIRED_TIMESTEPS" if coarse else ""))
        if coarse:
            flags.append(f"DOF {d} has only {n_cyc:.2g} samples per cycle")
        f_main = main_peaks(t, states[:, 2 * d + 1, :])
        vq = channel_agreement(t, states[:, 2 * d, :], states[:, 2 * d + 1, :],
                               f_main)
        av = channel_agreement(t, states[:, 2 * d + 1, :], accs[:, d, :],
                               f_main)
        for (f, g1, p1), (_f, g2, p2) in zip(vq, av):
            print(f"     at {f:7.3g} Hz:  velocity vs d/dt displacement gain "
                  f"{g1:.3f} phase {p1:+6.1f} deg;  acceleration vs d/dt "
                  f"velocity gain {g2:.3f} phase {p2:+6.1f} deg")
            for what, g, p in (('velocity', g1, p1), ('acceleration', g2, p2)):
                if abs(g - 1.0) > 0.05 or abs(p) > 5.0:
                    flags.append(f"DOF {d} {what} disagrees with the derivative "
                                 f"of the channel below it at {f:.3g} Hz "
                                 f"(gain {g:.2f}, phase {p:+.0f} deg)")
    return flags


# ── 2. the list ─────────────────────────────────────────────────────────────
def monomials(n_vars, max_degree):
    """Exponent tuples of every monomial up to ``max_degree``, constant first."""
    out = [(0,) * n_vars]
    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), deg):
            e = [0] * n_vars
            for i in combo:
                e[i] += 1
            out.append(tuple(e))
    return out


def theta(Z, exps):
    """Library matrix: one column per monomial of the columns of ``Z``."""
    return np.stack([np.prod(Z ** np.array(e)[None, :], axis=1) for e in exps],
                    axis=1)


def term_name(e, names):
    parts = [n if p == 1 else f"{n}**{p}" for n, p in zip(names, e) if p]
    return '*'.join(parts) if parts else '1'


def stlsq_path(Th, y, max_terms, n_lam=60):
    """Sequentially thresholded least squares over a sweep of thresholds:
    ``{n_terms: coef}``, the best fit found for each equation size, plus
    ``'dense'``.  Columns are scaled to unit RMS so one threshold fits all."""
    norm = np.sqrt(np.mean(Th ** 2, axis=0)) + 1e-300
    A = Th / norm
    dense = np.linalg.lstsq(A, y, rcond=None)[0]
    best = {'dense': (dense / norm, float(np.sum((A @ dense - y) ** 2)))}
    for lam in np.geomspace(1e-4, 1.0, n_lam) * float(np.std(y)):
        xi = dense.copy()
        for _ in range(20):
            small = np.abs(xi) < lam
            xi[small] = 0.0
            big = ~small
            if not big.any():
                break
            new = np.zeros_like(xi)
            new[big] = np.linalg.lstsq(A[:, big], y, rcond=None)[0]
            if np.array_equal(np.abs(new) < lam, small):
                xi = new
                break
            xi = new
        k = int(np.count_nonzero(xi))
        if 1 <= k <= max_terms:
            sse = float(np.sum((A @ xi - y) ** 2))
            if k not in best or sse < best[k][1]:
                best[k] = (xi / norm, sse)
    return best


def as_expr(dof, coefs, exps, names, var_names):
    """A physical equation string the scorers and plot scripts accept."""
    body = ' '.join(f"{c:+.6g}*{term_name(e, names)}"
                    for c, e in zip(coefs, exps) if c != 0.0) or '0'
    return f"{var_names[dof]}ddot = {body}"


class Scorer:
    """The training's simulation reward for physical equation strings, each
    DOF alone (partners read off the record) or several together."""

    def __init__(self, system, t, states, accs):
        self.system, self.t, self.states, self.accs = system, t, states, accs
        self.N = system.n_dof
        self.stride, self.sub = ro.auto_steps(t, states, accs)
        self.w = ro.check_weights(SIM_WEIGHTS)
        self.tfs = ({d: ro.tf_setup(t, states[:, 2 * d, :], self.stride)
                     for d in range(self.N)} if self.w[3] > 0 else {})
        self.specs = ({d: ro.spec_setup(t, states[:, 2 * d + 1, :], self.stride)
                       for d in range(self.N)} if self.w[4] > 0 else {})

    def _accels(self, exprs):
        return expr_accels(exprs, self.system.var_names)

    def _window(self, dofs):
        if SIM_WINDOW == 'auto':
            return ro.auto_window(self.t, self.states, self.accs, dofs=dofs)
        return SIM_WINDOW

    def alone(self, d, expr):
        """``(reward, per-trial residuals)`` of DOF ``d``'s equation."""
        exprs = [expr if j == d else '0' for j in range(self.N)]
        res = ro.rollout_residuals(self._accels(exprs)[d], self.t, self.states,
                                   d, window=self._window(d),
                                   stride=self.stride, substeps=self.sub,
                                   weights=SIM_WEIGHTS, accs=self.accs,
                                   tf=self.tfs.get(d), spec=self.specs.get(d))
        if not np.all(np.isfinite(res)):
            return 0.0, res
        return 1.0 / (1.0 + ro.combine_trials(res, TRIAL_DECAY)), res

    def together(self, exprs):
        """Each DOF's reward with every DOF integrated together."""
        res = ro.rollout_set_residuals(
            dict(enumerate(self._accels(exprs))), self.t, self.states,
            window=self._window(None), stride=self.stride, substeps=self.sub,
            weights=SIM_WEIGHTS, accs=self.accs, tfs=self.tfs or None,
            specs=self.specs or None)
        return [1.0 / (1.0 + ro.combine_trials(res[d], TRIAL_DECAY))
                if np.all(np.isfinite(res[d])) else 0.0 for d in range(self.N)]


def r2s(system, states, targets, d, expr):
    """R^2 of an equation's acceleration on the measured states against each
    target (measured acceleration, d/dt velocity): ``{label: R^2}``."""
    N = system.n_dof
    a_hat = expr_accels([expr if j == d else '0' for j in range(N)],
                        system.var_names)[d]
    pred = np.concatenate([np.broadcast_to(a_hat(states[p]),
                                           states[p, 0].shape).astype(float)
                           for p in range(states.shape[0])])
    out = {}
    for label, tgt in targets.items():
        meas = tgt[:, d, :].ravel()
        out[label] = (float('nan') if not np.all(np.isfinite(pred)) else
                      1.0 - float(np.sum((pred - meas) ** 2)
                                  / np.sum((meas - meas.mean()) ** 2)))
    return out


def tune(scorer, d, coefs, exps, names, var_names, evals):
    """Powell on the coefficients (each within +-50%), maximising the
    simulation reward -- what the trainer's sim_refine does."""
    from scipy.optimize import minimize
    c0 = np.asarray(coefs, dtype=float)
    on = np.flatnonzero(c0 != 0.0)
    best = {'r': scorer.alone(d, as_expr(d, c0, exps, names, var_names))[0],
            'c': c0}

    def loss(x):
        c = c0.copy()
        c[on] = c0[on] * (1.0 + x)
        r = scorer.alone(d, as_expr(d, c, exps, names, var_names))[0]
        if r > best['r']:
            best.update(r=r, c=c)
        return 1.0 / max(r, 1e-12) - 1.0
    if on.size and evals > 0:
        minimize(loss, np.zeros(on.size), method='Powell',
                 bounds=[(-0.5, 0.5)] * on.size,
                 options={'maxfev': int(evals), 'xtol': 1e-3, 'ftol': 1e-4})
    return best['r'], best['c']


def main():
    system = build_system()
    N, P = system.n_dof, system.n_trials
    t = np.asarray(system.time, dtype=float)
    states, accs = record_states(system, range(P))
    names = [n for v in system.var_names for n in (v, v + 'dot')]

    file_system = system
    if SOURCE == 'experimental' and CHANNELS != 'measured':
        file_system = build_system('measured')
    flags = check_data(file_system, t, *record_states(file_system, range(P)))
    if file_system is not system:
        print("  (the file's own channels, above; with CHANNELS='disp' the fit "
              "and the reward below use the displacement's derivatives, which "
              "agree by construction)")

    exps = monomials(2 * N, MAX_DEGREE)
    X = np.concatenate([states[p].T for p in range(P)])          # (rows, 2N)
    scale = X.std(axis=0) + 1e-300
    Th = theta(X / scale, exps)
    col_scale = np.array([np.prod(scale ** np.array(e)) for e in exps])
    scorer = Scorer(system, t, states, accs)
    targets = {'acceleration': accs}
    dv = np.stack([time_derivative(states[:, 2 * d + 1, :], t, axis=-1)
                   for d in range(N)], axis=1)
    if not np.allclose(dv, accs, rtol=0.0,
                       atol=1e-6 * float(np.abs(accs).max() + 1e-300)):
        targets['d/dt velocity'] = dv          # the channels disagree: fit both

    print(f"\n[2. the list] every monomial of {', '.join(names)} up to degree "
          f"{MAX_DEGREE} ({len(exps) - 1}) and a constant; sparse fits of each "
          f"size up to {MAX_TERMS} terms, simulated and scored with "
          f"sim_window={SIM_WINDOW!r}, sim_weights={tuple(SIM_WEIGHTS)}, "
          f"trial_decay={TRIAL_DECAY}; 'dense' (every monomial) is for "
          f"reference only")
    best_list, by_target = {}, {}
    for d in range(N):
        print(f"\n  DOF {d} ({system.var_names[d]})")
        print("    fitted to         terms   "
              + "".join(f"{'R^2 vs ' + lb:>22s}" for lb in targets)
              + "   sim reward alone   worst trial residual")
        rows = []
        for label, tgt in targets.items():
            y = np.concatenate([tgt[p, d] for p in range(P)])
            ys = float(np.std(y)) + 1e-300
            path = stlsq_path(Th, y / ys, MAX_TERMS)
            for k in sorted(path, key=lambda z: (z == 'dense', z)):
                coefs = path[k][0] * ys / col_scale
                expr = as_expr(d, coefs, exps, names, system.var_names)
                r, res = scorer.alone(d, expr)
                fit = r2s(system, states, targets, d, expr)
                n_terms = int(np.count_nonzero(coefs))
                rows.append((r, label, n_terms, coefs, expr))
                print(f"    {label:16s} {str(k):>6s}   "
                      + "".join(f"{fit[lb]:22.4f}" for lb in targets)
                      + f"   {r:16.4f}   {np.nanmax(res):20.3f}")
        # what the search can write: at most MAX_TERMS terms
        fits = [row for row in rows if row[2] <= MAX_TERMS] or rows
        by_target[d] = {lb: max((r for r, l2, *_ in fits if l2 == lb),
                                default=0.0)
                        for lb in targets}
        # the sparsest equation within 0.01 of the best reward
        r_top = max(r for r, *_ in fits)
        r_best, label, n_terms, coefs, expr = min(
            (row for row in fits if row[0] >= r_top - 0.01),
            key=lambda z: (z[2], -z[0]))
        if TUNE_EVALS > 0:
            r_t, c_t = tune(scorer, d, coefs, exps, names, system.var_names,
                            TUNE_EVALS)
            print(f"    best of the list ({n_terms} terms, fitted to {label}) "
                  f"tuned on the simulation: {r_best:.4f} -> {r_t:.4f}")
            if r_t > r_best:
                r_best, coefs = r_t, c_t
                expr = as_expr(d, coefs, exps, names, system.var_names)
        best_list[d] = (r_best, expr)
        print(f"    best: {expr}")

    together = scorer.together([best_list[d][1] for d in range(N)])
    print("\n  the list's best per DOF, simulated together: "
          + ", ".join(f"DOF {d} {r:.4f}" for d, r in enumerate(together)))

    yours = None
    exprs = [e for e in EXPRS if str(e).strip()]
    if exprs:
        if len(exprs) != N:
            raise ValueError(f"EXPRS needs one equation per DOF ({N})")
        print("\n[3. your equations] scored the same way")
        yours = []
        for d in range(N):
            r, res = scorer.alone(d, clean_expr(exprs[d]))
            fit = r2s(system, states, targets, d, exprs[d])
            yours.append(r)
            print(f"  DOF {d}: R^2 "
                  + ", ".join(f"vs {lb} {fit[lb]:.4f}" for lb in targets)
                  + f"; sim reward alone {r:.4f}, worst trial residual "
                    f"{np.nanmax(res):.3f}")
        tog = scorer.together(exprs)
        print("  simulated together: "
              + ", ".join(f"DOF {d} {r:.4f}" for d, r in enumerate(tog)))

    print("\n[reading]")
    if flags:
        print("  data: " + "\n        ".join(flags))
        if file_system is not system:
            print("        (the file's own channels: with CHANNELS='disp' the "
                  "run uses the displacement's derivatives instead, so a "
                  "velocity or acceleration flag no longer reaches the fit or "
                  "the reward)")
    else:
        print("  data: the channels agree at the main peaks (within 5% and 5 "
              "deg) and every DOF gets 10+ samples per cycle")
    for d in range(N):
        r_list = best_list[d][0]
        line = f"  DOF {d}: the list's best simulates at {r_list:.3f}"
        if yours is not None:
            line += f", your run's at {yours[d]:.3f}"
            if r_list > yours[d] + 0.1:
                line += (" -> an equation the search space holds does clearly "
                         "better: the search is falling short (settings or "
                         "method)")
            elif r_list < 0.5:
                line += (" -> nothing in the search space does well either: "
                         "the limit is the data or physics these states do "
                         "not hold, not the search")
            else:
                line += (" -> the search found about what the space allows")
        elif r_list < 0.5:
            line += (" -> nothing in the search space does well: the data, or "
                     "physics these states do not hold")
        print(line)
        if 'd/dt velocity' not in targets:
            continue
        r_acc, r_dv = (by_target[d]['acceleration'],
                       by_target[d]['d/dt velocity'])
        if r_dv > r_acc + 0.1:
            print(f"  DOF {d}: fitted to d/dt velocity the list simulates at "
                  f"{r_dv:.3f}, fitted to the measured acceleration at "
                  f"{r_acc:.3f} -> the acceleration channel is biased, and the "
                  f"training's constant fit leans on it (w_acc): its constants "
                  f"come out wrong before the search even starts")


if __name__ == '__main__':
    main()
