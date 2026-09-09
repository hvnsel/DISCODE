"""
lono_comparison_plot.py
=======================
Compare the PUBLISHED LO-NO equations of motion against a DISCOVER-discovered
pair, on every trial of the experimental record.

Truth
-----
Wang & Moore (2021) Eq. (7), with the parameters identified by SIVA
(Lopez & Moore 2026, Table 2, Approach III).  Written for the LO (x = q1) and
NO (y = q2) with F(t) = 0 and both masses equal to m:

    m*x'' + b1*x' + b2*(x' - y') + k1*x + k2*(x - y) + a*(x - y)*|x - y|^beta = 0
    m*y''         - b2*(x' - y')         - k2*(x - y) - a*(x - y)*|x - y|^beta = 0

Dividing by m gives the per-DOF accelerations this script builds and plots.
The published coefficients are taken as truth as-is — they are NOT refitted to
the data.

Channel convention
------------------
Column 0 of the .mat is the LO, column 1 is the NO.  This is not an assumption:
the linear modes of Eq. (7) sit at 5.69 Hz and 19.90 Hz with mode shapes
[LO, NO] = [0.100, 1] and [1, -0.100], and column 0 carries ~26% of its energy
in the 20 Hz band against ~0.004% for column 1.

What is plotted, per trial
--------------------------
Three rows per DOF, because they fail in different ways and disagreeing rows
are informative:

  1. DISPLACEMENT   — forward simulation from that trial's initial condition.
     Compounds error over time; unforgiving of any frequency error.
  2. ACCELERATION   — each equation evaluated on the MEASURED states.
     A one-step-local check, independent of integration stability.
  3. CUMULATIVE ENERGY — integral(v*a) against the measured change in kinetic
     energy.  This is the quantity the DISCOVER reward is built on.

An equation can win row 2 and lose row 3, or vice versa; the console table
reports both so the comparison is not decided by whichever panel is prettiest.

USAGE
-----
Exactly ONE entry in GENERATED_EXPRS is active; the rest are commented out.
Uncomment the pair you want and comment the previous one.  Then::

    python lono_comparison_plot.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')

from discover_analysis import (clean_exprs, cumtrapz, forward_simulate,
                              predict_accel)
from discover_data import load_mat_data

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

MAT_PATH  = "AllData_ProcessedNOhit.mat"
VAR_NAMES = ['q1', 'q2']          # q1 = LO (column 0), q2 = NO (column 1)
DOF_LABELS = ['LO  (q1)', 'NO  (q2)']

# Data conditioning — match the training run being assessed.
TRIM_FRONT        = 600
TRIM_BACK         = 120_000
DESIRED_TIMESTEPS = 2000

TRIALS = None                     # None = all trials in the record

# ── Published LO-NO parameters ─────────────────────────────────────────────
# Lopez & Moore (2026) Table 2.  Swap the block to compare a different
# approach; masses from Wang & Moore (2021), m_LO = m_NO = 1.370 kg.
SIVA = dict(
    m     = 1.370,                # kg          both masses
    b1    = 3.8405,               # Ns/m        LO grounding damping
    b2    = 0.78563,              # Ns/m        LO-NO coupling damping
    k1    = 19273.0,              # N/m         LO grounding stiffness
    k2    = 1947.6,               # N/m         LO-NO coupling stiffness
    alpha = 2.7641e7,             # N/m^(b+1)   nonlinear coupling stiffness
    beta  = 2.0012,               # -           nonlinear exponent
)
# Approach I : b1=3.7270, b2=0.75904, k1=18690, k2=1920.2, alpha=2.5815e7, beta=2.0010
# Approach II: b1=3.7751, b2=0.7751,  k1=18883, k2=1933.8, alpha=2.6805e7, beta=2.0012

# ── Discovered equations — keep exactly ONE pair uncommented ───────────────
GENERATED_LABEL = "DISCOVER run E"

GENERATED_EXPRS = [
    "q1ddot = -1.279e+4*q1**3 - 8.551*q1**2*q1dot - 4.268e+6*q1**2*q2 + 0.8043*q1**2*q2dot + 2.084*q1**2 - 0.001906*q1*q1dot**2 - 1902.0*q1*q1dot*q2 + 0.0003585*q1*q1dot*q2dot + 0.000929*q1*q1dot - 4.748e+8*q1*q2**2 + 178.9*q1*q2*q2dot + 463.7*q1*q2 - 1.686e-5*q1*q2dot**2 - 8.738e-5*q1*q2dot - 6598.0*q1 - 1.416e-7*q1dot**3 - 0.212*q1dot**2*q2 + 3.995e-8*q1dot**2*q2dot + 1.035e-7*q1dot**2 - 1.058e+5*q1dot*q2**2 + 0.03988*q1dot*q2*q2dot + 0.1033*q1dot*q2 - 3.758e-9*q1dot*q2dot**2 - 1.947e-8*q1dot*q2dot - 1.951*q1dot + 6.6e+7*q2**3 + 9953.0*q2**2*q2dot + 2.579e+4*q2**2 - 0.001876*q2*q2dot**2 - 0.00972*q2*q2dot + 484.6*q2 + 1.178e-10*q2dot**3 + 9.158e-10*q2dot**2 + 0.3482*q2dot + 0.04959",
    "q2ddot = 6.629e+6*q1**3 + 34.47*q1**2*q1dot + 8.152e+5*q1**2*q2 + 2.768*q1**2*q2dot + 0.1081*q1*q1dot**2 + 5112.0*q1*q1dot*q2 + 0.01736*q1*q1dot*q2dot + 6.044e+7*q1*q2**2 + 410.4*q1*q2*q2dot + 0.0006968*q1*q2dot**2 + 1383.0*q1 + 0.000113*q1dot**3 + 8.014*q1dot**2*q2 + 2.721e-5*q1dot**2*q2dot + 1.895e+5*q1dot*q2**2 + 1.287*q1dot*q2*q2dot + 2.185e-6*q1dot*q2dot**2 - 1.193*q1dot - 2.945e+7*q2**3 + 1.522e+4*q2**2*q2dot + 0.05166*q2*q2dot**2 - 1673.0*q2 + 5.847e-8*q2dot**3 - 0.5637*q2dot + 0.008289",
]

# ── run A ──────────────────────────────────────────────────────────────────
# GENERATED_LABEL = "DISCOVER run A"
# GENERATED_EXPRS = [
#     "q1ddot = -1.659e+9*q1**3 + 1.051e+5*q1**2*q1dot + 3.093e+8*q1**2*q2 - 2.92e+4*q1**2 - 1.044e+4*q1*q1dot**2 + 4.107e+6*q1*q1dot*q2 - 122.5*q1*q1dot - 4.257e+8*q1*q2**2 + 2.805e+4*q1*q2 - 1588.0*q1 - 3.709*q1dot**3 + 2888.0*q1dot**2*q2 - 0.1285*q1dot**2 - 7.06e+5*q1dot*q2**2 + 58.84*q1dot*q2 - 0.001129*q1dot + 5.584e+7*q2**3 - 6735.0*q2**2 + 92.36*q2 - 3.304e-6",
#     "q2ddot = 3.224e+5*q1**3 + 3282.0*q1**2*q1dot - 7.63e+6*q1**2*q2 + 186.2*q1**2*q2dot + 11.13*q1*q1dot**2 - 5.177e+4*q1*q1dot*q2 + 1.263*q1*q1dot*q2dot + 6.019e+7*q1*q2**2 - 2937.0*q1*q2*q2dot + 0.03583*q1*q2dot**2 + 1339.0*q1 + 0.01259*q1dot**3 - 87.82*q1dot**2*q2 + 0.002143*q1dot**2*q2dot + 2.042e+5*q1dot*q2**2 - 9.964*q1dot*q2*q2dot + 0.0001215*q1dot*q2dot**2 - 1.684*q1dot - 2.794e+7*q2**3 + 1.158e+4*q2**2*q2dot - 0.2826*q2*q2dot**2 - 1606.0*q2 + 2.298e-6*q2dot**3 - 0.4418*q2dot",
# ]

# ── run B ──────────────────────────────────────────────────────────────────
# GENERATED_LABEL = "DISCOVER run B"
# GENERATED_EXPRS = [
#     "q1ddot = -1.618e+9*q1**3 + 3616.0*q1*q1dot - 7880.0*q1 - 0.05822*q1dot**3 + 118.0*q1dot**2*q2 + 0.03304*q1dot**2*q2dot + 7.05*q1dot**2 - 7.972e+4*q1dot*q2**2 - 44.65*q1dot*q2*q2dot - 641.8*q1dot*q2 - 0.006251*q1dot*q2dot**2 + 25.74*q1dot*q2dot - 1.661*q1dot + 1.795e+7*q2**3 + 1.508e+4*q2**2*q2dot + 2.756e+4*q2**2 + 4.224*q2*q2dot**2 + 15.44*q2*q2dot + 602.7*q2 + 0.0003942*q2dot**3 + 0.002161*q2dot**2 + 0.3853*q2dot + 0.002406",
#     "q2ddot = 1.48e+7*q1**3 + 1.179e+5*q1**2*q1dot - 5.736e+7*q1**2*q2 + 1.417e+4*q1**2*q2dot - 1801.0*q1**2 + 313.3*q1*q1dot**2 - 3.047e+5*q1*q1dot*q2 + 75.26*q1*q1dot*q2dot - 9.569*q1*q1dot + 7.41e+7*q1*q2**2 - 3.66e+4*q1*q2*q2dot + 4653.0*q1*q2 + 4.52*q1*q2dot**2 - 1.149*q1*q2dot + 1324.0*q1 + 0.2774*q1dot**3 - 404.7*q1dot**2*q2 + 0.09996*q1dot**2*q2dot - 0.01271*q1dot**2 + 1.968e+5*q1dot*q2**2 - 97.22*q1dot*q2*q2dot + 12.36*q1dot*q2 + 0.01201*q1dot*q2dot**2 - 0.003053*q1dot*q2dot - 0.8906*q1dot - 3.191e+7*q2**3 + 2.364e+4*q2**2*q2dot - 3006.0*q2**2 - 5.839*q2*q2dot**2 + 1.485*q2*q2dot - 1608.0*q2 + 0.0004807*q2dot**3 - 0.0001834*q2dot**2 - 0.5757*q2dot - 9.879e-7",
# ]

# ── run C ──────────────────────────────────────────────────────────────────
# GENERATED_LABEL = "DISCOVER run C"
# GENERATED_EXPRS = [
#     "q1ddot = -1.279e+4*q1**3 - 8.551*q1**2*q1dot - 4.268e+6*q1**2*q2 + 0.8043*q1**2*q2dot + 2.084*q1**2 - 0.001906*q1*q1dot**2 - 1902.0*q1*q1dot*q2 + 0.0003585*q1*q1dot*q2dot + 0.000929*q1*q1dot - 4.748e+8*q1*q2**2 + 178.9*q1*q2*q2dot + 463.7*q1*q2 - 1.686e-5*q1*q2dot**2 - 8.738e-5*q1*q2dot - 6598.0*q1 - 1.416e-7*q1dot**3 - 0.212*q1dot**2*q2 + 3.995e-8*q1dot**2*q2dot + 1.035e-7*q1dot**2 - 1.058e+5*q1dot*q2**2 + 0.03988*q1dot*q2*q2dot + 0.1033*q1dot*q2 - 3.758e-9*q1dot*q2dot**2 - 1.947e-8*q1dot*q2dot - 1.951*q1dot + 6.6e+7*q2**3 + 9953.0*q2**2*q2dot + 2.579e+4*q2**2 - 0.001876*q2*q2dot**2 - 0.00972*q2*q2dot + 484.6*q2 + 1.178e-10*q2dot**3 + 9.158e-10*q2dot**2 + 0.3482*q2dot + 0.04959",
#     "q2ddot = 6.629e+6*q1**3 + 34.47*q1**2*q1dot + 8.152e+5*q1**2*q2 + 2.768*q1**2*q2dot + 0.1081*q1*q1dot**2 + 5112.0*q1*q1dot*q2 + 0.01736*q1*q1dot*q2dot + 6.044e+7*q1*q2**2 + 410.4*q1*q2*q2dot + 0.0006968*q1*q2dot**2 + 1383.0*q1 + 0.000113*q1dot**3 + 8.014*q1dot**2*q2 + 2.721e-5*q1dot**2*q2dot + 1.895e+5*q1dot*q2**2 + 1.287*q1dot*q2*q2dot + 2.185e-6*q1dot*q2dot**2 - 1.193*q1dot - 2.945e+7*q2**3 + 1.522e+4*q2**2*q2dot + 0.05166*q2*q2dot**2 - 1673.0*q2 + 5.847e-8*q2dot**3 - 0.5637*q2dot + 0.008289",
# ]

# ── run D ──────────────────────────────────────────────────────────────────
# GENERATED_LABEL = "DISCOVER run D"
# GENERATED_EXPRS = [
#     "q1ddot = -1.783e+9*q1**3 - 1.028e+6*q1**2*q1dot + 5.082e+8*q1**2*q2 - 2.693e+4*q1**2*q2dot + 1.198e+5*q1**2 - 1107.0*q1*q1dot**2 + 1.094e+6*q1*q1dot*q2 - 57.98*q1*q1dot*q2dot + 258.0*q1*q1dot - 2.705e+8*q1*q2**2 + 2.867e+4*q1*q2*q2dot - 1.275e+5*q1*q2 - 0.7596*q1*q2dot**2 + 6.759*q1*q2dot - 6723.0*q1 - 0.3971*q1dot**3 + 589.0*q1dot**2*q2 - 0.03121*q1dot**2*q2dot + 0.1389*q1dot**2 - 2.912e+5*q1dot*q2**2 + 30.86*q1dot*q2*q2dot - 137.3*q1dot*q2 - 0.0008177*q1dot*q2dot**2 + 0.007276*q1dot*q2dot - 0.01619*q1dot + 4.799e+7*q2**3 - 7629.0*q2**2*q2dot + 3.394e+4*q2**2 + 0.4043*q2*q2dot**2 - 3.597*q2*q2dot + 448.6*q2 - 34.57*q2dot**3 + 9.531e-5*q2dot**2 + 0.3591*q2dot + 0.05788",
#     "q2ddot = 1.46e+7*q1**3 + 1.139e+5*q1**2*q1dot - 5.682e+7*q1**2*q2 + 1.638e+4*q1**2*q2dot - 5411.0*q1**2 + 296.1*q1*q1dot**2 - 2.954e+5*q1*q1dot*q2 + 85.15*q1*q1dot*q2dot - 28.13*q1*q1dot + 7.368e+7*q1*q2**2 - 4.248e+4*q1*q2*q2dot + 1.403e+4*q1*q2 + 6.122*q1*q2dot**2 - 4.045*q1*q2dot + 1323.0*q1 + 0.2566*q1dot**3 - 384.0*q1dot**2*q2 + 0.1107*q1dot**2*q2dot - 0.03657*q1dot**2 + 1.916e+5*q1dot*q2**2 - 110.4*q1dot*q2*q2dot + 36.48*q1dot*q2 + 0.01592*q1dot*q2dot**2 - 0.01052*q1dot*q2dot - 0.9003*q1dot - 3.185e+7*q2**3 + 2.754e+4*q2**2*q2dot - 9099.0*q2**2 - 7.939*q2*q2dot**2 + 5.246*q2*q2dot - 1608.0*q2 + 0.0007628*q2dot**3 - 0.000756*q2dot**2 - 0.5966*q2dot + 0.05462",
# ]

# ── Plot / simulation options ──────────────────────────────────────────────
T_END            = None       # crop the comparison window (s); None = full record
SOLVER           = 'LSODA'
BLOW_UP_LIMIT    = 1e4        # stop integration when any |state| exceeds this
ENERGY_NORMALIZE = True       # scale the energy residual by std(dKE)
SAVE_PNG         = True
SHOW             = True
DPI              = 150

C_EXP   = '#111111'
C_TRUTH = '#1f77b4'
C_GEN   = '#d62728'

# ═══════════════════════════════════════════════════════════════════════════


def truth_exprs(p=SIVA):
    """Eq. (7) rearranged to accelerations and divided by the mass.

    The nonlinear term is kept in its published ``(x-y)*|x-y|^beta`` form
    rather than snapped to a cubic, so a non-integer beta is honoured.
    """
    m = p['m']
    c_v1 = -(p['b1'] + p['b2']) / m      # q1dot in the LO equation
    c_v2 = p['b2'] / m                   # q2dot in the LO equation
    c_x1 = -(p['k1'] + p['k2']) / m      # q1    in the LO equation
    c_x2 = p['k2'] / m                   # q2    in the LO equation
    c_nl = p['alpha'] / m                # nonlinear coupling, magnitude
    b    = p['beta']
    nl   = f"{c_nl:.6g}*(q1-q2)*abs(q1-q2)**{b:.6g}"
    return [
        f"q1ddot = {c_v1:.6g}*q1dot + {c_v2:.6g}*q2dot "
        f"+ {c_x1:.6g}*q1 + {c_x2:.6g}*q2 - {nl}",
        f"q2ddot = {-c_v2:.6g}*q2dot + {c_v2:.6g}*q1dot "
        f"+ {-c_x2:.6g}*q2 + {c_x2:.6g}*q1 + {nl}",
    ]


def _nrmse(pred, ref):
    s = np.std(ref)
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / (s if s > 0 else 1.0))


def _impact_forces(mat_path):
    """Peak impact force per trial, for panel labels.  Optional."""
    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            grp = f if 'MaxForce' in f else next(
                v for v in f.values() if hasattr(v, 'keys') and 'MaxForce' in v)
            return np.array(grp['MaxForce']).ravel()
    except Exception:
        return None


def compare_trial(system, cl_truth, cl_gen, trial, forces=None):
    """Build the 3 x n_dof comparison figure for one trial and return metrics."""
    import matplotlib.pyplot as plt

    N  = system.n_dof
    vn = system.var_names
    t  = np.asarray(system.time, dtype=float)
    disp = system.disp[:, :, trial]
    vel  = system.vel[:, :, trial]
    acc  = system.acc[:, :, trial]

    if T_END is not None:
        msk = t <= float(T_END)
        t, disp, vel, acc = t[msk], disp[msk], vel[msk], acc[msk]

    ic = []
    for d in range(N):
        ic.extend([float(disp[0, d]), float(vel[0, d])])

    print(f"\n{'=' * 72}")
    hdr = f"TRIAL {trial}"
    if forces is not None and trial < len(forces):
        hdr += f"   (impact {forces[trial]:.0f} N)"
    print(f"{hdr}   IC = {np.array(ic)}")
    print(f"{'=' * 72}")

    print("  [truth] forward simulation")
    sol_t = forward_simulate(cl_truth, vn, t, ic, solver=SOLVER,
                             blow_up_limit=BLOW_UP_LIMIT, progress=False)
    print("  [generated] forward simulation")
    sol_g = forward_simulate(cl_gen, vn, t, ic, solver=SOLVER,
                             blow_up_limit=BLOW_UP_LIMIT, progress=False)

    a_truth, nb_t = predict_accel(cl_truth, vn, disp, vel)
    a_gen,   nb_g = predict_accel(cl_gen,   vn, disp, vel)
    if nb_t or nb_g:
        print(f"  [warn] non-finite predicted samples zeroed — "
              f"truth {nb_t}, generated {nb_g}")

    metrics = {'trial': trial, 'disp_nrmse': {}, 'acc_nrmse': {}, 'r_energy': {}}

    fig, axes = plt.subplots(3, N, figsize=(7.0 * N, 9.5), sharex='col',
                             squeeze=False)
    title = f"LO-NO   trial {trial}"
    if forces is not None and trial < len(forces):
        title += f"   (impact {forces[trial]:.0f} N)"
    fig.suptitle(f"{title}     published SIVA truth  vs  {GENERATED_LABEL}",
                 fontsize=12)

    for d in range(N):
        lbl = DOF_LABELS[d] if d < len(DOF_LABELS) else vn[d]

        # ── row 1: displacement from forward simulation ────────────────────
        ax = axes[0][d]
        ax.plot(t, disp[:, d], color=C_EXP, lw=1.0, ls='--', label='Experiment')
        for sol, col, name, key in ((sol_t, C_TRUTH, 'Truth (SIVA)', 'truth'),
                                    (sol_g, C_GEN, GENERATED_LABEL, 'gen')):
            if sol.t.size > 1:
                y = sol.y[2 * d]
                ax.plot(sol.t, y, color=col, lw=1.2, ls='--', label=name)
                ref = np.interp(sol.t, t, disp[:, d])
                metrics['disp_nrmse'].setdefault(key, {})[d] = _nrmse(y, ref)
                frac = (sol.t[-1] - sol.t[0]) / (t[-1] - t[0])
                if frac < 0.99:
                    ax.text(0.98, 0.05 if key == 'gen' else 0.15,
                            f"{name}: stopped at {100*frac:.0f}%",
                            transform=ax.transAxes, ha='right', fontsize=7,
                            color=col)
        ax.set_title(f"{lbl}", fontsize=11)
        ax.set_ylabel('displacement  [m]', fontsize=9)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, loc='upper right')

        # ── row 2: acceleration on measured states ─────────────────────────
        ax = axes[1][d]
        ax.plot(t, acc[:, d], color=C_EXP, lw=1.0, ls='--', label='Measured')
        ax.plot(t, a_truth[d], color=C_TRUTH, lw=1.1, ls='--', label='Truth (SIVA)')
        ax.plot(t, a_gen[d], color=C_GEN, lw=1.1, ls=':', label=GENERATED_LABEL)
        metrics['acc_nrmse'].setdefault('truth', {})[d] = _nrmse(a_truth[d], acc[:, d])
        metrics['acc_nrmse'].setdefault('gen', {})[d] = _nrmse(a_gen[d], acc[:, d])
        ax.set_ylabel(r'acceleration  [m/s$^2$]', fontsize=9)
        ax.set_title(f"NRMSE   truth {metrics['acc_nrmse']['truth'][d]:.3f}   "
                     f"gen {metrics['acc_nrmse']['gen'][d]:.3f}", fontsize=9)
        ax.grid(alpha=0.3)

        # ── row 3: cumulative work-energy balance ──────────────────────────
        ax = axes[2][d]
        v = vel[:, d]
        dke = 0.5 * (v ** 2 - v[0] ** 2)
        scale = float(np.std(dke))
        if scale < 1e-12:
            scale = float(np.mean(np.abs(dke))) + 1e-12
        ax.plot(t, dke, color=C_EXP, lw=1.2, label=r'measured $\Delta$KE')
        for a_pred, col, name, key, ls in (
                (a_truth[d], C_TRUTH, 'Truth (SIVA)', 'truth', '--'),
                (a_gen[d], C_GEN, GENERATED_LABEL, 'gen', ':')):
            cp = cumtrapz(v * a_pred, t)
            ax.plot(t, cp, color=col, lw=1.2, ls=ls, label=f'{name}  ' + r'$\int v\,a$')
            res = np.abs(cp - dke)
            if ENERGY_NORMALIZE:
                res = res / scale
            metrics['r_energy'].setdefault(key, {})[d] = 1.0 / (1.0 + float(np.mean(res)))
        ax.set_ylabel(r'energy  [J/kg]', fontsize=9)
        ax.set_xlabel('Time  [s]', fontsize=9)
        ax.set_title(f"r_energy   truth {metrics['r_energy']['truth'][d]:.3f}   "
                     f"gen {metrics['r_energy']['gen'][d]:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    if SAVE_PNG:
        out = f"lono_comparison_trial{trial}.png"
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f"  saved {out}")
    if not SHOW:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    return metrics


def summarize(all_metrics, n_dof):
    """Console table across trials.  The two metrics can disagree — that is the
    point of printing both."""
    print(f"\n{'=' * 72}")
    print(f"SUMMARY   published SIVA truth  vs  {GENERATED_LABEL}")
    print(f"{'=' * 72}")

    for name, key, fmt in (('acceleration NRMSE  (lower is better)', 'acc_nrmse', '.3f'),
                           ('energy reward       (higher is better)', 'r_energy', '.3f'),
                           ('sim displacement NRMSE (lower is better)', 'disp_nrmse', '.3f')):
        print(f"\n{name}")
        head = f"{'trial':>6}"
        for d in range(n_dof):
            head += f"{'truth DOF'+str(d):>14}{'gen DOF'+str(d):>14}"
        print(head)
        for mt in all_metrics:
            row = f"{mt['trial']:>6}"
            for d in range(n_dof):
                for k in ('truth', 'gen'):
                    v = mt[key].get(k, {}).get(d)
                    row += f"{'--':>14}" if v is None else f"{v:>14{fmt}}"
            print(row)
        for d in range(n_dof):
            for k in ('truth', 'gen'):
                vals = [m[key][k][d] for m in all_metrics
                        if d in m[key].get(k, {})]
                if vals:
                    print(f"        mean {k:>5} DOF{d}: {np.mean(vals):.4f}")


def main():
    system = load_mat_data(MAT_PATH,
                           trim_timesteps_front=TRIM_FRONT,
                           trim_timesteps_back=TRIM_BACK,
                           desired_timesteps=DESIRED_TIMESTEPS,
                           var_names=VAR_NAMES,
                           plot_data=False)
    print(f"[data] {system!r}")
    forces = _impact_forces(MAT_PATH)

    te = truth_exprs()
    print("\n[truth]  Wang & Moore Eq. (7) / m, SIVA parameters:")
    for e in te:
        print(f"   {e}")
    print(f"\n[{GENERATED_LABEL}]")
    for e in GENERATED_EXPRS:
        s = str(e).strip()
        print(f"   {s if len(s) <= 100 else s[:100] + ' ...'}")

    cl_truth = clean_exprs(te, system.var_names)
    cl_gen = clean_exprs(GENERATED_EXPRS, system.var_names)

    keep = list(range(system.n_trials)) if TRIALS is None else list(TRIALS)
    all_metrics = [compare_trial(system, cl_truth, cl_gen, tr, forces)
                   for tr in keep]
    summarize(all_metrics, system.n_dof)

    if SHOW:
        import matplotlib.pyplot as plt
        plt.show()
    return all_metrics


if __name__ == '__main__':
    main()