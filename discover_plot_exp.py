"""
discover_plot_exp.py
===================
Simulate discovered equations forward from an EXPERIMENTAL initial condition
and compare against the record.

Works for any number of DOFs — set ``DOF_INDEX`` to an integer to look at a
single channel (matching a :mod:`discover_sdof_exp` run), or leave it ``None``
for the full multi-DOF record.

USAGE
-----
1. Paste one discovered physical expression per DOF into ``DISCOVERED_EXPRS``.
   The trainer prints a paste-ready block when it finishes.
2. Point the config at the same data the run used — the trim and downsample
   settings matter, since they change the time grid the equations were fitted
   on.
3. ``python discover_plot_exp.py``

Variable names in expressions follow ``VAR_NAMES``::

    VAR_NAMES = ['x', 'y']   ->  x, xdot, y, ydot
    VAR_NAMES = ['q1','q2']  ->  q1, q1dot, q2, q2dot

Supported syntax:  ``+ - * / ** ^``, ``sin cos exp sqrt abs Abs sign sgn``.

Reading the output
------------------
Two figures are produced and they answer different questions.  The forward
simulation compounds error over time, so a small frequency error diverges
visibly even when the structure is right; the energy residual is evaluated on
the MEASURED states and is exactly what the search optimised.  Trust the
residual for "is this equation right", the simulation for "is this equation
usable as a model".
"""

from __future__ import annotations

import sys

sys.path.insert(0, '.')

from discover_analysis import plot_discovered
from discover_data import load_mat_data, select_dof

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIG ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

MAT_PATH  = "AllData_ProcessedNOhit.mat"
# MAT_PATH = "SN_data.mat"
VAR_NAMES = ['q1', 'q2']    # one name per DOF, in data-column order
# VAR_NAMES = ['x']
DOF_INDEX = None               # None = all channels; an int = single-DOF slice
#DOF_INDEX = 0
TRIAL     = 3              # which trial supplies the IC and the comparison

# One discovered expression per DOF (RHS only, or with "<var>ddot = " prefix).
DISCOVERED_EXPRS = [
    #"q1ddot = -1.535e+9*q1**3 - 4.083e+5*q1**2*q1dot + 4.777e+8*q1**2*q2 - 6.179e+5*q1**2*q2dot + 6.714e+4*q1**2 - 4270.0*q1*q1dot**2 + 3.14e+6*q1*q1dot*q2 - 1485.0*q1*q1dot*q2dot + 161.3*q1*q1dot - 6.009e+8*q1*q2**2 + 6.245e+5*q1*q2*q2dot - 6.785e+4*q1*q2 - 194.6*q1*q2dot**2 + 42.29*q1*q2dot - 1299.0*q1 - 0.781*q1dot**3 + 1152.0*q1dot**2*q2 - 0.892*q1dot**2*q2dot + 0.09692*q1dot**2 - 5.283e+5*q1dot*q2**2 + 750.3*q1dot*q2*q2dot - 81.53*q1dot*q2 - 0.2338*q1dot*q2dot**2 + 0.05081*q1dot*q2dot - 0.00276*q1dot + 7.792e+7*q2**3 - 1.578e+5*q2**2*q2dot + 1.714e+4*q2**2 + 98.34*q2*q2dot**2 - 21.37*q2*q2dot + 1.161*q2 + 11.94*q2dot**3 + 0.006659*q2dot**2 - 0.0007236*q2dot + 2.621e-5",
    #"q2ddot = 3.516e+7*q1**3 + 71.37*q1**2*q1dot + 1.066e+6*q1**2*q2 + 3.417*q1**2*q2dot + 1.439*q1**2 + 0.2518*q1*q1dot**2 + 7521.0*q1*q1dot*q2 + 0.02411*q1*q1dot*q2dot + 0.01015*q1*q1dot + 5.617e+7*q1*q2**2 + 360.1*q1*q2*q2dot + 151.6*q1*q2 + 0.0005772*q1*q2dot**2 + 0.000486*q1*q2dot + 1146.0*q1 + 0.0002961*q1dot**3 + 13.27*q1dot**2*q2 + 4.253e-5*q1dot**2*q2dot + 1.79e-5*q1dot**2 + 1.982e+5*q1dot*q2**2 + 1.27*q1dot*q2*q2dot + 0.5348*q1dot*q2 + 2.036e-6*q1dot*q2dot**2 + 1.714e-6*q1dot*q2dot - 1.571*q1dot - 2.945e+7*q2**3 + 9489.0*q2**2*q2dot + 3994.0*q2**2 + 0.03042*q2*q2dot**2 + 0.02561*q2*q2dot - 1620.0*q2 + 3.25e-8*q2dot**3 + 4.105e-8*q2dot**2 - 0.442*q2dot + 0.04046"
    
    #below is the best    
    #"q1ddot = -1.618e+9*q1**3 + 3616.0*q1*q1dot - 7880.0*q1 - 0.05822*q1dot**3 + 118.0*q1dot**2*q2 + 0.03304*q1dot**2*q2dot + 7.05*q1dot**2 - 7.972e+4*q1dot*q2**2 - 44.65*q1dot*q2*q2dot - 641.8*q1dot*q2 - 0.006251*q1dot*q2dot**2 + 25.74*q1dot*q2dot - 1.661*q1dot + 1.795e+7*q2**3 + 1.508e+4*q2**2*q2dot + 2.756e+4*q2**2 + 4.224*q2*q2dot**2 + 15.44*q2*q2dot + 602.7*q2 + 0.0003942*q2dot**3 + 0.002161*q2dot**2 + 0.3853*q2dot + 0.002406",
    #" q2ddot = 1.48e+7*q1**3 + 1.179e+5*q1**2*q1dot - 5.736e+7*q1**2*q2 + 1.417e+4*q1**2*q2dot - 1801.0*q1**2 + 313.3*q1*q1dot**2 - 3.047e+5*q1*q1dot*q2 + 75.26*q1*q1dot*q2dot - 9.569*q1*q1dot + 7.41e+7*q1*q2**2 - 3.66e+4*q1*q2*q2dot + 4653.0*q1*q2 + 4.52*q1*q2dot**2 - 1.149*q1*q2dot + 1324.0*q1 + 0.2774*q1dot**3 - 404.7*q1dot**2*q2 + 0.09996*q1dot**2*q2dot - 0.01271*q1dot**2 + 1.968e+5*q1dot*q2**2 - 97.22*q1dot*q2*q2dot + 12.36*q1dot*q2 + 0.01201*q1dot*q2dot**2 - 0.003053*q1dot*q2dot - 0.8906*q1dot - 3.191e+7*q2**3 + 2.364e+4*q2**2*q2dot - 3006.0*q2**2 - 5.839*q2*q2dot**2 + 1.485*q2*q2dot - 1608.0*q2 + 0.0004807*q2dot**3 - 0.0001834*q2dot**2 - 0.5757*q2dot - 9.879e-7"

    #"q1ddot = -1.279e+4*q1**3 - 8.551*q1**2*q1dot - 4.268e+6*q1**2*q2 + 0.8043*q1**2*q2dot + 2.084*q1**2 - 0.001906*q1*q1dot**2 - 1902.0*q1*q1dot*q2 + 0.0003585*q1*q1dot*q2dot + 0.000929*q1*q1dot - 4.748e+8*q1*q2**2 + 178.9*q1*q2*q2dot + 463.7*q1*q2 - 1.686e-5*q1*q2dot**2 - 8.738e-5*q1*q2dot - 6598.0*q1 - 1.416e-7*q1dot**3 - 0.212*q1dot**2*q2 + 3.995e-8*q1dot**2*q2dot + 1.035e-7*q1dot**2 - 1.058e+5*q1dot*q2**2 + 0.03988*q1dot*q2*q2dot + 0.1033*q1dot*q2 - 3.758e-9*q1dot*q2dot**2 - 1.947e-8*q1dot*q2dot - 1.951*q1dot + 6.6e+7*q2**3 + 9953.0*q2**2*q2dot + 2.579e+4*q2**2 - 0.001876*q2*q2dot**2 - 0.00972*q2*q2dot + 484.6*q2 + 1.178e-10*q2dot**3 + 9.158e-10*q2dot**2 + 0.3482*q2dot + 0.04959",
    #"q2ddot = 6.629e+6*q1**3 + 34.47*q1**2*q1dot + 8.152e+5*q1**2*q2 + 2.768*q1**2*q2dot + 0.1081*q1*q1dot**2 + 5112.0*q1*q1dot*q2 + 0.01736*q1*q1dot*q2dot + 6.044e+7*q1*q2**2 + 410.4*q1*q2*q2dot + 0.0006968*q1*q2dot**2 + 1383.0*q1 + 0.000113*q1dot**3 + 8.014*q1dot**2*q2 + 2.721e-5*q1dot**2*q2dot + 1.895e+5*q1dot*q2**2 + 1.287*q1dot*q2*q2dot + 2.185e-6*q1dot*q2dot**2 - 1.193*q1dot - 2.945e+7*q2**3 + 1.522e+4*q2**2*q2dot + 0.05166*q2*q2dot**2 - 1673.0*q2 + 5.847e-8*q2dot**3 - 0.5637*q2dot + 0.008289"

    #"q1ddot = -1.783e+9*q1**3 - 1.028e+6*q1**2*q1dot + 5.082e+8*q1**2*q2 - 2.693e+4*q1**2*q2dot + 1.198e+5*q1**2 - 1107.0*q1*q1dot**2 + 1.094e+6*q1*q1dot*q2 - 57.98*q1*q1dot*q2dot + 258.0*q1*q1dot - 2.705e+8*q1*q2**2 + 2.867e+4*q1*q2*q2dot - 1.275e+5*q1*q2 - 0.7596*q1*q2dot**2 + 6.759*q1*q2dot - 6723.0*q1 - 0.3971*q1dot**3 + 589.0*q1dot**2*q2 - 0.03121*q1dot**2*q2dot + 0.1389*q1dot**2 - 2.912e+5*q1dot*q2**2 + 30.86*q1dot*q2*q2dot - 137.3*q1dot*q2 - 0.0008177*q1dot*q2dot**2 + 0.007276*q1dot*q2dot - 0.01619*q1dot + 4.799e+7*q2**3 - 7629.0*q2**2*q2dot + 3.394e+4*q2**2 + 0.4043*q2*q2dot**2 - 3.597*q2*q2dot + 448.6*q2 - 34.57*q2dot**3 + 9.531e-5*q2dot**2 + 0.3591*q2dot + 0.05788",
    #"q2ddot = 1.46e+7*q1**3 + 1.139e+5*q1**2*q1dot - 5.682e+7*q1**2*q2 + 1.638e+4*q1**2*q2dot - 5411.0*q1**2 + 296.1*q1*q1dot**2 - 2.954e+5*q1*q1dot*q2 + 85.15*q1*q1dot*q2dot - 28.13*q1*q1dot + 7.368e+7*q1*q2**2 - 4.248e+4*q1*q2*q2dot + 1.403e+4*q1*q2 + 6.122*q1*q2dot**2 - 4.045*q1*q2dot + 1323.0*q1 + 0.2566*q1dot**3 - 384.0*q1dot**2*q2 + 0.1107*q1dot**2*q2dot - 0.03657*q1dot**2 + 1.916e+5*q1dot*q2**2 - 110.4*q1dot*q2*q2dot + 36.48*q1dot*q2 + 0.01592*q1dot*q2dot**2 - 0.01052*q1dot*q2dot - 0.9003*q1dot - 3.185e+7*q2**3 + 2.754e+4*q2**2*q2dot - 9099.0*q2**2 - 7.939*q2*q2dot**2 + 5.246*q2*q2dot - 1608.0*q2 + 0.0007628*q2dot**3 - 0.000756*q2dot**2 - 0.5966*q2dot + 0.05462"

    #"q1ddot = -1.932e+9*q1**3 + 6.705e+8*q1**2*q2 - 2.462e+6*q1**2*q2dot + 1.307e+5*q1**2 - 3.125e+8*q1*q2**2 - 6.292e+4*q1*q2*q2dot - 1.218e+5*q1*q2 - 1471.0*q1*q2dot**2 - 12.26*q1*q2dot - 6373.0*q1 - 3.02*q1dot + 4.856e+7*q2**3 + 1.467e+4*q2**2*q2dot + 2.839e+4*q2**2 + 1.476*q2*q2dot**2 + 5.715*q2*q2dot + 418.8*q2 - 0.2838*q2dot**3 + 0.0002877*q2dot**2 + 0.4971*q2dot + 0.06357",
    #" q2ddot = 4940.0*q1**3 + 46.5*q1**2*q1dot + 9.5e+5*q1**2*q2 + 3.806*q1**2*q2dot - 1.595*q1**2 + 0.1459*q1*q1dot**2 + 5962.0*q1*q1dot*q2 + 0.02388*q1*q1dot*q2dot - 0.01001*q1*q1dot + 6.09e+7*q1*q2**2 + 487.9*q1*q2*q2dot - 204.5*q1*q2 + 0.0009774*q1*q2dot**2 - 0.0008191*q1*q2dot + 1406.0*q1 + 0.0001526*q1dot**3 + 9.354*q1dot**2*q2 + 3.747e-5*q1dot**2*q2dot - 1.57e-5*q1dot**2 + 1.911e+5*q1dot*q2**2 + 1.531*q1dot*q2*q2dot - 0.6416*q1dot*q2 + 3.067e-6*q1dot*q2dot**2 - 2.57e-6*q1dot*q2dot - 1.199*q1dot - 2.953e+7*q2**3 + 1.564e+4*q2**2*q2dot - 6554.0*q2**2 + 0.06266*q2*q2dot**2 - 0.05251*q2*q2dot - 1672.0*q2 + 8.367e-8*q2dot**3 - 1.052e-7*q2dot**2 - 0.569*q2dot + 0.03509"

    #THE BEST for NO hit
     #"q1ddot = 4.723e+5*q1*q1dot*q2 - 4.769e+8*q1*q2**2 - 6632.0*q1 + 104.2*q1dot**2*q2 - 1.715e+5*q1dot*q2**2 - 26.39*q1dot*q2 - 1.758*q1dot + 6.693e+7*q2**3 + 2.665e+4*q2**2 + 484.6*q2 + 0.3474*q2dot + 0.04955",
    #" q2ddot = 1.443e+7*q1**3 + 1.435e+5*q1**2*q1dot - 5.699e+7*q1**2*q2 + 2688.0*q1**2 + 475.7*q1*q1dot**2 - 3.778e+5*q1*q1dot*q2 - 57.34*q1*q1dot + 7.503e+7*q1*q2**2 + 1788.0*q1*q2 + 1262.0*q1 + 0.5256*q1dot**3 - 626.3*q1dot**2*q2 + 0.1242*q1dot**2 + 2.487e+5*q1dot*q2**2 + 59.76*q1dot*q2 + 0.02674*q1dot*q2dot - 1.589*q1dot - 3.293e+7*q2**3 - 6474.0*q2**2 - 3.154*q2*q2dot - 1584.0*q2 - 0.4488*q2dot + 0.01904"

    #"q1ddot = -4.778e+8*q1*q2**2 - 6613.0*q1 - 9.689e+4*q1dot*q2**2 - 2.025*q1dot + 6.68e+7*q2**3 + 2.622e+4*q2**2 + 483.5*q2 + 0.375*q2dot + 0.05084",
    #"q2ddot = 1.372e+7*q1**3 + 1.367e+5*q1**2*q1dot - 5.481e+7*q1**2*q2 - 2474.0*q1**2 + 454.0*q1*q1dot**2 - 3.641e+5*q1*q1dot*q2 - 16.44*q1*q1dot + 7.301e+7*q1*q2**2 + 6592.0*q1*q2 + 1272.0*q1 + 7.566*q1dot**3 - 604.7*q1dot**2*q2 - 0.0273*q1dot**2 + 2.425e+5*q1dot*q2**2 + 21.9*q1dot*q2 - 1.765*q1dot - 3.242e+7*q2**3 - 4391.0*q2**2 - 1592.0*q2 - 0.4284*q2dot - 2.982e-6"

    # using set transformer architecture
    #"q1ddot = -1.731e+9*q1**3 - 7318.0*q1 - 2.553*q1dot + 1.896e+7*q2**3 + 540.3*q2 + 0.543*q2dot",
    #"q2ddot = 1681.0*q1 + 2.019*q1dot - 2.356e+7*q2**3 + 9.069e+4*q2**2*q2dot + 3109.0*q2**2 - 116.3*q2*q2dot**2 - 7.978*q2*q2dot - 1602.0*q2 + 0.04975*q2dot**3 + 0.005117*q2dot**2 - 0.9891*q2dot + 2.005e-6"

    "q1ddot = -1.642e+9*q1**3 + 2.856e+5*q1**2*q2dot + 1.786e+5*q1**2 + 613.1*q1*q2dot**2 + 766.8*q1*q2dot - 7791.0*q1 - 2.604*q1dot + 1.829e+7*q2**3 + 589.8*q2 + 0.4388*q2dot**3 + 0.8232*q2dot**2 + 0.5147*q2dot + 0.1073",
    "q2ddot = 4.142e+6*q1**3 + 4.371e+4*q1**2*q1dot - 2.209e+7*q1**2*q2 + 153.7*q1*q1dot**2 - 1.554e+5*q1*q1dot*q2 + 3.928e+7*q1*q2**2 + 1733.0*q1 - 2824.0*q1dot**9 + 3686.0*q1dot**7 - 1603.0*q1dot**5 + 232.6*q1dot**3 - 273.3*q1dot**2*q2 + 1.382e+5*q1dot*q2**2 - 6.355*q1dot - 2.328e+7*q2**3 - 1747.0*q2"

    # LO
    #"q1ddot = -2.323e+5*q1 - 12.62*q1dot + 1.392e+4*q2 - 21.35*q2dot - 0.5777",
    #"q2ddot = 1.992e+4*q1 - 190.7*q1dot**3 + 2.369*q1dot - 1.254e+7*q2**2 - 2.026e+4*q2 + 0.09504"

    #"q1ddot = 1.31e+26*q1**5*q1dot**5 - 1.22e+5*q1*q1dot - 2.36e+5*q1 - 11.4*q1dot + 6.937e+7*q2**2 + 1.475e+4*q2 - 0.7803",
    #"q2ddot = -1.548e+11*q1**3 - 8.955e+5*q1**2*q1dot - 1.307e+9*q1**2*q2 - 3.442e+4*q1**2*q2dot - 2.476e+6*q1**2 + 1275.0*q1*q1dot**2 + 3.724e+6*q1*q1dot*q2 + 98.04*q1*q1dot*q2dot + 7052.0*q1*q1dot + 2.718e+9*q1*q2**2 + 1.431e+5*q1*q2*q2dot + 1.029e+7*q1*q2 + 1.884*q1*q2dot**2 + 271.0*q1*q2dot + 2.229e+4*q1 - 0.6055*q1dot**3 - 2652.0*q1dot**2*q2 - 0.06981*q1dot**2*q2dot - 5.021*q1dot**2 - 3.871e+6*q1dot*q2**2 - 203.8*q1dot*q2*q2dot - 1.466e+4*q1dot*q2 - 0.002683*q1dot*q2dot**2 - 0.386*q1dot*q2dot + 1.51*q1dot - 1.884e+9*q2**3 - 1.488e+5*q2**2*q2dot - 1.07e+7*q2**2 - 3.917*q2*q2dot**2 - 563.5*q2*q2dot - 2.027e+4*q2 - 3.437e-5*q2dot**3 - 0.007418*q2dot**2 - 0.5335*q2dot + 0.1201"

    # using set transformer architecture
    #"q1ddot = 1.051e+9*q1*q1dot*q2dot**2 - 2.319e+5*q1 - 5.091e+8*q1dot*q2*q2dot**2 + 7.085e+5*q1dot*q2dot**3 - 10.07*q1dot + 1.9e+4*q2",
    #"q2ddot = 1.909e+4*q1 - 2.623e+8*q1dot*q2*q2dot**2 + 0.5198*q1dot - 1.97e+4*q2 - 1.085e+4*q2dot**3 - 0.001542*q2dot"
    #"q2ddot = 5.262e+11*q1*q2**2 + 1.675e+4*q1 + 6.768e+8*q2*q2dot**3 + 1.384e+6*q2*q2dot**2 - 1.94e+4*q2 - 2.003*q2dot"
    
    # SN_data
    # "xddot = 6.329e+10*x**4*xdot**3 - 2.122e+7*x**3 - 223.5*x*xdot - 464.8*x - 213.9*xdot**4 - 14.68*xdot**3 - 0.1694*xdot - 0.01385"


    # SH_data
    #"xddot = 2.711e+8*x**3 + 4.889e+7*x**2*xdot**3 - 6.892e+6*x**2*xdot - 2.413e+5*x**2 + 8570.0*x*xdot**2 - 1.318e+4*x*xdot - 1.316e+4*x - 154.5*xdot**3 - 13.75*xdot**2 - 3.627*xdot - 0.004036"
    #"xddot = 2.25e+8*x**3*xdot + 2.103e+8*x**3 - 1.979e+7*x**2*xdot**2 - 4.77e+6*x**2*xdot - 1.662e+5*x**2 - 2.104e+5*x*xdot**3 - 5.499e+4*x*xdot**2 - 707.6*x*xdot - 1.188e+4*x - 90.68*xdot**3 - 9.485*xdot**2 - 6.091*xdot - 0.01357"
    #"xddot = 3.191e+8*x**3 - 1.259e+4*x - 731.2*xdot**5 - 47.42*xdot**2 - 9.726*xdot"
    # LONO Combined
]


# Data loading — must match the training run.
TRIM_FRONT        = 1000 #600
TRIM_BACK         = 120_000
DESIRED_TIMESTEPS = 8000

# Simulation
T_END         = None        # override end time (None = use the record's)
SOLVER        = 'LSODA'     # 'RK45' | 'LSODA' | 'Radau'
MAX_STEP      = None        # override max step (None = 20 * dt)
BLOW_UP_LIMIT = 1e10         # stop when any |state| exceeds this (None = off)

# Energy residual
PLOT_ENERGY      = True
ENERGY_NORMALIZE = True

# ═══════════════════════════════════════════════════════════════════════════


def main():
    if not DISCOVERED_EXPRS or not any(str(e).strip() for e in DISCOVERED_EXPRS):
        print("Nothing to plot — paste one expression per DOF into "
              "DISCOVERED_EXPRS.")
        return

    system = load_mat_data(MAT_PATH,
                           trim_timesteps_front=TRIM_FRONT,
                           trim_timesteps_back=TRIM_BACK,
                           desired_timesteps=DESIRED_TIMESTEPS,
                           var_names=VAR_NAMES if DOF_INDEX is None else None,
                           plot_data=False)
    if DOF_INDEX is not None:
        system = select_dof(system, DOF_INDEX, var_name=VAR_NAMES[0])
    print(f"[data] {system!r}")

    return plot_discovered(
        system, DISCOVERED_EXPRS, trial=TRIAL, t_end=T_END,
        solver=SOLVER, max_step=MAX_STEP, blow_up_limit=BLOW_UP_LIMIT,
        plot_energy=PLOT_ENERGY, energy_normalize=ENERGY_NORMALIZE,
        out_prefix=f"{system.name}_trial{TRIAL}",
        ref_label='Experimental',
    )


if __name__ == '__main__':
    main()
