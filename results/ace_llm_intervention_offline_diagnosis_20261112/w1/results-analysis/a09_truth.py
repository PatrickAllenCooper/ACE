"""Recover the eight true target means per system from saved predictions and saved errors.
32*NMSE_i = sum_q (p_iq - t_q)^2  =>  32*NMSE_i - sum_q p_iq^2 = -2 sum_q p_iq t_q + s,  s = sum_q t_q^2.
Linear least squares in (t, s) using every saved curve point of the system (all 8 arms).
This is an INFERENCE from saved outputs, checked by the residual and by s == sum t^2. Structurally identical
queries (equal prediction columns) are not separately identified; minimum-norm lstsq assigns them equal values,
which is exact when the true means are structurally equal."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
T = {}
rows = []
for w in range(24):
    c = curves[curves.w == w]
    P = c[PRED].values; e = c.normalized_mse.values
    A = np.c_[-2 * P, np.ones(len(P))]
    b = 32 * e - (P ** 2).sum(axis=1)
    sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    t, s = sol[:8], sol[8]
    rec = ((P - t) ** 2).sum(axis=1) / 32
    T[w] = t
    rows.append({'w': w, 'graph': meta.graph_stratum[w], 'regime': meta.regime[w], 'cost': meta.cost_stratum[w], 'points': len(P), 'rank': rank,
                 'max_abs_err_recon': np.abs(rec - e).max(), 's_minus_sum_t2': s - (t ** 2).sum(),
                 **{'t%d' % i: t[i] for i in range(8)}})
R = pd.DataFrame(rows).set_index('w')
pd.set_option('display.width', 250)
print(R.to_string(float_format=lambda x: '%.4g' % x))
print('max reconstruction error of saved NMSE over all 4540 points: %.3g' % R.max_abs_err_recon.max())
np.save('truth_means.npy', np.array([T[w] for w in range(24)]))
# structural checks: chain q6==q7 ; fork q0==q4==q5==q6 and q1==q7 ; linearity q1,q2,q3
for w in range(24):
    t = T[w]; g = meta.graph_stratum[w]
    lin = abs(t[1] - 2 * t[2] + t[3])
    if g == 'chain':
        chk = (abs(t[6] - t[7]), abs(t[4] - 2 * t[5] + t[6]))
    elif g == 'fork':
        chk = (max(abs(t[0] - t[4]), abs(t[0] - t[5]), abs(t[0] - t[6])), abs(t[1] - t[7]))
    else:
        chk = (abs(t[4] - 2 * t[5] + t[6]), abs((t[7] - t[1]) - (t[6] - t[5])) )   # collider: q7 - q1 = b_ZY*(1 - ybar); loose
    print('w%02d %-8s linear-in-x residual %.2e ; structure checks %s' % (w, g, lin, tuple('%.2e' % v for v in chk)))
