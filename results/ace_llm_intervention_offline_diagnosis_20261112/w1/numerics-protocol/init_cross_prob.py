"""Posterior probability (under the learner's own saved post-initialization posterior) that the realized
NMSE of the posterior-mean predictor is <= 0.04. Pure posterior summary of saved public states."""
import math, numpy as np, pandas as pd
from acelib import *
K = 200000
rows = []
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    h = posterior(parents, noise, 1.0, hist[:8])
    rng = np.random.default_rng(777 + wi)
    th = {}
    for n in NODES:
        m, V, _ = h[n]
        L = np.linalg.cholesky(V)
        th[n] = m + np.einsum("kj,ij->ki", rng.standard_normal((K, len(m))), L)
    err2 = np.zeros(K)
    for q in QUERY_CLAMPS:
        pred = predict(h, parents, q)["Z"]
        vals = {}
        for n in NODES:
            if n in q:
                vals[n] = np.full(K, q[n]); continue
            phi = np.stack([np.ones(K)] + [vals[p] for p in parents[n]], 1)
            vals[n] = np.einsum("kd,kd->k", phi, th[n])
        err2 += ((pred - vals["Z"]) / SCALE) ** 2
    nm = err2 / 8
    regime, graph, cost = meta(wi)
    R8 = risk(h, parents)
    rows.append(dict(world=wi, regime=regime, graph=graph, R8=R8, mc_mean=nm.mean(), p_le_004=(nm <= 0.04).mean(), median=np.median(nm),
                     cv=nm.std() / nm.mean(), realized=nmse(h, parents, arm["world"])))
d = pd.DataFrame(rows)
d["crossed"] = d.realized <= 0.04
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
print("\nsum of posterior P(NMSE<=0.04 at init) over 24 systems = %.2f (observed crossers: %d)" % (d.p_le_004.sum(), d.crossed.sum()))
var = (d.p_le_004 * (1 - d.p_le_004)).sum()
print("Poisson-binomial sd = %.2f" % math.sqrt(var))
print("by regime family: matched expected %.2f observed %d ; shifted expected %.2f observed %d" % (
    d[d.regime.str.startswith("matched")].p_le_004.sum(), d[d.regime.str.startswith("matched")].crossed.sum(),
    d[~d.regime.str.startswith("matched")].p_le_004.sum(), d[~d.regime.str.startswith("matched")].crossed.sum()))
print("median coefficient of variation of realized NMSE given the state: %.2f ; median(median/mean): %.2f" % (d.cv.median(), (d["median"] / d.mc_mean).median()))
print("check MC mean vs exact integrated risk: max rel dev %.4f" % (np.abs(d.mc_mean / d.R8 - 1)).max())
d.to_csv("out_init_cross_prob.csv", index=False)
