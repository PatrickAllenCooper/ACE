"""Verifier step 3: reconstruct the private noise-disabled target means from saved (prediction, NMSE) pairs.
NMSE = (1/8) * sum_q ((p_q - t_q)/2)^2  =>  32*NMSE - |p|^2 = -2 p.t + |t|^2  (linear in (t, c=|t|^2)).
Then: all-zero-prediction NMSE = |t|^2/32. Fresh code."""
import pandas as pd, numpy as np
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
PC = [f"prediction_{i}" for i in range(8)]
meta = cells.drop_duplicates("world_id").set_index("world_id")[["cost_stratum", "graph_stratum", "regime"]]
rows = []
for w, g in curves.groupby("world_id"):
    P = g[PC].values
    y = 32 * g.normalized_mse.values - (P ** 2).sum(1)
    A = np.hstack([-2 * P, np.ones((len(P), 1))])
    sol, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    t, c = sol[:8], sol[8]
    # nonlinear check: recompute NMSE from t only
    nm = ((P - t) ** 2).sum(1) / 32
    maxres = np.abs(nm - g.normalized_mse.values).max()
    rows.append(dict(world_id=w, rank=rank, n=len(P), max_abs_nmse_residual=maxres, c_minus_tt=c - t @ t,
                     zero_pred_nmse=(t @ t) / 32, **{f"t{i}": t[i] for i in range(8)}))
T = pd.DataFrame(rows).set_index("world_id").join(meta)
T["short"] = [f"w{int(str(i)[-2:]):02d}" for i in T.index]
pd.set_option("display.width", 250)
print(T[["short", "graph_stratum", "regime", "cost_stratum", "rank", "n", "max_abs_nmse_residual", "c_minus_tt", "zero_pred_nmse"]].to_string(index=False))
print("max residual over all worlds:", T.max_abs_nmse_residual.max())
print("min rank:", T["rank"].min())
z = T[T.zero_pred_nmse <= 0.04]
print("systems where all-zero prediction meets 0.04:", len(z), z.short.tolist(), z.zero_pred_nmse.round(5).tolist())
init = curves[curves.total_cost_units == 8].groupby("world_id").normalized_mse.first()
T["init_nmse"] = init
print("of these, at target after init:", z.index.map(lambda i: init[i] <= 0.04).tolist())
print("systems where init NMSE is WORSE than all-zero prediction:", int((T.init_nmse > T.zero_pred_nmse).sum()), T[T.init_nmse > T.zero_pred_nmse].short.tolist())
print(T[["short", "graph_stratum", "regime"] + [f"t{i}" for i in range(8)]].round(4).to_string(index=False))
# structure check on recovered truth: chain t6==t7; fork t4==t5==t6==t0, t7==t1; x-linear t1,t2,t3; y-linear t4,t5,t6
for w, r in T.iterrows():
    t = r[[f"t{i}" for i in range(8)]].values.astype(float)
    xlin = abs(t[1] + t[3] - 2 * t[2]); ylin = abs(t[4] + t[6] - 2 * t[5])
    extra = {"chain": abs(t[6] - t[7]), "fork": max(abs(t[4] - t[0]), abs(t[5] - t[0]), abs(t[6] - t[0]), abs(t[7] - t[1])),
             "collider": abs((t[1] - t[2]) + (t[6] - t[5]) + t[0] - t[7] - (t[0]-t[2]) - (t[0]-t[5]))}[r.graph_stratum]
    if max(xlin, ylin, extra) > 1e-9:
        print("STRUCTURE VIOLATION", r.short, xlin, ylin, extra)
print("structure checks done")
T.to_csv(OUT + "out_truth.csv")
