"""Verifier step 15: does target_risk's action MIX vary between systems inside a (graph x cost-menu) cell?
If yes, 'an action mix plus one sign bit' understates its state dependence. Heuristic chi-square (actions within a run are not independent)."""
import pandas as pd, numpy as np, pickle
from scipy import stats
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
seqs = pickle.load(open(OUT + "out_seqs.pkl", "rb"))
W = meta.index.tolist()
def tok(w, a):
    far = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
    if a == far: return "x_far"
    if a in ("set_x_neg", "set_x_pos"): return "x_near"
    if a in ("set_y_neg", "set_y_zero", "set_y_pos"): return "y_any"
    return {"observe_all": "obs", "set_x_zero": "x0", "set_xy": "xy"}[a]
for p in ("target_risk", "local_information"):
    print("=====", p)
    rows = []
    for w in W:
        s = pd.Series([tok(w, a) for a in seqs[(w, p)]]).value_counts()
        rows.append(dict(world=f"w{int(str(w)[-2:]):02d}", graph=meta.loc[w, "graph_stratum"], cost=meta.loc[w, "cost_stratum"], regime=meta.loc[w, "regime"], n=len(seqs[(w, p)]), **s.to_dict()))
    T = pd.DataFrame(rows).fillna(0)
    cols = [c for c in ("obs", "x_far", "x_near", "x0", "y_any", "xy") if c in T.columns]
    T[cols] = T[cols].astype(int)
    pd.set_option("display.width", 200)
    print(T.sort_values(["graph", "cost", "world"]).to_string(index=False))
    for (g, c), sub in T.groupby(["graph", "cost"]):
        M = sub[cols].values
        M = M[:, M.sum(axis=0) > 0]
        chi2, pv, dof, _ = stats.chi2_contingency(M)
        n = M.sum(); v = np.sqrt(chi2 / (n * (min(M.shape) - 1)))
        ysh = (sub[[c2 for c2 in ("y_any", "xy") if c2 in sub]].sum(axis=1) / sub.n).round(2).tolist()
        osh = (sub.obs / sub.n).round(2).tolist()
        print(f"  cell {g:8s} {c:13s}: chi2={chi2:6.1f} dof={dof:2d} heuristic p={pv:.2g} Cramer V={v:.2f} | observe share by system {osh} | Y-involving share by system {ysh}")
