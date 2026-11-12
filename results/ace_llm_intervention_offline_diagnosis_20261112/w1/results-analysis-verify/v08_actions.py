"""Verifier step 8: action-sequence structure of target_risk and local_information.
- recover the SCM's estimated natural X mean from saved predictions: mu_X = (p0 - p2) / (p3 - p2) (valid for all three graphs, since the 8 target queries are the 8 menu actions; see llm_intervention_screen_protocol.py _public()).
- first action vs the farther X extreme; action mixes; Y-involving share by cost menu.
Fresh code."""
import pandas as pd, numpy as np
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
meta = cells.drop_duplicates("world_id").set_index("world_id")[["cost_stratum", "graph_stratum", "regime"]].sort_index()
W = meta.index.tolist()
sh = lambda w: f"w{int(str(w)[-2:]):02d}"
curves["mux"] = (curves.prediction_0 - curves.prediction_2) / (curves.prediction_3 - curves.prediction_2)
init = curves[curves.total_cost_units == 8].drop_duplicates("world_id").set_index("world_id")
meta["mux0"] = init.mux
meta["slope_x0"] = init.prediction_3 - init.prediction_2
meta["far_extreme"] = np.where(meta.mux0 > 0, "set_x_neg", "set_x_pos")
seqs = {(w, p): g.action_id.dropna().tolist() for (w, p), g in curves.groupby(["world_id", "policy_id"])}
for p in ("target_risk", "local_information"):
    meta[f"first_{p}"] = [seqs[(w, p)][0] for w in W]
    meta[f"second_{p}"] = [seqs[(w, p)][1] for w in W]
meta["short"] = [sh(w) for w in W]
pd.set_option("display.width", 250)
print(meta[["short", "graph_stratum", "regime", "cost_stratum", "mux0", "slope_x0", "far_extreme", "first_target_risk", "second_target_risk", "first_local_information", "second_local_information"]].round(4).to_string(index=False))
print("systems with estimated natural X mean < 0 (X=-1 is the NEAR side):", int((meta.mux0 < 0).sum()), meta[meta.mux0 < 0].short.tolist())
for p in ("target_risk", "local_information"):
    f = meta[f"first_{p}"]
    print(f"{p}: first action is an X-extreme: {int(f.isin(['set_x_neg','set_x_pos']).sum())}/24; equals farther extreme: {int((f == meta.far_extreme).sum())}/24; exceptions: {meta[f != meta.far_extreme][['short', 'mux0', 'far_extreme', f'first_{p}']].values.tolist()}")
    s = meta[f"second_{p}"]
    print(f"   second action distribution: {s.value_counts().to_dict()}")
    # ever uses the near extreme?
    near = np.where(meta.far_extreme == "set_x_neg", "set_x_pos", "set_x_neg")
    nfar = sum(seqs[(w, p)].count(meta.loc[w, "far_extreme"]) for w in W)
    nnear = sum(seqs[(w, p)].count(n) for w, n in zip(W, near))
    print(f"   X-extreme uses over all steps: far(init-estimate) {nfar}, near {nnear}")
# does mu_X change over runs for dpo / ranker / others
print("\n--- does the recovered X-mean estimate change during the run? ---")
for p in POL:
    nconst = 0
    for w in W:
        g = curves[(curves.world_id == w) & (curves.policy_id == p)]
        if np.ptp(g.mux.values) < 1e-9 * max(1, abs(g.mux.values[0])): nconst += 1
    print(f"{p}: X-mean estimate constant in {nconst}/24 runs")
# action mix
print("\n--- action mix, target_risk and local_information ---")
A = curves.dropna(subset=["action_id"]).merge(meta[["cost_stratum", "graph_stratum", "regime"]], left_on="world_id", right_index=True)
A["y_involving"] = A.action_id.isin(["set_y_neg", "set_y_zero", "set_y_pos", "set_xy"])
A["step"] = A.groupby(["world_id", "policy_id"]).cumcount()
for p in ("target_risk", "local_information"):
    a = A[A.policy_id == p]
    print(p, "n actions", len(a), "observe share", round((a.action_id == "observe_all").mean(), 4), "Y-involving share overall", round(a.y_involving.mean(), 4))
    print("   Y-involving share by cost menu:", a.groupby("cost_stratum").y_involving.agg(["sum", "count", "mean"]).round(4).to_dict("index"))
    print("   Y-involving share by graph x menu:", a.groupby(["graph_stratum", "cost_stratum"]).y_involving.mean().round(3).to_dict())
    print("   Y-involving share of COST by cost menu:", {k: round(float(v), 4) for k, v in a.assign(c=np.where(a.y_involving & (a.cost_stratum == 'heterogeneous'), 2, 1)).groupby("cost_stratum").apply(lambda g: (g.c * g.y_involving).sum() / g.c.sum()).items()})
    print("   mix by graph:", a.groupby("graph_stratum").action_id.value_counts(normalize=True).round(3).unstack(fill_value=0).to_string().replace("\n", "\n      "))
    # position of first observe
    fo = [seqs[(w, p)].index("observe_all") if "observe_all" in seqs[(w, p)] else None for w in W]
    print("   index of first observe by system:", fo)
print("\n--- full sequences, target_risk ---")
ab = {"observe_all": "o", "set_x_neg": "x-", "set_x_zero": "x0", "set_x_pos": "x+", "set_y_neg": "y-", "set_y_zero": "y0", "set_y_pos": "y+", "set_xy": "xy"}
for w in W:
    print(sh(w), meta.loc[w, "graph_stratum"][:4], meta.loc[w, "regime"], meta.loc[w, "cost_stratum"][:3], f"mux0={meta.loc[w,'mux0']:+.2f}", " ".join(ab[a] for a in seqs[(w, "target_risk")]))
print("\n--- full sequences, local_information ---")
for w in W:
    print(sh(w), meta.loc[w, "graph_stratum"][:4], meta.loc[w, "regime"], meta.loc[w, "cost_stratum"][:3], f"mux0={meta.loc[w,'mux0']:+.2f}", " ".join(ab[a] for a in seqs[(w, "local_information")]))
meta.to_csv(OUT + "out_meta_mux.csv")
import pickle; pickle.dump(seqs, open(OUT + "out_seqs.pkl", "wb"))
