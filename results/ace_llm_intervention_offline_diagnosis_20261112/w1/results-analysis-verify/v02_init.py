"""Verifier step 2: systems at target after initialization; identity and strata; near-threshold; threshold sweep counts."""
import pandas as pd, numpy as np
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
PC = [f"prediction_{i}" for i in range(8)]
init = curves[curves.total_cost_units == 8]
assert len(init) == 192 and init.action_id.isna().all()
# is the init state shared across policies within a world?
sp = init.groupby("world_id").normalized_mse.agg(["min", "max", "nunique"])
print("max within-world spread of init NMSE across the 8 policies:", (sp["max"] - sp["min"]).max())
predspread = init.groupby("world_id")[PC].agg(lambda s: s.max() - s.min()).max().max()
print("max within-world spread of init predictions:", predspread)
w = cells.drop_duplicates("world_id")[["world_id", "cost_stratum", "graph_stratum", "regime"]].set_index("world_id").sort_index()
w["init_nmse"] = init.groupby("world_id").normalized_mse.first()
w["at_target_0.04"] = w.init_nmse <= 0.04
w["short"] = [f"w{int(str(i)[-2:]):02d}" for i in w.index]
w["match"] = w.regime.str.split("_").str[0]
w["noise"] = w.regime.str.split("_").str[1]
pd.set_option("display.width", 200)
print(w.to_string())
print()
print("N at target after init (<=0.04):", int(w["at_target_0.04"].sum()))
print("identity:", w[w["at_target_0.04"]].short.tolist())
print("by graph:", w.groupby("graph_stratum")["at_target_0.04"].agg(["sum", "count"]).to_dict("index"))
print("by match:", w.groupby("match")["at_target_0.04"].agg(["sum", "count"]).to_dict("index"))
print("by noise:", w.groupby("noise")["at_target_0.04"].agg(["sum", "count"]).to_dict("index"))
print("by regime:", w.groupby("regime")["at_target_0.04"].agg(["sum", "count"]).to_dict("index"))
print("by cost menu:", w.groupby("cost_stratum")["at_target_0.04"].agg(["sum", "count"]).to_dict("index"))
# also verify via cells: every policy capped_total_cost == 8 on those worlds
c8 = cells.groupby("world_id").capped_total_cost.agg(lambda s: (s == 8).all())
print("worlds where all 8 policies have capped cost 8:", int(c8.sum()), "matches init flag:", bool((c8 == w["at_target_0.04"]).all()))
print()
print("init NMSE min/median/max:", w.init_nmse.min(), w.init_nmse.median(), w.init_nmse.max())
print("sorted init NMSE:")
print(w.sort_values("init_nmse")[["short", "graph_stratum", "regime", "cost_stratum", "init_nmse"]].to_string(index=False))
print()
# 'within 12.5 percent of the threshold': above 0.04 but <= 0.045
near = w[(w.init_nmse > 0.04) & (w.init_nmse <= 0.04 * 1.125)]
print("not at target but within +12.5% of 0.04 (<=0.045):", near.short.tolist(), near.init_nmse.round(5).tolist())
for pct in (0.05, 0.10, 0.125, 0.15, 0.20, 0.25):
    n_above = int(((w.init_nmse > 0.04) & (w.init_nmse <= 0.04 * (1 + pct))).sum())
    n_below = int(((w.init_nmse <= 0.04) & (w.init_nmse >= 0.04 * (1 - pct))).sum())
    print(f"  within +{pct:.3f}: {n_above} above-threshold systems; within -{pct:.3f}: {n_below} at-target systems")
print()
for th in (0.01, 0.02, 0.04):
    print(f"threshold {th}: at target after init = {int((w.init_nmse <= th).sum())}")
w.to_csv("/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/out_worlds.csv")
