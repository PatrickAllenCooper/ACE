"""Verifier step 6: curve-area endpoints under several defensible definitions; ranking of policies; paired stats vs TR.
Fresh code. Curves are step functions of total cost (state after each accepted response); last observation carried forward (LOCF)
to the common budget of 32 (matters for sft_llm early stops and for 2-unit actions that leave an unspendable unit)."""
import pandas as pd, numpy as np, itertools
from scipy import stats
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
W = sorted(cells.world_id.unique())
# step[w][p][c] for c in 8..32 = NMSE of the latest state with total cost <= c
grid = np.arange(8, 33)
step = {}
for (w, p), g in curves.groupby(["world_id", "policy_id"]):
    s = pd.Series(g.normalized_mse.values, index=g.total_cost_units.values)
    step[(w, p)] = s.reindex(grid).ffill().values
    assert not np.isnan(step[(w, p)]).any()
S = np.array([[step[(w, p)] for p in POL] for w in W])   # (24, 8, 25)
np.save(OUT + "out_step_nmse.npy", S)
L = np.log10(S)
defs = {
    "A: mean log10 NMSE, grid 8..32 (25 pts)": L.mean(axis=2),
    "B: mean log10 NMSE, grid 9..32 (24 pts)": L[:, :, 1:].mean(axis=2),
    "C: step integral over [8,32) = grid 8..31 (24 pts)": L[:, :, :-1].mean(axis=2),
    "D: trapezoid over [8,32] / 24": ((L[:, :, :-1] + L[:, :, 1:]) / 2).mean(axis=2),
    "E: log10 of mean raw NMSE, grid 8..32": np.log10(S.mean(axis=2)),
    "F: log10 of mean raw NMSE, grid 9..32": np.log10(S[:, :, 1:].mean(axis=2)),
}
claimed = dict(uniform_random=-1.709, menu_coverage=-1.842, local_information=-1.861, target_risk=-1.892, learned_ranker=-1.761, frozen_llm=-1.700, sft_llm=-1.790, dpo_llm=-1.760)
pd.set_option("display.width", 250)
tab = pd.DataFrame({k: v.mean(axis=0) for k, v in defs.items()}, index=POL)
tab["claimed"] = pd.Series(claimed)
print(tab.round(4).to_string())
print()
for k, v in defs.items():
    m = pd.Series(v.mean(axis=0), index=POL)
    print(k, "| max abs diff from claimed:", round(float((m - pd.Series(claimed)).abs().max()), 4), "| ranking:", " < ".join(m.sort_values().index))
print("claimed ranking:", " < ".join(pd.Series(claimed).sort_values().index))
# raw-NMSE areas for reference
raw = pd.DataFrame({"mean raw NMSE area 8..32": S.mean(axis=2).mean(axis=0), "mean raw NMSE area 9..32": S[:, :, 1:].mean(axis=2).mean(axis=0),
                    "median log-area(A)": np.median(defs["A: mean log10 NMSE, grid 8..32 (25 pts)"], axis=0)}, index=POL)
print(raw.round(4).to_string())
print("raw-area ranking:", " < ".join(raw.iloc[:, 0].sort_values().index))
print("median log-area(A) ranking:", " < ".join(raw.iloc[:, 2].sort_values().index))
