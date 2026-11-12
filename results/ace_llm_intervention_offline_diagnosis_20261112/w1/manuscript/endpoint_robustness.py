"""Descriptive endpoint-robustness check on the saved 24-system curves (no model or evaluator rerun).
Compares the reported retrospective first-crossing endpoint with alternatives computed from the same saved curve points.
Usage: python3 endpoint_robustness.py curves.csv cells.csv
"""
import sys, math
import numpy as np, pandas as pd
cur = pd.read_csv(sys.argv[1]); cells = pd.read_csv(sys.argv[2])
TARGET, CAP = 0.04, 32
order = ["uniform_random","menu_coverage","local_information","target_risk","learned_ranker","frozen_llm","sft_llm","dpo_llm"]
rows = []
for (w, p), g in cur.groupby(["world_id", "policy_id"]):
    g = g.sort_values("total_cost_units")
    c = g.total_cost_units.to_numpy(); e = g.normalized_mse.to_numpy()
    below = e <= TARGET
    first = int(c[below][0]) if below.any() else CAP
    # sustained: first index from which all later evaluated points are <= target
    sustained = CAP
    for i in range(len(c)):
        if below[i:].all():
            sustained = int(c[i]); break
    # step-function area under error curve from cost 8 to 32 (hold last value to the cap)
    grid = np.arange(8, CAP + 1)
    idx = np.searchsorted(c, grid, side="right") - 1
    idx = np.clip(idx, 0, len(c) - 1)
    auc = float(e[idx].mean()); lauc = float(np.log(e[idx]).mean())
    # number of crossings back above target after first attainment
    relapses = int(((~below[1:]) & below[:-1]).sum())
    rows.append(dict(world_id=w, policy_id=p, first=first, sustained=sustained, auc=auc, lauc=lauc,
                     final=e[-1], npts=len(c), last_cost=int(c[-1]), relapses=relapses, init_err=e[0]))
r = pd.DataFrame(rows)
chk = r.merge(cells[["world_id", "policy_id", "capped_total_cost", "final_error"]], on=["world_id", "policy_id"])
print("first-crossing reproduces reported capped_total_cost in", int((chk["first"] == chk.capped_total_cost).sum()), "of", len(chk), "cells")
print("final error reproduces in", int(np.isclose(chk.final, chk.final_error).sum()), "of", len(chk))
agg = r.groupby("policy_id").agg(first=("first", "mean"), sustained=("sustained", "mean"), auc=("auc", "mean"),
                                 geo_auc=("lauc", lambda s: math.exp(s.mean())), mean_final=("final", "mean"),
                                 relapses=("relapses", "sum"), runs_with_relapse=("relapses", lambda s: int((s > 0).sum()))).loc[order]
print(agg.round(4).to_string())
print("\nrank by each endpoint (1=best):")
print(agg[["first", "sustained", "auc", "geo_auc", "mean_final"]].rank().astype(int).to_string())
print("\ninitial (cost 8) error: systems with init error <= target:", int((r[r.policy_id == 'target_risk'].init_err <= TARGET).sum()),
      "| init error identical across policies within world:", bool(r.groupby('world_id').init_err.nunique().max() == 1))
print("init error quantiles over 24 systems:", r[r.policy_id == 'target_risk'].init_err.quantile([0, .25, .5, .75, 1]).round(4).to_dict())
# paired sustained-cost differences vs target risk
piv = r.pivot(index="world_id", columns="policy_id", values="sustained")
from scipy import stats
print("\npaired sustained-cost difference vs target_risk: mean [95% t-CI], lower/tie/higher")
for p in order:
    if p == "target_risk": continue
    d = piv[p] - piv["target_risk"]; n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(.975, n - 1)
    print(f"  {p:18s} {m:+.3f} [{m - t*se:+.3f},{m + t*se:+.3f}] {(d<0).sum()}/{(d==0).sum()}/{(d>0).sum()}")
