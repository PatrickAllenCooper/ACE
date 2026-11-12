"""Descriptive offline look at the 24-system screen endpoint sensitivity (saved cells.csv only).
NOT a new significance claim; used only to size future experiments.
Usage: python3 screen_power.py cells.csv
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1])
cost = df.pivot(index="world_id", columns="policy_id", values="capped_total_cost")
err = df.pivot(index="world_id", columns="policy_id", values="final_error")
order = ["uniform_random","menu_coverage","local_information","target_risk","learned_ranker","frozen_llm","sft_llm","dpo_llm"]
cost = cost[order]; err = err[order]
print("n systems", len(cost))
init_met = (cost == 8).all(axis=1)
print("systems where every policy has capped cost 8 (target met at init):", int(init_met.sum()))
print("systems where at least one policy capped at 32:", int((cost == 32).any(axis=1).sum()))
print("\nmean capped cost, all 24:"); print(cost.mean().round(3).to_string())
print("\nmean capped cost, the", int((~init_met).sum()), "systems not met at init:"); print(cost[~init_met].mean().round(3).to_string())
print("\nper-policy distribution of capped cost (value: count)")
for p in order:
    print(f"  {p:18s}", dict(sorted(cost[p].value_counts().to_dict().items())))
print("\nPaired differences vs target_risk on capped cost (policy - target_risk): mean, sd, t-CI, wins/ties/losses")
for p in order:
    if p == "target_risk": continue
    d = cost[p] - cost["target_risk"]
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    # sample size for 80% power to detect the observed mean diff at two-sided 0.05 (normal approx)
    need = math.ceil(((1.96 + 0.8416) * sd / abs(m)) ** 2) if m != 0 else float("inf")
    print(f"  {p:18s} mean={m:+.3f} sd={sd:.3f} 95%CI=[{m - t*se:+.3f},{m + t*se:+.3f}] lower/tie/higher={(d<0).sum()}/{(d==0).sum()}/{(d>0).sum()}  n_for_80%power_at_observed_effect~{need}")
print("\nPaired log final-error ratio vs target_risk (policy/target_risk): geo ratio, CI, wins")
for p in order:
    if p == "target_risk": continue
    z = np.log(err[p]) - np.log(err["target_risk"])
    n = len(z); m = z.mean(); se = z.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    print(f"  {p:18s} geo ratio={math.exp(m):.3f} [{math.exp(m - t*se):.3f},{math.exp(m + t*se):.3f}] policy lower error in {(z<0).sum()}/{n}")
print("\ngeometric mean final error per policy:"); print(np.exp(np.log(err).mean()).round(5).to_string())
print("\nmedian final error per policy:"); print(err.median().round(5).to_string())
print("\nStrata: mean capped cost by graph and cost stratum")
print(df.pivot_table(index="policy_id", columns="graph_stratum", values="capped_total_cost", aggfunc="mean").loc[order].round(2).to_string())
print(df.pivot_table(index="policy_id", columns="cost_stratum", values="capped_total_cost", aggfunc="mean").loc[order].round(2).to_string())
print(df.pivot_table(index="policy_id", columns="regime", values="capped_total_cost", aggfunc="mean").loc[order].round(2).to_string())
print("\nbest policy per system on capped cost (ties share):")
best = cost.eq(cost.min(axis=1), axis=0).sum().to_dict(); print(best)
print("unique-best counts:", {p: int(((cost[p] == cost.min(axis=1)) & (cost.eq(cost.min(axis=1), axis=0).sum(axis=1) == 1)).sum()) for p in order})
