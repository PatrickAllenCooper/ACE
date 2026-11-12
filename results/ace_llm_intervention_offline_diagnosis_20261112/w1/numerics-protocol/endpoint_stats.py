"""Offline descriptive analysis of saved 24-system reporting CSVs (read-only)."""
import itertools, math
import numpy as np, pandas as pd
from scipy import stats

D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
POL = ["uniform_random", "menu_coverage", "local_information", "target_risk",
       "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

print("cells", cells.shape, "curves", curves.shape)
wide = cells.pivot(index="world_id", columns="policy_id", values="capped_total_cost")[POL]
meta = cells.drop_duplicates("world_id").set_index("world_id")[["regime", "graph_stratum", "cost_stratum"]]
init = curves[curves.acquisition_cost_units == 0].groupby("world_id").normalized_mse.agg(["min", "max"])
assert np.allclose(init["min"], init["max"])
meta["init_nmse"] = init["min"]
tab = meta.join(wide)
print(tab.to_string())
print("\nmean capped total:\n", wide.mean().round(3).to_string())
print("\n# systems at/below 0.04 after initialization:", int((meta.init_nmse <= 0.04).sum()))
print(meta.assign(init_ok=meta.init_nmse <= 0.04).groupby("regime").init_ok.sum().to_string())
print(meta.assign(init_ok=meta.init_nmse <= 0.04).groupby("graph_stratum").init_ok.sum().to_string())
print(meta.assign(init_ok=meta.init_nmse <= 0.04).groupby("cost_stratum").init_ok.sum().to_string())
print("init NMSE quantiles:", meta.init_nmse.quantile([0, .25, .5, .75, 1]).round(4).to_dict())
print("init NMSE mean by regime:\n", meta.groupby("regime").init_nmse.agg(["mean", "median", "min", "max"]).round(4).to_string())

# envelope / floor
env = wide.min(axis=1)
print("\nlower envelope (min over 8 arms) mean capped total:", round(env.mean(), 3))
print("lower envelope over 5 numeric arms:", round(wide[POL[:5]].min(axis=1).mean(), 3))
n_init = int((meta.init_nmse <= 0.04).sum())
print("absolute floor given init crossers (8 + >=1 unit for others):", round(8 + (24 - n_init) / 24, 3))
print("number of systems where ALL 8 arms have identical capped cost:", int((wide.nunique(axis=1) == 1).sum()))
print("number of systems where all 8 arms == 8:", int((wide.max(axis=1) == 8).sum()))
print("systems where no arm ever attains:", int((wide.min(axis=1) == 32).sum()))
print("per-policy count at floor 8 / at cap 32:")
for p in POL:
    att = cells[cells.policy_id == p]
    print(f"  {p:18s} floor8={int((wide[p]==8).sum()):2d} cap32={int((wide[p]==32).sum()):2d} ever={int(att.ever_attained.sum())} final={int(att.final_attained.sum())}")
noninit = meta.index[meta.init_nmse > 0.04]
print("\nOn the", len(noninit), "systems not attained at init: mean capped total by policy")
print(wide.loc[noninit].mean().round(3).to_string())

# paired contrasts
rows = []
for a, b in itertools.combinations(POL, 2):
    d = (wide[a] - wide[b]).values
    nz = d[d != 0]
    sd = d.std(ddof=1); se = sd / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("nan")
    wins, losses = int((d < 0).sum()), int((d > 0).sum())
    p_sign = stats.binomtest(min(wins, losses), wins + losses, 0.5).pvalue if wins + losses else float("nan")
    try:
        p_w = stats.wilcoxon(d, zero_method="wilcox").pvalue if len(nz) else float("nan")
    except Exception:
        p_w = float("nan")
    rows.append((a, b, d.mean(), sd, se, t, wins, int((d == 0).sum()), losses, p_sign, p_w))
pairs = pd.DataFrame(rows, columns=["left", "right", "mean_diff", "sd", "se", "t", "left_lower", "ties", "left_higher", "p_sign", "p_wilcoxon"])
print("\nPaired contrasts on capped total cost (left - right), n=24 systems:")
print(pairs.round(3).to_string())
print("\nmedian SD of paired differences:", round(pairs.sd.median(), 3), " median SE:", round(pairs.se.median(), 3))
tr = pairs[(pairs.left == "target_risk") | (pairs.right == "target_risk")]
print("\nContrasts involving target_risk:\n", tr.round(3).to_string())

# bootstrap CI for key contrasts
rng = np.random.default_rng(20261112)
def boot(a, b, B=20000):
    d = (wide[a] - wide[b]).values
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return d.mean(), np.percentile(m, [2.5, 97.5])
for a, b in [("target_risk", "menu_coverage"), ("target_risk", "uniform_random"), ("target_risk", "local_information"),
             ("dpo_llm", "target_risk"), ("dpo_llm", "uniform_random"), ("dpo_llm", "frozen_llm"), ("menu_coverage", "uniform_random")]:
    m, ci = boot(a, b)
    print(f"bootstrap {a} - {b}: mean {m:.3f}, 95% percentile CI [{ci[0]:.3f}, {ci[1]:.3f}]")
tab.to_csv("/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/out_capped_cost_by_world.csv")
pairs.to_csv("/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/out_pairs_stats.csv", index=False)
