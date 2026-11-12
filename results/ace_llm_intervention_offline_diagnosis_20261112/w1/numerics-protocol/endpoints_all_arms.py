"""Alternative realized-error endpoints for all 8 arms from the saved curves.csv (no new evaluation)."""
import math, itertools
import numpy as np, pandas as pd
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
curves = pd.read_csv(D + "curves.csv")
POL = ["uniform_random", "menu_coverage", "local_information", "target_risk", "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
recs = []
for (w, p), g in curves.groupby(["world_id", "policy_id"]):
    g = g.sort_values("accepted_receipts")
    c = g.acquisition_cost_units.values; v = g.normalized_mse.values
    ok = v <= 0.04
    first = int(c[np.argmax(ok)]) if ok.any() else 24
    if ok[-1]:
        k = len(ok) - 1
        while k > 0 and ok[k - 1]: k -= 1
        sustained = int(c[k])
    else:
        sustained = 24
    cc = np.r_[c, 24]  # carry the last state forward to the common cap (early stop keeps its last predictions)
    w_ = np.diff(cc)
    recs.append(dict(world=w, policy=p, capped=8 + first, sustained=8 + sustained, final=v[-1], auc=float((v * w_).sum() / 24),
                     auc_log=float((np.log(v) * w_).sum() / 24), frac_time_ok=float((ok * w_).sum() / 24),
                     recross=int(np.sum(np.diff(ok.astype(int)) != 0)), ever=bool(ok.any()), final_ok=bool(ok[-1]),
                     n_points=len(v), max_after_first=float(v[np.argmax(ok):].max()) if ok.any() else np.nan,
                     nmse_c12=float(v[c <= 12][-1])))
e = pd.DataFrame(recs)
e.to_csv("out_endpoints_all_arms.csv", index=False)
gm = lambda s: math.exp(np.log(s).mean())
t = e.groupby("policy").agg(capped=("capped", "mean"), sustained=("sustained", "mean"), ever=("ever", "sum"), final_ok=("final_ok", "sum"),
        final_mean=("final", "mean"), final_geo=("final", gm), auc=("auc", "mean"), auc_geo=("auc", gm), auc_log=("auc_log", "mean"),
        frac_time_ok=("frac_time_ok", "mean"), recross=("recross", "mean"), lost_after_cross=("recross", lambda s: 0)).loc[POL]
t["crossed_then_ended_above"] = (e.assign(x=e.ever & ~e.final_ok).groupby("policy").x.sum()).loc[POL]
print(t.drop(columns="lost_after_cross").round(4).to_string())
tot = e.assign(x=e.ever & ~e.final_ok)
print("\nruns that crossed at least once but ended above threshold:", int(tot.x.sum()), "of", int(e.ever.sum()), "ever-attained runs (192 total)")
print("runs with >=2 threshold direction changes:", int((e.recross >= 2).sum()), "; total direction changes:", int(e.recross.sum()))
# among the 10 init-crossing systems (80 runs): how many end above threshold / leave the band
init_ok = e[e.capped == 8].world.value_counts()
ws = [w for w, n in init_ok.items() if n == 8]
sub = e[e.world.isin(ws)]
print("init-crossing systems:", len(ws), "; runs:", len(sub), "; runs ending above threshold:", int((~sub.final_ok).sum()), "; runs leaving the band at least once:", int((sub.recross >= 1).sum()))
print(sub.groupby("policy").agg(end_above=("final_ok", lambda s: int((~s).sum())), left_band=("recross", lambda s: int((s >= 1).sum())), max_after=("max_after_first", "max")).loc[POL].to_string())

def paired(col, a, b, log=False):
    A = e[e.policy == a].set_index("world")[col]; B = e[e.policy == b].set_index("world")[col]
    d = (np.log(A) - np.log(B)) if log else (A - B)
    se = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.std(ddof=1), se, d.mean() / se
print("\npaired t-like statistics (descriptive) for selected contrasts under each realized endpoint:")
for a, b in [("target_risk", "uniform_random"), ("target_risk", "menu_coverage"), ("dpo_llm", "target_risk"), ("dpo_llm", "frozen_llm"), ("dpo_llm", "uniform_random"), ("sft_llm", "frozen_llm")]:
    out = []
    for col, log in [("capped", False), ("sustained", False), ("final", True), ("auc", True), ("auc_log", False), ("frac_time_ok", False)]:
        m, sd, se, tt = paired(col, a, b, log)
        out.append(f"{col}{'(log)' if log else ''}: {m:+.3f}±{se:.3f} t={tt:+.2f}")
    print(f"  {a} - {b}: " + " | ".join(out))
