import itertools, math
import numpy as np, pandas as pd
from scipy import stats
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/"
df = pd.read_pickle(OUT + "replay_states.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
POL = ["uniform_random", "menu_coverage", "local_information", "target_risk", "learned_ranker"]

init = df[(df.n == 8) & (df.policy == "target_risk")].set_index("world")[["regime", "graph", "cost", "post_risk", "nmse"] + [f"rq{q}" for q in range(8)]]
init["ratio"] = init.nmse / init.post_risk
print("== Initialization (n=8) per system: learner posterior expected NMSE vs realized NMSE ==")
print(init.round(4).to_string())
print("\n# init: posterior risk <= 0.04:", int((init.post_risk <= 0.04).sum()), "; realized NMSE <= 0.04:", int((init.nmse <= 0.04).sum()))
print("init posterior risk: mean %.4f median %.4f min %.4f max %.4f" % (init.post_risk.mean(), init.post_risk.median(), init.post_risk.min(), init.post_risk.max()))
print(init.groupby("regime")[["post_risk", "nmse"]].mean().round(4).to_string())
print(init.groupby("graph")[["post_risk", "nmse"]].mean().round(4).to_string())
noise = init.regime.str.contains("low").map({True: "low(0.25)", False: "high(1.0)"})
print(init.groupby(noise)[["post_risk", "nmse"]].agg(["mean", "min", "max"]).round(4).to_string())

# calibration over all replayed states of the 5 numeric arms
df["matched"] = df.regime.str.startswith("matched")
cal = df.groupby("matched").apply(lambda g: pd.Series({"mean_nmse": g.nmse.mean(), "mean_post_risk": g.post_risk.mean(),
        "ratio_of_means": g.nmse.mean() / g.post_risk.mean(), "median_ratio": (g.nmse / g.post_risk).median(), "states": len(g)}))
print("\n== Calibration of posterior risk vs realized NMSE over all replayed states ==\n", cal.round(4).to_string())
fin = df.sort_values("n").groupby(["policy", "world"]).tail(1)
calf = fin.groupby("matched").apply(lambda g: pd.Series({"mean_final_nmse": g.nmse.mean(), "mean_final_post_risk": g.post_risk.mean(),
        "ratio": g.nmse.mean() / g.post_risk.mean(), "n": len(g)}))
print("final states only:\n", calf.round(4).to_string())

# endpoints based on posterior risk
def cross_cost(g, col, thr=0.04):
    g = g.sort_values("n")
    hit = g[g[col] <= thr]
    return 24 if hit.empty else int(hit.acq_cost.iloc[0])
def sustained_cost(g, col, thr=0.04):
    g = g.sort_values("n")
    ok = (g[col] <= thr).values
    if not ok[-1]:
        return 24
    k = len(ok) - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    return int(g.acq_cost.iloc[k])
def auc(g, col, log=False):
    # step-function area over acquisition cost 0..24 (value after each paid response holds until the next)
    g = g.sort_values("n")
    c = np.r_[g.acq_cost.values, 24]
    v = np.log(g[col].values) if log else g[col].values
    return float(np.sum(v * np.diff(c)) / 24)
recs = []
for (pol, w), g in df.groupby(["policy", "world"]):
    recs.append(dict(policy=pol, world=w, regime=g.regime.iloc[0], graph=g.graph.iloc[0], cost=g.cost.iloc[0],
        capped_real=8 + cross_cost(g, "nmse"), capped_post=8 + cross_cost(g, "post_risk"),
        sustained_real=8 + sustained_cost(g, "nmse"), sustained_post=8 + sustained_cost(g, "post_risk"),
        final_nmse=g.sort_values("n").nmse.iloc[-1], final_post=g.sort_values("n").post_risk.iloc[-1],
        auc_nmse=auc(g, "nmse"), auc_post=auc(g, "post_risk"), auc_lognmse=auc(g, "nmse", True), auc_logpost=auc(g, "post_risk", True),
        recross=int(np.sum(np.diff((g.sort_values("n").nmse.values <= 0.04).astype(int)) != 0))))
ep = pd.DataFrame(recs)
ep.to_csv(OUT + "out_endpoints_cpu_arms.csv", index=False)
agg = ep.groupby("policy")[["capped_real", "capped_post", "sustained_real", "sustained_post", "final_nmse", "final_post", "auc_nmse", "auc_post", "recross"]].mean().loc[POL]
print("\n== Mean endpoints for the 5 CPU arms (24 systems) ==\n", agg.round(4).to_string())
print("\ngeometric-mean final NMSE / final posterior risk:\n", ep.groupby("policy")[["final_nmse", "final_post"]].agg(lambda s: math.exp(np.log(s).mean())).loc[POL].round(5).to_string())
print("\n# systems with posterior risk <=0.04 at final state:\n", ep.assign(ok=ep.final_post <= 0.04).groupby("policy").ok.sum().loc[POL].to_string())
print("\n# systems with posterior-risk threshold ever crossed:\n", ep.assign(ok=ep.capped_post < 32).groupby("policy").ok.sum().loc[POL].to_string())

# paired effect sizes for several endpoints: target_risk vs each other arm
def paired(col, a, b, log=False):
    A = ep[ep.policy == a].set_index("world")[col]; B = ep[ep.policy == b].set_index("world")[col]
    d = (np.log(A) - np.log(B)) if log else (A - B)
    sd = d.std(ddof=1); se = sd / math.sqrt(len(d))
    return d.mean(), sd, se, (d.mean() / se if se > 0 else float("nan")), int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
print("\n== Paired contrasts target_risk - other (n=24): mean, sd, se, t, TR lower / tie / TR higher ==")
for col, log in [("capped_real", False), ("sustained_real", False), ("capped_post", False), ("sustained_post", False),
                 ("final_nmse", True), ("final_post", True), ("auc_nmse", True), ("auc_post", True), ("auc_lognmse", False), ("auc_logpost", False)]:
    print(f"-- endpoint {col}{' (log ratio)' if log else ''}")
    for b in ["uniform_random", "menu_coverage", "local_information", "learned_ranker"]:
        m, sd, se, t, lo, ti, hi = paired(col, "target_risk", b, log)
        print(f"   vs {b:18s} mean {m:8.4f} sd {sd:7.4f} se {se:7.4f} t {t:6.2f}  lower/tie/higher {lo}/{ti}/{hi}")

# per-graph breakdown of final posterior risk
print("\n== final posterior risk by graph x policy (mean) ==")
print(ep.pivot_table(index="graph", columns="policy", values="final_post", aggfunc="mean")[POL].round(4).to_string())
print("\n== final posterior risk by regime x policy (mean) ==")
print(ep.pivot_table(index="regime", columns="policy", values="final_post", aggfunc="mean")[POL].round(4).to_string())
print("\n== capped_post by graph x policy (mean) ==")
print(ep.pivot_table(index="graph", columns="policy", values="capped_post", aggfunc="mean")[POL].round(3).to_string())
print("\n== posterior risk at acquisition cost checkpoints (mean over 24 systems; last state with acq_cost<=c) ==")
chk = []
for (pol, w), g in df.groupby(["policy", "world"]):
    g = g.sort_values("n")
    for c in [0, 4, 8, 12, 16, 20, 24]:
        gg = g[g.acq_cost <= c]
        chk.append(dict(policy=pol, world=w, c=c, post=gg.post_risk.iloc[-1], nmse=gg.nmse.iloc[-1]))
chk = pd.DataFrame(chk)
print(chk.pivot_table(index="c", columns="policy", values="post", aggfunc="mean")[POL].round(4).to_string())
print("realized NMSE at the same checkpoints:")
print(chk.pivot_table(index="c", columns="policy", values="nmse", aggfunc="mean")[POL].round(4).to_string())
