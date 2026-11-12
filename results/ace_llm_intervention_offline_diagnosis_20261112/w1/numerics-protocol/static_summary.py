import math
import numpy as np, pandas as pd
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/"
df = pd.read_pickle(OUT + "static_designs.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
gm = lambda s: math.exp(np.log(s).mean())
cols = ["R8", "real_target_risk", "real_local_information", "real_menu_coverage", "real_uniform_random", "real_learned_ranker",
        "J_planonce", "J_target_risk", "J_local_information", "J_TRpartner", "J_TRothercell_mean", "J_menu_coverage", "J_uniform_random", "J_learned_ranker",
        "J_obs12_xneg6_xpos6", "J_obs8_xneg8_xpos8", "J_obs24", "J_xneg24", "J_xpos24"]
print("== arithmetic and geometric means over 24 systems (final posterior expected NMSE) ==")
print(pd.DataFrame({"mean": df[cols].mean(), "geomean": df[cols].apply(gm), "n<=0.04": (df[cols] <= 0.04).sum()}).round(5).to_string())
print("\nby graph (geometric mean):")
print(df.groupby("graph")[cols].agg(gm).T.round(5).to_string())

def lr(a, b, label):
    d = np.log(df[a]) - np.log(df[b])
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"{label:70s} mean log-ratio {d.mean():+.4f} (x{math.exp(d.mean()):.3f})  sd {d.std(ddof=1):.4f} se {se:.4f} t {d.mean()/se:+.2f}  a<b in {int((d<0).sum())}/24")
print("\n== paired log-ratios (a/b) over 24 systems ==")
lr("real_target_risk", "J_target_risk", "TR realized (adaptive) / same action multiset replayed open-loop")
lr("J_target_risk", "J_planonce", "TR's realized multiset (open-loop) / plan-once design from s8")
lr("real_target_risk", "J_planonce", "TR realized (adaptive) / plan-once open-loop design from s8")
lr("real_local_information", "J_planonce", "LI realized (adaptive) / plan-once open-loop design from s8")
lr("J_TRpartner", "J_target_risk", "TR multiset transplanted from partner system (public-only) / own TR multiset")
lr("J_TRothercell_mean", "J_target_risk", "TR multisets from the 2 other same-graph/cost systems / own TR multiset")
lr("J_TRpartner", "J_planonce", "partner-transplant (public-only static) / plan-once")
lr("J_obs12_xneg6_xpos6", "J_planonce", "fixed public schedule 12 obs + 6 x- + 6 x+ / plan-once")
lr("J_obs8_xneg8_xpos8", "J_planonce", "fixed public schedule 8 obs + 8 x- + 8 x+ / plan-once")
lr("J_menu_coverage", "J_planonce", "coverage schedule / plan-once")
lr("J_uniform_random", "J_planonce", "random's realized multiset (open-loop) / plan-once")
lr("J_obs24", "J_planonce", "24 observations (frozen-LLM design) / plan-once")
lr("J_xneg24", "J_planonce", "24 x set_x_neg (DPO design) / plan-once")
lr("J_learned_ranker", "J_planonce", "ranker's realized multiset (open-loop) / plan-once")
lr("J_obs12_xneg6_xpos6", "real_target_risk", "fixed public schedule 12 obs + 6 x- + 6 x+ / TR realized")
lr("J_obs12_xneg6_xpos6", "J_menu_coverage", "fixed public schedule 12/6/6 / coverage schedule")
lr("real_menu_coverage", "J_menu_coverage", "coverage realized / coverage model-expected (calibration)")
lr("real_uniform_random", "J_uniform_random", "random realized / random-multiset model-expected (calibration)")

print("\n== standardized residuals z=(realized - J)/fantasy_sd, by policy and regime family ==")
for pol in ["menu_coverage", "uniform_random", "target_risk", "local_information", "learned_ranker"]:
    z = (df[f"real_{pol}"] - df[f"J_{pol}"]) / df[f"sd_{pol}"]
    m = df.regime.str.startswith("matched")
    print(f"{pol:18s} all: mean z {z.mean():+.3f} (se {z.std(ddof=1)/math.sqrt(24):.3f}), sd z {z.std(ddof=1):.3f} | matched mean z {z[m].mean():+.3f} | shifted mean z {z[~m].mean():+.3f} | frac |z|>2: {(z.abs()>2).mean():.3f}")
print("\nMC standard error of J (fantasy sd / sqrt(4000)) relative to J: median %.4f" % (df.sd_target_risk / math.sqrt(4000) / df.J_target_risk).median())
print("relative fantasy sd of final risk (sd/J) for TR multiset: median %.3f, range %.3f-%.3f" % ((df.sd_target_risk / df.J_target_risk).median(), (df.sd_target_risk / df.J_target_risk).min(), (df.sd_target_risk / df.J_target_risk).max()))
print("\nplan-once designs:")
print(df[["world", "graph", "cost", "regime", "planonce_design", "J_planonce", "planonce_insample", "J_target_risk", "real_target_risk"]].round(4).to_string(index=False))
# How many systems reach model-expected risk <= 0.04 under each design
print("\n# systems with J<=0.04:", {c: int((df[c] <= 0.04).sum()) for c in cols})
