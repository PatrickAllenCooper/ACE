"""How good can a PUBLIC-ONLY static schedule (a function of graph and cost menu, not of the system's data) be,
measured by model-expected final NMSE from the saved post-initialization states?  Greedy forward search of one
shared schedule per (graph, cost) cell: (a) in-sample over the cell's 4 systems (optimistic for static),
(b) leave-one-system-out (honest).  Model-internal fantasy integration only; no environment responses."""
import math, numpy as np, pandas as pd
from acelib import *
import static_designs as sd
sd.K = 2000
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
plan = pd.read_pickle("static_designs.pkl").set_index("world")
S = {}
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    S[wi] = (posterior(parents, noise, 1.0, hist[:8]), parents, noise)

def shared_greedy(worlds, cost, seed_base=31000, budget=24):
    bs = {w: sd.Batch(*S[w], seed_base + w) for w in worlds}
    cur = {w: float(bs[w].risk().mean()) for w in worlds}
    design, left = [], budget
    while True:
        feas = [a for a in ACTIONS if action_cost(a, cost) <= left]
        if not feas: break
        best = None
        for a in feas:
            vals = {w: float(bs[w].copy().add(a).risk().mean()) for w in worlds}
            score = np.mean([math.log(cur[w]) - math.log(vals[w]) for w in worlds]) / action_cost(a, cost)  # mean log-risk reduction per cost
            if best is None or score > best[0] + 1e-15:
                best = (score, a, vals)
        design.append(best[1]); left -= action_cost(best[1], cost)
        for w in worlds: bs[w].add(best[1])
        cur = best[2]
    return design

rows = []
cells = {}
for wi in range(24):
    cells.setdefault(meta(wi)[1:], []).append(wi)
for (graph, cost), ws in cells.items():
    d_in = shared_greedy(ws, cost)
    desc = " ".join(f"{a}:{d_in.count(a)}" for a in ACTIONS if d_in.count(a))
    print(f"cell {graph:8s} {cost:13s} systems {ws} in-sample shared schedule: {desc}", flush=True)
    for w in ws:
        d_loo = shared_greedy([x for x in ws if x != w], cost)
        partner = (w + 12) % 24
        d_pp = shared_greedy([partner], cost)
        rows.append(dict(world=w, graph=graph, cost=cost, regime=meta(w)[0],
                         J_cell_insample=sd.J(*S[w], d_in, 9000 + w)[0], J_cell_loo=sd.J(*S[w], d_loo, 9000 + w)[0],
                         J_partner_plan=sd.J(*S[w], d_pp, 9000 + w)[0],
                         loo_design=" ".join(f"{a}:{d_loo.count(a)}" for a in ACTIONS if d_loo.count(a)),
                         J_planonce=plan.loc[w, "J_planonce"], J_TR=plan.loc[w, "J_target_risk"], real_TR=plan.loc[w, "real_target_risk"],
                         J_cov=plan.loc[w, "J_menu_coverage"], J_fixed1266=plan.loc[w, "J_obs12_xneg6_xpos6"]))
d = pd.DataFrame(rows).sort_values("world")
d.to_csv("out_public_static_search.csv", index=False)
gm = lambda s: math.exp(np.log(s).mean())
cols = ["J_planonce", "J_TR", "real_TR", "J_cell_insample", "J_cell_loo", "J_partner_plan", "J_fixed1266", "J_cov"]
print("\narithmetic / geometric means over 24 systems:")
print(pd.DataFrame({"mean": d[cols].mean(), "geomean": d[cols].apply(gm)}).round(5).to_string())
def lr(a, b):
    x = np.log(d[a]) - np.log(d[b]); se = x.std(ddof=1) / math.sqrt(len(x))
    print(f"  log({a}/{b}): mean {x.mean():+.4f} (x{math.exp(x.mean()):.3f}) se {se:.4f} t {x.mean()/se:+.2f} ; a<b in {int((x<0).sum())}/24")
print("\npaired log-ratios:")
for a in ["J_cell_insample", "J_cell_loo", "J_partner_plan", "J_fixed1266", "J_cov"]:
    lr(a, "J_planonce")
for a in ["J_cell_insample", "J_cell_loo"]:
    lr(a, "real_TR")
lr("J_cell_loo", "J_cov")
print("\nby graph (geomean ratio to plan-once):")
for c in ["J_cell_insample", "J_cell_loo", "J_partner_plan", "J_fixed1266", "J_cov"]:
    print("  ", c, d.assign(r=d[c] / d.J_planonce).groupby("graph").r.agg(gm).round(3).to_dict())
print(d[["world", "graph", "cost", "regime", "loo_design", "J_cell_loo", "J_cell_insample", "J_planonce"]].round(4).to_string(index=False))
