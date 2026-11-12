"""(7) Teacher behaviour on the saved training states. Offline; reads my own scratch pickle (trusted)."""
import pickle, sys, math
from collections import Counter, defaultdict
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
D = pickle.load(open(sys.argv[1], "rb"))
R = [r for r in D["records"] if not r["terminal"]]
ACTIONS = ["observe_all","set_x_neg","set_x_zero","set_x_pos","set_y_neg","set_y_zero","set_y_pos","set_xy"]

def ent(counts):
    n = sum(counts.values()); return -sum(c/n*math.log(c/n) for c in counts.values() if c)

print("== concentration of each teacher's choices (nonterminal states)")
for t in ("uniform_random","menu_coverage","local_information","target_risk"):
    sub = [r for r in R if r["teacher_method"]==t]
    c = Counter(r["supervised_action_id"] for r in sub)
    print(f"{t:18s} n={len(sub)} distinct actions={len(c)} modal={c.most_common(1)[0]} modal share={c.most_common(1)[0][1]/len(sub):.3f} entropy={ent(c):.3f} nats (max ln8=2.079) dist={dict(c.most_common())}")
print("\n== by teacher x graph x cost menu: modal action, modal share, #distinct")
rows = []
for t in ("uniform_random","menu_coverage","local_information","target_risk"):
    for g in ("chain","fork","collider"):
        for cm in ("uniform","heterogeneous"):
            sub = [r for r in R if r["teacher_method"]==t and r["graph"]==g and r["cost_menu"]==cm]
            c = Counter(r["supervised_action_id"] for r in sub)
            rows.append({"teacher": t, "graph": g, "cost_menu": cm, "n": len(sub), "modal": c.most_common(1)[0][0], "modal_n": c.most_common(1)[0][1],
                         "distinct": len(c), "dist": dict(c.most_common())})
T = pd.DataFrame(rows); print(T.to_string())
for t in ("uniform_random","menu_coverage","local_information","target_risk"):
    tt = T[T.teacher==t]
    print(f"  {t}: in-sample accuracy of the graph x cost modal action = {tt.modal_n.sum()}/{tt.n.sum()} = {tt.modal_n.sum()/tt.n.sum():.3f}")
# leave-one-world-out for informed teachers
for t in ("local_information","target_risk","menu_coverage","uniform_random"):
    sub = [r for r in R if r["teacher_method"]==t]
    for keys in ((), ("graph",), ("graph","cost_menu"), ("snapshot",)):
        ok = 0
        for r in sub:
            tr = [x for x in sub if x["world_index"]!=r["world_index"] and all(x[k]==r[k] for k in keys)]
            if not tr: tr = [x for x in sub if x["world_index"]!=r["world_index"]]
            c = Counter(x["supervised_action_id"] for x in tr)
            best = sorted(c, key=lambda a: (-c[a], a))[0]
            ok += best == r["supervised_action_id"]
        print(f"  LOWO constant-rule accuracy for {t:18s} cells={'+'.join(keys) or '(none)':16s}: {ok}/{len(sub)} = {ok/len(sub):.3f}")

print("\n== menu_coverage: is the label a deterministic function of history counts? (recompute least-used, lexical tie-break)")
ok = 0
for r in R:
    if r["teacher_method"]!="menu_coverage": continue
    feas = sorted(a for a,c in r["menu_costs"].items() if c <= r["remaining_units"])
    cnt = {a: r["history_actions"].count(a) for a in feas}
    pick = min(feas, key=lambda a: (cnt[a], a))
    ok += pick == r["supervised_action_id"]
print("  reproduced:", ok, "/ 32")

print("\n== target_risk teacher: decisiveness of its own choice (saved risk_reduction_per_cost and Monte Carlo SE)")
rows = []
for r in R:
    if r["teacher_method"]!="target_risk": continue
    md = r["teacher_metadata"]; sc = md["risk_reduction_per_cost"]; se = md["monte_carlo_standard_errors"]
    order = sorted(sc, key=lambda a: (-sc[a], a))
    b, s = order[0], order[1]
    z = (sc[b]-sc[s]) / math.sqrt(se[b]**2 + se[s]**2)
    rows.append({"record": r["record_id"][-27:], "graph": r["graph"], "cost": r["cost_menu"], "snap": r["snapshot"], "best": b, "second": s,
                 "best_score": sc[b], "second/best": sc[s]/sc[b] if sc[b] else float('nan'), "z_gap": z, "risk_before": md["risk_before"],
                 "best_rel_reduction": sc[b]*r["menu_costs"][b]/md["risk_before"],
                 "n_within_2se_of_best": sum(1 for a in sc if a!=b and (sc[b]-sc[a]) < 2*math.sqrt(se[b]**2+se[a]**2)),
                 "worst/best": min(sc.values())/sc[b] if sc[b] else float('nan'),
                 "xneg_score": sc["set_x_neg"], "xpos_score": sc["set_x_pos"]})
TR = pd.DataFrame(rows)
print(TR.round(4).to_string())
print("\n  target_risk: top-vs-second z gap quantiles:", TR.z_gap.quantile([0,.25,.5,.75,1]).round(2).to_dict())
print("  states where the runner-up is within 2 MC SE of the chosen action:", int((TR.z_gap<2).sum()), "/", len(TR))
print("  second/best score ratio quantiles:", TR["second/best"].quantile([0,.25,.5,.75,1]).round(3).to_dict())
print("  states where second-best achieves >=90% of the best score:", int((TR["second/best"]>=0.9).sum()), "/", len(TR), "; >=80%:", int((TR["second/best"]>=0.8).sum()))
print("  best action's one-step relative risk reduction (fraction of current risk) quantiles:", TR.best_rel_reduction.quantile([0,.25,.5,.75,1]).round(4).to_dict())
print("  (best, second) pairs:", dict(Counter(zip(TR.best, TR.second)).most_common()))
sym = TR[(TR.best.isin(["set_x_neg","set_x_pos"]))]
print("  when a single-X intervention is chosen: |score(x_neg)-score(x_pos)|/max quantiles:",
      (abs(sym.xneg_score-sym.xpos_score)/sym[["xneg_score","xpos_score"]].max(axis=1)).quantile([0,.25,.5,.75,1]).round(3).to_dict())
print("  target_risk rank-1 action by teacher score, counts:", dict(Counter(TR.best)))
# rank correlation between teacher scores and measured mean gains at the same state
from scipy import stats
rhos = []
for r in R:
    if r["teacher_method"]!="target_risk": continue
    sc = r["teacher_metadata"]["risk_reduction_per_cost"]; mg = r["feedback"]["mean_gains_per_cost"]
    rhos.append(stats.spearmanr([sc[a] for a in ACTIONS],[mg[a] for a in ACTIONS]).correlation)
print("  Spearman corr between target_risk's predicted risk reduction/cost and the measured 4-replicate mean gain/cost across the 8 actions, per state: mean",
      round(float(np.mean(rhos)),3), " median", round(float(np.median(rhos)),3), " share>0", round(float(np.mean(np.array(rhos)>0)),3), " n=", len(rhos))
# magnitude comparison: teacher's predicted expected gain vs measured mean gain and its SE
ratio = []
for r in R:
    if r["teacher_method"]!="target_risk": continue
    sc = r["teacher_metadata"]["risk_reduction_per_cost"]; fb = r["feedback"]
    for a in ACTIONS:
        g = np.array(fb["per_replicate_gains_per_cost"][a]); ratio.append((sc[a], g.mean(), g.std(ddof=1)/2))
ratio = np.array(ratio)
print("  target_risk predicted expected gain/cost: median", float(f"{np.median(ratio[:,0]):.3e}"), "| measured 4-rep mean gain: median", float(f"{np.median(ratio[:,1]):.3e}"),
      "| SE of the measured mean: median", float(f"{np.median(ratio[:,2]):.3e}"), "| median (SE of measured mean / predicted expected gain):", round(float(np.median(ratio[:,2]/np.abs(ratio[:,0]))),2))
print("  predicted spread between best and worst action (per state) vs SE of measured mean: median ratio (best-worst predicted)/(median SE):",
      round(float(np.median([(max(r['teacher_metadata']['risk_reduction_per_cost'].values())-min(r['teacher_metadata']['risk_reduction_per_cost'].values())) /
                              np.median([np.std(r['feedback']['per_replicate_gains_per_cost'][a], ddof=1)/2 for a in ACTIONS]) for r in R if r['teacher_method']=='target_risk'])),3))

print("\n== local_information teacher: decisiveness (saved scores; no SE saved)")
rows = []
for r in R:
    if r["teacher_method"]!="local_information": continue
    sc = r["teacher_metadata"]["scores"]; order = sorted(sc, key=lambda a: (-sc[a], a)); b, s = order[0], order[1]
    rows.append({"graph": r["graph"], "cost": r["cost_menu"], "snap": r["snapshot"], "best": b, "second": s, "second/best": sc[s]/sc[b],
                 "x_neg/x_pos": sc["set_x_neg"]/sc["set_x_pos"]})
LI = pd.DataFrame(rows)
print("  second/best ratio quantiles:", LI["second/best"].quantile([0,.25,.5,.75,1]).round(3).to_dict(), " >=0.9:", int((LI["second/best"]>=0.9).sum()), "/", len(LI))
print("  (best, second) pairs:", dict(Counter(zip(LI.best, LI.second)).most_common()))
rhos = []
for r in R:
    if r["teacher_method"]!="local_information": continue
    sc = r["teacher_metadata"]["scores"]; mg = r["feedback"]["mean_gains_per_cost"]
    rhos.append(stats.spearmanr([sc[a] for a in ACTIONS],[mg[a] for a in ACTIONS]).correlation)
print("  Spearman corr between local_information scores and measured mean gains per state: mean", round(float(np.mean(rhos)),3), " median", round(float(np.median(rhos)),3), " share>0", round(float(np.mean(np.array(rhos)>0)),3))
