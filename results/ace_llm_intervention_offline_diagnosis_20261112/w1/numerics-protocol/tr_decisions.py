"""Analyze SAVED target-risk and local-information decision receipts (no recomputation of planning)."""
import json, math
import numpy as np, pandas as pd
from acelib import *
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
SHORT = dict(zip(ACTIONS, ["O", "x-", "x0", "x+", "y-", "y0", "y+", "xy"]))
rows, seqs = [], {}
for pi, pol in [(3, "target_risk"), (2, "local_information")]:
    for wi in range(24):
        arm = load_arm(pi, wi)
        regime, graph, cost = meta(wi)
        seq = []
        for k, (inp, proc) in enumerate(arm["decisions"]):
            rec = json.loads(proc["stderr"].strip().splitlines()[-1])
            pm = rec["planning_metadata"]
            chosen = rec["decision"].get("action_id")
            seq.append(chosen)
            if pol == "target_risk":
                sc, se = pm["risk_reduction_per_cost"], pm["monte_carlo_standard_errors"]
                order = sorted(sc, key=lambda a: (-sc[a], a))
                best, second = order[0], order[1]
                gap = sc[best] - sc[second]
                se_gap = math.sqrt(se[best] ** 2 + se[second] ** 2)
                rows.append(dict(policy=pol, world=wi, regime=regime, graph=graph, cost=cost, k=k, chosen=chosen,
                    remaining=inp["state"]["remaining_units"], risk_before=pm["risk_before"],
                    best_score=sc[best], second=second, second_score=sc[second], gap=gap, se_gap=se_gap, z=gap / se_gap if se_gap > 0 else float("inf"),
                    n_feasible=len(sc), n_negative=sum(v < 0 for v in sc.values()), best_negative=sc[best] < 0,
                    rel_gap=gap / abs(sc[best]) if sc[best] != 0 else float("nan"),
                    n_within_1se=sum((sc[best] - sc[a]) <= math.sqrt(se[best] ** 2 + se[a] ** 2) for a in sc) - 1,
                    n_within_2se=sum((sc[best] - sc[a]) <= 2 * math.sqrt(se[best] ** 2 + se[a] ** 2) for a in sc) - 1,
                    elapsed=proc["elapsed_seconds"]))
            else:
                sc = pm["scores"]
                order = sorted(sc, key=lambda a: (-sc[a], a))
                rows.append(dict(policy=pol, world=wi, regime=regime, graph=graph, cost=cost, k=k, chosen=chosen,
                                 remaining=inp["state"]["remaining_units"], best_score=sc[order[0]], second=order[1], second_score=sc[order[1]],
                                 gap=sc[order[0]] - sc[order[1]], elapsed=proc["elapsed_seconds"]))
        seqs[(pol, wi)] = seq
d = pd.DataFrame(rows)
d.to_csv(OUT + "out_decision_receipts.csv", index=False)
tr = d[d.policy == "target_risk"]
print("target-risk decisions:", len(tr), " mean elapsed s:", round(tr.elapsed.mean(), 3), " max:", round(tr.elapsed.max(), 3))
print("local-info decisions:", (d.policy == "local_information").sum(), " mean elapsed s:", round(d[d.policy == "local_information"].elapsed.mean(), 3))
print("\n== TR action sequences (after 8 initial observations) ==")
for wi in range(24):
    regime, graph, cost = meta(wi)
    print(f"w{wi:02d} {graph:8s} {cost:13s} {regime:12s} TR: " + " ".join(SHORT[a] for a in seqs[("target_risk", wi)]))
print("\n== LI action sequences ==")
for wi in range(24):
    regime, graph, cost = meta(wi)
    print(f"w{wi:02d} {graph:8s} {cost:13s} {regime:12s} LI: " + " ".join(SHORT[a] for a in seqs[("local_information", wi)]))
cnt = []
for (pol, wi), seq in seqs.items():
    regime, graph, cost = meta(wi)
    c = {SHORT[a]: seq.count(a) for a in ACTIONS}
    cnt.append(dict(policy=pol, world=wi, graph=graph, cost=cost, regime=regime, **c))
cnt = pd.DataFrame(cnt)
cnt.to_csv(OUT + "out_action_counts.csv", index=False)
print("\n== TR action counts per system ==")
print(cnt[cnt.policy == "target_risk"].sort_values(["graph", "cost", "world"]).drop(columns="policy").to_string(index=False))
print("\n== TR mean action counts by graph x cost ==")
print(cnt[cnt.policy == "target_risk"].groupby(["graph", "cost"])[list(SHORT.values())].mean().round(2).to_string())
print("\n== LI mean action counts by graph x cost ==")
print(cnt[cnt.policy == "local_information"].groupby(["graph", "cost"])[list(SHORT.values())].mean().round(2).to_string())
print("\n== TR first decision by system ==")
print(tr[tr.k == 0][["world", "graph", "cost", "regime", "chosen", "risk_before", "best_score", "second", "second_score", "z"]].round(4).to_string(index=False))

print("\n== TR decision-margin statistics (best vs second-best one-step score, in MC SE units of the gap) ==")
print("fraction with z<1: %.3f ; z<2: %.3f ; z<3: %.3f ; median z %.2f" % ((tr.z < 1).mean(), (tr.z < 2).mean(), (tr.z < 3).mean(), tr.z.median()))
print("mean #other actions within 1 SE of best: %.2f ; within 2 SE: %.2f" % (tr.n_within_1se.mean(), tr.n_within_2se.mean()))
print("fraction of decisions where best estimated gain is negative: %.3f ; mean # negative-gain actions: %.2f of mean %.2f feasible" % (tr.best_negative.mean(), tr.n_negative.mean(), tr.n_feasible.mean()))
print("median relative gap (best-second)/|best|: %.3f" % tr.rel_gap.median())
print(tr.groupby("graph")[["z"]].agg(lambda s: (s < 2).mean()).rename(columns={"z": "frac z<2"}).round(3).to_string())
tr2 = tr.assign(phase=pd.cut(tr.k, [-1, 5, 11, 17, 40], labels=["k0-5", "k6-11", "k12-17", "k18+"]))
print(tr2.groupby("phase").agg(frac_z_lt2=("z", lambda s: (s < 2).mean()), med_best=("best_score", "median"), med_gap=("gap", "median"), frac_best_neg=("best_negative", "mean"), n=("z", "size")).round(5).to_string())

# Within-public-design agreement: same graph, cost, noise => matched vs shifted partner (wi, wi+12)
print("\n== TR allocation distance between partner systems with identical public design (L1 distance of count vectors / 2) ==")
names = list(SHORT.values())
c = cnt[cnt.policy == "target_risk"].set_index("world")
dist = []
for wi in range(12):
    a, b = c.loc[wi, names].values.astype(float), c.loc[wi + 12, names].values.astype(float)
    dist.append(dict(world=wi, partner=wi + 12, graph=c.loc[wi, "graph"], cost=c.loc[wi, "cost"], l1_half=np.abs(a - b).sum() / 2, n_a=a.sum(), n_b=b.sum(),
                     seq_agree=np.mean([x == y for x, y in zip(seqs[("target_risk", wi)], seqs[("target_risk", wi + 12)])])))
dist = pd.DataFrame(dist)
print(dist.round(3).to_string(index=False))
print("mean moved actions between partners: %.2f of ~%.1f ; mean stepwise sequence agreement: %.3f" % (dist.l1_half.mean(), dist.n_a.mean(), dist.seq_agree.mean()))
