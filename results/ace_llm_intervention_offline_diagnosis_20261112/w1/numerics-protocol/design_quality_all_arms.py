"""Outcome-noise-free 'design quality' of every arm's REALIZED action multiset (from saved curves.csv),
evaluated as model-expected final posterior NMSE from the saved common post-initialization state.
Also first-decision quality at the common state using SAVED target-risk receipts."""
import json, math
import numpy as np, pandas as pd
from acelib import *
import static_designs as sd
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
curves = pd.read_csv(D + "curves.csv")
POL8 = ["uniform_random", "menu_coverage", "local_information", "target_risk", "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
plan = pd.read_pickle("static_designs.pkl").set_index("world")
rows = []
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    h8 = posterior(parents, noise, 1.0, hist[:8])
    tr0 = json.loads(load_arm(3, wi)["decisions"][0][1]["stderr"].strip().splitlines()[-1])["planning_metadata"]["risk_reduction_per_cost"]
    best0 = max(tr0.values())
    regime, graph, cost = meta(wi)
    for pol in POL8:
        c = curves[(curves.world_id == 202611070100 + wi) & (curves.policy_id == pol)].sort_values("accepted_receipts")
        seq = [a for a in c.action_id.tolist() if isinstance(a, str)]
        spent = sum(action_cost(a, cost) for a in seq)
        assert spent == int(c.acquisition_cost_units.max())
        j, s, _ = sd.J(h8, parents, noise, seq, 9000 + wi)
        rows.append(dict(world=wi, regime=regime, graph=graph, cost=cost, policy=pol, n_actions=len(seq), spent=spent, J=j, fantasy_sd=s,
                         realized_final_nmse=float(c.normalized_mse.iloc[-1]), first_action=seq[0] if seq else None,
                         first_frac=(tr0[seq[0]] / best0) if seq else 0.0, J_planonce=plan.loc[wi, "J_planonce"], R8=plan.loc[wi, "R8"]))
d = pd.DataFrame(rows)
d.to_csv("out_design_quality_all_arms.csv", index=False)
gm = lambda s: math.exp(np.log(s).mean())
g = d.groupby("policy").agg(J_mean=("J", "mean"), J_geomean=("J", gm), realized_final_nmse_mean=("realized_final_nmse", "mean"),
        realized_final_nmse_geomean=("realized_final_nmse", gm), n_J_le_004=("J", lambda s: int((s <= 0.04).sum())),
        mean_spent=("spent", "mean"), first_frac_mean=("first_frac", "mean")).loc[POL8]
d["ratio_plan"] = d.J / d.J_planonce
g["J_over_planonce_geomean"] = d.groupby("policy").ratio_plan.agg(gm).loc[POL8]
print("== Design quality of each arm's realized action multiset (model-expected final NMSE from the common saved post-init state) ==")
print(g.round(4).to_string())
print("\nfirst action chosen at the common post-initialization state, by arm:")
print(d.groupby(["policy", "first_action"]).size().unstack(fill_value=0).loc[POL8].to_string())
# paired contrasts on log J versus target_risk
print("\npaired log(J_arm / J_target_risk), n=24:")
tr = d[d.policy == "target_risk"].set_index("world").J
for pol in POL8:
    if pol == "target_risk": continue
    x = np.log(d[d.policy == pol].set_index("world").J) - np.log(tr)
    se = x.std(ddof=1) / math.sqrt(24)
    print(f"  {pol:18s} mean {x.mean():+.3f} (x{math.exp(x.mean()):.2f}) se {se:.3f} t {x.mean()/se:+.1f}  arm better in {int((x<0).sum())}/24")
print("\ncorrelation across the 8 arms between mean J and mean realized final NMSE: r = %.3f ; with mean capped cost:" % np.corrcoef(g.J_mean, g.realized_final_nmse_mean)[0, 1])
capped = {"uniform_random": 14.25, "menu_coverage": 12.458, "local_information": 11.958, "target_risk": 11.625, "learned_ranker": 13.583, "frozen_llm": 13.625, "sft_llm": 14.625, "dpo_llm": 15.167}
print("  r(J_mean, capped cost) = %.3f ; r(J_geomean, capped cost) = %.3f" % (np.corrcoef(g.J_mean, [capped[p] for p in POL8])[0, 1], np.corrcoef(g.J_geomean, [capped[p] for p in POL8])[0, 1]))
print("\nby graph, geometric mean J:")
print(d.pivot_table(index="policy", columns="graph", values="J", aggfunc=gm).loc[POL8].round(4).to_string())
