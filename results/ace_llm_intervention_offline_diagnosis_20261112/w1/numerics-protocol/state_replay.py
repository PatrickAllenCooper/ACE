"""Replay saved CPU-arm histories; cross-check against saved curves and saved target-risk receipts."""
import json, pickle
import numpy as np, pandas as pd
from acelib import *

D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
curves = pd.read_csv(D + "curves.csv")
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/numerics-protocol/"
rows = []
max_nmse_dev, max_risk_dev, max_state_dev, n_checked_risk = 0.0, 0.0, 0.0, 0
for pi, pol in enumerate(POLICIES):
    for wi in range(24):
        arm = load_arm(pi, wi)
        world, learner = arm["world"], arm["learner"]
        wid = world["world_id"]
        regime, graph, cost = meta(wi)
        parents = graph_parents(learner)
        noise, pv = learner["noise_variances"], learner["prior_variance"]
        hist = arm["terminal"]["final_state"]["history"]
        cur = curves[(curves.world_id == wid) & (curves.policy_id == pol)].sort_values("accepted_receipts")
        assert len(cur) == len(hist) - 8 + 1, (pol, wi, len(cur), len(hist))
        acq = 0
        for j, n in enumerate(range(8, len(hist) + 1)):
            heads = posterior(parents, noise, pv, hist[:n])
            R, perq = risk(heads, parents, per_query=True)
            e = nmse(heads, parents, world)
            if n > 8:
                acq += hist[n - 1]["cost_units"]
            crow = cur.iloc[j]
            assert int(crow.accepted_receipts) == n and int(crow.acquisition_cost_units) == acq
            max_nmse_dev = max(max_nmse_dev, abs(e - crow.normalized_mse))
            # cross-check with the saved public state / saved risk_before where available
            k = n - 8
            if k < len(arm["decisions"]):
                inp, proc = arm["decisions"][k]
                st = inp["state"]
                for s in st["scm"]:
                    m, V, _ = heads[s["node"]]
                    max_state_dev = max(max_state_dev, float(np.max(np.abs(m - np.array(s["parameter_mean"])))),
                                        float(np.max(np.abs(V - np.array(s["parameter_covariance"])))))
                if pol == "target_risk":
                    rec = json.loads(proc["stderr"].strip().splitlines()[-1])
                    rb = rec["planning_metadata"]["risk_before"]
                    max_risk_dev = max(max_risk_dev, abs(rb - R)); n_checked_risk += 1
            rows.append(dict(policy=pol, world=wi, world_id=wid, regime=regime, graph=graph, cost=cost,
                             n=n, acq_cost=acq, action=hist[n - 1]["action_id"] if n > 8 else None,
                             post_risk=R, nmse=e, **{f"rq{q}": perq[q] for q in range(8)}))
df = pd.DataFrame(rows)
df.to_pickle(OUT + "replay_states.pkl")
df.to_csv(OUT + "out_replay_states.csv", index=False)
print("rows", len(df))
print("max |replayed NMSE - saved curves.csv NMSE| =", max_nmse_dev)
print("max |replayed posterior - saved public state| =", max_state_dev)
print("max |my integrated risk - saved target-risk risk_before| =", max_risk_dev, "over", n_checked_risk, "decisions")
