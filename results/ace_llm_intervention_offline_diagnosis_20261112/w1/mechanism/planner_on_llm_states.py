"""Score every LLM-visited public state with the frozen target-risk planner (pure CPU function on saved
public envelopes; no environment response, no model, analysis seed is mine, 32 draws as in the protocol)."""
import sys, json, pickle, collections, time, statistics as S
sys.dont_write_bytecode=True
sys.path.insert(0,"/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/mechanism/frozen_src")
import numpy as np
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_contract import Action
rows=pickle.load(open("decisions.pkl","rb"))
def task_args(env):
    t=env["task"]
    q=tuple(Action(a["action_id"],a["kind"],tuple(tuple(x) for x in a["targets"]),tuple(a["measure"]),a["cost_units"]) for a in t["queries"])
    return dict(target=t["target"], queries=q, weights=tuple(t["weights"]), scale=t["scale"])
arm=sys.argv[1]
rs=sorted([r for r in rows if r["arm"]==arm], key=lambda r:r["ordinal"])
out=[]; t0=time.time()
for r in rs:
    dec,meta=choose_target_risk(r["env"]["state"], seed=20261112, draws=32, **task_args(r["env"]))
    out.append(dict(ordinal=r["ordinal"], llm=json.loads(r["completion"]).get("action_id","STOP"), best=dec.get("action_id","STOP"),
                    scores=meta.get("risk_reduction_per_cost"), se=meta.get("monte_carlo_standard_errors"), before=meta.get("risk_before"),
                    requests=r["requests"], remaining=r["remaining"], graph=r["graph"], cost=r["cost"]))
print(arm,"states",len(out),"seconds",round(time.time()-t0,1))
pickle.dump(out, open(f"planner_{arm}.pkl","wb"))
