import json, glob, collections, hashlib, pickle, statistics as S
import numpy as np
BASE="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/campaign"
camp=json.load(open(BASE+"/campaign.json"))
print("campaign keys:", list(camp.keys())[:12])
C=lambda v: json.dumps(v, sort_keys=True, separators=(",",":"), allow_nan=False)
names={}
out={}
for p in range(5):
    recs=[]
    for f in sorted(glob.glob(f"{BASE}/policy-{p:02d}/world-*/episode/decision-*.process.json")):
        d=json.load(open(f)); inp=json.load(open(f.replace(".process.json",".input.json")))
        m=json.loads(d["stderr"].strip().splitlines()[-1])
        w=int(f.split("/world-")[1][:2]); k=int(f.split("decision-")[1][:4])
        arm=m.get("arm") or m.get("method")
        recs.append(dict(world=w,k=k,decision=json.loads(d["stdout"]),meta=m,env_sha=hashlib.sha256(C(inp).encode()).hexdigest(),
                         remaining=inp["state"]["remaining_units"], requests=inp["state"]["requests"], nfeasible=sum(a["cost_units"]<=inp["state"]["remaining_units"] for a in inp["state"]["menu"])))
        names[p]=arm
    out[p]=recs
    acts=collections.Counter(r["decision"].get("action_id","STOP") for r in recs)
    print(f"policy-{p:02d} arm={names[p]} decisions={len(recs)} actions={dict(acts)}")
pickle.dump(out, open("numeric_decisions.pkl","wb"))
# shared first state?
llm=pickle.load(open("decisions.pkl","rb"))
first_llm=[r for r in llm if r["arm"]=="dpo_llm" and r["ordinal"]==0][0]["envelope_sha256"]
print("first envelope world-00: numeric arms:", {names[p]: out[p][0]["env_sha"][:12] for p in out if out[p][0]["world"]==0 and out[p][0]["k"]==0}, "LLM:", first_llm[:12])
# ranker score sensitivity
rk=out[4]
ids=rk[0]["meta"]["action_ids"]
M=[]; 
for r in rk:
    if r["meta"]["action_ids"]==ids and len(r["meta"]["scores"])==9: M.append(r["meta"]["scores"])
M=np.array(M); print("ranker full-menu decisions:", M.shape, "action order:", ids)
print(" mean score by action:", dict(zip([str(i) for i in ids], np.round(M.mean(0),3))))
print(" SD across states by action:", dict(zip([str(i) for i in ids], np.round(M.std(0),3))))
A=M[:,:8]; lp=A-np.log(np.exp(A).sum(1,keepdims=True))  # log-softmax over 8 acquisition actions
tot=((lp-lp.mean())**2).sum(); within=((lp-lp.mean(0,keepdims=True))**2).sum()
print(" ranker log-softmax over 8 actions: variance explained by action identity alone R2=%.3f ; between-action SD of means=%.3f ; mean within-action SD=%.3f"%(1-within/tot, lp.mean(0).std(), lp.std(0).mean()))
top=collections.Counter(ids[int(np.argmax(m))] if int(np.argmax(m))<8 else "STOP" for m in M)
print(" argmax distribution:", dict(top))
gap=np.sort(A,axis=1)[:,-1]-np.sort(A,axis=1)[:,-2]
print(" top1-top2 score gap: median %.4f min %.5f max %.4f; softmax prob of chosen action median %.3f"%(np.median(gap), gap.min(), gap.max(), np.median(np.exp(lp).max(1))))
# by decision index
byk=collections.defaultdict(collections.Counter)
for r in rk: byk[r["k"]][r["decision"].get("action_id","STOP")]+=1
print(" ranker action by decision index:", {k:dict(v) for k,v in sorted(byk.items()) if k in (0,1,2,3,4,6,8,12,16,20,23)})
# target risk: best-action distribution and how often set_x_neg is best / its regret
tr=out[3]
best=collections.Counter(); reg=[]; rank_neg=[]
for r in tr:
    s=r["meta"]["planning_metadata"].get("risk_reduction_per_cost")
    if not s: continue
    b=max(s, key=lambda a:(s[a], )); best[b]+=1
    if "set_x_neg" in s:
        reg.append(max(s.values())-s["set_x_neg"]); rank_neg.append(sorted(s.values(), reverse=True).index(s["set_x_neg"])+1)
print("target_risk decisions with scores:", sum(best.values()), "best action:", dict(best))
print(" on target_risk's own states: set_x_neg rank distribution:", dict(collections.Counter(rank_neg)), " mean regret (risk-reduction per cost) of always choosing set_x_neg: %.5f; mean best score %.5f"%(S.mean(reg), S.mean(max(r["meta"]["planning_metadata"]["risk_reduction_per_cost"].values()) for r in tr if r["meta"]["planning_metadata"].get("risk_reduction_per_cost"))))
