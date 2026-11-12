import json, math, collections, statistics as S
import numpy as np
d=json.load(open("train_scores.json")); dpo=d["dpo"]; sft=d["sft"]
print("observe_all chosen:", [(x["j"],x["rj"],round(x["delta"],2)) for x in dpo if x["ch"]=="observe_all"])
print("observe_all rejected:", [(x["j"],x["ch"],round(x["delta"],2)) for x in dpo if x["rj"]=="observe_all"])
print("set_x_neg chosen:", [(x["j"],x["rj"],round(x["delta"],2)) for x in dpo if x["ch"]=="set_x_neg"])
print("set_x_neg rejected:", [(x["j"],x["ch"],round(x["delta"],2)) for x in dpo if x["rj"]=="set_x_neg"])
# ANOVA of reference log-probs
vals=[]; 
for x in dpo: vals.append((x["ch"],x["si"],x["rc"],x["graph"])); vals.append((x["rj"],x["si"],x["rr"],x["graph"]))
# dedupe by (rid, command) to avoid double counting the same record/command
seen={}; 
for x in dpo:
    seen[(x["rid"],x["ch"])]=(x["ch"],x["si"],x["rc"],x["graph"]); seen[(x["rid"],x["rj"])]=(x["rj"],x["si"],x["rr"],x["graph"])
vals=list(seen.values())
y=np.array([v[2] for v in vals]); tot=((y-y.mean())**2).sum()
def r2(keyf):
    g=collections.defaultdict(list)
    for v in vals: g[keyf(v)].append(v[2])
    within=sum(((np.array(a)-np.mean(a))**2).sum() for a in g.values())
    return 1-within/tot, len(g)
print(f"unique (record,command) reference scores: {len(vals)} from {len({k[0] for k in seen})} records; total SD={y.std():.3f}")
print("variance of post-SFT sequence log-prob explained by command identity alone: R2=%.3f (groups=%d)"%r2(lambda v:v[0]))
print("  by command x snapshot index: R2=%.3f (groups=%d)"%r2(lambda v:(v[0],v[1])))
print("  by command x snapshot x graph: R2=%.3f (groups=%d)"%r2(lambda v:(v[0],v[1],v[3])))
g=collections.defaultdict(list)
for v in vals: g[v[0]].append(v[2])
print("  within-command SD:", {k: round(float(np.std(a)),3) for k,a in sorted(g.items())}, " between-command SD of means:", round(float(np.std([np.mean(a) for a in g.values()])),3))
# does delta track the reference's own margin? (prior sharpening)
m=np.array([x["rc"]-x["rr"] for x in dpo]); dl=np.array([x["delta"] for x in dpo]); tau=np.array([x["j"]/118 for x in dpo])
print("corr(delta, reference margin rc-rr) all:", round(float(np.corrcoef(dl,m)[0,1]),3), " late half:", round(float(np.corrcoef(dl[60:],m[60:])[0,1]),3))
print("corr(delta, (rc-rr)*tau):", round(float(np.corrcoef(dl,m*tau)[0,1]),3))
k=float((dl*(m*tau)).sum()/((m*tau)**2).sum()); pred=k*m*tau
print("  one-parameter fit delta = k*(rc-rr)*tau: k=%.3f R2(vs zero)=%.3f"%(k, 1-((dl-pred)**2).sum()/(dl**2).sum()))
# final-policy implied sign: for each pair, reference margin vs (reference margin + delta) at visit time
flip_to_agree=sum(1 for x in dpo if (x["rc"]-x["rr"])<=0 and (x["rc"]-x["rr"]+x["delta"])>0)
flip_to_disagree=sum(1 for x in dpo if (x["rc"]-x["rr"])>0 and (x["rc"]-x["rr"]+x["delta"])<=0)
agree_ref=sum(1 for x in dpo if (x["rc"]-x["rr"])>0); agree_pol=sum(1 for x in dpo if (x["rc"]-x["rr"]+x["delta"])>0)
print(f"sequence-level agreement with measured preference at visit time: reference {agree_ref}/119, current DPO policy before its update {agree_pol}/119 (flips to agree {flip_to_agree}, to disagree {flip_to_disagree})")
print("  second half only: reference", sum(1 for x in dpo[60:] if (x['rc']-x['rr'])>0), "/", len(dpo[60:]), " policy", sum(1 for x in dpo[60:] if (x['rc']-x['rr']+x['delta'])>0),"/",len(dpo[60:]))
# DPO schedule order
print("DPO schedule order (first 15): ", [(x["rid"].split("/")[1][-3:], x["si"]) for x in dpo[:15]])
print("DPO loss: first", round(dpo[0]["loss"],4), "mean all", round(S.mean(x["loss"] for x in dpo),4), "last20 mean", round(S.mean(x["loss"] for x in dpo[-20:]),4), "min", round(min(x["loss"] for x in dpo),4), "max", round(max(x["loss"] for x in dpo),4))
print("SFT per-token loss first 5:", [round(x["per_tok"],3) for x in sft[:5]], "last 10:", [round(x["per_tok"],3) for x in sft[-10:]])
