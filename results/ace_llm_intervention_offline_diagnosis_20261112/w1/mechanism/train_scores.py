import json, math, collections, statistics as S
import numpy as np
FIT="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training/"
rows={r["rid"]:r for r in json.load(open("train_rows.json"))}
plan=json.load(open(FIT+"plan.json"))
sched=plan["schedule"]; beta=plan["numeric_recipe"]["beta"]; lr=plan["numeric_recipe"]["learning_rate"]
print("beta",beta,"lr",lr,"steps",len(sched),"sft",plan["sft_steps"],"dpo",plan["dpo_steps"])
ref=json.load(open(FIT+"reference-scores.json"))["scores"]
# ---------- SFT: online NLL of each label (single pass => out-of-sample at time of visit)
sft=[]
for i,s in enumerate(sched[:plan["sft_steps"]]):
    c=json.load(open(FIT+f"step-{i:05d}.complete.json"))
    r=rows[s["record_id"]]
    sft.append(dict(i=i, si=r["si"], label=r["sup"], teacher=r["teacher"], graph=r["graph"], nll=c["loss"]*c["action_tokens"], per_tok=c["loss"], ntok=c["action_tokens"], gn=c["gradient_norm_before_clip"], input_tokens=c["input_tokens"]))
# empirical progress-conditional label prior (in-sample over all 160)
cnt=collections.defaultdict(collections.Counter)
for x in sft: cnt[x["si"]][x["label"]]+=1
def prior_nll(x): 
    c=cnt[x["si"]]; return -math.log(c[x["label"]]/sum(c.values()))
marg=collections.Counter(x["label"] for x in sft if x["si"]<4)
print("\nSFT online sequence NLL (nats) of the supervised command at the time of its single visit:")
for lo,hi in ((0,10),(10,40),(40,80),(80,120),(120,160)):
    sub=[x for x in sft if lo<=x["i"]<hi]
    nt=[x for x in sub if x["si"]<4]; tm=[x for x in sub if x["si"]==4]
    print(f"  steps {lo:3d}-{hi:3d}: nonterminal n={len(nt)} mean NLL={S.mean(x['nll'] for x in nt):.3f} median={S.median(x['nll'] for x in nt):.3f} | stage-prior baseline={S.mean(prior_nll(x) for x in nt):.3f} | uniform8=2.079 | terminal(STOP) n={len(tm)} mean NLL={S.mean(x['nll'] for x in tm):.4f}")
late=[x for x in sft if x["i"]>=80 and x["si"]<4]
print("  late (steps>=80) nonterminal NLL by teacher:", {t: round(S.mean(x["nll"] for x in late if x["teacher"]==t),3) for t in sorted({x["teacher"] for x in late})})
print("  late (steps>=80) nonterminal NLL by label:", {t: (round(S.mean(x["nll"] for x in late if x["label"]==t),3), sum(1 for x in late if x["label"]==t)) for t in sorted({x["label"] for x in late})})
print("  late (steps>=80) nonterminal NLL by snapshot:", {t: round(S.mean(x["nll"] for x in late if x["si"]==t),3) for t in range(4)})
print("  late stage-prior baseline by snapshot:", {t: round(S.mean(prior_nll(x) for x in late if x["si"]==t),3) for t in range(4)})
print("  marginal entropy of nonterminal labels (nats):", round(-sum(v/sum(marg.values())*math.log(v/sum(marg.values())) for v in marg.values()),3))
print("  SFT grad norm before clip: first10 mean", round(S.mean(x["gn"] for x in sft[:10]),2), "last40 mean", round(S.mean(x["gn"] for x in sft[-40:]),2))
# ---------- reference (post-SFT) scores for DPO pairs
dpo=[]
for j,s in enumerate(sched[plan["sft_steps"]:]):
    i=plan["sft_steps"]+j
    r=rows[s["record_id"]]; ch,rj=r["strict"][s["pair_index"]]
    key=json.dumps([s["record_id"],s["pair_index"]],separators=(",",":"))
    rc,rr=ref[key]
    c=json.load(open(FIT+f"step-{i:05d}.complete.json"))
    loss=c["loss"]
    delta=-math.log(math.expm1(loss))/beta   # (pc-pr)-(rc-rr) at time of visit
    dpo.append(dict(j=j,i=i,rid=s["record_id"],si=r["si"],ch=ch,rj=rj,rc=rc,rr=rr,loss=loss,delta=delta,gn=c["gradient_norm_before_clip"],graph=r["graph"]))
print("\nPost-SFT reference sequence log-prob by command (over the 238 scored pair members; same record may repeat):")
bc=collections.defaultdict(list)
for x in dpo: bc[x["ch"]].append(x["rc"]); bc[x["rj"]].append(x["rr"])
for k,v in sorted(bc.items(), key=lambda kv:-S.mean(kv[1])):
    print(f"  {k:12s} n={len(v):3d} mean={S.mean(v):7.3f} median={S.median(v):7.3f} min={min(v):7.3f} max={max(v):7.3f}  mean prob={S.mean(math.exp(a) for a in v):.4f}")
bs=collections.defaultdict(lambda: collections.defaultdict(list))
for x in dpo: bs[x["si"]][x["ch"]].append(x["rc"]); bs[x["si"]][x["rj"]].append(x["rr"])
print("  mean reference log-prob by snapshot index and command:")
for si in sorted(bs): print("   si",si,{k:(round(S.mean(v),2),len(v)) for k,v in sorted(bs[si].items())})
agree=sum(1 for x in dpo if x["rc"]>x["rr"])
print(f"  SFT reference already prefers the measured-better (chosen) command in {agree}/{len(dpo)} scheduled pairs; mean (rc-rr)={S.mean(x['rc']-x['rr'] for x in dpo):.3f}")
# ---------- DPO drift
print("\nDPO: delta_i = change in (chosen - rejected) log-prob margin relative to the SFT reference at the single visit of pair i")
for lo,hi in ((0,1),(1,20),(20,40),(40,60),(60,80),(80,100),(100,119)):
    sub=[x for x in dpo if lo<=x["j"]<hi]
    print(f"  dpo steps {lo:3d}-{hi:3d}: mean loss={S.mean(x['loss'] for x in sub):.4f} mean delta={S.mean(x['delta'] for x in sub):+.3f} min={min(x['delta'] for x in sub):+.3f} max={max(x['delta'] for x in sub):+.3f} frac delta>0={sum(x['delta']>0 for x in sub)/len(sub):.2f} gradnorm={S.mean(x['gn'] for x in sub):.2f}")
# least squares: delta_i = (a[ch]-a[rj]) * tau_i, tau=j/118  (state-independent linear command drift)
cmds=sorted({x["ch"] for x in dpo}|{x["rj"] for x in dpo})
A=np.zeros((len(dpo),len(cmds))); y=np.array([x["delta"] for x in dpo])
for n,x in enumerate(dpo):
    tau=x["j"]/118
    A[n,cmds.index(x["ch"])]+=tau; A[n,cmds.index(x["rj"])]-=tau
sol,res,rank,sv=np.linalg.lstsq(A,y,rcond=None)
pred=A@sol; ss_res=float(((y-pred)**2).sum()); ss_tot=float(((y-y.mean())**2).sum()); ss0=float((y**2).sum())
sol=sol-sol.mean()
print("  linear state-independent command-drift fit (centered; nats of log-prob shift by end of DPO):")
for c,a in sorted(zip(cmds,sol), key=lambda t:-t[1]): print(f"    {c:12s} {a:+.3f}")
print(f"  rank={rank} R2 (vs mean)={1-ss_res/ss_tot:.3f}  R2 (vs zero)={1-ss_res/ss0:.3f}  n={len(dpo)}")
# set_x_neg specific
print("  pairs with set_x_neg chosen: deltas", [round(x["delta"],2) for x in dpo if x["ch"]=="set_x_neg"])
print("  pairs with set_x_neg rejected: deltas", [round(x["delta"],2) for x in dpo if x["rj"]=="set_x_neg"])
json.dump(dict(sft=sft,dpo=dpo), open("train_scores.json","w"))
