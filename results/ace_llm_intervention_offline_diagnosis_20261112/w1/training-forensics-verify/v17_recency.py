"""Recency check: exponentially weighted label shares at the end of the single unshuffled SFT pass,
and a recency-weighted state-blind counter as a prequential predictor."""
import json, os, math
import numpy as np
from collections import Counter
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
byid={r['rid']:r for r in R}
def lab(r): return 'stop' if r['sup'] is None else r['sup']
steps=[json.load(open(FIT+'/training/step-%05d.complete.json'%i)) for i in range(160)]
labels=[lab(byid[s['record_id']]) for s in steps]; nats=[s['loss']*s['action_tokens'] for s in steps]
ntl=[(i,l) for i,l in enumerate(labels) if l!='stop']
print('nonterminal labels in the last 30 SFT steps:',dict(Counter(l for i,l in ntl if i>=130).most_common()))
print('nonterminal labels in the last 20 SFT steps:',dict(Counter(l for i,l in ntl if i>=140).most_common()))
print('uniform (unweighted) shares: x_neg %.3f x_pos %.3f observe %.3f'%(32/128,29/128,24/128))
for hl in [5,10,20,40,80,1e9]:
    lam=0.5**(1/hl); w=Counter()
    for k,(i,l) in enumerate(ntl):
        w[l]+=lam**(len(ntl)-1-k)
    tot=sum(w.values())
    print('half-life %5s nonterminal examples: end-of-pass weighted shares x_neg %.3f x_pos %.3f observe %.3f xy %.3f y_neg %.3f ; neg:pos ratio %.2f'%(('inf' if hl>1e6 else int(hl)),w['set_x_neg']/tot,w['set_x_pos']/tot,w['observe_all']/tot,w['set_xy']/tot,w['set_y_neg']/tot,w['set_x_neg']/max(w['set_x_pos'],1e-9)))
print('SFT reference geometric-mean probabilities for comparison (snapshots 1-3): observe 0.367, x_neg 0.345, y_neg 0.106, x_pos 0.061 ; (snapshot 0): x_neg 0.658, observe 0.155, y_neg 0.103, x_pos 0.016')
# prequential recency-weighted counter (add-one over 8 actions), on nonterminal examples in steps 80-159
def preq(hl,alpha=1.0):
    lam=0.5**(1/hl); w={a:0.0 for a in ACTS}; out=[]
    for i,l in ntl:
        tot=sum(w.values())+alpha*8; p=(w[l]+alpha)/tot
        if i>=80: out.append(-math.log(p))
        for a in ACTS: w[a]*=lam
        w[l]+=1
    return np.mean(out)
for hl in [5,10,20,40,80,1e9]:
    print('prequential nats, recency-weighted state-blind counter, half-life %5s: %.4f'%(('inf' if hl>1e6 else int(hl)),preq(hl)))
print('LLM prequential on the same examples: %.4f'%np.mean([nats[i] for i,l in ntl if i>=80]))
# snapshot-aware counter (uses only the snapshot index, which is visible via remaining_units)
def preq_snap(alpha=1.0):
    w={s:{a:0.0 for a in ACTS} for s in range(4)}; out=[]
    for i,l in ntl:
        s=byid[steps[i]['record_id']]['snapshot']; tot=sum(w[s].values())+alpha*8; p=(w[s][l]+alpha)/tot
        if i>=80: out.append(-math.log(p))
        w[s][l]+=1
    return np.mean(out)
print('prequential nats, counter conditioned on snapshot index only (add-one): %.4f'%preq_snap())
