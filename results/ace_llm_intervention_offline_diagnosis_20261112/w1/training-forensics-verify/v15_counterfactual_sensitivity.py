"""Sensitivity of the toy all-687-pairs counterfactual and of the fitted scale to the step/momentum convention."""
import json, os
import numpy as np
from collections import Counter
OUT=os.path.dirname(os.path.abspath(__file__))
exec(open(OUT+'/v14_counterfactuals.py').read().split("real=[(IDX")[0])   # reuse my own model definitions only
rows=json.load(open(OUT+'/pair_scores.json')); on=json.load(open(OUT+'/dpo_online.json')); m_act=np.array(on['online_margin'])
real=[(IDX[x['ch']],IDX[x['rj']]) for x in rows]
strict_by_rec=[[(IDX[s['chosen']],IDX[s['rejected']]) for s in r['strict']] for r in nt if r['strict']]
allp=[x for sp in strict_by_rec for x in sp]
def run(pairs,step,mom,ema,th_init=th_sft):
    th=th_init.copy(); vel=np.zeros(8); lp0=logp_all(th_init); pred=[]
    for c,r in pairs:
        lp=logp_all(th); pred.append((lp[c]-lp[r])-(lp0[c]-lp0[r]))
        g=grad_logp(th,ACTS[c])-grad_logp(th,ACTS[r]); n=np.linalg.norm(g)
        if n>0: g/=n
        vel=mom*vel+((1-mom)*g if ema else g); th=th+step*vel
    return np.array(pred),logp_all(th)-lp0,ACTS[int(np.argmax(logp_all(th)))]
ent=[]
for x in rows: ent.append((x['ch'],x['fin_c']-x['ref_c'])); ent.append((x['rj'],x['fin_r']-x['ref_r']))
act_shift=np.array([np.mean([e[1] for e in ent if e[0]==a]) for a in ACTS])
rng=np.random.default_rng(1)
for ema in [False,True]:
    for step,mom in [(0.9,0.5),(0.45,0.5),(0.9,0.0),(1.8,0.5)]:
        pred,shift,w=run(real,step,mom,ema)
        r2=1-((m_act-pred)**2).sum()/(m_act**2).sum()
        wc=Counter(run([allp[i] for i in rng.permutation(len(allp))],step,mom,ema)[2] for _ in range(200))
        print('ema=%s step %.2f mom %.1f: r_margin %.3f R2-vs-zero %.3f r_shift %.3f shift(set_y_zero) %.1f (obs %.1f) | all-687 random orders winners %s'%(ema,step,mom,np.corrcoef(pred,m_act)[0,1],r2,np.corrcoef(shift,act_shift)[0,1],shift[IDX['set_y_zero']],act_shift[IDX['set_y_zero']],dict(wc)))
