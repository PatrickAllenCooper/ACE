"""Toy counterfactuals in my re-implementation of the state-blind token-bias model (inference only, not evidence about the real network),
plus two extra label checks: selected-pair agreement with planners, and X-vs-Y preference by cost menu."""
import json, os, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
IDX={a:i for i,a in enumerate(ACTS)}
byid={r['rid']:r for r in R}
nt=[r for r in R if not r['terminal']]
rows=json.load(open(OUT+'/pair_scores.json'))
ent=[]
for x in rows: ent.append((x['ch'],x['ref_c'])); ent.append((x['rj'],x['ref_r']))
ref_mean=np.array([np.mean([e[1] for e in ent if e[0]==a]) for a in ACTS])
PATH={'observe_all':(0,None,None),'set_xy':(1,4,None),'set_x_neg':(1,2,5),'set_x_pos':(1,2,6),'set_x_zero':(1,2,7),'set_y_neg':(1,3,5),'set_y_pos':(1,3,6),'set_y_zero':(1,3,7)}
NODES=[(0,1),(2,3,4),(5,6,7)]
def logp_all(th):
    ls=np.zeros(8)
    for lo in NODES:
        idx=list(lo); z=th[idx]; ls[idx]=z-(np.log(np.exp(z-z.max()).sum())+z.max())
    return np.array([sum(ls[k] for k in PATH[a] if k is not None) for a in ACTS])
def grad_logp(th,a):
    g=np.zeros(8)
    for node,k in zip(NODES,PATH[a]):
        if k is None: continue
        idx=list(node); z=th[idx]; p=np.exp(z-z.max()); p/=p.sum(); g[idx]-=p; g[k]+=1
    return g
p=np.exp(ref_mean)
root=np.array([p[0],p[1:].sum()]); root/=root.sum()
sec=np.array([p[1]+p[2]+p[3],p[4]+p[5]+p[6],p[7]]); sec/=sec.sum()
suf=np.array([p[1]+p[4],p[3]+p[6],p[2]+p[5]]); suf/=suf.sum()
th_sft=np.log(np.concatenate([root,sec,suf])); th_uni=np.zeros(8)
def winner(pairs,step=0.9,mom=0.5,th_init=th_sft):
    th=th_init.copy(); vel=np.zeros(8)
    for c,r in pairs:
        g=grad_logp(th,ACTS[c])-grad_logp(th,ACTS[r]); n=np.linalg.norm(g)
        if n>0: g/=n
        vel=mom*vel+g; th=th+step*vel
    return ACTS[int(np.argmax(logp_all(th)))]
real=[(IDX[x['ch']],IDX[x['rj']]) for x in rows]
rng=np.random.default_rng(0)
print('actual 119 pairs, real order: winner from SFT init = %s ; from uniform init = %s'%(winner(real),winner(real,th_init=th_uni)))
for step in [0.1,0.3,0.9,2.0]:
    for mom in [0.0,0.5,0.9]:
        w=Counter(winner([real[i] for i in rng.permutation(119)],step,mom) for _ in range(200))
        print('  random orders step %.1f mom %.1f: %s'%(step,mom,dict(w)))
# different random one-strict-pair-per-record selection
strict_by_rec=[[(IDX[s['chosen']],IDX[s['rejected']]) for s in r['strict']] for r in nt if r['strict']]
w=Counter()
for _ in range(2000):
    w[winner([sp[rng.integers(len(sp))] for sp in strict_by_rec])]+=1
print('random one-strict-pair-per-record selections (2000, roster order): %s'%{k:round(v/2000,3) for k,v in w.most_common()})
allp=[x for sp in strict_by_rec for x in sp]
w=Counter(winner([allp[i] for i in rng.permutation(len(allp))]) for _ in range(500)); print('all 687 strict pairs in random orders (500):',dict(w))
print('all 687 strict pairs in roster order:',winner(allp))
w=Counter()
for _ in range(2000):
    w[winner([(c,r) if rng.random()<0.5 else (r,c) for c,r in real])]+=1
print('coin-flip directions on the actual 119 pairs (2000):',{k:round(v/2000,3) for k,v in w.most_common()})
w=Counter()
for _ in range(2000):
    w[winner([(c,r) if rng.random()<0.5 else (r,c) for c,r in real],th_init=th_uni)]+=1
print('coin-flip directions, uniform init (2000):',{k:round(v/2000,3) for k,v in w.most_common()})
# ---- extra label checks
sel=json.load(open(OUT+'/selected_pairs.json')); selmap={x[1]:(x[2],x[3]) for x in sel}
for m,key in [('target_risk','risk_reduction_per_cost'),('local_information','scores')]:
    a=n=0
    for r in nt:
        if r['teacher_method']!=m or r['rid'] not in selmap: continue
        sc=r['teacher_meta'][key]; c,rj=selmap[r['rid']]; n+=1; a+= sc[c]>sc[rj]
    print('DPO-SELECTED pair agrees with %s ordering in %d/%d states'%(m,a,n))
# X-type (cost 1 always) vs Y-type (set_y_*, set_xy; cost 2 under het) strict pairs, by cost menu
def typ(a): return 'X' if a.startswith('set_x_') else ('Y' if (a.startswith('set_y_') or a=='set_xy') else 'O')
for cm in ['uni','het']:
    xw=yw=0
    for r in nt:
        if (r['menu_costs']['set_y_neg']==2)!=(cm=='het'): continue
        for s in r['strict']:
            t=(typ(s['chosen']),typ(s['rejected']))
            if t==('X','Y'): xw+=1
            if t==('Y','X'): yw+=1
    print('cost menu %s: strict X-vs-Y pairs: X preferred %d, Y preferred %d (X share %.3f)'%(cm,xw,yw,xw/(xw+yw)))
# is that difference robust at the world level? per-world X share by menu
def world_share(cm):
    out=[]
    for w in sorted(set(r['world_id'] for r in nt)):
        xw=yw=0
        for r in nt:
            if r['world_id']!=w or (r['menu_costs']['set_y_neg']==2)!=(cm=='het'): continue
            for s in r['strict']:
                t=(typ(s['chosen']),typ(s['rejected']))
                xw+= t==('X','Y'); yw+= t==('Y','X')
        if xw+yw>0: out.append(xw/(xw+yw))
    return out
from scipy import stats
u,h=world_share('uni'),world_share('het')
print('per-world X share: uni mean %.3f (n=%d worlds) ; het mean %.3f (n=%d) ; Mann-Whitney p=%.4f'%(np.mean(u),len(u),np.mean(h),len(h),stats.mannwhitneyu(u,h)[1]))
# raw (not per-cost) reductions: would Y be preferred under het if cost were ignored? mean raw loss reduction by type and menu
for cm in ['uni','het']:
    red=defaultdict(list)
    for r in nt:
        if (r['menu_costs']['set_y_neg']==2)!=(cm=='het'): continue
        for a in ACTS:
            red[typ(a)].append(r['fb']['mean_gains_per_cost'][a]*r['fb']['candidate_costs'][a])
    print('cost menu %s: mean raw NMSE reduction per added sample by type: %s ; grand mean %.5f'%(cm,{k:round(float(np.mean(v)),5) for k,v in red.items()},np.mean([x for v in red.values() for x in v])))
allred=[r['fb']['mean_gains_per_cost'][a]*r['fb']['candidate_costs'][a] for r in nt for a in ACTS]
print('overall mean raw NMSE reduction from one extra sample: %.5f ; median %.5f ; share positive %.3f'%(np.mean(allred),np.median(allred),np.mean(np.array(allred)>0)))
for s in range(4):
    v=[r['fb']['mean_gains_per_cost'][a]*r['fb']['candidate_costs'][a] for r in nt if r['snapshot']==s for a in ACTS]
    print('  snapshot %d: mean raw reduction %.5f ; share positive %.3f'%(s,np.mean(v),np.mean(np.array(v)>0)))
