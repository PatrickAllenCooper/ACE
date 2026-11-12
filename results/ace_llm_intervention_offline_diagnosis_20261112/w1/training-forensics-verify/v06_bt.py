"""Check E: accuracy of a state-independent (constant) ranking on the strict preference labels.
Own implementation: Bradley-Terry by MM iterations with a tiny symmetric prior, plus exact best permutation, Borda, and nulls."""
import json, os, itertools, math, random
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
IDX={a:i for i,a in enumerate(ACTS)}
nt=[r for r in R if not r['terminal']]
def graph(r):
    p=r['parents']; key=(tuple(p['X']),tuple(p['Y']),tuple(p['Z']))
    return {((),('X',),('Y',)):'chain',((),('X',),('X',)):'fork',((),(),('X','Y')):'collider'}[key]
def costm(r): return 'het' if r['menu_costs']['set_y_neg']==2 else 'uni'
# strict pairs as (world, rid, winner idx, loser idx, record meta)
P=[]
for r in nt:
    for s in r['strict']:
        P.append({'w':r['world_id'],'rid':r['rid'],'win':IDX[s['chosen']],'lose':IDX[s['rejected']],'graph':graph(r),'cost':costm(r),'snap':r['snapshot']})
N=len(P); print('strict pairs',N,'worlds',len(set(p['w'] for p in P)))
def wins_matrix(pairs):
    W=np.zeros((8,8))
    for p in pairs: W[p['win'],p['lose']]+=1
    return W
W=wins_matrix(P)
# per-pair majority ceiling
ceil=sum(max(W[i,j],W[j,i]) for i in range(8) for j in range(i+1,8))
print('per-unordered-pair majority ceiling: %d/%d = %.4f'%(ceil,N,ceil/N))
print('wins matrix (row beats col):'); print(W.astype(int))
def bt_fit(W,prior=0.0,iters=5000,tol=1e-12):
    # MM algorithm (Hunter 2004); prior adds `prior` virtual wins in each direction for every pair
    Wp=W+prior*(1-np.eye(8))
    n=Wp+Wp.T; w=Wp.sum(1)
    p=np.ones(8)
    for _ in range(iters):
        denom=np.array([sum(n[i,j]/(p[i]+p[j]) for j in range(8) if j!=i) for i in range(8)])
        pn=w/denom; pn/=pn.sum()
        if np.max(np.abs(pn-p))<tol: p=pn; break
        p=pn
    return np.log(p)
def acc(pairs,score,tie='half'):
    h=0.0
    for p in pairs:
        d=score[p['win']]-score[p['lose']]
        if d>0: h+=1
        elif d==0: h+= 0.5 if tie=='half' else 0
    return h
for prior in [0.0,0.01,0.5,1.0]:
    s=bt_fit(W,prior)
    print('BT in-sample (prior %.2f): %.1f/%d = %.4f ; ranking: %s'%(prior,acc(P,s),N,acc(P,s)/N,' > '.join(ACTS[i] for i in np.argsort(-s))))
    print('     strengths',{ACTS[i]:round(float(s[i]-s.mean()),3) for i in range(8)})
# exact best permutation in-sample
best=-1;bestperm=None
Wd=W-W.T
for perm in itertools.permutations(range(8)):
    pos=[0]*8
    for rank,i in enumerate(perm): pos[i]=rank
    v=0
    for i in range(8):
        for j in range(8):
            if pos[i]<pos[j]: v+=W[i,j]
    if v>best: best=v; bestperm=perm
print('exact best constant total order in-sample: %d/%d = %.4f ; order %s'%(best,N,best/N,' > '.join(ACTS[i] for i in bestperm)))
# Borda (net wins) in-sample
borda=W.sum(1)-W.sum(0)
print('Borda net-wins ranking in-sample: %.1f/%d = %.4f'%(acc(P,borda),N,acc(P,borda)/N),dict(zip(ACTS,borda.astype(int))))
# leave-one-world-out
worlds=sorted(set(r['world_id'] for r in nt))
def lowo(fit,cellfn=lambda p:0):
    h=0.0;n=0
    for w in worlds:
        te=[p for p in P if p['w']==w]
        for cell in set(cellfn(p) for p in te):
            tr=[p for p in P if p['w']!=w and cellfn(p)==cell]
            tec=[p for p in te if cellfn(p)==cell]
            if not tr: h+=0.5*len(tec); n+=len(tec); continue
            s=fit(wins_matrix(tr))
            h+=acc(tec,s); n+=len(tec)
    return h,n
for prior in [0.0,0.01,0.5,1.0]:
    h,n=lowo(lambda W_:bt_fit(W_,prior))
    print('BT leave-one-world-out (prior %.2f): %.1f/%d = %.4f'%(prior,h,n,h/n))
h,n=lowo(lambda W_:W_.sum(1)-W_.sum(0)); print('Borda leave-one-world-out: %.1f/%d = %.4f'%(h,n,h/n))
def best_perm_scores(W_):
    best=-1;bp=None
    for perm in itertools.permutations(range(8)):
        v=0
        for a in range(8):
            for b in range(a+1,8): v+=W_[perm[a],perm[b]]
        if v>best: best=v;bp=perm
    s=np.zeros(8)
    for rank,i in enumerate(bp): s[i]=-rank
    return s
# per-pair majority LOWO (not a ranking, just the most flexible constant rule)
def pm_fit_acc():
    h=0.0
    for w in worlds:
        tr=wins_matrix([p for p in P if p['w']!=w])
        for p in [p for p in P if p['w']==w]:
            a,b=tr[p['win'],p['lose']],tr[p['lose'],p['win']]
            h+= 1 if a>b else (0.5 if a==b else 0)
    return h
print('per-pair-majority leave-one-world-out: %.1f/%d = %.4f'%(pm_fit_acc(),N,pm_fit_acc()/N))
# binomial-ish uncertainty treating worlds as clusters: per-world LOWO accuracy
per_w=[]
for w in worlds:
    tr=[p for p in P if p['w']!=w]; te=[p for p in P if p['w']==w]
    s=bt_fit(wins_matrix(tr),0.01)
    if te: per_w.append((acc(te,s),len(te)))
a=np.array([x for x,_ in per_w]); n_=np.array([y for _,y in per_w])
rng=np.random.default_rng(1)
boots=[]
for _ in range(4000):
    ii=rng.integers(0,len(a),len(a)); boots.append(a[ii].sum()/n_[ii].sum())
print('BT LOWO world-cluster bootstrap 95%% interval: %.3f - %.3f (worlds with strict pairs: %d)'%(np.percentile(boots,2.5),np.percentile(boots,97.5),len(a)))
# conditional rankings (is there learnable state dependence at the coarse level?)
for name,fn in [('graph',lambda p:p['graph']),('cost menu',lambda p:p['cost']),('graph x cost',lambda p:(p['graph'],p['cost'])),('snapshot',lambda p:p['snap'])]:
    ins=0.0
    for cell in set(fn(p) for p in P):
        sub=[p for p in P if fn(p)==cell]; ins+=acc(sub,bt_fit(wins_matrix(sub),0.01))
    h,n=lowo(lambda W_:bt_fit(W_,0.01),fn)
    print('BT conditional on %-13s in-sample %.1f/%d = %.4f ; LOWO %.1f/%d = %.4f'%(name,ins,N,ins/N,h,n,h/n))
# nulls
rng=np.random.default_rng(0)
def null_coin(reps=1000):
    out=[]
    for _ in range(reps):
        Q=[]
        for p in P:
            if rng.random()<0.5: Q.append(p)
            else: Q.append({**p,'win':p['lose'],'lose':p['win']})
        out.append(acc(Q,bt_fit(wins_matrix(Q),0.01,iters=500,tol=1e-9))/N)
    return np.array(out)
nc=null_coin(600)
print('NULL coin-flip direction, BT in-sample accuracy: mean %.4f sd %.4f ; P(null >= observed 390/687=%.4f) = %.4f'%(nc.mean(),nc.std(),390/687,(nc>=390/687-1e-9).mean()))
def null_perm_within_record(reps=1000):
    out=[]
    byrec=defaultdict(list)
    for p in P: byrec[p['rid']].append(p)
    for _ in range(reps):
        Q=[]
        for rid,ps in byrec.items():
            perm=rng.permutation(8)
            for p in ps: Q.append({**p,'win':int(perm[p['win']]),'lose':int(perm[p['lose']])})
        out.append(acc(Q,bt_fit(wins_matrix(Q),0.01,iters=500,tol=1e-9))/N)
    return np.array(out)
npm=null_perm_within_record(600)
obs=acc(P,bt_fit(W,0.01))/N
print('NULL random relabelling of the 8 actions within each record (keeps within-record dependence): mean %.4f sd %.4f ; P(null >= observed %.4f) = %.4f'%(npm.mean(),npm.std(),obs,(npm>=obs-1e-9).mean()))
# cost-menu note: Y and XY interventions cost 2 under het; so check net wins by menu
for cm in ['uni','het']:
    Wc=wins_matrix([p for p in P if p['cost']==cm])
    print('net wins under %s menu (n=%d):'%(cm,int(Wc.sum())),dict(zip(ACTS,(Wc.sum(1)-Wc.sum(0)).astype(int))))
for g in ['chain','fork','collider']:
    Wc=wins_matrix([p for p in P if p['graph']==g])
    print('net wins in %s (n=%d):'%(g,int(Wc.sum())),dict(zip(ACTS,(Wc.sum(1)-Wc.sum(0)).astype(int))))
