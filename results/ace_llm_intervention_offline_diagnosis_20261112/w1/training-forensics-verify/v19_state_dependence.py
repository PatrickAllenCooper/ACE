"""Is the planner-agreement signal in the strict labels state-dependent, or would a constant planner-average ranking do as well?
Also: per-unordered-pair binomial tests."""
import json, os, math
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
nt=[r for r in R if not r['terminal']]
for m,key in [('target_risk','risk_reduction_per_cost'),('local_information','scores')]:
    rows=[r for r in nt if r['teacher_method']==m]
    own=const=n=0; constrank=0
    for r in rows:
        others=[x for x in rows if x['world_id']!=r['world_id']]
        avg={a:np.mean([x['teacher_meta'][key][a] for x in others]) for a in ACTS}
        # rank-average version (scale-free)
        rk={a:np.mean([stats.rankdata([x['teacher_meta'][key][b] for b in ACTS])[ACTS.index(a)] for x in others]) for a in ACTS}
        sc=r['teacher_meta'][key]
        for s in r['strict']:
            n+=1; own+= sc[s['chosen']]>sc[s['rejected']]; const+= avg[s['chosen']]>avg[s['rejected']]; constrank+= rk[s['chosen']]>rk[s['rejected']]
    print('%s states: strict labels agree with the state\'s own planner ordering %d/%d = %.3f ; with the leave-one-world-out AVERAGE planner score ordering %d/%d = %.3f ; with average planner rank ordering %d/%d = %.3f'%(m,own,n,own/n,const,n,const/n,constrank,n,constrank/n))
    # same-menu average (cost menu matters for per-cost scores)
    c2=0
    for r in rows:
        others=[x for x in rows if x['world_id']!=r['world_id'] and x['menu_costs']==r['menu_costs']]
        avg={a:np.mean([x['teacher_meta'][key][a] for x in others]) for a in ACTS}
        for s in r['strict']: c2+= avg[s['chosen']]>avg[s['rejected']]
    print('    with the leave-one-world-out average over same-cost-menu states: %d/%d = %.3f'%(c2,n,c2/n))
    # how constant is the planner ordering itself? mean pairwise Spearman between states
    M=np.array([[r['teacher_meta'][key][a] for a in ACTS] for r in rows])
    rs=[stats.spearmanr(M[i],M[j])[0] for i in range(len(M)) for j in range(i+1,len(M))]
    print('    mean pairwise Spearman of planner score vectors across the %d states: %.3f'%(len(M),np.mean(rs)))
    # permutation: assign each state another state's planner ordering (same snapshot) -> agreement
    rng=np.random.default_rng(0); perm=[]
    for _ in range(2000):
        a_=0
        idx=rng.permutation(len(rows))
        for r,j in zip(rows,idx):
            sc=rows[j]['teacher_meta'][key]
            for s in r['strict']: a_+= sc[s['chosen']]>sc[s['rejected']]
        perm.append(a_/n)
    perm=np.array(perm)
    print('    agreement when each state is paired with ANOTHER state\'s planner scores (2000 shuffles): mean %.3f, 95th pct %.3f ; P(shuffled >= own) = %.4f'%(perm.mean(),np.percentile(perm,95),(perm>=own/n).mean()))
# per-unordered-pair binomial tests on all 687 strict pairs
W=np.zeros((8,8),int)
for r in nt:
    for s in r['strict']: W[ACTS.index(s['chosen']),ACTS.index(s['rejected'])]+=1
ps=[]
for i in range(8):
    for j in range(i+1,8):
        k,n=W[i,j],W[i,j]+W[j,i]
        p=stats.binomtest(int(k),int(n),0.5).pvalue if n else 1.0
        ps.append((p,ACTS[i],ACTS[j],int(W[i,j]),int(W[j,i])))
ps.sort()
print('\nmost lopsided unordered pairs (binomial p, a, b, a wins, b wins):',[(round(p,4),a,b,x,y) for p,a,b,x,y in ps[:4]])
print('min p x 28 (Bonferroni) = %.3f ; pairs with p<0.05 uncorrected: %d of 28'%(ps[0][0]*28,sum(1 for p in ps if p[0]<0.05)))
