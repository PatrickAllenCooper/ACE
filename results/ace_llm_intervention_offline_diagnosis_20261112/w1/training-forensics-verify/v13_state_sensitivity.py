"""Contradiction hunting: is there state-conditional structure in the saved scores that tracks labels?"""
import json, os, glob, math
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
IDX={a:i for i,a in enumerate(ACTS)}
byid={r['rid']:r for r in R}
nt=[r for r in R if not r['terminal']]
T=np.load(OUT+'/final_table.npy')            # 128 x 8 final-checkpoint log-likelihoods
rows=json.load(open(OUT+'/pair_scores.json'))
rng=np.random.default_rng(0)
snap=np.array([r['snapshot'] for r in nt]); y=np.array([IDX[r['sup']] for r in nt]); teach=np.array([r['teacher_method'] for r in nt])
L=T-np.log(np.exp(T).sum(1,keepdims=True))
def interaction_stat(M,yy,strata):
    # mean over states of M[s,y_s] minus the stratum mean of column y_s
    d=np.zeros(len(yy))
    for st in np.unique(strata):
        ii=np.where(strata==st)[0]; cm=M[ii].mean(0)
        d[ii]=M[ii,yy[ii]]-cm[yy[ii]]
    return d
def perm_test(M,yy,strata,reps=5000):
    obs=interaction_stat(M,yy,strata).mean(); cnt=0
    for _ in range(reps):
        yp=yy.copy()
        for st in np.unique(strata):
            ii=np.where(strata==st)[0]; yp[ii]=yy[rng.permutation(ii)]
        cnt+= interaction_stat(M,yp,strata).mean()>=obs
    return obs,(cnt+1)/(reps+1)
print('FINAL (DPO) checkpoint, 128 nonterminal training states:')
for name,M in [('raw log-likelihood',T),('renormalized over 8 actions',L)]:
    obs,p=perm_test(M,y,snap)
    print('  %s: mean [score of teacher label in its own state - mean score of that command over same-snapshot states] = %+.3f nats ; permutation p (labels shuffled within snapshot) = %.4f'%(name,obs,p))
d=interaction_stat(T,y,snap)
for m in ['uniform_random','menu_coverage','local_information','target_risk']:
    print('    teacher %-18s mean %+.3f (n=%d) ; share positive %.2f'%(m,d[teach==m].mean(),(teach==m).sum(),(d[teach==m]>0).mean()))
for a in ACTS:
    if (y==IDX[a]).sum()>=4: print('    label %-12s mean %+.3f (n=%d)'%(a,d[y==IDX[a]].mean(),(y==IDX[a]).sum()))
# same test against the measured-best action (does the policy score track measured gains?)
best=np.array([IDX[max(ACTS,key=lambda a:r['fb']['mean_gains_per_cost'][a])] for r in nt])
obs,p=perm_test(T,best,snap); print('  vs measured-best action: mean %+.3f ; p = %.4f'%(obs,p))
# per-state Spearman between final scores and measured gains / planner scores
rho=[stats.spearmanr(T[i],[r['fb']['mean_gains_per_cost'][a] for a in ACTS])[0] for i,r in enumerate(nt)]
print('  per-state Spearman(final scores, measured mean gains): mean %.3f (t-test p %.3f)'%(np.mean(rho),stats.ttest_1samp(rho,0)[1]))
# does history copying explain it? correlate final score of each command with count of that command in history, within snapshot>0
for a in ['set_x_pos','set_x_neg','observe_all','set_xy','set_y_neg']:
    ii=[i for i,r in enumerate(nt) if r['snapshot']>0]
    cnt=[nt[i]['history_actions'][8:].count(a) for i in ii]
    print('    corr(final lp(%s), count of %s among acquired history; snapshots 1-3) = %.3f'%(a,a,np.corrcoef(T[ii,IDX[a]],cnt)[0,1]))
# partial: within-state rank of label after removing column means -> and controlling for history count? simple regression
Xh=np.array([[r['history_actions'][8:].count(a) for a in ACTS] for r in nt],float)
dd=T-np.array([T[snap==s].mean(0) for s in range(4)])[snap]   # snapshot-centred scores
lab_onehot=np.zeros_like(T); lab_onehot[np.arange(128),y]=1
Xc=Xh-np.array([Xh[snap==s].mean(0) for s in range(4)])[snap]
A=np.column_stack([lab_onehot.ravel(),Xc.ravel(),np.ones(T.size)])
coef=np.linalg.lstsq(A,dd.ravel(),rcond=None)[0]
print('  regression of snapshot-centred final scores on [is-teacher-label, centred history count of that command]: coef label %+.3f nats ; per extra history occurrence %+.3f nats'%(coef[0],coef[1]))
# SFT reference entries: command equals that record's SFT label vs not (z within command x snapshot-group)
ent=[]
for x in rows:
    r=byid[x['rid']]
    ent.append((x['ch'],r['snapshot'],x['ref_c'],x['fin_c'],r['sup']==x['ch'],r['history_actions'][8:].count(x['ch'])))
    ent.append((x['rj'],r['snapshot'],x['ref_r'],x['fin_r'],r['sup']==x['rj'],r['history_actions'][8:].count(x['rj'])))
cell=defaultdict(list)
for e in ent: cell[(e[0],e[1]==0)].append(e[2])
zs=[];zo=[]
for e in ent:
    v=cell[(e[0],e[1]==0)]
    if len(v)<3: continue
    z=e[2]-np.mean(v)
    (zs if e[4] else zo).append(z)
print('\nSFT reference checkpoint (saved 238 entries): entries whose command is that record\'s own SFT label: n=%d mean deviation from command x (snapshot0/later) mean = %+.3f nats ; other entries n=%d mean %+.3f ; Welch p=%.3f'%(len(zs),np.mean(zs),len(zo),np.mean(zo),stats.ttest_ind(zs,zo,equal_var=False)[1]))
# residual pair-specific margin change after command-mean shifts
shift={a:np.mean([e[3]-e[2] for e in ent if e[0]==a]) for a in ACTS}
act=np.array([(x['fin_c']-x['ref_c'])-(x['fin_r']-x['ref_r']) for x in rows]); pred=np.array([shift[x['ch']]-shift[x['rj']] for x in rows])
res=act-pred
print('\nper-pair margin change: corr(actual, predicted from command-mean shifts) = %.3f ; residual mean %+.3f sd %.2f ; t-test p = %.4f'%(np.corrcoef(act,pred)[0,1],res.mean(),res.std(ddof=1),stats.ttest_1samp(res,0)[1]))
print('actual margin change: mean %+.3f ; positive in %d/119'%(act.mean(),(act>0).sum()))
# token factorisation of the final table
print('\nfinal table: corr over states of (x_pos-x_neg) with (y_pos-y_neg) = %.3f ; (x_zero-x_neg) with (y_zero-y_neg) = %.3f'%(np.corrcoef(T[:,3]-T[:,1],T[:,6]-T[:,4])[0,1],np.corrcoef(T[:,2]-T[:,1],T[:,5]-T[:,4])[0,1]))
print('final table (x_pos - x_neg): mean %.2f sd %.2f min %.2f max %.2f ; by snapshot %s'%((T[:,3]-T[:,1]).mean(),(T[:,3]-T[:,1]).std(),(T[:,3]-T[:,1]).min(),(T[:,3]-T[:,1]).max(),[round(float((T[snap==s,3]-T[snap==s,1]).mean()),2) for s in range(4)]))
# gradient files: do they contain norms?
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
g=json.load(open(TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit/training/step-00160.gradients.json'))
print('\ngradients.json keys:',list(g.keys()),'; tensor entry keys:',list(g['tensors'][0].keys()),'; tensors:',len(g['tensors']))
