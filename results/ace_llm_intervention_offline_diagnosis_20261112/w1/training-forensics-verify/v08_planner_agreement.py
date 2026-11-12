"""Check G and K: agreement between strict measured-preference labels and the informed teachers' own score orderings."""
import json, os, math
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
nt=[r for r in R if not r['terminal']]
def lab(r): return r['sup']
tr=[r for r in nt if r['teacher_method']=='target_risk']
li=[r for r in nt if r['teacher_method']=='local_information']
print('target-risk states',len(tr),'local-information states',len(li))
# is the decision the argmax of the planner's score?
print('target_risk: decision == argmax risk_reduction_per_cost in %d/%d'%(sum(1 for r in tr if max(r['teacher_meta']['risk_reduction_per_cost'],key=r['teacher_meta']['risk_reduction_per_cost'].get)==lab(r)),len(tr)))
print('local_information: decision == argmax scores in %d/%d ; == argmax scores/cost in %d/%d'%(sum(1 for r in li if max(r['teacher_meta']['scores'],key=r['teacher_meta']['scores'].get)==lab(r)),len(li),sum(1 for r in li if max(ACTS,key=lambda a:r['teacher_meta']['scores'][a]/r['menu_costs'][a])==lab(r)),len(li)))
def agree(rows,scorefn,name):
    a=n=t=0
    for r in rows:
        sc=scorefn(r)
        for s in r['strict']:
            d=sc[s['chosen']]-sc[s['rejected']]
            n+=1
            if d>0: a+=1
            elif d==0: t+=1
    print('%s: strict labels agreeing with planner ordering %d/%d = %.4f (score ties %d) ; records %d with strict %d'%(name,a,n,a/n if n else float('nan'),t,len(rows),sum(1 for r in rows if r['n_strict']>0)))
    return a,n
a1,n1=agree(tr,lambda r:r['teacher_meta']['risk_reduction_per_cost'],'target_risk (risk_reduction_per_cost)')
a2,n2=agree(li,lambda r:r['teacher_meta']['scores'],'local_information (scores as saved)')
a3,n3=agree(li,lambda r:{a:r['teacher_meta']['scores'][a]/r['menu_costs'][a] for a in ACTS},'local_information (scores / cost)')
# binomial CI ignoring dependence, and cluster bootstrap by record
def cluster_boot(rows,scorefn,reps=4000,seed=0):
    rng=np.random.default_rng(seed)
    per=[]
    for r in rows:
        sc=scorefn(r); a=sum(1 for s in r['strict'] if sc[s['chosen']]>sc[s['rejected']]); per.append((a,len(r['strict'])))
    per=np.array(per); out=[]
    for _ in range(reps):
        ii=rng.integers(0,len(per),len(per)); tot=per[ii,1].sum()
        if tot: out.append(per[ii,0].sum()/tot)
    return np.percentile(out,[2.5,97.5])
print('  target_risk agreement record-cluster bootstrap 95%%: %s'%np.round(cluster_boot(tr,lambda r:r['teacher_meta']['risk_reduction_per_cost']),3))
print('  local_information agreement record-cluster bootstrap 95%%: %s'%np.round(cluster_boot(li,lambda r:r['teacher_meta']['scores']),3))
# per-state Spearman between predicted (planner) and measured mean gains per cost
def spear(rows,scorefn,name):
    rs=[]
    for r in rows:
        sc=scorefn(r); x=[sc[a] for a in ACTS]; y=[r['fb']['mean_gains_per_cost'][a] for a in ACTS]
        rho=stats.spearmanr(x,y)[0]; rs.append(rho)
    rs=np.array(rs)
    t=rs.mean()/(rs.std(ddof=1)/math.sqrt(len(rs)))
    print('%s: per-state Spearman(pred, measured mean gain/cost): mean %.4f median %.4f sd %.3f ; share >0 %.3f ; one-sample t=%.2f p=%.3f'%(name,rs.mean(),np.median(rs),rs.std(ddof=1),(rs>0).mean(),t,2*stats.t.sf(abs(t),len(rs)-1)))
    return rs
spear(tr,lambda r:r['teacher_meta']['risk_reduction_per_cost'],'target_risk')
spear(li,lambda r:r['teacher_meta']['scores'],'local_information(saved scores)')
spear(li,lambda r:{a:r['teacher_meta']['scores'][a]/r['menu_costs'][a] for a in ACTS},'local_information(scores/cost)')
# does the measured-best action agree with the planner-best action?
def top_agree(rows,scorefn,name):
    k=sum(1 for r in rows if max(ACTS,key=lambda a:scorefn(r)[a])==max(ACTS,key=lambda a:r['fb']['mean_gains_per_cost'][a]))
    # rank of the teacher's chosen action among the measured means (1=best)
    ranks=[sorted(ACTS,key=lambda a:-r['fb']['mean_gains_per_cost'][a]).index(lab(r))+1 for r in rows]
    print('%s: planner top == measured top in %d/%d (chance %.1f) ; mean measured rank of teacher choice %.2f (chance 4.5); measured gain of teacher choice >0 in %d/%d'%(name,k,len(rows),len(rows)/8,np.mean(ranks),sum(1 for r in rows if r['fb']['mean_gains_per_cost'][lab(r)]>0),len(rows)))
top_agree(tr,lambda r:r['teacher_meta']['risk_reduction_per_cost'],'target_risk')
top_agree(li,lambda r:r['teacher_meta']['scores'],'local_information')
for m in ['uniform_random','menu_coverage','local_information','target_risk']:
    rows=[r for r in nt if r['teacher_method']==m]
    ranks=[sorted(ACTS,key=lambda a:-r['fb']['mean_gains_per_cost'][a]).index(lab(r))+1 for r in rows]
    # is the SFT label ever rejected / chosen in a strict pair
    lab_pref=sum(1 for r in rows for s in r['strict'] if s['chosen']==lab(r)); lab_rej=sum(1 for r in rows for s in r['strict'] if s['rejected']==lab(r))
    print('teacher %-18s mean measured rank of label %.2f ; label is measured-top in %d/32 ; in strict pairs label preferred %d rejected %d'%(m,np.mean(ranks),sum(1 for x in ranks if x==1),lab_pref,lab_rej))
# SFT label vs DPO selected pair conflicts
sel=json.load(open(OUT+'/selected_pairs.json'))
selmap={x[1]:(x[2],x[3]) for x in sel}
conf=sum(1 for r in nt if r['rid'] in selmap and selmap[r['rid']][1]==lab(r)); sup=sum(1 for r in nt if r['rid'] in selmap and selmap[r['rid']][0]==lab(r))
print('DPO-selected pair rejects the SFT label in %d/119 records ; prefers the SFT label in %d/119 ; unrelated %d'%(conf,sup,119-conf-sup))
# Check K: target-risk labels and the best constant rule, leave-one-world-out
c=Counter(lab(r) for r in tr); print('target-risk labels',dict(c))
worlds=sorted(set(r['world_id'] for r in tr))
print('target-risk states per world',Counter(Counter(r['world_id'] for r in tr).values()))
hit=0;hitf=0
for w in worlds:
    trn=Counter(lab(r) for r in tr if r['world_id']!=w)
    mx=max(trn.values()); cands=[a for a in ACTS if trn[a]==mx]
    for r in tr:
        if r['world_id']==w:
            hit+= lab(r)==cands[0]; hitf+=(lab(r) in cands)/len(cands)
print('best constant rule LOWO on target-risk labels: first-tiebreak %d/32 ; fractional %.2f/32 ; in-sample %d/32'%(hit,hitf,max(c.values())))
# leave-one-STATE-out too
hit=0
for i,r in enumerate(tr):
    trn=Counter(lab(x) for j,x in enumerate(tr) if j!=i); mx=max(trn.values()); cands=[a for a in ACTS if trn[a]==mx]
    hit+=(lab(r) in cands)/len(cands)
print('best constant rule leave-one-state-out on target-risk labels: %.2f/32'%hit)
# runner-up within 2 MC SE
close=0; close_pooled=0
for r in tr:
    m=r['teacher_meta']; sc=m['risk_reduction_per_cost']; se=m['monte_carlo_standard_errors']
    order=sorted(ACTS,key=lambda a:-sc[a]); top,ru=order[0],order[1]
    gap=sc[top]-sc[ru]
    c1=r['menu_costs'][top]; c2=r['menu_costs'][ru]
    # SE of risk_after is per action; per-cost scaling
    se_top=se[top]/c1; se_ru=se[ru]/c2
    if gap<=2*se_top or gap<=2*se_ru: pass
    close+= gap<=2*max(se_top,se_ru)
    close_pooled+= gap<=2*math.sqrt(se_top**2+se_ru**2)
print('target-risk runner-up within 2 MC SE of the chosen action: %d/32 (2*max SE) ; %d/32 (2*sqrt(se1^2+se2^2)) ; unscaled-by-cost variants below'%(close,close_pooled))
for variant in ['max_raw','chosen_raw','pooled_raw','chosen_percost','runnerup_percost']:
    k=0
    for r in tr:
        m=r['teacher_meta']; sc=m['risk_reduction_per_cost']; se=m['monte_carlo_standard_errors']
        order=sorted(ACTS,key=lambda a:-sc[a]); top,ru=order[0],order[1]; gap=sc[top]-sc[ru]
        if variant=='max_raw': thr=2*max(se[top],se[ru])
        elif variant=='chosen_raw': thr=2*se[top]
        elif variant=='pooled_raw': thr=2*math.sqrt(se[top]**2+se[ru]**2)
        elif variant=='chosen_percost': thr=2*se[top]/r['menu_costs'][top]
        else: thr=2*se[ru]/r['menu_costs'][ru]
        k+= gap<=thr
    print('   variant %-18s %d/32'%(variant,k))
