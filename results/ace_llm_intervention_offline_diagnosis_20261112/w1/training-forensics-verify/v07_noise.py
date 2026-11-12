"""Check F: tie rule, chance rate of strict labels, reliability of replicate means, BH correction, base NMSE of DPO states."""
import json, os, math, itertools
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
nt=[r for r in R if not r['terminal']]
sel=json.load(open(OUT+'/selected_pairs.json'))
sel_rids=set(x[1] for x in sel)
print('tie_multiplier',Counter(r['fb']['tie_multiplier'] for r in nt),'min gap',Counter(r['fb']['minimum_gain_gap'] for r in nt),'replicates',Counter(r['fb']['replicate_count'] for r in nt))
# recompute every pair from per-replicate gains and check the saved tie flags
recheck=0; tvals=[]; pvals=[]; strict_flags=[]; meta=[]
signflip_expected=0.0
for r in nt:
    G=r['fb']['per_replicate_gains_per_cost']
    saved={(p['left'],p['right']):p for p in r['fb']['pairs']}
    for (l,rt),p in saved.items():
        d=np.array(G[l])-np.array(G[rt])
        m=d.mean(); se=d.std(ddof=1)/2.0
        strict=abs(m)>2*se
        if strict==p['tie'] or abs(m-p['mean_gain_difference'])>1e-12 or abs(se-p['paired_standard_error'])>1e-12: recheck+=1
        t=m/se if se>0 else float('inf')
        tvals.append(t); strict_flags.append(strict)
        pvals.append(2*stats.t.sf(abs(t),3))
        meta.append((r['rid'],l,rt))
        # sign-flip null: fraction of the 16 sign patterns that would be strict
        c=0
        for signs in itertools.product([1,-1],repeat=4):
            dd=d*np.array(signs); mm=dd.mean(); ss=dd.std(ddof=1)/2.0
            c+= abs(mm)>2*ss
        signflip_expected+=c/16
tvals=np.array(tvals); pvals=np.array(pvals); strict_flags=np.array(strict_flags)
print('pairs recomputed',len(tvals),'disagreements with saved flags/means/SE:',recheck)
print('strict',int(strict_flags.sum()),'of',len(strict_flags),'= %.4f'%strict_flags.mean())
alpha=2*stats.t.sf(2,3)
print('P(|t_3|>2) = %.4f ; expected strict under exact-normal null = %.1f'%(alpha,alpha*len(tvals)))
print('expected strict under within-pair sign-flip null (keeps observed |differences|): %.1f'%signflip_expected)
print('excess over normal null: %.1f = %.2f%% of comparisons ; over sign-flip null: %.1f = %.2f%%'%(strict_flags.sum()-alpha*len(tvals),100*(strict_flags.sum()-alpha*len(tvals))/len(tvals),strict_flags.sum()-signflip_expected,100*(strict_flags.sum()-signflip_expected)/len(tvals)))
print('implied share of strict labels that are chance-level under normal null: %.3f ; under sign-flip null: %.3f'%(alpha*len(tvals)/strict_flags.sum(),signflip_expected/strict_flags.sum()))
# simulation null that respects the 8-action structure: iid normal gains, 8 actions x 4 reps -> 28 pairs
rng=np.random.default_rng(0)
cnt=[]
for _ in range(4000):
    g=rng.standard_normal((8,4)); c=0
    for i in range(8):
        for j in range(i+1,8):
            d=g[i]-g[j]; c+= abs(d.mean())>2*d.std(ddof=1)/2
    cnt.append(c)
cnt=np.array(cnt)
print('iid-normal 8x4 simulation: mean strict per record %.3f (x128 = %.1f), sd per record %.2f -> sd of total about %.1f'%(cnt.mean(),cnt.mean()*128,cnt.std(),cnt.std()*math.sqrt(128)))
# heavy-tailed alternative
for name,gen in [('t3',lambda: rng.standard_t(3,(8,4))),('lognormal-ish',lambda: np.exp(rng.standard_normal((8,4)))),('chi2_1 (squared-error-like)',lambda: -(rng.standard_normal((8,4))**2))]:
    cnt=[]
    for _ in range(4000):
        g=gen(); c=0
        for i in range(8):
            for j in range(i+1,8):
                d=g[i]-g[j]; c+= abs(d.mean())>2*d.std(ddof=1)/2
        cnt.append(c)
    print('  iid %s null: expected strict total %.1f'%(name,np.mean(cnt)*128))
# BH
def bh(p,q):
    p=np.asarray(p); n=len(p); o=np.argsort(p); thr=q*np.arange(1,n+1)/n
    ok=np.where(p[o]<=thr)[0]
    return 0 if len(ok)==0 else ok.max()+1
for q in [0.05,0.10,0.20]:
    print('Benjamini-Hochberg q=%.2f keeps %d of %d'%(q,bh(pvals,q),len(pvals)))
print('smallest p-values:',np.sort(pvals)[:5], ' Bonferroni threshold 0.05/3584 = %.2e'%(0.05/3584))
print('p-value histogram (10 bins):',np.histogram(pvals,bins=10,range=(0,1))[0])
# reliability of the 4-replicate action means, per record
def reliab(r,mode):
    G=np.array([r['fb']['per_replicate_gains_per_cost'][a] for a in ACTS])  # 8 x 4
    a,k=G.shape
    gm=G.mean(); am=G.mean(1); rm=G.mean(0)
    ssb=k*((am-gm)**2).sum(); msb=ssb/(a-1)
    if mode=='oneway':
        ssw=((G-am[:,None])**2).sum(); msw=ssw/(a*(k-1))
    else: # two-way, replicate as block
        sse=((G-am[:,None]-rm[None,:]+gm)**2).sum(); msw=sse/((a-1)*(k-1))
    if msb<=0: return 0.0,msb,msw
    return max(0.0,(msb-msw)/msb),msb,msw
for mode in ['oneway','twoway']:
    rel=np.array([reliab(r,mode)[0] for r in nt])
    print('reliability (%s ANOVA, mean of 4): median %.3f mean %.3f zero in %d/128 ; quartiles %s'%(mode,np.median(rel),rel.mean(),int((rel==0).sum()),np.round(np.percentile(rel,[25,75]),3)))
    relsel=np.array([reliab(r,mode)[0] for r in nt if r['rid'] in sel_rids])
    print('     among the 119 DPO records: median %.3f zero in %d/119'%(np.median(relsel),int((relsel==0).sum())))
# pooled F statistics
F=[]
for r in nt:
    _,msb,msw=reliab(r,'twoway'); F.append(msb/msw)
F=np.array(F)
print('two-way F (actions; 7,21 df): median %.3f ; share with F<1: %.3f ; share significant at 0.05: %.3f (expected 0.05 under null)'%(np.median(F),(F<1).mean(),(stats.f.sf(F,7,21)<0.05).mean()))
# base NMSE of each state
def base_nmse(r):
    w=np.array(r['fb']['query_weights']); bl=np.array(r['fb']['base_losses'])
    v=(bl*w[None,:]).sum(1)
    return v
vals={}
same=0
for r in nt:
    v=base_nmse(r); vals[r['rid']]=v.mean(); same+= (np.ptp(v)<1e-15)
print('base NMSE identical across the 4 replicates in %d/128 records'%same)
n_le=sum(1 for r in nt if vals[r['rid']]<=0.04)
n_le_sel=sum(1 for rid in sel_rids if vals[rid]<=0.04)
print('states with base NMSE <= 0.04: %d/128 nonterminal ; among the 119 DPO records: %d/119'%(n_le,n_le_sel))
for s in range(4):
    rr=[r for r in nt if r['snapshot']==s]
    print('  snap %d: base NMSE<=0.04 in %d/32 ; median base NMSE %.4f'%(s,sum(1 for r in rr if vals[r['rid']]<=0.04),np.median([vals[r['rid']] for r in rr])))
# strict rate by whether the state is already below tolerance
for flag in [True,False]:
    rr=[r for r in nt if (vals[r['rid']]<=0.04)==flag]
    print('  base NMSE<=0.04 is %s: records %d strict share %.4f'%(flag,len(rr),sum(r['n_strict'] for r in rr)/(28*len(rr))))
# magnitude of gains relative to base loss
mg=[max(abs(v) for v in r['fb']['mean_gains_per_cost'].values()) for r in nt]
print('median over records of max |mean gain per cost|: %.5f ; median base NMSE %.5f'%(np.median(mg),np.median(list(vals.values()))))
# fraction of action means that are negative (an added sample made the estimate worse on average)
neg=sum(1 for r in nt for v in r['fb']['mean_gains_per_cost'].values() if v<0); tot=8*len(nt)
print('share of 4-replicate mean gains that are negative: %d/%d = %.3f'%(neg,tot,neg/tot))
json.dump(vals,open(OUT+'/base_nmse.json','w'))
