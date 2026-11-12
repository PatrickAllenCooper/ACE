"""Check J (own re-implementation): can a model with no state input reproduce the saved DPO online margins and final shifts?
Also examines how state-dependent the SFT reference and final scores are on the 238 scored (record, command) entries."""
import json, os, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
IDX={a:i for i,a in enumerate(ACTS)}
byid={r['rid']:r for r in R}
rows=json.load(open(OUT+'/pair_scores.json'))
on=json.load(open(OUT+'/dpo_online.json'))
m_act=np.array(on['online_margin'])           # (pi_c - ref_c) - (pi_r - ref_r) at the time of each update
beta=0.1
c=np.array([IDX[x['ch']] for x in rows]); r_=np.array([IDX[x['rj']] for x in rows])
refd=np.array([x['ref_c']-x['ref_r'] for x in rows])
find=np.array([x['fin_c']-x['fin_r'] for x in rows])
snap=np.array([byid[x['rid']]['snapshot'] for x in rows])
print('n pairs',len(rows))
# 1) eta^2 by command for reference, final and change on the 238 entries
ent=[]
for x in rows:
    s=byid[x['rid']]['snapshot']
    ent.append((x['ch'],s,x['ref_c'],x['fin_c'])); ent.append((x['rj'],s,x['ref_r'],x['fin_r']))
def eta2(vals,groups):
    vals=np.array(vals); gm=vals.mean(); sst=((vals-gm)**2).sum(); d=defaultdict(list)
    for v,g in zip(vals,groups): d[g].append(v)
    ssb=sum(len(v)*(np.mean(v)-gm)**2 for v in d.values()); return ssb/sst
for name,vals in [('SFT reference score',[e[2] for e in ent]),('final score',[e[3] for e in ent]),('change (final-ref)',[e[3]-e[2] for e in ent])]:
    print('%-22s eta^2 by command %.3f ; by command x (snapshot==0) %.3f ; by command x snapshot %.3f ; by snapshot only %.3f'%(name,eta2(vals,[e[0] for e in ent]),eta2(vals,[(e[0],e[1]==0) for e in ent]),eta2(vals,[(e[0],e[1]) for e in ent]),eta2(vals,[e[1] for e in ent])))
print('SFT reference mean score by command:',{a:round(float(np.mean([e[2] for e in ent if e[0]==a])),2) for a in ACTS})
print('final mean score by command (same entries):',{a:round(float(np.mean([e[3] for e in ent if e[0]==a])),2) for a in ACTS})
print('SFT reference sd within command:',{a:round(float(np.std([e[2] for e in ent if e[0]==a],ddof=1)),2) for a in ACTS})
print('final sd within command:',{a:round(float(np.std([e[3] for e in ent if e[0]==a],ddof=1)),2) for a in ACTS})
# change by command x snapshot0
print('change by command, snapshot 0 vs later:')
for a in ACTS:
    v0=[e[3]-e[2] for e in ent if e[0]==a and e[1]==0]; v1=[e[3]-e[2] for e in ent if e[0]==a and e[1]>0]
    print('   %-12s snap0 n=%2d mean %+7.2f | later n=%2d mean %+7.2f'%(a,len(v0),np.mean(v0) if v0 else float('nan'),len(v1),np.mean(v1) if v1 else float('nan')))
# 2) actual online margins: descriptive
print('\nactual online relative margin m_t: mean %.3f sd %.3f'%(m_act.mean(),m_act.std()))
print('corr(m_t, -(ref_c-ref_r)) = %.3f ; corr(m_t, time) = %.3f'%(np.corrcoef(m_act,-refd)[0,1],np.corrcoef(m_act,np.arange(119))[0,1]))
pi_d=m_act+refd     # actual online pi_c - pi_r
print('actual online (pi_c - pi_r): corr with (ref_c-ref_r) = %.3f'%np.corrcoef(pi_d,refd)[0,1])
print('final (fin_c-fin_r) vs (ref_c-ref_r): corr %.3f ; slope %.3f'%(np.corrcoef(find,refd)[0,1],np.polyfit(refd,find,1)[0]))
# 3) state-blind models, stepped once per pair in the real order
def simulate(kind,opt,lr,b0=None,adam_state=None):
    """kind='shift': log pi(c|s)=ref_c(s)+b_c -> margin=b_c-b_r (zero start).
       kind='abs'  : log pi(c|s)=b_c (same in every state) -> margin=(b_c-b_r)-(ref_c(s)-ref_r(s)).
       opt in {'sgd','sign','adam'}"""
    b=np.zeros(8) if b0 is None else b0.copy()
    m1=np.zeros(8); v=np.zeros(8); t=0
    pred=[]
    for i in range(119):
        d=b[c[i]]-b[r_[i]]
        m=d if kind=='shift' else d-refd[i]
        pred.append(m)
        g=np.zeros(8); s=beta/(1+math.exp(beta*m))   # -dL/dm
        g[c[i]]-=s; g[r_[i]]+=s                      # dL/db
        if opt=='sgd': b-=lr*g
        elif opt=='sign': b-=lr*np.sign(g)
        else:
            t+=1; m1=0.9*m1+0.1*g; v=0.999*v+0.001*g*g
            b-=lr*(m1/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
    return np.array(pred),b
# actual per-command mean change and final-table command means as targets
act_shift=np.array([np.mean([e[3]-e[2] for e in ent if e[0]==a]) for a in ACTS])
T=np.load(OUT+'/final_table.npy'); fin_cmd_mean=T.mean(0)
ref_cmd_mean=np.array([np.mean([e[2] for e in ent if e[0]==a]) for a in ACTS])
print('\nstate-blind model variants (r_margin = corr with the 119 saved online margins; r_shift = corr with the 8 actual per-command mean changes)')
for kind in ['shift','abs']:
    for opt,lrs in [('sgd',[0.3,1,3,10,30,100]),('sign',[0.03,0.1,0.3,1.0]),('adam',[0.03,0.1,0.3,1.0])]:
        best=None
        for lr in lrs:
            b0=None if kind=='shift' else ref_cmd_mean
            p,b=simulate(kind,opt,lr,b0)
            rm=np.corrcoef(p,m_act)[0,1] if np.std(p)>0 else float('nan')
            shift=b if kind=='shift' else b-ref_cmd_mean
            rs=np.corrcoef(shift,act_shift)[0,1]
            rmse=math.sqrt(np.mean((p-m_act)**2))
            if best is None or (rm==rm and rm>best[1]): best=(lr,rm,rs,rmse,shift)
            print('  %-5s %-4s lr=%-6g r_margin %.3f r_shift %.3f rmse %.2f'%(kind,opt,lr,rm,rs,rmse))
# 4) the trivial no-learning state-blind benchmarks for the online margin
print('\nbenchmarks for the online margin:')
print('  zero model (no movement from SFT): rmse %.2f'%math.sqrt(np.mean(m_act**2)))
p_abs0=-(refd-(ref_cmd_mean[c]-ref_cmd_mean[r_]))
print('  frozen state-blind policy at SFT command means, margin=(bbar_c-bbar_r)-(ref_c-ref_r): r %.3f rmse %.2f'%(np.corrcoef(p_abs0,m_act)[0,1],math.sqrt(np.mean((p_abs0-m_act)**2))))
p_absF=(fin_cmd_mean[c]-fin_cmd_mean[r_])-refd
print('  frozen state-blind policy at FINAL command means: r %.3f rmse %.2f'%(np.corrcoef(p_absF,m_act)[0,1],math.sqrt(np.mean((p_absF-m_act)**2))))
# how do online margins evolve: first 20 / later
for lo,hi in [(0,20),(20,40),(40,80),(80,119)]:
    print('  steps %3d-%3d: mean |m| %.2f ; mean m %.2f ; corr(m,-(refdiff)) %.3f'%(lo,hi,np.abs(m_act[lo:hi]).mean(),m_act[lo:hi].mean(),np.corrcoef(m_act[lo:hi],-refd[lo:hi])[0,1]))
# linear regression of online pi-difference on ref difference by phase -> how much state-specific SFT structure survives
for lo,hi in [(0,20),(20,40),(40,80),(80,119)]:
    sl=np.polyfit(refd[lo:hi],pi_d[lo:hi],1)[0]
    print('  steps %3d-%3d: slope of online (pi_c-pi_r) on (ref_c-ref_r) = %.3f (1 = SFT structure intact, 0 = erased)'%(lo,hi,sl))
print('  final table on the 119 pairs: slope %.3f'%np.polyfit(refd,find,1)[0])
