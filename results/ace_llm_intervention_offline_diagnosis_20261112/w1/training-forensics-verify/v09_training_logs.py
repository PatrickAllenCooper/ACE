"""Check H and I: per-step training logs, SFT prequential loss, DPO online loss, score tables."""
import json, os, glob, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
byid={r['rid']:r for r in R}
def lab(r): return 'stop' if r['sup'] is None else r['sup']
steps=[]
for i in range(279):
    c=json.load(open(FIT+'/training/step-%05d.complete.json'%i))
    l=json.load(open(FIT+'/training/step-%05d.loss.json'%i))
    u=json.load(open(FIT+'/training/step-%05d.update-intent.json'%i))
    assert c['index']==i and abs(c['loss']-l['loss'])<1e-12 and abs(c['gradient_norm_before_clip']-u['gradient_norm_before_clip'])<1e-9
    steps.append(c)
sft=[s for s in steps if s['stage']=='sft']; dpo=[s for s in steps if s['stage']=='dpo']
print('steps',len(steps),'sft',len(sft),'dpo',len(dpo),'rows',Counter((s['stage'],s['rows']) for s in steps))
print('all optimizer steps returned:',all(s['optimizer_step_returned'] for s in steps))
# ---------- SFT
print('\n--- SFT')
print('action_tokens by label',Counter((lab(byid[s['record_id']]),s['action_tokens']) for s in sft))
nats=np.array([s['loss']*s['action_tokens'] for s in sft])     # loss is mean per action token (incl. EOS)
term=np.array([byid[s['record_id']]['terminal'] for s in sft])
idx=np.arange(160)
print('block means of nats per NONTERMINAL example (20-step blocks):')
for b in range(8):
    m=(idx>=20*b)&(idx<20*b+20)&(~term)
    mt=(idx>=20*b)&(idx<20*b+20)&(term)
    print('  steps %3d-%3d: nonterminal n=%d mean %.3f (per-token %.3f) | terminal n=%d mean %.3f'%(20*b,20*b+19,m.sum(),nats[m].mean(),np.mean([s['loss'] for s,k in zip(sft,m) if k]),mt.sum(),nats[mt].mean()))
m2=(idx>=80)&(~term)
print('LLM prequential nats per nonterminal example, steps 80-159: %.4f (n=%d) ; steps 60-159: %.4f ; steps 0-79: %.4f'%(nats[m2].mean(),m2.sum(),nats[(idx>=60)&(~term)].mean(),nats[(idx<80)&(~term)].mean()))
print('reference points: ln 8 = %.4f ; ln 9 = %.4f ; entropy of the nonterminal label distribution = %.4f'%(math.log(8),math.log(9),-sum((c/128)*math.log(c/128) for c in Counter(lab(r) for r in R if not r['terminal']).values())))
# state-blind prequential label-frequency counter
labels=[lab(byid[s['record_id']]) for s in sft]
def counter_loss(alpha,support,cond_nonterminal):
    cnt=Counter(); out=[]
    for i,(l,t) in enumerate(zip(labels,term)):
        if not t:
            if cond_nonterminal:
                tot=sum(cnt[a] for a in ACTS)+alpha*len(support); p=(cnt[l]+alpha)/tot
            else:
                tot=sum(cnt.values())+alpha*len(support); p=(cnt[l]+alpha)/tot
            out.append((i,-math.log(p)))
        cnt[l]+=1
    return out
for alpha in [1.0,0.5,0.1]:
    for support,cn,name in [(ACTS,True,'8 actions, counts from nonterminal only'),(ACTS+['stop'],False,'9 commands, all counts'),(ACTS+['stop'],True,'9 commands support, nonterminal counts')]:
        o=counter_loss(alpha,support,cn)
        v=np.mean([x for i,x in o if i>=80]); v2=np.mean([x for i,x in o])
        print('  state-blind counter alpha=%.1f (%s): steps 80-159 %.4f ; all %.4f'%(alpha,name,v,v2))
# best in-hindsight constant distribution on steps 80-159 nonterminal
c2=Counter(l for i,(l,t) in enumerate(zip(labels,term)) if i>=80 and not t); n2=sum(c2.values())
print('  in-hindsight entropy of labels in steps 80-159 (nonterminal): %.4f (n=%d)'%(-sum((c/n2)*math.log(c/n2) for c in c2.values()),n2))
# note on comparability: LLM loss is over the FULL vocabulary incl. the 4 fixed prefix tokens, closing token and EOS
print('gradient norm before clip (SFT): min %.3f median %.3f max %.3f ; steps with norm<=1 (not clipped): %d'%(min(s['gradient_norm_before_clip'] for s in sft),np.median([s['gradient_norm_before_clip'] for s in sft]),max(s['gradient_norm_before_clip'] for s in sft),sum(1 for s in sft if s['gradient_norm_before_clip']<=1.0)))
# ---------- DPO
print('\n--- DPO')
dl=np.array([s['loss'] for s in dpo])
print('mean online DPO loss %.4f ; ln2 %.4f ; first 10 mean %.4f ; last 10 mean %.4f ; steps with loss>ln2: %d/119'%(dl.mean(),math.log(2),dl[:10].mean(),dl[-10:].mean(),int((dl>math.log(2)+1e-9).sum())))
gn=np.array([s['gradient_norm_before_clip'] for s in dpo])
print('pre-clip gradient norm: min %.3f max %.3f median %.3f ; clipped (norm>1): %d/119'%(gn.min(),gn.max(),np.median(gn),int((gn>1.0).sum())))
beta=0.1
online_margin=np.array([-math.log(math.expm1(l))/beta if l>0 else float('inf') for l in dl])   # (pi_c-ref_c)-(pi_r-ref_r)
print('online relative margins (nats): first %.4f ; mean %.3f ; min %.3f max %.3f ; negative in %d/119'%(online_margin[0],online_margin.mean(),online_margin.min(),online_margin.max(),int((online_margin<0).sum())))
ref=json.load(open(FIT+'/training/reference-scores.json'))['scores']
fin=json.load(open(FIT+'/final-before-reload.json'))
sel=json.load(open(OUT+'/selected_pairs.json'))
rows=[]
for (stepidx,rid,ch,rj),s in zip(sel,dpo):
    assert s['record_id']==rid and s['index']==stepidx
    key=json.dumps([rid,s['pair_index']],separators=(',',':'))
    rc,rr=ref[key]
    cmds=fin['commands'][rid]; sc=fin['scores'][rid]
    fc=sc[cmds.index(json.dumps({'action_id':ch},separators=(',',':')))]; fr=sc[cmds.index(json.dumps({'action_id':rj},separators=(',',':')))]
    rows.append({'rid':rid,'ch':ch,'rj':rj,'ref_c':rc,'ref_r':rr,'fin_c':fc,'fin_r':fr})
print('pairs with reference (SFT) scores ranked against the label (ref chosen < ref rejected): %d/119'%sum(1 for x in rows if x['ref_c']<x['ref_r']))
print('pairs with FINAL scores ranked against the label (final chosen < final rejected): %d/119'%sum(1 for x in rows if x['fin_c']<x['fin_r']))
print('pairs whose final RELATIVE margin (final-ref) is negative, i.e. DPO moved them the wrong way: %d/119'%sum(1 for x in rows if (x['fin_c']-x['ref_c'])-(x['fin_r']-x['ref_r'])<0))
fl=np.array([-math.log(1/(1+math.exp(-beta*((x['fin_c']-x['ref_c'])-(x['fin_r']-x['ref_r']))))) for x in rows])
print('DPO loss recomputed on the 119 trained pairs with the FINAL adapter: mean %.4f (ln2 %.4f)'%(fl.mean(),math.log(2)))
# change by command over the 238 (record, command) entries
ch_by=defaultdict(list)
for x in rows:
    ch_by[x['ch']].append((x['fin_c']-x['ref_c'],'pref')); ch_by[x['rj']].append((x['fin_r']-x['ref_r'],'rej'))
print('DPO-stage change in command log-likelihood (final - SFT reference), by command, over the 238 scored entries:')
allv=[];alll=[]
for a in sorted(ACTS,key=lambda a:-np.mean([v for v,_ in ch_by[a]])):
    v=[v for v,_ in ch_by[a]]
    print('  %-12s n=%2d mean %+7.2f sd %.2f min %+.2f max %+.2f'%(a,len(v),np.mean(v),np.std(v,ddof=1),min(v),max(v)))
    allv+=v; alll+=[a]*len(v)
allv=np.array(allv); gm=allv.mean()
ssb=sum(len(ch_by[a])*(np.mean([v for v,_ in ch_by[a]])-gm)**2 for a in ACTS); sst=((allv-gm)**2).sum()
print('  share of variance in the change explained by command identity (eta^2): %.4f'%(ssb/sst))
print('  mean change preferred %.3f ; rejected %.3f ; difference %.3f'%(np.mean([x['fin_c']-x['ref_c'] for x in rows]),np.mean([x['fin_r']-x['ref_r'] for x in rows]),np.mean([(x['fin_c']-x['ref_c'])-(x['fin_r']-x['ref_r']) for x in rows])))
# adding role (pref/rej) on top of command
import itertools
X=np.zeros((len(allv),8+1)); roles=[rl for a in sorted(ACTS,key=lambda a:-np.mean([v for v,_ in ch_by[a]])) for _,rl in ch_by[a]]
for i,(a,rl) in enumerate(zip(alll,roles)):
    X[i,ACTS.index(a)]=1; X[i,8]=1 if rl=='pref' else 0
bcoef,res,_,_=np.linalg.lstsq(X,allv,rcond=None)
pred=X@bcoef
print('  command + preferred-role regression: R^2 %.4f ; role coefficient (preferred vs rejected, same command) %+.3f nats'%(1-((allv-pred)**2).sum()/sst,bcoef[8]))
# also add record effects? state-level shift
# ---------- final table
print('\n--- final full table (128 nonterminal states x 8 action commands + stop)')
nt=[r for r in R if not r['terminal']]
T=np.zeros((128,8)); stopcol=[]
for i,r in enumerate(nt):
    cmds=fin['commands'][r['rid']]; sc=fin['scores'][r['rid']]
    for j,a in enumerate(ACTS): T[i,j]=sc[cmds.index(json.dumps({'action_id':a},separators=(',',':')))]
    stopcol.append(sc[cmds.index('{"stop":true}')])
arg=T.argmax(1)
print('argmax command by full-sequence score:',Counter(ACTS[j] for j in arg))
print('stop ever outranks the best action:',sum(1 for i in range(128) if stopcol[i]>T[i].max()))
srt=np.sort(T,1)
marg=srt[:,-1]-srt[:,-2]
print('set_x_neg is argmax in %d/128 ; margin of set_x_neg over best other command: median %.3f min %.3f'%((arg==1).sum(),np.median(T[:,1]-np.max(np.delete(T,1,1),1)),np.min(T[:,1]-np.max(np.delete(T,1,1),1))))
print('margin top1-top2 where set_x_neg wins: median %.3f'%np.median(marg[arg==1]))
# two-way decomposition
gm=T.mean(); rm=T.mean(1); cm=T.mean(0)
ss_t=((T-gm)**2).sum(); ss_r=8*((rm-gm)**2).sum(); ss_c=128*((cm-gm)**2).sum(); ss_i=ss_t-ss_r-ss_c
print('two-way decomposition of the final table: command main %.4f ; state main %.4f ; state x command %.4f'%(ss_c/ss_t,ss_r/ss_t,ss_i/ss_t))
print('command means (nats):',{a:round(float(c),2) for a,c in zip(ACTS,cm)})
# same decomposition on normalized log-probs over the 8 action commands (removes state main effect by construction)
L=T-np.log(np.exp(T).sum(1,keepdims=True))
gm=L.mean(); rm=L.mean(1); cm_=L.mean(0); ss_t=((L-gm)**2).sum(); ss_r=8*((rm-gm)**2).sum(); ss_c=128*((cm_-gm)**2).sum()
print('after renormalizing over the 8 action commands: command main %.4f ; state main %.4f ; interaction %.4f'%(ss_c/ss_t,ss_r/ss_t,(ss_t-ss_r-ss_c)/ss_t))
P=np.exp(L)
print('final policy probability (renormalized over 8 actions) of set_x_neg: median %.3f min %.3f max %.3f ; mean prob by command %s'%(np.median(P[:,1]),P[:,1].min(),P[:,1].max(),{a:round(float(x),3) for a,x in zip(ACTS,P.mean(0))}))
# does the final table's state x command interaction track the labels at all?
lab_idx=np.array([ACTS.index(lab(r)) for r in nt])
ll=L[np.arange(128),lab_idx]
print('final-policy mean log-prob of the SFT label (renormalized over 8): %.3f ; constant-policy (column means) log-prob of labels: %.3f'%(ll.mean(),(np.log(P.mean(0)))[lab_idx].mean()))
# rank of teacher label among commands
ranks=(T>=T[np.arange(128),lab_idx][:,None]).sum(1)
print('mean rank of SFT label in final table: %.2f ; label is top-1 in %d ; top-2 in %d'%(ranks.mean(),(ranks==1).sum(),(ranks<=2).sum()))
# by snapshot: residual interaction - does the runner-up vary with state?
second=np.argsort(-T,1)[:,1]
print('runner-up command distribution:',Counter(ACTS[j] for j in second))
for s in range(4):
    ii=[i for i,r in enumerate(nt) if r['snapshot']==s]
    print('  snap %d: mean final log-score by command %s'%(s,{a[-5:]:round(float(x),1) for a,x in zip(ACTS,T[ii].mean(0))}))
np.save(OUT+'/final_table.npy',T)
json.dump(rows,open(OUT+'/pair_scores.json','w'))
json.dump({'dpo_losses':dl.tolist(),'online_margin':online_margin.tolist()},open(OUT+'/dpo_online.json','w'))
