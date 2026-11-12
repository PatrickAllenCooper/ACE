"""Check D: composition of the strict preference pairs and of the 119 pairs used for DPO."""
import json, os, glob, math
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
PREP=TG+'/terminal01/remote/cpu/preparation/session/worker/artifacts'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
byid={r['rid']:r for r in R}
nt=[r for r in R if not r['terminal']]
tot=sum(r['n_prefs'] for r in R); strict=sum(r['n_strict'] for r in R)
print('comparisons',tot,'strict',strict,'ties',tot-strict,'records with >=1 strict',sum(1 for r in R if r['n_strict']>0),'nonterminal without strict',sum(1 for r in nt if r['n_strict']==0))
print('strict per record distribution',sorted(Counter(r['n_strict'] for r in nt).items()))
# consistency of record.json preferences with feedback terminal pairs
mism=0
for r in nt:
    fp=r['fb']['pairs']
    # map (left,right)->tie/chosen
    m={(p['left'],p['right']):p for p in fp}
    for s in r['strict']:
        p=m.get((s['left'],s['right']))
        if p is None or p['tie'] or p['chosen']!=s['chosen']: mism+=1
    if sum(1 for p in fp if not p['tie'])!=r['n_strict']: mism+=1
print('record.json vs feedback/terminal.json strict mismatches:',mism)
# (2) exposure.json
exp=json.load(open(PREP+'/exposure.json'))
print('exposure counts',exp['counts'])
exp_sel={}
for er in exp['records']:
    sel=[p for p in er['pairs'] if p['status']=='selected']
    assert len(sel)<=1
    if sel: exp_sel[er['record_id']]=sel[0]
# (3) training step start files
steps=[]
for f in sorted(glob.glob(FIT+'/training/step-*.start.json')):
    steps.append(json.load(open(f)))
dpo_steps=[s for s in steps if s['stage']=='dpo']; sft_steps=[s for s in steps if s['stage']=='sft']
print('training steps',len(steps),'sft',len(sft_steps),'dpo',len(dpo_steps))
print('sft order == roster order:',[s['record_id'] for s in sft_steps]==[r['rid'] for r in R], '; indices contiguous:',[s['index'] for s in steps]==list(range(len(steps))))
print('dpo order follows roster order (subset):',[s['record_id'] for s in dpo_steps]==[r['rid'] for r in R if r['n_strict']>0])
# compare the three routes
agree=0
sel_pairs=[]
for s in dpo_steps:
    r=byid[s['record_id']]
    mine=r['my_selected_strict_index']
    e=exp_sel[s['record_id']]
    ok=(mine==s['pair_index']==e['pair_index'])
    agree+=ok
    sp=r['strict'][s['pair_index']]
    # language.json completions for that strict index
    ch=json.loads(r['pref_completions'][s['pair_index']][0])['action_id']; rj=json.loads(r['pref_completions'][s['pair_index']][1])['action_id']
    assert ch==sp['chosen'] and rj==sp['rejected'], (ch,rj,sp)
    assert json.loads(e['chosen_command'])['action_id']==ch and json.loads(e['rejected_command'])['action_id']==rj
    sel_pairs.append((s['index'],r['rid'],ch,rj))
print('selected pair agrees across (my hash recomputation, exposure.json, step start files): %d/%d'%(agree,len(dpo_steps)))
json.dump(sel_pairs,open(OUT+'/selected_pairs.json','w'))
def table(pairs,title):
    pc=Counter(c for c,_ in pairs); rc=Counter(r for _,r in pairs)
    print('\n'+title+' (n=%d)'%len(pairs))
    print('  %-12s %9s %9s %6s'%('action','preferred','rejected','net'))
    for a in ACTS: print('  %-12s %9d %9d %+6d'%(a,pc[a],rc[a],pc[a]-rc[a]))
    return pc,rc
allp=[(s['chosen'],s['rejected']) for r in nt for s in r['strict']]
pc_all,rc_all=table(allp,'ALL strict pairs')
selp=[(c,rj) for _,_,c,rj in sel_pairs]
pc,rc=table(selp,'The DPO-selected pairs')
print('  max preferred share in selected: %s %.3f ; max rejected share: %s %.3f'%(pc.most_common(1)[0][0],pc.most_common(1)[0][1]/len(selp),rc.most_common(1)[0][0],rc.most_common(1)[0][1]/len(selp)))
print('  pairs involving set_x_neg at all: %d of %d'%(sum(1 for c,rj in selp if 'set_x_neg' in (c,rj)),len(selp)))
# token-level tallies in the selected pairs
def toks(a):
    if a=='observe_all': return ('observe','_all',None)
    if a=='set_xy': return ('set','_xy',None)
    _,v,s=a.split('_'); return ('set','_'+v,'_'+s)
for pos,name in [(0,'first'),(1,'second'),(2,'third')]:
    pcT=Counter(toks(c)[pos] for c,_ in selp); rcT=Counter(toks(rj)[pos] for _,rj in selp)
    print('  token %s: preferred %s | rejected %s'%(name,dict(pcT),dict(rcT)))
# sign tests on token tallies
from math import comb
def binom_two_sided(k,n):
    p=sum(comb(n,i) for i in range(0,n+1) if comb(n,i)<=comb(n,k))/2**n
    return p
for tok,posn in [('_neg',2),('_x',1),('_pos',2),('_y',1),('set',0),('observe',0)]:
    a=sum(1 for c,rj in selp if toks(c)[posn]==tok and toks(rj)[posn]!=tok)
    b=sum(1 for c,rj in selp if toks(rj)[posn]==tok and toks(c)[posn]!=tok)
    tp=sum(1 for c,rj in selp if toks(c)[posn]==tok); tr=sum(1 for c,rj in selp if toks(rj)[posn]==tok)
    print('  token %-8s preferred-total %3d rejected-total %3d | discordant pref-only %3d rej-only %3d sign-test p=%.3f'%(tok,tp,tr,a,b,binom_two_sided(a,a+b) if a+b else float('nan')))
# sign tests per command in selected and all
print('\nper-command two-sided sign test (preferred vs rejected appearances)')
for a in ACTS:
    print('  %-12s selected %2d/%2d p=%.3f | all %3d/%3d p=%.4f'%(a,pc[a],rc[a],binom_two_sided(pc[a],pc[a]+rc[a]),pc_all[a],rc_all[a],binom_two_sided(pc_all[a],pc_all[a]+rc_all[a])))
# pair matrix for the selected
print('\nselected (preferred -> rejected) pair counts:')
m=Counter(selp)
print('  %-12s'%'pref\\rej'+' '.join('%7s'%a[-6:] for a in ACTS))
for a in ACTS: print('  %-12s'%a+' '.join('%7d'%m[(a,b)] for b in ACTS))
# strict share by snapshot
print('\nstrict share by snapshot:')
for s in range(4):
    rr=[r for r in nt if r['snapshot']==s]
    print('  snap %d: %d/%d = %.4f ; records with no strict %d ; selected pairs from this snapshot %d'%(s,sum(r['n_strict'] for r in rr),28*len(rr),sum(r['n_strict'] for r in rr)/(28*len(rr)),sum(1 for r in rr if r['n_strict']==0),sum(1 for r in rr if r['n_strict']>0)))
# feedback cost
print('\nfeedback cost: per-record totals',Counter( sum(4*c for c in r['fb']['candidate_costs'].values())+32 for r in nt))
