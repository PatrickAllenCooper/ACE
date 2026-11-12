"""Headline check: greedy generations saved for the SFT and final (DPO) checkpoints on the training prompts."""
import json, os, glob
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
byid={r['rid']:r for r in R}
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
def lab(r): return 'stop' if r['sup'] is None else r['sup']
def parse(c):
    d=json.loads(c); return 'stop' if d.get('stop') else d['action_id']
gen={}
for stage in ['sft','final']:
    g={}
    files=glob.glob(FIT+'/%s-generation-*.json'%stage)
    extra=[]
    for f in files:
        d=json.load(open(f))
        assert d['stage']==stage
        if d['ordinal']>=160: extra.append(d); continue
        assert d['record_id'] not in g
        g[d['record_id']]=d
    print(stage,'generation files',len(files),'distinct records',len(g),'extra ordinals',[ (e['ordinal'],e['record_id'],e['completion']) for e in extra])
    for e in extra:
        print('  repeat matches first pass:',g[e['record_id']]['completion']==e['completion'], g[e['record_id']]['sequence_ids']==e['sequence_ids'])
    # order = roster?
    ordmap={d['ordinal']:d['record_id'] for d in g.values()}
    print('  ordinal order == roster order:',[ordmap[i] for i in range(160)]==[r['rid'] for r in R])
    gen[stage]=g
    nt=[r for r in R if not r['terminal']]; t=[r for r in R if r['terminal']]
    c=Counter(parse(g[r['rid']]['completion']) for r in nt)
    print('  NONTERMINAL generations:',dict(c))
    print('  TERMINAL generations:',dict(Counter(parse(g[r['rid']]['completion']) for r in t)), 'legal_commands at terminal',Counter(g[r['rid']]['legal_commands'] for r in t))
    print('  legal_commands at nonterminal',Counter(g[r['rid']]['legal_commands'] for r in nt))
    eq=sum(1 for r in nt if parse(g[r['rid']]['completion'])==lab(r))
    print('  equals teacher label (nonterminal): %d/128 ; all-160: %d/160'%(eq,sum(1 for r in R if parse(g[r['rid']]['completion'])==lab(r))))
    # by snapshot
    for s in range(4):
        cc=Counter(parse(g[r['rid']]['completion']) for r in nt if r['snapshot']==s)
        print('    snap',s,dict(cc), 'match',sum(1 for r in nt if r['snapshot']==s and parse(g[r['rid']]['completion'])==lab(r)))
    # where is it not set_x_neg?
    for r in nt:
        a=parse(g[r['rid']]['completion'])
        if a!='set_x_neg' and stage=='final':
            print('    non-set_x_neg:',r['rid'],a,'label',lab(r),'teacher',r['teacher_method'],'history tail',r['history_actions'][-4:])
    # match by teacher
    for m in ['uniform_random','menu_coverage','local_information','target_risk']:
        rr=[r for r in nt if r['teacher_method']==m]
        print('    teacher %-18s match %d/%d'%(m,sum(1 for r in rr if parse(g[r['rid']]['completion'])==lab(r)),len(rr)))
    # token lengths of prompts
    plen={rid:len(d['sequence_ids']) for rid,d in g.items()}
    gen[stage+'_seq']=plen
# constant baselines
nt=[r for r in R if not r['terminal']]
for a in ACTS: print('constant %-12s matches %d/128'%(a,sum(1 for r in nt if lab(r)==a)))
# SFT vs final agreement
sg,fg=gen['sft'],gen['final']
print('SFT and final generate the same command on %d/128 nonterminal'%sum(1 for r in nt if sg[r['rid']]['completion']==fg[r['rid']]['completion']))
print('crosstab sft->final',Counter((parse(sg[r['rid']]['completion']),parse(fg[r['rid']]['completion'])) for r in nt))
# characterise where SFT says observe_all
print('SFT observe_all cases by snapshot',Counter(r['snapshot'] for r in nt if parse(sg[r['rid']]['completion'])=='observe_all'),'by teacher',Counter(r['teacher_method'] for r in nt if parse(sg[r['rid']]['completion'])=='observe_all'))
print('SFT observe_all cases: label',Counter(lab(r) for r in nt if parse(sg[r['rid']]['completion'])=='observe_all'))
print('SFT observe_all cases: last history action',Counter(r['history_actions'][-1] for r in nt if parse(sg[r['rid']]['completion'])=='observe_all'))
print('SFT set_x_pos cases',[(r['rid'],lab(r)) for r in nt if parse(sg[r['rid']]['completion'])=='set_x_pos'])
# does SFT output depend on anything visible? count of observe_all in history, graph
def graph(r):
    p=r['parents']; key=(tuple(p['X']),tuple(p['Y']),tuple(p['Z']))
    return {((),('X',),('Y',)):'chain',((),('X',),('X',)):'fork',((),(),('X','Y')):'collider'}.get(key,str(key))
print('SFT gen by graph',Counter((graph(r),parse(sg[r['rid']]['completion'])) for r in nt))
print('final gen by graph',Counter((graph(r),parse(fg[r['rid']]['completion'])) for r in nt))
print('SFT gen by cost menu',Counter((('het' if r['menu_costs']['set_y_neg']==2 else 'uni'),parse(sg[r['rid']]['completion'])) for r in nt))
# sequence lengths (prompt + completion tokens)
import statistics
seqs=[len(d['sequence_ids']) for d in sg.values()]
print('sft generation sequence_ids length min/max',min(seqs),max(seqs))
