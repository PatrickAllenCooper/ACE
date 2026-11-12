"""Extra checks on the supervised stage: SFT reference probabilities by snapshot, prequential probability of the label by label type,
label order/recency, behavior policy of each world, and whether teacher identity is recoverable from the prompt text."""
import json, os, glob, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
COLL='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/session/worker/task/collection'
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
byid={r['rid']:r for r in R}
def lab(r): return 'stop' if r['sup'] is None else r['sup']
rows=json.load(open(OUT+'/pair_scores.json'))
# 1) SFT reference by snapshot group (geometric mean probability = exp(mean log-lik)) on the 238 saved entries
ent=[]
for x in rows:
    s=byid[x['rid']]['snapshot']
    ent.append((x['ch'],s,x['ref_c'])); ent.append((x['rj'],s,x['ref_r']))
print('SFT reference geometric-mean probability by command (n in brackets) on the saved 238 entries:')
for grp,name in [([0],'snapshot 0'),([1,2,3],'snapshots 1-3'),([1],'snap 1'),([2],'snap 2'),([3],'snap 3')]:
    out={}
    for a in ACTS:
        v=[e[2] for e in ent if e[0]==a and e[1] in grp]
        out[a]='%.3f[%d]'%(math.exp(np.mean(v)),len(v)) if v else 'NA'
    print('  %-14s %s'%(name,out))
# 2) prequential probability of the teacher label during SFT (exp(-nats)); by label and by step block
steps=[json.load(open(FIT+'/training/step-%05d.complete.json'%i)) for i in range(160)]
nats=np.array([s['loss']*s['action_tokens'] for s in steps]); labels=[lab(byid[s['record_id']]) for s in steps]; snaps=[byid[s['record_id']]['snapshot'] for s in steps]
print('\nprequential mean nats (and geometric-mean prob) of the teacher label, steps 80-159, by label:')
for a in ACTS+['stop']:
    v=[n for i,(n,l) in enumerate(zip(nats,labels)) if l==a and i>=80]
    if v: print('  %-12s n=%2d mean nats %.2f gm-prob %.3f ; share with p>0.5: %.2f'%(a,len(v),np.mean(v),math.exp(-np.mean(v)),np.mean([math.exp(-x)>0.5 for x in v])))
v=[math.exp(-n)>0.5 for i,(n,l) in enumerate(zip(nats,labels)) if l!='stop' and i>=80]
print('  share of nonterminal steps 80-159 where p(label)>0.5: %.3f (n=%d)'%(np.mean(v),len(v)))
print('prequential mean nats by snapshot, steps 80-159:',{s:round(float(np.mean([n for i,(n,sn) in enumerate(zip(nats,snaps)) if sn==s and i>=80])),2) for s in range(5)})
for s in [0,1,2,3]:
    for a in ['set_x_neg','set_x_pos','observe_all']:
        v=[n for i,(n,l,sn) in enumerate(zip(nats,labels,snaps)) if l==a and sn==s and i>=80]
        if v: print('   snap %d label %-11s n=%2d gm-prob %.3f'%(s,a,len(v),math.exp(-np.mean(v))))
# by teacher
for m in ['uniform_random','menu_coverage','local_information','target_risk']:
    v=[n for i,(n,s) in enumerate(zip(nats,steps)) if byid[s['record_id']]['teacher_method']==m and not byid[s['record_id']]['terminal'] and i>=80]
    print('  teacher %-18s steps 80-159: n=%d LLM mean nats %.2f'%(m,len(v),np.mean(v)))
# 3) label order
print('\nlabel sequence, last 40 SFT steps:',[l.replace('set_','') for l in labels[-40:]])
print('label counts in the last 20 / 40 / 80 nonterminal SFT examples:')
ntlab=[l for l in labels if l!='stop']
for k in [16,32,64]:
    print('   last %d nonterminal:'%k,dict(Counter(ntlab[-k:]).most_common()))
print('   first 64 nonterminal:',dict(Counter(ntlab[:64]).most_common()))
# 4) behavior policy per world
beh={}
for w in sorted(glob.glob(COLL+'/world-[0-9][0-9][0-9]')):
    p=json.load(open(w+'/plan.json')); beh[p['world_id']]=p['behavior_method']
print('\nbehavior_method per world:',Counter(beh.values()))
print('behavior vs snapshot-0 teacher:',Counter((beh[r['world_id']],r['teacher_method']) for r in R if r['snapshot']==0))
# acquired actions in histories by behavior method (history beyond 8 initial observations)
hist=defaultdict(Counter)
for r in R:
    if r['snapshot']==4:
        hist[beh[r['world_id']]].update(r['history_actions'][8:])
for k,v in hist.items(): print('  behavior %-18s acquired actions in full trajectories: %s'%(k,dict(v.most_common())))
print('initial 8 history actions all observe_all:',all(r['history_actions'][:8]==['observe_all']*8 for r in R))
# does the label equal the next executed behavior action when teacher==behavior?
# (next action after the snapshot state = history of the next snapshot at index n_history)
nxt={}
for r in R:
    if r['snapshot']<4:
        r2=byid['policy_train/%d/snapshot-%03d'%(r['world_id'],r['snapshot']+1)]
        nxt[r['rid']]=r2['history_actions'][r['n_history']]
same=[(r['teacher_method'],lab(r)==nxt[r['rid']]) for r in R if r['snapshot']<4 and beh[r['world_id']]==r['teacher_method']]
print('teacher==behavior states: label equals the action actually executed next:',Counter(same))
diff=[(lab(r)==nxt[r['rid']]) for r in R if r['snapshot']<4 and beh[r['world_id']]!=r['teacher_method']]
print('teacher!=behavior states: label equals executed next action in %d/%d'%(sum(diff),len(diff)))
# 5) teacher identity in the prompt?
lang=json.load(open(COLL+'/world-000/snapshot-001/language.json'))['prompt']
for kw in ['uniform_random','menu_coverage','local_information','target_risk','teacher','behavior','method','planner']:
    print('  prompt contains %-18s: %s'%(kw,kw in lang))
# 6) SFT generations vs behavior policy and SFT label
gen={}
for f in glob.glob(FIT+'/sft-generation-*.json'):
    d=json.load(open(f))
    if d['ordinal']<160: gen[d['record_id']]=json.loads(d['completion']).get('action_id','stop')
print('\nSFT generations by behavior policy of the world (nonterminal):',dict(Counter((beh[r['world_id']],gen[r['rid']]) for r in R if not r['terminal'])))
print('SFT generation equals last history action in %d/128 ; equals modal acquired action (snap>0) in %d/96'%(sum(1 for r in R if not r['terminal'] and gen[r['rid']]==r['history_actions'][-1]),sum(1 for r in R if not r['terminal'] and r['snapshot']>0 and gen[r['rid']]==Counter(r['history_actions'][8:]).most_common(1)[0][0])))
# counts of set_x_neg in history vs generation
print('teacher label equals the modal acquired action in history (snap>0): %d/96'%sum(1 for r in R if not r['terminal'] and r['snapshot']>0 and lab(r)==Counter(r['history_actions'][8:]).most_common(1)[0][0]))
