"""Check A/B/C: supervised target-command distribution, by teacher, snapshot, graph, cost menu; majority baselines."""
import json, os, collections
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
def lab(r): return 'stop' if r['sup'] is None else r['sup']
# consistency checks across the three files
bad=0
for r in R:
    comp=json.loads(r['sup_completion'])
    l2='stop' if comp.get('stop') else comp['action_id']
    td=r['teacher_decision']
    l3='stop' if td.get('stop') else td.get('action_id')
    if not (lab(r)==l2==l3): bad+=1; print('MISMATCH',r['rid'],lab(r),l2,l3)
    if r['teacher_method']!=r['sel_teacher_method']: print('teacher method mismatch',r['rid'])
print('label mismatches across record/language/teacher files:',bad)
c=Counter(lab(r) for r in R)
print('ALL 160:',{k:c[k] for k in ACTS+['stop']}, 'sum',sum(c.values()))
nt=[r for r in R if not r['terminal']]
t=[r for r in R if r['terminal']]
cn=Counter(lab(r) for r in nt)
print('NONTERMINAL 128:',{k:cn[k] for k in ACTS+['stop']}, 'sum',sum(cn.values()))
print('TERMINAL 32:',Counter(lab(r) for r in t), 'remaining units',Counter(r['remaining'] for r in t))
print('stop labels at remaining>0:',sum(1 for r in R if lab(r)=='stop' and r['remaining']>0), '; nonstop labels at terminal:',sum(1 for r in t if lab(r)!='stop'))
print('share set_x_neg of 160: %.4f ; of 128 nonterminal: %.4f'%(c['set_x_neg']/160,cn['set_x_neg']/128))
print('share stop of 160: %.4f'%(c['stop']/160))
print('second/third nonterminal: set_x_pos %d (%.3f), observe_all %d (%.3f)'%(cn['set_x_pos'],cn['set_x_pos']/128,cn['observe_all'],cn['observe_all']/128))
# by teacher
print('\nBy teacher (nonterminal):')
for m in ['uniform_random','menu_coverage','local_information','target_risk']:
    cc=Counter(lab(r) for r in nt if r['teacher_method']==m)
    print(' ',m,sum(cc.values()),{k:cc[k] for k in ACTS if cc[k]})
print('By teacher (terminal):',Counter(r['teacher_method'] for r in t))
# teacher assignment structure
print('\nTeacher by snapshot index:')
for s in range(5):
    print(' snap',s,Counter(r['teacher_method'] for r in R if r['snapshot']==s))
print('teachers per world:',Counter(tuple(r['teacher_method'] for r in sorted([x for x in R if x['world_id']==w],key=lambda x:x['snapshot'])) for w in sorted(set(r['world_id'] for r in R))))
# by snapshot
print('\nBy snapshot (labels):')
for s in range(5):
    cc=Counter(lab(r) for r in R if r['snapshot']==s)
    print(' snap',s,{k:cc[k] for k in ACTS+['stop'] if cc[k]}, 'remaining',Counter(r['remaining'] for r in R if r['snapshot']==s), 'spent',sorted(Counter(r['spent'] for r in R if r['snapshot']==s).items()))
c13=Counter(lab(r) for r in R if r['snapshot'] in (1,2,3))
print(' snaps 1-3 combined:',{k:c13[k] for k in ACTS if c13[k]},'n',sum(c13.values()))
# menu coverage at snapshot 0
mc0=[r for r in R if r['teacher_method']=='menu_coverage' and r['snapshot']==0]
print('\nmenu_coverage at snapshot 0: n=%d labels=%s'%(len(mc0),Counter(lab(r) for r in mc0)))
for s in range(4):
    for m in ['uniform_random','menu_coverage','local_information','target_risk']:
        rr=[r for r in R if r['teacher_method']==m and r['snapshot']==s]
        print('  snap',s,m,len(rr),dict(Counter(lab(r) for r in rr)))
# graph + cost cells
def graph(r):
    p=r['parents']
    key=(tuple(p['X']),tuple(p['Y']),tuple(p['Z']))
    return {((),('X',),('Y',)):'chain',((),('X',),('X',)):'fork',((),(),('X','Y')):'collider'}.get(key,str(key))
def costm(r):
    return 'het' if r['menu_costs']['set_y_neg']==2 else 'uni'
print('\ngraphs:',Counter(graph(r) for r in R),' cost menus:',Counter(costm(r) for r in R))
print('target:',Counter(r['target'] for r in R))
cells=defaultdict(list)
for r in R: cells[(graph(r),costm(r))].append(r)
for k,v in sorted(cells.items()):
    cc=Counter(lab(r) for r in v if not r['terminal'])
    print(' cell',k,'n=%d worlds=%d'%(len(v),len(set(r['world_id'] for r in v))),{a:cc[a] for a in ACTS if cc[a]})
# majority accuracy
print('\nMajority baseline: overall %d/160 ; nonterminal %d/128'%(max(c.values()),max(cn.values())))
ins=sum(max(Counter(lab(r) for r in v).values()) for v in cells.values())
print('within graph x cost cell in-sample majority (all 160): %d/160'%ins)
ins_nt=sum(max(Counter(lab(r) for r in v if not r['terminal']).values()) for v in cells.values())
print('within graph x cost cell in-sample majority (nonterminal): %d/128'%ins_nt)
# leave-one-world-out within cell, nonterminal; ties broken several ways
def lowo(rows,cellfn,tiebreak):
    hit=0
    for w in sorted(set(r['world_id'] for r in rows)):
        test=[r for r in rows if r['world_id']==w]
        for r in test:
            train=[x for x in rows if x['world_id']!=w and cellfn(x)==cellfn(r)]
            cc=Counter(lab(x) for x in train)
            if not cc: continue
            mx=max(cc.values()); cands=[a for a in ACTS+['stop'] if cc[a]==mx]
            if tiebreak=='first': pred=cands[0]
            elif tiebreak=='last': pred=cands[-1]
            elif tiebreak=='frac':
                hit+= (lab(r) in cands)/len(cands); continue
            hit+= pred==lab(r)
    return hit
for tb in ['first','last','frac']:
    print('LOWO nonterminal within graph x cost cell (tiebreak %s): %.2f/128'%(tb,lowo(nt,lambda r:(graph(r),costm(r)),tb)))
    print('LOWO nonterminal global majority (tiebreak %s): %.2f/128'%(tb,lowo(nt,lambda r:0,tb)))
    print('LOWO all-160 within graph x cost cell (tiebreak %s): %.2f/160'%(tb,lowo(R,lambda r:(graph(r),costm(r)),tb)))
# finer cells: + snapshot index, + teacher
for name,fn in [('snapshot',lambda r:r['snapshot']),('teacher',lambda r:r['teacher_method']),('teacher x snapshot',lambda r:(r['teacher_method'],r['snapshot'])),('graph',graph),('cost',costm),('graph x cost x snapshot',lambda r:(graph(r),costm(r),r['snapshot']))]:
    cl=defaultdict(list)
    for r in nt: cl[fn(r)].append(r)
    print(' in-sample majority by %s: %d/128 ; LOWO frac %.2f'%(name,sum(max(Counter(lab(r) for r in v).values()) for v in cl.values()),lowo(nt,fn,'frac')))
