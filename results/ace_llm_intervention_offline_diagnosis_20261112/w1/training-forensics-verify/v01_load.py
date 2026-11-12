"""Independent loader of the raw saved training collection (read-only).
Builds a compact table of the 160 records and writes it to my scratch dir as records.json.
"""
import json, os, glob, hashlib, collections
COLL='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/session/worker/task/collection'
OUT=os.path.dirname(os.path.abspath(__file__))

def canonical(v): return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(v): return hashlib.sha256(canonical(v).encode()).hexdigest()
PAIR_DOMAIN="ace-matched-training-exposure-proposal-v1/unordered-command-pair"
def cmd(a): return canonical({"stop": True} if a is None else {"action_id": a})

worlds=sorted(glob.glob(COLL+'/world-[0-9][0-9][0-9]'))
print('world dirs',len(worlds))
records=[]
for w in worlds:
    snaps=sorted(glob.glob(w+'/snapshot-[0-9][0-9][0-9]'))
    priv=json.load(open(w+'/private-world.json'))
    for s in snaps:
        rec=json.load(open(s+'/record.json'))
        teacher=json.load(open(s+'/teacher.json'))
        sel=json.load(open(s+'/selection.json'))
        lang=json.load(open(s+'/language.json'))
        numl=json.load(open(s+'/numeric-labels.json'))
        align=json.load(open(s+'/alignment.json'))
        st=rec['envelope']['state']; task=rec['envelope']['task']
        wid=rec['world_id']; si=sel['snapshot_index']
        rid='%s/%d/snapshot-%03d'%(rec['scope'],wid,si)
        # graph type from parents
        par={n['name']:tuple(n['parents']) for n in st['graph']}
        menu={m['action_id']:m['cost_units'] for m in st['menu']}
        fb=None
        fpath=s+'/feedback/terminal.json'
        if os.path.exists(fpath):
            fb=json.load(open(fpath))['feedback']
        prefs=rec['preferences']
        strict=[]
        k=0
        for i,p in enumerate(prefs):
            if not p['tie']:
                commands=sorted((cmd(p['left']),cmd(p['right'])))
                strict.append({'pref_index':i,'strict_index':k,'left':p['left'],'right':p['right'],
                               'chosen':p['chosen'],'rejected':p['rejected'],
                               'sel_sha':digest([PAIR_DOMAIN,rid,commands]),'commands':commands})
                k+=1
        # my own recomputation of the hash-selected pair
        selected=min(strict,key=lambda r:(r['sel_sha'],tuple(r['commands']))) if strict else None
        r={'rid':rid,'world_dir':os.path.basename(w),'snap_dir':os.path.basename(s),'world_id':wid,'snapshot':si,
           'terminal':st['terminal'],'remaining':st['remaining_units'],'spent':st['spent_units'],
           'requests':st['requests'],'n_history':len(st['history']),
           'parents':par,'menu_costs':menu,'target':task['target'],
           'sup':rec['supervised_action_id'],'sup_completion':lang['supervised_completion'],
           'teacher_method':teacher['method'],'teacher_decision':teacher['decision'],'teacher_meta':teacher.get('metadata'),
           'sel_teacher_method':sel.get('teacher_method'),
           'n_prefs':len(prefs),'n_strict':len(strict),'strict':strict,
           'my_selected_strict_index':None if selected is None else selected['strict_index'],
           'pref_completions':lang['preference_completions'],
           'numeric_labels':numl,'align_action_ids':align['action_ids'],
           'prompt_len_chars':len(lang['prompt']),
           'private':priv,
           'history_actions':[h['action_id'] for h in st['history']],
           }
        if fb is not None:
            r['fb']={'candidate_costs':fb['candidate_costs'],'mean_gains_per_cost':fb['mean_gains_per_cost'],
                     'per_replicate_gains_per_cost':fb['per_replicate_gains_per_cost'],
                     'pairs':fb['pairs'],'base_losses':fb['base_losses'],'query_weights':fb['query_weights'],
                     'tie_multiplier':fb['tie_multiplier'],'minimum_gain_gap':fb['minimum_gain_gap'],
                     'replicate_count':fb['replicate_count'],'branch_losses':fb['branch_losses'],
                     'normalized_loss_reductions':fb['normalized_loss_reductions'],'gains_per_cost':fb['gains_per_cost']}
        records.append(r)
print('records',len(records))
json.dump(records,open(OUT+'/records.json','w'))
print('teacher methods',collections.Counter(r['teacher_method'] for r in records))
print('terminal',collections.Counter(r['terminal'] for r in records))
print('snapshots per world',collections.Counter(collections.Counter(r['world_id'] for r in records).values()))
print('n_prefs',collections.Counter(r['n_prefs'] for r in records))
print('private keys',list(records[0]['private'].keys()))
print(records[0]['private'])
