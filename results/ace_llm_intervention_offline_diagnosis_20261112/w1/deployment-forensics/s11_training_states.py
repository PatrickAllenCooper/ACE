"""Training states embedded in cpu-phase/numeric-training-plan.json
(raw_records_utf8): where in an episode were the 160 supervised snapshots taken,
what did the teacher label, and what do the STOP examples look like."""
import json, collections, os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/cpu-phase/numeric-training-plan.json'
p = json.load(open(P))
recs = {k: json.loads(v) for k, v in p['raw_records_utf8'].items()}
ex = {r['record_id']: r for r in p['exposure']['records']}
rows = []
for rid, r in recs.items():
    st = r['envelope']['state']
    snap = int(rid.rsplit('-', 1)[1])
    costs = {a['action_id']: a['cost_units'] for a in st['menu']}
    graph = tuple((n['name'], tuple(n['parents'])) for n in st['graph'])
    gname = {(('X', ()), ('Y', ('X',)), ('Z', ('Y',))): 'chain', (('X', ()), ('Y', ('X',)), ('Z', ('X',))): 'fork', (('X', ()), ('Y', ()), ('Z', ('X', 'Y'))): 'collider'}.get(graph, str(graph))
    hist = [h['action_id'] for h in st['history']]
    rows.append({'rid': rid, 'world': r['world_id'], 'snap': snap, 'requests': st['requests'], 'remaining': st['remaining_units'], 'spent': st['spent_units'],
                 'terminal': st['terminal'], 'label': r['supervised_action_id'] if r.get('supervised_action_id') else 'STOP', 'graph': gname,
                 'cost': 'heterogeneous' if costs.get('set_y_neg', 1) == 2 else 'uniform', 'noise': st['scm'][0]['noise_variance'],
                 'hist': hist, 'n_pref': len(r['preferences']), 'ties': sum(q['tie'] for q in r['preferences']), 'env': r['envelope'],
                 'prefs': r['preferences'], 'keys': sorted(r.keys())})
print('record keys:', rows[0]['keys'])
print('worlds', len({r['world'] for r in rows}), 'graphs', collections.Counter(r['graph'] for r in rows if r['snap'] == 0), 'cost', collections.Counter(r['cost'] for r in rows if r['snap'] == 0), 'noise', collections.Counter(r['noise'] for r in rows if r['snap'] == 0))
for s in range(5):
    R = [r for r in rows if r['snap'] == s]
    print('snapshot %d: requests %s remaining %s terminal %s labels %s ties/pairs %d/%d' % (s, dict(collections.Counter(r['requests'] for r in R)), dict(sorted(collections.Counter(r['remaining'] for r in R).items())), dict(collections.Counter(r['terminal'] for r in R)), dict(collections.Counter(r['label'] for r in R).most_common()), sum(r['ties'] for r in R), sum(r['n_pref'] for r in R)))
print('\nhistory action composition of training snapshots (who generated the behaviour before the snapshot?):')
SH = {'observe_all': 'O', 'set_x_neg': 'N', 'set_x_zero': 'Z', 'set_x_pos': 'P', 'set_y_neg': 'a', 'set_y_zero': 'b', 'set_y_pos': 'c', 'set_xy': 'J'}
for r in rows[:15]:
    print('  ', r['rid'], r['graph'], r['cost'], 'req', r['requests'], 'rem', r['remaining'], 'label', r['label'], 'hist', ''.join(SH[h] for h in r['hist']))
allh = collections.Counter(h for r in rows for h in r['hist'][8:])
print('post-initialization history actions across all 160 training snapshots:', dict(allh.most_common()))
print('\nlabel by graph (non-stop):')
for g in ['chain', 'fork', 'collider']:
    print('  ', g, dict(collections.Counter(r['label'] for r in rows if r['graph'] == g and r['label'] != 'STOP').most_common()))
print('label by cost menu (non-stop):')
for c in ['uniform', 'heterogeneous']:
    print('  ', c, dict(collections.Counter(r['label'] for r in rows if r['cost'] == c and r['label'] != 'STOP').most_common()))
print('label by requests (history length):')
for q in sorted({r['requests'] for r in rows}):
    print('   requests=%2d' % q, dict(collections.Counter(r['label'] for r in rows if r['requests'] == q).most_common()))
pickle.dump(rows, open(HERE + '/out/train_states.pkl', 'wb'))
