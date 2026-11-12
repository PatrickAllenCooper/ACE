import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
orders = collections.Counter(tuple(a['action_id'] for a in r['envelope']['state']['menu']) for r in rows)
print('deployment menu orders (all 4371 decision states):', len(orders)); [print('   ', v, [SHORT[x] for x in k]) for k, v in orders.items()]
num = json.load(open(CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'))
raws = {k: json.loads(v) for k, v in num['raw_records_utf8'].items()}
o2 = collections.Counter(tuple(a['action_id'] for a in raw['envelope']['state']['menu']) for raw in raws.values())
print('training menu orders (160 records):', len(o2)); [print('   ', v, [SHORT[x] for x in k]) for k, v in o2.items()]
# query order and objective text constant?
print('distinct objective strings (deploy):', len({r['envelope']['state']['objective'] for r in rows}), '| distinct domain_notes:', len({r['envelope']['state']['domain_notes'] for r in rows}),
      '| distinct task payloads:', len({canonical(r['envelope']['task']) for r in rows}))
print('distinct task payloads (train):', len({canonical(raw['envelope']['task']) for raw in raws.values()}), '| train task == deploy task:', {canonical(raw['envelope']['task']) for raw in raws.values()} == {canonical(r['envelope']['task']) for r in rows})
# training worlds: graph/cost/noise mix
g = collections.Counter()
for k, raw in raws.items():
    if k.endswith('snapshot-000'):
        st = raw['envelope']['state']
        gr = tuple((n['name'], tuple(n['parents'])) for n in st['graph'])
        kind = 'chain' if gr[2][1] == ('Y',) else 'fork' if gr[2][1] == ('X',) else 'collider'
        cost = 'hetero' if any(a['cost_units'] == 2 for a in st['menu']) else 'uniform'
        g[(kind, cost, st['scm'][0]['noise_variance'])] += 1
print('training worlds by (graph, cost, noise):', dict(g))
# teacher identity recorded?
r0 = raws[next(iter(raws))]
print('training raw record keys:', list(r0.keys()))
