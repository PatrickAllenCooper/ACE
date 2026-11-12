import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import collections, pickle
num = json.load(open(CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'))
raws = {k: json.loads(v) for k, v in num['raw_records_utf8'].items()}
c = collections.Counter(); n = 0
for raw in raws.values():
    for p in raw['preferences']:
        n += 1
        for side in ('left', 'right'):
            c[p[side]] += 1
print('pairs', n, 'appearances by command (None = stop):', {str(k): v for k, v in c.items()})
print('pairs containing stop:', sum(1 for raw in raws.values() for p in raw['preferences'] if p['left'] is None or p['right'] is None))
# deployment: decisions where stop would have been 'reasonable' = none; count DPO/SFT/frozen stops
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
print('deployment stops by arm:', dict(collections.Counter(r['arm'] for r in rows if r['action'] == 'STOP')))
# number of pairs per non-terminal record
print('pairs per record:', collections.Counter(len(raw['preferences']) for raw in raws.values()))
# strict pairs per snapshot level and tie rate
by = collections.defaultdict(lambda: [0, 0])
for k, raw in raws.items():
    for p in raw['preferences']:
        by[k.split('/')[-1]][0] += not p['tie']; by[k.split('/')[-1]][1] += 1
print('strict/total pairs by snapshot:', {k: tuple(v) for k, v in sorted(by.items())})
