import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
P = CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'
d = json.load(open(P))
ex = d['exposure']
print('counts', ex['counts'])
r0 = ex['records'][0]
print({k: (v if k != 'pairs' else v[:3]) for k, v in r0.items()})
raw = json.loads(d['raw_records_utf8'][r0['record_id']])
def show(o, depth=0, maxdepth=3, prefix=''):
    if isinstance(o, dict):
        for i,(k, v) in enumerate(o.items()):
            if i > 30: print(prefix + '... (%d keys)' % len(o)); break
            if isinstance(v, list):
                print(prefix + f"{k}: list[{len(v)}]" + (f" first={repr(v[0])[:200]}" if v and not isinstance(v[0], (dict, list)) else ''))
                if v and isinstance(v[0], (dict, list)) and depth < maxdepth: show(v[0], depth+1, maxdepth, prefix + '    ')
            elif isinstance(v, dict):
                print(prefix + f"{k}: dict[{len(v)}]")
                if depth < maxdepth: show(v, depth+1, maxdepth, prefix + '    ')
            else: print(prefix + f"{k}: {repr(v)[:300]}")
show({k: v for k, v in raw.items() if k != 'envelope'}, maxdepth=4)
print('envelope state keys', list(raw['envelope']['state'].keys()), 'remaining', raw['envelope']['state']['remaining_units'], 'requests', raw['envelope']['state']['requests'])
print(ex['numeric_schedule'][0], ex['numeric_schedule'][160], ex['schedule'][160])
