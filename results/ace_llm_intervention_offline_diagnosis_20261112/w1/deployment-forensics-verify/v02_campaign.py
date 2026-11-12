import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
c = json.load(open(CAMP + '/campaign.json'))
print('policy_ids', c['policy_ids'])
print('policy_phases', c['policy_phases'])
print('strata_fields', c['strata_fields'])
import collections
cnt = collections.Counter()
for i, s in enumerate(c['sources']):
    p = s['plan']
    w = p['world']
    menu = {a['action_id']: a['cost_units'] for a in p['menu']}
    mech = w['mechanisms']
    print(i, w['world_id'], s['strata'], 'graph', [(g['name'], g['parents']) for g in w['graph']],
          'noise', p['learner']['noise_variances'], 'costs', [menu[a] for a in ACTIONS])
    cnt[(s['strata']['regime'], s['strata']['graph_stratum'], s['strata']['cost_stratum'])] += 1
print(len(cnt), set(cnt.values()))
print(json.dumps(c['sources'][0]['plan']['world'], indent=1)[:3000])
print(json.dumps(c['sources'][0]['plan']['task'], indent=1)[:3000])
print(json.dumps(c['sources'][0]['plan']['measurement_bindings'], indent=1)[:1500])
print(json.dumps(c['sources'][0]['plan']['policy_specs'][5], indent=1)[:3000])
