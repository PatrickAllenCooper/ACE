import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
for a in ARMS:
    R = [r for r in rows if r['arm'] == a]
    c = collections.Counter()
    for r in R:
        m = r['meta']
        if m is None: c['no numeric metadata (LLM worker record)'] += 1; continue
        if 'planning_metadata' in m:
            pm = m['planning_metadata']; c['planning_metadata keys: ' + ','.join(sorted(pm.keys()))] += 1
            if 'risk_reduction_per_cost' in pm: c['n scored actions=%d' % len(pm['risk_reduction_per_cost'])] += 1; c['declared draws=%s fantasy=%s' % (m['declared_planning_draws'], pm['fantasy_responses'] // max(1, len(pm['risk_reduction_per_cost'])))] += 1
            if 'scores' in pm: c['n scored actions=%d' % len(pm['scores'])] += 1; c['planning_draws/actions=%s' % (pm['planning_draws'] // len(pm['scores']))] += 1
        else:
            c['ranker keys: ' + ','.join(sorted(m.keys()))] += 1; c['n scores=%d' % len(m['scores'])] += 1
    print(a, len(R)); [print('     ', k, v) for k, v in c.items()]
