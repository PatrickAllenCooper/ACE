import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb')); ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
d0 = {r['world']: r for r in rows if r['d'] == 0 and r['arm'] == 'target_risk'}
allr = {(r['arm'], r['world']): r for r in rows if r['d'] == 0}
for k in ['ref', 0, 1, 2, 3, 4]:
    opt = {}; below = 0; gaps = []
    for w in range(24):
        sc = ref[d0[w]['envelope_sha256']]['ref'] if k == 'ref' else ref[d0[w]['envelope_sha256']]['rep256'][k]
        order = sorted(sc, key=lambda a: (-sc[a][0], a)); opt[w] = order[0]
        gap = sc[order[0]][0] - sc[order[1]][0]; gaps.append(gap)
        below += gap < 2 * (sc[order[0]][2] ** 2 + sc[order[1]][2] ** 2) ** 0.5
    hits = {a: sum(allr[(a, w)]['action'] == opt[w] for w in range(24)) for a in ARMS}
    print(k, {SHORT[a]: v for a, v in collections.Counter(opt.values()).most_common()}, '| gap<2se in', below, '| median gap %.5f' % np.median(gaps), '| hits', {a[:6]: v for a, v in hits.items()})
