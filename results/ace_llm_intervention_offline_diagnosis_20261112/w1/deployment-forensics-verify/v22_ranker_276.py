import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle
import numpy as np
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb')); ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
rk = [r for r in rows if r['arm'] == 'learned_ranker' and r['d'] >= 12]
full = [r for r in rk if len(r['meta']['action_ids']) == 9]
print('ranker decisions d>=12:', len(rk), 'full-menu:', len(full))
for k in ['ref', 0, 1, 2, 3, 4]:
    def S(r): return {a: v[0] for a, v in (ref[r['envelope_sha256']]['ref'] if k == 'ref' else ref[r['envelope_sha256']]['rep256'][k]).items()}
    ag = 0
    for r in full:
        ids, sc = r['meta']['action_ids'], r['meta']['scores']
        rc = sc[ids.index('set_x_neg')] - sc[ids.index('set_x_pos')]; tc = S(r)['set_x_neg'] - S(r)['set_x_pos']
        ag += np.sign(rc) == np.sign(tc)
    print(k, 'sign agreement on full-menu d>=12:', int(ag), 'of', len(full))
