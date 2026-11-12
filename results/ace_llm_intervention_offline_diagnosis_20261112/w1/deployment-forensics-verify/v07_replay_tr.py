"""Replay every saved target-risk decision (560) with the verifier's own sequential implementation and saved seeds."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
from vtr import *
import pickle, time
rows = [r for r in pickle.load(open(SCR + '/all_rows.pkl', 'rb')) if r['arm'] == 'target_risk']
print(len(rows))
t0 = time.time()
maxd = dict(before=0.0, score=0.0, after=0.0, se=0.0)
argmax_same = 0; nact = 0
worst = None
for i, r in enumerate(rows):
    meta = r['meta']; pm = meta['planning_metadata']
    seed = int(meta['seed_decimal'])
    before, out = seq_scores(r['envelope'], seed, 32)
    maxd['before'] = max(maxd['before'], abs(before - pm['risk_before']))
    assert set(out) == set(pm['risk_reduction_per_cost']), (r['world'], r['d'])
    for a, (s, af, se) in out.items():
        nact += 1
        ds = abs(s - pm['risk_reduction_per_cost'][a])
        if ds > maxd['score']: maxd['score'] = ds; worst = (r['world'], r['d'], a, s, pm['risk_reduction_per_cost'][a])
        maxd['after'] = max(maxd['after'], abs(af - pm['risk_after'][a]))
        maxd['se'] = max(maxd['se'], abs(se - pm['monte_carlo_standard_errors'][a]))
    best = min(out, key=lambda a: (-out[a][0], a))
    argmax_same += best == r['action']
print('decisions replayed', len(rows), 'action-scores compared', nact, 'elapsed %.1fs' % (time.time() - t0))
print('max abs diff', maxd, 'worst', worst)
print('argmax equals saved choice', argmax_same, 'of', len(rows))
