"""Validate the vectorised reference against saved 32-draw values (z-scores should be ~N(0,1))."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
from vtr import *
import pickle, time
rows = [r for r in pickle.load(open(SCR + '/all_rows.pkl', 'rb')) if r['arm'] == 'target_risk']
rng = np.random.default_rng(20261112)
sel = rows[::7]
zs = []
t0 = time.time()
for r in sel:
    pm = r['meta']['planning_metadata']
    before, out = vec_scores(r['envelope'], 16384, rng)
    assert abs(before - pm['risk_before']) < 1e-15
    for a, (s, af, se) in out.items():
        se32 = pm['monte_carlo_standard_errors'][a]
        tot = (se32 ** 2 + se ** 2) ** 0.5
        if tot > 0:
            zs.append((pm['risk_reduction_per_cost'][a] - s) / tot)
zs = np.array(zs)
print('states', len(sel), 'scores', len(zs), 'time %.1fs (%.3fs per state at 16384 draws)' % (time.time() - t0, (time.time() - t0) / len(sel)))
print('z mean %.3f sd %.3f; |z|>2: %.3f; |z|>3: %.4f; max|z| %.2f' % (zs.mean(), zs.std(), (abs(zs) > 2).mean(), (abs(zs) > 3).mean(), abs(zs).max()))
