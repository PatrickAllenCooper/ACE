"""Compute verifier's reference target-risk scores for every unique saved decision state.
   - ref: 16384 vectorised draws (MC s.e. about 1/8 of a 256-draw estimate)
   - rep256: five independent 256-draw estimates (to quantify seed sensitivity of any 256-draw statistic)
Pure arithmetic on saved public states; no model, no environment."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
from vtr import *
import pickle, time
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
uniq = {}
for r in rows:
    uniq.setdefault(r['envelope_sha256'], r['envelope'])
print('unique states', len(uniq)); sys.stdout.flush()
res = {}
t0 = time.time()
with np.errstate(all='ignore'):
    for i, (sha, env) in enumerate(uniq.items()):
        if env['state']['terminal']:
            res[sha] = None; continue
        ss = np.random.SeedSequence([20261112, int(sha[:16], 16)])
        children = ss.spawn(6)
        before, ref = vec_scores(env, 16384, np.random.default_rng(children[0]))
        reps = []
        for k in range(5):
            b2, o2 = vec_scores(env, 256, np.random.default_rng(children[1 + k]))
            reps.append(o2)
        for o in [ref] + reps:
            for a, v in o.items():
                assert all(np.isfinite(v)), (sha, a, v)
        res[sha] = dict(before=before, ref=ref, rep256=reps)
        if i % 500 == 0:
            print(i, '%.0fs' % (time.time() - t0)); sys.stdout.flush()
pickle.dump(res, open(SCR + '/ref_scores.pkl', 'wb'))
print('done', len(res), '%.0fs' % (time.time() - t0))
