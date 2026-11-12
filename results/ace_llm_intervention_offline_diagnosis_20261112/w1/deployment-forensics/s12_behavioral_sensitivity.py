"""Behavioural state sensitivity for every arm (no score vectors needed):
action entropy and mutual information between the chosen command and public
state features, plus SFT budget-level structure."""
import pickle, collections, json, os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
def H(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c)
def MI(pairs):
    a = collections.Counter(x for x, _ in pairs); b = collections.Counter(y for _, y in pairs); ab = collections.Counter(pairs)
    return H(a) + H(b) - H(ab)
def mi_corrected(pairs, nperm=300, seed=0):
    """MI minus the mean MI under permutation of the feature (bias correction), and permutation p."""
    rng = np.random.default_rng(seed)
    obs = MI(pairs)
    xs = [x for x, _ in pairs]; ys = [y for _, y in pairs]
    null = []
    for _ in range(nperm):
        perm = rng.permutation(len(ys))
        null.append(MI(list(zip(xs, [ys[i] for i in perm]))))
    null = np.array(null)
    return obs, obs - null.mean(), (1 + (null >= obs).sum()) / (nperm + 1)
def feats(r):
    st = r['envelope']['state']; w = worlds[r['world_idx']]
    return {'graph': w['graph'], 'cost': w['cost'], 'regime': w['regime'], 'noise': w['noise'], 'world': r['world_idx'],
            'step': r['decision_index'], 'remaining': st['remaining_units'], 'step_tercile': min(2, r['decision_index'] // 8),
            'last_action': st['history'][-1]['action_id'] if st['requests'] > 8 else 'INIT',
            'signX': st['scm'][0]['parameter_mean'][0] > 0}
print('Action entropy H(A) in bits (max log2(9)=3.17) and mutual information I(A; feature) in bits.')
print('MI shown as raw / permutation-bias-corrected (p). 300 permutations.')
print('%-18s %5s %6s | %-22s %-22s %-22s %-22s %-22s' % ('arm', 'n', 'H(A)', 'graph', 'cost menu', 'regime', 'step tercile', 'remaining units'))
res = {}
for arm in pids:
    R = [r for r in rows if r['arm'] == arm]
    F = [feats(r) for r in R]
    acts = [r['action'] for r in R]
    h = H(collections.Counter(acts))
    out = []
    for k in ['graph', 'cost', 'regime', 'step_tercile', 'remaining']:
        if h == 0:
            out.append('0 / 0 (n/a)'); continue
        o, c, pv = mi_corrected(list(zip(acts, [f[k] for f in F])))
        out.append('%.3f / %.3f (p=%.3f)' % (o, c, pv))
        res[(arm, k)] = (o, c, pv)
    print('%-18s %5d %6.3f | ' % (arm, len(R), h) + ' '.join('%-22s' % s for s in out))
# number of distinct actions per world, and switches
print('\nPer-arm within-episode variability: mean number of distinct commands per world, mean number of action switches per world')
for arm in pids:
    nd = []; sw = []
    for w in range(24):
        s = [r['action'] for r in sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == w), key=lambda r: r['decision_index'])]
        nd.append(len(set(s))); sw.append(sum(a != b for a, b in zip(s, s[1:])))
    print('  %-18s distinct %.2f (min %d max %d)  switches %.2f (min %d max %d)' % (arm, np.mean(nd), min(nd), max(nd), np.mean(sw), min(sw), max(sw)))
# across-world variability of the action sequence: are two worlds' sequences identical?
print('\nNumber of distinct complete action sequences across the 24 worlds:')
for arm in pids:
    seqs = collections.Counter(tuple(r['action'] for r in sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == w), key=lambda r: r['decision_index'])) for w in range(24))
    print('  %-18s %d distinct sequences (largest group %d worlds)' % (arm, len(seqs), max(seqs.values())))
# SFT budget-level structure
S = [r for r in rows if r['arm'] == 'sft_llm']
print('\nSFT: command by remaining_units (training snapshots were taken at remaining 24, 18(17), 12(11), 6(5), 0):')
by = collections.defaultdict(collections.Counter)
for r in S: by[r['envelope']['state']['remaining_units']][r['action']] += 1
for k in sorted(by, reverse=True):
    c = by[k]; n = sum(c.values())
    print('   remaining=%2d n=%2d  x_neg=%2d  other_acq=%2d  stop=%2d   %s' % (k, n, c['set_x_neg'], n - c['set_x_neg'] - c['STOP'], c['STOP'], {a: v for a, v in c.items() if a != 'set_x_neg'}))
near = lambda k: k in (11, 12, 13)
a = sum(1 for r in S if near(r['envelope']['state']['remaining_units']) and r['action'] not in ('set_x_neg', 'STOP'))
b = sum(1 for r in S if near(r['envelope']['state']['remaining_units']) and r['action'] == 'set_x_neg')
c = sum(1 for r in S if not near(r['envelope']['state']['remaining_units']) and r['action'] not in ('set_x_neg', 'STOP'))
d = sum(1 for r in S if not near(r['envelope']['state']['remaining_units']) and r['action'] == 'set_x_neg')
print('   non-x_neg acquisitions at remaining in {11,12,13}: %d of %d (%.1f%%) ; elsewhere: %d of %d (%.1f%%) ; Fisher exact p=%.2e' % (a, a + b, 100 * a / (a + b), c, c + d, 100 * c / (c + d), stats.fisher_exact([[a, b], [c, d]])[1]))
# graph dependence of SFT's set_y_neg / observe
print('\nSFT: worlds (of 8 per graph) with >=1 set_y_neg / >=1 observe_all / >=1 set_x_pos:')
for g in ['chain', 'fork', 'collider']:
    ws = [w for w in range(24) if worlds[w]['graph'] == g]
    print('   %-8s y_neg %d/8  observe %d/8  x_pos %d/8' % (g, sum(any(r['action'] == 'set_y_neg' for r in S if r['world_idx'] == w) for w in ws), sum(any(r['action'] == 'observe_all' for r in S if r['world_idx'] == w) for w in ws), sum(any(r['action'] == 'set_x_pos' for r in S if r['world_idx'] == w) for w in ws)))
t = [[8, 0], [0, 8]]
print('   Fisher exact, set_y_neg present: collider 8/8 vs fork 0/8: p=%.5f ; collider 8/8 vs chain 3/8: p=%.4f' % (stats.fisher_exact(t)[1], stats.fisher_exact([[8, 0], [3, 5]])[1]))
print('   cost of set_y_neg when chosen:', dict(collections.Counter((worlds[r['world_idx']]['cost'], r['charged_units']) for r in S if r['action'] == 'set_y_neg')))
