"""Target-risk arm: its SAVED 32-draw per-action scores at all 560 of its own
decisions. How peaked are decisions, how does the optimum move with the state."""
import pickle, collections, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
T = [r for r in rows if r['arm'] == 'target_risk']
A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
print('target_risk decisions', len(T), 'feasible-count distribution', collections.Counter(len(r['meta']['planning_metadata']['risk_reduction_per_cost']) for r in T))
g12 = []; rel = []; z = []; xrank = []; best = []; neg = 0; negbest = 0
for r in T:
    pm = r['meta']['planning_metadata']; s = pm['risk_reduction_per_cost']; se = pm['monte_carlo_standard_errors']
    order = sorted(s, key=lambda a: (-s[a], a))
    assert order[0] == r['action']
    g = s[order[0]] - s[order[1]]
    g12.append(g); rel.append(g / s[order[0]] if s[order[0]] > 0 else np.nan); z.append(g / np.hypot(se[order[0]], se[order[1]]))
    xrank.append(order.index('set_x_neg') + 1); best.append(order[0])
    neg += sum(v < 0 for v in s.values()); negbest += s[order[0]] <= 0
g12 = np.array(g12); z = np.array(z); rel = np.array(rel)
print('gap best-second (saved 32-draw scores): median %.5f mean %.5f ; relative to best score: median %.3f' % (np.median(g12), g12.mean(), np.nanmedian(rel)))
print('gap in combined MC s.e. units: median %.2f ; <1 s.e.: %d (%.1f%%) ; <2 s.e.: %d (%.1f%%) ; >=3 s.e.: %d (%.1f%%)' % (np.median(z), (z < 1).sum(), 100 * (z < 1).mean(), (z < 2).sum(), 100 * (z < 2).mean(), (z >= 3).sum(), 100 * (z >= 3).mean()))
print('negative estimated gains among candidate scores: %d ; decisions whose best score <= 0: %d' % (neg, negbest))
print('rank of set_x_neg at target_risk own states:', sorted(collections.Counter(xrank).items()))
print('\nby decision index: chosen action counts, median z-gap')
bd = collections.defaultdict(list)
for r, zz in zip(T, z): bd[r['decision_index']].append((r['action'], zz))
for k in sorted(bd):
    c = collections.Counter(a for a, _ in bd[k])
    print('   d=%2d n=%2d medz=%5.2f %s' % (k, len(bd[k]), np.median([zz for _, zz in bd[k]]), dict(c.most_common())))
rb = np.array([r['meta']['planning_metadata']['risk_before'] for r in T])
print('\nrisk_before along target_risk episodes: mean by decision index 0,4,8,12,16,20:', [round(float(np.mean([r['meta']['planning_metadata']['risk_before'] for r in T if r['decision_index'] == k])), 5) for k in (0, 4, 8, 12, 16, 20)])
# local information arm saved scores
L = [r for r in rows if r['arm'] == 'local_information']
zl = []
for r in L:
    s = r['meta']['planning_metadata']['scores']; order = sorted(s, key=lambda a: (-s[a], a)); assert order[0] == r['action']
    zl.append((s[order[0]] - s[order[1]]) / s[order[0]])
print('\nlocal_information: relative gap best-second (saved scores): median %.3f ; <5%%: %d of %d' % (np.median(zl), sum(v < 0.05 for v in zl), len(zl)))
