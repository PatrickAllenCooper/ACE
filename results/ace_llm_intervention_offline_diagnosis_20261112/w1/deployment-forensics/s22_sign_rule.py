"""A single readable state feature: the sign of the posterior mean of the X
intercept (scm[0].parameter_mean[0]). How well does it predict the planner's
preferred X-intervention, and did labels / learned policies encode it?"""
import pickle, collections, os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TR = pickle.load(open(HERE + '/out/tr_scores_256.pkl', 'rb'))
T = pickle.load(open(HERE + '/out/train_states.pkl', 'rb'))
TT = pickle.load(open(HERE + '/out/train_tr_256.pkl', 'rb'))
first = {(r['arm'], r['world_idx']): r for r in rows if r['decision_index'] == 0}
print('DEVELOPMENT, shared first decision (24 worlds): sign of posterior mean of X vs 256-draw TR optimum and each policy\'s choice')
tab = collections.Counter(); pol = {p: collections.Counter() for p in pids}
for w in range(24):
    r = first[('target_risk', w)]; mx = r['envelope']['state']['scm'][0]['parameter_mean'][0]; s = TR[r['envelope_sha256']]['scores']
    best = max(s, key=lambda a: s[a]); pref = 'x_neg' if s['set_x_neg'] > s['set_x_pos'] else 'x_pos'
    tab[('meanX>0' if mx > 0 else 'meanX<0', 'TR-opt=' + best)] += 1; tab[('meanX>0' if mx > 0 else 'meanX<0', 'TR prefers ' + pref)] += 1
    for p in pids: pol[p][('meanX>0' if mx > 0 else 'meanX<0', first[(p, w)]['action'])] += 1
for k in sorted(tab): print('   ', k, tab[k])
for p in pids: print('   %-18s %s' % (p, dict(sorted(pol[p].items()))))
rule = sum(1 for w in range(24) if (('set_x_neg' if first[('target_risk', w)]['envelope']['state']['scm'][0]['parameter_mean'][0] > 0 else 'set_x_pos') == max(TR[first[('target_risk', w)]['envelope_sha256']]['scores'], key=lambda a: TR[first[('target_risk', w)]['envelope_sha256']]['scores'][a])))
reg = np.mean([max(TR[first[('target_risk', w)]['envelope_sha256']]['scores'].values()) - TR[first[('target_risk', w)]['envelope_sha256']]['scores']['set_x_neg' if first[('target_risk', w)]['envelope']['state']['scm'][0]['parameter_mean'][0] > 0 else 'set_x_pos'] for w in range(24)])
rel = np.mean([(max(TR[first[('target_risk', w)]['envelope_sha256']]['scores'].values()) - TR[first[('target_risk', w)]['envelope_sha256']]['scores']['set_x_neg' if first[('target_risk', w)]['envelope']['state']['scm'][0]['parameter_mean'][0] > 0 else 'set_x_pos']) / max(TR[first[('target_risk', w)]['envelope_sha256']]['scores'].values()) for w in range(24)])
print('   one-feature rule "X mean > 0 -> set_x_neg else set_x_pos": TR-optimal in %d/24 ; mean regret %.5f ; mean relative regret %.3f  (always set_x_neg: 11/24, 0.00896, 0.256)' % (rule, reg, rel))
print('\nTRAINING snapshot 0 (32 worlds, same kind of state: 8 observations): sign of posterior mean of X vs TR optimum vs supervised label')
c1 = collections.Counter(); c2 = collections.Counter(); c3 = collections.Counter()
for r in T:
    if r['snap'] != 0: continue
    mx = r['env']['state']['scm'][0]['parameter_mean'][0]; s = TT[r['rid']]['scores']; best = max(s, key=lambda a: s[a])
    g = 'meanX>0' if mx > 0 else 'meanX<0'
    c1[(g, 'TR-opt=' + best)] += 1; c2[(g, 'label=' + r['label'])] += 1; c3[(g, 'TR prefers ' + ('x_neg' if s['set_x_neg'] > s['set_x_pos'] else 'x_pos'))] += 1
for c in (c3, c1, c2):
    for k in sorted(c): print('   ', k, c[k])
print('\nALL STATES (development, every arm, every decision with both X actions feasible): does sign(meanX) predict the TR preference between set_x_neg and set_x_pos?')
a = b = n = 0
bysteps = collections.defaultdict(lambda: [0, 0])
for r in rows:
    s = TR[r['envelope_sha256']]['scores']; mx = r['envelope']['state']['scm'][0]['parameter_mean'][0]
    ok = (s['set_x_neg'] > s['set_x_pos']) == (mx > 0); n += 1; a += ok
    k = min(r['decision_index'] // 4, 5); bysteps[k][0] += ok; bysteps[k][1] += 1
print('   agreement %d/%d (%.1f%%) ; by decision-index block of 4: %s' % (a, n, 100 * a / n, {k * 4: '%.2f' % (v[0] / v[1]) for k, v in sorted(bysteps.items())}))
print('   (after the first steps the preference is driven by which X value has already been sampled, not by the sign of the mean)')
# imbalance-aware rule: prefer the X extreme sampled less often so far (ties -> sign rule)
a2 = n2 = 0
for r in rows:
    st = r['envelope']['state']; s = TR[r['envelope_sha256']]['scores']; mx = st['scm'][0]['parameter_mean'][0]
    nn = sum(h['action_id'] in ('set_x_neg', 'set_xy') for h in st['history']); npos = sum(h['action_id'] == 'set_x_pos' for h in st['history'])
    pred_neg = (nn < npos) if nn != npos else (mx > 0)
    a2 += (s['set_x_neg'] > s['set_x_pos']) == pred_neg; n2 += 1
print('   rule "choose the X extreme sampled less often so far; if tied use the sign rule": agreement with TR preference %d/%d (%.1f%%)' % (a2, n2, 100 * a2 / n2))
