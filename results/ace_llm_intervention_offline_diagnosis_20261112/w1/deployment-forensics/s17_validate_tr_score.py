"""Is the target-risk score a meaningful decision-level endpoint?  Compare the
model-based predicted risk (risk_before) and predicted one-step reduction with
the REALISED private NMSE and NMSE change saved in the 192 cost reports."""
import pickle, collections, os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TR = pickle.load(open(HERE + '/out/tr_scores_256.pkl', 'rb'))
reports = pickle.load(open(HERE + '/out/reports.pkl', 'rb'))
recs = []
for arm in pids:
    for w in range(24):
        curve = reports[(arm, w)]['curve']
        seq = sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == w), key=lambda r: r['decision_index'])
        k = 0
        for r in seq:
            t = TR[r['envelope_sha256']]
            nm_before = curve[k]['normalized_mse']
            if r['action'] == 'STOP':
                continue
            nm_after = curve[k + 1]['normalized_mse']
            assert curve[k + 1]['action_id'] == r['action']
            cost = worlds[w]['menu_costs'][r['action']]
            recs.append({'arm': arm, 'w': w, 'd': r['decision_index'], 'risk_before': t['risk_before'], 'nmse_before': nm_before, 'nmse_after': nm_after,
                         'pred_red': t['risk_before'] - t['risk_after'][r['action']], 'real_red': nm_before - nm_after, 'cost': cost,
                         'regime': worlds[w]['regime'], 'graph': worlds[w]['graph'], 'action': r['action']})
            k += 1
print('acquisitions matched to curve points:', len(recs))
rb = np.array([x['risk_before'] for x in recs]); nb = np.array([x['nmse_before'] for x in recs])
pr = np.array([x['pred_red'] for x in recs]); rr = np.array([x['real_red'] for x in recs])
print('\nCalibration of the SCM posterior risk against private NMSE at the same state (all arms, all decisions):')
print('   mean risk_before %.5f ; mean realised NMSE %.5f ; ratio NMSE/risk %.3f ; Spearman rho %.3f ; Pearson r(log,log) %.3f' % (rb.mean(), nb.mean(), nb.mean() / rb.mean(), stats.spearmanr(rb, nb)[0], stats.pearsonr(np.log(rb), np.log(np.maximum(nb, 1e-12)))[0]))
for key in ('regime', 'graph'):
    for v in sorted({x[key] for x in recs}):
        sel = np.array([x[key] == v for x in recs])
        print('   %-7s %-13s n=%4d mean risk %.5f mean NMSE %.5f ratio %.2f' % (key, v, sel.sum(), rb[sel].mean(), nb[sel].mean(), nb[sel].mean() / rb[sel].mean()))
# per-world at the shared initial state
print('   at the shared initial state (24 worlds): risk_before vs NMSE0: Spearman rho %.3f ; mean risk %.4f mean NMSE %.4f' % (
    stats.spearmanr([TR[next(r for r in rows if r['arm'] == 'target_risk' and r['world_idx'] == w and r['decision_index'] == 0)['envelope_sha256']]['risk_before'] for w in range(24)],
                    [reports[('target_risk', w)]['curve'][0]['normalized_mse'] for w in range(24)])[0],
    np.mean([TR[next(r for r in rows if r['arm'] == 'target_risk' and r['world_idx'] == w and r['decision_index'] == 0)['envelope_sha256']]['risk_before'] for w in range(24)]),
    np.mean([reports[('target_risk', w)]['curve'][0]['normalized_mse'] for w in range(24)])))
print('\nPredicted one-step risk reduction (model) vs realised one-step NMSE reduction (single noisy response):')
print('   Spearman rho %.3f ; Pearson r %.3f ; mean predicted %.5f ; mean realised %.5f ; realised<0 (NMSE got worse) in %.1f%% of acquisitions' % (stats.spearmanr(pr, rr)[0], stats.pearsonr(pr, rr)[0], pr.mean(), rr.mean(), 100 * (rr < 0).mean()))
order = np.argsort(pr); bins = np.array_split(order, 10)
print('   deciles of predicted reduction: mean predicted -> mean realised (n)')
for b in bins:
    print('      %.5f -> %+.5f (n=%d, realised>0 in %.0f%%)' % (pr[b].mean(), rr[b].mean(), len(b), 100 * (rr[b] > 0).mean()))
# sum of predicted reductions over an episode vs realised total reduction, per arm x world
print('\nEpisode level: sum of predicted reductions vs total realised NMSE reduction (192 cells): ')
cells = collections.defaultdict(lambda: [0.0, 0.0])
for x in recs:
    cells[(x['arm'], x['w'])][0] += x['pred_red']; cells[(x['arm'], x['w'])][1] += x['real_red']
a = np.array([v[0] for v in cells.values()]); b = np.array([v[1] for v in cells.values()])
print('   Spearman rho %.3f ; Pearson r %.3f ; mean predicted %.4f mean realised %.4f' % (stats.spearmanr(a, b)[0], stats.pearsonr(a, b)[0], a.mean(), b.mean()))
# does lower decision-level regret go with better outcome across arms?
reg = pickle.load(open(HERE + '/out/regret.pkl', 'rb'))['Q']
print('\nArm level: mean relative regret (own trajectory) vs the report endpoints:')
import json
POL = {}
import csv
for row in csv.DictReader(open('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/policies.csv')):
    POL[row['policy_id']] = row
print('   policies.csv columns:', list(next(iter(POL.values())).keys()))
xs = []; ys = []; zs = []
for arm in pids:
    rel = np.mean([q['rel'] for (a, w, d), q in reg.items() if a == arm])
    cap = np.mean([8 + reports[(arm, w)]['capped_cost_units'] for w in range(24)])
    fin = np.mean([reports[(arm, w)]['final_error'] for w in range(24)])
    xs.append(rel); ys.append(cap); zs.append(fin)
    print('   %-18s mean rel regret %.3f | mean capped total cost %.3f | mean final NMSE %.5f' % (arm, rel, cap, fin))
print('   across the 8 arms: Spearman(rel regret, capped cost) = %.3f ; Spearman(rel regret, final NMSE) = %.3f' % (stats.spearmanr(xs, ys)[0], stats.spearmanr(xs, zs)[0]))
