"""First-decision analysis from SAVED 32-draw target-risk scores. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
rows = pickle.load(open(SCR + '/cpu_rows.pkl', 'rb')) + pickle.load(open(SCR + '/llm_rows.pkl', 'rb'))
print('total decisions', len(rows), dict(collections.Counter(r['arm'] for r in rows)))
camp = json.load(open(CAMP + '/campaign.json'))
strata = [s['strata'] for s in camp['sources']]
# unique states
shas = collections.Counter(r['envelope_sha256'] for r in rows)
print('unique envelope states', len(shas))
multi = {k: v for k, v in shas.items() if v > 1}
print('states shared by >1 decision:', len(multi), 'multiplicities', collections.Counter(multi.values()))
shared_ds = collections.Counter()
for r in rows:
    if shas[r['envelope_sha256']] > 1: shared_ds[r['d']] += 1
print('decision indices of shared-state rows', dict(shared_ds))
# decision-0 check
d0 = {(r['arm'], r['world']): r for r in rows if r['d'] == 0}
print('d0 rows', len(d0))
for w in range(24):
    s = {d0[(a, w)]['envelope_sha256'] for a in ARMS}
    assert len(s) == 1, (w, s)
print('all 8 arms share one envelope at d=0 in all 24 worlds: True')
# first decision TR optimum
opt = {}
tab = []
for w in range(24):
    m = d0[('target_risk', w)]['meta']['planning_metadata']
    sc = m['risk_reduction_per_cost']; se = m['monte_carlo_standard_errors']
    order = sorted(sc, key=lambda a: (-sc[a], a))
    best, second = order[0], order[1]
    assert best == d0[('target_risk', w)]['action']
    opt[w] = best
    gap = sc[best] - sc[second]
    # s.e. of the difference: combine s.e. of best and second (independent streams)
    se_diff = (se[best]**2 + se[second]**2) ** 0.5
    tab.append(dict(w=w, best=best, second=second, gap=gap, rel=gap/sc[best], se_best=se[best], se_second=se[second], se_diff=se_diff,
                    xneg=sc['set_x_neg'], xneg_rank=order.index('set_x_neg') + 1,
                    xneg_gap=sc[best] - sc['set_x_neg'], xneg_se_diff=(se[best]**2 + se['set_x_neg']**2) ** 0.5, scores=sc, se=se))
print('TR optimum at d0:', {SHORT[k]: v for k, v in collections.Counter(opt.values()).most_common()})
for t in tab:
    print(t['w'], strata[t['w']]['regime'], strata[t['w']]['graph_stratum'], strata[t['w']]['cost_stratum'], 'best', SHORT[t['best']], 'second', SHORT[t['second']],
          'gap %.5f rel %.3f se_best %.5f se_2nd %.5f se_diff %.5f' % (t['gap'], t['rel'], t['se_best'], t['se_second'], t['se_diff']), 'x- rank', t['xneg_rank'],
          'x- gap/se_diff %.2f' % (t['xneg_gap'] / t['xneg_se_diff'] if t['xneg_gap'] > 0 else 0.0))
gaps = np.array([t['gap'] for t in tab]); rels = np.array([t['rel'] for t in tab])
print('gap median %.5f mean %.5f; rel median %.4f; median gap / median best %.4f' % (np.median(gaps), gaps.mean(), np.median(rels), np.median(gaps) / np.median([t['scores'][t['best']] for t in tab])))
# "below 2 MC s.e." under alternative definitions
print('gap < 2*se_diff (combined):', sum(t['gap'] < 2 * t['se_diff'] for t in tab))
print('gap < 2*se_best:', sum(t['gap'] < 2 * t['se_best'] for t in tab))
print('gap < 2*max(se):', sum(t['gap'] < 2 * max(t['se_best'], t['se_second']) for t in tab))
print('gap < 2*(se_best+se_second):', sum(t['gap'] < 2 * (t['se_best'] + t['se_second']) for t in tab))
print('x- within 2 se_diff of best (incl. being best):', sum(t['xneg_gap'] <= 2 * t['xneg_se_diff'] for t in tab))
print('x- within 2 se_best of best:', sum(t['xneg_gap'] <= 2 * t['se'][t['best']] for t in tab))
print('x- within 2*(se sum):', sum(t['xneg_gap'] <= 2 * (t['se'][t['best']] + t['se']['set_x_neg']) for t in tab))
print('x- rank distribution', dict(collections.Counter(t['xneg_rank'] for t in tab)))
# agreement with optimum per arm and regret under saved scores
print('\narm | hits | mean abs regret (saved 32-draw scores) | mean relative regret (saved)')
for a in ARMS:
    hits = 0; regs = []; rr = []
    acts = collections.Counter()
    for w in range(24):
        act = d0[(a, w)]['action']; acts[act] += 1
        sc = tab[w]['scores']
        hits += act == opt[w]
        regs.append(sc[opt[w]] - sc[act]); rr.append((sc[opt[w]] - sc[act]) / sc[opt[w]])
    print('%-18s %2d  %.5f  %.3f  actions %s' % (a, hits, np.mean(regs), np.mean(rr), {SHORT[k]: v for k, v in acts.most_common()}))
# best constant action at d0
print('\nconstant-action policy at d0: hits, mean regret')
for act in ACTIONS:
    print(SHORT[act], sum(opt[w] == act for w in range(24)), '%.5f' % np.mean([tab[w]['scores'][opt[w]] - tab[w]['scores'][act] for w in range(24)]),
          'rel %.3f' % np.mean([(tab[w]['scores'][opt[w]] - tab[w]['scores'][act]) / tab[w]['scores'][opt[w]] for w in range(24)]))
# by stratum
for key in ['graph_stratum', 'regime', 'cost_stratum']:
    cc = collections.defaultdict(collections.Counter)
    for w in range(24): cc[strata[w][key]][SHORT[opt[w]]] += 1
    print(key, {k: dict(v) for k, v in cc.items()})
pickle.dump(rows, open(SCR + '/all_rows.pkl', 'wb'))
