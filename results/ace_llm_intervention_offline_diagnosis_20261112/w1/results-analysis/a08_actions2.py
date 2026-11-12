"""State dependence of the strong policies after removing the mirror-image sign choice,
and a test of a one-feature rule for the sign of X interventions."""
import numpy as np, pandas as pd
from collections import Counter
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
acq = curves[curves.step > 0].copy()
seqs = {(w, p): list(r.sort_values('step').action_id.values) for (w, p), r in acq.groupby(['w', 'policy_id'])}
CAN = {'observe_all': 'obs', 'set_x_neg': 'Xext', 'set_x_pos': 'Xext', 'set_x_zero': 'x0',
       'set_y_neg': 'Yext', 'set_y_pos': 'Yext', 'set_y_zero': 'y0', 'set_xy': 'xy'}
cseqs = {k: [CAN[a] for a in v] for k, v in seqs.items()}

def agreement(S, p, keys, loo, maxstep=24):
    hits = tot = 0
    for cell, ws in meta.groupby(keys).groups.items():
        for s in range(maxstep):
            acts = [(w, S[(w, p)][s]) for w in ws if len(S[(w, p)]) > s]
            if len(acts) < 2: continue
            if not loo:
                hits += Counter(a for _, a in acts).most_common(1)[0][1]; tot += len(acts)
            else:
                for w, a in acts:
                    c = Counter(b for v, b in acts if v != w); top = max(c.values())
                    hits += (sorted(k for k, v in c.items() if v == top)[0] == a); tot += 1
    return hits / tot
rows = []
for p in POL:
    rows.append({'policy': SHORT[p],
                 'raw_modal_graphcost': agreement(seqs, p, ['graph_stratum', 'cost_stratum'], False),
                 'raw_loo_graphcost': agreement(seqs, p, ['graph_stratum', 'cost_stratum'], True),
                 'signfree_modal_graphcost': agreement(cseqs, p, ['graph_stratum', 'cost_stratum'], False),
                 'signfree_loo_graphcost': agreement(cseqs, p, ['graph_stratum', 'cost_stratum'], True),
                 'signfree_loo_graph': agreement(cseqs, p, ['graph_stratum'], True),
                 'signfree_loo_all': agreement(cseqs, p, ['one'], True) if 'one' in meta else np.nan})
meta['one'] = 1
for r, p in zip(rows, POL):
    r['signfree_loo_all'] = agreement(cseqs, p, ['one'], True)
    r['raw_loo_all'] = agreement(seqs, p, ['one'], True)
print(pd.DataFrame(rows).set_index('policy').round(3).to_string())

# sign of X interventions per run: share x- among {x-, x+}
print('\n=== within-run consistency of the X sign (share of the run-majority sign among x-/x+ choices)')
for p in ['target_risk', 'local_information', 'learned_ranker', 'uniform_random', 'menu_coverage']:
    maj = tot = 0; pure = 0; nruns = 0
    for w in range(24):
        c = Counter(a for a in seqs[(w, p)] if a in ('set_x_neg', 'set_x_pos'))
        n = sum(c.values())
        if n == 0: continue
        nruns += 1; maj += max(c.values()); tot += n; pure += (len(c) == 1)
    print('  %-17s majority-sign share %.3f over %d X-extreme actions ; runs using a single sign: %d of %d' % (p, maj / tot, tot, pure, nruns))

# one-feature rule: estimated natural mean of X from initial predictions: xbar = (p0 - p2)/(p3 - p2)
ini = curves[curves.step == 0].drop_duplicates('w').set_index('w')
xbar = (ini.prediction_0 - ini.prediction_2) / (ini.prediction_3 - ini.prediction_2)
slope = (ini.prediction_3 - ini.prediction_1) / 2
print('\n=== X-sign rule: majority X sign of the run vs sign of the posterior mean of X after initialization')
print('(xbar recovered from the saved initial predictions as (q_obs - q_do(X=0)) / (q_do(X=1) - q_do(X=0)); an inference from saved predictions)')
tab = []
for w in range(24):
    row = {'w': w, 'graph': meta.graph_stratum[w], 'cost': meta.cost_stratum[w], 'regime': meta.regime[w], 'xbar_hat': xbar[w], 'slope_hat': slope[w]}
    for p in ['target_risk', 'local_information', 'learned_ranker']:
        c = Counter(a for a in seqs[(w, p)] if a in ('set_x_neg', 'set_x_pos'))
        row[SHORT[p] + '_x-'] = c['set_x_neg']; row[SHORT[p] + '_x+'] = c['set_x_pos']
        first = next((a for a in seqs[(w, p)] if a in ('set_x_neg', 'set_x_pos')), None)
        row[SHORT[p] + '_first'] = {'set_x_neg': '-', 'set_x_pos': '+', None: '.'}[first]
    tab.append(row)
T = pd.DataFrame(tab).set_index('w')
print(T.to_string(float_format=lambda x: '%.3f' % x))
for p in ['target_risk', 'local_info', 'ranker']:
    f = T[p + '_first']
    valid = f != '.'
    far = np.where(T.xbar_hat > 0, '-', '+')        # extreme farther from the estimated natural mean of X
    print('  %-12s first X-extreme action is on the side farther from xbar_hat in %d of %d runs ; x- chosen first in %d ; xbar_hat>0 in %d of 24 systems'
          % (p, (f[valid] == far[valid.values]).sum(), valid.sum(), (f[valid] == '-').sum(), (T.xbar_hat > 0).sum()))
    maj = np.where(T[p + '_x-'] >= T[p + '_x+'], '-', '+')
    print('               run-majority sign on the far side in %d of %d' % ((maj[valid.values] == far[valid.values]).sum(), valid.sum()))

# how much of target_risk / local_information is 'observe_all', and how much is X-only
print('\n=== class shares (sign-free) by policy')
cl = pd.DataFrame({SHORT[p]: pd.Series(Counter(a for w in range(24) for a in cseqs[(w, p)])) for p in POL}).fillna(0).astype(int)
print(cl.to_string()); print((cl / cl.sum() * 100).round(1).to_string())
# share by cost stratum for Y-involving actions
for p in ['target_risk', 'local_information']:
    for c in ['uniform', 'heterogeneous']:
        k = Counter(a for w in meta.index[meta.cost_stratum == c] for a in cseqs[(w, p)])
        n = sum(k.values())
        print('  %-17s %-13s Y-involving share %.3f ; obs share %.3f ; Xext share %.3f (n=%d)' % (p, c, (k['Yext'] + k['y0'] + k['xy']) / n, k['obs'] / n, k['Xext'] / n, n))
