import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
ini = curves[curves.step == 0].groupby('w').normalized_mse.first()
wide = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')[POL]
inf = ini > THRESH            # informative subset (not at target after initialization)
tr = wide['target_risk']
best8 = wide.min(axis=1)
best7 = wide.drop(columns='target_risk').min(axis=1)
numer = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker']
llm = ['frozen_llm', 'sft_llm', 'dpo_llm']
best_num = wide[numer].min(axis=1); best_llm = wide[llm].min(axis=1)
# trivial floor: 8 if at target at init, else 8 + cheapest action (1)
floor = np.where(inf, 9, 8)
def rep(mask, name):
    n = int(mask.sum())
    print('---', name, 'n =', n)
    print('  target_risk: sum %d  mean %.3f (acquisition mean %.3f)' % (tr[mask].sum(), tr[mask].mean(), tr[mask].mean() - 8))
    print('  best-of-8 envelope: sum %d  mean %.3f' % (best8[mask].sum(), best8[mask].mean()))
    print('  hindsight saving vs target_risk (target_risk - best8): sum %d  mean %.3f ; systems where some policy beats target_risk: %d ; ties with envelope: %d'
          % ((tr - best8)[mask].sum(), (tr - best8)[mask].mean(), ((tr - best8)[mask] > 0).sum(), ((tr - best8)[mask] == 0).sum()))
    print('  best-of-other-7 envelope: mean %.3f ; target_risk minus best7: mean %.3f (wins %d ties %d losses %d for target_risk)'
          % (best7[mask].mean(), (tr - best7)[mask].mean(), ((tr - best7)[mask] < 0).sum(), ((tr - best7)[mask] == 0).sum(), ((tr - best7)[mask] > 0).sum()))
    print('  trivial floor (8, or 9 if one unit-cost response is needed): mean %.3f ; target_risk minus floor: sum %d mean %.3f'
          % (floor[mask.values].mean(), (tr.values - floor)[mask.values].sum(), (tr.values - floor)[mask.values].mean()))
    print('  best-of-5-numerical envelope mean %.3f ; best-of-3-LLM envelope mean %.3f ; best LLM arm strictly below best numerical on %d systems, tie %d, above %d'
          % (best_num[mask].mean(), best_llm[mask].mean(), (best_llm < best_num)[mask].sum(), (best_llm == best_num)[mask].sum(), (best_llm > best_num)[mask].sum()))
    for p in POL:
        d = (wide[p] - best8)[mask]
        print('    %-12s mean cost %.3f ; regret to envelope mean %.3f ; at envelope on %d ; strictly below target_risk on %d'
              % (SHORT[p], wide[p][mask].mean(), d.mean(), (d == 0).sum(), (wide[p] < tr)[mask].sum()))
rep(pd.Series(True, index=wide.index), 'all 24 systems')
rep(inf, 'informative subset (init error > 0.04)')
print()
d = (tr - best8)
print('per-system target_risk minus best-of-8 (informative only):')
print(pd.DataFrame({'graph': meta.graph_stratum, 'regime': meta.regime, 'cost': meta.cost_stratum, 'init_err': ini, 'target_risk': tr,
                    'best8': best8, 'saving': d,
                    'winner': wide.apply(lambda r: ','.join(SHORT[p] for p in POL if r[p] == r.min()), axis=1)})[inf].to_string(float_format=lambda x: '%.4f' % x))
print()
print('Which policies supply the 14 informative-envelope minima (ties counted for each):')
print(pd.Series({SHORT[p]: int(((wide[p] == best8) & inf).sum()) for p in POL}).to_string())
print('unique (sole) winners:', pd.Series({SHORT[p]: int(((wide[p] == best8) & inf & (wide.eq(best8, axis=0).sum(axis=1) == 1)).sum()) for p in POL}).to_dict())
# how much of the envelope's advantage over target_risk comes from few systems
print('saving distribution:', sorted(d[inf].values.tolist(), reverse=True))
# Systems where target_risk is at cap
print('target_risk capped (never attained) systems:', list(wide.index[tr == CAP]), '; systems where no policy attained:', list(wide.index[best8 == CAP]))
# expected minimum-of-8 selection effect: compare envelope with the mean of the best single policy
print('best single policy mean over 24:', wide.mean().min(), wide.mean().idxmin(), ' envelope mean:', best8.mean())
# leave-one-out style: for each policy, how often is it the unique minimum
