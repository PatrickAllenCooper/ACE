import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
ini = curves[curves.step == 0].groupby('w').normalized_mse.first()
wide = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')[POL]
t = meta[['graph_stratum', 'regime', 'cost_stratum']].copy()
t['init_err'] = ini
t['at_target_init'] = ini <= THRESH
for p in POL:
    t[SHORT[p]] = wide[p]
t['best8'] = wide.min(axis=1)
t['n_at_best'] = (wide.eq(wide.min(axis=1), axis=0)).sum(axis=1)
t['argbest'] = wide.apply(lambda r: ','.join(SHORT[p] for p in POL if r[p] == r.min()) if (r == r.min()).sum() < 8 else 'all', axis=1)
print(t.to_string(float_format=lambda x: '%.5f' % x))
t.to_csv('persystem.csv')
print()
print('systems at target after initialization:', int(t.at_target_init.sum()), sorted(t.index[t.at_target_init]))
print('all 8 policies at cost 8 on these:', bool((wide[t.at_target_init] == 8).all().all()))
print('any policy at cost 8 on other systems:', bool((wide[~t.at_target_init] == 8).any().any()))
for k in ['graph_stratum', 'regime', 'cost_stratum']:
    print(pd.crosstab(t[k], t.at_target_init).to_string()); print()
print(pd.crosstab([meta.prior, meta.noise], t.at_target_init).to_string())
print()
print(pd.crosstab([t.graph_stratum], [t.regime], values=t.init_err, aggfunc=lambda s: ' / '.join('%.4f' % v for v in s)).to_string())
print()
print('init error summary: min %.5f median %.5f mean %.5f max %.5f' % (ini.min(), ini.median(), ini.mean(), ini.max()))
print('init error by regime (mean, median, n<=0.04):')
print(t.groupby('regime').init_err.agg(['mean', 'median', 'min', 'max', lambda s: (s <= THRESH).sum()]).to_string())
print(t.groupby('graph_stratum').init_err.agg(['mean', 'median', 'min', 'max', lambda s: (s <= THRESH).sum()]).to_string())
print(t.groupby('cost_stratum').init_err.agg(['mean', 'median', 'min', 'max', lambda s: (s <= THRESH).sum()]).to_string())
print()
# thresholds: how many systems at target at init for alternative thresholds
for th in [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.16]:
    print('threshold %.3f: systems already at target after init = %d' % (th, (ini <= th).sum()))
print(sorted(np.round(ini.values, 5)))
