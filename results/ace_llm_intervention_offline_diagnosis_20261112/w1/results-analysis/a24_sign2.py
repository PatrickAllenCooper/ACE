import numpy as np, pandas as pd, itertools
from common import *
cells, curves, meta = load()
S = wide_steps(curves)
inf = S['dpo_llm'][:, 0] > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
ini = curves[curves.step == 0].drop_duplicates('w').set_index('w')
xbar = ((ini.prediction_0 - ini.prediction_2) / (ini.prediction_3 - ini.prediction_2)).values
near = xbar < 0
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
lf = {p: np.log10(S[p][:, -1]) for p in POL}
rows = []
for p in POL:
    if p == 'target_risk': continue
    da = aulc[p] - aulc['target_risk']; dc = (cw[p] - cw['target_risk']).values.astype(float); dl = lf[p] - lf['target_risk']
    rows.append({'policy': SHORT[p], 'area_near(n=7)': da[near].mean(), 'area_far(n=17)': da[~near].mean(), 'area_gap': da[near].mean() - da[~near].mean(),
                 'cost_near': dc[near].mean(), 'cost_far': dc[~near].mean(), 'final_near': dl[near].mean(), 'final_far': dl[~near].mean()})
print('policy minus target_risk, split by whether X=-1 is the near (low-leverage) or far side of the estimated natural X mean')
print(pd.DataFrame(rows).set_index('policy').round(3).to_string())
# exact permutation test for DPO's area gap (near vs far), all C(24,7) relabelings
da = aulc['dpo_llm'] - aulc['target_risk']
obs = da[near].mean() - da[~near].mean()
tot = da.sum(); cnt = n = 0
for comb in itertools.combinations(range(24), 7):
    s = da[list(comb)].sum()
    g = s / 7 - (tot - s) / 17
    n += 1; cnt += abs(g) >= abs(obs) - 1e-12
print('DPO log-area deficit vs target_risk: near %.3f, far %.3f, gap %.3f ; exact two-sided permutation p over %d relabelings = %.4f (post hoc, descriptive)' % (da[near].mean(), da[~near].mean(), obs, n, cnt / n))
dl = lf['dpo_llm'] - lf['target_risk']; obs = dl[near].mean() - dl[~near].mean(); tot = dl.sum(); cnt = n = 0
for comb in itertools.combinations(range(24), 7):
    s = dl[list(comb)].sum(); g = s / 7 - (tot - s) / 17; n += 1; cnt += abs(g) >= abs(obs) - 1e-12
print('DPO log-final-error deficit: gap %.3f ; exact permutation p = %.4f' % (obs, cnt / n))
# raw DPO performance (not relative to TR)
print('DPO geometric-mean final error: near %.4f, far %.4f ; target_risk: near %.4f, far %.4f ; frozen: near %.4f, far %.4f'
      % (10 ** lf['dpo_llm'][near].mean(), 10 ** lf['dpo_llm'][~near].mean(), 10 ** lf['target_risk'][near].mean(), 10 ** lf['target_risk'][~near].mean(), 10 ** lf['frozen_llm'][near].mean(), 10 ** lf['frozen_llm'][~near].mean()))
print('init error geometric mean: near %.4f, far %.4f' % (10 ** np.log10(S['dpo_llm'][near, 0]).mean(), 10 ** np.log10(S['dpo_llm'][~near, 0]).mean()))
print('near systems by graph/regime:'); print(meta[near][['graph_stratum', 'regime', 'cost_stratum']].to_string())
