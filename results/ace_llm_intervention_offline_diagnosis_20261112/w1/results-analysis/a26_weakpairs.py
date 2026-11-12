import numpy as np, pandas as pd, itertools, warnings
from scipy import stats
from common import *
warnings.filterwarnings('ignore')
cells, curves, meta = load()
S = wide_steps(curves)
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
def capped(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w] <= th)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
E = {p: {'capped_cost': cw[p].values.astype(float), 'capped@0.02': capped(S[p], .02), 'log_area': np.log10(S[p][:, 1:]).mean(axis=1),
         'log_e12': np.log10(S[p][:, 4]), 'log_e16': np.log10(S[p][:, 8]), 'log_e24': np.log10(S[p][:, 16]), 'log_final': np.log10(S[p][:, -1])} for p in POL}
weak = ['uniform_random', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
rows = []
for a, b in itertools.combinations(weak, 2):
    for e in E[a]:
        d = E[a][e] - E[b][e]
        rows.append({'pair': '%s - %s' % (SHORT[a], SHORT[b]), 'endpoint': e, 'mean': d.mean(), 'sd': d.std(ddof=1), 'signflip_p': signflip_exact(d)})
R = pd.DataFrame(rows)
print('pairs among random, ranker, frozen, sft, dpo x 7 endpoints: %d tests ; smallest sign-flip p values:' % len(R))
print(R.sort_values('signflip_p').head(6).to_string(index=False, float_format=lambda x: '%.4f' % x))
print('number with p < 0.05:', (R.signflip_p < 0.05).sum(), ' p < 0.10:', (R.signflip_p < 0.10).sum())
