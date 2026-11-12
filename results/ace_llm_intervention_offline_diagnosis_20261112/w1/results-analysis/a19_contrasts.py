"""Key scientific contrasts among learned arms. DESCRIPTIVE POST HOC, development data, uncorrected p-values."""
import numpy as np, pandas as pd, warnings
from scipy import stats
from common import *
warnings.filterwarnings('ignore')
cells, curves, meta = load()
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
def capped(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w] <= th)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
E = {p: {'capped_cost': cw[p].values.astype(float), 'capped_cost@0.02': capped(S[p], .02), 'log_area': np.log10(S[p][:, 1:]).mean(axis=1), 'log_final': np.log10(S[p][:, -1])} for p in POL}
pairs = [('dpo_llm', 'frozen_llm', 'effect of the whole training recipe'), ('sft_llm', 'frozen_llm', 'effect of supervised stage'),
         ('dpo_llm', 'sft_llm', 'effect of preference stage'), ('dpo_llm', 'learned_ranker', 'LLM vs numerical ranker, same supervision'),
         ('sft_llm', 'learned_ranker', 'supervised LLM vs numerical ranker'), ('learned_ranker', 'target_risk', 'learned numerical vs planner'),
         ('learned_ranker', 'uniform_random', 'learned numerical vs random'), ('frozen_llm', 'uniform_random', 'observe-only vs random'),
         ('menu_coverage', 'target_risk', 'fixed open-loop cycle vs planner'), ('local_information', 'target_risk', 'information vs target-risk planner'),
         ('menu_coverage', 'uniform_random', 'fixed cycle vs random')]
rows = []
for a, b, why in pairs:
    for e in ['capped_cost', 'capped_cost@0.02', 'log_area', 'log_final']:
        d = E[a][e] - E[b][e]
        nz = d[d != 0]
        rows.append({'contrast': '%s - %s' % (SHORT[a], SHORT[b]), 'endpoint': e, 'mean': d.mean(), 'sd': d.std(ddof=1), 'neg': int((d < 0).sum()), 'zero': int((d == 0).sum()), 'pos': int((d > 0).sum()),
                     'wilcoxon_p': stats.wilcoxon(nz).pvalue if len(nz) else np.nan, 'signflip_p': signflip_exact(d), 'mean_inf14': d[inf].mean(), 'sd_inf14': d[inf].std(ddof=1), 'why': why})
R = pd.DataFrame(rows)
pd.set_option('display.width', 260)
print(R.to_string(index=False, float_format=lambda x: '%.4f' % x))
R.to_csv('key_contrasts.csv', index=False)
# Holm over the seven comparisons with target_risk on the predeclared endpoint
ps = {}
for p in POL:
    if p == 'target_risk': continue
    ps[SHORT[p]] = signflip_exact(E[p]['capped_cost'] - E['target_risk']['capped_cost'])
o = sorted(ps.items(), key=lambda kv: kv[1]); m = len(o); adj = {}; run = 0
for i, (k, v) in enumerate(o):
    run = max(run, min(1, (m - i) * v)); adj[k] = run
print('\nexact sign-flip p (capped cost, policy vs target_risk) and Holm-adjusted over the 7 comparisons:')
for k, v in o: print('   %-12s raw %.4f  Holm %.4f' % (k, v, adj[k]))
