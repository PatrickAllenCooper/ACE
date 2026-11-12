"""Replicate noise of the capped cost endpoint, from DPO/SFT runs that executed the identical
action prefix (set_x_neg repeatedly) on the same system with independent measurement noise."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
S = wide_steps(curves)      # policy -> [24, 25] step errors at integer costs 8..32
ini = S['dpo_llm'][:, 0]
inf = ini > THRESH
def capped_at(arr, cap):
    # first integer cost c in 8..cap with error <= threshold, else cap
    out = np.full(arr.shape[0], cap)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w, :cap - INIT + 1] <= THRESH)[0]
        if len(idx):
            out[w] = INIT + idx[0]
    return out
# the identical DPO/SFT prefix covers total cost <= 18 in every system (minimum prefix: 10 unit-cost actions)
for cap in [18]:
    c = {p: capped_at(S[p], cap) for p in POL}
    d = (c['dpo_llm'] - c['sft_llm'])
    print('cap %d: twin (DPO - SFT, identical actions) capped-cost differences, informative systems:' % cap, d[inf].tolist())
    print('   all 24: mean %+.3f SD %.3f ; informative 14: mean %+.3f SD %.3f ; nonzero %d of 14'
          % (d.mean(), d.std(ddof=1), d[inf].mean(), d[inf].std(ddof=1), (d[inf] != 0).sum()))
    print('   implied single-run replicate SD of the cap-%d cost endpoint: all-24 %.3f ; informative %.3f' % (cap, np.sqrt((d ** 2).mean() / 2), np.sqrt((d[inf] ** 2).mean() / 2)))
    print('   for comparison, SD over systems of paired differences between DIFFERENT policies at the same cap %d (all 24 / informative 14):' % cap)
    for p in POL:
        for q in ['target_risk']:
            if p == q: continue
            dd = c[p] - c[q]
            print('     %-12s - target_risk: mean %+.3f SD %.3f | informative mean %+.3f SD %.3f' % (SHORT[p], dd.mean(), dd.std(ddof=1), dd[inf].mean(), dd[inf].std(ddof=1)))
    tw_var = (d ** 2).mean()          # E[(a-b)^2] = 2 sigma_rep^2 for identical policy
    print('   twin difference variance (2*sigma_rep^2) = %.2f ; DPO - target_risk difference variance = %.2f ; ratio %.2f'
          % (tw_var, (c['dpo_llm'] - c['target_risk']).var(ddof=1), tw_var / (c['dpo_llm'] - c['target_risk']).var(ddof=1)))
# log error twin noise at fixed cost within prefix (cost 12, 16, 18)
print()
for cost in [9, 10, 12, 14, 16, 18]:
    d = np.log10(S['dpo_llm'][:, cost - 8]) - np.log10(S['sft_llm'][:, cost - 8])
    print('twin diff in log10 NMSE at total cost %2d: all-24 SD %.3f (replicate SD %.3f) ; informative SD %.3f (replicate SD %.3f)'
          % (cost, d.std(ddof=1), np.sqrt((d ** 2).mean() / 2), d[inf].std(ddof=1), np.sqrt((d[inf] ** 2).mean() / 2)))
    for p in ['target_risk', 'uniform_random', 'local_information']:
        dd = np.log10(S['dpo_llm'][:, cost - 8]) - np.log10(S[p][:, cost - 8])
        print('      vs different policy DPO - %-17s: mean %+.3f SD %.3f' % (p, dd.mean(), dd.std(ddof=1)))
# mean log error over costs 9..18 (prefix-area endpoint): twin replicate noise vs between-policy differences
A = {p: np.log10(S[p][:, 1:11]).mean(axis=1) for p in POL}
d = A['dpo_llm'] - A['sft_llm']
print()
print('prefix curve-area endpoint (mean log10 NMSE over total cost 9..18): twin diff mean %+.3f SD %.3f -> replicate SD %.3f' % (d.mean(), d.std(ddof=1), np.sqrt((d ** 2).mean() / 2)))
for p in POL:
    if p in ('dpo_llm',): continue
    dd = A['dpo_llm'] - A[p]
    print('   DPO - %-17s mean %+.3f SD %.3f' % (p, dd.mean(), dd.std(ddof=1)))
for p in POL:
    if p == 'target_risk': continue
    dd = A[p] - A['target_risk']
    print('   %-17s - target_risk mean %+.3f SD %.3f' % (p, dd.mean(), dd.std(ddof=1)))
