"""Which endpoint separates policies best per unit of noise? Standardized paired effects d_z = mean/SD.
POST HOC on development data; shown to compare endpoint sensitivity, not to select a favourable endpoint."""
import numpy as np, pandas as pd
from scipy import stats
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
def capped(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w] <= th)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
E = {}
for p in POL:
    L = np.log10(S[p])
    E[p] = {'capped_cost@0.04': cw[p].values.astype(float), 'capped_cost@0.02': capped(S[p], 0.02), 'capped_cost@0.01': capped(S[p], 0.01),
            'log_area 9..16': L[:, 1:9].mean(axis=1), 'log_area 9..24': L[:, 1:17].mean(axis=1), 'log_area 9..32': L[:, 1:].mean(axis=1),
            'raw_area 9..16': S[p][:, 1:9].mean(axis=1), 'raw_area 9..32': S[p][:, 1:].mean(axis=1),
            'log_e12': L[:, 4], 'log_e16': L[:, 8], 'log_e24': L[:, 16], 'log_e32': L[:, 24],
            'share_budgets_at_target': -(S[p][:, 1:] <= THRESH).mean(axis=1)}   # negated so that lower = better
rows = []
for e in E['dpo_llm']:
    r = {'endpoint': e}
    for p in ['uniform_random', 'dpo_llm', 'frozen_llm', 'sft_llm', 'learned_ranker', 'menu_coverage', 'local_information']:
        d = E[p][e] - E['target_risk'][e]
        r[SHORT[p]] = d.mean() / d.std(ddof=1)
    rows.append(r)
print('standardized paired effect d_z = mean/SD of (policy - target_risk), all 24 systems (positive = target_risk better)')
print(pd.DataFrame(rows).set_index('endpoint').round(3).to_string())
rows = []
for e in E['dpo_llm']:
    r = {'endpoint': e}
    for p in ['uniform_random', 'dpo_llm', 'frozen_llm', 'sft_llm', 'learned_ranker', 'menu_coverage', 'local_information']:
        d = (E[p][e] - E['target_risk'][e])[inf]
        r[SHORT[p]] = d.mean() / d.std(ddof=1)
    rows.append(r)
print('\nsame, informative 14 systems')
print(pd.DataFrame(rows).set_index('endpoint').round(3).to_string())
print('\n(n needed for 80%% power at two-sided 0.05 is about 7.85 / d_z^2 + 2: d_z 0.2 -> %d, 0.3 -> %d, 0.5 -> %d, 0.7 -> %d)' % (7.85 / .04 + 2, 7.85 / .09 + 2, 7.85 / .25 + 2, 7.85 / .49 + 2))

# oracle headroom and its power in this 24-system screen
tr = cw['target_risk'].values.astype(float)
floor = np.where(inf, 9., 8.)
best8 = cw[POL].min(axis=1).values.astype(float)
def power_t(d, n=None):
    n = len(d) if n is None else n
    ncp = abs(d.mean()) * np.sqrt(n) / d.std(ddof=1); tc = stats.t.ppf(.975, n - 1)
    return 1 - stats.nct.cdf(tc, n - 1, ncp) + stats.nct.cdf(-tc, n - 1, ncp)
for name, o in [('oracle at the trivial floor (8 if at target at init, else 9)', floor), ('hindsight best-of-8 envelope', best8)]:
    d = o - tr
    print('\n%s minus target_risk: mean %.3f SD %.3f ; t = %.2f ; plug-in power of a 24-system paired t-test %.2f ; of 48 systems %.2f'
          % (name, d.mean(), d.std(ddof=1), d.mean() / (d.std(ddof=1) / np.sqrt(24)), power_t(d), power_t(d, 48)))
for frac in [0.25, 0.5, 0.75]:
    # a policy that removes a fixed fraction of target_risk's excess over the floor on every system (deterministic, no added noise): optimistic
    d = -frac * (tr - floor)
    print('   policy capturing %.0f%% of the floor headroom with NO extra noise: mean %.3f SD %.3f ; power at n=24 %.2f' % (100 * frac, d.mean(), d.std(ddof=1), power_t(d)))
    # with independent replicate noise like a second policy: use SD 6.2
    ncp = abs(d.mean()) * np.sqrt(24) / 6.2; tc = stats.t.ppf(.975, 23)
    print('      same mean effect with the observed between-policy SD 6.2: power at n=24 %.2f ; systems for 80%%: %d' % (1 - stats.nct.cdf(tc, 23, ncp) + stats.nct.cdf(-tc, 23, ncp), int(np.ceil((2.8016 * 6.2 / abs(d.mean())) ** 2)) + 2))

# best-of-two selection effect from the identical-policy twins (DPO vs SFT, cost <= 18)
def capped_at(arr, cap):
    out = np.full(arr.shape[0], float(cap))
    for w in range(arr.shape[0]):
        idx = np.where(arr[w, :cap - INIT + 1] <= THRESH)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
a, b = capped_at(S['dpo_llm'], 18), capped_at(S['sft_llm'], 18)
print('\nidentical-policy twins (cap 18): mean |difference| all-24 %.3f, informative %.3f ; expected gain of best-of-2 over one run = half of that: %.3f / %.3f cost units'
      % (np.abs(a - b).mean(), np.abs(a - b)[inf].mean(), np.abs(a - b).mean() / 2, np.abs(a - b)[inf].mean() / 2))
c18 = {p: capped_at(S[p], 18) for p in POL}
print('cap-18 endpoint: target_risk mean %.3f ; best-of-8 mean %.3f ; envelope saving %.3f (informative %.3f)'
      % (c18['target_risk'].mean(), np.min([c18[p] for p in POL], axis=0).mean(), (c18['target_risk'] - np.min([c18[p] for p in POL], axis=0)).mean(),
         (c18['target_risk'] - np.min([c18[p] for p in POL], axis=0))[inf].mean()))
