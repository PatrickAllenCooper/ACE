"""More sensitive endpoints from curves.csv. DESCRIPTIVE POST HOC NUMBERS ON DEVELOPMENT DATA.
No endpoint here was predeclared. p-values are uncorrected for the many comparisons shown."""
import numpy as np, pandas as pd, warnings, json
from scipy import stats
from common import *
warnings.filterwarnings('ignore')
cells, curves, meta = load()
pd.set_option('display.width', 250)
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]
inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')

def sustained(arr):
    """Cost of sustained attainment: smallest integer cost c such that error <= threshold at every
    integer cost in [c, 32] (step function, so this covers every evaluated state from c on). Cap 32 if the
    final state is above target. (A run that is at target only at exactly cost 32 gets 32.)"""
    out = np.full(arr.shape[0], CAP)
    for w in range(arr.shape[0]):
        ok = arr[w] <= THRESH
        if ok[-1]:
            k = len(ok) - 1
            while k > 0 and ok[k - 1]:
                k -= 1
            out[w] = INIT + k
    return out

E = {}
for p in POL:
    L = np.log10(S[p])
    E[p] = pd.DataFrame({
        'capped_cost': cw[p].values.astype(float),
        'sustained_cost': sustained(S[p]).astype(float),
        'aulc': L[:, 1:].mean(axis=1),            # mean log10 NMSE at integer total costs 9..32
        'aulc_left': L[:, :-1].mean(axis=1),      # integral of the step function over [8,32] / 24
        'auc_raw': S[p][:, 1:].mean(axis=1),      # mean NMSE at costs 9..32 (natural scale)
        'log_e12': L[:, 12 - 8], 'log_e16': L[:, 16 - 8], 'log_e24': L[:, 24 - 8], 'log_e32': L[:, 32 - 8],
        'e12': S[p][:, 4], 'e16': S[p][:, 8], 'e24': S[p][:, 16], 'e32': S[p][:, 24],
        'frac_cost_at_target': (S[p][:, 1:] <= THRESH).mean(axis=1),   # share of budgets 9..32 at which the state meets target
    })
# final error equals carried-forward error at cost 32
fe = cells.pivot(index='w', columns='policy_id', values='final_error')
print('final_error equals step-function error at cost 32 for every run:', all(np.allclose(fe[p].values, S[p][:, -1]) for p in POL))
ENDS = ['capped_cost', 'sustained_cost', 'aulc', 'aulc_left', 'auc_raw', 'log_e12', 'log_e16', 'log_e24', 'log_e32', 'frac_cost_at_target']
out = {}
for name, mask in [('all24', np.ones(24, bool)), ('informative14', inf)]:
    print('\n================ policy means,', name, '(lower is better except frac_cost_at_target)')
    tab = pd.DataFrame({SHORT[p]: E[p][mask][ENDS].mean() for p in POL}).T
    tab['geo_e12'] = [10 ** E[p][mask].log_e12.mean() for p in POL]
    tab['geo_e16'] = [10 ** E[p][mask].log_e16.mean() for p in POL]
    tab['geo_e24'] = [10 ** E[p][mask].log_e24.mean() for p in POL]
    tab['geo_e32'] = [10 ** E[p][mask].log_e32.mean() for p in POL]
    tab['mean_e12'] = [E[p][mask].e12.mean() for p in POL]
    tab['mean_e16'] = [E[p][mask].e16.mean() for p in POL]
    tab['mean_e24'] = [E[p][mask].e24.mean() for p in POL]
    tab['mean_e32'] = [E[p][mask].e32.mean() for p in POL]
    tab['med_e32'] = [E[p][mask].e32.median() for p in POL]
    print(tab.to_string(float_format=lambda x: '%.4f' % x))
    asc = pd.Series(True, index=tab.columns); asc['frac_cost_at_target'] = False
    ranks = pd.DataFrame({c: tab[c].rank(ascending=bool(asc[c]), method='min').astype(int) for c in tab.columns})
    print('\nranks (1 = best):')
    print(ranks.to_string())
    out[name] = {'means': tab, 'ranks': ranks}
    tab.to_csv('endpoint_means_%s.csv' % name); ranks.to_csv('endpoint_ranks_%s.csv' % name)

def paired(d):
    d = np.asarray(d, float)
    n = len(d); nz = d[d != 0]
    res = {'n': n, 'mean': d.mean(), 'sd': d.std(ddof=1), 'median': np.median(d), 'neg': int((d < 0).sum()), 'zero': int((d == 0).sum()), 'pos': int((d > 0).sum())}
    res['t_p'] = stats.ttest_1samp(d, 0).pvalue if d.std(ddof=1) > 0 else np.nan
    try:
        res['wilcoxon_p'] = stats.wilcoxon(nz, zero_method='wilcox', alternative='two-sided', mode='auto').pvalue if len(nz) > 0 else np.nan
    except Exception as ex:
        res['wilcoxon_p'] = np.nan
    try:
        res['wilcoxon_pratt_p'] = stats.wilcoxon(d, zero_method='pratt', alternative='two-sided').pvalue if len(nz) > 0 else np.nan
    except Exception:
        res['wilcoxon_pratt_p'] = np.nan
    res['signflip_p'] = signflip_exact(d)
    k = len(nz)
    res['sign_p'] = stats.binomtest(int((nz < 0).sum()), k, 0.5).pvalue if k > 0 else np.nan
    return res

rows = []
for name, mask in [('all24', np.ones(24, bool)), ('informative14', inf)]:
    for ref in ['target_risk', 'uniform_random']:
        for e in ['capped_cost', 'sustained_cost', 'aulc', 'aulc_left', 'auc_raw', 'log_e12', 'log_e16', 'log_e24', 'log_e32', 'frac_cost_at_target']:
            for p in POL:
                if p == ref: continue
                d = (E[p][e] - E[ref][e]).values[mask]
                r = paired(d); r.update({'subset': name, 'ref': ref, 'endpoint': e, 'policy': SHORT[p]})
                rows.append(r)
P = pd.DataFrame(rows)[['subset', 'ref', 'endpoint', 'policy', 'n', 'mean', 'sd', 'median', 'neg', 'zero', 'pos', 't_p', 'wilcoxon_p', 'wilcoxon_pratt_p', 'signflip_p', 'sign_p']]
P.to_csv('paired_endpoints.csv', index=False)
fmt = lambda x: '%.4f' % x
for name in ['all24', 'informative14']:
    for ref in ['target_risk', 'uniform_random']:
        for e in ['capped_cost', 'sustained_cost', 'aulc', 'auc_raw', 'log_e12', 'log_e16', 'log_e24', 'log_e32']:
            q = P[(P.subset == name) & (P.ref == ref) & (P.endpoint == e)].drop(columns=['subset', 'ref', 'endpoint', 'wilcoxon_pratt_p', 't_p'])
            print('\n--- %s | difference = policy - %s | endpoint %s  (neg = policy lower/better)' % (name, ref, e))
            print(q.to_string(index=False, float_format=fmt))
