"""Threshold sensitivity of the retrospective cost-to-target endpoint. POST HOC DESIGN DIAGNOSTIC on development data.
It is used to judge how many systems are informative at each threshold, not to pick a threshold that favours a policy."""
import numpy as np, pandas as pd
from scipy import stats
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 280)
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]
def capped(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w] <= th[w])[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
def sustained(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        ok = arr[w] <= th[w]
        if ok[-1]:
            k = len(ok) - 1
            while k > 0 and ok[k - 1]: k -= 1
            out[w] = INIT + k
    return out
def mde(sd, n=24):
    # minimum detectable mean difference, paired t, two-sided 0.05, 80% power
    lo, hi = 0.0, 20 * sd
    for _ in range(60):
        mid = (lo + hi) / 2; df = n - 1; ncp = mid * np.sqrt(n) / sd; tc = stats.t.ppf(.975, df)
        pw = 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
        lo, hi = (mid, hi) if pw < .8 else (lo, mid)
    return hi
rows = []; rank_rows = []
for th in [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]:
    tv = np.full(24, th)
    C = {p: capped(S[p], tv) for p in POL}
    M = np.array([C[p] for p in POL])
    at_init = (ini <= th).sum()
    none = (M.min(axis=0) == CAP).sum()
    tr_never = (C['target_risk'] == CAP).sum()
    all_tied = (M.max(axis=0) == M.min(axis=0)).sum()
    informative_tr = ((ini > th) & (C['target_risk'] < CAP)).sum()
    d_r = C['uniform_random'] - C['target_risk']; d_d = C['dpo_llm'] - C['target_risk']; d_l = C['local_information'] - C['target_risk']
    floor = np.where(ini > th, 9, 8)
    rows.append({'threshold': th, 'at_target_at_init': at_init, 'no_policy_attains': none, 'target_risk_never': tr_never, 'all8_tied': all_tied,
                 'informative(init>th & TR attains)': informative_tr,
                 'TR_mean': C['target_risk'].mean(), 'best8_mean': M.min(axis=0).mean(), 'floor_mean': floor.mean(),
                 'TR_minus_best8': (C['target_risk'] - M.min(axis=0)).mean(), 'TR_minus_floor': (C['target_risk'] - floor).mean(),
                 'random-TR mean': d_r.mean(), 'random-TR sd': d_r.std(ddof=1), 'random-TR d/sd': d_r.mean() / d_r.std(ddof=1),
                 'dpo-TR mean': d_d.mean(), 'dpo-TR sd': d_d.std(ddof=1), 'local-TR mean': d_l.mean(), 'local-TR sd': d_l.std(ddof=1),
                 'MDE_n24(sd local-TR)': mde(d_l.std(ddof=1))})
    rr = {'threshold': th}
    for p in POL: rr[SHORT[p]] = C[p].mean()
    rank_rows.append(rr)
R = pd.DataFrame(rows).set_index('threshold')
print(R.T.to_string(float_format=lambda x: '%.3f' % x))
print('\nmean capped first-crossing cost by policy and threshold')
RR = pd.DataFrame(rank_rows).set_index('threshold'); print(RR.round(3).to_string())
print('ranks'); print(RR.rank(axis=1, method='min').astype(int).to_string())
R.to_csv('threshold_sweep.csv'); RR.to_csv('threshold_sweep_policy_means.csv')
print('\nmean capped SUSTAINED cost by policy and threshold')
rs = []
for th in [0.005, 0.01, 0.02, 0.04, 0.08]:
    rr = {'threshold': th}
    for p in POL: rr[SHORT[p]] = sustained(S[p], np.full(24, th)).mean()
    rs.append(rr)
print(pd.DataFrame(rs).set_index('threshold').round(3).to_string())

# relative thresholds: reduce the system's own initial error by a factor
print('\nrelative thresholds (target = f x the system\'s own post-initialization error)')
rs = []; rk = []
for f in [0.5, 0.25, 0.1, 0.05]:
    tv = f * ini
    C = {p: capped(S[p], tv) for p in POL}; M = np.array([C[p] for p in POL])
    d_l = C['local_information'] - C['target_risk']; d_r = C['uniform_random'] - C['target_risk']
    rs.append({'f': f, 'no_policy_attains': (M.min(axis=0) == CAP).sum(), 'target_risk_never': (C['target_risk'] == CAP).sum(), 'all8_tied': (M.max(axis=0) == M.min(axis=0)).sum(),
               'TR_mean': C['target_risk'].mean(), 'best8_mean': M.min(axis=0).mean(), 'random-TR mean': d_r.mean(), 'random-TR sd': d_r.std(ddof=1),
               'local-TR sd': d_l.std(ddof=1)})
    rr = {'f': f}
    for p in POL: rr[SHORT[p]] = C[p].mean()
    rk.append(rr)
print(pd.DataFrame(rs).set_index('f').round(3).to_string()); print(pd.DataFrame(rk).set_index('f').round(3).to_string())

# what the frozen arm (observe-only) says about a larger observational initialization
print('\nfrozen LLM = 8 initial observations plus up to 24 more observations (574 of 576 actions are observe_all):')
F = S['frozen_llm']
for c in [8, 12, 16, 20, 24, 28, 32]:
    print('   after %2d total observations: systems at or below 0.04: %2d ; <= 0.02: %2d ; <= 0.01: %2d ; geometric-mean NMSE %.4f' % (c, (F[:, c - 8] <= .04).sum(), (F[:, c - 8] <= .02).sum(), (F[:, c - 8] <= .01).sum(), 10 ** np.log10(F[:, c - 8]).mean()))
print('   (so MORE observational initialization makes MORE systems uninformative; a smaller initialization is the direction that helps)')
# minimum error reached by any policy at any cost per system: reachability
allmin = np.min([S[p].min(axis=1) for p in POL], axis=0)
trmin = S['target_risk'].min(axis=1); trfinal = S['target_risk'][:, -1]
print('\nreachability: per-system minimum error over all policies and costs; quantiles:', np.round(np.quantile(allmin, [0, .25, .5, .75, 1]), 5))
print('target_risk final error quantiles:', np.round(np.quantile(trfinal, [0, .25, .5, .75, 1]), 5))
for th in [0.005, 0.01, 0.02, 0.04]:
    print('   threshold %.3f: target_risk final <= th in %d systems ; sustained by target_risk from some cost on: %d ; init <= th: %d' % (th, (trfinal <= th).sum(), (sustained(S['target_risk'], np.full(24, th)) < CAP).sum(), (ini <= th).sum()))
# minimum detectable effects at n=24 for the predeclared endpoint
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
print('\nminimum detectable mean difference at n=24 (paired t, two-sided 0.05, 80% power):')
for p in POL:
    if p == 'target_risk': continue
    sd = (cw[p] - cw['target_risk']).std(ddof=1)
    print('   %-12s SD %.3f -> MDE %.2f cost units' % (SHORT[p], sd, mde(sd)))
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
for p in ['menu_coverage', 'local_information', 'dpo_llm']:
    sd = (aulc[p] - aulc['target_risk']).std(ddof=1)
    m = mde(sd)
    print('   curve area, SD of %-17s %.3f -> MDE %.3f log10 units = %.0f%% lower geometric-mean error' % (p, sd, m, 100 * (1 - 10 ** (-m))))
