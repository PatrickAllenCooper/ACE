"""Additional generated tables: power, endpoint sensitivity, twins, X-sign split, key contrasts."""
import numpy as np, pandas as pd, warnings, json
from scipy import stats
from common import *
warnings.filterwarnings('ignore')
cells, curves, meta = load()
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')[POL]
TAB = json.load(open('tables.json'))
def md(df, floatfmt='%.3f', index=True):
    df = df.copy()
    if index: df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out = []
        for v in r.values:
            if isinstance(v, (float, np.floating)): out.append('' if np.isnan(v) else (floatfmt % v))
            else: out.append(str(v))
        lines.append('| ' + ' | '.join(out) + ' |')
    return '\n'.join(lines)
def n_t(sd, delta, power, alpha=0.05, two_sided=True):
    n = 3
    while True:
        df = n - 1; ncp = delta * np.sqrt(n) / sd
        tc = stats.t.ppf(1 - alpha / 2, df) if two_sided else stats.t.ppf(1 - alpha, df)
        pw = 1 - stats.nct.cdf(tc, df, ncp) + (stats.nct.cdf(-tc, df, ncp) if two_sided else 0)
        if pw >= power: return n
        n += 1
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
rows = []
for p in POL:
    if p == 'target_risk': continue
    d = (cw[p] - cw['target_risk']).values.astype(float); a = aulc[p] - aulc['target_risk']
    rows.append({'policy - target_risk': SHORT[p], 'cost mean': d.mean(), 'cost SD (24)': d.std(ddof=1), 'zeros of 24': int((d == 0).sum()), 'cost SD (inf 14)': d[inf].std(ddof=1),
                 'log-area mean': a.mean(), 'log-area SD (24)': a.std(ddof=1), 'log-area SD (inf 14)': a[inf].std(ddof=1)})
SD = pd.DataFrame(rows).set_index('policy - target_risk')
TAB['paired_sd'] = md(SD)
rows = []
for name, sd in [('3.07 (coverage, smallest)', SD['cost SD (24)']['coverage']), ('6.21 (local_info = median of 7)', SD['cost SD (24)']['local_info']),
                 ('6.43 (dpo)', SD['cost SD (24)']['dpo']), ('6.76 (frozen, largest)', SD['cost SD (24)']['frozen']),
                 ('7.53 (median, informative-only systems)', SD['cost SD (inf 14)'].median())]:
    rows.append({'paired SD used': name, 'superiority 1 unit, 80%': n_t(sd, 1, .8), '1 unit, 90%': n_t(sd, 1, .9), '2 units, 80%': n_t(sd, 2, .8), '2 units, 90%': n_t(sd, 2, .9),
                 'noninferiority margin 1, true diff 0, 80%': n_t(sd, 1, .8, 0.025, False), 'margin 1, 90%': n_t(sd, 1, .9, 0.025, False),
                 'margin 1, true diff +0.33, 80%': n_t(sd, 1 - 1 / 3, .8, 0.025, False)})
TAB['power_cost'] = md(pd.DataFrame(rows).set_index('paired SD used'))
rows = []
for name, sd in [('0.351 (coverage, smallest)', SD['log-area SD (24)']['coverage']), ('0.478 (ranker = median of 7)', SD['log-area SD (24)'].median()), ('0.490 (local_info, largest)', SD['log-area SD (24)']['local_info'])]:
    rows.append({'paired SD used (log10 units)': name, '10% lower, 80%': n_t(sd, -np.log10(.9), .8), '10%, 90%': n_t(sd, -np.log10(.9), .9),
                 '20% lower, 80%': n_t(sd, -np.log10(.8), .8), '20%, 90%': n_t(sd, -np.log10(.8), .9), '50% lower, 80%': n_t(sd, -np.log10(.5), .8)})
TAB['power_area'] = md(pd.DataFrame(rows).set_index('paired SD used (log10 units)'))
rows = []
for rho in [0.5, 0.8]:
    r = {'replicate-noise share of single-run paired variance': rho}
    for R in [1, 2, 4, 8, 16]:
        r['R=%d' % R] = '%d / %d' % (n_t(6.2 * np.sqrt((1 - rho) + rho / R), 1, .8), n_t(0.478 * np.sqrt((1 - rho) + rho / R), -np.log10(.8), .8))
    rows.append(r)
TAB['power_reps'] = md(pd.DataFrame(rows).set_index('replicate-noise share of single-run paired variance'), '%.1f')

# endpoint sensitivity d_z
def capped(arr, th):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        idx = np.where(arr[w] <= th)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
E = {p: {'capped cost, threshold 0.04 (predeclared)': cw[p].values.astype(float), 'capped cost, threshold 0.02': capped(S[p], .02), 'capped cost, threshold 0.01': capped(S[p], .01),
         'log-area, costs 9..16': np.log10(S[p][:, 1:9]).mean(axis=1), 'log-area, costs 9..32': np.log10(S[p][:, 1:]).mean(axis=1),
         'raw area, costs 9..32': S[p][:, 1:].mean(axis=1), 'log NMSE at cost 12': np.log10(S[p][:, 4]), 'log NMSE at cost 16': np.log10(S[p][:, 8]),
         'log NMSE at cost 24': np.log10(S[p][:, 16]), 'log final NMSE (cost 32)': np.log10(S[p][:, 24])} for p in POL}
rows = []
for e in E['dpo_llm']:
    r = {'endpoint': e}
    for p in ['uniform_random', 'dpo_llm', 'frozen_llm', 'sft_llm', 'learned_ranker', 'menu_coverage', 'local_information']:
        d = E[p][e] - E['target_risk'][e]; r[SHORT[p]] = d.mean() / d.std(ddof=1)
    rows.append(r)
TAB['dz'] = md(pd.DataFrame(rows).set_index('endpoint'), '%.2f')

# near / far
inir = curves[curves.step == 0].drop_duplicates('w').set_index('w')
xbar = ((inir.prediction_0 - inir.prediction_2) / (inir.prediction_3 - inir.prediction_2)).values
near = xbar < 0
lf = {p: np.log10(S[p][:, -1]) for p in POL}
rows = []
for p in POL:
    if p == 'target_risk': continue
    da = aulc[p] - aulc['target_risk']; dc = (cw[p] - cw['target_risk']).values.astype(float); dl = lf[p] - lf['target_risk']
    rows.append({'policy - target_risk': SHORT[p], 'log-area, near (n=7)': da[near].mean(), 'log-area, far (n=17)': da[~near].mean(), 'cost, near': dc[near].mean(), 'cost, far': dc[~near].mean(),
                 'log final, near': dl[near].mean(), 'log final, far': dl[~near].mean()})
TAB['nearfar'] = md(pd.DataFrame(rows).set_index('policy - target_risk'))
k = pd.read_csv('key_contrasts.csv')
k = k[k.endpoint.isin(['capped_cost', 'log_area'])].copy()
k['-/0/+'] = k.neg.astype(str) + '/' + k.zero.astype(str) + '/' + k.pos.astype(str)
TAB['contrasts'] = md(k[['contrast', 'endpoint', 'mean', 'sd', '-/0/+', 'signflip_p', 'why']], '%.4f', index=False)
json.dump(TAB, open('tables.json', 'w'))
print(sorted(TAB.keys()))
print(TAB['power_cost']); print(TAB['power_area']); print(TAB['power_reps']); print(TAB['dz'])
