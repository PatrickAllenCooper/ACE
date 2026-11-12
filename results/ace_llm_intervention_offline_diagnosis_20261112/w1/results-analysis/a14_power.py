"""Power planning from the observed paired SDs. PLANNING ARITHMETIC ON DEVELOPMENT DATA, not a result.
Formulae (paired design, d_i = policy_i - reference_i, SD s_d, n independent systems):
  superiority, two-sided alpha:   n = ((z_{1-alpha/2} + z_{1-beta}) * s_d / delta)^2
  noninferiority, margin M, one-sided alpha, assumed true difference D0 (0 = truly equal):
                                   n = ((z_{1-alpha} + z_{1-beta}) * s_d / (M - D0))^2
  exact paired-t version: smallest n with noncentral-t power >= target (df = n-1, ncp = delta*sqrt(n)/s_d).
  with R replicate episodes per (system, policy) and replicate-noise share rho of the single-run paired variance:
                                   s_d(R)^2 = s_d^2 * ((1 - rho) + rho / R)
"""
import numpy as np, pandas as pd
from scipy import stats
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
aucr = {p: S[p][:, 1:].mean(axis=1) for p in POL}

def n_z(sd, delta, power, alpha=0.05, two_sided=True):
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    return ((za + stats.norm.ppf(power)) * sd / delta) ** 2
def n_t(sd, delta, power, alpha=0.05, two_sided=True):
    n = 3
    while True:
        df = n - 1; ncp = delta * np.sqrt(n) / sd
        tc = stats.t.ppf(1 - alpha / 2, df) if two_sided else stats.t.ppf(1 - alpha, df)
        pw = 1 - stats.nct.cdf(tc, df, ncp) + (stats.nct.cdf(-tc, df, ncp) if two_sided else 0)
        if pw >= power: return n
        n += 1
        if n > 200000: return np.nan

print('z multipliers: two-sided 0.05 & 80%% power: %.4f ; 90%%: %.4f ; one-sided 0.025 identical' % (stats.norm.ppf(.975) + stats.norm.ppf(.8), stats.norm.ppf(.975) + stats.norm.ppf(.9)))
rows = []
for p in POL:
    if p == 'target_risk': continue
    d = (cw[p] - cw['target_risk']).values.astype(float)
    a = aulc[p] - aulc['target_risk']
    r = aucr[p] - aucr['target_risk']
    rows.append({'policy': SHORT[p], 'cost_mean': d.mean(), 'cost_sd_all24': d.std(ddof=1), 'cost_sd_inf14': d[inf].std(ddof=1), 'zeros_all24': int((d == 0).sum()),
                 'aulc_mean': a.mean(), 'aulc_sd_all24': a.std(ddof=1), 'aulc_sd_inf14': a[inf].std(ddof=1),
                 'aucraw_sd_all24': r.std(ddof=1), 'aucraw_sd_inf14': r[inf].std(ddof=1)})
SD = pd.DataFrame(rows).set_index('policy')
print('\nobserved paired SDs of (policy - target_risk)'); print(SD.round(4).to_string())
print('pooled (root-mean-square over the 7 comparisons): cost all-24 %.3f ; cost informative-14 %.3f ; aulc all-24 %.4f ; aulc informative-14 %.4f'
      % (np.sqrt((SD.cost_sd_all24 ** 2).mean()), np.sqrt((SD.cost_sd_inf14 ** 2).mean()), np.sqrt((SD.aulc_sd_all24 ** 2).mean()), np.sqrt((SD.aulc_sd_inf14 ** 2).mean())))
print('median over the 7 comparisons: cost all-24 %.3f ; informative-14 %.3f ; aulc all-24 %.4f ; informative-14 %.4f'
      % (SD.cost_sd_all24.median(), SD.cost_sd_inf14.median(), SD.aulc_sd_all24.median(), SD.aulc_sd_inf14.median()))

def table_cost(sds, label):
    print('\n--- capped cost-to-target: systems needed (paired t, two-sided alpha 0.05; z-approximation in brackets) ---', label)
    out = []
    for name, sd in sds:
        out.append({'SD source': name, 'SD': sd,
                    'd=1, 80%': '%d (%.0f)' % (n_t(sd, 1, .8), n_z(sd, 1, .8)), 'd=1, 90%': '%d (%.0f)' % (n_t(sd, 1, .9), n_z(sd, 1, .9)),
                    'd=2, 80%': '%d (%.0f)' % (n_t(sd, 2, .8), n_z(sd, 2, .8)), 'd=2, 90%': '%d (%.0f)' % (n_t(sd, 2, .9), n_z(sd, 2, .9)),
                    'NI M=1 (true diff 0), 80%': '%d' % n_t(sd, 1, .8, alpha=0.025, two_sided=False), 'NI M=1, 90%': '%d' % n_t(sd, 1, .9, alpha=0.025, two_sided=False),
                    'NI M=1 (true diff +0.33), 80%': '%d' % n_t(sd, 1 - 0.3333, .8, alpha=0.025, two_sided=False)})
    print(pd.DataFrame(out).set_index('SD source').to_string())
table_cost([('coverage - target_risk (smallest)', SD.cost_sd_all24['coverage']), ('local_info - target_risk (closest competitor)', SD.cost_sd_all24['local_info']),
            ('dpo - target_risk', SD.cost_sd_all24['dpo']), ('median of 7', SD.cost_sd_all24.median()), ('frozen - target_risk (largest)', SD.cost_sd_all24['frozen'])],
           'all-24 mix (10 of 24 systems tied at 8 by construction)')
table_cost([('coverage - target_risk', SD.cost_sd_inf14['coverage']), ('local_info - target_risk', SD.cost_sd_inf14['local_info']),
            ('dpo - target_risk', SD.cost_sd_inf14['dpo']), ('median of 7', SD.cost_sd_inf14.median())],
           'informative-only systems (SD from the 14 systems above target after initialization)')

def table_aulc(sds, label):
    print('\n--- curve-area endpoint (mean log10 NMSE over costs 9..32): systems needed ---', label)
    print('    10%% lower geometric-mean error = %.4f log10 units ; 20%% = %.4f' % (-np.log10(.9), -np.log10(.8)))
    out = []
    for name, sd in sds:
        out.append({'SD source': name, 'SD': round(sd, 4),
                    '10%, 80%': n_t(sd, -np.log10(.9), .8), '10%, 90%': n_t(sd, -np.log10(.9), .9),
                    '20%, 80%': n_t(sd, -np.log10(.8), .8), '20%, 90%': n_t(sd, -np.log10(.8), .9),
                    '30%, 80%': n_t(sd, -np.log10(.7), .8), '50%, 80%': n_t(sd, -np.log10(.5), .8)})
    print(pd.DataFrame(out).set_index('SD source').to_string())
table_aulc([('coverage - target_risk (smallest)', SD.aulc_sd_all24['coverage']), ('local_info - target_risk', SD.aulc_sd_all24['local_info']),
            ('dpo - target_risk', SD.aulc_sd_all24['dpo']), ('median of 7', SD.aulc_sd_all24.median())], 'all 24')
table_aulc([('coverage - target_risk', SD.aulc_sd_inf14['coverage']), ('local_info - target_risk', SD.aulc_sd_inf14['local_info']),
            ('dpo - target_risk', SD.aulc_sd_inf14['dpo']), ('median of 7', SD.aulc_sd_inf14.median())], 'informative 14')

# natural-scale area: 10% / 20% of target_risk's mean area
m = aucr['target_risk'].mean()
print('\n--- natural-scale area (mean NMSE over costs 9..32); target_risk mean %.5f ; 10%% = %.5f, 20%% = %.5f' % (m, .1 * m, .2 * m))
for name in ['coverage', 'local_info', 'dpo']:
    sd = SD.aucraw_sd_all24[name]
    print('   SD %-10s %.4f : 10%% 80%%: %d ; 20%% 80%%: %d ; 20%% 90%%: %d' % (name, sd, n_t(sd, .1 * m, .8), n_t(sd, .2 * m, .8), n_t(sd, .2 * m, .9)))

# replicate design effect
print('\n--- effect of R replicate episodes per (system, policy) on required systems, capped cost, delta=1, 80% power, single-run SD 6.2')
for rho in [0.5, 0.8]:
    print('   replicate-noise share rho = %.1f : ' % rho + ' ; '.join('R=%d -> SD %.2f, n=%d' % (R, 6.2 * np.sqrt((1 - rho) + rho / R), n_t(6.2 * np.sqrt((1 - rho) + rho / R), 1, .8)) for R in [1, 2, 4, 8, 16, 32]))
print('--- same for the curve-area endpoint, 20% (0.0969), single-run SD 0.45')
for rho in [0.5, 0.8]:
    print('   rho = %.1f : ' % rho + ' ; '.join('R=%d -> SD %.3f, n=%d' % (R, 0.45 * np.sqrt((1 - rho) + rho / R), n_t(0.45 * np.sqrt((1 - rho) + rho / R), -np.log10(.8), .8)) for R in [1, 2, 4, 8, 16, 32]))

# bootstrap power for the effects actually observed
rng = np.random.default_rng(20261112)
def boot_power(d, ns, B=4000):
    out = {}
    for n in ns:
        idx = rng.integers(0, len(d), size=(B, n)); x = d[idx]
        m = x.mean(axis=1); s = x.std(axis=1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = m / (s / np.sqrt(n))
        pt = 2 * stats.t.sf(np.abs(t), n - 1)
        rej_t = np.nanmean(np.where(s > 0, pt < 0.05, False))
        # Wilcoxon (zeros dropped), normal approximation with tie correction via scipy on each sample is slow; do 600 samples
        rej_w = np.mean([ (stats.wilcoxon(xx[xx != 0]).pvalue < 0.05) if (xx != 0).sum() >= 6 else False for xx in x[:600]])
        out[n] = (rej_t, rej_w)
    return out
import warnings; warnings.filterwarnings('ignore')
print('\n--- bootstrap power (resampling the 24 observed paired differences with replacement; two-sided 0.05) to detect effects of the OBSERVED size')
for name, d in [('random - target_risk, capped cost (obs mean +2.625)', (cw['uniform_random'] - cw['target_risk']).values.astype(float)),
                ('dpo - target_risk, capped cost (obs mean +3.542)', (cw['dpo_llm'] - cw['target_risk']).values.astype(float)),
                ('frozen - target_risk, capped cost (obs mean +2.000)', (cw['frozen_llm'] - cw['target_risk']).values.astype(float)),
                ('ranker - target_risk, capped cost (obs mean +1.958)', (cw['learned_ranker'] - cw['target_risk']).values.astype(float)),
                ('coverage - target_risk, capped cost (obs mean +0.833)', (cw['menu_coverage'] - cw['target_risk']).values.astype(float)),
                ('random - target_risk, curve area (obs mean +0.183)', aulc['uniform_random'] - aulc['target_risk']),
                ('dpo - target_risk, curve area (obs mean +0.132)', aulc['dpo_llm'] - aulc['target_risk'])]:
    bp = boot_power(d, [24, 48, 96, 192])
    print('   %-55s ' % name + ' ; '.join('n=%d: t %.2f, Wilcoxon %.2f' % (n, bp[n][0], bp[n][1]) for n in bp))
