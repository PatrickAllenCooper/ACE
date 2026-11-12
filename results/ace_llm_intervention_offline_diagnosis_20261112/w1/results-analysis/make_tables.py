"""Generate the markdown tables used in the notes file, directly from the saved CSVs (no transcription)."""
import numpy as np, pandas as pd, warnings, json
from scipy import stats
from collections import Counter
from common import *
warnings.filterwarnings('ignore')
cells, curves, meta = load()
S = wide_steps(curves)
T = np.load('truth_means.npy')
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')[POL]
TAB = {}

def md(df, floatfmt='%.3f', index=True):
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells_ = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells_.append('' if np.isnan(v) else (floatfmt % v))
            else:
                cells_.append(str(v))
        lines.append('| ' + ' | '.join(cells_) + ' |')
    return '\n'.join(lines)

# 1 policy summary
g = cells.groupby('policy_id')
t = pd.DataFrame({'mean capped total cost': g.capped_total_cost.mean(), 'ever /24': g.ever_attained.sum(), 'final /24': g.final_attained.sum(),
                  'mean final NMSE': g.final_error.mean(), 'median final NMSE': g.final_error.median(),
                  'geo-mean final NMSE': g.final_error.apply(lambda s: 10 ** np.log10(s).mean()), 'paid cost': g.actual_total_cost.sum(),
                  'stops': g.termination_reason.apply(lambda s: (s == 'policy_stop').sum())}).loc[POL]
t.index = [SHORT[p] for p in POL]; t.index.name = 'policy'
TAB['policy_summary'] = md(t, '%.5f')

# 2 per-system
t = pd.DataFrame({'graph': meta.graph_stratum, 'regime': meta.regime, 'menu': meta.cost_stratum.str[:3], 'init NMSE': ini})
for p in POL: t[SHORT[p]] = cw[p].values
t['best8'] = cw.min(axis=1).values
t.index = ['w%02d' % w for w in t.index]; t.index.name = 'sys'
TAB['per_system'] = md(t, '%.4f')
at = ~inf
_u = pd.DataFrame({'at target after init': pd.concat([meta[at].groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']]),
                   'of': pd.concat([meta.groupby(k).size() for k in ['graph_stratum', 'regime', 'cost_stratum']])}).fillna(0).astype(int)
_u = _u.loc[['chain', 'fork', 'collider', 'matched_low', 'matched_high', 'shifted_low', 'shifted_high', 'uniform', 'heterogeneous']]
_u.index.name = 'stratum'
TAB['uninformative'] = md(_u)

# 3 headroom
tr = cw['target_risk']; best8 = cw.min(axis=1); floor = pd.Series(np.where(inf, 9, 8), index=cw.index)
rows = []
for name, m in [('all 24', np.ones(24, bool)), ('informative 14', inf)]:
    rows.append({'subset': name, 'target_risk mean': tr[m].mean(), 'best-of-8 mean': best8[m].mean(), 'floor mean': floor[m].mean(),
                 'TR - best8 (sum)': int((tr - best8)[m].sum()), 'TR - best8 (mean)': (tr - best8)[m].mean(),
                 'TR - floor (sum)': int((tr - floor)[m].sum()), 'TR - floor (mean)': (tr - floor)[m].mean(),
                 'systems where some arm beats TR': int(((tr - best8)[m] > 0).sum())})
TAB['headroom'] = md(pd.DataFrame(rows).set_index('subset'))

# 4 endpoints
def sustained(arr):
    out = np.full(arr.shape[0], CAP, float)
    for w in range(arr.shape[0]):
        ok = arr[w] <= THRESH
        if ok[-1]:
            k = len(ok) - 1
            while k > 0 and ok[k - 1]: k -= 1
            out[w] = INIT + k
    return out
E = {}
for p in POL:
    L = np.log10(S[p])
    E[p] = pd.DataFrame({'capped_cost': cw[p].values.astype(float), 'sustained_cost': sustained(S[p]), 'log_area': L[:, 1:].mean(axis=1), 'raw_area': S[p][:, 1:].mean(axis=1),
                         'log_e12': L[:, 4], 'log_e16': L[:, 8], 'log_e24': L[:, 16], 'log_e32': L[:, 24]})
for name, m in [('all24', np.ones(24, bool)), ('inf14', inf)]:
    t = pd.DataFrame({SHORT[p]: {'capped cost': E[p].capped_cost[m].mean(), 'sustained cost': E[p].sustained_cost[m].mean(), 'log-area': E[p].log_area[m].mean(),
                                 'raw area': E[p].raw_area[m].mean(), 'geo NMSE@12': 10 ** E[p].log_e12[m].mean(), 'geo NMSE@16': 10 ** E[p].log_e16[m].mean(),
                                 'geo NMSE@24': 10 ** E[p].log_e24[m].mean(), 'geo NMSE@32 (final)': 10 ** E[p].log_e32[m].mean()} for p in POL}).T
    r = t.rank(method='min').astype(int)
    out = t.copy().astype(object)
    for c in t.columns:
        fmt = '%.3f' if c in ('capped cost', 'sustained cost', 'log-area') else '%.4f'
        out[c] = [(fmt % v) + ' (%d)' % k for v, k in zip(t[c], r[c])]
    out.index.name = 'policy'
    TAB['endpoint_means_' + name] = md(out)

def paired(d):
    d = np.asarray(d, float); nz = d[d != 0]
    return {'mean': d.mean(), 'SD': d.std(ddof=1), '-/0/+': '%d/%d/%d' % ((d < 0).sum(), (d == 0).sum(), (d > 0).sum()),
            'Wilcoxon p': stats.wilcoxon(nz).pvalue if len(nz) else np.nan, 'sign-flip p': signflip_exact(d)}
for ref in ['target_risk', 'uniform_random']:
    for e in ['capped_cost', 'log_area', 'log_e12', 'log_e16', 'log_e24', 'log_e32', 'sustained_cost']:
        rows = []
        for p in POL:
            if p == ref: continue
            r = {'policy': SHORT[p]}
            a = paired((E[p][e] - E[ref][e]).values); b = paired((E[p][e] - E[ref][e]).values[inf])
            r.update({'mean': a['mean'], 'SD': a['SD'], '-/0/+': a['-/0/+'], 'Wilcoxon p': a['Wilcoxon p'], 'sign-flip p': a['sign-flip p'],
                      'inf14 mean': b['mean'], 'inf14 SD': b['SD'], 'inf14 -/0/+': b['-/0/+'], 'inf14 sign-flip p': b['sign-flip p']})
            rows.append(r)
        TAB['paired_%s_%s' % (e, SHORT[ref])] = md(pd.DataFrame(rows).set_index('policy'), '%.4f')

# 6 actions
acq = curves[curves.step > 0].join(meta[['graph_stratum', 'cost_stratum', 'regime']], on='w')
for p in ['target_risk', 'local_information']:
    d = acq[acq.policy_id == p]
    for by, nm in [(['graph_stratum', 'cost_stratum'], 'graphcost'), (['regime'], 'regime')]:
        t = pd.crosstab([d[b] for b in by], d.action_id).reindex(columns=ACTIONS, fill_value=0)
        t.columns = [ASHORT[a] for a in ACTIONS]; t['n'] = t.sum(axis=1)
        if isinstance(t.index, pd.MultiIndex):
            t.index = [' / '.join(i) for i in t.index]
        t.index.name = 'cell'
        TAB['actions_%s_%s' % (SHORT[p], nm)] = md(t)
    d2 = d.copy(); d2['steps'] = pd.cut(d2.step, [0, 2, 4, 8, 12, 16, 24], labels=['1-2', '3-4', '5-8', '9-12', '13-16', '17-24'])
    t = pd.crosstab(d2.steps, d2.action_id).reindex(columns=ACTIONS, fill_value=0); t.columns = [ASHORT[a] for a in ACTIONS]; t['n'] = t.sum(axis=1)
    TAB['actions_%s_step' % SHORT[p]] = md(t)
TAB['fixed_seq'] = md(pd.read_csv('fixed_sequence_agreement.csv').set_index('policy')[['graph_x_cost_modal', 'graph_x_cost_loo', 'graph_modal', 'graph_loo', 'global_constant_action_share']])

# 7 dpo strata
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
df = meta.copy()
df['dpo'] = cw['dpo_llm'].values; df['tr'] = cw['target_risk'].values; df['rnd'] = cw['uniform_random'].values; df['frz'] = cw['frozen_llm'].values
df['da'] = aulc['dpo_llm'] - aulc['target_risk']; df['dfin'] = np.log10(S['dpo_llm'][:, -1]) - np.log10(S['target_risk'][:, -1])
df['never'] = (df.dpo == 32).astype(int); df['infm'] = inf.astype(int)
rows = []
for by in ['graph_stratum', 'regime', 'cost_stratum']:
    for k, g_ in df.groupby(by):
        gi = g_[g_.infm == 1]
        rows.append({'stratum': k, 'n': len(g_), 'informative n': len(gi), 'DPO cost': g_.dpo.mean(), 'TR cost': g_.tr.mean(), 'DPO-TR': (g_.dpo - g_.tr).mean(),
                     'DPO-TR (informative)': (gi.dpo - gi.tr).mean() if len(gi) else np.nan, 'DPO-random': (g_.dpo - g_.rnd).mean(), 'DPO-frozen': (g_.dpo - g_.frz).mean(),
                     'log-area DPO-TR': g_.da.mean(), 'log final DPO-TR': g_.dfin.mean(), 'DPO never attains': int(g_.never.sum())})
TAB['dpo_strata'] = md(pd.DataFrame(rows).set_index('stratum'))
rows = []
for k, g_ in df.groupby(['graph_stratum', 'cost_stratum']):
    rows.append({'cell': ' / '.join(k), 'n': len(g_), 'informative n': int(g_.infm.sum()), 'DPO cost': g_.dpo.mean(), 'TR cost': g_.tr.mean(), 'DPO-TR': (g_.dpo - g_.tr).mean(),
                 'log-area DPO-TR': g_.da.mean(), 'DPO never attains': int(g_.never.sum())})
TAB['dpo_graphcost'] = md(pd.DataFrame(rows).set_index('cell'))
def perq(p, cost):
    out = np.empty((24, 8))
    for w in range(24):
        r = curves[(curves.policy_id == p) & (curves.w == w)].sort_values('step')
        k = np.searchsorted(r.total_cost_units.values, cost, side='right') - 1
        out[w] = ((r[PRED].values[k] - T[w]) / 2) ** 2 / 8
    return out
q0 = perq('dpo_llm', 8)
grp = {'q1,q7 (X=-1)': [1, 7], 'q2,q3 (X=0,+1)': [2, 3], 'q0 (observe)': [0], 'q4-q6 (do Y)': [4, 5, 6]}
rows = []
for p in ['dpo_llm', 'sft_llm', 'learned_ranker', 'frozen_llm', 'target_risk', 'local_information', 'menu_coverage', 'uniform_random']:
    qq = perq(p, 32)
    for g_ in ['chain', 'fork', 'collider']:
        m = (meta.graph_stratum == g_).values
        r = {'policy': SHORT[p], 'graph': g_}
        for name, idx in grp.items(): r[name] = qq[m][:, idx].sum() / q0[m][:, idx].sum()
        r['all queries'] = qq[m].sum() / q0[m].sum()
        rows.append(r)
TAB['retained'] = md(pd.DataFrame(rows).set_index('policy'))

# 8 non-monotonicity
TAB['nonmono'] = md(pd.read_csv('nonmonotonicity.csv').set_index('policy')[['acquisitions', 'error_rose', 'rose_share', 'rose_gt2x', 'runs_crossed', 'crossed_then_left', 'crossed_but_final_above', 'exit_events', 'first_cross_lost_next_step']])
# 9 sft
s = pd.read_csv('sft_stops.csv')
s['sys'] = ['w%02d' % w for w in s.w]
TAB['sft'] = md(s.set_index('sys')[['graph', 'regime', 'cost', 'term', 'stop_total_cost', 'acquisitions', 'init_err', 'err_at_stop', 'at_target_at_stop', 'ever', 'capped_cost', 'tr_err_at_same_cost', 'non_xneg_actions']], '%.4f')
# 10 threshold
th = pd.read_csv('threshold_sweep.csv')
th['threshold'] = th.threshold.map(lambda v: ('%.4f' % v))
for c in ['at_target_at_init', 'target_risk_never', 'no_policy_attains', 'informative(init>th & TR attains)']:
    th[c] = th[c].astype(int)
th = th.rename(columns={'at_target_at_init': 'at target after init', 'target_risk_never': 'TR never attains', 'no_policy_attains': 'no policy attains',
                        'informative(init>th & TR attains)': 'informative for TR', 'TR_mean': 'TR mean cost', 'best8_mean': 'best-of-8 mean', 'TR_minus_floor': 'TR minus floor',
                        'random-TR mean': 'random-TR mean', 'random-TR sd': 'random-TR SD', 'random-TR d/sd': 'random-TR d_z', 'local-TR sd': 'local_info-TR SD'})
TAB['threshold'] = md(th.set_index('threshold')[['at target after init', 'TR never attains', 'no policy attains', 'informative for TR', 'TR mean cost', 'best-of-8 mean', 'TR minus floor', 'random-TR mean', 'random-TR SD', 'random-TR d_z', 'local_info-TR SD']])
tm = pd.read_csv('threshold_sweep_policy_means.csv'); tm['threshold'] = tm.threshold.map(lambda v: ('%.4f' % v))
TAB['threshold_means'] = md(tm.set_index('threshold'))
json.dump(TAB, open('tables.json', 'w'))
print({k: len(v) for k, v in TAB.items()})
