"""Where the constant set_x_neg policy (DPO arm) loses most and least, with a per-query error decomposition.
Per-query truth is the least-squares reconstruction from a09_truth.py (an inference from saved outputs, exact to 1e-14)."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
T = np.load('truth_means.npy')
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
lf = {p: np.log10(S[p][:, -1]) for p in POL}
df = meta.copy()
df['init_err'] = ini
for ref in ['target_risk', 'uniform_random', 'frozen_llm']:
    df['dcost_' + SHORT[ref]] = (cw['dpo_llm'] - cw[ref]).values
    df['daulc_' + SHORT[ref]] = aulc['dpo_llm'] - aulc[ref]
    df['dlogfinal_' + SHORT[ref]] = lf['dpo_llm'] - lf[ref]
df['dpo_cost'] = cw['dpo_llm'].values; df['tr_cost'] = cw['target_risk'].values
df['dpo_final'] = S['dpo_llm'][:, -1]; df['tr_final'] = S['target_risk'][:, -1]
df['dpo_final_ok'] = df.dpo_final <= THRESH
df['dpo_ever'] = cw['dpo_llm'].values < 32
print('per-system DPO vs target_risk')
print(df[['graph_stratum', 'regime', 'cost_stratum', 'init_err', 'dpo_cost', 'tr_cost', 'dcost_target_risk', 'daulc_target_risk', 'dpo_final', 'tr_final', 'dlogfinal_target_risk']].to_string(float_format=lambda x: '%.4f' % x))
def agg(by, mask=None, label=''):
    d = df if mask is None else df[mask]
    g = d.groupby(by)
    t = pd.DataFrame({'n': g.size(), 'dpo_cost': g.dpo_cost.mean(), 'tr_cost': g.tr_cost.mean(),
                      'dcost_vs_tr': g.dcost_target_risk.mean(), 'dcost_vs_random': g.dcost_random.mean(), 'dcost_vs_frozen': g.dcost_frozen.mean(),
                      'daulc_vs_tr': g.daulc_target_risk.mean(), 'daulc_vs_random': g.daulc_random.mean(),
                      'dlogfinal_vs_tr': g.dlogfinal_target_risk.mean(),
                      'dpo_geo_final': g.dpo_final.apply(lambda s: 10 ** np.log10(s).mean()), 'tr_geo_final': g.tr_final.apply(lambda s: 10 ** np.log10(s).mean()),
                      'dpo_mean_final': g.dpo_final.mean(), 'tr_mean_final': g.tr_final.mean(),
                      'dpo_never': g.dpo_ever.apply(lambda s: (~s).sum()), 'dpo_final_above': g.dpo_final_ok.apply(lambda s: (~s).sum())})
    print('\n--- by', by, label); print(t.to_string(float_format=lambda x: '%.3f' % x))
for by in ['graph_stratum', 'regime', 'cost_stratum', 'prior', 'noise']:
    agg(by, None, '(all 24)')
agg(['graph_stratum', 'cost_stratum'], None, '(all 24)')
agg(['graph_stratum', 'prior'], None, '(all 24)')
for by in ['graph_stratum', 'regime', 'cost_stratum']:
    agg(by, inf, '(informative 14)')

# per-query decomposition
QN = ['q0 obs', 'q1 X=-1', 'q2 X=0', 'q3 X=+1', 'q4 Y=-1', 'q5 Y=0', 'q6 Y=+1', 'q7 X=-1,Y=+1']
def perq(p, cost):
    out = np.empty((24, 8))
    for w in range(24):
        r = curves[(curves.policy_id == p) & (curves.w == w)].sort_values('step')
        k = np.searchsorted(r.total_cost_units.values, cost, side='right') - 1
        out[w] = ((r[PRED].values[k] - T[w]) / 2) ** 2 / 8
    return out
chk = perq('dpo_llm', 32).sum(axis=1)
print('\nper-query contributions sum to saved final NMSE (DPO): max abs diff %.2e' % np.abs(chk - S['dpo_llm'][:, -1]).max())
for cost in [8, 16, 32]:
    print('\n=== mean per-query contribution to NMSE at total cost %d (each column = mean over the 8 systems of the graph; sums to mean NMSE)' % cost)
    for p in (['dpo_llm'] if cost == 8 else ['dpo_llm', 'target_risk', 'local_information', 'frozen_llm', 'uniform_random']):
        q = perq(p, cost)
        t = pd.DataFrame(q, columns=QN).join(meta.graph_stratum).groupby('graph_stratum').mean()
        t['total'] = t.sum(axis=1)
        print(('initial state (all arms)' if cost == 8 else p)); print((t * 1000).round(2).to_string() + '   (x 1e-3)')
# share of DPO final error by query group and graph
q = perq('dpo_llm', 32); q0 = perq('dpo_llm', 8)
grp = {'q1,q7 (X=-1 queries)': [1, 7], 'q2,q3 (X=0,+1)': [2, 3], 'q0 (observational)': [0], 'q4-q6 (do Y)': [4, 5, 6]}
print('\n=== DPO: retained fraction of initial per-query error at the final state (sum over systems of final / sum of initial), by graph')
rows = []
for g in ['chain', 'fork', 'collider']:
    m = (meta.graph_stratum == g).values
    r = {'graph': g}
    for name, idx in grp.items():
        r[name] = q[m][:, idx].sum() / q0[m][:, idx].sum()
    r['all'] = q[m].sum() / q0[m].sum()
    rows.append(r)
print(pd.DataFrame(rows).set_index('graph').round(3).to_string())
for p in ['target_risk', 'local_information', 'frozen_llm', 'uniform_random', 'menu_coverage', 'learned_ranker', 'sft_llm']:
    qq = perq(p, 32)
    rows = []
    for g in ['chain', 'fork', 'collider']:
        m = (meta.graph_stratum == g).values
        r = {'graph': g}
        for name, idx in grp.items():
            r[name] = qq[m][:, idx].sum() / q0[m][:, idx].sum()
        r['all'] = qq[m].sum() / q0[m].sum()
        rows.append(r)
    print(p); print(pd.DataFrame(rows).set_index('graph').round(3).to_string())
# geometric version: median over systems of final/initial ratio per query group
print('\n=== median over systems of (final / initial) group error ratio')
for p in ['dpo_llm', 'target_risk', 'local_information', 'frozen_llm']:
    qq = perq(p, 32)
    rows = []
    for g in ['chain', 'fork', 'collider']:
        m = (meta.graph_stratum == g).values
        r = {'graph': g}
        for name, idx in grp.items():
            r[name] = np.median(qq[m][:, idx].sum(axis=1) / q0[m][:, idx].sum(axis=1))
        r['all'] = np.median(qq[m].sum(axis=1) / q0[m].sum(axis=1))
        rows.append(r)
    print(p); print(pd.DataFrame(rows).set_index('graph').round(3).to_string())
