"""Per-query absolute contributions and median retained ratios (robustness view for section 7.1)."""
import numpy as np, pandas as pd, json
from common import *
cells, curves, meta = load()
T = np.load('truth_means.npy')
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
def perq(p, cost):
    out = np.empty((24, 8))
    for w in range(24):
        r = curves[(curves.policy_id == p) & (curves.w == w)].sort_values('step')
        k = np.searchsorted(r.total_cost_units.values, cost, side='right') - 1
        out[w] = ((r[PRED].values[k] - T[w]) / 2) ** 2 / 8
    return out
QN = ['q0 obs', 'q1 X=-1', 'q2 X=0', 'q3 X=+1', 'q4 Y=-1', 'q5 Y=0', 'q6 Y=+1', 'q7 X=-1,Y=+1']
q0 = perq('dpo_llm', 8)
rows = []
for label, arr in [('initial (all arms)', q0)] + [(SHORT[p], perq(p, 32)) for p in ['dpo_llm', 'sft_llm', 'learned_ranker', 'frozen_llm', 'target_risk', 'local_information', 'menu_coverage', 'uniform_random']]:
    for g in ['chain', 'fork', 'collider']:
        m = (meta.graph_stratum == g).values
        r = {'state': label, 'graph': g}
        for i, n in enumerate(QN): r[n] = arr[m][:, i].mean() * 1000
        r['total'] = arr[m].sum(axis=1).mean() * 1000
        # queries whose true value depends on the natural X mean: q0 always; q4-q6 in fork and collider
        dep = [0] if g == 'chain' else [0, 4, 5, 6]
        r['X-mean-dependent share'] = arr[m][:, dep].sum() / arr[m].sum()
        rows.append(r)
A = pd.DataFrame(rows).set_index('state')
TAB['perq_abs'] = md(A, '%.2f')
grp = {'q1,q7 (X=-1)': [1, 7], 'q2,q3 (X=0,+1)': [2, 3], 'q0 (observe)': [0], 'q4-q6 (do Y)': [4, 5, 6]}
rows = []
for p in ['dpo_llm', 'sft_llm', 'learned_ranker', 'frozen_llm', 'target_risk', 'local_information', 'menu_coverage', 'uniform_random']:
    qq = perq(p, 32)
    for g in ['chain', 'fork', 'collider']:
        m = (meta.graph_stratum == g).values
        r = {'policy': SHORT[p], 'graph': g}
        for name, idx in grp.items(): r[name] = np.median(qq[m][:, idx].sum(axis=1) / q0[m][:, idx].sum(axis=1))
        r['all queries'] = np.median(qq[m].sum(axis=1) / q0[m].sum(axis=1))
        rows.append(r)
TAB['retained_median'] = md(pd.DataFrame(rows).set_index('policy'))
json.dump(TAB, open('tables.json', 'w'))
print(TAB['perq_abs']); print(TAB['retained_median'])
# which systems dominate DPO's q0 error by graph
q = perq('dpo_llm', 32)
for g in ['chain', 'fork', 'collider']:
    m = np.where((meta.graph_stratum == g).values)[0]
    dep = [0] if g == 'chain' else [0, 4, 5, 6]
    v = q[m][:, dep].sum(axis=1)
    print(g, 'X-mean-dependent final error by system (x1e-3):', {int(w): round(float(x) * 1000, 2) for w, x in zip(m, v)}, 'top-2 share %.2f' % (np.sort(v)[-2:].sum() / v.sum()))
