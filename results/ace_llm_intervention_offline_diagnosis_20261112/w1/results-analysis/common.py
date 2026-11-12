"""Shared loaders for the read-only re-analysis of the saved 24-system comparison.
Reads only the saved CSVs. Writes nothing outside this scratch directory."""
import numpy as np, pandas as pd

D = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/'
POL = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk',
       'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']
SHORT = {'uniform_random': 'random', 'menu_coverage': 'coverage', 'local_information': 'local_info',
         'target_risk': 'target_risk', 'learned_ranker': 'ranker', 'frozen_llm': 'frozen',
         'sft_llm': 'sft', 'dpo_llm': 'dpo'}
THRESH = 0.04
INIT, CAP = 8, 32
PRED = ['prediction_%d' % i for i in range(8)]
ACTIONS = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
ASHORT = {'observe_all': 'obs', 'set_x_neg': 'x-', 'set_x_zero': 'x0', 'set_x_pos': 'x+',
          'set_y_neg': 'y-', 'set_y_zero': 'y0', 'set_y_pos': 'y+', 'set_xy': 'xy'}


def load():
    cells = pd.read_csv(D + 'cells.csv')
    curves = pd.read_csv(D + 'curves.csv')
    cells['w'] = cells.world_id - 202611070100
    curves['w'] = curves.world_id - 202611070100
    curves = curves.reset_index().rename(columns={'index': 'row'})
    # step index inside each run, in file order (0 = post-initialization evaluation)
    curves['step'] = curves.groupby(['world_id', 'policy_id']).cumcount()
    meta = cells[['w', 'world_id', 'cost_stratum', 'graph_stratum', 'regime']].drop_duplicates().set_index('w').sort_index()
    meta['noise'] = meta.regime.map(lambda r: 'low' if r.endswith('low') else 'high')
    meta['prior'] = meta.regime.map(lambda r: r.split('_')[0])
    return cells, curves, meta


def step_curve(run):
    """Error as a right-continuous step function of total paid cost, on integer costs 8..32.
    e(c) = error after the last accepted response with total_cost_units <= c.
    After a policy stop the last SCM state is carried forward (the model does not change)."""
    run = run.sort_values('step')
    costs = run.total_cost_units.values
    errs = run.normalized_mse.values
    out = np.empty(CAP - INIT + 1)
    for i, c in enumerate(range(INIT, CAP + 1)):
        k = np.searchsorted(costs, c, side='right') - 1
        out[i] = errs[k]
    return out


def wide_steps(curves):
    """dict policy -> array [24 worlds, 25 integer costs 8..32] of step-function errors."""
    res = {}
    for p in POL:
        arr = np.empty((24, CAP - INIT + 1))
        for w in range(24):
            arr[w] = step_curve(curves[(curves.policy_id == p) & (curves.w == w)])
        res[p] = arr
    return res


def signflip_exact(d, tol=1e-12):
    """Exact two-sided sign-flip (paired randomization) p-value for the mean of d.
    Zeros are kept (they contribute nothing to the sum, so p is the same with or without them).
    Enumerates all 2^k sign patterns of the k nonzero differences by meet-in-the-middle."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 0]
    k = len(d)
    if k == 0:
        return 1.0
    obs = abs(d.sum())
    h = k // 2
    def sums(x):
        s = np.zeros(1)
        for v in x:
            s = np.concatenate([s + v, s - v])
        return s
    a, b = sums(d[:h]), np.sort(sums(d[h:]))
    thr = obs * (1 - 1e-12) - tol
    # count pairs with |a+b| >= thr
    hi = len(b) - np.searchsorted(b, thr - a, side='left')      # b >= thr - a
    lo = np.searchsorted(b, -thr - a, side='right')              # b <= -thr - a
    cnt = hi.sum() + lo.sum()
    if thr <= 0:
        return 1.0
    return float(cnt) / (len(a) * len(b))
