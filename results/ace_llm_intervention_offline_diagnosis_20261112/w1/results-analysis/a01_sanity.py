import numpy as np, pandas as pd, json, ast
from common import *
cells, curves, meta = load()
pol = pd.read_csv(D + 'policies.csv').set_index('policy_id')
g = cells.groupby('policy_id')
rep = pd.DataFrame({
    'mean_capped_total_cost': g.capped_total_cost.mean(),
    'mean_capped_acq_cost': g.capped_acquisition_cost.mean(),
    'ever': g.ever_attained.sum(), 'final': g.final_attained.sum(),
    'mean_final_error': g.final_error.mean(), 'median_final_error': g.final_error.median(),
    'actual_total_cost': g.actual_total_cost.sum(),
    'stopped': g.termination_reason.apply(lambda s: (s == 'policy_stop').sum()),
    'failed': g.failed.sum(), 'unresolved': g.unresolved.sum()}).loc[POL]
print(rep.to_string(float_format=lambda x: '%.5f' % x))
# compare with policies.csv
chk = {
 'cost': np.allclose(rep.mean_capped_total_cost, pol.loc[POL].mean_capped_total_cost),
 'acq': np.allclose(rep.mean_capped_acq_cost, pol.loc[POL].mean_capped_acquisition_cost),
 'ever': (rep.ever.values == pol.loc[POL].ever_attained.values).all(),
 'final': (rep.final.values == pol.loc[POL].final_attained.values).all(),
 'ferr': np.allclose(rep.mean_final_error, pol.loc[POL].mean_final_error, rtol=1e-12, atol=0),
 'actual': (rep.actual_total_cost.values == pol.loc[POL].actual_total_cost.values).all(),
 'stopped': (rep.stopped.values == pol.loc[POL].stopped.values).all()}
print('policies.csv reproduced from cells.csv:', chk)

# cells vs curves consistency
bad = 0
for (w, p), run in curves.groupby(['w', 'policy_id']):
    run = run.sort_values('step')
    c = cells[(cells.w == w) & (cells.policy_id == p)].iloc[0]
    e = run.normalized_mse.values; tc = run.total_cost_units.values
    cross = tc[e <= THRESH]
    first = cross[0] if len(cross) else None
    capped = first if first is not None else CAP
    ok = (capped == c.capped_total_cost) and (abs(e[-1] - c.final_error) < 1e-15) \
        and ((first is not None) == bool(c.ever_attained)) and ((e[-1] <= THRESH) == bool(c.final_attained)) \
        and (tc[-1] == c.actual_total_cost) and (tc[0] == 8) and (run.acquisition_cost_units.values[0] == 0) \
        and (run.accepted_receipts.values[0] == 8) and pd.isna(run.action_id.values[0]) \
        and (np.diff(tc) > 0).all() and (np.diff(run.accepted_receipts.values) == 1).all() \
        and (run.total_cost_units.values - run.acquisition_cost_units.values == 8).all()
    if first is not None:
        ok = ok and (c.first_crossing_total == first)
    else:
        ok = ok and pd.isna(c.first_crossing_total)
    bad += (not ok)
print('runs with any cells/curves inconsistency:', bad, 'of', curves.groupby(['w', 'policy_id']).ngroups)

# cost increments vs menu
cm = {'uniform': {a: 1 for a in ACTIONS}, 'heterogeneous': {a: (2 if i >= 4 else 1) for i, a in enumerate(ACTIONS)}}
bad = 0
for (w, p), run in curves.groupby(['w', 'policy_id']):
    run = run.sort_values('step')
    inc = np.diff(run.total_cost_units.values)
    acts = run.action_id.values[1:]
    exp = np.array([cm[meta.loc[w, 'cost_stratum']][a] for a in acts])
    bad += int((inc != exp).any())
print('runs whose cost increments disagree with the declared menu costs:', bad)

# NMSE recomputation needs private truth; check init predictions are identical across arms
ini = curves[curves.step == 0]
print('init error identical across 8 arms in every world:', ini.groupby('w').normalized_mse.nunique().eq(1).all(),
      '; init predictions identical:', all(ini.groupby('w')[c].nunique().eq(1).all() for c in PRED))

# action counts vs actions.csv
ac = curves[curves.step > 0].groupby(['policy_id', 'action_id']).size()
ref = pd.read_csv(D + 'actions.csv').set_index(['policy_id', 'action_id']).accepted_acquisitions
print('actions.csv reproduced:', ac.sort_index().equals(ref.sort_index().astype('int64')) or (ac.sort_index().values == ref.sort_index().values).all())
print('acquisitions per policy:', curves[curves.step > 0].groupby('policy_id').size().loc[POL].to_dict())
print('total curve points', len(curves), '; accepted responses incl init:', 192 * 8 + (curves.step > 0).sum(),
      '; total paid cost', cells.actual_total_cost.sum())

# pairs.csv
pairs = pd.read_csv(D + 'pairs.csv')
wide = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')[POL]
bad = 0
for _, r in pairs.iterrows():
    d = (wide[r.left] - wide[r.right]).values
    ok = (list(d) == ast.literal_eval(r.differences)) and abs(d.mean() - r.mean_difference) < 1e-12 \
        and (d < 0).sum() == r.wins and (d == 0).sum() == r.ties and (d > 0).sum() == r.losses
    bad += (not ok)
print('pairs.csv rows not reproduced:', bad, 'of', len(pairs), '(difference = left - right; wins = left lower)')
