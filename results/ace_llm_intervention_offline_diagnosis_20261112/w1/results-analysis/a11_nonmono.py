"""Non-monotonicity of the realized error curves and stability of first crossing."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
ini = curves[curves.step == 0].groupby('w').normalized_mse.first()
inf = ini > THRESH
rows = []; per_run = []
for p in POL:
    n_steps = rises = 0; ratios = []
    big2 = big10pct = 0
    cross_runs = leave_runs = final_out_after_cross = 0
    reentry = 0; n_exits = 0
    rises_above = 0; steps_from_below = 0   # transitions starting at/below target that end above
    lucky = 0  # first crossing immediately followed by an exit at the next evaluated state
    frac_after = []
    for w in range(24):
        r = curves[(curves.policy_id == p) & (curves.w == w)].sort_values('step')
        e = r.normalized_mse.values
        d = np.diff(e)
        n_steps += len(d); rises += (d > 0).sum()
        ratios += list(e[1:] / e[:-1])
        big2 += (e[1:] / e[:-1] > 2).sum(); big10pct += (e[1:] / e[:-1] > 1.1).sum()
        ok = e <= THRESH
        steps_from_below += ok[:-1].sum(); rises_above += (ok[:-1] & ~ok[1:]).sum()
        if ok.any():
            cross_runs += 1
            k = np.argmax(ok)
            later = ok[k:]
            left = (~later).any()
            leave_runs += left
            final_out_after_cross += (not ok[-1])
            n_exits += (later[:-1] & ~later[1:]).sum()
            if k + 1 < len(ok) and not ok[k + 1]:
                lucky += 1
            frac_after.append(later.mean())
        per_run.append({'w': w, 'policy': SHORT[p], 'steps': len(d), 'rises': int((d > 0).sum()), 'ever': bool(ok.any()),
                        'left_after_cross': bool(ok.any() and (~ok[np.argmax(ok):]).any()), 'final_ok': bool(ok[-1]), 'informative': bool(inf[w])})
    ratios = np.array(ratios)
    rows.append({'policy': SHORT[p], 'acquisitions': n_steps, 'error_rose': int(rises), 'rose_share': rises / n_steps,
                 'rose_gt10pct': int(big10pct), 'rose_gt2x': int(big2), 'median_ratio_when_rising': np.median(ratios[ratios > 1]),
                 'runs_crossed': cross_runs, 'crossed_then_left': int(leave_runs), 'crossed_but_final_above': int(final_out_after_cross),
                 'exit_events': int(n_exits), 'first_cross_lost_next_step': int(lucky),
                 'mean_share_of_later_states_at_target': np.mean(frac_after),
                 'P(leave | at target, one more response)': rises_above / steps_from_below})
R = pd.DataFrame(rows).set_index('policy')
print(R.to_string(float_format=lambda x: '%.3f' % x))
R.to_csv('nonmonotonicity.csv')
print('\nTOTAL: error rose after %d of %d accepted acquisitions (%.1f%%); crossed-then-left in %d of %d runs that ever crossed; final above target after crossing in %d'
      % (R.error_rose.sum(), R.acquisitions.sum(), 100 * R.error_rose.sum() / R.acquisitions.sum(), R.crossed_then_left.sum(), R.runs_crossed.sum(), R.crossed_but_final_above.sum()))
PR = pd.DataFrame(per_run)
print('\ninformative systems only (14 x 8 = 112 runs):')
g = PR[PR.informative].groupby('policy')
print(pd.DataFrame({'ever': g.ever.sum(), 'left_after_cross': g.left_after_cross.sum(), 'final_ok': g.final_ok.sum(),
                    'rise_share': g.rises.sum() / g.steps.sum()}).loc[[SHORT[p] for p in POL]].to_string(float_format=lambda x: '%.3f' % x))
print('\nsystems at target after initialization (10 x 8 = 80 runs): runs that later left the target at least once, and runs whose final state is above target')
g = PR[~PR.informative].groupby('policy')
print(pd.DataFrame({'left_after_cross': g.left_after_cross.sum(), 'final_above': g.final_ok.apply(lambda s: (~s).sum()),
                    'rise_share': g.rises.sum() / g.steps.sum()}).loc[[SHORT[p] for p in POL]].to_string(float_format=lambda x: '%.3f' % x))
tot = PR[~PR.informative]
print('of the 80 runs on the 10 already-at-target systems: %d left the target at some later point; %d ended above target' % (tot.left_after_cross.sum(), (~tot.final_ok).sum()))
# by world: on already-attained systems, which worlds see exits
print(PR[~PR.informative].groupby('w').agg(left=('left_after_cross', 'sum'), final_above=('final_ok', lambda s: (~s).sum())).join(ini.rename('init_err')).join(meta[['graph_stratum', 'regime', 'cost_stratum']]).to_string())

# does error at init crossing reflect luck: distribution of error ratio at step 1
e1 = curves[curves.step == 1].set_index(['w', 'policy_id']).normalized_mse
e0 = curves[curves.step == 0].set_index(['w', 'policy_id']).normalized_mse
ratio = (e1 / e0)
print('\nfirst acquisition: error rose in %d of %d runs (%.1f%%); median ratio %.3f; 10th-90th pct %.3f-%.3f'
      % ((ratio > 1).sum(), len(ratio), 100 * (ratio > 1).mean(), ratio.median(), ratio.quantile(.1), ratio.quantile(.9)))
print(ratio.groupby('policy_id').apply(lambda s: pd.Series({'rose': (s > 1).sum(), 'median_ratio': s.median()})).unstack().loc[POL].to_string())
