"""The supervised arm: where and when it stopped, and the error at stopping."""
import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
S = wide_steps(curves)
c = cells[cells.policy_id == 'sft_llm'].set_index('w')
ini = curves[curves.step == 0].groupby('w').normalized_mse.first()
rows = []
for w in range(24):
    r = curves[(curves.policy_id == 'sft_llm') & (curves.w == w)].sort_values('step')
    acts = list(r.action_id.values[1:])
    e = r.normalized_mse.values
    other = {p: S[p][w, -1] for p in POL}
    rows.append({'w': w, 'graph': meta.graph_stratum[w], 'regime': meta.regime[w], 'cost': meta.cost_stratum[w],
                 'term': c.termination_reason[w], 'stop_total_cost': int(c.actual_total_cost[w]), 'unspent': 32 - int(c.actual_total_cost[w]),
                 'acquisitions': len(acts), 'init_err': ini[w], 'err_at_stop': e[-1], 'at_target_at_stop': e[-1] <= THRESH,
                 'ever': bool(c.ever_attained[w]), 'capped_cost': int(c.capped_total_cost[w]),
                 'min_err_in_run': e.min(), 'dpo_err_at_same_cost': S['dpo_llm'][w, int(c.actual_total_cost[w]) - 8],
                 'tr_err_at_same_cost': S['target_risk'][w, int(c.actual_total_cost[w]) - 8], 'tr_final': S['target_risk'][w, -1],
                 'non_xneg_actions': ','.join('%d:%s' % (i + 1, ASHORT[a]) for i, a in enumerate(acts) if a != 'set_x_neg'),
                 'last_action': ASHORT[acts[-1]]})
T = pd.DataFrame(rows).set_index('w')
print(T.to_string(float_format=lambda x: '%.4f' % x))
T.to_csv('sft_stops.csv')
st = T[T.term == 'policy_stop']
print('\nstopped runs: %d ; not stopped: %s' % (len(st), list(T.index[T.term != 'policy_stop'])))
print('stop total cost distribution:', st.stop_total_cost.value_counts().sort_index().to_dict())
print('unspent units: total %d (= 768 - %d) ; distribution %s' % (st.unspent.sum(), T.stop_total_cost.sum(), st.unspent.value_counts().sort_index().to_dict()))
print('acquisition count at stop:', st.acquisitions.value_counts().sort_index().to_dict())
print('stopped above target: %d systems %s' % ((~st.at_target_at_stop).sum(), list(st.index[~st.at_target_at_stop])))
print(st[~st.at_target_at_stop][['graph', 'regime', 'cost', 'stop_total_cost', 'init_err', 'err_at_stop', 'ever', 'capped_cost', 'tr_final']].to_string(float_format=lambda x: '%.4f' % x))
print('of the 5 stopped-above-target systems, ever attained earlier:', int(st[~st.at_target_at_stop].ever.sum()))
print('stop cost by cost menu:'); print(st.groupby('cost').stop_total_cost.agg(['mean', 'min', 'max', 'count']).to_string())
print('stop cost by graph:'); print(st.groupby('graph').stop_total_cost.agg(['mean', 'min', 'max', 'count']).to_string())
print('acquisitions at stop by cost menu:'); print(st.groupby('cost').acquisitions.agg(['mean', 'min', 'max']).to_string())
print('error at stop: geometric mean %.5f ; median %.5f ; mean %.5f ; stopped & at target %d of %d'
      % (10 ** np.log10(st.err_at_stop).mean(), st.err_at_stop.median(), st.err_at_stop.mean(), st.at_target_at_stop.sum(), len(st)))
print('correlation of stop cost with log10 init error: %.3f ; with log10 error at stop: %.3f'
      % (np.corrcoef(st.stop_total_cost, np.log10(st.init_err))[0, 1], np.corrcoef(st.stop_total_cost, np.log10(st.err_at_stop))[0, 1]))
print('mean log10 ratio (sft error at stop / target_risk error at the same paid cost): %+.3f ; (sft at stop / target_risk final at 32): %+.3f'
      % (np.log10(st.err_at_stop / st.tr_err_at_same_cost).mean(), np.log10(st.err_at_stop / st.tr_final).mean()))
# is the stop triggered by acquisition count rather than state? distribution of step index of stop by menu
print('\nsequence of SFT action classes, all 24 runs:')
for w in range(24):
    r = curves[(curves.policy_id == 'sft_llm') & (curves.w == w)].sort_values('step')
    print('  w%02d %-8s %-13s %-12s cost@stop=%2d  %s  [%s]' % (w, meta.graph_stratum[w], meta.cost_stratum[w], meta.regime[w], c.actual_total_cost[w],
          ' '.join(ASHORT[a] for a in r.action_id.values[1:]), c.termination_reason[w]))
