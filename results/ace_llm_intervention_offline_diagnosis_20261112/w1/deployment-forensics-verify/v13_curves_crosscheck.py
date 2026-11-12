"""Cross-check the decision table against the independent reporting CSVs, and recompute NMSE from saved public states
   plus the private mechanisms stored in campaign.json. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np, pandas as pd
D = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01'
cur = pd.read_csv(D + '/curves.csv'); cells = pd.read_csv(D + '/cells.csv')
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json'))
print('curve points', len(cur), 'cells', len(cells))
# 1. action sequence agreement
seqs = collections.defaultdict(dict)
for r in rows:
    if r['action'] != 'STOP': seqs[(r['world_id'], r['arm'])][r['d']] = r['action']
mism = 0; n = 0
for (wid, arm), g in cur.groupby(['world_id', 'policy_id']):
    g = g.sort_values('accepted_receipts')
    acts = [a for a in g['action_id'].tolist() if isinstance(a, str)]
    mine = [seqs[(wid, arm)][d] for d in sorted(seqs[(wid, arm)])]
    n += 1
    if acts != mine: mism += 1; print('MISMATCH', wid, arm, acts[:5], mine[:5])
print('cells compared', n, 'action-sequence mismatches', mism)
# 2. NMSE recomputation
def compose(mech, graph, clamps):
    vals = dict(clamps)
    for node in graph:
        name = node['name']
        if name in vals: continue
        m = mech[name]
        vals[name] = m[0] + sum(c * vals[p] for p, c in zip(node['parents'], m[1:]))
    return vals
def nmse(world, env):
    graph = world['graph']; task = env['task']
    true = {m['node']: [m['intercept']] + [dict(m['coefficients'])[p] for p in next(n for n in graph if n['name'] == m['node'])['parents']] for m in world['mechanisms']}
    est = {s['node']: s['parameter_mean'] for s in env['state']['scm']}
    tot = 0.0
    for q, w in zip(task['queries'], task['weights']):
        cl = {k: v for k, v in q['targets']}
        # observe query: no clamps -> X is its (noise-disabled) intercept
        zt = compose(true, graph, cl)[task['target']]; ze = compose(est, graph, cl)[task['target']]
        tot += w * ((ze - zt) / task['scale']) ** 2
    return tot
cur_idx = {(int(r.world_id), r.policy_id, int(r.accepted_receipts)): r.normalized_mse for r in cur.itertuples()}
maxdiff = 0.0; cnt = 0
for r in rows:
    world = camp['sources'][r['world']]['plan']['world']
    v = nmse(world, r['envelope'])
    r['nmse'] = v
    key = (r['world_id'], r['arm'], r['envelope']['state']['requests'])
    if key in cur_idx:
        maxdiff = max(maxdiff, abs(v - cur_idx[key])); cnt += 1
    else:
        print('no curve point for', key)
print('NMSE recomputed for', cnt, 'states; max abs diff vs curves.csv', maxdiff)
pickle.dump({(r['arm'], r['world'], r['d']): r['nmse'] for r in rows}, open(SCR + '/nmse_by_state.pkl', 'wb'))
# 3. SFT stopped runs above target
s = cells[cells.policy_id == 'sft_llm']
print('SFT termination reasons', dict(s.termination_reason.value_counts()), '| stopped & final_error>0.04:', int(((s.termination_reason != 'budget_exhausted') & (s.final_error > 0.04)).sum()),
      '| worlds', s[(s.termination_reason != 'budget_exhausted') & (s.final_error > 0.04)].world_id.tolist())
print('SFT actual_total_cost sum', int(s.actual_total_cost.sum()))
# 4. NMSE change after single acquisitions
for arm in ARMS + ['ALL']:
    g = cur if arm == 'ALL' else cur[cur.policy_id == arm]
    up = 0; tot = 0; eq = 0
    for (wid, a), gg in g.groupby(['world_id', 'policy_id']):
        v = gg.sort_values('accepted_receipts').normalized_mse.values
        d = np.diff(v); up += (d > 0).sum(); tot += len(d); eq += (d == 0).sum()
    print('%-18s acquisitions %4d NMSE increased after %4d (%.1f%%)' % (arm, tot, up, 100 * up / tot))
# 5. ten systems met target after initialisation
init = cur[cur.accepted_receipts == 8].groupby('world_id').normalized_mse.first()
print('worlds with init NMSE<=0.04:', int((init <= 0.04).sum()), sorted((init[init <= 0.04].index - 202611070100).tolist()))
print('init NMSE per world', [round(x, 4) for x in init.values])
