"""Endpoint noise floor: SFT and DPO issue identical action prefixes (all x-) but receive independent measurement streams.
   Any endpoint difference decided inside the common prefix is measurement noise alone. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np, pandas as pd
D = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01'
cur = pd.read_csv(D + '/curves.csv'); cells = pd.read_csv(D + '/cells.csv')
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
seq = collections.defaultdict(dict)
for r in rows: seq[(r['arm'], r['world'])][r['d']] = r
# do arms with the same first action get different measurements?
w = 0
for a in ['menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'sft_llm', 'dpo_llm']:
    r1 = seq[(a, w)].get(1)
    h = r1['envelope']['state']['history'][8]
    print('world 0', a, 'first acquisition', h['action_id'], {k: round(v, 4) for k, v in h['values'].items()})
cell = {(int(r.world_id) - 202611070100, r.policy_id): r for r in cells.itertuples()}
def nm_curve(arm, w):
    g = cur[(cur.world_id == 202611070100 + w) & (cur.policy_id == arm)].sort_values('accepted_receipts')
    return g.normalized_mse.values, g.total_cost_units.values
print('\nworld | common x- prefix (acquisitions) | SFT capped, DPO capped | crossing inside common prefix? | NMSE after 10 acquisitions SFT vs DPO')
diff_in_prefix = 0; diffs = 0; n10 = []
for w in range(24):
    s = [seq[('sft_llm', w)][d]['action'] for d in sorted(seq[('sft_llm', w)])]; dp = [seq[('dpo_llm', w)][d]['action'] for d in sorted(seq[('dpo_llm', w)])]
    k = 0
    while k < min(len(s), len(dp)) and s[k] == dp[k]: k += 1
    cs, cd = cell[(w, 'sft_llm')], cell[(w, 'dpo_llm')]
    ns, ts = nm_curve('sft_llm', w); nd, td = nm_curve('dpo_llm', w)
    # index of first crossing (number of acquisitions) or None
    fs = next((i for i, v in enumerate(ns) if v <= 0.04), None); fd = next((i for i, v in enumerate(nd) if v <= 0.04), None)
    inside = (fs is not None and fs <= k) or (fd is not None and fd <= k)
    differ = cs.capped_total_cost != cd.capped_total_cost
    diffs += differ
    decided_by_noise = differ and ((fs is not None and fs <= k) != (fd is not None and fd <= k) or (fs is not None and fd is not None and fs <= k and fd <= k))
    diff_in_prefix += bool(decided_by_noise)
    n10.append((ns[10], nd[10]))
    print('%2d %-8s %-4s | prefix %2d | %2d %2d | first crossing idx SFT %s DPO %s | %s | %.4f %.4f' % (w, strata[w]['graph_stratum'], strata[w]['cost_stratum'][:4], k, cs.capped_total_cost, cd.capped_total_cost, fs, fd,
          'DIFFER-in-prefix' if decided_by_noise else ('differ' if differ else 'tie'), ns[10], nd[10]))
print('worlds where SFT and DPO capped cost differ:', diffs, '| of these, decided by a crossing inside the identical-action prefix:', diff_in_prefix)
n10 = np.array(n10)
print('NMSE after 10 identical x- acquisitions: SFT mean %.4f DPO mean %.4f ; mean |diff| %.4f ; correlation %.3f ; worlds where exactly one is <=0.04: %d' % (
    n10[:, 0].mean(), n10[:, 1].mean(), np.abs(n10[:, 0] - n10[:, 1]).mean(), np.corrcoef(n10[:, 0], n10[:, 1])[0, 1], int(((n10[:, 0] <= 0.04) != (n10[:, 1] <= 0.04)).sum())))
# also menu coverage vs DPO vs SFT after the first acquisition (all three take x- first)
a1 = np.array([[nm_curve(a, w)[0][1] for a in ['menu_coverage', 'sft_llm', 'dpo_llm']] for w in range(24)])
print('NMSE after the first (identical x-) acquisition: mean abs pairwise difference %.4f ; mean level %.4f' % (np.mean([abs(a1[:, i] - a1[:, j]).mean() for i, j in [(0, 1), (0, 2), (1, 2)]]), a1.mean()))
