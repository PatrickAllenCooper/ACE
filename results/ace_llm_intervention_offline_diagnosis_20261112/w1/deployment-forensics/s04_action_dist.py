"""Action distributions per arm; consistency checks against the documents and the
actions.csv / cost-report curves."""
import pickle, collections, json, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
ACTIONS = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy', 'STOP']
print('%-18s %5s | ' % ('arm', 'n') + ' '.join('%10s' % a for a in ACTIONS))
tab = {}
for arm in pids:
    c = collections.Counter(r['action'] for r in rows if r['arm'] == arm)
    tab[arm] = c
    n = sum(c.values())
    print('%-18s %5d | ' % (arm, n) + ' '.join('%10d' % c.get(a, 0) for a in ACTIONS))
print()
print('acquired (non-STOP) totals:', {a: sum(v for k, v in tab[a].items() if k != 'STOP') for a in pids})
# per graph / cost
for key in ['graph', 'cost', 'regime']:
    print('\n--- by', key)
    for arm in pids:
        for val in sorted(set(w[key] for w in worlds)):
            c = collections.Counter(r['action'] for r in rows if r['arm'] == arm and worlds[r['world_idx']][key] == val)
            print('%-18s %-14s n=%4d | ' % (arm, val, sum(c.values())) + ' '.join('%s=%d' % (a, c[a]) for a in ACTIONS if c.get(a)))
# cross-check with cost-report curves
REP = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque/cpu/cpu/worker/artifacts/campaign/private'
mism = 0; total = 0
reports = {}
for wi in range(24):
    for ai, arm in enumerate(pids):
        rep = json.load(open('%s/world-%02d/arm-%02d/cost-report.json' % (REP, wi, ai)))
        assert rep['policy_id'] == arm and rep['world_id'] == worlds[wi]['world_id']
        reports[(arm, wi)] = rep
        seq_rep = [p['action_id'] for p in rep['curve'] if p['kind'] != 'shared_initial']
        seq = [r['action'] for r in sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == wi), key=lambda r: r['decision_index']) if r['action'] != 'STOP']
        total += 1
        if seq != seq_rep:
            mism += 1
            print('MISMATCH', arm, wi, seq, seq_rep)
print('\ncost-report action sequence cross-check: cells', total, 'mismatches', mism)
pickle.dump(reports, open(HERE + '/out/reports.pkl', 'wb'))
# actions.csv
p = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/actions.csv'
print(open(p).read()[:3000])
