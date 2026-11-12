import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, csv
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb')); ev = pickle.load(open(SCR + '/ev_ref.pkl', 'rb')); nm = pickle.load(open(SCR + '/nmse_by_state.pkl', 'rb'))
ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
with open(SCR + '/decisions_verifier.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['arm', 'world', 'world_id', 'regime', 'graph', 'cost_menu', 'decision_index', 'remaining_units', 'action', 'envelope_sha256', 'private_nmse_before',
                'posterior_risk_before', 'tr_ref_best_action', 'tr_ref_best_score', 'tr_ref_chosen_score', 'tr_ref_rel_regret', 'tr_ref_rank_of_chosen', 'tr_ref_rank_of_set_x_neg', 'n_affordable'] +
               ['tr_ref_score_' + a for a in ACTIONS])
    for r, e in zip(rows, ev):
        assert r['arm'] == e['arm'] and r['world'] == e['world'] and r['d'] == e['d']
        s = strata[r['world']]
        w.writerow([r['arm'], r['world'], r['world_id'], s['regime'], s['graph_stratum'], s['cost_stratum'], r['d'], r['envelope']['state']['remaining_units'], r['action'], r['envelope_sha256'],
                    nm[(r['arm'], r['world'], r['d'])], ref[r['envelope_sha256']]['before'], e['best'], e['best_score'], e['chosen_score'], e['rel'], e['rank'], e['xneg_rank'], e['n']] +
                   [e['scores'].get(a, '') for a in ACTIONS])
print('wrote', SCR + '/decisions_verifier.csv', len(rows))
