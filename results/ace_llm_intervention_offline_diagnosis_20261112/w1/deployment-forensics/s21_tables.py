"""Compact tables for the notes: SFT stop table; first-decision table with both
saved 32-draw and recomputed 256-draw target-risk scores."""
import pickle, collections, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TR = pickle.load(open(HERE + '/out/tr_scores_256.pkl', 'rb'))
reports = pickle.load(open(HERE + '/out/reports.pkl', 'rb'))
SH = {'observe_all': 'obs', 'set_x_neg': 'x-', 'set_x_zero': 'x0', 'set_x_pos': 'x+', 'set_y_neg': 'y-', 'set_y_zero': 'y0', 'set_y_pos': 'y+', 'set_xy': 'xy', 'STOP': 'STOP'}
print('SFT stop table: world | graph | cost | regime | stop decision index | requests | spent | remaining | legal cmds | prompt tokens | posterior risk at stop | private final NMSE | final target met')
for r in sorted((r for r in rows if r['arm'] == 'sft_llm' and r['action'] == 'STOP'), key=lambda r: r['world_idx']):
    st = r['envelope']['state']; W = worlds[r['world_idx']]; rep = reports[('sft_llm', r['world_idx'])]
    print('| %2d | %-8s | %-13s | %-12s | %2d | %2d | %2d | %d | %d | %4d | %.4f | %.4f | %s |' % (r['world_idx'], W['graph'], W['cost'], W['regime'], r['decision_index'], st['requests'], st['spent_units'], st['remaining_units'], r['legal_commands'], r['prompt_tokens'], TR[r['envelope_sha256']]['risk_before'], rep['final_error'], rep['final_attained']))
print('\nFirst-decision table (shared state). s32 = saved deployed scores (planning_metadata.risk_reduction_per_cost); s256 = recomputed.')
print('| w | graph | cost | regime | NMSE0 | TR-opt (s32) | TR-opt (s256) | gap best-2nd s256 | in s.e. | gap best - x_neg s256 | rank x_neg s256 | unif | menu | local | TR | ranker | frozen | sft | dpo |')
first = {(r['arm'], r['world_idx']): r for r in rows if r['decision_index'] == 0}
for w in range(24):
    r = first[('target_risk', w)]; s32 = r['meta']['planning_metadata']['risk_reduction_per_cost']; t = TR[r['envelope_sha256']]; s = t['scores']
    o32 = sorted(s32, key=lambda a: (-s32[a], a)); o = sorted(s, key=lambda a: (-s[a], a)); W = worlds[w]
    z = (s[o[0]] - s[o[1]]) / math.hypot(t['se'][o[0]], t['se'][o[1]])
    print('| %2d | %-8s | %-5s | %-12s | %.4f | %-3s | %-3s | %.5f | %5.1f | %.5f | %d | %s |' % (w, W['graph'], W['cost'][:5], W['regime'], reports[('target_risk', w)]['curve'][0]['normalized_mse'], SH[o32[0]], SH[o[0]], s[o[0]] - s[o[1]], z, s[o[0]] - s['set_x_neg'], o.index('set_x_neg') + 1, ' | '.join(SH[first[(p, w)]['action']] for p in pids)))
