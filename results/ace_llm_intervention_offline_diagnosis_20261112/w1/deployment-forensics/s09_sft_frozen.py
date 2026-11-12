"""Supervised (SFT) arm: the 42 non-set_x_neg acquisitions and the 23 stops.
Frozen arm: the 2 non-observe actions."""
import pickle, collections, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
reports = pickle.load(open(HERE + '/out/reports.pkl', 'rb'))
SH = {'observe_all': 'O', 'set_x_neg': 'N', 'set_x_zero': 'Z', 'set_x_pos': 'P', 'set_y_neg': 'a', 'set_y_zero': 'b', 'set_y_pos': 'c', 'set_xy': 'J', 'STOP': '#'}
print('legend:', SH)
def seq(arm, w):
    return sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == w), key=lambda r: r['decision_index'])
for arm in ['sft_llm', 'frozen_llm', 'dpo_llm']:
    print('\n=====', arm, 'per-world action strings (decision order)')
    for w in range(24):
        s = seq(arm, w); W = worlds[w]; rep = reports[(arm, w)]
        st_last = s[-1]['envelope']['state']
        print('  %2d %-8s %-5s %-12s n_dec=%2d %s  | last-decision remaining=%2d  final_nmse=%.4f final_attained=%s paid=%d' % (
            w, W['graph'], W['cost'][:5], W['regime'], len(s), ''.join(SH[r['action']] for r in s), st_last['remaining_units'],
            rep['final_error'], rep['final_attained'], rep['charges']['actual_total_cost_units']))
S = [r for r in rows if r['arm'] == 'sft_llm']
print('\n===== SFT: stops')
stops = [r for r in S if r['action'] == 'STOP']
print('n stops', len(stops))
print('decision_index of stop:', sorted(collections.Counter(r['decision_index'] for r in stops).items()))
print('remaining_units at stop:', sorted(collections.Counter(r['envelope']['state']['remaining_units'] for r in stops).items()))
print('requests (history length) at stop:', sorted(collections.Counter(r['envelope']['state']['requests'] for r in stops).items()))
print('spent_units at stop:', sorted(collections.Counter(r['envelope']['state']['spent_units'] for r in stops).items()))
print('legal_commands at stop:', sorted(collections.Counter(r['legal_commands'] for r in stops).items()))
print('prompt_tokens at stop: min %d median %d max %d' % (min(r['prompt_tokens'] for r in stops), int(np.median([r['prompt_tokens'] for r in stops])), max(r['prompt_tokens'] for r in stops)))
print('stops by graph:', dict(collections.Counter(worlds[r['world_idx']]['graph'] for r in stops)), 'by cost:', dict(collections.Counter(worlds[r['world_idx']]['cost'] for r in stops)), 'by regime:', dict(collections.Counter(worlds[r['world_idx']]['regime'] for r in stops)))
nostop = [w for w in range(24) if not any(r['world_idx'] == w for r in stops)]
print('world(s) without a policy stop:', nostop, [(worlds[w]['graph'], worlds[w]['cost'], worlds[w]['regime']) for w in nostop])
print('final NMSE at stopped worlds above target (0.04):', [(r['world_idx'], round(reports[('sft_llm', r['world_idx'])]['final_error'], 4)) for r in stops if reports[('sft_llm', r['world_idx'])]['final_error'] > 0.04])
# was the stop the first decision at which that state condition held?  compare remaining at stop to all non-stop decisions
print('\nSFT: P(stop | remaining_units) over ALL sft decisions:')
byrem = collections.defaultdict(collections.Counter)
for r in S: byrem[r['envelope']['state']['remaining_units']][r['action'] == 'STOP'] += 1
for k in sorted(byrem): print('   remaining=%2d: stop %d / %d' % (k, byrem[k][True], byrem[k][True] + byrem[k][False]))
print('SFT: P(stop | requests=history length):')
byreq = collections.defaultdict(collections.Counter)
for r in S: byreq[r['envelope']['state']['requests']][r['action'] == 'STOP'] += 1
for k in sorted(byreq): print('   requests=%2d: stop %d / %d' % (k, byreq[k][True], byreq[k][True] + byreq[k][False]))
print('SFT: P(stop | prompt_tokens bucket of 250):')
bytok = collections.defaultdict(collections.Counter)
for r in S: bytok[r['prompt_tokens'] // 250 * 250][r['action'] == 'STOP'] += 1
for k in sorted(bytok): print('   tokens %4d-%4d: stop %d / %d' % (k, k + 249, bytok[k][True], bytok[k][True] + bytok[k][False]))
print('\n===== SFT: the 42 non-set_x_neg acquisitions')
non = [r for r in S if r['action'] not in ('set_x_neg', 'STOP')]
print('n', len(non), dict(collections.Counter(r['action'] for r in non)))
print('by decision_index:', sorted(collections.Counter(r['decision_index'] for r in non).items()))
print('by action x decision_index:')
for a in ['observe_all', 'set_x_pos', 'set_y_neg']:
    print('   %-11s' % a, sorted(collections.Counter(r['decision_index'] for r in non if r['action'] == a).items()))
print('by remaining_units:', sorted(collections.Counter(r['envelope']['state']['remaining_units'] for r in non).items()))
print('by graph:', dict(collections.Counter((worlds[r['world_idx']]['graph'], r['action']) for r in non)))
print('by cost menu:', dict(collections.Counter((worlds[r['world_idx']]['cost'], r['action']) for r in non)))
print('by regime:', dict(collections.Counter((worlds[r['world_idx']]['regime'], r['action']) for r in non)))
print('worlds with >=1 non-x_neg acquisition:', sorted(collections.Counter(r['world_idx'] for r in non).items()))
print('charged units of the non-x_neg acquisitions:', dict(collections.Counter(r['charged_units'] for r in non)))
print('\ndetail (world, graph, cost, regime, decision, remaining, action, #prior x_neg, posterior mean X intercept, trace cov Z):')
for r in sorted(non, key=lambda r: (r['world_idx'], r['decision_index'])):
    st = r['envelope']['state']; W = worlds[r['world_idx']]
    nx = sum(h['action_id'] == 'set_x_neg' for h in st['history'])
    print('   w%2d %-8s %-5s %-12s d=%2d rem=%2d %-11s prior_xneg=%2d meanX=%+.3f trZ=%.4f tok=%d' % (r['world_idx'], W['graph'], W['cost'][:5], W['regime'], r['decision_index'], st['remaining_units'], r['action'], nx, st['scm'][0]['parameter_mean'][0], float(np.trace(np.array(st['scm'][2]['parameter_covariance']))), r['prompt_tokens']))
# SFT position pattern: fraction non-x_neg by decision index
print('\nSFT action by decision index (N=x_neg, P=x_pos, a=y_neg, O=observe, #=stop):')
bd = collections.defaultdict(collections.Counter)
for r in S: bd[r['decision_index']][SH[r['action']]] += 1
for k in sorted(bd): print('   d=%2d %s' % (k, dict(bd[k])))
print('\n===== FROZEN: non-observe actions')
Fz = [r for r in rows if r['arm'] == 'frozen_llm']
for r in Fz:
    if r['action'] != 'observe_all':
        st = r['envelope']['state']; W = worlds[r['world_idx']]
        print('   w%2d %-8s %-5s %-12s decision=%d remaining=%d requests=%d action=%s prompt_tokens=%d legal=%d ordinal=%d' % (r['world_idx'], W['graph'], W['cost'], W['regime'], r['decision_index'], st['remaining_units'], st['requests'], r['action'], r['prompt_tokens'], r['legal_commands'], r['ordinal']))
        print('      history actions:', ''.join(SH[h['action_id']] for h in st['history']))
        print('      scm:', json.dumps([{k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in s.items() if k in ('node', 'parameter_mean', 'eligible_rows')} for s in st['scm']]))
        print('      last 2 history rows:', json.dumps(st['history'][-2:]))
print('frozen legal_commands distribution:', dict(collections.Counter(r['legal_commands'] for r in Fz)))
