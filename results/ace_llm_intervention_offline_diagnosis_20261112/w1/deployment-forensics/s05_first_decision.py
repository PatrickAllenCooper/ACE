"""Shared first post-initialization decision (decision_index 0): all arms see the
identical public state. Use target_risk's SAVED per-action scores
(planning_metadata.risk_reduction_per_cost, 32 fantasy draws) as the reference."""
import pickle, collections, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
reports = pickle.load(open(HERE + '/out/reports.pkl', 'rb'))
A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
first = {(r['arm'], r['world_idx']): r for r in rows if r['decision_index'] == 0}
# shared state check
ok = all(len({first[(a, w)]['envelope_sha256'] for a in pids}) == 1 for w in range(24))
print('all 8 arms share identical envelope sha256 at decision 0 in all 24 worlds:', ok)
out = []
print('\nPer-world table at shared first decision. tr = saved target-risk score (risk reduction per cost unit).')
print('w  graph    cost  regime        nmse0   risk_before | ' + ' '.join('%9s' % a[:9] for a in A) + ' | best      gap12   gap_xneg  se_best | chosen by: ' + ' '.join(p[:6] for p in pids))
agree = collections.Counter(); regret = collections.defaultdict(list); nreg = collections.defaultdict(list); rankpos = collections.defaultdict(list)
best_counter = collections.Counter(); xneg_rank = []
for w in range(24):
    tr = first[('target_risk', w)]['meta']['planning_metadata']
    s = tr['risk_reduction_per_cost']; se = tr['monte_carlo_standard_errors']
    order = sorted(A, key=lambda a: (-s[a], a))
    best, second = order[0], order[1]
    best_counter[best] += 1
    gap12 = s[best] - s[second]
    gapx = s[best] - s['set_x_neg']
    xneg_rank.append(order.index('set_x_neg') + 1)
    nmse0 = reports[('target_risk', w)]['curve'][0]['normalized_mse']
    chosen = {p: first[(p, w)]['action'] for p in pids}
    rng = s[best] - min(s.values())
    for p in pids:
        c = chosen[p]
        if c == 'STOP':
            regret[p].append(float('nan')); continue
        regret[p].append(s[best] - s[c])
        nreg[p].append((s[best] - s[c]) / rng if rng > 0 else 0.0)
        rankpos[p].append(order.index(c) + 1)
        agree[p] += (c == best)
    W = worlds[w]
    print('%2d %-8s %-5s %-12s %.4f  %.5f     | ' % (w, W['graph'], W['cost'][:5], W['regime'], nmse0, tr['risk_before'])
          + ' '.join('%9.5f' % s[a] for a in A) + ' | %-9s %.5f %.5f  %.5f | ' % (best, gap12, gapx, se[best])
          + ' '.join('%-6s' % {'observe_all': 'obs', 'set_x_neg': 'x-', 'set_x_zero': 'x0', 'set_x_pos': 'x+', 'set_y_neg': 'y-', 'set_y_zero': 'y0', 'set_y_pos': 'y+', 'set_xy': 'xy', 'STOP': 'STOP'}[chosen[p]] for p in pids))
    out.append({'w': w, 'scores': s, 'se': se, 'risk_before': tr['risk_before'], 'best': best, 'second': second, 'gap12': gap12, 'gap_xneg': gapx, 'chosen': chosen, 'nmse0': nmse0, 'order': order})
print('\ntarget-risk optimum at shared first decision, counts over 24 worlds:', dict(best_counter))
print('rank of set_x_neg under target-risk score at first decision (1=best):', collections.Counter(xneg_rank))
g12 = np.array([o['gap12'] for o in out]); gx = np.array([o['gap_xneg'] for o in out]); rb = np.array([o['risk_before'] for o in out])
print('gap best-second: min %.5f median %.5f mean %.5f max %.5f' % (g12.min(), np.median(g12), g12.mean(), g12.max()))
print('gap best-set_x_neg: min %.5f median %.5f mean %.5f max %.5f ; zero in %d/24' % (gx.min(), np.median(gx), gx.mean(), gx.max(), int((gx == 0).sum())))
print('relative gap12 / best score: median %.3f' % np.median([o['gap12'] / o['scores'][o['best']] for o in out]))
seb = np.array([np.hypot(o['se'][o['best']], o['se'][o['second']]) for o in out])
print('gap12 in units of combined MC s.e.: ', np.round(g12 / seb, 2).tolist())
print('  worlds where gap12 < 2 combined MC s.e.:', int((g12 / seb < 2).sum()), 'of 24')
print('\nPer-policy summary at the shared first decision (24 worlds):')
print('%-18s agree_with_TR_opt  mean_regret  median_regret  mean_norm_regret  mean_rank  chose_x_neg' % 'arm')
for p in pids:
    r = np.array(regret[p]); nr = np.array(nreg[p]); rk = np.array(rankpos[p])
    cx = sum(first[(p, w)]['action'] == 'set_x_neg' for w in range(24))
    print('%-18s %2d/24             %.5f      %.5f        %.3f             %.2f       %2d/24' % (p, agree[p], np.nanmean(r), np.nanmedian(r), nr.mean(), rk.mean(), cx))
# constant-action reference policies
print('\nConstant-action reference policies at the shared first decision:')
for a in A:
    r = np.array([o['scores'][o['best']] - o['scores'][a] for o in out])
    print('  always %-11s: optimal in %2d/24, mean regret %.5f, median %.5f' % (a, int((r == 0).sum()), r.mean(), np.median(r)))
r_unif = np.array([np.mean([o['scores'][o['best']] - o['scores'][a] for a in A]) for o in out])
print('  uniform-random expectation: mean regret %.5f' % r_unif.mean())
# worlds already at target after init
init_ok = [w for w in range(24) if out[w]['nmse0'] <= 0.04]
print('\nworlds already at NMSE<=0.04 after init:', init_ok, len(init_ok))
for sub, name in ((init_ok, 'already-at-target (10)'), ([w for w in range(24) if w not in init_ok], 'not-at-target (14)')):
    print(' subset', name)
    print('   TR optimum counts:', dict(collections.Counter(out[w]['best'] for w in sub)))
    for p in pids:
        r = np.array([regret[p][w] for w in sub])
        print('   %-18s mean regret %.5f  agree %d/%d' % (p, np.nanmean(r), sum(first[(p, w)]['action'] == out[w]['best'] for w in sub), len(sub)))
pickle.dump(out, open(HERE + '/out/first_decision.pkl', 'wb'))
