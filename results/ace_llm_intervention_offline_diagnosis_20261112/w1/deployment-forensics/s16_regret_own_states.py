"""Decision-level regret of every arm at EVERY decision on its own trajectory,
under the target-risk score recomputed with 256 fantasy draws on the saved
public state (out/tr_scores_256.pkl). Also: robustness of the shared-first-
decision results to Monte Carlo noise, validation of the score against realised
NMSE changes, and what the planner would have wanted at the LLM arms' states."""
import pickle, collections, json, os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TR = pickle.load(open(HERE + '/out/tr_scores_256.pkl', 'rb'))
reports = pickle.load(open(HERE + '/out/reports.pkl', 'rb'))
A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
SH = {'observe_all': 'obs', 'set_x_neg': 'x-', 'set_x_zero': 'x0', 'set_x_pos': 'x+', 'set_y_neg': 'y-', 'set_y_zero': 'y0', 'set_y_pos': 'y+', 'set_xy': 'xy', 'STOP': 'STOP'}
print('states with recomputed scores:', len(TR))
# ---- consistency of 256-draw recomputation with the saved 32-draw scores of the target_risk arm
dif = []; same = 0; n = 0
for r in rows:
    if r['arm'] == 'target_risk':
        s32 = r['meta']['planning_metadata']['risk_reduction_per_cost']; s256 = TR[r['envelope_sha256']]['scores']
        dif += [s32[a] - s256[a] for a in s32]
        b256 = min(s256, key=lambda a: (-s256[a], a)); same += (b256 == r['action']); n += 1
dif = np.array(dif)
print('saved 32-draw vs recomputed 256-draw scores on target_risk states: mean diff %.6f, sd %.6f ; 256-draw argmax equals deployed (32-draw) choice in %d/%d (%.1f%%)' % (dif.mean(), dif.std(), same, n, 100 * same / n))
# ---- per decision quantities
def q(r):
    t = TR[r['envelope_sha256']]; s = t['scores']; se = t['se']
    order = sorted(s, key=lambda a: (-s[a], a)); best = order[0]
    out = {'best': best, 'sbest': s[best], 'order': order, 'scores': s, 'se': se, 'risk_before': t['risk_before'], 'nfeas': len(s)}
    a = r['action']
    if a == 'STOP':
        out.update(regret=s[best], rank=len(s) + 1, opt=False, near=False, rel=1.0)   # stopping forgoes the best available gain
    else:
        reg = s[best] - s[a]
        out.update(regret=reg, rank=order.index(a) + 1, opt=(a == best), near=(reg <= 2 * math.hypot(se[best], se[a]) if a != best else True),
                   rel=(reg / s[best] if s[best] > 0 else 0.0))
    return out
Q = {id(r): q(r) for r in rows}
print('\nDecision-level endpoint on each arm\'s OWN trajectory (all its decisions; STOP counted as forgoing the best available gain).')
print('regret = best score - chosen score (risk reduction per cost unit, 256 draws). rel = regret / best score. near-opt = within 2 combined MC s.e. of the best.')
print('%-18s %4s | %9s %9s | %7s | %8s %8s | %6s | %s' % ('arm', 'n', 'mean_reg', 'med_reg', 'mean_rel', 'TR-opt%', 'near-opt%', 'm_rank', 'sum regret over episode (mean per world)'))
summ = {}
for arm in pids:
    R = [r for r in rows if r['arm'] == arm]
    reg = np.array([Q[id(r)]['regret'] for r in R]); rel = np.array([Q[id(r)]['rel'] for r in R])
    opt = np.mean([Q[id(r)]['opt'] for r in R]); near = np.mean([Q[id(r)]['near'] for r in R]); rk = np.mean([Q[id(r)]['rank'] for r in R])
    perw = [sum(Q[id(r)]['regret'] for r in R if r['world_idx'] == w) for w in range(24)]
    summ[arm] = dict(mean=reg.mean(), med=np.median(reg), rel=rel.mean(), opt=opt, near=near, rank=rk, perw=perw)
    print('%-18s %4d | %9.5f %9.5f | %7.3f | %7.1f%% %7.1f%% | %6.2f | %.4f' % (arm, len(R), reg.mean(), np.median(reg), rel.mean(), 100 * opt, 100 * near, rk, np.mean(perw)))
# paired per-world comparison of mean relative regret
print('\nPer-world mean relative regret (rel), paired across the 24 worlds: arm vs target_risk and vs uniform_random (Wilcoxon signed-rank, descriptive)')
perw_rel = {arm: np.array([np.mean([Q[id(r)]['rel'] for r in rows if r['arm'] == arm and r['world_idx'] == w]) for w in range(24)]) for arm in pids}
for arm in pids:
    x = perw_rel[arm]
    line = '%-18s mean %.3f sd %.3f' % (arm, x.mean(), x.std(ddof=1))
    for ref in ('target_risk', 'uniform_random'):
        if arm != ref:
            dd = x - perw_rel[ref]
            line += ' | vs %s: diff %+.3f, worse in %d/24, W p=%.2g' % (ref, dd.mean(), int((dd > 0).sum()), stats.wilcoxon(dd).pvalue)
    print(line)
# by decision-index block
print('\nMean relative regret by decision-index block (0-3, 4-7, 8-15, 16-23):')
for arm in pids:
    out = []
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 24)):
        v = [Q[id(r)]['rel'] for r in rows if r['arm'] == arm and lo <= r['decision_index'] < hi]
        out.append('%.3f' % np.mean(v))
    print('  %-18s %s' % (arm, '  '.join(out)))
print('\nFraction of decisions where the chosen action is the 256-draw TR optimum, by decision-index block (0, 1-3, 4-7, 8-15, 16-23):')
for arm in pids:
    out = []
    for lo, hi in ((0, 1), (1, 4), (4, 8), (8, 16), (16, 24)):
        v = [Q[id(r)]['opt'] for r in rows if r['arm'] == arm and lo <= r['decision_index'] < hi]
        out.append('%.2f' % np.mean(v))
    print('  %-18s %s' % (arm, '  '.join(out)))
# ---- what did the planner want at the DPO / frozen / SFT states?
for arm, const in (('dpo_llm', 'set_x_neg'), ('frozen_llm', 'observe_all'), ('sft_llm', 'set_x_neg'), ('learned_ranker', None)):
    R = [r for r in rows if r['arm'] == arm]
    print('\n%s states: 256-draw TR optimum distribution: %s' % (arm, dict(collections.Counter(Q[id(r)]['best'] for r in R).most_common())))
    if const:
        rk = collections.Counter(Q[id(r)]['order'].index(const) + 1 if const in Q[id(r)]['order'] else 99 for r in R)
        print('   rank of %s under TR at these states: %s' % (const, sorted(rk.items())))
        byd = collections.defaultdict(list)
        for r in R: byd[r['decision_index']].append(Q[id(r)])
        print('   by decision index: share of worlds where %s is TR-optimal / within 2 s.e. ; mean relative regret of %s ; modal TR optimum' % (const, const))
        for k in sorted(byd):
            qs = byd[k]
            opt = np.mean([x['best'] == const for x in qs])
            near = np.mean([(x['sbest'] - x['scores'][const]) <= 2 * math.hypot(x['se'][x['best']], x['se'][const]) for x in qs])
            rel = np.mean([(x['sbest'] - x['scores'][const]) / x['sbest'] if x['sbest'] > 0 else 0 for x in qs])
            print('     d=%2d n=%2d  opt %.2f  near %.2f  rel-regret %.3f  modal-opt %s' % (k, len(qs), opt, near, rel, collections.Counter(x['best'] for x in qs).most_common(2)))
# ---- first decision robustness with 256 draws
print('\nShared first decision with 256-draw scores:')
first = {(r['arm'], r['world_idx']): r for r in rows if r['decision_index'] == 0}
bc = collections.Counter(); xr = []; g12 = []; gx = []; zz = []
for w in range(24):
    x = Q[id(first[('target_risk', w)])]; s = x['scores']; o = x['order']
    bc[o[0]] += 1; xr.append(o.index('set_x_neg') + 1); g12.append(s[o[0]] - s[o[1]]); gx.append(s[o[0]] - s['set_x_neg'])
    zz.append((s[o[0]] - s[o[1]]) / math.hypot(x['se'][o[0]], x['se'][o[1]]))
print('   TR optimum counts (256 draws):', dict(bc), ' | set_x_neg rank:', sorted(collections.Counter(xr).items()))
print('   gap best-second: median %.5f mean %.5f ; in MC s.e.: %s ; <2 s.e. in %d/24' % (np.median(g12), np.mean(g12), np.round(zz, 1).tolist(), sum(z < 2 for z in zz)))
print('   gap best-set_x_neg: median %.5f mean %.5f ; set_x_neg within 2 s.e. of best in %d/24' % (np.median(gx), np.mean(gx), sum((Q[id(first[('target_risk', w)])]['sbest'] - Q[id(first[('target_risk', w)])]['scores']['set_x_neg']) <= 2 * math.hypot(Q[id(first[('target_risk', w)])]['se'][Q[id(first[('target_risk', w)])]['best']], Q[id(first[('target_risk', w)])]['se']['set_x_neg']) for w in range(24))))
print('   %-18s agree  near-opt  mean_regret  mean_rel' % 'arm')
for arm in pids:
    qs = [Q[id(first[(arm, w)])] for w in range(24)]
    print('   %-18s %2d/24  %2d/24    %.5f     %.3f' % (arm, sum(x['opt'] for x in qs), sum(x['near'] for x in qs), np.mean([x['regret'] for x in qs]), np.mean([x['rel'] for x in qs])))
pickle.dump({'Q': {(r['arm'], r['world_idx'], r['decision_index']): Q[id(r)] for r in rows}, 'summ': summ}, open(HERE + '/out/regret.pkl', 'wb'))
