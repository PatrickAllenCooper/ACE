"""Is the observed state-dependence of the ranker and the SFT adapter in the
RIGHT direction according to the (256-draw) target-risk score at their own states?"""
import pickle, collections, os, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TR = pickle.load(open(HERE + '/out/tr_scores_256.pkl', 'rb'))
def S(r): return TR[r['envelope_sha256']]['scores']
def SE(r): return TR[r['envelope_sha256']]['se']
# ---------- ranker
R = [r for r in rows if r['arm'] == 'learned_ranker']
ids = ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero', None]
full = [r for r in R if r['meta']['action_ids'] == ids]
rc = np.array([r['meta']['scores'][1] - r['meta']['scores'][2] for r in full])   # x_neg - x_pos (ranker)
tc = np.array([S(r)['set_x_neg'] - S(r)['set_x_pos'] for r in full])             # x_neg - x_pos (TR)
print('RANKER (564 full-menu decisions): contrast set_x_neg - set_x_pos, ranker score vs TR score at the same state')
print('   Pearson r=%+.3f Spearman rho=%+.3f ; sign agreement %d/%d (%.1f%%)' % (stats.pearsonr(rc, tc)[0], stats.spearmanr(rc, tc)[0], int((np.sign(rc) == np.sign(tc)).sum()), len(rc), 100 * (np.sign(rc) == np.sign(tc)).mean()))
for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 24)):
    sel = np.array([lo <= r['decision_index'] < hi for r in full])
    print('   decisions %2d-%2d: sign agreement %d/%d ; TR prefers x_neg over x_pos in %d ; ranker prefers x_neg in %d' % (lo, hi - 1, int((np.sign(rc[sel]) == np.sign(tc[sel])).sum()), int(sel.sum()), int((tc[sel] > 0).sum()), int((rc[sel] > 0).sum())))
# whole-vector rank correlation between ranker scores (8 actions) and TR scores at each state
rhos = []
for r in full:
    a = [r['meta']['scores'][ids.index(x)] for x in ids[:8]]; b = [S(r)[x] for x in ids[:8]]
    rhos.append(stats.spearmanr(a, b)[0])
rhos = np.array(rhos)
print('   per-state Spearman rho between the ranker\'s 8 action scores and the 8 TR scores: mean %+.3f median %+.3f ; >0 in %.1f%%' % (rhos.mean(), np.median(rhos), 100 * (rhos > 0).mean()))
for lo, hi in ((0, 1), (1, 4), (4, 8), (8, 24)):
    sel = np.array([lo <= r['decision_index'] < hi for r in full])
    print('      decisions %2d-%2d: mean rho %+.3f' % (lo, hi - 1, rhos[sel].mean()))
# a state-independent reference: the ranker's AVERAGE score vector used at every state
avg = np.mean([[r['meta']['scores'][ids.index(x)] for x in ids[:8]] for r in full], axis=0)
rh2 = np.array([stats.spearmanr(avg, [S(r)[x] for x in ids[:8]])[0] for r in full])
print('   same, using the ranker\'s state-AVERAGED score vector at every state: mean rho %+.3f (difference from state-specific: %+.4f)' % (rh2.mean(), rhos.mean() - rh2.mean()))
# ---------- SFT deviations
Sf = [r for r in rows if r['arm'] == 'sft_llm']
dev = [r for r in Sf if r['action'] not in ('set_x_neg', 'STOP')]
gain = np.array([S(r)[r['action']] - S(r)['set_x_neg'] for r in dev])
sig = np.array([abs(S(r)[r['action']] - S(r)['set_x_neg']) > 2 * math.hypot(SE(r)[r['action']], SE(r)['set_x_neg']) for r in dev])
print('\nSFT: the 42 non-set_x_neg acquisitions. TR score(chosen) - TR score(set_x_neg) at the same state (per cost unit):')
print('   chosen better than set_x_neg in %d/42 ; mean gain %+.5f ; median %+.5f ; |diff| > 2 MC s.e. in %d (of which chosen better: %d)' % (int((gain > 0).sum()), gain.mean(), np.median(gain), int(sig.sum()), int((gain[sig] > 0).sum())))
for a in ('set_x_pos', 'set_y_neg', 'observe_all'):
    g = np.array([S(r)[a] - S(r)['set_x_neg'] for r in dev if r['action'] == a])
    rk = [sorted(S(r), key=lambda k: -S(r)[k]).index(a) + 1 for r in dev if r['action'] == a]
    print('   %-11s n=%2d better than x_neg in %2d ; mean gain %+.5f ; TR rank of the chosen action: %s' % (a, len(g), int((g > 0).sum()), g.mean(), sorted(collections.Counter(rk).items())))
# compare: at ALL sft non-stop decisions with d>=6, how often would switching to x_pos / observe have beaten x_neg?
late = [r for r in Sf if r['action'] != 'STOP' and r['decision_index'] >= 6 and 'set_y_neg' in S(r)]
print('   reference: at all %d SFT acquisition states with decision index >= 6, TR prefers set_x_pos over set_x_neg in %.1f%%, observe_all over set_x_neg in %.1f%%, set_y_neg over set_x_neg in %.1f%%' % (
    len(late), 100 * np.mean([S(r)['set_x_pos'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['observe_all'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['set_y_neg'] > S(r)['set_x_neg'] for r in late])))
# rel regret at deviation states vs matched non-deviation states
def rel(r):
    s = S(r); b = max(s.values()); return (b - s[r['action']]) / b if b > 0 else 0
print('   mean relative regret: at the 42 deviation decisions %.3f ; at SFT set_x_neg decisions with the same decision indexes (10-21) %.3f' % (np.mean([rel(r) for r in dev]), np.mean([rel(r) for r in Sf if r['action'] == 'set_x_neg' and 10 <= r['decision_index'] <= 21])))
# set_y_neg by graph: is Y-intervention valuable there under TR?
print('   TR value of set_y_neg relative to best, by graph, over all SFT acquisition states (mean of score/best):', {g: round(float(np.mean([S(r)['set_y_neg'] / max(S(r).values()) for r in Sf if r['action'] != 'STOP' and worlds[r['world_idx']]['graph'] == g and 'set_y_neg' in S(r)])), 3) for g in ('chain', 'fork', 'collider')})
# ---------- SFT stops
st = [r for r in Sf if r['action'] == 'STOP']
print('\nSFT stops (23): best available TR gain per cost unit forgone at the stop state: mean %.5f median %.5f ; risk_before at stop: mean %.5f ; for comparison risk_before at the shared initial states: mean %.5f' % (
    np.mean([max(S(r).values()) for r in st]), np.median([max(S(r).values()) for r in st]), np.mean([TR[r['envelope_sha256']]['risk_before'] for r in st]),
    np.mean([TR[r['envelope_sha256']]['risk_before'] for r in Sf if r['decision_index'] == 0])))
rb_stop = np.array([TR[r['envelope_sha256']]['risk_before'] for r in st])
print('   posterior integrated risk at stop <= 0.04 (public analogue of the private target) in %d/23 ; min %.4f max %.4f' % (int((rb_stop <= 0.04).sum()), rb_stop.min(), rb_stop.max()))
# did SFT stop earlier when posterior risk was lower? (state-dependence of stop time on uncertainty)
x = np.array([r['decision_index'] for r in st]); 
print('   stop decision index vs posterior risk at stop: Spearman rho %+.3f (p=%.2f) ; vs remaining units: all stops at remaining<=3' % stats.spearmanr(x, rb_stop))
