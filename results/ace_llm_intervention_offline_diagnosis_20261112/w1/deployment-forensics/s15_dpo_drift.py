"""SUPPLEMENT: what the DPO stage did to command log-odds, from saved per-step
losses + saved SFT reference scores only (training states, prequential)."""
import pickle, os, collections
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(HERE + '/out/dpo_steps.pkl', 'rb'))
A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
d = np.array([x['delta'] for x in R]); m = np.array([x['ref_margin'] for x in R]); n = np.array([x['n'] for x in R])
print('Prequential (first-exposure) evaluation of the DPO-stage policy on the 119 scheduled pairs:')
print('   policy log-odds favour the preference-chosen command: %d/119 ; SFT reference: %d/119' % (sum(x['policy_margin'] > 0 for x in R), sum(x['ref_margin'] > 0 for x in R)))
print('   Delta > 0 (shift toward chosen on a not-yet-trained pair): %d/119 (%.1f%%), binomial p vs 0.5 = %.3f' % ((d > 0).sum(), 100 * (d > 0).mean(), stats.binomtest(int((d > 0).sum()), 119, 0.5).pvalue))
late2 = [x for x in R if x['n'] >= 60]
print('   second half (steps 60-118): Delta>0 %d/%d ; policy favours chosen %d/%d ; SFT favours chosen %d/%d' % (sum(x['delta'] > 0 for x in late2), len(late2), sum(x['policy_margin'] > 0 for x in late2), len(late2), sum(x['ref_margin'] > 0 for x in late2), len(late2)))
print('\nIs the DPO shift a sharpening of the SFT ordering?  Delta vs SFT reference margin (chosen - rejected):')
for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 119), (0, 119)):
    sel = (n >= lo) & (n < hi)
    sl = stats.linregress(m[sel], d[sel])
    print('   steps %3d-%3d: Pearson r=%+.3f Spearman rho=%+.3f slope=%+.2f (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in %d/%d' % (lo, hi - 1, stats.pearsonr(m[sel], d[sel])[0], stats.spearmanr(m[sel], d[sel])[0], sl.slope, int((np.sign(d[sel]) == np.sign(m[sel])).sum()), int(sel.sum())))
print('   mean |Delta| by phase:', [round(float(np.abs(d[(n >= lo) & (n < hi)]).mean()), 2) for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 119))], ' mean |SFT ref margin|: %.2f' % np.abs(m).mean())
print('   mean |policy margin| = |ref margin + Delta| by phase:', [round(float(np.abs((m + d)[(n >= lo) & (n < hi)]).mean()), 2) for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 119))])
# per-command drift by least squares on late steps: Delta = drift[chosen] - drift[rejected]
for lo, hi, name in ((30, 119, 'steps 30-118'), (60, 119, 'steps 60-118'), (90, 119, 'steps 90-118')):
    rows = [x for x in R if lo <= x['n'] < hi]
    X = np.zeros((len(rows), 8)); y = np.array([x['delta'] for x in rows])
    for i, x in enumerate(rows):
        X[i, A.index(x['chosen'])] += 1; X[i, A.index(x['rejected'])] -= 1
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    coef = coef - coef.mean()
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print('\n   per-command DPO log-odds drift (zero-mean, nats) fitted on %s (n=%d, rank %d, R^2=%.3f):' % (name, len(rows), rank, r2))
    print('      ' + '  '.join('%s %+6.2f' % (a, c) for a, c in sorted(zip(A, coef), key=lambda t: -t[1])))
sft_mean = {'observe_all': -1.092, 'set_x_neg': -0.952, 'set_x_zero': -4.899, 'set_x_pos': -3.220, 'set_y_neg': -2.256, 'set_y_zero': -5.561, 'set_y_pos': -4.328, 'set_xy': -3.877}
net = {'observe_all': -7, 'set_x_neg': 5, 'set_x_zero': -6, 'set_x_pos': 9, 'set_y_neg': 5, 'set_y_zero': 3, 'set_y_pos': -13, 'set_xy': 4}
rows = [x for x in R if x['n'] >= 60]
X = np.zeros((len(rows), 8)); y = np.array([x['delta'] for x in rows])
for i, x in enumerate(rows):
    X[i, A.index(x['chosen'])] += 1; X[i, A.index(x['rejected'])] -= 1
coef = np.linalg.lstsq(X, y, rcond=None)[0]; coef -= coef.mean()
print('\n   correlation of fitted drift (steps 60-118) with SFT mean command log-lik: r=%+.3f rho=%+.3f' % (stats.pearsonr(coef, [sft_mean[a] for a in A])[0], stats.spearmanr(coef, [sft_mean[a] for a in A])[0]))
print('   correlation of fitted drift with net exposed label count (chosen - rejected): r=%+.3f rho=%+.3f' % (stats.pearsonr(coef, [net[a] for a in A])[0], stats.spearmanr(coef, [net[a] for a in A])[0]))
# cumulative net label count BEFORE step 60 only
cnt = collections.Counter()
for x in R:
    if x['n'] < 60: cnt[x['chosen']] += 1; cnt[x['rejected']] -= 1
print('   net label count in DPO steps 0-59:', {a: cnt[a] for a in A})
print('   correlation of drift with net label count in steps 0-59: r=%+.3f rho=%+.3f' % (stats.pearsonr(coef, [cnt[a] for a in A])[0], stats.spearmanr(coef, [cnt[a] for a in A])[0]))
