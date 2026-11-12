"""Numerical ranker: saved per-candidate scores at every decision (576).
Margins, between-state variance vs between-action gaps, rank changes,
correlation with state features."""
import pickle, collections, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
R = [r for r in rows if r['arm'] == 'learned_ranker']
IDS = ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero', None]
print('decisions', len(R))
print('candidate-count distribution:', collections.Counter(len(r['meta']['action_ids']) for r in R))
print('meta keys:', sorted(R[0]['meta'].keys()))
full = [r for r in R if r['meta']['action_ids'] == IDS]
S = np.array([r['meta']['scores'] for r in full])  # n x 9
print('decisions with all 9 candidates:', len(full))
names = [a or 'STOP' for a in IDS]
# softmax probabilities
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
print('\nper-candidate raw score: mean, sd across decisions (between-state), min, max ; mean softmax prob')
for j, a in enumerate(names):
    print('  %-11s mean %8.4f sd %7.4f min %8.4f max %8.4f | mean p %.4f' % (a, S[:, j].mean(), S[:, j].std(ddof=1), S[:, j].min(), S[:, j].max(), P[:, j].mean()))
srt = np.sort(S, 1)[:, ::-1]
m12 = srt[:, 0] - srt[:, 1]
print('\nmargin best-second: min %.5f p10 %.5f median %.5f mean %.5f p90 %.5f max %.5f' % (m12.min(), np.percentile(m12, 10), np.median(m12), m12.mean(), np.percentile(m12, 90), m12.max()))
print('score range (best-worst non-stop): median %.4f ; best - STOP: median %.4f' % (np.median(S[:, :8].max(1) - S[:, :8].min(1)), np.median(S[:, :8].max(1) - S[:, 8])))
# centre scores per decision (remove state-level offset) to isolate relative preferences
C = S[:, :8] - S[:, :8].mean(1, keepdims=True)
print('\ncentred (per-decision mean removed, 8 actions): mean and between-state sd')
for j, a in enumerate(names[:8]):
    print('  %-11s mean %+8.4f sd %7.4f' % (a, C[:, j].mean(), C[:, j].std(ddof=1)))
between_action_sd = C.mean(0).std(ddof=1)
within_action_sd = np.sqrt((C.std(0, ddof=1) ** 2).mean())
print('SD of action means (between-action gaps): %.4f ; RMS between-state SD of centred scores: %.4f ; ratio state/action = %.3f' % (between_action_sd, within_action_sd, within_action_sd / between_action_sd))
# variance decomposition of centred scores: action main effect vs state x action interaction
tot = ((C - C.mean()) ** 2).sum()
act = C.shape[0] * ((C.mean(0) - C.mean()) ** 2).sum()
print('share of centred-score variance explained by the action main effect (state-independent): %.4f ; state-dependent (interaction) share: %.4f' % (act / tot, 1 - act / tot))
# rank orders
orders = collections.Counter(tuple(np.array(names)[np.argsort(-s, kind='stable')]) for s in S)
print('\ndistinct complete rank orders of the 9 candidates:', len(orders), 'over', len(S), 'decisions')
for o, c in orders.most_common(6):
    print('  ', c, ' > '.join(o))
top = collections.Counter(names[i] for i in S.argmax(1)); second = collections.Counter(names[i] for i in np.argsort(-S, 1)[:, 1])
print('top-1:', dict(top), ' second:', dict(second))
print('rank position of STOP:', collections.Counter((np.argsort(-S, 1) == 8).argmax(1) + 1))
# the x_neg vs x_pos contrast
d = S[:, 1] - S[:, 2]
print('\nset_x_neg - set_x_pos score: mean %.4f sd %.4f min %.4f max %.4f ; >0 in %d/%d' % (d.mean(), d.std(ddof=1), d.min(), d.max(), int((d > 0).sum()), len(d)))
# features
def feats(r):
    st = r['envelope']['state']; w = worlds[r['world_idx']]
    f = {'step': r['decision_index'], 'remaining': st['remaining_units'], 'requests': st['requests'],
         'graph': w['graph'], 'cost': w['cost'], 'regime': w['regime'], 'noise': w['noise']}
    hist = st['history']
    f['n_xneg'] = sum(h['action_id'] == 'set_x_neg' for h in hist); f['n_xpos'] = sum(h['action_id'] == 'set_x_pos' for h in hist)
    for s in st['scm']:
        f['trcov_' + s['node']] = float(np.trace(np.array(s['parameter_covariance'])))
        f['mean0_' + s['node']] = s['parameter_mean'][0]
    f['mean_x'] = st['scm'][0]['parameter_mean'][0]
    return f
F = [feats(r) for r in full]
print('\ncorrelation of (x_neg - x_pos) score contrast with state features (Pearson r, Spearman rho):')
from scipy import stats
for k in ['step', 'remaining', 'n_xneg', 'n_xpos', 'mean_x', 'trcov_X', 'trcov_Y', 'trcov_Z', 'noise']:
    x = np.array([f[k] for f in F], float)
    print('  %-10s r=%+.3f rho=%+.3f' % (k, stats.pearsonr(x, d)[0], stats.spearmanr(x, d)[0]))
for key in ['graph', 'cost', 'regime']:
    print('  by', key, {v: (round(float(np.mean([dd for dd, f in zip(d, F) if f[key] == v])), 4), sum(1 for dd, f in zip(d, F) if f[key] == v and dd > 0), sum(1 for f in F if f[key] == v)) for v in sorted({f[key] for f in F})}, '(mean contrast, n x_neg chosen, n)')
# what determines the choice x_neg vs x_pos: by step and by counts of prior actions
print('\nchoice by decision index (all 576): index: (x_neg, x_pos)')
bystep = collections.defaultdict(collections.Counter)
for r in R: bystep[r['decision_index']][r['action']] += 1
print('  ', {k: (v['set_x_neg'], v['set_x_pos']) for k, v in sorted(bystep.items())})
# per world sequences
print('\nper-world ranker action sequences (N = set_x_neg, P = set_x_pos):')
for w in range(24):
    seq = ''.join({'set_x_neg': 'N', 'set_x_pos': 'P'}.get(r['action'], '?') for r in sorted((r for r in R if r['world_idx'] == w), key=lambda r: r['decision_index']))
    W = worlds[w]
    mx = [r for r in R if r['world_idx'] == w and r['decision_index'] == 0][0]['envelope']['state']['scm'][0]['parameter_mean'][0]
    print('  %2d %-8s %-5s %-12s postmean_X@0=%+.3f  %s' % (w, W['graph'], W['cost'][:5], W['regime'], mx, seq))
# logistic-ish: does posterior mean of X intercept predict x_pos at decision 0?
print('\nfirst decision: posterior mean of X intercept vs ranker choice and TR optimum')
FD = pickle.load(open(HERE + '/out/first_decision.pkl', 'rb'))
for w in range(24):
    r0 = [r for r in R if r['world_idx'] == w and r['decision_index'] == 0][0]
    print('  w%2d meanX %+.3f ranker %-10s TR-opt %-10s  ranker(xneg-xpos) %+.4f  TR(xneg-xpos) %+.5f' % (w, r0['envelope']['state']['scm'][0]['parameter_mean'][0], r0['action'], FD[w]['best'], r0['meta']['scores'][1] - r0['meta']['scores'][2], FD[w]['scores']['set_x_neg'] - FD[w]['scores']['set_x_pos']))
x = np.array([[r for r in R if r['world_idx'] == w and r['decision_index'] == 0][0]['meta']['scores'][1] - [r for r in R if r['world_idx'] == w and r['decision_index'] == 0][0]['meta']['scores'][2] for w in range(24)])
y = np.array([FD[w]['scores']['set_x_neg'] - FD[w]['scores']['set_x_pos'] for w in range(24)])
print('first-decision correlation between ranker contrast and TR contrast (x_neg - x_pos): r=%.3f rho=%.3f' % (stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]))
