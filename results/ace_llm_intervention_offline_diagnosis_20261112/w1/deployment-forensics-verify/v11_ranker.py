"""Learned numerical ranker: saved score vectors. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
rk = sorted([r for r in rows if r['arm'] == 'learned_ranker'], key=lambda r: (r['world'], r['d']))
print('ranker decisions', len(rk), 'actions', {SHORT[k]: v for k, v in collections.Counter(r['action'] for r in rk).items()})
nfeas = collections.Counter(len(r['meta']['action_ids']) for r in rk)
print('len(action_ids) distribution (includes stop=None):', dict(nfeas))
keys = collections.Counter(tuple(sorted(r['meta'].keys())) for r in rk)
print('meta keys', keys)
full = [r for r in rk if len(r['meta']['action_ids']) == 9]
print('decisions with all 8 actions + stop:', len(full))
# check argmax == action, where stop included
bad = 0
for r in rk:
    ids, sc = r['meta']['action_ids'], r['meta']['scores']
    best = ids[int(np.argmax(sc))]
    if (best or 'STOP') != r['action']: bad += 1
print('argmax(score) != action:', bad)
# rank orders
orders = collections.Counter()
orders_acq = collections.Counter()
for r in full:
    ids, sc = r['meta']['action_ids'], np.array(r['meta']['scores'])
    o = tuple((ids[i] or 'STOP') for i in np.argsort(-sc, kind='stable'))
    orders[o] += 1
    orders_acq[tuple(x for x in o if x != 'STOP')] += 1
print('distinct full rank orders (9 incl. stop) among full decisions:', len(orders))
for o, c in orders.most_common(6): print('   ', c, [SHORT[x] for x in o])
print('distinct rank orders over 8 acquisitions:', len(orders_acq))
orders_all = collections.Counter()
for r in rk:
    ids, sc = r['meta']['action_ids'], np.array(r['meta']['scores'])
    orders_all[tuple((ids[i] or 'STOP') for i in np.argsort(-sc, kind='stable'))] += 1
print('distinct rank orders over all 576 (restricted menus included):', len(orders_all), [ (c, [SHORT[x] for x in o]) for o, c in orders_all.most_common()])
# top-2 margin
marg = []; marg_acq = []
for r in rk:
    sc = np.array(r['meta']['scores']); ids = r['meta']['action_ids']
    s = np.sort(sc)[::-1]; marg.append(s[0] - s[1])
    s2 = np.sort(np.array([x for x, i in zip(sc, ids) if i is not None]))[::-1]; marg_acq.append(s2[0] - s2[1])
print('top-2 margin: median %.4f mean %.4f min %.5f max %.4f' % (np.median(marg), np.mean(marg), np.min(marg), np.max(marg)))
print('top-2 margin among acquisitions: median %.4f' % np.median(marg_acq))
# variance decomposition on full decisions: scores matrix (n, 9) and (n, 8)
for label, cols in [('9 incl stop', 9), ('8 acquisitions', 8)]:
    M = np.array([r['meta']['scores'][:cols] for r in full])
    Mc = M - M.mean(axis=1, keepdims=True)          # centre within decision
    action_effect = Mc.mean(axis=0, keepdims=True)  # state-independent action effect
    resid = Mc - action_effect
    tot = (Mc ** 2).sum(); print(label, 'share of centred variance explained by state-independent action effect: %.4f' % (1 - (resid ** 2).sum() / tot),
                                 '| sd of action effects %.4f, residual sd %.4f' % (action_effect.std(), resid.std()))
    print('    mean centred score per action', dict(zip([SHORT[a] for a in ACTIONS] + ['stop'], np.round(action_effect[0], 4))))
# x- minus x+ contrast over decisions
con = np.array([r['meta']['scores'][r['meta']['action_ids'].index('set_x_neg')] - r['meta']['scores'][r['meta']['action_ids'].index('set_x_pos')] for r in rk])
print('x- minus x+ contrast: min %.4f median %.4f max %.4f; positive in %d of %d' % (con.min(), np.median(con), con.max(), (con > 0).sum(), len(con)))
byd = collections.defaultdict(list)
for r, c in zip(rk, con): byd[r['d']].append(c)
print('x- chosen by decision index:', [(d, sum(1 for c in byd[d] if c > 0)) for d in sorted(byd)])
# per world sequence
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
for w in range(24):
    print(w, strata[w]['graph_stratum'][:4], strata[w]['cost_stratum'][:4], strata[w]['regime'], ''.join('-' if r['action'] == 'set_x_neg' else '+' for r in rk if r['world'] == w))
pickle.dump({(r['world'], r['d']): c for r, c in zip(rk, con)}, open(SCR + '/ranker_contrast.pkl', 'wb'))
