"""Own-trajectory regret under the verifier's reference target-risk scores. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
from scipy import stats
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
nm = pickle.load(open(SCR + '/nmse_by_state.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]

def evaluate(scoreset):
    """scoreset: function row -> dict action -> (score, after, se). Adds fields to a copy of rows."""
    out = []
    for r in rows:
        sc = scoreset(r)
        order = sorted(sc, key=lambda a: (-sc[a][0], a))
        best = order[0]
        e = dict(arm=r['arm'], world=r['world'], d=r['d'], action=r['action'], best=best, n=len(sc))
        chosen = sc[r['action']][0] if r['action'] != 'STOP' else 0.0
        e['best_score'] = sc[best][0]; e['chosen_score'] = chosen
        e['regret'] = sc[best][0] - chosen
        e['rel'] = (sc[best][0] - chosen) / sc[best][0] if sc[best][0] > 0 else np.nan
        e['hit'] = r['action'] == best
        se_c = sc[r['action']][2] if r['action'] != 'STOP' else 0.0
        e['within2'] = e['regret'] <= 2 * (sc[best][2] ** 2 + se_c ** 2) ** 0.5
        e['xneg_rank'] = order.index('set_x_neg') + 1
        e['xneg_best'] = best == 'set_x_neg'
        e['second_gap'] = sc[order[0]][0] - sc[order[1]][0] if len(order) > 1 else np.nan
        e['second_se'] = (sc[order[0]][2] ** 2 + sc[order[1]][2] ** 2) ** 0.5 if len(order) > 1 else np.nan
        e['rank'] = order.index(r['action']) + 1 if r['action'] != 'STOP' else np.nan
        e['scores'] = {a: v[0] for a, v in sc.items()}
        out.append(e)
    return out

def summarize(ev, label):
    print('\n=== %s ===' % label)
    print('arm                 n   relregret(all) relregret(noSTOP) absregret  hit%%   within2se%%  [n best<=0]')
    res = {}
    for a in ARMS:
        E = [e for e in ev if e['arm'] == a]
        rel = np.array([e['rel'] for e in E]); ok = ~np.isnan(rel)
        E2 = [e for e in E if e['action'] != 'STOP']; rel2 = np.array([e['rel'] for e in E2]); ok2 = ~np.isnan(rel2)
        res[a] = rel[ok].mean()
        print('%-18s %4d   %.3f          %.3f            %.5f   %5.1f   %5.1f      %d' % (a, len(E), rel[ok].mean(), rel2[ok2].mean(), np.mean([e['regret'] for e in E]),
              100 * np.mean([e['hit'] for e in E]), 100 * np.mean([e['within2'] for e in E]), (~ok).sum()))
    return res

ev_ref = evaluate(lambda r: ref[r['envelope_sha256']]['ref'])
res_ref = summarize(ev_ref, 'reference 16384 draws')
res_256 = []
for k in range(5):
    ev = evaluate(lambda r, k=k: ref[r['envelope_sha256']]['rep256'][k])
    res_256.append((summarize(ev, '256-draw replicate %d' % k), ev))
pickle.dump(ev_ref, open(SCR + '/ev_ref.pkl', 'wb'))
pickle.dump([e for _, e in res_256], open(SCR + '/ev_256.pkl', 'wb'))
