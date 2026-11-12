"""High-precision (262,144-draw) target-risk scores for the 24 shared first-decision states. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
from vtr import *
import pickle, collections
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
d0 = {(r['arm'], r['world']): r for r in rows if r['d'] == 0}
rng = np.random.default_rng(987654321)
hp = {}
with np.errstate(all='ignore'):
    for w in range(24):
        env = d0[('target_risk', w)]['envelope']
        acc = None
        for rep in range(16):                       # 16 x 16384 = 262144 draws
            before, out = vec_scores(env, 16384, rng)
            if acc is None: acc = {a: [] for a in out}
            for a, v in out.items(): acc[a].append(v[0])
        hp[w] = {a: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) for a, v in acc.items()}
pickle.dump(hp, open(SCR + '/d0_highprec.pkl', 'wb'))
opt = {}
print('w  stratum                         best(hp) 2nd  gap      se_gap   z     | saved32 best | x- rank | x- gap/se')
flips = 0
for w in range(24):
    sc = hp[w]; order = sorted(sc, key=lambda a: (-sc[a][0], a)); b, s2 = order[0], order[1]
    opt[w] = b
    gap = sc[b][0] - sc[s2][0]; se = (sc[b][1] ** 2 + sc[s2][1] ** 2) ** 0.5
    saved = d0[('target_risk', w)]['action']
    flips += saved != b
    xg = sc[b][0] - sc['set_x_neg'][0]; xse = (sc[b][1] ** 2 + sc['set_x_neg'][1] ** 2) ** 0.5
    print('%2d %-12s %-8s %-13s  %-4s %-4s %.5f  %.5f  %5.1f | %-4s %s | %d | %.1f' % (w, strata[w]['regime'], strata[w]['graph_stratum'], strata[w]['cost_stratum'], SHORT[b], SHORT[s2], gap, se, gap / se,
          SHORT[saved], '' if saved == b else '<-- differs', order.index('set_x_neg') + 1, xg / xse if xg > 0 else 0))
print('high-precision optimum distribution:', {SHORT[k]: v for k, v in collections.Counter(opt.values()).most_common()})
print('TR deployed (32-draw) first choice differs from high-precision optimum in', flips, 'of 24 worlds')
print('\narm | hits vs high-precision optimum | mean abs regret | mean relative regret')
for a in ARMS:
    hits = sum(d0[(a, w)]['action'] == opt[w] for w in range(24))
    reg = [hp[w][opt[w]][0] - hp[w][d0[(a, w)]['action']][0] for w in range(24)]
    rel = [(hp[w][opt[w]][0] - hp[w][d0[(a, w)]['action']][0]) / hp[w][opt[w]][0] for w in range(24)]
    print('%-18s %2d  %.5f  %.3f' % (a, hits, np.mean(reg), np.mean(rel)))
print('\nconstant action at d0: hits, mean abs regret, mean rel regret')
for act in ACTIONS:
    print('%-4s %2d  %.5f  %.3f' % (SHORT[act], sum(opt[w] == act for w in range(24)), np.mean([hp[w][opt[w]][0] - hp[w][act][0] for w in range(24)]),
                                 np.mean([(hp[w][opt[w]][0] - hp[w][act][0]) / hp[w][opt[w]][0] for w in range(24)])))
# worlds where x- is statistically indistinguishable from the best at high precision (|z|<2)
close = [w for w in range(24) if opt[w] != 'set_x_neg' and (hp[w][opt[w]][0] - hp[w]['set_x_neg'][0]) / ((hp[w][opt[w]][1] ** 2 + hp[w]['set_x_neg'][1] ** 2) ** 0.5) < 2]
print('worlds where x- is not best but within 2 s.e. at high precision:', close)
rel_x = [(hp[w][opt[w]][0] - hp[w]['set_x_neg'][0]) / hp[w][opt[w]][0] for w in range(24)]
print('x- relative regret per world', [round(x, 3) for x in rel_x])
print('worlds with x- within 5%% of best: %d ; within 10%%: %d' % (sum(x <= 0.05 for x in rel_x), sum(x <= 0.10 for x in rel_x)))
