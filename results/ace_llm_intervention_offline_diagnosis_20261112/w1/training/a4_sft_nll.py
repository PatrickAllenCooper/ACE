import json, sys, os, math, statistics, collections
root = sys.argv[1]; recs = json.load(open(sys.argv[2]))
T = os.path.join(root, 'training')
plan = json.load(open(os.path.join(T, 'plan.json')))
ev = [json.load(open(os.path.join(T, 'step-%05d.complete.json' % i))) for i in range(279)]
C = collections.Counter
# command token structure
rec0 = plan['records'][0]
cmds = rec0['commands']
print('command token ids at record 0:')
for c in sorted(cmds): print('  %-28s %s' % (c, cmds[c]))
same = all(json.dumps(r['commands'], sort_keys=True) == json.dumps(rec0['commands'], sort_keys=True) for r in plan['records'] if len(r['commands']) == 9)
print('all 128 nonterminal records share identical command token tables:', same)
# marginal entropy of nonterminal labels
lab = C(r['label'] for r in recs if not r['terminal']); n = sum(lab.values())
H = -sum(v/n*math.log(v/n) for v in lab.values())
print('nonterminal label marginal entropy (nats): %.4f ; ln8=%.4f ln9=%.4f ; majority share %.4f' % (H, math.log(8), math.log(9), max(lab.values())/n))
# per-teacher marginal entropy
for t in ('uniform_random', 'menu_coverage', 'local_information', 'target_risk'):
    l = C(r['label'] for r in recs if not r['terminal'] and r['teacher'] == t); m = sum(l.values())
    print('   teacher %-18s entropy %.3f  n=%d' % (t, -sum(v/m*math.log(v/m) for v in l.values()), m))
# prequential total NLL per example
tot = [(i, recs[i], ev[i]['loss'] * ev[i]['action_tokens']) for i in range(160)]
for a, b in ((0, 40), (40, 80), (80, 120), (120, 160)):
    nt = [x for i, r, x in tot[a:b] if not r['terminal']]; tm = [x for i, r, x in tot[a:b] if r['terminal']]
    print('LLM SFT steps %3d-%3d: nonterminal mean total NLL %.3f nats (median %.3f, n=%d); terminal(stop) mean total NLL %.3f' % (
        a, b-1, statistics.mean(nt), statistics.median(nt), len(nt), statistics.mean(tm)))
# by teacher in last 80 steps
for t in ('uniform_random', 'menu_coverage', 'local_information', 'target_risk'):
    x = [v for i, r, v in tot[80:] if not r['terminal'] and r['teacher'] == t]
    print('   last-80 nonterminal by teacher %-18s mean total NLL %.3f n=%d' % (t, statistics.mean(x), len(x)))
# by label in last 80 steps
for l in sorted(lab):
    x = [v for i, r, v in tot[80:] if not r['terminal'] and r['label'] == l]
    if x: print('   last-80 nonterminal by label %-12s mean total NLL %.3f n=%d  => implied mean prob %.3f' % (l, statistics.mean(x), len(x), statistics.mean(math.exp(-v) for v in x)))
# a marginal-frequency online predictor (Laplace) as a state-blind baseline, same order
counts = C(); base = []
acts = sorted(lab)
for i, r, v in tot:
    if r['terminal']: continue
    p = (counts[r['label']] + 1) / (sum(counts.values()) + len(acts))
    base.append((i, -math.log(p), v)); counts[r['label']] += 1
for a, b in ((0, 40), (40, 80), (80, 120), (120, 160)):
    seg = [(x, y) for i, x, y in base if a <= i < b]
    print('state-blind online frequency baseline steps %3d-%3d: mean NLL %.3f vs LLM %.3f' % (a, b-1, statistics.mean(x for x, _ in seg), statistics.mean(y for _, y in seg)))
print('whole-run nonterminal mean: baseline %.3f LLM %.3f ; second half: baseline %.3f LLM %.3f' % (
    statistics.mean(x for _, x, _ in base), statistics.mean(y for _, _, y in base),
    statistics.mean(x for i, x, _ in base if i >= 80), statistics.mean(y for i, _, y in base if i >= 80)))
try:
    from scipy import stats
    print('scipy t_3 two-sided tails:', {k: round(2*stats.t.sf(k, 3), 4) for k in (1, 2, 3, 4, 6, 10)})
    print('binomial: strict 687 of 3584 vs null p=0.1393 -> expected %.1f' % (3584*2*stats.t.sf(2, 3)))
except Exception as e:
    print('scipy unavailable', e)
