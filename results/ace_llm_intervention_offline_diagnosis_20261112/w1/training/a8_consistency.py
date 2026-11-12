import json, sys, collections, itertools, random, statistics
recs = json.load(open(sys.argv[1])); C = collections.Counter
def consistency(pairs):
    d = C()
    for ch, rj in pairs: d[(ch, rj)] += 1
    keys = {tuple(sorted(k)) for k in d}
    maj = sum(max(d[(a, b)], d[(b, a)]) for a, b in keys); tot = sum(d.values())
    return maj, tot
def null(pairs, reps=2000, seed=1):
    rng = random.Random(seed); vals = []
    for _ in range(reps):
        flipped = [(a, b) if rng.random() < .5 else (b, a) for a, b in pairs]
        m, t = consistency(flipped); vals.append(m / t)
    vals.sort(); return statistics.mean(vals), vals[int(.975 * reps)]
allp = [p for r in recs for p in r['strict']]
m, t = consistency(allp); nm, n975 = null(allp)
print('all strict pairs: majority-direction share %d/%d = %.3f ; random-direction null mean %.3f, 97.5%% %.3f' % (m, t, m/t, nm, n975))
for g in ('chain', 'fork', 'collider'):
    ps = [p for r in recs if r['graph'] == g for p in r['strict']]
    m, t = consistency(ps); nm, n975 = null(ps)
    print('  graph %-8s majority share %d/%d = %.3f ; null mean %.3f, 97.5%% %.3f' % (g, m, t, m/t, nm, n975))
for g in ('chain', 'fork', 'collider'):
    for c in ('uniform', 'heterogeneous'):
        ps = [p for r in recs if r['graph'] == g and r['cost'] == c for p in r['strict']]
        m, t = consistency(ps); nm, n975 = null(ps)
        print('  graph %-8s cost %-13s majority share %d/%d = %.3f ; null mean %.3f, 97.5%% %.3f' % (g, c, m, t, m/t, nm, n975))
# per-action win share by graph and by cost
for key in ('graph', 'cost'):
    for val in sorted({r[key] for r in recs}):
        w, l = C(), C()
        for r in recs:
            if r[key] != val: continue
            for ch, rj in r['strict']: w[ch] += 1; l[rj] += 1
        print(' win share by %s=%s:' % (key, val), {a: '%d/%d' % (w[a], w[a]+l[a]) for a in sorted(set(w) | set(l))})
