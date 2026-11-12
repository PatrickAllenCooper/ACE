import json, sys, collections
recs = json.load(open(sys.argv[1])); plan = json.load(open(sys.argv[2])); byid = {r['record_id']: r for r in recs}
C = collections.Counter
def parts(a):
    if a == 'observe_all': return ('observe', '_all', None)
    if a == 'set_xy': return ('set', '_xy', None)
    v, val = a.split('_')[1], a.split('_')[2]
    return ('set', '_' + v, '_' + val)
sel = [byid[s['record_id']]['strict'][s['pair_index']] for s in plan['schedule'][160:]]
allp = [p for r in recs for p in r['strict']]
for name, pairs in (('119 exposed pairs', sel), ('all 687 strict pairs', allp)):
    net = [C(), C(), C()]; up = [C(), C(), C()]; dn = [C(), C(), C()]
    for ch, rj in pairs:
        pc, pr = parts(ch), parts(rj)
        for lvl in range(3):
            # compare first diverging level only: later tokens are in different contexts
            if pc[lvl] != pr[lvl]:
                if pc[lvl]: net[lvl][pc[lvl]] += 1; up[lvl][pc[lvl]] += 1
                if pr[lvl]: net[lvl][pr[lvl]] -= 1; dn[lvl][pr[lvl]] += 1
                break
    print(name)
    for lvl, lab in enumerate(('verb token (set vs observe)', 'variable token given set (_x/_y/_xy)', 'value token given same variable (_neg/_pos/_zero)')):
        print('   first-divergence at %-52s' % lab, {k: (up[lvl][k], dn[lvl][k], net[lvl][k]) for k in sorted(set(up[lvl]) | set(dn[lvl]))}, '(up, down, net)')
    # context-free value-token tally: every pair where value tokens differ at all
    v = C()
    for ch, rj in pairs:
        a, b = parts(ch)[2], parts(rj)[2]
        if a != b:
            if a: v[a] += 1
            if b: v[b] -= 1
    print('   context-free value-token net (chosen value != rejected value):', dict(v))
# SFT label token tallies for nonterminal labels
lab = C(r['label'] for r in recs if not r['terminal'])
verb = C(); var = C(); val = C(); valx = C()
for a, n in lab.items():
    p = parts(a); verb[p[0]] += n
    if p[0] == 'set': var[p[1]] += n
    if p[2]: val[p[2]] += n
    if p[1] == '_x' and p[2]: valx[p[2]] += n
print('SFT nonterminal label tokens: verb', dict(verb), '| variable given set', dict(var), '| value given _x', dict(valx), '| value any', dict(val))
print('greedy token-path mode from SFT label marginal: set (%.3f) -> _x (%.3f) -> _neg (%.3f)' % (verb['set']/128, var['_x']/verb['set'], valx['_neg']/var['_x']))
