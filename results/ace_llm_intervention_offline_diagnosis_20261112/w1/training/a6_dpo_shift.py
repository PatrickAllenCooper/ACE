import json, sys, os, math, statistics, collections
root = sys.argv[1]; recs = json.load(open(sys.argv[2])); byid = {r['record_id']: r for r in recs}
T = os.path.join(root, 'training'); plan = json.load(open(os.path.join(T, 'plan.json'))); sched = plan['schedule']
def act(c):
    d = json.loads(c); return 'stop' if d.get('stop') else d['action_id']
ref = json.load(open(os.path.join(T, 'reference-scores.json')))['scores']
fin = json.load(open(os.path.join(root, 'final-before-reload.json')))
final = {rid: dict(zip([act(c) for c in fin['commands'][rid]], fin['scores'][rid])) for rid in fin['scores']}
by_action = collections.defaultdict(list); refp = collections.defaultdict(list); finp = collections.defaultdict(list)
for s in sched[160:]:
    rid, pi = s['record_id'], s['pair_index']
    ch, rj = byid[rid]['strict'][pi]
    rc, rr = ref[json.dumps([rid, pi], sort_keys=True, separators=(',', ':'))]
    for a, rv in ((ch, rc), (rj, rr)):
        by_action[a].append(final[rid][a] - rv); refp[a].append(rv); finp[a].append(final[rid][a])
print('per action, over exposed pairs containing it: n, mean ref logp (post-SFT), mean final logp, mean change')
for a in sorted(by_action):
    print('  %-12s n=%3d  ref %.3f  final %.3f  change %+.3f  (ref prob gm %.4f -> final gm %.4f)' % (
        a, len(by_action[a]), statistics.mean(refp[a]), statistics.mean(finp[a]), statistics.mean(by_action[a]),
        math.exp(statistics.mean(refp[a])), math.exp(statistics.mean(finp[a]))))
# stop probability at nonterminal under final
st = [final[r['record_id']]['stop'] for r in recs if not r['terminal']]
print('final logp(stop) at nonterminal states: mean %.3f max %.3f' % (statistics.mean(st), max(st)))
tm = [final[r['record_id']]['stop'] for r in recs if r['terminal']]
print('final logp(stop) at terminal states (only legal command): mean %.3f min %.3f' % (statistics.mean(tm), min(tm)))
# SFT reload scores: same pair keys, sanity
sr = json.load(open(os.path.join(root, 'sft-reload-scores.json')))
print('sft-reload-scores keys:', sorted(sr.keys()), 'n score keys', len(sr['scores']))
# how do final per-state distributions vary: entropy over legal commands, and top-2 gap
ent = []; gap = []
for r in recs:
    if r['terminal']: continue
    sc = final[r['record_id']]; z = math.log(sum(math.exp(v) for v in sc.values()))
    p = sorted((math.exp(v - z) for v in sc.values()), reverse=True)
    ent.append(-sum(x*math.log(x) for x in p if x > 0)); o = sorted(sc.values(), reverse=True); gap.append(o[0]-o[1])
print('final policy entropy over 9 legal commands (nats): mean %.3f min %.3f max %.3f ; top1-top2 logp gap: median %.2f min %.2f' % (
    statistics.mean(ent), min(ent), max(ent), statistics.median(gap), min(gap)))
# state-sensitivity proxy: across-state SD of final logp per action (128 states)
print('across-state SD of final logp by action (128 nonterminal states):')
for a in ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero', 'stop']:
    v = [final[r['record_id']][a] for r in recs if not r['terminal']]
    print('  %-12s mean %.3f sd %.3f min %.3f max %.3f' % (a, statistics.mean(v), statistics.pstdev(v), min(v), max(v)))
# does final logp(observe_all)-logp(set_x_neg) track snapshot index / graph?
for key in ('si', 'graph', 'cost', 'regime', 'behavior'):
    g = collections.defaultdict(list)
    for r in recs:
        if r['terminal']: continue
        g[r[key]].append(final[r['record_id']]['observe_all'] - final[r['record_id']]['set_x_neg'])
    print(' final logp(observe_all)-logp(set_x_neg) by', key, {k: round(statistics.mean(v), 2) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))})
g = collections.defaultdict(list)
for r in recs:
    if r['terminal']: continue
    g[r['si']].append(final[r['record_id']]['set_x_pos'] - final[r['record_id']]['set_x_neg'])
print(' final logp(set_x_pos)-logp(set_x_neg) by si', {k: round(statistics.mean(v), 2) for k, v in sorted(g.items())})
