"""Offline read-only analysis of the executed scientific training plan (JSON only)."""
import json, sys, collections, statistics
plan_path = sys.argv[1]
p = json.load(open(plan_path))
recs = p['records']
METHODS = ("uniform_random", "menu_coverage", "local_information", "target_risk")
REGIMES = (("matched_low", 0., .25), ("matched_high", 0., 1.), ("shifted_low", 1.5, .25), ("shifted_high", 1.5, 1.))
GRAPHS = ("chain", "fork", "collider")
COSTS = ("uniform", "heterogeneous")
extra = ("chain", "fork", "collider", "chain")
# Rebuild the prospective training roster exactly as _proposal does (pure arithmetic).
rows = []
for ri in range(4):
    for pi, graph in enumerate(GRAPHS + (extra[ri],)):
        for cost in COSTS:
            idx = len(rows)
            ci = COSTS.index(cost)
            behavior = METHODS[(pi + ci + ri) % 4]
            tb = (pi + 2*ci + 2*ri) % 4
            rows.append(dict(world_id=202611070000+idx, regime=REGIMES[ri][0], graph=graph, cost=cost,
                             behavior=behavior, teachers=[METHODS[(tb+si) % 4] for si in range(5)]))
by_world = {r['world_id']: r for r in rows}

def act(cmd):
    d = json.loads(cmd)
    return 'stop' if d.get('stop') else d['action_id']

out = []
for i, rec in enumerate(recs):
    raw = json.loads(rec['raw_record_utf8'])
    st = raw['envelope']['state']
    wid = raw['world_id']; si = i % 5
    row = by_world[wid]
    parents = {n['name']: n['parents'] for n in st['graph']}
    g = ('chain' if parents['Z'] == ['Y'] and parents['Y'] == ['X'] else
         'fork' if parents['Z'] == ['X'] and parents['Y'] == ['X'] else
         'collider' if parents['Z'] == ['X', 'Y'] else 'unknown')
    costs = {a['action_id']: a['cost_units'] for a in st['menu']}
    cost = 'heterogeneous' if costs['set_y_neg'] == 2 else 'uniform'
    assert g == row['graph'] and cost == row['cost'], (wid, g, row['graph'], cost, row['cost'])
    noise = {s['node']: s['noise_variance'] for s in st['scm']}
    hist = [h['action_id'] for h in st['history']]
    out.append(dict(i=i, record_id=rec['record_id'], world_id=wid, si=si, graph=g, cost=cost, regime=row['regime'],
                    behavior=row['behavior'], teacher=row['teachers'][si], label=act(rec['supervised_command']),
                    n_strict=len(rec['strict_pairs']), n_tied=len(rec['tied_pairs']),
                    n_pref=len(raw['preferences']), spent=st['spent_units'], remaining=st['remaining_units'],
                    terminal=st['terminal'], requests=st['requests'], prompt_tokens=len(rec['prompt_ids']),
                    n_commands=len(rec['commands']), hist_acq=hist[8:], noise=noise['Z'],
                    strict=[(act(a), act(b)) for a, b in rec['strict_pairs']],
                    cmd_tokens={act(c): len(t) for c, t in rec['commands'].items()}))
json.dump(out, open(sys.argv[2], 'w'))
C = collections.Counter
print('records', len(out), 'terminal', sum(r['terminal'] for r in out))
print('label marginal (all 160):', C(r['label'] for r in out).most_common())
print('label marginal (128 nonterminal):', C(r['label'] for r in out if not r['terminal']).most_common())
for t in METHODS:
    print(' teacher', t, 'nonterminal:', C(r['label'] for r in out if r['teacher'] == t and not r['terminal']).most_common(),
          '| terminal labels:', C(r['label'] for r in out if r['teacher'] == t and r['terminal']).most_common())
print('teacher x snapshot counts:', sorted(C((r['si'], r['teacher']) for r in out).items()))
print('behavior world counts:', C(r['behavior'] for r in out if r['si'] == 0))
print('graph world counts:', C(r['graph'] for r in out if r['si'] == 0), 'cost:', C(r['cost'] for r in out if r['si'] == 0),
      'regime:', C(r['regime'] for r in out if r['si'] == 0))
for si in range(5):
    rs = [r for r in out if r['si'] == si]
    print(' snapshot', si, 'labels', C(r['label'] for r in rs).most_common(), 'spent', sorted(C(r['spent'] for r in rs).items()),
          'remaining', sorted(C(r['remaining'] for r in rs).items()), 'requests', sorted(C(r['requests'] for r in rs).items()))
pt = [r['prompt_tokens'] for r in out]
print('prompt tokens min/median/max', min(pt), statistics.median(pt), max(pt))
print('command token lengths (incl EOS):', C(tuple(sorted(r['cmd_tokens'].items())) for r in out).most_common(3))
print('n_commands dist', C(r['n_commands'] for r in out))
print('total prefs', sum(r['n_pref'] for r in out), 'strict', sum(r['n_strict'] for r in out), 'tied', sum(r['n_tied'] for r in out))
print('records with >=1 strict among nonterminal:', sum(1 for r in out if not r['terminal'] and r['n_strict'] > 0),
      'with 0 strict:', [(r['record_id'], r['graph'], r['cost'], r['regime'], r['si']) for r in out if not r['terminal'] and r['n_strict'] == 0])
print('strict per nonterminal record: min/median/max', min(r['n_strict'] for r in out if not r['terminal']),
      statistics.median(r['n_strict'] for r in out if not r['terminal']), max(r['n_strict'] for r in out if not r['terminal']))
print('strict per snapshot index:', {si: sum(r['n_strict'] for r in out if r['si'] == si) for si in range(5)})
print('strict by graph:', {g: sum(r['n_strict'] for r in out if r['graph'] == g) for g in GRAPHS},
      'nonterminal recs by graph:', {g: sum(1 for r in out if r['graph'] == g and not r['terminal']) for g in GRAPHS})
print('strict by cost:', {c: sum(r['n_strict'] for r in out if r['cost'] == c) for c in COSTS})
print('strict by regime:', {x[0]: sum(r['n_strict'] for r in out if r['regime'] == x[0]) for x in REGIMES})
wins, losses = C(), C()
for r in out:
    for ch, rj in r['strict']:
        wins[ch] += 1; losses[rj] += 1
acts = sorted(set(wins) | set(losses))
print('ALL strict pairs: action -> wins, losses, win share')
for a in acts:
    print('  %-12s wins %4d losses %4d share %.3f' % (a, wins[a], losses[a], wins[a]/max(1, wins[a]+losses[a])))
# schedule
sched = p['schedule']
dpo = [s for s in sched if s['stage'] == 'dpo']
byid = {r['record_id']: r for r in out}
sel = [(byid[s['record_id']], byid[s['record_id']]['strict'][s['pair_index']]) for s in dpo]
print('DPO selected pairs:', len(sel))
cw, cl = C(ch for _, (ch, rj) in sel), C(rj for _, (ch, rj) in sel)
print('SELECTED 119: action -> chosen, rejected')
for a in sorted(set(cw) | set(cl)):
    print('  %-12s chosen %3d rejected %3d net %+d' % (a, cw[a], cl[a], cw[a]-cl[a]))
print('selected pair types (chosen>rejected):', C((ch, rj) for _, (ch, rj) in sel).most_common(60))
print('selected by graph:', C(r['graph'] for r, _ in sel), 'by si:', C(r['si'] for r, _ in sel), 'by cost:', C(r['cost'] for r, _ in sel))
print('SFT order first 12:', [(s['record_id'].split('/')[1][-2:], s['record_id'][-3:]) for s in sched[:12]])
print('exposures one_pass_dpo False because unselected strict pairs get 0; total strict unexposed:',
      sum(sum(1 for n in e['dpo_pairs'] if n == 0) for e in p['exposures'].values()),
      'exposed:', sum(sum(e['dpo_pairs']) for e in p['exposures'].values()),
      'max exposure of any pair:', max([n for e in p['exposures'].values() for n in e['dpo_pairs']] or [0]))
