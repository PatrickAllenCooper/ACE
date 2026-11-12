"""Offline read-only analysis of the saved training collection (JSON only)."""
import json, sys, os, collections, statistics, math, itertools
root = sys.argv[1]
recs = json.load(open(sys.argv[2]))
C = collections.Counter
def q(xs):
    xs = sorted(xs); n = len(xs)
    return dict(n=n, min=xs[0], q25=xs[n//4], med=statistics.median(xs), q75=xs[(3*n)//4], max=xs[-1], mean=sum(xs)/n)
term = json.load(open(os.path.join(root, 'terminal.json')))
print('collection terminal: status', term['status'], 'accounting', term['accounting'])
kinds = collections.defaultdict(lambda: C())
traj = []
for w in term['worlds']:
    a = w['accounting']
    for k in ('trajectory', 'branch', 'scoring'):
        kinds[k]['cost'] += a[k]['reserved_cost_units']; kinds[k]['responses'] += a[k]['executor_accepted_responses']
    traj.append(a['trajectory']['executor_accepted_responses'])
print('by kind:', {k: dict(v) for k, v in kinds.items()})
print('trajectory responses per world:', sorted(C(traj).items()))
T, strict_t = [], []
teach = []
allpairs = []
cyc = 0; cyc_possible = 0
neg_gain = 0; n_gain = 0
snr = []
for i, r in enumerate(recs):
    d = os.path.join(root, 'world-%03d' % (i // 5), 'snapshot-%03d' % (i % 5))
    t = json.load(open(os.path.join(d, 'teacher.json')))
    sel = json.load(open(os.path.join(d, 'selection.json')))
    assert t['method'] == r['teacher'], (t['method'], r['teacher'], r['record_id'])
    dec = t['decision']; lab = 'stop' if dec.get('stop') else dec['action_id']
    assert lab == r['label']
    md = t['metadata']
    sc = md.get('scores') or md.get('risk_reduction_per_cost')
    entry = dict(r=r, method=t['method'], label=lab, scores=sc, mc_se=md.get('monte_carlo_standard_errors'),
                 actual_cost=sel['actual_acquisition_cost_units'], threshold=sel['acquisition_cost_units'])
    teach.append(entry)
    if r['terminal']: continue
    fb = json.load(open(os.path.join(d, 'feedback', 'terminal.json')))['feedback']
    assert fb['tie_multiplier'] == 2.0 and fb['minimum_gain_gap'] == 0.0 and fb['replicate_count'] == 4 and fb['scale'] == 2.0
    assert len(fb['candidate_ids']) == 8 and len(fb['query_action_ids']) == 8 and fb['query_weights'] == [0.125]*8
    assert all(row == fb['scoring_truths'][0] for row in fb['scoring_truths'])
    entry['fb'] = fb
    mg = fb['mean_gains_per_cost']; pr = fb['per_replicate_gains_per_cost']
    for c in fb['candidate_ids']:
        n_gain += 1; neg_gain += mg[c] < 0
    # within-candidate replicate SD vs spread of candidate means
    within = statistics.mean(statistics.variance(pr[c]) for c in fb['candidate_ids'])
    between = statistics.variance([mg[c] for c in fb['candidate_ids']])
    snr.append(between / (within / 4) if within > 0 else float('nan'))   # ~F statistic, null expectation ~1
    pref = {}
    for p in fb['pairs']:
        se = p['paired_standard_error']
        tval = p['mean_gain_difference'] / se if se > 0 else float('inf') * (1 if p['mean_gain_difference'] > 0 else -1)
        T.append(abs(tval))
        allpairs.append((r, p, tval))
        if not p['tie']:
            pref[(p['chosen'], p['rejected'])] = True
    # count 3-cycles among strict prefs
    acts = fb['candidate_ids']
    beats = set(pref)
    for a, b, c in itertools.permutations(acts, 3):
        if a < b and a < c:  # each rotation counted once
            if (a, b) in beats and (b, c) in beats and (c, a) in beats: cyc += 1
    for a, b, c in itertools.combinations(acts, 3):
        e = sum(((x, y) in beats or (y, x) in beats) for x, y in ((a, b), (b, c), (a, c)))
        cyc_possible += e == 3
print('pairs', len(T), 'strict (|t|>2 rule):', sum(not p['tie'] for _, p, _ in allpairs), ' share %.4f' % (sum(not p['tie'] for _, p, _ in allpairs)/len(T)))
for thr in (1, 2, 3, 4, 6, 10):
    print('  share |t|>%d: %.4f' % (thr, sum(t > thr for t in T)/len(T)))
print('  t_3 null reference: P(|T|>1)=0.3910 P(|T|>2)=0.1393 P(|T|>3)=0.0577 P(|T|>4)=0.0280 P(|T|>6)=0.0093 P(|T|>10)=0.0021')
print('candidate mean gains per cost: negative %d of %d (%.3f)' % (neg_gain, n_gain, neg_gain/n_gain))
print('between-candidate / within F-like ratio across 128 states (null ~1):', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q([s for s in snr if s == s]).items()},
      ' share>2.36 (F(7,24) 5% crit ~2.42):', sum(s > 2.42 for s in snr), '/', len(snr))
print('strict triads: 3-cycles %d among fully-strict triads %d' % (cyc, cyc_possible))
# magnitude of gains
allmg = [v for e in teach if 'fb' in e for v in e['fb']['mean_gains_per_cost'].values()]
print('mean gain per cost (NMSE units) distribution:', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in q(allmg).items()})
basel = [sum(0.125*x for x in e['fb']['base_losses'][0]) for e in teach if 'fb' in e]
print('base NMSE at snapshot states:', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in q(basel).items()}, ' share already <=0.04:', sum(b <= 0.04 for b in basel), '/', len(basel))
for si in range(4):
    b = [sum(0.125*x for x in e['fb']['base_losses'][0]) for e in teach if 'fb' in e and e['r']['si'] == si]
    print('   snapshot', si, 'base NMSE median %.5f share<=0.04 %d/%d' % (statistics.median(b), sum(x <= 0.04 for x in b), len(b)),
          ' strict pairs', sum(e['r']['n_strict'] for e in teach if 'fb' in e and e['r']['si'] == si))
# best measured action per state (argmax mean gain per cost) distribution
best = C(max(e['fb']['mean_gains_per_cost'], key=e['fb']['mean_gains_per_cost'].get) for e in teach if 'fb' in e)
print('argmax measured mean gain per cost, 128 states:', best.most_common())
copeland = C()
for e in teach:
    if 'fb' not in e: continue
    w = C()
    for p in e['fb']['pairs']:
        if not p['tie']: w[p['chosen']] += 1; w[p['rejected']] -= 1
    if w:
        top = max(w.values()); tops = [a for a, v in w.items() if v == top]
        for a in tops: copeland[a] += 1/len(tops)
print('Copeland winner (strict wins-losses) per state, fractional for ties:', {k: round(v, 2) for k, v in copeland.most_common()})
# Agreement of strict measured preferences with teacher model scores where the teacher saved full-menu scores
for method in ('target_risk', 'local_information'):
    agree = tot = 0; top_in_strict = 0; lab_w = lab_l = 0
    for e in teach:
        if e['method'] != method or 'fb' not in e or not e['scores']: continue
        s = e['scores']
        for p in e['fb']['pairs']:
            if p['tie']: continue
            if s[p['chosen']] == s[p['rejected']]: continue
            tot += 1; agree += s[p['chosen']] > s[p['rejected']]
            if p['chosen'] == e['label']: lab_w += 1
            if p['rejected'] == e['label']: lab_l += 1
    print('teacher=%s states: strict measured pairs agreeing with teacher score order: %d/%d (%.3f); teacher label wins %d loses %d of its strict pairs' % (
        method, agree, tot, agree/max(1, tot), lab_w, lab_l))
# All-pairs (incl. ties) sign agreement of mean diff with target-risk score order
for method in ('target_risk', 'local_information'):
    agree = tot = 0
    for e in teach:
        if e['method'] != method or 'fb' not in e or not e['scores']: continue
        s = e['scores']
        for p in e['fb']['pairs']:
            ds = s[p['left']] - s[p['right']]
            if ds == 0 or p['mean_gain_difference'] == 0: continue
            tot += 1; agree += (ds > 0) == (p['mean_gain_difference'] > 0)
    print('teacher=%s states: sign(mean measured diff) agrees with teacher score order on all pairs: %d/%d (%.3f)' % (method, agree, tot, agree/max(1, tot)))
# teacher label vs SFT label across all four teachers at each state: margin top1-top2 relative to MC SE for target_risk
m = []
for e in teach:
    if e['method'] == 'target_risk' and e['scores']:
        s = sorted(e['scores'].items(), key=lambda kv: -kv[1])
        se = e['mc_se']
        m.append(((s[0][1]-s[1][1]) / max(1e-18, math.hypot(se[s[0][0]], se[s[1][0]])), s[0][0], s[1][0], e['r']['graph'], e['r']['si']))
print('target_risk teacher: top1-top2 margin in MC-SE units:', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in q([x[0] for x in m]).items()},
      ' share margin<2 SE:', sum(x[0] < 2 for x in m), '/', len(m))
print('target_risk teacher label x graph:', sorted(C((x[3], x[1]) for x in m).items()))
print('target_risk teacher label x snapshot:', sorted(C((x[4], x[1]) for x in m).items()))
li = [e for e in teach if e['method'] == 'local_information' and e['scores']]
print('local_information teacher label x graph:', sorted(C((e['r']['graph'], e['label']) for e in li).items()))
print('local_information teacher label x snapshot:', sorted(C((e['r']['si'], e['label']) for e in li).items()))
mc = [e for e in teach if e['method'] == 'menu_coverage' and not e['r']['terminal']]
print('menu_coverage teacher label x snapshot:', sorted(C((e['r']['si'], e['label']) for e in mc).items()))
print('menu_coverage teacher label x behavior:', sorted(C((e['r']['behavior'], e['label']) for e in mc).items()))
ur = [e for e in teach if e['method'] == 'uniform_random' and not e['r']['terminal']]
print('uniform_random teacher label x snapshot:', sorted(C((e['r']['si'], e['label']) for e in ur).items()))
print('label x snapshot x teacher (set_x_neg only):', sorted(C((e['r']['si'], e['method']) for e in teach if e['label'] == 'set_x_neg').items()))
# milestone overshoot
print('actual acquisition cost at snapshot vs threshold:', sorted(C((e['threshold'], e['actual_cost']) for e in teach).items()))
# behavior histories: action marginal of acquisition actions visited in states (history beyond init) at snapshot 4
hist = C(); hist_by_beh = collections.defaultdict(C)
for e in teach:
    if e['r']['si'] == 4:
        for a in e['r']['hist_acq']: hist[a] += 1; hist_by_beh[e['r']['behavior']][a] += 1
print('behavior-policy acquisition actions over 32 training trajectories:', hist.most_common(), 'total', sum(hist.values()))
for b, c in hist_by_beh.items(): print('   ', b, c.most_common(), sum(c.values()))
# fraction of training states whose acquisition history is all one action
print('training states (si 1-3) whose acquisition history has a single distinct action:',
      sum(1 for e in teach if e['r']['si'] in (1, 2, 3) and len(set(e['r']['hist_acq'])) == 1), 'of 96;',
      'all-set_x_neg histories:', sum(1 for e in teach if e['r']['si'] in (1, 2, 3) and set(e['r']['hist_acq']) == {'set_x_neg'}))
print('max share of set_x_neg in any si>=1 state history:', max(e['r']['hist_acq'].count('set_x_neg')/len(e['r']['hist_acq']) for e in teach if e['r']['si'] >= 1))
