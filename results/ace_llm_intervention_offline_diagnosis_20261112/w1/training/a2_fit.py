"""Offline read-only analysis of saved fit evidence JSON (no tensors, no model)."""
import json, sys, collections, statistics, math, os
root = sys.argv[1]            # .../artifacts/fit
recs = json.load(open(sys.argv[2]))
byid = {r['record_id']: r for r in recs}
T = os.path.join(root, 'training')
plan = json.load(open(os.path.join(T, 'plan.json')))
sched = plan['schedule']
C = collections.Counter
def act(cmd):
    d = json.loads(cmd); return 'stop' if d.get('stop') else d['action_id']
ev = [json.load(open(os.path.join(T, 'step-%05d.complete.json' % i))) for i in range(len(sched))]
sft, dpo = ev[:160], ev[160:]
def q(xs):
    xs = sorted(xs); n = len(xs)
    return dict(min=xs[0], q25=xs[n//4], med=statistics.median(xs), q75=xs[(3*n)//4], max=xs[-1], mean=sum(xs)/n)
print('SFT loss (mean per-token NLL of label command):', {k: round(v, 4) for k, v in q([e['loss'] for e in sft]).items()})
for a, b in ((0, 20), (20, 40), (40, 80), (80, 120), (120, 160)):
    seg = sft[a:b]
    print('  SFT steps %3d-%3d mean loss %.4f  mean grad norm pre-clip %.3f  (nonterminal mean loss %.4f, terminal mean loss %.4f)' % (
        a, b-1, sum(e['loss'] for e in seg)/len(seg), sum(e['gradient_norm_before_clip'] for e in seg)/len(seg),
        statistics.mean([e['loss'] for i, e in enumerate(seg, a) if i % 5 != 4]), statistics.mean([e['loss'] for i, e in enumerate(seg, a) if i % 5 == 4])))
print('SFT grad norm pre-clip:', {k: round(v, 3) for k, v in q([e['gradient_norm_before_clip'] for e in sft]).items()})
print('SFT steps clipped (norm>1):', sum(e['gradient_norm_before_clip'] > 1 for e in sft), 'of', len(sft))
print('DPO loss:', {k: round(v, 5) for k, v in q([e['loss'] for e in dpo]).items()}, ' ln2=%.5f' % math.log(2))
print('DPO loss first 10:', [round(e['loss'], 5) for e in dpo[:10]])
print('DPO loss last 10:', [round(e['loss'], 5) for e in dpo[-10:]])
print('DPO grad norm pre-clip:', {k: round(v, 5) for k, v in q([e['gradient_norm_before_clip'] for e in dpo]).items()})
print('DPO steps clipped (norm>1):', sum(e['gradient_norm_before_clip'] > 1 for e in dpo), 'of', len(dpo))
print('DPO implied u=(dpol-dref) from loss via -logsigmoid(0.1*u): first 5',
      [round(-math.log(math.exp(e['loss'])-1)/0.1, 3) for e in dpo[:5]], 'last 5', [round(-math.log(math.exp(e['loss'])-1)/0.1, 3) for e in dpo[-5:]])
us = [-math.log(math.exp(e['loss'])-1)/0.1 for e in dpo]
print('DPO implied u at time of each step (nats): ', {k: round(v, 3) for k, v in q(us).items()}, ' count u>0:', sum(u > 0 for u in us))
print('rows/input tokens: sft rows', C(e['rows'] for e in sft), 'dpo rows', C(e['rows'] for e in dpo),
      'action_tokens sft', C(e['action_tokens'] for e in sft).most_common(), 'dpo', C(e['action_tokens'] for e in dpo).most_common())
print('total cuda ms sft %.1f dpo %.1f' % (sum(e['cuda_elapsed_ms'] for e in sft), sum(e['cuda_elapsed_ms'] for e in dpo)))
term = json.load(open(os.path.join(T, 'terminal.json')))
print('terminal:', {k: term[k] for k in ('status', 'completed_steps', 'sft_steps', 'dpo_steps', 'one_pass_sft', 'one_pass_dpo', 'reference_pair_count', 'elapsed_seconds', 'execution_kind')})
# Reference (post-SFT) scores vs final scores on the 119 pairs
ref = json.load(open(os.path.join(T, 'reference-scores.json')))['scores']
fin = json.load(open(os.path.join(root, 'final-before-reload.json')))
fscores, forder = fin['scores'], fin['commands']
final = {rid: dict(zip([act(c) for c in forder[rid]], fscores[rid])) for rid in fscores}
rows = []
for s in sched[160:]:
    rid, pi = s['record_id'], s['pair_index']
    key = json.dumps([rid, pi], sort_keys=True, separators=(',', ':'))
    r = byid[rid]; ch, rj = r['strict'][pi]
    rc, rr = ref[key]
    fc, fr = final[rid][ch], final[rid][rj]
    rows.append(dict(rid=rid, ch=ch, rj=rj, ref_margin=rc-rr, fin_margin=fc-fr, d_ch=fc-rc, d_rj=fr-rr))
print('pairs:', len(rows))
print('reference margin logp(chosen)-logp(rejected) under post-SFT model:', {k: round(v, 3) for k, v in q([x['ref_margin'] for x in rows]).items()},
      ' share ref already prefers chosen:', sum(x['ref_margin'] > 0 for x in rows), '/', len(rows))
print('final margin under final DPO model:', {k: round(v, 3) for k, v in q([x['fin_margin'] for x in rows]).items()},
      ' share final prefers chosen:', sum(x['fin_margin'] > 0 for x in rows), '/', len(rows))
print('margin change final-ref (u at end):', {k: round(v, 3) for k, v in q([x['fin_margin']-x['ref_margin'] for x in rows]).items()},
      ' improved:', sum(x['fin_margin'] > x['ref_margin'] for x in rows), '/', len(rows))
print('change in logp(chosen) final-ref:', {k: round(v, 3) for k, v in q([x['d_ch'] for x in rows]).items()})
print('change in logp(rejected) final-ref:', {k: round(v, 3) for k, v in q([x['d_rj'] for x in rows]).items()})
# Final model full-sequence scores at each nonterminal training record
arg = C(); rank_of_label = C(); arg_by_graph = collections.defaultdict(C)
mean_lp = collections.defaultdict(list); probs_xneg = []
for r in recs:
    if r['terminal']: continue
    sc = final[r['record_id']]
    best = max(sc, key=sc.get); arg[best] += 1; arg_by_graph[r['graph']][best] += 1
    order = sorted(sc, key=sc.get, reverse=True); rank_of_label[order.index(r['label'])] += 1
    z = math.log(sum(math.exp(v) for v in sc.values()))
    for a, v in sc.items(): mean_lp[a].append(v - z)
    probs_xneg.append(math.exp(sc['set_x_neg'] - z))
print('FINAL(DPO) adapter: argmax full-sequence logp over 9 legal commands at 128 nonterminal training states:', arg.most_common())
print('  by graph:', {g: c.most_common() for g, c in arg_by_graph.items()})
print('  rank of the SFT teacher label under final model (0=top):', sorted(rank_of_label.items()))
print('  mean renormalized prob over legal commands:', {a: round(statistics.mean(math.exp(x) for x in v), 4) for a, v in sorted(mean_lp.items())})
print('  P(set_x_neg | legal) across 128 states:', {k: round(v, 4) for k, v in q(probs_xneg).items()})
tot = [math.log(sum(math.exp(v) for v in final[r['record_id']].values())) for r in recs if not r['terminal']]
print('  log total mass on legal commands (unconstrained):', {k: round(v, 4) for k, v in q(tot).items()})
# Greedy constrained generations on the 160 training states (+1 repeat), both checkpoints
for stage in ('sft', 'final'):
    gens = [json.load(open(os.path.join(root, '%s-generation-%02d.json' % (stage, i)))) for i in range(161)]
    comp = [act(g['completion']) for g in gens[:160]]
    nt = [(r, c) for r, c in zip(recs, comp) if not r['terminal']]
    print(stage.upper(), 'greedy generation on 128 nonterminal training states:', C(c for _, c in nt).most_common(),
          '| terminal states:', C(c for r, c in zip(recs, comp) if r['terminal']).most_common())
    print('   agreement with SFT teacher label (nonterminal): %d/128' % sum(c == r['label'] for r, c in nt),
          ' by teacher:', {t: '%d/%d' % (sum(c == r['label'] for r, c in nt if r['teacher'] == t), sum(1 for r, _ in nt if r['teacher'] == t)) for t in ('uniform_random', 'menu_coverage', 'local_information', 'target_risk')})
    print('   by snapshot index:', {si: C(c for r, c in nt if r['si'] == si).most_common() for si in range(4)})
    print('   by graph:', {g: C(c for r, c in nt if r['graph'] == g).most_common() for g in ('chain', 'fork', 'collider')})
    if stage == 'final':
        sel = {s['record_id']: byid[s['record_id']]['strict'][s['pair_index']] for s in sched[160:]}
        print('   final greedy == DPO chosen action on its own exposed pair: %d/119; == rejected: %d/119' % (
            sum(c == sel[r['record_id']][0] for r, c in nt if r['record_id'] in sel), sum(c == sel[r['record_id']][1] for r, c in nt if r['record_id'] in sel)))
        print('   final greedy == argmax full-sequence score: %d/128' % sum(c == max(final[r['record_id']], key=final[r['record_id']].get) for r, c in nt))
summ = json.load(open(os.path.join(root, 'summary.json')))
print('summary:', {k: summ[k] for k in ('records', 'sft_steps', 'dpo_steps', 'training_updates', 'training_input_tokens', 'training_action_tokens', 'stop_count', 'action_count', 'generation_count', 'training_wall_seconds', 'elapsed_seconds', 'peak_gpu_allocated_bytes', 'device_name', 'started_utc', 'completed_utc')})
