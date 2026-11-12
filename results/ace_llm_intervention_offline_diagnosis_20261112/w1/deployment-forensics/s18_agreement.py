"""Pairwise agreement between arms at the shared first decision (identical state)
and, for reference only, at matched (world, decision index) where states differ."""
import pickle, collections, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
act = {(r['arm'], r['world_idx'], r['decision_index']): r['action'] for r in rows}
sha = {(r['arm'], r['world_idx'], r['decision_index']): r['envelope_sha256'] for r in rows}
short = ['unif', 'menu', 'local', 'TR', 'ranker', 'frozen', 'sft', 'dpo']
print('Shared first decision (identical public state in all 8 arms): pairwise agreement counts out of 24 worlds')
print('%-8s' % '' + ' '.join('%6s' % s for s in short))
for i, a in enumerate(pids):
    print('%-8s' % short[i] + ' '.join('%6d' % sum(act[(a, w, 0)] == act[(b, w, 0)] for w in range(24)) for b in pids))
# shared states beyond decision 0?
bysha = collections.defaultdict(set)
for k, v in sha.items(): bysha[v].add(k)
multi = [ks for ks in bysha.values() if len({k[0] for k in ks}) > 1]
print('\nstates (envelope sha256) shared by more than one arm: %d ; all at decision 0: %s ; arms per shared state: %s' % (len(multi), all(k[2] == 0 for ks in multi for k in ks), collections.Counter(len(ks) for ks in multi)))
print('arms choosing the same first action still diverge at decision 1 (independent per-arm measurement streams): e.g. world 0, decision-1 envelope hashes of menu_coverage / sft / dpo (all chose set_x_neg at d=0):', [sha[(a, 0, 1)][:10] for a in ('menu_coverage', 'sft_llm', 'dpo_llm')])
print('\nMatched (world, decision index) agreement over all indexes where both arms made a decision (states differ after d=0; descriptive only):')
print('%-8s' % '' + ' '.join('%6s' % s for s in short))
for i, a in enumerate(pids):
    out = []
    for b in pids:
        ks = [(w, d) for w in range(24) for d in range(24) if (a, w, d) in act and (b, w, d) in act]
        out.append('%5.1f%%' % (100 * np.mean([act[(a, w, d)] == act[(b, w, d)] for w, d in ks])))
    print('%-8s' % short[i] + ' '.join(out))
print('\nranker vs target_risk at the first decision, by ranker choice:')
for c in ('set_x_neg', 'set_x_pos'):
    ws = [w for w in range(24) if act[('learned_ranker', w, 0)] == c]
    print('   ranker chose %s in %d worlds; target_risk chose there: %s' % (c, len(ws), dict(collections.Counter(act[('target_risk', w, 0)] for w in ws))))
