"""SUPPLEMENT (training-side, not deployment): the only saved LLM command
log-likelihoods. reference-scores.json = SFT-adapter log-likelihood of the
chosen and rejected command for each of the 119 exposed DPO pairs.
step-NNNNN.loss.json = scalar loss per update; for DPO steps the implied
policy-vs-reference log-odds shift is recoverable: Delta = logit(exp(-loss))/beta."""
import json, os, collections, math, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
T = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training'
plan = json.load(open(T + '/plan.json'))
print('plan keys:', list(plan.keys()))
rec = plan.get('numeric_recipe') or plan.get('recipe')
print('recipe:', json.dumps(rec))
sched = plan['schedule']
print('schedule len', len(sched), 'first', json.dumps(sched[0])[:300])
print('dpo example', json.dumps(next(s for s in sched if s['stage'] == 'dpo'))[:400])
ref = json.load(open(T + '/reference-scores.json'))['scores']
ref = {(json.loads(k)[0], json.loads(k)[1]): v for k, v in ref.items()}
# pair identity from the numeric training plan (same exposure)
NP = json.load(open('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/cpu-phase/numeric-training-plan.json'))
recs = {r['record_id']: r for r in NP['exposure']['records']}
def cmd(c):
    d = json.loads(c); return d.get('action_id', 'STOP')
pairinfo = {}
for rid, r in recs.items():
    for q in r['pairs']:
        if q['exposure_count'] > 0:
            pairinfo[(rid, q['pair_index'])] = (cmd(q['chosen_command']), cmd(q['rejected_command']))
print('exposed pairs in numeric plan', len(pairinfo), 'reference score keys', len(ref), 'key match', set(pairinfo) == set(ref))
losses = [json.load(open(T + '/step-%05d.loss.json' % i))['loss'] for i in range(len(sched))]
beta = rec['beta']
print('beta', beta, 'lr', rec.get('learning_rate'))
# ---------- SFT stage
sft = [(i, s) for i, s in enumerate(sched) if s['stage'] == 'sft']
lab = [recs[s['record_id']]['supervised_command'] for _, s in sft]
L = np.array([losses[i] for i, _ in sft])
print('\nSFT per-step loss = mean per-token NLL of the label command+EOS (before that update). 160 single-example steps, one pass.')
for a, b in ((0, 20), (20, 40), (40, 80), (80, 120), (120, 160)):
    print('   steps %3d-%3d mean loss %.3f (implied mean command log-lik ~ -%.2f for an 9-token command)' % (a, b - 1, L[a:b].mean(), L[a:b].mean() * 9))
bylab = collections.defaultdict(list)
for (i, s), l in zip(sft, lab): bylab[cmd(l)].append((i, losses[i]))
print('   by label (all steps; last-80-steps mean):')
for k, v in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
    late = [x for i, x in v if i >= 80]
    print('     %-11s n=%2d mean %.3f ; late n=%2d mean %s' % (k, len(v), np.mean([x for _, x in v]), len(late), ('%.3f' % np.mean(late)) if late else 'n/a'))
# ---------- reference (SFT) scores on the 119 DPO pairs
print('\nSFT reference log-likelihoods (full command + EOS, nats) at the 119 exposed DPO training pairs:')
byc = collections.defaultdict(list)
margins = []; agree = 0
for key, (c, r) in pairinfo.items():
    lc, lr = ref[key]
    byc[c].append(lc); byc[r].append(lr)
    margins.append(lc - lr); agree += lc > lr
margins = np.array(margins)
print('   SFT prefers the preference-chosen command in %d of 119 pairs (%.1f%%) ; mean margin (chosen - rejected) %.3f, median %.3f, sd %.3f' % (agree, 100 * agree / 119, margins.mean(), np.median(margins), margins.std(ddof=1)))
print('   binomial test vs 0.5: p=%.3f' % stats.binomtest(agree, 119, 0.5).pvalue)
print('   per-command SFT log-lik across training states: n, mean, sd (between-state), min, max, mean prob')
cm = {}
for a in ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']:
    v = np.array(byc[a]); cm[a] = v.mean()
    print('     %-11s n=%2d mean %6.3f sd %.3f min %6.3f max %6.3f  mean p %.3f' % (a, len(v), v.mean(), v.std(ddof=1), v.min(), v.max(), np.exp(v).mean()))
means = np.array(list(cm.values()))
pooled_sd = np.sqrt(np.mean([np.var(byc[a], ddof=1) for a in cm]))
print('   SD of the 8 command means (between-command gaps): %.3f ; pooled within-command between-state SD: %.3f ; ratio state/command = %.2f' % (means.std(ddof=1), pooled_sd, pooled_sd / means.std(ddof=1)))
# by snapshot index (history length)
print('   SFT log-lik of set_x_neg and set_x_pos by training snapshot index (0: remaining 24, 1: ~18, 2: ~12, 3: ~6):')
for a in ['set_x_neg', 'set_x_pos', 'observe_all', 'set_xy', 'set_y_neg']:
    d = collections.defaultdict(list)
    for key, (c, r) in pairinfo.items():
        snap = int(key[0].rsplit('-', 1)[1])
        if c == a: d[snap].append(ref[key][0])
        if r == a: d[snap].append(ref[key][1])
    print('     %-11s' % a, {k: (len(v), round(float(np.mean(v)), 2)) for k, v in sorted(d.items())})
# ---------- DPO stage
dpo = [(i, s) for i, s in enumerate(sched) if s['stage'] == 'dpo']
print('\nDPO stage: 119 single-pair steps. Delta = policy log-odds(chosen vs rejected) minus SFT-reference log-odds, at the time of the step (before the update).')
rowsd = []
for n, (i, s) in enumerate(dpo):
    key = (s['record_id'], s['pair_index'])
    c, r = pairinfo[key]
    l = losses[i]
    delta = math.log(math.exp(-l) / (1 - math.exp(-l))) / beta if l > 1e-12 else float('inf')
    lc, lr = ref[key]
    rowsd.append({'n': n, 'step': i, 'loss': l, 'delta': delta, 'ref_margin': lc - lr, 'policy_margin': lc - lr + delta, 'chosen': c, 'rejected': r, 'rid': key[0]})
Dl = np.array([x['delta'] for x in rowsd]); Ls = np.array([x['loss'] for x in rowsd])
for a, b in ((0, 10), (10, 30), (30, 60), (60, 90), (90, 119)):
    seg = rowsd[a:b]
    print('   dpo steps %3d-%3d: mean loss %.3f ; Delta mean %+7.2f median %+7.2f min %+7.2f max %+7.2f ; Delta>0 in %d/%d ; policy prefers chosen in %d/%d (SFT ref: %d/%d)' % (
        a, b - 1, np.mean([x['loss'] for x in seg]), np.mean([x['delta'] for x in seg]), np.median([x['delta'] for x in seg]), min(x['delta'] for x in seg), max(x['delta'] for x in seg),
        sum(x['delta'] > 0 for x in seg), len(seg), sum(x['policy_margin'] > 0 for x in seg), len(seg), sum(x['ref_margin'] > 0 for x in seg), len(seg)))
print('   loss: first %.4f (=ln2 %.4f), min %.4f, max %.4f, last %.4f' % (Ls[0], math.log(2), Ls.min(), Ls.max(), Ls[-1]))
# does Delta depend on whether set_x_neg is the chosen or the rejected command?
print('\n   Delta by role of each command in the pair (late DPO steps 60-118 only): mean Delta when command is CHOSEN vs when REJECTED')
late = rowsd[60:]
for a in ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']:
    ch = [x['delta'] for x in late if x['chosen'] == a]; rj = [x['delta'] for x in late if x['rejected'] == a]
    print('     %-11s chosen n=%2d mean %s | rejected n=%2d mean %s' % (a, len(ch), ('%+7.2f' % np.mean(ch)) if ch else '   n/a', len(rj), ('%+7.2f' % np.mean(rj)) if rj else '   n/a'))
print('\n   all 119 DPO steps (n, loss, Delta, SFT ref margin, chosen > rejected, record):')
for x in rowsd:
    print('     %3d loss %.4f Delta %+8.2f refmargin %+6.2f  %-10s > %-10s %s' % (x['n'], x['loss'], x['delta'], x['ref_margin'], x['chosen'], x['rejected'], x['rid'][-25:]))
pickle.dump(rowsd, open(HERE + '/out/dpo_steps.pkl', 'wb'))
