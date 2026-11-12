"""DPO-stage records: reference scores and per-step losses. Verifier's own code; no weights loaded."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import collections, math
import numpy as np
T = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training'
plan = json.load(open(T + '/plan.json'))
print('plan keys', list(plan.keys())[:30])
for k in ['recipe', 'counts', 'checkpoint_steps']:
    if k in plan: print(k, json.dumps(plan[k])[:1200])
ref = json.load(open(T + '/reference-scores.json'))['scores']
P = CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'
num = json.load(open(P))
# does the LLM plan share the exposure with the numeric plan?
for k in ['exposure_sha256']:
    print(k, plan.get(k), num.get(k))
ex = plan['exposure'] if 'exposure' in plan else num['exposure']
sched = ex['schedule']
recs = {r['record_id']: r for r in ex['records']}
beta = None
for path in [('recipe', 'beta'), ('recipe', 'dpo_beta')]:
    try: beta = plan[path[0]][path[1]]; print('beta from plan', path, beta)
    except Exception: pass
rows = []
for step, s in enumerate(sched):
    loss = json.load(open(T + '/step-%05d.loss.json' % step))['loss']
    if s['stage'] != 'dpo':
        rows.append(dict(step=step, stage='sft', loss=loss)); continue
    rec = recs[s['record_id']]
    pair = next(p for p in rec['pairs'] if p['pair_index'] == s['pair_index'])
    key = json.dumps([s['record_id'], s['pair_index']], separators=(',', ':'))
    rc, rr = ref[key]
    c = json.loads(pair['chosen_command'])['action_id']; r = json.loads(pair['rejected_command'])['action_id']
    rows.append(dict(step=step, stage='dpo', loss=loss, ref_c=rc, ref_r=rr, chosen=c, rejected=r, record=s['record_id']))
sft = [r for r in rows if r['stage'] == 'sft']; dpo = [r for r in rows if r['stage'] == 'dpo']
print('sft steps', len(sft), 'dpo steps', len(dpo))
print('SFT loss: first 5', [round(r['loss'], 3) for r in sft[:5]], 'last 5', [round(r['loss'], 3) for r in sft[-5:]], 'mean first 32 %.3f mean last 32 %.3f' % (np.mean([r['loss'] for r in sft[:32]]), np.mean([r['loss'] for r in sft[-32:]])))
print('DPO loss: first 3', [round(r['loss'], 4) for r in dpo[:3]], 'last 3', [round(r['loss'], 4) for r in dpo[-3:]], 'min %.4f max %.4f' % (min(r['loss'] for r in dpo), max(r['loss'] for r in dpo)))
b = beta if beta is not None else 0.1
for r in dpo:
    # loss = -log sigmoid(b * delta)  ->  delta = -(1/b) * log(exp(loss) - 1)
    r['delta'] = -math.log(math.expm1(r['loss'])) / b
    r['ref_margin'] = r['ref_c'] - r['ref_r']
    r['pol_margin'] = r['ref_margin'] + r['delta']
d = np.array([r['delta'] for r in dpo])
print('beta used', b, '| delta (policy log-odds shift vs reference at first exposure): min %.2f max %.2f mean %.2f; |delta| max %.2f' % (d.min(), d.max(), d.mean(), abs(d).max()))
print('SFT reference prefers chosen (ref_c>ref_r):', sum(r['ref_margin'] > 0 for r in dpo), 'of', len(dpo))
print('current policy at first exposure prefers chosen (pol_margin>0):', sum(r['pol_margin'] > 0 for r in dpo), 'of', len(dpo))
print('delta>0 (drift already favours chosen before its own update):', sum(r['delta'] > 0 for r in dpo), 'of', len(dpo))
# per-command drift model: delta_t = g(t) * (u[chosen] - u[rejected])
cmds = sorted({r['chosen'] for r in dpo} | {r['rejected'] for r in dpo})
ix = {c: i for i, c in enumerate(cmds)}
def fit(weights):
    X = np.zeros((len(dpo), len(cmds)))
    for i, r in enumerate(dpo):
        X[i, ix[r['chosen']]] += weights[i]; X[i, ix[r['rejected']]] -= weights[i]
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred = X @ coef
    r2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum()
    r2_0 = 1 - ((d - pred) ** 2).sum() / (d ** 2).sum()
    return coef - coef.mean(), r2, r2_0
t = np.arange(len(dpo), dtype=float)
sftmean = collections.defaultdict(list)
for r in dpo:
    sftmean[r['chosen']].append(r['ref_c']); sftmean[r['rejected']].append(r['ref_r'])
m = np.array([np.mean(sftmean[c]) for c in cmds])
for label, w in [('constant', np.ones(len(dpo))), ('linear in step', t / t.max()), ('sqrt step', np.sqrt(t / t.max()))]:
    coef, r2, r2_0 = fit(w)
    print('drift model [%s]: R2 (centred) %.3f, R2 (uncentred) %.3f; corr(per-command drift, SFT mean command loglik) r = %.3f' % (label, r2, r2_0, np.corrcoef(coef, m)[0, 1]))
    print('    drift', {SHORT[c]: round(float(x), 2) for c, x in zip(cmds, coef)})
print('SFT mean command log-likelihood (over appearances in exposed pairs)', {SHORT[c]: round(float(x), 2) for c, x in zip(cmds, m)}, 'n', {SHORT[c]: len(sftmean[c]) for c in cmds})
# last-third deltas by command pair involving x-
late = dpo[80:]
xs = [r['delta'] if r['chosen'] == 'set_x_neg' else -r['delta'] for r in late if 'set_x_neg' in (r['chosen'], r['rejected'])]
print('late pairs involving x-: n=%d, shift toward x- positive in %d, mean %.2f nats' % (len(xs), sum(x > 0 for x in xs), np.mean(xs) if xs else float('nan')))
xs_all = [(r['step'], round(r['delta'] if r['chosen'] == 'set_x_neg' else -r['delta'], 2)) for r in dpo if 'set_x_neg' in (r['chosen'], r['rejected'])]
print('all pairs involving x-: (step, shift toward x-)', xs_all)
