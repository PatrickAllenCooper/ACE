"""DPO stage: per-command signed shifts over time, and beta provenance. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import collections, math, re
import numpy as np
T = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training'
plan = json.load(open(T + '/plan.json'))
print('numeric_recipe', json.dumps(plan['numeric_recipe'])[:1500])
print('exposures type', type(plan['exposures']).__name__, 'len', len(plan['exposures']), '| schedule len', len(plan['schedule']), plan['schedule'][0], plan['schedule'][160])
print('records type', type(plan['records']).__name__, len(plan['records']))
r0 = plan['records'][0] if isinstance(plan['records'], list) else next(iter(plan['records'].values()))
print('record keys', list(r0.keys()) if isinstance(r0, dict) else type(r0))
e0 = plan['exposures'][160] if isinstance(plan['exposures'], list) else None
print('exposure[160]', json.dumps(e0)[:600] if e0 else None)
ref = json.load(open(T + '/reference-scores.json'))['scores']
num = json.load(open(CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'))
recs = {r['record_id']: r for r in num['exposure']['records']}
raws = {k: json.loads(v) for k, v in num['raw_records_utf8'].items()}
# check the LLM plan schedule equals the numeric plan's exposure schedule
same = all(a['record_id'] == b['record_id'] and a.get('pair_index') == b.get('pair_index') and a['stage'] == b['stage'] for a, b in zip(plan['schedule'], num['exposure']['schedule']))
print('LLM plan schedule == numeric exposure schedule:', same, len(plan['schedule']), len(num['exposure']['schedule']))
b = 0.1
dpo = []
for step, s in enumerate(plan['schedule']):
    if s['stage'] != 'dpo': continue
    loss = json.load(open(T + '/step-%05d.loss.json' % step))['loss']
    rec = recs[s['record_id']]
    pair = next(p for p in rec['pairs'] if p['pair_index'] == s['pair_index'])
    rc, rr = ref[json.dumps([s['record_id'], s['pair_index']], separators=(',', ':'))]
    c = json.loads(pair['chosen_command'])['action_id']; r = json.loads(pair['rejected_command'])['action_id']
    st = raws[s['record_id']]['envelope']['state']
    dpo.append(dict(step=step, loss=loss, delta=-math.log(math.expm1(loss)) / b, ref_c=rc, ref_r=rr, chosen=c, rejected=r, rem=st['remaining_units'], record=s['record_id']))
for cmd in ['observe_all', 'set_x_neg', 'set_x_pos']:
    xs = [(r['step'], r['rem'], SHORT[r['rejected'] if r['chosen'] == cmd else r['chosen']], 'C' if r['chosen'] == cmd else 'R', round(r['delta'] if r['chosen'] == cmd else -r['delta'], 2)) for r in dpo if cmd in (r['chosen'], r['rejected'])]
    print(SHORT[cmd], 'pairs', len(xs), 'chosen in', sum(1 for x in xs if x[3] == 'C'), '| shift toward it >0 in', sum(1 for x in xs if x[4] > 0), '| mean shift %.2f' % np.mean([x[4] for x in xs]))
    print('    (step, remaining, other, C/R, shift toward cmd):', xs)
# direct obs-vs-x- pairs
print('direct obs vs x- pairs:', [(r['step'], r['rem'], SHORT[r['chosen']], round(r['delta'], 2), round(r['ref_c'], 2), round(r['ref_r'], 2)) for r in dpo if {r['chosen'], r['rejected']} == {'observe_all', 'set_x_neg'}])
# SFT reference log-likelihood of x- and obs by remaining budget
for cmd in ['observe_all', 'set_x_neg', 'set_x_pos', 'set_y_neg']:
    byrem = collections.defaultdict(list)
    for r in dpo:
        if r['chosen'] == cmd: byrem[r['rem']].append(r['ref_c'])
        if r['rejected'] == cmd: byrem[r['rem']].append(r['ref_r'])
    print('SFT reference loglik of', SHORT[cmd], 'by remaining budget:', {k: (len(v), round(float(np.mean(v)), 2)) for k, v in sorted(byrem.items())})
# richer drift models
cmds = sorted({r['chosen'] for r in dpo} | {r['rejected'] for r in dpo}); ix = {c: i for i, c in enumerate(cmds)}
d = np.array([r['delta'] for r in dpo]); t = np.array([r['step'] - 160 for r in dpo], float) / 118
def r2_of(X):
    coef, *_ = np.linalg.lstsq(X, d, rcond=None); pred = X @ coef
    return 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum(), coef
X0 = np.zeros((len(dpo), len(cmds)))
for i, r in enumerate(dpo): X0[i, ix[r['chosen']]] += 1; X0[i, ix[r['rejected']]] -= 1
print('R2 constant drift %.3f | constant+linear-in-step per command %.3f | linear only %.3f | quadratic only %.3f' % (
    r2_of(X0)[0], r2_of(np.hstack([X0, X0 * t[:, None]]))[0], r2_of(X0 * t[:, None])[0], r2_of(X0 * (t ** 2)[:, None])[0]))
for lo in [0, 40, 60, 80]:
    sel = np.arange(len(dpo)) >= lo
    Xs = X0[sel]; ds = d[sel]
    coef, *_ = np.linalg.lstsq(Xs, ds, rcond=None); pred = Xs @ coef
    print('  steps >= DPO-%d (n=%d): constant-drift R2 %.3f' % (lo, sel.sum(), 1 - ((ds - pred) ** 2).sum() / ((ds - ds.mean()) ** 2).sum()))
