import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import math, collections, pickle
import numpy as np
T = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training'
plan = json.load(open(T + '/plan.json')); ref = json.load(open(T + '/reference-scores.json'))['scores']
num = json.load(open(CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'))
recs = {r['record_id']: r for r in num['exposure']['records']}
tr = pickle.load(open(SCR + '/train_ref.pkl', 'rb'))
dpo = []
for step, s in enumerate(plan['schedule']):
    if s['stage'] != 'dpo': continue
    loss = json.load(open(T + '/step-%05d.loss.json' % step))['loss']
    pair = next(p for p in recs[s['record_id']]['pairs'] if p['pair_index'] == s['pair_index'])
    rc, rr = ref[json.dumps([s['record_id'], s['pair_index']], separators=(',', ':'))]
    c = json.loads(pair['chosen_command'])['action_id']; r = json.loads(pair['rejected_command'])['action_id']
    delta = -math.log(math.expm1(loss)) / 0.1
    trs = tr[s['record_id']]
    dpo.append(dict(step=step, ref_m=rc - rr, pol_m=rc - rr + delta, delta=delta, chosen=c, rejected=r, tr_agree=trs[c][0] > trs[r][0]))
n = len(dpo)
for lo, hi in [(0, 40), (40, 80), (80, 119), (0, 119)]:
    D = dpo[lo:hi]
    print('DPO steps %3d-%3d (n=%d): SFT-reference prefers chosen %d ; policy-at-first-exposure prefers chosen %d ; label agrees with TR ordering %d ; mean |delta| %.2f' % (
        lo, hi, len(D), sum(d['ref_m'] > 0 for d in D), sum(d['pol_m'] > 0 for d in D), sum(d['tr_agree'] for d in D), np.mean([abs(d['delta']) for d in D])))
flip_good = sum(1 for d in dpo if d['ref_m'] <= 0 < d['pol_m']); flip_bad = sum(1 for d in dpo if d['pol_m'] <= 0 < d['ref_m'])
print('pairs flipped to agree with label: %d ; flipped to disagree: %d' % (flip_good, flip_bad))
# does the policy's first-exposure preference agree with the TR ordering (rather than the noisy label)?
print('SFT-reference preference agrees with TR ordering: %d/119 ; policy-at-first-exposure agrees with TR ordering: %d/119' % (
    sum((d['ref_m'] > 0) == d['tr_agree'] for d in dpo), sum((d['pol_m'] > 0) == d['tr_agree'] for d in dpo)))
