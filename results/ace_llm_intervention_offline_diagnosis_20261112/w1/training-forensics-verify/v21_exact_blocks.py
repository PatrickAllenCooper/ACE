"""Exact (unrounded) block geometric-mean prequential probabilities by label; replaces the rounded figures printed by v16."""
import json, os, math
import numpy as np
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
byid={r['rid']:r for r in R}
def lab(r): return 'stop' if r['sup'] is None else r['sup']
steps=[json.load(open(FIT+'/training/step-%05d.complete.json'%i)) for i in range(160)]
for a in ['set_x_pos','set_x_neg','observe_all']:
    for lo,hi in [(0,40),(40,80),(80,120),(120,160)]:
        v=[-s['loss']*s['action_tokens'] for i,s in enumerate(steps) if lab(byid[s['record_id']])==a and lo<=i<hi]
        print('%-12s steps %3d-%3d: n=%2d mean lp %.3f gm-prob %.4f'%(a,lo,hi,len(v),np.mean(v),math.exp(np.mean(v))))
# labels in the last 30 steps
print('labels steps 130-159:',[lab(byid[s['record_id']]) for s in steps[130:]])
print('world of steps 135-137:',[s['record_id'] for s in steps[135:138]])
