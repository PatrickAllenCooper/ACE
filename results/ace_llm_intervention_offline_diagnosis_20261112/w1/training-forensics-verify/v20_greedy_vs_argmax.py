"""Sanity: final greedy generation vs argmax of the saved final full-command table; and what the saved SFT reference scores imply for greedy-vs-argmax."""
import json, os, glob, math
import numpy as np
from collections import Counter
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
nt=[r for r in R if not r['terminal']]
byid={r['rid']:r for r in R}
T=np.load(OUT+'/final_table.npy')
gen={}
for stage in ['sft','final']:
    for f in glob.glob(FIT+'/%s-generation-*.json'%stage):
        d=json.load(open(f))
        if d['ordinal']<160: gen[(stage,d['record_id'])]=json.loads(d['completion']).get('action_id','stop')
same=sum(1 for i,r in enumerate(nt) if gen[('final',r['rid'])]==ACTS[int(T[i].argmax())])
print('final greedy generation equals argmax of the final full-command table in %d/128 states'%same)
# first-branch token masses in the final table: P(set*) vs P(observe)
P=np.exp(T); pset=P[:,1:].sum(1); pobs=P[:,0]
print('final: P(observe_all) > sum of P(set_*) in %d/128 states ; median P(set_*)=%.3f P(observe)=%.3f ; total mass on 8 actions median %.3f'%((pobs>pset).sum(),np.median(pset),np.median(pobs),np.median(P.sum(1))))
# SFT checkpoint: saved reference entries
rows=json.load(open(OUT+'/pair_scores.json'))
obs_hi=[(x['rid'],sc) for x in rows for cmd,sc in [(x['ch'],x['ref_c']),(x['rj'],x['ref_r'])] if cmd=='observe_all']
print('SFT reference entries for observe_all: n=%d ; with P>0.5: %d ; SFT greedy output in those states: %s'%(len(obs_hi),sum(1 for _,s in obs_hi if s>math.log(0.5)),Counter(gen[('sft',rid)] for rid,s in obs_hi if s>math.log(0.5))))
print('SFT greedy output in states where the saved P(observe_all) is between 0.3 and 0.5:',Counter(gen[('sft',rid)] for rid,s in obs_hi if math.log(0.3)<s<=math.log(0.5)),' ; below 0.3:',Counter(gen[('sft',rid)] for rid,s in obs_hi if s<=math.log(0.3)))
xn=[(x['rid'],sc) for x in rows for cmd,sc in [(x['ch'],x['ref_c']),(x['rj'],x['ref_r'])] if cmd=='set_x_neg']
print('SFT reference entries for set_x_neg: n=%d ; P(set_x_neg) by SFT greedy output: %s'%(len(xn),{g:round(float(np.exp(np.mean([s for rid,s in xn if gen[("sft",rid)]==g]))),3) for g in set(gen[("sft",rid)] for rid,_ in xn)}))
print('   states where SFT greedy output is observe_all and set_x_neg is scored: P(set_x_neg) values',[round(math.exp(s),3) for rid,s in xn if gen[('sft',rid)]=='observe_all'])
print('   states where SFT greedy output is set_x_neg and observe_all is scored: P(observe_all) values',sorted([round(math.exp(s),3) for rid,s in obs_hi if gen[('sft',rid)]=='set_x_neg']))
