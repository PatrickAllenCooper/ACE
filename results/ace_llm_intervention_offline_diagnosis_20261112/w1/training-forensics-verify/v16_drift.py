"""Is the SFT model's set_x_pos neglect at the end of the pass drift (recency) or state-conditioning? Uses only saved per-step losses and reference scores."""
import json, os, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
byid={r['rid']:r for r in R}
def lab(r): return 'stop' if r['sup'] is None else r['sup']
steps=[json.load(open(FIT+'/training/step-%05d.complete.json'%i)) for i in range(160)]
for a in ['set_x_pos','set_x_neg','observe_all']:
    seq=[(i,byid[s['record_id']]['snapshot'],round(math.exp(-s['loss']*s['action_tokens']),3)) for i,s in enumerate(steps) if lab(byid[s['record_id']])==a]
    print(a,'(step, snapshot, prequential p(label)):')
    print('   ',seq)
    for lo,hi in [(0,40),(40,80),(80,120),(120,160)]:
        v=[math.log(p) for i,_,p in seq if lo<=i<hi and p>0]
        if v: print('    steps %3d-%3d: n=%2d gm-prob %.3f'%(lo,hi,len(v),math.exp(np.mean(v))))
rows=json.load(open(OUT+'/pair_scores.json'))
print('\nSFT reference (end of SFT) scores for set_x_pos / set_x_neg / observe_all entries, split by whether that command is the record\'s own SFT label:')
for a in ['set_x_pos','set_x_neg','observe_all','set_xy','set_y_neg','set_y_pos']:
    own=[];oth=[]
    for x in rows:
        r=byid[x['rid']]
        for cmd,sc in [(x['ch'],x['ref_c']),(x['rj'],x['ref_r'])]:
            if cmd==a and r['snapshot']>0: (own if r['sup']==a else oth).append(sc)
    print('  %-12s snapshots 1-3: own-label states n=%d gm-prob %s | other states n=%d gm-prob %.3f'%(a,len(own),('%.3f'%math.exp(np.mean(own))) if own else 'NA',len(oth),math.exp(np.mean(oth)) if oth else float('nan')))
# position in the stream of the reference entries: world index of the record vs ref prob for x_pos (later worlds were trained on more recently)
xp=[(byid[x['rid']]['world_id']%100,byid[x['rid']]['snapshot'],sc) for x in rows for cmd,sc in [(x['ch'],x['ref_c']),(x['rj'],x['ref_r'])] if cmd=='set_x_pos']
w=np.array([a for a,_,_ in xp]); s=np.array([c for _,_,c in xp])
print('\ncorr(world index, SFT reference lp(set_x_pos)) over %d entries = %.3f'%(len(xp),np.corrcoef(w,s)[0,1]))
xn=[(byid[x['rid']]['world_id']%100,sc) for x in rows for cmd,sc in [(x['ch'],x['ref_c']),(x['rj'],x['ref_r'])] if cmd=='set_x_neg']
print('corr(world index, SFT reference lp(set_x_neg)) over %d entries = %.3f'%(len(xn),np.corrcoef([a for a,_ in xn],[b for _,b in xn])[0,1]))
