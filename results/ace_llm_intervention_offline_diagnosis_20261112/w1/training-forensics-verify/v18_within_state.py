"""Within-state comparison for the 27 records whose own SFT label also appears in their DPO pair:
log-likelihood of the label (a) just before the SFT update on that record (per-step loss), (b) at the end of SFT (reference score), (c) at the final checkpoint."""
import json, os, math
import numpy as np
from scipy import stats
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
FIT=TG+'/terminal01/remote/launch/training/session/worker/artifacts/fit'
byid={r['rid']:r for r in R}
steps={}
for i in range(160):
    s=json.load(open(FIT+'/training/step-%05d.complete.json'%i)); steps[s['record_id']]=(i,-s['loss']*s['action_tokens'])
rows=json.load(open(OUT+'/pair_scores.json'))
out=[]
for x in rows:
    r=byid[x['rid']]
    for cmd,ref,fin,role in [(x['ch'],x['ref_c'],x['fin_c'],'pref'),(x['rj'],x['ref_r'],x['fin_r'],'rej')]:
        if cmd==r['sup']:
            i,pre=steps[x['rid']]; out.append((i,r['snapshot'],cmd,role,pre,ref,fin))
out.sort()
print('n records',len(out))
print('step snap label        role  lp_before_own_SFT_update  lp_end_of_SFT  lp_final')
for o in out: print('%4d %4d %-12s %-4s %10.2f %14.2f %10.2f'%o)
pre=np.array([o[4] for o in out]); ref=np.array([o[5] for o in out]); fin=np.array([o[6] for o in out]); st=np.array([o[0] for o in out])
print('\nmean lp before own update %.3f ; end of SFT %.3f ; final %.3f'%(pre.mean(),ref.mean(),fin.mean()))
late=st>=60
print('records trained at step>=60 (n=%d): before %.3f -> end of SFT %.3f ; paired t p=%.3f ; improved in %d/%d'%(late.sum(),pre[late].mean(),ref[late].mean(),stats.ttest_rel(ref[late],pre[late])[1],(ref[late]>pre[late]).sum(),late.sum()))
for a in ['set_x_pos','set_x_neg','observe_all']:
    m=np.array([o[2]==a for o in out])&late
    if m.sum(): print('   %-12s n=%d before %.2f (gm-p %.3f) -> end of SFT %.2f (gm-p %.3f)'%(a,m.sum(),pre[m].mean(),math.exp(pre[m].mean()),ref[m].mean(),math.exp(ref[m].mean())))
