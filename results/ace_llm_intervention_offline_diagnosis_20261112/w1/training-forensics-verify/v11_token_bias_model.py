"""Check J, second pass. Independent re-implementation of the earlier agent's DESCRIBED model
(from their notes section 6.7, not from their script): 8 token logits in a 3-node trie softmax
(observe/set ; _x/_y/_xy ; _neg/_pos/_zero shared by X and Y), no state input, initialised at the node
probabilities implied by mean SFT log-likelihoods, one unit-norm step per pair in the real order,
step size 0.9, momentum 0.5.  Then: sensitivity and null controls of my own."""
import json, os, math
import numpy as np
from collections import Counter, defaultdict
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
ACTS=['observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy']
IDX={a:i for i,a in enumerate(ACTS)}
rows=json.load(open(OUT+'/pair_scores.json'))
on=json.load(open(OUT+'/dpo_online.json'))
m_act=np.array(on['online_margin'])
c=np.array([IDX[x['ch']] for x in rows]); r_=np.array([IDX[x['rj']] for x in rows])
ent=[]
for x in rows:
    ent.append((x['ch'],x['ref_c'],x['fin_c'])); ent.append((x['rj'],x['ref_r'],x['fin_r']))
ref_mean=np.array([np.mean([e[1] for e in ent if e[0]==a]) for a in ACTS])
act_shift=np.array([np.mean([e[2]-e[1] for e in ent if e[0]==a]) for a in ACTS])
# token structure: logits order [observe,set | _x,_y,_xy | _neg,_pos,_zero]
PATH={'observe_all':(0,None,None),'set_xy':(1,4,None),'set_x_neg':(1,2,5),'set_x_pos':(1,2,6),'set_x_zero':(1,2,7),'set_y_neg':(1,3,5),'set_y_pos':(1,3,6),'set_y_zero':(1,3,7)}
NODES=[(0,1),(2,3,4),(5,6,7)]
def logp_all(th):
    ls=np.zeros(8)
    for lo in NODES:
        idx=list(lo); z=th[idx]; ls[idx]=z-(np.log(np.exp(z-z.max()).sum())+z.max())
    return np.array([sum(ls[k] for k in PATH[a] if k is not None) for a in ACTS]),ls
def grad_logp(th,a):
    g=np.zeros(8)
    for node,k in zip(NODES,PATH[a]):
        if k is None: continue
        idx=list(node); z=th[idx]; p=np.exp(z-z.max()); p/=p.sum()
        g[idx]-=p; g[k]+=1
    return g
# init from mean SFT log-likelihoods
p=np.exp(ref_mean); 
root=np.array([p[IDX['observe_all']],sum(p[IDX[a]] for a in ACTS if a!='observe_all')]); root/=root.sum()
sec=np.array([sum(p[IDX[a]] for a in ['set_x_neg','set_x_pos','set_x_zero']),sum(p[IDX[a]] for a in ['set_y_neg','set_y_pos','set_y_zero']),p[IDX['set_xy']]]); sec/=sec.sum()
suf=np.array([p[IDX['set_x_neg']]+p[IDX['set_y_neg']],p[IDX['set_x_pos']]+p[IDX['set_y_pos']],p[IDX['set_x_zero']]+p[IDX['set_y_zero']]]); suf/=suf.sum()
th0=np.log(np.concatenate([root,sec,suf]))
print('init node probabilities: root %s second %s suffix %s'%(np.round(root,3),np.round(sec,3),np.round(suf,3)))
def run(step,mom,th_init=th0,cc=c,rr=r_,order=None,unit=True):
    th=th_init.copy(); vel=np.zeros(8); lp0,_=logp_all(th_init); pred=[]
    n=len(cc); order=range(n) if order is None else order
    for i in order:
        lp,_=logp_all(th)
        pred.append((lp[cc[i]]-lp[rr[i]])-(lp0[cc[i]]-lp0[rr[i]]))
        g=grad_logp(th,ACTS[cc[i]])-grad_logp(th,ACTS[rr[i]])
        nrm=np.linalg.norm(g)
        if unit and nrm>0: g=g/nrm
        vel=mom*vel+g; th=th+step*vel
    lpf,_=logp_all(th)
    return np.array(pred),lpf-lp0,th
pred,shift,th=run(0.9,0.5)
print('\nAs described (step 0.9, momentum 0.5):')
print('  corr(predicted pre-update margins, 119 saved online margins) = %.4f ; sign agreement %d/119 ; R^2 vs zero prediction %.3f'%(np.corrcoef(pred,m_act)[0,1],int((np.sign(pred)==np.sign(m_act)).sum()),1-((m_act-pred)**2).sum()/(m_act**2).sum()))
print('  final per-command shift model vs observed:',{a:(round(float(s),2),round(float(o),2)) for a,s,o in zip(ACTS,shift,act_shift)})
print('  corr(final per-command shift) = %.4f ; winner: %s'%(np.corrcoef(shift,act_shift)[0,1],ACTS[int(np.argmax(logp_all(th)[0]))]))
# sensitivity
print('\nSensitivity to the two fitted hyperparameters (r_margin / r_shift):')
print('        mom '+' '.join('%11.1f'%m for m in [0.0,0.3,0.5,0.7,0.9]))
for step in [0.1,0.3,0.6,0.9,1.2,2.0]:
    line=[]
    for mom in [0.0,0.3,0.5,0.7,0.9]:
        pr,sh,_=run(step,mom); line.append('%5.2f/%5.2f'%(np.corrcoef(pr,m_act)[0,1],np.corrcoef(sh,act_shift)[0,1]))
    print('  step %.1f  '%step+' '.join(line))
# null controls
rng=np.random.default_rng(0)
def nullstats(kind,reps=2000):
    rm=[];rs=[];win=Counter()
    for _ in range(reps):
        if kind=='flip':
            f=rng.random(119)<0.5; cc=np.where(f,r_,c); rr=np.where(f,c,r_); order=None
            pr,sh,th=run(0.9,0.5,cc=cc,rr=rr)
            # compare model margins (for the flipped labels) with the ACTUAL saved margins oriented by the real labels
            pr_real=np.where(f,-pr,pr)
        elif kind=='order':
            order=rng.permutation(119); pr_,sh,th=run(0.9,0.5,order=order)
            pr_real=np.empty(119); pr_real[order]=pr_
        elif kind=='randpairs':
            cc=rng.integers(0,8,119); rr=(cc+rng.integers(1,8,119))%8
            pr,sh,th=run(0.9,0.5,cc=cc,rr=rr); pr_real=pr
        rm.append(np.corrcoef(pr_real,m_act)[0,1]); rs.append(np.corrcoef(sh,act_shift)[0,1]); win[ACTS[int(np.argmax(logp_all(th)[0]))]]+=1
    return np.array(rm),np.array(rs),win
for kind,desc in [('flip','coin-flip label directions on the actual pairs (margin compared in real orientation)'),('order','actual pairs in random order (margins re-aligned to real steps)'),('randpairs','entirely random command pairs')]:
    rm,rs,win=nullstats(kind,1000)
    print('NULL %s:\n   r_margin mean %.3f [5%%,95%%]=[%.3f,%.3f] ; r_shift mean %.3f [5%%,95%%]=[%.3f,%.3f] ; share r_shift>=0.93: %.3f ; winners %s'%(desc,np.nanmean(rm),np.nanpercentile(rm,5),np.nanpercentile(rm,95),np.nanmean(rs),np.nanpercentile(rs,5),np.nanpercentile(rs,95),(rs>=0.93).mean(),dict(win.most_common(4))))
# static baseline: how well does "rich get richer" alone (no pairs at all) predict the shifts?
print('\ncorr(SFT mean log-likelihood by command, observed DPO change) = %.3f'%np.corrcoef(ref_mean,act_shift)[0,1])
net=np.array([sum(1 for x in rows if x['ch']==a)-sum(1 for x in rows if x['rj']==a) for a in ACTS])
print('corr(label net preferred-rejected among the 119, observed change) = %.3f'%np.corrcoef(net,act_shift)[0,1])
# pure sharpening model: log pi_final = T * log pi_SFT (+const) -> margin_t predicted by (tau_t - 1)*(ref_c-ref_r) using the pair's own state-specific reference
refd=np.array([x['ref_c']-x['ref_r'] for x in rows])
print('corr(saved online margins, state-specific (ref_c-ref_r)) = %.3f ; corr with time-weighted t*(ref_c-ref_r) = %.3f'%(np.corrcoef(m_act,refd)[0,1],np.corrcoef(m_act,np.arange(119)*refd)[0,1]))
# compare: model prediction vs the state-specific sharpening predictor in a joint regression
X=np.column_stack([pred,np.arange(119)*refd/119.0,np.ones(119)])
coef,res,_,_=np.linalg.lstsq(X,m_act,rcond=None)
r2=lambda cols: 1-((m_act-X[:,cols]@np.linalg.lstsq(X[:,cols],m_act,rcond=None)[0])**2).sum()/((m_act-m_act.mean())**2).sum()
print('R^2 for saved online margins: token-bias model only %.3f ; time-weighted state-specific sharpening only %.3f ; both %.3f'%(r2([0,2]),r2([1,2]),r2([0,1,2])))
# residual of margins after the state-blind model: is it related to the state-specific reference difference?
resid=m_act-pred
print('corr(residual after state-blind model, state-specific deviation of ref difference from command means) = %.3f'%np.corrcoef(resid,refd-(ref_mean[c]-ref_mean[r_]))[0,1])
