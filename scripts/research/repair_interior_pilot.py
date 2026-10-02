"""Fixed two-case engineering pilot of independent diagnostic blocks."""
import pathlib,json,hashlib,time,itertools
from dataclasses import asdict
import numpy as np
from connected_motif import make_system,action_menu,sample
from connected_transfer_bridge_dev import posterior
from heldout_rbf_residual_screen import fit,predict
from conservative_repair_gate import decide
start=time.time();s=make_system(930217,3,1,.15);menu=[(0,(0,1),v) for v in itertools.product([-2.,-1.,0.,1.,2.],repeat=2)]
sv,sf,_=sample(s,np.random.default_rng(930218),32)
pm,pc=posterior(sf[:,0],sv[:,2],np.zeros(3),np.eye(3),.15)
arrays={'source_values':sv,'source_features':sf};rows=[]
for case,amp in enumerate([0.,.85]):
 noise=np.random.default_rng(940000+case);actions=np.random.default_rng(950000+case)
 vals=[];features=[];fit_actions=[]
 for k in range(16):
  ai=int(actions.integers(25));v,f,_=sample(s,noise,4,menu[ai],heldout_terms={0:amp});vals.append(v);features.append(f[:,0]);fit_actions.append(ai)
 z=np.concatenate(features);y=np.concatenate(vals)[:,2];lm,_=posterior(z,y,pm,pc,.15);coef=fit(z,y,pm,pc,1.5,10.)
 noise=np.random.default_rng(960000+case);actions=np.random.default_rng(970000+case);blocks=[];diagnostic=[];action_ids=[]
 for k in range(128):
  ai=int(actions.integers(25));v,f,_=sample(s,noise,4,menu[ai],heldout_terms={0:amp});q=f[:,0];truth=v[:,2]
  l=(np.sum(q*lm,axis=1)-truth)**2;c=(predict(q,coef,1.5)-truth)**2
  blocks.append([np.minimum(l,1).mean(),np.minimum(c,1).mean(),l.mean(),c.mean()]);diagnostic.append(v);action_ids.append(ai)
 b=np.asarray(blocks);assert np.isfinite(b).all();decision=decide(b[:,0],b[:,1],comparisons=2)
 rows.append({'case':'unchanged' if case==0 else 'changed','decision':asdict(decision),'paired_block_variance':float(np.var(b[:,1]-b[:,0],ddof=1)),'raw_baseline_mse':float(b[:,2].mean()),'raw_candidate_mse':float(b[:,3].mean())})
 arrays.update({f'case{case}_fit_features':z,f'case{case}_fit_y':y,f'case{case}_fit_actions':np.array(fit_actions),f'case{case}_diagnostic_values':np.array(diagnostic),f'case{case}_diagnostic_actions':np.array(action_ids),f'case{case}_block_losses':b})
p=pathlib.Path('results/repair_interior_pilot_20261002');p.mkdir(exist_ok=False);np.savez_compressed(p/'observations.npz',**arrays)
r={'protocol_file':'docs/development/guidance/repair_interior_pilot_2026-10-02.md','rows':rows,'generated_rows':1184,'seconds':time.time()-start,'scope':'one-system engineering pilot; no population safety claim','model_calls':0};(p/'result.json').write_text(json.dumps(r,indent=2)+'\n')
(p/'complete.json').write_text(json.dumps({'files':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['observations.npz','result.json']},'sources':{n:hashlib.sha256(pathlib.Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['repair_interior_pilot.py','connected_motif.py','connected_transfer_bridge_dev.py','heldout_rbf_residual_screen.py','conservative_repair_gate.py']}},indent=2)+'\n');print(json.dumps(r,indent=2))
