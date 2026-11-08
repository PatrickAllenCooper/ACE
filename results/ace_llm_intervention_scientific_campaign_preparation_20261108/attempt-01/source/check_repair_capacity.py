"""Noiseless fixed-basis capacity check; no environment queries or tuning."""
import hashlib,itertools,json,pathlib
import numpy as np
from heldout_rbf_residual_screen import basis
p=np.array(list(itertools.product([-2.,-1.,0.,1.,2.],repeat=2)));x,y=p.T
base=np.column_stack([np.ones(len(x)),x,y,x*y]);z=np.column_stack([x,y,x*y]);full=np.column_stack([base,basis(z,1.5)])
candidates={'quadratic_x':x*x,'quadratic_y':y*y,'cubic_x':x**3,'saturating_odd':np.tanh(1.7*x+.8*y),'radial':np.exp(-.5*(x*x+y*y))}
rows=[]
for name,h in candidates.items():
 errors={}
 for label,a in [('baseline',base),('fixed_rbf',full)]:
  c=np.linalg.lstsq(a,h,rcond=None)[0];res=h-np.einsum('ij,j->i',a,c);errors[label]=float(np.sum(res**2)/np.sum(h**2))
 rows.append({'function':name,**errors,'capacity_reduction_fraction':1-errors['fixed_rbf']/errors['baseline']})
assert all(r['fixed_rbf']<=r['baseline']+1e-12 for r in rows)
out=pathlib.Path('results/repair_capacity_check_20261002');out.mkdir(exist_ok=False)
r={'rows':rows,'baseline_rank':int(np.linalg.matrix_rank(base)),'candidate_rank':int(np.linalg.matrix_rank(full)),'points':len(p),'width':1.5,'scope':'in-menu unregularized approximation ceiling, not trained repair performance or generalization','new_queries':0,'model_calls':0};raw=(json.dumps(r,indent=2)+'\n').encode();(out/'result.json').write_bytes(raw);(out/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(raw).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'basis_source_sha256':hashlib.sha256(pathlib.Path(__file__).with_name('heldout_rbf_residual_screen.py').read_bytes()).hexdigest()},indent=2)+'\n');print(raw.decode())
