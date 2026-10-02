"""Post hoc comparison on saved independent blocks; generates no new responses."""
import hashlib,json,pathlib,math
from dataclasses import asdict
import numpy as np
from conservative_repair_gate import empirical_bernstein
root=pathlib.Path('results/repair_interior_pilot_20261002')
r=json.loads((root/'complete.json').read_text())
for n,h in r['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
x=np.load(root/'observations.npz');rows=[]
for case in [0,1]:
 b=x[f'case{case}_block_losses'];decision=empirical_bernstein(b[:,0],b[:,1],comparisons=2)
 rows.append({'case':case,'decision':asdict(decision)})
# Independent direct transformed [0,1] formula validates range scaling.
b=x['case1_block_losses'];d=b[:,1]-b[:,0];z=(d+1)/2
u=2*(float(z.mean())+math.sqrt(2*float(z.var(ddof=1))*math.log(80)/len(z))+7*math.log(80)/(3*(len(z)-1)))-1
assert abs(u-rows[1]['decision']['upper_bound'])<1e-12
assert not empirical_bernstein([.2]*128,[.3]*128,comparisons=2).replace
assert empirical_bernstein([.8]*128,[.1]*128,comparisons=2).replace
p=pathlib.Path('results/repair_bernstein_diagnostic_20261002');p.mkdir(exist_ok=False)
data={'source':'https://arxiv.org/abs/0907.3740','rows':rows,'scope':'post hoc development; no raw-MSE safety certificate','input_receipt_sha256':hashlib.sha256((root/'complete.json').read_bytes()).hexdigest(),'new_queries':0,'model_calls':0,'range_scaling_check':'passed'}
raw=(json.dumps(data,indent=2)+'\n').encode();(p/'result.json').write_bytes(raw);(p/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(raw).hexdigest(),'sources':{n:hashlib.sha256(pathlib.Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['conservative_repair_gate.py','diagnose_repair_bernstein.py']}},indent=2)+'\n');print(raw.decode())
