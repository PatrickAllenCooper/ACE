"""Deterministic contract/edge-case checks and resource calculation; no SCM queries."""
import hashlib,json,pathlib
from dataclasses import asdict
from conservative_repair_gate import decide,sufficient_blocks
assert not decide([0.2]*6,[0.1]*6,comparisons=10).replace
assert not decide([0.2]*10000,[0.2]*10000,comparisons=10).replace
assert not decide([0.2]*10000,[0.3]*10000,comparisons=10).replace
assert decide([0.8]*1000,[0.1]*1000,comparisons=10).replace
for a,b in [([],[]),([0],[float('nan')]),([0],[-.1]),([0],[1.1]),([0,0],[0])]:
 try:decide(a,b,comparisons=10)
 except ValueError:pass
 else:raise AssertionError('invalid input accepted')
a=decide([.5]*500,[.3]*500,comparisons=1);b=decide([.5]*500,[.3]*500,comparisons=10);assert b.upper_bound>a.upper_bound
out=pathlib.Path('results/repair_gate_resource_diagnostic_20261002');out.mkdir(exist_ok=False)
r={'checks':'passed','scope':'deterministic API checks and analytic sufficient sample counts, not empirical safety validation','alpha':.05,'beta':.2,'comparisons':10,'counts':{str(g):sufficient_blocks(g) for g in [.2,.1,.05]},'six_block_example':asdict(decide([.2]*6,[.1]*6,comparisons=10)),'new_queries':0,'model_calls':0}
b=(json.dumps(r,indent=2)+'\n').encode();(out/'result.json').write_bytes(b);(out/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(b).hexdigest(),'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('conservative_repair_gate.py')]}},indent=2)+'\n');print(b.decode())
