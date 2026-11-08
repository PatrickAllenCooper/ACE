"""Internal scorer checks; no independent benchmark membership implied."""
import copy,hashlib,json,pathlib
from temporal_action_benchmark import audit_task,score,public_task
p=pathlib.Path('results/action_state_contract_check_20261002/contract.json');contract=json.loads(p.read_text())
t={'id':'engineering1','description':'Internally authored test description','provenance':{'publisher':'engineering fixture','source_url':'internal:test','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_group':'internal','context_author':'fixture author','gold_adjudicator':'fixture checker','adjudication_status':'verified'},'contract':contract,'initial_state':{'latch':'open','shaking':False},'candidate_sequences':[['start'],['close_latch','start'],['close_latch','start','open_latch']],'gold_valid_sequences':[['close_latch','start']],'ambiguous':False}
assert score(t,{'id':t['id'],'abstain':False,'sequences':[['close_latch','start']]})['exact_menu']
assert score(t,{'id':t['id'],'abstain':False,'sequences':[['start']]})['false_legal']==1
assert score(t,{'id':t['id'],'abstain':False,'sequences':[['close_latch','start']]*2})['duplicates']==1
bad=copy.deepcopy(t);bad['gold_valid_sequences']=[['start']]
try:audit_task(bad)
except ValueError:pass
else:raise AssertionError('bad gold accepted')
a=copy.deepcopy(t);a['ambiguous']=True;a['gold_valid_sequences']=None
assert score(a,{'id':a['id'],'abstain':True,'sequences':[]})['ambiguity_abstention_correct']
assert 'contract' not in public_task(t) and 'gold_valid_sequences' not in public_task(t)
out=pathlib.Path('results/temporal_action_benchmark_check_20261002');out.mkdir(exist_ok=False)
r={'checks':'passed','independent_adjudicated_tasks':0,'model_calls':0,'hardware_calls':0,'scope':'internal software fixture; provenance metadata is not proof of independence'};b=(json.dumps(r,indent=2)+'\n').encode();(out/'result.json').write_bytes(b);(out/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(b).hexdigest(),'source_hashes':{n:hashlib.sha256(pathlib.Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['temporal_action_benchmark.py','check_temporal_action_benchmark.py']}},indent=2)+'\n');print(b.decode())
