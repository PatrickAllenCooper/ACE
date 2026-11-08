"""Saved-byte result checks only. No runtime staging, tokenization or models."""
from pathlib import Path
import datetime,hashlib,json,subprocess,types
REPO=Path('/Users/pat/code/ACE')
BASE=REPO/'results/ace_llm_intervention_training_cpu_20261108'
LAUNCH=BASE/'launch01'
LOCAL=BASE/'validation01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-cpu-validation-20261108-01')
RAW=BASE/'terminal01/remote'
TERMINAL_EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-cpu-terminal-20261108-01')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def module(name):
 raw=(LAUNCH/name).read_bytes();assert sha(raw)==packet['files'][name]
 obj=types.ModuleType('saved_'+name[:-3]);obj.__file__=str(LAUNCH/name)
 exec(compile(raw,str(LAUNCH/name),'exec'),vars(obj));return obj
packet=read(LAUNCH/'preparation.json');packet_pin=sha((LAUNCH/'preparation.json').read_bytes())
outer=read(LAUNCH/'packet.json');disposition=read(BASE/'terminal01/disposition.json')
assert disposition['job_id']=='35374034' and disposition['scheduler_state']=='COMPLETED' and disposition['scheduler_exit']=='0:0'
assert disposition['resource_contract_matched'] and disposition['accounting_errors']==[] and disposition['input_pin_mismatches']==[]
for root in [BASE/'terminal01',TERMINAL_EXT]:
 actual={str(p.relative_to(root/'remote')) for p in (root/'remote').rglob('*') if p.is_file()}
 assert actual==set(disposition['files'])
 for name,row in disposition['files'].items():
  p=root/'remote'/name;b=p.read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'],name
for name,pin in {**outer['new_input_files'],'packet.json':sha((LAUNCH/'packet.json').read_bytes())}.items():
 assert sha((RAW/name).read_bytes())==pin
p=module('prepare_llm_intervention_training_cpu.py');s=module('llm_intervention_model_service.py')
c=module('qualify_llm_intervention_service_cpu.py');d=module('qualify_llm_intervention_runtime_metadata.py')
r=module('llm_intervention_service_runtime.py');p.packet_shape(packet,s,c,d)
q=RAW/'qualification';stage=read(q/'stage.json');stage_pin=sha((q/'stage.json').read_bytes())
data={n:(RAW/n).read_bytes() for n in packet['files']}
parent=read(q/'terminal.json');allocation=read(q/'allocation.json')
assert allocation['returncode']==0;c.allocation_fields(allocation['stdout'],'35374034')
worker=q/'session/worker/artifacts'
worker_pin=p.validate_worker(worker,packet,packet_pin,data,s,c,d,stage,stage_pin,read(q/'independent-provenance-sources.json'),r)
assert not list(q.rglob('failure.json'))
assert parent['schema']==p.VERSION and parent['status']=='cpu_preparation_complete'
assert parent['packet_sha256']==packet_pin and parent['stage_sha256']==stage_pin
assert parent['session']['terminal_sha256']==worker_pin and parent['elapsed_seconds']<580
assert all(parent[k] is False for k in ['model_weights_loaded','training_execution_qualified','scientific_qualified'])
assert parent['new_training_updates']==parent['environment_responses']==0
cleanup=read(q/'session/worker.cleanup.json');returned=read(q/'session/worker.returned.json')
assert cleanup==parent['session']['cleanup'] and cleanup['returncode']==0 and not cleanup['errors'] and cleanup['group_still_exists'] is False
assert returned['returncode']==0 and returned['terminal_sha256']==worker_pin
prepared=read(worker/'prepared.json');plan=prepared['training_plan']
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'saved_byte_checks_passed_independent_review_pending',
 'job_id':'35374034','source_commit':outer['source_commit'],'packet_sha256':sha((LAUNCH/'packet.json').read_bytes()),
 'preparation_sha256':packet_pin,'stage_sha256':stage_pin,'worker_terminal_sha256':worker_pin,
 'supervisor_terminal_sha256':sha((q/'terminal.json').read_bytes()),'prepared_file_sha256':sha((worker/'prepared.json').read_bytes()),
 'training_plan_sha256':prepared['training_plan_sha256'],'exposure_sha256':prepared['exposure_sha256'],
 'records':len(plan['records']),'prompt_token_counts':[r['token_metadata']['prompt_tokens'] for r in plan['records']],
 'completion_token_lengths':[[len(ids) for ids in r['commands'].values()] for r in plan['records']],
 'stored_tokens':plan['stored_tokens'],'sft_steps':plan['sft_steps'],'dpo_steps':plan['dpo_steps'],
 'exposure_counts':prepared['exposure']['counts'],'raw_files':disposition['raw_files'],'raw_bytes':disposition['raw_bytes'],
 'allocated_cpu_seconds':disposition['allocated_cpu_seconds'],'gpu_slice_seconds':disposition['gpu_slice_seconds'],
 'closed_cpu_seconds':disposition['known_closed_cpu_seconds'],'cpu_admission_after_release':disposition['cpu_admission_after_release'],
 'closed_gpu_seconds':disposition['gpu_admission_seconds'],'supervisor':parent,
 'new_training_updates':0,'new_environment_responses':0,'scientific_qualified':False,'training_qualified':False,
 'actual_training_plan_qualified':False,'next_gate':'independent result acceptance then bounded integrated fresh-adapter training',
 'scope':'No tokenizer/model rerun in this checker. Copies and retained original execution evidence only.'}
EXT.mkdir(mode=0o700,exist_ok=False)
for root in [EXT,LOCAL]:
 with (root/'saved_validation.json').open('x') as f:f.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
with (EXT/'check_saved.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in report.items() if k!='supervisor'},indent=2))
