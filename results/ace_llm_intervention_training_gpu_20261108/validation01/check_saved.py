"""Read complete saved bytes, without staging, models, tokenization or tensors."""
from pathlib import Path
import datetime,hashlib,json,types
REPO=Path('/Users/pat/code/ACE');BASE=REPO/'results/ace_llm_intervention_training_gpu_20261108'
LAUNCH=BASE/'launch01';RAW=BASE/'terminal01/remote';LOCAL=BASE/'validation01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-gpu-validation-20261108-01')
TERMINAL_EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-gpu-terminal-20261108-01')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def file_pin(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
outer=read(LAUNCH/'packet.json');packet=read(LAUNCH/'training.json');packet_pin=file_pin(LAUNCH/'training.json')
job=read(LAUNCH/'submission.result.json')['job_id'];assert job.isdigit()
disposition=read(BASE/'terminal01/disposition.json')
assert disposition['job_id']==job and disposition['scheduler_state']=='COMPLETED' and disposition['scheduler_exit']=='0:0'
assert disposition['resource_contract_matched'] and disposition['accounting_errors']==[] and disposition['input_pin_mismatches']==[]
for root in [BASE/'terminal01',TERMINAL_EXT]:
 actual={str(p.relative_to(root/'remote')) for p in (root/'remote').rglob('*') if p.is_file()}
 assert actual==set(disposition['files'])
 for name,row in disposition['files'].items():
  path=root/'remote'/name
  assert path.stat().st_size==row['bytes'] and file_pin(path)==row['sha256'],name
for n,h in {**outer['new_input_files'],'packet.json':file_pin(LAUNCH/'packet.json')}.items():assert file_pin(RAW/n)==h
entry='qualify_llm_intervention_training_gpu.py';raw=(LAUNCH/entry).read_bytes();assert sha(raw)==packet['files'][entry]
g=types.ModuleType('saved_training_gpu');g.__file__=str(LAUNCH/entry)
exec(compile(raw,str(LAUNCH/entry),'exec'),vars(g))
root,authenticated,data,old,prepared,service,cpu,d,gpu,helper=g.authenticate(LAUNCH/'training.json',packet_pin)
assert authenticated==packet
q=RAW/'qualification';stage=read(q/'stage.json');stage_pin=file_pin(q/'stage.json');parent=read(q/'terminal.json')
allocation=read(q/'allocation.json');scheduler=read(q/'scheduler_raw.json')
assert scheduler['returncode']==0 and scheduler['job_id']==job
assert allocation==gpu.authenticate_allocation(scheduler['stdout'],job,packet['resources'],helper)
validation=g.validate_worker(q/'session/worker/artifacts',packet,packet_pin,data,old,prepared,service,cpu,d,
 stage,stage_pin,read(q/'independent-provenance-sources.json'))
assert not list(q.rglob('failure.json'))
assert parent['schema']==g.VERSION and parent['status']=='artificial_training_complete'
assert parent['packet_sha256']==packet_pin and parent['stage_sha256']==stage_pin and parent['validation']==validation
assert parent['environment_responses']==0 and parent['scientific_qualified'] is False
assert parent['external_scheduler_and_custody_acceptance_required'] is True and parent['elapsed_seconds']<580
cleanup=read(q/'cleanup.json');g.clean_success(cleanup);assert cleanup==parent['cleanup']
gpu.require_gpu_sample(read(q/'post-exit-gpu.json'),active=False)
samples=[read(p) for p in sorted(q.glob('gpu-sample-*.json'))]
positive=[s for s in samples if s.get('returncode')==0 and s.get('owned_rows')]
assert positive and len(samples)==parent['gpu_observation_count']
first=read(q/'first-verified-gpu-work.json');assert first['owned_process_observed'] is True
assert first['elapsed_seconds']==parent['first_verified_gpu_seconds'] and 0<first['elapsed_seconds']<240
summary=read(q/'session/worker/artifacts/fit/summary.json')
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'saved_checks_passed_independent_review_pending',
 'job_id':job,'source_commit':outer['source_commit'],'packet_sha256':file_pin(LAUNCH/'packet.json'),
 'training_sha256':packet_pin,'stage_sha256':stage_pin,'supervisor_terminal_sha256':file_pin(q/'terminal.json'),
 'validation':validation,'worker_summary':summary,'supervisor':parent,'positive_owned_gpu_samples':len(positive),
 'raw_files':disposition['raw_files'],'raw_bytes':disposition['raw_bytes'],
 'allocated_cpu_seconds':disposition['allocated_cpu_seconds'],'gpu_slice_seconds':disposition['gpu_slice_seconds'],
 'closed_cpu_seconds':disposition['known_closed_cpu_seconds'],'closed_gpu_seconds':disposition['gpu_admission_seconds'],
 'cpu_admission_after_release':disposition['cpu_admission_after_release'],
 'new_environment_responses':0,'scientific_worlds':0,'scientific_qualified':False,'training_qualified':False,
 'scope':'saved execution/source/custody checks only; no model or tensor deserialization, numerical replay or scientific evaluation',
 'next_gate':'independent result acceptance then complete paired artificial campaign before scientific generation'}
EXT.mkdir(mode=0o700,exist_ok=False)
for r in [EXT,LOCAL]:
 with (r/'saved_validation.json').open('x') as f:f.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
with (EXT/'check_saved.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in report.items() if k not in ('validation','worker_summary','supervisor')},indent=2))
