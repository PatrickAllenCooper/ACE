from pathlib import Path
import json,hashlib,datetime
R=Path('/Users/pat/code/ACE');T=R/'results/ace_llm_intervention_scientific_training_gpu_20261108/launch01';A=R/'results/ace_llm_intervention_scientific_campaign_cpu_preparation_20261108/token02-history01/authoring';O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(n,x):
 with (O/n).open('x') as f:f.write(json.dumps(x,sort_keys=True,indent=2)+'\n' if not isinstance(x,str) else x)
builder=A/'build_packet.py';assert sha(builder)=='3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'
original=load(A/'argument-map.json');packet=load(T/'packet.json');assert sha(T/'packet.json')=='33fa6064bd224a64736bf466623bd3ae9399c81022dac6023c689515764a9ec0'
assert sha(T/'inputs/training.json')=='b12fd2d7c0879618b48232cdd46bada5a3c1c16a9773680d3769a24a99e7aaf0'
f=load(T/'freeze.json');cpu=load(T/'cpu-acceptance.json');result=load(T/'submission.result.json');assert result['job_id']=='35420219'
expected=dict(job_id=result['job_id'],source_commit=f['source_commit'],prepared_sha256=cpu['prepared_sha256'],training_plan_sha256=cpu['plan_sha256'],exposure_sha256=cpu['exposure_sha256'],scientific_freeze_sha256=sha(T/'freeze.json'),collection_acceptance_sha256=sha(T/'collection-acceptance.json'),cpu_preparation_acceptance_sha256=sha(T/'cpu-acceptance.json'),model_files_sha256=digest(f['model_files']),runtime_contract_sha256=digest(f['runtime']))
write('expected-training.json',expected)
real=R/'results/ace_llm_intervention_scientific_realization_20261108';future=R/'results/ace_llm_intervention_scientific_training_gpu_20261108'
overrides={'training-packet':T/'packet.json','training-result':T/'submission.result.json','expected-training':O/'expected-training.json','realization-acceptance':real/'acceptance02/acceptance.json','realization-terminal':real/'terminal02/remote/launch/realization/session/worker/task/terminal.json','realization-inventory':real/'terminal02/inventory_transport.json','cpu-preparation-acceptance':T/'cpu-acceptance.json','previous-baseline':T/'baseline.json','collection-baseline':T/'previous-baseline.json'}
unknown={'gpu-training-acceptance':future/'acceptance01/acceptance.json','baseline':future/'acceptance01/accounting.json','training-disposition':future/'terminal01/disposition.json','training-inventory':future/'terminal01/inventory_transport.json','training-review':future/'acceptance01/review.md','site-readiness':None}
args={}
for n,old in original['metadata'].items():
 if n in unknown:
  args[n]={'path':None,'sha256':None,'status':'pending_independent_actual_training_audit' if n!='site-readiness' else 'pending_parent_cpu_site_readiness','suggested_path':str(unknown[n]) if unknown[n] else None};continue
 p=overrides.get(n)
 if p is None:
  matches=[T/(n+x) for x in ('.json','.md') if (T/(n+x)).is_file()];assert len(matches)==1,(n,matches);p=matches[0]
 h=sha(p)
 if old['sha256'] is not None:assert h==old['sha256'],n
 if p.parent==T and p.name in packet['new_input_files']:assert h==packet['new_input_files'][p.name],n
 args[n]={'path':str(p),'sha256':h,'status':'resolved_metadata_bytes'}
requirements={'schema':'ace-scientific-training-gpu-acceptance-v1','status':'scientific_training_gpu_accepted','scope':'policy_train','required_findings':0,'scientific_utility_qualified':False,**expected,**{k:True for k in ['training_gpu_qualified','checkpoint_reload_inference_qualified','complete_custody_verified','runtime_qualified','owned_process_cleanup_verified','scheduler_verified']},'completed_steps':279,'sft_steps':160,'dpo_steps':119,'environment_responses':0,**{k:None for k in ['engine_terminal_sha256','sft_checkpoint_sha256','sft_adapter_sha256','final_checkpoint_sha256','final_adapter_sha256']}}
write('training-acceptance-requirements.json',{'not_an_acceptance':True,'required_fields_and_expected_values':requirements,'null_means_actual_bytes_unknown':True,'extra_acceptance_fields':'consumer does not require exact field-set equality; no new schema imposed'})
artifacts={'prepared.json':'launch/inputs/prepared.json','numerical-plan.json':'cpu/preparation/session/worker/artifacts/numerical-plan.json','engine-terminal.json':'launch/training/session/worker/artifacts/fit/training/terminal.json',**{n:'launch/training/session/worker/artifacts/fit/training/'+n for n in ['sft.checkpoint.json','final.checkpoint.json','sft.adapter.pt','final.adapter.pt']}}
mapvalue={'schema':'prospective-cpu-campaign-3893-argument-map-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'builder':str(builder),'builder_sha256':sha(builder),'metadata':args,'cli_rule':'Every metadata name requires --NAME PATH --NAME-sha256 SHA; pending entries must be independently supplied before execution.','additional_required':{'training-remote-custody':{'path':None,'suggested_path':str(future/'terminal01/remote'),'status':'pending_complete_independently_accepted_training_custody'},'realization-task':{'path':str(real/'terminal02/remote/launch/realization/session/worker/task'),'status':'existing_accepted_directory; no descriptor payload opened'}},'training_job_id':'35420219','training_packet_commit':'9c54a77b3ee0ba3988a2dcdd0fe733ee0562ffe7','training_outer_sha256':sha(T/'packet.json'),'training_inner_path':str(T/'inputs/training.json'),'training_inner_sha256':sha(T/'inputs/training.json'),'artifact_paths_relative_to_training_remote_custody':artifacts,'history':{'collection_baseline':'42/86','token_failure_baseline':'43/88','previous_baseline':'44/90; closed14189CPU/1601GPU; preparation3600','baseline':'45/92 ONLY after accepted training; costs unknown','prospective_cpu_terminal':'46/94; no result assumed'},'cpu_stage':next(s for s in f['stages'] if s['stage']=='cpu_campaign'),'packet_built':False,'actual_training_accepted':False,'pending_arguments':list(unknown),'source_members':{n:{'path':str(A/n),'sha256':sha(A/n)} for n in ['build_packet.py','launch_once.py','monitor_once.py']},'launch_future_arguments':{'packet-directory':str(R/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/launch01'),'packet-commit':None,'packet-sha256':None,'packet-review':None,'packet-review-sha256':None},'monitor_future_arguments':{'packet-directory':str(R/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/launch01'),'packet-sha256':None,'submission-result-sha256':None,'evidence-directory':None}}
write('argument-map.json',mapvalue)
write('handoff.md',f'''# Prospective CPUcampaign3893 argument map

Builder: `{builder}` SHA `{sha(builder)}`. No build, launch, remote call, scientific execution, checkpoint/model/private-outcome decoding or tests occurred. Only existing interface source and public saved metadata were read. Existing authoring and acceptance capsules are unchanged.

`argument-map.json` resolves {len(args)-len(unknown)} of {len(args)} metadata inputs with exact paths and hashes. Pending path/pins remain null; suggested future paths are conventions for parent/auditor agreement, not evidence of existence or acceptance. `expected-training.json` is the exact independent10-field producer contract from accepted TOKEN02, adopted freeze and the actual submitted job/result; it asserts no completed work. `training-acceptance-requirements.json` states existing consumer requirements and is not an acceptance or new schema.

## Actual training producer required

Parent/Godel independent actual training audit must supply acceptance (schema ace-scientific-training-gpu-acceptance-v1), review, disposition, first inventory_transport and actual45/92 accounting. Supervisor55be5ba7 runs workerab2c999a via frozena934; successful outer status alone is insufficient and neither emits independent acceptance. The expected engine terminal is `remote/launch/training/session/worker/artifacts/fit/training/terminal.json`, not outer `training/terminal.json` or worker `artifacts/terminal.json`. SFT/final checkpoint JSON and opaque adapter files are its siblings. Exact five output pins remain unknown. SFT checkpoint must complete160 steps,index159,stage sft; final279 steps,index278,stage dpo, using accepted119DPO steps. Parent/Godel must establish real GPU work, reload inference, full runtime/cleanup/scheduler/custody qualification with zero environment responses and no utility claim. No output was read for this map.

All10 expected fields must match acceptance. Model/runtime pins use canonical sorted compact JSON without newline over freeze.model_files/runtime. prepared_sha256 is raw bytes; training_plan_sha256/exposure_sha256 are semantic pins from TOKEN02 acceptance. Output hashes are raw file hashes. No acceptance pin is derived from a worker's claimed status.

## Remaining admission joins

Training outer33fa6064 and innerb12fd2d7 are pinned; inner path is `inputs/training.json`. Actual submission result binds job35420219; this does not establish terminal success. Independent baseline must retain all44 prior rows/90journals and append exactly this job and its2journals. For independently accepted elapsed t, closed CPU=14189+4t,GPU=1601+t; t is UNKNOWN here, must be observed0..2700. Add single3600 preparation allowance; CPU candidate adds12000,zeroGPU. Builder also checks all remaining frozen stage reservations against57600/7200; no planned cost substitutes for allocation. Preserve collection failure41/84, collection success42/86, TOKEN failure43/88 and TOKEN success44/90 plus all original claims/operational amendments unchanged.

Actual training inventory must pin every original launch input, packet and bothjournals,7 required artifacts, whole TOKEN02/failedTOKEN01/collection02/failedcollection01 roots and directory rosters. `--training-remote-custody` must be its complete accepted local remote directory. `--realization-task` remains original accepted realization02 task; builder later captures24 prescribed development descriptors after admission. No descriptor was opened here.

CPU stage remains frozen2CPU7680MiB6000sec,acpu/cpu-normal,ucb736_asc2,zeroGPU,startup180/work5940/hard5980,numerical50/55. Root and destinations are recorded directly from freeze. `--site-readiness` is independently supplied nonempty text (captured as site-readiness.md); do not substitute GPU readiness. Any fresh remote check remains parent-owned.

Launcher needs future CPU packetcommit/pin and independent concretepacket review/path/pin, not GPU's packetcommit. Monitor needs actual future CPU submission-result pin and a fresh exclusive evidence directory. This map does not authorize either operation.

Initial map-generation attempt stopped read-only at a nonexistent assumed inputs/packet.json before writing any map. Actual training inner filename inputs/training.json was then resolved; no production or coordinator defect. Failure note retained in attempt01-failure.json; no packet operation occurred.
''')
write('receipt.json',{'status':'prospective_argument_map_only','source_builder_sha256':sha(builder),'files':{p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file() and not p.name.endswith('.log')},'resolved_metadata':len(args)-len(unknown),'pending_metadata':len(unknown),'actual_training_acceptance_created':False,'tests_run':False,'build_run':False,'remote_actions':False})
print(json.dumps({'directory':str(O),'map_sha256':sha(O/'argument-map.json'),'expected_training_sha256':sha(O/'expected-training.json'),'receipt_sha256':sha(O/'receipt.json'),'pending':list(unknown)},indent=2))
