from pathlib import Path
import json,hashlib,sys,types,time,resource,traceback,math
OUT=Path(__file__).resolve().parents[1];T=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01');L=T/'remote/launch';I=L/'inputs';O=L/'training';W=O/'session/worker/artifacts';start=time.monotonic()
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(b,m):
 if not b:raise ValueError(m)
def save(n,o):
 with (OUT/'attempts'/n).open('x') as f:json.dump(o,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def module(n):
 b=(I/n).read_bytes();need(hashlib.sha256(b).hexdigest()==packet['files'][n],'captured source '+n);m=types.ModuleType(n[:-3]);m.__file__=str(I/n);exec(compile(b,str(I/n),'exec'),vars(m));return m
try:
 packet=read(I/'training.json');term=read(O/'terminal.json');previous=read(OUT/'attempts/validate02-failure.json')
 need(previous['type']=='TypeError' and 'allocation=s.authenticate_allocation' in previous['traceback'],'exact retained post-validation driver failure')
 save('saved-validation-completed-before-driver-error.json',dict(status='verified_saved_worker_runtime_and_source',validation=term['validation'],evidence='validate02.py independently completed validate_worker and exact parent validation equality before the separately retained scheduler raw-dict/string driver error.',script_sha256=sha(OUT/'scripts/validate02.py'),failure_receipt_sha256=sha(OUT/'attempts/validate02-failure.json'),outer_terminal_sha256=sha(O/'terminal.json'),tensor_files_deserialized=False,heavy_imports=False))
 s=module('supervise_llm_intervention_scientific_training_gpu.py');h=module(s.SUPERVISOR);g=module(s.GPU)
 scheduler=read(O/'scheduler_raw.json');need(scheduler['returncode']==0 and scheduler['job_id']=='35420219','original scheduler transport')
 alloc=s.authenticate_allocation(scheduler['stdout'],'35420219',packet['resources'],h);need(alloc==read(O/'allocation.json'),'raw allocation reconstruction')
 hand=read(O/'handoff.json');command=read(O/'child-command.json');child=read(O/'child-start.json');first=read(O/'first-verified-gpu-work.json');fw=read(W/'first-gpu-work.json');origin=hand['started'];pid=child['pid']
 need(hand['packet_sha256']==term['packet_sha256']==sha(I/'training.json') and hand['runtime_stage_sha256']==term['stage_sha256']==sha(O/'stage.json') and hand['admission_sha256']==term['admission_sha256']==sha(I/'admission.json') and hand['job_id']=='35420219','one handoff identity')
 need(hand['deadlines']==packet['deadlines']==s.DEADLINES and command['absolute_worker_deadline']==origin+2620 and command['absolute_supervisor_deadline']==origin+2670 and command['first_gpu_deadline_from_start_seconds']==240,'original absolute timers')
 cmd=command['command'];need(cmd[:3]==[str(Path(read(O/'stage.json')['staged_prefix'])/'bin/python'),'-I','-B'] and cmd[3]=='-c','isolated bytecode-free child interpreter')
 need(cmd[4]==module(s.DIAGNOSTIC).BOOTSTRAP and cmd[5].endswith('/inputs/'+s.ENTRY) and cmd[6]==packet['files'][s.ENTRY],'captured entry bootstrap')
 for flag,pin in (('--packet-sha256',sha(I/'training.json')),('--admission-sha256',sha(I/'admission.json')),('--runtime-stage-sha256',sha(O/'stage.json')),('--handoff-sha256',sha(O/'handoff.json'))):need(cmd[cmd.index(flag)+1]==pin,'child command pin '+flag)
 need(child['elapsed_seconds']<fw['monotonic_seconds']-origin<=first['elapsed_seconds']<240 and first['elapsed_seconds']==term['first_verified_gpu_seconds'] and first['owned_process_observed'] is True,'first actual work clock')
 need(s.positive_work(W/'first-gpu-work.json') and (W/'first-gpu-work.json').read_bytes()==(W/'fit/first-gpu-work.json').read_bytes(),'positive CUDA first work original')
 samples=[read(p) for p in sorted(O.glob('gpu-sample-*.json'))];need(len(samples)==term['gpu_observation_count']==698,'exact sample count')
 times=[v['elapsed_seconds'] for v in samples];need(times==sorted(times) and times[0]>=child['elapsed_seconds'] and times[-1]<term['elapsed_seconds']<min(904,2670),'sample/terminal deadline chronology')
 positive=[v for v in samples if s.positive_owned_memory(v,pid)];need(positive and any(v['elapsed_seconds']<=first['elapsed_seconds'] and first['elapsed_seconds']-v['elapsed_seconds']<1 for v in positive),'first paired memory sample')
 progress=sorted(O.glob('progress-*.json'));previous_progress=set();lastpg=origin;lastgpu=origin;maxgpuage=0;maxprogage=0
 for path in progress:
  index=int(path.stem.split('-')[1])-1;pr=read(path);sample=samples[index];now=origin+sample['elapsed_seconds'];upper=origin+(samples[index+1]['elapsed_seconds'] if index+1<len(samples) else term['elapsed_seconds'])
  need(pr['progress_bound_seconds']==180 and pr['telemetry_bound_seconds']==90,'progress contract')
  need(lastpg<=pr['last_progress_monotonic']<=upper and lastgpu<=pr['last_gpu_monotonic']<=upper,'progress clocks');lastpg=pr['last_progress_monotonic'];lastgpu=pr['last_gpu_monotonic']
  need(upper-lastpg<=180 and upper-lastgpu<=90,'conservative progress/telemetry upper bounds')
  if s.positive_owned_memory(sample,pid):need(now<=lastgpu<=upper,'positive owned observation clock')
  rows=set(pr['progress']);need(previous_progress<=rows and len(rows)==len(pr['progress']),'monotone unique progress')
  for n in rows-previous_progress:need(not Path(n).is_absolute() and '..' not in Path(n).parts and (W/n).is_file(),'progress points to actual retained record')
  previous_progress=rows;maxgpuage=max(maxgpuage,upper-lastgpu);maxprogage=max(maxprogage,upper-lastpg)
 need(len(progress)==685 and len(previous_progress)==1701,'full progress roster')
 clean=read(O/'cleanup.json');s.clean_success(clean);need(clean==term['cleanup'] and clean['pgid']==pid,'owned PID cleanup');g.require_gpu_sample(read(O/'post-exit-gpu.json'),active=False)
 need(not list(O.rglob('*failure*.json')),'no current failure markers')
 phases=[read(p) for p in sorted(W.glob('phase-*.json'))];major=[v for v in phases if v['phase'] in ('service_cpu_import_start','service_cpu_import_complete','training_start','training_complete','checkpoint_inference_complete')]
 need([v['phase'] for v in major]==['service_cpu_import_start','service_cpu_import_complete','training_start','training_complete','checkpoint_inference_complete','checkpoint_inference_complete'],'major phase chronology')
 need(all(v.get('new_training_updates',0)==0 for v in major[-2:]) and [v['checkpoint'] for v in major[-2:]]==['sft','final'],'checkpoint stage order')
 stageph=[read(p) for p in sorted(O.glob('phase-*.json'))];need(stageph[0]['phase']=='stage_start' and stageph[-1]['phase']=='stage_complete' and stageph[-1]['elapsed_seconds']<child['elapsed_seconds'],'complete staging before child')
 env=read(O/'child-environment.json');need(env['loaded_parent_libraries_changed'] is False and env['after']['python']==env['after']['c'] and env['after']['python']['CUDA_VISIBLE_DEVICES']=='0' and env['after']['python']['MKL_THREADING_LAYER']=='GNU','saved CPU/runtime environment')
 ob=read(T/'inventory_transport.json');obs=json.loads(ob['stdout']);need(obs['upstream_errors']=={} and all(obs[k] is True for k in ('root_present','cpu_present','failed_token_present','common_present','failed_present','namespace_present','original_namespace_present')),'original roots+claims present')
 row=obs['parent_accounting'];need(row[0]=='35420219' and row[2:6]==['COMPLETED','0:0','4','904'] and row[6]=='ucb736_asc2','actual Slurm row')
 for k in ('job','steps','details','related_accounting'):need(obs['results'][k]['returncode']==0,'actual accounting query '+k)
 fields=h.scheduler_fields(obs['results']['details']['stdout']);need(fields['JobId']=='35420219' and fields['JobState']=='COMPLETED' and fields['ExitCode']=='0:0' and fields['Requeue']=='0' and fields['Restarts']=='0','actual terminal scheduler identity')
 for k in ('UserId','Account','Partition','QOS','TimeLimit','ReqTRES','AllocTRES','Command','NumCPUs','NumNodes','NumTasks'):need(fields[k]==alloc['fields'][k],'terminal allocation unchanged '+k)
 need(term['environment_responses']==0 and term['scientific_qualified'] is False,'terminal scope')
 memory=[float(v['owned_rows'][0].split(',')[2]) for v in positive]
 summary=dict(status='verified_saved_lifecycle_gpu_and_scheduler',job_id='35420219',allocated_cpu_seconds=3616,gpu_slice_seconds=904,slurm_elapsed_seconds=904,outer_elapsed_seconds=term['elapsed_seconds'],stage_elapsed_seconds=stageph[-1]['elapsed_seconds']-stageph[0]['elapsed_seconds'],child_start_seconds=child['elapsed_seconds'],first_verified_gpu_seconds=first['elapsed_seconds'],cuda_first_forward_ms=fw['cuda_interval_ms'],positive_owned_memory_samples=len(positive),gpu_samples=len(samples),max_owned_process_memory_mib=max(memory),max_sample_interval_seconds=max(b-a for a,b in zip(times,times[1:])),progress_snapshots=len(progress),max_conservative_gpu_freshness_age_seconds=maxgpuage,max_conservative_progress_age_seconds=maxprogage,major_phases=major,cleanup=clean,post_exit_owned_gpu_processes=0,runtime_directory_deletion_verified=False,gpu_utilization_percentage=None,wall_seconds=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 save('lifecycle03.json',summary);print('lifecycle/scheduler/continuing GPU work verified')
except BaseException as e:
 save('lifecycle03-failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
