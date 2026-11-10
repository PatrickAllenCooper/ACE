"""Saved finalizer execution metadata only; no scientific source callable imports."""
from pathlib import Path
import json,hashlib,ast,time,subprocess
O=Path(__file__).parent;R=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108');L=R/'launch02';T=R/'terminal01';F=T/'remote/launch/finalization';checks=[];pins={};facts={}
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
 checks.append(m)
def read(p):
 b=Path(p).read_bytes();pins[str(p)]={'bytes':len(b),'sha256':sha(b)};return b
def j(p):return json.loads(read(p))
def within(p,root):return p==root or p.startswith(root+'/')
def run():
 inner=j(L/'inputs/packet.json');old=j(L/'inputs/cpu-packet.json');p=sha(read(L/'inputs/packet.json'));start=j(F/'start.json');ws=j(F/'worker/start.json');proc=j(F/'worker.started.json');auth=j(F/'worker/authentication-complete.json');work=j(F/'worker/work-start.json');hand=j(F/'handoff.json');stage=j(F/'stage.json');cl=j(F/'worker.cleanup.json');use=j(F/'worker.resources.json');pa=j(F/'attempt.json');wa=j(F/'worker/attempt.json');term=j(F/'worker/terminal.json');origin=start['started_monotonic'];clock=j(L/'finalizer-clock-adoption.json')['deadlines'];entry='finalize_llm_intervention_scientific_campaign_cpu.py'
 need(start==ws or all(start[k]==ws[k] for k in ('started_monotonic','packet_sha256','deadlines','report_startup_seconds','resources','schema')),'parent worker common origin and authority')
 need(start['deadlines']=={k:origin+v for k,v in clock.items() if k!='report_startup'},'absolute adopted clock')
 need(proc['pid']==proc['pgid']==auth['pid']==work['pid']==cl['pid']==cl['pgid'],'single owned worker identity')
 need(hand['started_monotonic']==origin and hand['parent_pid']!=proc['pid'] and hand['job_id']=='35533604' and hand['stage_sha256']==sha(read(F/'stage.json')),'owned handoff and stage')
 need(origin<=auth['monotonic']<origin+clock['preparation'],'full preparation before deadline')
 need(auth['report_startup_deadline']==work['report_startup_deadline']==min(auth['monotonic']+clock['report_startup'],origin+clock['report_startup_cap']),'report startup bound anchored to original clock')
 need(auth['monotonic']<=work['monotonic']<work['report_startup_deadline'] and work['authentication_sha256']==sha(read(F/'worker/authentication-complete.json')) and work['private_finalization_calls']==1,'authenticated once reporting start')
 need(all(x['packet_sha256']==p for x in (start,ws,auth,work,hand,pa,wa,term)),'all execution packet pins exact')
 need(proc['command'][:4]==[stage['staged_prefix']+'/bin/python','-I','-B','-c'] and proc['command'][6]==inner['files'][entry] and proc['command'][-1]==sha(read(F/'handoff.json')),'isolated authenticated bootstrap and handoff arguments')
 for x in (pa,wa):need(x['status']=='completed' and x['error'] is None and x['retry_allowed'] is False and x['scientific_acceptance'] is False and x['independent_acceptance_required'] is True,'actual successful attempt without self qualification')
 need(pa['elapsed_seconds']<clock['hard'] and wa['elapsed_seconds']<clock['work'] and use['elapsed_seconds']<clock['cleanup'],'actual completion and cleanup within absolute bounds')
 need(cl['errors']==[] and cl['returncode']==0 and cl['kill_sent'] is False and cl['group_still_exists'] is False and pa['result']['cleanup']==cl and use['first_failure'] is None and use['cleanup_failure'] is None,'natural complete owned leader group closure')
 need(pa['result']['job_id']=='35533604' and pa['result']['stage_sha256']==sha(read(F/'stage.json')) and pa['result']['report']==term['report'],'parent independent report validation exact worker agreement')
 report=term['report'];camp=j(O/'campaign-file-pins.json');need(report['campaign_files']==camp,'parent and worker report pins match full after custody')
 need(report['cell_count']==192 and report['descriptive_policy_comparisons']==28 and report['scientific_acceptance'] is False and report['new_environment_responses']==report['new_training_updates']==0,'fixed descriptive report envelope no new responses or updates')
 need(report['summary_sha256']==camp['private/summary.json'] and report['finalization_sha256']==camp['finalization.json']==camp['finalization.candidate.json'],'exact summary and finalization publication pins')
 need(term['status']=='descriptive_finalization_complete' and term['independent_acceptance_required'] is True,'worker completion marker independent acceptance boundary')
 # Inspect the unchanged runtime snapshot contract using saved observations.
 prefix=stage['staged_prefix'];need(prefix=='/scratch/local/35533604/ace-campaign-finalizer-35533604/runtime','allocated private staged runtime path')
 need(stage['archive_sha256']==old['runtime']['archive_sha256'] and stage['manifest_sha256']==old['runtime']['manifest_sha256'] and stage['objects']==49966 and stage['file_bytes']==8397575073 and stage['model_loaded'] is False,'exact qualified archive staging')
 priorstage=j(R.parent/'ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/stage.json');need(stage['cache_source_sha256']==priorstage['cache_source_sha256'],'unchanged accepted six runtime source cache pins')
 snaps=[];module_counts=[];native_counts=[]
 for name in ('runtime-before.json','runtime-ready.json','runtime-after.json'):
  rec=j(F/'worker'/name);s=rec['snapshot'];snaps.append(s)
  need(s['prefix']==s['base_prefix']==prefix and s['executable']==prefix+'/bin/python' and s['python_version']=='3.10.19','actual copied interpreter '+name)
  need(all(within(q,prefix) for q in s['sys_path']) and s['sys_path']==snaps[0]['sys_path'],'no search path drift '+name)
  need(s['generated_provenance']=={'present':False,'verified':False} and s['mixed_modules']=={} and s['mixed_module_paths']==[] and s['mixed_native_paths']==[],'no mixed or generated runtime '+name)
  need(all(within(q,prefix) for q in s['module_paths'].values()),'independent loaded module path check '+name)
  mapped=[line.split(None,5)[5].removesuffix(' (deleted)') for line in s['process_maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')]
  need(all(within(q,prefix) or any(within(q,k) for k in ('/lib','/lib64','/usr/lib','/usr/lib64')) for q in mapped),'independent native mapping check '+name)
  need(not any(within(q,old['runtime_source']) for q in mapped),'original runtime never mapped '+name)
  need(not any(n.split('.')[0] in {'torch','transformers','tokenizers','safetensors','scipy'} for n in s['module_paths']),'no model tokenizer or non NumPy package loaded '+name)
  need(all(s['packages'][k]['loaded'] is False for k in ('torch','transformers','scipy')),'forbidden package observations '+name)
  need(s['packages']['numpy']['loaded']==(name!='runtime-before.json'),'NumPy startup order '+name)
  if name!='runtime-before.json':
   need(s['packages']['numpy']['version']==old['runtime']['imported_versions']['numpy']=='2.2.5' and any(n.startswith('numpy.') and q.endswith('.so') for n,q in s['module_paths'].items()),'actual pinned NumPy extension '+name)
  env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','MKL_THREADING_LAYER':'GNU','LD_LIBRARY_PATH':prefix+'/lib'}
  need(rec['environment']==env,'CPU environment exact '+name)
  for n,v in rec['captured_project_modules'].items():
   file=Path(v['path']).name;need(file in inner['source_names'] and v['sha256']==inner['files'][file] and v['path']==str(Path(proc['command'][5]).parent/file) and v['loader'] in ('authenticated_captured_bytes','outer_authenticated_entrypoint'),'actual loaded captured source '+name+':'+n)
  module_counts.append(dict(record=name,project_modules=len(rec['captured_project_modules']),runtime_modules=len(s['module_paths'])));native_counts.append(len(set(mapped)))
 # Saved phase evidence, including both complete historical passes and final postchecks.
 timings=[]
 for role,directory in [('parent',F),('worker',F/'worker')]:
  ends=sorted(directory.glob('timing-*.end.json'));need(len(ends)==(11 if role=='parent' else 9),'complete timing roster '+role)
  for i,path in enumerate(ends):
   end=j(path);beg=j(directory/path.name.replace('.end.','.start.'))
   need(end['completed'] is True and end['failure'] is None and end['sequence']==i and end['role']==role and end['origin_monotonic']==origin and all(end[k]==v for k,v in beg.items()),'complete identical timing start '+role+str(i))
   need(abs(end['elapsed_seconds']-(end['finished_monotonic']-end['started_monotonic']))<1e-6 and end['elapsed_seconds']>=0,'saved monotonic timing arithmetic '+role+str(i))
   timings.append(dict(role=role,phase=end['phase'],sequence=i,start=end['started_elapsed_seconds'],finish=end['finished_elapsed_seconds'],elapsed=end['elapsed_seconds'],cpu=end['process_cpu_seconds'],details=end['details']))
 need([x['details'].get('files') for x in timings if x['phase']=='inventory_cpu_output']==[20564,20564,20564],'all 3 full CPU historical inventories')
 need([x['details'].get('files') for x in timings if x['phase']=='inventory_gpu_output']==[13262,13262,13262],'all 3 full GPU inventories')
 need([x['details'].get('files') for x in timings if x['phase']=='inventory_campaign']==[27869,35726,27869,35726],'all before after campaign inventories')
 stages=[j(F/f'phase-{i:04d}.json') for i in range(102)];need(stages[0]['phase']=='stage_start' and stages[-1]['phase']=='stage_complete' and all(stages[i]['elapsed_seconds']<=stages[i+1]['elapsed_seconds'] for i in range(101)),'all 102 ordered runtime staging observations')
 need(read(F/'worker.stderr.log')==b'' and read(F/'worker.stdout.log')==b'','worker no hidden stderr failures')
 # Verify the saved backup execution and source without any remote action.
 rc=j(T/'remote-custody.json');need(read(T/'backup-operations/receipt.json')==read(T/'remote-custody.json'),'original backend receipt exact outer copy')
 tx=j(T/'backup-operations/transport.json');need(tx['returncode']==0 and tx['remote_outcome_ambiguous'] is False and tx['retry_allowed'] is False and tx['wall_seconds']<1800,'backup transport completed once bounded')
 need(tx['command']==j(T/'backup-operations/command.json') and 'python3 -I -B -c ' in tx['command'][-1] and tx['command'][-1].endswith(rc['helper_sha256']),'isolated actual backend bootstrap pin')
 need(sha(read(T/'backup-operations/helper.py'))==rc['helper_sha256'],'actual captured backend bytes')
 request=j(T/'backup-operations/request.json');intent=j(T/'backup-operations/intent.json');need(sha(read(T/'backup-operations/request.json'))==intent['request_sha256'] and request['observation']==json.loads(j(T/'inventory_transport.json')['stdout']) and request['roots']==rc['roots'],'actual backup request exact observation and roots')
 # Bind exact frozen source to actual callable entry via captured bootstrap, already source-reviewed.
 text=read(L/'inputs'/entry).decode();tree=ast.parse(text);funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 call=[n for n in ast.walk(funcs['worker']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='campaign' and n.func.attr=='finalize_campaign'];need(len(call)==1,'one saved reporter call in actual authenticated worker')
 facts.update(parent_elapsed_seconds=pa['elapsed_seconds'],worker_elapsed_from_shared_origin=wa['elapsed_seconds'],authentication_elapsed_seconds=auth['monotonic']-origin,report_startup_elapsed_seconds=work['monotonic']-auth['monotonic'],stage_elapsed_seconds=stages[-1]['elapsed_seconds']-stages[0]['elapsed_seconds'],phase_timings=timings,cleanup=cl,worker_resources=use,runtime_records=module_counts,native_mapping_counts=native_counts,cell_count=report['cell_count'],policy_comparisons=report['descriptive_policy_comparisons'],finalization_sha256=report['finalization_sha256'],summary_sha256=report['summary_sha256'],new_environment_responses=0,new_training_updates=0,report_content_decoded=False,full_semantic_report_audit_pending=True,terminal_sha256=sha(read(F/'worker/terminal.json')),parent_attempt_sha256=sha(read(F/'attempt.json')),stage_sha256=sha(read(F/'stage.json')),backup_transport_seconds=tx['wall_seconds'])
if __name__=='__main__':
 started=time.monotonic()
 try:run();(O/'runtime-facts.json').write_bytes(raw(facts));print(json.dumps({'status':'runtime_execution_custody_checks_pass','checks':len(checks)}))
 except BaseException as e:(O/'runtime-inspection-failure.json').write_bytes(raw({'type':type(e).__name__,'message':str(e),'checks_completed':len(checks)}));raise
 finally:(O/'runtime-read-pins.json').write_bytes(raw(pins));(O/'runtime-checks.json').write_bytes(raw(checks));(O/'runtime-inspection-cost.json').write_bytes(raw({'elapsed_seconds':time.monotonic()-started,'process_cpu_seconds':time.process_time()}))
