from pathlib import Path
import ast,json,hashlib,subprocess,time,traceback,stat,re
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/pat/code/ACE')
T=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/terminal02'
R=T/'remote/launch';P=R/'preparation';JOB='35419195'

def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def data(p):return json.loads(p.read_bytes())
def literal(source,key):
 defs={}
 for n in ast.parse(source).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):defs[n.targets[0].id]=n.value
 def ev(n):
  if isinstance(n,ast.Name):return ev(defs[n.id])
  if isinstance(n,ast.Constant):return n.value
  if isinstance(n,ast.Dict):return {ev(k):ev(v) for k,v in zip(n.keys,n.values)}
  if isinstance(n,(ast.List,ast.Tuple)):return [ev(x) for x in n.elts]
  raise ValueError('nonliteral')
 return ev(defs[key])

def main():
 previous=data(ROOT/'custody-attempt01/receipt.json');need(previous['status']=='custody_source_history_checks_passed','full custody prerequisite')
 p=data(R/'inputs/packet.json');outer=data(R/'packet.json');source=p['source_commit']
 # Actual code bytes were authenticated to packet commit in custody pass; now
 # join their declared scientific origins to the independent frozen Git source.
 for origin,h in outer['source_pins'].items():
  q=subprocess.run(['git','--no-replace-objects','show',source+':'+origin],cwd=REPO,capture_output=True)
  need(q.returncode==0 and sha(q.stdout)==h,'original scientific source '+origin)
  need(p['files'][Path(origin).name]==h,'flat captured scientific input '+origin)
 for n,h in p['files'].items():need(sha((R/'inputs'/n).read_bytes())==h,'post-custody input '+n)
 need(sha((P/'packet.json').read_bytes())==sha((R/'inputs/packet.json').read_bytes()),'parent packet copy')
 f=data(R/'freeze.json');closure=f['stage_closure']['stages']['token']
 need(p['source_names']==closure['source_names'] and all(p['files'][n]==h for n,h in closure['input_source_files'].items()),'full frozen token closure')
 need(p['resources']==dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,wall_seconds=600,gpus=0,requeue=False),'CPU resource contract')
 need(p['deadlines']==dict(startup=300,work=550,cleanup=570,hard=590),'adopted absolute offsets')
 first=json.loads(data(T/'inventory_transport.json')['stdout']);last=json.loads(data(T/'inventory_after_transport.json')['stdout'])
 for obs in (first,last):
  for n in ('details','steps','job','queue'):need(obs['results'][n]['returncode']==0,'saved scheduler '+n)
  fields=dict(re.findall(r'([^\s=]+)=(\S+)',obs['results']['job']['stdout']))
  expected=dict(JobId=JOB,JobName='ace_llm_scientific_token02',Account='ucb736_asc2',QOS='cpu-normal',Partition='acpu',
     NumCPUs='3',NumNodes='1',NumTasks='1',Requeue='0',Restarts='0',TimeLimit='00:10:00',RunTime='00:04:32',JobState='COMPLETED',ExitCode='0:0')
  need(all(fields.get(k)==v for k,v in expected.items()) and fields['UserId']=='paco0228(348288)','actual scheduler fields')
  for k in ('AllocTRES','ReqTRES'):
   tres=dict(x.split('=',1) for x in fields[k].split(','));need(tres==dict(cpu='3',mem='11520M',node='1',billing='3'),'actual requested/allocated TRES')
  details=[x.split('|') for x in obs['results']['details']['stdout'].splitlines() if x]
  need(len(details)==1 and details[0][:4]==[JOB,'acpu','cpu-normal','10'] and details[0][5]=='paco0228','details identity/wall')
  need(not any(x.split('|')[0]==JOB or x.split('|')[1].lower().startswith('ace') for x in obs['results']['queue']['stdout'].splitlines()),'active related job after terminal')
 allocation=data(P/'allocation.json');need(allocation['returncode']==0,'running allocation observed')
 running=dict(re.findall(r'([^\s=]+)=(\S+)',allocation['stdout']))
 need(running['JobId']==JOB and running['JobState']=='RUNNING' and running['NumCPUs']=='3' and running['Requeue']==running['Restarts']=='0','saved live allocation')
 stage=data(P/'stage.json');prefix=stage['staged_prefix'];provenance=data(P/'independent-provenance-sources.json')
 diagnostic=(R/'inputs/qualify_llm_intervention_runtime_metadata.py').read_bytes()
 expected_provenance=literal(diagnostic,'PROVENANCE_SOURCES')
 need(set(provenance['sources'])==set(expected_provenance),'exact parent provenance source roster')
 for name,(rel,h) in expected_provenance.items():
  row=provenance['sources'][name];s=row['stat']
  need(row['sha256']==h and row['path']==prefix+'/lib/python3.10/site-packages/'+rel,'independent runtime provenance pin')
  need(stat.S_ISREG(s['mode']) and s['uid']==348288 and s['nlink']==1 and not s['mode']&0o022,'owned provenance source metadata')
 need(sha(provenance['expected_source'].encode())==provenance['expected_source_sha256'],'derived generated source byte hash')
 # Authority: fixed provenance_contract authenticates six source bytes and uses
 # its literal-AST-only generator/template derivation before worker spawn.
 # No template or runtime code is executed by this audit.
 source_text=(R/'inputs/prepare_llm_intervention_scientific_training_cpu.py').read_text()
 need('provenance=d.provenance_contract(prefix)' in source_text and 'ownership.supervise(command' in source_text,'pinned provenance before supervision path')
 parent=data(P/'terminal.json');started=data(P/'session/worker.started.json');marker=data(P/'session/worker/work-start.json');resources=data(P/'session/worker.resources.json')
 origin=started['started_monotonic'];inner=sha((R/'inputs/packet.json').read_bytes());stagepin=sha((P/'stage.json').read_bytes())
 expected_command=[prefix+'/bin/python','-I','-B','-c',literal(diagnostic,'BOOTSTRAP'),outer['remote_root']+'/inputs/prepare_llm_intervention_scientific_training_cpu.py',p['files']['prepare_llm_intervention_scientific_training_cpu.py'],
 '--packet',outer['remote_root']+'/inputs/packet.json','--packet-sha256',inner,'--source-commit',source,'--output',outer['remote_root']+'/preparation/session/worker','--worker','--runtime-stage',outer['remote_root']+'/preparation/stage.json','--runtime-stage-sha256',stagepin,'--parent-started',str(origin)]
 need(started['command']==expected_command,'actual child captured command/deadline/source join')
 need(origin<=marker['monotonic']<origin+300,'absolute startup marker')
 # Unlike worker Journal.elapsed, these two receipts are explicitly relative
 # to the same original parent origin in the pinned parent/owned source.
 need(0<resources['elapsed_including_cleanup_seconds']<=parent['elapsed_seconds']<550,'whole child+cleanup and parent completion before absolute work bound')
 failures=[n for n in first['files'] if n.startswith('launch/preparation/') and any(x in Path(n).name for x in ('failure','interruption','uncertain','lifecycle-errors'))]
 need(not failures,'new execution failure evidence '+str(failures))
 for root in (P,P/'session/worker/artifacts'):
  need((root/'terminal.json').read_bytes()==(root/'terminal-candidate.json').read_bytes(),'original terminal/candidate publication')
 phases=[data(x) for x in sorted(P.glob('phase-*.json'))]
 need(phases[0]['phase']=='stage_start' and phases[-1]['phase']=='stage_complete','actual parent staging phases')
 steps=first['results']['steps']['stdout'];batch=[r.split('|') for r in steps.splitlines() if r.startswith(JOB+'.batch|')][0]
 return dict(status='source_scheduler_original_clock_checks_passed',scientific_source_commit=source,source_files=len(outer['source_pins']),
   source_names=len(p['source_names']),entry_sha256=p['files']['prepare_llm_intervention_scientific_training_cpu.py'],
   stage_sha256=stagepin,provenance_sources_sha256=sha((P/'independent-provenance-sources.json').read_bytes()),
   generated_source_sha256=provenance['expected_source_sha256'],stage_start_parent_elapsed=phases[0]['elapsed_seconds'],
   stage_complete_parent_elapsed=phases[-1]['elapsed_seconds'],stage_duration=phases[-1]['elapsed_seconds']-phases[0]['elapsed_seconds'],
   startup_marker_parent_elapsed=marker['monotonic']-origin,child_cleanup_parent_elapsed=resources['elapsed_including_cleanup_seconds'],
   parent_complete_elapsed=parent['elapsed_seconds'],work_bound550_established_by_parent_completion=True,
   worker_journal_clock_not_used_for_parent_deadline=True,exact_cleanup_duration_measured=False,
   runtime_directory_deletion_verified=False,parent_stage_peak_rss_bytes=max(x['peak_self_rss_bytes'] for x in phases),
   child_peak_rss_bytes=resources['cumulative_child_peak_rss_bytes'],slurm_batch_maxrss=batch[8],slurm_batch_total_cpu=batch[9],
   requested_cpus=3,allocated_cpus=3,requested_memory_mib=11520,wall_limit_seconds=600,elapsed_seconds=272,gpus=0,
   actual_artifact_aid_checks_pending=True,acceptance=False)

if __name__=='__main__':
 out=ROOT/'source-scheduler-attempt01';out.mkdir(mode=0o700,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes());start=time.monotonic();cpu=time.process_time()
 try:r=main();code=0
 except BaseException as exc:r=dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),acceptance=False);code=1
 r.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,cost_scope='this local checker process; Git child overhead not measured')
 (out/'receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2));raise SystemExit(code)
