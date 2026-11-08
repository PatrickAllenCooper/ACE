"""Independent saved-custody audit. Stdlib bytes/JSON/AST/Git only; no project imports."""
import ast, hashlib, json, os, stat, subprocess, sys, time
from pathlib import Path
REPO=Path('/Users/pat/code/ACE')
REL='results/ace_llm_intervention_scientific_collection_20261108/terminal01'
LOCAL=REPO/REL
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-terminal-20261108-01')
JOB='35407428'
COMMIT='67680d3eb9fb0276a8cf9fd01cfddded8ba46c97'
SOURCE='a9346022f145cc29439325b6f291597b2e6ec037'
PACKET='aaf0f21110c0906ba0a6a0ba23faa9e0b5ffc6098f1802682901d47e28302961'
BASELINE='69cc97784637b0ec522da6e00e00a1f08bdfabe9712e2104a43010ee19f2035f'
INTENT='61b1d2a2043764a8d82e04dcfafc95baf6dd66598597ed2e9bfa0e94b42ce9b8'
FAILURE='9b34ca42142fd4ce8fb79f1a3319f2a2efde8df0bb90f5ad32d18dcf4c237d59'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(p):
 need(not p.is_symlink(),'symlink file')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode),'not regular');b=f.read();z=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(identity(a)==identity(z)==identity(p.lstat()),'changed during read')
 return b

def decode(b):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'duplicate key');d[k]=v
  return d
 def reject(_):raise ValueError('nonfinite')
 v=json.loads(b,object_pairs_hook=pairs,parse_constant=reject)
 json.dumps(v,allow_nan=False)
 return v

def read(p):return decode(raw(p))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def inventory(root):
 files={};dirs=[]
 need(root.is_dir() and not root.is_symlink(),'ordinary root required')
 for p in sorted(root.rglob('*')):
  need(not p.is_symlink(),'symlink in capsule')
  n=p.relative_to(root).as_posix()
  if p.is_dir():dirs.append(n)
  else:
   need(p.stat().st_nlink==1,'independent regular custody required')
   b=raw(p);files[n]={'bytes':len(b),'sha256':sha(b)}
 return dict(files=files,directories=dirs)
def normalize(rows):
 out=[];ids=set()
 for row in rows:
  r=list(row)
  while r and r[-1]=='':r.pop()
  need(len(r)==8 and r[0] not in ids and r[4].isdigit() and r[5].isdigit(),'invalid accounting row');ids.add(r[0])
  pairs=[s.split('=',1) for s in r[7].split(',') if s]
  need(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'invalid TRES')
  r[7]=','.join(k+'='+v for k,v in sorted(pairs));out.append(r)
 return sorted(out)
def git(commit,path):
 return subprocess.run(['git','--no-replace-objects','show',commit+':'+path],cwd=REPO,capture_output=True,check=True).stdout

def main(out):
 start=time.monotonic();cpu=time.process_time();result=dict(status='incomplete',acceptance=False)
 try:
  inv=inventory(LOCAL);external=inventory(EXT);need(inv==external,'capsule copies differ')
  sealed=read(LOCAL/'custody-inventory.json');check={**inv,'files':{k:v for k,v in inv['files'].items() if k!='custody-inventory.json'}}
  need(sealed['identical'] is True and sealed['acceptance'] is False and len(sealed['copies'])==2,'sealed inventory shape')
  need(all(v==check for v in sealed['copies'].values()),'sealed inventories differ')
  observations=[]
  for name in ['inventory_transport.json','inventory_after_transport.json']:
   transport=read(LOCAL/name);need(transport['returncode']==0,'failed transport')
   v=decode(transport['stdout'].encode());need(v['terminal_gate'] is True and not v.get('observation_error'),'terminal observation failed')
   # Captured source is parsed only, never evaluated.
   tree=ast.parse(transport['python_source']);constants={}
   for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
     try:constants[n.targets[0].id]=ast.literal_eval(n.value)
     except (ValueError,TypeError):pass
   need(constants['JOB']==JOB,'observed job differs');observations.append(v)
  before,after=observations
  for k in ('files','directories','journal_pins','parent_accounting','original_hardlink_groups','upstream_pins','upstream_errors'):
   need(before[k]==after[k],'second observation changed '+k)
  need(normalize(before['related_accounting'])==normalize(after['related_accounting']),'accounting snapshots differ')
  remote=inventory(LOCAL/'remote');need(remote==dict(files=before['files'],directories=before['directories']),'raw mirror differs')
  need(len(remote['files'])==146 and sum(v['bytes'] for v in remote['files'].values())==2336700,'raw size differs')
  d=read(LOCAL/'disposition.json');need(d['remote_files']==remote['files'] and d['remote_directories']==remote['directories'],'disposition custody differs')
  need(d['status']=='terminal_failed_outputs_preserved' and d['errors']==d['input_pin_mismatches']==[],'collector conflicts')
  need(all(d[k] is True for k in ['custody_bytes_verified','current_lineage_verified','exact_prior_accounting_matched','exact_prior_and_current_journals_matched','upstream_training_inputs_verified']),'collector lineage flags')
  need(all(d[k] is False for k in ['acceptance','collection_accepted','scientific_qualified','retry_allowed']),'collector original qualification flags')
  launch=LOCAL/'remote/launch';p=read(launch/'packet.json');need(sha(raw(launch/'packet.json'))==PACKET and p['source_commit']==SOURCE,'packet differs')
  packetrel='results/ace_llm_intervention_scientific_collection_20261108/launch01/'
  for n,h in {'packet.json':PACKET,**p['new_input_files']}.items():
   b=raw(launch/n);need(sha(b)==h,'input mismatch '+n);need(git(COMMIT,packetrel+n)==b,'packet Git mismatch '+n)
   for label in ['local-launch','external-launch']:need(raw(LOCAL/'coordinator'/label/n)==b,'coordinator input differs '+n)
  for rel,h in p['source_pins'].items():
   b=raw(launch/'inputs'/Path(rel).name);need(sha(b)==h and git(SOURCE,rel)==b,'source Git mismatch')
  need(sha(raw(launch/'baseline.json'))==BASELINE,'baseline pin differs');base=read(launch/'baseline.json')
  ri=read(launch/'submission.intent.json');rr=read(launch/'submission.result.json')
  need(sha(raw(launch/'submission.intent.json'))==INTENT==rr['intent_sha256'],'intent pin differs')
  need(rr['job_id']==JOB and rr['returncode']==0 and rr['ambiguous'] is False and rr['stdout'].strip()==JOB,'submission result differs')
  need(ri['packet_commit']==COMMIT and ri['packet_sha256']==PACKET and ri['source_commit']==SOURCE,'intent lineage')
  need(ri['baseline_sha256']==BASELINE and ri['pins']==p['new_input_files'],'intent inputs')
  for label in ['local-launch','external-launch']:
   lr=read(LOCAL/'coordinator'/label/'submission.result.json');li=read(LOCAL/'coordinator'/label/'submission.intent.json')
   need(lr==rr and set(ri)==set(li)|{'remote_utc','fresh_readiness'} and all(ri[k]==v for k,v in li.items()),'local/remote intent semantics')
  review_rel='docs/development/guidance/reviews/ace_llm_intervention_scientific_collection01_packet_review_2026-11-08.md'
  need(sha(git(COMMIT,review_rel))==ri['review_sha256'],'committed independent packet review')
  need(sha(raw(LOCAL/'pull_terminal.py'))=='754efa824d7589795f127adf07f66a46baa9f281f13758ca98ba3ad7a28ba83e','collector pin')
  # Accepted upstream transport is metadata only; no realization descriptors decoded.
  original=decode(read(launch/'realization-inventory.json')['stdout'].encode())
  upstream=before['upstream_pins'];need(len(upstream)==36 and before['upstream_errors']=={},'upstream errors')
  oldroot=p['remote_root'].replace('scientific_collection_20261108_01','scientific_realization_20261108_02')
  for path,h in upstream.items():need(original['files']['launch/'+path.removeprefix(oldroot+'/')]['sha256']==h,'accepted upstream mismatch')
  for name,pin in [('namespace.json',p['namespace_claim_sha256']),('original-namespace.json',ri['original_namespace_claim_sha256'])]:
   need(remote['files'][name]['sha256']==pin,'remote namespace changed')
  need(sha(raw(LOCAL/'namespaces/local.namespace.json'))==p['namespace_claim_sha256'],'local namespace changed')
  # Exact closed allocation rows, not count-only checks.
  rows=normalize(before['related_accounting']);prior=[r for r in rows if r[0]!=JOB]
  need(len(rows)==41 and len(prior)==40 and prior==normalize(base['related_accounting']),'prior rows differ')
  actual=next(r for r in rows if r[0]==JOB)
  need(actual[:7]==[JOB,'ace_llm_scientific_collect01','FAILED','1:0','2','126','ucb736_asc2'],'actual row differs')
  tres=dict(x.split('=',1) for x in actual[7].split(','));need(tres==dict(billing='2',cpu='2',mem='7.50G',node='1'),'actual TRES differs')
  journals={**base['journal_pins'],p['remote_root']+'/submission.intent.json':INTENT,p['remote_root']+'/submission.result.json':sha(raw(launch/'submission.result.json'))}
  need(len(journals)==84 and journals==before['journal_pins'],'journal map differs')
  cpu_total=sum(int(r[4])*int(r[5]) for r in rows);gpu_total=0
  for r in rows:
   need(r[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'},'active allocation')
   t=dict(x.split('=',1) for x in r[7].split(',') if x)
   count=int(t['gres/gpu']) if 'gres/gpu' in t else sum(int(v) for k,v in t.items() if k.startswith('gres/gpu:'))
   gpu_total+=count*int(r[5])
  need(cpu_total==base['closed_cpu_seconds']+252==11386 and gpu_total==base['closed_gpu_slice_seconds']==1601,'cumulative charges differ')
  for v in observations:
   need(all(v['results'][n]['returncode']==0 for n in ('steps','details','job','queue','related_accounting')),'scheduler query failed')
   need(all(not l.split('|')[1].lower().startswith('ace') for l in v['results']['queue']['stdout'].splitlines() if l),'ACE job remains active')
   detail=v['results']['details']['stdout'].strip().split('|')
   need(detail[:4]==[JOB,'acpu','cpu-normal','15'] and detail[5]=='paco0228','requested resource profile')
   need(dict(x.split('=',1) for x in detail[4].split(','))==tres,'requested TRES differs')
   job=v['results']['job']['stdout'];need(all(x in job.split() for x in ['JobState=FAILED','ExitCode=1:0','Requeue=0','Restarts=0','Reason=NonZeroExitCode']),'scontrol differs')
   steps=[l.split('|') for l in v['results']['steps']['stdout'].splitlines() if l]
   need(len(steps)==3 and normalize([steps[0][:8]])==[actual] and {s[0] for s in steps}=={JOB,JOB+'.batch',JOB+'.extern'},'job steps differ')
  need(d['closed_cpu_seconds']==cpu_total and d['closed_gpu_slice_seconds']==gpu_total and d['cpu_admission_after_release']==14986,'disposition totals differ')
  # Positive traceback and exact source call order support the no-response boundary.
  error=raw(launch/'logs'/f'{JOB}.err').decode();failure=read(launch/'collection/failure.json')
  need(sha(raw(launch/'collection/failure.json'))==FAILURE,'failure pin differs')
  need(failure['type']=='InterruptedError' and failure['message']=='scientific CPU stage interrupted','failure type differs')
  for fragment in ['line 398, in parent','line 194, in stage_runtime','line 241, in stage','line 540, in event','line 86, in write','line 428, in interrupted']:
   need(fragment in error,'traceback boundary absent '+fragment)
  source=raw(launch/'inputs/supervise_llm_intervention_scientific_cpu.py').decode();tree=ast.parse(source)
  parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parent')
  calls=[(ast.unparse(n.func),n.lineno) for n in ast.walk(parent) if isinstance(n,ast.Call)]
  stage_line=next(l for n,l in calls if n=='d.stage_runtime');spawn_line=next(l for n,l in calls if n=='ownership.supervise')
  need(stage_line==398 and spawn_line==409 and stage_line<spawn_line,'stage before worker spawn source changed')
  absent=['stage.json','independent-provenance-sources.json','session','terminal.json','terminal-candidate.json','task']
  need(all(not (launch/'collection'/n).exists() for n in absent),'unexpected post-stage artifact')
  need(not any(k.endswith('.jsonl') or 'work-start' in k or '/task/' in k for k in remote['files']),'unexpected scientific work artifact')
  phases=[read(q) for q in sorted((launch/'collection').glob('phase-*.json'))]
  need(len(phases)==55 and phases[0]['phase']=='stage_start' and all(q['phase']=='stage_progress' for q in phases[1:]),'staging phase lineage')
  need(phases[-1]['detail']==dict(completed_objects=26501,objects=49966,phase='stage') and 119.9<phases[-1]['elapsed_seconds']<120,'last stage observation differs')
  need(all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(phases,phases[1:])),'phase clock order')
  start_record=read(launch/'collection/start.json');inner=read(launch/'inputs/packet.json')
  need(start_record['packet_sha256']==p['new_input_files']['inputs/packet.json'] and start_record['resources']==p['resources'],'start binding differs')
  origin=start_record['deadlines']['startup']-120
  need(all(abs(start_record['deadlines'][k]-origin-v)<1e-7 for k,v in p['deadlines'].items()),'deadlines differ')
  need(all(b['mount_alias'] is True and b['root_device']==62 and b['root_inode']==8388611 for b in start_record['output_path_bindings']),'observed mount alias differs')
  # Final readback protects the audit from evolving custody.
  need(inventory(LOCAL)==inv and inventory(EXT)==external,'custody changed during audit')
  result.update(status='saved_failure_custody_and_no_response_checks_passed',acceptance=False,
   raw_files=146,raw_bytes=2336700,capsule_files=len(inv['files']),capsule_bytes=sum(v['bytes'] for v in inv['files'].values()),
   source_files=len(p['source_pins']),input_objects=len(p['new_input_files'])+1,terminal_jobs=41,submission_journals=84,
   prior_cpu_seconds=11134,allocated_cpu_seconds=252,closed_cpu_seconds=cpu_total,closed_gpu_slice_seconds=gpu_total,
   preparation_allowance_cpu_seconds=3600,cpu_admission_after_release=14986,
   scientific_worker_spawned=False,collection_invoked=False,environment_response_attempts=0,environment_responses=0,
   scientific_worlds_generated=0,records_created=0,pairs_created=0,training_updates=0,
   runtime_staging_started=True,runtime_staging_completed=False,runtime_qualified_for_this_execution=False,
   owned_worker_cleanup_applicable=False,node_local_partial_runtime_cleanup_verified=False,
   failure_type=failure['type'],last_phase=phases[-1],phase_count=len(phases),
   signal_number_recorded=False,startup_alarm_explanation='Strongly supported by exact source, 120s timer and last receipt at119.991s; signal number unrecorded.',
   peak_stage_progress_self_rss_bytes=max(q['peak_self_rss_bytes'] for q in phases),
   scheduler_steps=before['results']['steps']['stdout'],
   evidence_pins={n:inv['files'][n]['sha256'] for n in ['disposition.json','custody-inventory.json','inventory_transport.json','inventory_after_transport.json','pull_terminal.py','remote/launch/submission.intent.json','remote/launch/submission.result.json','remote/launch/logs/'+JOB+'.err','remote/launch/collection/failure.json','remote/launch/collection/start.json']},
   all_capsule_files=inv,actual_rows=rows,actual_journal_pins=journals)
 except BaseException as exc:
  result.update(exception_type=type(exc).__name__,message=str(exc));raise
 finally:
  result.update(process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-start,
    audit_source_sha256=sha(raw(Path(__file__))))
  save(out/'audit.json',result)
if __name__=='__main__':
 out=Path(sys.argv[1]);out.mkdir(mode=0o700,exist_ok=False);main(out)
