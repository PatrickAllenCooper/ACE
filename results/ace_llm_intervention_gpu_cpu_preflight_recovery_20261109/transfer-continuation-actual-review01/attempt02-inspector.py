"""Independent saved-byte/JSON/Git audit only; no project imports or callable execution."""
import ast,base64,datetime,hashlib,json,os,pathlib,shlex,stat,subprocess,time,traceback
R=pathlib.Path('/Users/pat/code/ACE');B=pathlib.Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
P=R/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109';G=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
L=G/'launch02';E=B/'ACE-LLM-intervention-scientific-campaign-gpu-20261109-02';O=L/'transfer-continuation01';X=E/O.name
W=pathlib.Path(__file__).parent;checks=[];pins={};facts={};start=time.monotonic()
def ck(name,good,details=None):
 checks.append(dict(check=name,passed=bool(good),details=details))
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(d):return (json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def raw(p):
 p=pathlib.Path(p);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('not sole regular file: '+str(p))
 with p.open('rb') as f:b=f.read()
 t=p.lstat()
 if (s.st_ino,s.st_size,s.st_mtime_ns)!=(t.st_ino,t.st_size,t.st_mtime_ns):raise ValueError('changed during read: '+str(p))
 pins[str(p)]=dict(bytes=len(b),sha256=sha(b));return b
def pin(p):
 p=pathlib.Path(p);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('not sole regular file: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 t=p.lstat()
 if (s.st_ino,s.st_size,s.st_mtime_ns)!=(t.st_ino,t.st_size,t.st_mtime_ns):raise ValueError('changed during hash: '+str(p))
 v=dict(bytes=s.st_size,sha256=h.hexdigest());pins[str(p)]=v;return v
def pairs(xs):
 d={}
 for k,v in xs:
  if k in d:raise ValueError('duplicate JSON key '+k)
  d[k]=v
 return d
def parse(b):return json.loads(b,object_pairs_hook=pairs)
def js(p):return parse(raw(p))
def gitpin(commit,p):
 rel=str(pathlib.Path(p).relative_to(R));r=subprocess.run(['git','--no-replace-objects','show',commit+':'+rel],cwd=R,capture_output=True,timeout=30)
 if r.returncode:raise ValueError('Git read '+rel+': '+r.stderr.decode())
 return dict(bytes=len(r.stdout),sha256=sha(r.stdout))
def equal_copy(a,b):
 av=pin(a);bv=pin(b);return av==bv and a.stat().st_ino!=b.stat().st_ino
try:
 old=js(W/'attempt01-report.json')
 checks.extend(v for v in old['checks'] if v['check'] not in ('missing_only_roster','audit_inspector_completed'))
 pins.update(json.loads((W/'input-pins.json').read_text()))
 intent=js(O/'intent.json');packet=js(L/'packet.json');req={s:js(O/(s+'.request.json')) for s in ('preserve','complete','preflight','submit')};base=req['preserve'];ctx=base['context'];si=js(L/'submission.intent.json');sr=js(L/'submission.result.json')
 source=raw(O/'continue_once.py');tree=ast.parse(source);constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   try:constants[n.targets[0].id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 fullpins={**packet['new_input_files'],'packet.json':intent['packet_sha256']}
 obs=base['observation'];local_names={p.name for p in O.iterdir()};command=G/'transfer-continuation-command01';ci=js(command/'intent.json');cr=js(command/'result.json');cout=parse(raw(command/'stdout.log'))
 members=base['members'];completed=js(O/'completion.json');pres=parse(raw(O/'preserve.stdout'));ack=js(O/'partial-local-readback.json');part=raw(O/'partial.original')
 pf=js(O/'pre-submission.json');fresh=pf['fresh'];baseline=ctx['baseline'];cpu=fresh['closed_cpu_seconds'];gpu=fresh['closed_gpu_slice_seconds']
 extline=dict(intent,preservation_sha256=pres['record_sha256'],local_readback_sha256=sha(canonical(ack)),completion_sha256=sha(canonical(completed)),preflight_sha256=sha(canonical(pf)))
 ck('missing_only_roster',members=={n:dict(bytes=pins[str(L/n)]['bytes'],sha256=fullpins[n]) for n in sorted(set(obs['missing'])|{'gpu-failure-disposition.json'})} and len(members)==168 and len(obs['missing'])==167 and set(obs['files'])|set(obs['missing'])==set(fullpins)|{'readiness.json'})
 ck('fresh_namespace_runtime_failure_joins',fresh['existing_namespace_claim_sha256']==ctx['namespace_claim_sha256']==si['existing_namespace_claim_sha256'] and fresh['original_namespace_claim_sha256']==ctx['original_namespace_claim_sha256']==si['original_namespace_claim_sha256'] and fresh['model_files']==ctx['model_files'] and fresh['manifest_sha256']==ctx['manifest']['sha256'] and fresh['archive_size']==js(L/'readiness.json')['archive_size']==8438650880 and fresh['failed_gpu_original_bytes_verified'] is True and fresh['frozen_output_absent'] is True and fresh['gpu_recovery']==ctx['gpu_recovery']==packet['gpu_recovery']==si['gpu_recovery'])
 for k in ['resources','limits','admission','source_commit','storage_binding','operational_source_commit','token_recovery','cpu_acceptance_sha256','cpu_job_id','failed_token_root','failed_token_job_id']:
  ck('unchanged_intent_'+k,si[k]==packet[k])
 ck('no_new_namespace_or_worlds',si['new_namespace_claimed'] is False and si['replacement_worlds'] is False and si['blind_retry_forbidden'] is True and packet['new_training_updates']==0 and packet['new_worlds']==0)
 # All remote phase executions used the independently pinned exact driver/bootstrap; no caller imports.
 phases=[]
 for phase in req:
  t=js(O/(phase+'.transport.json'));cmd=t['command'];remote=shlex.split(cmd[-1]);limit=630 if phase=='complete' else 1230
  ck(phase+'_transport_bound_cleanup',t['returncode']==0 and t['ambiguous'] is False and t['retry_allowed'] is False and t['stream_errors']==[] and t['stream_threads_closed'] is True and t['owned_process_group_closed'] is True and t['owned_group_probe']['closed'] is True and t['owned_group_probe']['observations'][-1]=='ESRCH' and 0<t['elapsed_seconds']<limit)
  ck(phase+'_bootstrap_source_pin',remote==['python3','-I','-B','-c',constants['BOOTSTRAP'],sha(source),str(len(source))])
  phases.append(dict(phase=phase,start_utc=t['utc'],elapsed_seconds=t['elapsed_seconds'],transport_bound_seconds=limit,returncode=t['returncode'],owned_process_group_closed=t['owned_process_group_closed']))
 ck('one_unambiguous_result_join',sr==parse(raw(O/'submit.stdout'))==cout and sr['job_id']=='35516125' and sr['returncode']==0 and sr['ambiguous'] is False and sr['stdout']=='35516125\n' and sr['stderr']=='' and sr['intent_sha256']=='dd7148afb1fcad65834c717b39c1d4728860277e9e0ef75278c65378e6de0d1b' and pin(L/'submission.result.json')['sha256']=='ab4719a46cb68901829bc5617d1321a94021ffca7be89046554c69e9518f51fc' and equal_copy(L/'submission.result.json',E/'submission.result.json'))
 terminal=js(O/'terminal.json');ck('operation_terminal_join',terminal==dict(status='one_submission_recorded',job_id=sr['job_id'],lineage=extline,acceptance=False) and not any('failure' in n or 'retry' in n for n in local_names))
 times=[ci['utc'],pres['record']['utc'],js(O/'complete.transport.json')['utc'],pf['utc'],si['utc'],sr['utc'],cr['utc']];ck('phase_timestamp_order',times==sorted(times),times)
 # External-only storage evidence is explicitly referenced from both operation copies.
 storage=X/'storage-operations';sti=js(storage/'intent.json');stref=js(O/'storage-use.intent.json')
 ck('storage_intent_join',stref['intent_sha256']==pin(storage/'intent.json')['sha256'] and stref['path']==str(storage) and sti['identity']['packet_commit']==intent['packet_commit'] and sti['identity']['packet_sha256']==intent['packet_sha256'] and sti['acceptance'] is False)
 inventories=[]
 for i in range(98):
  ref=js(O/f'storage-use-{i:03}.json');invpath=storage/f'inventory-{i:03}.json';inv=js(invpath);srref=js(storage/f'reference-{i:03}.json')
  ck(f'storage_operation_{i:03}',ref==srref and ref['inventory_path']==str(invpath) and ref['inventory_sha256']==pin(invpath)['sha256'] and ref['intent_sha256']==stref['intent_sha256']==inv['intent_sha256'] and inv['ordinal']==i and inv['status']==ref['status']=='completed' and inv['error'] is None and inv['acceptance'] is False and ref['acceptance'] is False and inv['retry_allowed'] is False)
  inventories.append(inv)
 finalmap=inventories[-1]['inventory'];ops=storage/'operations';opfiles={str(p.relative_to(ops)) for p in ops.rglob('*') if p.is_file()};opdirs=sorted(str(p.relative_to(ops)) for p in ops.rglob('*') if p.is_dir())
 ck('storage_final_current_operation_membership',opfiles==set(finalmap['files']) and opdirs==finalmap['directories'])
 bad=[]
 for n,info in finalmap['files'].items():
  if pin(ops/n)!=info:bad.append(n)
 ck('storage_final_current_operation_bytes',not bad,dict(files=len(finalmap['files']),mismatches=bad))
 for folder in ['verify-000','verify-001','fetch-log-028','fetch-log-091']:
  t=js(ops/folder/'transport.json');rc=js(ops/folder/'receipt.json');facts.setdefault('storage_phase_summaries',[]).append(dict(phase=folder,transport={k:v for k,v in t.items() if k!='command'},receipt_keys=list(rc)))
 # These are local current-operation bytes, not repeated remote/ancestor scientific acceptance.
 facts.update(dict(operation_local_files=len(local_names),completed_members=len(members),completed_bytes=completed['total_bytes'],preserved_partial_bytes=len(part),preserved_partial_sha256=sha(part),packet_files=len(fullpins),original_matching_members=len(obs['files'])-1,prior_terminal_jobs=47,prior_journals=96,closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,candidate_cpu_admission=45361,candidate_gpu_admission=5630,candidate_with_finalizer_cpu_admission=47761,job_id=sr['job_id'],submission_utc=sr['utc'],remote_intent_sha256=sr['intent_sha256'],local_intent_sha256=pin(L/'submission.intent.json')['sha256'],submission_result_sha256=pin(L/'submission.result.json')['sha256'],invocation=cr,phase_timings=phases,storage_operations=98,storage_final_inventory_sha256=pin(storage/'inventory-097.json')['sha256'],source_driver_commit=intent['driver_commit'],original_packet_commit=intent['packet_commit'],failure_proof_commit=intent['proof_commit']))
except BaseException as exc:
 checks.append(dict(check='audit_inspector_completed',passed=False,details=repr(exc)));(W/'attempt02-traceback.txt').write_text(traceback.format_exc())
finally:
 result=dict(scope='saved operational metadata/byte/Git joins only; no scientific acceptance',checks=checks,failed_checks=[v for v in checks if not v['passed']],facts=facts,elapsed_seconds=time.monotonic()-start)
 (W/'attempt02-report.json').write_bytes(canonical(result));(W/'input-pins-final.json').write_bytes(canonical(pins))
 print(json.dumps(dict(checks=len(checks),failed_checks=result['failed_checks'],elapsed_seconds=result['elapsed_seconds'],workspace=str(W)),indent=2))
