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
 intent=js(O/'intent.json');packet=js(L/'packet.json');req={s:js(O/(s+'.request.json')) for s in ('preserve','complete','preflight','submit')};base=req['preserve'];ctx=base['context'];si=js(L/'submission.intent.json');sr=js(L/'submission.result.json')
 # Every completed local phase record is independently copied; the external storage supplement is explicit.
 local_names={p.name for p in O.iterdir()};external_names={p.name for p in X.iterdir()}
 ck('operation_membership',external_names-local_names=={'storage-operations'} and not local_names-external_names,dict(local_files=len(local_names),external_only=sorted(external_names-local_names)))
 ck('operation_two_copy_bytes',all(equal_copy(O/n,X/n) for n in sorted(local_names)))
 command=G/'transfer-continuation-command01';ec=B/'ACE-LLM-GPU02-transfer-continuation-command-20261110-01'
 ck('invocation_mirrors',set(p.name for p in command.iterdir())==set(p.name for p in ec.iterdir()) and all(equal_copy(p,ec/p.name) for p in command.iterdir()))
 ci=js(command/'intent.json');cr=js(command/'result.json');cout=parse(raw(command/'stdout.log'));ck('invocation_returned_zero',cr['returncode']==0 and cout==sr)
 # Local Git only: three immutable commit roles, no fetch, checkout, imported project or remote action.
 driver=P/'transfer-continuation01/continue_once.py';source=raw(driver);tree=ast.parse(source)
 constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   try:constants[n.targets[0].id]=ast.literal_eval(n.value)
   except (ValueError,TypeError):pass
 ck('driver_bytes',sha(source)==intent['driver_sha256']=='a6996da4b7b4998fa0f576097a52cd72e25312d05697fc45ca83e25fccfab8c5' and pin(driver)==gitpin(intent['driver_commit'],driver) and raw(O/'continue_once.py')==source)
 review=P/'transfer-continuation-review01/review.md';design=R/constants['DESIGN_REVIEW_PATH'];packet_review=R/constants['REVIEW_PATH']
 for label,p,commit,h in [('source_review',review,intent['driver_commit'],intent['driver_review_sha256']),('design_review',design,intent['driver_commit'],intent['design_review_sha256']),('packet_review',packet_review,intent['packet_commit'],intent['packet_review_sha256'])]:
  ck(label+'_git',pin(p)==gitpin(commit,p) and pin(p)['sha256']==h)
 ck('captured_independent_review',raw(O/'independent-review.md')==raw(review))
 ck('invocation_exact_identity',ci['driver_commit']==intent['driver_commit'] and ci['original_packet_commit']==intent['packet_commit'] and ci['argv'][1:3]==['-I','-B'] and ci['argv'][3]==str(driver))
 args=dict(zip(ci['argv'][4::2],ci['argv'][5::2]));ck('invocation_explicit_pins',args=={'--self-sha256':intent['driver_sha256'],'--driver-commit':intent['driver_commit'],'--driver-review':str(review),'--driver-review-sha256':intent['driver_review_sha256'],'--projects-backup':intent['projects_backup']})
 fullpins={**packet['new_input_files'],'packet.json':intent['packet_sha256']};bad=[]
 for name,h in fullpins.items():
  a=pin(L/name);b=pin(E/name);g=gitpin(intent['packet_commit'],L/name)
  if a!=b or a!=g or a['sha256']!=h:bad.append(name)
 ck('all_original_packet_bytes_copies_and_git',not bad,dict(files=len(fullpins),mismatches=bad))
 proofs={}
 for role,(rel,h) in constants['PROOFS'].items():
  p=R/rel;proofs[role]=js(p);ck('original_proof_'+role,pin(p)==gitpin(intent['proof_commit'],p) and pin(p)['sha256']==h==intent['proof_pins'][role])
  if role in ('transfer','failure'):ck('original_proof_mirror_'+role,equal_copy(p,E/p.name))
 ck('original_failed_disposition_unchanged',proofs['transfer']['timeout'] is True and proofs['transfer']['retry_allowed'] is False and proofs['failure']['type']=='ValueError')
 obs=proofs['observation'];ck('captured_original_observation',base['observation']==obs)
 ck('original_readiness_bytes',equal_copy(L/'readiness.json',E/'readiness.json') and pin(L/'readiness.json')['sha256']==obs['files']['readiness.json']['sha256'])
 ck('original_controller_release_retained',equal_copy(L/'controller_released.json',E/'controller_released.json') and pin(L/'controller_released.json')==gitpin(intent['proof_commit'],L/'controller_released.json'))
 # Same exact request context, not fresh inferred authority; copies and canonical serialized joins.
 for phase,r in req.items():
  ck(phase+'_shared_request_context',all(r[k]==base[k] for k in base if k!='phase') and r['phase']==phase)
  ck(phase+'_exact_launcher_bytes',base64.b64decode(r['launcher_source'],validate=True)==raw(L/'launch_once.py'))
 ck('request_pins_packet',ctx['pins']==fullpins and si['pins']==fullpins)
 ck('request_lineage',base['lineage']==intent and base['projects_backup']==intent['projects_backup'])
 neednames=set(obs['missing'])|{'gpu-failure-disposition.json'}
 members={n:dict(bytes=pins[str(L/n)]['bytes'],sha256=fullpins[n]) for n in sorted(neednames)}
 ck('missing_only_roster',base['members']==members and len(members)==168 and len(obs['missing'])==167 and set(obs['files'])|set(obs['missing'])==set(fullpins))
 completed=js(O/'completion.json');compout=parse(raw(O/'complete.stdout'))
 ck('completion_exact_members',completed==compout and completed['completed_members']==members and completed['matching_original_members_unchanged'] is True and completed['acceptance'] is False and completed['total_bytes']==sum(v['bytes'] for v in members.values())==414954743)
 pres=parse(raw(O/'preserve.stdout'));ack=js(O/'partial-local-readback.json');part=base64.b64decode(pres['partial_base64'],validate=True)
 ck('preservation_record_hash',pres['record_sha256']==sha(canonical(pres['record'])) and pres['record']['lineage']==intent and pres['record']['acceptance'] is False)
 ck('preserved_partial_exact',len(part)==2392064 and sha(part)==constants['PARTIAL_SHA'] and part==raw(O/'partial.original')==raw(X/'partial.original'))
 # No decoding original opaque disposition: the prefix is byte-compared only.
 with (L/'gpu-failure-disposition.json').open('rb') as f:prefix=f.read(len(part))
 ck('preserved_original_prefix',part==prefix)
 ck('projects_preservation_join',pres['record']['projects_backup']==intent['projects_backup']+'/partial.original' and pres['record']['original_path']==intent['remote_root']+'/gpu-failure-disposition.json' and pres['record']['bytes']==len(part) and pres['record']['sha256']==sha(part))
 expectedack=dict(partial_sha256=sha(part),partial_bytes=len(part),local_copies={str(p):dict(bytes=len(part),sha256=sha(part)) for p in (O/'partial.original',X/'partial.original')},preservation_sha256=pres['record_sha256'])
 ck('both_local_ack_before_complete',ack==expectedack==req['complete']['local_ack'] and req['complete']['preservation_sha256']==pres['record_sha256'])
 pf=js(O/'pre-submission.json');ck('preflight_record_stdout',pf==parse(raw(O/'preflight.stdout')))
 extline=dict(intent,preservation_sha256=pres['record_sha256'],local_readback_sha256=sha(canonical(ack)),completion_sha256=sha(canonical(completed)),preflight_sha256=sha(canonical(pf)))
 ck('transitive_phase_bindings',req['preflight']['completion_sha256']==extline['completion_sha256'] and req['submit']['completion_sha256']==extline['completion_sha256'] and req['submit']['preflight_sha256']==extline['preflight_sha256'] and req['submit']['submission_lineage']==extline==si['transfer_continuation'])
 ck('actual_local_intent_exact',req['submit']['submission_intent']==si and equal_copy(L/'submission.intent.json',E/'submission.intent.json'))
 # Exact rows and path/hash maps, with totals independently summed; do not rereview predecessors' scientific work.
 fresh=pf['fresh'];baseline=ctx['baseline'];bfile=js(L/'gpu-failure-baseline.json')
 ck('baseline_actual_pin_and_content',baseline==bfile and pin(L/'gpu-failure-baseline.json')['sha256']==packet['baseline_sha256']==si['baseline_sha256'])
 def normalize(rows):
  out={}
  for row in rows:
   if len(row)!=8 or row[0] in out:raise ValueError('duplicate/malformed accounting row')
   tres={}
   for s in row[7].split(','):
    k,v=s.split('=',1)
    if k in tres:raise ValueError('duplicate TRES')
    tres[k]=v
   out[row[0]]=tuple(row[:7])+(tuple(sorted(tres.items())),)
  return out
 ck('fresh_exact_47_accounting_rows',normalize(fresh['related_accounting'])==normalize(baseline['related_accounting']) and len(fresh['related_accounting'])==47)
 ck('fresh_exact_96_journal_pins',fresh['journal_pins']==baseline['journal_pins'] and len(fresh['journal_pins'])==96)
 cpu=gpu=0
 for row in fresh['related_accounting']:
  cpu+=int(row[4])*int(row[5]);t=dict(x.split('=',1) for x in row[7].split(','));gpu+=int(row[5])*sum(int(v) for k,v in t.items() if k.startswith('gres/gpu:'))
 ck('closed_cost_and_candidate_reservations',(cpu,gpu)==(30241,2750) and cpu==fresh['closed_cpu_seconds']==baseline['closed_cpu_seconds'] and gpu==fresh['closed_gpu_slice_seconds']==baseline['closed_gpu_slice_seconds'] and fresh['cpu_reserved_total']==cpu+3600+4*2880==45361 and fresh['gpu_reserved_total']==gpu+2880==5630)
 ck('no_future_actual_cost_assumed',baseline['outstanding_reservation']==0 and baseline['terminal_jobs']==47 and baseline['submission_journals']==96 and all(r[0]!='35516125' for r in baseline['related_accounting']))
 ck('full_preflight_200_input_pins',pf['verified']['verified']==fullpins)
 ck('fresh_namespace_runtime_failure_joins',fresh['existing_namespace_claim_sha256']==ctx['namespace_claim_sha256']==si['existing_namespace_claim_sha256'] and fresh['original_namespace_claim_sha256']==ctx['original_namespace_claim_sha256']==si['original_namespace_claim_sha256'] and fresh['model_files']==ctx['model_files'] and fresh['manifest_sha256']==ctx['manifest']['sha256'] and fresh['archive_size']==ctx['archive']['bytes'] and fresh['failed_gpu_original_bytes_verified'] is True and fresh['frozen_output_absent'] is True and fresh['gpu_recovery']==ctx['gpu_recovery']==packet['gpu_recovery']==si['gpu_recovery'])
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
 checks.append(dict(check='audit_inspector_completed',passed=False,details=repr(exc)));(W/'attempt01-traceback.txt').write_text(traceback.format_exc())
finally:
 result=dict(scope='saved operational metadata/byte/Git joins only; no scientific acceptance',checks=checks,failed_checks=[v for v in checks if not v['passed']],facts=facts,elapsed_seconds=time.monotonic()-start)
 (W/'attempt01-report.json').write_bytes(canonical(result));(W/'input-pins.json').write_bytes(canonical(pins))
 print(json.dumps(dict(checks=len(checks),failed_checks=result['failed_checks'],elapsed_seconds=result['elapsed_seconds'],workspace=str(W)),indent=2))
