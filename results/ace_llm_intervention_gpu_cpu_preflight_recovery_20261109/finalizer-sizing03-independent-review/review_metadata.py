"""Independent saved timing/metadata arithmetic only; no producer imports or replay."""
from pathlib import Path, PurePosixPath
import ast,hashlib,json,math,re,shutil,stat,subprocess,traceback
W=Path(__file__).parent;R=Path('/Users/pat/code/ACE');P=R/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109';S=P/'finalizer-sizing03';E=Path((S/'external-path.txt').read_text().strip());checks=[];observed_pins={}
def raw(x):return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def ck(n,b,details=None):checks.append(dict(check=n,passed=bool(b),details=details))
def unique(items):
 d={}
 for k,v in items:
  if k in d:raise ValueError('duplicate key: '+k)
  d[k]=v
 return d
def read(p):
 p=Path(p);st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1
 b=p.read_bytes();after=p.lstat();assert (st.st_ino,st.st_mtime_ns,st.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size)
 observed_pins[str(p)]=dict(bytes=len(b),sha256=sha(b));return b
def j(p):return json.loads(read(p),object_pairs_hook=unique)
def manifest(root):return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(read(p))) for p in sorted(root.rglob('*')) if p.is_file()}
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-7)
try:
 sm=manifest(S);em=manifest(E);ck('preparation_two_complete_copies',sm==em);shutil.copytree(S,W/'snapshot',copy_function=shutil.copyfile)
 prep=j(S/'preparation.json');paired=j(S/'attempt01/derived/derived.json');d=j(S/'attempt02/derived/disposition.json');old=j(S/'closed-timing-evidence.json')
 ck('announced_preparation_pin',sha(read(S/'preparation.json'))=='895cc0b59f07d768ca627569af236a5d7c92cbb545276ea7853dffac9086e276')
 ck('announced_disposition_pin',sha(read(S/'DISPOSITION.md'))=='be3fe18e35c5a6f5af7d2b59e3099e79660e522b42fbe0ae443f7fd64da401fd')
 ck('machine_disposition_pin',sha(read(S/'attempt02/derived/disposition.json'))=='9443fadca1c824410af732b27462eb1870a0718645f2103143c7975b1c924e27')
 for n,v in prep['objects'].items():ck('preparation_object:'+n,observed_pins.get(str(S/n),dict(bytes=len(read(S/n)),sha256=sha(read(S/n))))==v)
 ck('derivation_links',d['paired_derivation_sha256']==sha(read(S/'attempt01/derived/derived.json')) and d['closed_timing_evidence_sha256']==sha(read(S/'closed-timing-evidence.json')))
 for a in ['attempt01','attempt02']:
  run=j(S/a/'run.json');ck(a+'_saved_success',run['returncode']==0 and run['error'] is None and run['remote_actions']==0 and run['scientific_functions_executed']==0)
 for src,attempt in [('derive_inventory_timing.py','attempt01'),('derive_disposition.py','attempt02'),('derive_inventory_timing.py','attempt02'),('closed-timing-evidence.json','attempt02')]:ck('executed_capture:'+attempt+'/'+src,read(S/src)==read(S/attempt/src))
 ck('copied_paired_derivation',read(S/'attempt01/derived/derived.json')==read(S/'attempt02/attempt01/derived/derived.json'))
 srcpin=j(S/'source-pin.json');source=read(S/'committed-finalizer-entry.py');git=subprocess.run(['git','--no-replace-objects','show',srcpin['source_commit']+':'+srcpin['repository_path']],cwd=R,capture_output=True,timeout=30)
 ck('entry_exact_committed_profile',git.returncode==0 and git.stdout==source and sha(source)==srcpin['sha256']==d['entry_sha256']==prep['entry_sha256']=='8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f')
 ck('closed_numeric_review_pin',sha(read(P/'finalizer-clock-review03/review.md'))==d['source_review_sha256']==prep['source_review_sha256']=='39ef3af657fd64e8174d6c0b462fdb1519c1978ee3ea9d5288f11c998d4a5bc9')
 inputs={}
 for n,v in prep['input_pins'].items():
  b=read(v['path']);ck('actual_metadata_pin:'+n,observed_pins[v['path']]=={k:v[k] for k in ('bytes','sha256')} and d['input_pins'][n]==v);inputs[n]=json.loads(b,object_pairs_hook=unique)
 before=json.loads(inputs['inventory_transport.json']['stdout'],object_pairs_hook=unique);after=json.loads(inputs['inventory_after_transport.json']['stdout'],object_pairs_hook=unique);receipt=inputs['remote-custody.json'];disp=inputs['disposition.json'];seal=inputs['custody-inventory.json']
 ck('exact_full_saved_map_association',before['files']==after['files']==receipt['full_inventory']['files']==disp['remote_files'])
 ck('unique_directory_rosters',all(type(v['directories']) is list and len(v['directories'])==len(set(v['directories'])) for v in [before,after,receipt['full_inventory']]) and sorted(before['directories'])==sorted(after['directories'])==sorted(receipt['full_inventory']['directories']))
 ck('known_corpus_size',len(before['files'])==99039 and sum(v['bytes'] for v in before['files'].values())==2507157901 and len(before['directories'])==3782)
 ck('metadata_members_in_both_capsule_seals',seal['identical'] is True and len(seal['copies'])==2 and all(all(copy['files'][n]=={k:info[k] for k in ('bytes','sha256')} for n,info in prep['input_pins'].items() if n!='custody-inventory.json') for copy in seal['copies'].values()))
 ck('receipt_disposition_join',disp['remote_custody_receipt_sha256']==prep['input_pins']['remote-custody.json']['sha256'] and disp['errors']==disp['input_pin_mismatches']==[] and disp['job_id']=='35516125')
 # Recompute only saved progress-bin associations; no filesystem walk of original outputs.
 group_defs={
 'cpu_output_excluding_campaign':lambda n:n.startswith('cpu/cpu/') and not n.startswith('cpu/cpu/worker/artifacts/campaign/'),
 'gpu_output':lambda n:n.startswith('launch/gpu/'),
 'campaign':lambda n:n.startswith('cpu/cpu/worker/artifacts/campaign/'),
 'cpu_inputs':lambda n:n.startswith('cpu/inputs/'),
 'gpu_inputs':lambda n:n.startswith('control/inputs/'),
 'attempt_hashes':lambda n:bool(re.match(r'cpu/cpu/worker/artifacts/campaign/policy-0[0-7]/',n)) and not re.fullmatch(r'cpu/cpu/worker/artifacts/campaign/policy-0[0-7]/attempt(?:\.candidate)?\.json',n),
 'arm_files':lambda n:bool(re.match(r'cpu/cpu/worker/artifacts/campaign/policy-0[0-7]/world-[0-9]{2}/',n)),
 'sealed_input_rereads':lambda n:n.startswith('cpu/cpu/worker/artifacts/campaign/') and '/sealed-inputs/' in n,
 'arm_receipts':lambda n:bool(re.fullmatch(r'cpu/cpu/worker/artifacts/campaign/policy-0[0-7]/world-[0-9]{2}/arm-receipt\.json',n))}
 groups={}
 for role,name,obs in [('before','inventory_transport.json',before),('after','inventory_after_transport.json',after)]:
  t=inputs[name];saved=paired['observations'][role];ck(role+'_completed_observation',t['returncode']==0 and obs['terminal_gate'] is True and not obs.get('observation_error') and obs['upstream_errors']=={})
  ck(role+'_captured_stderr',t['stderr']==(S/'attempt01/derived'/(role+'.stderr')).read_text())
  tree=ast.parse(t['python_source']);loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(x,'id',None) for x in n.target.elts]==['label','tree','present']];assert len(loops)==1
  labels=[ast.literal_eval(x.elts[0]) for x in loops[0].iter.elts];ck(role+'_captured_traversal_order',labels==['control','launch','cpu','training','token','failed-token01','common','failed01'] and sha(t['python_source'].encode())==saved['captured_source_sha256'])
  paths=[n for label in labels for n in sorted([n for n in obs['files'] if n.startswith(label+'/')],key=lambda s:tuple(s.split('/')))];ordinals={p:i+1 for i,p in enumerate(paths)}
  samples=[]
  for line in t['stderr'].splitlines():
   if not line.startswith('{'):continue
   v=json.loads(line)
   if v.get('phase')=='terminal_inventory_verification':samples.append((v['files_read'],v['elapsed_seconds']))
  ck(role+'_all_progress_samples',list(map(list,samples))==saved['progress_samples'] and all(n==(i+1)*1000 and (i==0 or sec>=samples[i-1][1]) for i,(n,sec) in enumerate(samples)))
  elapsed=[0.0]+[v for _,v in samples];groups[role]={}
  for g,predicate in group_defs.items():
   selected=[n for n in paths if predicate(n)];indexes=[ordinals[n] for n in selected];bins=sorted({(i-1)//1000 for i in indexes});assert max(indexes)<=samples[-1][0]
   seconds=sum(elapsed[i+1]-elapsed[i] for i in bins);v=saved['groups'][g];groups[role][g]=dict(files=len(selected),bytes=sum(obs['files'][n]['bytes'] for n in selected),seconds=seconds)
   ck(role+'_progress_group:'+g,len(selected)==v['files'] and groups[role][g]['bytes']==v['bytes'] and min(indexes)==v['first_counter'] and max(indexes)==v['last_counter'] and close(seconds,v['intersecting_sample_seconds']) and [[i*1000,(i+1)*1000] for i in bins]==v['intersecting_counter_intervals'])
 slower={g:max(groups[role][g]['seconds'] for role in groups) for g in group_defs};ck('slower_per_group_without_fast_sample_selection',all(close(v,d['slower_group_seconds'][g]) for g,v in slower.items()))
 c=groups['before'];outer_files=3*(c['cpu_output_excluding_campaign']['files']+c['gpu_output']['files'])+4*c['campaign']['files'];outer_bytes=3*(c['cpu_output_excluding_campaign']['bytes']+c['gpu_output']['bytes'])+4*c['campaign']['bytes'];reads=sum(c[g]['files'] for g in ['attempt_hashes','arm_files','sealed_input_rereads','arm_receipts'])+16
 ck('source_multiplicity_arithmetic',outer_files==212954 and outer_bytes==1746508319 and reads==56240 and 192*9+24*10+5==1973)
 # Reuse the accepted historical timing receipt, not its old component tests/workloads.
 historical=R/'results/ace_llm_intervention_scientific_resource_preparation_20261108/preparation01/saved-timing-analysis/attempt03/derived/receipt.json';h=read(historical);ck('accepted_report_timing_reference',sha(h)==old['all_input_pins'][str(historical.relative_to(R))]['sha256'] and json.loads(h)['finalization']==old['accepted_artificial_finalization'])
 phases=[]
 for n in ['phase-0000.json','phase-0101.json']:
  p=R/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu'/n;b=read(p);ck('accepted_staging_phase_pin:'+n,sha(b)==old['all_input_pins'][str(p.relative_to(R))]['sha256']);phases.append(json.loads(b))
 ck('accepted_staging_elapsed',close(phases[1]['elapsed_seconds']-phases[0]['elapsed_seconds'],old['accepted_scientific_cpu_runtime_staging']['elapsed_seconds']))
 scan=sum(slower[g] for g in ['cpu_output_excluding_campaign','gpu_output','campaign']);reread=2*(slower['cpu_inputs']+slower['gpu_inputs']);prep_payload=2*scan+reread;worker_io=sum(slower[g] for g in ['attempt_hashes','arm_files','sealed_input_rereads','arm_receipts','campaign']);report=old['accepted_artificial_finalization'];lower=report['report_through_terminal_utc_seconds']*4800/report['curve_points'];higher=report['report_through_terminal_utc_seconds']*38400/report['point_query_predictions']
 ck('io_and_report_components',close(scan,d['per_authentication_payload_proxy_seconds']) and close(reread,d['original_input_reread_proxy_seconds']) and close(prep_payload,d['preparation_payload_proxy_seconds']) and close(worker_io,d['worker_existing_io_proxy_seconds']))
 results={}
 for name,mult,component in [('point_count',1,lower),('selected_slower_group_and_point_query',1,higher),('extra_25_percent_io_stress',1.25,higher)]:
  a=prep_payload*mult+150+100+60;w=component+worker_io*mult+100+50+100;p=scan*mult+100+50+100;start=a+90;we=start+w;ce=we+30;end=ce+p;late=1620+w+30+p
  z=d['scenarios'][name];expected=dict(preparation_seconds=a,worker_seconds=w,parent_post_seconds=p,preparation_residual_seconds=1500-a,worker_residual_at_latest_start_seconds=1700-w,parent_post_residual_seconds=840-p,joint_hard_residual_seconds=4190-end,latest_start_joint_parent_end=late,latest_start_joint_hard_residual_seconds=4190-late)
  ck('scenario_arithmetic:'+name,all(close(z[k],v) for k,v in expected.items()) and all(close(z['joint_schedule'][k],v) for k,v in dict(auth_complete=a,report_start=start,worker_end=we,cleanup_end=ce,parent_end=end).items()))
  results[name]=dict(expected,parent_end=end,phase_fit=a<1500 and 90<120 and w<1700 and p<840 and end<4190)
 ck('selected_fits_stress_explicitly_fails',results['selected_slower_group_and_point_query']['phase_fit'] and not results['extra_25_percent_io_stress']['phase_fit'])
 ck('profile_and_pending_authority',d['profile']==dict(cpus=2,memory_mib=7680,gpus=0,wall_seconds=4200,preparation=1500,report_startup=120,report_startup_cap=1620,work=3320,cleanup=3350,hard=4190) and all(d[k] is False for k in ['full_path_timing_supported','resource_admission_accepted','execution_qualified','scientific_qualified']))
 ck('provisional_accounting_arithmetic_only',disp['observed_related_jobs']==48 and disp['observed_submission_journals']==98 and disp['closed_cpu_seconds']==37073 and disp['closed_gpu_slice_seconds']==4458 and 37073+3600+2*4200==49073 and 57600-49073==8527 and d['collector_reported_accounting']['independent_actual_baseline_pending'] is True)
 facts=dict(groups=groups,slower=slower,outer_files=outer_files,outer_bytes=outer_bytes,reporter_additional_reads=reads,report_output_growth_scenario_files=1973,scenarios=results,source_commit=srcpin['source_commit'],source_sha256=srcpin['sha256'],input_pins=prep['input_pins'],preparation_files=len(sm),preparation_bytes=sum(v['bytes'] for v in sm.values()))
 (W/'facts.json').write_bytes(raw(facts));(W/'source-manifest.json').write_bytes(raw(sm))
except BaseException as exc:
 ck('review_inspector_completed',False,repr(exc));(W/'attempt01-traceback.txt').write_text(traceback.format_exc())
finally:
 result=dict(checks=checks,failed_checks=[v for v in checks if not v['passed']],project_execution=False,tests_rerun=0,remote_actions=False,scope='new saved timing association/arithmetic only, not GPU acceptance')
 (W/'verification.json').write_bytes(raw(result));(W/'input-pins.json').write_bytes(raw(observed_pins));print(json.dumps(dict(checks=len(checks),failed_checks=result['failed_checks'],workspace=str(W)),indent=2))
