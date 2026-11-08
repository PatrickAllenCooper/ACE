"""Saved failure/phase metadata and exact source-order proof; never calls production code."""
from pathlib import Path
import json,hashlib,ast,math,datetime,time
HERE=Path(__file__).resolve().parent
TERM=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/terminal01')
RAW=TERM/'remote/launch';OUT=RAW/'preparation'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def main():
 start=time.monotonic();cpu=time.process_time();custody=load(HERE/'custody-result.json')
 need(custody['status']=='complete_saved_custody_and_failure_accounting_verified','complete independent custody first')
 partial=load(HERE/'source-diagnostic-review.json');d=load(TERM/'disposition.json');files=d['remote_files']
 for n,b in partial['source_bindings'].items():
  value=(RAW/'inputs'/n).read_bytes();need(sha(value)==b['sha256'] and value==(HERE/'source'/n).read_bytes(),'original source mismatch')
 entry='prepare_llm_intervention_scientific_training_cpu.py';owned='llm_intervention_cpu_owned_supervision.py';diagnostic='qualify_llm_intervention_runtime_metadata.py';archive='llm_intervention_runtime_archive.py'
 raw=(HERE/'metadata/inner-packet.json').read_bytes();p=load(HERE/'metadata/inner-packet.json')
 need((OUT/'packet.json').read_bytes()==raw and sha(raw)==partial['inner_packet_sha256'],'actual parent packet differs')
 s=load(OUT/'start.json');need(s['deadlines']==p['deadlines'] and s['resources']==p['resources'] and s['packet_sha256']==sha(raw),'actual start contract differs')
 f=load(OUT/'failure.json');need(f['type']=='ValueError' and f['message']=='original absolute deadlines required','observed failure type changed')
 need(sha((OUT/'failure.json').read_bytes())=='acc3c88781d333719154cf2f774bb6633e9cc99b87e62bf4ec54a74c529ae03a','original bounded diagnostic differs')
 need(sha((RAW/'logs/35415334.err').read_bytes())=='6dc0407e0ab1db4e0b148b585a4a63d9ef378cfcd7bcad4c74fc57f7bd0700a1','original stderr differs')
 need('line 536, in parent' in f['traceback'] and 'line 121, in supervise' in f['traceback'] and 'line 26, in require' in f['traceback'],'positive stack boundary differs')
 source=(RAW/'inputs'/entry).read_text();ot=(RAW/'inputs'/owned).read_text();diag=(RAW/'inputs'/diagnostic).read_text();ar=(RAW/'inputs'/archive).read_text()
 a=ast.parse(source);o=ast.parse(ot)
 defs={x.name:x for x in a.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))};supervise=next(x for x in o.body if isinstance(x,ast.FunctionDef) and x.name=='supervise')
 guard=next(x for x in supervise.body if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and getattr(x.value.func,'id',None)=='require')
 popen=[x for x in ast.walk(supervise) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='Popen'];need(len(popen)==1 and guard.lineno==121 and popen[0].lineno==139,'first guard/Popen source ordering differs')
 parent_calls=[x for x in ast.walk(defs['parent']) if isinstance(x,ast.Call)]
 owncall=next(x for x in parent_calls if isinstance(x.func,ast.Attribute) and x.func.attr=='supervise');need(owncall.lineno==536,'parent callsite differs')
 need(ast.get_source_segment(source,next(k.value for k in owncall.keywords if k.arg=='started'))=='args.started','original clock reset')
 for kw,key in [('startup_deadline','startup'),('work_deadline','work')]:
  expr=ast.get_source_segment(source,next(k.value for k in owncall.keywords if k.arg==kw))
  need(expr=="args.started+p['deadlines']['"+key+"']",'deadline origin differs')
 # New execution is ruled out by positive first-guard stack plus complete absence of any post-spawn artifacts.
 prefix='launch/preparation/';actual=[n[len(prefix):] for n in files if n.startswith(prefix)]
 expected={'packet.json','allocation.json','start.json','failure.json','stage.json','independent-provenance-sources.json'}|{f'phase-{i:04d}.json' for i in range(102)}
 need(set(actual)==expected,'unexpected missing/additional parent output')
 dirs={n[len(prefix):] for n in d['remote_directories'] if n.startswith(prefix)};need(dirs=={'session'},'worker/session directory exists')
 need(not list((OUT/'session').iterdir()),'worker session not empty')
 need(not any(n.startswith('launch/preparation/session/') for n in files),'worker spawned evidence exists')
 # Exact six public tokenizer input bytes; neither deserialize nor invoke a tokenizer.
 tokens={n.split('/',2)[2]:r['sha256'] for n,r in files.items() if n.startswith('launch/tokenizer/')}
 need(tokens==p['tokenizer_files'] and len(tokens)==6,'public tokenizer input membership differs')
 stage=load(OUT/'stage.json')
 need(stage['archive_sha256']==p['archive']['sha256'] and stage['manifest_sha256']==p['manifest']['sha256']
      and stage['objects']==49966 and stage['file_bytes']==8397575073 and stage['runtime_qualified'] is stage['model_loaded'] is False,'staging receipt differs')
 need(stage['staged_prefix']=='/scratch/local/35415334/ace-scientific-preparation-35415334/runtime','staged runtime scope differs')
 provenance=load(OUT/'independent-provenance-sources.json')
 need(sha(provenance['expected_source'].encode())==provenance['expected_source_sha256'],'pure provenance source bytes differ')
 # Original phase receipt sequence; no reread/extraction of staged runtime.
 phases=[load(OUT/f'phase-{i:04d}.json') for i in range(102)]
 need(phases[0]['phase']=='stage_start' and phases[-1]['phase']=='stage_complete','phase endpoints')
 elapsed=[x['elapsed_seconds'] for x in phases];pcpu=[x['process_cpu_seconds'] for x in phases]
 need(all(type(x) in (int,float) and math.isfinite(x) for x in elapsed+pcpu),'nonfinite timings')
 need(elapsed==sorted(elapsed) and pcpu==sorted(pcpu),'nonmonotonic phase receipts')
 need([x['detail']['completed_objects'] for x in phases[1:-1]]==list(range(1,49966,500)) and all(x['phase']=='stage_progress' and x['detail']['objects']==49966 for x in phases[1:-1]),'exact100 staging progress receipts differ')
 need(all(x['peak_self_rss_bytes']<=512*1024**2 for x in phases),'parent diagnostic RSS cap exceeded')
 times=[datetime.datetime.fromisoformat(x['utc']) for x in phases];need(times==sorted(times),'nonmonotonic UTC phase receipts')
 for i in range(1,len(phases)):need(abs((times[i]-times[0]).total_seconds()-(elapsed[i]-elapsed[0]))<.01,'wall/monotonic phase interval mismatch')
 intervals=dict(pre_stage_inclusive_seconds=elapsed[0],stage_start_to_first_object_seconds=elapsed[1]-elapsed[0],first_object_to_stage_complete_seconds=elapsed[-1]-elapsed[1],
   total_staging_seconds=elapsed[-1]-elapsed[0],stage_complete_elapsed_seconds=elapsed[-1],remaining_before_original_startup_at_stage_complete_seconds=120-elapsed[-1],
   stage_start_process_cpu_seconds=pcpu[0],stage_complete_process_cpu_seconds=pcpu[-1],staging_process_cpu_seconds=pcpu[-1]-pcpu[0],
   parent_max_phase_rss_bytes=max(x['peak_self_rss_bytes'] for x in phases),
   allocation_elapsed_seconds=129,allocation_cpu_seconds=387,scheduler_total_cpu_seconds=31.342,scheduler_batch_maxrss_kib=8819932,
   scheduler_batch_maxrss_mib=8819932/1024)
 need(elapsed[-1]<120 and 120-elapsed[-1]<.05,'expected near-startup completed stage differs')
 proof=dict(status='verified_failed_before_token_worker',scope='Only new work in failed job35415334; prior accepted collection work remains preserved and is not zeroed.',
   job_id='35415334',complete_custody_verified=True,source_control_flow_and_traceback_verified=True,missing_outputs_are_not_sole_evidence=True,
   observed_error=dict(type=f['type'],message=f['message']),recorded_sigalrm=False,
   failure_boundary='First shared owned-supervision absolute-deadline predicate, after stage/provenance publication and session mkdir, before launched flag/log-open/Popen.',
   exact_elapsed_at_guard_recorded=False,startup_expired_at_guard_source_deduction=True,
   owned_worker_spawned=False,owned_worker_cleanup_applicable=False,node_local_staged_runtime_deletion_verified=False,
   runtime_staging_completed=True,runtime_qualified_for_this_execution=False,tokenizer_instantiations=0,tokenization_calls=0,
   prepared_artifacts_created=0,exposure_proposals_created=0,numerical_plans_created=0,training_updates=0,model_weights_loaded=False,
   inference_calls=0,environment_response_attempts=0,environment_responses=0,unknown_return_attempts=0,scientific_response_cost_units=0,scientific_worlds_generated=0,
   original_collection_unchanged=True,collection_acceptance_sha256='e217199099b81480f0e85c09f0432b60dd5edb2aad1b6b30c8ed86ca736e34e5',
   source_commit=partial['source_commit'],source_pins={n:b['sha256'] for n,b in partial['source_bindings'].items()},
   source_codepaths={entry:{'authenticate':392,'parent':510,'supervise_call':536,'main':549,'child':438,'work_start_before_tokenizer':462,'tokenizer_factory_call':468},
                    owned:{'initial_guard':121,'logs_open':137,'launched_flag':138,'Popen':139,'unlaunched_confirm_closed':41},
                    diagnostic:{'stage_runtime':182,'provenance_contract':343,'events':535},archive:{'stage':202,'full_archive_hash_before_extract':211,'progress_callback':241}},
   evidence_pins={n:files[n]['sha256'] for n in files if n.startswith('launch/preparation/') or n in ('launch/logs/35415334.err','launch/submission.intent.json','launch/submission.result.json')},
   disposition_sha256=sha((TERM/'disposition.json').read_bytes()),custody_inventory_sha256=sha((TERM/'custody-inventory.json').read_bytes()),
   custody_audit_sha256=sha((HERE/'custody-result.json').read_bytes()),phase_intervals=intervals,
   diagnosis_limits=['Pre-stage51.55s includes input/collection byte authentication, allocation query and setup; no per-operation timing.',
     'First-object gap includes manifest validation/full archive hashing/first member work; no isolated hash or I/O timing.',
     'Stage-complete timestamp precedes its fsync and provenance capture. Only28.4ms remained; no exact post-stage or guard elapsed receipt.',
     'Do not infer contention, idle time or node-runtime deletion from these observations. Slurm and parent RSS scopes differ.'],
   success_or_recovery_authorization=False,scientific_qualified=False)
 (HERE/'no-worker-no-work-proof.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
 (HERE/'phase-timings.json').write_text(json.dumps(dict(phases=phases,intervals=intervals,source='102 original saved receipts; no execution'),sort_keys=True,indent=2)+'\n')
 print(json.dumps(dict(proof_sha256=sha((HERE/'no-worker-no-work-proof.json').read_bytes()),phase_timings_sha256=sha((HERE/'phase-timings.json').read_bytes()),intervals=intervals,local_audit_wall_seconds=time.monotonic()-start,local_audit_process_cpu_seconds=time.process_time()-cpu)))
if __name__=='__main__':main()
