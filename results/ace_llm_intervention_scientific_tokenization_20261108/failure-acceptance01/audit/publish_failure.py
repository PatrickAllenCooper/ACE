"""Publish completed independent failure audit; metadata serialization/copying only."""
from pathlib import Path
import json,hashlib,shutil,os
HERE=Path(__file__).resolve().parent;REPO=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/failure-acceptance01'
EXTERNAL=BASE/'ACE-LLM-intervention-scientific-tokenization-failure-acceptance-20261108-01'
TERM=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/terminal01'
EXTTERM=BASE/'ACE-LLM-intervention-scientific-tokenization-terminal-20261108-01'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def raw(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(x,m):
 if not x:raise ValueError(m)
def write(p,b):
 with Path(p).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
a=read(HERE/'custody-result.json');boundary=read(HERE/'no-worker-no-work-proof.json');partial=read(HERE/'source-diagnostic-review.json')
need(a['status']=='complete_saved_custody_and_failure_accounting_verified' and boundary['status']=='verified_failed_before_token_worker','completed actual audits required')
review=REPO/'docs/development/guidance/reviews/ace_llm_intervention_scientific_tokenization_failure01_review_2026-11-08.md'
design=REPO/'docs/development/guidance/reviews/ace_llm_intervention_scientific_tokenization_recovery_design_review_2026-11-08.md'
reviewpin=sha(review.read_bytes());designpin=sha(design.read_bytes())
freeze='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
collection='e217199099b81480f0e85c09f0432b60dd5edb2aad1b6b30c8ed86ca736e34e5';previous='ab8efc04b63d1598b5416a1c62f2c346b3fef5ab46842119b3a62ed89c3de62b'
zeros={k:0 for k in ('scientific_worlds_generated','environment_response_attempts','environment_responses','unknown_return_attempts','tokenizer_calls','model_loads','inference_calls','training_updates','policy_updates')}
evidence=dict(a['artifact_pins']);evidence.update({'remote/'+n:h for n,h in boundary['evidence_pins'].items()})
for n in ('pull_terminal.py','stream_custody.py','stream_custody_review.md','stream-transport.json'):
 evidence[n]=sha((TERM/n).read_bytes())
accept=dict(schema='ace-scientific-tokenization-failure-acceptance-v1',status='accepted_failed_before_token_worker',failure_accepted=True,
 tokenization_accepted=False,complete_custody_verified=True,required_findings=0,job_id='35415334',scope='Only failed TOKEN01 custody, no new scientific preparation/work, and actual allocated charges. No recovery authority.',
 packet_sha256=a['packet_sha256'],inner_packet_sha256=partial['inner_packet_sha256'],packet_commit=a['packet_commit'],disposition_sha256=a['artifact_pins']['disposition.json'],
 freeze_sha256=freeze,source_commit=partial['source_commit'],entry_sha256=partial['source_bindings']['prepare_llm_intervention_scientific_training_cpu.py']['sha256'],
 collection_acceptance_sha256=collection,predecessor_baseline_sha256=previous,source_control_flow_and_traceback_verified=True,two_remote_inventories_matched=True,
 original_inputs_and_claims_verified=True,original_collection_unchanged=True,owned_worker_spawned=False,owned_worker_cleanup_applicable=False,
 runtime_staging_started=True,runtime_staging_completed=True,runtime_qualified_for_this_execution=False,node_local_staged_runtime_cleanup_verified=False,
 token_preparation_pipeline_invoked=False,prepared_artifacts_created=0,exposure_proposals_created=0,numerical_plans_created=0,scientific_response_cost_units=0,
 model_weights_loaded=False,explicit_cuda_api_called=False,training_execution_qualified=False,scientific_qualified=False,**zeros,
 allocation=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',requested_cpus=3,allocated_cpus=3,memory_mib=11520,wall_seconds=600,
   gpus=0,state='FAILED',exit_code='1:0',requeue=False,restarts=0,elapsed_seconds=129,allocated_cpu_seconds=387,gpu_slice_seconds=0),
 scheduler_terminal_verified=True,accounting_complete=True,raw_files=a['raw_files'],raw_bytes=a['raw_bytes'],capsule_files=a['capsule_files'],capsule_bytes=a['capsule_bytes'],
 repository_terminal_root=str(TERM),external_terminal_root=str(EXTTERM),evidence_pins=evidence,
 audit_sha256=sha((HERE/'custody-result.json').read_bytes()),boundary_proof_sha256=sha((HERE/'no-worker-no-work-proof.json').read_bytes()),
 phase_timings_sha256=sha((HERE/'phase-timings.json').read_bytes()),review_path=str(review.relative_to(REPO)),review_sha256=reviewpin,
 diagnosis=dict(observed_error='ValueError: original absolute deadlines required',recorded_sigalrm=False,failure_boundary=boundary['failure_boundary'],
   missing_outputs_are_not_sole_evidence=True,exact_guard_elapsed_unmeasured=True,phase_intervals=boundary['phase_intervals'],
   second_controller_lookup_unavailable=True,matching_saved_sacct_resolves_terminal_state=True),
 recovery=dict(automatic_retry_allowed=False,prospective_reviewed_operational_amendment_required=True,new_once_only_attempt_intent_required=True,
   fresh_live_lock_scheduler_journal_site_admission_required=True,registry_must_be_preserved=True,replacement_worlds_allowed=False,tokenization_authorized=False))
acceptbytes=raw(accept);acceptpin=sha(acceptbytes)
account=dict(schema='ace-scientific-screen-accounting-baseline-v1',scope='Exact43terminal/88journal actual baseline including independently accepted failed pre-token-worker attempt35415334; no success or retry authority.',
 terminal_jobs=43,submission_journals=88,related_accounting=a['related_accounting'],journal_pins=a['journal_pins'],closed_cpu_seconds=13373,closed_gpu_slice_seconds=1601,
 preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=16973,outstanding_reservation=0,cpu_cap_seconds=57600,gpu_cap_seconds=7200,
 input_pins=dict(predecessor=previous,acceptance=acceptpin,disposition=a['artifact_pins']['disposition.json'],audit=accept['audit_sha256'],boundary_proof=accept['boundary_proof_sha256'],failure_review=reviewpin,
   terminal_inventory_before=a['artifact_pins']['inventory_transport.json'],terminal_inventory_after=a['artifact_pins']['inventory_after_transport.json'],custody_inventory=a['artifact_pins']['custody-inventory.json']),
 previous_baseline_sha256=previous,failure_acceptance_sha256=acceptpin,collection_acceptance_sha256=collection,
 recovery_namespace_claim_sha256='40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a',namespace_claim_preserved_sha256='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29',
 automatic_retry_allowed=False,fresh_lock_scheduler_journal_readback_required_before_mutation=True,tokenization_accepted=False,whole_cycle_process_cpu_known=False)
accountbytes=raw(account);accountpin=sha(accountbytes)
proof=dict(schema='ace-scientific-tokenization-no-learnedwork-proof-v1',status='verified_failed_before_token_worker',failed_job_id='35415334',
 failure_acceptance_sha256=acceptpin,accounting_sha256=accountpin,packet_sha256=a['packet_sha256'],disposition_sha256=a['artifact_pins']['disposition.json'],
 freeze_sha256=freeze,source_commit=partial['source_commit'],collection_acceptance_sha256=collection,**zeros,
 proof=dict(owned_worker_not_spawned=True,token_preparation_not_invoked=True,positive_source_trace_boundary=True,complete_custody_corroborrates_boundary=True,
   original_collection_unchanged=True,no_missing_output_only_inference=True),
 scope='New work only in35415334. token_preparation_not_invoked means the child prepare_pipeline; parent authentication/staging did execute.',
 original_outer_invoked=True,runtime_staging_completed=True,runtime_execution_qualified=False,node_local_staged_runtime_cleanup_verified=False,
 boundary_proof_sha256=accept['boundary_proof_sha256'],phase_timings_sha256=accept['phase_timings_sha256'],source_pins=boundary['source_pins'],
 source_codepaths=boundary['source_codepaths'],evidence_pins=evidence,observed_error=boundary['observed_error'],recorded_sigalrm=False,
 failure_boundary=boundary['failure_boundary'],review_sha256=reviewpin,automatic_retry_allowed=False,scientific_qualified=False)
proofbytes=raw(proof);proofpin=sha(proofbytes)
consumer={
 'token-failure-acceptance':dict(path=str(LOCAL/'acceptance.json'),sha256=acceptpin),
 'token-no-learnedwork-proof':dict(path=str(LOCAL/'no-learnedwork-proof.json'),sha256=proofpin),
 'token-failure-baseline':dict(path=str(LOCAL/'accounting.json'),sha256=accountpin),
 'token-failure-disposition':dict(path=str(TERM/'disposition.json'),sha256=a['artifact_pins']['disposition.json']),
 'token-failure-inventory':dict(path=str(TERM/'inventory_transport.json'),sha256=a['artifact_pins']['inventory_transport.json']),
 'token-failure-readback':dict(path=str(TERM/'inventory_after_transport.json'),sha256=a['artifact_pins']['inventory_after_transport.json']),
 'token-failure-custody-inventory':dict(path=str(TERM/'custody-inventory.json'),sha256=a['artifact_pins']['custody-inventory.json']),
 'failed-token-packet':dict(path=str(TERM/'remote/launch/packet.json'),sha256=a['packet_sha256']),
 'failed-token-stage':dict(path=str(TERM/'remote/launch/inputs/packet.json'),sha256=partial['inner_packet_sha256']),
 'failed-token-result':dict(path=str(TERM/'remote/launch/submission.result.json'),sha256=a['submission_result_sha256']),
 'token-operational-review':dict(path=str(design),sha256=designpin),
 'token-operational-guide':dict(path=str(REPO/'docs/development/guidance/ace_llm_intervention_scientific_tokenization_recovery_2026-11-08.md'),sha256='89f296f5d778b269e97338078feafe4333b9f61ded5d6870853e3633dd8d5d54'),
 'token-diagnosis':dict(path=str(LOCAL/'audit/diagnosis.json'),sha256='d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63')}
need(all(type(v[k]) is int and v[k]==0 for v in (accept,proof) for k in zeros),'exact consumer integer zeros')
need(proof['accounting_sha256']==sha(accountbytes) and account['input_pins']['acceptance']==sha(acceptbytes),'acyclic consumer bindings')
need(not any(os.path.lexists(p) for p in (LOCAL,EXTERNAL)),'new exclusive acceptance roots required')
# Retain all original audit attempts and corrections; do not copy the full unchanged terminal capsule again.
records={}
for x in sorted(HERE.rglob('*')):
 if x.is_file():records[x.relative_to(HERE).as_posix()]=dict(bytes=x.stat().st_size,sha256=sha(x.read_bytes()))
auditinventory=raw(dict(files=records,scope='All independent auditor files including failed attempts; original terminal custody remains at separately pinned roots.'))
for root in (EXTERNAL,LOCAL):
 root.mkdir(parents=True,mode=0o700,exist_ok=False)
 shutil.copytree(HERE,root/'audit',copy_function=shutil.copyfile)
 for n,b in {'acceptance.json':acceptbytes,'accounting.json':accountbytes,'no-learnedwork-proof.json':proofbytes,
    'consumer-arguments.json':raw(consumer),'audit-inventory.json':auditinventory,'review.md':review.read_bytes(),'recovery-design-review.md':design.read_bytes()}.items():write(root/n,b)

def inventory(root):
 return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(root.rglob('*')) if p.is_file()}
left=inventory(LOCAL);right=inventory(EXTERNAL);need(left==right,'independent publication copies differ')
for n in left:need(not os.path.samestat((LOCAL/n).stat(),(EXTERNAL/n).stat()),'publication files alias')
inv=raw(dict(files=left,excludes='inventory.json itself',identical_copies=[str(LOCAL),str(EXTERNAL)]))
for root in (EXTERNAL,LOCAL):write(root/'inventory.json',inv)
# Verify requested consumer paths after exclusive publication; no callable consumer invocation.
for name,v in consumer.items():need(sha(Path(v['path']).read_bytes())==v['sha256'],'published consumer path/pin differs '+name)
summary=dict(repository_root=str(LOCAL),external_root=str(EXTERNAL),acceptance_sha256=acceptpin,accounting_sha256=accountpin,
 no_learnedwork_proof_sha256=proofpin,review_path=str(review),review_sha256=reviewpin,operational_design_review_path=str(design),operational_design_review_sha256=designpin,
 guide_sha256=consumer['token-operational-guide']['sha256'],inventory_sha256=sha(inv),consumer_arguments_sha256=sha(raw(consumer)),
 required_findings=0,terminal_jobs=43,submission_journals=88,closed_cpu_seconds=13373,closed_gpu_slice_seconds=1601,cpu_admission_seconds=16973,
 operational_amendment_not_created=True,scientific_qualified=False)
print(raw(summary).decode())
