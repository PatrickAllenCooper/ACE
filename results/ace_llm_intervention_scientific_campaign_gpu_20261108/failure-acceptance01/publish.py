"""Publish only new independent review artifacts after local metadata verification."""
import copy
import datetime
import json
import os
from pathlib import Path
import time
import audit as a

start=time.monotonic(); cpu_start=time.process_time()
checks=a.decode((a.WORK/'checks.json').read_bytes())
source=a.decode((a.WORK/'source-boundary.json').read_bytes())
a.PINS.update(a.decode((a.WORK/'source-pins.json').read_bytes()))
assert checks['required_findings']==source['required_findings']==0
assert all(c['passed'] for c in checks['checks']+source['checks'])
disp=a.j(a.TERM/'disposition.json'); receipt=a.j(a.TERM/'remote-custody.json')
base=a.j(a.LAUNCH/'cpu-baseline.json'); original_acceptance=a.j(a.LAUNCH/'cpu-acceptance.json')
obs=a.j(a.ROOT/'failure-observation01/stdout.json'); intent=a.j(a.LAUNCH/'submission.intent.json')
collector_intent=a.j(a.TERM/'collector-intent.json')
expected_before=dict(a.PINS)

def p(path): return dict(path=str(path),**a.PINS[str(path)])
def exclusive(parent,stem,suffix=''):
    for i in range(1,100):
        candidate=parent/(stem+f'{i:02d}'+suffix)
        try: candidate.mkdir(mode=0o700); return candidate
        except FileExistsError: continue
    raise ValueError('exclusive destination unavailable')

# Check progress/receipt bindings without re-running the source or contacting CURC.
progress=[a.decode(line) for line in a.read(a.TERM/'backup-operations/stderr.bin').splitlines() if line.startswith(b'{')]
completed=[v for v in progress if v.get('state')=='complete']
archive_verifications=[v for v in completed if v['phase']=='archive_verify']
assert len(archive_verifications)>=2 and all(v['files']==72557 and v['members']==76094 and v['bytes']==1366843790 for v in archive_verifications)
assert all(any(v['phase']==phase for v in completed) for phase in ('archive_build','archive_copy','before_scan','after_archive_scan','after_project_scan'))

# Before writing new artifacts, every original byte pin must still match.
for path,meta in list(a.PINS.items()): assert a.read(path,False)==meta
original_stats={n:a.ident(Path(n).lstat()) for n in a.PINS}
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
local=exclusive(a.ROOT,'failure-acceptance')
external=exclusive(a.BASE,'ACE-LLM-intervention-scientific-campaign-gpu-failure-acceptance-20261109-')
locations=dict(repository=str(local),external=str(external),originals_untouched=True,exclusive_created=True)
files={}
def put(name,value):
    data=value if isinstance(value,bytes) else a.canon(value)
    for root in (local,external):
        with (root/name).open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    files[name]=dict(bytes=len(data),sha256=a.sha(data))
    return files[name]['sha256']

put('locations.json',locations)
put('audit.py',(a.WORK/'audit.py').read_bytes())
put('source_review.py',(a.WORK/'source_review.py').read_bytes())
put('publish.py',Path(__file__).read_bytes())
pinsha=put('source-pins.json',dict(schema='ace-independent-review-source-pins-v1',algorithm='sha256',
    scope='Every local evidence/source file read or hashed for this review. Git object byte pins are in source-boundary.json. Scientific raw payloads were opaque bytes, not decoded.',files=a.PINS))
checksha=put('checks.json',checks)
sourcesha=put('source-boundary.json',source)

custody=dict(schema='ace-gpu01-independent-failure-custody-review-v1',status='verified_saved_full_backup_and_two_selected_local_copies',required_findings=0,
    custody_mode='curc_projects_full_backup_with_compact_local_evidence',complete_local_raw_copy=False,
    terminal_failure_seal=p(a.TERM/'custody-failure-inventory.json'),original_disposition=p(a.TERM/'disposition.json'),
    before=p(a.TERM/'inventory_transport.json'),after=p(a.TERM/'inventory_after_transport.json'),backup_receipt=p(a.TERM/'remote-custody.json'),
    full_original_files=72557,full_original_bytes=1366843790,full_directories=3537,
    selected_local_files_per_copy=182,selected_local_bytes_per_copy=285130684,omitted_local_files=72375,omitted_local_bytes=1081713106,
    entire_sealed_capsule_files_per_copy=311,all_sealed_members_hashed_in_both_copies=True,all_selected_local_members_hashed_in_both_copies=True,
    exact_directory_membership_verified=True,copy_paths=[str(a.TERM),str(a.EXT)],
    full_inventory_sha256=checks['full_inventory_sha256'],original_before_after_and_project_readback_inventory_digests_equal=True,
    backup_archive=dict(path=receipt['backup_archive_path'],sha256=receipt['backup_archive_sha256'],bytes=receipt['backup_archive_bytes'],kind=receipt['backup_kind']),
    remote_backup_verification_basis='Pinned saved successful transport, helper source, framed request, receipt, full member maps and completed verification telemetry. No new remote read was performed.',
    backup_completed_progress=completed,backup_roots=receipt['roots'],original_hardlink_groups=132,hardlink_alias_metadata_unchanged=True,
    namespace_pins={n:receipt['full_inventory']['files'][n] for n in ('namespace.json','original-namespace.json')},
    both_actual_local_namespaces_match_saved_remote=True,all_original_coordinator_files_match_saved_references=True,
    full_predecessor_files_unchanged=72374,all_predecessor_directories_unchanged=True,
    predecessor_label_mapping={'launch':'cpu','cpu':'token','common':'common','training':'training','failed-token01':'failed-token01','failed01':'failed01','namespace.json':'namespace.json','original-namespace.json':'original-namespace.json'},
    label_mapping_basis='Exact absolute root equality between predecessor and current backup receipts; no membership or digest is dropped.',
    checks_sha256=checksha,source_pins_sha256=pinsha)
custodysha=put('custody-review.json',custody)

resource=dict(schema='ace-gpu01-independent-resource-reconciliation-v1',status='resource_disagreement_reconciled_from_saved_evidence',required_findings=0,
    job_id='35498218',resource_contract_matched=True,original_disposition=p(a.TERM/'disposition.json'),
    successful_scontrol_source=p(a.ROOT/'failure-observation01/stdout.json'),saved_scontrol_fields=checks['resource_fields'],
    later_scontrol_status='Both terminal observations returned Invalid job id specified; no fresh scontrol call performed.',
    original_resource_disagreement=disp['errors'][0],
    resolution='Earlier saved successful scontrol supplies exact job/resource/no-requeue/lifecycle evidence. The immutable source failed before out.mkdir and its own gpu/allocation.json. The collector only considers that missing file or its later unavailable scontrol; this independent review uses the earlier pinned observation and matching saved detailed sacct rows.',
    original_collector_returncode=1,retroactive_collector_success=False,original_disposition_changed=False,
    requested=intent['resources'],requested_tres=checks['requested_tres'],allocated_tres=checks['allocated_tres'],parent_accounting=checks['parent_accounting'],
    saved_job_step_rows=[s.split('|') for s in obs['commands']['accounting']['stdout'].splitlines()],
    accounting_rule='Charge the parent allocation exactly once; batch and extern are corroborating step rows, not additional charges.',
    observed=dict(allocated_cpus=4,gpu_slices=1,gpu_type='a100_3g.20gb',host_memory_mib=15360,elapsed_seconds=245,
        allocated_cpu_seconds=980,gpu_slice_seconds=245,state='FAILED',exit_code='1:0',requeue=0,restarts=0,
        process_total_cpu_seconds=6.755,batch_total_cpu_seconds=6.753,extern_total_cpu_seconds=0.001,batch_maxrss_saved='133176K'),
    memory_scope='15GiB is host allocation memory. a100_3g.20gb is the scheduler MIG type; device capacity was not measured.',
    lifecycle=dict(submit=checks['resource_fields']['SubmitTime'],start=checks['resource_fields']['StartTime'],end=checks['resource_fields']['EndTime'],
        timezone_note='Exact scheduler-displayed times; the saved text does not encode a UTC offset.',startup_alarm_seconds=240,
        cpu_authentication_timestamp='No separate phase timestamp; traceback places CPU authentication before output creation.',model_loading='not reached on recorded source path',gpu_work='not reached on recorded source path',final_output='not created'),
    no_requeue_verified=True,exclusive_submission_intent_verified=True,whole_node_exclusivity_claimed=False,
    exact_prior_46_rows_and_94_journals_unchanged=True,current_terminal_jobs=47,current_submission_journals=96,
    predecessor_cpu_seconds=29261,predecessor_gpu_slice_seconds=2505,closed_cpu_seconds=30241,closed_gpu_slice_seconds=2750,
    preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=33841,outstanding_reservation=0,
    local_collector_process_seconds=checks['original_collector_process_cpu_seconds'],local_collector_wall_seconds=checks['original_collector_wall_seconds'],
    collector_command_wrapper_metrics=checks['original_collector_result'],
    metrics_scope='Process CPU, elapsed time and allocated resource charges remain separate. Local review/collector process metrics are not additional Slurm jobs and do not duplicate the single preparation allowance.',
    gpu_utilization_percent=None,gpu_usage_qualification=False,runtime_qualified=False,scientific_qualified=False,retry_allowed=False,
    requested_versus_observed_note='Actual allocation matched the request. The recorded attempt ended during CPU authentication before staging or model work; no GPU telemetry or application compute progress was reached. This accepts failure accounting, not runtime capacity or performance.',
    corrective_action_not_authorized='A successor design must move bulk CPU custody authentication out of GPU allocation and retain exact input/source bindings and deadline semantics, including the second authentication after handoff. This review changes no source and authorizes no retry.',
    source_boundary_sha256=sourcesha,checks_sha256=checksha)
resourcesha=put('resource-reconciliation.json',resource)

proof=dict(schema='ace-scientific-campaign-gpu-no-policy-work-proof-v1',status='verified_no_new_policy_work',required_findings=0,job_id='35498218',
    proof_scope='Recorded failed invocation plus complete before/after saved inventories of all seven named roots and both namespace registries. No universal observation of unrelated paths or node-local state is claimed.',
    before=p(a.TERM/'inventory_transport.json'),after=p(a.TERM/'inventory_after_transport.json'),original_cpu_acceptance=p(a.LAUNCH/'cpu-acceptance.json'),
    immutable_outer_packet=p(a.LAUNCH/'packet.json'),source_boundary_sha256=sourcesha,custody_review_sha256=custodysha,checks_sha256=checksha,
    original_input_pins_verified=72372,all_72374_predecessor_files_and_directories_exact=True,
    accepted_cpu_noncampaign_files=20564,accepted_cpu_campaign_files=15030,cpu_completed_cells=120,declared_cells=192,
    cpu_attempt_pins=original_acceptance['attempt_pins'],cpu_terminal_sha256=original_acceptance['phase_terminal_sha256'],
    cpu_bundle_sha256=original_acceptance['bundle_sha256'],cpu_bundle_payload_decoded=False,
    forbidden_paths=['launch/gpu','**/policy-05','**/policy-06','**/policy-07'],forbidden_file_or_directory_matches=[],
    original_cpu_and_campaign_file_additions=0,original_cpu_and_campaign_file_deletions=0,original_cpu_and_campaign_digest_changes=0,
    all_other_predecessor_root_additions_deletions_and_digest_changes=0,
    entire_current_launch_file_roster='178 immutable input/packet members, unchanged readiness.json, two submission journals, two original log files: 183 files; only inputs/logs directories.',
    new_policy_attempts=0,new_scientific_responses=0,new_world_realizations=0,new_training_updates=0,
    source_boundary='main:677 -> authenticate:346 -> admit_cpu:198 -> inventory:151 -> read:98; signal-14 handler at673. First CPU inventory fails before out.mkdir681, allocation444, stage/controller/service/model/response work.',
    absence_is_not_sole_proof=True,opaque_byte_reads_may_have_occurred=True,model_or_adapter_tensors_decoded=False,scored_outcomes_decoded=False,
    no_node_local_cleanup_claim=True,gpu_utilization_percent=None,runtime_qualified=False,scientific_qualified=False,retry_allowed=False)
proofsha=put('no-policy-work-proof.json',proof)

acceptance=dict(schema='ace-scientific-campaign-gpu-failure-acceptance-v1',status='accepted_failed_prepolicy_attempt',required_findings=0,findings=[],
    acceptance=True,acceptance_scope='Independent acceptance of this terminal failed attempt, saved full CURC backup with two complete selected-local evidence copies, unchanged predecessor/input custody, no-new-policy-work boundary, and actual accounting only.',
    job_id='35498218',scheduler_state='FAILED',scheduler_exit_code='1:0',original_collector_returncode=1,retroactive_collector_success=False,
    failure_reason='SIGALRM signal14 during first CPU-output byte inventory, before GPU output creation or staging; individual file and latency cause are not identified.',
    input_pins=dict(original_disposition=p(a.TERM/'disposition.json'),original_failure_seal=p(a.TERM/'custody-failure-inventory.json'),
        predecessor_baseline=p(a.LAUNCH/'cpu-baseline.json'),predecessor_acceptance=p(a.LAUNCH/'cpu-acceptance.json'),
        outer_packet=p(a.LAUNCH/'packet.json'),submitted_entry=p(a.LAUNCH/'inputs/qualify_llm_intervention_scientific_campaign_gpu.py'),
        original_submission_result=p(a.LAUNCH/'submission.result.json'),failure_observation=p(a.ROOT/'failure-observation01/stdout.json'),
        preliminary_review=p(a.ROOT/'preliminary-failure-review01/review.md'),remote_custody=p(a.TERM/'remote-custody.json')),
    packet_commit=intent['packet_commit'],source_commit=intent['source_commit'],
    no_policy_work_proof_sha256=proofsha,resource_reconciliation_sha256=resourcesha,custody_review_sha256=custodysha,
    source_boundary_sha256=sourcesha,source_pins_sha256=pinsha,checks_sha256=checksha,
    complete_CURC_backup_verified_from_saved_records=True,complete_local_raw_copy=False,all_selected_local_files_verified_in_both_copies=True,
    related_terminal_jobs=47,submission_journals=96,closed_cpu_seconds=30241,closed_gpu_slice_seconds=2750,
    preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=33841,outstanding_reservation=0,
    new_policy_attempts=0,new_scientific_responses=0,new_world_realizations=0,new_training_updates=0,
    runtime_qualified=False,training_execution_qualified=False,scientific_qualified=False,gpu_usage_qualified=False,retry_allowed=False,
    new_gpu_testing_authorized=False,source_changes_authorized=False,originals_changed=False,remote_actions=0,original_source_executions=0,tests_run=0,
    review_method='Independent rederivation from pinned local bytes, complete metadata-map joins, actual saved scheduler rows, and static submitted-source/Git verification. No subagent results or parent acceptance conclusions substituted for evidence.',
    utc=stamp,locations=locations)
acceptsha=put('failure-acceptance.json',acceptance)
newbase=copy.deepcopy(base)
newbase.update(closed_cpu_seconds=30241,closed_gpu_slice_seconds=2750,cpu_admission_seconds=33841,
    terminal_jobs=47,submission_journals=96,related_accounting=checks['related_accounting'],journal_pins=checks['journal_pins'],outstanding_reservation=0,
    input_pins=dict(acceptance=acceptsha,disposition=a.PINS[str(a.TERM/'disposition.json')]['sha256'],predecessor=a.PINS[str(a.LAUNCH/'cpu-baseline.json')]['sha256']))
assert set(newbase)==set(base) and newbase['schema']==base['schema']
basesha=put('baseline.json',newbase)

review=f'''# Independent GPU01 failure, custody and accounting acceptance

**accepted_failed_prepolicy_attempt — required findings: 0.** Job **35498218** remains `FAILED`, exit `1:0`. The original collector returned **1** at 2026-11-09 22:47:36 UTC. This new review reconciles its sole resource-interface disagreement; it does not rewrite that exit, its disposition or its failure seal. It accepts failure custody/accounting and the pre-policy boundary. Runtime, GPU-use, training-execution and scientific qualification remain false, and no retry or new GPU test is authorized.

## Independent evidence and custody

Both original terminal capsules were independently rehashed against `custody-failure-inventory.json` (`8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0`). Each contains 310 sealed members plus the identical seal itself. Every selected raw file and every directory matched in both copies. Complete before/after transport records (`f9c7dc2e…` / `4303d7d4…`) parse successfully and have exactly equal full file maps, directory membership, namespaces, input pins, hardlink metadata, related jobs and journal pins. The review uses those maps themselves, not the parent's summary flags.

The saved receipt `5224bf51110996d0a72f334dc976db7644f24d99e7500b99730ddf00eff19214` covers **72,557 named files / 1,366,843,790 bytes / 3,537 directories**. Its independently recomputed full inventory digest is `{checks['full_inventory_sha256']}` and agrees with the original-before, original-after and project-readback digests. The source-pinned successful backup transport and complete member-verification telemetry support the recorded CURC USTAR backup at `{receipt['backup_archive_path']}`: **1,422,510,080 archive bytes**, SHA256 `{receipt['backup_archive_sha256']}`. All 132 original hardlink groups are retained as alias metadata, with every named archive member checked.

This is **complete saved CURC backup plus selected local evidence**, not a complete local raw copy. Each local copy contains **182 selected raw files / 285,130,684 bytes**. The explicit omitted partition is 72,375 files / 1,081,713,106 bytes; the selected and omitted maps exactly reconstruct the full inventory. No remote archive read was performed in this review. The previously saved full member verification is its remote-custody basis. No institutional-backup or retention guarantee is inferred.

## Input preservation and no new policy work

Every one of the predecessor's **72,374 files and all directories** survives unchanged. Absolute root identity resolves the predecessor's `launch -> cpu` and `cpu -> token` label changes without dropping or conflating paths. All **72,372 original input pins** match both saved observations and the full inventory. The original accepted CPU outputs match exactly: **20,564 noncampaign files plus 15,030 campaign files**, the five attempt/candidate pins, and the original phase terminal reporting **120 completed CPU cells out of 192 declared**. Both namespace registries and both complete coordinator launch copies also match their saved pins.

There is no `launch/gpu` tree and no `policy-05`, `policy-06` or `policy-07` file or directory anywhere in the complete recorded root maps. All upstream trees are identical to accepted predecessor maps, so there is no new response, policy attempt, world realization, training update or input drift in those trees. The current launch's only 183 files are its 178 immutable input/packet members, readiness record, two submission journals and two logs. Scientific payloads, private scores, model weights and adapter tensors were not decoded. Opaque bytes were hashed only.

The exact entry SHA256 is `2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1`. All 178 packet/input members were joined to packet commit `41bda8e1d9e925eca68f1aa6b16f4d616d1cf7db`, and the entry plus four imported helpers were joined to source commit `a9346022f145cc29439325b6f291597b2e6ec037` with Git replacement objects and lazy fetch disabled. The outer byte SHA `436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7` is distinct from the Git revision. The preliminary review SHA `45932dbb87627c21aed845716aada2d0db323c0fef26d2deb5a39b4802383bc0` and its source pins also match, but the conclusions here were independently rederived.

The original stderr bytes match the earlier failure observation (`cce561b31d38d34e8a2f0e2f626d741bd845decd5c8d31817558993f20a65aa7`) and both terminal copies. The trace is `main:677 -> authenticate:346 -> admit_cpu:198 -> inventory:151 -> read:98`, interrupted by signal14 at line673 after the source's 240-second startup alarm. This is the first CPU output inventory. The later campaign inventory, private bundle decode and training-lineage admission were not reached. Output creation at681, `stage_and_exec`, `allocation.json`, runtime staging, controller/service startup and model loading are downstream and were not reached on this recorded invocation. The four imported helpers have only standard-library top-level imports and constant construction; their main guards are inactive under the named module imports. Opaque adapter-byte reads do not establish tensor deserialization or GPU use.

No absence-only proof is used. Full unchanged root maps, the authenticated source and original failure trace jointly establish the no-new-policy-work boundary. The specific blocked file, latency cause and exact partial-inventory progress remain unknown. This review makes no universal claim about unrelated paths or node-local cleanup.

## Sole resource disagreement resolved

The collector looked for `gpu/allocation.json` or its own current `scontrol` result. The allocation file could not exist because authentication failed before output creation; both later collector `scontrol` calls returned `Invalid job id specified`. The earlier pinned failure observation successfully saved `scontrol` for the exact job, including `Requeue=0`, `Restarts=0`, state/exit, command, log paths and lifecycle. It agrees with both saved detailed `sacct` requests and all parent/step rows: account `ucb736_asc2`, partition `aa100`, QOS `gpu-testing`, **4 CPUs, one `a100_3g.20gb` slice, 15 GiB host memory, 2,880-second wall limit**.

Scheduler-displayed lifecycle: submit `2026-11-09T15:13:11`, start `2026-11-09T15:13:12`, end `2026-11-09T15:17:17`; the saved text has no UTC offset. Start-to-end is exactly **245 seconds**. Exclusive submission intent/result checks and `xb` writes precede the single successful `sbatch`, with no retry loop. This is submission-artifact exclusivity, not whole-node exclusivity.

Charge the parent once: **980 allocated CPU-seconds and 245 GPU-slice-seconds**. Batch/extern rows corroborate the parent and are not charged again. The full saved census is exactly the previous **46 jobs / 94 journals plus this job / two journals = 47 / 96**, all terminal and with no active related ACE job in either saved queue. Every predecessor row and journal pin is retained exactly.

| Accounting | CPU-seconds | GPU-slice-seconds |
|---|---:|---:|
| Pinned predecessor | 29,261 | 2,505 |
| Failed GPU01 parent | 980 | 245 |
| Closed total | **30,241** | **2,750** |
| Single unchanged preparation allowance | 3,600 | — |
| Admission total | **33,841** | **2,750** |

Outstanding reservation is **0**. Caps remain 57,600 CPU-seconds and 7,200 GPU-slice-seconds. No dollar-cost conversion is claimed.

Process and elapsed metrics remain separate: Slurm parent `TotalCPU=00:06.755`; collector process CPU `{checks['original_collector_process_cpu_seconds']}` seconds and collector elapsed `{checks['original_collector_wall_seconds']}` seconds; command wrapper child CPU `131.088909` and elapsed `1650.0954186249999` seconds. These are not alternative allocation charges or extra preparation allowances. Batch RSS is not a model-memory qualification. No GPU utilization percentage or GPU execution qualification is asserted.

The request and actual allocation agree, but the recorded attempt spent its startup path in CPU authentication inside the GPU allocation. A future design needs bulk CPU verification before GPU allocation while preserving exact bindings and deadline semantics, including the second authentication after handoff. This review does not implement or authorize that change, relax guards, modify the packet or locks, rerun the collector, or submit work.

## New independent artifacts

`failure-acceptance.json` SHA256 **{acceptsha}**.

`baseline.json` SHA256 **{basesha}** uses exactly the predecessor's baseline schema and field set. Its input pins bind this new acceptance, the untouched original disposition `545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507`, and predecessor `c21f43194ff61b30c0322acff99c20e71158d980a00a34a1685b87eac7f27125`.

`no-policy-work-proof.json`, `resource-reconciliation.json`, `custody-review.json`, `source-boundary.json`, `checks.json`, all read-source pins, and the metadata-only audit programs preserve the derivations. `originals-unchanged.json` records final readback and `output-inventory.json` seals the two identical new output copies, excluding only itself. New directories are `{local}` and `{external}`. No original source was executed, no models or tests were run, and no remote action or lock operation occurred.
'''
put('review.md',review.encode())

# Final readback: original bytes and metadata unchanged across authoring, and
# original sealed directory membership remains exact. These are hash checks only.
for path,meta in list(a.PINS.items()):
    assert a.read(path,False)==meta and a.ident(Path(path).lstat())==original_stats[path]
seal=a.j(a.TERM/'custody-failure-inventory.json')
for root in (a.TERM,a.EXT): assert a.inventory(root,('custody-failure-inventory.json',))==seal['copies'][str(root)]
for name in ('local-launch','external-launch'):
    ref=a.j(a.TERM/'coordinator'/f'{name}-reference.json');assert a.inventory(ref['path'])==ref['inventory']
assert a.PINS==expected_before
put('originals-unchanged.json',dict(schema='ace-independent-review-originals-unchanged-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_pins_sha256=pinsha,original_files_rehashed=len(a.PINS),every_byte_pin_unchanged=True,
    original_file_identity_size_mode_mtime_ctime_nlink_unchanged_across_authoring=True,
    terminal_both_seals_and_directory_membership_reverified=True,coordinator_both_directory_memberships_reverified=True,
    original_failure_disposition_and_collector_records_untouched=True,remote_actions=0,locks_touched=0,
    original_source_executions=0,models_imported=0,tests_run=0,scored_outcomes_decoded=False,
    author_process_cpu_seconds=time.process_time()-cpu_start,author_elapsed_seconds=time.monotonic()-start,
    metrics_scope='This local metadata publication process only; not a Slurm allocation or an additional preparation allowance.'))

def output_view(root):
    return dict(files={p.name:dict(bytes=p.stat().st_size,sha256=a.sha(p.read_bytes())) for p in sorted(root.iterdir()) if p.name!='output-inventory.json'},directories=[])
v=output_view(local)
assert v==output_view(external) and v['files']==files
put('output-inventory.json',dict(schema='ace-independent-failure-acceptance-output-inventory-v1',copies={str(local):v,str(external):v},identical=True,
    excludes='output-inventory.json itself',failure_acceptance_sha256=acceptsha,baseline_sha256=basesha))
assert (local/'output-inventory.json').read_bytes()==(external/'output-inventory.json').read_bytes()
assert output_view(local)==output_view(external)==v
for root in (local,external):
    for n,meta in files.items(): assert dict(bytes=(root/n).stat().st_size,sha256=a.sha((root/n).read_bytes()))==meta
final=dict(locations=locations,files=len(files),failure_acceptance_sha256=acceptsha,baseline_sha256=basesha,
    output_inventory_sha256=files['output-inventory.json']['sha256'],original_files_rehashed=len(a.PINS),two_copy_output_readback=True)
(a.WORK/'published.json').write_bytes(a.canon(final))
print(json.dumps(final,indent=2))
