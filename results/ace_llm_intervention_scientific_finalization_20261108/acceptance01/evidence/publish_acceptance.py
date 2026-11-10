"""Publish already-completed saved-evidence review; no remote/scientific calls."""
from pathlib import Path
import json,hashlib,shutil,os,datetime
O=Path(__file__).parent;REPO=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline');R=REPO/'results/ace_llm_intervention_scientific_finalization_20261108';T=R/'terminal01';L=R/'launch02';D=R/'acceptance01';E=BASE/'ACE-LLM-intervention-scientific-finalization-acceptance-20261110-01';S=O/'publication'
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def h(p):return sha(p.read_bytes())
def j(p):return json.loads(p.read_bytes())
def save(n,v):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(v if isinstance(v,bytes) else raw(v))
 return h(p)
assert not D.exists() and not E.exists() and not S.exists()
assert j(O/'facts.json')['accounting']['closed_cpu_seconds']==40265
assert j(O/'runtime-facts.json')['full_semantic_report_audit_pending'] is True
assert j(O/'control-lineage.json')['status']=='actual_control_lineage_pass'
assert h(T/'custody-inventory.json')=='8b05d07c16d9702107cdd984c8d4c0d0d8603c123ec011b7072d4cc08b3b6d62'
assert h(L/'packet.json')=='dcd34d5c040b7a9bf2769736a8ee5fa103090c4ef1af3f7c663cd961eabbc433'
facts=j(O/'facts.json');rt=j(O/'runtime-facts.json');outer=j(L/'packet.json');inner=j(L/'inputs/packet.json');stage=T/'remote/launch/finalization';custody=h(T/'custody-inventory.json');backend_receipt=h(T/'remote-custody.json');disp=h(T/'disposition.json');prebase=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/acceptance01/baseline.json';utc=datetime.datetime.now(datetime.timezone.utc).isoformat();S.mkdir()
allocation=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',requested_cpus=2,allocated_cpus=2,memory_mib=7680,wall_seconds=4200,elapsed_seconds=1596,gpus=0,state='COMPLETED',exit_code='0:0',allocated_cpu_seconds=3192,gpu_slice_seconds=0,batch_maxrss_kib=7861960,scheduler_total_cpu_seconds=74.082)
terminal_pins={n:h(stage/n) for n in ('start.json','attempt.json','allocation.json','stage.json','handoff.json','worker.started.json','worker.cleanup.json','worker.resources.json','worker/start.json','worker/authentication-complete.json','worker/work-start.json','worker/terminal.json','worker/attempt.json','worker/runtime-before.json','worker/runtime-ready.json','worker/runtime-after.json')}
acceptance=dict(schema='ace-scientific-campaign-finalization-acceptance-v1',status='development_finalization_execution_accepted',scope='development',job_id='35533604',launch_packet_sha256=h(L/'packet.json'),finalizer_packet_sha256=h(L/'inputs/packet.json'),source_commit=inner['source_commit'],operational_source_commit=inner['operational_source_commit'],finalizer_source_sha256=inner['files']['finalize_llm_intervention_scientific_campaign_cpu.py'],clock_adoption_sha256=inner['clock_adoption_sha256'],cpu_acceptance_sha256=inner['cpu_acceptance_sha256'],gpu_acceptance_sha256=inner['gpu_acceptance_sha256'],gpu_recovery_acceptance_sha256=inner['gpu_recovery_acceptance_sha256'],allocation=allocation,finalizer_output=outer['paths']['output'],campaign_root=outer['paths']['campaign'],terminal_pins=terminal_pins,finalization_sha256=rt['finalization_sha256'],summary_sha256=rt['summary_sha256'],cell_count=192,policy_comparisons=28,report_member_count=242,report_members_sha256=h(O/'report-members.json'),new_environment_responses=0,new_training_updates=0,accepted_checks={k:True for k in ('source','runtime','scheduler','supervision','cleanup','full_upstream_preservation','full_remote_custody','selected_local_custody','reporting_envelope','report_membership','all_attempt_accounting')},custody_inventory_sha256=custody,remote_custody_receipt_sha256=backend_receipt,disposition_sha256=disp,execution_qualified=True,report_arithmetic_accepted=False,scientific_qualified=False,retry_allowed=False)
accpin=save('acceptance.json',acceptance)
base=j(prebase);am=j(O/'accounting-material.json');base.update(closed_cpu_seconds=40265,closed_gpu_slice_seconds=4458,cpu_admission_seconds=43865,outstanding_reservation=0,terminal_jobs=49,submission_journals=100,related_accounting=am['related_accounting'],journal_pins=am['journal_pins'],input_pins={'predecessor':h(prebase),'acceptance':accpin,'disposition':disp})
assert set(base)==set(j(prebase)) and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
basepin=save('baseline.json',base);memberpin=save('report-members.json',(O/'report-members.json').read_bytes())
review=f'''# Actual scientific finalizer execution acceptance — 35533604

Required findings: **0**. Execution, runtime, ownership/cleanup, source lineage, fixed reporting envelope, complete remote custody and actual accounting are accepted. This decision permits the separately reviewed opaque 242-report retrieval. Report arithmetic, attainment, utility, scientific efficacy and public claims are **not accepted by this review**.

## Exact execution and authority

Job `35533604` completed `0:0`, 06:33:18–06:59:54 UTC on 2026-11-10, 1,596 seconds at 2 CPUs, 7,680 MiB, zero GPUs, `ucb736_asc2/acpu/cpu-normal`, 4,200-second limit and no requeue. Parent, batch and extern accounting all completed successfully. The source-authenticated allocation query agrees with terminal scheduler details; the later expired `scontrol` record is retained and the successful `sacct` job/steps/details establish termination.

Actual packet02 `{h(L/'packet.json')}` and inner `{h(L/'inputs/packet.json')}` bind original packet commit `fc57d6d796b30a02e12b3c7fc32d9f6394638912`, frozen scientific source `{inner['source_commit']}`, and operational entry `{inner['files']['finalize_llm_intervention_scientific_campaign_cpu.py']}` at commit `{inner['operational_source_commit']}`. The independently reviewed/adopted clock authority `{inner['clock_adoption_sha256']}` retains absolute preparation1500, reporting-startup120/cap1620, work3320, cleanup3350 and hard4190. All 117 captured input pins and all transferred packet objects match actual terminal full-map observations; the 58-name loaded-work source roster remains distinct from the input-byte closure. The captured immutable reporter was not changed or rerun during this audit.

Local intent, canonical augmented remote intent and result join exactly; result `{h(L/'submission.result.json')}` identifies one unambiguous submission. Fresh pre-submit rows and journal pins equal the accepted predecessor. Source, packet, collector and sealed-evidence Git bindings were checked locally, including custody commit `47aa3068b4b9bf951d44f293d93cf35ff607acab`. No remote Git/network recheck was performed here.

## Actual runtime, clocks and cleanup

Parent elapsed 1,585.773751 seconds. Full parent authentication took 870.601681 seconds, including complete CPU/GPU/campaign inventory passes. Runtime staging took 83.551155 seconds. The worker retained the original parent clock, completed full authentication at elapsed 1,096.230477, and started reporting 0.478767 seconds later. Actual reporting took 294.353026 seconds. Worker completed by elapsed 1,402.149442; parent postchecks took 179.052579 seconds. All 20 phase-timing pairs and 102 ordered staging observations are preserved and consistent. Full CPU and GPU inventories were each checked three times (parent admission, worker admission, parent postchecks); both report validators retained matching final campaign maps. No historical check was skipped or clock silently reset.

The archive/manifest and six cache/source pins match the previously accepted runtime. Actual isolated `-I -B` Python3.10.19 ran from owned node-local scratch. Three snapshots authenticate loaded project sources against captured input pins; all other module and native mappings are inside the staged runtime or the allowed system library roots. NumPy2.2.5 and an actual native extension appear only after runtime admission. No Torch, Transformers, tokenizer, SciPy or safetensors modules were loaded. Fixed offline/CPU/thread controls match. The worker PID equals its process group; exit0, no kill, no cleanup errors and `group_still_exists=false` join the parent result. Parent and worker attempts are completed without failure records. This is recorded trusted-process consistency, not hostile-code attestation.

Batch MaxRSS was **7,861,960 KiB (7,677.6953125 MiB)** against 7,680 MiB. This is near the allocation bound; **no memory headroom or future completion guarantee is claimed**. Scheduler TotalCPU74.082 seconds is observed usage, separate from the 3,192 allocated CPU-second charge. Worker/cleanup peak-RSS and process-time observations have their narrower scope and are not substituted for batch memory/cost.

## Complete custody and reporting boundary

Before, after and independently read-back project archive maps agree on **107,284 raw files / 3,308,319,856 bytes**. Every one of the **99,039 accepted upstream files** and all upstream directories remain unchanged, including failed attempts, training/token lineage, both GPU roots and namespaces. The only 8,245 added files are the exclusive finalizer control tree and original campaign reporting surface. The full campaign after-map matches both the worker receipt and parent independently validated result; original GPU acceptance is reused, not re-audited.

The full USTAR archive is `{facts['archive']['path']}`, {facts['archive']['bytes']:,} archive bytes, SHA256 `{facts['archive']['sha256']}`. Captured backend4161 source/bootstrap, one successful bounded backup transport, original request/receipt and three inventory digest joins are authenticated. This is saved proof of readback at collection time; current/future remote accessibility and institutional backup durability are not newly attested. The reviewed batch retrieval must revalidate its receipt/archive at use.

Each local raw selection has **379 files / 306,334,884 bytes**. The entire capsule has 411 files before its own seal. Every local file and directory in both capsules matches the seal and the copies have independent inodes. Explicit omission membership accounts for 3,001,984,972 bytes; neither local capsule is represented as full raw custody. Original collector false-acceptance flags remain unchanged.

The completed source-bound parent and worker both validate the frozen 192-cell, 24-world, 8-policy, 28-pair reporting envelope, original attempt pins, zero new environment responses/updates, and identical finalization/candidate receipt pins. The after-inventory independently contains all **242** required report members, total **4,403,819 bytes**, maximum member860,703 bytes, within the reviewed batch limits. Their exact byte map is `report-members.json` SHA256 `{memberpin}`. The finalization receipt pin is `{rt['finalization_sha256']}`; summary pin `{rt['summary_sha256']}`. This review does **not** decode the 242 private report payloads, independently recompute curves/pairs, or infer utility; their semantic review follows authenticated retrieval. Reporting-envelope acceptance here combines authenticated executed source, matching parent/worker validation and exact full-custody membership.

## All-attempt accounting and preserved failures

The exact 48 predecessor rows/98 journal path-hash entries are unchanged. One actual row adds `2 × 1596 = 3192` CPU seconds and zero GPU seconds; exactly the current intent/result extend to **49 terminal jobs / 100 journals**. Closed totals are **40,265 CPU / 4,458 GPU seconds**, plus the single existing 3,600 CPU preparation allowance = **43,865 CPU admission**, zero outstanding reservation. Remaining ledger ceilings: 13,735 CPU / 2,742 GPU seconds. The original no-remote local outer-read failure and monitor06 size-bound failure remain preserved and add no allocation. Scheduler07 established terminal state before the sole collector invocation. Whole collector invocation97.013593 child CPU/2765.272798125 wall seconds is retained separately; local review and other preparation overhead are not misrepresented as job charges.

Independent inspection used saved bytes, JSON/AST metadata, hashes and local Git objects only. No production callable, scientific replay, numerical/model/tokenizer execution, test rerun, remote operation, lock, job, report retrieval or commit occurred. Both primary inspection scripts passed once. Two small control-inspector schema/index errors and an exploratory formatting error are retained with the original scripts and explicit continuation notes; they are not production defects and did not alter any evidence.

## Consumer handoff

Acceptance `{accpin}`; baseline `{basepin}`; original custody seal `{custody}`; original backend receipt `{backend_receipt}`. `handoff.json` supplies absolute paths and exact keyword pins for reviewed batch source `f62337123abe2eec7c7624e9c32692309d4469db23d28a8e94902a7e1548837c` / backend4161. The accepted original receipt and its original local selection must not be rewritten. Required findings0; report semantics remain a separate gate.
'''
reviewpin=save('review.md',review.encode())
# Capture bounded review artifacts, preserving all inspector attempts. Original large custody stays referenced.
for pth in sorted(O.iterdir()):
 if pth.is_file() and pth.name!='publish_acceptance.py':save('evidence/'+pth.name,pth.read_bytes())
save('evidence/publish_acceptance.py',Path(__file__).read_bytes())
receipt=dict(status='accepted_execution_and_custody',required_findings=0,created_utc=utc,acceptance_sha256=accpin,baseline_sha256=basepin,review_sha256=reviewpin,report_members_sha256=memberpin,custody_inventory_sha256=custody,remote_custody_receipt_sha256=backend_receipt,disposition_sha256=disp,completion_sha256=h(R/'monitor-collection-control01/completion.json'),inspection_facts_sha256=h(O/'facts.json'),runtime_facts_sha256=h(O/'runtime-facts.json'),control_lineage_sha256=h(O/'control-lineage.json'),original_capsules_unchanged=True,scientific_qualified=False,report_arithmetic_accepted=False)
receiptpin=save('receipt.json',receipt)
handoff=dict(status='independent_finalizer_execution_accepted_report_retrieval_ready',required_findings=0,acceptance_path=str(D/'acceptance.json'),acceptance_sha256=accpin,baseline_path=str(D/'baseline.json'),baseline_sha256=basepin,review_path=str(D/'review.md'),review_sha256=reviewpin,independent_review_receipt_path=str(D/'receipt.json'),independent_review_receipt_sha256=receiptpin,receipt_path=str(T/'remote-custody.json'),receipt_sha256=backend_receipt,finalizer_custody_path=str(T/'custody-inventory.json'),finalizer_custody_sha256=custody,finalizer_acceptance_sha256=accpin,members_path=str(D/'report-members.json'),members_sha256=memberpin,member_count=242,member_bytes=4403819,batch_source_path=str(REPO/'results/ace_llm_intervention_remote_custody_20261109/final-reports-batch02/batch_reports.py'),batch_source_sha256='f62337123abe2eec7c7624e9c32692309d4469db23d28a8e94902a7e1548837c',backend_source_path=str(T/'remote_custody.py'),backend_source_sha256='4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f',archive=facts['archive'],repository_acceptance_root=str(D),external_acceptance_root=str(E),report_content_decoded=False,report_arithmetic_accepted=False,scientific_qualified=False)
assert h(Path(handoff['batch_source_path']))==handoff['batch_source_sha256'] and h(Path(handoff['backend_source_path']))==handoff['backend_source_sha256']
handoffpin=save('handoff.json',handoff)
files={p.relative_to(S).as_posix():{'bytes':p.stat().st_size,'sha256':h(p)} for p in sorted(S.rglob('*')) if p.is_file()}
inventorypin=save('inventory.json',dict(files=files,excludes=['inventory.json'],repository_root=str(D),external_root=str(E)))
for target in (D,E):
 target.mkdir(mode=0o700)
 for pth in sorted(S.rglob('*')):
  rel=pth.relative_to(S);q=target/rel
  if pth.is_dir():q.mkdir(mode=0o700)
  else:
   with q.open('xb') as out:out.write(pth.read_bytes())
for pth in S.rglob('*'):
 if pth.is_file():
  rel=pth.relative_to(S);assert h(D/rel)==h(E/rel)==h(pth);a=(D/rel).stat();b=(E/rel).stat();assert (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino) and a.st_nlink==b.st_nlink==1
result=dict(status='published_and_both_copies_verified',required_findings=0,repository=str(D),external=str(E),files=len(files)+1,bytes=sum(p.stat().st_size for p in S.rglob('*') if p.is_file()),acceptance_sha256=accpin,baseline_sha256=basepin,review_sha256=reviewpin,receipt_sha256=receiptpin,handoff_sha256=handoffpin,inventory_sha256=inventorypin,members_sha256=memberpin)
(O/'publication-result.json').write_bytes(raw(result));print(json.dumps(result,sort_keys=True))
