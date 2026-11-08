from pathlib import Path
import hashlib,json,importlib.util,shutil,time,traceback,os,stat
E=Path(__file__).resolve().parent
REPO=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
T=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/terminal02'
OUT=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/acceptance02'
EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-acceptance-20261108-02'
REVIEW=REPO/'docs/development/guidance/reviews/ace_llm_intervention_scientific_tokenization_actual02_review_2026-11-08.md'

def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(p):return json.loads(p.read_bytes())
def save(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def filemap(root):
 result={}
 for p in sorted(root.rglob('*')):
  need(not p.is_symlink(),'symlink evidence')
  if p.is_file():
   st=p.stat();need(st.st_nlink==1,'independent evidence file');result[p.relative_to(root).as_posix()]={'sha256':sha(p.read_bytes()),'bytes':st.st_size}
 return result

def main():
 need(not OUT.exists() and not EXT.exists() and not REVIEW.exists(),'exclusive actual acceptance/review destinations')
 custody=read(E/'custody-attempt01/receipt.json');source=read(E/'source-scheduler-attempt03/receipt.json');artifact=read(E/'artifact-runtime-attempt01/report.json')
 need(custody['status']=='custody_source_history_checks_passed' and source['status']=='source_scheduler_original_clock_checks_passed'
  and artifact['status']=='saved_artifacts_runtime_lifecycle_checks_passed','all actual checks required')
 A=E/'artifact-runtime-attempt01/aid';need(sha((A/'audit_saved.py').read_bytes())=='b3804801e460b9aff44465683dc0fb5cb627af299f883f666ec773c94ab0cf8c','cleared aid source')
 spec=importlib.util.spec_from_file_location('accepted_shape_check',A/'audit_saved.py');aid=importlib.util.module_from_spec(spec);spec.loader.exec_module(aid)
 facts=artifact['facts'];acceptance=dict(schema='ace-scientific-training-preparation-acceptance-v1',status='accepted',scope='policy_train',required_findings=0,
 cpu_packet_sha256=aid.INNER,prepared_sha256=facts['prepared_sha256'],plan_sha256=facts['plan_sha256'],freeze_sha256=aid.FREEZE,
 namespace_admission_sha256=aid.NAMESPACE,complete_custody_verified=True,runtime_qualified_for_this_cpu_execution=True,
 owned_process_cleanup_verified=True,training_execution_qualified=False,scientific_qualified=False,job_id='35419195',
 numerical_plan_file_sha256=facts['numerical_plan_file_sha256'],numerical_plan_sha256=facts['numerical_plan_sha256'],exposure_sha256=facts['exposure_sha256'])
 aid.validate_acceptance(acceptance,facts,'35419195');need(len(acceptance)==18,'strict18 acceptance')
 accepted_raw=raw(acceptance);accepted_pin=sha(accepted_raw)
 observed=json.loads(read(T/'inventory_transport.json')['stdout']);launch=T/'remote/launch';old=read(launch/'token-failure-baseline.json')
 result=read(launch/'submission.result.json');disposition=read(T/'disposition.json')
 review_text=f'''# Scientific TOKEN02 actual preparation review

**Accepted,0 required findings for TOKEN02 preparation.** Independently reviewed complete saved evidence for job35419195, packet69aaaccdf9f20b11665f4708638db9ba621df4c9881befc1d4bb4225a46f912f, innercab514d59959dd64c080d1414bd1dea6a47da59a93c55d3d23fe3b2fc338ee1c and full packet commit6f29023422b01db07412f609e12659fef065bf6c. Strict18-field acceptance SHA256 `{accepted_pin}`. Source a9346022,freeze156026,namespace5027 remain unchanged; separate TOKEN02 startup300 authoritya77f63 is retained. No new scientific-method or model qualification is asserted.

Custody: independent byte/size/roster verification of both complete capsules,63,123 files/1,626,168,012bytes each excluding its seal;31,608 raw files/336,832,982bytes. Every original raw file, empty directory, input, journal, archive and partial/failure history is retained. Original remote hardlink metadata survives while local materializations have independent files/inodes. Both remote inventory observations, full local/external copies, original local/remote intents, exact source/packet commit and reviewed collector68d19/helperfab01 bindings match. Original failed TOKEN01 capsule is fully authenticated against b02ecbbd, not replaced by selected raw files; its failed43/88 baseline and387CPU charge remain. No prior source/model/raw audit or tests were repeated.

Actual scheduler: COMPLETED0:0,272seconds,3CPU11520MiB,ucb736_asc2/acpu/cpu-normal,600-second cap,zeroGPU,no-requeue/no-restart. The44th job adds816CPU; all43 original rows/88 journals remain, producing44/90 and14,189CPU/1,601GPU,17,789CPU including the single3600 allowance,zero outstanding reservation. Original failed allocation costs remain charged. Second scontrol returned Invalid job id; this unavailable observation is retained. Both terminal sacct/request observations agree; first terminal scontrol and original running allocation independently bind the profile. Batch MaxRSS9,418,068K and TotalCPU61.853s are retained separately from parent/child RSS and allocation charge; no cache attribution or memory-headroom claim is made.

Actual runtime/source: all32 captured Python source bytes match the frozen source commit, with28 loaded source_names and observed captured module closure. Stage binds49,966objects/8,397,575,073bytes to the accepted archive/manifest, expected runtime/package/native-prefix records, six pinned provenance sources, exact derived generated-source8205b169 and saved cache reconciliation. No GPU mapping exemption. Parent staging began184.441s and finished218.958s; original-origin startup marker245.315s<300, child-plus-cleanup257.923s and parent completion262.601s<550/590. Parent/owned-source control flow and actual command establish the clock join; worker Journal durations are a different origin and were not used as parent deadlines. Natural owned leader/group exit is explicit,returncode0,no kill/errors. Exact cleanup duration and node-local runtime-directory deletion are not claimed.

Complete artifact arithmetic: all32training worlds×5snapshots=160 records and3,584 original pairs are retained and matched to accepted original record/receipt bytes. Independent stdlib recomputation finds687strict/2,897tied pairs,119selected/568unexposed strict pairs,41records without strict pairs (32terminal and9other). All160SFT precede119DPO, with no recollection, dropped row, tie rescue or replacement. Exact exposure `{facts['exposure_sha256']}` is shared by LLM and numerical plans. Checkpoints159/278 mean SFT reference after160SFT steps and final after119DPO steps; TOKEN creates plans, no trained checkpoint. one_pass_sft=true,one_pass_dpo=false correctly retains zero exposure for unselected pairs.

Saved prompts range1,998–4,199tokens, below the independently required4,744 bound, with471,731prompt tokens and481,651total stored tokens. Command lengths6/8/9, EOS/PAD/legal commands, raw-envelope/prompt identities and full recipe/scoring layout match. Numerical width8,891 and1,154,817parameters derive from pinned encoder constants; every feature shape matches. LLM plan retains112adapter shapes/1,146,880parameters,seed109 and canonical_selected_unpadded recipe. Token IDs were not retokenized and feature tensors were not reconstructed; their provenance is the independently authenticated CPU execution, with saved signatures checked as metadata.

No new environment responses, inference, model weights or training updates occurred in this phase. Original accepted collection accounting remains9,145known/valid response attempts,10,240reserved cost units,zero unknown returns. Preparation acceptance does not establish GPU training execution, trained-policy utility or scientific evaluation; both qualification flags remainfalse. No development answer, model/tensor or tokenizer was decoded/executed by this audit.

Checker history is fully retained. Source/scheduler attempt01 incorrectly demanded byte equality across source-declared packet reserialization; attempt02 demanded a second available scontrol result. Both failed checker sources/logs/receipts remain, with narrowly corrected attempt03 passing. Corrected aidb3804801 was used only after independent recheck9f6e0926 resolved R1/R2; actual artifact/runtime attempt01 passed. No experimental failure was removed and no actual job was rerun.

Separate downstream integration issue reported to parent: actual custody seal33,854,831bytes exceeds the reviewed GPU-training collector's32MiB metadata read bound (33,554,432). This requires a bounded collector correction before its use; it is not a TOKEN02 execution defect or permission to modify terminal custody. No downstream code was changed here.

Full audit, costs, source snapshots and failed attempts are retained at `{E}` and copied into acceptance02/evidence in both acceptance copies. Separate accounting/source/runtime/exposure/report sidecars accompany the strict acceptance. No remote action, new allocation, model/tokenizer/tensor rerun or commit occurred.
'''
 review_raw=review_text.encode();review_pin=sha(review_raw)
 account=dict(old)
 account.update(scope='Exact44terminal/90journal actual baseline including accepted TOKEN02 and all original failed attempts',terminal_jobs=44,submission_journals=90,
 related_accounting=observed['related_accounting'],journal_pins=observed['journal_pins'],closed_cpu_seconds=14189,closed_gpu_slice_seconds=1601,
 cpu_admission_seconds=17789,outstanding_reservation=0,previous_baseline_sha256=aid.BASELINE,tokenization_accepted=True,
 input_pins=dict(predecessor=aid.BASELINE,acceptance=accepted_pin,disposition=custody['independent_input_pins']['disposition.json'],
 review=review_pin,terminal_inventory_before=custody['independent_input_pins']['inventory_transport.json'],terminal_inventory_after=custody['independent_input_pins']['inventory_after_transport.json'],custody_inventory=custody['independent_input_pins']['custody-inventory.json']))
 accounting_check=aid.audit_accounting(account,(launch/'token-failure-baseline.json').read_bytes(),job_id='35419195',result=result,observed=observed,disposition=disposition,
 pins={'token-failure-baseline':aid.BASELINE,'cpu-acceptance':accepted_pin,'cpu-disposition':custody['independent_input_pins']['disposition.json'],
 'cpu-result':'6efcd47104ead43413043697846595bb831774e793f5ad5923e47cc9731bfa73'},remote_root=read(launch/'packet.json')['remote_root'])
 # Last anchor readbacks immediately before exclusive publication; original
 # complete copies were independently hashed and all consumed bytes rechecked.
 for root in (T,BASE/'ACE-LLM-intervention-scientific-tokenization-terminal-20261108-02'):
  for n,h in custody['independent_input_pins'].items():need(sha((root/n).read_bytes())==h,'final immutable anchor '+n)
 attempt_costs=[]
 for n in ('custody-attempt01','source-scheduler-attempt01','source-scheduler-attempt02','source-scheduler-attempt03','artifact-runtime-attempt01'):
  r=read(E/n/'receipt.json');attempt_costs.append(dict(attempt=n,status=r['status'],receipt_sha256=sha((E/n/'receipt.json').read_bytes()),
   wall_seconds=r['wall_seconds'],checker_process_cpu_seconds=r.get('process_cpu_seconds',r.get('checker_process_cpu_seconds')),
   git_child_cpu_seconds=r.get('git_child_cpu_seconds'),cost_scope=r['cost_scope']))
 report=dict(schema='ace-scientific-token02-independent-actual-audit-v1',status='accepted_cpu_preparation_only',required_findings=0,job_id='35419195',
 acceptance_sha256=accepted_pin,review_sha256=review_pin,source_scheduler=source,artifact_runtime=artifact,custody=custody,
 accounting_consumer_check=accounting_check,failed_checker_attempts_preserved=['source-scheduler-attempt01','source-scheduler-attempt02'],
 attempt_costs=attempt_costs,scientific_qualified=False,training_execution_qualified=False,
 no_new_allocation=True,remote_actions=0,tokenizer_model_tensor_reruns=0,
 limitations=['feature tensor hashes checked as metadata, not regenerated','token IDs not retokenized',
 'worker journal clock distinct from parent clock; parent completion supplies separate absolute deadline evidence',
 'second terminal scontrol unavailable; retained first terminal/running allocation plus both sacct observations used',
 'no exact cleanup-duration or node-local runtime deletion claim'],
 downstream_required_correction=dict(scope='GPU-training collector only',path='/tmp/ace-scientific-training-gpu-token02-mcu8h32f/collector/pull_terminal.py',
 source_line=707,actual_custody_inventory_bytes=33854831,metadata_limit_bytes=33554432,token02_acceptance_blocked=False))
 files={'acceptance.json':accepted_raw,'accounting.json':raw(account),'review.md':review_raw,'report.json':raw(report),
 'source-runtime.json':raw(dict(source=source,runtime=artifact['runtime'],lifecycle=artifact['lifecycle'],aid_sha256=artifact['aid_sha256'],aid_review_sha256=artifact['aid_review_sha256'])),
 'exposure-report.json':raw({k:artifact[k] for k in ('facts','exposure_counts','selected_pairs_by_record','numerical_width','numerical_parameter_count','command_token_lengths','one_pass_sft','one_pass_dpo','original_collection_accounting')}),
 'audit-costs.json':raw(dict(attempts=attempt_costs,scope='review costs separate from exact Slurm job charges; whole authoring overhead unmetered'))}
 # Record candidate bytes in the already exclusive audit area before publication.
 candidate=E/'publication-candidate';candidate.mkdir(mode=0o700,exist_ok=False)
 for n,b in files.items():save(candidate/n,b)
 save(candidate/'validation.json',raw(dict(strict_acceptance_keys=18,consumer_accounting_check=accounting_check,
   artifact_runtime_passed=True,source_scheduler_passed=True,full_custody_passed=True)))
 for root in (OUT,EXT):
  root.mkdir(mode=0o700,exist_ok=False)
  for n,b in files.items():save(root/n,b)
  shutil.copytree(E,root/'evidence')
 save(REVIEW,review_raw)
 # Publication receipt is outside strict acceptance; no self-referential hashes.
 receipt=dict(schema='ace-scientific-token02-acceptance-source-receipt-v1',status='published_independently_audited_cpu_preparation',
  required_findings=0,job_id='35419195',acceptance_sha256=accepted_pin,accounting_sha256=sha(files['accounting.json']),
  review_sha256=review_pin,report_sha256=sha(files['report.json']),aid_source_sha256='b3804801e460b9aff44465683dc0fb5cb627af299f883f666ec773c94ab0cf8c',
  aid_review_sha256='9f6e0926e7ec71d36f93dbfb3f33f2093eced9f6abe7b1acb1041844aca5cc2e',
  publication_source_sha256=sha(Path(__file__).read_bytes()),custody_inventory_sha256=custody['independent_input_pins']['custody-inventory.json'],
  disposition_sha256=custody['independent_input_pins']['disposition.json'],strict_acceptance_fields=18,
  historical_failed_token01_preserved=True,checker_failed_attempts_preserved=2,terminal_capsules_unchanged=True,
  artifact_checks_passed=True,runtime_for_this_execution_verified=True,owned_cleanup_verified=True,
  scientific_qualified=False,training_execution_qualified=False,repo_root=str(OUT),external_root=str(EXT))
 for root in (OUT,EXT):save(root/'receipt.json',raw(receipt))
 left,right=filemap(OUT),filemap(EXT);need(left==right,'complete acceptance mirrors differ')
 for n in left:need(not os.path.samestat((OUT/n).stat(),(EXT/n).stat()),'acceptance files alias')
 seal=dict(copies={str(OUT):left,str(EXT):right},identical=True,excludes='custody-inventory.json itself')
 for root in (OUT,EXT):save(root/'custody-inventory.json',raw(seal))
 need(filemap(OUT)==filemap(EXT),'sealed acceptance final readback')
 return dict(status='published',acceptance_sha256=accepted_pin,accounting_sha256=sha(files['accounting.json']),
  review_sha256=review_pin,receipt_sha256=sha(raw(receipt)),acceptance_custody_sha256=sha(raw(seal)),
  repo_root=str(OUT),external_root=str(EXT),prepared_sha256=facts['prepared_sha256'],plan_sha256=facts['plan_sha256'])

if __name__=='__main__':
 start=time.monotonic();cpu=time.process_time()
 try:result=main();code=0
 except BaseException as exc:result=dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());code=1
 result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
 save(E/'publication-result.json',raw(result));print(json.dumps(result,indent=2));raise SystemExit(code)
