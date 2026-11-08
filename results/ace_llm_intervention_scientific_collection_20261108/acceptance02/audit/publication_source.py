from pathlib import Path
import hashlib,json,os,shutil,datetime
repo=Path('/Users/pat/code/ACE');audit=Path(Path('/tmp/ace-scientific-collection02-audit-current.txt').read_text().strip())
root=repo/'results/ace_llm_intervention_scientific_collection_20261108/terminal02';launch=root/'remote/launch';collection=launch/'collection/session/worker/task/collection'
local=repo/'results/ace_llm_intervention_scientific_collection_20261108/acceptance02'
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-acceptance-20261108-02')
review=repo/'docs/development/guidance/reviews/ace_llm_intervention_scientific_collection02_results_review_2026-11-08.md'
def sha(b):return hashlib.sha256(b).hexdigest()
def data(p):return json.loads(p.read_bytes())
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def save(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert not local.exists() and not external.exists() and not review.exists()
custody=data(audit/'custody04/audit.json');runtime=data(audit/'runtime02/audit.json');records=data(audit/'records04/audit.json');census=data(audit/'journals02/journal-census.json');git=data(audit/'custody-git-binding.json')
assert custody['status']=='verified_saved_custody_and_accounting' and runtime['status']=='verified_saved_runtime_source_resources_cleanup' and records['status']=='verified_complete_saved_collection'
assert census['errors']==[] and not census['counts_are_lower_bounds'] and git['status']=='verified_full_terminal_git_tree'
disposition=data(root/'disposition.json');outer=data(launch/'packet.json');inner=data(launch/'inputs/collection-packet.json');proposal=data(launch/'inputs/proposal.json');intent=data(collection/'intent.json')
raw_map={n[len('launch/collection/session/worker/task/collection/'):]:v for n,v in disposition['remote_files'].items() if n.startswith('launch/collection/session/worker/task/collection/')}
assert len(raw_map)>0 and all((collection/n).is_file() for n in raw_map)
full_raw_files={n:v['sha256'] for n,v in raw_map.items()}
core=intent['expected_source_pins'];assert len(core)==13 and set(core)==set(proposal['source_pins'])|{'scripts/research/llm_intervention_training_collection.py'}
binding=dict(root=outer['remote_root']+'/collection/session/worker/task/collection',files=full_raw_files,
 freeze_sha256=inner['freeze_sha256'],realization_acceptance_sha256=sha((launch/'realization-acceptance.json').read_bytes()),
 world_pins=inner['world_pins'],configuration_pins=inner['configuration_pins'],source_pins=core,
 namespaces={str(w['world_id']):w['training']['request_namespace'] for w in proposal['training_worlds']})
assert set(binding)=={'root','files','freeze_sha256','realization_acceptance_sha256','world_pins','configuration_pins','source_pins','namespaces'}
# Exact gate evidence pins stay immutable; original collector false flags stay false.
pins={n:sha((root/n).read_bytes()) for n in ('disposition.json','custody-inventory.json','inventory_transport.json','inventory_after_transport.json')}
assert pins['disposition.json']=='9ab4b1f7fd8aff0e29ec3a8dc2849483d27a3da4071b397d60b95ee46b4e79a9'
assert pins['custody-inventory.json']=='9f9abed853fb3d3d935f5a848abef743ab09f07c731ee2b5ea5c4960a68ac00b'
assert disposition['acceptance'] is False and disposition['collection_accepted'] is False
# Final evidence statement includes additional full committed-tree readback, without changing original audits.
summary=data(audit/'audit-result.json')
final_summary={**summary,'custody_commit':git['commit'],'custody_git_binding_sha256':sha((audit/'custody-git-binding.json').read_bytes()),'raw_collection_files':len(raw_map),'raw_collection_bytes':sum(v['bytes'] for v in raw_map.values())}
save(audit/'final-audit-result.json',encoded(final_summary));audit_pin=sha((audit/'final-audit-result.json').read_bytes())
text=f'''# Scientific collection02 actual-result review — 2026-11-08

Required findings: **0**. Accept the complete original policy_train collection produced by job35409614 and its bounded CPU execution. This is independent saved-artifact acceptance, not a new run, exposure selection, tokenizer qualification, learned-policy training, development evaluation or utility result.

## Exact reviewed evidence and authority

The repository `results/ace_llm_intervention_scientific_collection_20261108/terminal02` and external `ACE-LLM-intervention-scientific-collection-terminal-20261108-02` are identical complete independent regular-byte copies. All30839 raw files/175574845bytes, complete directory rosters, all92 packet objects and23 frozen production-source files were verified. Every capsule file also matches the full committed Git tree `{git['commit']}`. Original collector5d2bdfdf… and its false acceptance/scientific flags are unchanged.

- Outer packet: `a5c36ddfd72de17965bc828d870fc352d1d40dbe86b1d473c8a67fd0bea4ae90`; packet commit `75e3b8221f36a82d2ac4183354ba6a76fdc1fdd4`.
- Supervisor packet: `5a5107d78f2042374b28bd0ca9afefbb7594ba0255e05bba903287670a113fde`; unchanged scientific inner packet `a86dfd705dba2ca213c3ac25884c3e3deca177e00d8ce8be72732a3baf878c35`.
- Source commit `a9346022f145cc29439325b6f291597b2e6ec037`; freeze `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`.
- Remote intent `a9ddf90cf92515ff9670c61c904290226b4d9e1a38a00cee50f3526aa843c87c`; submission result `669fc7544078afbc532813a71bca4644743dd9172aef3f0458c6f1bf5397bb46`.
- Disposition `{pins['disposition.json']}`; custody seal `{pins['custody-inventory.json']}`.
- First inventory `{pins['inventory_transport.json']}`; independent second inventory `{pins['inventory_after_transport.json']}`.
- Final independent audit `{audit_pin}`. All audit sources, failed audit assumptions, corrections and successful attempts are retained in acceptance02/audit; no production source was imported or invoked.

Collection02 retains adopted operational authorityfa94858d…, accepted realization5e28e1aa…, failed collection01 acceptance00c4557a…/proof59c21d63…, and exact predecessor baselineeeec362d…. All146 failed-custody originals remain:144 original job-root files plus two separately captured immutable namespace registries. Both original/recovery claims and all36 explicitly allowed realized-training input pins match; no development descriptor traversal is authorized by this audit.

## Complete scientific collection proof

All32 exact policy_train world IDs,160 ordered records,3584 planned pairs and all original candidate/terminal names are present and authenticated. The acceptance contains the entire original collection subtree ({len(raw_map)} files, {sum(v['bytes'] for v in raw_map.values())}bytes), not a filtered or selected training set. The13-file primitive collection source map is distinct from the14-file scientific glue input map and23-file supervisor source closure; all are joined to their actual scopes.

The independent census checked4640 hash-chained charged-executor journals and9145 raw-return receipts. Every original reservation is linked exactly once to its matching accepted response and a positive source-bound adapter mapping/outcome across4672 complete adapter journals. Counts are9145 reserved attempts=9145 known returns=9145 accepted responses, zero unknown or unaccepted returns; charged cost is10240. The9216 attempt ceiling and10240 cost-unit ceiling are distinct quantities. No failed/truncated/unplanned journals or hidden extra returns were found. Trajectory journals end with their explicit charged-executor `stop`; feedback executors close explicitly. Adapter closures and exact stream/replicate/namespace/noise-key/descriptor/reservation pins establish positive complete lineage, not absence-of-error inference.

All160 record/receipt/envelope/selection/teacher/plan/inventory pins match. Fixed public behavior methods, milestones0/6/12/18/24, four replicate IDs,8-action candidate rosters,128 feedback batches, shared scoring receipts, fixed base predictions, one-update branch mappings, all pair orderings and saved tie classifications match the frozen method. All2897 tied pairs remain alongside687 strict labels. These counts do not select D or establish utility. Teachers bind the original public envelopes and source-ordered pre-feedback decisions. The32 terminal records carry the budget-exhausted stop label; no gain-based stop inference or outcome filtering occurs.

Language and numerical label associations agree with the exact records. Numeric files were checked only as original opaque bytes plus non-object NPY headers/extent: width8891,9 action rows for nonterminal states and1 stop row at terminal. No array, tensor, tokenizer, model or production numerical function was loaded. Branch predictions and scoring truths were checked for exact saved-receipt association, not regenerated, refitted or recomputed. The original approved collection performs fixed Bayesian trajectory/branch learner updates; the verified zero training-update count means learned-policy optimization, not absence of those prescribed collection calculations.

## Runtime, bounds and cleanup

All four original completion/candidate receipt pairs match their supplied pins: outer2a8f4902…, worker9d4ea9cb…, task61c674d6…, collection47609e42…. The authenticated49966-object/8397575073-byte staged runtime binds archive6815ab97… and manifestb44610f6…. Actual isolated Python3.10.19 and imported NumPy2.2.5/native extension appear in the owned staged prefix. Runtime-before has no project/numerical imports; before_collection and after_collection have the exact13 captured work modules. Source hashes/loaders, stable search paths, offline/single-thread environment, absent mixed/generated provenance and zero accelerator evidence match. Torch/Transformers/tokenizers/SciPy are not admitted in this execution. The stage receipt's original runtime_qualified=false remains untouched; this independent review supplies the execution-specific qualification.

Observed staging span is{runtime['stage_seconds']:.6f}s; work begins{runtime['startup_seconds']:.6f}s after the common parent origin, within the authorized300s collection02 startup. Original work850/cleanup875/hard890 offsets remain absolute and unreset. Outer completion is{runtime['outer_elapsed_seconds']:.6f}s; Slurm records800s within the900s cap. The original supervisor cleanup positively reports the owned PID/PGID gone, return0, no kill needed and errors[]. No node-local staged-runtime deletion claim is made: owned-process cleanup and scratch-tree deletion are separate facts.

Slurm confirms once-only job35409614 COMPLETED0:0,2CPU,7680MiB,zeroGPU,ucb736_asc2/acpu/cpu-normal,15minutes,Requeue0/Restarts0. Batch MaxRSS is7862016KiB=7677.75MiB, only2.25MiB below the requested memory. Whole-job TotalCPU is126.338s, distinct from charged allocation1600CPU-seconds. Worker/parent recorded peak158830592bytes is a different measurement scope; it does not invalidate or explain away Slurm's near-limit peak. No OOM is recorded. This review does not establish memory headroom or authorize using the smaller process observation to size later stages.

## Actual accounting and consumer interface

All41 predecessor rows are unchanged in full (including unique IDs/TRES), with exactly the new terminal allocation row; all84 prior journal path/hash entries remain with exactly the new intent/result. Both observations agree on42 terminal study jobs/86 journals. Existing unrelated queued asc1 jobs are outside this study baseline and were neither queried anew nor changed. No current/related ACE reservation is present in the saved observations.

Actual increment:2×800=1600CPU-seconds,0GPU. Closed totals are11386+1600=12986CPU and1601GPU; the single3600CPU preparation allowance gives16586CPU, with zero outstanding study reservation. The new baseline preserves every actual row and journal pin and binds predecessoreeec362d…, this new acceptance, and disposition9ab4b1f7…. Fresh lock/queue/accounting/journal/readiness checks remain mandatory before any mutation; a saved baseline is not live launch authority.

Publish exclusively to repository `results/ace_llm_intervention_scientific_collection_20261108/acceptance02` and external `ACE-LLM-intervention-scientific-collection-acceptance-20261108-02`. `acceptance.json` uses the existing ace-scientific-collection-acceptance-v1 accepted/policy_train contract and exact eight-key collection binding required by TOKEN worker47bbf574… and coordinator88320878…. `accounting.json` uses the existing ace-scientific-screen-accounting-baseline-v1. No production acceptance schema or worker source changed. A separate consumer-arguments.json supplies only the three independently resolved pending inputs; it does not run or modify TOKEN preparation.

Local saved-evidence audit process CPU across retained attempts is26.797160s, plus the separately recorded complete Git binding. Git subprocess costs and interactive inspection/authoring/packaging overhead are not fully metered; these are preparation activities, not new allocation charges. Earlier audit-only assumptions about directory ordering, explicit stop endings, failed-root versus separate registry labels, unrelated queued jobs, inventory/source-map shapes, and terminal action rows were corrected with all predecessors preserved. One auditor syntax error stopped before execution. None changed original evidence or executed project/scientific code.

Limitations: this accepts one complete original training collection and its source/runtime/scheduler/custody lineage. It does not accept future token length/capacity, selected exposure/D, trained policies, development endpoints, scientific effectiveness, future resources, or automatic retry. No tests, models, tokenizer, project callable, world generation, response replay/refit or remote command were run by this audit.
'''
save(review,text.encode());review_pin=sha(review.read_bytes())
acceptance=dict(schema='ace-scientific-collection-acceptance-v1',status='accepted',required_findings=0,scope='policy_train',
 job_id='35409614',collection_accepted=True,complete_custody_verified=True,accounting_complete=True,collection=binding,
 disposition_sha256=pins['disposition.json'],collection_packet_sha256=sha((launch/'packet.json').read_bytes()),
 source_commit=outer['source_commit'],packet_commit='75e3b8221f36a82d2ac4183354ba6a76fdc1fdd4',custody_commit=git['commit'],
 freeze_sha256=inner['freeze_sha256'],realization_acceptance_sha256=binding['realization_acceptance_sha256'],
 namespace_admission_sha256=inner['namespace_admission_sha256'],namespace_claim_sha256=outer['namespace_claim_sha256'],
 operational_amendment_sha256=outer['operational_amendment_sha256'],
 collection_failure_acceptance_sha256='00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee',
 no_response_proof_sha256='59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998',
 predecessor_baseline_sha256='eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b',
 review_sha256=review_pin,review_path=review.relative_to(repo).as_posix(),audit_sha256=audit_pin,evidence_pins=pins,
 original_terminal_pins=runtime['terminal_pins'],runtime_qualified_for_this_cpu_execution=True,frozen_collection_method_verified=True,
 owned_worker_cleanup_verified=True,node_local_runtime_tree_deletion_verified=False,
 collection_accounting=records['accounting'],records=160,retained_pairs=3584,executor_journals=4640,adapter_journals=4672,
 all_original_response_returns_linked=True,worlds_generated=0,model_loads=0,tokenizer_calls=0,training_updates=0,development_evaluated=False,
 numerical_arrays_reexecuted=False,scientific_qualified=False,automatic_retry_allowed=False,
 actual_allocation=dict(cpus=2,memory_mib=7680,wall_limit_seconds=900,elapsed_seconds=800,gpus=0,cpu_seconds=1600,gpu_slice_seconds=0,
 batch_maxrss_kib=7862016,scheduler_total_cpu_seconds=126.338),
 limits='Collection-only acceptance; future fixed exposure/D and tokenizer/model/training/development qualification remain separate. No utility or memory-headroom claim.')
acceptance_bytes=encoded(acceptance);acceptance_pin=sha(acceptance_bytes)
old=data(launch/'failure-baseline.json');assert sha((launch/'failure-baseline.json').read_bytes())==acceptance['predecessor_baseline_sha256']
baseline=dict(schema='ace-scientific-screen-accounting-baseline-v1',scope='exact42-terminal-study-job/86-journal baseline including independently accepted complete collection02 job35409614; not live readiness or retry authority',
 terminal_jobs=42,submission_journals=86,related_accounting=custody['related_accounting'],journal_pins=custody['journal_pins'],
 closed_cpu_seconds=custody['closed_cpu_seconds'],closed_gpu_slice_seconds=custody['closed_gpu_slice_seconds'],
 preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=custody['cpu_admission_after_release'],outstanding_reservation=0,
 cpu_cap_seconds=57600,gpu_cap_seconds=7200,collection_accepted=True,automatic_retry_allowed=False,
 fresh_lock_scheduler_journal_readback_required_before_mutation=True,whole_cycle_process_cpu_known=False,
 previous_baseline_sha256=acceptance['predecessor_baseline_sha256'],realization_acceptance_sha256=binding['realization_acceptance_sha256'],
 failure_acceptance_sha256=acceptance['collection_failure_acceptance_sha256'],
 namespace_claim_preserved_sha256='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29',
 recovery_namespace_claim_sha256=outer['namespace_claim_sha256'],
 input_pins=dict(predecessor=acceptance['predecessor_baseline_sha256'],acceptance=acceptance_pin,disposition=pins['disposition.json'],
 audit=audit_pin,review=review_pin,terminal_inventory_before=pins['inventory_transport.json'],terminal_inventory_after=pins['inventory_after_transport.json']))
baseline_bytes=encoded(baseline);baseline_pin=sha(baseline_bytes)
# Static JSON/byte interface verification only; never invoke TOKEN/worker functions.
assert acceptance['status']=='accepted' and acceptance['scope']=='policy_train' and acceptance['required_findings']==0
assert set(binding['world_pins'])==set(binding['configuration_pins'])==set(binding['namespaces'])=={str(w) for w in range(202611070000,202611070032)}
assert set(binding['source_pins'])==set(proposal['source_pins'])|{'scripts/research/llm_intervention_training_collection.py'}
assert baseline['input_pins']['acceptance']==acceptance_pin and baseline['closed_cpu_seconds']==old['closed_cpu_seconds']+1600==12986
assert baseline['closed_gpu_slice_seconds']==old['closed_gpu_slice_seconds']==1601 and baseline['cpu_admission_seconds']==16586
assert baseline['journal_pins']=={**old['journal_pins'],outer['remote_root']+'/submission.intent.json':sha((launch/'submission.intent.json').read_bytes()),outer['remote_root']+'/submission.result.json':sha((launch/'submission.result.json').read_bytes())}
assert len(baseline['related_accounting'])==42 and len(baseline['journal_pins'])==86 and baseline['outstanding_reservation']==0
interface={'collection-acceptance':dict(path=str(local/'acceptance.json'),sha256=acceptance_pin),
 'baseline':dict(path=str(local/'accounting.json'),sha256=baseline_pin),
 'collection-disposition':dict(path=str(root/'disposition.json'),sha256=pins['disposition.json'])}
# Exclusive original publication; source custody is never written.
local.mkdir(mode=0o700);external.mkdir(mode=0o700)
publication_bytes=encoded(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),acceptance_sha256=acceptance_pin,accounting_sha256=baseline_pin,review_sha256=review_pin,audit_sha256=audit_pin,originals_modified=False,project_callable_executions=0,remote_actions=0))
for dest in (local,external):
 save(dest/'acceptance.json',acceptance_bytes);save(dest/'accounting.json',baseline_bytes)
 save(dest/'review.md',review.read_bytes());save(dest/'consumer-arguments.json',encoded(interface))
 shutil.copytree(audit,dest/'audit',copy_function=shutil.copyfile)
 save(dest/'publication.json',publication_bytes)
def inventory(dest):
 return {x.relative_to(dest).as_posix():dict(bytes=x.stat().st_size,sha256=sha(x.read_bytes())) for x in sorted(dest.rglob('*')) if x.is_file()}
a=inventory(local);b=inventory(external);assert a==b,'acceptance mirrors must be byte-identical'
seal=encoded(dict(files=a,excludes=['inventory.json'],mirrors_identical=True,original_custody_unchanged=True))
for dest in (local,external):save(dest/'inventory.json',seal)
assert inventory(local)==inventory(external)
print(json.dumps(dict(repository=str(local),external=str(external),acceptance_sha256=acceptance_pin,accounting_sha256=baseline_pin,review_sha256=review_pin,review=str(review),inventory_sha256=sha(seal),consumer_arguments=interface),indent=2))
