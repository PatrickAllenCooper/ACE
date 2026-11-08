from pathlib import Path
import json,hashlib,types,sys,time,traceback,resource
OUT=Path(__file__).resolve().parents[1];T=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01');R=T/'remote';L=R/'launch';O=L/'training';W=O/'session/worker/artifacts';F=W/'fit';start=time.monotonic()
def need(b,m):
 if not b:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def encode(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(p,b):
 with p.open('xb') as f:f.write(b)
def save(n,v):write(OUT/'attempts'/n,encode(v))
try:
 reports={n:read(OUT/'attempts'/n) for n in ('custody01.json','source01.json','saved-validation-completed-before-driver-error.json','lifecycle04.json','history06.json')}
 statuses=('verified_complete_original_custody','verified','verified_saved_worker_runtime_and_source','verified_saved_lifecycle_gpu_and_scheduler','verified_original_lineage_and_cpu_consumer_history')
 need(tuple(v['status'] for v in reports.values())==statuses,'all independent actual gates required')
 evidencepins={n:sha((OUT/'attempts'/n).read_bytes()) for n in reports};checked=reports['saved-validation-completed-before-driver-error.json']['validation']['saved_validation'];life=reports['lifecycle04.json'];history=reports['history06.json']
 authority=read(OUT/'authority.json');disp=read(T/'disposition.json');obs=json.loads(read(T/'inventory_transport.json')['stdout']);rawpins={n:v['sha256'] for n,v in obs['files'].items()}
 def pins(names):return {n:rawpins[n] for n in names}
 pref='launch/training/';wp=pref+'session/worker/artifacts/';fp=wp+'fit/'
 findings={
 'training_gpu_qualified':dict(passed=True,rationale='Independently revalidated frozen stdlib saved-worker contract: exact160SFT+119DPO, all279seven-part update receipts,112adapter matrices,11144recompute calls, base/projection unchanged, fixed reference238forwards. First CUDA forward and actual step intervals paired with owned PID memory before240s; continuing progress/memory gaps within frozen bounds.',evidence_pins=pins([fp+'training/terminal.json',fp+'training/plan.json',wp+'first-gpu-work.json',pref+'first-verified-gpu-work.json',pref+'gpu-sample-0020.json'])),
 'checkpoint_reload_inference_qualified':dict(passed=True,rationale='Prescribed SFT160 and final279 role aliases and opaque bytes independently hashed; fresh base reloads, all119referencepair replays and1184final legal-command scores exact.322saved legal generations including two first-record repeats, unchanged inference signatures and zero added updates/environment responses. No checkpoint selected by outcome.',evidence_pins=pins([fp+'sft.reload.json',fp+'final.reload.json']) if False else pins([fp+'sft-reload.json',fp+'final-reload.json',fp+'sft-reload-scores.json',fp+'final-reload-scores.json',fp+'summary.json'])),
 'complete_custody_verified':dict(passed=True,rationale='Independently streamed every file hash in both original complete sealed capsules, each131227members excluding its seal; exact36623rawfiles538401564bytes, directory rosters and both original remote inventories. Full prior failed/successful TOKEN and collection roots and capsules, original claims, all45jobs92journals retained. Original archive pin and565248000bytes verified.',evidence_pins=pins(['launch/packet.json','launch/submission.intent.json','launch/submission.result.json','namespace.json','original-namespace.json'])),
 'runtime_qualified':dict(passed=True,rationale='All67frozen Git source hashes and actual committed child inputs independently matched. Exact runtime stage/archive/manifest, provenance, cache reconciliation, imported versions, captured project closure and native-map policy independently passed the original saved CPU/runtime validator for after-model-load and after-requests. No ambient model/runtime import was used in this audit.',evidence_pins=pins([pref+'stage.json',pref+'independent-provenance-sources.json',wp+'runtime-stage-binding.json',wp+'runtime-after-service-imports.json',wp+'runtime-after-model-load.json',wp+'runtime-after-requests.json',wp+'authenticated.json'])),
 'owned_process_cleanup_verified':dict(passed=True,rationale='Natural leader exit0 and owned group absent, TERM/KILL false, errors empty, parent/worker terminals consistent before absolute hard deadline; post-exit owned GPU rows empty. This qualifies process cleanup only, not node-local runtime-directory deletion.',evidence_pins=pins([pref+'cleanup.json',pref+'post-exit-gpu.json',pref+'terminal.json',wp+'terminal.json'])),
 'scheduler_verified':dict(passed=True,rationale='Original before-child scontrol allocation independently reconstructed and joined to terminal scontrol/sacct: job35420219 COMPLETED0:0,ucb736_asc2/aa100/gpu-testing,4CPU15360MiB onea100_3g.20gb,2700sec cap,no-requeue/restarts.904actualseconds charges3616CPU904GPU once; prior44rows90journals unchanged.',evidence_pins=pins([pref+'scheduler_raw.json',pref+'allocation.json','launch/job.sbatch','launch/submission.intent.json','launch/submission.result.json']))}
 modulepath=Path('/tmp/ace-gputrain35420219-audit-prep-zqkcv893/audit_saved_training.py');b=modulepath.read_bytes();need(sha(b)=='bf378f64ca534438d6060e7b4d4125420d539df45470b1771ea54f665dcbd11f','audit aid pin');aid=types.ModuleType('audit_aid');aid.__file__=str(modulepath);exec(compile(b,str(modulepath),'exec'),vars(aid))
 with aid.closed_validators() as mods:
  expected=history['expected_training'];prepared_raw=(L/'inputs/prepared.json').read_bytes()
  acceptance,roles=aid.assemble_after_independent_audit(lineage=mods['llm_intervention_scientific_training_lineage'],expected=expected,prepared_raw=prepared_raw,fit_path=F,raw_file_pins=rawpins,checked=checked,findings=findings)
  # The role gate is the exact frozen downstream consumer validator, not a new tensor reader.
  need(acceptance['completed_steps']==279 and acceptance['sft_steps']==160 and acceptance['dpo_steps']==119,'actual exposure counts')
  outputpins={k:v for k,v in acceptance.items() if k in ('engine_terminal_sha256','sft_checkpoint_sha256','sft_adapter_sha256','final_checkpoint_sha256','final_adapter_sha256')}
  versions=read(W/'runtime-after-requests.json');versions={**{k:v['version'] for k,v in versions['packages'].items()},**versions['additional_package_versions']}
  review=f'''# Independent actual scientific GPU training review — 35420219

Status: **accepted for the prescribed policy_train training/reload roles; 0 required findings**. This review covers the completed job and its two original sealed terminal copies, not a future campaign admission or scientific-utility result. All new audit work remains outside the repository; original capsules were not modified.

## Actual identity and authority

Frozen source commit `a9346022f145cc29439325b6f291597b2e6ec037`, freeze `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`, submitted packet commit `9c54a77b3ee0ba3988a2dcdd0fe733ee0562ffe7`. Independently checked all67frozen source hashes against Git without replacement objects, exact29source+archivehelper+3data child roster, every outer input and actual committed packet member. Outer `33fa6064bd224a64736bf466623bd3ae9399c81022dac6023c689515764a9ec0`; inner `b12fd2d7c0879618b48232cdd46bada5a3c1c16a9773680d3769a24a99e7aaf0` at `remote/launch/inputs/training.json`. Local/external original intent bytes agree; remote intent differs only by the reviewed fresh_readiness/remote_utc augmentation. The one result binds job35420219 and its remote intent. Both original namespace claims remain unchanged.

## Complete custody and history

Original disposition `{authority['pins']['disposition.json']}`, seal `{authority['pins']['custody-inventory.json']}` ({authority['custody_seal_bytes']}bytes), original before `{authority['pins']['inventory_transport.json']}` and after `{authority['pins']['inventory_after_transport.json']}` inventories. Both local whole capsules independently rehashed:131227members excluding the seal,1940directories each. Raw36623files/538401564bytes match both remote inventories exactly. Original565248000byte archive hash `{authority['pins']['stream-attempt/archive.tar']}` verified; no extraction, recompression or mutation. Current seal used a narrowly bounded96MiB audit read; no collector/runtime cap was edited.

The closed CPU3893 `campaign_token_history` validator independently joined collection42/86 → failedTOKEN43/88 → acceptedTOKEN44/90 → training45/92, including31606upstream file pins and four complete prior raw roots/directory rosters. Full prior capsules, failed attempts and both claims were included in the whole-custody hash checks. The auxiliary inventory upstream_pins map is an input subset; complete originals are bound by the full raw files maps, not inferred from that subset. No audit of accepted scientific outcomes was repeated.

## Actual execution and prescribed checkpoints

Exactly32original policy_train worlds×5records=160, all3584pairs retained, immutable accepted119DPO exposure:279continuous optimizer updates. Frozen worker saved-evidence validation independently passed for2630evidence files, all seven receipts per step,112adapter matrices/1146880parameters,11144checkpoint recompute calls, unchanged base/projection and fixed post-SFT reference (119pairs/238decoderforwards). The seed109/rank8q/v/AdamW/unpadded singleton recipe and source routing remain unchanged.

Both prescribed checkpoint roles passed the closed pure lineage admission: SFT at160/index159/stage sft and final at279/index278/stage dpo. No intermediate checkpoint or output was selected. Three fresh base loads total, exact119referencepair replays for SFT and1184final legal-command scores before/reload; recorded maximum difference0 for both.161saved legal generations per checkpoint,322total, with first/intervening/first repeatability, zero new optimizer updates during inference, zero environment responses. Opaque adapter/optimizer/reference files were fully hashed but never deserialized by this audit; numerical evidence comes from the authenticated actual execution receipts and unchanged implementation, not an independent model replay.

Exact downstream byte pins:

'''+'\n'.join('- `'+k+'`: `'+v+'`' for k,v in outputpins.items())+f'''

## Runtime, measured GPU work and lifecycle

Original captured Python3.10.19 runtime/archive6815ab97…/manifestb44610f6…, independent provenance/cache/source bindings, full captured project-module closure, imported versions and original GPU-native mapping policy independently passed `validate_worker`/`validate_runtime_records` on saved records. Imported package roster: `{json.dumps(versions,sort_keys=True)}`. A no-heavy-import finder guarded the saved validation; no model, Torch, tokenizer, tensor loader, inference or environment interaction was invoked here.

Stage extraction took35.027s; child start37.555s. First actual CUDA forward249.919ms paired with owned child PID NVIDIA memory at51.481s from the original origin, below240s.698samples/686positive owned-memory observations; maximum observed owned process memory3404MiB.685progress snapshots; conservative maximum telemetry age4.010s (<90) and progress age31.701s (<180). Actual CUDA step intervals total227398.370ms and generation intervals151755.155ms accompany saved work. GPU utilization percentage was unavailable; memory/CUDA/progress establish actual work, not percentage utilization or saturation.

Training wall291.505s; outer terminal895.710s, inside all original absolute worker2620/soft2630/cleanup2645/hard2670 boundaries. The scheduler confirms COMPLETED0:0 after904s,4CPU,15360MiBhost memory,onea100_3g.20gb,ucb736_asc2/aa100/gpu-testing,2700s requested cap,no requeue/restarts. Batch MaxRSS12118404KiB (about11834.38MiB); this is host RSS, distinct from framework/device memory. One requested and observed MIG slice was used. The allocation ended naturally well before its cap; no scaling/throughput guarantee follows from this run.

Natural child leader exit0, group absent, no TERM/KILL, errors[], parent/worker terminal consistency and empty post-exit owned GPU rows verified. **No direct evidence establishes deletion of the node-local runtime directory; no such claim is made.**

## Consumer scope, actual accounting and limitations

The exact10-field expected-training record is independently derived from accepted TOKEN02, original freeze/model/runtime maps and actual submitted job, and matches the separately supplied Kuhn record. The closed pure training-lineage validator verifies the acceptance fields/roles against opaque actual bytes. CPU3893's complete original-history function passes; its full future49-input `admit` call remains deferred until parent supplies actual CPU site readiness and the completed review/output pins. No GPU readiness text or invented CPU readiness was substituted. This is a strict training acceptance suitable as its input, not a CPU packet or permission to bypass storage/readiness gates.

Actual charge3616CPU-seconds/904GPU-slice-seconds. All44original rows/90journals retained and exactly one row/two journals appended: **45terminal jobs/92journals,17805closed CPU-seconds/2505GPU-slice-seconds; single3600CPU preparation allowance gives21405CPU admission-seconds, zero outstanding reservation.** Existing57600/7200caps remain unchanged. Parent-reported storage constraint continues to block subsequent campaigns independently of this accepted completed execution.

Scientific utility, development/confirmation endpoints and future campaign execution remain unqualified. No scientific world was generated, response obtained, tensor deserialized, model replayed or test suite repeated in this audit. No remote action or repository edit occurred.

## Retained audit attempts

All scripts, source snapshots, stdout/stderr and failure receipts remain in this exclusive directory. Four local audit-glue attempts were retained: validate01 used a nonexistent adapter function (actual closed supervisor passes packet directly); validate02 completed the full worker/runtime validator and equality check, then passed scheduler transport dict instead of stdout; lifecycle03 used the sacct details record instead of scontrol job record; history05 overrequired every full-file upstream member in the auxiliary input-only upstream_pins map. These were audit interface errors, not training/runtime failures. Corrected narrow continuations established the remaining gates; no production source/input was changed. `saved-validation-completed-before-driver-error.json` explicitly records the successful validator segment and its exact script/failure evidence rather than claiming a clean initial audit process.

Audit report pins:

'''+'\n'.join('- `attempts/'+n+'`: `'+h+'`' for n,h in evidencepins.items())+'\n'
  product=OUT/'acceptance01';product.mkdir();write(product/'review.md',review.encode());reviewpin=sha(review.encode())
  acceptance.update(audit_review_sha256=reviewpin,audit_evidence_pins=evidencepins,disposition_sha256=authority['pins']['disposition.json'],custody_inventory_sha256=authority['pins']['custody-inventory.json'],inventory_transport_sha256=authority['pins']['inventory_transport.json'],inventory_after_transport_sha256=authority['pins']['inventory_after_transport.json'],training_packet_sha256=sha((L/'packet.json').read_bytes()),training_inner_packet_sha256=sha((L/'inputs/training.json').read_bytes()),submission_result_sha256=aid.RESULT_SHA,parent_terminal_sha256=sha((O/'terminal.json').read_bytes()),worker_terminal_sha256=sha((W/'terminal.json').read_bytes()),runtime_directory_deletion_verified=False,gpu_utilization_percentage=None,audit_models_loaded=False,audit_tensors_deserialized=False,next_cpu_campaign_readiness_qualified=False,storage_admission_qualified=False)
  acceptance_raw=encode(acceptance)
  # Recheck the final published object with all additive evidence fields; no model loads.
  role_args={stage+'_'+kind+'_raw':(F/'training'/f'{stage}.{suffix}').read_bytes() for stage in ('sft','final') for kind,suffix in (('checkpoint','checkpoint.json'),('adapter','adapter.pt'))}
  roles=mods['llm_intervention_scientific_training_lineage'].admit_training(acceptance_raw=acceptance_raw,acceptance_sha256=sha(acceptance_raw),expected=expected,prepared_raw=prepared_raw,engine_terminal_raw=(F/'training/terminal.json').read_bytes(),**role_args)
  accounting=aid.accounting_after_acceptance(cpu=mods['cpu_consumer'],prior_raw=(L/'baseline.json').read_bytes(),result=read(L/'submission.result.json'),disp=disp,obs=obs,acceptance_raw=acceptance_raw,disposition_sha256=authority['pins']['disposition.json'])
  write(product/'acceptance.json',acceptance_raw);write(product/'expected-training.json',encode(expected));write(product/'accounting.json',encode(accounting));write(product/'role-validation.json',encode(roles))
  pinsout={p.name:sha(p.read_bytes()) for p in product.iterdir()}
  write(product/'consumer-arguments.json',encode(dict(schema='ace-scientific-training-gpu-consumer-arguments-v1',metadata={n:dict(path=str(p),sha256=sha(p.read_bytes())) for n,p in {'gpu-training-acceptance':product/'acceptance.json','expected-training':product/'expected-training.json','baseline':product/'accounting.json','training-review':product/'review.md','training-disposition':T/'disposition.json','training-inventory':T/'inventory_transport.json'}.items()},training_remote_custody=str(R),site_readiness=None,site_readiness_status='future_actual_CPU_readiness_required_no_substitution',cpu_campaign_built=False,storage_admission_qualified=False)))
  save('accept07.json',dict(status='accepted_actual_training_roles',required_findings=0,outputs=pinsout,wall_seconds=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,closed_saved_lineage_gate_passed=True,full_future_cpu_admit_not_called=True))
  print(json.dumps(pinsout,sort_keys=True))
except BaseException as e:
 save('accept07-failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
