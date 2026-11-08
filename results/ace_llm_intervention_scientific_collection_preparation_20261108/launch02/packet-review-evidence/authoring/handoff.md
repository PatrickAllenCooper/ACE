# Collection authoring — adopted recovery02 interface

Stable BUILD/LAUNCH/MONITOR authoring, ready for independent review and later actual acceptance inputs. No concrete collection packet has been built. The entire previous authoring tree, handoffs and static attempts are preserved as independent byte-compared copies under `pre-recovery02-update/`; earlier `pre-infra-amendment/`, `static01/`, and `static02-generic/` also remain untouched. `recovery02-update.diff` contains this source delta. No repository source or Git changes.

## Fixed metadata, actual inputs still absent

Pinned adopted freeze: `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`; source commit `a9346022f145cc29439325b6f291597b2e6ec037`. Entry `e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8`; collection glue `2bc872349f9fbfbafe1564b751cde899705f487f3bf822cce5518e0865bd9a4d`. Namespace descriptor `5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36`; authority `ba6ef3937d4aa3cad68a64ae7f02e098041c5bddff5e68d5e5b0833f30cd2a50`. Existing adoption02 review/adoption bytes are pinned too. Full67 metadata closure,23 actual collection input sources; all source bytes captured from the exact Git commit with replacement objects disabled.

Actual successful realization02 acceptance, claim, task custody, disposition, packet/result and next baseline remain required independent inputs. No current job/acceptance/baseline pin or cost is fabricated. Before any output publication, require complete accepted56 realization metadata and exactly40 terminal jobs/82 journals derived from the pinned39/80 failed-attempt baseline63cd36df... plus the actual successful recovery job. Failed35402725 cannot pass this gate and remains charged/in the full row and journal chain.

## Builder CLI

Required `--NAME PATH --NAME-sha256 SHA256` for every name:

```
freeze                 freeze-review
adoption               namespace-admission
namespace-claim        recovery-authority
realization-acceptance realization-disposition
realization-inventory
baseline               predecessor-baseline
realization-packet     realization-result
```

Also required: `--realization-task PATH`, the accepted local custody directory corresponding to realization02 `ROOT/realization/session/worker/task`. All CLI pins remain mandatory, including already fixed public metadata. `recovery-authority` consumes the existing adoption02 record. The focused inventory correction also requires `--realization-inventory PATH --realization-inventory-sha256 SHA256`, for the original terminal02/inventory_transport.json. Actual acceptance must independently include `inventory_transport_sha256` equal to those exact transport bytes; no field is added to or inferred into the original disposition.

The accepted actual realization record must satisfy the existing outer's `status="accepted"`, integer `required_findings=0`, `complete_custody_verified=true`, and `realization_terminal_sha256`, `freeze_sha256`, `namespace_admission_sha256` bindings. Existing packet joins also require `job_id`, `disposition_sha256`, `namespace_claim_sha256`, `realization_packet_sha256`; `inventory_transport_sha256` is also required for the corrected inventory interface; extra acceptance metadata is allowed. It is an independently supplied record, never authored here.

Disposition must establish complete custody/current input lineage, accepted complete failed01 custody (41 original objects), unchanged original local/remote namespace claims, same accepted recovery02 local/remote claim, exact prior accounting/journals, and matching COMPLETED0:0 resources. Actual recovery allocation must be2CPU/zeroGPU, elapsed0..300; charged CPU is2×elapsed. The result must be unambiguous and the new job must differ from failed35402725.

`predecessor-baseline` is the existing failure-acceptance01/accounting.json with pin63cd36df...,39/80,11048CPU/1601GPU. `baseline` is the future independently accepted40/82 record. Keep the existing baseline fields and `input_pins.predecessor`, `.acceptance`, `.disposition` joined to the separately supplied records. Its full row/journal sets must equal the predecessor plus exactly that actual realization job/two journals. Actual cost must agree with disposition; outstanding reservation0, preparation3600, ceilings57600CPU/7200GPU. No helper creates a baseline. Collection reserves1800CPU/zeroGPU, so admission is actual closedCPU+3600+1800, with actual GPU unchanged.

## Training-only boundary and packet shape

The future builder reads only task `terminal.json` and exactly `world-000.json`..`world-031.json`. It does not enumerate the task directory or open development payloads032..055. Terminal metadata checks the complete56 IDs/order/file/semantic pins; only32 policy_train descriptors are decoded and staged. No realizer/project helper is called. Original proposal and configuration bindings preserve160 records,3584 planned pairs,9216 response-attempt/10240 charged-cost caps. These are planned limits, not actual success claims. The independently reviewed collection/outer workers retain charged attempts and failures.

Fixed profile:900s,2CPU7680MiB,zeroGPU,ucb736_asc2/acpu/cpu-normal,no requeue. Absolute offsets startup120/work850/cleanup875/hard890. No memory-headroom claim. Outer inputs retain60 files:23 sources plus37 training-only data/inner admission files. Metadata authority additions stay outside the child inputs. Runtime staging, ownership, deadlines and cleanup use the unchanged captured supervisor/helpers.

Collection remote root: `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01`. Repo `results/ace_llm_intervention_scientific_collection_20261108/launch01`; external `ACE-LLM-intervention-scientific-collection-20261108-01` under the established BASE. Roots derive from the exactly pinned collect stage. Output `ROOT/collection`; task `ROOT/collection/session/worker/task`; raw collection `.../task/collection`.

Outer packet is `ROOT/packet.json`; supervisor packet/admission are `ROOT/inputs/packet.json` and `admission.json`; glue packet/admission are `ROOT/inputs/collection-packet.json` and `collection-admission.json`. Originals are realized only at accepted realization02 root. Original01 and recovery02 registries are read-only; neither is created/adopted/replaced/deleted. Authority/claim link the original58033e...6e6e478... pin and accepted failure/no-draw proof.

## Launcher and monitor

Launcher requires `--packet-directory`, `--packet-commit`, `--packet-sha256` (outer), `--packet-review`, `--packet-review-sha256`. Both existing coordinator locks cover packet authentication, two complete live accounting/journal checks, no-active-ACE admission, site feasibility, all original32 training descriptors/terminal bytes, both original/recovery remote registry pins, transfer/readback and ONE sbatch. Local claims are checked before/after. No new registry or alternate realization is permitted. Interrupted builds/launches and ambiguous submissions preserve evidence and prohibit retry; timeout stdout/stderr remains retained.

Monitor requires `--packet-directory`, `--packet-sha256`, `--submission-result-sha256`, `--evidence-directory` (new exclusive). It observes scheduler/output-marker metadata once, with no outcome JSON decode or acceptance. Monitor source is unchanged. No collection terminal collector was requested/authored in this scope.

## Focused inventory input correction

Previous stable authoring and evidence are independently preserved under `pre-inventory-input-fix/`. `inventory-input-fix.diff` is the narrow builder/launcher change plus the new fabricated test file. Builder no longer reads any `disposition.remote_files` field. `realization_files()` authenticates the required raw transport against both CLI pin and independently supplied acceptance inventory pin, requires integer returncode0, terminal gate/root present/no observation error, and validates literal ROOT/JOB assignments in the captured controller via AST without execution. The stdout parent accounting must identify the accepted COMPLETED0:0 recovery job. Its safe files map must match the independently pinned disposition's raw count and bytes (actual reported331/5,292,431). Terminal, intent and result pin joins, all32 training descriptor byte checks, and launcher original terminal/descriptor readbacks use this same authenticated map. All56 descriptor hashes may appear as metadata; no development payload is opened.

Read-only metadata observation found terminal02/inventory_transport.json SHA `88af54fed84094045d8701164e34a84801dc538a87df42e2da5d3103f716a0af`. This is recorded for handoff only; the required CLI and independent acceptance joins remain mandatory. The supplied immutable disposition SHA573468... was not modified. Local bytes at that SHA currently include a remote_files key; the corrected API does not require or consult it.

Four new stdlib fabricated methods passed in `inventory-fix04/`, covering no-disposition-map positive shape, missing/mismatched acceptance/transport pins, failed transport/terminal gate, literal root/job/terminal-row mismatches, count/byte mismatches and each of the three terminal/journal pin mismatches. Exact captured builder/launcher/monitor/test bytes, stdout/stderr, command, returncode and wall/child CPU costs are retained. No prior methods were repeated. The acceptance in these fixtures is artificial and is never used with actual scientific evidence.

## Validation and pending gates

`static03-adopted-recovery02/` retains exact source snapshots, checker, stdout/stderr and receipt. AST parse/compile (including embedded remote Python) passed; no target/project script executed. Only the four new fabricated metadata methods above ran; no old tests, world draws, collection, tokenizer/model, remote call, build, namespace claim, allocation, submission, commit or source-helper edit. This is not runtime qualification. Actual accepted realization56 and actual40/82 baseline must arrive before building; parent retains independent authoring/concrete packet review, Git and launch ownership.

## Exact authoring pins

- `build_packet.py`: `86e29d1b1a315168363bb8f8e917226f3c68b095f62b4672d0910fe6f9722e97`
- `launch_once.py`: `fd190804ef3979174510c97aa522be17cb669a078ea3b3c25d5b280dd4f2e5b5`
- `monitor_once.py`: `b621addc3df3af1013543189dac03941f5cf3a131c8a688351e7c66cf48658d6`

- `test_inventory_metadata.py`: `b58b2bc10fd69ee963e0d4e2f082c939f78bb979ddba42777c18bec6095091ba`
