# Scientific CPU campaign coordinator source review — 2026-11-08

**1 required finding, confined to the read-only monitor's campaign-bundle path.** No required builder or launcher finding was identified. This review covers Kuhn's prospective coordinator source, not a concrete packet, actual execution or acceptance. Future accepted training44/90 and CPU campaign execution remain separate gates. The reviewer's separately authored GPU campaign coordinator is outside this independent review.

Reviewed `/tmp/ace-scientific-cpu-campaign-authoring-7sihtfq3`:

- `build_packet.py`: `4f9aeefe929d89f70fd98a93d8368fbe1b150b4ef7189e2a7f258f306325e19e`.
- `launch_once.py`: `d2b731024b9dbc567f5254fa6b9ba65875909a9309c4c0dec88c735cdc08cec0`.
- `monitor_once.py`: `354cab58ecf5416293ee8464f133906d4cd05e6f080e47ff5a8160a0e93e0b5e`.

## R1 — Monitor names a bundle path the frozen worker does not write

`monitor_once.py:51` checks `cpu/worker/artifacts/campaign/bundle.json`. The frozen producer `prepare_llm_intervention_scientific_campaign_cpu.py`, SHA256 `ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893`, writes `out/'campaign-bundle.json'` in `run_phase`, where `out=loc['phase']` and the inherited `paths()` defines phase as `cpu/worker/artifacts/cpu-phase`. The actual marker is therefore **`cpu/worker/artifacts/cpu-phase/campaign-bundle.json`**. The campaign directory contains campaign.json/source-plan-pins.json/start.json and policy attempts, not bundle.json.

As written, the monitor reports the intended bundle marker absent even after successful creation. Replace this one path with the actual producer path. A focused static producer-path check is sufficient; no model, ranker, scientific execution or prior test repetition is needed. The correction does not change worker semantics or acceptance. The other monitored parent/worker/ranker and shared-owned-helper marker paths inspected here match their producers.

## Authority, captured inputs and scientific boundaries

The builder fixes sourcea9346022f145cc29439325b6f291597b2e6ec037 and freeze15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d, preserving the original adoption, namespace and recovery claim separately from collection-only amendmentfa94858d…. The frozen CPUcampaign closure has49 source names/51 source-helper inputs; all67 frozen source-map entries are compared to full Git commit bytes with replacement objects disabled. The original proposal's byte and semantic pins and12 scientific source pins remain unchanged. Constants are read by a narrow AST resolver, not project-module execution.

Admission requires independently supplied actual training acceptance, original training packet/result/disposition/inventory, separately pinned expected-training and accepted CPU preparation. Training scope, qualification flags, job/source/freeze/prepared/plan/exposure/model/runtime and collection/token acceptance identities must match. The numerical-plan file and semantic pins come from accepted CPU preparation. Adapter/checkpoint/engine/prepared/numerical-plan bytes are copied and hashed without tensor loading or payload decoding. Original accepted realization terminal/inventory metadata selects exactly world032..055 as24 development descriptors; their private payloads remain opaque to the coordinator.

The original public proposal generates96 binding declarations for the four numerical policies, retaining world/arm namespaces, eight charged initial observations and budget32. These declarations match the existing pure scientific builder's binding shape. The95 worker inputs preserve the complete24-world/eight-policy campaign and the fixed numerical exposure. The allocated worker remains responsible for exact prepared-plan semantics, one matched numerical ranker fit of160+D, and five CPU arms; authoring does not claim those operations occurred or select outcomes.

Prospective frozen/SFT/DPO service declarations use the unchanged service recipe, model/runtime/source identities and accepted checkpoint roles. They retain576 requests per service, startup120/request30/idle30/total1200, and caller startup125/request35/cleanup5. Service commands reference original CPU_ROOT/inputs; all future service outputs and stage bindings point to the frozen GPU_ROOT/gpu. No GPU service, model inference, tokenizer or new world construction is invoked here.

## Resources, accounting and once-only orchestration

The CPU stage remains2CPU7680MiB6000seconds,zeroGPU,ucb736_asc2/acpu/cpu-normal, no-requeue, startup180/work5940/hard5980 and numeric work50/timeout55. Its resource and source declarations match the frozen worker interface. The6000-second cap is a bounded choice, not a throughput or memory-headroom guarantee.

The independently supplied43/88 baseline and actual accepted training44/90 extension must match every original row/journal, current job/resources, original training inventory and actual costs. Failed charges remain; missing/inconsistent accelerator TRES reject in metadata admission. The original failed collection authority is pinned through the accepted training packet. The builder requires actual closed costs plus the single3600CPU preparation allowance and all remaining frozen CPU/GPU/finalizer caps to fit57600CPU/7200GPU. No future cost or acceptance default is supplied.

Builder and launcher hold both existing locks, preserve both claims, and require exclusive output/intent/controller state. The launcher reauthenticates metadata and reconstructs exact CPU input declarations, verifies committed local/external packet mirrors and independent packet review, and twice reconciles the complete live44/90 history, no active ACE job, original accepted input pins, both claims, runtime manifest/archive metadata and site/account/QoS/memory/wall feasibility. Full archive-byte/runtime/CPU-allocation/owned-process checks remain the unchanged allocated worker's responsibility. It transfers and reads back exact pins, writes local then exclusive remote intent and issues one sbatch. Failed/ambiguous transports preserve partial streams and prohibit blind retry. Neither launcher nor monitor cancels jobs.

## Evidence and limits

Read the author's retained three-method static01 evidence and one-method monitor-delta02 evidence without rerunning either. Both returned0; respective wall/childCPU times are0.067123708/0.061604 and0.0420155/0.038176 seconds. The first monitor snapshot was corrected for shared-helper cleanup/resource/start marker names; that evidence is preserved. R1 above concerns the separate bundle location, not those already-corrected helper markers.

Exact source/test/handoff/argument-map snapshots, original receipts and relevant frozen producer bytes are retained at `/Users/pat/.codex/tmp/ace-scientific-cpu-campaign-coordinator-review-kpunpsk8`. Review receipt SHA256 `4932d3f53561d4af219a38029086626180ae5ef62111d08773971d43584d85a7`. No coordinator/production callable, test rerun, model/tokenizer/ranker execution, scientific draw/response, remote action, build, source edit or commit occurred during this independent review. Concrete packet review and actual runtime/source/custody/cleanup/accounting acceptance remain required.

## Focused R1 closure — 2026-11-08

**R1 closed; 0 remaining required findings.** Corrected monitor SHA256 `cb63691494e51fd200b1d40105794126fc7cd1986e7521a1d667836225fd0012` differs only by the required campaign-bundle producer path. Builder4f9aeefe…/launcherd2b73102… and argument map are byte-identical. Read the one new AST regression and retained receipt `9af1ae617602cb638ec2d3693d8bc3fa031a5b6117bb835e1219ab71816ce46f` without rerunning: returncode0,0.0465385wall/0.042706childCPU seconds. It derives the path from the frozen producer and inherited paths helper. Original review SHA256 `94dee1c33ea357baac5d77d47f17e454e41340b2f52e510befecb6da497babae` and original source remain preserved. Corrected source/handoff/regression evidence is retained in the existing review evidence directory under `r1/`. No other review scope changed; no execution, test rerun, remote action or commit occurred.
