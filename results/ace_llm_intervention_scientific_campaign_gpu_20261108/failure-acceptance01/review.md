# Independent GPU01 failure, custody and accounting acceptance

**accepted_failed_prepolicy_attempt — required findings: 0.** Job **35498218** remains `FAILED`, exit `1:0`. The original collector returned **1** at 2026-11-09 22:47:36 UTC. This new review reconciles its sole resource-interface disagreement; it does not rewrite that exit, its disposition or its failure seal. It accepts failure custody/accounting and the pre-policy boundary. Runtime, GPU-use, training-execution and scientific qualification remain false, and no retry or new GPU test is authorized.

## Independent evidence and custody

Both original terminal capsules were independently rehashed against `custody-failure-inventory.json` (`8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0`). Each contains 310 sealed members plus the identical seal itself. Every selected raw file and every directory matched in both copies. Complete before/after transport records (`f9c7dc2e…` / `4303d7d4…`) parse successfully and have exactly equal full file maps, directory membership, namespaces, input pins, hardlink metadata, related jobs and journal pins. The review uses those maps themselves, not the parent's summary flags.

The saved receipt `5224bf51110996d0a72f334dc976db7644f24d99e7500b99730ddf00eff19214` covers **72,557 named files / 1,366,843,790 bytes / 3,537 directories**. Its independently recomputed full inventory digest is `70cfa7801604f4041f934ed74a36d54bd39817397a68dc09649e532e5391b483` and agrees with the original-before, original-after and project-readback digests. The source-pinned successful backup transport and complete member-verification telemetry support the recorded CURC USTAR backup at `/projects/paco0228/ACE_evidence/ace-llm-campaign-gpu-20261109-01/raw.tar`: **1,422,510,080 archive bytes**, SHA256 `ecfa5d2f8512ab1e14b3c578c0d3d606bfe696f8081babe4dce3cfed08229d95`. All 132 original hardlink groups are retained as alias metadata, with every named archive member checked.

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

Process and elapsed metrics remain separate: Slurm parent `TotalCPU=00:06.755`; collector process CPU `124.82727399999999` seconds and collector elapsed `1643.4779242079999` seconds; command wrapper child CPU `131.088909` and elapsed `1650.0954186249999` seconds. These are not alternative allocation charges or extra preparation allowances. Batch RSS is not a model-memory qualification. No GPU utilization percentage or GPU execution qualification is asserted.

The request and actual allocation agree, but the recorded attempt spent its startup path in CPU authentication inside the GPU allocation. A future design needs bulk CPU verification before GPU allocation while preserving exact bindings and deadline semantics, including the second authentication after handoff. This review does not implement or authorize that change, relax guards, modify the packet or locks, rerun the collector, or submit work.

## New independent artifacts

`failure-acceptance.json` SHA256 **cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0**.

`baseline.json` SHA256 **cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef** uses exactly the predecessor's baseline schema and field set. Its input pins bind this new acceptance, the untouched original disposition `545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507`, and predecessor `c21f43194ff61b30c0322acff99c20e71158d980a00a34a1685b87eac7f27125`.

`no-policy-work-proof.json`, `resource-reconciliation.json`, `custody-review.json`, `source-boundary.json`, `checks.json`, all read-source pins, and the metadata-only audit programs preserve the derivations. `originals-unchanged.json` records final readback and `output-inventory.json` seals the two identical new output copies, excluding only itself. New directories are `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/failure-acceptance01` and `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-failure-acceptance-20261109-01`. No original source was executed, no models or tests were run, and no remote action or lock operation occurred.
