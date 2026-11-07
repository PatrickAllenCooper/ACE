# Runtime generated-module diagnostic v2 results review

Date: 2026-11-07

**Disposition: CPU diagnostic passed. Required corrections: 0.** Saved evidence for job `35347001` supports completed staging, the declared package observations, the specific generated-module exception, deterministic arithmetic and owned-child cleanup. This does not adopt a runtime, qualify a model service, establish historical execution or produce a scientific result. Failed predecessor `35345616` remains failed and preserved.

This is a distinct saved-evidence review. No tests, diagnostic/package source, models or remote commands were executed. Inspection used byte hashes, JSON and static literal parsing, saved-path comparisons and arithmetic on receipts. Only this review was authored.

## Custody and execution identity

Both `results/ace_llm_intervention_runtime_metadata_20261107/terminal02` and `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-terminal-20261107-02` contain the same **148 raw files totaling 1,691,665 bytes**. Every raw file matches both the collector's fresh remote inventory and its disposition map. Exact membership and bytes also match for the complete two-copy collection: **152 files totaling 1,753,408 bytes**, including four local collection records. No raw symlinks or missing members were found.

The remote submission result binds its remote intent digest and job ID. All six intent input pins match the captured files and their committed bytes at `0bc0640281cf2874145cc524349e2211ed64b8b0`. The relevant identities are:

| Object | SHA-256 |
| --- | --- |
| Source | `31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329` |
| Launch packet | `d59a578560a6fbe4ce68f1403414300c0208d054b1c8669eb18d4faf7e4214db` |
| Diagnostic contract | `80107304e27dca7a1ff328487d3189210cd5509bdf2051e26641ab7c28dff151` |
| `terminal02/inventory_transport.json` | `9d39aabc3724610379b7ecc2a4822f333173b7a373e94a43c0d18fd7683227df` |
| `terminal02/disposition.json` | `ca8ae5336501dd382c73fb5ac7e2bfd047893f9b8f018971572615eedaef447a` |
| `qualification/stage.json` | `341881bf56c8fe3bb45b5e5c59aaf71c085dcfa661d622b0a62a4e0801ec965a` |
| `qualification/terminal.json` | `687c694aad2f3c200153b3e7725ace28e9ac30f3d21c819ac5cc43d0a23d40b0` |
| `qualification/worker/terminal.json` | `e52c5753a97eedf02b23c6c1030b99037db54150de59f44e16542bfce5860a8e` |
| `qualification/worker/after-imports.json` | `e3f851fc7edfe5e480e691500a03e58f1b76f1555e7371b3a027aede145b9023` |
| `qualification/worker.cleanup.json` | `8b5495f4ce4f4f910d79d73eb3b22135f867e93f49bef4e975c392396d81dece` |

The supervisor and worker terminal files each equal their candidate bytes. All 19 worker evidence pins and exact worker-file membership agree. Worker stage and diagnostic pins, the supervisor's worker-terminal pin, and the returned/cleanup records form a consistent chain. The collector's pending-review status is appropriate and does not itself assert acceptance.

## Staging, generated code and imports

The stage receipt records **49,966 objects and 8,397,575,073 regular-file bytes** under the new node-local runtime prefix. Its archive SHA is `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98` and manifest SHA is `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a`. These agree with the fixed packet and successful allocated-stage receipt. The bulk archive was not redundantly rehashed by this review. Node-local contents are ephemeral, with the observed stage and generated-source evidence retained locally.

The generated file is **2,355 bytes**, SHA-256 `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`. I independently reconstructed these bytes from the saved, pinned Torch template literals and seven fixed non-scriptable substitutions using static parsing only. They exactly equal both the recorded expected source and actual source content in the post-import and final snapshots.

Recorded root identity matches the stage-bound private `worker-tmp`. The generated directory is its direct child. Both directories have owner UID 348288 and mode 0700. The generated file is regular, single-link, owner-matched and not group/world writable. Its recorded module file, spec origin and three function code filenames agree with the one expected path. The captured six provenance-source hashes match the independently pinned saved sources. The reviewed running guard additionally enforces canonical no-symlink paths, exact directory membership and loader identity. This is post-import provenance under the stated trusted-owned process scope, not a sandbox or universal in-memory attestation.

Before imports there are 88 recorded module paths and no mixed modules. After imports and at the final snapshot there are 2,625 module paths with exactly one exception, `_remote_module_non_scriptable`. I recomputed the exceptions from the saved path maps. The search path gains exactly its verified directory. Parsing all three saved process maps finds no path outside the staged prefix or permitted system-library roots. These are observations of loaded paths, not a universal dependency-closure claim.

Actual observations are Python **3.10.19**, NumPy **2.2.5**, SciPy **1.15.3**, Torch **2.9.1+cu128**, and Transformers **4.57.3**. Transformers has no direct dictionary version and is observed through the authenticated LazyModule `_objects` value. Metadata is kept distinct: default NumPy metadata resolves to **2.2.6**, both 2.2.5 and 2.2.6 entries remain, and Torch metadata is **2.9.1**. This run neither repairs that mismatch nor justifies forcing metadata to masquerade as imported code.

All three saved cache headers have timestamp flags 0, correct recorded source sizes, and timestamps that differ from their staged sources. Their source/cache hashes agree with the stage pins. Standard-loader timestamp rejection is expected, but execution choice was not traced. No code-equivalence comparison occurred, no source-only execution is established, and the historical Conda cache hash differences remain unexplained.

Saved arithmetic gives dot product **11** and both NumPy and SciPy solutions **[2, 3]**. These satisfy the specified equations independently. Qwen class import completed, but no model weights or model inference were involved. Dependency capability checks were not instrumented. Zero GPU allocation is verified separately from any assumption about internal library capability checks.

## Scheduler, cleanup and costs

Saved running allocation and terminal scheduler records agree on `ucb736_asc2`, `acpu/cpu-normal`, one CPU/task/node, 3 GiB, ten minutes, no requeue, zero restarts and no GPU TRES. Parent, batch and extern steps ended **COMPLETED 0:0**. Scheduler start/end are 13:09:44 to 13:11:03 local cluster time, consistent with the UTC diagnostic phases and **79 allocated CPU seconds**.

All 106 supervisor and eight worker phase records have monotone elapsed and process-CPU values. The stage phase interval is **49.671118 seconds**. Qwen class import completed at worker elapsed **15.914295 seconds** and arithmetic at **16.837462 seconds**. Worker completion, receipt validation and cleanup fit their unchanged deadlines. Owned PID/PGID **2089407** was already gone, with return code 0, no surviving group and no cleanup errors.

| Measurement | Process CPU seconds | Elapsed seconds | Peak process RSS bytes |
| --- | ---: | ---: | ---: |
| Worker receipt | 7.835052649 | 17.700920565 | 706,961,408 |
| Supervisor receipt | 29.007660145 | 69.269371216 | 156,909,568 |

Supervisor elapsed includes the worker interval. Slurm reports TotalCPU **37.783 seconds** and batch MaxRSS **3,143,420 KiB**, approximately **99.927%** of 3 GiB. The batch peak is not explained by the individual process peaks. Its file-cache/cgroup attribution remains uncertain. The job completed without an OOM finding, but these records do not establish comfortable memory headroom for later work.

Closed allocation accounting is **6,287 + 79 = 6,366 CPU seconds**. With the unchanged 3,600-second preparation allowance, admission after release is **9,966 CPU seconds**. The earlier 600-second reservation is replaced by this attempt's actual allocation, not added again. Prior GPU accounting remains **836 seconds**, with zero added here. Process, scheduler and reservation quantities overlap and are not a whole-cycle CPU estimate. Collector measurements of 0.010387 process CPU seconds and 7.775547 elapsed seconds cover only its recorded local collection interval.

The evidence supports this bounded CPU diagnostic only. Runtime adoption, model-service qualification, full project closure, historical certification and scientific acceptance remain false. Required corrections remaining: **0**.
