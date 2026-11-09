# Scientific GPU01 preliminary failure/source-boundary review

Job **35498218** failed during CPU custody authentication, before the recorded invocation could create its GPU output, stage a runtime, start the GPU controller or a policy service, load a model, generate a response, or realize a world. This is a **preliminary source-backed boundary assessment**, not failure acceptance, complete-custody acceptance, a replacement accounting baseline, or authorization to retry.

The review reads only the named saved failure observation and immutable submitted metadata/source. It does not read the parent's active terminal capsule or perform remote operations, acquire locks, execute project source, run tests, import models, or decode private scientific payloads.

## Exact evidence association

- Observation: `results/ace_llm_intervention_scientific_campaign_gpu_20261108/failure-observation01/stdout.json`, SHA256 `cce561b31d38d34e8a2f0e2f626d741bd845decd5c8d31817558993f20a65aa7`. Its source, intent, result and stdout match the independently stored external observation copies byte for byte. Captured observation source SHA256: `80916a2932d28219368d6ec92f1c6eccc54e196ec5192b6246dfea94c3e31cf9`.
- Exact stderr: 2,099 bytes, SHA256 `a96fcb696a994fd624eddc61c84e908cfbf717285217f5838667b127f58c75cf`. The extracted bytes match the observation's declared size and hash. Stdout is empty.
- Saved submission result SHA256 `7c484f72ecafe9188cf4a71cb7023844c46390eb200ffe7507123d473ba43b86` binds job 35498218 and submission intent `46998e9b8a2d5cc1017b2540cebd533ef1a2637c7b0dd440eb81230eb99b4ca4`, reporting successful, nonambiguous submission once. It is not evidence of scientific success.
- Packet commit `41bda8e1d9e925eca68f1aa6b16f4d616d1cf7db` is a Git revision, distinct from outer packet byte SHA256 `436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7`. Inner packet SHA256: `85d2607d94114109819ca5e8150c5849c13f1bbf4b37765d3f7ee2ff64e77c58`.
- Submitted entry SHA256 `2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1` matches the saved launch copy, packet/intent pins and source commit `a9346022f145cc29439325b6f291597b2e6ec037`. The four helper modules reached before the failure also match that source commit. Git reads disabled replacement objects. The exact submitted sbatch authenticates captured entry bytes before execution and supplies no `--controller` argument.

`source-pins.json` and `checks.json` retain the full associations, helper pins, extracted logs and accounting calculations. This verifies the submitted-source association; a fresh remote rehash of all inputs remains a complete-custody task.

## Recorded boundary

The traceback is `main:677 → authenticate:346 → admit_cpu:198 → inventory:151 → read:98`, ending in `TimeoutError: campaign supervisor signal 14` from the installed handler at line 673. Signal 14 is explicitly recorded. The submitted source arms its SIGALRM handler against `started + 240` at line 675. The evidence supports expiry of that startup alarm during a CPU-output byte read; it does not identify the individual file, latency cause, storage fault, or completed inventory fraction.

The failing operation is the **first** CPU output inventory comparison, excluding `worker/artifacts/campaign` from file hashing. The subsequent campaign inventory, terminal validation, private campaign-bundle decoding, and training-lineage admission were not reached. Directory enumeration occurs before the exclusion filter, but the trace has already advanced into file reading. Opaque input bytes, including adapter `.pt` bytes and potentially other inventory members, may have been read; this is not model deserialization or model execution.

The four helper modules imported before this point have stdlib-only top-level imports; model/runtime work is in functions not called along the failing path. The invocation stops before `out.mkdir` at line 681 and before `stage_and_exec`. Thus runtime extraction, node-local staging, controller startup, policy initialization, inference and GPU computation are not reached on this recorded path. No new response, world realization, training fit or policy attempt is supported by this path. The saved observation independently reports that the `gpu` output path does not exist. Because `owned` remains false, the absence of a stage-local `failure.json` is consistent with the source's exception-publication rule; absence alone is not the boundary proof.

The saved observation inspects the target GPU path and four log/submission files. It is not a full traversal of upstream campaign roots or node-local scratch, and it does not certify the absence of every process or every possible new file elsewhere. Those broader zero-new-work and cleanup claims remain for the sealed failure audit. No node-local runtime-deletion claim is made.

## Accounting and GPU-work interpretation

The saved parent Slurm row is `FAILED`, `1:0`, 245 elapsed seconds, four allocated CPUs and one `a100_3g.20gb` slice. `scontrol` agrees on the account `ucb736_asc2`, partition `aa100`, QOS `gpu-testing`, 15 GiB allocation, 48-minute cap, no requeue and zero restarts. Charge the parent once: **980 allocated CPU-seconds and 245 GPU-slice-seconds**. Do not add batch/extern step rows again.

Pinned predecessor baseline `c21f43194ff61b30c0322acff99c20e71158d980a00a34a1685b87eac7f27125` contains 46 unique terminal jobs and 94 journals. Its recomputed totals are 29,261 CPU-seconds and 2,505 GPU-slice-seconds. Adding this failed job gives provisional closed totals **30,241 CPU-seconds / 2,750 GPU-slice-seconds**, or **33,841 CPU-seconds** including the unchanged 3,600-second preparation allowance. Expected history becomes **47 jobs / 96 journals**; this note does not certify that fresh complete census or publish a new baseline.

No GPU telemetry or application compute progress was reached. The source and failure trace establish CPU authentication work inside a GPU allocation; the exact GPU utilization percentage is unknown. Slurm's 6.755 seconds of reported CPU time is different from the allocated CPU charge and does not diagnose the latency cause. The batch RSS is not a model-memory qualification.

## Recovery contract consequences and remaining gates

Moving this bulk CPU authentication before GPU allocation is consistent with the shared-resource rules. The correction must preserve exact input/source/custody validation and scientific identity. A material source or execution-boundary change needs explicit reviewed binding; the old entry pin cannot describe changed code. This note does not specify the parent's implementation or grant operational adoption.

Two existing source constraints must be carried into that review: `authenticate()` runs both before staging and again after same-PID controller handoff; merely moving its first call would leave a second bulk read. Runtime staging also requires completion before the original `started + 120` boundary at line 485, with the 240-second first-GPU deadline and later work/final deadlines anchored to the original origin. Removing guards or resetting the clock is not established by this failure review, nor is extending the GPU idle window.

Before failure acceptance, require complete preserved original logs/inputs/source/submission and both custody observations/copies or their explicitly admitted compact-backup equivalents; unchanged predecessor claims and scientific inputs; reconciliation of target and shared campaign roots for new attempts/responses; actual terminal lifecycle evidence; and the fresh 47/96 journal/accounting join. Keep the failed attempt and its full charge. Any successor requires an exclusive attempt and explicit reviewed recovery authority, with no overwrite, blind retry, resampling or outcome-driven selection.

All acceptance/qualification flags remain false. No original evidence, repository file, active capsule or scientific source was modified by this review.
