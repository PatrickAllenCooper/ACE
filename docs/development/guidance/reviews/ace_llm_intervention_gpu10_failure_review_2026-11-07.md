**Disposition: GPU10 is a preserved failed qualification. The unresolved startup/import failure prevents service acceptance; there is no successful inference or GPU-computation result.** No additional concrete custody, allocation-contract, or owned-process-group cleanup defect was found in this terminal evidence. The parent's separately scoped CPU-only import diagnosis is the next step; this review does not establish a remedy or authorize a GPU retry.

Reviewed November 7, 2026. Scope: the retained GPU10 terminal, its launch identities, scheduler records, failure streams, phases, and cleanup/telemetry evidence. Relevant frozen source was read only to interpret those records, without repeating the closed implementation or packet reviews. Only this document was written. No tests, project/ML imports, model execution, remote commands, transfers, submissions, or implementation edits were performed.

**Custody and launch identity.**

The [local terminal capsule](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10) contains exactly the 217 regular files listed in its remote inventory, totaling 4,909,550 bytes. Independent local hashing found no missing, extra, symlinked, size-mismatched, or hash-mismatched remote members. The [external terminal capsule](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-model-service-gpu-20261104-10/terminal20261107-01) matches all 223 local capsule files, including the six capture/custody/transport files outside the mirrored remote tree. Transfer status is zero and transfer/inventory stderr is empty. This verifies custody against the saved remote inventory, not a new remote observation.

All 24 prelaunch pins match the retrieved bytes: 22 packet members, the packet itself, and the final job script. Retrieved input and script bytes also match [launch10](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/launch10). Submission intent binds source commit `209f701dd7e32337367aa5b4def22aedc6474ab7`, packet commit `58a0e539592085a1b2822cefa0fa06b0a21366fb`, and the already reviewed packet/script hashes below. The saved submission result identifies exactly job `35252306`; the relevant accounting snapshot contains one GPU10 job and no GPU10 retry.

Session0's authentication receipt matches the frozen manifest, the client's expected identity and identity digest, the 13-source map digest, all seven model-file hash records, and the declared package versions. It identifies the `frozen` treatment and artificial scope. This is recorded authentication, not a completed ready handshake. No model inputs were reread by this reviewer. As disclosed in the receipt, package versions do not authenticate the installed dependency bytes.

**Observed failure and execution boundary.**

The [worker failure](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10/remote/integration10/session0/worker/failure.json) and retained 4,561-byte stderr agree: `initialize_qwen` fails at the import of `Qwen3ForCausalLM` from `transformers.models.qwen3.modeling_qwen3`. The dependency chain proceeds through Transformers processing/model/loss modules into SciPy optimize, linalg, sparse, and sparse.linalg, ending with `ImportError: cannot import name 'eigsh' from partially initialized module 'scipy.sparse.linalg._eigen'`. The exception's “most likely due to a circular import” wording is a diagnostic suggestion, not an independently established cause. The earlier `TRANSFORMERS_CACHE` deprecation warning is retained; the evidence does not identify it as the cause.

This is before the frozen service's [model-load marker and from_pretrained calls](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10/remote/input/llm_intervention_model_service.py:281), model CUDA placement, projection installation, or generation. Authentication did hash model input bytes; that file I/O is not model initialization or GPU computation. The only worker phases are authentication start and completion. There is no `model_cpu_load_start`, model-ready record, observed/accepted handshake, request, raw generation, accepted generation, or successful terminal file. Only session0 exists; the saved-adapter session never started. The worker records zero new training updates and zero environment responses, with both raw/accepted generation lists empty. None of the eight planned requests completed; no scientific world execution occurred.

The [client failure](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10/remote/integration10/session0/client/failure.json) is a handshake-stage `EOFError`, “worker stdout closed before a complete response,” with zero completed requests and last completed ID `-1`. Driver failure and Slurm stderr propagate that EOF. It is the downstream symptom of the worker's failed initialization, not evidence of an accepted exchange. Worker/client/driver failure records retain `retry_allowed: false`; no fallback policy or replacement session is evidenced.

**Timing and the startup-alarm hypothesis.**

- Scheduler start/end are November 4, 18:14:40–18:18:04 Mountain time, corresponding to November 5, 01:14:40–01:18:04 UTC: 204 allocated seconds.
- Worker authentication starts at 01:14:59 UTC, approximately 19 seconds after allocation start. Authentication completes at 01:15:01 UTC, with worker monotonic elapsed `1.5885829241` seconds and process CPU time `1.01351192` seconds.
- Approximately 183 seconds remain between the rounded authentication-complete timestamp and scheduler end. This interval includes imports, failure propagation, cleanup, and shutdown; it is not a measured import duration.
- The sampler's timestamps use the driver's separate monotonic origin. Its 173 samples run from driver elapsed `12.2179411831` to `192.2747632298` seconds, spanning `180.0568220466` seconds. They must not be treated as worker elapsed timestamps.

Frozen source arms a 180-second worker startup alarm, rearmed against the original journal start after manifest validation/authentication; the client separately permits 185 seconds for its handshake. Driver/final-watchdog/Slurm limits are 560/580/600 seconds. The observed run ended within the outer caps, but the failure records contain no exact failure elapsed time, signal-delivery event, or `TimeoutError`. The traceback does not identify the alarm handler. An alarm interrupting a dependency import and contributing to a partially initialized module is a plausible hypothesis given the timing; these records neither prove nor exclude it. They also do not establish a defective SciPy installation, a deterministic circular import, filesystem latency, or another specific root cause. Do not relabel the failure as a proven timeout or claim that the startup alarm was independently verified to cause termination.

**Cleanup and GPU evidence.**

Client cleanup and the driver's [fallback cleanup record](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10/remote/integration10/fallback_cleanup.json) agree on owned PID/PGID `2756257`: leader return code 1, group no longer present, no errors, and neither TERM nor KILL sent. Thus the retained evidence supports natural failed exit and successful owned-group cleanup, not graceful successful service closure. Client retention records zero discarded observed bytes and no persistence errors; stdout is empty and stderr is fully within its bound. No `closed.json` or dedicated post-close GPU sample exists because the successful-session path was never reached. The terminal Slurm job/batch states are failed, while the extern step completed; the later November 7 saved queue snapshot is empty. These are historical observations, not a current process/queue probe.

All 173 telemetry records have return code zero, empty stderr, no sampling error, and empty `owned_rows`; intervals are approximately 1.034–1.071 seconds. The [frozen sampler](/Users/pat/code/ACE/results/ace_llm_intervention_model_service_gpu_20261104/terminal10/remote/input/supervise_llm_intervention_checkpoint_gpu.py:102) queries compute-process PID/UUID/memory and retains rows for the worker PID. Therefore the query was available and repeatedly observed no matching compute-process row. It does not measure utilization, provide a numeric zero-memory reading, cover every instant, or independently establish MIG reporting completeness. There are no framework CUDA timings, memory peaks, or inference-progress records. GPU work was not verified; no numerical utilization or throughput claim is justified. The pre-load traceback and absent application progress support a startup failure rather than completed GPU computation, while telemetry limitations remain explicit.

**Allocation and retained costs.**

Saved scheduler accounting and the startup allocation record agree on account `ucb736_asc1`, partition `aa100`, QoS `gpu-testing`, node `c3gpu-c2-u13`, four allocated CPUs, 12 GiB host memory, and one `a100_3g.20gb` MIG slice. The script requests those resources, a ten-minute cap, no requeue, and the distinct GPU10 log/output root. Four allocated CPUs are within the permitted maximum of eight. Startup `scontrol` reports `Requeue=0` and `Restarts=0`; its then-running status/exit code is superseded by terminal `sacct`: job and batch `FAILED`, exit `1:0`, elapsed 204 seconds. This is not a scheduler timeout or reported OOM.

GPU10 consumed **816 allocated CPU-seconds (`4×204`) and 204 GPU-slice seconds**. Preserve these failed-attempt costs despite absent verified GPU work. Scheduler `TotalCPU` is 8.718 seconds and batch `MaxRSS` is 468,984 K; CPU usage time is distinct from allocated CPU-seconds, and host RSS is not GPU memory. The low recorded CPU usage and missing application progress warrant the parent's CPU-only diagnosis; they do not identify the import failure's cause.

Restricting the saved related-accounting rows to the 13 ACE LLM jobs gives **4,450 allocated CPU-seconds and 836 GPU-slice seconds**, including GPU10 (`3634+816`, `632+204`). Unrelated projects and batch/extern duplicates are excluded. Retaining the previous 3,600 CPU-second preparation allowance yields 8,050 CPU-seconds against 57,600, with 836 GPU-slice seconds against 7,200. The allowance is not newly measured preparation usage. The original candidate reservations of 4,800 CPU/600 GPU seconds are not additional actual consumption. This reconciles this closed attempt; it is not admission for another allocation.

**Exact reviewed identities (SHA-256).**

| Artifact | SHA-256 |
| --- | --- |
| Remote inventory, binding all 217 retrieved files and scheduler snapshot | `a8037e8dcd965dd1ada6c046df9916997ca0c35be82700a35fbac75031003b82` |
| Custody receipt | `9f05a585365152630f2b4da72f72b9ee7073825a3970f2a83302b8b94acead05` |
| Packet | `e1a911fcf4d087742e1e0cc2ed87b351ea13ee966113c9b40245704a1afe5c68` |
| Final job script | `de099f6f52eb7982394bf7d9008e8420859837b99c1d2755760c8f356fee9538` |
| Frozen service manifest | `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37` |
| Adapter service manifest (not executed) | `4bd9afe1fe24cc9e83d413856dab0383fc3518c30090ee19f1a73059ad5225d2` |
| Model service source | `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1` |
| Persistent transport source | `8ccd3ae0bd28729af8ebb758c9bd474a926d05362799f45bf0133e1e4b5e4a5d` |
| Integration driver source | `fb3e95013cb46cd53f3b44ee72a141d75616c39b762b8e62ea90d89f3a9de717` |
| Allocation/telemetry/cleanup helper source | `ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec` |
| Worker failure | `315409ad46d1ce2e22bbd5ed8a257d8b566aa3b261472fd3cedaf7a5b8af44d1` |
| Client failure | `ca978cd817eeb7b06cfcec9f6f5d8ceb3d39dda6d6f7687d43b387d2356f1326` |
| Retained worker stderr | `16c88974d7afb7a825cbf32579d8cf1cbb2c475bc3729254d6aca23798888bdb` |
| Driver failure | `821701c5a83c53317ed1b43a3fd135d50aec931ecd7d3a275fafb93e1a28616d` |
| Driver fallback cleanup | `d7f66fbf266e8f17a22e02524aaa875f7c861c6875b9a617a960d8ce115981df` |
