# Independent saved-evidence runtime metadata results review

November 7, 2026. **Required corrections to the retained custody, accounting and stated failed disposition: 0. Diagnostic qualification remains failed.** CPU job `35345616` completed authenticated staging and reached the Qwen class import, then failed the interpreter search-path guard. It did not reach the arithmetic controls or complete the actual-version gate. These results do not qualify the full runtime, model service or a scientific study.

This is a new postrun evidence review. Only this document was authored. Review operations read, hashed and compared saved local files, parsed JSON and checked the recorded Git bindings. No test, diagnostic, launcher, runtime package or model was executed or imported. No remote operation, allocation, source change, ledger edit or commit occurred. Previous implementation reviews and their fixtures were not repeated. The separate ongoing cause diagnosis is outside this review.

## Custody and execution lineage

Both terminal `remote/` trees contain exactly **144 regular files totaling 1,653,619 bytes**, with no symlinks or additional file members. Independently hashed every member in each copy and matched its size and digest to both the captured remote inventory and the collector disposition. All four top-level collection files also match between copies.

- Repository custody: `/Users/pat/code/ACE/results/ace_llm_intervention_runtime_metadata_20261107/terminal01`
- Exclusive external custody: `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-terminal-20261107-01`

The captured inventory transport and transfer both returned zero. The recorded transfer elapsed time is 7.55787575 seconds. The collector disposition correctly labels complete custody as pending independent review, describes archive reuse as planned, and does not infer staging completion. This review establishes staging completion separately from the saved stage evidence.

The remote submission result names job `35345616` and authenticates the exact remote intent. The intent binds packet `c32df16f…`, diagnostic `a6f1fda5…`, script and three source files, launcher `c93e0791…`, and final prelaunch recheck `b7f8f610…`. All transferred input bytes match the launch copies. The corresponding Git objects at recorded commit `5a2b819a40fd6e8fc4b8d50069b226f5237ed773` match the recorded input, launcher and recheck hashes. This is a lineage check, not another source review.

The saved intent records reconciliation of 16 previous terminal jobs and 34 submission journals. Local controller receipts record both controller locks and their release. The retained submission result records one successful `sbatch` result at `2026-11-07T19:46:36.110414+00:00`. Local and remote clocks differ slightly, so cross-host UTC fields should not be used to infer subsecond lock/submission ordering. The recorded source flow and receipt bindings establish the submission sequence. No duplicate submission or requeue is evidenced.

## Scheduler and resource accounting

Captured `sacct` reports the parent job as **FAILED, exit 1:0**, with one allocated CPU for **72 seconds**, account `ucb736_asc2`, and `cpu=1,mem=3G,node=1` without GPU resources. Captured `scontrol` agrees on one node/task/CPU, `acpu/cpu-normal`, a ten-minute limit, `Requeue=0`, `Restarts=0`, and `Reason=NonZeroExitCode`. The job ran on `c3cpu-e2-u1` from 12:47:11 to 12:48:23 local scheduler time, consistent with 19:47:11 to 19:48:23 UTC. This was not a recorded timeout or OOM termination.

The `.batch` step also failed 1:0. The `.extern` step completed 0:0 and reports 73 seconds. That overlapping step is not a second allocation and is not added to the parent's 72 CPU-seconds. The captured terminal queue contains no active related ACE diagnostic. This is a saved observation, not a claim about the current queue or unrelated jobs.

The actual closed CPU total is **6,215 + 72 = 6,287 seconds**. Releasing the 600-second proposed reservation gives **6,287 + 3,600 preparation allowance = 9,887 / 57,600 admission CPU seconds**. Prior GPU accounting remains **836 / 7,200 seconds**, with zero added by this job.

The process and allocation measurements have different scopes:

| Measurement | Value | Scope |
| --- | ---: | --- |
| Supervisor process CPU | 36.034832729 s | Recorded process interval, excluding worker CPU |
| Worker process CPU | 10.662403861 s | Recorded worker interval |
| Sum of those two CPU intervals | 46.697236590 s | Does not include all interpreter startup/teardown |
| Scheduler parent TotalCPU | 48.422 s | Slurm's recorded job CPU measurement |
| Allocated CPU time | 72 s | One CPU multiplied by parent elapsed allocation |
| Supervisor elapsed | 64.754983416 s | Includes staging and waiting for the worker |
| Worker elapsed | 16.194194055 s | Contained within supervisor elapsed |

These values are not interchangeable or additive across columns/scopes. The collector separately records approximately 0.015804 local process CPU seconds and 7.571057 elapsed seconds. Neither the diagnostic nor collector receipts measure whole-cycle cost, and the 3,600-second preparation allowance is not a measured CPU total.

Supervisor peak self RSS was **156,094,464 bytes**. Worker peak self RSS was **703,332,352 bytes**. Both recorded process peaks are below their diagnostic observation thresholds. Batch `MaxRSS` was **3,143,440 KiB**, about **99.927%** of the 3-GiB request. That batch observation remains close to the allocation limit. These separate peaks do not establish aggregate simultaneous usage or reliable cgroup headroom. File-cache attribution remains unknown. The saved evidence does not support calling this an OOM failure or declaring the memory issue resolved.

## Archive, stage and cache evidence

The saved stage receipt names `/scratch/local/35345616/ace-runtime-metadata-35345616/runtime` and binds the original archive and manifest pins. It records **49,966 objects and 8,397,575,073 regular-file bytes**. The authenticated caller/helper chain and stage-complete receipt establish that the retained archive passed full allocated-stage authentication and extraction before the worker began. The stage receipt explicitly leaves runtime qualification and model loading false.

For the receipt relationship, independently hashed the retained 13-MB manifest and compared its archive identity, object count, file-byte total and six relevant cache/source hashes with the new stage receipt. No bulk archive was recopied or rehashed by this review. The archive pin is `6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98`, with retained size **8,438,650,880 bytes**. The manifest pin is `b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a`. Original archive custody remains prior evidence. The new node-local copy is ephemeral, not a separately retained 8-GB terminal artifact.

All three cache/source observations match the manifest-derived stage pins. Their headers have flags zero and source sizes matching the staged sources, but their header timestamps differ from the actual staged source integer mtimes. The saved report therefore records expected CPython rejection of these stale timestamp caches. No cache body comparison was performed, `code_equivalent` is null, and no comparison executed cached code. This is a header-based prediction, not a traced loader decision. The earlier Conda cache hash differences remain unexplained. The receipt does not claim source-only execution or publisher authenticity.

The phase inventory has 104 sequential supervisor records and seven sequential worker records. Elapsed time, process CPU, cumulative peak RSS and recorded UTC are nondecreasing within each process. Key observations are:

| Phase | UTC | Elapsed in its process |
| --- | --- | ---: |
| Supervisor stage start | 19:47:15.773912 | 1.323552479 s |
| Supervisor stage complete | 19:48:01.225594 | 46.775240423 s |
| Worker cache reconciliation complete | 19:48:01.497141 | 0.052863962 s |
| Worker metadata capture complete | 19:48:01.742260 | 0.297979729 s |
| Worker Torch/attention complete | 19:48:08.426626 | 6.982335808 s |
| Worker Transformers Auto classes complete | 19:48:14.474022 | 13.029729080 s |
| Worker Qwen class complete | 19:48:15.599017 | 14.154726606 s |
| Worker failed terminal measurement | 19:48:17.638476 | 16.194194055 s |
| Supervisor failed terminal measurement | 19:48:19.205331 | 64.754983416 s |

The new stage interval is **45.451687944 seconds**, obtained by subtracting stage-start from stage-complete. The packet's 35.303541-second value is the prior job's staging interval, not this run's measurement. Qwen completion at 14.154727 seconds is cumulative worker elapsed, not the duration of that import alone or the entire allocation.

## Actual imports and failure boundary

Before numerical imports, Python reports **3.10.19** with executable, prefix, base prefix and four search paths inside the staged runtime. No numerical package is yet loaded. Metadata capture preserves **72 distribution entries across 68 normalized names**, all with recorded metadata paths inside the staged prefix.

Both NumPy metadata entries remain present. The default metadata selection is NumPy **2.2.6**, while the subsequently imported NumPy module reports **2.2.5**. This actual observation supports the earlier captured-source diagnosis for this copied runtime. It does not retroactively certify historical execution or adopt a replacement runtime contract.

| Package | Recorded default metadata | Imported module observation |
| --- | --- | --- |
| NumPy | 2.2.6, with 2.2.5 also present | 2.2.5 |
| SciPy | 1.15.3 | 1.15.3 |
| Torch | 2.9.1 | 2.9.1+cu128 |
| Transformers | 4.57.3 | Loaded, recorded version null |

All four package origins are inside the staged runtime. The null Transformers value is an unobserved imported version, not evidence that the package lacks a version, has a different version, or passed the exact-version gate. It must not be replaced with its metadata value in a success claim.

The post-import snapshot contains a new search path, `/tmp/tmphqxw2qr_`, and a loaded module named `_remote_module_non_scriptable` with origin `/tmp/tmphqxw2qr_/_remote_module_non_scriptable.py`. The worker traceback stops at `check_prefix`'s search-path assertion. The final snapshot retains the same path and module, and its repeated containment check records the same error.

Independently compared the saved module-path inventory with the declared staged/caller roots and parsed the captured native mapping paths. The temporary Python module is the recorded outside-prefix module. No outside-allowlist native mapping is recorded before or after imports. This does not identify who created the temporary module, authenticate its bytes, prove an unwanted external dependency, or establish universal binary closure. That cause and provenance work belongs to the separate diagnosis. The recorded guard failure is real under the frozen policy.

The after-import snapshot was written before the guard failed. The full actual-version comparison lies after that guard, so it did not complete. No `arithmetic.json`, arithmetic-complete phase or `worker.returned.json` exists. The import-time stderr log independently reaches `transformers.models.qwen3.modeling_qwen3` before the same guard traceback. Importing the class is not model instantiation, inference, training or intervention selection.

## Failure authority, cleanup and limits of acceptance

Worker and supervisor each have one failed terminal whose bytes exactly match its candidate. The worker's complete evidence map authenticates its 17 other files, including metadata, caches, snapshots, maps, phases and failure details. Its stage and diagnostic pins match the independently retained inputs. The supervisor propagates the worker's nonzero result and remains failed. There is no competing success terminal.

Cleanup names the same PID/PGID `3743066` as the startup receipt. It reports return code 1, `already_gone: true`, `group_still_exists: false`, no cleanup errors and no kill needed. This confirms the owned group was gone at cleanup observation. It does not convert the failed child into a success. No deadline exhaustion is indicated. The cleanup receipt has no standalone duration field, so this review does not invent a measured cleanup interval.

The collector's top-level `failure: null` refers to absence of a legacy `failure.json` file. It is not an assertion of success. Both actual `failure-detail.json` records and failed terminals are present, authenticated and retained. No custody correction is required on that basis.

The recorded flags and executed path support zero model-weight loads, zero training updates, zero environment responses and no explicit caller CUDA API calls. Dependency capability checks during imports were not instrumented. Zero GPU allocation is independently supported by Slurm. There is no model-service qualification, full runtime acceptance, original-environment mutation, metadata cleanup or automatic scientific retry established here.

The necessary next diagnosis is narrow: explain and authenticate the temporary module/search-path creation and obtain the Transformers version through an appropriate observation mechanism. A later diagnostic must still complete its declared origin, version, arithmetic and cleanup gates. This evidence does not justify relabeling the failed run, substituting metadata for imported versions, or broadly allowing arbitrary temporary paths.

## Principal result pins

All values below are SHA-256 of the saved evidence checked in this review.

| Object | SHA-256 |
| --- | --- |
| Complete-custody disposition | `45560cb09c2c31354625c48e7e09f135295413d416fe0c6dcab352e55d636525` |
| Inventory transport | `8bd1db5386ceafac92aec76fe7a8b08d1b3f1e8fc8451b3795123343e7b4fa42` |
| Launch packet | `c32df16f0f0960f25d8871b40d2ecd970ef0d3b93ffd352dcb4075f3cb5f7c63` |
| Diagnostic JSON | `a6f1fda50fc238b8c4b19e2113fc5bbc6fafdcf315cf21cf45677a3a41b3e555` |
| Executed diagnostic source | `9f728a3df959a0e69cc110d31cf11e5c07b9dde84ffe74eb914084c713b91b49` |
| Remote submission intent | `d2e331d3bd068ea5fb3546710dc8f8dfdcdaaad62b02296c916784a784acf937` |
| Remote submission result | `14c0662224ec2131d1acccb0ae4620b50bd348139ba032164fe29d45881c9599` |
| Stage receipt | `3706121ecba81b0f18c485c5dea875d3a63fff16ea5d315d0e6087f956038763` |
| Cache observations | `f556513d4fe37b6a5904211cf6a059c93bdbd712c0a890ebe4c5aa7b42c29b0b` |
| Metadata before imports | `b3fd71e28f2db685df1d34f43a05279d7e95a4eacd4ae2f023cd06492dc76629` |
| After-import/final runtime JSON | `6d41574eaa38035d5c512da8488c21081102abcd177a8ec036ce250f73d7a1bf` |
| Worker failed terminal | `7a9a10c6d38859853ae91fa8f3d9abd834f27d686f81db3177d9845743036568` |
| Supervisor failed terminal | `7875181fd089f044e0b100e4b5bd8f4336b2efe87e4b42343ccd2ce25ce192ce` |
| Owned-group cleanup | `0935cef622185e37600a63a2002010f40ba597d4c9bced7411ab3e2c837a12ba` |
