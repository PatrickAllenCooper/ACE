# Scientific collection01 failed-attempt review — 2026-11-08

**0 required findings for acceptance of failed-attempt custody, actual charges and the no-response boundary. Collection itself is not accepted.** Job35407428 is FAILED1:0; no complete collection, token preparation input, runtime qualification, retry or recovery allocation is authorized by this review. No optional suggestions.

## Exact evidence and independent scope

Reviewed original saved terminal01 at `results/ace_llm_intervention_scientific_collection_20261108/terminal01` and external `ACE-LLM-intervention-scientific-collection-terminal-20261108-01`. Both capsules independently hash to the same341 files/7,678,825bytes. All146 raw files/2,336,700bytes match both remote inventories, including all directories and the separate original/recovery claims. The collector's original `terminal_failed_outputs_preserved` disposition and false acceptance/collection/scientific flags remain unchanged.

| Binding | SHA256 |
|---|---|
| Outer packet | `aaf0f21110c0906ba0a6a0ba23faa9e0b5ffc6098f1802682901d47e28302961` |
| Supervisor packet | `fddda2e1fe9c2110ed0ca9c770d4db85605e527af504778dff5957a62c990cfc` |
| Collection packet | `a86dfd705dba2ca213c3ac25884c3e3deca177e00d8ce8be72732a3baf878c35` |
| Original disposition | `fd3a52be3054c1dd8d0d2b324dea3e64bdf8bc3676c28e7367cc0bbf06a3c685` |
| First remote inventory transport | `ab556daa4bb0bd7b3433b168b21e07d0ef8fdbe3590c79c317a9eb747400b454` |
| Second remote inventory transport | `6b0aaa590d3f62e4196643f2b0ca0a5b5015e4214820ec7f9c47ff10a51fe202` |
| Original sealed custody inventory | `0b0ab5a287cd58a31eea828e25ac7ccfa599900287d665da93ea2bbac2054b4d` |
| Remote intent | `61b1d2a2043764a8d82e04dcfafc95baf6dd66598597ed2e9bfa0e94b42ce9b8` |
| Submission result | `92333af96e9f30d321b4f047e2ffcf3b9dce0f9351eb5979b83a027c27b74cbb` |
| Outer failure | `9b34ca42142fd4ce8fb79f1a3319f2a2efde8df0bb90f5ad32d18dcf4c237d59` |
| Original stderr | `112baf9502866eb6674d3fa34c8223efc328285abd4b30deb27a685af1619323` |
| Independent successful audit | `36029474e066c569e05491b1dfccf2ce601e3c2346bba7a9a06147acebe819ce` |

All80 packet objects match packetcommit `67680d3eb9fb0276a8cf9fd01cfddded8ba46c97` and both captured coordinator mirrors. All23 production source objects match sourcecommit `a9346022f145cc29439325b6f291597b2e6ec037`. The committed independent packet review matches the submitted intent. Supervisor source is `e5235f16dd17c72e0310004e7bd8df636008c789db8f793212d528fabfd587f8`; collection glue is `2bc872349f9fbfbafe1564b751cde899705f487f3bf822cce5518e0865bd9a4d`. Freeze remains `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`. The standalone collector is `754efa824d7589795f127adf07f66a46baa9f281f13758ca98ba3ad7a28ba83e`.

The original40-row/82-journal baseline `69cc97784637b0ec522da6e00e00a1f08bdfabe9712e2104a43010ee19f2035f` is unchanged. All36 explicitly allowed upstream realization objects match accepted realization transport88af54fe…; realization acceptance remains5e28e1aa…. Original claim58033e81… and recovery claim40c0937c… match both copied local claims and both remote observations. No development descriptor payload was opened. Training descriptors were hashed as opaque bytes for custody/Git checks, never decoded or regenerated.

## Failure location and no-response proof

The original stderr traces `main:455 -> parent:398 -> stage_runtime:194 -> archive.stage:241 -> event:540 -> write:86 -> interrupted:428`. The exception is `InterruptedError: scientific CPU stage interrupted` while writing/fsyncing a staging-progress receipt. It is not a saved model/collection exception, an output-alias rejection, a Slurm TIMEOUT or an observed OOM diagnosis.

The55 retained phase records begin at stage_start elapsed3.155045432s and end with26,501 of49,966 runtime objects observed staged at elapsed119.991398940s. The first object is observed at68.052392737s. The authenticated start record sets startup120/work850/cleanup875/hard890; source line447 arms the remaining absolute startup timer. The timing and exact call path strongly support startup-alarm interruption during staging. The shared signal handler does not record the signal number, so SIGALRM is an evidence-backed inference, not an independently logged signal identity. The final progress record is a lower bound on copied objects, not proof that exactly26,501 objects were written.

Positive source ordering is decisive: `parent:398` must return from synchronous runtime staging before saving stage.json/creating session/handoff and calling `ownership.supervise` at409. That helper is the only collection child spawn. Only the child can reach runtime-before_collection, work-start and `collection.run_collection` at318; only that glue invokes the response-producing collector. The pinned archive stage performs archive/hash/filesystem work and does not execute the staged runtime or generate scientific responses.

The complete immutable inventories contain no stage.json, provenance completion, session directory, child-start/resource/cleanup receipt, runtime-before, runtime-before_collection, work-start, task directory, executor/adapter journal or collection record. These absences corroborate the positive traceback/source boundary; absence alone is not the proof. Thus this attempt spawned no scientific worker, invoked no collection, generated no world, made zero environment response attempts/returns, charged zero scientific response units, and produced zero training records/pairs or policy updates. The planned32-world/160-record/3,584-pair schedule remains wholly unexecuted here; its10,240cost-unit/9,216attempt limits are not observed output counts.

The earlier alias correction was actually traversed: start.json records `/scratch/alpine` resolving to `/gpfs/alpine1/scratch`, equal root device62/inode8388611, for both logical output bindings on c3cpu-e2-u1. No new canonical-path assumption is needed for this failed attempt.

## Runtime, cleanup and resources

Runtime staging started but did not complete or qualify the runtime for this execution. No staged interpreter, NumPy collection worker, tokenizer, LLM weights or model computation was launched by the observed source path. The OwnedChild handle was constructed, but no child process was assigned; worker-group cleanup is therefore inapplicable, not a fabricated successful cleanup receipt. Node-local partial runtime removal was not inventoried by this whole-ROOT collector and is **not verified**. Scheduler terminal/queue observations establish closure of this allocation, not deletion of its temporary scratch bytes.

Saved Slurm request/allocation evidence agrees on2CPU,7680MiB,zeroGPU,ucb736_asc2/acpu/cpu-normal,900s,Requeue0/Restarts0. Actual elapsed is126s, with FAILED1:0 parent/batch and COMPLETED0:0 extern. Allocation charge is252CPU-seconds/0GPU-seconds regardless of the no-response result. Recorded TotalCPU is21.688s; batch MaxRSS is6,238,568K. Phase-sampled peak self RSS is159,240,192bytes. These measures have different observation paths/times and disagree materially; do not substitute one for the other or treat the phase samples as a whole-job peak. The capsule does not establish a safe enlarged timeout or future staging memory requirement.

Independently normalized every original accounting row, preserving all substantive fields and unique TRES keys, then added the one actual failed row. Independently matched all82 predecessor journal hashes plus this intent/result, giving exact41 terminal allocations/84 journals in both remote snapshots. The complete row sum is **11,386CPU/1,601GPU**; plus the single3,600CPU preparation allowance gives **14,986CPU** with zero outstanding allocation reservation. Original failures remain charged. Unrelated queued jobs are retained in the observation and were not acted on.

## Limits and handoff

Failure/no-response acceptance is a separate record, never `ace-scientific-collection-acceptance-v1`. It cannot authorize tokenization, expose a partial dataset, reset the namespace, replace worlds, repeat scientific collection or launch a retry. Any prospective operational recovery requires its own reviewed source/contract/resource/once-only packet and fresh exact baseline/claims under both locks. The frozen token authoring remains untouched; its earlier prospective successful41/84 transition cannot consume this failed41/84 baseline. A later successful collection would require explicit reviewed lineage/count reconciliation.

Preparation and both actual audit attempts are retained under `/tmp/ace-scientific-collection01-actual-audit-r92f2yuo`, and packaged in the separate failure-acceptance01 capsule. Attempt01 stopped on an auditor expectation error: intent.pins correctly contains79 new inputs plus packet.json, while the checker initially expected only79. The original checker, error, audit output and execution receipt are retained. Only that expectation was corrected; attempt02 passed. No original artifact changed. Attempt01 used0.704844 childCPU/0.861214208wall; attempt02 used0.735492 childCPU/0.877065792wall. Audit overhead outside these measured calls is not inferred. The prepared saved-journal census was never invoked because no collection task exists.

No tests, project/source-worker calls, model/tokenizer/numerical imports, scientific replay/draw/response, remote operation, new allocation, retry, staging or commit occurred in this independent review. The reviewer did not author the collection coordinator or production collector. Required corrections for this limited failed-attempt acceptance: none; operational recovery remains a separate unresolved task.
