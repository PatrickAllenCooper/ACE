# CPU import diagnostic: bounded static review

**Disposition: two required P2 corrections before CPU submission.** Reviewed
November 7, 2026. Scope is the new `probe.py`, accompanying `job.sbatch`, and
the saved GPU10 bootstrap/import prefix needed to assess the comparison.
No diagnostic, project/runtime imports, tests, workers, models, GPU queries,
scientific data generation or remote commands were executed. Only this review
document was written. Source inspection, JSON reads, hashes and AST parsing
were local read-only operations.

## Required corrections

1. **P2 — Preserve worker ownership and outcomes across exceptional exits.**
   At [probe.py:151](/Users/pat/code/ACE/results/ace_llm_intervention_import_diagnostic_20261107/probe.py:151),
   `Popen` creates a separate session, but its command/PID record is written only
   after waiting and any timeout cleanup. If the worker exits after
   `wait(timeout=220)` raises but before `killpg`, `ProcessLookupError` escapes;
   neither that mode's process record nor the diagnostic summary is written,
   and the second arm is skipped. A failed five-second reap or another wrapper
   exception has the same evidence gap. The wrapper also has no `finally`
   cleanup for an already launched worker. Save the command, PID/PGID and start
   time before waiting; tolerate an already-gone owned group; bound reaping;
   and persist the resulting exit/kill/uncertain-cleanup state on every path.
   Keep any fallback strictly scoped to the recorded owned group. The outer
   wait is necessary because the worker's Python alarm can be delayed or caught
   during imports; do not rely on the soft alarm alone. This finding concerns
   the new wrapper, not the previously reviewed service transport.

2. **P2 — Retain the selected installed-source evidence on failed imports.**
   The inventory at
   [probe.py:84](/Users/pat/code/ACE/results/ace_llm_intervention_import_diagnostic_20261107/probe.py:84)
   runs only after *all* imports succeed. If the targeted `eigsh` failure recurs,
   the exception path saves a traceback and remaining SciPy module names but
   no selected-source hashes/paths. Import unwinding can also remove precisely
   the failing modules from `sys.modules`, so merely moving the same loop to
   `finally` does not fully address the gap. Both arms could fail without
   preserving the source subset intended to explain that result. Resolve and
   retain the bounded selected installed files without importing those modules
   (for example through distribution metadata or established package paths),
   and preserve that evidence independently of successful imports. Keep any
   evidence-read errors separate from the primary import failure. If only
   hashes are retained, describe that as a source inventory, not retained source
   contents. No full installed-environment inventory is requested.

## Scope and side effects

The authenticated service bootstrap is compiled from the captured pinned bytes,
and its frozen `authenticate` path validates the original source/model inventory
and runtime versions. It reads and hashes cached weight files; it does **not**
deserialize or instantiate a model. The frozen/artificial/no-adapter guard is
explicit. The transport import's inspected project dependencies introduce NumPy,
not an earlier Torch/SciPy import that would defeat the intended ordering.

The six import statements after that bootstrap match the saved GPU10
`initialize_qwen` import prefix. The alternate arm inserts `scipy.linalg`,
`scipy.sparse.linalg` and `scipy.optimize` before Torch. Neither arm calls
`initialize_qwen`, `serve_policy`, a pretrained loader, an optimizer, an
environment callback or an explicit CUDA API. No installation or installed-file
mutation appears. Importing third-party packages still executes their module
initializers: the literal `cuda_queried=False` is a declaration of the probe's
explicit behavior, not instrumentation proving that no dependency performed
capability checks. Report that limit accurately.

Both arms run in fresh interpreter processes with separate output directories,
exclusive files, `-B` and `-X importtime`. The job supplies offline/no-pyc settings,
empty `CUDA_VISIBLE_DEVICES` and one-thread limits. Those are process-local
environment settings, not modifications to the installed environment. GPU10
used two-thread limits; this is a CPU import diagnostic, not an identical full
GPU-runtime reproduction. Fresh processes isolate Python module caches but not
filesystem caches: the fixed first/second ordering and repeated weight/source
hashing can warm shared files. Timing differences alone cannot establish that
SciPy-first import order fixes the original failure. No additional repetitions
are required by this bounded review.

The fifteen-second faulthandler dumps are periodic observations, not timeout
failures. `deadline_signal` is intended to distinguish a soft-deadline
interruption from an ordinary import exception. Interpret that marker, stacks,
phase times and stderr together; the GPU10 message's suggestion of a circular
import is not itself a causal diagnosis. Its saved trace reaches the SciPy
`_eigen`/`_svds` import path before any `model_cpu_load_start` event.

## Allocation, bounds and evidence interpretation

The script requests account `ucb736_asc1`, `acpu/cpu-normal`, one node/task,
one CPU, 3 GiB, eight minutes, no requeue and no GPU GRES. Before launching the
workers, the new probe records the completed `scontrol` response and rejects
incompatible allocation fields, including any allocated `gres/` entry. A site
allocation of two CPUs would correctly fail this fixed one-CPU guard; it must
not be silently accepted or charged as one. The scheduler timeout path does
not currently save partial output; that happens before either diagnostic arm
starts and must not be interpreted as an import result.

There are two 210-second worker soft alarms and two 220-second parent waits.
Allowing both five-second kill/reap waits plus the ten-second scheduler timeout
gives **460 seconds before launch/file-write overhead**, not just 440. The
remaining twenty seconds must cover that overhead within the 480-second Slurm
cap; the wrapper has no separate absolute whole-run watchdog. Preserve partial
results if the scheduler ends the run. The worker soft alarm and stack dumper
are cancelled at worker exit. Do not label an outer kill as a successful import.

`diagnostic_complete.json` deliberately says `diagnostics_retained` and
`runtime_qualified=False`, including when an arm returns nonzero. That is an
appropriate collection-status distinction. A diagnosis still requires checking
each process record, worker terminal/failure, phases, stacks and importtime
output. A successful wrapper exit alone establishes neither successful imports
nor readiness for another GPU allocation.

The supplied accounting arithmetic is consistent:
4,450 prior CPU-seconds + 3,600 cumulative preparation allowance + 480 proposed
CPU-seconds = **8,530**, below 57,600. Proposed GPU reservation is zero; the
reported prior 836 GPU-slice-seconds is unchanged. Those live prior totals,
queue/readiness and submission custody belong to the parent and were not
queried or independently re-established here.

## Exact inspected bytes

- `probe.py`: `ed42935c49a842e417dd083a2f5c4ad6d1d9f87b802d2a820b03f859516aee98`.
- `job.sbatch`: `11593ce8a4f74bb229b93eff440de08a0303d256afe3b598a078e30c6119bcf6`.
- Saved GPU10 service: `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1`.
- Saved GPU10 frozen manifest: `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37`.

All three job-script pins match those inspected bytes. The batch bootstrap
hashes and executes the same captured probe bytes. Worker launches reopen the
probe path, so the proposed immutable input directory remains a trusted launch
assumption. `preparation01.json` retains the earlier source pin
`f6f630b541a480f8ef63e4c37f321776b5a2632e770a168068738e2fb0c06594`;
its syntax-check declaration does not describe the newer allocation-gate bytes.
This review parsed the current source without executing it. The initial
preparation evidence should remain unchanged, with later custody identifying
the final source/job-script pair. No submission or runtime qualification is
approved by this outstanding-findings disposition.
