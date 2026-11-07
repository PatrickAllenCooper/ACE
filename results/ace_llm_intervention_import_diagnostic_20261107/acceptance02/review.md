# Independent saved-evidence review: CPU import diagnostic 35343088

Date: 2026-11-07. **Required findings: 0.** Accept the retained diagnostic evidence and accounting disposition. The original-sequence startup failure remains unresolved; this review does **not** qualify the model service or authorize a GPU retry. Confidence is high in custody, input identity, allocation and the two recorded worker outcomes; the cause of the timing difference remains uncertain.

Scope was the completed `terminal02` pull, reviewed launch02 input identity, saved worker phases/stacks/import-time output, selected source-capture metadata, scheduler records and cost arithmetic. Work consisted of local reads, SHA-256 comparisons and standard-library parsing of saved evidence. No project/dependency imports, tests, new counterexample probes, models, remote operations, installs, source/ledger edits or commits were performed. Prior component and account reviews remain closed and unchanged. Third-party source implementations were not audited.

## Custody, inputs and allocation

All **75 raw files, 1,958,758 bytes**, independently rehashed from `results/ace_llm_intervention_import_diagnostic_20261107/terminal02/remote`, agree exactly with both `disposition.json` and the remote inventory retained in `inventory_transport.json`. They also agree with the external mirror at `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-import-diagnostic-terminal-20261107-02/remote`. The four surrounding custody files agree between local copies. Saved inventory and transfer commands returned zero; no missing, extra or changed raw file was found. This verifies the retained capture, not continuing remote state.

Pulled probe, batch script and packet match local launch02 and committed bytes at `869e4cbfe4e9e3a37ed0098225723442e4faa293`. The submission result binds the exact intent hash and job ID 35343088. The intent's launcher and prior account-review pins match that commit. Worker commands use the packet's service and frozen-manifest pins; those hashes also match both saved launch10 and terminal10 input copies. Both workers recorded successful authentication before dependency imports. This is bounded input authentication, not a claim that every installed binary was frozen.

Live allocation capture and terminal `sacct`/`scontrol` agree: account `ucb736_asc2`, partition `acpu`, QOS `cpu-normal`, one node/task/CPU, 3 GiB, eight-minute limit, no GPU TRES, no requeue or restart. Terminal state is `COMPLETED 0:0`; runtime is 301 seconds, with start/end 11:02:36–11:07:37 scheduler time. This is successful diagnostic-wrapper completion despite one failed worker. `diagnostic_complete.json` explicitly retains both results and sets `runtime_qualified: false`.

## Worker evidence and interpretation

| Saved measurement | Original sequence | SciPy-first sequence |
| --- | ---: | ---: |
| Worker outcome | Timeout, outer kill, return code -9 | Returned, return code 0 |
| Failure/terminal elapsed seconds | 217.046833 | 49.060399 |
| Process CPU seconds at that record | 6.236775 | 5.692939 |
| Torch phase elapsed seconds | 90.270469 | 2.664037 |
| `transformers` import-time self seconds | 51.154697 | 1.734618 |
| Worker peak RSS, KiB | 640,788 | 700,504 |

The Torch durations above are differences of saved monotonic `elapsed_seconds` fields. They supersede provisional more precise figures in conversation; the ledger's approximately 2.665-second second duration is a minor precision discrepancy, without disposition impact. Import-time output is in microseconds, and nested cumulative times must not be added.

The original worker's `failure.json` records `TimeoutError`, `loaded_scipy: []` and no retry permission. Its traceback runs from Transformers auto-model/generation imports through Torch Dynamo and Triton to `triton._C.libtriton`; no successful Transformers-auto phase was emitted. The deadline handler ran at 216.733115 seconds despite the nominal 210-second alarm. The parent enforced its independent 220-second wait, sent SIGKILL to the owned group, reaped the child and recorded `group_still_exists: false` with no cleanup errors. The second process started about 220.085 seconds after the first. The soft alarm was not an exact hard wall; the outer bound and retained failure matter (`remote/probe.py:104–110, 166–175, 227–241`).

The SciPy-first worker recorded completion of explicit SciPy imports, Torch/attention, Transformers auto classes, Qwen class and project helpers, followed by its terminal receipt. It demonstrates successful imports in that process. The code and receipts show no model construction/weight loading, training or scientific responses, and no explicit CUDA API call. Dependency capability checks were not instrumented. Torch's native-library preload appears in the first worker's stacks, so “no explicit CUDA API” must not be expanded to “no CUDA-related libraries loaded.”

The 14 first-worker and three second-worker faulthandler samples show source reads, path checks and native-library loading. Their repeated 15-second “Timeout” headings are sampling markers, not additional failed attempts. Long elapsed time relative to process CPU is consistent with loading or filesystem waits, but does not identify filesystem latency, cache state, native-loader contention or another wait as the unique cause. No syscall/I/O or controlled cold-cache measurement was retained.

The comparison changes import order and runs sequentially on the same node/shared filesystem. Authentication, selected-source capture and transport import were already faster in the second process, before its SciPy preload. Consequently, these observations cannot establish an order-only correction or reproduce/resolve the previous GPU SciPy failure. The first worker had not reached that failure path. One CPU here and a 210-second soft cap also differ from the earlier two-CPU/180-second startup conditions.

Each arm's 16 selected installed-source captures matches its byte counts and hashes; both inventories are identical. All 16 successful-arm imported-source paths/hashes match those captures. Recorded versions are Torch 2.9.1, Transformers 4.57.3 and SciPy 1.15.3. This is selected Python-source evidence, not complete runtime closure: the observed Triton binary, Torch Dynamo and Transformers import-utility implementation are outside that capture set. No inference about their correctness or byte identity beyond the available evidence is made.

## Cost disposition and next CPU gate

Charge this job **301 allocated CPU-seconds and zero GPU-seconds**. Slurm `TotalCPU` is 14.374 seconds and batch maximum RSS is 1,073,460 KiB; these describe process work/memory, not additional allocation charges. Job/batch/extern records overlap. Worker CPU snapshots also overlap those measures and exclude portions of process startup/teardown; they are not additive. The retained timing does not fully attribute allocation time outside worker measurement windows.

Arithmetic reconciles: 4,462 + 301 = **4,763** closed CPU-seconds; with the unchanged 3,600 preparation allowance, admission is **8,363 / 57,600**. Replacing the 480 reservation with 301 actual releases 179. GPU admission remains **836 / 7,200**. These totals inherit the previously reconciled cycle baseline; older jobs were not re-reviewed.

The next smallest useful gate is one fresh **CPU-only original-sequence qualification** after explicit, measured preparation under the intended launch layout. Keep one CPU, 3 GiB, zero GPUs, exclusive outputs and bounded owned-process cleanup. Budget any copy/hash preparation separately; set the total wall bound from that preparation plus the worker bound. Check completion within the intended service startup limit, not merely the diagnostic's longer timeout. Retain actual interpreter/import paths, source/binary identity for any relocated runtime, phase times, stacks, process disposition and allocation cost. No new implementation or submission is part of this review.

The parent's provisional local-copy route is a reasonable hypothesis to qualify on CPU if pursued. A separately authenticated copy must preserve original evidence and expose any mixed original/copied import paths. Neither relocation nor warming a shared cache has yet been demonstrated to solve startup. A warm-cache success alone does not qualify the intended launch layout. Preserve a new failure rather than automatically retrying or enlarging a GPU reservation.

Remaining caller boundary: reconcile current jobs/journals/budgets and readiness before any new submission; review and pin any new preparation/launcher changes; establish bounded startup in the intended runtime/layout; and separately qualify model loading, generation and actual GPU work with the required independent activity signals before a later GPU run. This review accepts diagnostic custody and conclusions, not a causal fix, full installed-environment provenance, scientific performance or future resource authorization.

## Reviewed pins

All values are SHA-256.

| Artifact | Pin |
| --- | --- |
| `terminal02/disposition.json` | `80720261564b76abf2bc267a3aa057ec0146272718f00de12e648699623ac35e` |
| `terminal02/inventory_transport.json` | `f636749bdddfe323e5616e1bf849d8308daac4d86841b7590667d7e050fc9f22` |
| `remote/probe.py` | `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951` |
| `remote/job.sbatch` | `9d9f230db70214ef85283e634dbc5e99b4ac84a3a58bc866a984370c08b98ca3` |
| `remote/packet.json` | `546ffe5d314e790339b38bc86521b92992d56e481057a70a00a07780210a9057` |
| `remote/submission.intent.json` | `1e56a2c323a52caefdea2796b760285dcbaaaf51e02456ca152d4ad1c7f0e1ff` |
| `remote/diagnostic/diagnostic_complete.json` | `20bd39d3fc0a6ef82610d1e8246a27fe5e9869e8a5948403ebc4f8c7f1f34add` |
| `remote/diagnostic/matched_import_order/failure.json` | `0bdafc71fc60f19746d21f33eda1dc9cb48b271bb749bc61363efe204f919da6` |
| `remote/diagnostic/scipy_first/terminal.json` | `b9f680e6c6ec9d33ae4bb073675981c85e30e7229540e6988def6326657767bb` |

The inventory pin above binds all raw files, including the full failed/successful stderr and stack logs, phase records and selected installed-source contents. The sole file authored by this review is this review document.
