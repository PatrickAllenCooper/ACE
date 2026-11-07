# Independent staged-runtime caller review

November 7, 2026. **One required finding, P2. Do not treat the current caller as ready for launch until R1 is corrected.** The archive helper has a separate review and is outside this disposition. Its findings are not counted again here.

This review inspected the new caller, its three fabricated methods, the preparation01 receipt and stderr, the staged-runtime guide, and the retained original import probe. The original service authentication interface was read only to check the caller's dependency on it. No tests, reviewed source, models, real runtime, GPU work or remote commands were executed. No commit was made. Only this review document was written.

## Required correction

### R1. Start the outer child deadline at launch, before receipt I/O

At [caller lines 136 onward](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_staged_import.py:136), the parent records `child_start`, starts the child, then calls `write` for `worker.started.json`. That write performs a flush and `fsync`. Only after it returns does the parent call `proc.wait(timeout=185)`.

The wait therefore gives the child a new 185 seconds after the receipt write. For example, if the write takes 20 seconds and the child remains stuck in a native import, termination is not requested until roughly 205 seconds after launch. If the receipt write stalls, the only parent alarm in force is the overall 1,740-second alarm. The unchanged probe's 210-second alarm does not enforce the shorter caller limit and may itself be delayed during native loading, as the preceding diagnostic already showed.

The later `elapsed > 180` check correctly prevents acceptance of an overlong returned child. It does not enforce the promised 185-second outer termination deadline. This matters specifically in a qualification intended to diagnose slow shared-filesystem startup.

Required change: establish an absolute monotonic child deadline at launch and enforce it while startup-receipt I/O is in progress, not just while waiting for the process. Use the remaining duration for any subsequent wait. Preserve the earlier overall preparation deadline and the existing owned-group cleanup and failure evidence. A bounded watchdog or carefully managed shorter parent timer can implement this. Do not reset either deadline after recording a receipt. Keep the five-second cleanup allowance distinct from the instant at which termination is requested.

Add a focused new fabricated supervisor check in which the post-spawn receipt operation consumes time. It should verify that the original child deadline is not extended and that timeout leads to owned-group cleanup and failure, never a success terminal. No real runtime or remote job is needed for this regression. The three existing methods do not cover this path and need not be repeated merely to accumulate checks.

## Other inspected behavior

The scheduler gate queries `scontrol` for the supplied Slurm job ID and retains its stdout, stderr and return code before checking the allocation. It requires `ucb736_asc2`, `acpu`, `cpu-normal`, one CPU, one task, one node, `00:30:00`, no requeue, `cpu=1`, `mem=3G` and no `gres/` entry in AllocTRES. The proposed resource values are consistent with the stated CPU-only scope. This source inspection does not verify an actual allocation or the separate launch packet.

The proposal arithmetic is correct: 4,763 closed allocated CPU-seconds plus the cumulative 3,600-second preparation allowance plus the proposed 1,800-second cap equals 10,163 of 57,600 CPU-seconds. GPU use remains 836 of 7,200 GPU-slice-seconds, with no additional GPU reservation proposed. These figures come from the current ledger and proposal. They are not new scheduler measurements or whole-cycle process CPU.

Packing and staging execute before spawning the copied interpreter. Phase records include UTC, monotonic elapsed time and the supervisor's process CPU. Pack completion binds the archive manifest and counts. Staging returns its own record and a stage-complete event. Phase differences support separate packing and staging timing. The supervisor's `process_cpu_seconds` excludes the child's CPU. The original probe retains its own process CPU and RSS on ordinary success or caught failure. A child killed before writing its receipt still requires scheduler accounting and retained logs. No speedup or complete CPU total can be inferred from these fields alone.

The original probe is read, hashed and executed from the captured bytes. The copied worker invokes its `worker` function once with `mode='matched_import_order'`. The hook changes terminal writing to collect origin evidence. It does not replace the authentication calls or reorder the named transport, Torch, Transformers, Qwen and project-helper imports. No SciPy-first branch is selected. Authentication still reads and hashes the existing model files. That is model-file verification, not model instantiation, training or inference.

The copied worker requires the actual interpreter and `sys.prefix` to resolve under the intended staged prefix. It removes `PYTHONHOME`, `PYTHONPATH` and `LD_PRELOAD` from the child environment, and sets `LD_LIBRARY_PATH` to the staged library directory. Before an original successful terminal can be written, the hook records loaded module paths, `sys.path` and `/proc/self/maps`. It rejects resolved module paths under the original runtime, module paths containing `site-packages` or `lib-dynload` outside the staged prefix, and native mappings whose recorded path contains the original runtime prefix. These are scoped observed-path checks. They are not a proof that every external library, alias, module search path or future dependency is contained in the archive.

The child uses `start_new_session=True`. The existing `kill_owned` routine targets its process group, reaps the leader with a bounded wait and probes for remaining group members. Cleanup is also attempted after a normal leader return. Qualification rejects recorded cleanup errors or a surviving group. The main exception path attempts cleanup if no cleanup receipt exists, then writes failure evidence. It does not retry the child.

Success requires a zero child return code, parent-measured elapsed time no greater than 180 seconds, clean owned-group termination, the original terminal and the origin record. The original terminal must state completion, no model weights loaded, zero training updates and zero environment responses. The final caller terminal explicitly leaves model-service qualification false and automatic GPU retry false. These conditions support a narrowly described CPU import qualification once R1 and the separate helper findings are closed. They do not qualify a GPU service or scientific study.

Some bootstrap failures occur before the caller's output directory and main exception handler, including a rejected probe pin. A scheduler interruption or filesystem write failure can also prevent a complete terminal record. The separate launcher must preserve stdout, stderr, scheduler identity and accounting, and treat a missing or conflicting terminal as failure. An exit code or one partial receipt is insufficient for acceptance. That launcher is not reviewed here.

## Optional diagnostic improvement

Origin and process-map capture currently runs only on the attempted successful terminal write. A failing or timed-out import therefore has stack and phase evidence but may lack the complete origin snapshot most useful for diagnosing a mixed runtime. A best-effort failure-path snapshot would improve diagnosis, provided it does not delay cleanup, replace the original exception, or permit success after a failed origin check. This is not an additional launch blocker for the stated fail-closed acceptance rule.

## Evidence and review boundary

Preparation01's recorded stderr reports three methods passing. The receipt records 0.058243 child CPU seconds, 0.062811 elapsed seconds and 20,873,216 bytes of peak child RSS. The methods cover the exact allocation and seven altered fields, rejection of a mismatched interpreter prefix, and rejection of a wrong source pin before executing that source. They do not exercise a successful copied worker, origin-hook failure, the supervisor's timeout, process-group cleanup, receipt validation, or a real staged runtime. Those author-run measurements were inspected, not repeated.

Reviewed SHA-256 identities:

- Caller: `c18e955482add513f7dfedc85f573bc7ec480e02779133fe1665e732b0ba0fc4`.
- Test file: `05bc14d1abe0462dcae5afa33861960382992e3c26bddffb981b6f6e71084136`.
- Guide: `e80f89e47f9acb012d9b79b36846815ed26a381ba96986b31df2dec79f22c261`.
- Original launch02 probe: `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951`.
- Preparation01 receipt: `45b45388436881417d2db8cee84b3091518c3fb76350d76ae44e4a6cab99d4c0`.

The caller and test identities match the preparation receipt. This disposition applies to these bytes. It must not be relabeled as acceptance of a later correction. The separate archive-helper review, exact launch inputs, once-only submission handling, live resource admission and actual execution remain distinct gates.
