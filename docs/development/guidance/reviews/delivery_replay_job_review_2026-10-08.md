# Bounded prelaunch replay job review — 2026-10-08

**Disposition: changes required. Four required findings in the new job integration.** This is a static review, not replay qualification or launch authorization.

## Scope and method

Reviewed the new `supervise_replay.py`, `freeze.json`, and `replay.sbatch` in `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-replay-job-20261008-01`. Read the unchanged `scripts/research/runner_delivery_confirmation.py` only to establish the imported `utc()`/`supervise()` behavior, and the packaged `replay_delivery_prospective_release.py` only to establish its launch, authentication, runtime, receipt, and inference interfaces. The local package counterpart is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-anonymous-B-candidate-20261008-01`.

Inspected local text and calculated SHA-256 digests. A standard-library-only metadata read compared the manifest entries, registered dependency map, and original acceptance digest; it did not execute any reviewed module. No remote commands, submissions, launches, inference, model loads, tests, installations, or component review reruns occurred. No review agents were spawned. The only write is this report. Main owns final original phase-cost and attempt disposition; those are outside this review.

Original Stage B completion and accepted audit are supplied context, not independently re-audited here. Local manifest metadata contains 3,890 file entries and 5,065 bindings, matching the supplied package description. Transfer completion, expired SSH, unverified remote bytes, and no replay submission are supplied operational context. This review establishes neither remote custody nor successful replay.

## Required findings

### R1 — [P1] Authentication and resource gates disappear under Python optimization

`supervise_replay.py:4–9` implements every digest and configuration rejection with `assert`: freeze, worker, manifest, original supervisor, CPU count, account, RSS, and wall values. `replay.sbatch:13–16` inherits the Python environment and supplies only `-B`; it neither rejects `PYTHONOPTIMIZE` nor makes these checks independent of optimization. A submission inheriting `PYTHONOPTIMIZE=1` removes these checks. The wrapper then compiles and executes the captured supervisor bytes at line 10 without validating their digest, and proceeds with the unchecked freeze.

This is a static consequence of Python assertion semantics, not a tested mutation or evidence that the current environment is optimized. `-B` disables bytecode writes; it does not preserve assertions.

**Required correction:** replace authorization, digest, and allocation assertions with explicit comparisons that raise or return a recorded failure. Clear or reject inherited optimization settings as an additional launch constraint. Fail before executing supervisor or package code on any mismatch.

### R2 — [P1] The manifest pin is not used to authenticate executable entry points before launch

`supervise_replay.py:8` verifies only `manifest.json`; lines 12–13 then launch the on-disk replay CLI by path. The manifest already contains the CLI and verifier digests, but the wrapper does not compare their bytes before execution. The CLI imports `verify_delivery_release` at its line 23, before `replay()` verifies package bytes at lines 258–269. Thus an altered CLI can bypass its own checks, or an altered verifier can execute import-time code before the package check, while the manifest itself still has the expected digest. A substituted CLI could write a receipt and exit zero; the wrapper accepts receipt existence without checking its contents at lines 15–17.

There is also no independent worker check in `replay.sbatch:16` before Python executes `supervise_replay.py`. Its self-hash is useful for detecting later disk disagreement under a trusted worker, but an entirely substituted worker need not perform that check. In contrast, the original supervisor integration captures, hashes, and executes the same bytes at wrapper lines 9–10, subject to R1.

**Required correction:** establish a trusted, independently pinned launch bootstrap that authenticates the freeze and wrapper before executing the wrapper. Have the trusted wrapper authenticate the packaged CLI and its immediate verifier import against the pinned manifest before launching either. Bind execution to the authenticated snapshots or enforce a documented immutable execution directory so a check followed by reopening a mutable path does not lose custody. This can wrap the unchanged components; it does not require re-reviewing or editing their scientific implementation.

The pending remote byte verification remains a separate necessary prelaunch gate. Transfer completion alone does not close it. This finding concerns the new launcher’s missing executable checks, not a claim that remote files are corrupted.

### R3 — [P2] Recorded account and resource declarations are not reconciled with the allocation

The default `replay.sbatch:3–10` explicitly requests the intended account, CPU partition/QOS, one node/task/core, 3 GiB, and 900 seconds. However, wrapper line 5 checks only `SLURM_CPUS_PER_TASK`; line 6 compares constants inside the freeze. Line 14 records `account=reg['account']` rather than establishing the actual allocation’s account. An account override at submission can therefore run the job under a different account while its execution record still says `ucb736_asc1`. The same integration does not reconcile the actual task/node count or time/resource overrides with the frozen reservation.

**Required correction:** reconcile the allocated account and resource envelope with the freeze before the child starts, using scheduler-provided identity/resource fields and/or a separately authenticated submission/allocation receipt. At minimum check the actual account and single CPU task, and establish the frozen node, wall, memory, and zero-GPU allocation basis. Reject unsupported overrides; record verified allocation values separately from requested values. The present script defaults are correct, but they do not establish what was actually allocated.

This review does not recalculate main’s original reservation total. `0.25` reserved core-hours is consistent with this script’s default request of one core for 900 seconds; it is not a measured process-CPU value.

### R4 — [P2] Wrapper failures can leave no terminal record or a misleading successful status

After the exclusive start marker is created at wrapper line 11, `m.supervise()` and result/output handling at lines 13–16 have no exception disposition. For example, a child-spawn `OSError` propagates from original supervisor lines 125–127, leaving a start marker and Slurm traceback but no `replay_execution.json`. A pre-existing execution file is discovered only by the final `open('x')`, after child work has already occurred. The start marker still blocks a duplicate attempt, but it does not explain the terminal failure.

Separately, when a child exits zero without producing a receipt, the wrapper writes `status='complete', exit_code=0` to `replay_execution.json` and only then exits one at line 17. That terminal JSON describes success while the wrapper declares failure; no failure reason is attached. An existing receipt is handled by the CLI’s rejection, but the wrapper does not validate a newly emitted receipt’s required success fields.

**Required correction:** check existing terminal/receipt artifacts before child work and preserve them on rejection. Once attempt custody is acquired, record wrapper/launch exceptions with stage, reason, child status when available, and job/freeze identities. Validate the child receipt and reconcile missing, malformed, or non-success receipts into the terminal status before writing it. Preserve the exclusive start marker and logs. Document that scheduler termination or other uncatchable interruption requires final scheduler/log reconciliation; do not classify absent execution evidence as success or silently permit a replacement attempt.

## Checks that agree at the reviewed snapshot

- **Exclusive replay custody:** `replay_started.json` is opened with `x` before the child starts. Two callers using this frozen output path cannot both pass that operation. The CLI separately requires an exclusive receipt outside the package. Final execution output also uses `x`. R4 concerns incomplete terminal disposition, not an observed simultaneous replay.
- **Supervisor integration:** the captured original supervisor SHA matches both the repository file and the local frozen bundle copy. Its module has a different `__name__`, so its original `main()` is not invoked. Only `supervise()`/`utc()` are called. It creates a new child process group, samples child-tree RSS, kills that group for RSS/deadline/telemetry failure, and returns status, exit code, and samples. No original acquisition/fitting worker is called.
- **CPU/RSS/wall defaults:** Slurm requests one CPU and no GPUs. Parent and child set the three numerical-library thread variables to one and hide CUDA devices. The child receives a 3 GiB RSS threshold and an independent 850-second watchdog. The watchdog starts after wrapper preflight, while Slurm’s 900-second limit covers the entire job; startup/finalization consuming the nominal 50-second margin could therefore leave an interrupted attempt requiring reconciliation. No run-derived resource measurements are claimed here.
- **Runtime/source/input contracts:** the six `freeze.json` dependency versions match the pinned packaged replay contract and CLI constants. The CLI checks distribution versions before outcomes/model inference, then package byte membership and imported module version/origin. Archived learner/helpers, response snapshots, model bytes, and prediction bytes are checked through its existing authenticated interfaces. These are downstream safeguards conditional on trusted entry-point execution; they do not close R2. The interpreter path in Slurm matches the freeze; a path alone is not an independently measured runtime qualification.
- **Original acceptance link:** the independently frozen original acceptance digest agrees with the manifest-bound replay contract’s upstream gate. The wrapper does not re-audit original acceptance; the CLI consumes the existing accepted projection through its unchanged gate.
- **Zero fitting/response scope:** the command invokes only the supplemental replay CLI. That interface loads saved weights on CPU and compares inference with cached predictions, rather than invoking the original fit/acquisition functions. Its receipt declares zero optimizer updates and zero new responses. The CLI has an offline audit hook and emits per-world progress. The wrapper’s declaration of zero fits/responses is consistent with this reviewed source path, conditional on the authentication fixes; no inference was executed to verify it.

## Reviewed SHA-256 values

Job directory `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-replay-job-20261008-01`:

- `supervise_replay.py`: `5a12e644dfc3797250f9d5ad2c12e879e452c1c547b485ea3cfa6450a4635bc3`
- `freeze.json`: `08ff1863008a456a240f7474e0287fc36c8b23a6f1e07aa837be6cfcf0e816fa`
- `replay.sbatch`: `dcda8887d4956b963a4423e36250da22b1caccf300d49ace2abf4fe7ae089172`

Integration counterparts:

- `/Users/pat/code/ACE/scripts/research/runner_delivery_confirmation.py`: `2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee`
- `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/project/scripts/research/runner_delivery_confirmation.py`: same SHA-256 as above.
- Packaged `replay_delivery_prospective_release.py`: `80dc80239b403cf2b443987fb92640ea0eec938876d14ac6958c040bc8c7c1c3` (also matches the repository counterpart).
- Packaged `verify_delivery_release.py`: `9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52` (immediate import boundary only).
- Packaged `B/replay_contract.json`: `7375f7fe0837197eee8c43decb1c73ffcb74bd6c956367c10323558ffd98c35d` (runtime/original-acceptance metadata comparison only).
- Packaged `manifest.json`: `f5415e24cdb1eaf353c8e8fb3bb0f3510b3b2d476174bbc1dd8c7d3655a1c2da`, matching the freeze pin.
- Original acceptance pin recorded in the freeze and packaged gate: `dda7a9a4414524ed74e6e8955718c4be93177e452f68dcc961984a521060a3cb`; original acceptance contents were not opened or re-audited.

Package-relative entries above refer to the local package counterpart identified in Scope. The job’s declared `source_revision` is `4a1f954f89bf6ec529aae0488c1320faa915b50d`; its separately authored worker has no Git revision and is bound by the listed digest. No remote digest equality, historical revision closure, numerical replay result, or final attempt-cost disposition is inferred from these local hashes.

## Bounded static correction recheck of job 02 — 2026-10-08

**Disposition: R1, R2, and R4 are closed in the reviewed successor; R3 has one remaining required fix concerning absent GPU allocation evidence.** This disposition applies only to `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-replay-job-20261008-02`. It does not approve submission or retroactively change the predecessor’s disposition.

Rechecked only the corrections to the four findings through local source inspection and hashing. Read the unchanged packaged CLI’s receipt emission and verifier’s return/import interface solely to check the new captured-byte bootstrap and receipt validation. No tests, remote commands, launches, model loads, inference, installs, or repeated component audits occurred. Main’s concurrently prepared fabricated launcher regressions were not run or independently evaluated here. Only this existing report was appended. Independent hashes confirm predecessor 01’s three reviewed files remain unchanged; its unsubmitted status is supplied context.

### Closed corrections

- **R1:** `supervise_replay.py:7–12` now uses explicit `require()` failures. Its main rejects optimized Python at line 53. `replay.sbatch:14,17` clears `PYTHONOPTIMIZE`, `PYTHONPATH`, and `PYTHONHOME` and starts isolated Python with `-I -B`. The child also uses `-I -B`. The reviewed rejection gates no longer depend on assertions.
- **R2:** `replay.sbatch:20–25` independently compares worker and freeze bytes against embedded SHA-256 pins and executes the captured worker bytes. The wrapper rereads and authenticates the freeze before decoding it. At wrapper lines 67–73, the pinned manifest supplies CLI/verifier digests, and the wrapper captures and checks both sources and the unchanged supervisor. Child arguments contain base64-encoded authenticated CLI/verifier snapshots. The child bootstrap at lines 30–38 installs the verifier in `sys.modules` before executing captured CLI bytes, so the CLI’s existing verifier import uses that authenticated module. The unchanged verifier’s disk/source comparison and later package integrity checks remain additional fail-closed checks. No package CLI or verifier source is executed before the new wrapper authenticates it. This closes the identified entry-point gap; trust in the independently pinned Slurm script and pending remote transfer verification remains a prelaunch prerequisite.
- **R3, corrected portion:** wrapper lines 16–28 query `scontrol show job` with a timeout, bind the returned job identity, and explicitly compare account, CPU/task/node counts, QOS, partition, running state, wall time, and parsed node memory. Relevant task/node environment fields must also agree. The raw scheduler response, its hash, and verified fields are retained, and the recorded account comes from that response. The previous requested-account substitution is fixed. A missing required identity/CPU/memory/time field rejects rather than being silently accepted.
- **R4:** wrapper lines 58–60 reject pre-existing start/execution/receipt/log artifacts before child work and then acquire the exclusive start marker. Lines 61–84 initialize a failure disposition, catch allocation/authentication/spawn/child/receipt exceptions, and preserve any returned child supervision result. Missing or invalid receipts and failed children produce `wrapper_failure` with stage/type/reason rather than a successful terminal record. Receipt validation at lines 41–49 checks the existing CLI’s success flag, integer counters, dependency map, manifest pin, and 3,890/5,065 inventory values. The exclusive terminal record includes finish time and wrapper elapsed time; failures exit one. The previous complete-JSON/nonzero-wrapper contradiction is removed. Hard termination or failure to write the final file still requires scheduler/log reconciliation, as described in the original review.

The child deadline is now `started + 850`, so allocation and executable preflight consume the same wrapper budget rather than receiving an additional 850 seconds afterward. The successor uses a new exclusive output directory while retaining the existing pinned input package. No new fitting or response path was added by these corrections.

### Remaining R3 correction — [P2] Missing allocation evidence passes the zero-GPU check

`supervise_replay.py:25` searches for `gpu` in `AllocTRES`, `TresPerNode`, `TresPerTask`, and `Gres`, but each absent field defaults to an empty string. None of these fields is required elsewhere. A scheduler response containing all the accepted identity/CPU/task/node/time/memory fields but omitting these four allocation fields therefore passes this check and is recorded as verified allocation, with `AllocTRES: null`, even though GPU allocation evidence is unavailable. An empty or uninterpretable non-GPU `AllocTRES` value is likewise not validated. This follows directly from the inspected expression; no fabricated response was executed during this recheck.

**Required correction:** require an authoritative, present, well-formed allocated-resource record for this running job before inferring zero GPUs. For example, require and parse a complete `AllocTRES` field, reconcile its CPU/node/memory values with the other checked fields, and reject any allocated GPU resource. Optional request/GRES fields can remain optional; their absence must not substitute for the actual allocation record. If the site omits the authoritative field, reject as allocation telemetry unavailable and use an independently authenticated equivalent rather than interpreting missing evidence as zero. The existing post-marker exception handler will then preserve that rejection as an allocation-preflight failure.

This remaining finding does not assert that job 02 requests GPUs or that any remote allocation exists. Its Slurm directives request CPU resources only. It concerns the explicitly required verification of the actual allocation and is the only remaining required fix found in this bounded recheck.

### Successor hashes and limits

SHA-256 of job 02 files read:

- `supervise_replay.py`: `daf9f09b0f94c32cb17b43905a43a706238318769683f7b4ea93baa6501c914f`
- `freeze.json`: `e9b4799117de37ab74133f41400cfc015de2771fd971c1b063a721e28c438126`
- `replay.sbatch`: `b8fc3dd7af74c2315cb8cd90d886851c6a0d302f5241cb10cb8eaa2349466d1d`

The successor’s embedded worker/freeze pins match these local files. Its manifest and supervisor pins remain those listed in the original review. Independently rehashed packaged CLI/verifier sources remain `80dc80239b403cf2b443987fb92640ea0eec938876d14ac6958c040bc8c7c1c3` and `9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52`. This static recheck establishes source correspondence, not real CLI execution, site-specific scheduler output compatibility, target-runtime qualification, remote byte equality, numerical replay success, or final original phase-cost/attempt disposition.

### Reconciliation with main-reported fabricated launcher checks

Main reports six methods passing in 0.149 seconds, with evidence in external job 02 `test_launch.py` and `tests.log`. Reported coverage includes explicit rejection under `-O`; nine wrong scheduler-field subcases; captured CLI/verifier execution with deliberately substituted disk files; six receipt rejection cases; terminal dispositions for child-spawn `OSError`, missing/malformed receipts, and valid success; and refusal of a pre-existing terminal artifact. Main reports no real CLI, model, scheduler, or inference invocation. This reviewer did not run or independently inspect those test artifacts; these are attributed results, not additional static-review measurements.

These reported checks support the R1/R2/R4 closures and the corrected account/resource portions of R3. They do not establish rejection of missing or uninterpretable allocated-resource evidence, and the inspected job 02 expression still defaults absent GPU-related fields to empty strings. The remaining R3 finding therefore stands at the recorded hashes. No additional required issues were identified in this bounded correction recheck. Scientific workers, original acceptance, phase-cost accounting, and historical attempt disposition were not re-reviewed.

### Actual verifier import smoke evidence

Main additionally reports a minimal import smoke using the actual authenticated verifier through the captured-byte child bootstrap and a fabricated CLI that prints only `VERIFIER_SHA256`. This reviewer read, but did not rerun, `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-replay-job-20261008-02/verifier_import_smoke.json` (SHA-256 `a33f0a33b9ba369b9a29077d0bfa207c64234dcf6a74a72e7822666468628618`). The record has return code zero, empty stderr, and stdout equal to the reviewed actual verifier digest `9f80286faeb3266c53bba70a7cf2abc26c4129c61a56c265c263cb09a8f00d52`.

This supplies narrow execution evidence resolving the captured-source compile-frame compatibility concern for the verifier’s import-time check under the new bootstrap. It does not run a verifier package audit, the real replay CLI, model loading, inference, or scientific-value processing; the record reports zero new fits/responses. It supplements the static R2 closure without establishing numerical or target-runtime replay qualification. The remaining R3 allocation-evidence finding and all other scope limits remain unchanged. Only this report was appended.

## Final bounded static R3 recheck of job 03 — 2026-10-08

**Final disposition: R3 is closed. All four original findings are closed for successor job 03; zero remaining required fixes within this bounded review.** This supersedes the remaining-R3 disposition for the successor, preserving the earlier reports and predecessor findings as history.

Reviewed only the new allocation-evidence correction in `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-replay-job-20261008-03` and its regenerated freeze/script pins. Compared job 03 with job 02 locally: the worker change is confined to the allocated-resource check; freeze/script changes bind the new worker, exclusive output directory, freeze digest, and predecessor provenance. The prior authentication, custody, supervision, and terminal-failure corrections are retained. Rehashed job 02’s three files match its previously reviewed hashes exactly. No tests were rerun, no remote commands or launches occurred, and no scientific workers or historical accounting were re-reviewed. Only this report was appended.

At `supervise_replay.py:25–34`, `AllocTRES` is now mandatory. An absent/empty value or whole-record null/N/A/Unknown value rejects. Each comma-separated component must contain exactly one nonempty key/value pair; duplicate keys reject. The parsed record must explicitly contain `cpu=1`, `node=1`, and memory equal to 3 GiB through the existing strict M/G parser. Missing or invalid CPU/node/memory evidence therefore fails before child execution. Any allocated resource key beginning with `gres` rejects, including GPU type keys; the optional GPU request/GRES strings remain additional checks. This establishes zero allocated GPUs from the authoritative scheduler TRES record rather than defaulting missing evidence to zero. Failures remain inside the existing post-marker allocation-preflight exception disposition. The original account, identity, task, node, partition, QOS, running-state, and wall checks remain in place.

Main reports one focused scheduler test passing 15 negative subcases, including six new cases for missing/unknown allocation evidence, missing/wrong memory, duplicate keys, and a GPU-type resource string. This is attributed test evidence, not independently rerun verification. The earlier six-method fabricated launcher suite and actual-verifier import smoke were not repeated by this reviewer. Static inspection is sufficient to close the specific missing-allocation-evidence defect; compatibility with an actual scheduler response remains a runtime prerequisite.

Job 03 SHA-256 values:

- `supervise_replay.py`: `d7067a563c42a1a315f0156901adc4149d0e32f367d2d5cc8c5667732ce47466`
- `freeze.json`: `41bf3ba3627b644667cdb20adabe93d36021cd804165be7af9e15e4457884a6f`
- `replay.sbatch`: `7eddb0b2dbb38ac86b7b576ad52a192d8575b8b435292bc7d25c386c8d55acbf`

The script’s embedded worker/freeze pins and the freeze’s worker digest match the independently hashed local files. The freeze retains the previously reviewed package-manifest and original-supervisor pins and binds job 02’s unchanged freeze as its unsubmitted predecessor. Main reports no job 03 transfer or submission. Closing these static findings establishes neither remote byte custody nor numerical replay/target-runtime qualification, submission authorization, actual allocation measurements, or final original phase-cost/attempt disposition.
