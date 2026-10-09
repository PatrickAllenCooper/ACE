# Foundation component boundary recheck

Date: 2026-10-09. Read-only static recheck of the five findings in the preceding component review, concentrating on the added execution boundary. Only this new review was written. No reviewed modules or models were imported; no tests, smokes, installs, or scored runs were executed. The five passing tests are user-reported; their source was inspected, not rerun.

## Exact reviewed inputs

- `scripts/research/foundation_component_pilot.py` (318 lines): SHA-256 `811747bf1581c2a11827295d367f00b8fd4857ffc1916569a27e39c54827cb26`.
- `scripts/research/supervise_foundation_component.py` (116 lines): SHA-256 `c0c8fac08e88f6084438f829c95e97d640512185dba1a9179feae731fd9ff486`.
- `scripts/research/test_foundation_component_supervisor.py` (61 lines): SHA-256 `c5f5a00abd248bcba2d154eb34fb812d100ee0e2e7b674bb276b6af34b904676`.
- `docs/development/guidance/ace_foundation_component_pilot_2026-10-09.md` (40 lines): SHA-256 `c86f9b5a63161d72fb617a4a7dd4b37a8c6634ad23ae8aa0b76dec0009943da0`.

Hashes were checked again after reading. HEAD at recheck was `1044e4480c1c9d6ccbf13d3eb804863b3c01b111`; the worker/protocol were modified and supervisor/tests untracked. These hashes, not HEAD, identify the reviewed implementation. No concrete launch freeze or smoke receipt was supplied for artifact-level validation.

## Required findings

### B1 — SIGTERM bypasses cleanup and releases the lock while the child survives (P1; before either smoke)

`supervise_foundation_component.py:62,73–81,111–113`.

The child starts a new session, but the supervisor installs no SIGTERM handler. Default SIGTERM termination does not raise a Python exception or execute this `finally`. A normal controller cancellation directed at the supervisor can therefore leave its separate child process group running, omit the terminal ledger/resource receipt, and release the controller lock. Another attempt can then acquire that lock while the first child remains alive. The child's CPU limit does not restore the lost wall deadline if it is blocked or sleeping.

Handle catchable controller termination explicitly: preserve the cancellation reason, terminate and reap the owned child/group, publish terminal accounting, and retain the lock until cleanup ends. Include other catchable termination signals the actual launcher uses. State the limitation for SIGKILL/host failure rather than claiming cleanup on literally every terminal path; this finding does not demand that Python catch uncatchable signals.

### B2 — The independently checked freeze is reopened without carrying its trust anchor to the worker (P1; before treating a launch as frozen)

`supervise_foundation_component.py:42–45,58–60`; `foundation_component_pilot.py:204–230,236`.

The supervisor hashes the freeze path and then separately rereads that path for parsing. The worker later reopens it again, but receives no expected freeze SHA. Its `started.json` hashes the path yet again. A replacement/edit between these operations can make the worker validate different dependency, protocol, or model-file pins from the freeze accepted by the independent launcher pin, while `launch.json` still records the original SHA. Complete seven-file and nine-package checks do not establish that they came from the approved freeze bytes.

Read once, hash and parse the same bytes in the supervisor. Hand off those verified bytes through an immutable attempt snapshot or pass the independent expected SHA and have the worker hash and parse one captured byte sequence before using any field. Record that verified SHA, not a later reread. Reject mismatches before model loading. This is a concurrent-edit consistency defect; it is not an allegation that a current artifact was modified.

### B3 — Terminal success ignores supervisor errors and trusts completion-file presence (P2; before using smoke receipts as gates)

`supervise_foundation_component.py:73–100`.

`passed` is only `code == 0 and complete_path.is_file()`. It does not require a clean supervisor outcome or inspect the completion payload. For example, an exception/KeyboardInterrupt after the child has written its completion marker and exited but before the normal reap can be recorded in `error`; the `finally` reap can nevertheless obtain exit zero and publish terminal `status: complete`, returning zero. Invalid cell JSON is also labeled `invalid_record` without affecting that acceptance decision.

Require an error-free, noncancelled, non-timeout supervisor outcome for an accepted smoke. Validate the mode-appropriate completion record and reconcile its planned identities/statuses with the terminal ledger. Keep execution completion distinct from scientific success: a pilot may finish with explicitly recorded failed cells and undefined aggregates, as the protocol allows. Do not silently promote an invalid record or supervisory failure to an accepted receipt.

### B4 — The executable gate does not distinguish compatibility freezes from a qualified pilot freeze (P2; before any scored pilot)

`supervise_foundation_component.py:36–48,58–60`; `foundation_component_pilot.py:200–230,238–255`; protocol final paragraph.

The mode comes solely from CLI arguments. Neither component requires a freeze stage/allowed mode, successful smoke receipt pins, or the committed-source provenance required for the final scientific freeze. A freeze valid for either technical smoke can therefore be reused with `--mode pilot` and reach `world()` without establishing those additional prerequisites. Merely adding unused fields to the planned freeze would not enforce them.

Bind stage/mode in the freeze and validate it. Before pilot execution, require and verify both successful compatibility receipts against their relevant frozen worker/model/runtime identities and the declared committed source. Compatibility mode must remain executable before those receipts exist. This correction does not require running a pilot, expanding the smoke budget, or adding model-performance thresholds.

## Disposition of the preceding five findings

1. **Elapsed-time enforcement:** the independent polling supervisor now covers blocking worker imports, preflight, model loading, and cells at 120/1800 seconds. B1 remains a real cancellation escape; acceptance of terminal outcomes also needs B3.
2. **Failure ledger/publication:** the prior plan, per-cell start records, serialized-before-publication JSON, and reconciliation materially fix ordinary preflight failures and interrupted cells. Finite metrics are now checked. B1/B3 are the remaining boundary defects identified here.
3. **Freeze completeness:** explicit failures replace assertions; required package names, the exact seven-file set, protocol pin, checkpoint hash, and revision declarations are checked. B2 concerns binding these checks to the independent pin; B4 concerns the final pilot qualification gate. Actual launch pins and provenance were not supplied and are not certified by this review.
4. **Resource evidence:** whole-child `wait4` accounting, explicit RSS conversion, separate supervisor CPU, and the worker phase label address the previous accounting mismatch on handled paths. Retain the stated single-process trust model. B1 must preserve this evidence during controller cancellation.
5. **Summary definitions:** resolved for this development protocol. Comparator, ratio direction, arithmetic/geometric formulas, all-six eligibility, zero/nonfinite/failure handling, and labeled language fallback treatment are now specified. No further statistical or information-matched language comparison is required.

The existing five fixtures cover preflight failure, wall timeout, preservation after child failure, nonfinite serialization, and exclusive publication. They do not cover B1–B4. Target any further boundary checks at those cases; a broad scientific re-review or model execution is unnecessary to resolve them.

## Implementer disposition after bounded reviewer follow-ups

B1–B4 corrected before scoring. Deferred cancellation now avoids asynchronous exceptions across Popen ownership or wait4 result publication; SIGTERM/HUP/INT set a flag checked by the supervisor. Independent reviewer final B1 recheck reports zero required remaining, supervisorSHA629f3a35553aa46e3b212bc426d4147e7348dd380c720aa830545719d41ab8cf. Earlier final recheck addressed B2–B4, leaving only this transition correction. Eleven focused supervisor fixture methods pass; no models are used in these tests.

Smoke01 executed normally before the boundary recheck completed. Both ran compatibility inputs only, with clean original terminal receipts. They remain predecessor evidence and are NOT used to qualify the final pilot because worker/supervisor/protocol bytes changed. Preparation02 was never executed. Smoke03 qualifies the corrected bytes if its own gates pass. No scored worlds existed during these repairs.

A distinct reporter review found missing/nonfinite fields and then null/non-mapping metric containers could abort reporting. Corrected with guarded extraction and JSON-safe diagnostic preservation; zero/nonfinite/missing cases leave all-six aggregates undefined. Five fabricated reporter test methods pass. Independent final narrow recheck reports zero required issues. No scientific scores were used to select these reporting rules.
