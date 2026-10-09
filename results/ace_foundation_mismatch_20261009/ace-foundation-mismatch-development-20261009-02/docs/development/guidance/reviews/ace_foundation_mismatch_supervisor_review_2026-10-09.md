# Foundation mismatch supervisor/source review

Date: 2026-10-09. Bounded static review of the new supervisor and the worker's source authentication, initialization, main entry point, and response journals. No model imports, tests, fixtures, scientific runs, or private outcomes were used. Only this review file was written.

## Exact reviewed inputs

- `scripts/research/supervise_foundation_mismatch.py` (172 lines): SHA-256 `a3f6e02fd35e26fe4257684c0768ac1bed3b09ea291a0618abd54e04d32c8243`.
- `scripts/research/foundation_mismatch_pilot.py` (233 lines): SHA-256 `7d50891299fda616f9ac9befa3852c76027af7d65fc78c2dd1da6322ba4a586c`.
- Protocol resource/reporting excerpts: `docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md`, whole-file SHA-256 `92de4a08e7a933bedb69fe3c2993b327c84ddba4ee6111f6511496f5805e5318`.

The two new Python files were untracked. HEAD observed during review was `ea43efcc3a2480a5e61ffa5accc4ee7a3e8e57ad`; the hashes above identify the reviewed source, not that commit. The primary source hashes were unchanged on reread. The imported component module's initial environment/audit-hook setup was inspected only to establish its startup effects; its science and previous tests were not reviewed again.

## Required findings

### 1. Bind the declared checkpoint digest to the checkpoint actually authenticated (P2)

`foundation_mismatch_pilot.py:49–50,230`; `supervise_foundation_mismatch.py:59–60`.

Initialization checks the checkpoint file against the hardcoded accepted digest, but never compares that digest with `freeze['checkpoint_sha256']`. The fixture gate merely compares this declared field between two freezes. Thus a fixture and pilot freeze can share an incorrect declared checkpoint digest and still pass while executing the hardcoded accepted checkpoint. This does not permit a different checkpoint through the current worker; it does make the supposedly authenticated provenance claim false.

Require `actual_checkpoint_sha256 == freeze['checkpoint_sha256'] == accepted_checkpoint_sha256` before loading. Persist the verified checkpoint path and actual digest in preflight evidence. An independently pinned freeze is not a substitute for checking that its declared checkpoint digest describes the loaded input.

### 2. Apply the deadline to supervisor preflight and recheck it before worker creation (P2)

`supervise_foundation_mismatch.py:49,53,66,79,96–105`.

The full-budget UTC admission check runs before committed/current-source verification and fixture validation. The Git subprocesses have no timeout. Once validation returns, only cancellation is checked before `Popen`; elapsed/UTC limits are first checked in the worker polling loop. A slow preflight can therefore exceed the 120/900-second stage allowance or pass 22:00 UTC and still spawn the worker. Cancellation during a blocked preflight subprocess only sets the flag and cannot reach owned-process cleanup until that call returns.

Carry the remaining monotonic/UTC deadline through preflight, bound owned Git subprocess waits, and check cancellation plus both deadlines immediately before worker creation. If preparation has consumed the allowance, retain a failed preflight with all cells unattempted instead of starting work and killing it on the next poll. Preserve accounting for preparation separately from worker CPU. This correction does not require changing the 120/900-second budgets.

## Checks with no additional required finding

- **Source closure:** the supervisor requires exactly six declared source/protocol entries and checks both committed blob bytes and current files. The worker verifies the independently pinned captured freeze, checks current source pins before initialization, and compiles imported research modules from the same bytes it hashes. This fixes the earlier unbound freeze-handoff pattern. No concrete launch freeze was supplied, so actual pins/revision/artifact paths are not certified here.
- **Fixture gate:** the pilot requires a hashed successful fixture terminal and its hashed freeze, matching source/runtime/interpreter/checkpoint declarations, complete fixture cells, and fixture time/CPU/RSS qualification. Source changes invalidate reuse through the exact source mapping. Finding 1 concerns the unverified checkpoint declaration within this otherwise useful gate.
- **Terminal identity:** reconciliation checks mode, expected cell count, completion-versus-file ledger equality, and each planned seed/variant/method identity and disposition. The plan has 24 fixture cells or 144 pilot cells. Failed scientific cells may remain disclosed in a completed pilot; a qualifying fixture requires every cell complete.
- **Signal ownership:** handlers retain the reviewed deferred-flag pattern through `Popen` assignment and `wait4` status publication. Cancellation prevents terminal success. Cleanup/reaping and terminal publication occur while the caller retains the controller lock. No repetition of the old signal tests was performed. The additional preflight wait issue is Finding 2.
- **Clock and CPU constants:** `1791583200` denotes `2026-10-09T22:00:00Z`. Limits are 120 seconds for the fixture and 900 for the pilot, with process CPU soft/hard limits of limit/limit+1, single-thread settings, and whole-child `wait4` accounting. The normal worker loop checks both elapsed time and the UTC stop. This does not remedy the preflight gap above.
- **Response journals:** reservations precede each 32-response prehistory, each 32-response variant history, and each 768-response private evaluation batch. Training returns bind their saved arrays. Selection is sealed before evaluation reservation/generation, and evaluation reservation/return bind the selection seal. One seed's intended counts are 160 training and 3,072 private responses; six seeds give 960 and 18,432. On interruption, a reservation without a returned receipt remains unresolved and must not be represented as zero responses or as verified returned responses. These partial journals are preserved; normal completion totals are fixed design totals, not a separate journal audit.

## Scope limit

No review of generating mechanisms, adapter predictions, selection arithmetic, local/composed scientific estimands, private scores, or reporter statistics was performed. Those remain with the independent science reviewer. This review neither runs nor authorizes the prospective fixture/pilot and does not repeat the previously reviewed pure-selector or component tests.

## Same-review protocol clarification

The user subsequently supplied an updated protocol while this guard review was concluding. Its SHA-256 is `4b3f59584f0f8b1ab15202c35b37b7fe2949374b85f9b5ae95e65e345ad8b025`. Only the relevant additions concerning exact streams, artificial seed 123456, resource gates, response journals, and exclusive execution were consulted. Both reviewed Python source hashes remain unchanged, so required Findings 1 and 2 still apply.

The protocol now expressly distinguishes the 6 GiB fixture RSS sizing gate from an OS memory cap; there is no missing-memory-cap finding. It also distinguishes stub-model tests from the unexecuted actual-runtime fixture, separates artificial response accounting, and correctly treats unpaired reservation/return journal entries as unknown returns.

The user reports five passing new artificial tests. Inspection of the late-start test shows that it sets the clock to `STOP_UNIX-100` at entry to `validate_stage`, leaving less than the required 120 seconds. That exercises initial admission rejection, not a preflight that begins on time and overruns during source/fixture validation. It therefore does not close Finding 2. No tests were executed in this review, and the independently reviewed science/reporter semantics were not reopened.

## Same-review output/lock binding addition

The subsequent supervisor SHA-256 is `032dc250acafc0aaf0e41d9842b78e1a1972f1d824b07f73647510cacb68d615`; the worker remains `7d50891299fda616f9ac9befa3852c76027af7d65fc78c2dd1da6322ba4a586c`. The added supervisor check at line 85 compares resolved output and lock paths against the independently pinned freeze before worker launch. Together with exclusive output creation, this addresses accidental reuse of one freeze at a different output/controller lock. No additional finding on that change.

The test file at SHA-256 `bd1922303e154f33c136e2f924bf158cb4448222ffee4453a36a888d1268d16c` now includes the user-reported, not-yet-run 24-cell preflight-failure fixture. Only its guard-related additions and the late-start test were read. The new test is distinct from the five previously reported passing tests. The user reports that `ace-foundation-controller.lock` is the existing shared lock and that no pilot process is live; this review did not independently query process state. Required Findings 1 and 2 remain present in the inspected revised supervisor/unchanged worker.

## Narrow correction recheck — both required findings closed

Recheck date: 2026-10-09. This disposition supersedes the outstanding status of Findings 1 and 2 for the following exact source versions; the earlier observations remain as historical review evidence.

- Supervisor SHA-256: `8021b0edcbbef79c2ab998a68aca6bb5f3b1f0ad83868aa0ccf9ad066f3e5a62`.
- Worker SHA-256: `c561ad8dac60cd01a160717a3578447b507b72e58981ef5ba63ffda20c9a7172`.
- Artificial test source SHA-256: `88ba9e97190a7835e64bb35d1bf98ad934177bc6063e476bba043a66c5077c72`.

**Finding 1: closed.** Worker initialization now requires the actual checkpoint digest to equal the freeze declaration and the fixed accepted digest before authenticated helper execution or model loading. Preflight records the checkpoint path and actual digest. The new metadata-rejection fixture targets the previously missing comparison and prevents helper execution on that path.

**Finding 2: closed.** `deadline_check` combines the shared stage monotonic deadline, absolute UTC stop, and cancellation flag. `supervise` passes its original `start + limit` into stage validation. Owned Git subprocesses check before spawning, poll `communicate` with a timeout no greater than the remaining allowance, and kill/reap on an interrupted or expired wait. Validation checks the deadline again after each Git result. The worker launch checks it before opening logs and again immediately before `Popen`, preventing an overrun in preflight from silently starting the worker. Deferred signal handling is retained.

The inspected guard fixtures cover incorrect checkpoint metadata, an already-expired Git deadline that must not spawn, a sleeping owned subprocess exceeding a short deadline, and preservation of the 24-cell unattempted plan after bad-freeze failure. Their reported passing status is supplied by the user; this recheck read the definitions and implementation without executing them. No claim is made here that the timeout fixture independently asserts every process-lifecycle property; cleanup/reaping was also checked in the implementation.

No required guard correction remains from these two findings in the inspected versions. Scope was limited to their fixes and relevant fixtures. No source edits, model imports, test reruns, real-runtime fixture, scientific run, or private-outcome inspection occurred. Only this review disposition was appended. This is source-review closure, not authentication of a future freeze or qualification of an unexecuted runtime fixture.
