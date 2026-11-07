# Staged-runtime caller correction recheck

November 7, 2026. **R1 is closed. Zero required findings remain in this caller delta.** This is a static correction review, not qualification of an actual runtime, launch packet or GPU service.

The [original review](/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_staged_runtime_review_2026-11-07.md) is preserved unchanged. Its SHA-256 remains `0cdafb6286c157a50a4c76f4a7f6a3e5c0b67931ef171b89600d125007046f41`. The recheck compares the preparation01 caller and tests with the current files and preparation02 copies. It inspects only the changed supervision path and the two new fabricated fixture results. The helper's separate review and unchanged caller logic were not reviewed again. No tests, reviewed functions, processes under test, models, real runtime or remote commands were executed. No commit was made. Only this recheck document was written.

## Why R1 is closed

The new [run_bounded_child](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_staged_import.py:86) records one monotonic start before opening child logs or calling `Popen`. It rejects launch when less than 200 seconds remain under the original overall deadline. The child deadline is fixed at that start plus 185 seconds.

An `ITIMER_REAL` timer is armed before child creation and startup-receipt I/O. The existing parent SIGALRM handler raises into exception handling. After the startup receipt, the code computes `deadline - time.monotonic()`. A nonpositive remainder raises without calling `wait`. Otherwise, `wait` receives only that remaining duration. A slow receipt therefore no longer gives the child a fresh 185-second wait.

The `finally` block disables the phase timer while invoking the existing bounded owned-group cleanup for a returned process handle. It restores the remaining duration to the original overall deadline before writing the cleanup receipt. It does not grant a fresh 1,740 seconds. If the overall deadline has already elapsed, the minimum positive 0.001-second timer avoids silently disabling the timer with a zero value. Failure still propagates to the main failure writer.

The main caller passes `start + 1740`, where `start` is the original preparation start. Its unchanged acceptance check uses the elapsed value measured from the new pre-spawn start and rejects values over 180 seconds. Cleanup status, the original terminal, origin evidence and zero-update/response checks remain acceptance conditions. The correction does not change the original probe, its named import order, the account or resource gate, packing limits, or model-service qualification scope.

This closes the specific timer-reset and unguarded receipt-I/O gap. A Python real timer is not a hard kernel preemption guarantee. Native calls, signal delivery and scheduler termination can still delay observation. Actual elapsed times, terminal status, cleanup and scheduler accounting remain necessary evidence. No live timing guarantee is inferred from the fabricated checks.

## New fixture evidence

Preparation02's command selects only these two new methods:

- `test_receipt_latency_does_not_extend_absolute_child_deadline`
- `test_no_time_left_rejects_before_spawn`

The first method has two fabricated clock cases. A 20-second startup-receipt delay leaves a 165-second wait. A 186-second delay raises before any wait. Both cases assert one owned-cleanup call, a cleanup receipt, an initial 185-second timer, restoration of only the original overall remainder, and no success terminal. The second method supplies only 150 seconds of remaining overall time and verifies rejection before `Popen`.

The saved stderr reports two methods passing. The receipt records return code zero, 0.056215 child CPU seconds, 0.060124458 elapsed seconds and 21,250,048 bytes peak child RSS. These are the author's recorded fixture measurements, not review or whole-preparation costs. `Popen`, the clock and timer are mocked. The fixtures do not deliver a real SIGALRM or exercise kernel process cleanup. The source establishes where the timer is armed and how the `finally` path connects to the existing cleanup routine. None of the three preparation01 methods was rerun by the preparation02 command or by this review.

## Scope and identities

Current caller and preparation02 copy both have SHA-256 `9e63e644aba5c65a5c8b61d24018a8dad2282538b2834132b36da5d7516e72c7`. Current tests and the preparation02 copy both have SHA-256 `ee8a3342b6292a9d38b9d11b606aa03d56bdcb889171a69ffe6758cc1543fbc8`. Both match the receipt's source pins.

Preparation02 receipt SHA-256 is `105f8e31f02f3175c5bf5a1aeb2c5d8150fba9ad41b6124dbfb279034a6eeb69`. Its predecessor pin matches preparation01 receipt `45b45388436881417d2db8cee84b3091518c3fb76350d76ae44e4a6cab99d4c0`. The unchanged guide has SHA-256 `e80f89e47f9acb012d9b79b36846815ed26a381ba96986b31df2dec79f22c261`.

The optional failure-origin snapshot remains explicitly deferred. It was not a blocker and is not counted as an unresolved required finding. The helper correction review, independent packet pins, submission reconciliation, actual resource gate, execution and complete custody remain separate obligations. This recheck does not authorize a different resource envelope or certify that the copied runtime will import successfully.
