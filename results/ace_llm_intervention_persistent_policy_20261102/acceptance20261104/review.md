**Disposition: two required findings (P2).**

1. **P2 — Drain worker pipes while waiting for graceful shutdown.** [close()](/Users/pat/code/ACE/scripts/research/llm_intervention_persistent_policy.py:412) calls `cleanup_process_group(..., graceful=True)` before reading remaining stderr. That helper closes stdin and waits/signals without servicing either output pipe. A worker that writes shutdown diagnostics exceeding the pipe's capacity but remaining below the configured stderr limit blocks in its write and cannot exit. This is possible within supported settings: `max_stderr_bytes` allows up to 1 MiB, independently of the much smaller ordinary pipe capacity. The client then terminates the worker and rejects its nonzero exit; draining stderr afterward cannot recover the clean shutdown. Service bounded stderr and detect unsolicited stdout concurrently with the shutdown deadline and owned-group escalation. The current close test covers unsolicited stdout, but no fixture covers pipe-filling, within-limit shutdown stderr. This finding follows from the control flow; no reproduction was executed.

2. **P2 — Separate episode decision indices from session request IDs in `choose()`.** [choose()](/Users/pat/code/ACE/scripts/research/llm_intervention_persistent_policy.py:401) passes `decision_index` directly to `call()`, whose monotonicity is session-wide. The existing [episode driver](/Users/pat/code/ACE/scripts/research/llm_intervention_episode.py:91) resets `decisions = 0` for each episode. After a first episode has made any policy call, reusing this persistent policy for another episode rejects its first decision as a repeated/decreasing ID and permanently closes the worker. The same conflict occurs when a direct `call()` precedes `choose()` with an episode index starting at zero. Preserve strict wire IDs by allocating a fresh session ID inside `choose()` and retaining the supplied episode directory/index as client-side provenance. The existing compatibility test exercises only one `choose()` call and does not establish reuse across episodes. Direct `call()` callers supplying increasing IDs are unaffected.

Reviewed November 2, 2026. Scope is the new persistent transport, its artificial tests, and `results/ace_llm_intervention_persistent_policy_20261102/source01/review_request.md`. Existing contract/task/episode functions were read only where necessary to resolve imported validation and interface behavior. No tests, subprocess workers, real models, scientific execution, remote operations, or GPU work were run. Only this review document was written. GPU09 result review and production integration are outside this review.

**Protocol and public boundary.**

Both client and server detach and validate the complete public task envelope through `unpack_envelope`; the imported validator checks nested allowlists, task validity, history accounting, and public state consistency. The API passes neither an executor nor a private evaluator. Public prose and same-UID filesystem/environment access still require the trusted runner's discipline, as the request explicitly acknowledges.

Handshake validation binds the expected model/source/adapter pins and exact numerical-recipe content and digest. Requests and responses bind the handshake digest, envelope digest, and request ID. Duplicate JSON keys and nonfinite constants are rejected; IDs exclude booleans and must increase within the session. Completion parsing rejects repair/extraction, invalid menu choices, unaffordable actions, and active actions in terminal states while returning the original text unchanged. No fallback or retry was found.

The handshake establishes exact attestation, not actual artifact authentication. `serve_policy` accepts identity independently of `initialize()`, so a callback can attest to pins without loading those artifacts. This is explicitly disclosed, including `provenance_qualified: false` in evidence, and is not an additional finding. A future authenticated initializer must establish the actual model/source/adapter/recipe binding.

**Bounds, failure evidence, and ownership.**

Normal exchanges use nonblocking pipes, selectors, monotonic deadlines, bounded JSONL frames, a cumulative stderr cap, a request-count cap, and bounded retained stdout. Invalid frames or callback failures terminate the session; rejected response bytes are captured before validation, and decoded responses are saved before semantic acceptance. Output directories/files are exclusive, with flushed/fsynced file writes. Oversized streams retain bounded evidence rather than promising complete offending output.

Failure handling marks the session permanently closed, attempts owned process-group cleanup, and records cleanup and persistence errors. The worker is launched in a new session and cleanup targets its recorded group; no global process matching is used. Clean-close evidence requires a zero leader exit and confirmation that the group is gone. Explicit context-manager/`close()` ownership remains necessary. The shutdown-draining defect above prevents that lifecycle from reliably accepting all otherwise valid bounded stderr output.

The timeout and isolation claims remain appropriately limited: responsive local filesystem operations are assumed; this transport supplies neither an allocation supervisor nor a sandbox. `choose()` has the expected signature, task wrapping, and exact-text result for a single episode. The existing `run_task_episode` still constructs `TaskProcessPolicy`; integration and persistent lifetime ownership remain parent work.

**Preservation and exact reviewed bytes.**

The custody directory contains 137 files including its manifest. All 136 inventoried sizes and SHA-256 values match. Both working sources are byte-identical to the preserved `source/` copies. The retained report records 12 artificial methods passing once in 1.483 seconds; this review inspected their logic and receipts without rerunning them. It provides no real-model or GPU qualification.

- `llm_intervention_persistent_policy.py`: `c704a00f04a57397ef18db2a858bf7842a9cedc10557beaf162e812b9397cde1`.
- `test_llm_intervention_persistent_policy.py`: `dc3f674636aa360df7b3df7c0da889d88a9675d9860a135aeb6673e0e0a0acef`.
- `source01/review_request.md`: `9dce8728ef290ca4f13c6f0d43e8af2121c37762e3d35149717dd23ae9016765`.
- `source01/custody_manifest.json`: `bcb64cd7242fc416ef16f83f4cc718b007010589178fef51d18b7b9850aec5b7`.

This module remains separate from GPU09 and unqualified for production. The two findings concern this transport implementation and its advertised compatibility, not the pending model-authentication or GPU-integration work.

**Final disposition after the source02 delta recheck: zero required findings remain.**

Completed November 4, 2026. This addendum supersedes the initial two-P2 disposition for the exact source hashes below. The initial review remains intact above and in `source02/initial_review.md`. Scope was limited to the two corrections, their five new artificial methods and saved evidence, and the handoff record. No prior tests or full review were repeated; no tests, workers, models, scientific workloads, or remote operations were executed. Only this review file was appended.

The shutdown finding is resolved. [Cleanup](/Users/pat/code/ACE/scripts/research/llm_intervention_persistent_policy.py:156) now services both pipes while waiting, reserving half of the cleanup invocation's existing monotonic budget for graceful exit and the remainder for owned-group TERM/KILL escalation. Each service pass performs bounded reads; stderr still uses the cumulative cap, and unsolicited stdout is captured and rejected. A pipe-service error is retained in cleanup evidence and propagated through the failure path. Residual output is checked before clean-close evidence is written. The new 512 KiB stderr fixture checks exact retained bytes, zero exit, no signals, and no remaining group. The two negative shutdown fixtures check preserved unsolicited output or bounded stderr overflow and confirmed cleanup. Their saved receipts agree with those assertions.

The episode-ID finding is resolved. [choose()](/Users/pat/code/ACE/scripts/research/llm_intervention_persistent_policy.py:455) allocates `self.previous + 1` independently of the local decision index and saves an exclusive episode-to-session mapping before transmission. Direct `call()` retains its original strict, caller-supplied monotonic IDs. The mapping survives a rejected response. `bind_task()` saves the old/new task association before changing the bound task, without starting another worker; full task/state validation remains on the existing call path before transmission. Episode paths and binding metadata remain client-side. The new fixture checks wire IDs 40/41/42/43 across a direct call and local indices 0/0/1 in two episodes, changed task payloads, one initializer, and the same worker PID.

The saved patch exactly matches source01-to-current source differences. All 12 original test-method source texts are unchanged; exactly five focused methods were added. The retained command/report records one invocation of those five methods, 0.414 seconds, exit zero. These are inspected receipts, not a rerun. Both current files still match their source02 copies. The handoff's manifest pin also matches. Final comprehensive custody reconciliation remains with the parent.

Final reviewed SHA-256 values:

- `llm_intervention_persistent_policy.py`: `8ccd3ae0bd28729af8ebb758c9bd474a926d05362799f45bf0133e1e4b5e4a5d`.
- `test_llm_intervention_persistent_policy.py`: `d5a9c066bdeab7f37cd47e4fe7806f053e449f349b347915999ce41f36417980`.
- `source02/review_request.md`: `401a8dcadb4d7f1c165b2884b06d1d52b781f98f43efd143a998ecbe50a606b3`.
- `source02/changes.patch`: `883110e3e53d33599f797ebace9b987f9ce0c9bd70ebfb2912c7429c33f7b855`.
- `source02/custody_manifest.json`: `60b67ba362f9f2c1f948d77ee7f784535ef74b3e25fe2727f29c00d2c2e33c52`.
- Preserved initial review: `4d0060ee76903300b5e12b53712349db1768e9bd41bbedeb998c903148379f25`.

The narrow implementation recheck is complete. Actual artifact authentication, the parent's separately scoped model service, episode-runner integration, GPU/runtime qualification, and production readiness remain outside this disposition. Existing sequential-use, explicit lifecycle ownership, same-UID access, and responsive-filesystem limits still apply.
