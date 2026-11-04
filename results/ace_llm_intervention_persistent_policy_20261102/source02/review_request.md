# Persistent public policy transport: narrow review02 requested

Only the two P2 findings from Singer's preserved initial review are addressed. Please recheck these fixes and five new regression methods; do not repeat the closed 12-method suite or unrelated source/GPU reviews. Independent closure is pending.

1. Graceful close now services bounded stdout/stderr while waiting for owned-group exit and during escalation. Half the unchanged cleanup budget is reserved for graceful exit; the remainder covers TERM/KILL. Pipe-service errors retain evidence and trigger escalation. A 512 KiB shutdown stderr fixture exits cleanly without signals, while concurrent unsolicited stdout and oversized stderr remain failures with confirmed cleanup.
2. choose allocates previous-session-ID + 1, independently of episode-local indices. Exclusive request-N.episode.json records the episode directory, decision index, proposed session ID and task pin before transmission. bind_task(PredictionTask) persists a binding receipt and changes subsequent public envelopes without restarting the worker. Direct call IDs remain caller-supplied and strictly monotonic. Tests combine direct call(40), choose at local index0, rebinding, then a different episode at local indices0/1, yielding wire IDs40/41/42/43 with one initializer/PID. Rejected responses retain their local/session mapping.

## New validation only

Five explicitly named review02 methods ran once from /tmp with /Users/pat/code/ACE/.venv_test/bin/python, Python3.9.6. Saved output: Ran 5 tests in0.414s; OK; exit0. No failed test invocation or command occurred. Exact command, tool result and command history are in command_receipts.json. All new fixture artifacts are under test-attempt01/. The old12 method bodies are statically unchanged and were not executed.

## Preservation and limits

Both source01 copies retain all137 original files, original hashes, manifest and review-request bytes. preservation.json records their verification. This new source02 capsule is separate from all active GPU packets. It includes the unchanged initial review, the bounded patch, new source/test bytes and new fixtures. No Git operation, model, scientific namespace, remote action or allocation occurred.

Actual artifact authentication, Qwen serving, external allocation supervision and parent episode-runner integration remain unqualified. The handshake remains exact attestation, not proof of loaded weights. Calls/binding are sequential and not thread safe. Same-UID filesystem/environment access is not sandboxed. Responsive local filesystem I/O remains an assumption; existing timeout and lifecycle constraints otherwise remain in force.
