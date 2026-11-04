# New persistent public policy transport: review requested

This is a new bounded implementation, separate from GPU08/GPU09 sources, packets and qualification. Please review only the new transport and artificial tests; no old reviews or tests need repetition. No model server or scientific execution is authorized by this custody record.

## New files and interface

- scripts/research/llm_intervention_persistent_policy.py: PersistentPolicy.call(public_envelope, request_id), a PredictionTask-bound choose adapter for the existing ProcessPolicy interface, and serve_policy(initialize, identity=...) around a callback initialized once.
- scripts/research/test_llm_intervention_persistent_policy.py: tiny artificial subprocess workers and in-memory protocol checks. No model, environment executor, scientific generator, remote operation or allocation is invoked.

The trusted CPU process retains environment/ledger/private-evaluation ownership. Requests carry only a schema-validated public task envelope and its digest, a monotonic request ID and the handshake identity digest. Responses must carry the exact matching identities and legal, affordable completion text. No extraction, repair, fallback or replay occurs.

## Review targets

Review exact model/source/adapter/numerical-recipe handshake binding, duplicate keys and request identity handling, public allowlists on both sides, bounded JSONL and nonblocking pipe I/O, startup/per-request deadlines, bounded stderr capture, owned process-group cleanup, durable exclusive evidence and failure behavior, and existing choose-interface compatibility.

The handshake is an exact attestation, not proof that a real model initializer authenticated loaded artifacts. Model authentication, actual Qwen initialization, runtime/GPU qualification, GPU idle lifecycle/resource supervision and CPU episode integration remain parent work. Same-UID filesystem/environment access is not sandboxed. Calls are sequential; caller must use the context manager or close(). Transport deadlines assume responsive local filesystem operations; they are not an allocation supervisor or an interrupt mechanism for stalled filesystem I/O. Cleanup failures are explicitly recorded, never reported as clean success. Truncated oversized streams retain bounded evidence, not full offending output.

## Artificial validation receipt

Run exactly once from /tmp:

    /Users/pat/code/ACE/.venv_test/bin/python -B /Users/pat/code/ACE/scripts/research/test_llm_intervention_persistent_policy.py

Interpreter: Python 3.9.6 (default, Aug 25 2026, 21:26:21), Clang 21.0.0 (clang-2100.3.34.2).
Result: Ran 12 tests in 1.483s; OK; exit 0.
No failed test invocations or failed setup commands occurred in this sidecar. Intentional negative cases retain their protocol/failure artifacts. The cleanup-denial test subsequently reaped its own worker and saved test-owned-worker-final-cleanup.json. All test artifacts, including the full report, are copied under test-attempt01/.

No Git commit, package installation, real model, remote action, scientific namespace or shared allocation was used. Existing files were not edited. Parent independently reviews and integrates this implementation.
