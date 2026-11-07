# CPU service preflight review

Date: November 7, 2026. Independent reviewer: Mendel, separate from the implementation parent. This review covered the new CPU supervisor after the separate service runtime review. It did not repeat that closed review or execute any source, tests, remote commands or model operations.

The initial review required two fixes.

1. The packet and diagnostic source were read without a byte bound before authentication. An accidental large-file substitution could exhaust the CPU job's memory before rejection. The file map was also consumed before validating its complete shape and roster.
2. Session validation checked selected runtime fields but did not require the baseline, metadata, cache reconciliation or provenance-source records. Missing records or contradictory prefix/native-path evidence could pass. Merging version dictionaries also permitted an additional-package key to mask a contradictory main-package observation.

The narrow recheck closed both findings. Input reads now reject oversized or nonregular files before reading their contents. The CLI applies bounded authentication and validates the exact 25-file roster before consuming its entries. Runtime acceptance now requires all four previously omitted records, compares the actual parent stage and independently captured provenance, checks baseline and post-import prefix/path constraints, checks metadata/cache evidence, and requires separate exact version key sets. Both children invoke these checks.

The reviewer inspected the new focused regressions for oversized sparse input, missing records, contradictory runtime paths and version masking without rerunning them. No required fixes remained on wrapper SHA-256 `d70bc01efb8988cc637e355e37bfc94ca24d6960ba8df4a4b2e0caf59ebd2c5c`.

The parent ran three new correction methods and three affected methods from an unrelated directory. All six passed. Earlier allocation and child-command methods were not repeated. Original code snapshots, initial findings and test logs remain in the exclusive preparation capsule.

The telemetry addition records waited-for child CPU during successful calls, excluding supervisor cleanup CPU. The RSS field is a cumulative maximum over waited-for children, not a per-session or process-tree peak. Failed calls lack these success-only measurements and still require full scheduler accounting.

This closes the reviewed source issues. It does not establish actual CPU preflight success, CUDA execution, model-service qualification or scientific usefulness. Final packet review, fresh controller/journal reconciliation, once-only submission, complete custody and independent result acceptance remain necessary.
