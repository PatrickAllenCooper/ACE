# Delivery confirmation adapter: local validation

Implemented `scripts/research/runner_delivery_confirmation.py` against the unchanged
12-history registration and Runner revision e9f811f. Default invocation only
validates. This stage generated no confirmation histories, fits, or scores.

Eleven deterministic tests passed. Checks cover registration identity, pre-call
attempt accounting, no replacement, receipt parity and eligible-row counts,
byte seals and symlink rejection, evaluation refusal before grid access,
network/grid audit hooks, elapsed/RSS supervisor paths, independent watchdog,
and preservation of an unrelated process. Existing development bundle 5f89033d
was read for custody verification only: 4,803 ACE rows match metadata and the
saved refit receipt. Its canonical parsed-row hash is
`f6d87379563a9cf22e644d2870a7e5cf3c7568bce77ec9c9cba5b31bf0c34e23`.
No historical score was recalculated.

The companion freeze receipt records adapter, tests, registration, interpreter,
and installed dependency versions. A future execution archives the pinned Runner
revision, inventories the whole source snapshot, copies the protocol document,
and verifies source and dependency identity in each worker. Acquired artifacts
and all three initialization receipts must pass semantic checks before sealing;
scoring requires every registered case and an unchanged seal. Failures preserve
partial artifacts and mark the execution incomplete. There is no resume or seed
replacement.

## Remaining execution gate

Campaign authorization and reconciliation of additional seed registries are
required. A future approval receipt must bind registration SHA, adapter SHA,
and the exact dependency_versions mapping from the freeze receipt, and record
approved=true, registry_reconciled=true, and authorized_by. No such approval
receipt was created here. Merely possessing a receipt is a procedural check,
not an authentication system.

Once authorized, the first registered case acquires and fits without scoring.
Continue only if `1.2 * 12 * first_case_seconds + 300 <= 7200`. Otherwise stop,
preserve artifacts, and request a resource/protocol decision. Limits remain six
CPU threads, two hours, 8 GiB sampled RSS, zero GPU and model API calls.

## Practical limitations

The full acquisition/fitting/scoring integration has not been executed by these
fixture tests. The audit hook protects trusted adapter code; it is not an OS
sandbox. RSS is sampled and aggregate process-tree telemetry can miss brief
peaks. The independent watchdog bounds supervised computation, with OS scheduling
latency; archive/setup operations remain CPU-only and are not independently
watchdog-supervised. A long setup therefore consumes the campaign wall budget
before any child may start. No performance or accuracy claim follows from this
implementation validation.
