# Bounded independent review: episode sealing adapter

Source fixed at module SHA-256
`b614744cecdd055720b8ec2fb35f245a75db569acacada38e0b9e1f2403f8478`.
Review the new module/test/guide, using existing source only as interface context.
Parent owns the paired runner, independent finalizer integration and scientific
source/runtime/provenance qualification. No old source or parent ledger was edited.

Agreed API: keyword-only `inputs`, `expected_sha256`,
`shared_initial_receipt_sha256`, `scope`, `world_id`, `policy_id`,
`acquisition_cap_units`; exact result keys `bundle`, `bundle_sha256`, `provenance`.
Every emitted bundle member is UTF-8 canonical JSON **plus newline**, including
receipts and returned JSON .bin evidence. Original input byte hashes are preserved.

Check pin-before-decode, whole journal/hash/order/state-machine authentication,
typed identities/charges/clamps, proposal and terminal consistency, source schemas,
captured return versus acceptance distinction, full common-initial tuple checking,
partial initialization, complete-original/failed-outer override, no fabricated
non-JSON return bytes, and output compatibility with the exact cost-curve schema.

Six new handcrafted methods passed in one invocation from /tmp, Python 3.9.6:
`Ran 6 tests in 0.097s`; child CPU 0.144719 seconds; elapsed 0.16237649999999998
seconds. Exact command/environment/interpreter/source pins are retained in
test-attempt01/receipt.json. There were no failed test attempts. One preliminary
read-only discovery command exited 2 because a guessed separate response-adapter
file did not exist; the actual class was read in llm_intervention_world.py.
That command and error are retained in command_receipts.json.

Fixtures hand-author graph/mechanism JSON, journal records and placeholder summary
values. No actual executor, learner replay, reporter, model, world generation,
private evaluator, RNG draw, old test suite, remote execution or commit was run.
The parent separately owns actual-runner/new-layer integration testing.

Limitations are explicit: hashes do not authenticate source/runtime honesty;
policy-state proposal hashes and outer receipt hashes are format/binding checks,
not independent execution evidence. Terminal summaries are schema/count checked,
not numerically refitted. A malformed or unsealed chain stays caller-unresolved;
raw costs and the full roster cell must remain retained. No scientific scores or
acceptance are produced. Independent review remains pending.
