# Artificial campaign CPU finalizer review — 2026-11-08

**Disposition: no required findings in this bounded static review.**

Reviewed only the new `scripts/research/finalize_llm_intervention_campaign_cpu.py`, its focused test file, and `/tmp/ace-campaign-finalizer-handoff.md`. Read narrow reused-helper interfaces to check integration; their implementations and closed reviews were not reopened. No source edits, tests, staging, model execution, private finalization, remote actions, or commits were performed.

Verified SHA-256:

- Finalizer: `88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f`.
- Focused tests: `f705254a05e7720f7ae93c796851929a95977ccec8e9403bdeb19b4854000576`.

Static checks:

- `authenticate` binds the independently pinned packet, captured source closure, predecessor packets/acceptances, and original input bytes. `admit_evidence` checks CPU/GPU acceptance identities, scheduler envelopes, exact Boolean GPU scheduler/runtime/cleanup/custody attestations, full upstream inventories, terminal pins, original bundle pin, and all eight published attempt/candidate pins before private bundle decoding. Independent acceptance remains external trusted authority, as documented.
- Destination validation separates the new input/output roots from upstream roots and protects immutable runtime/model/archive/manifest inputs from the output claim. The campaign remains at its original CPU-output location. Exclusive directory creation prevents output reuse; existing campaign finalization artifacts block admission.
- The parent requires 2 CPUs, 7680 MiB, 300 seconds, the specified account/partition/QOS, zero GPUs, and no requeue/restart. Staging uses the pinned helper once. The owned isolated worker inherits the original absolute clock: startup 120s, work 260s, cleanup 280s, publication 290s. Cleanup authority comes from the process handle created by this supervisor. Runtime checks reuse the captured-module filter and prefix/native-path checks without added exemptions.
- The worker invokes the pinned original campaign finalizer once. No new fitting, model loading, policy responses, or GPU execution is introduced. Report validation matches the existing receipt/summary interfaces: all 16 ordered cells, all 28 policy pairs across both worlds, original evidence retention, reported-cell curves, failed/unresolved counts, and false scientific acceptance. Failure artifacts are retained; outer failure records explicitly defer campaign publication authority to the original `finalization.json` if present.

Read the retained `finalizer01/attempt-01` and `attempt-02` receipts/logs: the first records 21 tests with one scheduler parsing error; the second records 23 passing fabricated/mocked tests. The second source/test snapshots match the hashes above. Recorded combined test-child CPU is 0.617659 seconds. No suite was rerun.

This review does not establish actual archive staging speed, runtime qualification, successful private finalization, scheduler completion, or scientific/operational acceptance. The five-minute execution budget remains a bounded proposal; parent-owned upstream acceptance and subsequent execution/custody evaluation remain required.
