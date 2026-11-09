# CPU history metadata adapter — source-only handoff

Implementation: `llm_intervention_cpu_history_admission.py`, SHA256 `4ab59481dc3085fe43629cd5480d9cbff65b1c7d62ca96703b349788d4907a67`.

Tests: `test_cpu_history_admission.py`, SHA256 `2ec4de53ad4f9933b5521c5a919ff81485b8ffbe32cb422683c925bc03abdd39`.

This is prospective source for independent review. No original campaign, phase, entrypoint, producer, current capsule or scientific file was changed. No actual readset was captured, no original project callable was executed, and no runtime, model, response, world, remote operation, lock, allocation or commit was involved.

## Fixed public API

```python
history = capture_history(
    acceptance_raw=acceptance_bytes,
    acceptance_sha256=independent_acceptance_pin,
    bundle_raw=bundle_bytes,
    bundle_sha256=independent_bundle_pin,
    readset_raw=readset_bytes,
    readset_sha256=independent_readset_pin,
    campaign_source_raw=captured_original_campaign_source,
    campaign_source_sha256=independent_campaign_source_pin,
)
with history.install(campaign_module):
    # The parent's existing five-CPU barrier and GPU controller/phase only.
    ...
# Original callback identity is restored here, including on BaseException.
```

The module name is `llm_intervention_cpu_history_admission`. Capture is keyword-only, performs no live filesystem reads or writes, and returns a detached object rather than a caller-controlled qualification flag. Its public pins/root are `acceptance_sha256`, `bundle_sha256`, `readset_sha256`, `campaign_source_sha256`, `campaign_root`, and `campaign_sha256`. There is no CLI, producer, acceptance emitter, full-tree scanner or receipt framework.

The caller must independently authenticate the actual CPU acceptance, phase-terminal semantics, namespace/lineage and completed CPU preflight receipt **before** installing this adapter. The caller must require full CPU post-custody verification before any GPU execution acceptance. Capture checks its supplied exact byte pins and the CPU acceptance/bundle/readset joins; it does not manufacture independent acceptance or certify that a supplied preflight check happened. Kuhn's preflight binding remains a separate mandatory outer gate.

The original campaign source is restricted to SHA256 `9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe`. At installation, compilation of these authenticated bytes (without executing them) supplies expected code objects. The existing module's original `_load_attempt`, `_inventory`, `_closure`, `_need`, `_keys`, `_decode`, and `_source_pins` must match those code objects, filename, global dictionary and function identity structure before callback replacement. The module version must match. An already replaced callback/helper or nested installation is rejected.

Add this new module through the parent's explicit captured source closure; do not relax the loader or runtime source guard. Scope the context to controller barriers and `run_gpu_phase`, restore before final runtime observations where convenient, and always restore before any finalizer. Finalization is outside this API. The live guards also reject existing finalization/private markers. This is trusted-process integration, not protection against arbitrary concurrent mutation of Python globals by hostile code.

## Exact readset schema and bounds

```json
{
  "schema": "ace-scientific-cpu-history-readset-v1",
  "cpu_acceptance_sha256": "<independent acceptance byte pin>",
  "campaign_root": "<the exact accepted logical campaign root>",
  "files": {
    "<relative member>": {"sha256": "<accepted member pin>", "bytes": 123, "utf8": "<exact UTF8 bytes decoded once>"}
  }
}
```

Exactly 133 members: `campaign.json`, `source-plan-pins.json`, `start.json`; both `attempt.json` and `attempt.candidate.json` under each `policy-00` through `policy-04`; and `world-00/arm-receipt.json` through `world-23/arm-receipt.json` under each of those five policies. No phase/bundle duplicate belongs in this map. The separately supplied bundle is joined to the acceptance's `worker/artifacts/cpu-phase/campaign-bundle.json` pin. The existing phase-terminal path/pin is checked against the acceptance map; the outer entry remains responsible for decoding/admitting its captured bytes.

Encoded readset and decoded member aggregate: 16 MiB maximum each. Acceptance: 128 MiB; bundle: 32 MiB; original source: 1 MiB. These are named bounds, not a new global cap. The parent reports the actual 133-member metadata set as 5,548,472 raw bytes; this authoring task did not read or capture those actual members. UTF8 round-trip length, hash, exact fields and accepted map pins are checked for every member; duplicate JSON keys and nonfinite JSON numbers are rejected.

## Semantic and live-path behavior

The accepted bundle determines the exact 24 development IDs, source-plan digests, ordered eight policy IDs, phases and three persistent-service IDs. Campaign/source-plan/start bytes must agree with this binding. Each CPU attempt pin must match both accepted attempt/candidate map entries. Its declared full `evidence_sha256` must equal the corresponding accepted historical map, excluding only the same two attempt header files as the original inventory routine.

`_historical_attempt` retains the original `_load_attempt` header fields and identity, arm roster, world IDs, status/reason/path constraints, receipt pins and arm plan/policy identities. It specializes the original closure branch to the five service-free CPU task-process policies and retains the controller's no-errors/all-completed admission. The only omitted payload operation is the full historical `_inventory(path)` scan; the complete accepted evidence map is still compared. `semantic-delta.md` maps the source predicates.

Installation establishes identities for the fixed metadata directory parents. Each callback checks the exact campaign value/pin, policy ID, path and expected receipt pin. A CPU expected pin different from acceptance is rejected, including a live self-hash supplied by the original local barrier. CPU reads use no-follow descriptors, ordinary-file/size checks and before/after identity checks, compare against accepted captured bytes, then decode/use those same read buffers. Each of that CPU policy's 26 metadata files is read once per callback. Campaign/source-plan/start headers and the five service-absence guards are live. Parent/root replacement, symlinks (including dangling service links), finalization markers and wrong roots reject.

The one previously reviewed mount binding `/scratch/alpine` → `/gpfs/alpine1/scratch` is preserved with exact alias resolution and no-follow physical descendant traversal. Arbitrary aliases are not admitted. Root/directory identities use device/inode, not timestamps that legitimately change when new GPU attempts are added.

The three exact GPU IDs take the saved original `_load_attempt` with the original arguments after identity/path guards. Their full inventory, arm, closure and service logic is unchanged. No `_inventory` module global is mutated. A retained callback cannot be used after the context retires. Mutable return records do not modify the sealed captures.

These checks bind consumed bytes; they do not claim continuous immutability of unused historical files between CPU observations. Any historical post-check failure blocks acceptance and preserves the GPU attempt; it does not authorize a rerun.

## Retained new evidence

- `attempt01`: source/test snapshots and failure transcript. Local Python 3.9 rejected `dataclass(slots=True)` before any test method ran. The correction removes only that unsupported keyword; no gate changes.
- `attempt02`: nine new fabricated methods passed, 0.984370 child CPU-seconds, 1.009431667 wall seconds, 47,529,984-byte macOS child maximum RSS.
- `attempt03`: one added mount-alias/rebinding method passed, 0.238560 child CPU-seconds, 0.262285708 wall seconds, 28,082,176-byte child maximum RSS. The prior nine methods were not repeated.

The tests execute only the new stdlib adapter and an explicitly synthetic minimal campaign module. They temporarily substitute the synthetic source pin in the new module to test binding mechanics; the delivered production source retains the fixed original pin. No real model, plan validation, world, policy or scientific source execution occurs. Existing production integration and exact actual readset admission remain unexecuted and require independent review.

Both the initial failed import and all ten new passing methods, their exact source snapshots, stdout/stderr, argv, resource receipts and input design/source lineage are preserved. The external mirror path and final inventory are in `locations.json` and `inventory.json`.
