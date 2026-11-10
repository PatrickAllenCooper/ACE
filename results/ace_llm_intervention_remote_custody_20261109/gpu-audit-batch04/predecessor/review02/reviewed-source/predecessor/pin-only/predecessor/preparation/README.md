# Prospective GPU72 audit batch materializer

Source-only successor extension to unchanged final-reports-batch02 (`f6233712…`), retaining backend4161 without mutation. **Independent review required before actual use. No actual custody retrieval or acceptance occurred.** GPU audit9e3d remains unchanged and separately under Godel review. If that audit source changes, this explicit binding needs a separately reviewed successor; it cannot silently track a new source.

`batch_gpu_audit.py` SHA256 `5c918f959ca1a04b4515ca97a3dec7ae719cae3b8ee2b0fd91048a5b663e3332`. Exact original/draft/test sources, transcripts, costs and diff are retained. `static-inheritance.json` proves unchanged AST for the original source binder, remote emitter (two complete archive passes), bootstrap, framed transport transcript owner, verification constructor and bundle logic. Backend bytes are exact4161; remote1770s/local1800s,32MiB/member512MiB aggregate, canonical USTAR and all original verification remain unchanged.

## Input and API contract

Authenticate extension/backend bytes against independent reviewed pins before loading. The existing `bind_sources(extension_source, extension_sha256, backend_source, backend_sha256)` keyword API is unchanged. Import/source binding performs no remote operation or lock acquisition. The captured GPU audit source is authenticated as bytes only and is never imported.

1. From the independently pinned full GPU remote-custody receipt, call `select_members(receipt['full_inventory'])` to obtain the prospective exact map. Independently inspect/pin its canonical JSON bytes before transfer. No omitted-member map supplied by an arbitrary caller controls selection.
2. Call the new keyword-only API, once, after terminal custody:

```python
fetch_audit(
    receipt=original_gpu_receipt, receipt_sha256=independent_receipt_sha,
    custody_seal=original_gpu_seal, custody_seal_sha256=independent_seal_sha,
    cpu_acceptance=accepted_cpu16, cpu_acceptance_sha256=independent_cpu_sha,
    gpu_audit_source=reviewed_audit_bytes, gpu_audit_sha256=independent_audit_sha,
    gpu_audit_review=independent_audit_review_bytes,
    gpu_audit_review_sha256=independent_review_sha,
    members=exact_inventory_derived_map, members_sha256=independent_map_sha,
    destination=new_canonical_exclusive_selected_root,
    helper_path=authenticated_transport_helper,
    log_dir=new_canonical_exclusive_operation_root,
)
```

Receipt, seal, accepted CPU and member map must match original canonical backend JSON bytes and independent SHA-256 pins. Each named full-map object is capped128MiB; combined transport request remains256MiB. Audit source is capped256KiB, review128KiB, and must be the pinned9e3d source. The review pin is caller authority, not an inferred approval. No actual future receipt/acceptance/job/cost is fabricated here.

The GPU seal's actual schema has `copies`, `identical`, `excludes`, `acceptance`; the extension checks two identical compact views, receipt-byte membership and exact selected remote subtree. The receipt's full before/after/project inventories must agree, then original backend validation and remote durable readback authenticate the complete archive. The GPU's captured `launch/inputs/cpu-acceptance.json` must exactly match the independently accepted16-key CPU record, and original accepted CPU campaign file pins must remain unchanged. This is metadata binding, not a repeat CPU or GPU acceptance audit. No GPU execution acceptance is required before materialization, avoiding circular audit authorization.

The extension itself acquires both existing `ace-delivery-controller.lock` and `ace-foundation-controller.lock`, nonblocking, under the fixed BASE. Do not invoke while another owner holds either or nest it under a caller's lock acquisition. No lock files are created, no retry occurs, and no scientific job is submitted. Destinations must be canonical, absent and disjoint. Pre-admission failures make no transfer claim; caller retains their invocation evidence. Once admitted, exact source/request/receipt/map/review, raw framed stdout, stderr, partial files and failure records are preserved. Successful logs include `fetched.json`, directly usable as the aid config's `fetched` mapping after independent inventory/hash inspection. This additional selection never modifies the original receipt or compact capsule.

## Exact selection

Inventory-derived matches follow the pinned audit's fetch predicate, then reject candidates outside precise paths: `cpu/cpu/worker/artifacts/campaign/source-plan-pins.json`; all24 policy00 public `episode/plan.json`; policy05–07 attempt receipts, fixed service startup/close/failure metadata, request/task-binding JSON families, and world00–23 arm/task/paired-plan/diagnostic/episode/world-journal/return records. All candidate metadata is included; missing required public plans/pins or unexpected candidate paths fail. Failure records are retained, not filtered for favorable results.

No private subtree, private report, model/tensor/binary, raw world descriptor, sealed-input duplicate or arbitrary caller member is emitted. Paired-plan and response/return bytes are required by the existing audit's opaque joins; this transfer never JSON-decodes their scientific values. Everything unselected is still fully hashed in both archive passes. No full archive is written locally. Selection excess fails before transfer rather than truncating it.

## New fabricated checks only

Three new methods target this selection/authority/frame integration. Attempt01: selection and authority tests passed; the framing fixture failed at backend no-follow rejection of macOS `/var` alias. Attempt02 reran only framing after canonicalization, then failed at backend0700 check because the fixture used default0755. Attempt03 corrected only fixture mkdir mode and reran only framing: passed. No extension change across these three attempts. All failed attempts and partial originals remain, including six malformed/truncated/substituted frame cases. Prior passing methods and all older suites were not repeated.

Costs: attempt01 .919216 child CPU/.986949 elapsed; attempt02 .192436/.208958667; attempt03 1.992717/2.099289708. Total3.104369 child CPU seconds/3.295197375 elapsed. No remote, real coordinator locks, production-module imports, model/tokenizer/tensor, actual outcome reads or scientific calls. These fixtures do not qualify actual archive throughput or backend runtime.
