# Final report batch02: backend binding successor only

Source-only successor to batch adapter 0737c80ff5f896b2b4f5bf1931d9ab8099cf8cbe518c305ccb25145ff7670b2b. Independent review of this successor is pending; the parent owns review/integration. No build, remote operation, report retrieval, private payload decode, scientific execution, lock acquisition or Git commit occurred here.

## Exact scope

`batch_reports.py` changes exactly four lines: module BACKEND_SHA256 and the independently captured remote-bootstrap backend literal move from 527d to 4161; two explanatory labels name the captured 4161 backend. No other adapter bytes change. `source.diff` and `static-api-proof.json` demonstrate that reversing these exact pin/label replacements recovers the complete 0737 bytes. All adapter function signatures and all 28 used backend function/method signatures match. Backend alarms remain 1770s remote/1800s transport.

Adapter SHA256: f62337123abe2eec7c7624e9c32692309d4469db23d28a8e94902a7e1548837c
Backend SHA256: 4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f
Backend independent review: 8f858bef11028623627858020d00d4850d0c9afadd1de9a2472b386b5f65959a
Backend/review/authority committed revision: b7a44809604822d979c1ed05c3c0ce0128a597c0
Storage authority: 011c2d3bce9ac28e388b5dc91de73076ad42bbdf97819b20850554e8a9a8c29a

Captured backend, review and authority bytes are under dependencies/. They matched that full Git revision; no Git writes occurred. The original adapter, old backend and closed review/doc are preserved under predecessor/ for lineage only. Complete original batch01 remains unchanged at results/ace_llm_intervention_remote_custody_20261109/final-reports-batch01 and its previously sealed external preparation.

## Unchanged API and future invocation

Authenticate captured adapter and backend bytes with explicit conditional raises before exec (never assert). Then call:

```python
api["bind_sources"](extension_source=extension_bytes,
                    extension_sha256="f62337123abe2eec7c7624e9c32692309d4469db23d28a8e94902a7e1548837c",
                    backend_source=backend_bytes,
                    backend_sha256="4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f")
api["fetch_reports"](receipt=receipt, receipt_sha256=independent_receipt_pin,
                     members=members, members_sha256=independent_members_pin,
                     finalizer_acceptance_sha256=accepted_finalizer_pin,
                     finalizer_custody_sha256=complete_finalizer_custody_pin,
                     destination=new_exclusive_reports_root,
                     helper_path=authenticated_transport_helper,
                     log_dir=new_exclusive_operation_root)
```

The receipt must bind backend4161; old527d receipts are not reinterpreted or rewritten. Parent must first independently accept complete finalizer execution and custody and bind the exact new receipt/source/member pins. Actual CPU collection is still pending and no actual finalizer receipt exists in this preparation.

Selection is unchanged: all 192 cost reports, 24 world summaries, 24 world pin files and campaign summary/pins (242 files), JSON-named opaque bytes under cpu/cpu/worker/artifacts/campaign/private/. Exact missing members reject; no synthetic replacements, silent omissions or outcome decoding. One full verification plus one full archive stream/check remains two passes per successful batch. All capacity/size/framing, exclusive destination, first-failure/cleanup and additional-selection evidence behavior stays unchanged. The original custody receipt/local selection is never modified. This binding check establishes no actual archive throughput or scientific qualification.

## Single new fabricated check

Only `test_backend_binding.py` ran, one new method with three cases: actual authenticated captured4161 source binds; changed bytes with the4161 pin reject; valid old527d source with its old pin rejects before execution. Source authentication is explicit and negative cases block backend exec. No archive creation/verification/streaming, payload reads, remote bootstrap, subprocess transport or scientific/model functions ran. The old nine methods were not imported or rerun.

Evidence: evidence/attempt01/source/ contains exact executed snapshots and dependencies; source-pins.json, stdout.txt, stderr.txt and run.json retain the original attempt. Result: 1 test passed, 0.076806583s wall, 0.046783s child CPU, 24,690,688 bytes maxRSS. No failed test attempt. Independent review remains the parent's next step.

`inventory.json` pins all preparation files except itself; both repository and external copies match with independent file inodes. `handoff.json` records exact source/test/static/test-receipt pins and copy destinations. All prior failed CPU archive evidence stays with the parent's preserved custody; this preparation neither retries nor touches it.
