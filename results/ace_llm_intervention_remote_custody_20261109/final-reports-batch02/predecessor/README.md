# Opaque finalizer report batch — prospective source only

No actual retrieval, finalizer execution, remote operation, scientific outcome decode, or acceptance has occurred in this preparation. Parent must independently accept the complete finalizer execution and custody before invoking this adapter. Those acceptance/custody hashes are required inputs, not acceptance decisions made by this adapter.

## Scope and exact selection

The fixed prefix is `cpu/cpu/worker/artifacts/campaign/private/`. The exact additional selection contains 242 JSON-named opaque files: 192 `world-00..23/arm-00..07/cost-report.json`, 24 `world-00..23/summary.json`, 24 `world-00..23/report-pins.json`, plus campaign `summary.json` and `report-pins.json`. Member metadata is independently pinned; no report JSON is decoded. It rejects any missing, extra, changed, or oversize member. The frozen finalizer creates cost reports only for reported cells. Missing reports therefore reject this complete-roster retrieval; it never fills or silently drops unresolved cells. Other original failure evidence remains in the complete remote archive and original custody.

Both original source and storage authority remain unchanged: backend `527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b`, authority `02a7a56bcd563433b356cef708d88841f5af30b4f91ddae4d3820394f9b7bbe1`. This extension records a separate additional selection, never rewrites the original receipt, and does not claim a complete local raw copy.

## API / captured invocation

Load and authenticate extension bytes before executing them; also supply independently pinned original backend bytes. There is no executable main or automatic remote action. The parent uses its existing transport helper only after all acceptance/custody gates.

```python
from pathlib import Path
import hashlib

extension = Path(EXTENSION_PATH).read_bytes()
backend = Path(BACKEND_PATH).read_bytes()
if hashlib.sha256(extension).hexdigest() != EXTENSION_SHA256:
    raise ValueError("independent extension SHA differs before exec")
if hashlib.sha256(backend).hexdigest() != BACKEND_SHA256:
    raise ValueError("independent backend SHA differs before exec")
api = {"__name__": "captured_batch_reports", "__file__": "<captured-batch-reports>"}
exec(compile(extension, api["__file__"], "exec"), api)
api["bind_sources"](extension_source=extension, extension_sha256=EXTENSION_SHA256,
                    backend_source=backend, backend_sha256=BACKEND_SHA256)
b = api["_backend"]()
receipt_raw = Path(ORIGINAL_RECEIPT_PATH).read_bytes()
members_raw = Path(INDEPENDENT_MEMBERS_PATH).read_bytes()
if hashlib.sha256(receipt_raw).hexdigest() != ORIGINAL_RECEIPT_SHA256:
    raise ValueError("independent original receipt SHA differs")
if hashlib.sha256(members_raw).hexdigest() != MEMBERS_SHA256:
    raise ValueError("independent selection SHA differs")
receipt, members = b.decode(receipt_raw), b.decode(members_raw)
if b.raw(receipt) != receipt_raw or b.raw(members) != members_raw:
    raise ValueError("canonical pinned metadata bytes required")
result = api["fetch_reports"](
    receipt=receipt, receipt_sha256=ORIGINAL_RECEIPT_SHA256,
    members=members, members_sha256=MEMBERS_SHA256,
    finalizer_acceptance_sha256=ACCEPTED_FINALIZER_SHA256,
    finalizer_custody_sha256=COMPLETE_FINALIZER_CUSTODY_SHA256,
    destination=NEW_ABSOLUTE_REPORTS_DIR,
    helper_path=EXISTING_AUTHENTICATED_CURC_HELPER,
    log_dir=NEW_ABSOLUTE_OPERATION_DIR)
```

`members` is a canonical JSON map of exact archive-relative names to `{ "bytes": integer, "sha256": "64hex" }`. The parent independently derives/checks its 242 entries against the accepted full archive inventory; the adapter also checks equality against the original byte-pinned receipt. The two JSON inputs must be the backend's canonical byte spelling, preserving byte and semantic pins rather than silently normalizing unpinned input. Required actual future pins and paths have no defaults.

Parents of both new directories must already exist, be canonical, and have no symlink components. Report and operation destinations must be absent and disjoint; both are exclusively created owned0700. No overwrites, deletion, retries, or directory adoption. Parent must durably pin and retain `operation-records.json`, `receipt.json` on success, and every original/partial operation file on failure; a receipt file alone is not an independent finalizer acceptance. Any exception or evidence-publication failure makes invocation unsuccessful and requires reconciliation, never automatic retry.

## Reuse, passes, bounds and evidence

The adapter captures the original backend without modifying it. Small `types.FunctionType` specializations reuse its exact `transport` and `remote_main` code with isolated dictionaries supplying only the extension bootstrap/source bundle or remote emitter. Original backend globals and its binding stay unchanged. Both individual source hashes are verified before either source executes remotely; the combined bundle is authenticated by the independent derived bundle hash supplied to the remote bootstrap.

Remote pass1 is original `remote_verify`, including `validate_receipt`, `check_backup_record`, complete canonical USTAR member/hash checks, and full archive hash. Remote pass2 is original `check_tar` over the whole archive, emitting only the selected report members. The exact second archive hash/size must still match. A later unselected-member failure can occur after reports stream; it is retained as failure, not successful receipt. Exactly two archive passes for any successful batch; no per-report archive replay. No full archive is downloaded to the Mac.

Original `Receiver` validates frame/identity, byte lengths, hashes and trailing/truncated data; original process-group/transport cleanup and receiver cleanup are reused. The additional framed stdout transcript is retained including partial/malformed bytes. `operation-records.json` pins sources, requests, command, stdout/stderr, transport and success/failure records (excluding itself); publication errors preserve a pre-existing primary error. Reports are hashed again on local readback. Opaque payloads are never JSON-decoded, printed, deserialized, or used for scientific selection.

Bounds remain 16GiB full archive inventory payload, original USTAR/member/metadata limits, 32MiB per selected report, 512MiB selected report aggregate, 1800s transport and 1770s remote alarm. Capacity uses original per-filesystem accounting/headroom and adds the retained stream/source/request evidence budget. The retained stdout contains a second local copy of selected report bytes (not the full archive), explicitly included in capacity admission. The 242 tiny synthetic files do not establish actual transfer throughput, space reservation, institutional backup, or finalizer qualification. No changes to GPU/CPU jobs, source closure, science, storage authority or original selected-inventory policy.

## New focused evidence

Original preparation (path pinned in `predecessor.json`), `evidence/attempt01`: six new methods passed once; 1.435903166s wall, 1.343798s child CPU, 32,587,776 bytes child maxRSS. Two-pass opaque readback, original bindings/receipt unchanged, exact roster/pins/bounds/exclusive destinations, both-source pre-exec tampering, malformed/truncated/extra/hash framing, and archive corruption after the first pass while retaining delivered report bytes.

Original preparation, `evidence/attempt02`: only one newly added primary/secondary evidence-publication failure method passed; 0.237958917s wall, 0.200332s child CPU, 26,722,304 bytes child maxRSS. Earlier six methods were not repeated. The source delta only preserves a cleanup exception if inventory publication also fails. All source snapshots, synthetic archives/payloads/partial failures and transcripts are retained. No failed test run, old test methods, SSH, real artifacts, models, tokenizers or project execution.

Frozen-source path observations are captured in `originals/` solely to establish the report path/conditional cost-report behavior; none were imported or executed. Independent source review is requested separately; preparation does not mark itself independently accepted.

## Focused review correction

Initial independent review SHA c7c4b92ed7fb7bc1d86efa1255e532878f4549a3ec0d599c705a506ca31479d0 is preserved in initial-review/. B1 now claims the report root with exclusive owned_dir before calling the original destination builder; an empty path appearing during begin is rejected. B2 now uses a narrow transcript context that retains the original transfer error across flush, fsync, buffered close and underlying-handle close; original backend Receiver/close_receiver own resource cleanup and save secondary errors under transcript-cleanup/. B3 captured invocation uses explicit conditional raises before any exec, unaffected by Python optimization. No original backend/source/receipt or archive behavior changed.

This successor's `evidence/attempt01` runs only two NEW focused methods: raced empty destination and simultaneous stream + flush/fsync/close failures. Both passed in 0.293575250 wall seconds, 0.249391 child CPU seconds, 27,688,960 bytes child maxRSS. The previous seven methods were not repeated; all original source/test/evidence and review files remain intact. Nine distinct new methods passed across the preserved lineage; no actual retrieval or acceptance.
