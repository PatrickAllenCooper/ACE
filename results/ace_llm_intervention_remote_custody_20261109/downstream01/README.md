# Downstream GPU campaign / finalizer preparation

**Final source reviews: 0 required findings. Source preparation only; no job, build, lock, test rerun, remote action or scientific execution occurred during consolidation.**

Use `final-implementation/gpu/` and `final-implementation/finalizer/`. Exact bytes are indexed by `final-source-pins.json`. Preserve the complete directory: `lineage/` holds every downstream draft, patching script, original source capture, failed-check transcript and test receipt; `reviews/` holds all initial findings and closures. The earlier17:10 draft is historical. Final narrow review: `reviews/05-final-deadline-closed/review.md`, SHA47f3accb75fa16e3b13dc0c514ea2ae22bdba4bbc8b91b0a84aa6b5fb1feb933. D1/D2/D3 are also independently closed. Source origins and exact copy checks are in `origin-inventories.json`; no predecessor was changed.

## Operational CLI

`CLI_ARGUMENTS.json` lists every required flag extracted statically from the exact final source. `CLI_TEMPLATES.txt` provides complete argument-by-argument templates with explicitly unfilled values. **Templates were not executed. Fill only actual independently pinned records.** No prospective acceptance/job/cost default is supplied.

- GPU builder: every metadata path/SHA pair in the argument map, plus `--cpu-remote-custody` (accepted CPU selected terminal/remote), `--cpu-terminal-mirror` (the second accepted whole compact capsule), `--storage-config`, `--storage-config-sha256`.
- Finalizer builder: its complete metadata path/SHA pairs, `--gpu-terminal-custody`, `--gpu-terminal-mirror` (the two accepted whole GPU compact capsules), and the same two storage-config arguments.
- Both launchers: `--packet-commit`, `--packet-sha256`, `--packet-review`, `--packet-review-sha256`, `--packet-directory`. Run only the reviewed, committed concrete-packet launcher; it rejects the preparation copy as a launch location.
- Both collectors: `--job-id`, `--packet-sha256`, `--submission-result-sha256`, `--local-intent-sha256`, `--packet-commit`, `--collector-sha256`, `--collector-repository-path`, `--collector-review`, `--collector-review-sha256`, `--packet-review`, `--packet-review-sha256`, `--backup-root`. GPU collector additionally requires `--cpu-terminal-custody` and `--cpu-custody-inventory-sha256`. Collector source/review must match their independently supplied full packet commit. `--backup-root` must be a new exclusive `/projects/paco0228/ACE_evidence/...` destination; existing or ambiguous attempts never retry blindly.
- Both read-only monitors: the five exact flags in the map. They are unchanged small-status monitors; they do not collect full history or qualify execution.

Builders need their same-directory files intact: build_packet.py, launch_once.py, monitor_once.py, compact_inputs.py; GPU also needs the unchanged cpu_history_contract.py3893. Standalone collectors remain separately source/review pinned. Runtime jobs still use the original frozen scientific source/inputs and resource/deadline contracts.

## Storage configuration

Schema `ace-compact-downstream-inputs-v1`; `predecessor` is `cpu` for GPU building or `gpu` for finalization. The **exact eight members** are `authority`, `authority_review`, `backend`, `backend_review`, `receipt`, `seal`, `after`, `reference`. Each entry requires independently pinned `path` and `sha256`. `backend` also requires `source_commit` (full Git commit) and `repository_path`; `backend_review` requires `repository_path` at that same commit. All values are explicit caller inputs.

The accepted authority bytes are copied to `dependencies/storage-authority.json`, SHA02a7a56bcd563433b356cef708d88841f5af30b4f91ddae4d3820394f9b7bbe1. Backend source SHA527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b and full reviewd59ffe14881d2921804bdd19ba2dfa17df7899013e7aa4fca6bfcbf28e9d28ef are copied unchanged; `dependency-pins.json` names their exact matching files. Supply the real committed repository paths/revision; copied paths alone do not establish Git binding.

The four actual predecessor custody members map as follows: `receipt`→remote-custody.json; `seal`→custody-inventory.json; `after`→inventory_after_transport.json; `reference`→predecessor-reference.json. The original before-observation/disposition are already independently pinned ordinary builder metadata. Both original selected capsule roots must match the seal; never rewrite receipt.local_root. Config is bounded128KiB, selected seal128MiB, other storage metadata32MiB. Fetches are restricted to independently pinned flat launch/inputs members,256MiB/member and unchanged scientific packet totals. CPU acceptance retains its exact16 keys.

No recursive predecessor copying: full raw bytes stay in the verified CURC USTAR backup, full inventory remains authoritative, and local views explicitly remain selected. Existing full training/failed originals remain retained in place. Independent archive verification uses the original captured observation; separate live stage append gates remain intact. The helper's current-launch text selection includes `.utf8`; omitted binaries and earlier-root files remain archived.

## Retain new operation records

Before any archive verification/fetch, both locks and exclusive attempt gates must pass. Caller evidence registers a durable original operation directory and pinned intent first. Each verify/read/fetch success or failure publishes a numbered pinned inventory/reference to both caller evidence destinations. Keep these originals; they are not disposable caches:

- Build: external sibling `<external-launch-name>-storage-build-preparation/records`, with references in its preparation parent and repository stage `storage-build-preparation01`. Absent canonical stage parents are supported; alias/file parents and existing preparation leaves reject.
- Launch: external sibling `<external-launch-name>-storage-launch-operations`, references in both packet roots.
- Collection: external sibling `<external-terminal-name>-storage-collector-operations`, references in both terminal capsules.

Finalizer reports live under archived `cpu/cpu/worker/artifacts/campaign/private/`, including summary.json and per-cell cost-report.json, outside current-launch selection. Actual audit/report retrieval must fetch required independently pinned members separately and preserve their fetch records; do not label them part of the original selected subset or deserialize tensors merely for hashes.

Metadata reconciliation and full-inventory collection now use absolute1200s remote/1230s local bounds, with count/elapsed-only stderr every1000 successful reads. Collector TimeoutError propagates to outer failure and cannot be swallowed by ordinary file-error handling. Progress never resets the alarm. This is a finite bound informed by the parent's974.6s/36k-file backup, not a throughput guarantee. GPU/job/service deadlines and scientific gates are unchanged.

## Evidence / integrity

No tests were repeated here. Original receipts preserve all focused methods and the initial D1/D2 fixture-path assertion failure, followed by its single-method correction; subsequent new path/deadline/propagation checks and independent source reviews remain separate. `consolidation.json` records local copy provenance. `inventory.json` inventories every file/directory except itself, with exact SHA256/bytes; its exclusions and totals are explicit. Original source trees were inventoried before/after copying and every destination matched, with independent file inodes. Repository integration/commit and any future operation remain the parent's work after the active controller releases.
