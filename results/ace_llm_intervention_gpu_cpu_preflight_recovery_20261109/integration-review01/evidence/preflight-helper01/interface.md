# Draft pure preflight validator interface

Public module: `llm_intervention_cpu_preflight_receipt.py`.

```python
validate_receipt(receipt_raw, receipt_sha256, expected_raw, expected_sha256)
```

Four positional arguments; raw arguments must be immutable UTF-8 JSON `bytes`, hashes independently supplied lowercase64-hex. No I/O, producer imports, clock reads, locks, resource requests or actual receipt generation. Exact keys/types, duplicate JSON keys and nonfinite numbers fail closed. Each JSON input is capped128MiB. Caller authenticates both pins to independent authority; this function cannot establish that authority itself. This is a draft, not integrated or actual-GPU qualification.

Canonical digest throughout is SHA256 of `json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')` (default ASCII escaping). Input byte pins always bind the ORIGINAL supplied byte strings, not this canonical serialization.

## Expected JSON (exact top-level keys)

- `schema`: `ace-gpu-cpu-preflight-expected-v1`
- `execution_context`: descriptor below
- `execution_context_sha256`: independent canonical descriptor digest
- `cpu_inventories`: exactly `cpu_output` and `campaign`, each `{files, directories}`
- `consumed_read_set`: exact133 relative campaign filenames, values `{bytes, sha256}`
- `receipt_not_after_utc`: independently anchored source-context UTC observation/submission bound; not inferred from the receipt. No age TTL or scheduler-wait expiry.

Each inventory `files` is the FULL independently authenticated relative-name→`{bytes: nonnegative int, sha256: 64hex}` map. `directories` is the FULL independently authenticated sorted unique list, including empty directories, excluding the root itself. Parents must be present, files/directories cannot collide, and path aliases/traversal are rejected lexically. CPU-output inventory excludes `worker/artifacts/campaign` and its subtree; that tree has its own campaign inventory. This directory roster must descend from sealed original collector metadata; the older producer's file-only hash map is NOT a directory-roster proof.

The campaign inventory has exactly the original root entries `campaign.json`, `source-plan-pins.json`, `start.json`, `policy-00` through `policy-04`, with no persistent `service` subtree under those five policies. Read set is exactly those three root files + five `attempt.json`/`attempt.candidate.json` pairs + `policy-00..04/world-00..23/arm-receipt.json` (133). Every read-set byte/hash entry must equal its independent full campaign inventory entry. Total read-set payload is bounded16MiB; expected size5,548,472 is an observation supplied by the parent, not a hard-coded actual result. Bundle/CPU-phase-terminal at-use checks remain the outer consumer's responsibility; they are not silently added to this133-member barrier set.

## Execution context descriptor (exact keys)

- `schema`: `ace-gpu-cpu-preflight-context-v1`
- `control_root`: new exclusive GPU02 control path
- `gpu_output`: original frozen `failed_gpu01_root + '/gpu'`
- `failed_gpu01_root`: original failed GPU01 root
- `cpu_roots`: exactly `{cpu_output, campaign}`, absolute paths; campaign equals `cpu_output + '/worker/artifacts/campaign'`
- `lineage`: exact pin keys `cpu_acceptance_sha256`, `failed_gpu01_acceptance_sha256`, `recovery_authority_sha256`, `source_sha256`, `checker_sha256`, `namespace_sha256`, `original_namespace_sha256`
- `baseline`: exactly `{sha256, terminal_jobs:47, submission_journals:96}`. No charge or actual-baseline pin defaults.
- `proof_pins`: exact existing-evidence pin keys `custody_sha256`, `before_observation_sha256`, `after_observation_sha256`, `read_set_artifact_sha256`. These can bind the accepted failed-GPU collector's original sealed CPU observations and independently captured read-set artifact. No additional remote inventory or new preparation job is required by this API.
- `source_files_sha256`, `model_files_sha256`, `resources_sha256`, `limits_sha256`: independent contract digests, opaque to this helper. The outer must derive/compare them to its exact source/model/resource/limit objects; this helper does not qualify those contracts.
- `cpu_inventory_sha256`: canonical digest of `{cpu_roots, cpu_output_excluded_directory:'worker/artifacts/campaign', inventories:cpu_inventories}`
- `consumed_read_set_sha256`: canonical digest of the133-member map.

No future packet SHA is present. This context and the preparation receipt exist BEFORE packet freeze. The new packet later binds both pins and compares its execution fields to this descriptor. Original GPU01 files remain immutable; later creation is restricted by the integration authority to the previously absent GPU01/gpu subtree. This pure validator neither creates nor proves exclusive creation of that subtree.

## Receipt JSON (exact keys)

- `schema`: `ace-gpu-cpu-preflight-receipt-v1`
- `phase`: `CPU-prelaunch-complete`
- `execution_context_sha256`: exact independently supplied descriptor pin
- `inventory_before`, `inventory_after`: each `{canonical_sha256, file_count, directory_count, total_bytes}`. Both must equal the full independently supplied inventory digest and exact sums. Directory counts exclude the two roots themselves. No `verified:true` substitute is accepted.
- `consumed_read_set`: `{canonical_sha256, file_count:133, total_bytes}` matching the independent read set exactly.
- `absent_paths`: exact sorted six paths: frozen GPU01/gpu and the five original CPU policy service directories. These are authenticated recorded absence assertions, not local filesystem probes or proof of future absence.
- `started_utc`, `finished_utc`: strict UTC `YYYY-MM-DDTHH:MM:SS[.fraction]Z`, up to six fractional digits. Validate started≤finished≤independent `receipt_not_after_utc`; no cross-host monotonic values and no queue-age expiry.

Returns a new detached `ace-gpu-cpu-preflight-scoped-evidence-v1` record with the two original byte pins, context digest, inventory/read-set summaries, UTC bound and observed absence scope. Always `full_external_postcustody_required=true`, `continuous_immutability_proven=false`, `gpu_execution_qualified=false`, `acceptance=false`. It proves agreement of independently pinned metadata only; reviewed producer/authority must establish the underlying exhaustive verification and pin origins. All actual historical post-custody, consumed-byte authentication, original output creation and GPU runtime gates remain with the parent implementation/independent audit.
