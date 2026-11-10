# CPU preflight metadata assembler — stable source-only handoff

Source: `assemble_cpu_preflight.py`.

SHA256: `eb5894f449c30d3da5c57e4772117cd7f63835822cbd00484506d598e89cd6c7`.

This source is frozen for Kuhn's independent review. **No assembler fixture, test method, or actual assembly has been executed.** Planned structural fixtures were not authored or run before the parent's instruction to hand off immediately. The parent's separate actual-metadata harness is the intended next qualification after source review; this preparation does not substitute for that qualification. No helper/history tests were repeated.

## Exact API

```python
outputs = assemble(*, inputs=inputs, pins=pins, assembled_utc=assembly_utc)
```

The notation above denotes keyword-only arguments; a Python invocation is:

```python
outputs = assemble(inputs=inputs, pins=pins, assembled_utc=assembly_utc)
```

`inputs` is an exact role-to-immutable-UTF8-JSON-bytes dictionary. `pins` has exactly the same keys and independently supplied SHA256 strings. Return keys are exactly `expected_raw`, `receipt_raw`, and `assembly_record_raw`, each immutable JSON bytes. No CLI, filesystem access, clock reading, network, subprocess, source import, model, tokenizer or original project callable is part of this module.

| Role | Required original metadata |
|---|---|
| `authority` | Adopted `ace-gpu-cpu-history-recovery-authority-v1`, created before context/receipt. Current expected authority is b3630d…; the assembler does not hardcode that mutable integration choice. |
| `failure_acceptance` | Independently accepted failed GPU01 attempt, job35498218; published pin cacfe32a… |
| `failure_proof` | Its original no-policy-work proof, pinned by failure acceptance; published pin164ab633… |
| `baseline` | Accepted47-job/96-journal accounting with failure acceptance/disposition/predecessor joins; published pincf0982dc… |
| `cpu_acceptance` | Original accepted CPU campaign metadata, pin3c5ce737… |
| `before_observation` | Original `terminal01/inventory_transport.json` bytes; pinf9c7dc2e… |
| `after_observation` | Original `terminal01/inventory_after_transport.json` bytes; pin4303d7d4… |
| `custody` | Original `terminal01/custody-failure-inventory.json`; pin8e0ce58a… . This is not a manufactured successful collector seal or the separate audit's custody-review record. |
| `remote_custody` | Original `terminal01/remote-custody.json`, pin5224bf51…; full original CURC archive/map/readback record. |
| `readset` | Actual `readset01/cpu-history-readset.json`, pin9be82230…; schema `ace-cpu-history-readset-v1`. |
| `original_gpu_packet` | Original **inner** GPU packet, `launch01/inputs/packet.json`; not the outer coordinator packet. Its byte pin is joined to the archived `launch/inputs/packet.json` entry. |
| `original_cpu_packet` | Original CPU child packet, as archived at `launch/inputs/cpu-packet.json`; supplies the unchanged model-file map. |

The original outer packet43680732… is joined through independent failure proof/authority. The original entry26940935…, source revisiona9346022… and job35498218 are fixed predecessor identities. All actual role pins remain caller inputs; none are derived from live paths. The parent's invocation manifest/harness owns file reading, source authentication, exclusive output creation and retention of actual failures. This pure function issues no execution authorization.

## Authority and schema contract

The authority must have the normal fields consumed by orchestration02's `recovery_contract`, plus independently anchored `receipt_not_after_utc`. Extra independent fields such as `source_review_sha256` and `scope` are permitted. Future context/receipt/packet hashes are rejected under their ordinary output-key names. No authority output is produced or modified here.

`historical_proof_pins` must be exactly the four helper roles: `custody_sha256` binds the original failure seal; `before_observation_sha256` and `after_observation_sha256` bind the two original transport records; `read_set_artifact_sha256` binds the captured133-member artifact. The seal must retain two identical original copies and exact before/after/remote-custody members. Its retained `pull_terminal.py` pin must match `historical_verifier_sha256` (collector0ecf0f97…). Thus context lineage `source_sha256` denotes the historical verifier, not this assembler.

The authority's `operational_source_files` must have exactly56 entries: the original54 source names, with only the GPU entry hash replaced, plus `llm_intervention_cpu_preflight_receipt.py` and `llm_intervention_cpu_history_admission.py`. The assembler accepts the authority's reviewed new hashes; it does not hardcode history997 or any superseded entry/helper pin. Its expected context derives model-map digest from the pinned original CPU packet and resource/limit digests from the pinned original GPU packet. Optional authority `resources`/`limits` objects, if present, must match those originals.

The output `expected_raw` and `receipt_raw` use the exact schemas in `references/preflight-interface.md` (source interface pin2dc4ffc7…). Canonical semantic digests are sorted, compact JSON with no newline; output byte strings append a newline and have separate byte pins. Backend4161 archive inventory pins instead use sorted, indented JSON plus newline; this deliberate difference is implemented explicitly.

## Saved evidence derivation and timing

Original before/after directory containers must be lists with unique normalized names and complete parents. Their traversal ordering is normalized only after those checks; dictionary/string coercion and duplicates are rejected. Both complete maps must equal the archive's full map/readback. The `cpu` archive label supplies the CPU origin; the accepted logical CPU output is sliced from it, then `worker/artifacts/campaign` is separated. The resulting full file hashes must match the two exact CPU acceptance maps. Directory rosters, including empty directories, descend from the sealed observations rather than being invented from the old file-only acceptance.

The assembler checks absence of GPU01/gpu and all five CPU service paths in both observations. The fixed133-member readset's UTF8 is re-encoded without decoding its scientific contents, and its exact byte lengths and hashes must match the full accepted campaign map. It does not recapture historical files or inspect models/private outcomes.

Receipt `started_utc` is the saved before-scan UTC; `finished_utc` is the saved after-transport completion UTC. The currently observed exact bounds are22:24:12.370227Z and22:46:32.850513Z, not an invented22:20/22:47 pair. All four scan/transport timestamps are retained separately in the assembly record. `assembled_utc` appears only in that record. The receipt's independent upper bound comes from authority `receipt_not_after_utc` (current authority23:13:52Z). No current-clock call, age TTL or scheduler-wait expiry is introduced.

Only named full-map input roles (`before_observation`, `after_observation`, `custody`, `remote_custody`, `cpu_acceptance`) have128MiB caps; readset has16MiB, other metadata32MiB, total inputs512MiB. Expected/receipt outputs retain the helper's128MiB bound. No payload/artifact size, resources, runtime or deadline limit changes.

The assembly record retains original collector return code and says `retroactive_collector_success=false`. It also states acceptance/GPU/scientific qualification false, full external post-custody required, continuous immutability unproven, and no fresh remote scan. No additional CPU Slurm scan or allocation is requested. Live submission/namespace/output guards, consumed-byte admission, actual runtime/GPU work, cleanup and final full historical post-custody remain independent gates.

## Preparation lineage and limitations

`references/` captures the exact interface/helper and the recovery-entry draft used to align field names. They are read-only references, not imports or authority for the future operational source closure. The parent's current committed source pins are supplied by the adopted authority, not these draft reference copies.

Development included static source reads and inspection of published accepted failure/custody metadata shapes. No assembler function was invoked against actual metadata. No project/scientific source or helper tests ran. One initial filename-only search listed terminal01 paths while locating acceptance files before the published seal; it did not read terminal contents. Later metadata inspection occurred after the parent supplied accepted complete failure custody. No remote access, locks, original file mutation, build, submission, model operation or commit occurred.

Independent source review and actual metadata qualification remain pending at this handoff. The external mirror retains exact source, references, this API and source pins; `receipt.json` truthfully records zero fixture/actual executions.
