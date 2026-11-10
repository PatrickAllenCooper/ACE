Batch04 prepares bounded GPU02 saved-evidence materialization for independently reviewed aid b76e13e2b70de3e1f638ef163427186e03ac6f69634a0d0c6914821db25b6d3a. Its closed aid review is cc2f9c4a82f9f85907cceb2f38a04c5512e5d0ebcc59ed894767830c1543c6ac. Exact bytes and pins are captured in references/ and reviewed-aid-binding.json. Batch04 itself still requires independent source review before use. No actual transfer, archive access, GPU audit, model/outcome read, lock acquisition or submission occurred during preparation.

The full batch03 source, dependencies, old tests/evidence and closed B1–B3 review are preserved under predecessor/. Backend4161 is unchanged. The scientific audit_names selector is unchanged. Captured bootstrap, source binding, both complete archive passes, framing, receiver/transcript cleanup, nonblocking two-lock implementation, exclusive no-adoption output claims, protected-capsule checks and deadlines are unchanged. The new public API adds explicit control-binding metadata; no implicit GPU acceptance is needed or created.

The exact new control binding is a canonical JSON object with seven keys:

```
schema outer outer_sha256 inner inner_sha256 result result_sha256
```

Schema is `ace-gpu02-audit-batch-control-v1`. `outer`, `inner` and `result` are the actual separately authenticated saved outer recovery packet, inner GPUv2 packet and submission result objects. Each SHA must match its original canonical saved bytes and the complete archived member at `control/packet.json`, `control/inputs/packet.json`, or `control/submission.result.json`. The entire binding gets its own independently supplied canonical SHA. The invalid template has null actual objects/pins; no future job or success is invented.

The binding enforces the v2 packet schemas, new control root20261109_02, frozen old20261108_01/gpu output, original failed GPU01 identity35498218,47/96 predecessor counts, exact three service freeze names, current CPU packet/acceptance pins, all inner/outer captured input pins, original failed acceptance/seal/disposition and required preflight/authority lineage. Current submission must be unambiguous, return0 and name a different job. Its intent pin must match the control journal. The archive receipt must label `control` and `launch` separately and bind the original CPU physical root. These are saved-byte/identity joins for selecting metadata, not acceptance of accounting, authority semantics, runtime or GPU work; the aid and independent reviewer perform those later checks.

Selection is the exact union of the original scientific inventory-derived metadata and these six members:

- `control/inputs/cpu-packet.json`
- `control/inputs/frozen_llm.freeze.json`
- `control/inputs/sft_llm.freeze.json`
- `control/inputs/dpo_llm.freeze.json`
- `control/logs/<ACTUAL_CURRENT_JOB>.out`
- `control/logs/<ACTUAL_CURRENT_JOB>.err`

The log names derive from the authenticated saved submission result. Both must exist and no foreign current-control log is allowed. The service names are the reviewed fixed three-policy names and must match the inner packet. Old `launch/inputs/cpu-acceptance.json` cannot substitute for the independently accepted CPU bytes at `control/inputs/cpu-acceptance.json`, even if an old matching copy exists. No arbitrary control subtree, private payload or extra member can be requested. Full archive membership remains verified twice, including omitted weights and unrelated archived members; only exact selected bytes are emitted.

Every selected member remains bounded32MiB. Scientific and control members share ONE combined512MiB selection bound. The binding/individual pinned metadata roles retain128MiB bounds, the combined request256MiB, aid source256KiB and review text128KiB. No separate control allowance increases the transfer budget. Large full-observation/context artifacts remain separate independently pinned role inputs to the aid under its explicit128MiB reader, not ordinary batch selections. Missing/oversized members fail; no truncation or partial-member substitution is allowed.

Public API changes, after the unchanged explicit source/backend `bind_sources` step:

```
members = select_members(full_inventory,
    control_binding=actual_binding,
    control_binding_sha256=independent_binding_pin)

fetch_audit(
    ...all original independently pinned receipt/seal/CPU/aid/review arguments...,
    control_binding=actual_binding,
    control_binding_sha256=independent_binding_pin,
    members=members, members_sha256=independent_member_map_pin,
    destination=NEW_EXCLUSIVE_PATH, helper_path=REVIEWED_HELPER_PATH,
    log_dir=NEW_EXCLUSIVE_LOG_PATH)
```

This pseudocode was not executed against real inputs. Original args and bounds are documented in predecessor/batch03. The action becomes `batch_gpu72_recovery_audit`; transfer/retrieval result schemas explicitly become `ace-gpu72-recovery-audit-batch-transfer-v2` and `ace-gpu72-recovery-audit-batch-retrieval-v2`. Results bind the control-binding SHA, current job, inner/outer packet pins, control/failed/output roots and original source/review/custody/member pins. Acceptance and retry_allowed remain false. The complete detached binding is retained as `recovery-control-binding.json`; `fetched.json` includes both scientific and control members for the aid's existing supplied-fetch interface. The caller's later mutations cannot alter the validated request or materialized records.

Aid/review arguments remain separately supplied bounded exact bytes with independent hashes, as required by closed B1. Batch04 does not import the aid or infer review authority from arbitrary text. Use the exact b76e/cc2f pair recorded in reviewed-aid-binding.json for this preparation. A future changed aid/pair requires its own reviewed selection compatibility; source flexibility is not an automatic approval. The newly supplied review was pinned after the three tests, without changing tested executable source or rerunning any method.

Three NEW focused methods passed once: current-control/old-copy/job rejection; exact31-member fabricated selection, over32MiB rejection and combined512MiB limit; mocked streaming/readback of all control bytes with post-detach caller-binding mutation and unchanged original fabricated capsules. Saved tests imported only this new batch and unchanged stdlib backend definitions; the aid remained opaque bytes. They bypassed the real-lock wrapper and replaced transport before invocation. No old test was repeated. The actual remote two-pass path was inspected statically, not executed; that is an explicit validation limit.

The17-key GPU acceptance and separate recovery acceptance interface remain unaffected. This is materialization preparation only. Parent must independently review this source and pin a concrete actual request before any later transfer, audit or acceptance.
