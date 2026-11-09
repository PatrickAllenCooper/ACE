# Separate saved-metadata CPU audit supplement

Authoring only, 2026-11-09. This closes the **implementation coverage** of the first two explicitly open items in the closed aid's contract: saved runtime predicates and campaign-bundle semantics. It does not assert that actual CPU execution passed them. The running job's outputs have not been inspected. Independent review of this supplement is pending.

`audit_saved_cpu.py` is an exact, unchanged copy of closed source `6d56672582f8c41bdd01c64b446c1d1607316650c6357bec95a889c357ab7eb4`. No production file, repository file, live launch, acceptance schema or closed audit contract was changed. `oracles/` preserves the six pinned production reference sources; `oracle-pins.json` records original locations. The supplement reads their literal constants through AST only. It never imports or executes a production module, generated runtime source, bytecode, numerical policy, model or simulator.

## Existing-input API

After complete terminal custody **and a successful actual invocation of the closed aid**, use the same independently pinned aid config. No new metadata authority/schema is needed:

```text
python3 -I -B /absolute/path/audit_supplement.py \
  --aid-source /absolute/path/audit_saved_cpu.py \
  --config /absolute/path/original-filled-aid-config.json \
  --config-sha256 <same-original-config-pin> \
  --aid-report /absolute/path/original-completed-aid-report.json \
  --aid-report-sha256 <independently-pinned-actual-report> \
  --oracles /absolute/path/oracles \
  --output /absolute/path/new-exclusive-supplement-attempt
```

No concrete job/report pins are fabricated here. The supplement requires the closed aid's exact source SHA, the same config SHA, all nine completed aid checks, no aid errors, and `acceptance=false`/`scientific_qualified=false`. It joins the original archive hash and inner packet pin, then reuses `Audit.b/j/h` for exact selected/fetched byte reads and full archived opaque hash membership. It does **not** call the aid's custody, accounting, runtime, ranker or other check methods again. No remote verification/fetch is performed. Any missing required saved text is reported through the existing `necessary_exact_fetches` mechanism; preserve the attempt and let the caller supply exact fetched evidence under the original contract.

The only metadata-aid module loaded is the pinned stdlib aid itself, under a non-main name. The supplement's own `main` is not invoked by tests. Actual use is prohibited until the job has terminal custody and the original actual aid report exists.

## Exact coverage added

| Frozen oracle | Supplement coverage |
| --- | --- |
| `qualify_llm_intervention_service_cpu.validate_runtime_records` | Stage/contract/device binding, initial baseline, complete metadata checks, independent provenance equality, all three ordered cache/source pairs and admissibility flags, both post-import snapshots, exact package rosters/imported versions and generated-source provenance joins. The CPU-phase filename replaces the old tokenization filename exactly as the scientific producer does. |
| `llm_intervention_service_runtime.check_stage/check_metadata` | Archive/manifest/prefix,49966objects/8397575073bytes and model/runtime qualification flags; exact cache-source roster; normalized metadata package names and all expected candidate versions. |
| `qualify_llm_intervention_runtime_metadata.check_prefix/check_metadata` | Prefix/base-prefix/executable, before/after sys-path, only the allowed generated module exception, mixed module/native-path rejection, package version errors, both NumPy metadata versions and all distribution/default paths. |
| Saved generated-provenance fields | Independent source paths/pins, expected source content/hash, temporary-root identity, generated directory/file ownership/mode/size observations, owner type, exact module/spec/function origins and original source pins. No generated code or cache is executed. |
| Scientific producer `numeric_prefix/policy_specs` | Original saved absolute work deadline and stage/packet/source/bootstrap arguments,96 world-policy bindings, exact four numerical policy arguments/provenance, seed109/draws32 final-only DPO ranker checkpoint/training receipt, and the three declared persistent service specifications. |
| Scientific producer `validate_bundle` | Exact24 ordered IDs, every descriptor canonical hash, each complete source-plan hash, public graph/menu/task/prior/initial8/acquisition24, all eight policies and CPU/GPU phases, three services, proposal/source/world/service pins, four regimes via exact proposal row joins, graph/cost strata, arm namespaces, unchanged scientific flags and original outer provenance. Whole published campaign/source-plan pins and all recorded CPU decision argv are joined back to these exact specs. |

Coefficient-bearing descriptors and source plans are JSON-decoded transiently only for canonical hashing/equality and public identity/binding checks. Coefficients are never numerically interpreted, printed or included in the report. No actual private scores, true query means, predictions, noisy responses, private cost curves or utility outcomes are read. There is no world/response generation or model execution. Adapters and ranker tensors remain opaque full-archive hash evidence through the closed aid API.

Remote path predicates use normalized absolute POSIX paths from saved evidence instead of resolving remote paths on the Mac. They do not claim a new live filesystem/symlink inspection. Original runtime ownership, module/loader types, temporary-directory membership and absence of mixed native mappings remain authenticated observations from the unchanged producer. The supplement checks every predicate exposed by the original saved validators and their saved provenance fields; it does not attempt to recreate live Python objects or repeat cache compilation.

## Output and remaining actual-review scope

The new attempt retains the original config/report, exact supplement source and `report.json`, including findings, costs and completed facts. Both `acceptance` and `scientific_qualified` are always false. Failure reports retain traceback and missing evidence; no overwrites, resume or automatic retry. Reports emit only IDs, counts and hashes for descriptor/campaign evidence.

Once an actual run passes, the two coverage items above can be discharged by the independent reviewer using this report. The other original obligations remain: authenticated collector/helper/packet/source commits and namespace/Slurm/history joins, original whole failures, actual cleanup/accounting/core aid findings, and the independent actual acceptance decision. The existing exact16-key GPU-consumer CPU acceptance stays separate and unchanged. This supplement neither writes it nor adds downstream fields or a new review/authority layer.

## Retained new evidence

`attempt01` contains one run of three NEW artificial methods, all passed: runtime missing/mixed/cache/provenance mutations;24 tiny hand-built descriptors and rehashed plan/menu/task/graph/checkpoint/namespace/strata mutations plus executed argv disagreement; and early rejection of incomplete or mismatched original aid reports before evidence access. Hand-built coefficients are only literal zero/one; no generator, scientific law draw, simulator or response call exists in the tests. Fixtures use synthetic proposal pins explicitly inside the test harness; they cannot qualify actual inputs.

Reported cost:0.343597child CPU seconds,0.401979833elapsed seconds,26,624,000peak child RSS bytes. No old tests or actual audit was run. A pre-test static transcription typo in the base-oracle hash was corrected; its exact unexecuted draft and note remain under `draft01`. No test failures or reruns occurred. The source oracle hashes are also independently checked by static byte inspection in the preparation receipt.
