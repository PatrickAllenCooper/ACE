# Prospective CPU campaign saved-evidence audit

Prepared 2026-11-09. **No actual CPU campaign audit or acceptance has been performed.** The new `audit_saved_cpu.py` is a stdlib, read-only evidence aid plus the mandatory source review plan below. It never imports a producer, executes a scientific function, contacts CURC, reads tensors, or writes acceptance. Only its new exclusive audit-attempt directory is writable. Its code has been parsed with `ast.parse`; it has not been imported or run. No tests were repeated.

The captured source references and their exact hashes are in `reference-source-pins.json`. They include the frozen CPU producer `ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893`, the actual GPU consumer's 16-key contract, the compact collector and the reviewed remote-custody backend. The scientific source commit remains `a9346022f145cc29439325b6f291597b2e6ec037`; freeze remains `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`. Capturing references does not qualify this new aid.

## Exact invocation and inputs

After complete actual terminal custody, make a **new** filled copy of `input-contract.template.json`, independently establish its SHA256, and run the reviewed aid once:

```text
python3 -I -B /private/tmp/ace-scientific-cpu-campaign-audit-preparation-v2qy5xca/audit_saved_cpu.py \
  --config /absolute/path/filled-input-contract.json \
  --config-sha256 <independently-supplied-64hex> \
  --output /absolute/path/new-exclusive-checker-attempt
```

`job_id` and both `capsules` paths are future actual values, deliberately null in the template. Supply repository and external compact terminal directories, each with the same original `custody-inventory.json`. The aid checks every local member against that two-copy seal and the selected raw subtree. It does not recursively copy the existing training capsule.

`artifacts` values are `{path: absolute_file, sha256: independent_64hex}`. Required entries:

| Key | Original object |
| --- | --- |
| `disposition` | Successful compact CPU collector `disposition.json`; custody only, never acceptance |
| `custody` | Compact capsule `custody-inventory.json` |
| `before`, `after` | Original `inventory_transport.json` and `inventory_after_transport.json`; transport return code 0 and JSON `stdout` observations |
| `remote_custody` | Original `remote-custody.json`, including full inventory, exact selected/omitted partition, roots, archive hash, source before/after and project readback inventory pins |
| `result` | Original successful `submission.result.json` |
| `local_intent`, `remote_intent` | Both distinct original intent byte records; semantic join allows only `remote_utc` and `fresh_readiness` augmentation |
| `outer` | Independently reviewed actual launch `packet.json` |
| `inner` | Exact `launch/inputs/packet.json` consumed by the CPU producer |
| `admission` | Exact `launch/inputs/admission.json` |
| `baseline` | Accepted training's 45-job/92-journal accounting baseline, already staged into the launch packet |
| `training_acceptance` | Existing independent actual GPU-training acceptance |
| `training_inventory` | Existing accepted training's original `inventory_transport.json` |
| `training_seal` | Existing full training ancestor `custody-inventory.json` (72,434,765 bytes, bound up to 128 MiB here); reuse its established acceptance, do not rerun its model/runtime audit |

Optional `artifacts.archive_verification` is a separately saved return object from the **existing** reviewed `verify_backup` API, if already available. Its canonical receipt hash, helper hash, full/selected inventories and archive hash must agree. It is not a new mandatory receipt layer or a request to run another transfer. Without it, the aid uses the original pinned collector/helper receipt's complete project readback and original before/after bindings. Independent review must authenticate the collector, helper, committed sources and transport records in either case.

`helper_sha256` binds the reviewed backend used in the actual collection; current template pins `527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b`. A legitimate later reviewed helper requires explicit caller pin update, not an inferred match.

`fetched` defaults to `{}`. Each necessary saved text member can be supplied as `"launch/...": {"path": absolute_exclusive_file, "sha256": original_full_inventory_hash}`. A fetch result is usable only if its size and hash match the full original inventory. The aid itself never fetches. It records the first missing required text member in `necessary_exact_fetches` and fails; preserve that attempt before supplying evidence to a new attempt. No automatic retry.

Existing input pairs in the template are current **packet metadata**, not future terminal pins or acceptance. Actual result/intent/terminal fields remain null. If the packet is replaced by an independently reviewed successor, update these pairs explicitly and reassess this frozen source scope.

## What the aid checks

1. Both compact copies, full original before/after inventories, authenticated remote archive receipt, deterministic local/omitted partition, and all claimed original input hashes. A failed/missing terminal, malformed inventory, changed inputs or preserved failed current-stage attempt blocks the success path.
2. Exact source/freeze, 2 CPU/7680 MiB/no GPU/6000 seconds, original startup180/work5940/hard5980 and numeric50/55 limits. It checks original running allocation and saved terminal scheduler request/actual resource evidence, then extends all 45 prior rows and 92 journal pins by exactly this job and its two original journals.
3. Full actual allocation charge arithmetic, including every earlier failed attempt: old accepted total plus `2 × actual elapsed seconds`, zero new GPU seconds; one existing 3600-second preparation allowance; 46 jobs/94 journals; 57600 CPU/7200 GPU global ceilings. CPU seconds here are allocated charges, not measured utilization. No future elapsed/cost is fabricated.
4. Original parent monotonic origin and inherited child deadlines, worker readiness before startup, zero exit, natural leader/group closure, no forced kill, no cleanup errors, saved parent output custody, staged archive/manifest binding and observed imported source/version closure. Process cleanup does **not** prove node-local runtime-directory deletion.
5. Matched numerical plan/exposure, exact source-derived width8891/parameter count1154817, checkpoint indices159/278, all 160 SFT +119 DPO updates, schedule order, completed-step roster, finite loss/gradient observations, gradient-presence records, fixed SFT reference, and both saved export/reload receipts. All six ranker `.pt` hashes bind the original full archive inventory; none are deserialized or claimed locally present.
6. Every one of five CPU attempts and every one of 24 development arms: complete original receipt/candidate/subtree pins, ordered identities, clean episode/adapter diagnostics, all sealed input hashes, all decisions and numeric-call records. No method or failed attempt is filtered away. Exact public response charges are reconstructed from original journals with their hash chains; response values remain opaque lexical spans. The eight shared initial responses cost eight units **per arm** (960 total across 120 arms); acquisition is bounded at24 per arm. Actual spend is summed, never assumed to exhaust the cap.
7. Numeric process/raw `.utf8`/stdout/stderr correspondence, unique call IDs, complete seven-file call inventories, original numeric source closure, runtime prefix/owned PGID, actual50-second bound and no LLM loads. Runtime observations are used as evidence, not model replay.
8. Exact CPU phase/worker terminal: 120 completed CPU cells, 192 declared cells, 279 ranker updates, no LLM training/tokenizer/world generation/GPU services/private scoring; original once-only phase order. GPU's remaining72 cells and final192-cell/28-pair evaluation remain future work.

## Mandatory independent source/runtime review plan

The aid is deliberately **not a complete automatic acceptance authority**. Its report always retains `acceptance: false` and `scientific_qualified: false`, even on exit0. `machine_checks_complete` means only the named checks above ran. The following original gates must still be assessed by the independent actual reviewer; they are repeated verbatim in every machine report so none silently disappear:

- Complete frozen `validate_runtime_records` behavior: `qualify_llm_intervention_service_cpu.py:136`, `llm_intervention_service_runtime.py:32`/`:45`, and `qualify_llm_intervention_runtime_metadata.py:473`. Check original before/after prefix/executable/base-prefix/sys-path, mixed module/native-path exclusions, cache/source pairs, metadata candidates, provenance source text/hash and generated-module exception, using saved records only. The aid checks the primary bindings/versions/captured closure but does not port every runtime predicate. Do not import the production entry: its top-level `_base()` executes another module.
- Frozen producer `validate_bundle` (`prepare_llm_intervention_scientific_campaign_cpu.py:526`) and `policy_specs`: exact 24 supplied world identities/pins, saved source-plan pins, public graph/menu/task/learner priors, shared8/remaining24, policy-specific namespaces, four regimes/cost strata, all eight ordered policy/phase/service identities, original numerical bindings and checkpoint roles. Reuse closed source qualification; inspect the actual saved binding evidence. The aid leaves the coefficient-bearing `campaign-bundle.json` and `campaign.json` opaque and therefore does not claim independent semantic recomputation of these objects.
- Reconcile all actual scientific input/model/runtime/source pins and the exact committed launch/collector/helper/review chain, both unchanged original/recovery namespace claims, full upstream preserved roots, and collector before/after scheduler observations. Check raw job/step accounting alongside the aid's allocation/row arithmetic. No assertion of immutable protection against same-UID/admin edits is made.
- Resolve `predecessor-reference.json` to the **existing** fully accepted training ancestor, including all failed TOKEN/collection attempts. Its strict seal is reused; this aid checks metadata/hash-role joins rather than repeating the full old raw audit. The compact capsule explicitly does not contain a recursive copy.
- Inspect every retained checker attempt and machine finding. Missing evidence or an unresolved gate blocks acceptance; exit0 alone cannot authorize it. No numerical model, tokenizer, tensor loader, environment callback or original finalizer is needed.

These are existing source/runtime/lineage gates, not requests for new scientific fixtures, training, review schemas or remote receipt layers.

## Exact downstream acceptance, after actual independent review only

The consumer's existing keys are copied to `cpu-acceptance-keys.json`; exactly16, no extras:

```text
schema,status,cpu_packet_sha256,job_id,allocation,cpu_output,campaign_root,
cpu_output_files,cpu_output_excluded_directory,campaign_files,bundle_path,
bundle_sha256,phase_terminal_path,phase_terminal_sha256,attempt_pins,
scientific_qualified
```

- `schema = ace-scientific-campaign-cpu-acceptance-v1`, `status = development_cpu_phase_accepted`, `scientific_qualified = false`.
- `cpu_packet_sha256` is the **inner** packet byte hash, not the outer launch hash.
- `allocation` is exactly `{account: ucb736_asc2, partition: acpu, qos: cpu-normal, cpus: 2, memory_mib: 7680, gpus: 0, state: COMPLETED, exit_code: "0:0"}`.
- `cpu_output` is the frozen logical remote CPU root ending `/cpu`; `campaign_root = cpu_output + /worker/artifacts/campaign`. Do not replace logical `/scratch/alpine/...` paths with Mac paths or observed `/gpfs` aliases.
- `cpu_output_files` is the **full original remote** hash map below `launch/cpu/`, relative to CPU root, excluding `worker/artifacts/campaign`. `cpu_output_excluded_directory` is exactly `worker/artifacts/campaign`. `campaign_files` is the full original remote hash map below that campaign root. Derive these from the authenticated full inventory, never the selected local tree.
- `bundle_path = worker/artifacts/cpu-phase/campaign-bundle.json`, `phase_terminal_path = worker/artifacts/cpu-phase/terminal.json`; their hashes come from that full inventory.
- `attempt_pins` contains the five CPU policy keys and their exact original attempt hashes, joined to the candidate and full campaign inventory.

Keep accounting, source/runtime/cleanup findings, custody references, costs, all120 per-cell counts and audit-source/review receipts in separate sidecars. Do not add fields to the acceptance schema, infer scientific utility or confuse allocation charges with response-cost units.

## Necessary fetches and integration gaps

- **No tensor fetch is necessary** for this metadata audit. `.pt` adapters/ranker states/export/reference files remain opaque hash evidence in the full authenticated archive. The aid reports their remote location and `locally_hashed: false` when omitted locally.
- Current-stage numeric `input.utf8` is in the reviewed text-selection policy; no2880 individual fetches should be needed. JSON/JSONL/log/source evidence under `launch/` is selected subject to32MiB/member and512MiB total. A selected-total overflow is a collector admission failure, not permission to silently omit evidence.
- Any required current-stage text object omitted because it exceeds32MiB requires a caller-supplied exact saved fetch; this aid permits up to128MiB per supplied metadata file. Larger objects require a separately reviewed bounded reader. It must not discard/truncate records. The complete training ancestor seal already uses the explicit128MiB metadata allowance.
- Full inventory acceptance maps remain compatible with the exact16-key consumer despite compact local storage. The real integration difference is that the old producer's `capture(.pt)`/filesystem `inventory()` validators cannot be called unchanged on a compact tree. This aid substitutes **authenticated full archive membership/hash joins**, not fabricated local files; remaining semantic/runtime predicates stay explicit above.
- The backend's project archive verification is receipt evidence, not a claim that Mac holds a full archive or that the institution guarantees backup retention. Scratch remains compute source; the archive is durable evidence. No additional remote operation is requested by this preparation.

Outputs of a future invocation are original filled config, an exact aid-source copy and `report.json`, including failure traceback, completed checks, first necessary fetch, opaque remote evidence, all available per-cell counts and local process/elapsed costs. Do not reuse an output directory or overwrite an unsuccessful attempt. No current actual output, outcome, cost or acceptance is asserted here.
