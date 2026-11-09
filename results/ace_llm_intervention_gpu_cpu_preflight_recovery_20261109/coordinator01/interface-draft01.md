GPU02 coordinator authoring interface (not qualified; no build performed)

Original builder CLI is retained. All original metadata/pin pairs remain, notably CPU acceptance/baseline46/94. New required `--NAME PATH --NAME-sha256 SHA` pairs:

- gpu-failure-acceptance, gpu-failure-baseline, gpu-failure-packet, gpu-failure-result
- gpu-failure-disposition, gpu-failure-seal, gpu-failure-inventory, gpu-failure-inventory-after
- gpu-recovery-authority, gpu-recovery-review (Markdown)
- operational-source-manifest, operational-source-review (Markdown)
- cpu-history-expected, cpu-history-preflight, cpu-history-readset, preflight-review (Markdown)

`operational-source-manifest` exact keys: `schema=ace-gpu02-operational-sources-v1`, full `source_commit`, `files`. Files exactly the unchanged entry basename plus the two helpers, each `{repository_path,sha256}`. The coordinator captures these exact Git bytes; it does not run the entry generator or science. Authority `operational_source_files` remains the COMPLETE child source roster, not just those three.

In addition to parent's entry-consumed authority fields, the coordinator requires:

- `failure_evidence_pins`: exact map of six metadata role names gpu-failure-packet/result/disposition/seal/inventory/inventory-after to their independently supplied hashes.
- `failure_acceptance_contract`: independently adopted exact public subset of the ACTUAL reconciliation acceptance, including its actual `schema`, `status`, `job_id=35498218`, integer `required_findings=0`. Every supplied field must equal the actual acceptance. This avoids guessing Ramanujan's unpublished schema; adoption must pin the actual record and its no-work/resource reconciliation, not relabel the original collector's error. No acceptance is manufactured here.

The original failed disposition/seal pins are fixed to 545cee3c… and 8e0ce58a…. Original status remains terminal_custody_with_binding_disagreements with its sole resource-contract error and current_lineage_verified=False. Independent reconciliation, full before/after maps and exact original46+failed47 accounting are required. The authority/expected/receipt contain no future packet hash (no hash cycle).

Control root is scientific_campaign_gpu_20261109_02; output remains scientific_campaign_gpu_20261108_01/gpu. Old root exactly unchanged before submission, output absent including dangling links; only that accepted failed sibling is allowed. Original CPU inputs/services/endpoint contracts remain unchanged. Launcher CLI unchanged; its concrete committed packet captures all new pinned authority bytes. Historical reconciliation is repeated before ONE sbatch; ambiguous results are never retried.

Monitor reads new-control intent/scheduler but old-root/gpu public telemetry. Collector separate completion pending; must include each whole root once, allow only gpu/ under old root, and preserve independent old failure custody/47-96 -> new48-98 history.
