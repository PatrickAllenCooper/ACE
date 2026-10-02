# CausalMan source/API feasibility audit

This is a **static source audit**, not a simulator result. The read-only runner `scripts/research/audit_causalman_source.py` checks 12 files against the Git blob hashes at upstream revision `17529dad5ec8b8c691494c617b9af4533aa44bf8`, records SHA-256 hashes, and inspects code without importing it or unpickling data. A second cached execution reproduced the three JSON artifacts byte for byte. No environment response, model call, or GPU allocation was used.

## Findings that change the planned experiment

1. The micro subtree contains 1,018 files totaling **144,812,833 bytes**. The full package is unnecessary for a micro audit. Inventory sizes are repository bytes, not measured runtime RAM.
2. Micro has one configured product and 51 distinct configured subbatches. The integer `batch_size_run` fields sum to 14,813, but actual generated counts depend on stored path-dataframe lengths and have **not** been measured. Asking the high-level sampler for a few returned rows still generates full batches before subsampling.
3. Seeds affect random sampling and the selected rows. Structural graphs and equations come from fixed batch/path-specific pickle files. A seed grid is repeated sampling of the configured production process, not a collection of independently parameterized SCMs.
4. The sequential runner concatenates results from all paths/batches but returns only the last path graph; the public high-level call uses that graph when constructing observable columns and graph projections. Its output must not be treated as samples from one fixed graph without verifying that assumption.
5. High-level observable-column selection depends on constancy and the intervention target set. The public feature schema could therefore change between actions. Freeze a public schema independently, and fail on unexplained missing/extra columns.
6. `sample()` returns six objects, including full hidden data and graphs. Its README example unpacks four. A policy process must receive only the declared public response and intervention mask, with all other objects retained privately.
7. `apply_interventions()` is a stub that clears the dictionary. Direct numeric `intervention_dict` is the implemented route. The FCM supports multiple intervention keys but evaluates string values; the adapter must reject strings, booleans, nonfinite values, unknown targets, and values outside the frozen numeric menu.
8. An upstream intervention table is constructed from the requested dictionary, so its presence alone does not prove a successful intervention. Check actual sampled target values and graph membership. Observable does not imply physically actuated: a simulator node intervention is not evidence of a feasible factory control.
9. The repository includes an AGPL-3.0 license. The audited packaging configuration supplies unbounded minimum dependency versions; runtime reproducibility will require an explicit environment lock. This audit records the license file, without resolving any separate redistribution terms for future derived datasets.

## Decision and next bounded step

Do not submit a multi-seed confirmation using the high-level mixture API. The next feasible route to investigate is the **unchanged lower-level upstream sampler on one pinned batch/path**, treated explicitly as one external system. This changes the estimand from the complete production mixture and must be named in the protocol.

Before a CPU custody submission: load only the required trusted pinned graph artifacts in an isolated environment; verify two observable node targets and their numerical ranges; freeze a stable public response schema; measure setup, query time, generated/returned rows, and host RAM in a bounded local engineering probe. Validate that changed sampling seeds preserve graph/parameter hashes and that distinct action calls use a declared fresh-noise policy. Separate any evaluation outcomes from acquired observations. A small static sample-count request is not yet a measured resource justification.

If the lower-level path cannot provide this boundary without rewriting mechanisms, stop CausalMan for this gate and audit the predefined Causal Chambers fallback. Even a successful single-path custody smoke would establish engineering feasibility only, not external generalization or a foundation-model contribution.

## Reproduction

Run `python3 scripts/research/audit_causalman_source.py --cache /tmp/ace-causalman-source --output /tmp/ace-causalman-audit-new`. The cache stores pinned upstream source and tree metadata. The output directory must not exist. `complete.json` records the exact audit-script hash and output hashes; `source_inventory.json` records each verified upstream blob. The audit downloads no bulk simulator pickles and executes no upstream code.

Primary source: [pinned CausalMan repository](https://github.com/boschresearch/CausalMan/tree/17529dad5ec8b8c691494c617b9af4533aa44bf8). Key inspected files: `causalman.py`, `sample_batch.py`, `utils/sampling.py`, `utils/graph.py`, `utils/data.py`, and `fcm.py`.
