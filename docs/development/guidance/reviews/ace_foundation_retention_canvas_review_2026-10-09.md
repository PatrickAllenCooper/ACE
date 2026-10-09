# Retention pilot canvas: artifact consistency review

Date: 2026-10-09. Disposition: **0 required findings** for the current embedded pilot result and its presentation mapping.

## Scope and exact inputs

Read-only review of the canvas dataset, React/SVG display logic, labels and scientific scope. Only this review file was written. No models, refits, scientific execution, prior tests, reporter execution, typecheck rerun, remote calls or commits. No rendered inspection was performed; this is not a pixel, layout or interactive-runtime approval. The separate full raw-result numerical review remains outside this review.

- Canvas: `/Users/pat/.cursor/projects/Users-pat-code-ACE/canvases/ACE-retention-pilot.canvas.tsx`
  - SHA256: `aa32302165d254b43d10b513e9f882597ea997aa58c5dd8d66eec7a2a2341be6`.
- Verified pilot summary: `results/ace_foundation_retention_20261009/ace-foundation-retention-pilot-20261009-01/verified_summary.json`
  - SHA256: `3d0903195419aa4d71b3e2e12bab178fd13735a337383a89300b9ad223eecfc4`.
  - Freeze lineage: `ef1f1f0907ba4aecb243819d18a50016afb6733a8b7372ac39e8ef4f5125dd80`.
  - Terminal lineage: `8ebe9573f2c2f172e6b3bb2da90286a9940848d7ae16289fd5f77c5081379711`.
- Results document: `docs/development/guidance/ace_foundation_retention_results_2026-10-09.md`
  - SHA256 at review: `e3d5bf1d8c7679469fdab22120d420d5e09d20bff7a39f18e5c6b15381b35733`.

## Dataset and display mapping

Stdlib parsing compared the embedded JSON to the saved summary. All 96 reference comparisons, 72 direct comparisons, 384 local-harm records and 24 selection dictionaries are exactly equal, including nested fields and ordering. All 216 embedded cells are exact projections of the source fields `seed`, `variant`, `method`, `status`, `metrics` and `diagnostics`; omitted prediction/parent hashes and timings are not altered or displayed as verified anew.

The full six-system × four-scenario × nine-method identity grid is preserved. Each of the 24 scenario/endpoint/comparison-mode views contains all six ordered seed pairs, with six direct or eight non-reference Grammar32 contrasts. The overview retains all four scenarios. Each selected-system detail retains all nine predictors and all five selector choices.

Ratio direction, geometric/arithmetic labels and strict wins/ties/losses agree with saved within-system ratios. Arithmetic and geometric transformations were independently checked from those saved ratios; no mismatch was found. Logarithmic coordinates and geometric-mean markers are finite and within the declared plot range for every current view. The headline counts match: PFN/RBF missing-Y 6/0/0, missing-M 1/0/5; Combined/Grammar32 null 1/0/5 and coefficient-M 0/0/6.

## Nulls, harm, support and claims

The summary has 216 complete cells, no verification issues or invalid numeric fields, and fully validated 960/18,432 response accounting. Every plotted ratio and unchanged-mechanism harm value is finite and defined. Thus current plotting/counting/averaging does not silently omit failed or undefined result pairs; this disposition does not qualify the UI for a different failed-result dataset.

All 216 outside-M partitions retain `count=0, mse=null`, displayed as raw JSON null and explicitly described as empty/undefined rather than zero error. The unchanged-mechanism filter matches the source flag; changed mechanisms are excluded from that specifically labeled harm table. Positive ΔMSE is correctly interpreted as worse than retention. Combined's null local-Y harm count is 4/6, and its displayed mean is the signed mean across all six systems. Predicted-parent composition support remains distinct from local measured-parent support in the displayed diagnostics.

The canvas states the fixed-history scope, unequal 32-versus-24 fitting rows, five composed calibration rows, fixed numerical-control limitation, and absence of intervention-efficiency or population-safety conclusions. It separates unchanged local preservation from upstream effects on composed output. Conditional model benefits are accompanied by the explicit warning that this RBF comparison does not isolate pretraining causally. The missing-Y guard-attribution warning agrees with identical composed endpoint errors across the four ablations in the summary and the results document; independent prediction-vector identity checking belongs to the separate raw-result review. Pilot CPU/RSS, source captions and summary links agree with the saved summary/document.

No required dataset, presentation-mapping, null/support or scope correction was found in these exact bytes.
