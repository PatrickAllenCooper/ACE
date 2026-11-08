# Screen training GPU supervisor conditional source review — 2026-11-08

**Required findings: 0 within the conditional qualifier-source scope below.** This document consolidates the completed source review and focused pin recheck; no review or tests were repeated to create it. It is not concrete-packet, worker, actual-preparation, resource-sufficiency or launch acceptance.

Reviewed source: `scripts/research/qualify_llm_intervention_screen_training_gpu.py`.

- Original conditional-review SHA-256: `e555b773de8a8b46086e19a0e131bc9195bbc4c953d40c0486c7fec0afb2551e`.
- Current source SHA-256 after focused pin recheck: `059bc0f426efc81dbab7bf7328e7f82c2285fd572ea243a69922429b431078fb`.

The original review assessed the new packet authentication, saved-token lineage, prepared-source bindings, import closure and parent/worker interface, while reusing the established supervisor/helpers. `PREPARED_SHA` and `PLAN_SHA` were deliberately unresolved at that time; authentication failed closed before staging or child launch. The conclusions were conditional on independently reviewed preparation/plan pins, the separately frozen and reviewed worker, and a later concrete packet/coordinator review.

## Conditional source-review conclusions

- Exact packet shape and file roster, independently pinned helper/data inputs, and captured input bytes bind the artificial request. Accepted capacity packet/runtime/model fields and CPU acceptance are checked explicitly. Prepared records bind the saved prompt IDs, complete command IDs, metadata, envelope digests, model-file digest and special token IDs. Four explicit artificial records and the pinned canonical plan are required.
- All 18 CPU-preparation sources, including `llm_intervention_gradient_probe.py`, are included in the qualifier's 28-file source roster. Static inspection found no uncaptured project imports in that roster. Captured-source loading rejects imports outside the closure; saved runtime snapshots must contain the complete expected project closure with matching hashes.
- GPU, supervisor, CPU and diagnostic helper hashes matched their fixed constants. The accepted allocation contract remains one `a100_3g.20gb` GPU, four requested CPUs (four to eight allocated), 12288 MiB, 600 seconds, `ucb736_asc2/aa100/gpu-testing`, no requeue or restarts. This bounds the proposed smoke run; it does not establish that the new full-shape workload fits those resources.
- Full archive staging, manifest/archive/source bindings, job-owned scratch, isolated runtime, model-file authentication, generated-source provenance and GPU mapping validation retain the established paths. The parent independently checks saved runtime and artifact records after the worker exits.
- Supervision logic is unchanged: verified GPU work before 240 seconds requires owned-process telemetry plus positive CUDA interval and allocated-memory evidence. After verification, the existing 90-second telemetry/progress watchdog applies. Absolute deadlines remain 540 seconds for the worker, 550 for supervisor work, 565 for cleanup and 580 for final completion, inside the 600-second allocation.
- Exclusive output creation, one child process, no internal retry, natural successful exit, owned-group cleanup and post-exit GPU checks remain intact. Global once-only submission, shared locks and accounting admission belong to the future concrete coordinator and are not supplied by this qualifier alone.
- Both `run_training` and `validate_outputs` receive `prepared_sha256` and `plan_sha256` keyword arguments; the inspected interfaces accept them. This interface check does not review or accept the separate worker implementation. Parent validation binds the worker's saved validation and fit-summary bytes, source/model authentication, runtime observations and ordered training/checkpoint phases. External scheduler and complete-custody acceptance remains explicitly required; scientific qualification remains false.

## Focused pin recheck

The focused recheck established that only `PREPARED_SHA`, `PLAN_SHA` and their adjacent explanatory comment changed from the original reviewed source. Restoring those two constants to `None` and restoring the original comment reproduced the original source SHA-256 exactly. No other qualifier logic changed.

- Prepared artifact: `results/ace_llm_intervention_screen_training_shape_20261108/preparation01/artifact/prepared.json`.
- Exact prepared-file SHA-256: `6ec85a1608d3e64c914b51b81605248be1a11dfbe879e0d463a9cc09b4da3712`.
- Canonical training-plan SHA-256: `0807b705f2bad022630629629e99a36f519b7e0f9e980464ec9a7b3385c10ded`, matching both the qualifier constant and the artifact's `training_plan_sha256`.
- CPU capacity-acceptance SHA-256: `69446fb29e24c34cff070d2f9f699a5d1d8d3e9a478c9d438adc01be7a290e59`, unchanged and matching the preparation's acceptance binding.
- Preparation's recorded source commit: `b070fe03dcc6c3ab32fefb961e2562417aaf8dcc`.

These checks authenticate the supplied bytes and digest bindings, not the actual preparation's substantive acceptance. That independent review remains separate. The subsequently reported worker metadata-field correction is also outside this qualifier review; a corrected, frozen worker must receive its own review before concrete packet acceptance.

**Optional suggestions: none.** The prior conditional source conclusion stands for current source `059bc0f4…`. No source execution, tokenizer/model work, numerical imports, tests, remote operations, submission or accepted-source edits were performed during these reviews. Only read-only source inspection, static AST comparison and byte/canonical-JSON hashing were used. This document records those existing findings without authorizing execution.

## Focused preparation-source roster correction

**Disposition: 0 remaining required findings for this focused source delta.** This appended disposition binds the latest qualifier SHA-256 `4665d03e2b910921e5b7e9bc2d74e24d07278852d9d7d5b89ec1cc56932d2b56`, superseding `059bc0f4…` as the reviewed source pin. Earlier sections retain the historical review and its scope.

The parent reported a newly exposed validator gap: the previous `all()` predicate accepted an empty preparation-source map and did not require the complete, exact preparation roster. A subset could pass; an extra map entry could also pass if its name/hash matched another packet file. The independently fixed prepared-file hash protected the actual accepted artifact from such mutation, but did not make the standalone roster predicate complete. This gap was not identified in the original conditional review.

The correction defines `PREPARATION_SOURCES` as the exact 18-name set and, at `validate_prepared` line 141, requires `type(prepared.get('source_sha256')) is dict`, then exact key-set equality, then every source hash to match the packet. Short-circuit evaluation rejects missing/non-dictionary maps before key iteration; exact set equality rejects empty, incomplete and extra maps before the per-source hash comparison. The 18 names match the source map in the unchanged, independently pinned prepared artifact, including `llm_intervention_gradient_probe.py`.

Static reconstruction removing only the new roster declaration and restoring only the old predicate reproduced prior source SHA-256 `059bc0f426efc81dbab7bf7328e7f82c2285fd572ea243a69922429b431078fb` exactly. Thus this recheck found no other source changes; existing resource, supervision, runtime, token-lineage and external-acceptance conclusions retain their conditional scope. The prepared artifact still hashes to `6ec85a1608d3e64c914b51b81605248be1a11dfbe879e0d463a9cc09b4da3712`.

No tests or source execution were performed for this recheck. The parent reported the other nine focused tests previously passed and Kuhn was rerunning only the affected method with three negative cases and a valid positive; this review does not assert completion of that rerun. The separately reported zero-required worker and actual-input reviews are recorded at `ace_llm_intervention_screen_training_gpu_worker_review_2026-11-08.md` and `ace_llm_intervention_screen_training_shape_review_2026-11-08.md` in this directory. Their existence was checked; their reviews were not repeated here. Concrete packet/coordinator acceptance remains separate and pending. Optional suggestions: none.
