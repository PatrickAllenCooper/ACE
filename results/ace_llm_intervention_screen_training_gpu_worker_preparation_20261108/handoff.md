# Screen singleton GPU worker handoff

The new worker is a separate file derived from the accepted historical worker. The old file is unchanged. Only this new worker, its new stdlib tests and preparation evidence were authored; parent-owned qualifier/engine/scorer/plan files were not changed.

API (both independent pins mandatory, no defaults):

```python
run_training(prepared, model_directory, output_path, observer, journal,
             *, prepared_sha256, plan_sha256)
validate_outputs(root, prepared, *, prepared_sha256, plan_sha256)
```

`prepared_sha256` hashes canonical JSON plus LF. `plan_sha256` is the semantic plan digest. Null/wrong pins fail before heavy imports. The parent must supply pins independently, authenticate actual saved tokenizer IDs and raw envelopes, and separately authenticate source/runtime/model bytes. The byte verifier does not load tensors. Neither the worker nor the verifier grants whole-run/runtime/scheduler/scientific acceptance.

The explicit artificial profile is four ordered records with prompt lengths 4638/4638/4744/4638. r0/r1/r3 share the supplied nonterminal prompt, commands and raw envelope. SFT labels are observe_all, set_x_neg, stop and set_xy. Scheduled DPO pairs are observe_all > set_x_neg (8/9 complete tokens) and its reverse (9/8); r3 retains unexposed set_xy > observe_all (8/8). Stop has six complete tokens. The first four updates are SFT, then two unique DPO pairs; one-pass SFT true, one-pass DPO false. Checkpoints are fixed at indices 3 and 5. The complete Qwen q/v inventory is 112 matrices / 1,146,880 parameters.

Execution uses `qwen_screen_singletons` and exact `engine.screen_numeric_recipe()` / `canonical_selected_unpadded`, with strict Flash and FP32 selected projection. Engine fitting owns reference/policy/final-reference scoring. Worker final and fresh SFT/final reload diagnostics use ordered unpadded singleton calls, including both mixed pair orders. The new reference receipt requires `unique_pair_score_calls=2`, `decoder_forwards=4`, `scoring_layout=ordered_unpadded_singletons`; legacy batched fields reject. Per-step checkpoint recomputation must be at least 28 for SFT and 56 for DPO, and at least 224 overall. Final/reload score counts are derived from the validated 28-command roster, not the old ten-command constant.

The inherited fresh-base/SFT/final reload sequence, exact score comparisons, ten legal generation requests, first/intervening/first repeat, signature checks, CUDA event/memory evidence, parent observer calls, zero-response accounting, cuBLAS workspace disposal, exclusive evidence and failure retention remain. CUDA intervals are work evidence, not utilization percentages. Parent owns process cleanup, real NVIDIA telemetry, inherited deadlines and resource admission.

Validation: 12 NEW stdlib methods passed in attempt-01, unittest 0.977 seconds, child CPU 1.066246 seconds, wrapper elapsed 1.069941541 seconds. Exact command/cwd/source pins are in attempt-01/execution.json; all source, stdout/stderr and fabricated output trees are retained and inventoried. No failed attempt occurred. The independent fabricated writer uses opaque checkpoint bytes and hand-written token IDs: it is not an actual-tokenizer or GPU qualification. No old tests, numerical libraries, models, tensors, tokenizer, remote operation, scientific data or response generation was executed. AST checks cover GPU wiring without executing it.

The actual parent-prepared artifact remains to be pinned and reviewed. This worker and the parent engine/scorer/plan/qualifier changes require independent review before any actual GPU qualification. No run, submission or commit is authorized by this handoff. Exact new and unchanged-old source pins are in handoff.json.
