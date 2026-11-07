# Sagan: narrow training-engine implementation review

New supplied-model core only, November 7, 2026. Protocol and frozen helpers
unchanged. No CLI, source/runtime authentication framework, full-study runner,
model initialization, pretrained weights, GPU, remote, scientific data,
installation, Git action or allocation. Parent owns CPU35341862/GPU10 handling.

Changed files:
- scripts/research/llm_intervention_training_engine.py
- scripts/research/test_llm_intervention_training_engine.py
- docs/development/guidance/ace_llm_intervention_training_engine_2026-11-07.md

Review the SFT->fixed post-SFT-reference DPO loop, exact explicit GPU09 canonical
selected/FP32-projection/checkpointing/AdamW recipe, supplied-model parameter
bounds, CPU-prepared matched state/token binding, fixed ordered exposures,
intermediate SFT and reference checkpoints, continuous optimizer moments,
finite gradients/clipping/recomputation evidence, and failed partial custody.

The engine source was unchanged after its first test invocation. Six new tiny
arithmetic methods ran: five passed; one test assertion expected a later loss
guard, but the existing DPO helper correctly rejected nonfinite scores first.
That sole test expectation was corrected (method name now says nonfinite_scores).
Only the corrected method reran and passed. The other five test methods are
AST-identical and were not repeated. No existing test suite was run.

Attempt01: Ran 6 tests in 0.728s, failures=1, exit1.
Attempt02: Ran 1 test in 0.446s, OK, exit0.
Python3.9.6, local torch2.8.0. Actual Qwen/CUDA scoring and checkpoint recomputation
are NOT exercised: scoring and counters are mocked around tiny LoRA arithmetic,
with real AdamW updates and saved adapter/reference/optimizer states.
Both source versions, commands, outputs, CPU/wall receipts and all failure
fixtures are retained. No failed attempt has been relabeled successful.

Mandatory checkpoints: initial, post-SFT, fixed reference and final; additional
step checkpoints only at explicitly declared indices. Failure retains current
adapter/optimizer/gradients/reference and history, with unknown/partial status
if optimizer.step raises. No rollback, retry, best-checkpoint selection or resume.

CPU preparation owns matched_views/tokenization/grammar and full plan checks.
fit performs cheap plan/hash checks, supplied-model admission and fitting. Model
loading, tokenizer/source/runtime admission, source/model/provenance/resource
freeze, meaningful exposure schedule, telemetry/deadlines, and inference/reload
remain outer integration obligations. Results remain provenance-unqualified;
an outer receipt pin is lineage, not verification.

Please report required defects only with exact lines. No edits, models, remote
work, scientific draws or repeated closed reviews/tests are requested. This
handoff is ready for review, not a claim that independent review already passed.
