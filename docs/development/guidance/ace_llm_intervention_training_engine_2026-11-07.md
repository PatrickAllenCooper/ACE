# Bounded supplied-model training engine

November 7, 2026. Reusable SFT→fixed-reference DPO control flow, with new tiny
arithmetic tests only. No pretrained weights, actual Qwen forward, GPU, scientific
data, responses, downloads, remote actions or optimizer-recipe experiments here.
The screen protocol and frozen GPU helpers remain unchanged. This is not a new
Qwen qualification, scientific training prescription, model initializer or runner.

`llm_intervention_training_engine.py` has three stages. On CPU, `prepare_record`
validates independently pinned matched-record bytes with `matched_views`, then
uses the existing `prepare_task_tokens` for exact chat prefixes and complete
commands including EOS. It stores each prefix once, the command token table, raw
record, state/task/receipt/tokenizer pins, supervised label, strict pairs and ties.
The caller must authenticate tokenizer bytes; a supplied hash is not proof.

On CPU, `prepare_training_plan` accepts records and their independent pins,
explicit numeric recipe, stage/record/pair sequence, adapter name/shape inventory,
parameter cap, fixed checkpoint indices and outer provenance receipt pin. Every
SFT step precedes every DPO step; both stages must be nonempty. Absent strict pairs,
changed state/label associations, altered recipes and parameter overages reject.
Every record/pair gets an exposure count, including zero/repeated exposure. Flags
report exact one-pass SFT/DPO coverage. Ties remain evidence, not strict targets.
This is one-record SFT or one chosen/rejected pair per step, without accumulation,
automatic batching, shuffling, inferred epochs, loss selection or early stopping.

`fit(model=..., plan=..., plan_sha256=..., output=...)` accepts an already loaded,
admitted model and exclusive output path. Model, tokenizer, data, source/runtime,
provenance and resources remain caller-owned. The model must already have LoRA
adapters and the qualified `FixedRowFP32Projection`. The engine installs the
existing `DecoderCheckpointing`, enables eval-mode gradients and uses the existing
canonical selected-position scorers throughout; it does not repeat full/selected
comparisons, reload models, generate actions or execute environments.

Pass `gpu09_numeric_recipe()` explicitly. It takes arithmetic, rank8/alpha16,
LR.0001, beta.1 and 8192/32 token caps from the frozen GPU09 qualification source,
with the same AdamW defaults made explicit: betas(.9,.999), eps1e-8, weight decay0,
clip norm1, no alternate fused/foreach selection. One optimizer carries moments
across SFT and DPO. GPU09's artificial 3+3 step count is **not** a scientific
exposure default; only the caller's ordered schedule determines updates.

`qwen_gpu09` requires the supplied single-CUDA-device Qwen3 layout, 28 layers,
112 adapter matrices/1,146,880 parameters, BF16 base and FP32 adapters/reference,
SDPA, strict determinism, highest float32 matmul precision, TF32 disabled and
CUBLAS_WORKSPACE_CONFIG=:4096:8. Caller sets that environment before imports and
the declared seed109 before initialization. The engine checks current settings
and holds FlashAttention-only SDPA during fitting. Source/runtime authentication,
setting-before-import evidence and actual hardware qualification stay external.
`artificial_control` is restricted to artificial scope and CPU tensors. Its tests
mock scoring and recomputation counters; they certify control flow only.

Initial supplied base/projection signatures and adapter checkpoint precede any
updates. At the exact SFT boundary the intermediate adapter/optimizer are saved;
adapter buffers are snapshotted once into the fixed post-SFT DPO reference, whose
tensors are saved before reference scoring. Each unique scheduled pair is scored
once for the cache; the reference signature is checked during DPO and its scores
are checked again at completion. Final base/projection signatures must equal the
initial ones. Whole-base/projection signature scans require transfers on GPU and
must be included in the parent's resource envelope; they run only at endpoints.

Each step retains start, loss, named finite-gradient checks, norm/clipping,
recomputation count, optimizer-attempt/return and completion events. GPU execution
also records CUDA elapsed time and peak allocated/reserved memory. These timings
include evidence overhead, not standalone throughput measurements. The parent
still owns telemetry, deadlines, allocation accounting and process supervision.

Mandatory checkpoints are initial, post-SFT and final. Additional adapter/optimizer
checkpoints occur only at declared step indices. Failed runs save the current
adapter, optimizer, gradients, reference and partial history without rolling back,
retrying or selecting a successful subset. An exception inside optimizer.step is
explicitly unknown/partial. Failure persistence assumes working storage; checkpoint
write errors are reported. The API provides no automatic resume. It leaves the
model in eval mode, with adapters trainable, and disables its checkpoint controller
on exit; outer inference/reload handling remains separate.

Bounds: 256 records, 16,384 steps, 4,194,304 stored token IDs, at most112 trainable
adapter matrices and1,146,880 parameters; caller's explicit parameter cap may be
smaller. Prepared-plan pins bind token bytes; they do not prove honest tokenization
or scientific world separation. Every engine result remains provenance-unqualified,
even when it carries an outer receipt pin. Independent outer admission is required
for any scientific claim. This is a trusted sequential primitive, not a sandbox.

Remaining integration: reviewed real tokenized matched-data packet; caller-approved
exposure/checkpoint cadence; authenticated already-initialized Qwen/runtime and
GPU09 recipe; outer source/provenance/resource freeze and measured supervision;
independent review and later real-model qualification of the integrated engine;
separate numerical-family optimization and production inference/reload custody.
No CLI, source-auth framework, full-study runner or allocation is supplied.

New test artifacts and exact CPU/command receipts are retained in
`results/ace_llm_intervention_training_engine_preparation_20261107` and exclusive
external `ACE-LLM-intervention-training-engine-preparation-20261107-01`.
