**Disposition: two required findings (P2).**

1. **P2 — Apply authenticated startup/session deadlines before hashing the remaining inputs.** [main()](/Users/pat/code/ACE/scripts/research/llm_intervention_model_service.py:308) arms a fixed 180-second timer, then calls `authenticate()`, which validates the freeze and hashes all source, model, adapter, and receipt files before returning. Only afterward are the manifest's startup and total limits applied. Valid configurations permit either limit below 180 seconds. For example, authentication lasting 20 seconds under a 5-second startup limit continues for those 20 seconds and rejects retrospectively; a shorter total limit can likewise be exceeded. The defect occurs with responsive reads of large files, independently of stalled-filesystem or GPU-kernel assumptions. Retain the bootstrap ceiling, but tighten it to the authenticated absolute startup/session deadlines immediately after validating the freeze, before the remaining file reads and runtime checks.

2. **P2 — Preserve generated evidence before rejection gates.** [decide()](/Users/pat/code/ACE/scripts/research/llm_intervention_model_service.py:279) receives the generated completion and token record, then synchronizes CUDA and rejects changed model tensors before saving that record. On a post-generation invariant failure, the surviving request file contains only an envelope digest and identity, and `failure.json` contains the exception/traceback; the already available prompt, sequence, and completion are lost. There is also no capture boundary around the shared [generation helper's sequence validation](/Users/pat/code/ACE/scripts/research/llm_intervention_language_policy.py:174), so a returned sequence rejected by the grammar is never persisted or returned to the client. Save bounded raw generation evidence as unaccepted output when it becomes available, before grammar/post-generation invariant rejection, and record acceptance separately. A failure before tokens are available should explicitly retain that fact. Keep rejection and no-fallback behavior intact.

Reviewed November 4, 2026. This is a bounded static review of the two new model-service files and the imported functions needed to resolve their behavior. The saved artificial receipt and preparation copies were read; no tests, service processes, models, GPUs, scientific workloads, or remote operations were run. Only this review document was written. The parent's guide and world-generator integration are outside scope.

**Authentication and captured imports.**

Static import inspection found no missing local dependency in the declared 13-source set. Authentication checks the pinned freeze, exact source/model inventories, seven model-file hashes, optional adapter and receipt bytes, and exact declared Python/package versions before importing project modules. Regular-file reads reject final-component symlinks and detect size/mtime/inode changes during reading. The captured-source loader rejects already imported members and uncaptured `llm_intervention_*` imports and compiles the captured bytes without consulting a `.pyc`. The service entrypoint itself is the already executing bootstrap; it is hashed and path-checked rather than re-executed through that loader.

Model inputs are subsequently reopened by Transformers. Consequently, model loading relies on the stated trusted-filesystem/no-hostile-concurrent-replacement assumption. Installed package contents and the interpreter are not authenticated byte closures; their versions are checked. These limits are disclosed and are not additional findings.

The handshake binds digests of the model-file and source-pin maps, the adapter bytes (or explicit frozen-treatment sentinel), and the complete numerical recipe. The authenticated journal separately binds the exact freeze digest, and adapter restoration records the receipt digest. Treatment labels, scope, limits, and receipt digest are not individually represented in the transport handshake; the caller must retain its exact freeze/command binding for those properties. Receipt bytes are authenticated without validating their scientific training meaning, consistent with the explicit `training_provenance_qualified: false` scope. No scientific training acceptance follows from the handshake alone.

**Generation and state isolation.**

The shared inference recipe fields and runtime versions match the frozen GPU09 packet: BF16 base, FP32 adapters, rank/alpha, strict determinism, highest FP32 precision, FlashAttention, canonical fixed-row FP32 output projection, and 8,192/32 token limits. The service also explicitly sets no TF32 and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables thinking, and uses greedy inference with zero training updates. This was a recipe comparison only, not a GPU09 result re-review or qualification transfer.

Initialization restores an optional adapter on CPU, freezes the model, selects eval mode, transfers it once, and installs the canonical projection. The decision closure retains that model/tokenizer. The generation helper validates the public envelope, constructs an unpadded input and fresh explicit generation configuration, supplies no preceding cache/history, and uses eval/no-grad with grammar-constrained decoding. No optimizer, reference-mode switch, private evaluator, or environment callback is introduced. Before/after tensor identity, storage, version, shape, dtype, and device checks detect the covered mutations; they are not full content hashes or proof against arbitrary runtime memory writes. Actual Qwen cache behavior, repeatability, throughput, and memory remain unqualified here.

**Lifetime, evidence, and artificial coverage.**

After authentication, stage alarms are capped by the absolute session end. Idle waits and decision work receive separate alarms, and the persistent protocol enforces request counts and bounded frames. Python signal delivery cannot forcibly interrupt arbitrary blocked native/GPU execution; external process/allocation supervision remains necessary. Exclusive journal writes are flushed/fsynced, and normal exceptions preserve failure metadata without retry. The two findings above limit the startup-bound and failure-output claims.

The saved report records seven new artificial methods passing, with no model/GPU execution. Those methods exercise file corruption, duplicate/nonfinite JSON, recipe rejection, adapter/receipt byte binding, captured-source execution, symlink/version rejection, and journal exclusivity. They do not exercise `initialize_qwen`, the main deadline transitions, or failed-generation retention. Their assertions are consistent with their narrow authentication scope; no additional execution was performed for this review.

**Exact reviewed bytes.**

Both current files match their `preparation01` copies.

- `llm_intervention_model_service.py`: `6cf85d93ee077d4e4ae5a768a766954a9c24d03a68f6503eacfd0ddf6d1d4623`.
- `test_llm_intervention_model_service.py`: `e642b04a58b8d9c6e9d1a6a78424bcf307ee3e601a12ca4220718a1f21fa96bb`.
- Current 13-source pin-map digest, using sorted compact JSON of filename to SHA-256: `41a14fa0e583e47f7ab3dae18661cfd8b357ad06ec975a6fd3abd6cd07d324a9`.
- `preparation01/test_receipt.json`: `47651923893deffc162db4051f666185f5ee0318d503260ce18e586337382ac1`.
- `preparation01/tests.stderr`: `fbc026d5ce2da9b8207e534ce4b1da4ab6ed5d5b5a37e54b6bd55651c3611a41`.

This disposition concerns the implementation as reviewed. It provides no real-model, GPU, allocation, training-provenance, or production qualification.
