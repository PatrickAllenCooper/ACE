**Disposition: one required finding (P1); implementation is not ready to freeze until it is corrected.**

Reviewed November 2, 2026. This is a bounded static implementation and OOM-design review of the four new checkpoint files, including the final capture-provenance and per-training-step recomputation changes. It does not repeat the GPU08 numerical-results review. Only this review document was written. No tests, model execution, allocations, SSH/Slurm operations, or implementation edits were performed. The user's report that three artificial methods passed was not independently rerun.

**Required finding — P1: supervisor and worker source inventories disagree.**

The worker's `SOURCES` contains 16 files, including `llm_intervention_checkpointing.py`. The supervisor's `required` set contains 15 and omits that helper. Consequently, every freeze accepted by the worker is rejected by the supervisor at its exact-membership check before child launch. Removing the helper from a freeze would instead make worker validation fail. Add the helper to the supervisor's authenticated source set; retain the exact-set and byte-pin checks. Static AST inspection found this precise set difference and no missing local imports in the worker's own 16-file closure. The worker declares 14 assets.

Locations: [supervisor membership](/Users/pat/code/ACE/scripts/research/supervise_llm_intervention_checkpoint_gpu.py:233), [worker membership](/Users/pat/code/ACE/scripts/research/qualify_llm_intervention_checkpoint_gpu.py:24).

**OOM diagnosis and smallest correction.**

The executed GPU08 source matches the current canonical worker and its scoring, learning, and capture helpers. Its [first training-step receipt](/Users/pat/code/ACE/results/ace_llm_intervention_task_runtime_20261102/gpu08/terminal/gpu08/worker/phase-027.json) records 16,565,905,920 peak allocated bytes for the 4,341-token SFT step. The [failure receipt](/Users/pat/code/ACE/results/ace_llm_intervention_task_runtime_20261102/gpu08/terminal/gpu08/worker/failure.json) places the next OOM in decoder input RMSNorm, before the vocabulary projection: 18.12 GiB allocated by PyTorch, 1.09 GiB reserved but unallocated, and a failed 58 MiB request on the 19.62 GiB slice.

No concrete cross-step graph-retention bug was found. Captures and their boundary tensors are detached CPU copies; hooks and gradients are cleared; comparison dictionaries contain scalar metadata; completed training losses are deleted; optimizer gradients are cleared before the next forward. Initial and reference scores use `no_grad`, and reference scores are kept on CPU. The singleton head reuses its one FP32 weight buffer. The OOM occurs before that head is reached. Frozen base weights still require activation gradients to propagate into upstream trainable adapters.

Live decoder activation pressure is therefore the best-supported diagnosis. Allocator fragmentation cannot be excluded from these receipts, but there is no evidence that cache clearing alone would make the complete long forward/backward fit. Explicit per-decoder activation checkpointing is the smallest supported correction preserving the two-row batches, token sequences, objectives, and six-update schedule. Its actual memory reduction and runtime remain unmeasured.

**Checkpoint implementation and state.**

[DecoderCheckpointing](/Users/pat/code/ACE/scripts/research/llm_intervention_checkpointing.py:12) wraps the existing decoder forward methods without adding module-name prefixes or changing parameter ownership. Each closure binds its own decoder and original bound forward method. It requests non-reentrant checkpointing with RNG preservation, independent of training mode, and bypasses checkpointing under `no_grad`. This preserves first-layer adapter gradients even when frozen embeddings produce inputs without `requires_grad`. The native Transformers checkpoint path is gated by `self.training`, which is unsuitable for the existing eval-mode scorer. These API properties were checked against [PyTorch 2.9.1 checkpoint source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/checkpoint.py) and [Transformers 4.57.3 checkpoint-layer source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_layers.py).

The wrapper checks eval mode and captures decoder/submodule training and LoRA-reference flags before forward. Its recomputation context rejects changed flags. The worker remains in eval mode between scoring and backward and retains its strict arithmetic/FlashAttention context. No optimizer update, reference snapshot, or reference-mode switch intervenes between a policy forward and its backward. The helper's flag guard is not a general detector of parameter or buffer mutation; the reviewed worker's ordering supplies that invariant.

Both `past_key_value` and `past_key_values` are rejected in checkpointed execution, as is `use_cache=True`. No-grad reference scoring bypasses recomputation. Generation runs on the fresh reloaded model with its ordinary cache behavior. No cache or frozen-reference defect was identified.

**Qualification, persistence, and terminal gates.**

Both comparison paths use canonical selected scoring; their capture `path` correctly remains `selected` for projection-boundary alignment. The new `checkpointing_enabled` and `checkpoint_recompute_delta` fields distinguish recomputation mode. Raw captures are saved before requiring exactly 28 recomputations when enabled and zero when disabled. A failed execution-count check is distinct from a numerical mismatch and aborts before optimization.

The eight SFT/DPO off/on repeat captures and eight comparisons retain the original loss/gradient tolerances and exact within-mode repeat checks. Numerical failures are accumulated; base, adapter, and projection preservation evidence is saved before the final numerical gate. AdamW is constructed only after that gate. The previous full/selected result is inherited through the pinned GPU08 disposition, whose independent acceptance is outside this review.

The latest training loop records starting allocation/context, checks at least 28 recomputation-context entries for each completed backward before `optimizer.step()`, and stores the per-step delta. `validate_terminal` also requires that delta for every one of the six unique training steps; the supervisor calls that validator. The old cumulative-counter-only weakness is therefore not a remaining finding. Counters attest to wrapper/recomputation entry, while CUDA events, progress, and independent process telemetry retain their separate roles.

The run starts from the authenticated GPU04 adapter, not GPU08's partly updated state. The original 4,341/7,343/7,343 schedule is retained for both SFT and DPO. DPO snapshots its reference after SFT and keeps reference scores and buffers fixed through its updates.

**Reload lifetime and limits.**

The controller holds flags/counters, not a list of bound decoder methods. Decoder closures do create local cycles with their own modules, but the worker deletes the controller, optimizer, trainable list, and model roots and invokes `gc.collect()` before allocating the fresh model. No surviving root retaining the full old model was identified. Captures are already detached and released. Actual post-release CUDA allocation still requires runtime observation.

Reload reinstalls the canonical projection and compares scores and base/adapter/projection signatures. It need not reinstall checkpoint wrappers for the no-grad score/generation checks, because those wrappers bypassed recomputation before reload as well. Existing 180/540/570/600-second startup/worker/supervisor/allocation limits remain unchanged. GPU fit, checkpoint gradient equivalence, long-context completion, reload memory, and scientific utility are not qualified by this review.

Reviewed source SHA-256 values, before the required correction:

- `llm_intervention_checkpointing.py`: `f8bd09b1c9ed296da0913ac81aa71d2ce57e89f0a7a448582f9c4298c28dccb5`
- `qualify_llm_intervention_checkpoint_gpu.py`: `fca1daced91b4e2c1de5e7e453800140638b776324114776ddded985a5919475`
- `supervise_llm_intervention_checkpoint_gpu.py`: `709e1b90790631cf9feb2f35d484430d982146d6ebe26e232d9662c88c7e5fdb`
- `test_llm_intervention_checkpointing.py`: `4cd2001815ad0d2d9724bf43353cfa79f948073f6ad376c1433d13369c875811`

**Final disposition after the narrow source-membership correction: zero required findings remain.**

This November 2, 2026 addendum supersedes the initial disposition above for the exact source hashes below. The initial finding and pre-correction pins remain preserved as review history.

The only implementation change from the reviewed snapshot is adding `llm_intervention_checkpointing.py` to the supervisor's `required` set. Static AST inspection confirms exact equality of the worker and supervisor source sets: 16 entries each, with no missing or extra member. The other three reviewed files are byte-identical to their reviewed versions. The four pre-correction source files and the initial review in `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-checkpoint-source-20261102-02` match the earlier recorded source pins and the complete pre-addendum review bytes.

Final reviewed source SHA-256 values:

- `llm_intervention_checkpointing.py`: `f8bd09b1c9ed296da0913ac81aa71d2ce57e89f0a7a448582f9c4298c28dccb5`
- `qualify_llm_intervention_checkpoint_gpu.py`: `fca1daced91b4e2c1de5e7e453800140638b776324114776ddded985a5919475`
- `supervise_llm_intervention_checkpoint_gpu.py`: `ad67abbb4929cdd8b41d90ea83a6197e8138e62340239a96f8161aee1a3091ec`
- `test_llm_intervention_checkpointing.py`: `4cd2001815ad0d2d9724bf43353cfa79f948073f6ad376c1433d13369c875811`

The implementation review is complete and the corrected source can proceed to a distinct frozen-packet review. This is not packet approval, submission authorization, or actual GPU numerical/memory qualification. Only the membership delta, source hashes, and preservation bytes were rechecked; no tests, model execution, remote operations, or full implementation re-review were performed.
