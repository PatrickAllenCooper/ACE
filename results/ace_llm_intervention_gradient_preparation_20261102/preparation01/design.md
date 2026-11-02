# Gradient-repeat diagnosis after failed GPU05

GPU05 stopped before updates because its identical losses had different parameter gradients. The original failure, tolerances and source remain unchanged. This new diagnostic separates repeat variation from path differences before changing precision or attempting longer-context training again.

## Frozen diagnostic design

Use the same pinned Qwen0.6B base, accepted artificial GPU04 adapter and613-token public task. There are no new worlds, responses, optimizer updates or parameter selection. CPU-prepared token and label bytes must match before model loading. The full source, runtime versions, prior receipts and new packet are authenticated.

The fixed order is defaultmode followed by strictmode. Within each mode, run full/full then selected/selected, for eight backward passes. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before imports in both modes. Reset to identical saved CPU/CUDA RNG states before every pass. Keep FlashAttention-only SDPA and BF16base/FP32adapter dtype. Strictmode uses torch.use_deterministic_algorithms(True, warn_only=False). Record the actual flags and prohibit fallback to another SDPA backend.

A non-mutating observer captures the output projection's input tensor and its actual backward gradient. It returns no replacement input or gradient. Save detached CPU copies, all112adapter gradients, loss, action-prediction positions, CUDA timing and hashes for each pass. Remove hooks and clear gradients even on failure. Verify base, adapters and reference buffers remain byte-identical.

All eight raw captures are saved before deriving ten comparisons. Compare repeat pairs, full/selected pairs in each mode and the first default/strict pair for each path. Align the full projection captures to the selected positions. Preserve each tensor's old absolute0.0001/relative0.01 tolerance result and descriptive norms. A numerical mismatch is a result of this diagnostic, not grounds to omit a repeat. Diagnostic completion does not qualify the failed longer-context recipe.

Repeat variation shows nondeterminism somewhere. Two stable repeats cannot prove its absence. Strictmode affects more than FlashAttention and cannot uniquely attribute a change to that kernel. Identical projection inputs with different input gradients localize divergence to the projection/loss boundary but do not identify the exact arithmetic cause. The separate design review has zero required findings.

The [versioned PyTorch2.9.1 implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu) selects deterministic Flash backward only under strict deterministic algorithms. This source fact motivates the control. It is not an empirical diagnosis of GPU05.

## Resource envelope

One A10020GB testing slice remains the smallest compatible slice. GPU05 observed5970MiB process memory and48scheduler seconds with40.15seconds to firstGPUforward. The same model and short context with eight backward observations justifies a bounded five-minute diagnostic request. No longer-context optimizer stage is included.

Request4CPUthreads, at most8allocated,12GiB,oneA10020GBslice,300seconds,accountucb736_asc1,aa100/gpu-testing andno-requeue. Workerstartup180seconds,worker250seconds,supervisor280seconds. Preserve25seconds for cleanup. The controller retains early and periodic owned NVIDIA process observations, actual CUDA events and phase progress. Unsupported strictmode behavior is a preserved failure, not silent fallback. Captured-tensor copies affect timing, so this is not production throughput.

A new attempt would reserve2400CPU-seconds and300GPU-seconds. Conservative cycle totals would become39600of57600CPU-seconds and3900of7200GPU-seconds. Known actual allocation before this diagnostic is2482CPU-seconds and344GPU-slice-seconds. These scopes overlap and must not be added. Both controller locks, previous journals, live queue, scratch write/read/cleanup, independent packet pins and review must pass before one exclusive submission.

Five new artificial methods pass from an unrelated cwd. They verify non-mutating observation against an unobserved backward pass, selected-position alignment, mismatch preservation, cleanup after failed forward and the shortened resource/deadline contract. No pretrained model or GPU was used by these local checks. Implementation and review overhead are not fully metered. The old completed checks and studies are not repeated.
