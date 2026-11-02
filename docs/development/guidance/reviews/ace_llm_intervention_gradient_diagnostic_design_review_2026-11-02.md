# Proposed gradient-repeat diagnostic review

Independent read-only reviewer Mencius, agent01a19e68-fd6b-7bc3-a9b3-98faf5f75dfd. Zero required design findings. No execution or edits. Implementation and actual packet remain unqualified.

A bounded eight-backward, zero-update diagnostic is reasonable after failedGPU05. Compare default and strict deterministic algorithms, each with two full-path and two selected-path backward passes. Keep the same valid CUBLAS_WORKSPACE_CONFIG set before imports in both arms. Record strictmode as torch.use_deterministic_algorithms(True, warn_only=False). Keep weights, input, RNGstate and Flash-only backend fixed, and clear gradients between passes.

Capture detached copies of projection inputs and their actual backward gradients without changing propagation. Align full-path captures to exactly the selected action-prediction positions. Save every one of the112parameter gradients and diagnostics before any comparison rejection. Do not adjust the failed attempt's tolerances.

A300second envelope requires new startup/worker/supervisor/cleanup deadlines that fit inside it. The observed48second run supports the bounded request but does not guarantee timing under strictmode. Resource and packet review remain necessary before submission.

[PyTorch2.9.1 source](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu) confirms that strictdeterministic algorithms enables deterministic Flash backward while warning-only mode does not. The existing scoring context controls evaluation mode, not kernel determinism.

Repeat variation establishes nondeterminism somewhere. Stability across two repeats does not prove its absence. Strictmode affects more than Flash, so improvement does not uniquely attribute the cause. Equal captured hidden states with different hidden-state gradients would localize divergence to the projection/loss boundary, without identifying its precise mechanism. An FP32 projection control can wait until these observations justify it.

GPU05 remains failed. The new diagnostic is not a scientific intervention study or a substitute for successful longer-context training and reload qualification.
