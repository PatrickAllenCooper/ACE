# Shared-task full-size GPU implementation review

Independent read-only reviewer Mencius, agent01a19e68-fd6b-7bc3-a9b3-98faf5f75dfd. Scope: new worker, derived supervisor, pure fixtures and long-context design. No models, tests, remote actions or edits were performed by the reviewer.

The initial review found one required issue. Tolerance checks raised before preserving the measured full/selected losses and gradient discrepancies. A failure would therefore lose the evidence needed to diagnose it.

The correction introduces compare_and_record. It saves each objective's losses, absolute difference, all named adapter tensor shapes and gradient differences, gradient magnitudes, tolerance decisions and the fixed tolerances before either loss or gradient rejection. Both supervised and preference comparisons use it. One focused method passed with a positive case and separate loss/gradient-failure cases. The initial five pure contract methods remain prior evidence and were not repeated.

The narrow independent recheck reported zero required corrections remaining. It verified the persistence ordering and both callers. It did not execute the target model, certify numerical equality or qualify a GPU run.

The implementation keeps the prior supervisor's exact allocation, startup, telemetry, progress, deadline and owned-process cleanup guards. New completion evidence covers full-size accepted-adapter restoration, short-context score/gradient comparison, six longer-context artificial updates, saved fixed-reference tensors, fresh-base policy reload and four constrained generations without environment callbacks. Final runtime and independent terminal-artifact acceptance remain pending.

Review planning also identified dtype/trainability, fixed-reference reload scope, actual SDPA backend, full token/mask freezing, fresh-base reload and measured complete-step memory as important. The new packet explicitly addresses these. FlashAttention-only execution is a new qualified recipe, not an inherited claim about arbitrary SDPA behavior. Memory peaks are accumulated before resets to include comparisons, reference scoring and reload.
