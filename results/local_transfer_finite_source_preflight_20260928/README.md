# Finite-source transfer preflight (28 September 2026)

Protocol: `docs/development/guidance/protocol_transfer_finite_source_preflight_2026-09-28.json`. Source revision: `f895116475a37dfb0e9137d8f905c3879ddda1db`.

Twelve reused development systems (seeds 100–111), each with 30 source nodes, were observed in two separate arms: 16 or 64 noisy source examples per node. This is 28,800 counted source training responses across both arms. Each cell also has 3,840 independently generated noisy holdout responses used only for evaluation, totaling 92,160 holdout responses. These are synthetic source observations, not target intervention outcomes.

| Source examples per node | Mean clean-source MSE | Pooled 90% predictive coverage |
| --- | ---: | ---: |
| 16 | 0.020316 | 0.902409 |
| 64 | 0.002314 | 0.900543 |

All 24 cells have exact source counts, finite positive-definite posterior covariances, and nonzero clean estimation error at every node. Both coverage values satisfy the frozen [0.85, 0.95] gate. The complete receipt records the source revision, protocol hash, example counts, and summary hash. A deterministic replay matched the SHA256 hashes of the source data, fitted posteriors, evaluations, and suite summary for all cells.

This establishes that finite source estimates and their uncertainty can be generated and audited. The six-feature regression basis and noise level match the synthetic generator, so calibration here does not establish transfer under model misspecification. The preflight generator constructs target systems to identify latent old mechanisms, while the fitter receives only separate source observations; a later target-policy experiment needs stricter process separation. No target policy, target-query advantage, fresh-system confirmation, CURC job, or closed-model call is claimed. The next comparison requires its own frozen protocol and gates.
