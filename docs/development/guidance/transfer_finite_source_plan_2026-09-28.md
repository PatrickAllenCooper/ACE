# Finite-source modular transfer: next development gate

The current synthetic target generator copies `old` coefficients directly into every unchanged target mechanism. An [evaluation audit](../../../results/local_transfer_exact_old_audit_20260928/README.md) confirms bitwise identity on all 1,824 unchanged nodes of the archived 12-seed development grid. This isolates sparse-change detection, but makes unchanged source modules perfect before adaptation. The next C-track setting must explicitly include source estimation error.

## Construction and information boundary

1. Draw a latent source system with the existing four feature families and independent node coefficients. For every node, acquire a fixed, counted source dataset using an independent random stream. Fit a Gaussian linear posterior for that node. The target policy receives only the posterior mean/covariance, source-data count, graph/roles, and target observations; it never receives latent coefficients or changed-node labels.
2. Generate a target system from the **same latent** source coefficients on unchanged nodes. Apply the existing family or coefficient changes to k∈{1,3,10} nodes. Keep source/target observation noise and held-out evaluation independent. The evaluator retains latent truth under a sealed interface.
3. Report source acquisition and training cost separately from each deployment's 200-example target budget. Vary source samples per node only at prespecified levels (initially 16 and 64) to expose calibration and amortization, not to tune after target scores.
4. Compare frozen source, uniform warm posterior, protected source switch, adaptive allocation, and scratch. All target arms receive the same legal actions and exact total target-example budget. An adaptive arm must preserve per-node receipt counts and cannot evaluate hidden changed-node labels while choosing actions.

## Development gates before CURC expansion

Run a small local preflight on the existing 12 development seeds with **source-data streams newly separated from target streams**. Require nonzero source-estimation error on unchanged nodes, reproducible source posterior hashes, exact source/target counts, and calibration of source predictive uncertainty on a source-only holdout. Then test changed-node recovery and untouched-node harm against the strongest uniform warm baseline. If the old exact-copy result disappears, retain that negative result and do not train a neural hypernetwork on this generator as a rescue attempt.

Only after this numerical gate is credible should a learned 1–5M-parameter module encoder be considered. A source posterior over fixed features is a necessary control for the neural architecture; model size or a semantic label must add value beyond that control. No closed-source model API is part of this stage.
