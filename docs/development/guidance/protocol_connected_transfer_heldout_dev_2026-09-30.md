# Connected transfer with an out-of-bank target mechanism: development protocol

Status: **specified, not run**. This development test follows the six-system connected pair pilot and the separate held-out-mechanism simulator audit. Freeze this document before looking at the new outcomes. This is not an independent confirmation of the earlier pilot.

## Scientific question

Does source-module reuse help when three mechanisms change in a connected 30-node SCM, one changed local conditional mean lies outside the fitted feature bank, and an upstream change shifts the input distribution of unchanged descendants? Separate the benefit of acquisition from the benefit of transfer.

## Systems and information boundary

- Generate 12 new systems with seeds 2000–2011, 30 nodes, 10 motifs, fanout topology, root SD .15 and child SD .15. These are development systems. Generate no result on these seeds before source code and this protocol are committed.
- For each source system, change motif 0's first coefficient by +1.0 and motif 5's interaction coefficient by +0.5. Keep motif 7's three fitted coefficients unchanged, but add `0.85*tanh(1.7*x1 + 0.8*x2)` to its **target-only** conditional mean. Thus motifs 0, 5, and 7 are locally changed; other motifs retain identical conditional functions even if their parent distributions move.
- Source training has 16 or 64 complete natural trajectories, treated as separate strata. Source acquisition cost is reported separately from target cost. No target label or evaluator value may enter the source prior, action selector, or model selection outside the counted assay and acquired batches.
- The evaluator uses a sealed, fixed feasible-intervention panel generated from an independent seed. It returns only final metrics, never loss gradients or individual truth values to the policy.

## Target acquisition and controls

Each arm receives the same four natural assay trajectories and then five batches of eight trajectories. Pair actions cost four units per actuated node *per trajectory*, plus one per trajectory: `4 + 5*(8 + 8*2*4) = 364` total cost, 44 acquired trajectories, 80 actuator uses. Natural motif labels enter all eligible local learners; when an action overwrites a motif child, its natural label is masked. Log natural and masked labels by action.

Compare two acquisition policies: the current posterior-risk pair selector and the prespecified fixed factorial-hub pair schedule `(motif 0, [-2,-2]), (0, [-2,2]), (0, [2,-2]), (0, [2,2]), (1, [-2,-2])`. They have the same menu and cost. Couple simulator noise by seed and step, without allowing candidate-scoring randomness to perturb the environment draw. Neither policy sees the sealed evaluation panel.

On **each identical acquired dataset**, fit (a) a source-warm linear posterior, (b) a broad-prior scratch linear posterior, and (c) a guarded source/broad mixture whose gate uses only acquired training labels. The mixture gate and its threshold must be fixed in code before any system 2000–2011 outcome is scored. If that gate is not implemented, run only (a)/(b) and explicitly mark (c) missing; do not infer a transfer advantage from a policy contrast.

## Correct outcome computation

For motif `j`, predict its conditional mean from its parent features. Score against the target **noise-free conditional mean on the same parent contexts**, not against a coefficient vector. For motif 7, target truth is `phi @ target_coefficients[7] + 0.85*tanh(1.7*x1 + 0.8*x2)`; a three-feature learner has an irreducible approximation residual. For all other motifs, truth is `phi @ target_coefficients[j]`. Separate changed motifs 0/5 (in-bank) from motif 7 (out-of-bank), unchanged descendants, and unchanged nondescendants. Evaluate raw MSE and a baseline-normalized MSE if scales differ. The existing `connected_transfer_pair_pilot.py` coefficient-only `score()` cannot be reused unchanged for this test.

## Validity and decision gates

Record system graph, full source/target conditional specification, source revision, protocol hash, seeds, action trace, exact cost, acquired trajectory identities, natural masks, finite per-motif errors, and SHA-256 receipts. Require independent recomputation of conditional truth and a byte-identical replay before interpretation. A Slurm completion or aggregate CSV is insufficient.

Development evidence warrants a fresh 20-system confirmation only if: (1) risk acquisition improves changed-motif feasible error over fixed schedule at equal cost for a **fixed estimator**; (2) source reuse improves changed-motif error over scratch on the **same acquired data** in both source-size strata, or a prespecified gate selects scratch when source harms; and (3) unchanged-node error is no more than 5% worse than scratch or fixed schedule under the same comparison. Report effect sizes and uncertainty; 12 systems do not certify population superiority. If source reuse fails, retain the numerical acquisition question separately and do not scale a neural module library.

This CPU development test can run locally. CURC is reserved for a larger frozen confirmation or a later neural prototype after the numerical gate. No Azure or other closed-model call is part of this protocol.
