# Transfer routing gate: development screen, 26 September 2026

This is a diagnostic on the existing development seeds 100–111, not a fresh confirmation or a full SCM result. The source library, target examples, and methods follow `protocol_learned_transfer_v1.md`. A fifth arm retains the old mechanism unless the best source prototype reduces the four-example passive-assay sum of squared errors by more than a fixed margin. It uses no changed-node label. All methods still receive the same target data and the source cost remains 2,560 samples.

At 200 target samples, mean changed-node MSE for warm/retrieval/guarded was:

| Margin | Change | k=1 | k=3 | k=10 |
| --- | --- | --- | --- | --- |
| 0.2 | family | .2497/.0441/.0441 | .1800/.0216/.0216 | .2139/.0430/.0476 |
| 0.2 | coefficient | .0539/.0570/.0565 | .0992/.1160/.1152 | .0622/.0889/.0878 |
| 0.8 | family | .2497/.0441/.0441 | .1800/.0216/.0228 | .2139/.0430/.0745 |
| 0.8 | coefficient | .0539/.0570/.0547 | .0992/.1160/.1152 | .0622/.0889/.0673 |
| 1.6 | family | .2497/.0441/.0592 | .1800/.0216/.0427 | .2139/.0430/.1110 |
| 1.6 | coefficient | .0539/.0570/.0547 | .0992/.1160/.1091 | .0622/.0889/.0660 |

Margins 0.1 and 0.4 were also checked and interpolated between the displayed outcomes. The simple gate fails: coefficient changes remain worse than warm start, notably k=3, while conservative thresholds discard family-change benefit. Unchanged-node means were close to warm in this development set, but this does not cure changed-node negative transfer. No fresh confirmation run is justified for this rule. A subsequent architecture needs to infer change type or combine an old-mechanism residual update with source reuse, then test on held-out forms before scaling.

Implementation: `scripts/research/learned_transfer.py --guard-margin` adds the diagnostic arm and schema-v2 receipt; `scripts/research/validate_cell.py` validates both v1 and v2 receipts. Smoke: one v2 coefficient/k=3/seed=100 cell validates at 15 rows; an archived v1 cell validates at 12 rows. The displayed sweep was computed in memory on the fixed development seeds; it did not generate a new results directory or use CURC resources.
