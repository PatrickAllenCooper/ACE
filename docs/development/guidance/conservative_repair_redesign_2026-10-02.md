# Bounded conservative repair redesign: conditional evidence before replacement

Status: specified, not implemented or validated. Existing 12 exposed systems are debugging only. Prior residual repair harmed unchanged mechanisms; no scaling authorized.

Separate a parent-support shift from a change in the conditional child response. Reserve target observations for three disjoint roles: fit, selection, and final scoring, grouped by whole intervention setting. Fit source-warm and scratch candidates identically to controls. Use a diagnostic parent-intervention menu frozen independently of change locations. Charge those diagnostic responses to all arms; do not grant the repair arm extra labels.

First establish overlapping parent support from acquired inputs alone. Outside that support, abstain to the predeclared scratch predictor; label this a fallback, not evidence of mechanism change. On overlap, use paired validation losses to compare an unchanged source mechanism and a locally refitted mechanism. Select a replacement only when a one-sided 95% paired loss bound favors replacement after Holm correction across tested modules. If independent intervention blocks are too few to estimate a bound, abstain. No threshold adjustment using the exposed final-scoring observations.

Report parent-shift-only, mechanism-change-only, mixed, and no-change strata separately. Include source-warm, scratch, do-nothing, previous repair, and separately labeled oracle-change-location controls. The main comparison is against the strongest simple control selected using selection data only. Require <=5% unchanged-mechanism harm in every predefined stratum and improvement on changed mechanisms. Also report coverage/abstention and false repair counts; universal abstention cannot establish successful repair.

Validation prerequisite: inspect saved artifacts for per-example inputs, intervention IDs and residuals. Aggregated MSE alone cannot implement this redesign or verify its uncertainty bound. If unavailable, specify one small fresh CPU experiment with independent systems and budget accounting before generating data. No rerun of the exposed 12-system threshold search and no neural repair job. This is a bounded diagnostic proposal, not a proven safety guarantee; dependence across samples must be handled at the intervention-block level.

## Saved-data sufficiency audit (02 October 18:28 UTC)

All 48 acquired-data/action receipts and linked source/system hashes validate. There are 2112 recorded target rows. Each campaign has six recorded blocks (4 observational responses plus five 8-response intervention batches); individual motifs have natural labels in only four to six blocks. Inputs, values and masks are available, and block IDs can be reconstructed from the recorded trajectory counts without guessing row boundaries.

These data permit implementation debugging, but not a fresh test of the redesign: systems were exposed repeatedly, and the risk-policy batches were adaptively selected. Splitting six blocks among fitting, selection and scoring does not create independent experimental replication or justify a conventional paired-loss confidence bound. Do not silently count individual rows as independent intervention blocks.

Before any fresh run, replace the underspecified confidence test with a prespecified randomized diagnostic campaign independent of the acquisition policy, define its intervention-block estimand and finite-sample assumptions, and determine its size from an explicit power calculation. Disjoint blocks alone are insufficient for a bound on adaptively selected data. Keep oracle change labels exclusively in the evaluator. The existing redesign remains unvalidated; no GPU escalation or exposed-world threshold rerun is authorized by this inventory.
