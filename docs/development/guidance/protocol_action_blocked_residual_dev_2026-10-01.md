# Action-blocked residual-repair development diagnostic

Freeze date: 1 October 2026, before computing this diagnostic's outcomes. This is post hoc development on the same 12 already-scored fanout systems (seeds 2000–2011) and 48 saved target campaigns in `results/local_connected_transfer_heldout_dev_20260930`. It cannot confirm a new repair rule or support a GPU scale-up.

## Hypothesis

Sample-wise four-fold CV shares nearly identical parent contexts from an intervention batch across training and validation. It can prefer a local RBF residual that interpolates an action batch yet extrapolates badly to other feasible actions. Hold out **whole acquired action blocks** instead: block 0 is the four observational assay responses; blocks 1–5 are the five eight-response pair interventions. For each motif, validate only natural child labels; some blocks have no such labels and are omitted. Require the assay and at least three total nonempty blocks.

Keep the prior source-warm linear fit, RBF widths and precisions, residual fit, CV selection threshold (at least 10% and 0.0025 absolute improvement), and acquired-context exponential taper unchanged. Select hyperparameters and activate a repair using acquired labels only. Score final predictions on the existing sealed feasible-action panel only after selection.

Primary development readout: number of selected repairs on the out-of-bank motif, number on unchanged motifs, all/changed/unchanged feasible MSE against the linear source-warm fallback, and worst per-motif degradation. The practical safety criterion is no >5% unchanged-motif degradation in each source-size/acquisition-policy stratum, plus improvement on the out-of-bank motif. These are diagnostic, not fresh inferential gates. Preserve every row, input/output hashes, and byte-identical replay. No new environment response, CURC job, GPU allocation, or model API call is needed.
