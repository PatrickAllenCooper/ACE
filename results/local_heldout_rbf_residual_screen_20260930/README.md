# Generic nonlinear residual capacity screen on saved target data

Post hoc exploratory analysis of the [connected held-out transfer development run](../local_connected_transfer_heldout_dev_20260930/README.md). Source revision `cf3eaa88`; 48 saved source-size × acquisition-policy cells. This screen targets motif 7 **because its failure was already known**, so it is an oracle-localized capacity diagnostic, not a deployable repair method or independent confirmation.

For each cell, a source-warm linear prior is fit on the same 44 acquired natural motif-7 labels as before. A 3×3 Gaussian RBF grid over parent coordinates is added as a residual basis. Width and ridge precision are selected by four-fold cross-validation on those **acquired labels only**; the sealed feasible-action panel is used only for final scoring. Widths are 0.75/1.5/3.0 and precisions 1/10/100/1000, fixed in code before this screen. Source and acquisition costs were already charged in the parent run. No new training query or model call was made.

Mean motif-7 feasible MSE, linear warm → RBF residual:

- Source 16, risk pair: **0.33247 → 0.10004**, 10/12 systems improved; paired 95% interval for RBF-minus-linear is [−0.34300, −0.12187].
- Source 16, fixed pair: 0.36668 → 0.22262, 9/12 improved; interval crosses zero.
- Source 64, risk pair: 0.32072 → 0.18452, 9/12 improved; interval crosses zero.
- Source 64, fixed pair: 0.31577 → 0.20350, 9/12 improved; interval crosses zero.

The source-16 risk mean falls below the three-feature oracle-linear floor of 0.12851 on this panel, demonstrating that the extra basis can represent relevant structure. The result does **not** show robust model selection or sparse repair, because motif 7 was identified using the prior result. A fresh test must select repair targets without oracle labels, train a nonlinear residual only where justified by acquired evidence, and report false repairs on unchanged descendants.

`metrics.csv` contains all 48 cells and cross-validation choices; `complete.json` pins the input-suite hash and output SHA-256. An independent rerun was byte-identical (SHA-256 `e2ce8977afff057679ca34ed96ec21d89e71cab0a2df0bd9baf4f3d1665257f9`). No CURC job, closed-model call, or non-ACE job was used; CURC SSH remained disconnected.
