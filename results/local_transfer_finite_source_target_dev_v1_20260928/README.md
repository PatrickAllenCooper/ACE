# Finite-source target development screen (28 September 2026)

Frozen [protocol](../../docs/development/guidance/protocol_transfer_finite_source_target_dev_2026-09-28.json); code revision `08620e9` (full revision in `suite_complete.json`). Source posteriors and their separate acquisition costs are in the [preflight](../local_transfer_finite_source_preflight_20260928/README.md). These 12 development seeds have been used before; this is not fresh confirmation.

For each of two source sample sizes, two change types, three changed-node counts, and 12 seeds, the screen gave frozen-source, scratch, and warm-source methods identical uniform target prefixes at budgets 120, 200, and 400. Each of the 144 cells uses at most 400 counted target responses, or 57,600 target responses across cells. The same source acquisitions are reused across change settings; their 28,800 training responses are a separate cost. The target generator draws 14 potential observations per node but only the prescribed prefixes are fitted or charged. The policy fitter receives source posteriors and acquired target x/y, while labels and truth are used only in scoring.

At **200 target responses**, mean MSE ratios for warm source versus scratch are:

| Source examples/node | Change | k=1 changed / unchanged | k=3 changed / unchanged | k=10 changed / unchanged |
| --- | --- | --- | --- | --- |
| 16 | Family | 3.815 / 0.028 | 5.766 / 0.033 | 4.212 / 0.026 |
| 16 | Coefficient | 1.507 / 0.030 | 0.956 / 0.028 | 0.951 / 0.033 |
| 64 | Family | 4.940 / 0.006 | 7.713 / 0.007 | 5.596 / 0.006 |
| 64 | Coefficient | 1.839 / 0.006 | 1.235 / 0.006 | 1.240 / 0.007 |

Smaller is better. The frozen warm-source usefulness gate fails: a strong source posterior protects unchanged nodes but locks the fit to the old mechanism at family-changed nodes. With 64 source examples the old module is more precise, and the changed-node failure is worse. Do not promote uniform warm transfer or a neural module encoder on this evidence. A local change detector and protected fallback are required, with both unchanged harm and changed recovery tested on the same counted target budget.

The run has 144 cell receipts, 38,880 node-method-budget rows, exact target counts, and source posterior hashes checked against the source receipts. A separate deterministic replay matched every cell and suite hash. An initial scoring implementation emitted numerical warnings but produced finite values; replacing its matrix multiplication with an explicit dot contraction removed warnings, and all 216 aggregate values changed by at most `4.5e-16`. The validated output here is from the corrected revision. No CURC job or closed-model call was made.
