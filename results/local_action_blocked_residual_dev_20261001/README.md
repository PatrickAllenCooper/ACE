# Whole-action-block residual repair diagnostic

Frozen diagnostic protocol: `docs/development/guidance/protocol_action_blocked_residual_dev_2026-10-01.md`; source revision `d9ba55dc`. This reuses the 12 already-scored fanout systems and 48 saved acquisition campaigns from `results/local_connected_transfer_heldout_dev_20260930`. It adds **zero** environment queries, CURC jobs, GPU allocations, or model calls. It is post hoc development, not fresh confirmation.

The earlier four-fold sample-wise CV let near-identical contexts from each pair-intervention batch appear in both training and validation. This diagnostic instead holds out whole acquired blocks: the four-response observational assay or one of five eight-response intervention batches. Only natural child labels are scored. The RBF family, CV improvement thresholds, source-warm fallback, and acquired-context taper are unchanged.

Across 480 motif cells, the new rule selected 139 repairs, including all 48 out-of-bank motif-7 cells and **66 unchanged-motif cells**. It improves mean changed-motif feasible error but fails unchanged-module protection in every source-size/policy stratum:

| Source responses | Pair policy | Changed linear → repaired MSE | Unchanged linear → repaired MSE | Unchanged ratio |
|---|---|---:|---:|---:|
| 16 | risk | .11446 → .02941 | .00906 → .01053 | 1.162 |
| 16 | fixed factorial hub | .12440 → .03936 | .01016 → .01150 | 1.131 |
| 64 | risk | .10965 → .03723 | .00597 → .00727 | 1.218 |
| 64 | fixed factorial hub | .10716 → .02921 | .00681 → .00741 | 1.088 |

The worst positive per-motif degradation is seed 2007, source64, risk policy, motif 0: .01427 linear versus .13106 repaired. The 5% unchanged-module protection criterion fails in every stratum. Holding out action blocks is therefore insufficient; this rule is not promoted and does not justify a neural GPU scale-up. A future repair architecture needs stronger abstention or explicit support-shift evidence, tested on new systems after design is frozen.

The input and output SHA-256 receipts match, all 480 unique cells are present, and every linear feasible score matches the earlier saved-data run within 1e-10. A second full execution was byte-identical (`diff -qr` returned no differences). Raw cell metrics are in `metrics.csv`.
