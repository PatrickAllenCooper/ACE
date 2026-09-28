# Direct PEV scoring-rule contrast on archived replication

This **exploratory secondary analysis** directly pairs integrated-variance PEV and naive-variance PEV on the 20 fresh shift30 systems from the [frozen endpoint replication](../research_pev_endpoint_replication_v1/README.md). It reuses only validated local files, makes **zero new environment queries**, and calls no model API. Analysis implementation revision: `6dc39a004ae3322d0a612a292507e8e334d69401`; source experiment revision: `49e1bb1ab1a7ab91eb30e2063feb3704382a77d0`.

The primary source experiment compared endpoint coverage to PEV. This direct PEV-versus-PEV-var contrast was inspected afterward, so its p-value is descriptive rather than a fresh prespecified test. The independent unit is the SCM seed, not one of the 32 repeated actions.

Both arms spent exactly 2,000 samples and used the same persistent ensemble and paired SCM at each seed. Final noise-free feasible nonroot MSE (lower is better) averaged **0.015347 PEV** and **0.017491 PEV-var**. PEV-var minus PEV averaged **+0.002144**, with a paired 95% t interval **[−0.000259, +0.004547]**, two-sided paired t **p=0.0773**, and PEV lower on **12/20** systems. Wilcoxon p=0.1536. The interval includes no difference and a small reversal; this is neither evidence of equivalence nor a resolved advantage for the integrated score.

The policies make materially different choices despite similar aggregate performance. At the same step they targeted the same node only **5.3/32** times on average and matched both target and value **1.55/32** times. Their selected target sets overlapped by 15.5 nodes on average; PEV visited 16.25 unique targets and PEV-var 17.5. Mean absolute intervention values were 4.539 and 4.780 respectively. These action summaries describe behavior; they do not identify which part of the integrated score caused the small outcome difference.

`per_system.csv` contains all 20 paired outcomes and action comparisons. `summary.json` and `complete.json` record the estimate, uncertainty, source hash, 40 revalidated cell receipts, and output hashes. The original 20 systems and 60-arm comparison remain in the source directory; this diagnostic did not alter them. The shared CURC connection was healthy with no ACE jobs queued at the check.

**Decision:** do not claim that integrated variance reduction outperforms naive variance. Do not run another same-generator seed sweep solely to reduce the p-value. A new PEV contribution needs a different environment or a separately frozen mechanism-level question with a clear effect threshold.
