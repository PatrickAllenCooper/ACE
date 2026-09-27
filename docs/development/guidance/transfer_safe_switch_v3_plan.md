# Development gate: protect unchanged mechanisms before source transfer

Date: 26 September 2026. Status: initial **development-only** odds-threshold screen completed locally; no v3 CURC jobs. Read alongside the [portfolio](research_portfolio_2026-09-25.md), [v2 protocol](protocol_transfer_bayes_v2.md), [v2 result](../../../results/research_transfer_bayes_v2/README.md), and [local switch diagnostic](../../../results/local_transfer_safe_switch_dev_20260926/README.md).

## Why change the rule

The v2 full-data Bayesian mixture improves family-change adaptation, but fails to establish 5% noninferiority on untouched nodes at 200 target samples and degrades their mean MSE by 19–33% at 120. Its `old_selection_mass` averages about 0.77–0.82 in k=1 settings, leaving material weight on source modules even though only one of 30 nodes changed. This aggregate includes the changed node and cannot by itself identify the cause of each untouched-node error. The earlier passive-SSE hard gate also failed to remove coefficient-change harm. A neural hypernetwork would add complexity before resolving this safety failure.

## Candidate architecture

Keep the old mechanism as an explicit protected expert. Use a *separate*, node-local evidence test to decide whether acquired data justify leaving it. If the test does not cross a threshold, fit the old-centered posterior only. If it does, fit an old/source-centered mixture and compare its predictive score on subsequently acquired examples against the old expert. Source selection must never receive the hidden changed-node label or the held-out test panel. Report detection delay and false switches as well as prediction error. The source library and its 2,560-example cost stay fixed.

Begin with the Gaussian feature-bank benchmark already implemented in `scripts/research/learned_transfer.py`; this tests the switching principle before any DNN build. Prequential evaluation can use the common ordered target examples: the first four form the passive assay, later examples arrive in the recorded order. A candidate switch at a given target budget may use only examples acquired by then. There is no free validation data. Later examples used to validate a candidate are counted in the same 120/200/400 target budget. Vary no simulator parameters based on fresh confirmation outcomes.

## Development screen and stop rule

Use seeds 100–111, both change types, k∈{1,3,10}, and budgets 120/200/400. Compare warm, v2 Bayesian mixture, and the protected switch on identical data and source library. Try only a small prespecified evidence-threshold grid (for example prior odds 4, 10, 25), then select a single threshold by the development criterion below. Save node-level switch decisions and per-node error, not just aggregate old-selection mass. Reconstruct the old and source predictive distributions from training data; no hidden mechanism parameters may enter switching.

Promotion to a *new frozen* 20-system confirmation requires at 120 and 200 examples: (1) untouched-node mean MSE no more than 5% above warm in every change type/k cell on development seeds; (2) family-change changed-node MSE at least 20% below warm at 200 in every k cell; (3) coefficient-change changed-node MSE no more than 5% above warm at 200; and (4) finite, receipt-validated per-node predictions and correct sample accounting. These development cutoffs are resource gates, not inferential claims. If no threshold passes, report the failure and keep warm start as the safe default. Do not add fresh seeds to the existing v2 grid to rescue its failed gate.

If the switch passes locally, freeze code, threshold, metrics, a fresh seed range and a paired-system interval rule before CURC submission. Use a pinned sparse checkout, account `ucb736_asc1`, ACE-only job names, exact revision/job/output manifest, full result custody and checksum validation. No Azure or other closed-source model API is needed.

The first odds-threshold diagnostic passes these development error gates, but uses the same acquired observations to select and fit the source model. Next implement a counted prequential confirmation segment and a misspecified-family negative control on development systems. Freeze a new independent-system protocol only if both preserve the gain and untouched-node protection. A passing in-sample gate alone is insufficient.
