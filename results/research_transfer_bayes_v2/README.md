# Full-data Bayesian source-mixture confirmation

Run on 26 September 2026 under frozen [protocol](../../docs/development/guidance/protocol_transfer_bayes_v2.md). Twenty CURC jobs (33010132–33010144, 33010146–33010152) completed with exit 0:0 on account `ucb736_asc1` from sparse checkout `/scratch/alpine/paco0228/ACE/code_transfer_bayes_efa4dc9`, pinned revision `efa4dc91eca0e8c674fd43709f99529370dc5313`. CURC output was `/scratch/alpine/paco0228/ACE/results/research_transfer_bayes_v2`. All 120 setting receipts and 1,800 method/budget rows validate locally; the learned source-library hash is identical across settings. A checksum dry run against CURC showed no differences. The submit manifest and per-job logs are preserved here.

At **200 target samples**, mean changed-node MSE for old warm start / passive source retrieval / full-data Bayesian mixture:

| Change | k=1 | k=3 | k=10 |
| --- | --- | --- | --- |
| Family | .36070/.02439/.04506 | .20669/.04613/.03744 | .21276/.04139/.03584 |
| Coefficient | .06629/.05987/.05098 | .07475/.11217/.07643 | .06189/.08808/.06173 |

The Bayesian mixture retains a large family-switch gain over old warm start and substantially reduces the coefficient-change harm of passive source selection. The fixed-data source library consumed **2,560 training examples** before any target system; target budgets exclude that cost. The source family/feature bank is shared with targets, so this remains a favorable numerical transfer screen, not evidence for a neural or language-model foundation prior.

Protection of untouched nodes is the limiting issue. At 120 target samples, the Bayesian mixture's untouched-node mean MSE is 19–33% higher than warm across the six settings. At 200, ratio-of-means Bayesian/warm ranges from 1.017 to 1.035; paired system bootstrap upper bounds exceed the required 1.05 threshold for family k=1 (1.069) and coefficient k=3 (1.071). Therefore the prespecified 5% noninferiority gate is **not established**. At 400, untouched-node ratios are near 1.00, but coefficient k=3 changed-node mean is 1.052 times warm (95% unadjusted bootstrap interval 1.010–1.122). Intervals are descriptive and unadjusted for the six setting comparisons. This run does not justify claiming reliable safe transfer or building a neural hypernetwork on the current routing rule.

Reproduce receipt checks, means, and paired bootstrap intervals with `python3 scripts/research/aggregate_transfer_bayes.py --root results/research_transfer_bayes_v2`. The next architecture should explicitly protect an unchanged old mechanism at low budget while allowing a source-family switch when evidence is strong; it must also account for source-training cost. Do not increase seed count on this same favorable generator merely to rescue the failed gate.
