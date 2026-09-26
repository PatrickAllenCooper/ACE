# Persistent SCM confirmation v1

Frozen protocol: [`docs/development/guidance/protocol_persistent_confirmation_v1.json`](../../docs/development/guidance/protocol_persistent_confirmation_v1.json). CURC source revision: `01a6eccf5e4b03e42ecdd7ef61b6cf327c1a9cc9`. Account: `ucb736_asc1`. Remote source: `/scratch/alpine/paco0228/ACE/results/research_persistent_confirmation_v1`. The 160 job IDs and output paths are in `submitted.tsv`.

All 160 jobs completed with exit code 0; 160/160 local receipts validate. Every arm used 32 intervention steps and exactly 2,000 environment samples, including observational refreshes, with zero candidate-probe samples. Four arms within each seed share a serialized system SHA-256 hash. There are 20 distinct system hashes per family. The local files match CURC under `rsync -nrc`.

## Prespecified primary result

The primary metric is final feasible-intervention **non-root** mechanism loss; lower is better. PEV integrated variance reduction versus graph-matched non-leaf coverage:

- Homogeneous 30-node: mean paired difference **−0.01360**, 95% t interval [−0.02151, −0.00569], 20/20 seeds favor PEV. Two-sided paired t `p=0.00192`, Holm-adjusted primary `p=0.00384`.
- Heterogeneous 30-node: mean paired difference **−0.10919**, 95% t interval [−0.18856, −0.02981], 16/20 seeds favor PEV. Two-sided paired t `p=0.00961`, Holm-adjusted primary `p=0.00961`.

Sensitivity: Wilcoxon two-sided p is approximately 0.000002 for homogeneous and 0.00315 for heterogeneous; 20% trimmed paired means are −0.00768 and −0.06549. This supports the direction of the primary result without making the t test the only lens. Development seeds 42/123/456 were excluded from this confirmation.

## Limits and ablation

PEV and naive variance scoring are essentially tied on the primary metric: mean PEV-minus-naive differences are −0.00026 (homogeneous; paired t p≈0.54) and −0.00079 (heterogeneous; p≈0.90). Thus these data support **graph-aware uncertainty acquisition**, but do not isolate the proposed integrated variance reduction score as the reason. On held-out observational loss, all four policies are also close. The test remains synthetic, with known fully observed graphs and a fixed learner, cost model, and evaluation panel; no external-domain or foundation-model claim follows.

The broad parent-product-domain loss shows larger PEV gains than the feasible-intervention metric. The latter is the frozen primary metric because it evaluates attainable contexts rather than the domain PEV itself scores. The five-node pilot is a single fixed system and is not part of this 40-system confirmation.

Reproduce the verified paired summary with `python3 scripts/research/aggregate_persistent.py --root results/research_persistent_confirmation_v1`. The submitted source revision remains checked out on CURC until any pending revision-pinned work is clear.
