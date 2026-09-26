# Learned numerical module-library transfer screen v1

Frozen protocol: [`docs/development/guidance/protocol_learned_transfer_v1.md`](../../docs/development/guidance/protocol_learned_transfer_v1.md). CURC source revision: `64ee14d236f00521d08d0ce633384f88397ba570`; account: `ucb736_asc1`; remote output: `/scratch/alpine/paco0228/ACE/results/research_learned_transfer_v1`. Job IDs 33004910–33004929 are in `submitted.tsv`.

All 20 jobs completed successfully. Each produced six settings (family/coefficient changes × k=1/3/10) and each setting has 12 rows (four methods × three target budgets). All 120 receipts and 360 budget-level comparisons validate. One source-library SHA-256 hash is shared across all targets. The source library cost **2,560 examples** and is reused; target budgets 120, 200, and 400 are additional. The local files match CURC under `rsync -nrc`.

## What the screen shows

At **200 target examples**, mean changed-node MSE for warm start versus nearest learned source retrieval:

- Family changes: k=1, **0.2717 → 0.0324**; k=3, **0.1627 → 0.0294**; k=10, **0.2048 → 0.0579**. Retrieval helps 18–19 of 20 fresh seeds in each setting.
- Coefficient-only changes: k=1, **0.0782 → 0.0893**; k=3, **0.0721 → 0.1257**; k=10, **0.0668 → 0.1054**. Retrieval generally harms these targets.

At the same budget, retrieval also raises unchanged-node MSE in every setting; for family k=3 the mean rises from 0.0300 to 0.0356, well beyond the planned 5% tolerance. The source library is therefore **not ready to promote** as a general transfer architecture. A change-type gate that retains the old mechanism unless evidence supports a family switch is the next diagnostic. It must be evaluated on fresh targets and must protect unchanged nodes; simply reporting the favorable family-change case would be misleading.

These are fixed-data local mechanism assays with a known feature bank shared by source and target. They do not test whole-SCM interventions, unknown function families, or a language model. Reproduce the full validated summary with `python3 scripts/research/aggregate_transfer.py --root results/research_learned_transfer_v1`.
