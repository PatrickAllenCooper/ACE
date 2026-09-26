# Fresh shift-family PEV confirmation

The [frozen protocol](../../docs/development/guidance/protocol_pev_shift30_mean_confirmation.json) specified 20 fresh coefficient systems (seeds 5000–5019), four policies, a 2,000-sample budget per policy, and final held-out noise-free feasible non-root mechanism error as the primary outcome. All systems share one fixed 30-node graph topology. Lower error is better; no checkpoint selection was used.

All 80 CURC jobs completed with exit `0:0` on account `ucb736_asc1`. The exact job IDs and output paths are in `submitted.tsv` (first 33013232, last 33013315, with gaps). The jobs ran from pinned scratch checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_confirm_0a5c43d`, revision `0a5c43d3c57b956715d2deea1278967f0d7288f0`, and wrote to `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_mean_confirmation`. Local validation passed 80/80 schema-v3 receipts, 32 steps and exactly 2,000 executed environment samples per cell, zero candidate-probe samples, matching source revision and system hash across the four arms of each seed, and finite final metrics. All logs and cell artifacts are retained here. A CURC/local checksum dry run found no differences. Slurm accounted 2.030 total elapsed job-hours and 10.149 allocated CPU-core-hours (five allocated CPUs per job).

| Policy | Mean final noise-free feasible non-root MSE, 20 systems |
| --- | ---: |
| Non-leaf coverage ensemble | 0.039791 |
| Non-leaf random ensemble | 0.043898 |
| PEV integrated variance reduction | 0.015430 |
| PEV naive variance | 0.015871 |

The **prespecified primary** paired contrast, PEV minus coverage, is −0.024360 with 95% paired t interval [−0.048847, +0.000126] and two-sided paired t p=0.05108. This does **not** meet the protocol's criterion of an interval excluding zero. PEV is lower on 19/20 individual systems; the median paired difference is −0.009720, and a post hoc two-sided sign test gives p=0.000040. One system (seed 5013) contributes a −0.239253 difference, making the paired mean and t interval sensitive to the tail. The sign test is a useful robustness description, not a replacement for the frozen primary test.

Secondary contrasts: PEV minus random is −0.028468, 95% paired t interval [−0.048386, −0.008550], p=0.00750, lower on 19/20 systems. PEV minus naive variance is −0.000440, interval [−0.002278, +0.001397], p=0.62176, lower on 10/20 systems. The uncertainty-based policies are very similar; these data do not attribute a gain to PEV's integrated-variance formula over simpler variance scoring. The corresponding noisy feasible non-root means are 0.60226 coverage, 0.60665 random, 0.57958 PEV, and 0.58046 naive variance; this metric includes irreducible observation noise. Held-out observational total loss is about 5.56 in all four arms. Broad total loss favors PEV but measures a different context distribution.

The result supports further study of uncertainty-guided intervention under function-family shift, while leaving the prespecified mean-effect claim unconfirmed. It establishes neither graph-topology transfer nor an external-domain result. Do not extend this grid solely to chase a threshold. A next PEV experiment should first address the naive-variance tie and use an independently chosen graph or environment with a frozen primary outcome.

Reproduce the validation and paired analysis with `python3 scripts/research/aggregate_persistent.py --root results/research_pev_shift30_mean_confirmation`.
