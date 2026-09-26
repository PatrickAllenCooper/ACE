# PEV shifted-mechanism pilot

Twelve CURC jobs (33012701–33012712) completed with exit 0:0 on account `ucb736_asc1` from sparse scratch checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_7b4b3b0`, pinned source revision `7b4b3b0ba6f3fab9e06cd3317f6599683882d2c9`. The CURC output root was `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_pilot`. All 12 local receipts validate, all four arms share the same system hash within each of the three seeds, every arm spends exactly 2,000 environment samples, and the CURC/local checksum dry run found no differences. Slurm accounted 2.236 allocated CPU-core-hours across the exact jobs. The manifest, trajectories, query ledgers, system specifications, receipts, and logs are preserved here.

Final feasible-intervention non-root loss, mean across three **development** systems:

| Non-leaf random | Non-leaf coverage | PEV | PEV naive variance |
| ---: | ---: | ---: | ---: |
| .58603 | .58294 | .57769 | .57655 |

PEV minus coverage was −.005715, −.008441, and −.001611 on seeds 42, 123, and 456. PEV minus naive variance was −.002580, +.002745, and +.003236. Broad-domain non-root loss was 3.5069 random, 2.1667 coverage, 1.0165 PEV, and .9893 naive variance. Observational total loss was nearly unchanged across arms. Reproduce the paired final-checkpoint report with `python3 scripts/research/aggregate_persistent.py --root results/research_pev_shift30_pilot`.

The feasible metric compares predictions to **noisy** held-out child outputs. This 30-node family has 25 non-root nodes and child noise SD .15, giving an expected irreducible sum of 25×.15²=.5625 before any model error. The raw PEV-versus-coverage mean difference is only .00525, about 0.9% of the recorded metric. A ratio after subtracting the *expected* noise floor would be unstable with this small panel and is not a valid new primary outcome. The larger broad-domain difference and consistent negative paired differences are useful diagnostics, but this three-system pilot cannot establish a shift-family advantage, nor does it isolate integrated variance reduction from naive variance.

**Decision:** do not launch a 20-system repeat on this generator merely to improve significance. First add a fixed, noise-free conditional-mechanism evaluation on the same feasible parent contexts, while keeping the noisy metric for continuity. Validate that evaluator on small systems, then freeze a genuinely new confirmation protocol if the scientific question remains worthwhile. This pilot is a synthetic function shift on a familiar graph, not an independent external environment.
