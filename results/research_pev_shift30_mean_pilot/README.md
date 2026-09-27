# Shifted-family noise-free metric replay

Twelve CURC jobs (33012860–33012871) completed with exit 0:0 on account `ucb736_asc1` from sparse scratch checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_mean_9e9ba24`, source revision `9e9ba247fe16ab0acf92be3cfb7dd18a6f122145`. Output root: `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_mean_pilot`. All schema-v3 receipts validate; every arm uses exactly 2,000 environment samples and system hashes match within each seed. A CURC/local checksum dry run showed no differences. All **12 original trajectories match exactly on every pre-existing per-step field** against `results/research_pev_shift30_pilot/`; this replay added only the sealed conditional-mean metric.

Final held-out, noise-free feasible-intervention non-root mechanism error, mean over three development systems:

| Non-leaf random | Non-leaf coverage | PEV | PEV naive variance |
| ---: | ---: | ---: | ---: |
| .020677 | .018849 | .011167 | .010556 |

PEV minus coverage was −.004168, −.014191, and −.004685 by seed 42/123/456. Mean PEV error is about 41% below coverage on this **development** metric. PEV minus naive variance was −.000076, +.001236, +.000674: no evidence here that the integrated-variance formula adds value over simpler variance scoring. The original noisy feasible losses remain .58294 coverage, .57769 PEV, and .57655 naive variance, and are retained in every trajectory. This is not a fresh confirmation because the metric was introduced after inspecting the first pilot, then replayed on the same systems.

The new evaluator computes the true conditional mechanism mean on the same fixed feasible parent contexts and never exposes it to acquisition. A one-step four-arm parity check passed, and the old trajectories' exact equality supports that policy/training behavior was unchanged. The next test uses fresh prespecified systems under `protocol_pev_shift30_mean_confirmation.json`. **Later provenance correction:** the five-layer graph *generator* remains fixed, but its parent edges vary by seed. See [the audit](../../docs/development/guidance/erratum_shift_graph_provenance_2026-09-27.md). The experiment still does not test an external graph family or real-domain transfer.
