# Corrected connected-motif development scale check

Fifteen CURC jobs (33008830–33008839, 33008841–33008844, 33008846) ran on account `ucb736_asc1` from sparse scratch checkout `/scratch/alpine/paco0228/ACE/code_connected_rng_v1_0fd626d`, pinned source revision `0fd626d9ae503c666b2d68f03ac2c45c8a5d9c42`. The output root on CURC was `/scratch/alpine/paco0228/ACE/results/research_connected_acquisition_dev_rng_v1`. All jobs completed; 15/15 schema-v2 receipts, system definitions, action ledgers, exact cost equations, and source revisions validate locally. A checksum dry run against CURC reported no differences. The `submitted.tsv` manifest maps each job to a setting.

This corrects the v0 random-stream coupling. At fixed k=3, all five arm outcomes, costs, masked labels, and risks match **exactly** for N=15, 30, and 100 for each of the three seeds. Additional nodes are downstream padding, so this equality is an implementation invariant and negative control, not proof of scalability.

Mean final feasible-motif MSE at root SD .15, actuator penalty λ=4, cost budget 400 (three development systems):

| N | k | Random single | Coverage single | Random pair | Coverage pair | Risk pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15/30/100 | 3 | .06882 | .06975 | .84866 | .02210 | .04720 |
| 30 | 1 | .02592 | .02230 | 1.05652 | .00096 | .00154 |
| 30 | 10 | .14469 | .05240 | .07515 | .13774 | .04211 |

At k=3, coverage pairs outperform the posterior-risk pair selector on average; at k=10, risk pairs outperform coverage pairs but are close to coverage singles. Per-system risk-pair versus coverage-pair values are k=3: .09934/.01863, .04013/.03898, .00213/.00869; k=10: .05118/.22669, .02618/.12879, .04895/.05774. These small and mixed development results fail the portfolio's within-menu promotion criterion. More seeds of the same favorable generator are not justified just to obtain a positive test. This screen uses an exact linear learner, known DAG, a single connected chain of motifs, and downstream padding; it does not establish a foundation-model or neural-architecture contribution.

The archived v0 output remains under `results/research_connected_acquisition_dev_v0/` with its invalid fixed-k N comparison clearly marked.
