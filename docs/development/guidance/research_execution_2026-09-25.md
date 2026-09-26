# ACE research execution ledger — 25 September 2026

Source plan: [research portfolio](research_portfolio_2026-09-25.md). This ledger describes what is implemented and what still needs work. Newer validated receipts and subsequent ledger updates supersede the initial state below.

## Scope and constraints

- Numerical research on CURC is authorized. No Azure or other closed-model API calls in this phase.
- Preserve all non-ACE jobs and remote untracked files. Use `ucb736_asc1` and ACE-only job names beginning `acer_`.
- Prototype results are development evidence. A 5-node run with one fixed SCM provides learner-seed variation, not independent-system replication.
- Current source revision is recorded by the submit manifest and every cell checks it at startup. `REQUIRED_REVISION` should equal the tested commit.
- Results root: `/scratch/alpine/paco0228/ACE/results/research_portfolio_20260925`.

## Implemented first wave

`scripts/research/agenda_runner.py` implements three **numerical prototypes**:

1. `prior`: a fixed-data Bayesian linear feature bank, correct and deliberately wrong priors, and a broad fallback. This isolates the prior-reliability question. It has no semantic descriptions or LM outputs yet.
2. `design`: a small interaction mechanism with exact linear posterior updates, single and paired interventions, cost penalties, and background-variance sweeps. This is a mechanism-level reachability diagnostic, not a full-graph controller.
3. `transfer`: source-family prototypes, a shared passive assay, sparse function changes, and local query allocation. The mechanism library is prescribed numerically; learned hypernetworks and physically constrained whole-SCM interventions remain future work.

The PEV canary uses the existing 5-node, homogeneous 30-node, and heterogeneous 30-node runners. Each family compares Random ensemble, non-leaf coverage ensemble, PEV integrated variance reduction, and PEV naive variance. All canary policies use a 3-member ensemble, 20 training epochs, eight intervention steps, one reset campaign, and identical observation schedules. This is a **functional and cost screen**, not the planned confirmatory 5-member/100-epoch comparison. PEV-var may be better; both outcomes remain open.

The canary is `12` PEV cells. The numerical prototype is `3` prior, `12` design, and `9` transfer cells, for `36` independent CPU jobs in the default first wave. Requests: 24 numerical jobs at one CPU/30 minutes/2G plus 12 PEV jobs at four CPUs/two hours/16G, an upper request of 108 CPU-core-hours. Actual running time will be measured.

## Submission and validation

The local code must be committed and pushed, then CURC `/projects/paco0228/ACE` advanced by fast-forward to the same revision. Submit from that remote checkout with `REQUIRED_REVISION=<full SHA> bash jobs/curc_submit_research_agendas.sh`. The script skips an output only when `scripts/research/validate_cell.py` validates it; it also avoids resubmitting a matching queued/running job. It appends each submission to `submitted.tsv` immediately with job id, revision, output path, and timestamp.

The worker verifies the revision and writes a receipt only after producing files. Validate numerical cells by `metrics.csv` hash, complete receipt, row count, and finite error. Validate PEV cells by the eight-step trajectory, summary, query breakdown, and zero candidate-probe samples. `python scripts/research/status.py --root <root>` summarizes verified cells and descriptive pilot outcomes. Slurm completion alone is never a result.

The numerical runner and both shell scripts were smoke-tested locally before remote submission. `scripts/runners/run_5node_baseline_seed.py --method nonleaf_coverage_ens` produced a one-step artifact. The three numerical prototype tracks each produced finite local receipts. The exact interaction sanity case recovered no interaction information under deterministic zero background with single-parent actions, and recovered it with joint actions; this is a controlled toy result, not evidence at scale.

## What the hourly check must do

1. Check shared SSH using `curc-access`; request private reconnect if the master expires. Do local work meanwhile.
2. Compare ACE `acer_` Slurm queue/accounting with the exact submitted ids. Read failed logs and inspect completed outputs.
3. Run receipt validation and the status summarizer. For completed numerical cells, pull the full small directory locally. For PEV, pull per-step files, summaries, query breakdowns, receipts, and logs. Verify remote/local file hashes.
4. Update this ledger with measured runtimes, OOM/timeouts, skipped/in-flight cells, verified results and decisions. Commit/push compact results and analysis; avoid indiscriminate large copies.
5. After all first-wave cells validate, determine whether to extend the PEV control to multiple seeds and full training, and whether B/C merit richer models. Develop the persistent learner and common harness described in the portfolio before making paper claims.
6. Notify Patrick on meaningful completion, failure, blocked access, or decisions. Remain quiet on unchanged queue states. Never call Azure or alter other projects' jobs.

## Open implementation work

- Common persistent campaign harness with an inaccessible test evaluator and pre-query budget enforcement; the historical reset runners do not provide this yet.
- Replay-based learner parity between historical ACE and baselines.
- Real independent-system confirmation for A/B/C, with at least 20 SCMs per primary setting once pilot variance is known.
- Learned modular encoder/adapters; physically constrained multi-target experiments on whole graphs; numerical symbolic-search controls; semantic metadata dataset designed without giving away equations.
- External validation on an independently selected task after a synthetic gate passes.

The first wave's numerical prototypes are meant to find and repair conceptual or engineering flaws cheaply. Do not interpret their small toy effects as the outcome of the full research agendas.

## First-wave outcome (25 September)

All 36 jobs (32986987–32987022) completed successfully. The complete outputs, submission ledger, and receipts were copied to `results/research_portfolio_20260925/`; 36/36 cells pass local artifact validation. PEV canaries each executed eight interventions and two observational refreshes, exactly 480 environment samples. The 12 PEV cells each used one fixed graph and seed 42, so they cannot establish a population effect.

The reachability prototype is informative but intentionally simple. With zero background variation in the uncontrolled parent, single-parent actions have end-budget interaction MSE about 1.44; paired actions are below 0.00025 across the tested costs. At background standard deviation 0.15, single actions also identify the interaction well. This means the next whole-SCM study must vary natural support, graph topology, actuator cost, and the allowed action set. A paired-action advantage is conditional on weak natural support, not a general result.

The prior prototype does not yet solve misspecification: at 64 samples the broad/correct/wrong/fallback end-budget MSEs are approximately 0.00177/0.00162/0.00200/0.00200 (three seeds). The fallback mixture offers negligible recovery here. This track needs a genuinely fallible structural prior and a stronger evidence gate before larger runs.

The local-repair prototype has small differences between module and ordinary warm-start residual sampling at one or three changed nodes. With ten changes, module residual is worse (0.02125 versus 0.01804). The current prescribed library is not evidence that learned modular transfer works. A learned source library and fresh independent target SCMs are needed.

The one-seed PEV canary is mixed. On the heterogeneous 30-node graph, ACE-style end loss is 5.92 for PEV versus 10.07 for eligible-node random and 10.62 for eligible-node coverage; on homogeneous 30-node it is 1.82 versus 3.78 and 4.07; on legacy five-node it is 1.63 versus 2.18 and 2.70. The broader total-loss metric is much closer among arms and sometimes favors random. These are eight-step transient values under the old reset runner, which is why the persistent learner and metric parity work remain mandatory before any scientific conclusion.

The immediate next gate is a single persistent learner per graph with strict pre-query budget accounting and sealed evaluation. Then test several independent graph seeds and longer trajectories, preserving all per-step outcomes in Git. Do not expand the toy A/C pilots into large job grids until their model assumptions are improved.

## Persistent campaign pilot

`scripts/research/persistent_scm.py` now implements that next gate separately from the historical reset runners. For each arm it retains one SCM and one ensemble learner throughout the campaign. The intervention policy receives only the student; the held-out observational data and broad mechanism contexts are created once by `SealedEvaluator` and are never passed to policy selection. The evaluator is fixed for paired arms of a given family and seed. Before every iteration, the runner checks that the executed batch **and any due observational refresh** fit within the remaining sample budget. `trajectory.csv` records both metrics and cumulative environment samples. A hash receipt and independent validator guard the output.

The planned CURC pilot is three seeds (42, 123, 456) by three graph families (legacy five-node, homogeneous 30-node, heterogeneous 30-node) by four graph-matched ensemble policies (non-leaf random, non-leaf coverage, PEV integrated variance reduction, and PEV naive variance). The default is 2,000 samples, K=3, 20 training epochs, 50 samples per intervention and 40 per observational refresh every three steps. This is 36 CPU jobs, up to 288 requested core-hours at their two-hour limits; actual consumption should be far less. The five-node seeds share one fixed SCM and must be interpreted as learner randomness only. Thirty-node seeds construct distinct graphs/mechanisms. Use `REQUIRED_REVISION=<tested SHA> bash jobs/curc_submit_persistent_scm.sh` on the fast-forwarded CURC checkout. The submitter skips only validated outputs or matching live jobs.

After validation, compare *final* and trajectory-based outcomes on each seed under the exact sample budget; do not select the best checkpoint. Expand only if the observed pilot effect is stable, budget parity holds, and the heterogeneous graph does not fail badly. The broad metric is a fixed-domain mechanism error, while the observational metric is computed from a fixed passive holdout; report both.

## Persistent pilot result (26 September)

All 36 persistent jobs (32987241–32987276) completed with exit code 0 on account `ucb736_asc1`, source revision `684f1181c181c8dedc13a3d46be0c89c6a75fd9d`, under `/scratch/alpine/paco0228/ACE/results/research_persistent_20260925`. Their trajectories, query ledgers, receipts, logs, and submit manifest were copied to `results/research_persistent_20260925/`; a checksum dry run found no differences. Independent local validation passes 36/36. Every arm consumed exactly 2,000 environment samples, with no candidate-probe samples. Actual per-job runtime was 23–113 seconds, far below the two-hour request.

On homogeneous 30-node development systems (seeds 42, 123, 456), mean final broad error is 4.4005 for eligible-node random, 2.3708 for eligible-node coverage, 1.6292 for PEV integrated variance reduction, and 1.6565 for PEV naive variance. On heterogeneous 30-node systems, corresponding means are 5.7493, 4.3597, 2.2150, and 2.5494. PEV integrated variance reduction is below both graph-matched controls and the naive variance ablation in all six individual 30-node development systems. This is promising *pilot* evidence only; these settings and seeds informed our choices. The broad evaluation domain overlaps PEV's design objective, so independent feasible-intervention evaluation is needed.

The legacy five-node SCM remains one fixed system across three learner seeds. Mean final broad error is 0.6594 random, 0.6307 coverage, 0.6028 PEV, 0.5968 naive variance; differences are small and PEV does not beat coverage on every seed. Held-out observational error barely differs across policies in any family. Do not claim general five-node or observational improvement from this pilot.

Next: freeze a fresh 20-system confirmation per 30-node family, add fixed feasible-intervention and non-root metrics, serialize system provenance, and preserve the four-arm matched 2,000-sample protocol. Avoid tuning on confirmation outcomes. The local command `python3 scripts/research/aggregate_persistent.py --root results/research_persistent_20260925` reproduces the pilot summary from receipts.

## Confirmation protocol (frozen 26 September)

The settings, 20 fresh seeds per 30-node family, primary feasible-intervention non-root metric, and paired analysis are frozen in [`protocol_persistent_confirmation_v1.json`](protocol_persistent_confirmation_v1.json). The runner now writes a fixed held-out feasible-intervention panel, broad-domain non-root loss, observational non-root loss, and a hash of the serialized SCM graph, forms, coefficients, and source revision. The original 36-cell pilot remains on schema v1; confirmation uses schema v2. PEV still receives only the student and never the held-out evaluator.

The submission set is two families × 20 fresh graph seeds × four policies = 160 CPU jobs. At the measured pilot mean of roughly 1–2 minutes per 30-node job, this is a modest compute request despite the two-hour wall limit. Submit only after code is pushed and the CURC checkout fast-forwarded: `ROOT=/scratch/alpine/paco0228/ACE/results/research_persistent_confirmation_v1 FAMILIES='hom30 hetero30' SEEDS='1000 1001 ... 1019' REQUIRED_REVISION=$(git rev-parse HEAD) bash jobs/curc_submit_persistent_scm.sh`. Keep the exact complete seed list in the frozen JSON; abbreviated ellipsis is explanatory, not a runnable command. The hourly check should verify receipts and system hashes across all four arms of each seed before aggregating outcomes.

Submitted on 26 September: 160 jobs recorded in `results/research_persistent_confirmation_v1/submitted.tsv`, exact frozen grid validated locally. Source revision `01a6eccf5e4b03e42ecdd7ef61b6cf327c1a9cc9`; account `ucb736_asc1`; CURC output `/scratch/alpine/paco0228/ACE/results/research_persistent_confirmation_v1`. Job IDs range from 33002475 to 33002637 with gaps assigned by Slurm to other submissions. At the last check, 111 ACE confirmation jobs were running and 49 pending; zero receipts existed. The upper requested allocation is 160 × 4 CPUs × 2 hours = 1,280 core-hours, within the portfolio's staged CPU cap; pilot runtimes suggest actual use will be much lower. Do not update the remote source checkout while these revision-pinned jobs are pending.
