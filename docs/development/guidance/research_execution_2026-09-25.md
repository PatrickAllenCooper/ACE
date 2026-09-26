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
