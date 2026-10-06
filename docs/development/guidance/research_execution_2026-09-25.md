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

Submitted on 26 September: 160 jobs recorded in `results/research_persistent_confirmation_v1/submitted.tsv`, exact frozen grid validated locally. Source revision `01a6eccf5e4b03e42ecdd7ef61b6cf327c1a9cc9`; account `ucb736_asc1`; CURC output `/scratch/alpine/paco0228/ACE/results/research_persistent_confirmation_v1`. Job IDs range from 33002475 to 33002637 with gaps assigned by Slurm to other submissions. The submitter requested four CPUs and two hours per job (1,280 requested task-core-hours); Slurm accounted **five allocated CPUs** per job, so a full two-hour allocation would have been 1,600 core-hours. Both are within the portfolio's staged CPU cap. The earlier queue snapshot of 111 running and 49 pending, with zero receipts, is superseded by the completion section below.

## Confirmation outcome (26 September, later check)

The earlier queue snapshot above is superseded. All 160 jobs completed with exit code 0; 160/160 receipts validate locally, with exact 2,000-sample budgets, 32 steps each, 40 paired systems, and matching CURC/local file checksums. Slurm reports 4.586 summed job-hours and **22.932 allocated CPU-core-hours** across these exact jobs. Full trajectories, system definitions, query ledgers, receipts, and logs are in `results/research_persistent_confirmation_v1/` with a concise [result summary](../../../results/research_persistent_confirmation_v1/README.md). PEV improves the frozen feasible-intervention non-root metric over graph-matched coverage in both families after Holm correction (homogeneous paired mean −0.01360, adjusted p=0.00384; heterogeneous −0.10919, adjusted p=0.00961). The naive variance ablation is effectively tied with PEV on that metric; do not attribute the gain specifically to integrated variance reduction. Observational error is also nearly unchanged. No confirmation cells require repair or resubmission.

The next scientific gate is an independently chosen external environment or a family shift selected before method scoring, plus a stronger acquisition ablation that isolates uncertainty from graph-aware coverage and action-value effects. Avoid additional seed sweeps on these same generators merely to make p-values smaller. The numerical prior and modular-transfer tracks remain open; their current toy pilots are not positive evidence and should be redesigned before committing large compute.

The [shifted-mechanism PEV pilot](protocol_pev_shift30_pilot.md) now freezes a synthetic function-family shift before scoring it. It keeps the prior graph and matched persistent campaign contract, but changes nonroot functions to saturation, bounded interactions, and ripples. One-step local smoke validates all four arms and matching system hashes. This is a 12-job pilot only; a larger fresh-system run is conditional on its results. It does not substitute for an independent external environment.

Shift-family pilot submitted on 26 September from pinned sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_7b4b3b0`, revision `7b4b3b0ba6f3fab9e06cd3317f6599683882d2c9`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_pilot`. Jobs 33012701–33012712 cover three seeds and four methods; each requests four CPUs, 16G, and two hours. Initial accounting showed all 12 RUNNING and no completed receipts. The output-root manifest maps jobs to cells. No non-ACE jobs were changed.

Later check: all 12 jobs COMPLETED with exit 0:0; 12/12 receipts validate, all arms use exactly 2,000 samples, and system hashes match within each seed. CURC/local checksums agree. Slurm accounted 2.236 allocated CPU-core-hours. The full [pilot result](../../../results/research_pev_shift30_pilot/README.md) has feasible non-root means .58294 coverage, .57769 PEV, .57655 naive variance. PEV beats coverage on each of the three development systems, but the raw difference is small relative to the .5625 expected observation-noise floor and naive variance remains tied. Do not run 20 more systems under this noisy primary metric; implement a sealed noise-free conditional-mechanism panel before a new confirmation decision.

The sealed noise-free conditional-mechanism metric is now implemented for `shift30` only. The SCM exposes a deterministic `mechanism_mean`, and the evaluator records its prediction error on the same fixed feasible parent contexts while retaining the noisy metric. The evaluator remains inaccessible to policies; a schema-v3 one-step smoke passed all four methods and system-hash parity. The [three-system replay protocol](protocol_pev_shift30_mean_replay.md) freezes an exact rerun of the pilot to diagnose the metric without new seeds or policy tuning. The original schema-v2 pilot is retained.

The 12 replay jobs were submitted on 26 September from sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_mean_9e9ba24`, pinned revision `9e9ba247fe16ab0acf92be3cfb7dd18a6f122145`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_mean_pilot`. Job IDs 33012860–33012871 cover the same three seeds and four arms. Each requests four CPUs, 16G, and two hours. These are correctness/measurement replays, not independent confirmation systems; validate receipts, exact budgets, hashes, and the new metric before comparison.

Later check: all 12 replay jobs COMPLETED with exit 0:0; 12/12 schema-v3 receipts validate, 2,000 samples per arm, matching system hashes, and matching CURC/local checksums. All 12 trajectories are exactly identical to the original pilot for every pre-existing per-step field. The [replay summary](../../../results/research_pev_shift30_mean_pilot/README.md) reports noise-free feasible mechanism means .018849 coverage, .011167 PEV, .010556 naive variance, with PEV below coverage on all three development systems. This supports a fresh test of the uncertainty-policy effect, not the distinctive IVR scoring formula. The [fresh confirmation protocol](protocol_pev_shift30_mean_confirmation.json) now freezes seeds 5000–5019, one primary paired contrast, and 80 cells before submission. Topology remains fixed across coefficient draws; this is a synthetic function shift, not an external environment.

Fresh shift-family confirmation submitted on 26 September from pinned sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_shift30_confirm_0a5c43d`, source revision `0a5c43d3c57b956715d2deea1278967f0d7288f0`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_shift30_mean_confirmation`. The 80 exact job IDs are in `results/research_pev_shift30_mean_confirmation/submitted.tsv` (first 33013232, last 33013315, with Slurm gaps); a local check confirms the manifest has every frozen seed/method pair exactly once and one source revision. Each job requests four CPUs, 16G, and 30 minutes; the upper requested allocation is 160 job-hours/640 requested CPU-core-hours, while prior pilot jobs actually used about two minutes each. Initial snapshot: 80 RUNNING, zero receipts. Do not score until all 80 validate and their local copies match CURC hashes.

Completion check: all 80 jobs finished `COMPLETED|0:0`; 80/80 schema-v3 cells validate locally after checksum-matching transfer, with 2,000 executed samples and matched system hashes across arms of every seed. Slurm accounted 2.030 elapsed job-hours and 10.149 allocated CPU-core-hours. The [full fresh result](../../../results/research_pev_shift30_mean_confirmation/README.md) gives final noise-free feasible non-root means 0.039791 coverage, 0.043898 random, 0.015430 PEV, and 0.015871 naive variance. The frozen primary PEV-minus-coverage paired mean is −0.024360, 95% t CI [−0.048847, +0.000126], p=0.05108; **the prespecified interval-excludes-zero gate fails**. PEV is lower on 19/20 systems, so the directional pattern is strong, but one large contrast affects the mean inference. PEV and naive variance remain tied (paired p=0.622); no IVR-specific claim follows. Do not extend this seed grid solely to cross a threshold. The next PEV design should target an independent topology/environment and explain the variance-policy equivalence before any larger confirmation.

Subsequent action-log diagnostic: PEV and naive variance selected the same target at a matched step in only 95/640 cases (18 exact target/value actions), although both visited all 25 eligible targets with similar marginal frequencies. The equal final error is therefore an outcome equivalence under distinct action sequences, not literal action identity. No ACE jobs remain queued; the shared CURC SSH master is healthy. This diagnostic does not alter the frozen primary inference.

**Graph-provenance correction (27 September):** the frozen shift30 protocol's “fixed topology” description was wrong. A [receipt-backed audit](erratum_shift_graph_provenance_2026-09-27.md) found 20 distinct DAGs, 35–41 edges, and 17–22 eligible targets per seed across the 20 fresh systems, with graph and coefficients matched across arms. The graph *family* and five-layer sizing are fixed; parent edges vary because the runner seeds NumPy before `LargeScaleSCM` draws its graph. The 25 target names above are pooled across systems, not eligible in each. The paired effects and failed primary inference remain unchanged. The next PEV shift should target a different graph generator or external domain, not merely new edges from this one.

The next small PEV control is [endpoint-valued graph-matched coverage and random acquisition](protocol_pev_extreme_value_control_dev.md) on the three old development systems. The completed uncertainty policies strongly favored endpoint intervention values, a factor the original coverage/random controls did not isolate. Both new arms have passed one-step local receipt and budget checks; six CPU cells are authorized by the staged research plan once this source revision is pinned on CURC. This is a development diagnostic, not a new confirmation or a test of the IVR formula.

Six endpoint-value control cells were submitted on 27 September from pinned sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_extreme_3a47374`, source revision `3a47374ca869a1be36c3fd26d99c3b0d9ca19be0`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_extreme_value_dev`. Job IDs 33031117–33031122 map to the exact two-method×three-seed grid in `results/research_pev_extreme_value_dev/submitted.tsv`. Each requests four CPUs, 16G, and 30 minutes. Initial snapshot: six RUNNING, zero receipts. Do not score until all six validate, the graph/mechanisms match the archived replay within serialization tolerance, and full artifacts are copied locally with checksum parity.

Before scoring, source comparison against the archived shift30 mean replay revision `9e9ba247fe16ab0acf92be3cfb7dd18a6f122145` found no changes to the SCM, learner, evaluator, or existing policy logic: only the two new policy classes and their runner dispatch were added. Local seed-42 smoke reproduced the archived graph and function forms exactly; coefficient JSON differs across local/CURC NumPy runtimes by at most 1.1e−16. `scripts/research/aggregate_pev_extreme_control.py` will validate six new and twelve archived cells, exact budgets, endpoint values, and graph/mechanism parity before reporting final outcomes.

**Endpoint-control outcome (27 September):** all six jobs COMPLETED with exit 0:0, and all six new plus twelve archived cells passed the comparison receipt's exact 2,000-sample, 32-step, system-parity, and endpoint-value checks. CURC/local rsync checksum dry run found no differing files. Slurm allocated five CPUs and summed 982 job seconds (1.364 CPU-core-hours); no `acer_` jobs remain in the queue. The [full development result](../../../results/research_pev_extreme_value_dev/README.md) shows endpoint coverage mean final noise-free feasible nonroot MSE 0.008802, below PEV 0.011167 and naive variance 0.010556 on each of the three reused systems. Endpoint random mean 0.034605 is worse, so action magnitude alone is not enough. These post hoc development systems cannot support a fresh superiority claim. Freeze endpoint coverage as a baseline in the next independent confirmation; treat previous PEV-versus-ordinary-coverage advantage as potentially explained by value design and target coverage, not by the integrated-variance formula.

The [endpoint-coverage replication protocol](protocol_pev_endpoint_replication_2026-09-27.json) freezes 20 new seeds (6000–6019), three arms, exact 2,000-sample budgets, and final noise-free feasible nonroot MSE before submission. Primary comparison is endpoint coverage minus PEV; naive-variance PEV is secondary. This is replication within the same shifted-function and hierarchical-graph generators, so even success will not establish external-domain or graph-generator transfer. The three-seed development result informed this protocol; do not pool it with the fresh analysis. New `aggregate_pev_endpoint_replication.py` validates all 60 cells and writes paired statistics only after complete local custody.

The source-pinned canary for the fresh replication was submitted on 27 September from sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_endpoint_49e1bb1`, revision `49e1bb1ab1a7ab91eb30e2063feb3704382a77d0`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_endpoint_replication_v1`. Seed 6000 jobs: endpoint coverage 33035212, PEV 33035213, PEV-var 33035214. Each requests four CPUs, 16G, and 30 minutes; Slurm may allocate five CPUs as in earlier runs. Initial status: all three PENDING, zero completion receipts. Validate exact budget, receipts, system parity, and runtime before submitting the other 19 systems. No Azure or other closed-source model API was used, and no other project's job was modified.

Later canary validation: all three seed-6000 jobs COMPLETED with exit 0:0 in 70, 71, and 83 seconds, respectively, at five allocated CPUs. Local receipts and hashed trajectories pass `validate_cell.py`; each has exactly 32 actions, 2,000 environment samples, and zero candidate-probe samples. All three system hashes agree (`4354e60eef4292b1b12a37553895c1f17625e59fa4fffdb000fdd767d225fbb7`), endpoint coverage uses only absolute value 5, and CURC/local checksum dry run is empty. The one-seed outcome was used for validation only, not as a promotion criterion.

After that validation, 57 remaining cells for seeds 6001–6019 were submitted from the same pinned revision and account to the same output root. Job IDs 33036077–33036133 map to methods and seeds in `results/research_pev_endpoint_replication_v1/submitted.tsv` (all 60 cells, including the canary). Each requests four CPUs, 16G, and 30 minutes. Initial post-submission snapshot: 57 RUNNING, three valid receipts. The manifest and canary artifacts were copied locally; finish checksum and receipt validation before aggregate scoring. No other project's jobs were modified.

**Fresh endpoint replication outcome (27 September):** all 60 jobs COMPLETED with exit 0:0, all schema-v3 cells and 20 paired system definitions validate, and CURC/local checksum dry run shows no differing files. Slurm accounted 5,416 job seconds × five allocated CPUs = 7.522 CPU-core-hours. The [full result](../../../results/research_pev_endpoint_replication_v1/README.md) gives endpoint coverage final noise-free feasible nonroot MSE 0.019032, PEV 0.015347, and naive variance 0.017491. The prespecified endpoint-coverage-minus-PEV paired mean is **+0.003685**, 95% CI [+0.000452, +0.006917], two-sided p=0.0276; the endpoint-coverage-superiority gate fails and the observed direction favors PEV on these 20 fresh systems. Endpoint coverage versus naive variance is unresolved (p=0.197). The favorable three-system endpoint diagnostic did not generalize. This is within the same graph/function generator and does not isolate IVR's contribution or prove external transfer. No further same-generator seed sweep is justified merely to strengthen p-values; the next PEV test should shift graph generator or use an independently chosen external environment.

The next staged graph-shift pilot is frozen in [random-DAG protocol](protocol_pev_random_dag_dev_2026-09-27.json): three development seeds, three arms, and the same exact learner/query/evaluation contract. `RandomOrderShiftedSCM` changes only the known DAG generator to select one or two parents uniformly among all earlier nodes, allowing long-range edges while retaining five roots and the shifted mechanism family. A local one-step smoke for all three policies passed exact 50-sample accounting and system/coefficients parity. For seed 42, the new DAG has 34 edges and 15 edges spanning more than ten node positions, unlike the archived layer-constrained graph. This is nine CPU cells only, with no fresh inference until a separate protocol is frozen after validation.

The random-DAG development pilot was submitted on 27 September from pinned sparse checkout `/scratch/alpine/paco0228/ACE/code_pev_random_dag_4819601`, source revision `48196012d6d93f4cac4fb77b4ef963de3e2d200e`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_pev_random_dag_dev`. Nine jobs 33036573–33036581 map to the exact seed/method grid in the locally copied `results/research_pev_random_dag_dev/submitted.tsv`. Each requests four CPUs, 16G, and 30 minutes. Initial snapshot: all nine RUNNING, zero receipts; do not interpret until complete validation and checksum parity. No closed-source model API or non-ACE job was touched.

**Random-DAG pilot outcome:** all nine jobs COMPLETED 0:0, nine receipts and exact budgets validate, three system hashes pair across arms, and CURC/local checksum parity holds. Slurm accounted 693 job seconds × five CPUs = 0.963 CPU-core-hours. The [full result](../../../results/research_pev_random_dag_dev/README.md) has mean final noise-free feasible nonroot MSE 0.014648 endpoint coverage, 0.013687 PEV, and 0.015965 naive variance. PEV beats coverage on two of three systems and loses on one; the small mean edge does not meet the prespecified consistent-lowering development gate. No fresh 20-system graph-shift grid is justified by this pilot. Retain this negative/ambiguous boundary and prioritize a distinct external environment or a revised mechanism-level question over another synthetic seed sweep.

The [direct PEV-versus-variance diagnostic](../../../results/local_pev_scoring_diagnostic_20260928/README.md) reuses the 20-system endpoint replication with no new query or job. Forty PEV/PEV-var cell receipts, paired system identity, exact 2,000-sample budgets, source/output hashes, and action logs validate. Final feasible mean nonroot MSE averages .015347 for PEV and .017491 for PEV-var; the paired PEV-var-minus-PEV mean is +.002144, 95% t interval [−.000259, +.004547], exploratory two-sided p=.0773, with PEV lower on 12/20 systems. Same-step targets agree only 5.3/32 times on average, so the policies differ behaviorally, but this post hoc contrast does not isolate a reliable benefit from the integrated score. No equivalence claim follows either. Do not chase another same-generator seed sweep for this scoring-rule distinction; require an external environment or separately frozen mechanism-level question.

## Numerical prior reliability screen (26 September)

The first-wave marginal-evidence mixture assigned only about 0.3–2.2% fallback weight at 64 samples even under wrong proposals. `prior_gate_experiment` now asks a narrower, falsifiable question: can eight acquired validation samples identify a misleading typed proposal without requiring extra environment samples? It compares a broad fit, a strong proposed fit, and a validation-weighted mixture on identical fixed data. Both correct and wrong metadata conditions use the same generated observations. The separate [protocol](protocol_prior_gate_v1.md) freezes 20 fresh numerical systems (seeds 2000–2019) after 12 local development systems (100–111). The local 12 development cells have verified receipts in `results/local_prior_gate_dev_20260926/`.

Development mean MSE at 16 acquired samples: correct proposal 0.01023, validation gate 0.01107, broad 0.01971; wrong proposal 0.02458, gate 0.01493, broad 0.01971. At 32 samples: correct 0.00395, gate 0.00431, broad 0.00532; wrong 0.00593, gate 0.00532, broad 0.00532. This is a numerical reliability screen with a known feature bank. It does **not** test whether a foundation model can produce informative priors or whether a full Bayesian mixture is calibrated. The fresh CURC run should be evaluated once, without tuning on its outcomes.

Fresh numerical prior check submitted on CURC: 20 jobs 33004856–33004875 under account `ucb736_asc1`, source revision `2f4b17c787f2dbea51a94da8cf9e44fd69bc14aa`, output `/scratch/alpine/paco0228/ACE/results/research_prior_gate_v1`. All 20 completed with exit code 0; 20/20 receipts validated after checksum-preserving local sync to `results/research_prior_gate_v1/`. At 16 samples, correct-condition mean MSE is 0.01024 proposal versus 0.01045 gate; wrong-condition MSE is 0.02564 proposal versus 0.02128 gate versus 0.01750 broad. At 32 samples, wrong-condition values are 0.00877 proposal, 0.00670 gate, and 0.00651 broad. The gate **partially** recovers from wrong proposals but does not beat the broad baseline reliably at low budget. Continue developing reliability tests; do not promote this as a successful semantic prior method.

A later [post hoc hard-threshold audit](../../../results/local_prior_gate_threshold_audit_20260926/README.md) used only the 12 archived development systems. Four held-out-SSE thresholds trade correct-proposal benefit against wrong-proposal protection; none dominates the original soft gate across budgets and conditions. The most conservative threshold slightly improves the wrong condition at 16 examples but worsens the correct condition. All 12 source receipts, 24 summary cells, and output hashes validate. This is a diagnostic, not an opportunity to tune on the 20 fresh v1 systems. A new A track should test independently authored, incomplete semantic metadata and misspecified forms, rather than spend CURC time on more thresholds of this favorable numerical grammar.

The [external benchmark feasibility audit](external_benchmark_feasibility_2026-09-27.md) identifies pinned BoxingGym Lotka–Volterra as the first A-track integration target. It has independently authored descriptive/abstract task messages and a two-population query API. This changes the task to dynamical forecasting and the familiar predator–prey story can cue memorized equations, both explicit limitations. The upstream module currently fails a local import because ArviZ is missing; other modules need PyMC, absent from CURC `ace`. Stage an isolated scratch runtime and an oracle/evaluator smoke before scheduling a cached open-Qwen proposal pilot. No new result, CURC job, or closed-source model call follows from this audit.

The [pinned Lotka oracle-custody smoke](protocol_boxing_lotka_smoke_2026-09-27.json) now has an ACE adapter that substitutes only the unused Box's Loop reporting import with a raising shim, leaving the upstream simulator untouched. Local seed 42 runs eight counted training queries and sixteen held-out queries; two repetitions produce identical receipt and source SHA-256. This resolves the ArviZ import obstacle for the numerical-only smoke without installing PyMC. Freeze and validate three small CURC cells before considering any open-model GPU job.

The three [BoxingGym custody cells](../../../results/research_boxing_lotka_smoke/README.md) were submitted and completed on 27 September: jobs 33037384–33037386, account `ucb736_asc1`, ACE source revision `35cd9e23be79197264762092694391304060ad9f`, upstream commit `b43e38cb03d09c13efa9cf4d9bae740d51157bfd`, output `/scratch/alpine/paco0228/ACE/results/research_boxing_lotka_smoke`. All three exit 0:0, use exactly eight public acquired and sixteen private held-out queries, and validate source hash, file hashes, finite responses, distinct worlds, and CURC/local checksum parity. Slurm accounts 73 seconds at one CPU per job (0.0203 CPU-core-hours). These are environment-adapter receipts only. No numerical baseline, open-model proposal, or closed-source API call has occurred.

## Matched-menu joint-intervention development screen (26 September)

The [B2 development screen](design_b2_dev_2026-09-26.md) implements exact-cost single and pair menus, a coverage-pair control, and a posterior-risk pair policy. All 108 local cells validated with receipt hashes and cost equations. Joint actions solve the zero-background interaction obstruction, but within the pair menu a simple coverage rule is close to the risk policy. No additional CURC jobs were submitted on this toy; the next justified scale test must embed motifs in a larger SCM and vary motif count independently of graph size.

The subsequent [disjoint-motif development screen](motif_reachability_dev_2026-09-26.md) did this first as a numerical control: 48/48 receipt-validated cells on N=15/30/100 with k=1/3/10 hard motifs. Independent root padding leaves outcomes identical by construction, while increasing k at fixed budget makes joint actions much less decisive. At weak nonzero background variation and high paired-actuator cost, coverage singles can outperform pair policies. This does not satisfy the within-menu promotion criterion; connected sparse SCMs are the next required gate. No CURC jobs were spent on the disjoint prototype.

The [connected-motif development protocol](protocol_connected_motif_v0.md) now has a DAG, forward sampler, action menu, exact-posterior matched acquisition harness, action/cost ledger, natural-child mask for interventions on upstream children, and a sealed feasible-action evaluator. Its 24-cell local development grid validates. At k=3 and λ=4, risk pairs substantially improve over coverage pairs when natural root variation is zero, but at root SD .15 coverage singles are competitive with risk pairs. The next scale-development grid is 15 CPU cells on fresh seeds 200–202 and prespecified (N,k) settings; it remains too small for a claim. The submitter records the source revision, job id, account, and output path. No Azure calls.

Submitted the 15 connected-motif scale-development cells on 26 September under account `ucb736_asc1`, source revision `7d5de8eff9f5d4fef1080dce18ba6d32ba407ade`, output root `/scratch/alpine/paco0228/ACE/results/research_connected_acquisition_dev_v0`. Job IDs: 33008724–33008730 and 33008732–33008739 (Slurm assigned 33008731 elsewhere). All 15 were RUNNING at the initial check, not yet receipt-validated. Each requests one CPU, 2G, 30 minutes. The remote `submitted.tsv` is the authoritative job-to-setting manifest. Pull and validate the complete metrics, actions, system definitions, receipts, and manifest before scoring.

Later check: all 15 v0 jobs COMPLETED; 15/15 complete receipts, action/cost ledgers, and system files validate locally after a checksum-matching transfer. The v0 N comparison is invalid because padding draws consumed the main random stream and changed later motif data/actions. Preserve these outputs as an audit trail; do not score them as a scale result. The corrected schema-v2 runner uses a separate padding random stream and passes exact fixed-k N-invariance checks on seeds 100 and 200–202. Twenty-four corrected local development cells validate. The same 15 development settings will be rerun under `research_connected_acquisition_dev_rng_v1`; their outcome should supersede v0, with no extra seeds or changed primary metric.

CURC `/projects` is currently 100% full (250G/250G) across the shared filesystem. A fast-forward of ACE from `7d5de8e` failed while creating a Git pack temporary file; the original checkout remains at that revision. ACE `.git` is 1.2G and its tracked results tree 1.7G, but this is a shared project filesystem and no unrelated files were removed. The corrected submitter and worker now accept `ACE_CODE_ROOT`, allowing a sparse, pinned ACE checkout on `/scratch/alpine` for this rerun. Verify that checkout revision and each job's source path before submission. Keep outputs under scratch and sync back to the local Git repository. This is an access/storage workaround, not a change to the experiment.

Corrected sparse checkout: `/scratch/alpine/paco0228/ACE/code_connected_rng_v1_0fd626d`, source revision `0fd626d9ae503c666b2d68f03ac2c45c8a5d9c42`. The 15 correctness-rerun jobs were submitted on account `ucb736_asc1` to `/scratch/alpine/paco0228/ACE/results/research_connected_acquisition_dev_rng_v1`: 33008830–33008839, 33008841–33008844, and 33008846. Each requests one CPU, 2G, and 30 minutes. All 15 were RUNNING with zero receipts at the initial check; this is not yet an outcome. The output-root `submitted.tsv` records the exact job-to-setting mapping. No non-ACE files or jobs were modified.

Later check: all 15 corrected jobs COMPLETED with 15/15 locally validated schema-v2 receipts and exact cost/action ledgers. The local copy matches CURC checksums. Fixed-k N-invariance passes exactly for N=15/30/100 on all three seeds. The full [corrected result summary](../../../results/research_connected_acquisition_dev_rng_v1/README.md) shows that risk pairs do not consistently beat both pair coverage and single coverage. At k=3, pair coverage has mean feasible-motif MSE .02210 versus risk pair .04720; at k=10, risk pair .04211 versus pair coverage .13774 and single coverage .05240, with only three systems. No B3 promotion claim follows; connected-chain and action-cost sensitivity remain useful diagnostics.

Post hoc [coverage-order audit](../../../results/local_connected_coverage_rotation_dev_20260926/README.md): the k=10 coverage-pair control executed only five actions and its original fixed ordering visited motifs 0–4. A local check rotated every starting motif on the same three corrected systems, preserving exact offset-zero action parity and risk actions. At k=10, risk-pair MSE .04211 beats all ten rotated coverage-pair means (.13774–.41681), 30/30 system×rotation comparisons; at k=3, risk wins only 5/9 comparisons and coverage ordering matters. All 195 metrics, 1,365 actions, unique cells, finite values, and hashes validate. Rotations are not independent systems; no new CURC job or promotion follows. A future B gate needs order-independent matched-pair controls and new graph structures.

A further [permutation audit](../../../results/local_connected_coverage_permutation_dev_20260927/README.md) weakens the all-rotations impression. On the same three systems, all six k=3 motif orders and 64 deterministic sampled k=10 orders per system preserve exact identity-order parity and fixed risk actions. At k=10, risk beats coverage pairs in 152/192 system×order scenarios but some coverage permutations beat risk on each system; at k=3, risk wins 11/18. These scenarios are repeated schedules on only three SCMs, not independent samples. Do not promote B on this generator. The next B design needs a cost-matched adaptive coverage control and a distinct graph structure before fresh confirmation.

On 28 September, the [balanced pair development control](../../../results/local_connected_balanced_dev_20260928/README.md) added a cost-matched posterior-risk pair policy that must acquire from a least-visited motif. Its six reused cells exactly reproduce all archived actions and outcomes for the original arms; all 36 metric rows, 240 action rows, cost ledgers, balance constraints, and hashes validate. At k=3 its mean feasible MSE is .02508 versus .02210 for fixed coverage pairs; at k=10 it is .03799 versus .04211 for unconstrained risk, .13774 for fixed coverage pairs, and .05240 for coverage singles. This is a local post hoc diagnostic with no CURC job, not confirmation. The connected-chain generator still lacks a robust within-menu advantage across motif counts. A distinct graph structure is required before spending on new B systems.

The [fanout distinct-topology development screen](../../../results/local_connected_fanout_dev_20260928/README.md) then used a shared upstream child for all later motifs, under frozen protocol `protocol_connected_fanout_dev_2026-09-28.json` and source revision `4cc17875ab11e40c334cabe75ad41c5d2d1717e3`. Six local cells (seeds 300–302, k=3/10, N=30, root SD .15, penalty 4, budget 400) validate all action/cost/mask ledgers, topology smoke checks, per-cell SHA-256 receipts, and suite receipt. Risk pair mean feasible MSE was .01370/.01034 at k=3/10, versus coverage pair .21590/.15392 and balanced risk pair .02882/.04015. The risk policy chose motif 0 in three of five actions on every cell, exploiting the fanout's shared upstream position. Random pair beats risk on one k=3 seed, and only three systems per setting are available. This is a promising numerical *development* signal, not a foundation-model result or confirmation. No CURC jobs or closed-model calls were made; the next staged step is more independent fanout development systems with the same settings and all arms, then a separate frozen confirmation family only if the within-menu advantage persists.

The independent fanout development continuation is frozen in `protocol_connected_fanout_extension_2026-09-28.json`: seeds 303–305 at k=3/10 with unchanged methods, costs, and metric. The six cells were submitted as one short CPU job **33093296** under account `ucb736_asc1`, job name `acer_fanout_ext`, source revision `ec820bb94e9bb9753b1a14d223e437a6096a4031`, code checkout `/scratch/alpine/paco0228/ACE/code_connected_fanout_ec820bb`, and output root `/scratch/alpine/paco0228/ACE/results/research_connected_fanout_extension_dev_v1`. `submitted.tsv` records the full job mapping. It was RUNNING at the initial scheduler check. No outcome is claimed until all six per-cell receipts, action/query-cost ledgers, source hashes, and local custody validate. No closed-model API was used; no non-ACE jobs were touched.

The job later COMPLETED 0:0 in 21 seconds. All [six additional fanout cells](../../../results/research_connected_fanout_extension_dev_v1/README.md) and the suite receipt validate locally, including protocol/source hashes, exact actions and query costs, natural-child masks, topology, finite scores, and checksum-equivalent remote/local custody. Across the combined six independent systems per k, risk pairs beat coverage pairs in 6/6 at both k=3 and k=10; random pairs in 5/6 and 6/6 respectively; coverage singles in 5/6 and 6/6. Mean risk MSE is .00995/.01137, versus coverage-pair .12396/.12376. However, risk chooses shared motif 0 for exactly three of five actions in **every one of the 12 cells**. A simple graph-hub policy may explain the apparent advantage. Freeze and run that cost-matched control on the existing development systems before any fresh confirmation or foundation-model framing.

The [frozen hub-first control](../../../results/local_connected_hub_control_dev_20260928/README.md) adds deterministic and seed-fixed random downstream schedules after three shared-motif pair probes, with the same five pair batches and 360 cost units as risk. On all 12 previously scored fanout cells, it reproduces every archived action exactly and archived metrics within 1e-10; 96 metric rows, 600 action rows, protocol and output hashes, cost ledgers, and finite outcomes validate. Risk mean feasible MSE remains below both hub controls: at k=3, .00995 versus .02677/.02390; at k=10, .01137 versus .02203/.02401. Risk wins 5/6 and 5/6 paired cells at k=3, and 5/6 and 4/6 at k=10, against coverage/random hub schedules respectively. At least one static schedule wins on two systems at each k. These reused, post hoc development systems do not establish that posterior adaptation is essential. No CURC job or closed-model API was used. Next B gate is a graph family without one shared parent, retaining all existing pair controls and the hub controls where applicable.

The [bounded-degree binary-tree development gate](../../../results/local_connected_binary_tree_dev_20260928/README.md) then removed the fanout's dominant direct parent while preserving a connected DAG, exact learner, eight-arm menu, and cost budget. Frozen protocol `protocol_connected_binary_tree_dev_2026-09-28.json`, implementation revision `9c5679571e0bbcd2e8fcc5f5c268ff3f4d92692e`, seeds 400–402, N=30, k=10. All three local cells validate source/protocol hashes, topology and outdegree, exact action/cost/mask ledgers, finite scores, and per-file receipts. Coverage pair beats posterior-risk pair in all three systems: mean feasible MSE .03418 versus .04820. Coverage pair also beats coverage single in all three at the same total cost, though the single arm uses twice as many environment samples. This fails the frozen within-pair-menu development gate for the current posterior-risk selector. **Stop this B selector's expansion and do not submit a larger CURC grid or claim adaptive design.** The potential joint-action advantage is a separate, unconfirmed hypothesis. No CURC jobs or closed-model calls were used; no ACE jobs were queued at the remote check.

## Learned numerical module-library screen (26 September)

The subsequent [development routing-gate diagnostic](transfer_guard_dev_2026-09-26.md) tested passive-assay SSE margins before any fresh CURC confirmation. It did not remove coefficient-change negative transfer, so the guarded source selector is retained as a reproducible failed diagnostic, not promoted to a new experimental arm. No additional CURC jobs were submitted for this gate.

`scripts/research/learned_transfer.py` learns one fixed 2,560-sample source library from separate labeled mechanisms, then compares four models on identical target data for 30 local mechanisms. It is a fixed-data screen, not a neural architecture, active controller, full SCM intervention study, or language-model result. The [protocol](protocol_learned_transfer_v1.md) freezes 20 fresh target seeds (3000–3019) and all six change settings. Development artifacts for 12 seeds × six settings are under `results/local_learned_transfer_dev_20260926/`; all 72 receipts validate.

Development all-node MSE at 400 target samples for **family changes**: warm/source-mixture = 0.01263/0.01230 at k=1, 0.01506/0.01305 at k=3, and 0.02016/0.01352 at k=10. For **coefficient-only changes**, source mixture is slightly worse than warm at k=3 (0.01322 versus 0.01310) and k=10 (0.01529 versus 0.01427). Changed-node MSE at 200 samples shows a larger family-change advantage, but the shared source library uses the same feature/family bank as targets. This is a favorable retrieval test and a negative-transfer control, not proof that a pretrained foundation model will adapt efficiently. Submit the 20 frozen seeds on CURC only after the code and results are pushed and the checkout is advanced.

Fresh check: 20 CURC jobs 33004910–33004929 on account `ucb736_asc1`, source revision `64ee14d236f00521d08d0ce633384f88397ba570`, output `/scratch/alpine/paco0228/ACE/results/research_learned_transfer_v1`. All completed successfully; 120/120 setting receipts and 360/360 budget-level comparisons validate after checksum-preserving local sync. The full [result summary](../../../results/research_learned_transfer_v1/README.md) reports large changed-node benefits for family switches at 200 target samples but consistent harm for coefficient changes and measurable damage on untouched nodes. The source library must be gated by evidence of a family change before C can satisfy its own promotion criterion. Do not build the proposed neural hypernetwork yet; solve this reliability issue first.

The next [Bayesian transfer protocol](protocol_transfer_bayes_v2.md) replaces passive-only source weighting with full-data marginal-evidence weighting over old and source-centered Gaussian mechanism priors. All 72 development cells (12 seeds × six change settings) validate with schema-v3 receipts. At 200 target samples it retains most family-switch benefit while reducing coefficient-change harm relative to passive retrieval. The fresh 20-system grid is frozen at seeds 4000–4019 with untouched-node noninferiority as a required gate. The CURC `/projects` filesystem remains full, so run from a pinned sparse ACE checkout under `/scratch/alpine` rather than touching unrelated project storage.

Fresh Bayesian-transfer grid submitted on 26 September from sparse checkout `/scratch/alpine/paco0228/ACE/code_transfer_bayes_efa4dc9`, source revision `efa4dc91eca0e8c674fd43709f99529370dc5313`, account `ucb736_asc1`, output `/scratch/alpine/paco0228/ACE/results/research_transfer_bayes_v2`. Twenty jobs: 33010132–33010144 and 33010146–33010152 (33010145 assigned elsewhere). Each requests one CPU, 2G, and 30 minutes. The output-root `submitted.tsv` gives exact job-to-seed mapping. Initial check: four COMPLETED, sixteen RUNNING, thirty of 120 setting receipts written. This is not yet an aggregate result; validate all six settings per system, source hashes, local checksum parity, and exact methods before interpretation.

Later check: all 20 jobs COMPLETED with exit 0:0; all 120 schema-v3 cells and 1,800 method/budget rows validate locally. One learned source hash is shared across the grid, and CURC/local file checksums match. The full [result summary](../../../results/research_transfer_bayes_v2/README.md) shows large family-switch gains and much less coefficient-change harm than passive retrieval, but the untouched-node noninferiority requirement is not established at 200 target samples and clearly fails in mean at 120. The full-data mixture is a useful diagnostic, not a promoted safe-transfer architecture. No ACE jobs remain queued after this run; no closed-source model calls were made.

Next C development gate: [protect the unchanged old mechanism with an evidence-triggered local switch](transfer_safe_switch_v3_plan.md). This is a planned diagnostic on existing development seeds, not a fresh outcome or a reason to allocate GPU time yet. On the completed v2 grid, average old-selection mass remains only about 0.77–0.82 for k=1 settings despite 29/30 unchanged nodes; this motivates inspecting node-level switches and errors without treating the aggregate mass as causal proof. CURC access checked healthy and no ACE `acer_` jobs were queued at the 27 September 01:30 UTC heartbeat.

Initial [protected-switch development screen](../../../results/local_transfer_safe_switch_dev_20260926/README.md) ran locally on the archived 12 development seeds, six change settings, three budgets, and odds thresholds 4/10/25. It reproduced all archived warm and v2 Bayesian-mixture metrics before evaluating the new method; 19,440 finite node-level rows and hashes validate. All three odds thresholds meet the development error gates. At odds 10 and 200 examples, family changed-node MSE ratios versus warm are 0.246/0.147/0.264 at k=1/3/10; coefficient ratios are 1.000/1.000/1.000; untouched ratios are 1.000, with no false switches on these development systems. This promising separation may be specific to the favorable shared feature bank. The current diagnostic reuses acquired samples for selection and fit; a counted prequential validation and misspecified-family control remain necessary before freezing fresh systems or submitting CURC jobs. No closed-source model call was made.

The subsequent [counted prequential check](../../../results/local_transfer_prequential_dev_20260926/README.md) fails its development gate. All 19,440 node-level rows, hashes, warm parity, and sample-order constraints validate locally. With odds 4 at the 200-sample budget, family changed-node ratios versus warm are 0.972/0.422/0.648 for k=1/3/10; only 2/12 changed nodes switch at k=1. Odds 10/25 are more conservative and also fail k=1. Untouched and coefficient-change nodes are protected, but the rule requires more confirmation evidence than the sparse-change low-budget regime supplies. No fresh transfer grid or GPU prototype is justified by this rule. The in-sample success and prequential failure must both be retained.

The [counted adaptive-allocation development check](../../../results/local_transfer_adaptive_prequential_dev_20260927/README.md) spends the same 200 target examples but redirects 80 post-assay examples toward nodes nominated by early source-versus-old evidence. All 72 settings have exact 200-example accounting and 2,160 node-level decisions. The protected switch still fails the written development gate: family changed-node MSE ratios versus adaptive warm are 0.985/0.425/0.852 for k=1/3/10. However, adaptive **warm** alone gives changed-node ratios 0.220/0.155/0.634 against the archived uniform-allocation warm baseline on family changes; untouched-node error rises as high as 1.099× at k=10. The data suggest an acquisition question—how to reserve baseline coverage while spending more samples where mechanisms disagree—rather than evidence for a source-library switch. These are reused favorable development systems, so no fresh confirmation or GPU prototype follows yet.

## NeuronBench public-control canary (28 September)

The truth-blind weak control frozen in `protocol_neuronbench_public_controls_dev_2026-09-28.json` has a successful one-world (`z_rebound`) CURC canary after a serialization-only repair. The random and public-forecast-coverage arms each used exactly four distinct pool actions, common fixed ridge forecaster, separate private scorer, and six complete heldout labels. Scores are 51.8401 and 3.4100 floored spike MSE respectively. Both receipts, per-file hashes, independent score parity, and remote/local custody validate. [Full run note](../../../results/research_neuronbench_public_controls_dev_v1_retry1/README.md). These scores do not justify changing the frozen rules. The other five worlds' public inputs/plans must be frozen before their score runs. No closed-model calls were made.

The follow-on public-question export (job `33085609`) exposed acquisition/forecast menu overlap on three of the five remaining NeuronBench worlds. The first export attempt (`33085541`) stopped without querying or scoring. The [v2 frozen protocol](protocol_neuronbench_public_controls_dev_v2_2026-09-28.json) excludes actions that coincide with forecast targets. Five question sets and ten plans were exported, locally validated, and committed before outcome submission; [custody note](../../../results/research_neuronbench_public_questions_dev_v2/README.md). The old `z_rebound` canary had no overlap and its plan is unchanged by this eligibility rule. This revision is an explicitly labeled development correction, not a post-score optimization on the five worlds.

The ten v2 NeuronBench control jobs (`33085652`–`33085661`, account `ucb736_asc1`, revision `a70c815cc190ee50999db338af9d9b1ad8b9f225`) all completed and [validated locally](../../../results/research_neuronbench_public_controls_dev_v2/README.md) with exact cost 4, no acquisition/forecast overlap, six-label score parity, receipts, and remote/local checksum parity. Coverage beats random on h_sag, na_fatigue, and textbook_M, but loses on ca_rebound and sharply on d_type. The prior z_rebound canary favored coverage. This is a mixed six-world designed development diagnostic; it does not establish a general acquisition gain. Further NeuronBench work should separate acquisition from forecaster quality using a stronger frozen mechanistic predictor and a shared-data comparison, rather than expanding this weak ridge control.

A [zero-query, same-observation forecaster sensitivity diagnostic](../../../results/local_neuronbench_forecaster_diagnostic_dev_20260928/README.md) froze 24 public-only predictions before separate scoring. Mean and nearest-waveform controls show that the direction of the coverage-versus-random comparison changes across predictors on ca_rebound and d_type. Ridge's huge na_fatigue random error also shrinks under a constant forecaster. All 24 cells validate exact public hashes and score parity. This reinforces that action choice cannot be credited until a stronger, independently specified mechanism predictor is assessed on common data. These are already-scored development worlds; no additional CURC job was warranted for arithmetic on archived public files.

The [allocation-floor development check](../../../results/research_transfer_floor_dev_20260928/README.md) ran frozen floors 5 and 6 on CURC (jobs `33088724`–`33088725`, revision `eb444eebb708e8c7a91a2ef24c2b89b878b243b1`, account `ucb736_asc1`). Both 72-setting/2,160-node outputs validate with exact 200-example budgets and checksum custody. A floor of six reduces family k=10 untouched-node harm from 1.099× to 1.063× versus uniform warm, still above the prespecified 1.05 gate. Both floors fail; the protected source switch also remains below its gate. Do not submit fresh confirmation for this simple floor adjustment.

## Second external A-track domain: signal localization (28 September)

The pinned BoxingGym `location_finding.Signal` task has incomplete public source metadata and a two-dimensional response, unlike the familiar Lotka–Volterra dynamics pilot. A [three-world custody smoke](../../../results/research_boxing_signal_smoke/README.md) ran on CURC jobs `33090302`–`33090304`, account `ucb736_asc1`, ACE revision `b062e5e27955c08d6765dcc906f030a74cedb88a`. Each world has exactly 16 counted public queries and 32 private held-out responses; source locations, output hashes, and local/CURC custody validate. No model was tested or API called. The next A step is a frozen fixed-data numerical baseline and privileged ceiling, with scorer separation, before another open-model proposal can be informative.

The [fixed-data signal-localization numerical screen](../../../results/research_boxing_signal_fixed_data_dev_v1/README.md) fitted three models from the same 16 public observations on each of the three custody worlds. CURC jobs `33091055`–`33091057` (revision `473bc820f4e46aba4323a02dc4b6b3cb2196029f`, account `ucb736_asc1`) completed and validated; forecasts were committed before private scoring. Generic three-Gaussian-source MAE beats flexible RBF on seeds 123 and 456, but not 42. The equation-specific fitted inverse-quadratic model is best on seeds 42 and 456, but loses to Gaussian on seed 123. Its exact family is privileged; its imperfect fitted performance is **not** a numerical ceiling, correcting the frozen protocol's wording. This is descriptive-prior development evidence only. No LM or API was used, and the next A study must include these numerical controls and misleading metadata.

The [signal-localization wrong-metadata gate](../../../results/research_boxing_signal_prior_gate_dev_v1/README.md) used the same 16 acquired responses and a frozen first-eight/next-eight split to compare a correct three-source Gaussian proposal, a false one-source Gaussian proposal, and an RBF fallback. CURC jobs `33091738`–`33091740` (revision `f28464be057d7843a4955efb5040c973c37ccae0`, account `ucb736_asc1`) completed; public-only predictions and receipts were committed before separate scoring. Validation accepted the wrong proposal on all three worlds. Its held-out MAE harms versus fallback on seed 42 but improves on seeds 123/456, even beating the correct-count proposal there. The correct source count is not a reliable discriminating prior at this query budget; this gate does not recover from misleading metadata. Do not expand it as-is to new seeds or claim semantic value.

The [equal-strength public diagnostic](../../../results/local_boxing_signal_equal_strength_dev_20260928/README.md) tested the other explicit descriptive cue using only a frozen first-eight/next-eight split of the same 16 public responses. Equal Gaussian-source amplitudes improved public validation MAE against freely fitted amplitudes on seed 123 but lost on seeds 42 and 456. This is a mixed, reused-world development outcome, so no open-Qwen GPU job was launched for this restricted cue. The shared CURC SSH master was healthy at the 28 September check; no ACE `acer_` jobs were queued or running then. No closed-model call or other project's job was touched.

## Bounded-degree joint-action confirmation (28 September)

The [20-system fresh confirmation](../../../results/local_connected_binary_tree_pair_confirmation_20260928/README.md) tested the separate joint-action capability question after the risk-selector development gate failed. Protocol and code were frozen at `2ab453fe103a4313a6b29ed112958362d229014d` before systems 500–519 were generated. All 160 method cells, 1,000 actions, exact costs, and receipt hashes validate locally. The primary coverage-pair-versus-coverage-single ratio of means is 0.705, but the paired difference interval [−0.03150, +0.00043] crosses zero, so the prespecified joint-action confirmation **fails**. Coverage pair beats random pair under its separate frozen contrast. Posterior-risk pair has the lowest mean error on these fresh systems, a reversal of the three-system development screen, but that post hoc observation does not promote the risk controller. This local numerical run took only seconds; no CURC job or model API was warranted. The shared CURC channel remained healthy and no ACE `acer_` jobs were active.

## NeuronBench trace-informed forecaster readiness (28 September)

The [public trace audit](../../../results/local_neuronbench_trace_readiness_20260928/README.md) validates 48 archived voltage traces across six designed worlds, both acquisition arms, and four counted actions per arm. All have finite values and uniform recorded-index stride; upward zero crossings after the public test-start offset reproduce every archived spike count. `h_sag` and `ca_rebound` return identical counts for the same four random-arm actions but have different voltage trajectories, so scalar-count forecasters discard potentially useful information. This is only a data-readiness finding. The [next model plan](neuronbench_trace_forecaster_plan_2026-09-28.md) fixes an approximate, world-agnostic state-space family and a public-only gate before private scoring. No new oracle query, CURC job, or closed-model call was made; the shared CURC connection was healthy and no ACE job was active.

A follow-on read of the pinned NeuronBench public protocol helper, followed by an independent reconstruction against all 48 archived actions, fixed the trace-forecaster timing contract: 0.01 ms integration, tenfold trace subsampling, 20 ms baseline, 80 ms tail, and a scored window beginning with the final contiguous positive run. Four `ca_rebound` release-only forecast protocols have no positive run and therefore count from index zero. Starting after the negative pre-pulse would mis-score them. No world-specific truth, hidden mapping, or private response was read; no new query, job, or API call occurred.

The first [trace-informed state-model public gate](../../../results/local_neuronbench_state_forecaster_public_dev_20260928/README.md) is complete. Source and protocol were frozen at `839e344d4141c6e9259d7a226598c13af4a482d4` before the 12-cell, 48-fold public screen. The approximate integrate-and-fire model beats all three simple controls in pooled public MAE (7.00 versus best-control ridge 8.11) but wins against the best control in only 5/12 cells, below the prespecified eight-cell consistency gate. Stop this model variant before private scoring; the earlier count-only acquisition results remain predictor-dependent. All public hashes/folds validate, with zero new oracle queries, CURC jobs, or closed-model calls.

## Exact-old transfer benchmark audit (28 September)

The [evaluation-only audit](../../../results/local_transfer_exact_old_audit_20260928/README.md), revision `81b7974e4a673c24f11a98702d331885d673a1e4`, confirms that all 1,824 unchanged node instances in the 12-seed development transfer grid have old coefficients bitwise equal to target truth; 336 changed instances do not. For family k=10 at floor six, all 240 untouched nodes receive six examples and no source switch; the 78 also receiving six under uniform have identical summed error, while the 162 receiving seven under uniform account for all aggregate untouched harm (their error ratio is 1.0939). This narrows the floor failure to allocation arithmetic on a deliberately exact-old generator. The [finite-source next gate](transfer_finite_source_plan_2026-09-28.md) will give policies only modules estimated from counted source data, leaving nonzero source uncertainty. No new target query, CURC job, or closed-model call was needed for this archive audit.

The [finite-source preflight](../../../results/local_transfer_finite_source_preflight_20260928/README.md) used frozen protocol and code at `f895116475a37dfb0e9137d8f905c3879ddda1db`. Across 12 reused development systems, 24 cells contain 28,800 counted synthetic source training responses and 92,160 separate evaluation holdout responses. At 16/64 source examples per node, mean clean-source MSE is 0.020316/0.002314 and pooled 90% predictive coverage is 0.902409/0.900543. All cells passed count, covariance, nonzero-error, coverage, and deterministic replay/hash checks. These results validate source-estimation custody and matched-family calibration only; they are not a target-policy result. The next target comparison requires a separately frozen protocol and stricter information separation. The shared CURC master was healthy with no ACE `acer_` jobs active; no CURC submission or closed-model call was made.

The [finite-source target development screen](../../../results/local_transfer_finite_source_target_dev_v1_20260928/README.md) froze its protocol at `c34fe98` and ran corrected code at `08620e9` on the same 12 reused seeds. All 144 setting cells, 57,600 maximum-budget target examples, 38,880 node-method-budget rows, source hashes, and deterministic replay receipts validate. At 200 target examples, warm source divided by scratch changed-node MSE is 3.82–5.77 for family changes with 16 source examples/node and 4.94–7.71 with 64, while untouched-node ratios are 0.026–0.033 and 0.006–0.007 respectively. This **fails** the frozen useful-prior gate despite strong untouched protection: a more precise old module makes family-change adaptation worse. Uniform warm transfer is not promoted. A protected change detector with a scratch fallback is the next local numerical question. The shared CURC master was healthy, no ACE `acer_` job was active, and no CURC job or closed-model call was used.

The [counted finite-source scratch-switch screen](../../../results/local_transfer_finite_source_switch_dev_v1_clean_20260928/README.md) froze its protocol at `f51d792` and ran corrected code at `41d1cc5` on the same development systems and target prefixes. All 144 cells, 12,960 node rows, parent-metric parity checks, source hashes, exact target counts, and deterministic replay receipts validate. At 200 target examples it makes zero false switches on unchanged nodes and preserves their warm-source error, but the frozen gate fails in six of 24 strata. Family changed-node MSE divided by scratch is 1.267/0.870/1.760 for k=1/3/10 with 16 source examples/node and 1.572/0.920/1.483 with 64. A few missed or poorly recovered family changes dominate error despite high switch rates. Retain this negative result; do not tune the threshold on these reused systems or promote the method. No ACE job was active on the healthy CURC channel, and no CURC submission or closed-model call was made.

An [evaluation-only miss audit](../../../results/local_transfer_switch_miss_audit_20260928/README.md) inspected all 144 switch receipts with byte-identical replay. At 200 target responses, 25/168 family-changed cases remain unswitched with 16 source examples/node and 14/168 with 64. The missed cases account for 29.604 and 20.395 summed excess MSE over scratch. Eleven and three failed the four-response nomination even though all subsequently accumulated scratch-favoring evidence above log(4). Removing nomination alone would flag 6/912 and 3/912 untouched family nodes, adding 2.397 and 0.539 summed MSE if switched. This is a post hoc diagnostic on reused systems, not a validated new controller; freeze a rule before any fresh test. No new observations, CURC job, or closed-model call were made.

The [fresh-system soft-mixture validation](../../../results/local_transfer_soft_mixture_fresh_20260928/README.md) froze protocol `a6d9740` and code `ce2f137` before generating 20 systems (700–719). It spent 48,000 counted source and 48,000 counted target responses across the prescribed arms and settings. All 40 source and 240 target receipts, hashes, exact budgets, finite metrics, deterministic replay, per-seed means, and paired bootstrap analysis validate. The soft source/scratch mixture with fixed 0.1 change prior protects untouched nodes but **fails four of 24 point-gate strata**, all at k=1: family changed-node ratios versus scratch are 1.207/1.299 for 16/64 source examples per node, and coefficient ratios versus the better control are 1.276/1.067. The k=3/10 point strata pass. Do not promote or retune this rule on the new systems. Source/target features match the synthetic generator; misspecified families and scale remain untested. The shared CURC channel was healthy and no ACE `acer_` job was active; this short numerical validation needed no CURC or closed-model call.

An [evaluation-only four-response rank audit](../../../results/local_transfer_assay_rank_audit_20260928/README.md) finds the k=1 changed node in the top eight evidence-ranked nodes in 80/80 source-size × type × system cases on the archived 700–719 worlds, with rank one in 79/80. All source hashes and replay receipts validate. This makes a fixed top-eight allocation test worth trying on untouched systems but says nothing yet about its downstream MSE or unchanged-node protection. The ranking audit added no query, CURC job, or model call.

The [fresh top-eight allocation study](../../../results/local_transfer_top8_allocation_fresh_20260929/README.md) froze protocol at `4f014730` and code at `72886204` before systems 800–819. All 40 source and 80 target cells, 48,000 source training responses, 32,000 target arm-response counts (20,693 unique acquired prefixes), action and metric hashes, per-arm 200-response budgets, and replay receipts validate. At k=1, adaptive soft mixture divided by uniform soft mixture changed-node MSE is 0.030/0.036 for family changes and 0.226/0.188 for coefficient changes at source sizes 16/64. The full point gate **fails two of 16 contrasts** in the 16-source coefficient arm: candidate versus adaptive scratch changed-node MSE 1.116 and candidate versus uniform soft unchanged-node MSE 1.102. Changed-node gains are mainly acquisition gains, since adaptive scratch also greatly beats uniform scratch. The result is a promising matched-family numerical controller diagnostic, not a promoted foundation-model result. A harder family and untouched-node protection are required next. The shared CURC connection was healthy with no ACE `acer_` job active; no CURC job or closed-model call was needed for this short local test.

The [B1 nuisance-adjusted interaction audit](../../../results/local_joint_interaction_information_20260929/README.md) resolves an action-menu ambiguity. In the linear interaction toy with unknown intercept and main effects, repeating a single-parent intervention at one fixed value leaves the interaction coefficient exactly confounded with the other parent's main effect, even for nonzero background variation. A single fixed joint pair is also rank deficient. Varying the single-target value across experiments yields information `tau²/sigma²` per response when the other parent varies with SD `tau`; a balanced joint factorial gives `1/sigma²`. At `tau=0.1`, `sigma=0.15`, those are 0.444 and 44.444. All 20 deterministic matrix cases and replay hashes validate. Future B claims require matched varied-value single and joint controls. No simulator oracle query, CURC job, or closed-model call was needed.

An [executed-action audit](../../../results/local_connected_action_values_audit_20260929/README.md) checked the hashes and action traces of all 20 archived B confirmation systems. Coverage single made 10 actions on 10 motifs at one value per motif; coverage pair made 5 actions on 5 motifs at one value vector per motif. Risk pair made 5 actions on 2.95 motifs on average and revisited a motif at different values in every system. Thus the archived risk-versus-coverage result mixes motif selection with value diversity. This is an action-trace finding, not proof that the full learner design is rank deficient. The B gate remains failed; freeze a matched varied-value pair control before any fresh confirmation. The audit added no simulator query, CURC job, or closed-model call.

The [fixed factorial-hub development control](../../../results/local_connected_factorial_hub_dev_20260929/README.md) was frozen at `2b8e112f` and run on three reused binary-tree systems. It covers all four pair-value combinations at motif 0 plus one motif-1 pair action, spending the same 360 cost and 40 responses as risk pair. Mean feasible MSE was 0.04318 versus risk pair 0.04820, with a highly uneven three-system pattern; coverage pair remained 0.03418. Archived hashes and all prior action/metric rows matched, and deterministic replay matched byte for byte. This is development only; no B claim is promoted. The next B decision is whether a fresh, powered and value-matched confirmation merits new systems. No CURC job or closed-model call was needed for the 120 synthetic responses.

The [fresh factorial-pair confirmation](../../../results/local_connected_factorial_pair_confirmation_20260929/README.md) froze code and protocol at `5fde2a2e` before systems 900–939. Risk pair achieved mean feasible MSE 0.026942 versus 0.039722 for the fixed full-factorial pair control at identical 360 acquisition cost and 40 responses. The ratio of means 0.6783 and paired difference 95% interval [−0.020726, −0.004834] pass both prespecified thresholds; risk won 26/40 systems. All 40 receipts, 360 arm rows, 17,600 arm-response counts, file hashes, costs, action schedules, and byte-identical replay validate. This confirms a policy advantage over that fixed varied-value control **within this synthetic binary-tree family**. It does not identify which part of the risk policy helps, establish joint-action superiority over single actions, or show an LM benefit. A motif-visit-matched fixed value schedule and broader family are next. No CURC job or closed-model call was needed for this short local simulation.

The [fixed-value motif-selection development screen](../../../results/local_connected_fixed_value_dev_20260929/README.md) froze a control before rerunning systems 400–402. It keeps posterior-risk motif scoring but cycles actuator values deterministically within each visited motif. Its mean feasible MSE was 0.03670 versus 0.04820 for risk pair; the three paired outcomes vary and the fixed-value arm wins only one. Archived metric/action parity, exact cost, hashes, and byte-identical replay validate. This is a small reused-system diagnostic, not evidence of value-optimization superiority or equivalence. It justifies a fresh within-policy comparison before attributing the earlier B result to optimized actuator values. No CURC job or closed-model call was needed for 120 new synthetic responses.

The [fresh value-selection confirmation](../../../results/local_connected_fixed_value_confirmation_20260929/README.md) froze the primary risk-versus-fixed-value comparison at `d5205857`, repaired a pre-outcome receipt-key compatibility issue at execution revision `087aa108`, and then generated 80 new binary-tree systems (1100–1179). Risk pair mean feasible MSE was 0.025486 versus 0.032938 for risk-scored motif selection with fixed, balanced values. The ratio 0.7738 and paired difference 95% interval [−0.011535, −0.003369] pass both prespecified thresholds; risk won 58/80. All 80 receipts, 800 method rows, 38,400 synthetic arm-response counts, costs, file hashes, value cycles, and byte-identical replay validate. This supports value optimization within the specified synthetic policy family. It remains a policy comparison because different acquired data can change later motif choices; it is not a foundation-model or general joint-actuation result. No CURC job or closed-model call was needed.

The next C implementation is specified in the [held-out-form gate](transfer_heldout_form_gate_2026-09-29.md): one changed node acquires an additive `tanh(1.7*x1 + 0.8*x2)` component absent from the fitted six-feature bank, with finite source observations, 200 target responses per arm, and the old top-eight rule. It retains unchanged-node protection and changed-node scratch comparisons as gates, and requires true conditional-mean scoring rather than coefficient-only scoring. This is a design, not an outcome; no new target observations, CURC job, or model call have occurred. The PEV shifted-family primary gate and NeuronBench state-forecaster public gate remain failed, so neither justifies an immediate same-generator expansion.

The [held-out-form result](../../../results/local_transfer_heldout_form_fresh_20260929/README.md) is now complete on 20 new systems at frozen revision `6c96c5c1`. The full gate **fails**: with 16 source responses per node, adaptive soft mixture versus uniform soft mixture has unchanged-node MSE ratio 1.1551 (limit 1.05), although changed-node ratio is 0.3174 and every changed node is nominated in the top eight. With 64 source responses, all point contrasts pass, but both source-size strata were required. Adaptive mixture essentially equals adaptive scratch on changed nodes; the out-of-bank form leaves a mean 0.04549 approximation residual. Independent score recomputation and byte-identical replay validate 40 source and 80 target receipts, 48,000 source training responses, 32,000 target arm-response counts, 10,988 global unique acquired response identities, all hashes, and 14,400 scores. This rejects the current safe-transfer rule on the new family; it does not negate the acquisition gain. No CURC job or closed-model call was needed.

An [evaluation-only allocation audit](../../../results/local_transfer_allocation_harm_audit_20260929/README.md) localizes the 16-source unchanged-node failure. The 22 unchanged nodes per system left with four target responses have 1.3276× uniform error and contribute +1.4080 summed error; the seven unchanged nodes given 14 responses contribute −0.5059, leaving +0.9021 net error across 20 systems. The unchanged-node soft mixture is numerically identical to source-warm to displayed precision, so the immediate problem is target allocation. The changed node ranks first in all 20 systems, making the benchmark unusually easy. A six-response floor plus five extra responses on each of the top four is a concrete next development rule, but its selection used these outcomes and requires a new frozen system set and harder localization before promotion. All 40 input receipts, hashes, 1,200 node rows, and byte-identical replay validate; no new query, CURC job, or model call was made.

The [fresh coverage-floor study](../../../results/local_transfer_floor6_fresh_20260929/README.md) froze its rule and two held-out strengths at `e17e69d1` before systems 1300–1319. It gives all 30 nodes six responses, then five extra to each top-four evidence candidate, for 200 target responses per policy. **All 16 prespecified point contrasts pass** across 16/64 source responses and strong/weak held-out changes. Changed-node floor-mixture versus uniform-mixture MSE ratios are 0.1977/0.1352 at source16 and 0.1977/0.1156 at source64; unchanged ratios are 1.0301/1.0298 and 0.9983/0.9983. The 16-source unchanged absolute difference is small but positive, and changed-node paired intervals include zero; this is a point-gate capability, not a resolved population gain. The changed node is top-four in 20/20 strong and 19/20 weak systems at source16. The mixture often equals adaptive scratch on changed nodes, so no special source-model benefit follows. Forty source and 80 target receipts, exact 48,000 source and 48,000 target arm-response counts, 11,064 global unique acquired identities, 21,600 independently recomputed scores, hashes, and byte-identical replay validate. No CURC job or closed-model call was needed.
The [fresh fanout pair comparison](../../../results/local_connected_fanout_pair_confirmation_20260929/README.md) froze protocol at `5fac127b` before generating 40 new shared-parent systems 1400–1439. Risk-pair mean feasible MSE 0.013271 versus fixed factorial-hub 0.015352 gave ratio 0.8645 and paired 95% interval for difference [-0.004491, +0.000329]. It **failed** both parts of the frozen gate; risk won 25/40. Static hub controls were 0.0160–0.0168. All 40 receipts, 400 metric rows, 19,200 arm-response counts, source/protocol/file hashes, exact costs, and byte-identical replay validate. This limits the earlier binary-tree result: the current numerical selector has no confirmed advantage over the stronger fixed hub schedule on fanout. No CURC jobs were in queue at the check (shared SSH healthy), so this seconds-long local test required no cluster allocation. No closed-model API or non-ACE job was touched.

The [next transfer design](transfer_hard_localization_design_2026-09-29.md) responds to the floor-six test's easy localization. Archived action records show the changed node received the top-four bonus in 79/80 source-size × strength cells, and mixture prediction often equaled scratch on the changed node. The next development screen uses three changes and separates an upstream distribution shift from a local mechanism change. This is a specification, not new outcome data. At this hourly check the shared CURC SSH master had expired; a private reconnect was requested. Local analysis continued, and no remote job was submitted or touched.

The [connected shift invariance audit](../../../results/local_connected_shift_invariance_audit_20260929/README.md) found that the previous transfer generator draws node inputs independently and cannot test downstream covariate drift. At pinned revision `a2939ca8`, the existing connected-motif simulator passed a 12-system gate: an upstream coefficient change altered descendant input/output distributions under common exogenous noise while the descendant's conditional mean stayed byte-identical at 1,024 common parent contexts per system. An unaffected root remained identical. The audit generated 6,144 source and 6,144 target diagnostic responses, acquired zero training responses, passed receipt hash and byte-identical replay, and made no closed-model call. This is simulator validation only; the connected graph still needs integration with the transfer learner. CURC SSH was healthy at the latest check with no ACE jobs queued; this subsecond audit needed no cluster job.

The [connected transfer evidence bridge](../../../results/local_connected_transfer_bridge_dev_20260929/README.md) implements a first learner connection at pinned revision `9d1d0329`. On 12 new 30-node fanout systems with three changed motifs each, source posterior fitting and a four-response target assay rank only 23/36 changed motifs in the top four at either 16 or 64 source trajectories; the four slots contain 2.083 unchanged motifs per system on average. All seven unchanged downstream motifs per system show parent-distribution shift. This is development evidence of harder localization, not a completed fixed-budget acquisition or held-out-form test. Twelve system/trace receipts, suite hash, 240 metric rows, independent checksum verification, and byte-identical replay validate. The shared CURC SSH was healthy with no ACE jobs queued; the short diagnostic ran locally. No closed-model API or non-ACE job was touched.

The [connected query-accounting gate](../../../results/local_connected_transfer_query_contract_20260929/README.md) corrected the proposed hard-transfer design: the connected simulator exposes all node values on each trajectory, so the old 200 per-node response budget cannot be carried over. At frozen revision `463bf3d9`, three new systems each used four shared observational assay trajectories plus five eight-trajectory pair batches. Exact per-system cost was 364, with 44 acquired trajectories, 80 actuator uses, 432 natural motif labels, and eight labels masked by intervention. All system/trace/action receipts, hashes, shapes, costs, and byte-identical replay validate. This is a query contract, not a transfer efficacy result. CURC SSH was healthy and no ACE job was queued; this short CPU gate needed no cluster allocation or closed-model call.

The [connected transfer pair pilot](../../../results/local_connected_transfer_pair_pilot_20260929/README.md) is complete as a six-system development run at pinned revision `71d6dd31`. Under equal 364 cost and 44 target trajectories per arm, risk-pair source-warm changed-motif MSE was 0.00528 versus fixed factorial-hub 0.01141 with 16 source trajectories, and 0.00474 versus 0.00668 with 64. Risk masks 16 natural labels per arm versus eight for fixed, but still scores lower in this sample. Source-warm exceeds scratch error on changed motifs at source_n=16, so the transfer component is not yet established in the low-source regime. Twenty-four cell receipts, 480 metric rows, full traces/actions, exact cost and mask ledgers, hashes, and byte-identical replay validate. This is not a fresh promotion test: the comparator/topology and system count are narrow, and all changes are in-bank. No CURC job or closed-model call was needed.

The [connected held-out mechanism gate](../../../results/local_connected_heldout_mechanism_audit_20260929/README.md) adds an optional target-only nonlinear local form to the connected simulator at revision `e9b347dc`. Twelve new systems validate exact term identity under common noise and an independent three-feature projection residual MSE of 0.05840–0.06125. Twenty old/new no-heldout simulator cells remain byte-identical; the new audit replays byte for byte and hashes its metrics. This enables the next hard transfer screen but provides no acquisition or source-library result by itself. CURC SSH was healthy with no ACE job queued; no closed-model API or non-ACE job was used.

The [SCM–foundation-model viability agenda](scm_foundation_viability_agenda_2026-09-29.md) now separates five tracks with matched controls and stop gates: semantic priors, joint interventions, modular repair, action-language compilation, and partial identification. The [exact partial-identification fixture](../../../results/local_partial_identification_gate_20260929/README.md) ran at source revision `ff1b9ba8300f7252653dec6082601d9e1854bc07`. Two candidate SCMs have the same observational law but different do(X=1) means (0 versus 1/2); a do(X=1), Y=1 result rules out one candidate. Exact enumeration, source pin, receipt hash, and byte-identical replay passed. This validates a task fixture, not foundation-model competence. No CURC job, acquired training query, or closed-model call was used.

At the 29 September hourly check, the shared CURC SSH master was healthy (remote `login-ci3`); no ACE `acer_` jobs were queued. Remote `/projects/paco0228/ACE` was at `7d5de8e`, so no remote execution was launched from that checkout. The [action-menu validator smoke gate](../../../results/local_action_menu_validator_smoke_20260929/README.md) ran locally at source revision `377b72c5` and passed 11 cases; all invalid proposals were rejected before any simulator query. Its result, receipt hash, and independent byte-identical replay validate. This is a compiler safety fixture, not evidence that an FM can compile natural-language constraints or improve action choice. No CURC job, closed-model call, or non-ACE job was touched.

At the next hourly check, the shared CURC SSH socket had expired; private reconnection was requested and remote scheduler state could not be reverified. Local work continued: the [exact action-menu identification gate](../../../results/local_partial_identification_menu_gate_20260929/README.md) ran at source revision `3c664855`. A discriminating do(X=1) action has 0.311278 bits of information and is selected when offered; a menu containing only irrelevant do(Z=1) yields abstention. Exact expectations, result hash, and byte-identical replay passed. This is an evaluator fixture, not a foundation-model outcome. No simulator query, closed-model call, CURC job, or non-ACE job was touched.

At the 30 September hourly check, the shared CURC SSH socket remained absent; the earlier private reconnect request remains outstanding, so no live scheduler claim or remote submission was made. The [connected out-of-bank transfer development protocol](protocol_connected_transfer_heldout_dev_2026-09-30.md) was frozen before new systems: 12 fanout systems, three local changes including a target-only tanh component, 16/64 source strata, two cost-matched pair policies, same-data warm/scratch comparisons, 364 target cost, and explicit conditional-mean scoring. The older coefficient-only scoring would misgrade the held-out form. This is a design, with zero new simulator queries or model calls. No non-ACE job was touched.

The [connected held-out-form development result](../../../results/local_connected_transfer_heldout_dev_20260930/README.md) ran locally at pinned source revision `76aaa3f6` on all 12 frozen systems. The acquisition promotion gate fails: at source_n=64, risk-pair warm changed-motif MSE .10965 versus fixed .10716, and both source-size risk-versus-fixed paired intervals include zero. Same-data warm starts improve the fixed-schedule changed-motif means (.12440 vs .14778 at source16; .10716 vs .14778 at source64), but out-of-bank motif 7 dominates error and no 2× adaptation result follows. All 48 cell receipts, exact cost and masks, 960 per-motif metrics, and 219 replayed files validated independently. The run generated 2,112 acquired arm trajectories across 48 campaigns, plus separately counted source and sealed evaluation trajectories. The mixture gate remains unimplemented and unclaimed. CURC SSH was still disconnected, so no remote job was submitted or touched; no closed-model call was made.

The [post hoc held-out failure audit](../../../results/local_connected_heldout_failure_audit_20260930/README.md) decomposes motif-7 error at audit revision `dc733562`. Its best three-feature linear projection on the sealed panel has mean MSE .12851, while warm learners retain .31577–.36668 MSE depending on source size and policy. Thus representational floor and excess estimation/coverage error both remain. Risk directly targeted motif 7 in only 3/60 pair decisions at source16 and 0/60 at source64; fixed never did, though all arms received 44 natural motif-7 labels each. This audit is diagnostic and post hoc, not a new policy or architecture result. Input hashes and byte-identical replay validate; no new training query, CURC job, or closed-model call was used. CURC SSH remained disconnected.

The [saved-data nonlinear residual screen](../../../results/local_heldout_rbf_residual_screen_20260930/README.md) at pinned revision `cf3eaa88` added a generic 3×3 Gaussian RBF residual basis to the source-warm linear model **only at the already-known failed motif 7**. Four-fold tuning used acquired labels, not the sealed panel. On the source16 risk-policy arm, mean motif-7 MSE fell .33247→.10004 with 10/12 paired wins and a difference interval below zero; the other three source-size/policy arms improved in mean but intervals crossed zero. All 48 input receipts validate and an independent output replay is byte-identical. This demonstrates feasible nonlinear capacity on saved data, not a node-agnostic repair rule or foundation-model benefit. No new query, CURC job, or closed-model call was used; the shared CURC SSH connection was still absent.

The [node-agnostic residual development gate](../../../results/local_node_agnostic_residual_dev_20260930/README.md) at revision `09fd1dbd` extended the acquired-label CV rule to every motif on the same 12 saved systems. It selected motif 7 in all 48 cells, but 11/336 unchanged-motif cells also received residual fits. At source16/risk, all-motif MSE fell .04068→.01813; at source64/risk, it rose .03708→.06251. Seed 2007 motif 0 is a decisive extrapolation failure: CV improved .02624→.02264 while feasible error rose .01427→2.34842. All 480 linear-reference scores matched the prior run within 1e-10; byte-identical replay and hashes validate. This blocks a fresh confirmation or neural scale-up of the current repair gate. No new simulator query, CURC job, or closed-model call was used; shared CURC SSH remained disconnected.

The [acquired-context support taper](../../../results/local_support_tapered_residual_dev_20260930/README.md) at revision `7a9317a7` blends the selected RBF residual toward the source-warm linear prediction as a prediction context moves away from acquired parent contexts. On the same 12 reused systems, source64/risk all-motif mean MSE improved from .03708 linear and .06251 untapered to .01292 tapered, but harmful repairs remain: seed 2006 motif 0 rose .00461→.10443 and unchanged-node means on fixed arms worsened by more than 5% as point ratios. The current repair rule still fails its protection gate. All 480 linear/untapered parity rows and byte-identical replay validate; no new acquisition, CURC job, or closed-model call was made. CURC SSH remained disconnected.

The [action-language fixture](../../../results/local_action_language_fixture_20260930/README.md) and [validator v2](../../../results/local_action_menu_validator_v2_20260930/README.md) ran locally at `0910df0e`: eight paired English/formal-schema texts across four constraint regimes, exact legal counts 6/6/10/2, and 13 validator cases including malformed non-object proposals. A [hand-written rule parser](../../../results/local_action_language_rule_baseline_20260930/README.md) at `36f239ef` solved 8/8 menus after one phrase was corrected on the same examples. This is in-sample and demonstrates that the smoke fixture is too easy for an FM contribution claim. Prompt/answer and parser output hashes replayed byte for byte. Next gate needs independent descriptions and a parser frozen before scoring. No simulator query, CURC job, or closed-model call was made; shared CURC SSH remained disconnected.

At the 30 September check, CURC SSH recovered and the scheduler again responded. The live queue contained no ACE `acer_` jobs, and no ACE job was recorded as starting since 29 September. The [twenty-task partial-identification suite](../../../results/local_partial_id_task_suite_20260930/README.md) was generated at pinned revision `85a164ec` from the frozen protocol. It contains ten informative and ten null menus; exact intervals and posteriors passed independent rational-arithmetic checks, and a fresh generation matched all three SHA-256 hashes. This is task construction only: zero simulator queries, open-model calls, closed-model calls, or CURC jobs. The next step is a frozen evaluator and small offline open-model canary.

The [public-text arithmetic control](../../../results/local_partial_id_rule_control_20260930/README.md) and separate scorer were frozen at revision `8cdad90a`; all 20 tasks scored exactly, and invalid-response smoke cases passed. The Qwen2.5-1.5B-Instruct checkpoint revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` was found in CURC's existing offline cache. `/projects/paco0228/ACE` could not fetch because its filesystem is full; no project files were removed. A Git bundle was transferred and a sparse, detached checkout was staged at `/scratch/alpine/paco0228/ACE/code_partial_id_8cdad90a` with HEAD `8cdad90a31ef3d433eca3b4d576aff179f745af3`. The public prompt and reveal hashes match local custody; the answer key is absent from this checkout. Four-task open-model canary job `33185296` was submitted on account `ucb736_asc1`, partition `aa100`, QoS `gpu-normal`, one `a100-40gb`, four CPUs, 32 GB RAM, one-hour cap. Output: `/scratch/alpine/paco0228/ACE/results/research_partial_id_open_canary_20260930`. Initial state PENDING (Priority). The first submission attempt used an invalid GRES spelling and was rejected without creating a job; the corrected submission produced the recorded ID. No Azure or other closed-source call or non-ACE job was touched. Validate logs, receipt, eight raw calls, hashes, and response scores before expanding.

Before receiving any model output, an answer-key balance audit found that all four queued canary tasks have point `do(X=1)` intervals; the 20-task suite overall has ten nonpoint intervals and three support-collapse outcomes. The queued canary remains a transport and updating smoke. A balanced follow-up must include nonpoint tasks before treating interval performance as measured. At the subsequent hourly check job `33185296` remained PENDING (Priority), with no log or receipt. The shared CURC SSH socket then expired, and a private reconnect request was sent; no newer job state is claimed.

On reconnection, live Slurm still showed `33185296` PENDING, initially `ReqNodeNotAvail` for the 40 GB A100 nodes; no output had begun. The GPU resource audit used prior offline Qwen jobs `33041111`–`33041112`: each allocated one 40 GB A100, measured peak GPU memory 3,688 MB, and finished in 2m22s and 19s. Accounting reported average gpuutil 3 and 31 respectively, which is too coarse to infer useful-compute fraction without application token counts. A 20 GB A100 MIG test submission was rejected by the selected `aa100`/`gpu-normal` combination, which permits only `a100-40gb` or `a100_80gb`; it created no job. The pending ACE job retained the minimum permitted GPU type and its wall limit was reduced from one hour to **15 minutes** with `scontrol update`, without canceling or duplicating it. Slurm verified the new limit; the job was still PENDING (Priority) at the next check. When it runs, verify inference token progress plus measured GPU memory/utilization and compare requested versus observed resources before any repeat.

Job `33185296` eventually started and **FAILED 1:0 after 22 seconds**. Its stderr shows `FileExistsError` because `CELL_OUTPUT` named the parent directory that had been created to hold Slurm logs; the runner correctly refused to overwrite it. Slurm batch accounting reports `gres/gpumem=0` and `gres/gpuutil=0`; there was no model loading or inference and no response receipt. The failure logs remain at `/scratch/alpine/paco0228/ACE/results/research_partial_id_open_canary_20260930/logs`. A corrected single ACE retry, `33202456`, uses the same pinned source revision and checkpoint, a **nonexistent** cell output `/scratch/alpine/paco0228/ACE/results/research_partial_id_open_canary_20261001_v2`, and a separate sibling log directory. Account `ucb736_asc1`, `aa100`/`gpu-normal`, one site-minimum `a100-40gb`, two CPUs, 16 GB host RAM, 15-minute cap. The old failed job is terminal; no duplicate running job or unrelated job was touched. Validate the retry's model/token/GPU telemetry, raw outputs, complete receipt, scores, and local custody before further submission.

The corrected [open-Qwen canary](../../../results/research_partial_id_open_canary_20261001_v2/README.md) completed `0:0` as job `33202456` in 3m51s. Receipt, eight reproduced prompt hashes, two output hashes, 373 generated tokens, and local raw/log custody validate. GPU accounting shows 3,690 MB peak and average utilization 22; inference occurred, although the receipt's 19.78-second generation-loop time leaves most startup time unattributed. **All eight direct responses fail the frozen JSON schema.** A diagnostic fenced-JSON extraction, kept separate from scoring, shows 0/4 correct actions and 4/4 incorrect claims of observational identification; null menus received irrelevant actions instead of abstention. The public-text arithmetic control was 20/20. This negative development result stops direct-prompt expansion to the remaining 16 tasks and does not warrant a balanced nonpoint follow-up with the same model/prompt. No closed-model call or unrelated project job was involved.

With no ACE job queued, the [later-authored action-language stress fixture](../../../results/local_action_language_temporal_stress_20261001/README.md) was frozen before running the old rule parser. The [evaluation](../../../results/local_action_language_temporal_stress_eval_20261001/README.md) found 0/8 exact menus, down from 8/8 on the original phrasing; six were cost-phrase parse errors and two generated the wrong 24-action menu. Gold menus, validator legality, hashes, and byte-identical replay passed. This is a diagnostic of a parser the author had already seen, not independent validation or an open-model win. No CURC GPU job, model call, or simulator query was used. The next useful gate is independently authored descriptions with a frozen model prompt and deterministic validator; do not fit the parser to this set and then call it held out.

At the 1 October hourly check, CURC SSH reached `login-ci4`; `squeue` showed no ACE `acer_` jobs. Accounting still lists the failed partial-identification first attempt `33185296` and completed retry `33202456`, with no newer ACE job. A read-only cross-topology comparison found that the **same** fixed factorial-hub pair control was already used in both fresh 40-system confirmations, so another run on those generators would be redundant. Binary-tree risk-pair/factorial-hub mean error ratio is 0.67826 (26/40 paired wins; frozen primary gate passed); fanout ratio is 0.86447 (25/40 wins; frozen primary gate failed). As an exploratory scale-free description, geometric per-system ratios are 0.68929 (95% paired-t interval on log ratios transformed back: [0.54138, 0.87762]) and 0.87617 ([0.74123, 1.03569]) respectively. These intervals are descriptive and do not establish a confirmed topology interaction: the graph families were studied sequentially, and this contrast was selected after seeing both results. A useful next Track B gate needs a *new* graph or environment selected and frozen before scoring, with the same cost, action menu, risk policy, factorial control, and feasible-action endpoint. No new simulator query, GPU allocation, model call, or non-ACE job was used for this audit.

The [fresh random-recursive graph confirmation](../../../results/local_connected_random_recursive_pair_20261001/README.md) then froze a new graph rule and 40 fresh seeds at `90aae16a` before scoring. Local CPU execution yielded risk-pair feasible MSE 0.0227035 versus fixed factorial-hub 0.0591759, ratio 0.38366, paired difference 95% interval [−0.0548996, −0.0180453], 31/40 wins: its frozen primary gate passes. The 40 graphs were distinct, both primary arms spent exactly 360 cost units per system, and all 17,600 acquired arm-responses across 360 cells were accounted for. Forty cell receipts and the suite hashes validated independently; a second run was byte-identical. This is a third synthetic graph rule in a sequential research program, with binary-tree pass and fanout failure already known. It supports a restricted numerical policy result, not a broad topology or foundation-model claim. CURC remained reachable with no ACE job queued; no remote job, GPU, closed-model call, or non-ACE job was touched. The next B gate should use a preselected external task or graph distribution with the same strong fixed comparator, rather than more favorable synthetic seeds.

At the following hourly check, CURC SSH again reached `login-ci4`; no ACE `acer_` job was queued or newly recorded. The Track 5 exact arithmetic control already scores 20/20 on the finite task suite, so it is an accuracy ceiling there; a tool-assisted FM cannot outperform it on those same exact tasks. Keep the suite as a calibration/refusal diagnostic and require a harder task family before spending another model call. For Track C, the [whole-action-block CV repair diagnostic](../../../results/local_action_blocked_residual_dev_20261001/README.md) froze its change at `d9ba55dc` before rerunning the 12 saved systems. It selected all 48 out-of-bank repairs but also 66 unchanged-motif repairs; unchanged feasible MSE rose by 8.8%–21.8% across the four source-size/policy strata, failing the 5% protection criterion in every stratum. All 480 cell hashes and linear-score parity validated, and a second run was byte-identical. This is another negative post hoc development result: no fresh confirmation, neural scale-up, new simulator query, CURC job, GPU, model call, or non-ACE job was undertaken.


## Broad experimental portfolio design (1 October, after recovery plan)

At Patrick's request, the [broad experimental agenda](broad_experimental_agenda_2026-10-01.md) and [registry](broad_experiment_registry_2026-10-01.json) now specify 11 work packages and 22 proposed/conditional experiments. They retain all negative results and add intervention-conditioned inference, learned design, causal experiment memory, representation tests, verified hypothesis search, and an optional real perturbation-data lane. Initial priorities are saved-result diagnosis, external task feasibility, intervention-context split/recoverability design, and independent language sources. No new experiment was run or CURC job submitted in this planning step; resource figures are ceilings, not observed demand.

Two design corrections matter for subsequent execution. The action-language scorer uses private gold schemas for evaluation; a runtime component with the same complete schema must be compared against exact enumeration and cannot be credited as a model capability. Also, a small learned prototype may be justified by recoverability, matched-control headroom, and measured resources without a hand-written method first solving the same learned subproblem. Existing failed semantic/repair methods keep their stop gates. Local document links and all 22 unique registry IDs validated; the literature check identifies prior art rather than asserting novelty. No closed-source model call or unrelated job was involved.


## CausalMan source audit and independent-language intake (2 October UTC / 1 October local)

Shared CURC SSH reached `login-ci4` during the 00:28 UTC heartbeat. The ACE-only queue filter was empty. Accounting since 1 October still shows only the recorded partial-ID failed first attempt `33185296` and completed retry `33202456`, both on `ucb736_asc1`; no new ACE submission or result was found. Existing validated local custody remains unchanged. No unrelated project job was modified.

The [CausalMan source audit](../../../results/local_causalman_source_audit_20261001/README.md) verified 12 source/configuration blobs against pinned upstream revision `17529dad5ec8b8c691494c617b9af4533aa44bf8`, recorded hashes, and replayed byte-identically. The micro subtree is 144,812,833 bytes rather than requiring the entire package. The high-level API mixes fixed batch/path models, returns only the last graph, and selects public columns partly according to the action; sampling seeds do not create independent SCMs. A naive multi-seed confirmation through that API is not justified. The next engineering gate is a bounded single-pinned-path sampler with verified public columns, target values, actual row counts and resource timing. No simulator query, model call, or GPU allocation occurred; source checks are not a runtime success.

The [language-source intake](action_language_source_intake_2026-10-01.md) identifies three official apparatus/API documents in one Opentrons source group. They contain state, adjacency, compatibility and exception constraints absent from our current static schema. There are zero adjudicated tasks: independent provenance and formal-state validation must precede a frozen comparison. The experiment registry now distinguishes these concrete source-audit/intake steps from still-proposed experiments. The bounded prior-safety and modular-repair studies remain pending; no old failed gate was relaxed by these audits.

## Fixed-path engineering feasibility and technical narrowing (2026-10-01)

Ran `scripts/research/probe_causalman_fixed_path.py` against verified upstream revision 17529dad5ec8b8c691494c617b9af4533aa44bf8. Receipt: `results/local_causalman_fixed_path_v2_20261001/complete.json`. Nine actions / 144 generated rows, 53 stable public columns, finite responses, exact intervention clamps, unchanged canonical mechanisms. Sampling ~0.97 s, peak RSS 221052928 bytes on macOS. Initial pickle-byte identity assertion failed; canonical symbolic mechanism/edge/metadata comparison replaced it, explicitly excluding NodeModel object identity. Preliminary four-row observational inspection also occurred (148 total generated rows across successful inspections, plus 16 from the initial failed assertion run = 164 across this local session). None are confirmation data. No Slurm submission or model API call.

Technical path documented in `technical_path_2026-10-01.md`: external CPU headroom gate → matched-history intervention-conditioned set predictor → conservative conditional-shift repair. Language compilation remains gated on independent adjudication. Mathematical intervention feasibility does not certify physical actuation; cross-system generalization remains untested.

## 02 October 01:29 UTC heartbeat: external task suitability gate

CURC shared SSH reached login-ci4; ACE `acer_` queue empty. Accounting still contains only failed 33185296 and completed 33202456; no new remote result or submission. Local static joint-task audit verified every cached upstream source/graph SHA-256 against the prior probe inventory. Four threshold checks all pass; only shared public descendant is the product quality result. Therefore the existing four paired probe actions do not exercise varying joint quality factors. Retain engineering success but stop the headroom comparison on this menu. Evidence: `results/local_causalman_joint_task_audit_20261001/{audit,complete}.json`; runner `scripts/research/audit_causalman_joint_task.py`. Zero generated rows/model calls/GPU usage this heartbeat. External suitability remains open; next inspect documented actuator/outcome motivation or the predeclared Causal Chambers fallback. Prior-safety diagnostic, language adjudication, and bounded repair validation remain pending. No non-ACE job changed.

## 02 October 02:30 UTC heartbeat: predeclared external fallback audit

CURC reachable on login-ci4; no queued ACE `acer_` jobs. Accounting unchanged (33185296 failed,33202456 completed). Audited Causal Chambers `lt_malus_v1` README, generator and variable dictionary at pinned repository commit 0fa8222dc761829270c8959e0ba53b261b075e1c, retained source bytes and SHA-256 receipt locally. Joint polarizer commands support a prospective offline prediction task; no arbitrary-action simulator or independent-world sample is claimed. Prose includes +90 whereas generator grid excludes it, so archive commands must settle the domain. Wrote bounded integrity/split/control gate in `causal_chambers_gate_2026-10-01.md`. This also supplies a second independent language-source group, with zero adjudicated tasks. No measurement download, outcome scoring, simulator queries, model call, GPU allocation or new Slurm job. Prior diagnostic and repair validation remain pending; no failed gate relaxed.

## 02 October 03:30 UTC heartbeat: physical archive integrity

CURC reachable on login-ci4 with no queued ACE jobs; accounting unchanged. Downloaded only the 602229-byte preselected lt_malus_v1 archive; published MD5 and local SHA-256 verified. Twelve files, 12000 total recorded rows, complete checked action metadata and increasing timestamps. Three duplicate occurrences across two conditions require grouped action splits. Runner `scripts/research/audit_chambers_archive.py` computes metadata only, with receipt in `results/causal_chambers_archive_audit_20261001`; raw local custody `/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip`. No outcome scoring, new experimental responses, model calls or jobs. Next gate is a frozen split and baseline protocol. Other safety/repair/language gates remain pending.

## 02 October 14:23 UTC heartbeat: first frozen physical prediction screen

Initial shared SSH check failed (socket absent); Patrick then reconnected and the live check reached login-ci4. ACE queue empty, accounting unchanged. Committed protocol c9649ecb before outcome access: white_64 only, vis_3 from commanded angles, fixed 30-degree joint blocks, 789 training and 211 test rows. Normalized held-out MSE: constant .984322, additive Fourier 2.951612, joint tensor Fourier .079841, fixed physical basis 1.629344. These are one-apparatus development results, not independent-world confirmation. Joint dependence is predictively useful relative to these particular additive features. The physical basis permits only scale/intercept and does not calibrate commanded-to-physical angle offsets; its failure does not refute Malus' law or beat a fully calibrated physical model. Next justify/calibrate that control using training data and a separately frozen development protocol before any neural escalation. Eleven conditions remain unscored.

The initial matrix-product prediction emitted floating-point warnings despite finite results. Recomputed all predictions via independent explicit einsum summation with finite checks; all four MSEs agree to 1e-12 relative tolerance and no warning. Initial and verified receipts retained; initial script hash represents pre-change code. Verified result path `results/chambers_prediction_screen_verified_20261002`; runner `scripts/research/chambers_prediction_screen.py`. No new experimental queries, Slurm submissions, GPUs or model API calls. Prior safety, repair validation and independent language adjudication still pending.

## 02 October 15:24 UTC heartbeat: bounded optical calibration diagnostic

CURC live check reached login-ci4; no queued ACE jobs, accounting unchanged (33185296 failed;33202456 completed). Frozen protocol ec2fbfad before this diagnostic: fit only white_64 training rows, 16 fixed bounded offset starts, nonnegative amplitude, minimum training error selects fit. Selected fit converged in 13 evaluations; training normalized MSE .375354, test 1.621578, versus the original fixed physical basis 1.629344 and joint Fourier .079841. Total execution .31 s. This reused development split does not provide fresh confirmation, and the simple formula is not a fully calibrated apparatus/sensor model. Offset adjustment alone does not resolve its misspecification. Stop parameter tweaking; next inspect the independently documented optical/sensor model and command conventions before any claim of advantage over physics or neural escalation. Other eleven conditions remain unscored.

Receipt and runner: `results/chambers_offset_diagnostic_20261002`, `scripts/research/chambers_offset_diagnostic.py`. All 16 starts recorded; finite predictions verified. Zero new experiment queries, model calls, GPU allocation or Slurm submission. Prior-safety and repair validation, and language gold adjudication, remain pending.

## 02 October 16:24 UTC heartbeat: corrected published physics control

Live CURC reached login-ci4, ACE queue empty and accounting unchanged. Inspected upstream causal-chamber-package at 9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6. Its model_e1 depends only on relative polarizer angle. **Our previous product formula added an unsupported first-angle attenuation factor.** Earlier fixed/offset physics scores must be labeled misspecified controls; they do not show a learned advantage over the published mechanism.

Committed correction protocol ba4f94b9 before corrected scoring. Fit two linear calibration coefficients using the same 789 training rows: training NMSE .0546565, held-out NMSE .0952700 on 211 rows. Joint Fourier remains .0798407 on this exposed development split, a much smaller gap; no statistical or independent-system advantage established. Correct baseline accounts for most predictable variation. No neural escalation justified by the earlier gap. Eleven conditions remain unscored. Preserve this task as a physical sanity/transfer control; next work should advance the pending safety/repair gates rather than repeatedly tune this exposed condition. Receipts: results/chambers_published_physics_screen_20261002, upstream MIT source retained with hash. Zero new queries/model calls/jobs/GPUs. Other pending gates unchanged.

## 02 October 17:26 UTC heartbeat: prior abstention diagnostic and repair redesign

CURC live channel reached login-ci4; ACE queue empty. Completed one archived-fit diagnostic under protocol 1a3b01d5: accept a prior only with >=20% validation-SSE improvement, otherwise broad fallback. Twelve source receipts verified. Correct-cue MSE ratios at budgets 16/32/64: .62975/.88849/1.0; wrong-cue ratios .74498/1.03669/1.0. Wrong-cue mean protection passes the 5% criterion, but correct-cue benefit fails at 64 because all proposals are rejected. Overall development gate fails. No threshold retuning or semantic-model escalation. These reused synthetic fits establish neither external safety nor a statistical guarantee. Receipt: results/prior_abstention_diagnostic_20261002.

Specified the bounded conditional-support repair redesign in conservative_repair_redesign_2026-10-02.md: disjoint fitting/selection/scoring blocks, acquired-input overlap checks, conservative corrected paired-loss evidence and fallback, unchanged-mechanism protection plus nontrivial repair benefit. Implementation/validation still pending artifact sufficiency audit. No simulator queries, model calls, GPU jobs or non-ACE modifications.

## 02 October 18:28 UTC heartbeat: repair custody and validation limits

CURC helper reports no shared master, so remote queue/accounting unavailable this hour; no reconnect attempt or unattended authentication. Continued local work. Verified all 48 acquired/action receipts plus linked source/system hashes in local_connected_transfer_heldout_dev_20260930. Raw target rows total 2112. Each campaign has six acquisition blocks; naturally labeled blocks per motif range four to six. Thus raw data is present, but the exposed, partly adaptive campaigns cannot supply the proposed fresh independent safety validation. Recorded the required randomized diagnostic design and confidence-assumption correction before any fresh experiment. Runner/receipt: scripts/research/audit_repair_custody.py and results/repair_custody_audit_20261002. No new samples, fits, model calls or GPU jobs. Repair remains specified but unvalidated; independent language adjudication remains pending.

## 02 October 19:30 UTC heartbeat: repair gate implementation and theory

Implemented fixed-sample bounded paired-loss repair selection in conservative_repair_gate.py and executable deterministic contract checks. Derived the Hoeffding/Bonferroni upper bound and worst-case sufficient power calculation, explicitly distinguishing bounded-loss protection from unbounded/relative MSE safety. At ten comparisons and alpha .05, 80%-power sufficient counts are 638/2550/10199 independent blocks for absolute bounded-loss gains .20/.10/.05. These are sufficient bounds, not required minima. All deterministic checks pass; no empirical safety claim. The earlier unspecified paired-loss bound is superseded for this reference control. Raw-MSE and unchanged-mechanism gates remain unmet. Result/receipt: results/repair_gate_resource_diagnostic_20261002. No samples or model calls generated and no jobs submitted. This advances code and theory locally; next needs a frozen independent-block variance pilot before choosing a less conservative practical test.

CURC reconciliation reached login-ci3; ACE queue empty and accounting unchanged (33185296 failed,33202456 completed). No new remote artifacts to collect or unrelated jobs modified.

## 02 October 20:30 UTC heartbeat: executed independent repair pilot

Frozen protocol 653348fc; ran one fresh three-node engineering system with unchanged/changed evaluator cases, 32 shared source rows, 64 fitting plus 512 independent diagnostic rows per case: exactly 1184 generated rows. All raw observations/actions/block losses retained at results/repair_variance_pilot_20261002 with source and output hashes. Both decisions abstain; candidate-minus-baseline means +.000281 and +.000173, no repair advantage. Derived why: any odd function on four symmetric corners is in the linear x,y span, including the selected tanh change. Therefore this diagnostic action distribution cannot expose its nonlinear nature. Stop without threshold relaxation or a seed sweep; next design must verify support-identifiability before scoring. This is a negative bounded engineering result, not population safety validation. No GPUs/model calls/jobs used.

CURC live check: login-ci3 reachable, no queued ACE jobs, accounting unchanged. No non-ACE job modified.

## 02 October 21:30 UTC heartbeat: action separation implemented

Implemented response-free finite-menu separation checks against the stronger [1,x,y,xy] comparator. Four corners admit any response; three levels still alias cubic x³ with 4x; five levels distinguish all five declared quadratic/cubic/saturating/radial examples. Assertions pass and hashes retained in results/action_identifiability_diagnostic_20261002. This provides an action-design criterion before another run, not empirical repair benefit. Wrote action_separation_design_2026-10-02.md with mathematical scope and candidate-capacity prerequisite. Zero simulator queries/model calls/GPU jobs. CURC login-ci3 reachable; ACE queue empty and accounting unchanged. Other uncompleted gates remain pending; no stop gate relaxed.

## 02 October 22:35 UTC heartbeat: capacity and executed interior-action pilot

CURC shared master absent; remote state unverified, local work continued. Fixed-width noiseless basis audit reduces residual approximation energy for all five declared functions, but includes an intercept and unregularized fitting; this is a capacity ceiling. Frozen implementation/protocol at 08f5c1aa then ran one fresh interior-action engineering pilot, exactly1184 generated rows. Changed case raw MSE .123061 linear versus .051379 candidate (about58% reduction); unchanged case .027450 versus .034379 (about25% harm if repaired). Conservative bounded-loss gate abstains in both cases; deployed fallback therefore does not realize the changed-case gain. No empirical population safety claim or gate relaxation. This demonstrates both candidate capacity and negative-transfer risk on a discriminating menu, while the gate remains too conservative in this pilot. Retain as diagnostic, do not scale or sweep seeds. Receipts/raw observations: results/repair_interior_pilot_20261002. No model calls/GPU/remote jobs. Next design question is how to obtain defensible efficient selection, not whether an unconstrained residual can fit the known change.

## 02 October 23:36 UTC heartbeat: variance-sensitive gate implementation

CURC shared socket missing; remote state unavailable. Implemented and primary-source-checked empirical Bernstein control from Maurer-Pontil2009 Theorem4, with correct range2 constant14 and simultaneous comparison correction. Independently checked transformed unit-range formula and decision edge cases. Reused saved128 independent blocks without generating responses. Upper bounds .17419 unchanged/.12367 changed, both abstain. Finite-sample range penalty alone exceeds observed changed-case gain; variance adaptation does not rescue this pilot. Retain negative result with explicit bounded-loss/fixed-sample scope. Sources and receipts: results/repair_bernstein_diagnostic_20261002. No new queries, model calls, jobs or GPU allocation. Next practical selection protocol must justify a stronger candidate, larger fixed diagnostic sample or narrower validated assumptions; no clipping/threshold retuning from exposed scores. Language gold adjudication and external transfer remain unresolved.

## 03 October 00:37 UTC heartbeat: temporal action-contract implementation

CURC shared master absent, remote state unavailable. Advanced the pending language infrastructure locally: implemented finite typed state transitions and exact legal command enumeration in action_state_contract.py. Internally authored latch/shaking fixture checks341 command sequences up to depth4, verifies illegal-command rejection/no state mutation and strict boolean typing. All pass; raw contract and source/output hashes retained at results/action_state_contract_check_20261002. No hardware/model call or simulation query. This resolves a temporal-schema blocker for eligible descriptions; independently adjudicated benchmark tasks remain zero, and spatial/unit/ambiguity clauses remain unsupported. No model or GPU gate relaxed. Repair redesign has completed bounded development with mixed candidate benefit/harm and failed certification; further repair work requires an explicitly justified protocol rather than more exposed-score tuning.

## 03 October 01:37 UTC heartbeat: temporal scorer integration

CURC socket absent; no remote-state claims. Implemented temporal benchmark scorer with private gold contract, strict proposals, complete finite candidate enumeration, source/adjudicator metadata checks and ambiguity abstention. Public projection excludes formal rules/gold. Internal contract tests pass; results/temporal_action_benchmark_check_20261002 records zero independently adjudicated tasks, model/hardware calls. This advances usable evaluator infrastructure but does not establish language performance. Next step is pinned source clauses and actual independent adjudication, not a GPU canary on the internal fixture. Repair stopped at its documented development limits; no new synthetic query or model call this heartbeat.

## Coordinated execution assignment: retained temporal source intake after d492f90b

Read latest Patrick steering and current ledger; no active local research runner found and no queued ACE jobs on live login-ci3. Retained Opentrons source HTML/text hashes plus two exact normalized source spans and an unadjudicated review packet: two context variants,39 candidate sequences each, explicit API bindings. Raw custody and reproducible preparation runner recorded. Packet has null reviewer/gold and cannot enter the verified scorer. Actual independent adjudicator unavailable/unassigned; API/robot scope and repeated-command semantics need review. One source group remains one statistical unit. No new scheduler submission, model/hardware call or simulation query. Do not recreate paused hourly schedule. Other numerical-prior/repair lanes retain their documented failed stop gates; no new run is justified solely to fill the queue.

## Explicit Build-it assignment: independent-review mechanism, 03 October local

Built blinded source-bound review pipeline,78 candidate sequence templates, explicit designation/author-conflict checks, complete coverage and source references, human/automated provenance separation and two-review reconciliation. No automatic gold release; unresolved semantics remain pending. Internal mock checks pass, but actual reviewer count remains zero. Details: independent_review_pipeline_2026-10-03.md and results/temporal_independent_review_20261003. Human reviewer designation/final approval is indispensable and unassigned; code cannot prove identity by changed names. No model spending or protocol relaxation; paused selfcheck unchanged. No external message sent, no jobs launched.

## 03 October: authorized ACE-Runner delivery development check

Patrick explicitly authorized the previously proposed bounded CPU check. Source ACE-Runner0767be28fa2ff4ab72277349abe60359e8585d24; seed7001, bundle5f89033d,4803 prepaid ACE observations; original persisted metadata and all input/grid/source hashes frozen before fitting. Six CPU threads,30000epochs, constantAdam.002, initialization seeds0/1/2. All completed in267.14seconds within7200second ceiling. Sampled worker RSS roughly389–467MB, about580–587%CPU; these samples are not peak-memory measurements. No new simulator/LLM calls, GPUs, cloud charges or CURC submissions.

Full-grid chain exact accuracy original69.692416%; refit init0/1/2=98.250240%/98.507264%/95.498752%, median98.250240%. Primary median exact error .0174976 versus original .30307584. Chain MSE .0000815686975/.0000817647260/.0000823090471, median .0000817647260 versus original .0003123575573. Flat exact accuracy77.915648%/80.076544%/81.053440%, median80.076544% versus original47.088896%. Eligibility counts3803flat,4303each link/IFTU,4803engagement/cost; all custody checks passed, original input hashes unchanged. Original score replay differs from archived exact score by one grid point; retained both values.

Development only on one already exposed seed/grid. This supports the practical delivery-model fix here, not fresh confirmation, an acquisition advantage, or population robustness. Init2 retains sensitivity despite near-identical MSE. Archived fixed-refit/Random/LHS seed7001 medians98.07488%/97.986048%/98.029824% are context only because recipes differ. Frozen study tools/numbers unchanged. PR57 remains draft and unmerged; no deployment. Compact frozen config/scores/custody manifest and exact execution script retained under results/runner_delivery_development_7001_20261003. Full weights/receipts/logs remain under /Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-development-7001-20261003. Next practical work is PR review; any broader accuracy campaign needs its own fixed protocol and justified scope.

## 04 October: agreed next-stage delivery PR review

Reviewed PR57 under the existing CPU-only development scope. Reproduced a real isolation defect: tensor/model initialization inherited the caller's default device, despite the documented CPU-only refit. Fixed via an explicit CPU device context alongside the CPU RNG fork, commit e9f811fbc68fb70681c3d89182d8ebe024882ce3. Meta-device regression requires no GPU and verifies CPU weights, restoration of caller context and bit-identical results against ordinary CPU fitting. Eleven focused tests pass. Full gate:1188unit tests passed/4skipped,100%coverage;13integration passed/3skipped;lint,docs,lock,package,frontend,dashboard all pass. Initial SSH connection expired during pre-push verification; keepalive retry reran the intact hook and pushed successfully.

PR57 updated and still draft/unmerged. Original development bundle and grid input hashes unchanged; no new accuracy campaign, simulator/LLM calls, GPU allocation, deployment or default acquisition change. The one-seed accuracy evidence remains attributed to frozen0767be2, not relabeled to the new source revision. Review receipt and verification log: results/runner_delivery_review_20261004.

## 04 October: fresh delivery confirmation registration prepared, not launched

Patrick explicitly prioritized independent confirmation; assignment authorizes protocol preparation only. Prepared docs/development/guidance/runner_delivery_confirmation_2026-10-04.md and machine registration/config/seed audit in protocols/runner_delivery_confirmation_20261004. Source ACE-Runner e9f811fbc68fb70681c3d89182d8ebe024882ce3 pinned with source hashes. Twelve prospectively hash-selected acquisition seeds have no collisions in5148 audited local metadata/manifest/CSV files containing487 known seeds, with zero read failures. Unknown remote/untracked histories still require reconciliation before launch.

Claim is delivery-recipe improvement on fresh randomized histories of one deterministic Mission-1 emulator, not independent physical systems, a never-seen benchmark, foundation-model benefit, or acquisition superiority. Frozen online comparator versus the three-init30000epoch all-paid-row delivery recipe; identical paid-call history, architecture and full-grid support, explicitly unequal fitting/data-use compute recorded as treatment differences. No new Random/LHS or historical refit comparator is presented as a fresh matched control. All12 models/logs/receipts must be finalized before scores are opened; adapter source/dependency hashes must be frozen first. One primary paired-log error comparison with R<=.80,95%upperCI<1,two-sided sign-flip p<.05 and all12valid cases; failures/partial campaign cannot yield a selected-subset positive verdict.

Explicit budget proposal: local CPU6threads,7200seconds total,8GiBRSS,5203calls/case and62436total; no GPUs/model APIs/cloud charge. Prior delivery timing267.14seconds supports about53.4minutes of delivery fits, but fresh online acquisition is unmeasured. Timing-only first-case gate includes20%reserve plus5minutes scoring allowance, with scores still sealed. Approval of this new campaign scope and outcome-blind execution/evaluator/supervisor implementation/validation remain before launch. Preparation made zero new emulator, fitting-campaign or model calls; PR57 remains draft/unmerged, frozen Peter-study analysis and results unchanged. No campaign launched.

## 2026-10-04 — outcome-blind delivery runner validation

Local adapter and sealed evaluator implemented against unchanged registration and
Runner e9f811f; eleven deterministic safety/custody tests passed. Existing
5f89033d history read for row/metadata/receipt parity only. Evidence:
`results/runner_delivery_adapter_validation_20261004/{tests.txt,freeze.json}`;
companion implementation note: `runner_delivery_adapter_2026-10-04.md`.
Zero new confirmation histories, fits, scores, emulator calls, GPU or API calls.
Campaign remains unlaunched. Gate: explicit adapter/dependency-bound authorization,
additional registry reconciliation, then first-case timing-only admission within
six CPU threads / 7200 seconds / 8 GiB sampled RSS. No PR57 merge.

## 2026-10-04 — additional delivery seed-registry reconciliation

Read-only local and live CURC audits completed. Local: 11,288 hashed files,
three proposal-only mentions, no prior-use matches. CURC broad pass: 2,690 files,
45-second partial bound; complete follow-up registry-only pass: 1,627 files,
no unread candidates or proposed-seed matches. Covered scope, oversized/raw-table
exclusions, absent remote protocol directory and per-file hashes retained in
compressed manifests under `results/runner_delivery_registry_reconciliation_20261004`.
Three reconciliation fixture tests passed; registration/adapter/dependency freeze
identities revalidated. Final readiness SHA:
81b949fdd45c6738ed28bdf55fc9bc0ed5c2972b9db7d01100381d7101e9b85e.
Campaign execution remains disabled, with zero new emulator rows or scores.
Next decision: authorize unchanged twelve-history local CPU campaign under
six-thread/7200-second/8-GiB ceilings, acknowledging audited scope; first-case
479.17-second timing admission then complete seal before scoring. No PR57 merge.

## 2026-10-05 — approved delivery confirmation started, stopped at timing gate

Patrick approved the two-hour CPU campaign in coordination transcript evidence
fco_01a10c89-47c1-7576-8fcb-05c1a24dd6bc, thread
01a10a1a-bc19-707e-95a8-46b7bb11a024. Approval/source/dependency/readiness
binding retained in results/runner_delivery_confirmation_20261005/approval.json.
Live checks showed no duplicate local workers or matching ACE/delivery Slurm jobs.
One local campaign started; supervisor PID51635, worker51660, observed 580–600%
CPU. Pinned source e9f811f and original registration/adapter unchanged.

Run terminated automatically with timing_gate_failed after536.448419seconds
aggregate elapsed; first case27424209 completed200steps/4803attempted calls and
all three30000epoch delivery fits. Refits115.84/118.85/136.17seconds. Sampled
peak process-tree RSS653066240bytes (~623MiB),1022samples. No GPU/API/remote jobs,
no second case, no evaluator invocation/seal/scores. Campaign incomplete: primary
>=20% error reduction claim untested, with no accuracy/generalization verdict.
All artifacts preserved at
/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-confirmation-20261005.
Compact start/audit/approval receipts committed under results.

Independent post-stop audit found an additional frozen adapter defect:
seal_cases sums derived startup/executed metadata counters alongside disjoint
charged roles, incorrectly rejecting real Runner metadata. Independent raw-row,
unique-index,role,total,startup/selected-counter,200step,canonical-row-hash,
metadata-hash and all three refit receipt/eligible-row checks passed. Historical
adapter and campaign source were not patched or restarted. The prospective
correction must distinguish charged roles from derived counters and add a real
metadata regression before any new approval. No score was accessed to diagnose
this defect. First-case gate failure remains valid independently of the defect.

Next decision requires a prospective amendment: repair the seal verifier and
explicitly justify revised runtime scope or a different separately registered
campaign. Do not automatically expand7200seconds, reduce epochs/inits/cases,
replace seeds, or resume after charged calls. PR57 stays draft/unmerged. The
scientific scope remains one deterministic emulator and an exposed grid, with
no acquisition-superiority or independent-system claim.

## 2026-10-05 — prospective adapter correction, no restart

Corrected disjoint charged-role versus derived startup/executed counter validation
in working adapter; added exact timing-admission receipt for future executions.
Thirteen tests pass, including read-only preserved first-case metadata and counter
mutations. Historical adapter/run/case hashes verified unchanged; no new calls,
fits, scores, campaign seal or execution. New hash/dependency/test review freeze
in results/runner_delivery_prospective_amendment_20261005. Original timing gate
failure remains independent and campaign incomplete.

Measured delivery fits370.855188seconds; remaining acquisition/setup/supervision
165.593231seconds. Aggregate elapsed proxy gives8024.857235seconds with registered
reserve/scoring allowance (~133.75minutes); exact old admission duration was not
persisted. Options: retain2h and stay stopped, or explicitly review135minute
prospective cap with separately authorized restart/history/seed policy. No cap,
seed,case,epoch orinit changes made; no automatic resumption. One-case estimate
is not guaranteed feasibility. Companion:runner_delivery_amendment_options_2026-10-05.md.

## 2026-10-05 — concrete two-hour-preserving engineering proposal

Read-only source/stage analysis identified repeated invariant normalization in
fixed full-batch delivery fits. Caching the exact float32 normalization once per
model is a plausible semantics-preserving candidate, conditional on bit-identical
fixture and real-data weights/optimizer checks. No speedup is claimed. Required
saving using unlogged-old-gate aggregate proxy:57.281752seconds (10.68%overall),
15.45%delivery reduction, three-fit target313.573436seconds with other costs fixed.
Proposed separately approved finite validation:CPU6threads/900s/8GiB,at most six
serial baseline/candidate30000epochfits on preserved rows, zero emulator/scorer/
GPU/API calls; exact checkpoints, source/custody and matched timing checks,stop
on any mismatch/cap/failure,no retry. Proposal only: no fits or execution occurred.
Companion:runner_delivery_budget_preserving_proposal_2026-10-05.md. Recommend
remain stopped under2h unless a preserving route is independently qualified and
new campaign/restart explicitly authorized. No seed/epoch/init/case/cap change.

## 2026-10-05 — authorized finite caching validation completed

Standing direct user authority for bounded existing-data local development applied;
f38f3a2a frozen before fits. Budget accounting536.448419+267.144395measured prior
seconds+1800explicit preparation reserve+900hard validation cap=3503.592814<7200.
Supervisor17210/worker17230 observed~6CPUcores; combined process-tree RSS guarded
at8GiB, sampled peak367820800bytes,1096samples. Observed sampling window574.431231s
(excludes small spawn latency), hard900s watchdog; all six serial fits complete.
20step fixture passed. All108 paired checkpoints matched parameters/buffers,
gradients/loss/Adam state; independent saved tensor checks passed for18models,
including bit-identical equality to historical first-case weights.

Three-fit totals baseline295.459031s/candidate276.747107s; saving18.711924s(6.33%),
below required57.281752s. Absolute313.573436s delivery target passed, but frozen
joint runtime criterion FAILED. Baseline itself75.40s faster than historical
fits, so cross-run changes cannot be attributed to caching. No new acquisition
or scoring, no grid/API/GPU access, originalcase unchanged, campaign stillstopped.
No automatic fitting repeats/tuning/restart/source substitution or PR57merge.
Evidence results/delivery_runtime_qualification_20261005; companion
delivery_runtime_qualification_2026-10-05.md. Outcome:equivalence qualified on
oneexisting history/runtime, performance insufficient for frozen admission.

## 2026-10-05 — saved-evidence prospective resource recommendation

Measured536.448419+267.144395+574.431231=1378.024045s; validation excludes small
spawn latency. With1800s explicit prep reserve3178.024045s; charging full900s
validation cap instead3503.592814s. Existing aggregate2h remaining3696.407186s.
No new fits/acquisition/scoring/grid access or historical changes.

Recommend explicit owner retention/no-resume amendment for original valid unscored
first case plus11unstarted originalseeds:projection7381.119132s,125minute stage
within185minute conservative aggregate, firstnewcase<=545.45s,6threads/8GiB,
originale9learner(no caching), originalcase hashes separately bound to prospective
counter/timing fix. Max aggregatecalls62036<=62436. Retention is based on fixed
seed/custody and no evaluated outcomes, not accuracy; requires explicit amendment
and validated continuation adapter, not an old approval or silent resumption.
Fresh12alternative:projection8024.857235s,135minute stage/195minute aggregate,
new prospective seed/exclusion registration and aggregatecalls67239 requiring
explicit call-cap amendment. All projectionsonehistoryproxy,notconfidencebounds.
Companion delivery_resource_amendment_recommendation_2026-10-05.md; exact bindings/
arithmetic resource_amendment_recommendation.json. Both paths remain unapproved;
campaign stopped/sealed scores unopened, PR57draft unchanged.

## 2026-10-05 — Option A explicitly approved, prospective continuation frozen

Patrick approved Option A at16:16UTC, verified in coordination record
fco_01a10cda-583b-7532-8ab8-1ca51400e4d8: retain original unscored case, complete
11original unusedseeds,125minute remaining/185minute aggregate,6threads8GiB
originalcalllimit. Immutable original case/source preserved and copied byte-for-
byte into separate delivery-option-a-20261005 output; no caching adopted.
2448local retained meta/complete/receipt registries rechecked: remaining seeds
unused, no read failures/matches. Source/archive hashes match original e9f811f.

Five continuation fixtures pass (duplicatecase refusal,firstnew timing gate,
casefailure stops without evaluator, timingstop before seal, owned descendant
process-group watchdog cleanup);13existing custody/resource tests pass. Approval,
source/case/adapter/implementation hashes and test receipts frozen prospectively
in results/delivery_option_a_20261005. Explicit separate continuation with one
7500s process-group watchdog includes nestedcase workers/evaluator; sampledRSS
covers supervisor+stage+case descendants. Prior charge/reserve3503.592814s plus
fullstage7500=11003.592814<11100,96.407186sheadroom; bounded setup/tests charged
within1800s prep reserve(no measured-historical-total claim). Firstnewseed
1726880744 admission<=545.454545s,aggregatecalls includes original4803.
No score opens until all12cases pass complete semantic/hash/modelseal; any
failure preserves partials, no retry/exclusion/replacement/epochchange.

## 2026-10-05 — Option A start verified; first new case admitted

Frozen0d6c1d13 before acquisition; verified start5a9c4fdb. Supervisor61498,
stageworker61499,firstcaseworker61501, actual~6CPUcores with combinedRSS guarded.
Firstnewseed1726880744 completed200steps/4803charged calls/all3original30000epoch
fits in492.335787s, passed545.454545s admission. Exact projection6798.832394s
<7500s remaining-stage cap. Independent rows/role/derivedcounter/digest/metadata/
eligibility/receipt audit passed. Retained originalcasehashes unchanged.
Two complete histories,9606calls; nextoriginalseed735595885 active(worker73178).
Other9unusedhistories still ordered. Scores and all12seal absent. No caching,
GPU/API/remote job/PRmerge. Stage automatically continues only under original
7500s watchdog/8GiBcombinedRSS/62436aggregatecall and fullcustody gates; no resets,
retries/exclusions/replacements. Current checkpoint receipt in
results/delivery_option_a_20261005/first_new_checkpoint.json. Final scientific
verdict pending complete12case seal and outcome-blind evaluator.

## 2026-10-05 — due Option A read-only progress audit, four complete histories

Completed original27424209 and new1726880744/735595885/983656467 independently
validated: canonical row hashes, disjoint/derived counters, sequential journals,
unique indices,200steps, finite eligible vectors, correct DO masks, all3original
30000epoch metadata/receipt/modelhash custody; original and retained-copy hashes
unchanged. Source/supervisor/case adapters still frozen. No discrepancies.
Completecalls19212; active124753321 journal snapshot4803 =>aggregate24015(lower
bound while runner remains live). Case durations492.335787/502.014370/493.844004s.
Supervisor61498/stage61499/active96450 verified, active~587%CPU; combinedRSS sample
509116416bytes(~486MiB),8GiBcap. Pinned adapter Torch6/inter-op1/OMP-MKL6; no
in-process counter introspection. Supervisor ps age1911s, conservative remaining
stage5589s, charged/reserved prior3503.592814+age=5414.592814<11100. Process age
includes tiny pre-watchdog setup, not an exact monotonic-clock reading. Same
supervisor since start, main+timer threads observed; frozen single7500s stage
watchdog covers inherited case workers, no deadline reset. All12seal and scores
still absent. No acquisition/fit/score restart, source/resource change, exclusion,
retry, caching or new jobs caused by audit. Runner continues approved seed order.
Receipt progress_audit_4_complete.json; next useful gate: activecase completion
and next ordered seed, then eventual full12custody seal before scoring.

## 2026-10-05 — due read-only audit, stage advanced to seven complete histories

Audited all seven completed cases, including new124753321/921441405/934168586:
canonical hashes, disjoint/derived counters, sequential journals/unique indices,
200steps, finite eligible vectors/correctDO masks, all3original30000epoch receipt/
metadata/model hashes passed. Original+retainedcopy and source/adapters/order
unchanged. Newly reported durations503.348625/504.843263s for firsttwo; latest
934168586 duration retained in machine case_reports receipt. No discrepancies.

Completecalls33621; activeoriginalseed546725957 journal3726 at snapshot =>37347
aggregateattempts lowerbound<62436,active<=5203. Same supervisor61498/stage61499
and live case descendant verified. Processage3080s (~51.33m); conservative
remaining4420s (~73.67m) of original7500s,noreset. Charged/reservedprior3503.592814
+age=6583.592814<11100. Current combinedRSS144834560bytes (~138MiB), below8GiB;
this is a current sample, not a peak. FrozenTorch6/inter-op1/OMP-MKL6 and observed
CPU activity; no invasive runtime counter injection. All12seal/scores stillabsent.
No retries/restarts/duplicates/source/cap/caching/threshold changes or scorer/grid
access caused by audit. Existing approved runner continues orderedremainingcases.
Receipt progress_audit_7_complete.json. Next usefulcheckpoint: activecase complete
then next originalseed520888668; final evaluation remains gated by all12custodyseal.

## 2026-10-05 — due audit, eight complete histories

Read-only re-audit passed all8complete histories, including546725957: canonical
rows/counters/journals/200steps/all3original30000epoch receipts/finiteeligibleDO
masks/modelhash custody, fixedorder and immutableoriginal/copy/source/adapters.
Active nextoriginalseed520888668(worker45172), observed498.5%CPU; frozencompute
configuration6threads/inter-op1/OMP-MKL6, no runtimecounter injection. Same
supervisor61498/stage61499; processage3563s, conservative remaining3937s (~65.62m)
of unchanged7500s deadline. Account prior3503.592814+age=7066.592814<11100.
Completedcalls38424+active3460snapshot=41884<62436; currentcombinedRSS365543424bytes
(~349MiB)<8GiB. No discrepancies, score/seal, gridaccess, restart, duplicate,
retry/exclusion/caching/source/capchange. Existingapprovedrunner continuesordered
cases. Receipt progress_audit_8_complete.json. Nextcheck: activecase completion/
nextoriginalseed1091699608; finalscoring onlyafter complete12custodyseal.

## 2026-10-05 — due audit, nine complete; observed timing slowdown

Full read-only audit passed all9completecases, including520888668: canonical
rows/counters/journals/uniqueindices/200steps/finiteeligibleDO masks/all3original
30000epoch fit receipts/metadata/modelhashes; retainedoriginal/copy/source/adapters
and orderedseedprefix unchanged. Latestcase520888668 took784.031937s versus
546725957503.740033s; slowdown observed, cause not established or corrected by
changing workload. Active1091699608(worker60062) observed441.3%CPU; frozen6thread/
inter-op1/OMP-MKL6 configured ceiling, no in-process thread counter introspection.

Completecalls43227+active4803snapshot=48030<62436; currentcombinedRSS337166336bytes
(~322MiB)<8GiB. Same supervisor61498/stage61499; processage4759s(~79.32m),remaining
2741s(~45.68m)of unchanged7500s stage. Prior3503.592814+age=8262.592814<11100.
No custody/source discrepancy; timing variability does not guarantee allremaining
cases finish. Watchdog unchanged; no reset/retry/duplicate/exclusion/caching/
source/capchange or scoring/gridaccess duringaudit. No fullseal/scores yet.
Receipt progress_audit_9_complete.json. Nextcheck activecase completion and
originalseed585254818,then884825602; scoring remains contingent on all12fullseal.

## 2026-10-05 — ten complete; noninvasive timing diagnosis and deadline risk

Read-only fullcustody audit passed all10complete cases, including1091699608:
canonicalrows/counters/journals/200steps/finiteeligibleDOmasks/all3original30000
fit receipt/hash checks. Source, original+copy and fixedorder unchanged. Complete
calls48030; active585254818(worker80064)3964journal lines =>51994snapshot<62436.
Observedworker549.5%CPU, frozen6threads; combinedRSS286490624bytes(~273MiB)<8GiB.
Same supervisor61498/stage61499, age5950s; remaining1550s(~25.83m) of original
7500s. Prior3503.592814+age=9453.592814<11100. No seal/scores or gridaccess.

Savedstage timings show fits slowing, not just setup:934168586359.18s fits/
519.97s case;546725957345.85/503.74;520888668562.98/784.03;
10916996081090.12/1381.97, with init times248.88/416.45/424.79s.
Generic host read-only diagnostics:18logicalCPU/36GiB memory; load~12.5;
~12.1GiB swap used/~18.4GiB compressor occupied, availablememory query35%;
short CPU sample42%idle. VM2secondcounterdeltas retained separately. These
observations do not establish causal attribution or prove CPU/memory saturation.
pmset reports no recorded thermal/performance warnings but no CPU power status;
not proof of no throttling. No unrelated process contents inspected or modified.

Completing both remaining histories at recent slower times is at risk before
unchangeddeadline. No parameter/priority/thread/env/watchdog/seed/source/cap
change, reset,retry,duplicate,exclusion or cancellation. Existing runner continues
only under approved guards; no partialscore. Receipts progress_audit_10_complete
and slowdown_diagnosis_10_complete.json. Next usefulcheck: activecase completion
versus remainingbudget, finaloriginalseed884825602 only if stage permits, otherwise
watchdog stop with preservedpartials and no selected-subset verdict.

## 2026-10-05 — continuation requested; existing stage remains active

Fresh read-only custody audit again passes ten complete histories. Active original
seed585254818 has4803 complete acquisition journal lines; aggregate snapshot52833
<62436. Supervisor age6184s gives conservative1316s(~21.93m) remaining under
original7500s stage clock. Combined owned-process RSS533676032bytes(~509MiB)
<8GiB. No fullseal or scores. Existing approved process continues fitting and
ordered remaining work; no new submission, duplicate, restart, source, thread,
threshold or budget changes. Recent timing risk remains; no partial scoring or
selected-subset verdict. Next checkpoint is case completion or guarded deadline.

## 2026-10-05 — Option A terminal watchdog stop; eleven valid histories

Same original supervisor enforced time_limit, exit-9 at2026-10-05T18:29:49.159899Z.
Measured stage7500.278414333996s; scheduling/cleanup overrun0.278414s recorded,
not an extension. Prior charged/reserved3503.592814290998s gives aggregate
11003.871228624994s<11100s.13942 samples, peak combined owned RSS744685568bytes
(~710MiB)<8GiB. Supervisor, stage and original process group are absent.
All11complete original histories independently pass canonical rows, counters,
200steps, finite eligible DO masks, all3original30000epoch fit receipts and model
hash custody. Original/copied case, source and adapters unchanged. Latest complete
585254818 took1730.397102s. Final original seed884825602 started automatically
in original order; watchdog preserved3326 charged attempted calls in its journal,
with no completed online dataset/model or fit receipt. Complete calls52833 plus
partial3326 =56159<62436. No fullseal, completed marker, scores or grid access.
Result remains incomplete/inconclusive; no selected-subset scientific verdict.
Terminal receipt results/delivery_option_a_20261005/terminal_audit.json records
all11 case hash inventories and partial journal hash. No restart performed.

Prospective recommendation, requiring explicit new authorization: retain all11
unscored completed histories with their immutable hashes, preserve interrupted
journal separately, and rerun only original seed884825602 from its initial state
in a new output. No supported acquisition checkpoint exists; a partial history
cannot be appended safely. Charge all3326 interrupted attempts even if unused.
Expected4803 further calls =>60962 aggregate; worst5203 =>61362, both below existing
62436 cap. Do not change seed/order, recipe, epochs, thresholds or scoring protocol.
Allow45minutes(2700s) CPU-only remaining stage including sealing/evaluation; latest
complete duration1730.397102s *1.2 +300s =2376.476522s, within2700s. Timing remains
variable; this is a bounded recommendation, not a completion guarantee. Aggregate
would reserve13703.871229s, so a230minute(13800s) total charged envelope is required,
beyond currently authorized185minutes. Keep6threads/8GiB; no GPU justified.
Before launch, freeze a separate explicit amendment covering final-seed restart,
new time envelope, exact call accounting and custody mapping; preserve both old
outputs. Validate a separate stage watchdog and aggregate ledger, duplicate-start
refusal and retained11 hash gates. Final evaluation may run only after all12 cases
are valid and sealed, and within the newly authorized stage. Do not score the11.
Without that amendment, remaining authorized work is custody/documentation only.

## 2026-10-05 — post-stop custody recheck

Independent read-only recheck matches all11 completed case inventories, partial
3326-call journal, execution hash and original/source/copy hashes to terminal
audit. No owned workers or seal/scoring/completion markers remain. Exact charged
attempts56159; only96.128771s remain in existing185minute aggregate envelope.
A final-seed restart cannot fit that remaining budget. The single pending decision
is still explicit authorization to retain11 histories and restart original884825602
under45minute stage/230minute aggregate, same6threads/8GiB and62436call cap.
Worst aggregate calls61362 including discarded interrupted attempts. No new
experiment, scoring or restart. Receipt custody_recheck.json.

## 2026-10-06 — explicitly approved final-history stage launched

Patrick approval13:33UTC Sentinel_2dbe33397af08191923b777fa9a8c2d0 relayed by
coordinator authorizes final884825602 restart,45minute additional/230minute total.
New output delivery-final-history-20261006 retains11 hash-verified unscored
histories; original stopped outputs/interrupted3326 attempts untouched. Prior
charged/reserved11003.871229s and56159calls. Preparation reserved300s within
authorized2700s; running watchdog2400s, combined6threads/8GiB, total13800s,
62436calls. Positive custody gate and negative seed/time/hash/discarded-call
checks passed before launch. Start13:35:54.449719UTC, supervisor99455, stage99460
(PGID99460), case99465; observed564.1%CPU, actual Python3.11 interpreter. No GPU
or APIs. Exact original recipe/copied adapter unchanged. Evaluation onlyafter
all12 fullseal; no partialscore. Frozen amendment/start receipts saved in
results/delivery_final_history_20261006. Existing exec session85136 owns supervisor.
