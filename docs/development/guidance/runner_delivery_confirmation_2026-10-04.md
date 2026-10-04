# Prospective delivery-refit confirmation registration

Status: **prepared for review; campaign launch is not authorized**. Date: 4 October 2026.
Machine registration and seed audit: `protocols/runner_delivery_confirmation_20261004/`.
No new emulator calls, fitting campaign, model calls or GPU allocation were made to prepare this registration.

## Exact claim and scope

On fresh randomized acquisition histories from the fixed `tlam_mission1` emulator, the frozen post-acquisition delivery recipe reduces geometric-mean full-grid exact-level error by at least 20% relative to the original online model, at identical charged emulator calls within each history.

This is a prospective replication of delivery improvement, **not acquisition-policy superiority**, a foundation-model benefit, external validation, or equal-compute superiority. Mission-1 is deterministic: the 12 independent acquisition RNG streams are the replication units, not 12 independent mechanisms or 390,625 independent test cases. The old grid and development outcomes have already been exposed. Future outcomes can be procedurally sealed, but this is not a newly unseen benchmark.

## Frozen source and comparators

Use ACE-Runner source `e9f811fbc68fb70681c3d89182d8ebe024882ce3`. Source file hashes and the complete acquisition configuration accompany the machine registration. PR57 remains draft; no merge or default change is required.

1. **Online comparator:** the actual online flat and causal-chain weights at step200, before refitting. Export and hash them before the delivery stage. Do not retrain, select, or replace this comparator.
2. **Delivery treatment:** `ace.delivery.refit_delivery` on all of that same case's retained ACE-paid observations, original persisted `meta.json` custody, 30000 epochs, constant Adam learning rate .002, initialization seeds0/1/2. Use training-only eligible-row input normalization and the frozen causal masks. Report all three models; the primary per-case treatment error is the median of their three errors. This is an evaluation of a three-fit recipe, not selection of a best model or a claim that a median-scoring exported model exists. Initialization0 is the prespecified single-model descriptive readout.

Both use the same 64-64 architecture, declared DAG, environment domains, acquisition history and evaluation support. The online model deliberately trains on its original50-row FIFO/selected-probe workflow; delivery sees all available paid rows and spends additional fitting compute. These are treatment differences, so the test cannot identify whether retention, optimization or normalization individually caused an improvement. Record online/acquisition compute and additional delivery compute separately; never claim equal fitting cost.

No new Random/LHS policies or frozen-refit fits are included: they are necessary for an acquisition claim, not this one primary delivery comparison. Archived study controls may appear in a clearly separate historical appendix, never as fresh paired controls or evidence of superiority.

## Prospective cases and exclusions

Exactly12 cases, in this order:

27424209,1726880744,735595885,983656467,124753321,921441405,
934168586,546725957,520888668,1091699608,585254818,884825602.

Rule: first32 SHA-256 bits of `ACE-delivery-confirmation-2026-10-04/v1/case/i`, modulo(2**31-1), i=0..11. This fixed rule was applied before any new outcomes. No favorable-case selection or replacements.

The structured local audit checked5148 metadata/manifest/CSV files and487 known seed values with zero read failures and zero collisions. Historical1234/2025/3141/4242/5555, confirmation7001–7011, smoke9999, and development7001 are excluded, along with all seed values discovered in the audited structured fields. This audit is not proof of absence from unknown remote or untracked runs. Before a launch, reconcile any additional known campaign registry; a collision blocks launch and requires a documented pre-outcome amendment, not silent replacement. Refit initialization seeds0/1/2 are optimization repeats, not fresh case seeds and not replication units.

## Acquisition and accounting

Use `ACEConfig.cloud()` with the exact registered JSON overrides: proposer=random, USE_LLM=false, pretrain_steps=0, pretrain_interval=0, ref_update_every=0, policy_update=none. Set seed to each listed case. Keep K4, n_val_samples500, seed rows3, learner_epochs100, learner_lr.002, buffer_steps50, observational refresh25/every5, mechanism selection, raw_sum reward, normalization enabled and train_on_lookahead=false. This is a CPU random-proposer configuration; it does not confirm the LLM-enabled shipped arm.

Construct the frozen environment and `ACEOracle.from_env`; run exactly200 steps with `run_random_baseline=false`. CPU-only process: CUDA_VISIBLE_DEVICES empty, six torch/BLAS threads, explicit CPU scope; assert every actual model tensor is on CPU.

Count every attempted environment call at the choke point, including startup validation, observations, seed rows, refreshes and all probes/fallbacks. Preserve full parameters, interventions, outcomes, role, query index and selection flag. Maximum5203 charged calls per case and62436 across12 cases; attempted calls on failed runs still count. Enforce the cap before making another call; do not assume200 steps equals200 calls. Completed cases may differ in measured calls; online and delivery share the same case's exact total and history.

Save original online weights, complete observation log, persisted custody metadata and full run configuration before fitting delivery. Verify total/per-role counts, unique query indices, schema, finiteness and correct node eligibility. Flat loss excludes non-root interventions; each node loss excludes do(node). No baseline observations enter delivery. No learner receives grid labels, private evaluator predictions or scoring summaries.

## Evaluation seal and metrics

Reuse the existing390625-point grid, file SHA-256 `0fed63740739fe0719036b528a9ee11a54c79b8f0a4085c719ed5963c5081807`; canonical array SHA-256 `a62aeac6f1c1e5e1c835cf537b3de3bfd87b6cb53c582dfe6c32cc090c49e090`. Do not rebuild or alter it.

Before outcomes, commit registration plus the execution/evaluator adapter, dependency receipt and adapter hashes. Adapter validation is restricted to mocks or previously exposed development artifacts; no fresh-case result may tune implementation or configuration. An isolated evaluator opens the grid only after **all12** online model hashes, all36 refit hashes, custody receipts and acquisition logs have been finalized. Training/acquisition processes have no grid file access. Evaluator results are withheld until the complete manifest is immutable; no tuning, additional epochs or stopping decisions based on score. Isolation is procedural protection for future outcomes, not a claim that this public benchmark has never been seen.

Chain predictions use root inputs and predicted intermediate values in frozen DAG order, never true intermediate outcomes. Score using source-pinned `ace.grid_eval` nearest-level/tie semantics, float64 score arrays and fixed8192-row inference chunks for **both** comparator and delivery. Compare all390625 points; no post-hoc split or subgroup selection. The grid includes any acquired root-design points: this is the fixed full-trade-space endpoint, not a zero-overlap held-out-row claim. Any batch-related threshold ties are disclosed. Test parser/custody/device/replay behavior before scoring.

Primary error E=max(1-exact,1/390625). Treatment E_s=median(E_s,0,E_s,1,E_s,2); comparator is original online E_s. Secondary descriptive endpoints: each initialization's exact error, chain MSE, flat exact accuracy/MSE, within-one-level, per-case wins, call counts, timings, fallback frequency and eligible rows. Neither grid points nor initializations increase n beyond12.

## Analysis and decision

One primary hypothesis, alpha .05; no other confirmatory claim. Use source-pinned `scripts.analysis.study_stats.paired_log_ratio` with all12 required pairs, floor1/390625, level.95. R=exp(mean(log(E_delivery/E_online))). Report all errors, ratios, geometric mean,95% paired-t interval on log ratios, wins, and the exact two-sided sign-flip p-value over4096 flips.

Confirm only if all12 cases are valid, R<=.80, the95% upper interval bound<1, and sign-flip p<.05. Otherwise report inconclusive/not confirmed; never interpret failure to reject as equivalence. The sign-flip test assumes null log ratios are symmetric; the t interval is model-based and not a distribution-free safety guarantee. Report this assumption and raw pairs. n12 is a bounded replication size, not an asserted80%-power calculation from the single development case. No adaptive sample-size expansion, threshold/seed changes or subgroup rescue.

## Failures and resource budget proposal

Proposed allocation: **local CPU only, six threads,7200seconds aggregate elapsed ceiling,8GiB observed-RSS ceiling**; zero cloud charge, GPUs, LLM/API calls or remote submissions. No spending is authorized by this document. Register exact UTC timestamps for preparation, acquisition, online fitting, each refit, evaluation and termination. Sample process CPU/RSS; report observed samples versus true peaks honestly.

Measured prior development cost: three delivery fits plus full-grid scoring267.14seconds on one4803-row history;12 such delivery stages project about53.4minutes. Fresh online acquisition cost is unmeasured. This is not a promised runtime or power estimate.

At launch, first complete the first registered case's acquisition and delivery fitting with scoring still sealed. Apply a timing-only feasibility gate: `1.2 * 12 * first_case_seconds + 300 <= 7200`. This allows20% timing reserve plus5minutes for final validation/scoring. If it fails, stop without opening fresh scores and request a revised resource protocol; do not reduce epochs/inits/cases to fit. A supervisor enforces the aggregate7200second cap independently, terminating only this campaign's child processes and preserving completed receipts and models. Exceeding8GiB RSS stops the campaign. Do not borrow idle GPUs or increase threads/time.

Custody/config mismatch, nonfinite eligible data/model/predictions, incomplete steps/inits, cap violation or missing artifacts blocks the primary verdict. No replacement seed or rerun after any emulator calls. A purely transport failure before any calls may retry the identical case once within the aggregate limits, with both attempts logged. An incomplete campaign reports each completed case descriptively and all failures; no selected-subset positive confirmation. Any source fix after fresh data starts requires an explicit amendment and a separately justified restart; preserve all prior data.

## What remains before launch

Approval of this exact proposed campaign/resource scope; outcome-blind implementation and validation of the execution/evaluator adapter and supervisor; dependency/adapter hash freeze; registry reconciliation. None requires fabricated labels or an arbitrary human gold-review gate. Protocol preparation is complete, but no campaign has started. PR57 stays draft/unmerged; frozen Peter-study tools, numbers and frontend contract remain untouched.
