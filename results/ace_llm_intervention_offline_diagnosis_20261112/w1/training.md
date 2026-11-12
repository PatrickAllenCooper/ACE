# ACE LLM intervention policy: training recipe reconstruction (label: training)

Prepared 2026-11-12 by a read-only subagent. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was modified. No model weights were loaded, no inference or training was run. All numbers below marked VERIFIED were read from code or recomputed from saved JSON with `/usr/bin/python3 -I` scripts kept in
`/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training/` (`a1_plan.py` to `a8_consistency.py`, plus the derived `records.json`). Items marked INFERENCE are my interpretation.

Abbreviations: `SR` = `/Users/pat/code/ACE/scripts/research`, `DG` = `/Users/pat/code/ACE/docs/development/guidance`, `RES` = `/Users/pat/code/ACE/results`.

## 0. Where the executed evidence lives (all local, all JSON readable without weights)

| What | Path |
| --- | --- |
| Executed LLM training plan (records, labels, pairs, schedule, recipe, token IDs) | `RES/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training/plan.json` |
| Per-step LLM evidence (loss, gradient norm, tokens), 279 steps | same directory, `step-NNNNN.{start,loss,gradients,update-intent,optimizer-attempt,update-returned,complete}.json` |
| Post-SFT reference log-likelihoods for the 119 DPO pairs | same directory, `reference-scores.json` |
| Final (DPO) adapter log-likelihood of every legal command at all 160 training states | `.../artifacts/fit/final-before-reload.json` |
| Greedy constrained generations of the SFT and final adapters on all 160 training states | `.../artifacts/fit/sft-generation-NN.json`, `final-generation-NN.json` (161 each, last is a repeat of state 0) |
| Worker summary | `.../artifacts/fit/summary.json` |
| Training acceptance review (job 35420219) | `RES/ace_llm_intervention_scientific_training_gpu_20261108/acceptance01/review.md` |
| Training collection (32 worlds, 160 snapshot records, 128 feedback blocks, 697 behavior decisions) | `RES/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/session/worker/task/collection/world-NNN/` |
| Exposure accounting accepted before training | `RES/ace_llm_intervention_scientific_tokenization_20261108/acceptance02/exposure-report.json` |
| Numerical ranker training evidence (279 steps) | `RES/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/cpu-phase/ranker/` |
| Deployment action counts for all 8 policies | `RES/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/actions.csv` |

Job facts (VERIFIED, `summary.json`, acceptance review): training job 35420219, one `a100_3g.20gb` MIG slice, started 2026-11-08T17:02:10Z, completed 17:16:11Z, training wall 291.5 s, CUDA step time 227.4 s total (SFT 99.9 s, DPO 127.5 s), peak GPU allocation 2.66 GB.

## 1. (a) Training data

Design source: `SR/llm_intervention_screen_protocol.py` (constants lines 27 to 33, public design 74 to 88, training rotation 124 to 145, balance 183 to 199), adopted unchanged by `DG/ace_llm_intervention_scientific_contract_2026-11-08.md` lines 9 to 31. Assembly code: `SR/llm_intervention_training_dataset.py` (`assemble_training_dataset`, lines 138 to 407).

- **Systems.** 32 training worlds, IDs 202611070000 to 202611070031 (`SR/llm_intervention_scientific_training_gpu_worker.py` lines 31 to 32). Graphs chain 12, fork 10, collider 10. Four regimes with 8 worlds each. 16 uniform-cost and 16 heterogeneous-cost worlds. VERIFIED against the saved records.
- **Episodes.** One behavior trajectory per world. Eight initial `observe_all` responses (cost 8), then acquisitions until 24 further cost units are spent. Total budget 32. Trajectory responses per world were 23 to 32 (22 worlds had 32). 697 acquisition decisions in total. VERIFIED from `collection/terminal.json`.
- **Behavior policy that visited the states.** One fixed numerical method per world, index `(pair_index + cost_index + regime_index) % 4` over `("uniform_random", "menu_coverage", "local_information", "target_risk")` (protocol line 126). Eight worlds per behavior. No LLM-visited states. Acquisition actions over the 32 trajectories: observe_all 199, set_x_neg 119, set_x_pos 114, set_y_pos 64, set_y_neg 59, set_xy 53, set_y_zero 49, set_x_zero 40. VERIFIED.
- **States used.** Five snapshots per world at post-initial acquisition cost milestones 0, 6, 12, 18, 24 (protocol line 32). Observed crossing costs were 0, 6 or 7, 12 or 13, 18 or 19, 24. The fifth is budget-terminal. So 160 records, of which 128 nonterminal and 32 terminal. All 160 prompts are distinct. Only 128 of the 697 visited decision states, plus the 32 terminal states, were used for supervision.
- **SFT examples.** 160, one per record. The label is one action chosen by the snapshot's preassigned teacher from the public envelope (`_select`, dataset lines 62 to 69, teacher call 251 to 255). Terminal records get `{"stop":true}` (dataset lines 274 to 277, "no gain-based stopping").
- **Teacher assignment.** `teacher_base = (pair_index + 2*cost_index + 2*regime_index) % 4`, snapshot `i` uses `METHODS[(teacher_base + i) % 4]` (protocol lines 127, 132). Each world sees all four teachers once across its four nonterminal snapshots. Each teacher labels 8 worlds at each snapshot index. So each teacher supplies 32 nonterminal labels and 8 terminal stop labels. The prompt does not say which teacher produced the label (`task_prompt`, `SR/llm_intervention_task.py` lines 144 to 152).
- **Label marginal (VERIFIED from `plan.json`).** All 160: stop 32, set_x_neg 32, set_x_pos 29, observe_all 24, set_xy 12, set_y_pos 10, set_y_neg 10, set_x_zero 7, set_y_zero 4. Among the 128 nonterminal labels set_x_neg is the plurality at exactly 25.0 percent, set_x_pos 22.7 percent, observe_all 18.8 percent. Marginal entropy 1.884 nats (ln 8 = 2.079).

| Teacher | Nonterminal labels (n = 32 each) | State dependence |
| --- | --- | --- |
| uniform_random | set_y_neg 7, set_x_zero 5, set_y_pos 5, observe_all 5, set_xy 4, set_y_zero 3, set_x_neg 3 | None by construction. Keyed RNG over affordable actions (`SR/llm_intervention_numerical_policy.py` lines 59 to 61). These are 25 percent pure-noise labels. |
| menu_coverage | set_x_pos 12, set_x_neg 12, set_y_pos 3, set_x_zero 2, set_y_neg 1, set_xy 1, set_y_zero 1 | Depends only on action counts in the history, alphabetical tie-break (lines 62 to 66). At snapshot 0 it is `set_x_neg` in 8 of 8 worlds by construction, because all interventions have count 0 and `set_x_neg` sorts first. |
| local_information | observe_all 11, set_x_neg 9, set_x_pos 8, set_y_neg 2, set_y_pos 2 | Depends on the SCM posterior. Ties broken alphabetically (line 86). |
| target_risk | set_x_pos 9, set_x_neg 8, observe_all 8, set_xy 7 | Depends on the SCM posterior, 32 fantasy draws per action (`SR/llm_intervention_target_risk.py` lines 91 to 136, tie-break line 130). In 17 of 32 states the top-1 minus top-2 score margin is below 2 Monte Carlo standard errors (median 1.8 SE). |

- Labels by snapshot index: snapshot 0 has set_x_neg in 16 of 32 states (8 menu_coverage, 4 local_information, 4 target_risk). Snapshot 1 mode is set_x_pos 13. Snapshots 2 and 3 mode is observe_all (9 and 7). Snapshot 4 is stop 32 of 32.
- **Legal set never varied.** All 128 nonterminal records have the same nine legal commands (8 actions plus stop) with identical token tables. Remaining budget at nonterminal snapshots was 24, 17 or 18, 11 or 12, 5 or 6, so every action was always affordable.

## 2. (c) Preference data

Code: `SR/llm_intervention_replicated_branches.py` (`build_replicated_training_preferences`, lines 122 to 285), `SR/llm_intervention_replicated_feedback.py` (`aggregate_replicated_feedback`, lines 100 to 190), `SR/llm_intervention_world.py` (`training_callbacks`, lines 345 to 393, noise keys 306 to 315).

- **Branches.** At each of the 128 nonterminal snapshots all 8 menu actions were candidates (protocol lines 128, 133). Each candidate was executed once per replicate from a fork of the same pre-action learner, with 4 replicates `r0..r3` (protocol line 33). The fork was updated with the single response and then predicted Z for the 8 public task queries.
- **Scoring.** Each replicate also bought 8 scoring responses, one per task query (the queries mirror the 8 menu actions and measure Z only, weights 0.125, scale 2). `structural_scores=True` (protocol line 138), so scoring truths are noise-disabled structural means. VERIFIED that truths are identical across the four replicates in every feedback file. Branch noise keys include replicate ID and action ID (world lines 307 to 314), so candidates do not share noise draws. There are no common random numbers.
- **Gain.** `G[c,r] = sum_q w_q * [((base_q - truth_q)/scale)^2 - ((branch_{c,r,q} - truth_q)/scale)^2] / cost[c]` (feedback docstring lines 112 to 118). This is replicated reduction in normalized prediction error per acquisition-cost unit after one response.
- **Pair label.** For every unordered pair within a state, `mean = mean_r(G[left,r] - G[right,r])`, `SE = sample_std(differences, ddof=1)/sqrt(4)`. Tie if `|mean| <= minimum_gain_gap + tie_multiplier * SE` with gap 0.0 and multiplier 2.0 (protocol line 140, feedback lines 163 to 171). Otherwise strict, chosen is the higher mean. Pairs exist only within a state, never across states.
- **Counts (VERIFIED).** 128 states times 28 pairs = 3,584 comparisons. 687 strict (19.17 percent), 2,897 tied. Strict pairs per nonterminal record: min 0, median 4.5, max 18. Nine nonterminal records had zero strict pairs.
- **Why only 119 were used.** The exposure rule (`SR/llm_intervention_training_exposure.py` lines 86 to 109, documented in `DG/ace_llm_intervention_training_exposure_2026-11-07.md` lines 33 to 51) gives at most ONE DPO pair per record: among that record's strict pairs it selects the pair with the smallest SHA-256 of the canonical JSON `["ace-matched-training-exposure-proposal-v1/unordered-command-pair", record_id, [two sorted command strings]]`, lexical tie-break. 128 nonterminal records minus 9 without a strict pair = 119. The other 568 strict pairs and all 2,897 ties were retained on disk and never exposed. The plan flag `one_pass_dpo` is `false` only because those 568 pairs have zero exposure. No pair was seen more than once.
- **Cost.** Collection charged 10,240 cost units and 9,145 accepted responses: trajectory 1,024 units and 953 responses, branches 5,120 units and 4,096 responses, scoring 4,096 units and 4,096 responses. So 9,216 of 10,240 units (90 percent) bought preference feedback that produced 119 gradient steps.

Composition of the preference labels (VERIFIED counts, my scripts `a1`, `a3`, `a8`):

| Action | Wins / losses in all 687 strict pairs | Chosen / rejected in the 119 exposed pairs |
| --- | --- | --- |
| observe_all | 53 / 71 | 6 / 13 |
| set_x_neg | 88 / 74 | 14 / 9 |
| set_x_pos | 123 / 78 | 23 / 14 |
| set_x_zero | 79 / 102 | 13 / 19 |
| set_xy | 93 / 98 | 20 / 16 |
| set_y_neg | 86 / 87 | 19 / 14 |
| set_y_pos | 75 / 98 | 8 / 21 |
| set_y_zero | 90 / 79 | 16 / 13 |

Noise evidence in the preference labels:

- 461 of 1,024 candidate mean gains (45 percent) are negative. Median mean gain is 0.00014 NMSE units against a 0.04 target. 78 of 128 snapshot states already had base NMSE at or below 0.04.
- With four replicates the tie rule is a two-sided t test at |t| > 2 on 3 degrees of freedom. Under a pure-noise null that rule declares 13.93 percent of pairs strict, which is 499 of 3,584. Observed 687. Observed tail shares versus the t3 null: |t|>3 9.9 versus 5.8 percent, |t|>4 5.9 versus 2.8, |t|>6 2.5 versus 0.9, |t|>10 0.6 versus 0.2.
- A per-state F-like ratio (between-candidate variance of mean gains over within-candidate variance divided by 4) has median 1.17 where the null expectation is about 1. Only 24 of 128 states exceed the 5 percent critical value near 2.42.
- In the 32 states where target_risk was the teacher its full score vector is saved. Strict measured pairs agree with the target-risk ordering in 123 of 179 cases (68.7 percent). Over all pairs including ties the sign agreement is 475 of 896 (53.0 percent). For local_information the figures are 118 of 176 (67.0 percent) and 493 of 896 (55.0 percent).
- Direction consistency across states: for each unordered action pair take the majority direction among its strict occurrences. The pooled majority share is 400 of 687 = 0.582. Random directions give 0.580 on average (97.5th percentile 0.603). No global pairwise ordering exists in the strict labels.
- No strict 3-cycles within a state (0 among 136 fully strict triads).
- INFERENCE. A simple mixture reading of the t-null excess and of the 69 percent planner agreement both suggest that roughly half to two thirds of the 687 strict labels are noise-driven. This assumes approximately normal replicate differences and treats the planner ordering as the reference. It is not a proof.

## 3. (b) SFT optimization, exact

Sources: `SR/llm_intervention_training_engine.py` (recipe lines 39 to 50, plan 117 to 219, `fit` 312 to 516), `SR/qualify_llm_intervention_checkpoint_gpu.py` lines 35 to 40, `SR/llm_intervention_screen_training_gpu_worker.py` lines 119 to 147, `SR/llm_intervention_learning.py`, `SR/llm_intervention_canonical_scoring.py`, contract lines 25 to 29. The executed `plan.json` carries the same recipe. VERIFIED.

| Item | Value |
| --- | --- |
| Base model | Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, 28 decoder layers |
| Adapters | LoRA rank 8, alpha 16 (scale 2.0), A Kaiming-uniform, B zeros, no dropout, on `q_proj` and `v_proj` of all 28 layers. 56 modules, 112 matrices, 1,146,880 trainable parameters (`learning.py` lines 106 to 153, `training_plan.py` lines 38 to 42) |
| Updates | 160, one per record, each record exactly once (`one_pass_sft: true`) |
| Batch size | 1 sequence per step. No accumulation (`engine` docstring lines 121 to 125, events show `rows: 1`) |
| Epochs | 1. Fixed roster order world 0 to 31, snapshot 0 to 4. No shuffling |
| Optimizer | AdamW, learning rate 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 (`engine` lines 387 to 389) |
| Schedule | Constant. No warmup, no decay. No scheduler object exists in the engine |
| Gradient clipping | Max norm 1.0 (`engine` line 445). 149 of 160 SFT steps were clipped. Pre-clip norm median 3.68, max 78.6 |
| Loss | Mean negative log-likelihood per command token, `-(sequence_logprob / token_count)` (`canonical_scoring.py` lines 99 to 102) |
| Loss mask | Command tokens plus EOS only. Prompt labels are -100 (`learning.py` `pack_actions` lines 25 to 50). The full prompt still conditions the likelihood |
| Sequence limit | 8,192 tokens with 32 reserved for generation. No truncation on either side. Oversize raises an error (`learning.py` lines 39 to 41, `language_policy.py` lines 105 to 106) |
| Actual lengths | Prompts 1,998 to 4,199 tokens (nonterminal 1,998 to 3,662, median 2,616). Commands 9 tokens for the six single-variable interventions, 8 for observe_all and set_xy, 6 for stop, each including EOS. Scored action tokens: 1,308 in SFT and 2,087 in DPO, 3,395 in total, against 1,119,066 input tokens over the 279 steps |
| Prompt format | Qwen chat template, one user turn, `enable_thinking=False`, fixed instruction plus canonical JSON envelope plus `"\nDecision:\n"` (`language_policy.py` lines 87 to 111, `task.py` lines 144 to 152) |
| Precision | Base bfloat16, adapters and reference float32, output projection float32 one row per call, strict deterministic algorithms, FlashAttention-only SDPA, non-reentrant activation checkpointing, model in eval mode during training, seed 109 |
| Checkpoint | Fixed, after step index 159. No selection |

Prompt composition at nonterminal states (VERIFIED from saved envelopes): measured history 43.8 percent of envelope characters, task block 15.0, menu 14.8, SCM posterior 11.7. About 103 full-precision decimal literals per prompt. The last 360 prompt tokens and first 204 are identical across all nonterminal prompts.

Command token structure (VERIFIED from `plan.json`): four shared prefix tokens for `{"action_id":"`, then `set` or `observe`, then `_x`, `_y`, `_xy` or `_all`, then `_neg`, `_pos` or `_zero`, then a shared `"}` token and EOS. The value tokens are shared between X and Y actions. Only two or three tokens per command carry action identity.

## 4. (d) DPO optimization, exact

| Item | Value |
| --- | --- |
| Loss | `-logsigmoid(beta * [(logp(chosen) - logp_ref(chosen)) - (logp(rejected) - logp_ref(rejected))])` (`learning.py` lines 94 to 103, `engine` lines 428 to 430) |
| beta | 0.1 |
| Likelihoods | Sum of log-probabilities over all command tokens plus EOS, conditioned on the full state prompt. Chosen and rejected share the same prompt |
| Reference | The post-SFT adapter, snapshotted once into fixed buffers at the SFT boundary. Reference scores cached before the first DPO update and rechecked at the end (`engine` lines 397 to 418, 475 to 483) |
| Batch | One pair per step, as two unpadded singleton forwards, one loss, one backward, one optimizer step |
| Updates | 119. Each exposed pair seen exactly once, roster order |
| Optimizer | Same AdamW object as SFT. Moments are not reset (`optimizer_state: continuous_across_sft_dpo`). Learning rate 1e-4 constant. Clip 1.0. All 119 DPO steps were clipped (pre-clip norm min 1.30, median 2.57, max 17.8) |
| Checkpoint | Fixed final iterate after step index 278. No development-score selection (contract line 27, worker line 96) |

## 5. (e) Numerical ranker

Sources: `SR/llm_intervention_ranker.py` (model lines 26 to 50, loss 60 to 63), `SR/llm_intervention_ranker_training.py` (recipe lines 32 to 41, loop 202 to 256), `SR/llm_intervention_task.py` `numerical_features` lines 192 to 263, `DG/ace_llm_intervention_ranker_training_2026-11-07.md`.

- **Features.** One row per feasible action plus stop. Each row is a shared state vector followed by a 28-wide candidate descriptor. The shared vector holds counters and budget, target one-hot, per-node posterior means and full covariances, adjacency and bounds, the padded menu, the padded paid history (80 rows), and the padded query law. Signed-log transform. Width 8,891. No names, units or prose. No planner scores.
- **Architecture.** `Linear(8891,128)`, Tanh, `Linear(128,128)`, Tanh, `Linear(128,1)`. 1,154,817 parameters. Seed 109. CPU float32.
- **Loss.** SFT is softmax cross-entropy over the candidate rows. DPO is the same `preference_loss` on categorical log-probabilities against a frozen post-SFT copy, beta 0.1.
- **Optimizer.** Adam, learning rate 1e-3, no clipping, moments continuous across stages.
- **Exposure.** The identical ordered schedule as the LLM: 160 single-record SFT steps then the same 119 selected pairs. That is the whole meaning of "matched example exposure". It does not match loss form, parameter count, pretraining, tokens or compute (ranker doc lines 36 to 41, contract line 27).
- **What the saved ranker evidence shows (VERIFIED).** Nonterminal SFT loss by quarter: 2.15, 2.03, 2.08, 2.11 nats, against ln 9 = 2.197. DPO loss mean 0.69339 against ln 2 = 0.69315. 32 zero-gradient steps (terminal rows with one candidate). The post-SFT ranker's maximum action probability averages 0.128 where uniform is 0.111. Its argmax over the 128 training states is set_x_pos 117 and set_x_neg 11, accuracy 32 of 128. In deployment it chose set_x_pos 435 and set_x_neg 141 of 576. The control was barely trained.

## 6. What the saved training evidence shows about the LLM (VERIFIED, scripts `a2`, `a4`, `a6`)

SFT stage:

- Per-token loss fell from 0.60 (steps 0 to 19) to 0.21 (steps 120 to 159). Because each record is seen once, each step's loss is a prequential, held-out-like measurement.
- Total NLL per nonterminal example by quarter: 3.88, 2.25, 2.57, 2.17 nats. A state-blind online label-frequency counter with add-one smoothing scores 2.12, 1.92, 1.98, 1.88 on the same sequence. The LLM never beat the state-blind baseline in any quarter (second half 2.37 versus 1.93).
- Terminal stop examples reached NLL 0.29 to 0.37 nats in the second half.
- In the last 80 steps the mean probability the model gave its label was 0.365 when the label was set_x_neg, 0.287 for set_x_pos, 0.127 for observe_all and at most 0.057 for every other action. By teacher, mean total NLL in the last 80 steps was 3.30 (uniform_random), 2.18 (menu_coverage), 1.85 (local_information), 2.15 (target_risk).
- The base model before any update gave log-probability -11.43 to the first record's label command (`first-gpu-work.json`).
- The SFT adapter's greedy output on the 128 nonterminal training states: set_x_neg 108, observe_all 19, set_x_pos 1. Agreement with its own teacher labels 36 of 128. Always predicting the modal label would score 32 of 128. By snapshot index: 32 of 32 set_x_neg at snapshots 0 and 1, observe_all appears at snapshot 2 (7) and 3 (12). This is coarse sensitivity to history length or budget, not to the posterior.

DPO stage:

- Loss over the 119 steps: mean 0.7395, median 0.6998, range 0.185 to 1.608, against ln 2 = 0.6931. After the first step, 55 steps were below ln 2 and 63 above. By segment of about 30 steps the means are 0.705, 0.813, 0.759, 0.679.
- On the 119 exposed pairs the post-SFT reference already preferred the chosen action in 66. The final adapter prefers it in 76. The margin improved on 74 pairs and worsened on 45. Margin changes range from -19.2 to +17.2 nats.
- Both members of a pair lost likelihood on average: chosen -4.69 nats, rejected -6.68 nats.
- Mean log-likelihood change by action from reference to final, on pairs containing it: set_x_neg +0.57, set_xy -0.03, observe_all -1.12, set_x_pos -3.84, set_y_neg -4.57, set_x_zero -9.57, set_y_pos -12.33, set_y_zero -13.34. The final log-probability of stop at terminal states is -3.58 on average and at nonterminal states -12.66.
- Final adapter on the 128 nonterminal training states: argmax and greedy output set_x_neg 123, set_x_pos 5 (the five are in two worlds). Mean probability of set_x_neg among legal commands 0.79 (range 0.13 to 0.93). 93 percent of unconstrained mass sits on the nine legal commands. Greedy decoding equals the full-sequence argmax at 128 of 128 states. The final greedy action equals the DPO chosen action on 16 of 119 exposed states and the rejected action on 8.
- The log-odds of observe_all versus set_x_neg move with snapshot index (-2.59, -1.97, -1.74, -1.49) and by at most 0.30 nats across graph, cost and regime group means. The gap never closes, so the argmax does not change.

The constant-action behavior was therefore visible in files written by the training job on 2026-11-08, before the development results. The worker validation and the acceptance review checked legality, exact reload and repeatability. They recorded `stop_count` and `action_count` but no action histogram.

## 7. (f) Recipe features that predict or permit a constant-action policy

| Candidate feature | Support | Contradiction or limit | Artifact that settles it |
| --- | --- | --- | --- |
| Marginal label imbalance | set_x_neg is the plurality nonterminal label (32 of 128) and 16 of 32 at the first decision. The token path set (0.81), then `_x` (0.65), then `_neg` (0.47) is the greedy marginal mode. SFT output equals that mode at 108 of 128 training states. | The imbalance is mild, 25.0 versus 22.7 percent for set_x_pos. Imbalance alone needs a state-blind learner and argmax decoding to produce a constant. | `plan.json` labels, `sft-generation-NN.json` |
| One pass over very few examples | 160 SFT and 119 DPO single-example updates. Prequential SFT loss never beat a state-blind frequency baseline. The numerical ranker under the same exposure stayed at ln 9 and ln 2. | The LLM did learn the output format and terminal stop within the pass. | `step-*.complete.json` for LLM and ranker |
| DPO pairs dominated by one preferred action | At token level the exposed pairs net favor set over observe (+7), `_x` and `_xy` over `_y` (+4, +5, -9) and `_neg` over `_zero` and `_pos` (+6, +2, -8 at first divergence). | At action level this is false. set_x_neg was chosen 14 and rejected 9 times. set_x_pos was the most chosen (23 versus 14) yet lost 3.8 nats. | `plan.json` schedule and `strict_pairs`, `exposure-report.json` |
| Preference labels approximately state-independent | None for a constant winner. | Pair directions flip across states at the coin-flip rate (0.582 versus null 0.580). The measured best action is spread over all eight actions (25, 23, 16, 15, 13, 12, 12, 12). The stronger reading is mostly-noise labels, see section 2. | `collection/world-NNN/snapshot-NNN/feedback/terminal.json` |
| Known DPO failure modes with few pairs | Likelihood displacement: chosen and rejected both fell, mass moved to set_x_neg, stop was erased. Prequential DPO loss above ln 2. Fit on the exposed pairs rose only from 66 to 76 of 119. beta 0.1 leaves the sigmoid near linear for margins under about 10 nats, so anchoring to the reference is weak. One pair per step with every step clipped. | The last 29 steps average 0.679, slightly below ln 2. | `reference-scores.json`, `final-before-reload.json`, `step-*.loss.json` |
| Learning rate relative to update count | Constant 1e-4 with no warmup or decay, batch size 1, 268 of 279 steps clipped. Updates moved likelihoods by many nats, so order and sampling noise matter. | The rate was not too small to change behavior. Frozen picks observe_all 574 of 576, SFT picks set_x_neg. | `step-*.update-intent.json`, adapter files (not loaded here) |
| Loss masked to a short command with shared prefixes | Six of nine tokens are shared prefix or suffix. Value tokens are shared between X and Y actions, which couples actions. All 128 nonterminal states have the identical legal set. | Masking the prompt is the correct state-conditioned objective. This is a coupling mechanism, not a demonstrated cause. | `plan.json` `commands` |

Additional features found while reading the code and artifacts:

1. **Mixed teachers without teacher identity.** A quarter of nonterminal labels are uniform-random noise and a quarter come from a counting rule. The two posterior-driven teachers contribute 32 labels each, and half of the target-risk labels are within Monte Carlo noise of the runner-up.
2. **Stop labels only at terminal states.** They are 20 percent of SFT steps although the grammar forces stop there anyway (`SR/llm_intervention_grammar.py` lines 73 to 75). INFERENCE: this plausibly explains the supervised arm stopping early in 23 systems. DPO then erased stop as a side effect.
3. **Alphabetical tie-breaks in every numerical teacher** favor set_x_neg among interventions.
4. **Greedy constrained decoding** (`language_policy.py` lines 163 to 165). Any stable marginal preference becomes a constant action.
5. **Off-policy states only.** None of the 96 post-initial training states has an all-set_x_neg acquisition history (maximum share 0.83). Deployment histories under the DPO arm are entirely set_x_neg. There is no on-policy relabeling.
6. **No behavioral gate at training time.** See section 6.
7. **The matched numerical control also degenerated**, so the failure is at the level of data and exposure and is not specific to the language model.

## 8. (g) Gradient diagnostics

Sources: `DG/ace_llm_intervention_gradient_diagnostics_2026-11-02.md`, `DG/ace_llm_intervention_checkpointing_2026-11-02.md`, `SR/diagnose_llm_intervention_gradients.py`, `SR/llm_intervention_gradient_probe.py`, execution ledger `DG/ace_llm_intervention_execution_2026-11-02.md` lines 781 to 885.

Established:

- GPU05 (job 35208932): full and selected-position supervised losses were identical at 4.7749, but 100 of 112 adapter gradient tensors failed the fixed tolerance (atol 1e-4, rtol 1e-2), largest difference 0.0236. The worker stopped before any update.
- GPU06 diagnostic (job 35209536): eight backward passes on one 613-token artificial task, zero updates. Default-mode repeats differed in 88 of 112 (full) and 82 of 112 (selected) tensors. Strict-determinism repeats were bitwise identical. Strict full versus selected still differed in 101 of 112, max 0.0266. Losses and aligned hidden inputs matched. This separated repeat nondeterminism from a stable path difference without identifying a kernel.
- GPU07 (FP32 projection alone) failed. GPU08 (fixed-row FP32 projection) gave identical losses and all 112 gradients across paths, then ran out of memory at 7,343 tokens. GPU09 (activation checkpointing) gave identical off and on gradients and completed 3 SFT plus 3 DPO artificial updates with 28 recomputations per backward, unchanged base and reference, and exact reload.
- In the scientific run every one of 279 steps recorded 112 present and finite adapter gradients with positive norm. Adapter signatures differ at initial, post-SFT and final. Base, projection and reference signatures were unchanged. Reload scores were exact.

Not established:

- That any gradient carried state-conditional signal, that loss fell on held-out states, or that the update direction was useful.
- That command rankings change with posterior uncertainty, costs or budget.
- That DPO improved preference margins.
- Action diversity. The GPU09 review already recorded that all four generations chose the same action `a00` and called this "not useful action selection" (ledger line 883).

## 9. (h) What a successor recipe can change at low cost, and what the code already supports

Already supported without new mechanisms:

- **More updates, multiple epochs, any order.** `prepare_training_plan` and `fit` accept an explicit schedule of up to 16,384 steps with repeated exposures counted, as long as all SFT precedes all DPO (`engine` lines 33, 187 to 204). The whole 279-step run cost under five minutes of one 20 GB slice.
- **All strict pairs.** The engine indexes any strict pair of a record. Only the exposure planner (`training_exposure.py` lines 17, 99 to 109, cap 512 steps) and the scientific worker contract (`scientific_training_gpu_worker.py` lines 85 to 99, 145 to 149) hard-code one SFT pass and at most one pair per record. Both need a new reviewed version, not new arithmetic.
- **More replicates and a stricter tie rule.** Up to 16 replicates and free `tie_multiplier` and `minimum_gain_gap` parameters (`training_dataset.py` line 36, protocol line 140).
- **Teacher distillation from target-risk scores at zero environment cost.** `choose_target_risk` returns `risk_reduction_per_cost` and Monte Carlo errors for every feasible action from the public envelope alone. Score vectors are already saved for 187 target-risk and 190 local-information behavior decisions and for the 32 plus 32 teacher snapshots. All 697 behavior-decision envelopes are saved (`world-NNN/decision-NNN.envelope.json`) and unused. They come from the same 32 worlds.
- **A converged numerical control.** The older `fit_ranker` (`ranker.py` lines 66 to 139) runs full-batch epochs over all strict pairs. It is CPU-only.
- **A training-time behavior gate.** The worker already saves full-menu scores for the final adapter and greedy generations for both adapters at every training state.

Needs a small code change:

- **Listwise ranking over the full menu.** `pack_actions` and `sequence_log_probs_unpadded` already return differentiable scores for any list of commands on one prompt. A softmax cross-entropy over the nine scores with hard or soft targets is not implemented for the LLM. It costs nine decoder forwards per step unless prefix sharing is added. The ranker's SFT loss is already this form with a one-hot target.
- **Soft targets.** The record schema carries one `supervised_action_id` (`supervision.py` lines 30 to 31). A new record version or side file is required.
- **Learning rate, beta, rank, schedule, batch accumulation.** The engine admits exactly two named recipes (`engine` lines 53 to 56) and has no scheduler or accumulation.
- **SFT scores for the supervised and frozen adapters over the full menu.** Only the 119 reference pairs were saved for the SFT adapter. A scoring-only GPU pass is needed.
- **On-policy states.** Assembly restricts behavior to the four numerical methods (`training_dataset.py` line 170).
- **Common random numbers across candidates** in branch noise (world lines 307 to 314).
- **Sampling at deployment.** Decoding is greedy only.

## 10. Document disagreements and supersessions

- The November 2 agenda proposed 16 initial responses, up to 64 additional interventions and seven policies. The executed protocol used 8 initial, 24 additional cost units and eight policies. The handoff (lines 19 and 23) says the agenda counts are superseded.
- The exposure guide (November 7, lines 102 to 105) anticipated "at most 128 DPO steps". The executed count is 119.
- The engine guide (November 7) says to pass `gpu09_numeric_recipe()`. The scientific run used `screen_numeric_recipe()` with unpadded singleton scoring, per the November 8 shape guide and contract. Newer documents win. Optimizer settings are the same in both.
- The plan flag `one_pass_dpo: false` could be misread as repeated pairs. It reflects 568 unexposed strict pairs. Maximum exposure of any pair is 1.
- `SR/llm_intervention_cpu_owned_supervision.py` is CPU child-process supervision. It contains no training supervision logic.
- I found no disagreement between documents on the executed recipe numbers.

## 11. Implications for the experimental agenda

1. **Zero-GPU diagnostics come first and are mostly already answerable.** The saved training evidence shows the constant action on the training states themselves, no prequential learning beyond a state-blind baseline, and a DPO stage that did not fit its own pairs. A new training run is not needed to learn this. Write it up as the diagnosis.
2. **Make a behavioral gate part of every future training freeze.** Before any deployment: action histogram on training and held-out training-world states, accuracy against the teacher versus the modal-label baseline, held-out or prequential loss versus a state-blind baseline, and for preference training the held-out pair accuracy versus 50 percent.
3. **Do not scale the measured-preference DPO stage as it stands.** It consumed 9,216 of 10,240 collection cost units, exposed 119 single-pair updates, and its labels look majority-noise by two independent estimates. Candidates that the code nearly supports: listwise soft targets from target-risk scores, all strict pairs with several epochs, 16 replicates with common random numbers, or an exact expected one-step gain computed from the private linear-Gaussian world at zero response cost to measure label noise directly.
4. **Repair the supervised target before judging the language model.** One competent teacher or explicit teacher identity, no uniform-random labels, no terminal stop examples, randomized or removed alphabetical tie-breaks, several shuffled epochs, held-out worlds.
5. **Use the free data.** 697 saved decision envelopes plus cheap target-risk relabeling give roughly five times more states with full score vectors and no new environment responses. They are from the same 32 worlds, so world-level cross-validation is required.
6. **Train the numerical ranker to convergence as the first learnability probe.** It is CPU-only and takes seconds. If a converged ranker on the full numeric state cannot learn the target-risk mapping across held-out worlds, the same data will not teach a 0.6B language model through JSON text. The current "learned numerical ranker" arm is not a strong learned control and should not be described as one.
7. **Test the token-factorization hypothesis cheaply.** Rename or permute action identifiers (single-token labels, different alphabetical order) and check whether the collapsed action follows the label marginal and tie-break order. This separates a supervision artifact from a property of X = -1.
8. **Cover on-policy states.** Teacher labels cost no environment responses, so relabeling states visited by the language policy is cheap in experiment units. Constant-action histories are currently outside the training distribution.
9. **Report the trained arms honestly in the paper.** The supervised and DPO arms of the 24-system screen test this specific recipe. They do not test whether a language model can condition on an SCM posterior.

## 12. Open questions

- Do the frozen and supervised adapters change command rankings when posterior uncertainty is varied with budget and history length held fixed? Full-menu scores exist only for the final adapter. This needs a scoring-only GPU pass.
- What fraction of the strict preference labels is noise? Exact expected gains from the private world descriptors, or 16 replicates, would answer it.
- Can any learner recover the target-risk action from the saved public state with 128 to 825 examples? The converged ranker answers this.
- Can Qwen3-0.6B read posterior covariances out of full-precision JSON at all? A probe with a derived, low-precision uncertainty summary per action would test representation separately from supervision.
- Was the supervised arm's early stopping tied to low remaining budget? The deployment logs are with the results-analysis and deployment-forensics agents.
- How much did Adam momentum carried from SFT into the first DPO steps matter? The optimizer state files exist but were not loaded here.
