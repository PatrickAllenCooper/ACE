# Training forensics: saved supervision and training outputs behind the constant-action LLM policies

Label: `training-forensics`. Prepared 2026-11-12. Read-only offline analysis with `/usr/bin/python3` (numpy, pandas, scipy). No model weights, tokenizer, tensors or `archive.tar` files were opened. No repository file was changed. All scripts and raw outputs are under
`/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training-forensics/` (see section 12).

Status words used below: **verified** = computed from or read in saved files; **inferred** = my interpretation or a toy-model result, not something the saved evidence proves.

## 0. Bottom line

1. **The constant action was already present on the training states and is saved in the training evidence.** Greedy generations saved at the two prescribed checkpoints, on the 128 nonterminal training prompts: supervised (SFT) checkpoint `set_x_neg` 108, `observe_all` 19, `set_x_pos` 1. Final (DPO) checkpoint `set_x_neg` 123, `set_x_pos` 5. Agreement with the teacher label: SFT 36/128, DPO 32/128. A constant `set_x_neg` gets 32/128. This is therefore not a training-to-development shift. The trained policies do not fit their own training supervision in a state-dependent way.
2. **Supervised labels are only mildly imbalanced at the command level** (`set_x_neg` 32/128 = 25.0% of nonterminal targets, `set_x_pos` 29, `observe_all` 24), but `set_x_neg` is the plurality at every token-level decision of the command (`set` 81.3%, `_x` 65.4%, `_neg` 45.7%) and is 50% of labels at the initial state. Greedy token-by-token decoding of a weakly state-conditioned model lands on that path.
3. **The 119 DPO pairs are not dominated by `set_x_neg`** (preferred in 14, rejected in 9; `set_x_pos` is the most preferred with 23). **Preference labels are not state-independent**: the best constant ranking predicts 56.8% of strict labels in-sample and 48.9% leave-one-world-out.
4. **Preference labels are very noisy.** The tie rule (|mean| ≤ 2 SE with 4 replicates) is a t-test with 3 degrees of freedom at α ≈ 0.139. Under no real difference it would still declare about 476 to 499 of the 3,584 comparisons strict. 687 were declared strict. Reliability of a 4-replicate action mean is about 0.15. On states where the target-risk planner's own scores are saved, strict labels agree with the planner's ordering in 123/179 = 68.7%, so roughly one strict label in three points the other way.
5. **The DPO stage moved state-independent command biases, not state-conditional preferences.** 79% of the variance of the DPO-stage log-likelihood change is explained by command identity alone. Both preferred and rejected commands fell on average (−4.69 and −6.68 nats). A model with no state input at all (8 token logits) reproduces the 119 saved online DPO margins with correlation 0.80 and the final per-command shifts with correlation 0.93. All 119 DPO steps were gradient-clipped to norm 1.0, and with β = 0.1 margins moved by up to ±19 nats.

## 1. Where things are and what the schema is (task item 1)

### 1.1 Directory map (verified)

| What | Path |
| --- | --- |
| Collection batch root | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/session/worker/task/collection/` |
| Per world | `world-000 … world-031/` with `plan.json`, `terminal.json`, `private-world.json`, `trajectory.jsonl`, `decision-NNN.{choice,envelope}.json` |
| Per snapshot (5 per world, 160 total) | `world-NNN/snapshot-00k/{record,teacher,selection,receipt,language,alignment,numeric-labels,envelope}.json`, `numeric-features.npy` |
| Per nonterminal snapshot (128) | `snapshot-00k/feedback/terminal.json` (complete aggregated feedback), plus raw branch/scoring journals |
| Batch files | `plan.json`, `inventory.json`, `terminal.json`, `supplied-proposal.json` in the collection root |
| Exposure (which examples and pairs are trained) | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/cpu/preparation/session/worker/artifacts/exposure.json` |
| Prepared tokens | same directory, `llm-prepared.json` (prompt token ids and command token ids for all 160 records) |
| Tokenization stage copy | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/terminal02/remote/launch/preparation/session/worker/artifacts/` (`exposure.json`, `llm-prepared.json`, `numerical-plan.json` are byte-identical to the training-gpu copies, SHA256 prefixes `43c48df2f581e2d3`, `edd8ffc6ccbbf487`, `a95f840bf8637c06`) |
| Training outputs | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/` |
| Per step (279) | `fit/training/step-NNNNN.{start,loss,gradients,update-intent,optimizer-attempt,update-returned,complete}.json` |
| Saved scores | `fit/training/reference-scores.json` (SFT checkpoint, 119 pairs × 2), `fit/final-before-reload.json` and `fit/final-reload-scores.json` (final checkpoint, 1,184 command scores), `fit/sft-reload-scores.json` |
| Saved generations | `fit/sft-generation-NN.json` and `fit/final-generation-NN.json`, ordinals 0 to 160 (160 records plus one revisit of record 0) |
| Recipe | `fit/arithmetic.json`, `fit/summary.json`, `fit/training/plan.json` |
| Acceptance reviews | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/acceptance01/review.md`, `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/acceptance02/review.md` |

File counts differ from the task brief. The local collection directory holds 30,596 files (the acceptance review states 30,418 in the original subtree), not about 12,400. The training directory holds 4,926 files (10 of them `.pt`), not about 235.

Source check: SHA256 of nine current files under `/Users/pat/code/ACE/scripts/research/` (`llm_intervention_replicated_feedback.py`, `_numerical_policy.py`, `_target_risk.py`, `_supervision.py`, `_training_dataset.py`, `_screen_protocol.py`, `_training_collection.py`, `_scm.py`, `_task.py`) match the pins recorded in the collection `plan.json`. `git diff --stat a9346022…  HEAD` is empty for the training engine, exposure, language policy, learning, canonical scoring and grammar modules. Code line references below therefore describe the executed source.

### 1.2 Design that produced the supervision (verified)

- 32 training worlds (`world_id` 202611070000 to 202611070031): 4 regimes (`matched_low`, `matched_high`, `shifted_low`, `shifted_high`; edge mean 0 or 1.5, known noise variance 0.25 or 1.0) × graph blocks (chain 12, fork 10, collider 10 worlds) × 2 cost menus (16 uniform, 16 heterogeneous). Source: `llm_intervention_screen_protocol.py:27-33,156-167`, `supplied-proposal.json`.
- Each world runs one 32-unit trajectory (8 initial `observe_all`, then 24 units chosen by a behavior policy: 8 worlds each for `uniform_random`, `menu_coverage`, `local_information`, `target_risk`).
- Snapshots are taken at acquisition-cost milestones 0, 6, 12, 18, 24. At snapshot index `si` the teacher is `METHODS[(teacher_base+si) % 4]` (`llm_intervention_screen_protocol.py:124-135`). Each snapshot index therefore has 8 worlds per teacher. Snapshot 4 has an empty candidate set and is a budget-terminal stop example.
- At snapshots 0 to 3 all 8 menu actions are candidates. Each candidate is executed once in each of 4 replicate blocks (`r0..r3`) from a fork of the same pre-action SCM learner, and the updated learner is scored against **noise-disabled structural truths** on the 8 public queries (`structural_scores: true`). I verified that scoring truths, base predictions and base losses are identical across the 4 replicates in all 128 records. The only replicate-to-replicate randomness is the single noisy response of the candidate action.
- Collection accounting (`terminal.json`): 10,240 cost units, 9,145 accepted responses. Trajectory 1,024 units / 953 responses, branch 5,120 / 4,096, scoring 4,096 / 4,096. Preference feedback therefore consumed 9,216 of 10,240 units (90%).

### 1.3 One supervised example (verified, `record.json`, schema `ace-matched-intervention-supervision-v1`, `llm_intervention_supervision.py:20-74`)

Keys: `version`, `scope` (`policy_train`), `world_id`, `envelope`, `supervised_action_id`, `preferences`, `supervision_receipt_sha256`.

- `envelope.state`: `domain_notes`, `graph` (X, Y, Z with parents and bounds), `history` (every paid response with action id, cost, targets and measured X, Y, Z at full float precision), `menu` (8 actions with costs), `objective`, `remaining_units`, `requests`, `returned_responses`, `scm` (per node: `eligible_rows`, `noise_variance`, `parameter_mean`, `parameter_covariance`, `terms`), `spent_units`, `terminal`.
- `envelope.task`: target `Z`, 8 queries `q0..q7` (observe, do(X = −1, 0, 1), do(Y = −1, 0, 1), do(X = −1, Y = 1)), weights 0.125 each, scale 2.0, `normalized_mse_tolerance` 0.04.
- `supervised_action_id`: the teacher's action id, or `null` for stop. The teacher and its metadata are in `teacher.json` (`method`, `decision`, `metadata`; for `target_risk` the metadata holds `risk_before`, `risk_after`, `risk_reduction_per_cost` and `monte_carlo_standard_errors` for all 8 actions; for `local_information` it holds `scores`). **The teacher identity is not in the prompt.**
- Language view (`language.json`): `prompt` = fixed instruction paragraph + canonical JSON of the envelope + `Decision:`; `supervised_completion` = `{"action_id":"…"}` or `{"stop":true}`; `preference_completions` = list of (chosen, rejected) command strings for strict pairs only.

### 1.4 One preference comparison (verified)

In `record.json`: `{left, right, tie, chosen, rejected}`. All C(8,2) = 28 unordered pairs per nonterminal record are retained. The numbers behind each are in `snapshot-00k/feedback/terminal.json → feedback.pairs[i]`: `replicate_differences` (4 values), `mean_gain_difference`, `paired_standard_error`, `tie_threshold`, `tie`, `chosen`, `rejected`. The same file holds `per_replicate_gains_per_cost[action]` (4 values), `mean_gains_per_cost`, `base_losses`, `branch_losses`, `scoring_truths`.

Gain definition (`llm_intervention_replicated_feedback.py:112-118,152-171`): for candidate c, replicate r, query q, `g = ((base−truth)/scale)² − ((branch−truth)/scale)²`, divided by the action cost; weighted over queries, then averaged over the 4 replicates. For a pair, `SE = sample_std(G_left − G_right, ddof=1)/sqrt(4)`. **Tie iff |mean difference| ≤ 0 + 2.0 × SE.** Otherwise the larger mean is `chosen`. Verified on all 3,584 rows: the flag equals this rule, threshold equals 2 × SE, and SE recomputed from the replicate differences matches to 1.4e-17.

### 1.5 Exposure rule (verified, `llm_intervention_training_exposure.py:75-124`, `exposure.json`)

- One SFT step for **every** record in roster order (world 0 to 31, snapshot 0 to 4). Then one DPO step for each record that has at least one strict pair, using the strict pair with the smallest SHA-256 of `[domain, record_id, sorted command pair]`. Ties are never trained on.
- `exposure.json → counts`: records 160, retained_pairs 3,584, strict_pairs 687, tied_pairs 2,897, selected_pairs 119, unselected_pairs 568, records_without_strict_pairs 41 (32 terminal + 9 nonterminal), sft_steps 160, dpo_steps 119.
- 119/3,584 = 3.3% of collected comparisons were used for DPO. That is 77.4 feedback cost units per used pair (9,216/119).

### 1.6 Training recipe (verified, `fit/arithmetic.json`, `fit/summary.json`, `fit/training/plan.json`)

Qwen3-0.6B, rank 8, alpha 16 LoRA on q/v projections of 28 layers (112 matrices, 1,146,880 parameters), base bfloat16, adapters float32. AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0, gradient clip norm 1.0, seed 109, **batch size 1 (SFT) or one pair (DPO), single pass, fixed roster order, no shuffling**, optimizer state continuous across SFT and DPO, DPO β = 0.1 against a fixed post-SFT reference. 279 updates in 291.5 s of training wall time. Whole job 904 s on one A100 20 GB MIG slice (job 35420219).

- SFT loss = mean negative log-likelihood per action token (`llm_intervention_canonical_scoring.py:99-102`). Commands have 9 tokens (`set_x_*`, `set_y_*`), 8 (`observe_all`, `set_xy`) or 6 (stop), including EOS.
- DPO loss = `−logsigmoid(β·[(lp_c − ref_c) − (lp_r − ref_r)])` on summed sequence log-probabilities (`llm_intervention_learning.py:94-103`).
- Deployment and the saved generations use greedy, grammar-constrained, token-by-token decoding (`llm_intervention_language_policy.py:132-184`, `do_sample=False, num_beams=1`, `prefix_allowed_tokens_fn`).

### 1.7 Command tokenization (verified, `llm-prepared.json → training_plan.records[0].commands`)

```
{"action_id":"observe_all"} [4913,1311,842,3252, 48033,5705,        9207,151645]
{"action_id":"set_x_neg"}   [4913,1311,842,3252, 746,3212,28209,    9207,151645]
{"action_id":"set_x_pos"}   [4913,1311,842,3252, 746,3212,6479,     9207,151645]
{"action_id":"set_x_zero"}  [4913,1311,842,3252, 746,3212,19359,    9207,151645]
{"action_id":"set_xy"}      [4913,1311,842,3252, 746,40920,         9207,151645]
{"action_id":"set_y_neg"}   [4913,1311,842,3252, 746,4178,28209,    9207,151645]
{"action_id":"set_y_pos"}   [4913,1311,842,3252, 746,4178,6479,     9207,151645]
{"action_id":"set_y_zero"}  [4913,1311,842,3252, 746,4178,19359,    9207,151645]
{"stop":true}               [4913,9495,788,1866,92,151645]
```

The nine commands form a trie with four decisions: `action` vs `stop`; `observe` vs `set`; `_x` vs `_y` vs `_xy`; `_neg` vs `_pos` vs `_zero`. **The suffix tokens are shared between the X and Y commands.** Everything else is deterministic.

### 1.8 Prompts (verified)

- 160 distinct prompt token sequences. Length 1,998 to 4,199 tokens (median 3,005; by snapshot index medians 2,017, 2,516, 3,018, 3,510, 4,012). `max_tokens` is 8,192, so nothing is truncated.
- The first 204 tokens and the **last 356 tokens are identical in all 160 prompts** (instruction; task block, `Decision:` and chat template). All state information sits in between.
- 18.2% of prompt characters are digits. A prompt holds 71 to 172 decimal numbers at about 17 significant digits.
- Nonterminal training states have `remaining_units` only in {5, 6, 11, 12, 17, 18, 24} (counts 2, 30, 3, 29, 7, 25, 32). All 32 stop examples have `remaining_units` 0 and `terminal: true`, where the grammar admits only `{"stop":true}`.

## 2. Supervised targets (task item 2)

Script `sft_targets.py`, output `sft_targets.out`. Independent recount by `grep` over the 160 raw `record.json` files gives the same numbers.

### 2.1 Totals (verified)

160 supervised examples, all used once for the 160 SFT updates in roster order (`step-00000 … step-00159` record ids equal `inventory.json → record_roster`; `plan.json → one_pass_sft: true`).

| Target | n | share of 160 | share of 128 nonterminal |
| --- | ---: | ---: | ---: |
| observe_all | 24 | 15.0% | 18.75% |
| **set_x_neg** | **32** | **20.0%** | **25.0%** |
| set_x_zero | 7 | 4.4% | 5.5% |
| set_x_pos | 29 | 18.1% | 22.7% |
| set_y_neg | 10 | 6.25% | 7.8% |
| set_y_zero | 4 | 2.5% | 3.1% |
| set_y_pos | 10 | 6.25% | 7.8% |
| set_xy | 12 | 7.5% | 9.4% |
| **stop** | **32** | **20.0%** | n/a |

All 32 stop targets are the snapshot-4 records. No stop label exists at any state with budget left.

### 2.2 By teacher (nonterminal, 32 each)

| Teacher | observe_all | set_x_neg | set_x_zero | set_x_pos | set_y_neg | set_y_zero | set_y_pos | set_xy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_random | 5 | 3 | 5 | 0 | 7 | 3 | 5 | 4 |
| menu_coverage | 0 | 12 | 2 | 12 | 1 | 1 | 3 | 1 |
| local_information | 11 | 9 | 0 | 8 | 2 | 0 | 2 | 0 |
| target_risk | 8 | 8 | 0 | 9 | 0 | 0 | 0 | 7 |

Half of the nonterminal labels (64) come from teachers that do not use the SCM posterior at all. 32 are uniformly random. 32 are menu coverage.

### 2.3 By snapshot index (nonterminal, 32 each)

| Snapshot | observe_all | set_x_neg | set_x_zero | set_x_pos | set_y_neg | set_y_zero | set_y_pos | set_xy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (initial state) | 3 | **16** | 3 | 2 | 3 | 2 | 0 | 3 |
| 1 | 5 | 7 | 1 | **13** | 1 | 1 | 3 | 1 |
| 2 | 9 | 4 | 2 | 8 | 3 | 0 | 4 | 2 |
| 3 | 7 | 5 | 1 | 6 | 3 | 1 | 3 | 6 |

At snapshot 0, `set_x_neg` comes from menu_coverage 8/8 (deterministic: least-used action with lexical tie-break, `llm_intervention_numerical_policy.py:62-66`; I reproduced all 32 menu-coverage labels from history counts), local_information 4/8 and target_risk 4/8. **At snapshots 1 to 3 the labels favor `set_x_pos` (27) and `observe_all` (21) over `set_x_neg` (16).**

### 2.4 By graph and cost menu (all 160)

| Graph, cost | observe_all | set_x_neg | set_x_zero | set_x_pos | set_y_neg | set_y_zero | set_y_pos | set_xy | stop | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain, heterogeneous | 2 | 6 | 1 | 8 | 3 | 1 | 2 | 1 | 6 | 30 |
| chain, uniform | 4 | 5 | 2 | 6 | 2 | 0 | 2 | 3 | 6 | 30 |
| collider, heterogeneous | 6 | 4 | 1 | 4 | 2 | 1 | 1 | 1 | 5 | 25 |
| collider, uniform | 3 | 0 | 1 | 4 | 3 | 1 | 4 | 4 | 5 | 25 |
| fork, heterogeneous | 5 | 9 | 1 | 3 | 0 | 1 | 0 | 1 | 5 | 25 |
| fork, uniform | 4 | 8 | 1 | 4 | 0 | 0 | 1 | 2 | 5 | 25 |

By graph: chain `set_x_neg` 11/60, fork 17/50, collider 4/50. By cost menu: heterogeneous 19/80, uniform 13/80. By regime: 8, 6, 10, 8 of 40 (matched_high, matched_low, shifted_high, shifted_low).

### 2.5 Majority-class accuracy (verified)

| Rule | All 160 | Nonterminal 128 |
| --- | ---: | ---: |
| Single constant | 32/160 = 20.0% (stop and set_x_neg tie at 32) | 32/128 = 25.0% (set_x_neg) |
| Modal within graph | 41/160 | 40/128 |
| Modal within cost menu | 35/160 | 33/128 |
| Modal within graph × cost (in-sample) | 42/160 = 26.3% | 41/128 = 32.0% |
| Modal within graph × cost (leave-one-world-out) | 36/160 = 22.5% | 29/128 = 22.7% |
| Modal within snapshot index | 77/160 = 48.1% (stop is determined) | 45/128 = 35.2% |
| Modal within teacher (not visible to the model) | 40/160 | 39/128 |

Modal targets within graph × cost cells (nonterminal): chain-het `set_x_pos` 8/24, chain-uni `set_x_pos` 6/24, collider-het `observe_all` 6/20, collider-uni `set_xy` 4/20 (three-way tie at 4), fork-het `set_x_neg` 9/20, fork-uni `set_x_neg` 8/20.

### 2.6 Token-level marginals of the labels (verified)

| Decision | Label counts |
| --- | --- |
| stop vs action | 32 vs 128 (perfectly determined by `terminal`) |
| observe vs set | 24 vs 104 (set 81.3%) |
| `_x` / `_y` / `_xy` after `set` | 68 / 24 / 12 (`_x` 65.4%) |
| `_neg` / `_pos` / `_zero` | 42 / 39 / 11 (`_neg` 45.7%; after `_x`: 32 / 29 / 7; after `_y`: 10 / 10 / 4) |

Per-node plurality path: action → set → `_x` → `_neg` = `set_x_neg`. By snapshot the suffix counts (neg/pos/zero) are 19/2/5, 8/16/2, 7/12/2, 8/9/2. So `_pos` is the plurality suffix at snapshots 1 to 3.

## 3. Preference comparisons (task item 3)

Script `prefs_analysis.py`, output `prefs_analysis.out`.

### 3.1 Totals (verified)

3,584 comparisons = 128 nonterminal records × 28 pairs. **Strict 687 (19.17%), ties 2,897 (80.83%).** Strict direction: chosen is `left` 348, `right` 339. Strict pairs per record: mean 5.37, median 4.5, max 18; 9 nonterminal records have none.

Strict share by stratum: snapshot 0 242/896 (27.0%), snapshot 1 161 (18.0%), snapshot 2 144 (16.1%), snapshot 3 140 (15.6%). Chain 239/1,344, collider 238/1,120, fork 210/1,120. Heterogeneous 367/1,792, uniform 320/1,792.

### 3.2 Ordered (preferred → rejected) counts, all 687 strict pairs

Rows preferred, columns rejected.

| preferred \ rejected | obs | x_neg | x_zero | x_pos | y_neg | y_zero | y_pos | xy | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| observe_all | 0 | 7 | 10 | 4 | 10 | 6 | 10 | 6 | 53 |
| set_x_neg | 8 | 0 | 12 | 14 | 18 | 10 | 13 | 13 | 88 |
| set_x_zero | 9 | 11 | 0 | 8 | 8 | 15 | 14 | 14 | 79 |
| set_x_pos | 16 | 15 | 13 | 0 | 16 | 20 | 23 | 20 | 123 |
| set_y_neg | 9 | 10 | 18 | 11 | 0 | 9 | 14 | 15 | 86 |
| set_y_zero | 11 | 15 | 15 | 14 | 9 | 0 | 9 | 17 | 90 |
| set_y_pos | 8 | 11 | 15 | 11 | 10 | 7 | 0 | 13 | 75 |
| set_xy | 10 | 5 | 19 | 16 | 16 | 12 | 15 | 0 | 93 |

| Action | preferred | rejected | win rate | share of preferred side |
| --- | ---: | ---: | ---: | ---: |
| observe_all | 53 | 71 | 0.427 | 7.7% |
| set_x_neg | 88 | 74 | 0.543 | 12.8% |
| set_x_zero | 79 | 102 | 0.436 | 11.5% |
| set_x_pos | 123 | 78 | 0.612 | 17.9% |
| set_y_neg | 86 | 87 | 0.497 | 12.5% |
| set_y_zero | 90 | 79 | 0.533 | 13.1% |
| set_y_pos | 75 | 98 | 0.434 | 10.9% |
| set_xy | 93 | 98 | 0.487 | 13.5% |

Only `set_x_pos` deviates from an even split at conventional levels (123 vs 78, binomial p = 0.0018). `set_x_neg` does not (88 vs 74, p = 0.31).

### 3.3 The 119 strict pairs used for DPO updates (verified)

| preferred \ rejected | obs | x_neg | x_zero | x_pos | y_neg | y_zero | y_pos | xy | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| observe_all | 0 | 1 | 0 | 3 | 0 | 0 | 1 | 1 | 6 |
| set_x_neg | 0 | 0 | 4 | 2 | 5 | 0 | 2 | 1 | 14 |
| set_x_zero | 3 | 0 | 0 | 2 | 0 | 3 | 2 | 3 | 13 |
| set_x_pos | 5 | 3 | 2 | 0 | 1 | 1 | 7 | 4 | 23 |
| set_y_neg | 2 | 0 | 5 | 2 | 0 | 5 | 2 | 3 | 19 |
| set_y_zero | 2 | 2 | 3 | 3 | 1 | 0 | 1 | 4 | 16 |
| set_y_pos | 1 | 2 | 0 | 1 | 3 | 1 | 0 | 0 | 8 |
| set_xy | 0 | 1 | 5 | 1 | 4 | 3 | 6 | 0 | 20 |

| Action | preferred | rejected | net | share of preferred side | share of rejected side |
| --- | ---: | ---: | ---: | ---: | ---: |
| observe_all | 6 | 13 | −7 | 5.0% | 10.9% |
| **set_x_neg** | **14** | **9** | **+5** | **11.8%** | 7.6% |
| set_x_zero | 13 | 19 | −6 | 10.9% | 16.0% |
| set_x_pos | 23 | 14 | +9 | 19.3% | 11.8% |
| set_y_neg | 19 | 14 | +5 | 16.0% | 11.8% |
| set_y_zero | 16 | 13 | +3 | 13.4% | 10.9% |
| set_y_pos | 8 | 21 | −13 | 6.7% | 17.6% |
| set_xy | 20 | 16 | +4 | 16.8% | 13.4% |

Uniform expectation per side is 14.9 pairs. **`set_x_neg` is not overrepresented on the preferred side** (14, slightly below uniform). `set_x_pos` is the most preferred. Head to head among the 119: `set_x_neg` over `set_x_pos` 2, `set_x_pos` over `set_x_neg` 3. Selected pairs by snapshot 31/30/29/29, by graph chain 44, collider 38, fork 37, by cost 60 heterogeneous and 59 uniform.

Token-level tallies of the same 119 pairs (chosen side, rejected side, net):

| Token decision | Tallies |
| --- | --- |
| observe / set | observe (6, 13, −7); set (113, 106, +7) |
| `_x` / `_y` / `_xy` | `_x` (50, 42, +8); `_y` (43, 48, −5); `_xy` (20, 16, +4) |
| `_neg` / `_pos` / `_zero` | `_neg` (33, 23, +10); `_pos` (31, 35, −4); `_zero` (29, 32, −3) |

None is individually significant (binomial p ≈ 0.17, 0.47, 0.23). Every net sign points along `set → _x → _neg`. In the full 687 strict set the suffix nets are `_pos` +22, `_neg` +13, `_zero` −12, and `_x` +36, `_y` −13, `_xy` −5, observe −18. The `_neg` over `_pos` tilt is therefore specific to the hash-selected 119 subsample.

### 3.4 Margins (verified)

| Quantity | 5% | 25% | 50% | 75% | 95% | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs mean gain difference, all | 8.5e-5 | 6.7e-4 | 2.4e-3 | 7.8e-3 | 4.1e-2 | 0.382 |
| abs mean gain difference, strict | 4.3e-4 | 3.0e-3 | 9.8e-3 | 2.8e-2 | 0.150 | 0.382 |
| abs mean gain difference, selected 119 | 3.5e-4 | 1.8e-3 | 5.0e-3 | 1.7e-2 | 7.9e-2 | 0.181 |
| paired SE, all | 2.1e-4 | 1.2e-3 | 3.2e-3 | 8.1e-3 | 3.4e-2 | 0.117 |
| t = abs mean / SE, all | 0.085 | 0.39 | 0.83 | 1.69 | 4.49 | 134.8 |
| t, strict | 2.08 | 2.42 | 3.08 | 4.60 | 8.11 | 134.8 |
| t, selected 119 | 2.09 | 2.36 | 3.08 | 4.40 | 7.25 | 134.8 |

Half of the pairs used for DPO have t below 3.08 on 3 degrees of freedom (two-sided p above 0.054). Cost structure: in the heterogeneous menu, 218 strict pairs compare actions of unequal cost and the cheaper one is preferred in 134 (61.5%).

## 4. State dependence of preference labels (task item 4)

Script `state_dependence_and_noise.py`, output `state_dependence_and_noise.out`.

### 4.1 Per unordered pair, share of the 128 states won by each side (verified)

| Pair | tie | first wins | second wins | majority share of strict | binomial p |
| --- | ---: | ---: | ---: | ---: | ---: |
| observe_all vs set_x_neg | 113 | 7 | 8 | 0.533 | 1.00 |
| observe_all vs set_x_zero | 109 | 10 | 9 | 0.526 | 1.00 |
| observe_all vs set_x_pos | 108 | 4 | 16 | 0.800 | 0.012 |
| observe_all vs set_y_neg | 109 | 10 | 9 | 0.526 | 1.00 |
| observe_all vs set_y_zero | 111 | 6 | 11 | 0.647 | 0.33 |
| observe_all vs set_y_pos | 110 | 10 | 8 | 0.556 | 0.81 |
| observe_all vs set_xy | 112 | 6 | 10 | 0.625 | 0.45 |
| set_x_neg vs set_x_zero | 105 | 12 | 11 | 0.522 | 1.00 |
| set_x_neg vs set_x_pos | 99 | 14 | 15 | 0.517 | 1.00 |
| set_x_neg vs set_y_neg | 100 | 18 | 10 | 0.643 | 0.18 |
| set_x_neg vs set_y_zero | 103 | 10 | 15 | 0.600 | 0.42 |
| set_x_neg vs set_y_pos | 104 | 13 | 11 | 0.542 | 0.84 |
| set_x_neg vs set_xy | 110 | 13 | 5 | 0.722 | 0.096 |
| set_x_zero vs set_x_pos | 107 | 8 | 13 | 0.619 | 0.38 |
| set_x_zero vs set_y_neg | 102 | 8 | 18 | 0.692 | 0.076 |
| set_x_zero vs set_y_zero | 98 | 15 | 15 | 0.500 | 1.00 |
| set_x_zero vs set_y_pos | 99 | 14 | 15 | 0.517 | 1.00 |
| set_x_zero vs set_xy | 95 | 14 | 19 | 0.576 | 0.49 |
| set_x_pos vs set_y_neg | 101 | 16 | 11 | 0.593 | 0.44 |
| set_x_pos vs set_y_zero | 94 | 20 | 14 | 0.588 | 0.39 |
| set_x_pos vs set_y_pos | 94 | 23 | 11 | 0.676 | 0.058 |
| set_x_pos vs set_xy | 92 | 20 | 16 | 0.556 | 0.62 |
| set_y_neg vs set_y_zero | 110 | 9 | 9 | 0.500 | 1.00 |
| set_y_neg vs set_y_pos | 104 | 14 | 10 | 0.583 | 0.54 |
| set_y_neg vs set_xy | 97 | 15 | 16 | 0.516 | 1.00 |
| set_y_zero vs set_y_pos | 112 | 9 | 7 | 0.562 | 0.80 |
| set_y_zero vs set_xy | 99 | 17 | 12 | 0.586 | 0.46 |
| set_y_pos vs set_xy | 100 | 13 | 15 | 0.536 | 0.85 |

Each side wins 3% to 18% of states and 72% to 88% are ties. One pair of 28 deviates from an even split at uncorrected p < 0.05, none after Bonferroni.

### 4.2 How well a constant ranking predicts strict labels (verified)

| Predictor | Accuracy on strict labels |
| --- | ---: |
| Per-pair majority direction (28 free parameters; ceiling for any state-independent pairwise rule, in-sample) | 400/687 = 58.2% |
| Bradley-Terry constant ranking, in-sample | 390/687 = 56.8% |
| Win-rate ranking, in-sample | 389/687 = 56.6% |
| Same in-sample fit when label directions are coin flips (300 draws) | mean 54.1%, 95th percentile 56.3% |
| **Bradley-Terry, leave-one-world-out** | **336/687 = 48.9%** |
| Win-rate ranking, leave-one-world-out | 337/687 = 49.1% |
| Ranking per cost menu: in-sample / leave-one-world-out | 61.9% / 51.8% |
| Ranking per graph × cost: in-sample / leave-one-world-out | 69.7% / 52.3% |
| Ranking per snapshot index: in-sample / leave-one-world-out | 63.8% / 51.7% |
| On the 119 selected pairs: BT fitted on them in-sample / leave-one-world-out | 78/119 = 65.5% (coin-flip null mean 59.8%, 95th percentile 65.6%) / 62/119 = 52.1% |

Bradley-Terry log-strengths on all strict pairs: `set_x_pos` +0.416, `set_x_neg` +0.179, `set_y_zero` +0.149, `set_y_neg` −0.002, `set_xy` −0.030, `set_y_pos` −0.222, `set_x_zero` −0.231, `observe_all` −0.259.

**The supervision does not reward a constant policy.** A constant ranking is at chance out of sample. Note that a per-cell ranking fitted with many parameters looks good in-sample (69.7%) and collapses to 52.3% out of sample, which is the signature of fitting noise.

### 4.3 What a constant policy earns under the measured feedback (verified)

Mean measured gain per cost over the 128 states: always `set_x_pos` 0.00891, always `set_x_neg` 0.00854, `set_x_zero` 0.00669, `set_y_neg` 0.00597, `set_y_zero` 0.00540, `set_xy` 0.00519, `observe_all` 0.00488, `set_y_pos` 0.00480; uniform random 0.00630. Standard errors across states are 0.0016 to 0.0036, so these differences are within noise. Mean within-state rank of every constant action is 4.30 to 4.75 (4.5 is chance). 78% of total summed gain comes from the 32 snapshot-0 states.

A cross-validated state-dependent picker (argmax of 2 replicates, scored on the other 2, averaged over 6 splits) earns 0.01267, against 0.00854 and 0.00891 for the two best constants. So the measured feedback contains some state-dependent signal, concentrated at the initial state.

Measured-best action per record (argmax of the 4-replicate mean): observe_all 12, set_x_neg 25, set_x_zero 12, set_x_pos 23, set_y_neg 15, set_y_zero 12, set_y_pos 16, set_xy 13. The supervised (teacher) action is the measured-best in 25/128 (chance 16). Its mean rank by measured gain is 3.98 (chance 4.5); by teacher: local_information 3.66, uniform_random 3.84, menu_coverage 4.03, **target_risk 4.38**.

## 5. Label noise (task item 5)

### 5.1 The tie rule as a test (verified arithmetic)

With 4 replicates, "strict iff |mean| > 2 SE" is a paired t-test on 3 degrees of freedom at two-sided α = P(|t₃| > 2) = 0.1393.

| Reference | Expected strict of 3,584 | Observed | Excess |
| --- | ---: | ---: | ---: |
| Gaussian null | 499.3 | 687 | 187.7 (27.3% of strict labels, 5.2% of comparisons) |
| Sign-flip randomization null, conditional on the observed replicate magnitudes | 476.0 | 687 | 211.0 (30.7% of strict labels, 5.9% of comparisons) |
| All four replicates agree in sign (symmetric null 12.5%) | 448 | 658 | 210 |

Storey estimates of the null proportion are 0.85 to 0.93 (λ from 0.9 to 0.5), which gives an estimated false-discovery share among strict labels of 62% to 68%.

| Threshold | Comparisons passing | Null expectation | Of the 119 DPO pairs |
| --- | ---: | ---: | ---: |
| Study rule, t > 2 | 687 (19.2%) | 476 to 499 | 119 |
| p < 0.05 (t > 3.182) | 329 (9.2%) | 179 | 56 |
| p < 0.01 (t > 5.841) | 95 (2.65%) | 36 | 13 |
| p < 0.001 (t > 12.92) | 12 (0.33%) | 3.6 | 2 |
| Benjamini-Hochberg q = 0.10 | 1 | n/a | 1 |
| Bonferroni 0.05 | 1 | n/a | 1 |

**Share distinguishable from noise:** about 5% to 6% of all comparisons are in excess of what the rule declares under no difference. Roughly two thirds of the 687 strict labels are at noise level by this accounting.

By snapshot (observed strict / sign-flip expectation / excess share of strict labels): 0: 242 / 117.0 / 51.7%; 1: 161 / 118.2 / 26.6%; 2: 144 / 121.6 / 15.5%; 3: 140 / 119.1 / 14.9%.

### 5.2 Replicate variability against margins (verified)

- Per comparison, SD of the 4 replicate differences divided by |mean difference|: median 2.41 for all pairs (5% 0.45, 95% 23.6); 0.65 for strict pairs.
- Per candidate, replicate SD of the gain divided by |mean gain|: median 2.08.
- Per-replicate gains (4,096 values): 42.2% are negative, skew 6.1, excess kurtosis 61. Of the 1,024 candidate means, 45.0% are negative. One extra response often makes the structural prediction worse.
- One-way variance components per record (8 candidates × 4 replicates): the candidate effect is significant at p < 0.05 in 24/128 records (6.4 expected by chance), p < 0.01 in 17, p < 0.001 in 7. **Reliability of the 4-replicate candidate mean: median 0.145, mean 0.252; exactly 0 (between ≤ within variance) in 62/128 records; above 0.5 in 28; above 0.8 in 7.** Single-replicate intraclass correlation: median 0.04.
- Split-half correlation of candidate means (2 vs 2 replicates): mean 0.087, median 0.079, negative in 52/128 records. Spearman-Brown reliability of the 4-replicate mean: 0.16.
- Without selection, the sign of a held-out replicate difference matches the sign of the mean of the other three in 7,625/14,336 = 53.2%. Split-half direction agreement is 53.5%. 50% is pure noise.
- By snapshot, reliability (median / mean / share exactly 0): 0: 0.32 / 0.39 / 37.5%; 1: 0.00 / 0.24 / 53%; 2: 0.00 / 0.18 / 53%; 3: 0.08 / 0.19 / 50%.

### 5.3 Many labels were collected where the task was already solved (verified)

The base normalized MSE already meets the 0.04 tolerance in 78/128 snapshot states (by snapshot 11, 18, 23, 26 of 32; median base NMSE 0.050, 0.035, 0.011, 0.010). In those states strict labels are 338/2,184 (15.5%) against a null expectation of 13.0%, and reliability has median 0.0. In the other 50 states strict labels are 349/1,400 (24.9%) against 13.7%, and reliability has median 0.33. **73 of the 119 DPO pairs come from states that already met the target.**

### 5.4 External check against the planners' own scores (verified)

For states whose teacher was `target_risk` or `local_information`, `teacher.json` holds that teacher's score for all 8 actions.

| States | Strict labels agreeing with the teacher's score ordering | By snapshot 0 / 1 / 2 / 3 | All comparisons, sign agreement | DPO-selected pairs agreeing |
| --- | ---: | --- | ---: | ---: |
| target_risk (32 states, 179 strict) | 123/179 = 68.7% | 45/54, 35/48, 14/38, 29/39 | 475/896 = 53.0% | 20/30 |
| local_information (32 states, 176 strict) | 118/176 = 67.0% | 64/78, 32/40, 8/22, 14/36 | 493/896 = 55.0% | 16/30 |

Strict labels with measured p < 0.01 in target-risk states agree with the planner in 22/24. Two independent estimates therefore agree: about one strict label in three has the wrong direction (noise accounting implies about 31% to 34% if noise-level labels are right half the time; planner disagreement is 31.3%). Taking a flip rate of 0.31 as given, a strict label carries at most about 0.10 bits about the ordering, so the 119 used pairs carry roughly 12 bits and all 687 roughly 71 bits (**inferred**, binary symmetric channel bound).

Per-state Spearman correlation between target-risk predicted risk reduction per cost and the measured 4-replicate mean gain across the 8 actions: mean 0.089, median 0.036, positive in 53%. For local_information scores: mean 0.136, median 0.167. In target-risk states the median standard error of a measured action mean is 1.44 times the planner's predicted expected gain, and the predicted best-to-worst spread is only 1.19 times that standard error. Four replicates cannot rank eight actions.

## 6. Training logs and saved evaluation bundles (task item 6)

Script `training_logs.py`, output `training_logs.out`.

### 6.1 What is saved (verified)

- Per step: loss, gradient norm before and after clipping, tokens, timing. `step-NNNNN.gradients.json` lists only `finite`, `present`, `shape` per tensor. **No per-tensor norms and no parameter-change records are saved.** Adapter and optimizer tensors exist as `.pt` files, which I did not open.
- SFT checkpoint: log-likelihoods only for the chosen and rejected commands of the 119 DPO pairs (238 values, `reference-scores.json`), plus 161 greedy generations.
- Final checkpoint: log-likelihoods of all 9 legal commands on the 128 nonterminal training states and of stop on the 32 terminal states (1,184 values), plus 161 greedy generations.
- **No base (frozen) model scores or generations on training prompts are saved.** The "28 scores and 10 generations" in test names correspond to the 4-record artificial qualification (3 × 9 + 1 and 2 × 5). The scientific run has 1,184 and 322.
- The acceptance review checked legality, reload exactness and repeatability of the generations. It did not inspect which commands were generated, and states that scientific utility is unqualified (`acceptance01/review.md:19,45`).

### 6.2 Supervised stage (verified)

Loss is mean NLL per action token; each example is scored before its own single update, so the trajectory is prequential.

| Steps | Mean loss | Median |
| --- | ---: | ---: |
| 0 to 19 | 0.597 | 0.520 |
| 20 to 39 | 0.367 | 0.344 |
| 40 to 59 | 0.302 | 0.260 |
| 60 to 79 | 0.244 | 0.234 |
| 80 to 99 | 0.239 | 0.197 |
| 100 to 119 | 0.257 | 0.207 |
| 120 to 139 | 0.212 | 0.147 |
| 140 to 159 | 0.217 | 0.237 |

The loss plateaus after about 60 updates. Gradient norm before clipping: median 3.68, max 78.6; 149/160 steps were clipped to 1.0.

Converted to nats per example (loss × action tokens):

| Steps | Nonterminal examples | Stop examples |
| --- | ---: | ---: |
| 0 to 39 | 3.88 | 3.82 |
| 40 to 79 | 2.25 | 1.92 |
| 80 to 119 | 2.57 | 0.29 |
| 120 to 159 | 2.17 | 0.37 |

- The stop rule at terminal states was learned.
- On nonterminal examples in steps 80 to 159 the LLM's prequential NLL is **2.37 nats**. A state-blind running-frequency counter with add-one smoothing scores **1.93** on the same examples (paired difference +0.44, SE 0.13, n = 64). Uniform over 8 actions is 2.08. In-sample label entropy is 1.88. The share of those examples where the model gives the teacher label more than 50% probability is 9.4%.
- By teacher, LLM against the state-blind baseline, steps 80 to 159: local_information 1.85 vs 1.74; menu_coverage 2.18 vs 1.87; target_risk 2.15 vs 1.72; uniform_random 3.30 vs 2.40. The LLM never beats the state-blind baseline.

**The supervised stage did not reach the predictive quality of a constant label-frequency model on new nonterminal states.**

### 6.3 Preference stage (verified)

- First DPO loss is exactly ln 2 = 0.693147. Mean online loss over the pass is **0.7395**, above ln 2. By block of 20 steps: 0.693, 0.742, 0.843, 0.766, 0.740, 0.649. 55 of 119 steps are below ln 2 and 64 above.
- Implied pre-update margin (change in log-odds of chosen over rejected relative to the reference, from the saved loss): quartiles −2.41, −0.13, +1.70; range −13.8 to +15.9; positive in 46.2% of steps. Each pair is seen once, before its own update, so this margin measures only what transferred from earlier pairs. Transfer hurt slightly more often than it helped.
- Gradient norm before clipping: min 1.30, median 2.57, max 17.8. **All 119 steps were clipped to norm 1.0.** The factor σ(−βΔ) that is supposed to shrink the update once a pair is satisfied was therefore cancelled by clipping at every step (**inferred** consequence; the clipping itself is verified).
- DPO objective on its own 119 pairs at the final checkpoint: 0.6811, against 0.6931 at the reference. 74/119 pairs improved.

### 6.4 SFT checkpoint against final checkpoint on the trained pairs (verified)

- The SFT checkpoint already ranks the preferred command above the rejected one in 66/119 pairs. The final checkpoint does so in 76/119. Cross-table: both agree 62, both disagree 39, fixed by DPO 14, broken by DPO 4. After training, 43 of the 119 trained pairs are still ranked against their label.
- In the 23 trained pairs that involve `set_x_neg`, the SFT checkpoint scores `set_x_neg` higher in 23/23 and the final checkpoint in 22/23. The label prefers `set_x_neg` in 14 of them.
- Mean log-likelihood per command over the 238 (state, command) points:

| Command | n | SFT mean | Final mean | Change | Change when preferred | Change when rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observe_all | 19 | −1.09 | −2.21 | −1.12 | −1.03 | −1.16 |
| set_x_neg | 23 | −0.95 | −0.38 | **+0.57** | +0.59 | +0.55 |
| set_x_zero | 32 | −4.90 | −14.47 | −9.57 | −9.07 | −9.91 |
| set_x_pos | 37 | −3.22 | −7.06 | −3.84 | −3.69 | −4.11 |
| set_y_neg | 33 | −2.26 | −6.83 | −4.57 | −4.40 | −4.80 |
| set_y_zero | 29 | −5.56 | −18.91 | −13.34 | −12.56 | −14.31 |
| set_y_pos | 29 | −4.33 | −16.66 | −12.33 | −9.72 | −13.32 |
| set_xy | 36 | −3.88 | −3.91 | −0.03 | +0.22 | −0.34 |

- **Variance of the change explained by command identity alone: R² = 0.791.** Command plus snapshot index: 0.877. Role (preferred or rejected) alone: 0.030. Mean change of preferred commands −4.69 nats, of rejected commands −6.68. Correlation between the actual per-pair margin change and the change predicted from command-mean shifts alone: 0.931. Residual state-specific margin change: mean +0.88 nats (SD 3.05, p = 0.002).
- "Rich get richer": correlation between a command's SFT mean log-likelihood and its DPO change is 0.80 over the 8 commands. Correlation between a command's label net (preferred minus rejected among the 119) and its DPO change is 0.44. `set_x_pos` has the largest label net (+9) and lost 3.84 nats.
- SFT log-likelihoods are themselves 82% command main effect on these points.
- SFT checkpoint by snapshot (geometric-mean probabilities on the saved subset): snapshot 0 `set_x_neg` 0.66, `observe_all` 0.16, `set_x_pos` 0.016. Snapshots 1 to 3 `observe_all` 0.37, `set_x_neg` 0.35, `set_x_pos` 0.061. **Yet the labels at snapshots 1 to 3 are `set_x_pos` 27, `observe_all` 21, `set_x_neg` 16.** The supervised model's neglect of `set_x_pos` is not what the label frequencies predict.

### 6.5 Final checkpoint: full command table on the training states (verified)

| Command | Mean log-likelihood | SD across 128 states | Min | Max | Mean probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| observe_all | −2.29 | 0.42 | −3.28 | −1.40 | 0.110 |
| **set_x_neg** | **−0.34** | **0.29** | −2.09 | −0.13 | **0.732** |
| set_x_zero | −14.57 | 2.39 | −17.57 | −3.91 | 0.0003 |
| set_x_pos | −7.24 | 3.23 | −10.88 | −0.37 | 0.044 |
| set_y_neg | −6.55 | 1.93 | −9.54 | −2.84 | 0.007 |
| set_y_zero | −20.50 | 4.50 | −26.30 | −5.68 | 0.0001 |
| set_y_pos | −14.01 | 5.57 | −21.02 | −2.28 | 0.003 |
| set_xy | −3.89 | 0.97 | −7.51 | −1.51 | 0.031 |
| stop | −12.66 | 2.20 | −16.51 | −6.69 | 0.00005 |

- Highest-scoring command: `set_x_neg` in 123 states, `set_x_pos` in 5 (all in worlds 202611070019 and 202611070020). Runner-up: `observe_all` 110, `set_x_pos` 9, `set_x_neg` 5, `set_xy` 4. Margin over the runner-up: min 0.44, quartiles 1.53, 1.93, 2.41, max 2.86 nats. Total mass on the 9 legal commands: median 0.954, min 0.693.
- Two-way decomposition of the 128 × 8 table: **command main effect 82.7%, state main effect 7.3%, state × command interaction 10.0%.** Only the interaction can change the ranking.
- The suffix decision is shared across X and Y: across states, (lp `x_pos` − lp `x_neg`) correlates 0.936 with (lp `y_pos` − lp `y_neg`), and the zero-versus-neg contrasts correlate 0.911. Means: `_pos` − `_neg` −6.90 after `_x` and −7.46 after `_y`; `_zero` − `_neg` −14.23 and −13.95.
- The policy is not blind to state. `_pos` − `_neg` has SD 3.4 nats across states (range −10.6 to +1.7) and is −10.1 at snapshot 0 against −5.7 to −6.0 later. Final lp(`set_x_pos`) correlates 0.575 with the number of `set_x_pos` entries in the history. lp(stop) rises from −15.1 at snapshot 0 to −9.9 at snapshot 3. **Inputs reach the scores, but the state-independent bias is larger than the state-dependent variation in 123 of 128 states.**
- On the 32 terminal states lp(stop) has median −2.26 and min −10.2 at the final checkpoint, against about −0.3 at the end of the supervised pass. DPO degraded the stop likelihood although no pair contains stop. The grammar still forces stop there.

### 6.6 Saved greedy generations on training states (verified)

| Checkpoint | Nonterminal completions (128) | Equals teacher label |
| --- | --- | ---: |
| SFT | set_x_neg 108, observe_all 19, set_x_pos 1 | 36/128 |
| Final (DPO) | set_x_neg 123, set_x_pos 5 | 32/128 |

SFT by snapshot: 0: 32 `set_x_neg`; 1: 32 `set_x_neg`; 2: 25 `set_x_neg`, 7 `observe_all`; 3: 19 `set_x_neg`, 12 `observe_all`, 1 `set_x_pos`. The 32 terminal completions are stop at both checkpoints, forced by the grammar (one legal command). The revisit of record 0 at ordinal 160 reproduces `set_x_neg`. These rates match deployment (SFT 481/523 = 92%, DPO 576/576).

History copying is not the mechanism: SFT generation equals the last history action in 14/128 and the modal acquired action in 33/96. The 19 `observe_all` generations occur in mixed histories, 15 of them under menu_coverage or uniform_random behavior.

### 6.7 A state-blind model of the DPO pass (verified arithmetic; interpretation inferred)

Script `bias_only_dpo_model.py`. Model: 8 logits only (observe/set; `_x`/`_y`/`_xy`; `_neg`/`_pos`/`_zero` shared by X and Y), no state input. Initialized at node probabilities implied by the mean SFT log-likelihoods (observe 0.37, set 0.63; `_x` 0.75, `_y` 0.21, `_xy` 0.04; `_neg` 0.88, `_pos` 0.10, `_zero` 0.02). Updated once per pair, in the real order, with a unit-norm step on that pair's margin. One fitted step size (0.9 logit units, momentum 0.5).

- Correlation between its predicted pre-update margins and the 119 margins implied by the saved online losses: **0.80** (R² 0.635 against the "no transfer" prediction of zero; sign agreement 93/119).
- Final per-command shift, model against observed: `set_x_neg` +0.84 / +0.57; `set_xy` −0.05 / −0.03; `set_y_pos` −12.0 / −12.3; `set_y_zero` −13.2 / −13.3; `set_x_pos` −5.0 / −3.8; `set_y_neg` −6.1 / −4.6; `set_x_zero` −6.3 / −9.6; `observe_all` −4.9 / −1.1. Correlation 0.93.

Toy counterfactuals in that model (`bias_only_counterfactuals.py`, **inference only, not evidence about the real network**): with the actual 119 pairs the winner is `set_x_neg` for every one of 2,000 random orders, every step size tried and even from uniform initial biases. With a different random one-strict-pair-per-record selection the winner is `set_x_neg` 48.5%, `set_x_pos` 19.8%, `observe_all` 13.2%, `set_xy` 8.4%, `set_y_neg` 6.9%. With all 687 strict pairs, `set_x_neg` in 478/500 orders and `set_x_pos` in 22. With coin-flip directions on the actual pairs: `observe_all` 47.7%, `set_x_neg` 12.6%, others below 12%. Reading: a learner that cannot use state collapses to some single command; which one depends on the token-level tallies of the particular pairs and on the SFT starting point.

## 7. Teacher behavior on the training states (task item 7)

Script `teachers.py`, output `teachers.out`.

| Teacher | Distinct actions | Modal action (share) | Entropy (max 2.08) | Best constant rule, leave-one-world-out |
| --- | ---: | --- | ---: | ---: |
| uniform_random | 7 | set_y_neg 7/32 | 1.91 | 7/32 |
| menu_coverage | 7 | set_x_pos and set_x_neg 12/32 each | 1.46 | 0/32 (ties flip); by snapshot 20/32 |
| local_information | 5 | observe_all 11/32 | 1.42 | 11/32; by snapshot 18/32 |
| target_risk | 4 | set_x_pos 9/32 | 1.38 | 0/32; by graph × cost 8/32 |

Within graph × cost cells (5 or 6 states each) the modal action of the two informed teachers never exceeds 3 of the cell. In-sample cell-modal accuracy: target_risk 15/32, local_information 13/32. **The informed teachers are state-dependent. No constant or cell-constant rule reproduces them.** Neither ever chose `set_x_zero` or `set_y_zero`, and target_risk never chose a single-Y action.

Decisiveness of target_risk by its own saved estimates: the runner-up is within 2 Monte Carlo standard errors of the chosen action in 17/32 states (z gap quartiles 0.39, 1.80, 5.94). The runner-up reaches at least 90% of the best score in 11/32 and at least 80% in 18/32. The best action's one-step relative risk reduction has median 8.9%. When a single-X intervention is chosen, `set_x_neg` and `set_x_pos` usually differ a lot (median relative difference 0.69), so that choice is driven by the state. For local_information the runner-up reaches at least 90% of the best score in 12/32.

Consequences for the supervised labels: 25% are random, 25% follow a counting rule with a lexical tie-break that emits `set_x_neg` at every initial state, and about half of the target-risk labels are near-ties by the planner's own estimate.

## 8. Verdict table (task item 8)

| # | Candidate explanation | Verdict | Supporting numbers | Contradicting numbers that must be kept |
| --- | --- | --- | --- | --- |
| i | Marginal label imbalance in supervised targets | **Partly supported: a necessary ingredient, not sufficient** | `set_x_neg` is the plurality nonterminal label (32/128) and the plurality at every token decision (set 81.3%, `_x` 65.4%, `_neg` 45.7%). It is 16/32 at the initial state, 8 of those from menu coverage's lexical tie-break. The SFT checkpoint emits it on 108/128 training states and matches labels in 36/128 against 32/128 for the constant. | The imbalance is small (25.0% vs 22.7% vs 18.8%). At snapshots 1 to 3 labels favor `set_x_pos` (27) and `observe_all` (21) over `set_x_neg` (16), yet the SFT checkpoint gives `set_x_pos` about 0.06 and `set_x_neg` about 0.35 there. Majority accuracy is only 25%, so a learner using state would not collapse. |
| ii | Preference pairs dominated by one preferred action | **Contradicted at the command level. A weak token-level tilt exists; its causal role is undetermined** | Token tallies in the 119 pairs: `_neg` 33 vs 23, `_x` 50 vs 42, observe 6 vs 13. All point along `set → _x → _neg`. A state-blind model driven by these pairs ends at `set_x_neg` (inferred). | `set_x_neg` is preferred in 14/119 (11.8%, below the uniform 12.5%) and rejected in 9. `set_x_pos` is preferred most (23) and still lost 3.8 nats. Token tilts are not significant (p 0.23, 0.47, 0.17). In all 687 strict pairs `_pos` (+22) leads `_neg` (+13). |
| iii | Preference labels approximately state-independent | **Contradicted** | none | Constant ranking: 56.8% in-sample (coin-flip null 54.1%), 48.9% leave-one-world-out; per-pair majority ceiling 58.2%; 0 of 28 pairs significant after correction. Strict labels agree with the state-dependent target-risk planner in 68.7%. A cross-validated state-dependent picker earns 0.0127 against 0.0085 to 0.0089 for the best constants. |
| iv | Label noise | **Supported** | Tie rule is a 3-df t-test at α 0.139: 476 to 499 strict expected under no difference, 687 observed. Excess is 5% to 6% of comparisons. Reliability of 4-replicate means: median 0.145, zero in 62/128 records, split-half 0.16. BH at q 0.10 keeps 1 comparison. 73/119 DPO pairs come from states that already met the target. About 31% of strict labels disagree with the target-risk ordering. | Labels are not pure noise: 68.7% and 67.0% agreement with the two planners (p < 1e-4), 22/24 for labels with p < 0.01, and signal at the initial state (excess share 52%, reliability median 0.32). |
| v | Too few updates | **Supervised stage: undetermined (too little informative data is supported; more passes untested). Preference stage: contradicted** | SFT: 160 single-example updates; prequential NLL on nonterminal states 2.37 against 1.93 for a state-blind frequency counter; only 64 labels from SCM-informed teachers. DPO: 43/119 trained pairs still ranked against the label; objective moved only from 0.6931 to 0.6811. | SFT loss plateaus after about 60 updates. DPO updates were large, not small: per-command mean shifts up to −13.3 nats, single-pair margins up to ±19 nats, all 119 steps clipped. More of the same would amplify the drift. |

Additional explanations the saved evidence raises:

| # | Explanation | Status |
| --- | --- | --- |
| vi | Command representation and decoding: shared suffix tokens plus greedy trie decoding turn weak conditioning into the per-node plurality path | Tokenization and decoding verified; final policy factorizes by token (r = 0.94 and 0.91). Causal weight inferred. |
| vii | DPO configuration (β = 0.1, clip 1.0, Adam, batch of one pair, fixed order) acts as an unanchored margin maximizer on global command biases | Clipping of all 119 steps, ±19 nat moves, R² 0.79 by command, both sides falling: verified. State-blind model fit 0.80 / 0.93: verified arithmetic. Mechanism: inferred. |
| viii | Base-model prior or menu-position bias toward the first listed intervention | Undetermined. No base-model scores on training prompts are saved. The SFT preference for `_neg` over `_pos` exceeds label frequencies, which is consistent with a prior but does not establish one. |
| ix | Supervised mixture of four unlabeled teachers | Verified as a design fact (25% random labels, 25% counting rule, teacher not in prompt). Its effect on the outcome is inferred. |
| x | Budget coverage of supervised states | Verified: nonterminal states only at 5, 6, 11, 12, 17, 18, 24 remaining units; stop only at 0. Relevant to the supervised arm's early stops in deployment (inferred link). |

## 9. Disagreements and corrections

- The task brief's file counts (about 12,400 collection files, about 235 training files) are wrong for the local copies: 30,596 and 4,926.
- The agenda (`ace_llm_intervention_agenda_2026-11-02.md:91`) asks for "uncertainty-aware ties rather than turning negligible noisy differences into strong preferences" and for scoring on "an independent training-world scoring set". The executed rule has a 13% to 14% false-strict rate by construction, and scoring used noise-disabled structural truths rather than a sampled scoring set. The handoff says the agenda's original counts are superseded, so this is a gap between intent and execution rather than a contradiction between current documents.
- The handoff's statement that the action counts "establish limited observed adaptation, not the cause" stands. What this analysis adds is that the same behavior is present on the training states and in the saved command scores.
- No numerical disagreement found between the handoff, the results report, `exposure.json`, the collection acceptance review and my recounts for 160 / 119 / 3,584 / 687 / 2,897 / 10,240 / 9,145.

## 10. Implications for the experimental agenda

1. **Do not present the completed DPO arm as a test of preference learning for intervention choice.** The saved evidence shows it never learned a state-conditional preference. It is a test of this particular label, exposure and optimizer recipe.
2. **Zero-cost paper asset available now:** the training-state behavior table (108/128, 123/128), the command-main-effect decomposition (82.7%) and the label-noise accounting can be reported from saved files without new compute.
3. **Add a behavioral gate before any deployment** of a trained policy: action distribution and teacher agreement on training states and on held-out training worlds, plus the state × command interaction share of the score table. The current acceptance gate checks legality only.
4. **Fix the preference label before training again.** Options to compare on CPU: expected (planner or exact posterior) gains instead of 4 realized single-sample outcomes; many more replicates with common random numbers; restricting labels to states that have not yet met the target; a stricter evidence threshold. A direct diagnostic is to compute high-precision expected gains for the 128 saved states from the saved private worlds and measure the true flip rate of the 687 strict labels. That needs explicit authorization because it draws new simulated responses from training worlds.
5. **Use one strong teacher, with soft labels.** A mixture that includes a random teacher and a counting rule caps what imitation can learn. Target-risk labels are diverse and state-dependent, but half are near-ties, so a distribution over actions from the planner's scores is the honest target.
6. **Change the action interface.** Give each action a single distinct token, randomize menu order per prompt, and choose by full-command scoring normalized over the legal set. Then the plurality-path artifact and any position prior can be measured rather than assumed.
7. **Make the preference optimizer incapable of pure bias drift**, and test it: balanced minibatches, stronger anchoring to the reference, all strict pairs rather than one per record. Run two cheap controls with the unchanged recipe: labels with directions flipped at random, and labels reassigned to shuffled states. If the outcome is the same single command, the recipe is state-blind. Each run is about five minutes of training on one 20 GB slice.
8. **Run the cheap forward-only diagnostic the handoff already asks for:** base, SFT and final command-score tables on the 160 training prompts plus designed artificial states (menu order permuted, costs changed, posterior covariance scaled, history truncated). The base table is the missing piece for explanation viii.
9. **Keep the numerical ranker as the decisive control.** It consumed the same 160 + 119 exposure and also lost to target risk in deployment (13.583 vs 11.625 per the handoff). If it shows the same bias-dominated pattern on training states, the supervision is the common cause. If it does not, the language interface is implicated. I did not analyze its saved training outputs.
10. **Count training cost honestly.** 9,216 of 10,240 collection cost units bought 119 used pairs. Any efficiency claim has to carry that bill.

## 11. Open questions

1. What does the frozen base model score on the training prompts, and does it already prefer `_neg` over `_pos` and `observe_all` over everything? Not saved.
2. What is the true flip rate of the 687 strict labels against exact expected gains? Estimated at about one third from two indirect routes.
3. Would the real network collapse to a different command under a different one-pair-per-record selection, or under random label directions? Predicted by the toy model, untested.
4. How large were the adapter updates in parameter space per stage? Computable from the saved `initial`, `sft` and `final` adapter files, which I did not open.
5. Does the supervised arm's early stopping in deployment occur at remaining budgets 1 to 4, values absent from nonterminal training states?
6. Does the numerical ranker trained on the same exposure show the same command-bias pattern?
7. Is the SFT model's excess preference for `set_x_neg` at snapshots 1 to 3 a base prior, a position effect, or generalization from the initial state?

## 12. Scripts and outputs (all scratch, nothing in the repository)

Directory: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training-forensics/`

| Script | Purpose | Output |
| --- | --- | --- |
| `load_collection.py` | Reads the 160 records, teachers, selections and 128 feedback aggregates into `collection.pkl` | `collection.pkl` |
| `sft_targets.py` | Supervised target distributions and majority accuracies | `sft_targets.out` |
| `prefs_analysis.py` | Pair counts, tie rule verification, ordered matrices, 119 selected pairs, margins | `prefs_analysis.out` |
| `state_dependence_and_noise.py` | Per-pair state dependence, Bradley-Terry fits with leave-one-world-out, noise tests, variance components | `state_dependence_and_noise.out` |
| `training_logs.py` | Loss and gradient trajectories, SFT and final score tables, generations | `training_logs.out` |
| `teachers.py` | Teacher concentration and decisiveness | `teachers.out` |
| `bias_only_dpo_model.py`, `bias_only_counterfactuals.py` | State-blind 8-logit model of the DPO pass and toy counterfactuals | `.out` files of the same names |

Several further checks were run inline and are reported above: sign-flip null, target-attainment split, planner agreement, prequential baselines, token tallies, prompt token statistics, final-table decomposition. The pickles in that directory were written by these scripts from the saved JSON and are not project data.

Limits of this analysis: no model was run, so every statement about the base model, about why the supervised model over-weights `_neg`, and about what would happen under a changed recipe is inference. Statistical tests treat the 28 pairs of a record as separate comparisons although they share 8 × 4 measurements; counts and expectations are unaffected by that dependence, p-values for aggregates are approximate.
