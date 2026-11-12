# Independent verification of the training-forensics claims

Label: `training-forensics-verify`. Prepared 2026-11-12. Read-only on all project data. No model weights were loaded, no inference or training was run, nothing was written under `/Users/pat/code/ACE`.

**Bottom line.** Every quantitative claim I re-derived reproduces from the raw saved files, most of them to the last digit. I found no number that is wrong. Five claims need a qualification (listed in section 2). My extra tests mostly strengthen the earlier verdicts and add four things the earlier analysis did not establish (section 3).

## 0. Method and independence

- All code is my own, written fresh under `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/training-forensics-verify/` (`v01_load.py` to `v21_exact_blocks.py`, each analysis script with a `.out` file of the same name). I did not open the earlier agent's scripts or `.out` files.
- I read one part of the earlier agent's prose notes (`w1/training-forensics.md`, sections 6.4 to 6.7) **after** finishing my own derivations, only to learn the specification of the "8-logit" model. My first, blind attempt at that model used a different specification and did not reproduce the claim (section 1, row J). With the described specification it reproduces.
- Path roots:
  - `COLL` = `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/terminal02/remote/launch/collection/session/worker/task/collection`
  - `TG` = `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108`
  - `PREP` = `TG/terminal01/remote/cpu/preparation/session/worker/artifacts`
  - `FIT` = `TG/terminal01/remote/launch/training/session/worker/artifacts/fit`
  - `SRC` = `/Users/pat/code/ACE/scripts/research`
- Inputs actually read: `COLL/world-*/snapshot-*/{record,teacher,selection,language,numeric-labels,alignment}.json`, `COLL/world-*/snapshot-*/feedback/terminal.json`, `COLL/world-*/{plan,private-world}.json`, `COLL/terminal.json`, `PREP/{exposure,llm-prepared}.json`, `FIT/training/step-*.{start,loss,complete,update-intent,gradients}.json`, `FIT/training/{reference-scores,terminal,reference-snapshot}.json`, `FIT/{final-before-reload,sft-reload-scores,final-reload-scores,arithmetic,summary}.json`, `FIT/{sft,final}-generation-*.json`, `TG/acceptance01/review.md`.
- The selected DPO pair for each record was established three independent ways that agree 119/119: my own recomputation of the selection hash from `SRC/llm_intervention_training_exposure.py:37-39,100`, the `status == "selected"` rows of `PREP/exposure.json`, and `pair_index` in `FIT/training/step-*.start.json`.

## 1. Claim-by-claim verdicts

| # | Claim | Verdict | My value | Script |
| --- | --- | --- | --- | --- |
| A | Supervised targets: observe_all 24, set_x_neg 32, set_x_zero 7, set_x_pos 29, set_y_neg 10, set_y_zero 4, set_y_pos 10, set_xy 12, stop 32 | **Confirmed** | Identical. Labels agree across `record.json`, `language.json` and `teacher.json` in 160/160 | `v02` |
| A2 | set_x_neg is 20.0% of 160 and 25.0% of 128 nonterminal. Stop is 20.0%, all at zero budget | **Confirmed** | 32/160, 32/128. All 32 stop labels have `remaining_units` 0 and one legal command | `v02` |
| B | Majority accuracy 20% overall, 25% nonterminal. Graph x cost cells 42/160 in-sample, 29/128 nonterminal leave-one-world-out | **Confirmed** | 32/160, 32/128, 42/160. Leave-one-world-out 29/128 with first-listed tie-break, 30 with last-listed, 29.75 with fractional credit. Nonterminal in-sample is 41/128 | `v02` |
| C | Half of nonterminal labels come from teachers that ignore the SCM (32 + 32). Menu coverage emits set_x_neg at all 8 initial states | **Confirmed** | 32 uniform_random, 32 menu_coverage. 8/8 set_x_neg at snapshot 0. Source: `SRC/llm_intervention_numerical_policy.py:59-66` (least-used count, then lexical `action_id`) | `v02` |
| C2 | set_x_neg is 16/32 at snapshot 0. Snapshots 1-3 favor set_x_pos 27 and observe_all 21 over set_x_neg 16 | **Confirmed** | 16/32. Then 27, 21, 16 of 96 | `v02` |
| C3 | Token-level pluralities: set 104/128, _x 68/104, _neg 42/92 | **Partially confirmed** | 104/128 and 68/104 exact. 42/92 pools `_neg` over the `_x` and `_y` prefixes. On the actual greedy path (after `set`, `_x`) it is 32/68 = 47.1% against `_pos` 29 and `_zero` 7. The plurality claim holds, by a margin of 3 labels. Token ids from `PREP/llm-prepared.json` | `v05` |
| C4 | Teacher identity is not in the prompt | **Confirmed** | No teacher, behavior or method string in the prompt text | `v12` |
| D | Among the 119 DPO pairs set_x_neg preferred 14, rejected 9. set_x_pos 23/14, set_y_pos 8/21, observe_all 6/13 | **Confirmed** | Identical. Full table in section 4 | `v03` |
| D2 | In all 687 strict pairs set_x_neg 88/74, set_x_pos 123/78 | **Confirmed** | Identical | `v03` |
| D3 | Token tallies in the 119: _neg 33 vs 23, _x 50 vs 42, none significant | **Confirmed** | 33/23 (sign test p 0.18), 50/42 (p 0.39), set 113/106 and observe 6/13 (p 0.17). One uncorrected exception at command level: set_y_pos 8/21, p 0.024, not significant after correction for 8 commands | `v03` |
| E | Constant Bradley-Terry ranking: 390/687 in-sample, 336/687 = 48.9% leave-one-world-out. Coin-flip null 54.1%. Per-pair majority ceiling 400/687 | **Confirmed** | 390/687 = 56.77%. Leave-one-world-out 336/687 = 48.91% (337 with a heavier prior). Coin-flip null mean 54.06%. Ceiling 400/687, and the exact best total order over all 8! permutations is also 400. World-cluster bootstrap 95% interval for leave-one-world-out accuracy 44.7% to 53.5% | `v06` |
| E2 | No unordered pair has a consistent winner after correction | **Confirmed** | Most lopsided pair observe_all vs set_x_pos 4 to 16, p 0.012, Bonferroni 0.33. 1 of 28 pairs below 0.05 uncorrected | `v19` |
| F | Tie rule is strict iff abs(mean) > 2 SE over 4 replicates, a 3-df t-test at alpha 0.139. 476-499 strict expected under no difference, 687 observed, excess 5-6% | **Confirmed** | Recomputed all 3,584 pairs from per-replicate gains with 0 disagreements. P(abs(t3) > 2) = 0.1393. Normal null 499.3, within-pair sign-flip null 476.0. Excess 5.24% to 5.89% of comparisons. Source: `SRC/llm_intervention_replicated_feedback.py:160-171` | `v07` |
| F2 | Reliability of a 4-replicate action mean: median 0.145, zero in 62/128 | **Confirmed** | One-way ANOVA: median 0.145, zero in 62/128. With replicate treated as a block: median 0.064, zero in 59/128 | `v07` |
| F3 | Benjamini-Hochberg at q 0.10 keeps 1 of 3,584 | **Confirmed** | 1 at q 0.05 and 0.10, 4 at q 0.20. Smallest p 9.0e-7 | `v07` |
| F4 | 73 of the 119 DPO pairs come from states already at NMSE <= 0.04 | **Confirmed** | 73/119. Over all nonterminal states 78/128 | `v07` |
| G | Strict labels agree with target-risk score ordering in 123/179 (68.7%), local information 118/176 (67.0%). Per-state Spearman mean 0.089 | **Confirmed** | 123/179 and 118/176 exact. Spearman mean 0.0885 (t-test p 0.26). Record-cluster bootstrap 95% intervals 0.58 to 0.78 and 0.53 to 0.78 | `v08` |
| H | SFT plateaus after about 60 updates. LLM 2.37 nats per nonterminal example in steps 80-159 against 1.93 for a state-blind counter | **Confirmed** | 20-step block means 4.76, 3.00, 2.48, 2.03, 2.41, 2.73, 2.03, 2.31. LLM 2.3700 (n 64). Add-one counter over 8 actions 1.9294. ln 8 = 2.079. The saved loss is a per-token mean over the full vocabulary, so nats per example = loss x `action_tokens` (`SRC/llm_intervention_canonical_scoring.py:99-102`) | `v09` |
| H2 | DPO mean online loss 0.7395, above ln 2. All 119 steps clipped (1.30-17.8). 43/119 trained pairs remain ranked against their label | **Confirmed** | 0.7395. Pre-clip norm 1.302 to 17.824, 119/119 above 1. Final chosen below final rejected in 43/119. Under the SFT reference it was 53/119 | `v09` |
| I | DPO-stage change by command: +0.57, -0.03, -1.12, -3.84, -4.57, -9.57, -12.33, -13.34. Command identity explains 79%. Preferred fell 4.69, rejected 6.68 | **Confirmed** | All eight means identical to two decimals. Eta-squared 0.7906. -4.694 and -6.675 | `v09` |
| I2 | Final table: set_x_neg wins 123/128, median margin 1.93 nats. Command main effect 82.7%, interaction 10.0% | **Confirmed** | 123/128, the other 5 are set_x_pos. Median margin 1.933. 82.68%, state 7.29%, interaction 10.03% | `v09` |
| J | A no-state 8-logit model reproduces the 119 online margins (r 0.80) and the per-command shifts (r 0.93) | **Confirmed as arithmetic, with one specification caveat** | With the model as described in the earlier notes (three trie nodes with 2 + 3 + 3 token logits, suffix shared by X and Y, unit-norm step 0.9, momentum 0.5 applied as a moving average): r 0.797, R-squared against zero 0.635, sign agreement 93/119, shift correlation 0.932. The correlations barely depend on the two fitted numbers (0.75 to 0.82 and 0.90 to 0.95 over a 6 x 5 grid). The claim text "8-logit model" does not say the logits are tokens. A model with one logit per command gives only r 0.16 to 0.36 and 0.35 to 0.48 | `v10`, `v11`, `v15` |
| K | Target-risk labels: set_x_pos 9, set_x_neg 8, observe_all 8, set_xy 7. Best constant rule 0/32 leave-one-world-out. Runner-up within 2 Monte Carlo SE in 17/32 | **Partially confirmed** | Counts exact. 0/32 holds only under a first-listed tie-break. With fractional credit it is 3/32, and in-sample it is 9/32. The 0 is a tie artifact of near-uniform labels, not evidence of anti-predictability. 17/32 with a pooled SE, 16/32 with the larger SE, 12 to 15 under other reasonable definitions | `v08` |
| L | Not saved: base-model scores or generations, the full SFT table, per-tensor gradient norms, parameter deltas. The review checked legality, not content | **Partially confirmed** | No base generations. One base score exists: `FIT/first-gpu-work.json` holds -11.4279 for record 0's label (equals step 0 loss x 9). SFT scores: 238 only (`FIT/sft-reload-scores.json`). `step-*.gradients.json` holds finite, present and shape only. **Deltas are recoverable:** `FIT/training/{initial,sft,final}.adapter.pt` are in the local tree and their SHA-256 values match the recorded pins (6018f56a..., 583b1659..., 6f540dfe...). Review: `TG/acceptance01/review.md:19,45` | `v13`, shell |
| M | Headline: on the 128 nonterminal training prompts SFT generates set_x_neg 108, observe_all 19, set_x_pos 1 and matches the label 36 times. DPO generates set_x_neg 123, set_x_pos 5 and matches 32 times. A constant matches 32 | **Confirmed** | Identical. Ordinal 160 is a repeat of record 0 and reproduces it. All 32 terminal outputs are stop with one legal command | `v04` |
| N | Exposure: 160 records, fixed roster order, batch 1, no shuffling. 3,584 = 128 x 28, 687 strict, 2,897 ties. 119 DPO updates = 3.3%. Feedback cost 9,216 of 10,240 | **Confirmed** | Step order equals roster order. 119/3,584 = 3.32% (17.3% of strict pairs). 64 records x 64 units + 64 records x 80 units = 9,216. `COLL/terminal.json`: 10,240 units, 9,145 accepted responses | `v03` |
| O | Recipe, strict share by snapshot, prompt lengths | **Confirmed** | Rank 8, 1,146,880 parameters, AdamW 1e-4, clip 1.0, beta 0.1, 291.50 s. Strict share 27.01%, 17.97%, 16.07%, 15.62%. Prompts 1,998 to 4,199 tokens, largest plus 32 is far below 8,192. Common suffix 356 tokens over all 160 (360 over the 128 nonterminal) | `v05`, `v03` |

## 2. Qualifications to the earlier claims

1. **C3.** The third token share of 45.7% is not the share at the decision the decoder actually faces. On the greedy path it is 47.1% (32 of 68), against 42.6% for `_pos`.
2. **J.** "8-logit model" means eight token logits in a three-node trie, not one logit per command. This matters. A per-command bias model cannot reproduce the data, because set_y_zero has a positive label net (+3) and fell the most (-13.3 nats). The fit needs the shared `_neg`/`_pos`/`_zero` suffix node. So the supported statement is narrower: **a state-blind model with token-factorized biases** tracks the DPO pass. It explains about 64% of the variance of the online margins, not all of it.
3. **K.** "0/32" is a tie-break artifact.
4. **L.** Parameter deltas and the missing score tables are recoverable from saved local files without any new training (section 3.6).
5. **Summary sentence "it is not a training-to-development shift".** True for the constant action. Not established for the supervised arm's early stopping. On training prompts the SFT checkpoint never generates stop at a nonterminal state (0/128). Stop labels exist only at `remaining_units` 0, and nonterminal training states have 5 or more units left (24, 17-18, 11-12, 5-6). Early stopping in deployment therefore happens in budget states that training never covered. This last step is an inference, since I did not open deployment data.

## 3. Additional findings (mine, not in the earlier claims)

### 3.1 The in-sample 56.8% of a constant ranking is exactly what noise alone produces

The earlier comparison used a coin-flip-direction null (54.1%), under which 390/687 looks marginal (my estimate p about 0.05). That null ignores that strict pairs inside one record are dependent (one noisy high mean wins several pairs). Under a null that relabels the 8 actions at random within each record, the in-sample Bradley-Terry accuracy has mean 56.5% (SD 1.9), and the observed 56.8% has p 0.45. There is no detectable state-independent preference signal. Conditional rankings do not rescue it under leave-one-world-out: by cost menu 51.8%, by graph x cost 52.3%, by snapshot 51.7%, by graph 41.8%. (`v06`)

### 3.2 The label signal that exists is state-specific and tracks the SCM planners

This is the positive evidence that verdict iii lacked. Strict labels agree with the state's **own** target-risk ordering in 68.7% of 179 pairs. Paired with **another** state's target-risk scores they agree in 50.2% (2,000 shuffles, p 0.0005). With the leave-one-world-out average planner ordering they agree in 51.4%. Local information: 67.0% own, 51.6% shuffled (p 0.0055), 51.1% average. So the measured preferences carry real state-dependent information, and a learner that cannot condition on state cannot use any of it. Among the DPO-selected pairs only, agreement is 20/30 for target risk and 16/30 for local information. (`v19`, `v14`)

### 3.3 Neither checkpoint shows any label-tracking on its own training states

- Final checkpoint, full 128 x 8 table: the teacher's label scores -0.16 nats relative to the same command's mean over same-snapshot states (permutation p 0.86). By teacher: target risk +0.17, local information -0.06, uniform random -0.11, menu coverage -0.63.
- SFT checkpoint, 27 saved entries where the scored command is that record's own label: +0.09 nats against other entries (p 0.18).
- Within-state, 14 records trained at step 60 or later: label log-likelihood -2.168 just before the model's own update on that record and -2.177 at the end of SFT (paired p 0.98). The supervised pass retained nothing measurable about individual training states.
- Scores do respond to history: each extra occurrence of a command in the acquired history adds about +0.19 nats to that command's final score, and lp(set_x_pos) correlates 0.49 with the count of set_x_pos in history. Inputs reach the scores, but not in a way that tracks labels. (`v13`, `v18`)

### 3.4 The supervised neglect of set_x_pos appeared late in the pass. Order is an untested contributor

Saved per-step losses give the probability the model assigned to each label just before training on it. For set_x_pos labels the geometric-mean probability was 0.007 (steps 0-39, n 9), 0.124 (40-79, n 8), **0.341 (80-119, n 6)** and 0.173 (120-159, n 6). For set_x_neg labels it was 0.103, 0.256, 0.180 and 0.419. At the end of SFT the saved reference scores give set_x_pos 0.061 at snapshots 1-3 and 0.016 at snapshot 0. Within the same states (6 records labelled set_x_pos and trained after step 60), the probability fell from 0.195 before the update to 0.084 at the end of the pass. Over the same period set_x_neg rose (0.27 to 0.39, n 3). In the last 30 SFT steps the labels contain set_x_neg 8 times and set_x_pos twice. Five of those set_x_neg labels fall within six consecutive steps (132, 133, 135, 136, 137, the last three in world 27). Exact values are in `v21_exact_blocks.out`. The rounded block figure for steps 0-39 printed in `v16_drift.out` drops two near-zero examples and should not be used.

Limits: n is small. Simple recency weighting of label counts gives at most a 1.6 to 1 ratio of neg to pos (half-life 10 examples), far below the 5.7 to 1 (snapshots 1-3) or 41 to 1 (snapshot 0) in the SFT scores. So recency is not sufficient either. The earlier statement that the set_x_pos neglect "is not what the label frequencies predict" stands. What the saved data add is that the model was not always neglecting set_x_pos, so a single unshuffled pass with batch size 1 is a live explanation that only a reshuffled rerun can test. (`v12`, `v16`, `v17`, `v18`)

### 3.5 Token-greedy decoding is not command argmax, and it inflates the supervised arm's set_x_neg rate

At the SFT checkpoint the saved geometric-mean probabilities at snapshots 1-3 are observe_all 0.367 and set_x_neg 0.345, a near tie. Yet SFT greedy output there is set_x_neg in 76 of 96 states. In the 11 saved states where P(observe_all) is between 0.30 and 0.44, the greedy output is set_x_neg in all 11. In the 2 states where P(observe_all) exceeds 0.5, the output is observe_all. The decoder (`SRC/llm_intervention_language_policy.py:163-170`, `do_sample=False`, `num_beams=1`, prefix-constrained) first chooses between `observe` and `set`, and `set` carries the summed mass of seven commands. This is an inference from averages, because set_x_neg and observe_all are almost never both scored in the same SFT state. At the final checkpoint the issue does not arise: greedy output equals the full-table argmax in 128/128 states. (`v20`, `v12`)

### 3.6 Missing diagnostics can be regenerated by inference alone

`PREP/llm-prepared.json` stores `prompt_ids` and all nine command token sequences for every record. `FIT/training/` stores the initial, SFT and final adapters and both optimizer states with matching hashes. The base-model and SFT 128 x 9 command tables, and per-layer LoRA deltas for each stage, therefore need only the pinned Qwen3-0.6B base weights and forward passes. No training and no environment responses are required. I did not do this because weight loading was out of scope.

### 3.7 Other observations

- **DPO barely fit its own pairs.** The DPO loss on the 119 trained pairs is 0.6811 at the final checkpoint against 0.6931 at the reference. 45/119 pairs moved against their label relative to the reference. The pair-specific effect beyond command-mean shifts is +0.88 nats (SD 3.06, p 0.002), which at beta 0.1 is negligible. (`v09`, `v13`)
- **DPO amplified the SFT ordering.** Over the 119 pairs the final score difference regressed on the SFT score difference has slope 3.45 (r 0.79). The correlation between a command's SFT mean log-likelihood and its DPO change is 0.80. The correlation between its label net (preferred minus rejected) and its change is 0.44. (`v10`, `v11`)
- **Null controls for the token-bias model.** With coin-flip label directions the margin correlation is 0.15 on average (5th to 95th percentile -0.37 to 0.63) and the shift correlation exceeds 0.93 in 1.6% of runs. With the real pairs in random order the margin correlation is 0.37 and the shift correlation 0.92. So the fit is specific to the real labels, and the margin fit also to the real order. (`v11`)
- **One toy counterfactual did not reproduce.** The earlier notes report that with all 687 strict pairs the toy model ends at set_x_neg in 478 of 500 random orders. In my re-implementation it ends at set_x_pos in about 65% and set_x_neg in about 35%, under every step convention I tried. In roster order it ends at set_x_neg. With a different random pair per record I get set_x_neg 40.6% (earlier notes 48.5%), set_x_pos 21.1%, set_xy 15.0%, observe_all 13.9%. With the actual 119 pairs it is set_x_neg in every order, as reported. Treat which command wins under altered supervision as not robust. (`v14`, `v15`)
- **One-step gains are at noise level after the first snapshot.** Mean raw NMSE reduction from one extra response is 0.0230 at snapshot 0 and 0.0035, 0.0018, 0.0013 at snapshots 1-3. The share of positive 4-replicate mean gains falls from 60.5% to 50.0%. Overall 45.0% of the 1,024 action means are negative. 69% to 73% of the 687 strict labels are what the tie rule passes under no difference at all (476 to 499). (`v07`, `v14`)
- **The informed teachers are not equally supported by the measured feedback.** Where a teacher's own label appears in a strict pair, it is preferred 45 times and rejected 11 for local information, 35/19 for target risk, 30/26 for menu coverage and 16/13 for uniform random. The teacher's choice is the measured-best action in 8/32, 7/32, 7/32 and 3/32 states (chance 4). (`v08`)
- **Cost-menu dependence of labels is suggestive only.** In strict pairs between an X action and a Y or joint action, X is preferred in 44.4% under the uniform menu and 61.5% under the heterogeneous menu. At world level this is not significant (Mann-Whitney p 0.40, 16 worlds each). (`v14`)
- **History comes from a per-world behavior policy.** Each world's trajectory is generated by one of the four numerical policies (8 worlds each, `COLL/world-*/plan.json` field `behavior_method`), independent of the rotating teacher. The teacher label equals the action actually executed next in 16/96 states where teacher and behavior differ. No training state has a history produced by an LLM policy, so deployment histories of 24 identical set_x_neg actions are outside the training state distribution. (`v12`)
- **Agenda intent against executed rule.** The research agenda asked for "uncertainty-aware ties rather than turning negligible noisy differences into strong preferences" (`/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md:91`). The executed 2-SE rule on 4 replicates passes 13.9% of null comparisons. This is a gap between intent and execution, not a disagreement between documents.

## 4. Reference tables

Composition of strict pairs (preferred / rejected):

| Command | All 687 | The 119 used for DPO |
| --- | ---: | ---: |
| observe_all | 53 / 71 | 6 / 13 |
| set_x_neg | 88 / 74 | 14 / 9 |
| set_x_zero | 79 / 102 | 13 / 19 |
| set_x_pos | 123 / 78 | 23 / 14 |
| set_y_neg | 86 / 87 | 19 / 14 |
| set_y_zero | 90 / 79 | 16 / 13 |
| set_y_pos | 75 / 98 | 8 / 21 |
| set_xy | 93 / 98 | 20 / 16 |

Supervised labels by teacher (nonterminal, 32 each): uniform_random observe 5, x_neg 3, x_zero 5, y_neg 7, y_zero 3, y_pos 5, xy 4. menu_coverage x_neg 12, x_zero 2, x_pos 12, y_neg 1, y_zero 1, y_pos 3, xy 1. local_information observe 11, x_neg 9, x_pos 8, y_neg 2, y_pos 2. target_risk observe 8, x_neg 8, x_pos 9, xy 7.

Saved greedy generations by snapshot (nonterminal): SFT 32 x_neg, 32 x_neg, 25 x_neg + 7 observe, 19 x_neg + 12 observe + 1 x_pos. Final 32, 31 + 1 x_pos, 30 + 2 x_pos, 30 + 2 x_pos. The 5 final set_x_pos outputs are all in worlds 202611070019 and 202611070020.

Command tokens (`PREP/llm-prepared.json`): shared prefix 4913, 1311, 842, 3252. Then observe 48033 + 5705, or set 746 followed by `_x` 3212, `_y` 4178 or `_xy` 40920, then `_neg` 28209, `_pos` 6479 or `_zero` 19359. Close 9207, end 151645. Stop is 4913, 9495, 788, 1866, 92, 151645 and is legal at every nonterminal state.

## 4b. The five explanation verdicts, after verification

| Explanation | Earlier verdict | My reading |
| --- | --- | --- |
| i. Supervised label imbalance | Partly supported, not sufficient | Agree. If anything weaker than stated: the last greedy decision rests on 32 against 29 labels, while the SFT scores favor `_neg` over `_pos` by 5.7 to 1 or more. Order of the single pass is an added untested contributor (3.4), and token-greedy decoding inflates the visible rate (3.5) |
| ii. Pairs dominated by one preferred action | Contradicted at command level | Agree. set_x_neg is preferred in 14/119 = 11.8%, below the uniform 12.5% |
| iii. Labels state-independent | Contradicted | Agree, and now with positive evidence: the signal is state-specific (3.2) and a constant ranking has no detectable signal (3.1) |
| iv. Label noise | Supported | Agree. 69% to 73% of strict labels are what the rule passes under no difference |
| v. Too few updates | SFT undetermined, DPO contradicted | Agree on DPO. For SFT the saved data lean toward "one pass did not fit": worse than a state-blind counter and no retention of its own examples (3.3). More passes remain untested |

No disagreement was found between the handoff of 2026-11-12, the results report of 2026-11-10 and the saved files on any training count (160 and 119 updates, 3,584 comparisons, 10,240 units, 9,145 responses).

## 5. What I could not check

- Whether the real network implements the token-bias mechanism. The fit is arithmetic on saved losses. The mechanism remains an inference.
- Base-model behavior on training prompts, and the full SFT command table. Not saved (see 3.6).
- Any deployment-side number quoted in the earlier summary (481/523, 576/576). Out of scope here.
- The earlier agent's Storey estimates and its other toy counterfactuals beyond those in 3.7.

## 6. Implications for the experimental agenda (my reading)

1. The preference data hold a state-specific signal that agrees with the SCM planner about two times in three, buried under labels that are about 70% chance-level. Supervision built from planner scores or from exact expected gains (closed form for linear-Gaussian worlds, using `private-world.json`) would remove most of that noise. Measuring label noise against exact expected gains is a zero-GPU analysis.
2. Before any retraining, regenerate the base and SFT command tables from saved adapters and prompts (3.6). This settles the base prior, the size of the token-greedy effect (3.5) and how much of the collapse precedes DPO.
3. Cheap controls that the saved evidence cannot replace: the unchanged recipe with shuffled SFT order and several seeds (3.4), with flipped or state-shuffled preference labels, and with command-level argmax decoding. Each is minutes of training on one 20 GB slice.
4. Any successor needs a training-set check that it beats a state-blind frequency counter (1.93 nats here) and shows a positive label-tracking statistic (3.3) before deployment scoring. The current recipe fails both on its own training states.
