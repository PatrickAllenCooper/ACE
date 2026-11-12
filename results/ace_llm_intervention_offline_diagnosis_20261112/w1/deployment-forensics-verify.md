# Independent verification of the deployment-forensics claims

Label: `deployment-forensics-verify`. Written 2026-11-12. Read-only on the project. All code is my own and lives in
`/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify/`.
I did not open the earlier agent's scripts or notes. No model weights were loaded, no episode was run, nothing was written under the repository.

## Bottom line

Every headline quantitative claim of the earlier agent reproduces from the raw records. I found no refuted claim.
Three items are only partially confirmed (a seed-dependent "identical" statement, the risk/NMSE ratio, one R2 value).
I also found evidence that qualifies four of the earlier agent's interpretations. Those qualifications matter for the agenda and are listed in the section "Evidence that qualifies or cuts against the earlier conclusions".

## Raw sources used

- GPU deployment records: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu`
  - `services/{frozen_llm,sft_llm,dpo_llm}/worker/generation-NNNN.json` (576 / 546 / 576 files)
  - `controller/{arm}-evidence.json`, field `joins[]` (ordinal to world_id and decision_index)
- CPU deployment records: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu`
  - `worker/artifacts/campaign/policy-0{0..4}/world-NN/episode/decision-NNNN.{input,process}.json`
  - `worker/artifacts/campaign/campaign.json` (24 plans, strata, public task, private mechanisms)
  - `worker/artifacts/cpu-phase/numeric-training-plan.json` (160 training records, exposure schedule)
- Training records beside deployment: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training` (`plan.json`, `reference-scores.json`, `step-NNNNN.loss.json`, adapters)
- Independent cross-check source: `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/{curves,cells,policies,actions}.csv`
- Source read: `/Users/pat/code/ACE/scripts/research/llm_intervention_{language_policy,grammar,task,target_risk,scm,numerical_policy,screen_policy,learning,model_service,screen_protocol}.py`. `git diff --stat a9346022f145 HEAD` on these files is empty, so the checkout equals the frozen scientific source for them.
- Tokenizer only (no weights): `/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json` (sha256 `aeb13307...`).

## How I verified

1. `v01_keys.py`: walked all 11,564 JSON files under the GPU tree and listed every key path.
2. `v03_cpu_dataset.py`, `v05_llm_dataset.py`: built my own table of all decisions. LLM states were recovered by decoding `prompt_ids` and checking `sha256(prompt) == prompt_sha256` and `sha256(canonical(envelope)) == envelope_sha256`.
3. `v13_curves_crosscheck.py`: my action sequences equal the action sequences in `curves.csv` for all 192 cells. My own NMSE recomputation from saved posterior means and the private mechanisms equals `curves.csv` to 2.2e-16 on all 4,371 decision states.
4. `vtr.py`: my own re-implementation of the one-step target-risk score (written from the math, repository not imported).
   - `v07_replay_tr.py`: with the saved decision seeds it reproduces all 560 saved target-risk decisions, 4,432 action scores, max absolute difference 1.1e-16, argmax identical in 560/560.
   - `v08_validate_vec.py`: a vectorised sampler agrees with saved 32-draw values (z-scores mean -0.02, sd 1.05).
   - `v09_reference_scores.py`: reference scores for all 4,203 unique states with 16,384 draws, plus five independent 256-draw estimates per state so that I can report how much any 256-draw statistic moves with the seed.
   - `v19_d0_highprec.py`: 262,144 draws for the 24 shared first states.
5. Per-claim scripts `v06, v10, v11, v12, v15 to v33`. Per-decision export for later agents: `.../deployment-forensics-verify/decisions_verifier.csv` (4,371 rows, reference scores for all eight actions, private NMSE, posterior risk).

## Claim-by-claim verdicts

### A. Do per-command score vectors exist? CONFIRMED absent for LLM arms

- No key in any of the 11,564 GPU-tree JSON files matches score, logit, logprob, prob, margin, likelihood, entropy, nll or loss, apart from unrelated module paths and LoRA `rank`.
- `generation-NNNN.json` has 22 keys: completion, cross_request_cache_supplied, cuda_interval_ms, elapsed_seconds, enable_thinking, envelope_sha256, environment_callbacks_available, legal_commands, longest_command_tokens, max_new_tokens, max_tokens, new_training_updates, ordinal, peak_gpu_allocated_bytes, peak_gpu_reserved_bytes, prompt_ids, prompt_sha256, prompt_tokens, sequence_ids, task_sha256, tensor_versions_unchanged, version. The earlier agent listed seven of them. The substance is right: none is a score.
- Source: `llm_intervention_language_policy.py:132-184` uses `GenerationConfig(do_sample=False, num_beams=1)` with `prefix_allowed_tokens_fn`. `check_legal_scores` only asserts finiteness and discards the logits. `llm_intervention_model_service.py:364-374` persists only the returned record.
- Per-action scores are saved at every decision for target risk (560 decisions, 32 draws per action, with `risk_before`, `risk_after`, `risk_reduction_per_cost`, `monte_carlo_standard_errors`), local information (570, `planning_metadata.scores`, 32 draws per action) and the ranker (576, `scores[]` aligned with `action_ids[]`, stop included). Menu coverage saves counts. Uniform random saves nothing but its seed. (`v28_score_presence.py`)
- Consequence: score-level state sensitivity of the LLM arms cannot be measured from deployment records. Only emitted commands can be analysed.
- New fact that matters for the agenda: the scores are recoverable without any new environment response. The local files `sft.adapter.pt` and `final.adapter.pt` in the training directory hash to `583b1659...` and `6f540dfe...`, which are exactly the adapter pins of the deployed `sft_llm` and `dpo_llm` services (`request-NNNN.json` identity). All 1,698 exact prompt token sequences are saved. A Qwen3-0.6B snapshot is in the local Hugging Face cache (its pin against `model_sha256 052d8c0b...` is not verified by me). I did not run it.

### B. Table of decisions. CONFIRMED exactly

Decisions per arm including stops: random 487, menu 480, local information 570, target risk 560, ranker 576, frozen 576, SFT 546, DPO 576. Total 4,371.
Actions: target risk obs 215, x- 129, x+ 76, y+ 62, xy 31, y- 29, y0 16, x0 2. Local information obs 318, x- 149, x+ 67, y+ 24, y- 7, xy 5. Ranker x+ 435, x- 141. Frozen obs 574, x0 2. SFT x- 481, x+ 23, y- 11, obs 8, stop 23. DPO x- 576.
1,698 of 1,698 LLM prompts decode to text whose SHA256 equals `prompt_sha256` and whose envelope equals `envelope_sha256`. 4,203 unique states. The only shared states are the 24 first-decision states, each shared by all eight arms. No `policy-05..07` directory exists under the CPU campaign, GPU campaign or finalization trees.
Command entropy in bits: random 2.995, menu 2.791, target risk 2.389, local 1.669, ranker 0.803, SFT 0.749 (0.519 without stops), frozen 0.033, DPO 0.

### C. First decision: how often is set_x_neg the target-risk optimum? CONFIRMED, with a tie caveat

Saved 32-draw scores (`v06_first_decision.py`): optimum x- 11, x+ 6, xy 5, y+ 1, y- 1. Best-minus-second gap median 0.00488, median relative gap 24.5%. Gap below 2 combined MC s.e. in 8/24. x- within 2 s.e. of best in 14/24.
High precision, 262,144 draws (`v19_d0_highprec.py`): optimum x- 11, x+ 6, xy 5, y+ 1, obs 1. The count 11 is stable but the membership is not. World 2 leaves the x- set and world 14 enters it. World 20 is a three-way tie.
Reason: in fork worlds with the uniform cost menu (worlds 2, 8, 14) `set_x_neg` and `set_xy` are exactly equivalent for the target Z, because Y does not feed Z. The deployed choice between them is Monte Carlo noise.
So x- is the unique or exactly tied optimum in 13/24 first states, within 5% of the best in 14/24, and loses 32% to 93% of the attainable one-step reduction in the other 10.
Under five independent 256-draw estimates the x- count is 12, 10, 13, 12, 12 (`v29_d0_256.py`). The earlier agent's statement that the 256-draw distribution is "identical" to the 32-draw one is therefore luck of its seed. PARTIALLY CONFIRMED for that sub-statement only.
x- is the best constant action at the first decision: mean relative regret 0.257, next best x0 0.428, xy 0.485.

### D. First-decision agreement with target risk. CONFIRMED

| arm | hits vs saved 32-draw optimum | mean regret, saved scores | relative regret, saved | hits vs 262k-draw optimum | relative regret, 262k |
| --- | ---: | ---: | ---: | ---: | ---: |
| target risk | 24 | 0 | 0 | 21 | 0.000 |
| local information | 17 | 0.00091 | 0.042 | 17 | 0.042 |
| ranker | 14 | 0.00254 | 0.160 | 14 | 0.159 |
| DPO = SFT = menu coverage | 11 | 0.00920 | 0.260 | 11 | 0.257 |
| uniform random | 6 | 0.01720 | 0.465 | 4 | 0.463 |
| frozen | 0 | 0.02796 | 0.602 | 1 | 0.606 |

First actions: DPO, SFT and menu coverage choose x- in all 24 worlds. Frozen chooses observe in all 24. Ranker chooses x- in 21 and x+ in 3 (worlds 4, 12, 15), and x+ is the optimum in all three.
Menu coverage's first action is x- by its alphabetical tie rule, so "11 hits" for menu coverage is an artifact of the same constant.

### E. Own-trajectory regret under the target-risk score. CONFIRMED

Mean relative regret, stops scored as zero gain:

| arm | earlier agent, 256 draws | mine, five 256-draw seeds | mine, 16,384 draws | chose optimum, mine | within 2 s.e., mine (256) |
| --- | ---: | ---: | ---: | ---: | ---: |
| target risk | 0.016 | 0.015 to 0.017 | 0.013 | 66.4 to 70.2% | 90.0 to 91.8% |
| local information | 0.154 | 0.153 to 0.155 | 0.152 | 34.4 to 35.8% | 47.7 to 49.6% |
| menu coverage | 0.441 | 0.440 to 0.442 | 0.441 | 17.1 to 18.8% | 23.3 to 24.4% |
| uniform random | 0.504 | 0.503 to 0.505 | 0.502 | 11.9 to 14.2% | 17.7 to 18.7% |
| ranker | 0.575 | 0.574 to 0.575 | 0.574 | 8.9 to 10.4% | 11.6 to 12.3% |
| frozen | 0.655 | 0.655 to 0.657 | 0.654 | 3.1 to 4.0% | 6.8 to 8.2% |
| SFT | 0.665 | 0.664 to 0.665 | 0.663 | 8.2 to 8.4% | 10.3 to 10.8% |
| DPO | 0.700 | 0.698 to 0.700 | 0.699 | 6.9 to 7.6% | 8.9 to 9.5% |

- DPO worse than random in 20/24 worlds, Wilcoxon p 0.0005 to 0.0007. Decisions 0-3: DPO 0.337 vs random 0.475. DPO by block 0-3, 4-7, 8-15, 16-23: 0.337, 0.584, 0.775, 0.862. All confirmed.
- x- optimal on DPO's own states: 38/576 at 16,384 draws (by decision 11, 9, 7, 6, 3, 2, then none). 40 to 44 across 256-draw seeds, with a single noise hit at decision 6 in two of five seeds. Claimed 39. Confirmed within Monte Carlo noise.
- x- ranks 7th or 8th of 8 on DPO states: 382/564 full-menu states (375 to 382 across seeds). Claimed 381. Confirmed.
- Target-risk arm: saved top gap below 2 combined s.e. in 419/560 = 74.8%. Its 32-draw choice equals my large-draw argmax in 66.6%. Confirmed. The cost of that noise is small by its own criterion (relative regret 0.013).
- Spearman between arm-level regret and mean capped cost: 0.929. Confirmed.
- NMSE increases after 1,909 of 4,348 single acquisitions = 43.9%. Confirmed.
- Posterior risk vs private NMSE: I get a ratio of means of 1.008 over the 4,371 decision states (1.023 over unique states, 0.932 on target-risk states). Claimed 0.994. PARTIALLY CONFIRMED: about one on average, exact figure not reproduced. Arm-level ratios range from 0.74 (random) to 1.26 (SFT), and the per-state Spearman is 0.64.

### F. Supervised adapter and frozen model. CONFIRMED exactly

- 42 non-x- acquisitions: x+ 23, y- 11, obs 8. Earliest at decision 10. 24/68 at remaining budget 11-13 vs 18/455 elsewhere, Fisher p 6.4e-13.
- y- in 8/8 collider worlds, 3/8 chain, 0/8 fork. Observe only in colliders (worlds 4, 5, 22).
- 23 stops, at remaining 1, 2, 3 in 4/5, 14/19, 5/24 decisions, and 0/498 at remaining 4 or more. Five stopped runs end above the target (worlds 10, 11, 12, 19, 21). World 17 is the only run without a stop. Paid cost 721.
- No cost-menu effect: y- 5 uniform vs 6 heterogeneous (Fisher p 1.0). Non-x- 23 vs 19 (p 0.52).
- Frozen: both `set_x_zero` are in world 15 (fork, heterogeneous, shifted_low), decisions 7 and 10, service ordinals 367 and 370.

### G. Ranker. CONFIRMED

x+ 435, x- 141. Two distinct rank orders over the 564 full-menu decisions (423 times x+ > x0 > x- > obs > xy > y- > y0 > y+ > stop, and 141 times the same with x- and x+ swapped). Top-two margin median 0.0031. State-independent action effect explains 98.1% of centred score variance over the eight acquisitions. Spearman of the x-/x+ contrast with the target-risk contrast at decision 0: 0.846. Sign agreement on full-menu decisions 12 and later: 27 to 31 of 276 depending on seed (claimed 30).

### H. Prompts, latency, token tree. CONFIRMED

Prompt tokens 1,990 to 4,160, median 2,998. Tokens added per response: 83 for SFT and DPO, 91 for frozen (observe rows are longer), 85.7 pooled. Posterior summary share 19.2% at decision 0 and 9.9% at decision 23. History share 35.5% and 67.2%. One 2x2 covariance: median 81 tokens, range 76 to 87 (the earlier "78" is one example). Median GPU ms per decision: frozen 424.6, SFT 536.0, DPO 709.6.
Token tree from the tokenizer and from the recorded command ids in the training plan: first branch `action` (8 commands) vs `stop`. Second branch `observe` (1) vs `set` (7). Then `_x` (3), `_xy` (1), `_y` (3). Then `_neg`, `_pos`, `_zero`.

### I. Training exposure. CONFIRMED

Labels: stop 32, x- 32, x+ 29, obs 24, xy 12, y+ 10, y- 10, x0 7, y0 4. All 32 stop labels sit on terminal states where stop is the only legal command. Five snapshots per world at nominal remaining budget 24, 18, 12, 6, 0 (exact values 24, 18 or 17, 12 or 11, 6 or 5, 0). Pairs 3,584, tied 2,897, strict 687, exposed 119. Exposed pairs chose x- 14 times and rejected it 9 times.
Labels equal my 16,384-draw target-risk optimum in 56/128 states (claimed 55). Strict pairs agree with the target-risk ordering in 470/687 = 68.4% (claimed 68.6%). Exposed pairs 69/119 = 58.0% (claimed 58.8%). Differences are one item each and within Monte Carlo noise.

### J. DPO stage. CONFIRMED in substance, one number PARTIALLY CONFIRMED

`beta = 0.1` (`plan.json numeric_recipe`, `llm_intervention_learning.py:103`). Inverting the logged single-pair loss gives the log-odds shift relative to the SFT reference at first exposure. Largest absolute shift 15.91 nats. The policy at first exposure prefers the labelled command in 66/119 pairs. The SFT reference prefers it in 66/119. No improvement by third of the DPO phase (22 vs 24, 20 vs 18, 24 vs 24). Three pairs flipped toward the label and three away.
Per-command drift model: correlation of fitted per-command drift with SFT mean command log-likelihood is 0.91 (claimed 0.89). My R2 is 0.60 for constant drift, 0.73 with a per-command linear trend, 0.69 to 0.75 on late windows. I could not reach the claimed 0.84 with these specifications.

## Evidence that qualifies or cuts against the earlier conclusions

1. **"The supervised adapter IS state-dependent" is true only for the budget counter and graph type.** (`v23_sft_triggers.py`, `v21_ranker_sft_extra.py`)
   - x+ is emitted exactly once per world in 23 of 24 worlds, after 10 to 18 consecutive x- (mode 12 to 13), and the policy then returns to x-.
   - On SFT states at decision 10 or later, the reference score prefers x+ over x- in 275/283 states. At remaining budget 11-13 it prefers x+ in 15/16 states where SFT chose x+ and in 42/44 states where SFT chose x-. The timing of x+ carries no posterior information (Mann-Whitney p 0.51 on risk and on the x+ advantage).
   - Stops are keyed to remaining budget of 3 or less. In 6 of 23 stops the public posterior risk was above the 0.04 tolerance (0.043, 0.050, 0.245, 0.124, 0.066, 0.069). At every stop the best available action still had positive expected gain.
   - The deviations did help a little: the chosen action beats x- on the reference score in 36/42 deviations (x+ 22/23, obs 8/8, y- 6/11), but it is the optimum in only 5/42.
   - Reading (inference): SFT learned a budget-keyed schedule anchored near the training snapshot levels 12 and 6. That is not evidence of posterior use.

2. **The headline endpoint cannot resolve SFT from DPO. Measurement noise alone moves it by several units.** (`v27_noise_floor.py`)
   - Each arm has its own measurement stream. In world 0 six arms take x- first and receive six different responses.
   - SFT and DPO issue identical actions for their first 10 to 22 acquisitions in every world. Their capped cost-to-target still differs in 11/24 worlds, and in 8 of those 11 the difference was decided by a crossing inside the identical-action prefix. Examples: world 21 SFT 32 vs DPO 12, world 16 SFT 10 vs DPO 22, world 10 SFT 23 vs DPO 13, world 5 SFT 11 vs DPO 18.
   - Those eight noise-only differences sum to 54 cost units, 2.25 per world averaged over 24 worlds. The reported SFT minus DPO difference in means is 0.54.
   - After ten identical x- acquisitions the two arms' NMSE differ by 0.0215 on average and fall on opposite sides of the 0.04 target in 3 worlds.
   - Ten worlds already meet the target after initialization (worlds 0, 1, 2, 6, 7, 8, 9, 15, 18, 20), so only 14 worlds carry endpoint information.

3. **The regret proxy tracks capped cost but is weaker against other outcomes.** (`v20_regret_details.py`)
   - Spearman of arm-level regret with mean capped cost 0.929, with ever-attained 0.84, with mean final NMSE 0.69. n is 8 arms.
   - Random and frozen are mis-ordered: random has lower regret (0.50 vs 0.65) but worse capped cost (14.25 vs 13.63) and worse final NMSE (0.041 vs 0.020). An all-observe policy looks poor on one-step regret yet ends with the fourth-best final error.
   - DPO is not distinguishable from frozen on regret (worse in 12/24 worlds, p 0.79). DPO is worse than SFT in 19 to 22 of 24 worlds (p 0.0003 to 0.001).
   - The proxy is the planner's own working-model criterion, so target risk scores near zero by construction. It is myopic. It should be validated before it is promoted to a primary endpoint.

4. **The ranker is not "equally state-blind" early on.** At decision 0 it deviates to x+ in three worlds and is right in all three. In decisions 0-3 its relative regret is 0.219 against 0.337 (DPO) and 0.340 (SFT). With the same supervision, the numeric encoding expressed a correct state-dependent first choice that neither trained LLM expressed. Later it follows a step-indexed switch: x- chosen in 21, 19, 19, 16, 15, 16, 15, 5, 5, 4, 3, 3 worlds at decisions 0 to 11 and never afterwards, while the reference prefers x- over x+ in 260/288 of its late states.

5. **The logged DPO drift does not by itself explain "x- only".** (`v17_dpo_stage2.py`, `v31_stop_pairs.py`)
   - Shift toward observe is positive in 19/19 exposed pairs that contain observe (mean +6.7 nats), although observe was the rejected command in 13 of them. Shift toward x- is positive in 20/23 (mean +3.3). Shift toward x+ is positive in only 13/37 (mean -1.5) although x+ was the chosen command in 23.
   - The single direct observe vs x- pair (step 225) moved 1.37 nats toward observe.
   - Yet the deployed DPO adapter emitted observe 0/576 times, while SFT emitted it 8 times.
   - Stop appears in 0 of 3,584 preference pairs, yet DPO removed every stop (SFT stopped in 23/24 worlds, DPO in 0/24). The DPO stage therefore changed behaviour that no preference label addressed.
   - Candidate explanations that the records cannot separate: greedy token-tree pooling (`set` carries the mass of seven commands against `observe`), off-distribution deployment states, or later updates. This needs command scores on the saved deployment prompts.

6. **Position, identity and tie rule are perfectly confounded for set_x_neg.** (`v30_menu_order.py`)
   - The menu order is identical in all 4,371 deployment states and all 160 training records: obs, x-, x0, x+, y-, y0, y+, xy. The task payload, objective and domain notes are byte-identical everywhere.
   - Frozen emits the first menu entry. SFT and DPO emit the first listed intervention, which is also the alphabetically first `set_*` id and therefore the winner of every numerical tie rule.
   - Protocol source `llm_intervention_screen_protocol.py:126-132,188`: at each snapshot 8 of 32 training worlds are labelled by each of uniform random, menu coverage, local information and target risk. Menu coverage always emits x- at the first state. This is consistent with x- being the label in 16/32 first-snapshot records while my reference optimum is x- in 11/32. (Teacher assignment read in source, not recomputed per record.)

7. **Deployment states leave the training distribution after about ten repeats.** (`v25_pins.py`) The longest run of one repeated acquisition in any of the 160 training histories is 10, and 129 of 160 have runs of 3 or less. DPO's states at decision 11 or later (312/576) contain 11 to 23 consecutive x- rows. "Never adapts" is therefore partly a statement about off-distribution inputs. (Inference from verified counts.)

8. **"DPO starts right" is generous.** x- is the unique or tied optimum in 13/24 first states and a large loss in 10. By decisions 4-7 the reference optimum on DPO's own states is x- in only 5/96, and from decision 8 it is x+, observe or y+ in 358/384.

## Caveats on my own work

- Relative regret is (best minus chosen) / best under the one-step working-model score. No state had a non-positive best score.
- 256-draw statistics depend on the seed. I report ranges over five seeds and a 16,384-draw reference. I could not reuse the earlier agent's seeds, so agreement is judged within those ranges.
- The DPO shift is an exact inversion of the logged loss under batch size one and beta 0.1. It is a pre-update value for each pair. I did not observe any post-update or final-model score.
- The teacher mixture and the tokenizer identity rest on source reading and functional checks (all 1,698 prompt hashes match, command token ids equal those recorded in the training plan). The plan's `tokenizer_sha256` field equals the model-set pin `052d8c0b...`, not a per-file hash.
- I ran `git diff --stat`, `git log -1` and `git status --short` only. Final `git status` equals the initial snapshot.

## Files written (scratch only)

- Notes: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify.md`
- Scripts `v01` to `v33`, `vcommon.py`, `vtr.py` and intermediate pickles in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify/`
- Per-decision table: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify/decisions_verifier.csv`
