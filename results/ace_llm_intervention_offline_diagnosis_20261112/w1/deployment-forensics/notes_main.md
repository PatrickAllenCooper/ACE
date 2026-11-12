# Deployment forensics of the 24-system ACE intervention comparison

Label: `deployment-forensics`. Prepared 2026-11-12. Read-only offline analysis with `/usr/bin/python3` (3.9). No model weights were loaded, no inference, no episode, no SLURM/SSH. Nothing under `/Users/pat/code/ACE` was modified (git status identical before and after).

All scripts and outputs are in
`/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics/`
(scripts `s01_*.py` to `s22_*.py`, outputs in `out/*.out`, pickles in `out/*.pkl`). Verbatim outputs of the main scripts are appended at the end of this file.

Abbreviations: TR = direct SCM target-risk planner. x-, x0, x+ = set X to -1, 0, +1. y-, y0, y+ = set Y. xy = joint X=-1, Y=+1. obs = observe_all. "d" = policy decision index within an episode (0 = first decision after the 8 shared initial observations).

## 0. Headline findings

1. **No per-command score is saved for any LLM arm at deployment.** Only the greedy constrained completion and the full prompt token ids are saved. Item (2) of the brief cannot be answered at score level from deployment records. Scores ARE saved for the numerical ranker, the target-risk planner and local information.
2. **The exact public state of every LLM decision is recoverable** by decoding the saved `prompt_ids` with the pinned tokenizer vocabulary. I verified all 1,698 decodes against the saved `prompt_sha256` and `envelope_sha256` (zero mismatches). This gives a unified table of all 4,371 decisions of all 8 arms with full states.
3. **At the shared first decision set_x_neg is the target-risk optimum in 11 of 24 systems** and the best constant action. DPO, SFT and menu coverage all choose it 24/24. So the DPO constant is the modal right answer at the start and becomes wrong later: on DPO's own states x- is TR-optimal in 39/576 decisions (11, 8, 9, 6, 3, 2 at d=0..5, then 0 for d>=6), and it is ranked 7th or 8th of 8 in 381/576.
4. **The supervised adapter is measurably state-dependent; DPO removed that.** SFT deviates from x- mostly when the remaining budget is 11 to 13 (24/68 vs 18/455 elsewhere, Fisher p = 6e-13), uses y- in 8/8 collider worlds and 0/8 fork worlds, and stops only when remaining budget <= 3. These patterns line up with the training snapshot budget levels (24, 18, 12, 6, 0) and with label statistics. They are shallow, but they prove that state information reaches the command.
5. **The numerical ranker is almost as state-blind as the LLMs**, with direct numeric state access. 98.1% of its centred score variance is a state-independent action effect; it produces only 2 distinct rank orders in 564 decisions; best-minus-second margin has median 0.0031.
6. **Decision-level regret under a 256-draw target-risk score separates the arms far better than cost-to-target.** Mean relative regret: TR 0.016, local information 0.154, menu 0.441, random 0.504, ranker 0.575, frozen 0.655, SFT 0.665, DPO 0.700. DPO is worse than uniform random in 20/24 worlds (Wilcoxon p = 0.0006), where the headline endpoint gave 3 lower, 13 tied, 8 higher.
7. **Training-side records saved next to the deployment data explain much of the behaviour** (supplement, section 10): 32 of 160 supervised updates are `{"stop":true}` on terminal states where stop is the only legal command; supervised labels equal the 256-draw TR optimum in only 55/128 states; the DPO stage did not improve first-exposure preference accuracy (66/119 before, 66/119 during) and instead applied a state-independent sharpening of the SFT command ordering (per-command drift correlates r = +0.89 with SFT mean log-likelihood).

## 1. Layout and what is recorded per decision

### 1.1 Locations (verified by listing)

| Content | Path |
| --- | --- |
| Numerical arms, per-world episodes | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/campaign/policy-{00..04}/world-{00..23}/episode/` |
| Campaign definition incl. private worlds, strata, menus | `.../cpu/worker/artifacts/campaign/campaign.json` |
| Duplicate per-call custody for numerical arms (2,673 dirs) | `.../cpu/worker/artifacts/numeric-calls/<sha>/{input.utf8,stdout.log,stderr.log,command.json,before.json,after.json,terminal.json}` |
| Matched ranker training plan with all 160 raw training records | `.../cpu/worker/artifacts/cpu-phase/numeric-training-plan.json` (key `raw_records_utf8`, `exposure`) |
| Ranker training steps | `.../cpu/worker/artifacts/cpu-phase/ranker/` |
| LLM service records | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services/{frozen_llm,sft_llm,dpo_llm}/worker/` |
| Map from LLM service ordinal to world and decision index | `.../gpu/controller/{frozen_llm,sft_llm,dpo_llm}-evidence.json`, key `joins[]` |
| 192 cost reports (curves, NMSE, predictions) | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque/cpu/cpu/worker/artifacts/campaign/private/world-NN/arm-NN/cost-report.json` |
| Display data | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/{display-data.json,curves.csv,actions.csv,cells.csv,policies.csv}` |
| LLM training step losses and SFT reference scores (supplement) | `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training/{reference-scores.json,step-NNNNN.loss.json,plan.json}` |

Policy order (`campaign.json: policy_ids`): 0 uniform_random, 1 menu_coverage, 2 local_information, 3 target_risk, 4 learned_ranker, 5 frozen_llm, 6 sft_llm, 7 dpo_llm. World index w = 0..23 maps to world_id 202611070100 + w. Strata cycle: regime block of 6 (matched_low 0-5, matched_high 6-11, shifted_low 12-17, shifted_high 18-23), within a block graph = chain, chain, fork, fork, collider, collider, and cost = uniform (even w), heterogeneous (odd w).

Discrepancies with the brief:
- File counts are larger than stated: 35,582 files under `.../launch/cpu` (campaign 15,030, numeric-calls 18,711, cpu-phase 1,696) and 11,564 under `.../launch/gpu` (services 10,271, controller 1,181). The brief said about 7,100 and 2,100.
- **The LLM arms' episode directories (policy-05..07 with `decision-NNNN.input.json` and `journal.jsonl`) are not in the local copies.** The local CPU custody was collected at the CPU phase and has policy-00..04 only. The GPU custody has service records only. `dpo_llm-evidence.json: recorded_campaign_root` points to `/gpfs/alpine1/scratch/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu/worker/artifacts/campaign`. This did not block the analysis (see 1.3).

### 1.2 Fields per decision

Numerical arms. Two files per decision:
- `decision-NNNN.input.json`: the complete public envelope `{state, task, version}`. `state` has `graph`, `menu`, `history` (every paid response with values), `scm` (per node `parameter_mean`, `parameter_covariance`, `noise_variance`, `eligible_rows`, `terms`), `remaining_units`, `spent_units`, `requests`, `returned_responses`, `terminal`, `objective`, `domain_notes`.
- `decision-NNNN.process.json`: `command`, `input_sha256`, `returncode`, `stdout` (the chosen command JSON), `stderr` (a JSON receipt), `elapsed_seconds`.

Per-action scores inside the `stderr` receipt:

| Arm | Per-action scores saved at every decision? | Field |
| --- | --- | --- |
| target_risk | Yes, every feasible action | `planning_metadata.risk_reduction_per_cost{action}`, plus `risk_after{action}`, `risk_before`, `monte_carlo_standard_errors{action}` (32 fantasy draws) |
| local_information | Yes | `planning_metadata.scores{action}` |
| learned_ranker | Yes, incl. stop | top-level `scores[]` aligned with `action_ids[]` (sorted ids then `null` = stop), also `feature_width` 8891, `encoding` |
| menu_coverage | Counts only | `planning_metadata.counts{action}` |
| uniform_random | No | seed only |

LLM arms. Per request ordinal NNNN in `services/<arm>/worker/`:
- `request-NNNN.json`: `envelope_sha256`, `identity`, `ordinal`.
- `raw-generation-NNNN.json`: `token_prefix` (returned token ids), `shape`, `raw_sha256`.
- `generated-NNNN.unaccepted.json` and `generation-NNNN.json`: `completion`, `prompt_ids`, `sequence_ids`, `prompt_tokens`, `prompt_sha256`, `envelope_sha256`, `task_sha256`, `legal_commands`, `longest_command_tokens`, `max_tokens`, `max_new_tokens`; `generation-` adds `elapsed_seconds`, `cuda_interval_ms`, `peak_gpu_allocated_bytes`, `peak_gpu_reserved_bytes`, `ordinal`.
- `phase-NNN.json`: timestamps.

**I enumerated every key of every worker record for the three arms (`s01_llm_keys.py`). No key matches score, logit, logprob, likelihood, prob, margin or entropy.** Reason in code: `generate_task_decision` in `/Users/pat/code/ACE/scripts/research/llm_intervention_language_policy.py` lines 132-184 runs greedy `model.generate` with `prefix_allowed_tokens_fn` and returns only the sequence. Request counts: frozen 576, SFT 546 (523 responses + 23 policy stops), DPO 576.

### 1.3 State reconstruction for LLM arms

The prompt is `task_prompt(envelope)` = fixed preamble + canonical JSON of the envelope + `"\nDecision:\n"` (`llm_intervention_task.py` lines 144-152) inside the Qwen chat template with thinking disabled. The saved `prompt_ids` are the whole prompt. Decoding them with the local tokenizer file (`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de2.../tokenizer.json`, SHA256 `aeb13307...` equal to the pin in `inputs/dpo_llm.freeze.json: model_files`) reproduces the envelope. Checks passed for all 1,698 LLM decisions: decoded user text SHA256 == saved `prompt_sha256`; canonical envelope SHA256 == saved `envelope_sha256`. Only the tokenizer vocabulary was used. Exactly one preamble variant occurs.

Unified dataset: `out/decisions.pkl`, 4,371 decisions (uniform 487, menu 480, local 570, TR 560, ranker 576, frozen 576, SFT 546, DPO 576). Cross-checks: action sequences equal the `curve[].action_id` sequences of all 192 cost reports (0 mismatches) and reproduce `actions.csv` exactly.

Shared-state facts (verified by envelope hash): all 8 arms see the identical state at d=0 in all 24 worlds. **No state is shared between any two arms after d=0** (4,203 unique states = 4,371 minus the 168 d=0 duplicates). Arms that choose the same first action still diverge at d=1 because each arm has its own measurement stream (`measurement_bindings` are distinct per arm in the paired plan). So d=0 is the only clean matched comparison.

## 2. LLM score vectors (brief item 2)

Not saved at deployment, for any of frozen, SFT, DPO. Margins, between-state variance of command scores, rank changes and correlations with state features cannot be computed for the LLMs from the deployment records. What the records do support:

(a) Input side. All prompts are distinct (576/576, 546/546, 576/576 distinct token sequences), so different states did survive serialization and tokenization.

(b) Behavioural state sensitivity for every arm (`s12_behavioral_sensitivity.py`). Entropy of the chosen command H(A) and permutation-bias-corrected mutual information with public state features, in bits:

| Arm | n | H(A) | I(A; graph) | I(A; cost menu) | I(A; step tercile) | I(A; remaining units) | distinct sequences over 24 worlds | switches per world |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_random | 487 | 2.995 | -0.007 (p=0.79) | 0.001 | 0.012 | -0.013 | 24 | 17.08 |
| menu_coverage | 480 | 2.791 | -0.019 | -0.006 | 0.022 | 2.008 | 2 | 19.00 |
| local_information | 570 | 1.669 | 0.089 (p=0.003) | 0.020 (p=0.003) | 0.062 | 0.123 | 24 | 12.88 |
| target_risk | 560 | 2.389 | 0.091 (p=0.003) | 0.158 (p=0.003) | 0.039 | 0.067 | 24 | 14.83 |
| learned_ranker | 576 | 0.803 | -0.002 (p=0.71) | -0.001 | 0.359 | 0.399 | 9 | 1.12 |
| frozen_llm | 576 | 0.033 | 0.002 (p=0.34) | 0.002 | -0.001 | -0.000 | 2 | 0.17 |
| sft_llm | 546 | 0.749 | 0.028 (p=0.003) | 0.001 (p=0.31) | 0.147 (p=0.003) | 0.241 (p=0.003) | 19 | 3.71 |
| dpo_llm | 576 | 0.000 | 0 | 0 | 0 | 0 | 1 | 0.00 |

Reading: at the level of the emitted command, supervised training increased state dependence relative to the frozen base (H 0.033 to 0.749 bits; significant dependence on graph, step and remaining budget). The DPO stage removed all of it. Whether DPO scores still vary with the state below the argmax is unknown from these records.

(c) Training-side LLM scores exist and are analysed in section 10.

What a score diagnostic needs: the 1,698 deployed prompts are saved as token ids, so frozen, SFT and DPO can be scored on all 9 commands at exactly the deployed inputs with no re-serialization risk. From the recorded latencies (median 0.42 to 0.71 s per ~3,000-token prompt) a naive pass over 3 models x 1,698 prompts is roughly 50 to 70 minutes of one 20 GB MIG slice, less with batching (my estimate, not a measurement).

## 3. Supervised arm (brief item 3)

Per-world action strings (N = x-, P = x+, a = y-, O = observe, # = stop), from `s09_sft_frozen.py`:

```
 0 chain    unif  matched_low   NNNNNNNNNNNNPaNNNNNNN#      12 chain    unif  shifted_low   NNNNNNNNNNNNNNNNNNNNNN#
 1 chain    heter matched_low   NNNNNNNNNNNNPNNNNaNNN#      13 chain    heter shifted_low   NNNNNNNNNNNNPNNNNNNNNNN#
 2 fork     unif  matched_low   NNNNNNNNNNNNPNNNNNNNNN#     14 fork     unif  shifted_low   NNNNNNNNNNNNNNNNNPNNNN#
 3 fork     heter matched_low   NNNNNNNNNNNNNPNNNNNNN#      15 fork     heter shifted_low   NNNNNNNNNNNNNPNNNNNNNN#
 4 collider unif  matched_low   NNNNNNNNNNNaNNNNPNOOO#      16 collider unif  shifted_low   NNNNNNNNNNNaPNNNNNNNNN#
 5 collider heter matched_low   NNNNNNNNNNNNaPNNNNONN#      17 collider heter shifted_low   NNNNNNNNNNNNaNPNNNNNNNN   (no stop)
 6 chain    unif  matched_high  NNNNNNNNNNNNNPNNNNNNN#      18 chain    unif  shifted_high  NNNNNNNNNNNNPNNNNNNNN#
 7 chain    heter matched_high  NNNNNNNNNNNNPNNNNaNNNN#     19 chain    heter shifted_high  NNNNNNNNNNNNNPNNNNNNNN#
 8 fork     unif  matched_high  NNNNNNNNNNNNPNNNNNNNNN#     20 fork     unif  shifted_high  NNNNNNNNNNNNNNNNNNPNNN#
 9 fork     heter matched_high  NNNNNNNNNNNNNPNNNNNNNN#     21 fork     heter shifted_high  NNNNNNNNNNNNNNNNNNPNNN#
10 collider unif  matched_high  NNNNNNNNNNNaNNNNPNNNNNN#    22 collider unif  shifted_high  NNNNNNNNNNNPaNNNNNOOOO#
11 collider heter matched_high  NNNNNNNNNNaPNNNNNNNNN#      23 collider heter shifted_high  NNNNNNNNNNNPaNNNNNNNNN#
```

The 42 non-x- acquisitions: x+ 23, y- 11, obs 8.
- By decision index: d=10 (1), 11 (6), 12 (12), 13 (7), 14 (1), 16 (2), 17 (3), 18 (5), 19 (2), 20 (2), 21 (1). None before d=10.
- By remaining budget: 24 of 68 decisions at remaining 11-13 are non-x- (35.3%) against 18 of 455 elsewhere (4.0%), Fisher exact p = 6.4e-13. At remaining 12 the split is 11 x- vs 13 other. Observes occur only at remaining 3-6.
- By graph: chain x+ 7, y- 3. Fork x+ 8 and nothing else. Collider x+ 8, y- 8, obs 8. Worlds with at least one y-: collider 8/8, chain 3/8, fork 0/8 (collider vs fork Fisher p = 0.00016). Observe only in collider worlds 4, 5, 22.
- By cost menu: uniform x+ 11, y- 5, obs 7. Heterogeneous x+ 12, y- 6, obs 1. y- was chosen 6 times at cost 2 and 5 times at cost 1, so no visible cost sensitivity.
- Every world except world 12 has at least one deviation.
- Under the 256-draw TR score at the same state, the chosen action beats x- in 36/42 (x+ 22/23, obs 8/8, y- 6/11). This is not evidence of good timing: at all SFT states with d>=6, x+ beats x- in 93.7% and obs beats x- in 96.8%. Mean relative regret is 0.469 at the 42 deviations against 0.813 at SFT x- decisions with d=10..21.
- The y- pattern matches the training labels, not the planner: supervised labels contain y- in chain 5, collider 5, fork 0 (section 10). Under TR, y- is relatively more valuable in fork worlds on SFT states (score/best 0.614) than in collider (0.331) or chain (0.269).

Stops: 23, one per world except world 17. Decision index 21 (8 worlds), 22 (13), 23 (2). History length 29 (8), 30 (13), 31 (2). Remaining units 3 (5), 2 (14), 1 (4). P(stop | remaining) = 4/5 at 1, 14/19 at 2, 5/24 at 3, 0/498 at 4 or more. By graph chain 8, fork 8, collider 7. By cost uniform 12, heterogeneous 11. By regime 6, 6, 5, 6. Legal commands at stop 9 (20 cases) or 5 (3 cases). Prompt tokens 3,757 to 3,954. Unspent budget 47 units in total (768 - 721). Five stopped runs are above target at their final state: worlds 10, 11, 12, 19, 21 (final NMSE 0.0493, 0.0472, 0.0917, 0.1991, 0.0583). The public posterior integrated risk at the stop state is <= 0.04 in 17/23. Stop index correlates with posterior risk at stop (Spearman +0.54, p = 0.01, n = 23): a weak hint that later stops go with higher uncertainty, confounded with cost structure, not established. Full table in the appendix (`s21`).

Inference (strongly supported, not proven): the stops come from the 32 supervised `{"stop":true}` examples, all on terminal states with remaining 0 where stop is the only legal command (section 10). The deployed episode loop never calls a policy in such a state (`llm_intervention_episode.py` lines 100-106). The adapter generalised "budget near zero or long history" to remaining 1-3.

## 4. Frozen arm (brief item 4)

574 observe_all and 2 set_x_zero. Both set_x_zero are in world 15 (202611070115, fork, heterogeneous, shifted_low), at d=7 (requests 15, remaining 17, service ordinal 367, 2,637 prompt tokens) and d=10 (requests 18, remaining 14, ordinal 370, 2,904 tokens). The history before d=7 is 15 observations. World 15 has posterior mean of X about -1.04 and several observed Y values below -3.6. **How close the other commands were cannot be determined: no scores are saved.** Under TR, observe is optimal in 18/576 frozen states (3.1%); mean relative regret 0.655; at d=0 observe is never optimal (0/24) and has mean relative regret 0.611, worse than the uniform-random expectation.

## 5. Numerical ranker (brief item 5)

Scores are saved at all 576 decisions (`s08_ranker.py`). 564 decisions have all 9 candidates, 12 have 5 (remaining 1 in heterogeneous menus).
- Action distribution: x+ 435 (75.5%), x- 141 (24.5%). Never observe, never a Y action, never stop.
- Only **2 distinct complete rank orders** in 564 decisions: `x+ > x0 > x- > obs > xy > y- > y0 > y+ > STOP` (423) and `x- > x0 > x+ > obs > xy > y- > y0 > y+ > STOP` (141). Second place is always x0, stop always last.
- Margin best minus second: min 0.00002, p10 0.00059, median 0.00307, mean 0.00292, p90 0.00513, max 0.00593 (raw network output). Softmax over the 9 scores is near-flat: mean probabilities x+ 0.1293, x0 0.1290, x- 0.1287, obs 0.1196, xy 0.1144, y- 0.1123, y0 0.1098, y+ 0.1075, stop 0.0495.
- Between-state variance vs between-action gaps (scores centred per decision over the 8 actions): SD of action means 0.0766; RMS between-state SD 0.0100; ratio 0.131. **Action main effect explains 98.09% of the centred-score variance; the state-by-action part is 1.91%.** Raw scores also carry a large state offset common to all actions (SD about 0.94), irrelevant to the choice.
- The only state-dependent contrast is x- vs x+: mean -0.0042, SD 0.0054. It correlates with decision index (r = -0.79), number of earlier x+ (r = -0.82), posterior mean of X (r = +0.37), covariance traces of Y and Z (+0.45, +0.46). Not with graph, cost menu or regime (I(A; graph) and I(A; cost) are zero within noise).
- Behaviour: x- for the first steps then x+ forever. x- counts by decision index: 21, 19, 19, 16, 15, 16, 15, 5, 5, 4, 3, 3, then 0 for d>=12. Typical world: `NNNNNNNPPPPPPPPPPPPPPPPP`. Worlds with negative posterior mean of X start with or switch early to x+ (worlds 4, 12, 15 are all x+).
- At d=0 the ranker picks x- in 21 worlds and x+ in 3 (worlds 4, 12, 15). All three x+ picks are TR-optimal. Its x- minus x+ contrast correlates with the TR contrast at d=0 (r = 0.70, rho = 0.85). Agreement with the TR optimum at d=0: 14/24, against 11/24 for DPO and SFT.
- The direction is right only early. Sign agreement of the ranker contrast with the 256-draw TR contrast on the ranker's own states: d=0-3 83/96, d=4-7 55/96, d=8-11 37/96, d=12-23 30/276. Late in the episode TR prefers x- over x+ in 246/276 ranker states because x+ has been repeated. Per-state Spearman correlation between the ranker's 8 action scores and the 8 TR scores: mean -0.241 (+0.339 at d=0, -0.354 at d=8-23). Using the ranker's state-averaged score vector at every state gives -0.257, so its state dependence adds +0.016.

Pairwise agreement at the shared first decision (out of 24): DPO = SFT = menu coverage 24/24. Ranker vs DPO 21. Ranker vs TR 14. Local information vs TR 17. TR vs DPO 11. Frozen agrees with no arm except uniform random once. Matrix in the appendix (`s18`).

## 6. Decision-level regret (brief item 6)

### 6.1 Shared first decision, saved scores

Saved deployed TR scores (`planning_metadata.risk_reduction_per_cost`, 32 draws) for all 8 actions in all 24 systems are in the appendix table (`s05`). Per-policy results on the shared state:

| Arm | chose TR optimum | mean regret (saved 32-draw) | mean regret (256-draw) | mean relative regret (256) | chose x- |
| --- | ---: | ---: | ---: | ---: | ---: |
| target_risk | 24/24 | 0 | 0 | 0 | 11/24 |
| local_information | 17/24 | 0.00091 | 0.00077 | 0.042 | 16/24 |
| learned_ranker | 14/24 | 0.00254 | 0.00248 | 0.158 | 21/24 |
| menu_coverage | 11/24 | 0.00920 | 0.00896 | 0.256 | 24/24 |
| sft_llm | 11/24 | 0.00920 | 0.00896 | 0.256 | 24/24 |
| dpo_llm | 11/24 | 0.00920 | 0.00896 | 0.256 | 24/24 |
| uniform_random | 6/24 | 0.01720 | 0.01671 | 0.461 | 4/24 |
| frozen_llm | 0/24 | 0.02796 | 0.02859 | 0.611 | 0/24 |

Constant-action references at d=0 (saved scores): always x- optimal 11/24, mean regret 0.00920. Always xy 5/24, 0.02465. Always x+ 6/24, 0.02200. Always x0 0/24, 0.01575. Always observe 0/24, 0.02796. Always y- 1/24, 0.03235. Always y0 0/24, 0.03137. Always y+ 1/24, 0.02978. Uniform-random expectation 0.02413.

### 6.2 Every decision, each arm's own trajectory

I recomputed the deployed planner's score vector on all 4,203 distinct saved states with 256 fantasy draws (`s06_recompute_tr.py`, pure numpy, byte-identical copies of the pinned modules). With the saved seed and 32 draws the recomputation reproduces the saved score vectors on a random sample of 80 TR decisions (identical decision in 80/80, maximum absolute score difference 2.8e-17, `s23_exact_replay_check.py`), so the path is faithful. On the TR arm's own states the 256-draw scores differ from the saved 32-draw scores by mean 0.000007, SD 0.000414.

| Arm | n | mean regret | median regret | mean relative regret | chose 256-draw optimum | within 2 MC s.e. of optimum | mean TR rank of choice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_risk | 560 | 0.00003 | 0.00000 | 0.016 | 65.9% | 90.4% | 1.55 |
| local_information | 570 | 0.00034 | 0.00005 | 0.154 | 34.6% | 48.6% | 2.63 |
| menu_coverage | 480 | 0.00285 | 0.00071 | 0.441 | 17.9% | 23.8% | 4.42 |
| uniform_random | 487 | 0.00526 | 0.00113 | 0.504 | 13.3% | 18.9% | 4.51 |
| learned_ranker | 576 | 0.00263 | 0.00128 | 0.575 | 9.7% | 12.2% | 5.37 |
| frozen_llm | 576 | 0.01309 | 0.00575 | 0.655 | 3.1% | 7.3% | 4.52 |
| sft_llm | 546 | 0.00799 | 0.00256 | 0.665 | 8.6% | 11.0% | 5.92 |
| dpo_llm | 576 | 0.00838 | 0.00312 | 0.700 | 6.8% | 9.7% | 6.22 |

Regret = best score minus chosen score, in risk reduction per cost unit. Relative regret = regret / best score. A stop is scored as forgoing the best available gain (SFT without its 23 stops: about 0.650).

Paired over the 24 worlds (per-world mean relative regret, Wilcoxon signed-rank, descriptive): every arm is worse than TR in 23 or 24 worlds (p about 1e-7). Against uniform random: DPO worse in 20/24 (diff +0.194, p = 0.00057), SFT 20/24 (+0.158, p = 0.0022), frozen 20/24 (+0.149, p = 0.012), ranker 16/24 (+0.069, p = 0.16), menu coverage better in 16/24 (-0.055, p = 0.029), local information better in 24/24.

By decision block (mean relative regret, d=0-3, 4-7, 8-15, 16-23): DPO 0.340, 0.586, 0.776, 0.862. SFT 0.340, 0.594, 0.724, 0.829. Ranker 0.220, 0.456, 0.623, 0.763. Frozen 0.604, 0.634, 0.662, 0.684. Random 0.475, 0.501, 0.522, 0.501. TR 0.009, 0.021, 0.018, 0.016. **DPO and SFT are better than random in the first four decisions and much worse afterwards.**

What the planner wanted on DPO's own 576 states: x+ 177, obs 151, y+ 139, x- 39, xy 31, y- 21, y0 18. Rank of x- there: 1st 39, 2nd 19, 3rd 21, 4th 33, 5th 37, 6th 46, 7th 201, 8th 180. On frozen's states: x- 238, x+ 163, xy 123, y+ 20, obs 18, y- 11, y0 3.

### 6.3 Is this endpoint meaningful? (`s17_validate_tr_score.py`)

- The SCM posterior integrated risk is calibrated on average against the private NMSE at the same state: mean risk 0.04564 vs mean NMSE 0.04534 over 4,348 acquisition states (ratio 0.994, Spearman 0.64). By regime the ratio NMSE/risk is 0.65 matched_high, 0.95 matched_low, 1.16 shifted_high, 1.04 shifted_low. By graph 1.12 chain, 1.04 collider, 0.66 fork.
- One step: predicted reduction vs realised NMSE reduction has Pearson 0.547 but Spearman 0.123. **NMSE got worse after 43.9% of single acquisitions.** Decile means are calibrated (top decile predicted 0.0229, realised 0.0302; overall predicted 0.00362, realised 0.00423).
- Episode level over 192 cells: sum of predicted reductions vs total realised reduction, Spearman 0.70.
- Arm level: mean relative regret vs mean capped cost-to-target across the 8 arms, Spearman 0.929 (0.690 vs mean final NMSE). n = 8, descriptive.

Caveats. The score is the TR planner's own one-step objective, so TR is the reference, not an independent winner. It is a model-based proxy, not ground truth. Relative regret is computed at different states for different arms, and a constant policy depletes the value of its own action.

## 7. Peakedness at the shared first decision (brief item 7)

- TR optimum: x- 11, x+ 6, xy 5, y+ 1, y- 1. Identical with saved 32-draw and recomputed 256-draw scores in all 24 worlds. Observe is never optimal at d=0.
- Rank of x- (256 draws): 1st 11, 2nd 3, 3rd 2, 4th 2, 5th 1, 7th 3, 8th 2. x- is within 2 MC s.e. of the best in 14/24.
- Gap best minus second (saved scores): min 0.00008, median 0.00488, mean 0.01186, max 0.08590; median 24.5% of the best score. With 256 draws median 0.00471, mean 0.01194. The gap is below 2 combined MC s.e. in 8/24 worlds at 32 draws and 5/24 at 256 draws (worlds 2, 5, 8, 14, 20).
- Gap best minus x-: zero in 11/24; median 0.00020 (0.00008 at 256 draws), mean 0.00920 (0.00896), max 0.13269 (world 12).
- A single readable feature decides the X choice at d=0: with posterior mean of X > 0 TR prefers x- over x+ in 17/17 worlds; with mean < 0 it prefers x+ in 7/7. The one-feature rule "mean of X > 0 then x-, else x+" is TR-optimal in 17/24 with mean relative regret 0.055 (always x-: 11/24 and 0.256). That number, `scm[0].parameter_mean[0]`, is in every prompt. SFT and DPO ignore it (x- in 7/7 negative-mean worlds). The ranker follows it in 3 of 7. Local information follows it in 6 of 7.
- Beyond the first steps the planner's own decisions are flat. On the TR arm's 560 decisions the saved best-minus-second gap is below 1 combined MC s.e. in 51.6% and below 2 in 74.8%. Median gap in s.e. units by decision: 4.83, 2.83, 2.05, 1.72 at d=0..3, then between 0.44 and 1.17. The 256-draw argmax equals the deployed 32-draw choice in only 369/560 (65.9%), yet the deployed choice is within 2 s.e. of the 256-draw best in 90.4%.

## 8. Prompt length, tokens, latency (brief item 8)

From `generation-NNNN.json` (`s10_tokens_latency.py`):

| Arm | prompt tokens min / median / mean / max | tokens added per accepted response | new tokens incl. EOS | cuda ms median / mean / p95 / max | total cuda s |
| --- | --- | ---: | --- | --- | ---: |
| frozen_llm | 1,990 / 3,066 / 3,069.7 / 4,160 | 90.9 | 8 (574), 9 (2) | 424.6 / 580.5 / 1,462.9 / 4,878.3 | 334.3 |
| sft_llm | 1,990 / 2,933 / 2,929.8 / 3,954 | 83.1 | 9 (515), 6 (23 stops), 8 (8) | 536.0 / 594.3 / 927.7 / 2,546.4 | 324.5 |
| dpo_llm | 1,990 / 2,984 / 2,981.0 / 3,985 | 83.0 | 9 (576) | 709.6 / 849.2 / 1,645.8 / 4,046.2 | 489.1 |

- 2.33 to 2.39 characters per token. Numbers are serialized at 16-17 significant digits and tokenized digit by digit: one 2x2 covariance matrix is 78 tokens.
- Token budget by section (DPO, world 0): at d=0 of 2,012 tokens, history 718 (35.7%), SCM posterior 377 (18.7%), task 321 (16.0%), menu 287 (14.3%), graph 71, notes 60, budget counters 25 (1.2%). At d=23 of 3,927 tokens, history 2,622 (66.8%), SCM 386 (9.8%), counters 27 (0.7%).
- Latency is nearly independent of prompt length (correlation 0.07 to 0.09). Peak GPU allocation 2.0 to 2.3 GiB. `legal_commands` is 9 except 5 at remaining 1 in heterogeneous menus (12, 4, 12 decisions).
- Numerical arms for reference, one fresh subprocess per decision: median 1.07 to 1.19 s (ranker 3.13 s).
- Decoding path. Commands tokenize as `{"` `action` `_id` `":"` then `observe` `_all` or `set` + `_x|_y|_xy` + `_neg|_zero|_pos`, then `"}`; stop is `{"` `stop` `":` `true` `}`. Greedy constrained decoding therefore makes up to four sequential choices: action vs stop, `set` vs `observe`, `_x` vs `_y` vs `_xy`, then the value. This is not the argmax over whole-command likelihood: `set` carries the pooled mass of seven commands against one for `observe`.

## 9. Evidence against the simple story "the DPO adapter ignores the state" (brief item 9)

For the simple story: one command on 576 of 576 decisions, across 576 distinct prompts, 3 graphs, 2 cost menus, 4 regimes and every budget level; one action sequence in all 24 worlds.

Against it, or complicating it:
1. The records cannot show that scores ignore the state. No scores are saved. A state-dependent score vector with command gaps too large to flip would look the same. Training-side loss records point that way: the DPO stage moved pairwise log-odds by up to 16 nats relative to SFT (section 10).
2. x- is the planner's choice at the start: TR-optimal in 11/24 at d=0 and the best constant action. DPO's regret at d=0 equals SFT's and menu coverage's, and its relative regret over d=0-3 (0.340) beats uniform random (0.475).
3. The same pipeline does transmit state into behaviour in the SFT adapter (budget level, graph, stopping). The loss of variation is specific to the DPO stage.
4. A numerical model with direct state access trained on the same exposure is also nearly state-blind (98.1% action main effect, 2 rank orders). The insensitivity is not specific to language.
5. The supervision carries little state-conditional signal. Labels equal the 256-draw TR optimum in 55/128 training states. The label mix at the first snapshot says x- in 16/32 worlds irrespective of the sign of the X mean (6 of 14 negative-mean worlds are labelled x-, 2 are labelled x+), while TR prefers x+ in 12 of those 14.
6. The planner itself is near-indifferent after the first steps (75% of its decisions within 2 MC s.e.), so argmax labels and one-step preferences are noisy.
7. Deployment states are off the training distribution. Training histories come from mixed behaviour (mean share of x- after initialization 0.204, max 0.833; 2 of 96 later snapshots have a single repeated action). DPO's states have 100% x-.
8. The exposed DPO pairs did not favour x- strongly: chosen in 14 of 119 and rejected in 9. x+ had the largest net count (+9).
9. DPO is not worse on every outcome: mean final NMSE 0.03642 against 0.04113 for uniform random.

## 10. Supplement: training-side records that explain the deployment behaviour

These are training records, outside the deployment scope, used only because they are the sole saved LLM command scores and they sit beside the deployment data. Another agent may cover them in more depth.

### 10.1 Matched exposure (`cpu-phase/numeric-training-plan.json`; `s07`, `s11`)

- 160 records = 32 training worlds x 5 snapshots. Snapshots are at remaining budget 24, 18 (17 in 7), 12 (11 in 3), 6 (5 in 2) and 0. The fifth snapshot is terminal with a single candidate (stop).
- Supervised labels: STOP 32, x- 32, x+ 29, obs 24, xy 12, y+ 10, y- 10, x0 7, y0 4. Snapshot 0: x- 16, x0 3, y- 3, xy 3, obs 3, x+ 2, y0 2. Snapshot 1: x+ 13, x- 7, obs 5. Snapshot 2: obs 9, x+ 8, x- 4. Snapshot 3: obs 7, x+ 6, xy 6, x- 5.
- Labels by graph: y- appears in chain 5, collider 5, fork 0.
- Preferences: 3,584 pairs, 2,897 tied, 687 strict, 119 exposed (one per record with a strict pair; 41 records have none). Exposed chosen: x+ 23, xy 20, y- 19, y0 16, x- 14, x0 13, y+ 8, obs 6. Exposed rejected: y+ 21, x0 19, xy 16, x+ 14, y- 14, y0 13, obs 13, x- 9. Net: x+ +9, x- +5, y- +5, xy +4, y0 +3, x0 -6, obs -7, y+ -13.
- Ranker recipe: hidden 128, 1,154,817 parameters, feature width 8,891, Adam lr 0.001, beta 0.1, 160 + 119 single-example steps.

### 10.2 Target-risk scores on the 128 non-terminal training states (`s20`)

- Supervised label equals the 256-draw TR optimum in 55/128 (43.0%); within 2 MC s.e. in 67/128. Mean relative regret of the labels 0.225 (uniform expectation 0.438, always x- 0.302). By snapshot 15/32, 19/32, 12/32, 9/32.
- TR optimum at training states: x+ 34, obs 31, x- 30, xy 16, y+ 12, y- 5.
- Strict preference pairs: the preferred action has the higher TR score in 471/687 (68.6%). For the 119 exposed pairs 70/119 (58.8%). By snapshot 76.4%, 75.8%, 51.4%, 64.3%.

### 10.3 SFT reference log-likelihoods and DPO step losses (`s14`, `s15`)

Recipe (`plan.json: numeric_recipe`): AdamW lr 1e-4, beta 0.1, gradient clip 1.0, rank 8, alpha 16, single-example steps, one pass.
- `reference-scores.json` holds the SFT adapter's full-vocabulary log-likelihood of the chosen and the rejected command for each of the 119 exposed pairs. Per command across training states (mean, between-state SD, mean probability): x- -0.952, 0.386, 0.413. obs -1.092, 0.344, 0.353. y- -2.256, 0.265, 0.108. x+ -3.220, 0.791, 0.053. xy -3.877, 0.502, 0.023. y+ -4.328, 0.955, 0.020. x0 -4.899, 1.066, 0.012. y0 -5.561, 0.784, 0.005. SD of the 8 command means 1.713 against pooled between-state SD 0.696 (ratio 0.41). Each command is averaged over the pairs in which it occurs, so the state sets differ.
- SFT state dependence on training states: x- log-likelihood falls with snapshot index (-0.42, -0.90, -1.16, -1.35) while observe rises (-1.86, -1.20, -0.96, -0.82).
- The SFT model prefers the preference-chosen command in 66/119 pairs (55.5%, binomial p = 0.27).
- DPO steps. Each loss is for one pair seen for the first time, so Delta = logit(exp(-loss)) / beta is the policy's log-odds shift relative to SFT on a not-yet-trained pair. Delta > 0 in 55/119 (46.2%). The policy favours the chosen command in 66/119, the same as SFT. Loss goes from 0.6931 to a range of 0.185 to 1.608; |Delta| reaches 15.9 nats.
- Delta tracks the SFT margin: Pearson +0.63 overall and +0.78 in the last 29 steps; sign agreement 92/119; slope +1.28 nats of Delta per nat of SFT margin overall (0.44, 1.83, 1.01, 1.97 in four successive blocks of about 30 steps).
- A per-command additive drift model explains R^2 = 0.65 (steps 30-118), 0.75 (60-118) and 0.84 (90-118) of Delta. Drift over the last 29 steps (zero-mean nats): obs +4.02, x- +3.54, xy +2.91, y- +1.53, x+ +0.96, y+ -2.73, x0 -3.54, y0 -6.70. Drift correlates r = +0.89 with the SFT mean command log-likelihood and r = +0.20 with the net exposed preference counts.

Inference: the DPO stage acted mainly as a state-independent sharpening of the SFT command ordering, not as preference learning. This is prequential evidence on training states, derived from loss scalars. The x- vs observe contrast rests on few pairs. The final adapter was not scored.

## 11. Limits of this analysis

- No LLM command scores at deployment. All LLM statements are about emitted commands, inputs, or training-side scalars.
- Target-risk regret is a model-based one-step proxy and is circular for the TR arm.
- Monte Carlo noise: 256 draws; "within 2 s.e." uses the function's own standard errors.
- 24 development systems, one trajectory each per arm. P-values are descriptive.
- Training-side conclusions are inferences from records not designed for this purpose.
