# Deployment forensics of the 24-system ACE intervention comparison

Label: `deployment-forensics`. Prepared 2026-11-12. Read-only offline analysis with `/usr/bin/python3` (3.9). No model weights were loaded, no inference, no episode, no SLURM/SSH. Nothing under `/Users/pat/code/ACE` was modified (git status identical before and after).

Reading guide: sections 0 to 11 (about the first 36 KB) are the findings. Section 12 lists scripts and reusable data. Everything after "# Appendix" is verbatim script output for checking exact numbers.

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

## 12. Scripts, reusable data and how to rerun

Directory: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics/`. Run with `/usr/bin/python3 -B <script>` from that directory (`-B` avoids bytecode files). `lib/` holds byte-identical copies of seven pure numpy repository modules (`llm_intervention_contract.py`, `_scm.py`, `_numerical_policy.py`, `_target_risk.py`, `_task.py`, `_episode.py`, `_grammar.py`), which I confirmed with `cmp` to equal both `/Users/pat/code/ACE/scripts/research/` and the pinned copies in `.../campaign_gpu_20261108/terminal02/remote/launch/inputs/`.

| Script | Purpose | Output |
| --- | --- | --- |
| `s01_llm_keys.py` | Enumerate all keys in LLM worker records | `out/s01_llm_keys.out` |
| `s02_decode_check.py` | Tokenizer pin and decode check on one prompt | stdout |
| `s03_build_dataset.py` | Unified 4,371-decision dataset with verified envelopes | `out/decisions.pkl` |
| `s04_action_dist.py` | Action distributions, cross-check with cost reports | `out/s04_action_dist.out`, `out/reports.pkl` |
| `s05_first_decision.py` | Shared first decision with saved TR scores | `out/s05_first_decision.out` |
| `s06_recompute_tr.py` | 256-draw TR scores on all 4,203 states (about 16 min on 9 processes) | `out/tr_scores_256.pkl` |
| `s07_training_exposure.py` | SFT label and DPO pair distributions | `out/s07_training_exposure.out` |
| `s08_ranker.py` | Ranker score vectors | `out/s08_ranker.out` |
| `s09_sft_frozen.py` | SFT deviations and stops, frozen non-observe actions | `out/s09_sft_frozen.out` |
| `s10_tokens_latency.py` | Tokens, latency, command token tree | `out/s10_tokens_latency.out` |
| `s11_training_states.py` | Embedded training states | `out/s11_training_states.out`, `out/train_states.pkl` |
| `s12_behavioral_sensitivity.py` | Entropy and mutual information per arm | `out/s12_behavioral_sensitivity.out` |
| `s13_tr_own_saved.py` | Peakedness of TR's own saved decisions | `out/s13_tr_own_saved.out` |
| `s14_llm_training_scores.py`, `s15_dpo_drift.py` | SFT reference scores, DPO step losses, drift | `out/s14_*.out`, `out/s15_*.out`, `out/dpo_steps.pkl` |
| `s16_regret_own_states.py` | Decision-level regret for all arms | `out/s16.out`, `out/regret.pkl` |
| `s17_validate_tr_score.py` | Calibration against realised NMSE | `out/s17.out` |
| `s18_agreement.py` | Pairwise agreement matrices | `out/s18_agreement.out` |
| `s19_direction_of_sensitivity.py` | Whether ranker and SFT state dependence points the right way | `out/s19.out` |
| `s20_training_states_tr.py` | TR scores on training states vs labels and preferences | `out/s20.out`, `out/train_tr_256.pkl` |
| `s21_tables.py` | Stop table and first-decision table | `out/s21.out` |
| `s22_sign_rule.py` | One-feature rule on the sign of the X posterior mean | `out/s22.out` |
| `s23_exact_replay_check.py` | Replay of saved TR scores with saved seeds | `out/s23.out` |

`out/decisions.pkl` is a dict with `rows` (one dict per decision: `arm`, `world_idx`, `decision_index`, `envelope`, `envelope_sha256`, `action`, `meta` for numerical arms, token and latency fields for LLM arms), `worlds` (strata, costs, true mechanisms) and `policy_ids`. `out/tr_scores_256.pkl` maps `envelope_sha256` to `scores`, `se`, `risk_after`, `risk_before`.

# Appendix: verbatim script outputs


## Appendix: out/s05_first_decision.out

```
all 8 arms share identical envelope sha256 at decision 0 in all 24 worlds: True

Per-world table at shared first decision. tr = saved target-risk score (risk reduction per cost unit).
w  graph    cost  regime        nmse0   risk_before | observe_a set_x_neg set_x_zer set_x_pos set_y_neg set_y_zer set_y_pos    set_xy | best      gap12   gap_xneg  se_best | chosen by: unifor menu_c local_ target learne frozen sft_ll dpo_ll
 0 chain    unifo matched_low  0.0298  0.07952     |   0.00884   0.04977   0.03192   0.00229   0.00278   0.00078   0.00485  -0.00035 | set_x_neg 0.01786 0.00000  0.00053 | x0     x-     x-     x-     x-     obs    x-     x-    
 1 chain    heter matched_low  0.0060  0.04545     |   0.00361   0.01499   0.01033   0.00364   0.00483   0.00037   0.00395   0.00294 | set_x_neg 0.00466 0.00000  0.00069 | y0     x-     x-     x-     x-     obs    x-     x-    
 2 fork     unifo matched_low  0.0377  0.06867     |   0.01556   0.04893   0.02503   0.00358   0.01290   0.01195   0.01021   0.04856 | set_x_neg 0.00036 0.00000  0.00018 | x-     x-     x-     x-     x-     obs    x-     x-    
 3 fork     heter matched_low  0.0681  0.02897     |   0.00273   0.00133   0.00934   0.01515   0.00136   0.00120   0.00114   0.00067 | set_x_pos 0.00582 0.01382  0.00015 | xy     x-     x+     x+     x-     obs    x-     x-    
 4 collider unifo matched_low  0.0450  0.06644     |   0.00491   0.01329   0.02286   0.02926   0.00596   0.00468   0.01388   0.01249 | set_x_pos 0.00640 0.01597  0.00025 | y-     x-     x+     x+     x+     obs    x-     x-    
 5 collider heter matched_low  0.0430  0.05915     |   0.00583   0.01596   0.00854   0.00296   0.00167   0.00666   0.01066   0.01629 | set_xy    0.00033 0.00033  0.00016 | x-     x-     x-     xy     x-     obs    x-     x-    
 6 chain    unifo matched_high 0.0003  0.05489     |   0.00420   0.01323   0.00830   0.00428   0.00494   0.00802   0.01141   0.01143 | set_x_neg 0.00179 0.00000  0.00033 | obs    x-     x-     x-     x-     obs    x-     x-    
 7 chain    heter matched_high 0.0367  0.05188     |   0.00522   0.00227   0.00535   0.01044   0.00183   0.00297   0.00441   0.00415 | set_x_pos 0.00509 0.00817  0.00034 | x+     x-     x+     x+     x-     obs    x-     x-    
 8 fork     unifo matched_high 0.0059  0.05239     |   0.00420   0.01114   0.00571   0.00219   0.00454   0.00446   0.00482   0.01122 | set_xy    0.00008 0.00008  0.00005 | xy     x-     x-     xy     x-     obs    x-     x-    
 9 fork     heter matched_high 0.0379  0.08178     |   0.00588   0.01813   0.01065   0.00387   0.00307   0.00349   0.00411   0.00929 | set_x_neg 0.00748 0.00000  0.00109 | y-     x-     x-     x-     x-     obs    x-     x-    
10 collider unifo matched_high 0.1250  0.13346     |   0.01177   0.01098   0.01388   0.01782   0.01340   0.01114   0.00437   0.00322 | set_x_pos 0.00395 0.00685  0.00114 | y0     x-     y-     x+     x-     obs    x-     x-    
11 collider heter matched_high 0.1107  0.13949     |   0.01642   0.03736   0.01564   0.00605   0.00735   0.00878   0.01028   0.02156 | set_x_neg 0.01580 0.00000  0.00148 | y-     x-     x-     x-     x-     obs    x-     x-    
12 chain    unifo shifted_low  0.5219  0.25753     |   0.03897   0.01243   0.05922   0.14512   0.02351   0.03809   0.03302   0.02922 | set_x_pos 0.08590 0.13269  0.00165 | xy     x-     x+     x+     x+     obs    x-     x-    
13 chain    heter shifted_low  0.3160  0.21760     |   0.01581   0.14766   0.10971   0.03652   0.02182   0.01832   0.00977   0.01127 | set_x_neg 0.03796 0.00000  0.00138 | x+     x-     x-     x-     x-     obs    x-     x-    
14 fork     unifo shifted_low  0.0411  0.11990     |   0.01118   0.07685   0.06807   0.04504   0.01013   0.01029   0.01155   0.07873 | set_xy    0.00188 0.00188  0.00090 | xy     x-     x-     xy     x-     obs    x-     x-    
15 fork     heter shifted_low  0.0129  0.03218     |   0.00283   0.00085   0.00614   0.01201   0.00127   0.00155   0.00164   0.00043 | set_x_pos 0.00587 0.01116  0.00041 | x0     x-     x+     x+     x+     obs    x-     x-    
16 collider unifo shifted_low  0.0916  0.07741     |   0.00883   0.01763   0.01056   0.01187   0.00426   0.01383   0.02798   0.01793 | set_y_pos 0.01005 0.01035  0.00090 | xy     x-     y+     y+     x-     obs    x-     x-    
17 collider heter shifted_low  0.1115  0.08680     |   0.00703   0.01014   0.00423   0.00822   0.00489   0.01016   0.01454   0.01520 | set_xy    0.00066 0.00506  0.00040 | x+     x-     x-     xy     x-     obs    x-     x-    
18 chain    unifo shifted_high 0.0154  0.04430     |   0.00391   0.00660   0.00448   0.00291   0.00486   0.00342   0.00253   0.00250 | set_x_neg 0.00175 0.00000  0.00039 | y0     x-     x-     x-     x-     obs    x-     x-    
19 chain    heter shifted_high 0.5449  0.44067     |   0.05436   0.16393   0.10340   0.02967   0.00734   0.00242   0.00242   0.00294 | set_x_neg 0.06052 0.00000  0.00352 | x-     x-     x-     x-     x-     obs    x-     x-    
20 fork     unifo shifted_high 0.0209  0.07964     |   0.00673   0.00261   0.00470   0.00715   0.00727   0.00604   0.00692   0.00307 | set_y_neg 0.00012 0.00465  0.00079 | x-     x-     x+     y-     x-     obs    x-     x-    
21 fork     heter shifted_high 0.0778  0.05575     |   0.00677   0.00697   0.00350   0.00285   0.00261   0.00268   0.00293   0.00346 | set_x_neg 0.00019 0.00000  0.00058 | y0     x-     x-     x-     x-     obs    x-     x-    
22 collider unifo shifted_high 0.4139  0.13209     |   0.01212   0.01711   0.00972   0.00494   0.01121   0.01359   0.02068   0.02698 | set_xy    0.00630 0.00987  0.00149 | xy     x-     x-     xy     x-     obs    x-     x-    
23 collider heter shifted_high 0.2332  0.17179     |   0.01749   0.02525   0.01702   0.01042   0.00602   0.00858   0.01357   0.02152 | set_x_neg 0.00373 0.00000  0.00169 | y-     x-     x-     x-     x-     obs    x-     x-    

target-risk optimum at shared first decision, counts over 24 worlds: {'set_x_neg': 11, 'set_x_pos': 6, 'set_xy': 5, 'set_y_pos': 1, 'set_y_neg': 1}
rank of set_x_neg under target-risk score at first decision (1=best): Counter({1: 11, 2: 3, 4: 2, 7: 2, 8: 2, 3: 2, 5: 1, 6: 1})
gap best-second: min 0.00008 median 0.00488 mean 0.01186 max 0.08590
gap best-set_x_neg: min 0.00000 median 0.00020 mean 0.00920 max 0.13269 ; zero in 11/24
relative gap12 / best score: median 0.245
gap12 in units of combined MC s.e.:  [19.8, 4.07, 1.06, 28.72, 17.69, 0.33, 5.48, 12.62, 0.8, 6.47, 2.41, 10.42, 40.38, 13.9, 1.57, 11.5, 8.91, 1.23, 4.19, 11.94, 0.08, 0.21, 3.19, 1.78]
  worlds where gap12 < 2 combined MC s.e.: 8 of 24

Per-policy summary at the shared first decision (24 worlds):
arm                agree_with_TR_opt  mean_regret  median_regret  mean_norm_regret  mean_rank  chose_x_neg
uniform_random      6/24             0.01720      0.00683        0.573             4.42        4/24
menu_coverage      11/24             0.00920      0.00020        0.308             3.00       24/24
local_information  17/24             0.00091      0.00000        0.054             1.46       16/24
target_risk        24/24             0.00000      0.00000        0.000             1.00       11/24
learned_ranker     14/24             0.00254      0.00000        0.199             2.33       21/24
frozen_llm          0/24             0.02796      0.01182        0.701             4.58        0/24
sft_llm            11/24             0.00920      0.00020        0.308             3.00       24/24
dpo_llm            11/24             0.00920      0.00020        0.308             3.00       24/24

Constant-action reference policies at the shared first decision:
  always observe_all: optimal in  0/24, mean regret 0.02796, median 0.01182
  always set_x_neg  : optimal in 11/24, mean regret 0.00920, median 0.00020
  always set_x_zero : optimal in  0/24, mean regret 0.01575, median 0.00761
  always set_x_pos  : optimal in  6/24, mean regret 0.02200, median 0.01019
  always set_y_neg  : optimal in  1/24, mean regret 0.03235, median 0.01484
  always set_y_zero : optimal in  0/24, mean regret 0.03137, median 0.01405
  always set_y_pos  : optimal in  1/24, mean regret 0.02978, median 0.01136
  always set_xy     : optimal in  5/24, mean regret 0.02465, median 0.00756
  uniform-random expectation: mean regret 0.02413

worlds already at NMSE<=0.04 after init: [0, 1, 2, 6, 7, 8, 9, 15, 18, 20] 10
 subset already-at-target (10)
   TR optimum counts: {'set_x_neg': 6, 'set_x_pos': 2, 'set_xy': 1, 'set_y_neg': 1}
   uniform_random     mean regret 0.00703  agree 3/10
   menu_coverage      mean regret 0.00241  agree 6/10
   local_information  mean regret 0.00002  agree 8/10
   target_risk        mean regret 0.00000  agree 10/10
   learned_ranker     mean regret 0.00129  agree 7/10
   frozen_llm         mean regret 0.01316  agree 0/10
   sft_llm            mean regret 0.00241  agree 6/10
   dpo_llm            mean regret 0.00241  agree 6/10
 subset not-at-target (14)
   TR optimum counts: {'set_x_pos': 4, 'set_xy': 4, 'set_x_neg': 5, 'set_y_pos': 1}
   uniform_random     mean regret 0.02446  agree 3/14
   menu_coverage      mean regret 0.01406  agree 5/14
   local_information  mean regret 0.00154  agree 9/14
   target_risk        mean regret 0.00000  agree 14/14
   learned_ranker     mean regret 0.00344  agree 7/14
   frozen_llm         mean regret 0.03853  agree 0/14
   sft_llm            mean regret 0.01406  agree 5/14
   dpo_llm            mean regret 0.01406  agree 5/14
```

## Appendix: out/s21.out

```
SFT stop table: world | graph | cost | regime | stop decision index | requests | spent | remaining | legal cmds | prompt tokens | posterior risk at stop | private final NMSE | final target met
|  0 | chain    | uniform       | matched_low  | 21 | 29 | 29 | 3 | 9 | 3763 | 0.0096 | 0.0076 | True |
|  1 | chain    | heterogeneous | matched_low  | 21 | 29 | 30 | 2 | 9 | 3757 | 0.0084 | 0.0013 | True |
|  2 | fork     | uniform       | matched_low  | 22 | 30 | 30 | 2 | 9 | 3842 | 0.0065 | 0.0033 | True |
|  3 | fork     | heterogeneous | matched_low  | 21 | 29 | 29 | 3 | 9 | 3783 | 0.0086 | 0.0374 | True |
|  4 | collider | uniform       | matched_low  | 21 | 29 | 29 | 3 | 9 | 3824 | 0.0156 | 0.0045 | True |
|  5 | collider | heterogeneous | matched_low  | 21 | 29 | 30 | 2 | 9 | 3818 | 0.0159 | 0.0045 | True |
|  6 | chain    | uniform       | matched_high | 21 | 29 | 29 | 3 | 9 | 3763 | 0.0115 | 0.0075 | True |
|  7 | chain    | heterogeneous | matched_high | 22 | 30 | 31 | 1 | 5 | 3824 | 0.0226 | 0.0069 | True |
|  8 | fork     | uniform       | matched_high | 22 | 30 | 30 | 2 | 9 | 3837 | 0.0174 | 0.0090 | True |
|  9 | fork     | heterogeneous | matched_high | 22 | 30 | 30 | 2 | 9 | 3835 | 0.0255 | 0.0016 | True |
| 10 | collider | uniform       | matched_high | 23 | 31 | 31 | 1 | 9 | 3954 | 0.0428 | 0.0493 | False |
| 11 | collider | heterogeneous | matched_high | 21 | 29 | 30 | 2 | 9 | 3799 | 0.0501 | 0.0472 | False |
| 12 | chain    | uniform       | shifted_low  | 22 | 30 | 30 | 2 | 9 | 3805 | 0.2446 | 0.0917 | False |
| 13 | chain    | heterogeneous | shifted_low  | 23 | 31 | 31 | 1 | 5 | 3921 | 0.0219 | 0.0039 | True |
| 14 | fork     | uniform       | shifted_low  | 22 | 30 | 30 | 2 | 9 | 3822 | 0.0273 | 0.0064 | True |
| 15 | fork     | heterogeneous | shifted_low  | 22 | 30 | 30 | 2 | 9 | 3818 | 0.0168 | 0.0126 | True |
| 16 | collider | uniform       | shifted_low  | 22 | 30 | 30 | 2 | 9 | 3871 | 0.0196 | 0.0033 | True |
| 18 | chain    | uniform       | shifted_high | 21 | 29 | 29 | 3 | 9 | 3762 | 0.0149 | 0.0038 | True |
| 19 | chain    | heterogeneous | shifted_high | 22 | 30 | 30 | 2 | 9 | 3825 | 0.1243 | 0.1991 | False |
| 20 | fork     | uniform       | shifted_high | 22 | 30 | 30 | 2 | 9 | 3831 | 0.0661 | 0.0123 | True |
| 21 | fork     | heterogeneous | shifted_high | 22 | 30 | 30 | 2 | 9 | 3829 | 0.0328 | 0.0583 | False |
| 22 | collider | uniform       | shifted_high | 22 | 30 | 30 | 2 | 9 | 3920 | 0.0294 | 0.0304 | True |
| 23 | collider | heterogeneous | shifted_high | 22 | 30 | 31 | 1 | 5 | 3873 | 0.0693 | 0.0349 | True |

First-decision table (shared state). s32 = saved deployed scores (planning_metadata.risk_reduction_per_cost); s256 = recomputed.
| w | graph | cost | regime | NMSE0 | TR-opt (s32) | TR-opt (s256) | gap best-2nd s256 | in s.e. | gap best - x_neg s256 | rank x_neg s256 | unif | menu | local | TR | ranker | frozen | sft | dpo |
|  0 | chain    | unifo | matched_low  | 0.0298 | x-  | x-  | 0.01777 |  40.9 | 0.00000 | 1 | x0 | x- | x- | x- | x- | obs | x- | x- |
|  1 | chain    | heter | matched_low  | 0.0060 | x-  | x-  | 0.00535 |  11.4 | 0.00000 | 1 | y0 | x- | x- | x- | x- | obs | x- | x- |
|  2 | fork     | unifo | matched_low  | 0.0377 | x-  | x-  | 0.00007 |   0.6 | 0.00000 | 1 | x- | x- | x- | x- | x- | obs | x- | x- |
|  3 | fork     | heter | matched_low  | 0.0681 | x+  | x+  | 0.00598 |  89.5 | 0.01409 | 7 | xy | x- | x+ | x+ | x- | obs | x- | x- |
|  4 | collider | unifo | matched_low  | 0.0450 | x+  | x+  | 0.00603 |  42.4 | 0.01563 | 4 | y- | x- | x+ | x+ | x+ | obs | x- | x- |
|  5 | collider | heter | matched_low  | 0.0430 | xy  | xy  | 0.00041 |   1.3 | 0.00041 | 2 | x- | x- | x- | xy | x- | obs | x- | x- |
|  6 | chain    | unifo | matched_high | 0.0003 | x-  | x-  | 0.00172 |  15.5 | 0.00000 | 1 | obs | x- | x- | x- | x- | obs | x- | x- |
|  7 | chain    | heter | matched_high | 0.0367 | x+  | x+  | 0.00449 |  30.2 | 0.00779 | 7 | x+ | x- | x+ | x+ | x- | obs | x- | x- |
|  8 | fork     | unifo | matched_high | 0.0059 | xy  | xy  | 0.00001 |   0.2 | 0.00001 | 2 | xy | x- | x- | xy | x- | obs | x- | x- |
|  9 | fork     | heter | matched_high | 0.0379 | x-  | x-  | 0.00801 |  21.1 | 0.00000 | 1 | y- | x- | x- | x- | x- | obs | x- | x- |
| 10 | collider | unifo | matched_high | 0.1250 | x+  | x+  | 0.00177 |   2.3 | 0.00541 | 5 | y0 | x- | y- | x+ | x- | obs | x- | x- |
| 11 | collider | heter | matched_high | 0.1107 | x-  | x-  | 0.01241 |  17.8 | 0.00000 | 1 | y- | x- | x- | x- | x- | obs | x- | x- |
| 12 | chain    | unifo | shifted_low  | 0.5219 | x+  | x+  | 0.08374 |  91.0 | 0.12951 | 8 | xy | x- | x+ | x+ | x+ | obs | x- | x- |
| 13 | chain    | heter | shifted_low  | 0.3160 | x-  | x-  | 0.03988 |  51.9 | 0.00000 | 1 | x+ | x- | x- | x- | x- | obs | x- | x- |
| 14 | fork     | unifo | shifted_low  | 0.0411 | xy  | xy  | 0.00015 |   0.4 | 0.00015 | 2 | xy | x- | x- | xy | x- | obs | x- | x- |
| 15 | fork     | heter | shifted_low  | 0.0129 | x+  | x+  | 0.00493 |  23.3 | 0.01058 | 7 | x0 | x- | x+ | x+ | x+ | obs | x- | x- |
| 16 | collider | unifo | shifted_low  | 0.0916 | y+  | y+  | 0.00993 |  21.1 | 0.01144 | 3 | xy | x- | y+ | y+ | x- | obs | x- | x- |
| 17 | collider | heter | shifted_low  | 0.1115 | xy  | xy  | 0.00103 |   5.2 | 0.00498 | 4 | x+ | x- | x- | xy | x- | obs | x- | x- |
| 18 | chain    | unifo | shifted_high | 0.0154 | x-  | x-  | 0.00250 |  20.1 | 0.00000 | 1 | y0 | x- | x- | x- | x- | obs | x- | x- |
| 19 | chain    | heter | shifted_high | 0.5449 | x-  | x-  | 0.06829 |  29.2 | 0.00000 | 1 | x- | x- | x- | x- | x- | obs | x- | x- |
| 20 | fork     | unifo | shifted_high | 0.0209 | y-  | y-  | 0.00036 |   0.8 | 0.00569 | 8 | x- | x- | x+ | y- | x- | obs | x- | x- |
| 21 | fork     | heter | shifted_high | 0.0778 | x-  | x-  | 0.00148 |   4.6 | 0.00000 | 1 | y0 | x- | x- | x- | x- | obs | x- | x- |
| 22 | collider | unifo | shifted_high | 0.4139 | xy  | xy  | 0.00776 |  11.9 | 0.00945 | 3 | xy | x- | x- | xy | x- | obs | x- | x- |
| 23 | collider | heter | shifted_high | 0.2332 | x-  | x-  | 0.00257 |   3.3 | 0.00000 | 1 | y- | x- | x- | x- | x- | obs | x- | x- |
```

## Appendix: out/s16.out

```
states with recomputed scores: 4203
saved 32-draw vs recomputed 256-draw scores on target_risk states: mean diff 0.000007, sd 0.000414 ; 256-draw argmax equals deployed (32-draw) choice in 369/560 (65.9%)

Decision-level endpoint on each arm's OWN trajectory (all its decisions; STOP counted as forgoing the best available gain).
regret = best score - chosen score (risk reduction per cost unit, 256 draws). rel = regret / best score. near-opt = within 2 combined MC s.e. of the best.
arm                   n |  mean_reg   med_reg | mean_rel |  TR-opt% near-opt% | m_rank | sum regret over episode (mean per world)
uniform_random      487 |   0.00526   0.00113 |   0.504 |    13.3%    18.9% |   4.51 | 0.1068
menu_coverage       480 |   0.00285   0.00071 |   0.441 |    17.9%    23.8% |   4.42 | 0.0570
local_information   570 |   0.00034   0.00005 |   0.154 |    34.6%    48.6% |   2.63 | 0.0080
target_risk         560 |   0.00003   0.00000 |   0.016 |    65.9%    90.4% |   1.55 | 0.0008
learned_ranker      576 |   0.00263   0.00128 |   0.575 |     9.7%    12.2% |   5.37 | 0.0631
frozen_llm          576 |   0.01309   0.00575 |   0.655 |     3.1%     7.3% |   4.52 | 0.3141
sft_llm             546 |   0.00799   0.00256 |   0.665 |     8.6%    11.0% |   5.92 | 0.1818
dpo_llm             576 |   0.00838   0.00312 |   0.700 |     6.8%     9.7% |   6.22 | 0.2011

Per-world mean relative regret (rel), paired across the 24 worlds: arm vs target_risk and vs uniform_random (Wilcoxon signed-rank, descriptive)
uniform_random     mean 0.506 sd 0.152 | vs target_risk: diff +0.489, worse in 24/24, W p=1.2e-07
menu_coverage      mean 0.451 sd 0.122 | vs target_risk: diff +0.435, worse in 24/24, W p=1.2e-07 | vs uniform_random: diff -0.055, worse in 8/24, W p=0.029
local_information  mean 0.156 sd 0.112 | vs target_risk: diff +0.139, worse in 23/24, W p=3.6e-07 | vs uniform_random: diff -0.350, worse in 0/24, W p=1.2e-07
target_risk        mean 0.017 sd 0.013 | vs uniform_random: diff -0.489, worse in 0/24, W p=1.2e-07
learned_ranker     mean 0.575 sd 0.192 | vs target_risk: diff +0.558, worse in 24/24, W p=1.2e-07 | vs uniform_random: diff +0.069, worse in 16/24, W p=0.16
frozen_llm         mean 0.655 sd 0.270 | vs target_risk: diff +0.639, worse in 24/24, W p=1.2e-07 | vs uniform_random: diff +0.149, worse in 20/24, W p=0.012
sft_llm            mean 0.664 sd 0.187 | vs target_risk: diff +0.647, worse in 24/24, W p=1.2e-07 | vs uniform_random: diff +0.158, worse in 20/24, W p=0.0022
dpo_llm            mean 0.700 sd 0.196 | vs target_risk: diff +0.684, worse in 24/24, W p=1.2e-07 | vs uniform_random: diff +0.194, worse in 20/24, W p=0.00057

Mean relative regret by decision-index block (0-3, 4-7, 8-15, 16-23):
  uniform_random     0.475  0.501  0.522  0.501
  menu_coverage      0.416  0.470  0.454  0.412
  local_information  0.074  0.178  0.162  0.176
  target_risk        0.009  0.021  0.018  0.016
  learned_ranker     0.220  0.456  0.623  0.763
  frozen_llm         0.604  0.634  0.662  0.684
  sft_llm            0.340  0.594  0.724  0.829
  dpo_llm            0.340  0.586  0.776  0.862

Fraction of decisions where the chosen action is the 256-draw TR optimum, by decision-index block (0, 1-3, 4-7, 8-15, 16-23):
  uniform_random     0.25  0.19  0.15  0.10  0.12
  menu_coverage      0.46  0.17  0.21  0.16  0.14
  local_information  0.71  0.54  0.36  0.31  0.25
  target_risk        1.00  0.79  0.64  0.62  0.61
  learned_ranker     0.58  0.42  0.11  0.01  0.00
  frozen_llm         0.00  0.01  0.03  0.04  0.04
  sft_llm            0.46  0.35  0.05  0.02  0.01
  dpo_llm            0.46  0.32  0.05  0.00  0.00

dpo_llm states: 256-draw TR optimum distribution: {'set_x_pos': 177, 'observe_all': 151, 'set_y_pos': 139, 'set_x_neg': 39, 'set_xy': 31, 'set_y_neg': 21, 'set_y_zero': 18}
   rank of set_x_neg under TR at these states: [(1, 39), (2, 19), (3, 21), (4, 33), (5, 37), (6, 46), (7, 201), (8, 180)]
   by decision index: share of worlds where set_x_neg is TR-optimal / within 2 s.e. ; mean relative regret of set_x_neg ; modal TR optimum
     d= 0 n=24  opt 0.46  near 0.58  rel-regret 0.256  modal-opt [('set_x_neg', 11), ('set_x_pos', 6)]
     d= 1 n=24  opt 0.33  near 0.50  rel-regret 0.322  modal-opt [('set_x_neg', 8), ('set_xy', 7)]
     d= 2 n=24  opt 0.38  near 0.46  rel-regret 0.359  modal-opt [('set_x_neg', 9), ('set_x_pos', 6)]
     d= 3 n=24  opt 0.25  near 0.33  rel-regret 0.423  modal-opt [('set_x_neg', 6), ('set_x_pos', 5)]
     d= 4 n=24  opt 0.12  near 0.17  rel-regret 0.497  modal-opt [('set_y_pos', 9), ('observe_all', 6)]
     d= 5 n=24  opt 0.08  near 0.08  rel-regret 0.563  modal-opt [('set_x_pos', 7), ('set_y_pos', 5)]
     d= 6 n=24  opt 0.00  near 0.04  rel-regret 0.615  modal-opt [('observe_all', 7), ('set_x_pos', 7)]
     d= 7 n=24  opt 0.00  near 0.04  rel-regret 0.668  modal-opt [('set_x_pos', 7), ('set_y_pos', 6)]
     d= 8 n=24  opt 0.00  near 0.04  rel-regret 0.702  modal-opt [('set_y_pos', 7), ('observe_all', 7)]
     d= 9 n=24  opt 0.00  near 0.00  rel-regret 0.741  modal-opt [('set_x_pos', 8), ('observe_all', 7)]
     d=10 n=24  opt 0.00  near 0.00  rel-regret 0.758  modal-opt [('set_y_pos', 8), ('set_x_pos', 7)]
     d=11 n=24  opt 0.00  near 0.04  rel-regret 0.769  modal-opt [('set_x_pos', 8), ('observe_all', 7)]
     d=12 n=24  opt 0.00  near 0.04  rel-regret 0.786  modal-opt [('observe_all', 7), ('set_x_pos', 7)]
     d=13 n=24  opt 0.00  near 0.00  rel-regret 0.807  modal-opt [('set_x_pos', 9), ('observe_all', 8)]
     d=14 n=24  opt 0.00  near 0.00  rel-regret 0.819  modal-opt [('set_x_pos', 8), ('observe_all', 7)]
     d=15 n=24  opt 0.00  near 0.00  rel-regret 0.826  modal-opt [('observe_all', 9), ('set_x_pos', 8)]
     d=16 n=24  opt 0.00  near 0.00  rel-regret 0.831  modal-opt [('observe_all', 8), ('set_x_pos', 8)]
     d=17 n=24  opt 0.00  near 0.00  rel-regret 0.846  modal-opt [('observe_all', 8), ('set_y_pos', 7)]
     d=18 n=24  opt 0.00  near 0.00  rel-regret 0.853  modal-opt [('observe_all', 8), ('set_x_pos', 8)]
     d=19 n=24  opt 0.00  near 0.00  rel-regret 0.862  modal-opt [('set_x_pos', 8), ('set_y_pos', 7)]
     d=20 n=24  opt 0.00  near 0.00  rel-regret 0.871  modal-opt [('observe_all', 9), ('set_y_pos', 7)]
     d=21 n=24  opt 0.00  near 0.00  rel-regret 0.874  modal-opt [('set_x_pos', 9), ('observe_all', 7)]
     d=22 n=24  opt 0.00  near 0.00  rel-regret 0.883  modal-opt [('set_y_pos', 9), ('set_x_pos', 8)]
     d=23 n=24  opt 0.00  near 0.00  rel-regret 0.872  modal-opt [('set_x_pos', 11), ('observe_all', 8)]

frozen_llm states: 256-draw TR optimum distribution: {'set_x_neg': 238, 'set_x_pos': 163, 'set_xy': 123, 'set_y_pos': 20, 'observe_all': 18, 'set_y_neg': 11, 'set_y_zero': 3}
   rank of observe_all under TR at these states: [(1, 18), (2, 45), (3, 97), (4, 150), (5, 91), (6, 92), (7, 62), (8, 21)]
   by decision index: share of worlds where observe_all is TR-optimal / within 2 s.e. ; mean relative regret of observe_all ; modal TR optimum
     d= 0 n=24  opt 0.00  near 0.04  rel-regret 0.611  modal-opt [('set_x_neg', 11), ('set_x_pos', 6)]
     d= 1 n=24  opt 0.00  near 0.04  rel-regret 0.602  modal-opt [('set_x_neg', 13), ('set_x_pos', 7)]
     d= 2 n=24  opt 0.04  near 0.08  rel-regret 0.598  modal-opt [('set_x_neg', 12), ('set_x_pos', 5)]
     d= 3 n=24  opt 0.00  near 0.08  rel-regret 0.606  modal-opt [('set_x_neg', 12), ('set_x_pos', 6)]
     d= 4 n=24  opt 0.04  near 0.08  rel-regret 0.621  modal-opt [('set_x_neg', 9), ('set_xy', 6)]
     d= 5 n=24  opt 0.04  near 0.04  rel-regret 0.636  modal-opt [('set_x_neg', 12), ('set_x_pos', 5)]
     d= 6 n=24  opt 0.04  near 0.08  rel-regret 0.642  modal-opt [('set_x_neg', 11), ('set_x_pos', 6)]
     d= 7 n=24  opt 0.00  near 0.00  rel-regret 0.652  modal-opt [('set_x_neg', 11), ('set_x_pos', 7)]
     d= 8 n=24  opt 0.04  near 0.08  rel-regret 0.641  modal-opt [('set_x_neg', 10), ('set_x_pos', 7)]
     d= 9 n=24  opt 0.04  near 0.08  rel-regret 0.657  modal-opt [('set_x_neg', 11), ('set_x_pos', 7)]
     d=10 n=24  opt 0.04  near 0.08  rel-regret 0.660  modal-opt [('set_x_neg', 10), ('set_x_pos', 7)]
     d=11 n=24  opt 0.00  near 0.12  rel-regret 0.659  modal-opt [('set_x_neg', 11), ('set_x_pos', 7)]
     d=12 n=24  opt 0.04  near 0.12  rel-regret 0.669  modal-opt [('set_x_neg', 8), ('set_x_pos', 8)]
     d=13 n=24  opt 0.08  near 0.08  rel-regret 0.669  modal-opt [('set_x_neg', 8), ('set_x_pos', 7)]
     d=14 n=24  opt 0.04  near 0.08  rel-regret 0.677  modal-opt [('set_x_neg', 8), ('set_x_pos', 6)]
     d=15 n=24  opt 0.00  near 0.08  rel-regret 0.679  modal-opt [('set_x_neg', 9), ('set_xy', 6)]
     d=16 n=24  opt 0.04  near 0.08  rel-regret 0.689  modal-opt [('set_x_neg', 8), ('set_x_pos', 7)]
     d=17 n=24  opt 0.00  near 0.08  rel-regret 0.684  modal-opt [('set_x_neg', 10), ('set_x_pos', 7)]
     d=18 n=24  opt 0.04  near 0.08  rel-regret 0.681  modal-opt [('set_x_neg', 8), ('set_xy', 7)]
     d=19 n=24  opt 0.04  near 0.04  rel-regret 0.687  modal-opt [('set_x_neg', 10), ('set_x_pos', 8)]
     d=20 n=24  opt 0.04  near 0.08  rel-regret 0.683  modal-opt [('set_x_neg', 8), ('set_x_pos', 8)]
     d=21 n=24  opt 0.04  near 0.04  rel-regret 0.688  modal-opt [('set_x_neg', 9), ('set_x_pos', 8)]
     d=22 n=24  opt 0.04  near 0.04  rel-regret 0.689  modal-opt [('set_x_neg', 9), ('set_x_pos', 8)]
     d=23 n=24  opt 0.04  near 0.08  rel-regret 0.668  modal-opt [('set_x_neg', 10), ('set_x_pos', 8)]

sft_llm states: 256-draw TR optimum distribution: {'set_x_pos': 145, 'observe_all': 141, 'set_y_pos': 100, 'set_xy': 80, 'set_x_neg': 41, 'set_y_neg': 26, 'set_y_zero': 13}
   rank of set_x_neg under TR at these states: [(1, 41), (2, 19), (3, 25), (4, 24), (5, 29), (6, 68), (7, 209), (8, 131)]
   by decision index: share of worlds where set_x_neg is TR-optimal / within 2 s.e. ; mean relative regret of set_x_neg ; modal TR optimum
     d= 0 n=24  opt 0.46  near 0.58  rel-regret 0.256  modal-opt [('set_x_neg', 11), ('set_x_pos', 6)]
     d= 1 n=24  opt 0.46  near 0.54  rel-regret 0.312  modal-opt [('set_x_neg', 11), ('set_x_pos', 6)]
     d= 2 n=24  opt 0.42  near 0.42  rel-regret 0.354  modal-opt [('set_x_neg', 10), ('set_x_pos', 6)]
     d= 3 n=24  opt 0.17  near 0.25  rel-regret 0.436  modal-opt [('set_xy', 7), ('set_x_pos', 5)]
     d= 4 n=24  opt 0.17  near 0.21  rel-regret 0.502  modal-opt [('set_xy', 7), ('observe_all', 5)]
     d= 5 n=24  opt 0.04  near 0.12  rel-regret 0.567  modal-opt [('observe_all', 7), ('set_xy', 6)]
     d= 6 n=24  opt 0.00  near 0.00  rel-regret 0.633  modal-opt [('observe_all', 7), ('set_y_pos', 6)]
     d= 7 n=24  opt 0.00  near 0.00  rel-regret 0.674  modal-opt [('set_x_pos', 7), ('observe_all', 6)]
     d= 8 n=24  opt 0.00  near 0.00  rel-regret 0.710  modal-opt [('observe_all', 8), ('set_x_pos', 6)]
     d= 9 n=24  opt 0.00  near 0.00  rel-regret 0.739  modal-opt [('observe_all', 7), ('set_y_pos', 6)]
     d=10 n=24  opt 0.00  near 0.04  rel-regret 0.755  modal-opt [('observe_all', 9), ('set_x_pos', 6)]
     d=11 n=24  opt 0.00  near 0.00  rel-regret 0.779  modal-opt [('observe_all', 8), ('set_x_pos', 7)]
     d=12 n=24  opt 0.00  near 0.00  rel-regret 0.787  modal-opt [('set_x_pos', 9), ('observe_all', 6)]
     d=13 n=24  opt 0.00  near 0.00  rel-regret 0.786  modal-opt [('set_x_pos', 8), ('observe_all', 7)]
     d=14 n=24  opt 0.00  near 0.00  rel-regret 0.790  modal-opt [('set_x_pos', 7), ('set_y_pos', 6)]
     d=15 n=24  opt 0.00  near 0.00  rel-regret 0.802  modal-opt [('set_x_pos', 7), ('set_y_pos', 6)]
     d=16 n=24  opt 0.00  near 0.00  rel-regret 0.812  modal-opt [('observe_all', 7), ('set_x_pos', 6)]
     d=17 n=24  opt 0.00  near 0.00  rel-regret 0.820  modal-opt [('observe_all', 7), ('set_x_pos', 7)]
     d=18 n=24  opt 0.00  near 0.00  rel-regret 0.833  modal-opt [('observe_all', 8), ('set_x_pos', 7)]
     d=19 n=24  opt 0.00  near 0.00  rel-regret 0.839  modal-opt [('observe_all', 9), ('set_y_pos', 6)]
     d=20 n=24  opt 0.00  near 0.00  rel-regret 0.848  modal-opt [('observe_all', 9), ('set_x_pos', 6)]
     d=21 n=24  opt 0.00  near 0.00  rel-regret 0.855  modal-opt [('set_y_pos', 7), ('observe_all', 7)]
     d=22 n=16  opt 0.00  near 0.00  rel-regret 0.894  modal-opt [('observe_all', 6), ('set_x_pos', 5)]
     d=23 n= 2  opt 0.00  near 0.00  rel-regret 0.840  modal-opt [('set_x_pos', 1), ('observe_all', 1)]

learned_ranker states: 256-draw TR optimum distribution: {'observe_all': 211, 'set_y_pos': 128, 'set_xy': 77, 'set_x_neg': 72, 'set_y_neg': 44, 'set_x_pos': 28, 'set_y_zero': 16}

Shared first decision with 256-draw scores:
   TR optimum counts (256 draws): {'set_x_neg': 11, 'set_x_pos': 6, 'set_xy': 5, 'set_y_pos': 1, 'set_y_neg': 1}  | set_x_neg rank: [(1, 11), (2, 3), (3, 2), (4, 2), (5, 1), (7, 3), (8, 2)]
   gap best-second: median 0.00471 mean 0.01194 ; in MC s.e.: [40.9, 11.4, 0.6, 89.5, 42.4, 1.3, 15.5, 30.2, 0.2, 21.1, 2.3, 17.8, 91.0, 51.9, 0.4, 23.3, 21.1, 5.2, 20.1, 29.2, 0.8, 4.6, 11.9, 3.3] ; <2 s.e. in 5/24
   gap best-set_x_neg: median 0.00008 mean 0.00896 ; set_x_neg within 2 s.e. of best in 14/24
   arm                agree  near-opt  mean_regret  mean_rel
   uniform_random      6/24   7/24    0.01671     0.461
   menu_coverage      11/24  14/24    0.00896     0.256
   local_information  17/24  20/24    0.00077     0.042
   target_risk        24/24  24/24    0.00000     0.000
   learned_ranker     14/24  17/24    0.00248     0.158
   frozen_llm          0/24   1/24    0.02859     0.611
   sft_llm            11/24  14/24    0.00896     0.256
   dpo_llm            11/24  14/24    0.00896     0.256
```

## Appendix: out/s17.out

```
acquisitions matched to curve points: 4348

Calibration of the SCM posterior risk against private NMSE at the same state (all arms, all decisions):
   mean risk_before 0.04564 ; mean realised NMSE 0.04534 ; ratio NMSE/risk 0.994 ; Spearman rho 0.640 ; Pearson r(log,log) 0.654
   regime  matched_high  n=1090 mean risk 0.04024 mean NMSE 0.02621 ratio 0.65
   regime  matched_low   n=1080 mean risk 0.01801 mean NMSE 0.01718 ratio 0.95
   regime  shifted_high  n=1088 mean risk 0.07328 mean NMSE 0.08503 ratio 1.16
   regime  shifted_low   n=1090 mean risk 0.05083 mean NMSE 0.05277 ratio 1.04
   graph   chain         n=1456 mean risk 0.05954 mean NMSE 0.06641 ratio 1.12
   graph   collider      n=1436 mean risk 0.04932 mean NMSE 0.05129 ratio 1.04
   graph   fork          n=1456 mean risk 0.02811 mean NMSE 0.01842 ratio 0.66
   at the shared initial state (24 worlds): risk_before vs NMSE0: Spearman rho 0.775 ; mean risk 0.1074 mean NMSE 0.1228

Predicted one-step risk reduction (model) vs realised one-step NMSE reduction (single noisy response):
   Spearman rho 0.123 ; Pearson r 0.547 ; mean predicted 0.00362 ; mean realised 0.00423 ; realised<0 (NMSE got worse) in 43.9% of acquisitions
   deciles of predicted reduction: mean predicted -> mean realised (n)
      0.00014 -> +0.00009 (n=435, realised>0 in 54%)
      0.00031 -> +0.00044 (n=435, realised>0 in 58%)
      0.00048 -> +0.00063 (n=435, realised>0 in 57%)
      0.00069 -> +0.00056 (n=435, realised>0 in 52%)
      0.00096 -> +0.00114 (n=435, realised>0 in 53%)
      0.00134 -> +0.00063 (n=435, realised>0 in 53%)
      0.00188 -> +0.00218 (n=435, realised>0 in 56%)
      0.00276 -> +0.00235 (n=435, realised>0 in 55%)
      0.00482 -> +0.00417 (n=434, realised>0 in 59%)
      0.02286 -> +0.03020 (n=434, realised>0 in 65%)

Episode level: sum of predicted reductions vs total realised NMSE reduction (192 cells): 
   Spearman rho 0.702 ; Pearson r 0.726 ; mean predicted 0.0820 mean realised 0.0959

Arm level: mean relative regret (own trajectory) vs the report endpoints:
   policies.csv columns: ['policy_id', 'label', 'world_count', 'mean_capped_total_cost', 'mean_capped_acquisition_cost', 'ever_attained', 'final_attained', 'mean_final_error', 'failed', 'unresolved', 'actual_total_cost', 'stopped']
   uniform_random     mean rel regret 0.504 | mean capped total cost 14.250 | mean final NMSE 0.04113
   menu_coverage      mean rel regret 0.441 | mean capped total cost 12.458 | mean final NMSE 0.01974
   local_information  mean rel regret 0.154 | mean capped total cost 11.958 | mean final NMSE 0.01800
   target_risk        mean rel regret 0.016 | mean capped total cost 11.625 | mean final NMSE 0.01722
   learned_ranker     mean rel regret 0.575 | mean capped total cost 13.583 | mean final NMSE 0.03572
   frozen_llm         mean rel regret 0.655 | mean capped total cost 13.625 | mean final NMSE 0.02008
   sft_llm            mean rel regret 0.665 | mean capped total cost 14.625 | mean final NMSE 0.02708
   dpo_llm            mean rel regret 0.700 | mean capped total cost 15.167 | mean final NMSE 0.03642
   across the 8 arms: Spearman(rel regret, capped cost) = 0.929 ; Spearman(rel regret, final NMSE) = 0.690
```

## Appendix: out/s12_behavioral_sensitivity.out

```
Action entropy H(A) in bits (max log2(9)=3.17) and mutual information I(A; feature) in bits.
MI shown as raw / permutation-bias-corrected (p). 300 permutations.
arm                    n   H(A) | graph                  cost menu              regime                 step tercile           remaining units       
uniform_random       487  2.995 | 0.014 / -0.007 (p=0.787) 0.012 / 0.001 (p=0.322) 0.023 / -0.008 (p=0.811) 0.032 / 0.012 (p=0.083) 0.257 / -0.013 (p=0.691)
menu_coverage        480  2.791 | 0.000 / -0.019 (p=1.000) 0.003 / -0.006 (p=0.920) 0.000 / -0.027 (p=1.000) 0.041 / 0.022 (p=0.010) 2.241 / 2.008 (p=0.003)
local_information    570  1.669 | 0.102 / 0.089 (p=0.003) 0.027 / 0.020 (p=0.003) 0.096 / 0.075 (p=0.003) 0.076 / 0.062 (p=0.003) 0.267 / 0.123 (p=0.003)
target_risk          560  2.389 | 0.110 / 0.091 (p=0.003) 0.167 / 0.158 (p=0.003) 0.076 / 0.047 (p=0.003) 0.058 / 0.039 (p=0.003) 0.279 / 0.067 (p=0.003)
learned_ranker       576  0.803 | 0.001 / -0.002 (p=0.711) 0.001 / -0.001 (p=0.442) 0.009 / 0.006 (p=0.073) 0.362 / 0.359 (p=0.003) 0.429 / 0.399 (p=0.003)
frozen_llm           576  0.033 | 0.006 / 0.002 (p=0.336) 0.003 / 0.002 (p=0.565) 0.007 / 0.003 (p=0.233) 0.002 / -0.001 (p=1.000) 0.013 / -0.000 (p=0.857)
sft_llm              546  0.749 | 0.039 / 0.028 (p=0.003) 0.007 / 0.001 (p=0.306) 0.017 / 0.001 (p=0.399) 0.158 / 0.147 (p=0.003) 0.367 / 0.241 (p=0.003)
dpo_llm              576 -0.000 | 0 / 0 (n/a)            0 / 0 (n/a)            0 / 0 (n/a)            0 / 0 (n/a)            0 / 0 (n/a)           

Per-arm within-episode variability: mean number of distinct commands per world, mean number of action switches per world
  uniform_random     distinct 7.50 (min 6 max 8)  switches 17.08 (min 11 max 23)
  menu_coverage      distinct 7.00 (min 7 max 7)  switches 19.00 (min 15 max 23)
  local_information  distinct 2.58 (min 2 max 4)  switches 12.88 (min 3 max 21)
  target_risk        distinct 3.67 (min 2 max 6)  switches 14.83 (min 9 max 20)
  learned_ranker     distinct 1.88 (min 1 max 2)  switches 1.12 (min 0 max 3)
  frozen_llm         distinct 1.04 (min 1 max 2)  switches 0.17 (min 0 max 4)
  sft_llm            distinct 3.50 (min 2 max 5)  switches 3.71 (min 1 max 6)
  dpo_llm            distinct 1.00 (min 1 max 1)  switches 0.00 (min 0 max 0)

Number of distinct complete action sequences across the 24 worlds:
  uniform_random     24 distinct sequences (largest group 1 worlds)
  menu_coverage      2 distinct sequences (largest group 12 worlds)
  local_information  24 distinct sequences (largest group 1 worlds)
  target_risk        24 distinct sequences (largest group 1 worlds)
  learned_ranker     9 distinct sequences (largest group 10 worlds)
  frozen_llm         2 distinct sequences (largest group 23 worlds)
  sft_llm            19 distinct sequences (largest group 3 worlds)
  dpo_llm            1 distinct sequences (largest group 24 worlds)

SFT: command by remaining_units (training snapshots were taken at remaining 24, 18(17), 12(11), 6(5), 0):
   remaining=24 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=23 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=22 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=21 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=20 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=19 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=18 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=17 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=16 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=15 n=24  x_neg=24  other_acq= 0  stop= 0   {}
   remaining=14 n=24  x_neg=23  other_acq= 1  stop= 0   {'set_y_neg': 1}
   remaining=13 n=23  x_neg=18  other_acq= 5  stop= 0   {'set_y_neg': 3, 'set_x_pos': 2}
   remaining=12 n=24  x_neg=11  other_acq=13  stop= 0   {'set_x_pos': 9, 'set_y_neg': 4}
   remaining=11 n=21  x_neg=15  other_acq= 6  stop= 0   {'set_y_neg': 1, 'set_x_pos': 5}
   remaining=10 n=24  x_neg=23  other_acq= 1  stop= 0   {'set_x_pos': 1}
   remaining= 9 n=24  x_neg=23  other_acq= 1  stop= 0   {'set_x_pos': 1}
   remaining= 8 n=24  x_neg=22  other_acq= 2  stop= 0   {'set_x_pos': 2}
   remaining= 7 n=24  x_neg=21  other_acq= 3  stop= 0   {'set_y_neg': 2, 'set_x_pos': 1}
   remaining= 6 n=22  x_neg=18  other_acq= 4  stop= 0   {'observe_all': 2, 'set_x_pos': 2}
   remaining= 5 n=24  x_neg=21  other_acq= 3  stop= 0   {'observe_all': 3}
   remaining= 4 n=24  x_neg=22  other_acq= 2  stop= 0   {'observe_all': 2}
   remaining= 3 n=24  x_neg=18  other_acq= 1  stop= 5   {'STOP': 5, 'observe_all': 1}
   remaining= 2 n=19  x_neg= 5  other_acq= 0  stop=14   {'STOP': 14}
   remaining= 1 n= 5  x_neg= 1  other_acq= 0  stop= 4   {'STOP': 4}
   non-x_neg acquisitions at remaining in {11,12,13}: 24 of 68 (35.3%) ; elsewhere: 18 of 455 (4.0%) ; Fisher exact p=6.38e-13

SFT: worlds (of 8 per graph) with >=1 set_y_neg / >=1 observe_all / >=1 set_x_pos:
   chain    y_neg 3/8  observe 0/8  x_pos 7/8
   fork     y_neg 0/8  observe 0/8  x_pos 8/8
   collider y_neg 8/8  observe 3/8  x_pos 8/8
   Fisher exact, set_y_neg present: collider 8/8 vs fork 0/8: p=0.00016 ; collider 8/8 vs chain 3/8: p=0.0256
   cost of set_y_neg when chosen: {('uniform', 1): 5, ('heterogeneous', 2): 6}
```

## Appendix: out/s18_agreement.out

```
Shared first decision (identical public state in all 8 arms): pairwise agreement counts out of 24 worlds
          unif   menu  local     TR ranker frozen    sft    dpo
unif        24      4      4      6      4      1      4      4
menu         4     24     16     11     21      0     24     24
local        4     16     24     17     19      0     16     16
TR           6     11     17     24     14      0     11     11
ranker       4     21     19     14     24      0     21     21
frozen       1      0      0      0      0     24      0      0
sft          4     24     16     11     21      0     24     24
dpo          4     24     16     11     21      0     24     24

states (envelope sha256) shared by more than one arm: 24 ; all at decision 0: True ; arms per shared state: Counter({8: 24})
arms choosing the same first action still diverge at decision 1 (independent per-arm measurement streams): e.g. world 0, decision-1 envelope hashes of menu_coverage / sft / dpo (all chose set_x_neg at d=0): ['302dd33ffc', 'e86f2eea1f', 'ea0b4cd0a4']

Matched (world, decision index) agreement over all indexes where both arms made a decision (states differ after d=0; descriptive only):
          unif   menu  local     TR ranker frozen    sft    dpo
unif    100.0%  13.2%  14.4%  12.5%  14.4%  13.3%  12.1%  12.1%
menu     13.2% 100.0%   9.4%  12.5%  17.9%   0.0%  17.4%  17.5%
local    14.4%   9.4% 100.0%  42.5%  22.5%  55.4%  24.9%  26.1%
TR       12.5%  12.5%  42.5% 100.0%  22.7%  38.2%  20.9%  23.0%
ranker   14.4%  17.9%  22.5%  22.7% 100.0%   0.0%  29.7%  24.5%
frozen   13.3%   0.0%  55.4%  38.2%   0.0% 100.0%   1.5%   0.0%
sft      12.1%  17.4%  24.9%  20.9%  29.7%   1.5% 100.0%  88.1%
dpo      12.1%  17.5%  26.1%  23.0%  24.5%   0.0%  88.1% 100.0%

ranker vs target_risk at the first decision, by ranker choice:
   ranker chose set_x_neg in 21 worlds; target_risk chose there: {'set_x_neg': 11, 'set_x_pos': 3, 'set_xy': 5, 'set_y_pos': 1, 'set_y_neg': 1}
   ranker chose set_x_pos in 3 worlds; target_risk chose there: {'set_x_pos': 3}
```

## Appendix: out/s13_tr_own_saved.out

```
target_risk decisions 560 feasible-count distribution Counter({8: 548, 4: 12})
gap best-second (saved 32-draw scores): median 0.00006 mean 0.00089 ; relative to best score: median 0.066
gap in combined MC s.e. units: median 0.96 ; <1 s.e.: 289 (51.6%) ; <2 s.e.: 419 (74.8%) ; >=3 s.e.: 82 (14.6%)
negative estimated gains among candidate scores: 1 ; decisions whose best score <= 0: 0
rank of set_x_neg at target_risk own states: [(1, 129), (2, 124), (3, 96), (4, 46), (5, 41), (6, 57), (7, 44), (8, 23)]

by decision index: chosen action counts, median z-gap
   d= 0 n=24 medz= 4.83 {'set_x_neg': 11, 'set_x_pos': 6, 'set_xy': 5, 'set_y_pos': 1, 'set_y_neg': 1}
   d= 1 n=24 medz= 2.83 {'set_x_neg': 14, 'set_x_pos': 5, 'set_xy': 2, 'set_y_pos': 1, 'set_y_neg': 1, 'observe_all': 1}
   d= 2 n=24 medz= 2.05 {'set_x_neg': 9, 'set_x_pos': 5, 'observe_all': 4, 'set_xy': 3, 'set_y_pos': 3}
   d= 3 n=24 medz= 1.72 {'observe_all': 6, 'set_x_neg': 5, 'set_y_pos': 4, 'set_x_pos': 4, 'set_xy': 3, 'set_x_zero': 1, 'set_y_neg': 1}
   d= 4 n=24 medz= 0.72 {'set_x_neg': 9, 'observe_all': 5, 'set_x_pos': 3, 'set_y_pos': 3, 'set_y_neg': 2, 'set_xy': 1, 'set_y_zero': 1}
   d= 5 n=24 medz= 1.14 {'set_y_pos': 7, 'observe_all': 7, 'set_x_neg': 6, 'set_x_pos': 3, 'set_y_neg': 1}
   d= 6 n=24 medz= 0.84 {'observe_all': 11, 'set_x_neg': 5, 'set_x_pos': 3, 'set_y_zero': 2, 'set_y_pos': 1, 'set_xy': 1, 'set_x_zero': 1}
   d= 7 n=24 medz= 0.92 {'observe_all': 11, 'set_y_pos': 5, 'set_x_neg': 5, 'set_x_pos': 2, 'set_y_neg': 1}
   d= 8 n=24 medz= 0.73 {'observe_all': 10, 'set_x_neg': 4, 'set_x_pos': 4, 'set_y_pos': 3, 'set_xy': 1, 'set_y_neg': 1, 'set_y_zero': 1}
   d= 9 n=24 medz= 0.70 {'observe_all': 11, 'set_x_neg': 8, 'set_y_pos': 3, 'set_y_neg': 1, 'set_x_pos': 1}
   d=10 n=24 medz= 1.04 {'observe_all': 9, 'set_x_pos': 4, 'set_x_neg': 3, 'set_xy': 3, 'set_y_pos': 2, 'set_y_neg': 2, 'set_y_zero': 1}
   d=11 n=24 medz= 1.13 {'observe_all': 14, 'set_x_neg': 3, 'set_y_pos': 3, 'set_y_zero': 1, 'set_xy': 1, 'set_x_pos': 1, 'set_y_neg': 1}
   d=12 n=24 medz= 0.80 {'observe_all': 12, 'set_x_neg': 4, 'set_x_pos': 3, 'set_y_zero': 2, 'set_y_pos': 2, 'set_y_neg': 1}
   d=13 n=24 medz= 1.17 {'observe_all': 8, 'set_x_neg': 7, 'set_y_pos': 3, 'set_xy': 2, 'set_x_pos': 2, 'set_y_neg': 2}
   d=14 n=24 medz= 0.65 {'observe_all': 11, 'set_x_neg': 4, 'set_y_neg': 3, 'set_x_pos': 3, 'set_y_zero': 2, 'set_y_pos': 1}
   d=15 n=24 medz= 1.17 {'observe_all': 12, 'set_x_neg': 3, 'set_x_pos': 3, 'set_y_pos': 2, 'set_xy': 2, 'set_y_zero': 1, 'set_y_neg': 1}
   d=16 n=24 medz= 1.08 {'observe_all': 13, 'set_x_pos': 3, 'set_x_neg': 3, 'set_xy': 2, 'set_y_neg': 2, 'set_y_pos': 1}
   d=17 n=24 medz= 0.67 {'observe_all': 12, 'set_x_pos': 4, 'set_x_neg': 3, 'set_y_pos': 2, 'set_y_neg': 2, 'set_y_zero': 1}
   d=18 n=24 medz= 1.12 {'observe_all': 10, 'set_x_neg': 6, 'set_y_pos': 3, 'set_x_pos': 3, 'set_y_neg': 1, 'set_y_zero': 1}
   d=19 n=23 medz= 0.99 {'observe_all': 10, 'set_x_neg': 6, 'set_x_pos': 3, 'set_y_pos': 3, 'set_y_zero': 1}
   d=20 n=22 medz= 0.53 {'observe_all': 10, 'set_x_neg': 3, 'set_y_pos': 3, 'set_y_neg': 3, 'set_xy': 3}
   d=21 n=20 medz= 0.87 {'observe_all': 9, 'set_x_pos': 4, 'set_x_neg': 3, 'set_y_pos': 2, 'set_y_neg': 1, 'set_y_zero': 1}
   d=22 n=20 medz= 0.44 {'observe_all': 10, 'set_x_pos': 5, 'set_xy': 2, 'set_x_neg': 2, 'set_y_neg': 1}
   d=23 n=19 medz= 0.54 {'observe_all': 9, 'set_y_pos': 4, 'set_x_neg': 3, 'set_x_pos': 2, 'set_y_zero': 1}

risk_before along target_risk episodes: mean by decision index 0,4,8,12,16,20: [0.10741, 0.04682, 0.0349, 0.02684, 0.02278, 0.02028]

local_information: relative gap best-second (saved scores): median 0.064 ; <5%: 231 of 570
```

## Appendix: out/s19.out

```
RANKER (564 full-menu decisions): contrast set_x_neg - set_x_pos, ranker score vs TR score at the same state
   Pearson r=+0.198 Spearman rho=-0.137 ; sign agreement 205/564 (36.3%)
   decisions  0- 3: sign agreement 83/96 ; TR prefers x_neg over x_pos in 64 ; ranker prefers x_neg in 75
   decisions  4- 7: sign agreement 55/96 ; TR prefers x_neg over x_pos in 46 ; ranker prefers x_neg in 51
   decisions  8-11: sign agreement 37/96 ; TR prefers x_neg over x_pos in 50 ; ranker prefers x_neg in 15
   decisions 12-23: sign agreement 30/276 ; TR prefers x_neg over x_pos in 246 ; ranker prefers x_neg in 0
   per-state Spearman rho between the ranker's 8 action scores and the 8 TR scores: mean -0.241 median -0.405 ; >0 in 26.4%
      decisions  0- 0: mean rho +0.339
      decisions  1- 3: mean rho +0.118
      decisions  4- 7: mean rho -0.216
      decisions  8-23: mean rho -0.354
   same, using the ranker's state-AVERAGED score vector at every state: mean rho -0.257 (difference from state-specific: +0.0158)

SFT: the 42 non-set_x_neg acquisitions. TR score(chosen) - TR score(set_x_neg) at the same state (per cost unit):
   chosen better than set_x_neg in 36/42 ; mean gain +0.00163 ; median +0.00069 ; |diff| > 2 MC s.e. in 40 (of which chosen better: 35)
   set_x_pos   n=23 better than x_neg in 22 ; mean gain +0.00246 ; TR rank of the chosen action: [(1, 6), (2, 2), (3, 2), (4, 6), (5, 5), (6, 2)]
   set_y_neg   n=11 better than x_neg in  6 ; mean gain +0.00057 ; TR rank of the chosen action: [(2, 1), (5, 2), (6, 2), (7, 1), (8, 5)]
   observe_all n= 8 better than x_neg in  8 ; mean gain +0.00072 ; TR rank of the chosen action: [(4, 4), (5, 1), (6, 3)]
Traceback (most recent call last):
  File "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics/s19_direction_of_sensitivity.py", line 51, in <module>
    len(late), 100 * np.mean([S(r)['set_x_pos'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['observe_all'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['set_y_neg'] > S(r)['set_x_neg'] for r in late])))
  File "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics/s19_direction_of_sensitivity.py", line 51, in <listcomp>
    len(late), 100 * np.mean([S(r)['set_x_pos'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['observe_all'] > S(r)['set_x_neg'] for r in late]), 100 * np.mean([S(r)['set_y_neg'] > S(r)['set_x_neg'] for r in late])))
KeyError: 'set_y_neg'
SFT: the 42 non-set_x_neg acquisitions. TR score(chosen) - TR score(set_x_neg) at the same state (per cost unit):
   chosen better than set_x_neg in 36/42 ; mean gain +0.00163 ; median +0.00069 ; |diff| > 2 MC s.e. in 40 (of which chosen better: 35)
   set_x_pos   n=23 better than x_neg in 22 ; mean gain +0.00246 ; TR rank of the chosen action: [(1, 6), (2, 2), (3, 2), (4, 6), (5, 5), (6, 2)]
   set_y_neg   n=11 better than x_neg in  6 ; mean gain +0.00057 ; TR rank of the chosen action: [(2, 1), (5, 2), (6, 2), (7, 1), (8, 5)]
   observe_all n= 8 better than x_neg in  8 ; mean gain +0.00072 ; TR rank of the chosen action: [(4, 4), (5, 1), (6, 3)]
   reference: at all 378 SFT acquisition states with decision index >= 6, TR prefers set_x_pos over set_x_neg in 93.7%, observe_all over set_x_neg in 96.8%, set_y_neg over set_x_neg in 68.5%
   mean relative regret: at the 42 deviation decisions 0.469 ; at SFT set_x_neg decisions with the same decision indexes (10-21) 0.813
   TR value of set_y_neg relative to best, by graph, over all SFT acquisition states (mean of score/best): {'chain': 0.269, 'fork': 0.614, 'collider': 0.331}

SFT stops (23): best available TR gain per cost unit forgone at the stop state: mean 0.00747 median 0.00253 ; risk_before at stop: mean 0.03920 ; for comparison risk_before at the shared initial states: mean 0.10741
   posterior integrated risk at stop <= 0.04 (public analogue of the private target) in 17/23 ; min 0.0065 max 0.2446
   stop decision index vs posterior risk at stop: Spearman rho +0.544 (p=0.01) ; vs remaining units: all stops at remaining<=3
```

## Appendix: out/s22.out

```
DEVELOPMENT, shared first decision (24 worlds): sign of posterior mean of X vs 256-draw TR optimum and each policy's choice
    ('meanX<0', 'TR prefers x_pos') 7
    ('meanX<0', 'TR-opt=set_x_pos') 6
    ('meanX<0', 'TR-opt=set_y_neg') 1
    ('meanX>0', 'TR prefers x_neg') 17
    ('meanX>0', 'TR-opt=set_x_neg') 11
    ('meanX>0', 'TR-opt=set_xy') 5
    ('meanX>0', 'TR-opt=set_y_pos') 1
   uniform_random     {('meanX<0', 'set_x_neg'): 1, ('meanX<0', 'set_x_pos'): 1, ('meanX<0', 'set_x_zero'): 1, ('meanX<0', 'set_xy'): 2, ('meanX<0', 'set_y_neg'): 1, ('meanX<0', 'set_y_zero'): 1, ('meanX>0', 'observe_all'): 1, ('meanX>0', 'set_x_neg'): 3, ('meanX>0', 'set_x_pos'): 2, ('meanX>0', 'set_x_zero'): 1, ('meanX>0', 'set_xy'): 4, ('meanX>0', 'set_y_neg'): 3, ('meanX>0', 'set_y_zero'): 3}
   menu_coverage      {('meanX<0', 'set_x_neg'): 7, ('meanX>0', 'set_x_neg'): 17}
   local_information  {('meanX<0', 'set_x_pos'): 6, ('meanX<0', 'set_y_neg'): 1, ('meanX>0', 'set_x_neg'): 16, ('meanX>0', 'set_y_pos'): 1}
   target_risk        {('meanX<0', 'set_x_pos'): 6, ('meanX<0', 'set_y_neg'): 1, ('meanX>0', 'set_x_neg'): 11, ('meanX>0', 'set_xy'): 5, ('meanX>0', 'set_y_pos'): 1}
   learned_ranker     {('meanX<0', 'set_x_neg'): 4, ('meanX<0', 'set_x_pos'): 3, ('meanX>0', 'set_x_neg'): 17}
   frozen_llm         {('meanX<0', 'observe_all'): 7, ('meanX>0', 'observe_all'): 17}
   sft_llm            {('meanX<0', 'set_x_neg'): 7, ('meanX>0', 'set_x_neg'): 17}
   dpo_llm            {('meanX<0', 'set_x_neg'): 7, ('meanX>0', 'set_x_neg'): 17}
   one-feature rule "X mean > 0 -> set_x_neg else set_x_pos": TR-optimal in 17/24 ; mean regret 0.00117 ; mean relative regret 0.055  (always set_x_neg: 11/24, 0.00896, 0.256)

TRAINING snapshot 0 (32 worlds, same kind of state: 8 observations): sign of posterior mean of X vs TR optimum vs supervised label
    ('meanX<0', 'TR prefers x_neg') 2
    ('meanX<0', 'TR prefers x_pos') 12
    ('meanX>0', 'TR prefers x_neg') 18
    ('meanX<0', 'TR-opt=set_x_neg') 1
    ('meanX<0', 'TR-opt=set_x_pos') 10
    ('meanX<0', 'TR-opt=set_xy') 2
    ('meanX<0', 'TR-opt=set_y_neg') 1
    ('meanX>0', 'TR-opt=observe_all') 1
    ('meanX>0', 'TR-opt=set_x_neg') 10
    ('meanX>0', 'TR-opt=set_xy') 3
    ('meanX>0', 'TR-opt=set_y_neg') 1
    ('meanX>0', 'TR-opt=set_y_pos') 3
    ('meanX<0', 'label=observe_all') 1
    ('meanX<0', 'label=set_x_neg') 6
    ('meanX<0', 'label=set_x_pos') 2
    ('meanX<0', 'label=set_x_zero') 1
    ('meanX<0', 'label=set_xy') 1
    ('meanX<0', 'label=set_y_neg') 2
    ('meanX<0', 'label=set_y_zero') 1
    ('meanX>0', 'label=observe_all') 2
    ('meanX>0', 'label=set_x_neg') 10
    ('meanX>0', 'label=set_x_zero') 2
    ('meanX>0', 'label=set_xy') 2
    ('meanX>0', 'label=set_y_neg') 1
    ('meanX>0', 'label=set_y_zero') 1

ALL STATES (development, every arm, every decision with both X actions feasible): does sign(meanX) predict the TR preference between set_x_neg and set_x_pos?
   agreement 3313/4371 (75.8%) ; by decision-index block of 4: {0: '0.96', 4: '0.81', 8: '0.68', 12: '0.70', 16: '0.68', 20: '0.70'}
   (after the first steps the preference is driven by which X value has already been sampled, not by the sign of the mean)
   rule "choose the X extreme sampled less often so far; if tied use the sign rule": agreement with TR preference 2617/4371 (59.9%)
```

## Appendix: out/s20.out

```
training states scored: 128

(a) supervised (SFT) label vs 256-draw TR optimum at the 128 non-terminal training states:
   label == TR optimum: 55/128 (43.0%) ; within 2 MC s.e. of optimum: 67/128 ; mean relative regret of the label 0.225 ; TR rank of label: [(1, 55), (2, 14), (3, 12), (4, 17), (5, 7), (6, 9), (7, 8), (8, 6)]
   TR optimum distribution at training states: {'set_x_pos': 34, 'observe_all': 31, 'set_x_neg': 30, 'set_xy': 16, 'set_y_pos': 12, 'set_y_neg': 5}
   snapshot 0: label==TR-opt 15/32 ; near-opt 19 ; mean rel regret 0.255 ; TR-opt dist {'set_x_neg': 11, 'set_x_pos': 10, 'set_xy': 5, 'set_y_pos': 3}
   snapshot 1: label==TR-opt 19/32 ; near-opt 21 ; mean rel regret 0.114 ; TR-opt dist {'set_x_neg': 8, 'observe_all': 8, 'set_x_pos': 8, 'set_xy': 4}
   snapshot 2: label==TR-opt 12/32 ; near-opt 15 ; mean rel regret 0.223 ; TR-opt dist {'observe_all': 10, 'set_x_pos': 8, 'set_x_neg': 6, 'set_xy': 3}
   snapshot 3: label==TR-opt 9/32 ; near-opt 12 ; mean rel regret 0.307 ; TR-opt dist {'observe_all': 12, 'set_x_pos': 8, 'set_x_neg': 5, 'set_xy': 4}
   reference: uniform-random expected relative regret at these states: 0.438 ; always-set_x_neg: 0.302

(b) measured preference pairs vs TR score at the same training state:
   strict pairs: 687 ; preference-chosen action has the higher TR score in 471 (68.6%) ; binomial p vs 0.5 = 1.1e-22
   among strict pairs whose TR difference exceeds 2 MC s.e. (n=639): agreement 69.8%
   the 119 pairs actually exposed in DPO: preference-chosen has higher TR score in 70/119 (58.8%)
   |TR score difference| : strict pairs median 0.00085 mean 0.00438 ; tied pairs (n=2897) median 0.00043 mean 0.00246
   snapshot 0: strict pairs 242, TR agreement 76.4%
   snapshot 1: strict pairs 161, TR agreement 75.8%
   snapshot 2: strict pairs 144, TR agreement 51.4%
   snapshot 3: strict pairs 140, TR agreement 64.3%

(c) per training state with >=3 strict pairs (n=88): Spearman rho between net preference wins and TR score across the 8 actions: mean +0.246 median +0.269 ; >0 in 71.6%

(d) training snapshots (index>=1, n=96) whose post-initialization history is one repeated action: 2 ; with >=4 distinct post-init actions: 55
    share of set_x_neg in post-init training histories: mean 0.204 max 0.833  (deployment DPO: 1.000 at every decision after the first)
```

## Appendix: out/s15_dpo_drift.out

```
Prequential (first-exposure) evaluation of the DPO-stage policy on the 119 scheduled pairs:
   policy log-odds favour the preference-chosen command: 66/119 ; SFT reference: 66/119
   Delta > 0 (shift toward chosen on a not-yet-trained pair): 55/119 (46.2%), binomial p vs 0.5 = 0.463
   second half (steps 60-118): Delta>0 31/59 ; policy favours chosen 33/59 ; SFT favours chosen 31/59

Is the DPO shift a sharpening of the SFT ordering?  Delta vs SFT reference margin (chosen - rejected):
   steps   0- 29: Pearson r=+0.493 Spearman rho=+0.505 slope=+0.44 (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in 20/30
   steps  30- 59: Pearson r=+0.663 Spearman rho=+0.732 slope=+1.83 (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in 23/30
   steps  60- 89: Pearson r=+0.707 Spearman rho=+0.806 slope=+1.01 (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in 24/30
   steps  90-118: Pearson r=+0.781 Spearman rho=+0.811 slope=+1.97 (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in 25/29
   steps   0-118: Pearson r=+0.626 Spearman rho=+0.689 slope=+1.28 (Delta per nat of SFT margin) ; sign(Delta)==sign(ref margin) in 92/119
   mean |Delta| by phase: [1.21, 4.0, 2.99, 4.92]  mean |SFT ref margin|: 1.95
   mean |policy margin| = |ref margin + Delta| by phase: [2.77, 5.52, 4.75, 6.7]

   per-command DPO log-odds drift (zero-mean, nats) fitted on steps 30-118 (n=89, rank 7, R^2=0.653):
      observe_all  +6.36  set_x_neg  +2.78  set_y_neg  +0.29  set_xy  -0.03  set_x_pos  -0.48  set_y_pos  -2.29  set_x_zero  -2.66  set_y_zero  -3.97

   per-command DPO log-odds drift (zero-mean, nats) fitted on steps 60-118 (n=59, rank 7, R^2=0.748):
      observe_all  +3.99  set_x_neg  +2.87  set_xy  +1.74  set_x_pos  +0.64  set_y_neg  +0.41  set_x_zero  -1.83  set_y_pos  -2.71  set_y_zero  -5.11

   per-command DPO log-odds drift (zero-mean, nats) fitted on steps 90-118 (n=29, rank 7, R^2=0.844):
      observe_all  +4.02  set_x_neg  +3.54  set_xy  +2.91  set_y_neg  +1.53  set_x_pos  +0.96  set_y_pos  -2.73  set_x_zero  -3.54  set_y_zero  -6.70

   correlation of fitted drift (steps 60-118) with SFT mean command log-lik: r=+0.887 rho=+0.857
   correlation of fitted drift with net exposed label count (chosen - rejected): r=+0.197 rho=+0.252
   net label count in DPO steps 0-59: {'observe_all': -1, 'set_x_neg': 3, 'set_x_zero': -2, 'set_x_pos': 2, 'set_y_neg': 6, 'set_y_zero': 1, 'set_y_pos': -7, 'set_xy': -2}
   correlation of drift with net label count in steps 0-59: r=+0.268 rho=+0.252
```

## Appendix: out/s07_training_exposure.out

```
top keys: ['checkpoint_steps', 'encoding', 'exposure', 'exposure_sha256', 'feature_pins', 'feature_width', 'outer_provenance_receipt_sha256', 'parameter_count', 'raw_records_utf8', 'recipe', 'record_pins', 'record_roster', 'retry_allowed', 'scientific_provenance_qualified', 'scope', 'supervision_receipt_pins', 'version']
counts: {"dpo_action_rows": 238, "dpo_steps": 119, "records": 160, "records_without_strict_pairs": 41, "retained_pairs": 3584, "selected_pairs": 119, "sft_action_rows": 160, "sft_steps": 160, "strict_pairs": 687, "tied_pairs": 2897, "unselected_pairs": 568}
records 160 worlds 32
record keys: ['action_ids', 'dpo_exposure_count', 'envelope_sha256', 'numeric_supervised_index', 'pairs', 'record_id', 'record_sha256', 'sft_exposure_count', 'strict_pair_exposures', 'supervised_command', 'supervision_receipt_sha256', 'task_sha256', 'world_id']
pair keys: ['chosen_command', 'exposure_count', 'numeric_chosen_index', 'numeric_rejected_index', 'pair_index', 'preference_index', 'preference_sha256', 'rejected_command', 'selection_sha256', 'status', 'unordered_commands']

SFT supervised_command distribution over 160 records: {'STOP': 32, 'set_x_neg': 32, 'set_x_pos': 29, 'observe_all': 24, 'set_xy': 12, 'set_y_pos': 10, 'set_y_neg': 10, 'set_x_zero': 7, 'set_y_zero': 4}
SFT exposure counts: Counter({1: 160})
  snapshot 0 n_candidates {9: 32} labels {'set_x_neg': 16, 'set_x_zero': 3, 'set_y_neg': 3, 'set_xy': 3, 'observe_all': 3, 'set_x_pos': 2, 'set_y_zero': 2}
  snapshot 1 n_candidates {9: 32} labels {'set_x_pos': 13, 'set_x_neg': 7, 'observe_all': 5, 'set_y_pos': 3, 'set_xy': 1, 'set_x_zero': 1, 'set_y_zero': 1, 'set_y_neg': 1}
  snapshot 2 n_candidates {9: 32} labels {'observe_all': 9, 'set_x_pos': 8, 'set_x_neg': 4, 'set_y_pos': 4, 'set_y_neg': 3, 'set_x_zero': 2, 'set_xy': 2}
  snapshot 3 n_candidates {9: 32} labels {'observe_all': 7, 'set_x_pos': 6, 'set_xy': 6, 'set_x_neg': 5, 'set_y_pos': 3, 'set_y_neg': 3, 'set_x_zero': 1, 'set_y_zero': 1}
  snapshot 4 n_candidates {1: 32} labels {'STOP': 32}
action_ids example: ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero', None] | snapshot-004 example: [None] {"stop":true} pairs 0

pairs total 3584
pair example strict: {"chosen_command": "{\"action_id\":\"set_y_neg\"}", "exposure_count": 0, "numeric_chosen_index": 5, "numeric_rejected_index": 1, "pair_index": 0, "preference_index": 9, "preference_sha256": "50cfacd0fc9ac23510dc01df5afd2d1d60379693c9d00ac5506925127f3063bf", "rejected_command": "{\"action_id\":\"set_x_neg\"}", "selection_sha256": "d58367975522945c35f93840820e0cbed4230366ffa34e8cd8a4aee77e32e73e", "status": "unselected", "unordered_commands": ["{\"action_id\":\"set_x_neg\"}", "{\"action_id\":\"set_y_neg\"}"]}
strict pairs 687 exposed (exposure_count>0): 119 total exposures 119
ALL strict pairs: chosen {'set_x_pos': 123, 'set_xy': 93, 'set_y_zero': 90, 'set_x_neg': 88, 'set_y_neg': 86, 'set_x_zero': 79, 'set_y_pos': 75, 'observe_all': 53}
ALL strict pairs: rejected {'set_x_zero': 102, 'set_y_pos': 98, 'set_xy': 98, 'set_y_neg': 87, 'set_y_zero': 79, 'set_x_pos': 78, 'set_x_neg': 74, 'observe_all': 71}
EXPOSED DPO pairs (119): chosen {'set_x_pos': 23, 'set_xy': 20, 'set_y_neg': 19, 'set_y_zero': 16, 'set_x_neg': 14, 'set_x_zero': 13, 'set_y_pos': 8, 'observe_all': 6}
EXPOSED DPO pairs (119): rejected {'set_y_pos': 21, 'set_x_zero': 19, 'set_xy': 16, 'set_x_pos': 14, 'set_y_neg': 14, 'set_y_zero': 13, 'observe_all': 13, 'set_x_neg': 9}

net (chosen - rejected) among exposed DPO pairs, and among all strict pairs:
  observe_all exposed: +6 / -13 = -7    all strict: +53 / -71 = -18
  set_x_neg   exposed: +14 / -9 = +5    all strict: +88 / -74 = +14
  set_x_zero  exposed: +13 / -19 = -6    all strict: +79 / -102 = -23
  set_x_pos   exposed: +23 / -14 = +9    all strict: +123 / -78 = +45
  set_y_neg   exposed: +19 / -14 = +5    all strict: +86 / -87 = -1
  set_y_zero  exposed: +16 / -13 = +3    all strict: +90 / -79 = +11
  set_y_pos   exposed: +8 / -21 = -13    all strict: +75 / -98 = -23
  set_xy      exposed: +20 / -16 = +4    all strict: +93 / -98 = -5

per-action strict-pair win rate (wins / strict pairs involving the action), all 687 strict pairs:
  observe_all 53/124 = 0.427
  set_x_neg   88/162 = 0.543
  set_x_zero  79/181 = 0.436
  set_x_pos   123/201 = 0.612
  set_y_neg   86/173 = 0.497
  set_y_zero  90/169 = 0.533
  set_y_pos   75/173 = 0.434
  set_xy      93/191 = 0.487

world ids: [202611070000, 202611070001, 202611070002] ... 202611070031
schedule stages: Counter({'sft': 160, 'dpo': 119})
SFT schedule label distribution (160 updates): {'STOP': 32, 'set_x_neg': 32, 'set_x_pos': 29, 'observe_all': 24, 'set_xy': 12, 'set_y_pos': 10, 'set_y_neg': 10, 'set_x_zero': 7, 'set_y_zero': 4}
DPO schedule chosen (119 updates): {'set_x_pos': 23, 'set_xy': 20, 'set_y_neg': 19, 'set_y_zero': 16, 'set_x_neg': 14, 'set_x_zero': 13, 'set_y_pos': 8, 'observe_all': 6}
DPO schedule rejected (119 updates): {'set_y_pos': 21, 'set_x_zero': 19, 'set_xy': 16, 'set_x_pos': 14, 'set_y_neg': 14, 'set_y_zero': 13, 'observe_all': 13, 'set_x_neg': 9}
DPO (chosen > rejected) pair types: [(('set_x_pos', 'set_y_pos'), 7), (('set_xy', 'set_y_pos'), 6), (('set_y_neg', 'set_x_zero'), 5), (('set_xy', 'set_x_zero'), 5), (('set_x_pos', 'observe_all'), 5), (('set_y_neg', 'set_y_zero'), 5), (('set_x_neg', 'set_y_neg'), 5), (('set_xy', 'set_y_neg'), 4), (('set_x_pos', 'set_xy'), 4), (('set_x_neg', 'set_x_zero'), 4), (('set_y_zero', 'set_xy'), 4), (('set_y_zero', 'set_x_zero'), 3), (('observe_all', 'set_x_pos'), 3), (('set_xy', 'set_y_zero'), 3), (('set_x_zero', 'set_y_zero'), 3), (('set_x_zero', 'set_xy'), 3), (('set_y_zero', 'set_x_pos'), 3), (('set_y_neg', 'set_xy'), 3), (('set_x_zero', 'observe_all'), 3), (('set_x_pos', 'set_x_neg'), 3), (('set_y_pos', 'set_y_neg'), 3), (('set_x_zero', 'set_x_pos'), 2), (('set_y_neg', 'set_x_pos'), 2), (('set_x_neg', 'set_x_pos'), 2), (('set_x_pos', 'set_x_zero'), 2), (('set_y_neg', 'observe_all'), 2), (('set_x_zero', 'set_y_pos'), 2), (('set_y_neg', 'set_y_pos'), 2), (('set_y_pos', 'set_x_neg'), 2), (('set_y_zero', 'set_x_neg'), 2), (('set_y_zero', 'observe_all'), 2), (('set_x_neg', 'set_y_pos'), 2), (('observe_all', 'set_y_pos'), 1), (('set_xy', 'set_x_pos'), 1), (('observe_all', 'set_xy'), 1), (('set_x_neg', 'set_xy'), 1), (('set_xy', 'set_x_neg'), 1), (('set_y_zero', 'set_y_neg'), 1), (('set_x_pos', 'set_y_neg'), 1), (('set_y_pos', 'set_y_zero'), 1)]
```

## Appendix: out/s11_training_states.out

```
record keys: ['envelope', 'preferences', 'scope', 'supervised_action_id', 'supervision_receipt_sha256', 'version', 'world_id']
worlds 32 graphs Counter({'chain': 12, 'fork': 10, 'collider': 10}) cost Counter({'uniform': 16, 'heterogeneous': 16}) noise Counter({0.25: 16, 1.0: 16})
snapshot 0: requests {8: 32} remaining {24: 32} terminal {False: 32} labels {'set_x_neg': 16, 'set_x_zero': 3, 'set_y_neg': 3, 'set_xy': 3, 'observe_all': 3, 'set_x_pos': 2, 'set_y_zero': 2} ties/pairs 654/896
snapshot 1: requests {14: 22, 13: 7, 12: 3} remaining {17: 7, 18: 25} terminal {False: 32} labels {'set_x_pos': 13, 'set_x_neg': 7, 'observe_all': 5, 'set_y_pos': 3, 'set_xy': 1, 'set_x_zero': 1, 'set_y_zero': 1, 'set_y_neg': 1} ties/pairs 735/896
snapshot 2: requests {20: 22, 16: 7, 17: 2, 19: 1} remaining {11: 3, 12: 29} terminal {False: 32} labels {'observe_all': 9, 'set_x_pos': 8, 'set_x_neg': 4, 'set_y_pos': 4, 'set_y_neg': 3, 'set_x_zero': 2, 'set_xy': 2} ties/pairs 752/896
snapshot 3: requests {26: 22, 20: 7, 22: 1, 21: 1, 24: 1} remaining {5: 2, 6: 30} terminal {False: 32} labels {'observe_all': 7, 'set_x_pos': 6, 'set_xy': 6, 'set_x_neg': 5, 'set_y_pos': 3, 'set_y_neg': 3, 'set_x_zero': 1, 'set_y_zero': 1} ties/pairs 756/896
snapshot 4: requests {32: 22, 24: 6, 27: 1, 23: 1, 30: 1, 25: 1} remaining {0: 32} terminal {True: 32} labels {'STOP': 32} ties/pairs 0/0

history action composition of training snapshots (who generated the behaviour before the snapshot?):
   policy_train/202611070000/snapshot-000 chain uniform req 8 rem 24 label set_x_zero hist OOOOOOOO
   policy_train/202611070000/snapshot-001 chain uniform req 14 rem 18 label set_x_pos hist OOOOOOOObaNbJO
   policy_train/202611070000/snapshot-002 chain uniform req 20 rem 12 label set_x_pos hist OOOOOOOObaNbJObZbaJb
   policy_train/202611070000/snapshot-003 chain uniform req 26 rem 6 label set_x_pos hist OOOOOOOObaNbJObZbaJbbccbZN
   policy_train/202611070000/snapshot-004 chain uniform req 32 rem 0 label STOP hist OOOOOOOObaNbJObZbaJbbccbZNcJaJcN
   policy_train/202611070001/snapshot-000 chain heterogeneous req 8 rem 24 label set_x_neg hist OOOOOOOO
   policy_train/202611070001/snapshot-001 chain heterogeneous req 13 rem 17 label set_x_neg hist OOOOOOOONPZJa
   policy_train/202611070001/snapshot-002 chain heterogeneous req 16 rem 12 label set_x_zero hist OOOOOOOONPZJacbN
   policy_train/202611070001/snapshot-003 chain heterogeneous req 20 rem 6 label set_y_pos hist OOOOOOOONPZJacbNPZJa
   policy_train/202611070001/snapshot-004 chain heterogeneous req 24 rem 0 label STOP hist OOOOOOOONPZJacbNPZJacbNP
   policy_train/202611070002/snapshot-000 fork uniform req 8 rem 24 label set_x_neg hist OOOOOOOO
   policy_train/202611070002/snapshot-001 fork uniform req 14 rem 18 label set_x_neg hist OOOOOOOONPZJac
   policy_train/202611070002/snapshot-002 fork uniform req 20 rem 12 label set_xy hist OOOOOOOONPZJacbNPZJa
   policy_train/202611070002/snapshot-003 fork uniform req 26 rem 6 label set_y_pos hist OOOOOOOONPZJacbNPZJacbNPZJ
   policy_train/202611070002/snapshot-004 fork uniform req 32 rem 0 label STOP hist OOOOOOOONPZJacbNPZJacbNPZJacbNPZ
post-initialization history actions across all 160 training snapshots: {'observe_all': 447, 'set_x_neg': 320, 'set_x_pos': 297, 'set_y_pos': 162, 'set_y_neg': 161, 'set_xy': 144, 'set_y_zero': 118, 'set_x_zero': 99}

label by graph (non-stop):
   chain {'set_x_pos': 14, 'set_x_neg': 11, 'observe_all': 6, 'set_y_neg': 5, 'set_y_pos': 4, 'set_xy': 4, 'set_x_zero': 3, 'set_y_zero': 1}
   fork {'set_x_neg': 17, 'observe_all': 9, 'set_x_pos': 7, 'set_xy': 3, 'set_x_zero': 2, 'set_y_pos': 1, 'set_y_zero': 1}
   collider {'observe_all': 9, 'set_x_pos': 8, 'set_y_neg': 5, 'set_xy': 5, 'set_y_pos': 5, 'set_x_neg': 4, 'set_y_zero': 2, 'set_x_zero': 2}
label by cost menu (non-stop):
   uniform {'set_x_pos': 14, 'set_x_neg': 13, 'observe_all': 11, 'set_xy': 9, 'set_y_pos': 7, 'set_y_neg': 5, 'set_x_zero': 4, 'set_y_zero': 1}
   heterogeneous {'set_x_neg': 19, 'set_x_pos': 15, 'observe_all': 13, 'set_y_neg': 5, 'set_x_zero': 3, 'set_y_pos': 3, 'set_y_zero': 3, 'set_xy': 3}
label by requests (history length):
   requests= 8 {'set_x_neg': 16, 'set_x_zero': 3, 'set_y_neg': 3, 'set_xy': 3, 'observe_all': 3, 'set_x_pos': 2, 'set_y_zero': 2}
   requests=12 {'set_x_pos': 2, 'set_x_neg': 1}
   requests=13 {'set_x_pos': 3, 'set_x_neg': 2, 'observe_all': 1, 'set_y_pos': 1}
   requests=14 {'set_x_pos': 8, 'set_x_neg': 4, 'observe_all': 4, 'set_y_pos': 2, 'set_xy': 1, 'set_x_zero': 1, 'set_y_zero': 1, 'set_y_neg': 1}
   requests=16 {'set_x_pos': 3, 'set_x_zero': 2, 'set_x_neg': 2}
   requests=17 {'observe_all': 1, 'set_y_neg': 1}
   requests=19 {'observe_all': 1}
   requests=20 {'observe_all': 9, 'set_x_pos': 6, 'set_y_pos': 5, 'set_x_neg': 3, 'set_y_neg': 3, 'set_xy': 2, 'set_y_zero': 1}
   requests=21 {'set_x_zero': 1}
   requests=22 {'set_xy': 1}
   requests=23 {'STOP': 1}
   requests=24 {'STOP': 6, 'set_x_neg': 1}
   requests=25 {'STOP': 1}
   requests=26 {'set_x_pos': 5, 'observe_all': 5, 'set_xy': 5, 'set_x_neg': 3, 'set_y_pos': 2, 'set_y_neg': 2}
   requests=27 {'STOP': 1}
   requests=30 {'STOP': 1}
   requests=32 {'STOP': 22}
```

## Appendix: out/s10_tokens_latency.out

```
Command tokenizations (pinned tokenizer, add_special_tokens=False); EOS 151645 appended at deployment:
  {"action_id":"observe_all"}     7 tokens  [4913, 1311, 842, 3252, 48033, 5705, 9207]  ['{"', 'action', '_id', '":"', 'observe', '_all', '"}']
  {"action_id":"set_x_neg"}       8 tokens  [4913, 1311, 842, 3252, 746, 3212, 28209, 9207]  ['{"', 'action', '_id', '":"', 'set', '_x', '_neg', '"}']
  {"action_id":"set_x_pos"}       8 tokens  [4913, 1311, 842, 3252, 746, 3212, 6479, 9207]  ['{"', 'action', '_id', '":"', 'set', '_x', '_pos', '"}']
  {"action_id":"set_x_zero"}      8 tokens  [4913, 1311, 842, 3252, 746, 3212, 19359, 9207]  ['{"', 'action', '_id', '":"', 'set', '_x', '_zero', '"}']
  {"action_id":"set_xy"}          7 tokens  [4913, 1311, 842, 3252, 746, 40920, 9207]  ['{"', 'action', '_id', '":"', 'set', '_xy', '"}']
  {"action_id":"set_y_neg"}       8 tokens  [4913, 1311, 842, 3252, 746, 4178, 28209, 9207]  ['{"', 'action', '_id', '":"', 'set', '_y', '_neg', '"}']
  {"action_id":"set_y_pos"}       8 tokens  [4913, 1311, 842, 3252, 746, 4178, 6479, 9207]  ['{"', 'action', '_id', '":"', 'set', '_y', '_pos', '"}']
  {"action_id":"set_y_zero"}      8 tokens  [4913, 1311, 842, 3252, 746, 4178, 19359, 9207]  ['{"', 'action', '_id', '":"', 'set', '_y', '_zero', '"}']
  {"stop":true}                   5 tokens  [4913, 9495, 788, 1866, 92]  ['{"', 'stop', '":', 'true', '}']

Branch points of the constrained greedy decode (prefix -> allowed next tokens):
  after '{"'                                     -> ['action', 'stop']
  after '{"action_id":"'                         -> ['set', 'observe']
  after '{"action_id":"set'                      -> ['_x', '_y', '_xy']
  after '{"action_id":"set_x'                    -> ['_pos', '_zero', '_neg']
  after '{"action_id":"set_y'                    -> ['_pos', '_zero', '_neg']

===== frozen_llm n 576
  prompt_tokens: min 1990 p25 2526 median 3066 mean 3069.7 p75 3609 max 4160
  prompt_chars : min 5151 median 7146 mean 7157.1 max 9185 ; chars/token 2.332
  elapsed_seconds: min 0.348 median 0.426 mean 0.582 p95 1.464 max 4.881 ; total 335.2 s
  cuda_interval_ms: min 346.8 median 424.6 mean 580.5 p95 1462.9 max 4878.3 ; total 334342.1 ms
  new tokens generated (incl EOS): {8: 574, 9: 2}  legal_commands: {9: 564, 5: 12}  longest_command_tokens: {9: 576}
  peak_gpu_allocated MiB: min 2032 max 2327
  mean prompt tokens by decision index 0,5,10,15,20,23: [2024.0, 2483.0, 2934.0, 3388.0, 3841.0, 4114.0]
  mean cuda ms by decision index 0,1,5,10,15,20,23: [559.0, 444.0, 522.0, 609.0, 593.0, 615.0, 651.0]
  latency ~ tokens linear fit: 0.0625 ms/token + 388.6 ms ; corr 0.083
  first request of session (ordinal 0) cuda ms: 735.8
  distinct prompts (by prompt-id hash): 576 of 576 ; distinct envelopes: 576

===== sft_llm n 546
  prompt_tokens: min 1990 p25 2467 median 2933 mean 2929.8 p75 3402 max 3954
  prompt_chars : min 5151 median 6997 mean 6988.1 max 8970 ; chars/token 2.385
  elapsed_seconds: min 0.410 median 0.538 mean 0.596 p95 0.930 max 2.549 ; total 325.5 s
  cuda_interval_ms: min 407.9 median 536.0 mean 594.3 p95 927.7 max 2546.4 ; total 324477.0 ms
  new tokens generated (incl EOS): {9: 515, 6: 23, 8: 8}  legal_commands: {9: 542, 5: 4}  longest_command_tokens: {9: 546}
  peak_gpu_allocated MiB: min 2041 max 2307
  mean prompt tokens by decision index 0,5,10,15,20,23: [2024.0, 2443.0, 2858.0, 3271.0, 3686.0, 3938.0]
  mean cuda ms by decision index 0,1,5,10,15,20,23: [632.0, 577.0, 570.0, 528.0, 683.0, 669.0, 565.0]
  latency ~ tokens linear fit: 0.0254 ms/token + 520.0 ms ; corr 0.069
  first request of session (ordinal 0) cuda ms: 786.5
  distinct prompts (by prompt-id hash): 546 of 546 ; distinct envelopes: 546

===== dpo_llm n 576
  prompt_tokens: min 1990 p25 2493 median 2984 mean 2981.0 p75 3481 max 3985
  prompt_chars : min 5151 median 7096 mean 7090.2 max 9039 ; chars/token 2.378
  elapsed_seconds: min 0.476 median 0.712 mean 0.851 p95 1.648 max 4.049 ; total 490.4 s
  cuda_interval_ms: min 474.7 median 709.6 mean 849.2 p95 1645.8 max 4046.2 ; total 489129.4 ms
  new tokens generated (incl EOS): {9: 576}  legal_commands: {9: 564, 5: 12}  longest_command_tokens: {9: 576}
  peak_gpu_allocated MiB: min 2041 max 2311
  mean prompt tokens by decision index 0,5,10,15,20,23: [2024.0, 2443.0, 2858.0, 3271.0, 3685.0, 3932.0]
  mean cuda ms by decision index 0,1,5,10,15,20,23: [811.0, 758.0, 828.0, 803.0, 708.0, 951.0, 859.0]
  latency ~ tokens linear fit: 0.0674 ms/token + 648.2 ms ; corr 0.089
  first request of session (ordinal 0) cuda ms: 817.6
  distinct prompts (by prompt-id hash): 576 of 576 ; distinct envelopes: 576

prompt token growth per additional accepted response (consecutive decisions in the same world):
  frozen_llm  mean 90.9 sd 7.7 min 53 max 132
  sft_llm     mean 83.1 sd 4.7 min 64 max 106
  dpo_llm     mean 83.0 sd 4.5 min 64 max 103

  token budget by section, DPO world 0 decision 0 (total prompt tokens 2012):
     history                  ~  718 tokens (35.7%)
     scm                      ~  377 tokens (18.7%)
     menu                     ~  287 tokens (14.3%)
     graph                    ~   71 tokens (3.5%)
     task                     ~  321 tokens (16.0%)
     domain_notes+objective   ~   60 tokens (3.0%)
     counters                 ~   25 tokens (1.2%)

  token budget by section, DPO world 0 decision 23 (total prompt tokens 3927):
     history                  ~ 2622 tokens (66.8%)
     scm                      ~  386 tokens (9.8%)
     menu                     ~  287 tokens (7.3%)
     graph                    ~   71 tokens (1.8%)
     task                     ~  321 tokens (8.2%)
     domain_notes+objective   ~   60 tokens (1.5%)
     counters                 ~   27 tokens (0.7%)

example scm JSON as the model sees it: [{"eligible_rows":8,"node":"X","noise_variance":0.25,"parameter_covariance":[[0.030303030303030304]],"parameter_mean":[1.0415656718463846],"terms":["intercept"]},{"eligible_rows":8,"node":"Y","noise_variance":0.25,"parameter_covariance":[[0.2653213106466827,-0.2256394260066532],[-0.2256394260066532,0.21663485280449185]],"parameter_mean":[-0.5478667338366421,0.42589362595851954],"terms":["intercept","X"]},{"eligible_rows":8,"node":"Z","noise_variance":0.25,"parameter_covariance":[[0.031112650920001025,0.007764614191435232],[0.007764614191435232,0.07446603048155506]],"parameter_mean":[-1.609175737627502,-1.220565774823312],"terms":["intercept","Y"]}]

example remaining/requests JSON fragment: {"remaining_units":24,"requests":8,"returned_responses":8}
one 2x2 covariance matrix = 78 tokens: ['[[', '0', '.', '2', '6', '5', '3', '2', '1', '3', '1', '0', '6', '4', '6', '6', '8', '2', '7', ',-', '0', '.', '2', '2', '5', '6', '3', '9', '4', '2', '6', '0', '0', '6', '6', '5', '3', '2', '],[-', '0', '.', '2', '2', '5', '6', '3', '9', '4', '2', '6', '0', '0', '6', '6', '5', '3', '2', ',', '0', '.']
```

Further verbatim outputs not reproduced here: out/s01_llm_keys.out, out/s04_action_dist.out, out/s08_ranker.out, out/s09_sft_frozen.out, out/s14_llm_training_scores.out.
