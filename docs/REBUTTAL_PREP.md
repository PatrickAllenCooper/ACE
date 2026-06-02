# Rebuttal Preparation Notes

Tracker for reviewer-anticipated experiments to run **after submission** so we
have answers ready by the rebuttal window. Numbered by reviewer concern.

## Concern 1: Budget fairness (already addressed in submission)

Status: **CLOSED**. The Method's "Lookahead evaluation" paragraph plus the
Limitations' "Budget accounting" clause both clarify that:
- "Intervention budget" counts only executed interventions.
- Lookahead is a training-time DPO reward signal, not used at deployment.
- Bayesian OED has the same many-candidate-per-step structure (M=10 posterior
  draws per candidate).
- 30-node baselines plateau at 5.86 with seed-std < 0.08 regardless of how
  many additional environment queries they receive.

If pressed harder during rebuttal, optional experiment:
- **Replace lookahead with student-SCM-only predictions.** Run ACE with the
  reward computed against the student's predicted outcome rather than a fresh
  environment query. Hypothesis: ACE still wins (because the LM's prior is
  doing the heavy lifting), with some degradation in DPO signal quality.

## Concern 2: Stronger 30-node baselines (RUN BEFORE REBUTTAL)

Reviewer asked for: Bayesian OED, PPO, CEM/BO-style acquisition, LM-only/no-DPO
controls on the 30-node system.

**Priority order for rebuttal experiments:**

1. **Bayesian OED on 30-node** (~12-24 hours per seed; needs adapted EIG over the
   30-node mechanism class). Highest value; directly addresses the "ACE only
   beats principled baselines on 5-node" point.
2. **PPO on 30-node** (~6-8 hours per seed; needs PPO state encoding adapted
   from 5-node hardcoded shape to variable shape). Same reward signal as ACE,
   so this isolates DPO's contribution at scale.
3. **LM-only / no-DPO on 30-node** (~6 hours per seed). Pretrained Qwen with
   lookahead but no DPO update. Hypothesis: matches passive baselines.
4. **CEM-style acquisition on 30-node** (~4 hours per seed). Cross-entropy
   method over intervention proposals, refining the proposal distribution from
   reward feedback. Useful adversarial baseline for the LM's learned strategy.

Job script template: copy `jobs/curc_large_scale_seed.sh` and modify the python
invocation per method.

## Concern 3: Final-vs-final and best-vs-best (addressed in submission)

Status: **CLOSED**. Table 2 now reports both Best MSE and Final MSE columns
for ACE and all baselines, with caption explaining why best is the natural
summary for non-monotone training. ACE wins on both metrics:
- Best-vs-best: 1.95 vs 5.80 (3.0×)
- Final-vs-final: 4.13 vs 5.86 (1.4×; ACE final is dominated by seed 42's
  late-training instability)

If reviewer presses further: add a stable-seed-only column (excluding seed 42)
giving ACE final 2.08 ± 0.87 vs baseline 5.86, a 2.8× improvement.

## Concern 4: Known-graph assumption (DEFENDED in submission, not a flaw)

Reviewer treated this as a "sharp limitation". Our position: **the known-graph
case is the central problem in mechanism estimation**, not a failing.

Status: **DEFENDED**. The reframed Limitations paragraph now explicitly states:
"ACE addresses mechanism estimation given known causal structure, the central
task whenever a graph is available from domain expertise, prior literature, or
an upstream structure-learning algorithm. Joint structure-and-mechanism methods
target a different problem... ACE complements rather than competes with them."

For rebuttal, additional support points:
- Many real experimental settings DO have known structure (gene knockout
  studies often start from KEGG pathways; physics experiments have known
  mechanism diagrams; economics has well-established theoretical models).
- The 30-node regime is provably out of reach for posterior-maintaining
  Bayesian methods at comparable compute (NP-hard, super-exponential DAG
  space; CBED's largest published evaluation is 20 nodes).
- A structure-learning frontend can sit upstream of ACE; this is the natural
  pipeline. Composing them is a follow-up experiment, not a flaw of ACE.

## Concern 5: Additional ablations (RUN BEFORE REBUTTAL)

Reviewer asked for: frozen LM, random-init LM/MLP policy, DPO without
pretrained prior, zero-shot LM.

**Status of each:**

- **No DPO ablation** is in Table 3 already (random proposals + lookahead;
  result 2.10±0.11 ≈ Round-Robin baseline). This is the same as
  "zero-shot-policy + lookahead-select" but with random rather than LM
  proposals. Reviewer wants the LM-zero-shot version specifically.

**Priority order for rebuttal experiments:**

1. **Zero-shot LM + lookahead** (~4 hours). Same as "No DPO" but use the
   pretrained Qwen2.5-1.5B for proposals (no DPO updates). Tests whether the
   LM prior alone is enough.
2. **Frozen LM + DPO on adapter only** (~6 hours). Adds a small LoRA adapter
   on top of frozen Qwen, trains only the adapter with DPO. Tests how much of
   ACE's improvement requires full fine-tuning vs. cheap adaptation.
3. **Random-init LM + DPO** (~12 hours; long because no useful prior). Train a
   from-scratch transformer with the same architecture as Qwen2.5-1.5B. Tests
   whether ACE's wins come from the LM's pretrained world-model prior versus
   just from DPO.

If only ONE rebuttal ablation is feasible: run #1 (zero-shot LM + lookahead).
This is the cleanest control for "is the LM adding anything beyond the
lookahead's evaluate-then-select mechanism?"

## Logistics

- All rebuttal experiments require GPU access. Submit to CURC `aa100`
  partition with the existing `curc_large_scale_seed.sh` worker, modified per
  experiment.
- Document new results in a separate "Rebuttal Addendum" PDF if the page
  budget for the rebuttal text is tight.
- Expected aggregate cost: ~80-120 GPU-hours across all rebuttal experiments.

## Concern 6: LM-prior-vs-DPO contribution (RAISED BY zero-shot-LM RESULT)

Status: **RUN AS FOLLOW-UP**. The zero-shot-LM N=5 result on canonical
30-node came back at best=1.73 ± 0.22, statistically tied with ACE
(1.95 ± 0.77). This validates Section 3.3's "LM as forward-model prior"
thesis but raises a follow-up: how much does DPO contribute when the LM
prior is *mismatched*?

**Two follow-up experiments**, both implemented and ready to submit
via `bash jobs/curc_submit_30node_followup.sh`:

1. **anon30** -- 30-node SCM, anonymised node names (`n_xxxx` instead of
   `X1..X30`). Removes the LM's semantic prior; the LM has only graph
   structure and per-node losses to reason from. *Hypothesis*: ACE >
   Zero-shot LM by a wider margin than canonical 30-node, because DPO
   must compensate for the missing semantic prior.
2. **nodes50** -- 50-node hierarchical SCM (canonical names). Stresses
   the prompt context length; LM prior may dilute. *Hypothesis*: both
   methods degrade, but ACE less so. (8 roots, 8 layer-1, 18 layer-2
   with colliders, 8 layer-3 colliders, 8 leaves.)

**Compute budget:** 5 ACE x 8h GPU + 5 ZSL x 8h GPU per condition x
2 conditions = ~160 GPU-hours total. Submitted as 20 jobs on aa100;
ACE jobs cap at 120 episodes, ZSL jobs at 40 (fixed-policy asymptote).

### RESOLVED (June 2026): hypotheses NOT confirmed -> LM-prior thesis is stronger

After two contaminated batches (see "Lessons" below) the clean N=5 anon30
numbers are:

| cell                | best MSE (N=5) | n_ep (mean) |
|---------------------|----------------|-------------|
| anon30 ACE          | 1.49 +/- 0.16  | ~19 (timeout, converged early) |
| anon30 zero-shot LM | 1.49 +/- 0.18  | ~17 |
| nodes50 zero-shot LM| 4.55 +/- 1.10  | ~38 (clean) |
| nodes50 ACE         | (contaminated) | dropped |

**anon30 ACE and anon30 zero-shot LM are statistically identical (1.49 vs
1.49).** Anonymising node names did NOT open the gap we hypothesised; the
LM's structural+loss reasoning plus lookahead drives performance even with
no semantic node names. This echoes the canonical-30 tie (ACE 1.95 vs ZSL
1.73). DPO loss confirmed to move off 0.693 (variable 0.59-2.56), so the
runs genuinely learned -- the result is real, not a no-DPO artifact.

Decision (user, June 2 2026): **reframe around the LM-prior thesis**
(Section 3.3) -- report anon30 as evidence the pretrained LM prior, not
DPO, drives 30-node performance; DPO is a refinement that matters at
smaller scale / harder horizons. nodes50 ACE decision **deferred** until
the followup figure is reviewed.

Caveat for honest reporting: anon30 ACE timed out at ~19 episodes (vs the
120 the canonical-30 ACE used). Best-MSE had converged by ~ep 20 for both
methods, so the best-MSE comparison is fair, but this must be stated.

### Lessons (two wasted batches)

1. `int(node[1:])` in BOTH LargeScaleSCM.generate AND the GroundTruthSCM
   mechanism crashed on anonymised `n_xxxx` names. Fix: node_idx lookup.
2. 50-node DPO OOMs a 40 GB A100 (4 forward passes, long prompt). Fix:
   bf16 + gradient checkpointing.
3. Anonymised 30-node ALSO OOMs (n_xxxx is 3x longer than X1..X30), so
   checkpointing is needed at 30 nodes too when anonymised.
4. The "None of the inputs have requires_grad=True" + "DPO loss 0.693"
   warnings are BENIGN -- they fire from the reference-model forward under
   torch.no_grad(). Proven by scripts/analysis/test_grad_checkpoint.py.
   Do not panic-cancel on these warnings; verify learning via the DPO-loss
   and mechanism-loss trajectory instead.
5. Working-DPO 50-node ACE runs ~57 min/episode; 120 episodes is infeasible
   inside CURC's 24h wall-time cap. Reduce episode budget or drop the cell.

Output structure (after pulling locally):
```
results/curc_30node_followup/
  anon30/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
  nodes50/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
  aggregate.csv          <- scripts/analysis/aggregate_followup_results.py
  fig_followup.png/pdf   <- scripts/analysis/plot_followup_results.py
```

### Follow-ups applied (June 2 2026)

- **Relabel** "zero-shot LM" -> "ACE w/o DPO (LM + lookahead)" everywhere
  (figure + paper), since it is an ablation of ACE (LM proposer + lookahead,
  DPO disabled), NOT naive zero-shot prompting. This removes the misread that
  "no training is naturally better".
- **Per-node figure**: `plot_followup_results.py` Panel (b) now plots per-node
  best MSE (total/N) with a 30-vs-50 separator, so the 50-node cell reads as a
  larger/harder graph rather than a worse method. The convergence-plateau
  episode is annotated in Panel (a).
- **Convergence caveat baked into prose**: appendix `app:scaling` + the
  "How much is the LM prior versus DPO?" paragraph in Section
  `sec:results-scale` state plainly that anon30 ACE truncated at ~ep 19 but
  best-MSE plateaued by ~ep 20 for both methods.
- **Convergence rerun** available: `jobs/curc_resubmit_anon30_ace_converge.sh`
  resumes anon30 ACE (STABLE_DIR) to ~40 episodes to show the plateau outright.
  `aggregate_followup_results.py` now records `best_episode` for this.

## Concern 7: Scaling principles + pipeline (30 -> 50 -> 100+)

Status: **PIPELINE BUILT; SWEEP READY TO RUN**. Frames the headline message
"ACE scales to larger N without architectural change". Full principles and the
100+ design spec live in `docs/development/guidance/guidance_doc.txt` (Scaling
section). Concrete artifacts:

- **Binding-cost fix (compact prompt)**: `ace_experiments.py --prompt_strategy
  compact --prompt_top_m M` surfaces only the top-M highest-loss nodes + parents,
  holding prompt length ~O(M) instead of O(N+E+H). Unit-tested in
  `scripts/analysis/test_compact_prompt.py` (50-node prompt 41% shorter).
- **Instrumentation**: prompt-token mean/max + peak VRAM + n_nodes now in
  `metrics.csv`; per-node MSE + `best_episode` in the followup aggregate.
- **Sweep**: `jobs/curc_submit_scaling.sh` runs N in {15,30,50} x
  {ace, zero_shot_lm, random}; `jobs/curc_submit_k_ablation.sh` sweeps K at 50.
  Worker `jobs/curc_scaling_seed.sh` uses STABLE dirs for checkpoint-resume.
- **Main-text figure**: `scripts/analysis/plot_scaling.py` ->
  `figs/fig_scaling.pdf`, per-node best MSE vs N (ACE, ACE-w/o-DPO, Random);
  N=5 diagnostic shown as a separate-family anchor with `--show-n5`.
- **Scope**: run N in {15,30,50} now; N=100+ specified as future work (compact
  prompt mandatory, salience-targeted lookahead, plateau-budgeted episodes).
