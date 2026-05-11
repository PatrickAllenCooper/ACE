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

**If results confirm the hypotheses**, the abstract and Section 5.2
prose can be tightened from "DPO's marginal contribution at this scale
is within seed-variance bounds" to "DPO's contribution emerges
when the LM prior is mismatched, validated by Anonymised-30 (gap
widens) and 50-node (gap widens) follow-up experiments".

**If results don't confirm**, the LM-prior thesis is even stronger
than the paper currently claims (Section 3.3 needs a small expansion);
DPO's contribution becomes harder to defend at this scale and the
paper's contribution emphasis should shift to the LM-prior + lookahead
combination, with DPO framed as a refinement step.

Output structure (after pulling locally):
```
results/curc_30node_followup/
  anon30/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
  nodes50/{ace,zero_shot_lm}/seed_{seed}/job_{jobid}/
```

Plot/aggregate: extend `scripts/analysis/plot_30node_results.py` with
two more conditions (or build a separate plot for the 4-cell
ACE-vs-ZSL grid).
