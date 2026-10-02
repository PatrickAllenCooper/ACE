# Technical path: intervention-conditioned prediction with conservative adaptation

## Decision

Focus the learned architecture on predicting the consequences of interventions from a history of experiments. Earn an acquisition claim only after establishing a predictive advantage on identical histories. The present evidence supports studying joint-action coverage and conditional mechanism reuse; it does not establish that language priors, DPO, or residual repair improve causal learning.

The central hypothesis is that a reusable model can infer local mechanisms from experimental context, compose those mechanisms for a previously unobserved joint intervention, and recognize when its context is insufficient. The foundation-model connection is amortized learning across systems and experimental histories. A small prototype tests this hypothesis without pretending to be a foundation model already.

## Resolved engineering uncertainty

The pinned, unchanged CausalMan lower-level sampler successfully executed nine actions on one fixed path: observation, four single interventions, and four paired interventions. Each returned 16 rows, for 144 generated rows. All 53 metadata-declared public columns remained present and finite; all requested target values were clamped correctly. Canonical graph edges, equations, distributions, and non-object metadata remained unchanged. Sampling took about 0.97 seconds on this Mac, with peak process RSS about 221 MB. No model calls or GPU allocation occurred.

These are engineering observations, not prediction or policy results. The graph contains 157 nodes and 170 edges. The two probe targets are PF_M1_T1_sgrad and PF_M1_T2_sgrad. Probe levels come from their monitoring thresholds; thresholds are not certified actuator ranges. This establishes a mathematical intervention benchmark only. The original pickle-byte equality check failed after sampling; canonical mechanism checks passed. Object serialization is therefore not used as a mechanism-identity test.

The complete production mixture remains outside scope. Different noise seeds are repeated samples of one system. CausalMan must never supply a fictitious population confidence interval through seed counting. No confirmation outcomes were used to select a learner.

## Architecture to implement

1. Represent every observed experimental row with variable identity, value, observation mask, intervention mask, intervention value, and experiment membership. Use normalization fitted on acquisition data only. Keep hidden graphs and private evaluation responses outside the learner process.
2. Encode unordered rows within each experiment and exchange information across variable tokens. Compare a small permutation-equivariant set encoder with a parameter-matched flat predictor. A graph version must use inferred structure; an oracle graph is a separately labeled upper bound.
3. Decode a predictive distribution for each requested outcome conditional on a proposed action. Train on held-out intervention blocks rather than random rows. Joint-action composition is the primary target, with observational and single-action predictions as controls.
4. Use shared mechanism representations plus a small context-conditioned residual. Include a scratch-trained version, a frozen pretrained version, and a fine-tuned version. Transfer is established only by improvement over the same architecture without pretraining under equal target data and compute accounting.
5. Make uncertainty operational: a conservative fallback to a nontransfer predictor when calibration/support checks fail. Do not attach an unvalidated confidence score and call it safe. Calibrate switching on disjoint development systems; report selective risk and fallback frequency.

This architecture is a hypothesis, not a novelty claim. Its testable contribution would be reliable compositional transfer under restricted intervention histories, with explicit negative-transfer control.

## Ordered execution and stopping rules

### 1. CPU external headroom gate

Freeze a finite mathematical action menu and outcome set before learner comparisons. First inspect which public outcomes are descendants of the probe targets privately, solely to establish that the task has a causal response. Record that task construction used evaluator information; do not expose edges to policies. If the two chosen targets have no joint predictive task beyond disconnected responses, select a documented task once and freeze it, or stop this path. Do not search repeatedly for a favorable policy result.

Build an adapter returning only the frozen public columns and masks. Validate rejected actions, deterministic replay, fresh-noise calls, exact row accounting, and separation of acquisition/evaluation seeds. Compare fixed factorial coverage, uniform legal actions, and the existing numerical policy with the same learner, action menu, and query budget. Evaluate joint-action held-out prediction, calibration, and worst-action error. Fit preprocessing on acquired rows. Report elapsed CPU cost as well as query cost.

Proceed to an acquisition claim only if there is nontrivial headroom over fixed coverage and the numerical policy improves under the frozen comparison. If fixed coverage is already sufficient, retain CausalMan as a transfer/prediction test; do not manufacture an acquisition problem.

### 2. Matched-history learned prototype

Prepare training histories on CPU from explicitly separate mechanism families. Split entire parameterized systems and mechanism families before training. Existing exposed synthetic systems are development data. Keep the external CausalMan path out of training and hyperparameter selection. One external path gives a case study, not broad external validity.

Begin with a small set model and a flat model on exactly the same histories. Controls: simple regression, flexible nontransfer regression, scratch set model, pretrained set model, shuffled intervention labels, and separately labeled oracle structure. Primary endpoint: normalized error on held-out joint-action blocks at matched target sample counts. Secondary: calibration, worst-action loss, compute, and degradation relative to scratch. Freeze a practically meaningful margin before opening held-out systems, using development variance for sample-size planning.

GPU submission requires measured CPU preparation, parameter/memory estimates and a bounded single-device utilization smoke. The previous portfolio ceilings remain ceilings, not allocations. Do not request GPU for the external adapter or data preparation.

### 3. Conservative repair branch

Use the same predictor to distinguish parent-distribution shift from conditional mechanism change. Compare unchanged mechanisms under shifted inputs, genuinely changed mechanisms, and mixed shifts. Select diagnostic interventions on parents, then test conditional residual changes on disjoint observations. Freeze repair decisions before scoring.

Require unchanged-mechanism harm to stay within the existing 5% margin. Include do-nothing, full refit, previous residual repair, and oracle-change-location controls. Report false repair rate and changed-mechanism benefit separately. A failed safety gate stops repair scaling even if average loss improves. The twelve exposed worlds can diagnose implementation, not confirm recovery.

### 4. Language branch remains separable

Continue gathering independently authored action constraints and adjudicated gold. Use language only to compile a typed action contract, with abstention for ambiguity and a deterministic full-schema baseline. Keep temporal/spatial constraints explicit. No language model trial is justified by the current zero-adjudicated-task intake. This branch cannot rescue a failed numerical predictor through privileged schema information.

## What would make a coherent paper

A defensible positive paper needs: (a) a matched-history transfer advantage on unseen systems, (b) compositional joint-action evaluation, (c) bounded negative transfer, and (d) at least one external mechanism test with carefully limited claims. An acquisition section is optional and requires its own same-learner result. A negative result remains useful if it isolates whether failures come from support coverage, mechanism mismatch, uncertainty, or acquisition rather than blending them.

Immediate implementation priority is the CPU adapter/headroom gate, followed by the matched-history predictor. The language and repair branches have bounded independent deliverables. No additional favorable synthetic seed sweep is needed to begin this path.

## Subsequent task gate: probe menu is insufficient (02 October 01:29 UTC heartbeat)

The static private-evaluator audit in `results/local_causalman_joint_task_audit_20261001` found just one shared public descendant of the two probe targets: `Sec_C2_Machine1_ProcessResult`, a product of local quality flags. Both tested threshold endpoints satisfy both inclusive quality checks for each target. Thus all four paired probe settings fix the target quality factors to one; they do not vary this joint quality mechanism. The continuous `smax` descendants respond separately as `delta_smax + sgrad`. Observational versus interventional effects can remain, and this audit does not prove that every task on this simulator is trivial.

**Do not launch the proposed headroom matrix on this menu.** Engineering feasibility passed; scientific task suitability did not. Preserve this finding and avoid searching the graph for a favorable policy score. The next bounded source-level decision is whether a documented actuator pair has meaningful interacting continuous outcomes under a justified action domain. If no such independently motivated pair is documented, use the already declared Causal Chambers fallback for external task audit. Prediction architecture development can proceed with system/family splits, but external validation remains unresolved. No new samples or GPU jobs were needed for this decision.
