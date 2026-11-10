# ACE intervention policies on 24 synthetic systems

November 10, 2026. Development comparison, with all eight policies retained.

**This training recipe did not improve intervention efficiency.** Direct SCM target-risk planning had the lowest mean capped cost to target, 11.625 units. The preference-trained LLM required 15.167 units, compared with 14.250 for uniform random and 13.625 for the frozen LLM. These values include the eight-unit initialization and the common cap for nonattainment.

All 192 runs completed with evaluable reports. There are no failed or unresolved comparison cells. All 24 systems and all eight policies remain in the analysis. The operational pipeline and CURC backup are independently accepted. The separate numerical review is in progress.

## Results across all 24 systems

The table uses equal weight for each system. Lower cost and lower error are better. “Ever” means the run crossed the 0.04 error target at least once. “Final” means it met that target after its last accepted response.

| Intervention policy | Mean capped total cost to target | Ever / 24 | Final / 24 | Mean final NMSE | Mean actual paid cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform random | 14.250 | 21 | 20 | 0.04113 | 32.000 |
| Menu coverage | 12.458 | 22 | 21 | 0.01974 | 32.000 |
| Local information | 11.958 | 23 | 18 | 0.01800 | 32.000 |
| Target risk | 11.625 | 23 | 21 | 0.01722 | 32.000 |
| Numerical ranker | 13.583 | 21 | 20 | 0.03572 | 32.000 |
| Frozen LLM | 13.625 | 22 | 20 | 0.02008 | 32.000 |
| Supervised LLM | 14.625 | 21 | 19 | 0.02708 | 30.042 |
| DPO LLM | 15.167 | 19 | 18 | 0.03642 | 32.000 |

DPO reached the target at least once on 19 systems and met it at the final state on 18. Target-risk planning reached it on 23 and met it at the final state on 21. DPO's mean final error was also higher than target risk, the frozen LLM and the supervised LLM. It was lower than random's mean final error, so the comparison is not a loss on every metric.

For paired capped cost, DPO was lower than target risk on 2 systems, tied on 13 and higher on 9. Against random, DPO was lower on 3, tied on 13 and higher on 8. Against the frozen LLM, it was lower on 5, tied on 11 and higher on 8. Against the supervised LLM, it was lower on 4, tied on 13 and higher on 7. Against the numerical ranker, it was lower on 2, tied on 14 and higher on 8. All 28 unordered comparisons are retained in the linked CSV.

Ten systems already met the error target after the common initial observations. Those systems contribute an eight-unit cost-to-target endpoint to every policy. This accounts for many paired ties and limits how much the average endpoint can distinguish the intervention policies. No systems are removed because their initialization was already accurate.

Actual spending tells a different story from retrospective first crossing. Seven policies spent the full 768 units across their 24 runs. The supervised LLM spent 721 units and stopped before the full budget in 23 systems. Five of those stopped runs were above the target at their final state. Its lower spending therefore does not establish an equally accurate autonomous stopping rule. Across the comparison, the logical per-arm bill was 6,097 cost units and 5,884 accepted responses, including each arm's separately charged initialization.

## What the recorded decisions show

The preference-trained LLM chose **X = −1 on all 576 acquired actions**. The frozen LLM chose observation on 574 of 576 actions and X = 0 on the remaining two. The supervised LLM chose X = −1 on 481 of its 523 acquired actions. These counts exclude the prescribed initial observations and stop commands.

Training changed the policy's behavior, but the final preference-trained policy did not vary its acquired action across the observed states. This is a concrete limitation of this recipe. It does not identify why the training produced that behavior, and it does not establish that every LLM intervention policy would fail.

All four coefficient/noise regimes, all three graph groups and both cost groups are retained in the companion data. DPO's mean capped cost is worse than target risk in each of the four regimes. Its mean capped cost is also worse than the frozen LLM in each regime. Differences between graph groups and cost groups remain descriptive. The cost strata use different realized systems, so they are not an intervention on price alone.

## Research recommendation

Keep direct SCM target-risk planning as the reference for the next development cycle. Do not scale or promote the current DPO policy as an intervention-efficiency improvement. The immediate research issue is whether the policy uses the changing SCM state and the action menu, rather than simply emitting a frequent command.

A useful next diagnostic would compare the frozen, supervised and preference-trained command distributions on saved training examples and separately declared artificial public states. It should measure sensitivity to posterior uncertainty, target queries, action costs and remaining budget. Such diagnostics can distinguish failures of state sensitivity, supervision and optimization without selecting a new policy using these 24 development outcomes. Any new training recipe needs fresh held-out systems, a separate freeze and the same strong numerical controls. This report does not authorize or launch that additional study.

This screen supplies a known graph, linear mechanisms, known noise levels and a small fixed action menu. Its public prose is constant across regimes. It does not test whether an LLM can turn a scientist’s natural-language objective into a model, interpret instruments, or propose new intervention types. Those possible roles remain separate research questions.

The current evidence supports the SCM planning baseline. It does not yet establish an added experimental-efficiency benefit from the trained LLM component. No automatic manuscript submission or favorable replacement experiment follows this result.

## What was compared

The experiment asks whether a trained language model can choose useful interventions from a public SCM state. It compares eight policies on the same 24 systems. Every policy uses the same SCM learner, task, action menu and initial observations within a system. This isolates the intervention chooser. It does not compare SCMs with an SCM-free learner.

The systems have three variables, X, Y and Z. Their known causal graphs are a chain, X → Y → Z, a fork, X → Y and X → Z, or a collider, X → Z and Y → Z. Mechanisms are linear with independent additive Gaussian noise in each structural equation. The 24 systems cross these three graphs with two coefficient regimes, two known noise levels and two intervention-cost menus. Each combination contributes one system. Policies are paired within a system. The two cost strata use different parameter draws.

At each decision, the policy sees the public target task, causal graph, remaining budget, legal intervention menu, measured history and SCM posterior means and covariances. It does not see the true mechanism coefficients or the private evaluation errors. It returns one legal action identifier or a stop command. The executor charges the action and obtains X, Y and Z measurements. Each accepted response updates every nonintervened mechanism whose outcome and parents are available. Intervened variables can still serve as known inputs to downstream mechanisms.

The eight acquisition choices are observation, setting X to −1, 0 or 1, setting Y to −1, 0 or 1, and jointly setting X to −1 and Y to 1. All cost one unit in the uniform menu. Y and joint interventions cost two units in the heterogeneous menu. Other actions cost one. Cost units therefore need not equal the number of interventions.

The policies are:

- **Uniform random:** samples uniformly from affordable actions.
- **Menu coverage:** chooses a least-used affordable action with a fixed tie rule.
- **Local information:** estimates information gained about eligible SCM mechanisms per cost unit.
- **Target risk:** estimates the reduction in uncertainty about the target predictions per cost unit.
- **Learned numerical ranker:** learns action preferences from the corresponding structured training examples.
- **Frozen LLM:** the pretrained Qwen3-0.6B model without the new adapters.
- **Supervised LLM:** the same base with its fixed final supervised adapter.
- **Preference-trained LLM:** the same base with its fixed final DPO adapter.

The comparison with the frozen LLM tests the effect of this training recipe on the same pretrained base. Comparing the preference-trained and supervised adapters tests the additional preference stage. Comparing either trained LLM with the numerical ranker tests whether the language-model route helps beyond a smaller learned action scorer with matched example exposure. Target risk provides a direct SCM planning control. None of these contrasts alone isolates the causal contribution of pretraining, model size or architecture.

## What adapts, and when

The Qwen3-0.6B adapters were trained offline on 32 separate training systems, using 160 supervised updates followed by 119 preference updates. The supervised and preference-trained arms use the prescribed final checkpoints, without development-score checkpoint selection. The supervised stage teaches the model to output legal commands selected by preassigned numerical teachers from SCM states. Preference labels compare replicated reductions in prediction error per acquisition-cost unit. The preference stage adjusts command probabilities using these chosen and rejected intervention pairs. The numerical ranker receives matched example exposure, which does not imply equal model size or compute.

Training collection consumed 9,145 accepted responses and 10,240 response-cost units, with no unknown returns. It retained all 3,584 preference comparisons. The fixed exposure rule selected 119 strict pairs for preference updates. This one-time training bill is separate from deployment intervention cost and from GPU computation. A deployment advantage alone would not establish lower total experimental cost.

During a development episode, the policy weights stay fixed. The SCM posterior changes after each accepted response, so the next policy input changes. There is no online language-model fine-tuning in this comparison. All 24 development systems are separate from the 32 systems used to create the training examples.

## How to read the endpoints

Each policy pays eight units for the shared initial observation schedule and receives up to 24 further acquisition-cost units. Every prediction error averages eight equally weighted target queries, using the fixed public scale of 2. The private evaluator compares the SCM prediction with the true noise-disabled target mean. The error is \(\frac18\sum_q((\hat z_q-z_q^*)/2)^2\), where \(z_q^*\) is the true noise-disabled target mean. The target is normalized mean squared error at most 0.04.

The evaluator checks for crossing after the complete eight-observation initialization and after each accepted acquisition. The capped cost-to-target endpoint is the first such crossing, plus the eight initial units. Nonattaining, failed or unresolved runs receive the common total cap of 32. Failed and unresolved runs keep that cap even if an earlier crossing was recorded. This is a retrospective endpoint. The policy cannot use the private error to stop, and the endpoint is not the amount actually charged. Actual paid cost is retained separately.

A run can cross the target and later leave it. Report both ever reaching the target and meeting it at the final evaluable SCM state after accepted responses. An early stop without the target is not experimental success. Paired comparisons use all 24 systems, including cap assignments. These are descriptive development results, without new significance tests or a claim of transfer to real compressors or laboratories.

## Where the evidence lives

Large artifacts remain on CURC. The complete finalization archive is `/projects/paco0228/ACE_evidence/ace-llm-finalization-20261110-01/raw.tar`. Its readback matched the original file inventory. It contains 107,284 files with 3,308,319,856 bytes of original content, stored in a 3,390,535,680-byte archive.

The Mac retains selected evidence, all 242 report files and verification records. Report retrieval completed after independent finalizer acceptance and rechecked the full CURC archive twice. The unaltered report payloads total 4,403,819 bytes. The two finalizer evidence capsules each contain 379 selected files totaling 306,334,884 bytes, plus verification metadata. This storage design removes the former 85 GB local download requirement. It preserves failed and interrupted attempts alongside successful runs.

The scientific methods were frozen before development outcomes. The source is `a9346022f145cc29439325b6f291597b2e6ec037` and the scientific freeze is `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`. Later operational recovery changed execution and collection handling, without selecting new worlds, methods, training checkpoints or thresholds from the development results.

## Compute accounting

The saved cycle reconciliation contains 49 jobs and 100 submission journals, including unsuccessful attempts and qualification work. It records 40,265 allocated CPU seconds, or 11.1847 CPU-core-hours, and 4,458 allocated GPU-slice seconds, or 1.2383 slice-hours. GPU allocations used one 20 GB A100 MIG slice. A slice-hour is not a full A100 GPU-hour.

The admission budget also retains a single 3,600-second CPU preparation allowance. That gives 43,865 of 57,600 CPU seconds and 4,458 of 7,200 GPU-slice seconds. No compute reservation remains outstanding. This allowance does not measure all development, review, transfer or reporting overhead. Process CPU, wall time, allocated CPU and reservations describe different quantities and must not be added together as independent costs.

The final CPU reporting job used two CPUs for 1,596 elapsed seconds, totaling 3,192 allocated CPU seconds, with no GPU. The subsequent archive collector used 97.013593 local child CPU seconds and 2,765.272798125 elapsed seconds. These collector measurements are separate from Slurm allocation and are not a measurement of the entire research cycle.

## Saved outputs and verification

- [Interactive comparison](/Users/pat/.cursor/projects/Users-pat-code-ACE/canvases/ACE-24-system-comparison.canvas.tsx), with all eight policies, every predeclared group, all 28 paired contrasts and all 24 system curves.
- [Policy summary CSV](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/policies.csv).
- [All 192 comparison cells](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/cells.csv).
- [All 28 paired comparisons](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/pairs.csv).
- [All predeclared group summaries](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/strata.csv).
- [All 4,540 evaluated curve points and target predictions](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/curves.csv).
- [Acquisition counts](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/actions.csv).
- [Unaltered campaign summary](../../../results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/opaque/cpu/cpu/worker/artifacts/campaign/private/summary.json).
- [Independent finalizer execution review](../../../results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/review.md) and [verified report retrieval](../../../results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/handoff.json).

The original summary SHA256 is `3f17abf3428411c44095dadcc3d80515267734d3729a0dbaef0989746dd7be0a`. The [display derivation](../../../results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/derivation.json) binds its input receipts, all report members, extraction source and CSV/display outputs. No model or scientific evaluator was rerun to create this writeup. The original reports retain their formatter-level qualification flags. The separate execution and numerical reviews establish the narrower provenance and arithmetic claims.
