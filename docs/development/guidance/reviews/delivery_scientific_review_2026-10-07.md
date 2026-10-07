# Scientific review of the delivery manuscript — 7 October 2026

## Scope and assessment

Reviewed the anonymous TMLR working draft in `paper/aistats_ace_2027/paper.tex`, beginning at documentclass (line 2404), its authoritative theory/table/bibliography companions, the delivery handoff and execution guidance, and accepted confirmation/Stage A/Stage C receipts. Bundled style code was excluded. Line anchors below refer to this working version; companion anchors are more stable when the generated bundle changes.

This review supports a **scoped empirical and methodological paper, conditional on completing the already registered evidence and tightening attribution and novelty**. The accepted results are credible evidence of differences between specified fitting recipes. They do not yet justify a broad preference for structured delivery, a causal decomposition of the online-to-delivery gain, or a new learning algorithm. The physical negative result is scientifically useful and should remain prominent. No acceptance outcome can be promised, and neither universally favorable results nor universal method superiority is a reasonable completion criterion.

Applied the user-supplied AGENTS.md resource and preservation instructions. No on-disk AGENTS.md was found in the repository or checked ancestor locations. No experiments, training, scheduler actions, closed APIs, environment changes, frozen-worker edits, manuscript edits, or commits were performed. No unopened Stage B responses or scores were inspected. Stage B status is taken from the supplied handoff: source `45ebeb89d2c76daa97a55f07728239245e0c4f60`, last verified 560/640 fits, no accepted results. Public primary literature was consulted for attribution. This file is the only written artifact.

Severity: **P1** materially affects scientific interpretation or submission readiness; **P2** affects precision, reproducibility, or assessment of scope. Each recommendation distinguishes **existing Stage B work**, **prose/theory/reporting using accepted evidence**, and **new experiments or resource decisions**.

## Prioritized findings

### 1. P1 — Independent-system evidence remains an existing completion gate

**Anchors:** [Abstract](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2424), lines 2428–2446; [Frozen prospective evaluation](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2693), lines 2695–2766; [Conclusion](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2857).

The notice and conclusions correctly keep Stage B pending. The completed 0.183 confirmation ratio is based on twelve histories of one emulator, an exposed grid containing acquired inputs, and a median of three scored initializations. Those constraints materially limit the headline even though the paired arithmetic is independently accepted. Stage A assesses the same exposed grid; it cannot independently validate the selected recipe. Stage C provides an action-block holdout, but one physical mechanism and losses to Fourier in all eleven conditions cannot establish independent-system structured-surrogate superiority.

**Recommendation — existing Stage B work:** integrate the complete frozen matrix only after the 640-fit seal, held-out evaluation, independent result audit and successful audit execution. Report all four registered contrasts, both strata, the short-fit ablation and initialization sensitivity. If the flat control erases the advantage, narrow the empirical contribution accordingly. A failure to meet a superiority gate is a reportable result, not a reason to replace worlds, modify the recipe or run a favorable sweep. No additional study is required by this finding.

**Evidence:** [handoff](/Users/pat/code/ACE/docs/development/guidance/handoff_delivery_2026-10-07.md), [full registration](/Users/pat/code/ACE/results/delivery_prospective_preparation_20261006/full_registration.json), [independent confirmation statistics](/Users/pat/code/ACE/results/delivery_final_history_20261006/independent_statistics.json).

### 2. P1 — The nearest precedent for delivery-time scratch fitting is missing

**Anchors:** [Introduction/contributions](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2470), lines 2470–2500; [Online learning and final prediction](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2568), lines 2568–2577; [bibliography](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_references.bib:1).

The paper acknowledges that batch refitting and replay are established, but cites online-to-batch theory as its only direct precedent for final prediction. **GDumb** already stores examples from a stream and trains a model from scratch at test time; the principal temporal separation is therefore not novel. Its classification setting and memory policy differ from this paper's interventional mechanism fits. That difference should be explained directly. [Prabhu, Torr and Dokania, ECCV 2020, abstract and Section 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf).

Replay of earlier observations is also directly represented by [Rolnick et al., Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682). These precedents do not establish results for the present causal setting, but they make a generic “refitting retained history improves final prediction” contribution insufficiently differentiated.

**Recommendation — prose/literature:** add the closest precedent and state the defensible distinction: intervention-specific eligibility, measured-parent fitting versus predicted-parent deployment, explicit reconstruction of admitted versus paid rows, recipe/compute controls, and retained failures under independent custody. Present the propositions as explanatory results, as the draft already does. Explain what transferable methodological lesson the controlled contrasts establish beyond the expectation that a fresh batch fit can outperform a bounded replay learner. Do not demand a new architecture or a novel theorem simply to manufacture novelty. Stage B can strengthen the empirical contribution, but cannot make scratch refitting itself new.

### 3. P1 — Scratch ablations do not partition the online-to-delivery gain

**Anchors:** [Interpretation in the introduction](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2492); [Stage A results](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2659), lines 2659–2667; [Discussion](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2812); [Conclusion](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2858).

The 0.096 final-buffer contrast and 0.034 short-fit contrast are large **conditional scratch-fit contrasts**, rather than shares of the original online-to-delivery improvement. The online model retains information in weights and Adam states from earlier updates. A scratch model trained on its final buffer does not retain that information. The table makes this difference visible: final-buffer long SCM is 5.466 times online NMSE, whereas delivery is 0.522 times online NMSE. One cannot infer that the online learner had only the final buffer's learned information.

In addition, changing the row set changes eligible training minima/maxima and therefore network input normalization. Equal epochs hold update counts fixed, but do not hold process CPU, processed examples, or optimization conditioning fixed. The admitted long fit costs 0.783 CPU hours versus 1.455 for the all-paid long fit. Replay weighting and scratch optimization also differ from online training, as the draft correctly acknowledges. These differences prevent treating the contrast magnitudes as an additive or uniquely identified causal decomposition.

**Recommendation — prose:** replace “explain more of the observed improvement” with a precise statement such as: “Within the scratch-fit matrix, long fitting on retained histories substantially outperforms fitting on the final buffer or using 100 updates; adding unused paid rows changes aggregate continuous error much less.” Explicitly acknowledge normalization and optimization-state differences. The existing factorial supports this narrower statement without further fits.

**Evidence:** [attribution table](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_attribution_table.tex:8), [accepted summary](/Users/pat/code/ACE/results/delivery_attribution_20261006/summary.json), [normalization and fitting implementation](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:178), lines 178–193.

**New experiment boundary:** a literal decomposition of the original gain would need additional controls for normalizers, replay weights, retained optimizer/model states and fitting budgets. That would be a new design decision; it is unnecessary for the narrowed paper and must not be implemented in frozen Stage B.

### 4. P2 — The 0.980 unused-row average conceals substantial heterogeneity

**Anchors:** [Stage A interpretation](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2661), lines 2661–2665; [retained failure](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2681); [Discussion](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2815).

“Little evidence of a benefit” is reasonable as a statement about the geometric mean of continuous error, but readers may interpret it as negligible effects across histories. Direct arithmetic on the accepted summary gives **10/12** continuous-error improvements for all-paid versus admitted long SCM, with ratios ranging from **0.644433 to 2.873995**. History 124753321 has the 2.873995 ratio; another history has 1.091362. Thus the near-unit aggregate combines improvements and a large adverse effect. Moreover, the table's snapped ratios imply a descriptive all-paid/admitted geometric ratio of approximately **0.469**, unlike the continuous-error ratio of 0.980. Endpoint dependence is part of the finding.

**Recommendation — accepted-evidence reporting:** say “little aggregate continuous-error improvement,” include the win count and adverse range, and explicitly distinguish the snapped contrast. Provide the primary initialization's per-history errors or a receipt-backed distribution display in the final paper. Include a compact sensitivity summary: accepted SCM/flat continuous ratios are approximately 0.551, 0.458 and 0.519 for initializations 0, 1 and 2, respectively. Keep initialization zero primary and retain all failures; these summaries are not new selection rules.

**Evidence:** [accepted summary, `fits` and `configurations`](/Users/pat/code/ACE/results/delivery_attribution_20261006/summary.json), [worsening-history scores](/Users/pat/code/ACE/results/delivery_attribution_20261006/scores/124753321.json). Calculations here only divide already accepted errors; no models or outcomes were regenerated.

### 5. P2 — Matched process CPU establishes a budgeted recipe contrast, not optimized baseline superiority

**Anchors:** [Abstract](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2435); [Stage A controls and scope](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2669); [matched-CPU implementation description](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2896), lines 2896–2905.

The matched-CPU flat control is appropriately labeled and its irrelevant SCM branch is disclosed. However, the accepted matched flat fits take **135,682–165,895 updates**, versus 30,000 in the equal-epoch flat arm. Their geometric NMSE/online ratio is **1.234**, compared with **0.948** for the 30,000-update flat arm. This establishes worse performance for that fixed long-training rule; it does not demonstrate that flat prediction cannot use the same compute effectively. Nor does process-CPU equality equate intermediate labels or capacity: the archived SCM has **22,341** parameters across five heads versus **4,801** in flat prediction, and primary all-paid eligible counts are 4,303/4,803 per SCM head versus 3,803 flat rows.

**Recommendation — prose/reporting:** state that this is a comparison against a specified CPU-budgeted flat recipe; retain both flat controls and their costs. Avoid wording that generalizes from this control to an optimally allocated equal-compute advantage. “Development-selected” is justified, but “strongest simpler” should mean strongest among the frozen candidates under that development procedure. The prospective flat comparison also remains a whole-recipe comparison with unequal supervision and compute.

**New experiment boundary:** validation-selected early stopping, a parameter-matched flat model, or action-aware flat regression would require a separately designed study. They are not prerequisites for the existing scoped recipe comparison and must not be added after Stage B outcome exposure.

### 6. P2 — The outside-box diagnostic measures true parents, not deployed predicted parents

**Anchors:** [Failure diagnostic](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2683), lines 2683–2690; [diagnostic implementation](/Users/pat/code/ACE/scripts/research/delivery_attribution.py:250), lines 250–256.

The implementation computes `outside_training_parent_box_fraction` from `truth.nodes` parent vectors. Therefore the reported **0.020%** is the fraction of **true evaluation parent vectors** outside coordinate training ranges. It is not the fraction of free-running predicted vectors outside those ranges. The present phrase “evaluated target parent values” is ambiguous in a paragraph concerned with deployment shifts. The propagation proposition requires control on a region containing both true and predicted parent vectors.

**Recommendation — prose:** name the true-vector quantity explicitly and say predicted-parent coverage is not established by it. Keep the already correct caveat that coordinate boxes do not certify joint support. Define local residual and prediction shift algebraically: with `r = fhat(p_true) - y` and `d = fhat(p_pred) - fhat(p_true)`, chain MSE is `E[r²] + E[d²] + 2E[rd]`. This explains the recorded nonadditivity without asserting a unique cause. No new diagnostic evaluation is needed for this correction.

### 7. P2 — The archival headline lacks essential task and endpoint details in the manuscript itself

**Anchors:** [Endpoints](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2542); [Completed confirmation](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2591); [Archived implementation](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2886); [attribution table caption](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_attribution_table.tex:26).

The main confirmation describes only a “single deterministic emulator” and an exposed discrete grid. Its actual graph/mechanisms, root action domain, quantization levels/tie rule, history construction, and grid size/weighting are not specified there or in the implementation appendix. The detailed prospective equations do not define the distinct archived task. The endpoint section defines training-variance NMSE specifically for prospective results, while the archival table uses NMSE without giving its normalizer. This is especially relevant because quantization is the primary archival result and continuous error leads the attribution narrative.

**Recommendation — reporting:** add a compact archived-task specification from frozen metadata, including the target, graph, action support, grid/quantizer and score normalizer. Provide the verified unique eligible counts: final buffer 50; admitted rows 1,178 per head; all-paid 4,303 or 4,803 per SCM head; flat 3,803. Distinguish counts of paid responses from eligible rows and update counts. Make the sampling basis for twelve independently seeded histories explicit. References to hashes and local custody support reproducibility but cannot substitute for describing the experimental task.

**Evidence:** [accepted attribution summary](/Users/pat/code/ACE/results/delivery_attribution_20261006/summary.json), [input protocol](/Users/pat/code/ACE/results/delivery_paper_implementation_20261006/stage_a_input_protocol.json), [confirmation scores](/Users/pat/code/ACE/results/delivery_final_history_20261006/scores.json).

### 8. P2 — Prospective graph size should not be interpreted as general task complexity or scaling

**Anchors:** [System scope](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2701); [prospective equations](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2918), lines 2918–2938.

Substituting the five-node equations gives `X3 = (c-da)X1 - db + s sin(aX1+b)`. The scored map depends on one root; `X4` and the `X5` branch are irrelevant to `X3`. The thirty-node family is layered, with mostly linear mechanisms and selected `u + 0.2 sin(u)` perturbations, positive bounded weights, and a single terminal target. Independent parameterizations are valid experimental units, but these families are restricted and differ in more than node count. A contrast between strata cannot identify the effect of increasing graph size alone.

**Recommendation — prose:** disclose the effective one-root five-node target and keep claims within the declared generators, targets and root-intervention distribution. Report graph size as a stratum descriptor, not evidence of broad scalability, varied nonlinear causal difficulty, or unrestricted intervention generalization. The current root-support counterexample is appropriate and should stay.

**New experiment boundary:** new generator families, richer nonlinearities, multiple terminal targets, noisy systems and internal interventions would broaden the claim through new experiments. They are optional future research requiring a separate scientific/resource decision, not grounds to delay the existing bounded study.

### 9. P2 — Completed compute and failure accounting belongs in the paper, beyond a promise of per-fit receipts

**Anchors:** [Reproducibility and compute](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2795); [Implementation appendix](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2874); [provenance](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2983).

Stage A's 10.87 fit CPU hours and Stage C's 0.053 fit/evaluation CPU hours are clearly scoped. The paper lacks a consolidated account of the original confirmation's charged responses, interrupted attempt/amendment, total execution costs, and the failed prospective qualification pilot. Without it, readers cannot assess the acquisition-versus-compute tradeoff motivating “delivery.”

**Recommendation — existing receipts/reporting:** include a compact accounting appendix distinguishing newly acquired responses, reuse/evaluation of pre-existing responses, fit process CPU, supervised wall time, memory and scheduler allocation. The accepted confirmation terminal audit records **60,962 aggregate charged calls including discarded attempts**; do not label that aggregate as the per-history or per-arm count. Preserve the failed pilot's **126 allocated CPU-seconds** and the replacement pilot's **119**. Stage B's **24.540833 accrued allocated CPU hours** is only the last supplied checkpoint; obtain final totals through the main agent's existing acceptance work. The **86.868333** reservation is not observed compute. No new resource use is justified by this reporting request.

**Evidence:** [confirmation terminal audit](/Users/pat/code/ACE/results/delivery_final_history_20261006/terminal_audit.json), [Stage A completion](/Users/pat/code/ACE/results/delivery_attribution_20261006/complete.json), [physical acceptance](/Users/pat/code/ACE/results/delivery_chambers_20261006/acceptance.json), [handoff accounting](/Users/pat/code/ACE/docs/development/guidance/handoff_delivery_2026-10-07.md).

## Statistics and causal theory: what is sound and what to clarify

The accepted confirmation ratio, interval and exhaustive sign-flip p-value agree with the independent statistics receipt. Histories are correctly treated as paired units; the manuscript already states the fixed-emulator scope and symmetry assumption. Enumeration makes the sign-flip calculation exact conditional on its null assumptions; it is not a randomized assignment test or a distribution-free test of all mean-zero alternatives. Retain that interpretation rather than changing the primary test because the draft needs stronger evidence.

The frozen Stage B system-level mean of two history log ratios, four-test Holm correction, fixed initialization and complete-matrix requirement are appropriate for the declared estimand. Marginal intervals are correctly distinguished from simultaneous intervals. A ratio estimate at most 0.8 together with an upper interval below 1 establishes the registered superiority gate; it **does not establish with 95% confidence that the population improvement is at least 20%**. Preserve the distinction in future abstract wording. Report absolute errors and individual system log ratios alongside ratios, floor activations and history-specific summaries. Do not redesign the registered inferential analysis after looking at outcomes.

The [eligibility proposition](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex:8) is valid under its stated invariant, independent additive-noise, pre-noise intervention and nonselective-retention assumptions, with finite second moments for squared loss. It gives a conditional-mean estimand on supported parents, not finite-sample identification. For formal precision, replace “retention does not select” with an explicit conditional mean-zero condition or conditional independence given parents/action/history, including any terminal selection of rows. Fixed planned all-paid collection is compatible with that condition; noisy response-selected probes or outcome-dependent stopping need not be. Empty eligible sets also require an explicit rule before applying the empirical objective in paper lines 2512–2520.

The [deterministic propagation proposition](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex:52) and finite triangular path sum are correct under the stated uniform approximation and learned-function Lipschitz assumptions. The draft properly refuses to substitute observed MSE for a uniform bound. The [root-support counterexample](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex:85) is mathematically sound and relevant to the restricted prospective design. Neither proves the cause of the empirical failure.

The [quantization proposition](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_theory.tex:109) is sound with fixed ties and at least two distinct levels; state the evaluation probability measure when using its probability notation. In theory lines 42–50, clarify that conditional-mean composition can fail **even for the preceding additive-noise class**: `X=W+U`, `Y=X²`, with zero child noise, already supplies that example. The failure is not restricted to nonadditive stochastic SCMs. These are small theory/prose repairs, not requests for a general identification theorem.

The physical relative-angle basis and Fourier specification are correctly reported, and the **7/11 rolling, 3/11 physics, 0/11 Fourier** counts match the accepted receipt. The conditional bootstrap is transparently block-weighted rather than row-weighted. With eight deterministic held-out blocks and possible archive-order dependence, describe its intervals as a conditional resampling assessment under block exchangeability, not uncertainty over apparatuses or the entire action domain. Keep all eleven rows and the unadjusted, descriptive interpretation. No extra apparatus study or significance fishing is needed.

Existing citations to [Xia et al.](https://arxiv.org/abs/2107.00793) and [Ross et al.](https://proceedings.mlr.press/v15/ross11a.html) are appropriately scoped: the former separates expressiveness from learnability/identification, and the latter addresses prediction-induced distribution shifts under its own reduction assumptions. The draft does not transfer their guarantees improperly. The [Causal Chambers attribution](https://arxiv.org/abs/2404.11341) is also appropriate. The priority literature problem is the missing direct delivery/refitting precedent, rather than those existing citations.

## Recommended order of action

1. **Before outcome integration:** repair novelty positioning, narrow the attribution wording, specify the archival task/normalizer, and name the true-parent coverage diagnostic. These are manuscript changes for the main agent; this reviewer has not edited manuscript sources.
2. **Using completed evidence:** report unused-row heterogeneity, fixed-initialization sensitivity, eligible-row/parameter/update counts, all physical negatives and complete historical accounting. Export numerical additions through the established receipt-backed claim generator rather than hand-editing generated values.
3. **Through existing Stage B gates:** complete the original frozen study and independent acceptance, then integrate all primary and secondary outcomes and actual final costs. Preserve the pending notice until those gates pass; if results are unfavorable, finish the narrower report.
4. **Only if the authors choose broader claims:** separately decide whether new matched-supervision controls, noisy/distributional estimands, internal interventions or new physical systems warrant another study and resource budget. None is authorized or required by this review for the scoped delivery/accounting paper.

## Evidence checks and completion

Read-only SHA256 checks confirmed that the current claim index binds the confirmation scores, independent statistics, Stage A summary/completion/gate and Stage C acceptance files to their recorded hashes. Scientific acceptance relies on the linked validated receipts; this bounded review did not replay fits or conduct a new raw-outcome audit. The scope does not include final anonymity, author disclosures, visual layout or submission packaging review.

Review complete. All unfavorable histories/conditions and failed attempts remain part of the recommended reporting. Only `docs/development/guidance/reviews/delivery_scientific_review_2026-10-07.md` was created; no other file was edited and no commit was made.
