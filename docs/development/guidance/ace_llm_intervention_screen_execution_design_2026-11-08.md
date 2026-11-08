# First scientific screen: execution decisions still to qualify

November 8, 2026. Prospective implementation design. No scientific world has been generated and this document is not an execution freeze.

The screen asks whether the trained intervention policy improves the cost of reaching a fixed prediction threshold on new synthetic systems. It is a development comparison, not a submission-ready confirmation study or evidence of engineering transfer. All eight methods remain in the comparison even when an LLM performs poorly.

Use the existing [32-training/24-development proposal](ace_llm_intervention_screen_protocol_2026-11-07.md), including its supplied graphs, private coefficient laws, action costs, response namespaces, eight shared initial responses and 24 additional cost units. Preserve its preassigned teacher and behavior rotations. Neither the two artificial integration systems nor their outcomes choose scientific worlds, thresholds or training examples.

## Training and deployment

Adopt the already implemented [matched exposure rule](ace_llm_intervention_training_exposure_2026-11-07.md) only after complete collection acceptance. It gives one supervised step for each of the 160 records, followed by at most one strict preference pair from each of 128 nonterminal records. The unordered pair is chosen by the fixed record/command hash. Preserve all 3,584 comparisons, ties and unexposed pairs. A tie does not trigger recollection or another training recipe. A complete dataset with no strict pairs is an explicit training failure for this design.

The candidate recipe uses the qualified Qwen3-0.6B base and tokenizer, fresh seed109 rank8 query/value adapters with scale16, learning rate0.0001 and DPO beta0.1. AdamW uses zero weight decay, betas(0.9,0.999) and epsilon1e-8. The reference is fixed after the supervised phase. The optimizer continues into DPO as in the qualified engine. Final supervised and final DPO checkpoints define the two trained LLM arms. No checkpoint is selected using development errors. The frozen arm uses the same original base. Every deployed policy stays fixed within and across its evaluation episodes.

The numerical ranker uses the same admitted public states, labels and selected preference pairs in the same order. Its existing fixed128-hidden-unit recipe remains the numerical learning control. Matching example exposure does not equate parameter count, pretraining, FLOPs or inference latency. The four numerical acquisition controls remain uniform random, coverage, local information and target risk.

## Endpoints and interpretation

Keep all 192 development world/policy cells and all 28 descriptive policy pairs. The primary practical quantities are capped acquisition cost and attainment together, with complete error-versus-cost curves and final error. Retain actual charged cost separately from retrospective first threshold crossing. A method that stops early without meeting the target has not saved useful experiments. No private score may influence a policy action or stop.

Report **ever crossing the threshold** and **meeting the threshold at the final paid state** separately. They need not agree. The two-world artificial integration exposed this distinction: every method crossed at initialization on one supplied system, while several ended above the threshold after continued acquisition. That fixture is not a scientific performance estimate. It motivates clear reporting, not a change to the predeclared error threshold or a favorable replacement world. All actual charges remain visible. A retrospective first crossing cannot establish usable autonomous stopping.

Report the trained DPO policy against the frozen LLM, the supervised LLM, the learned numerical ranker and target risk explicitly. Also retain random and the other controls. The world is the paired unit. Describe all four preassigned regimes and both cost strata without selecting favorable subsets. This development screen will diagnose behavior and support a later frozen confirmation design. It does not by itself identify a pretraining effect, a causal-structure ablation or utility on a compressor.

## Remaining admission work

The larger public prompts contain three nodes, eight task queries, eight acquisition actions and up to 32 history rows. The accepted artificial campaign covered only 844–1338 prompt tokens. Its success does not establish full-screen context capacity. First tokenize representative complete artificial states at the planned screen size using the pinned actual tokenizer on CPU. Include long numeric spellings, all graph shapes, both cost menus, near-terminal budgets and terminal stop states. Keep every field and token. The 8192-token limit includes the generation allowance. A finite stress fixture cannot prove a bound for every future floating-point history. Every actual scientific record must independently pass the unchanged token limit before training or deployment.

If full states do not fit, diagnose and review a prospective representation or context change before scientific generation. Never silently truncate histories or fields. Obtain a measured runtime and memory projection for the actual selected representation. Use the smallest qualified GPU slice and split CPU collection/tokenization from training and inference. Current resource totals must be reconciled before any allocation. The artificial training cost cannot be multiplied by record count and presented as a guaranteed full-study cost.

The full scientific collector, trained-checkpoint handoff and paired evaluator still need a single reviewed, frozen source/runtime/resource contract. Namespace admission, complete collection accounting, actual model/runtime pins, failure preservation and independent reporting remain necessary. The current 16 CPU-core-hour / 2 GPU-hour envelope covers only stages whose measured projection fits. No allocation is requested by this draft.

The initial source/design review binds the predecessor draft preserved in `results/ace_llm_intervention_screen_prompt_preparation_20261108/preparation01/prospective-design.md`. The final-versus-ever attainment paragraph and spacing above were clarified after inspecting the complete artificial finalization. They remain prospective for the ungenerated scientific screen and belong in its final integrated design review.
