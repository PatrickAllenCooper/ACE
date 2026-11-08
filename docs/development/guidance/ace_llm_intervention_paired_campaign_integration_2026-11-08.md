# Proposed artificial paired-campaign integration

November 8, 2026. **Proposed plan only: not frozen, approved for execution, submitted, run or accepted.**

Training GPU03, job 35379434, is now independently accepted with zero required defects. It completed fresh four-step SFT and two-step DPO training, both exact checkpoint reloads, ten legal generations and clean shutdown. Its acceptance is in `results/ace_llm_intervention_training_gpu_20261108/acceptance03`. The two failed predecessors remain unchanged.

Current closed allocation is **9,210 CPU-seconds and 1,347 GPU-slice-seconds**. With the unchanged 3,600 CPU-second preparation allowance, admission is 12,810/57,600 CPU and 1,347/7,200 GPU. No campaign resources are allocated. The prospective caps below would bring admission to 19,410 CPU seconds and 1,947 GPU-slice seconds. Reconcile current jobs and journals before any submission.

## Next gate and scope

Run one new **two-world × eight-policy artificial campaign**: CPU preparation and numerical controls, three sequential GPU inference services, then CPU private reporting. The deliverable is actual charged trajectories and a complete 16-cell cost comparison, not another component review or receipts-only milestone.

Reuse the accepted collection, exposure, training-plan, ranker trainer, screen policies, paired runner, campaign, model service, sealing and reporting components. Do not recollect the saved training batch, repeat LLM training, replay old test suites, install packages or generate the scientific 32-training/24-development descriptors. Artificial results establish integration, not scientific usefulness or permission to begin the scientific screen.

## Small supplied plans

Propose artificial IDs **9101 and 9102**, separately admitted before execution and distinct from training IDs 17/4 and earlier campaign fixtures 51/52. Construct descriptors directly from these hand-written laws; do not call the scientific generator.

| Field | Artificial 9101 | Artificial 9102 |
|---|---|---|
| Private laws | X = epsilon_X; Y = 1 + 2X + epsilon_Y | X = epsilon_X; Y = -0.5 + X + epsilon_Y; Z = 0.25 - 2Y + epsilon_Z |
| Independent noise variance | 0.25 per node | 0.25 per node |
| Target | Y | Z |
| Menu, in order | observe_all, x_minus: do(X=-1), x_plus: do(X=1) | Same actions |
| Measurements | All nodes | All nodes |
| Costs, in menu order | 1, 1, 1 | 1, 1, 2 |
| Fixed queries | do(X=-1), do(X=1), measuring target | Same |
| Analytical structural query means | -1, 3 | 3.25, -0.75 |

Both plans use X intervention bounds [-2,2], public learner noise 0.25, zero prior coefficient means and prior variance 1. Queries have equal weights, unit query cost, scale 2 and tolerance 0.04. Every arm receives two separately charged `observe_all` initial responses, followed by an acquisition cap of four cost units. Initial response keys are shared within a world; acquisition namespaces are arm-specific. Public prose contains no private coefficients, world IDs, seeds, regime labels or private scores.

Freeze the ordered roster: `uniform_random`, `menu_coverage`, `local_information`, `target_risk`, `learned_ranker`, `frozen_llm`, `sft_llm`, `dpo_llm`. Use a case label for descriptive grouping; graph and cost changes are confounded and do not identify a separate cost effect.

Bounds: **16 cells; eight requests per LLM service; 24 total LLM decisions; at most 96 campaign response attempts and 96 response-cost units**, including initialization. Stops reduce actual spending. Deterministic private structural evaluation adds no paid responses.

## Necessary integration work

Implement one phase-aware outer driver with a pinned launch shim and saved-evidence validation. Leave accepted component semantics intact.

1. **Bind runtime staging without changing campaign plans.** Plans freeze exact commands before CPU arms execute, whereas the service launcher needs the later node-local interpreter path and staging-receipt hash. Freeze a stable shim command; the supervisor supplies and authenticates the derived runtime binding and records the executed command. Preserve the accepted explicit GNU child environment and scheduler GPU visibility. Do not patch campaign commands between phases.
2. **Admit CPU evidence independently.** Supply retained CPU attempt pins and successful CPU-phase acceptance to the GPU supervisor. The campaign's local phase barrier alone hashes currently present receipts and is not independent custody admission.
3. **Bind checkpoint roles to accepted training.** Verify SFT after four updates and final DPO after six from the same successful fresh run. Both artificial checkpoints must use service treatment `artificial_adapter`, training scope `artificial`, and separate campaign policy IDs. Do not relabel artificial training as `policy_train` or relax the service contract.
4. **Join dynamic service evidence to execution.** The old GPU service validator hard-codes four requests numbered 40–43. The new validator must reconcile world/task, episode decision index, session request ID/ordinal, envelope/prompt hash, raw tokens/completion, executor proposal and charged response. Verify one model load per service, task rebinding across worlds, unchanged model tensors and clean closure. No repeat-state service replay is needed.
5. **Qualify only the new input/closure boundary.** Capture the additional campaign/numerical source closure and tokenize the new public inputs through `prepare_task_tokens`. The old `case0/case3/stress` preflight is not a campaign preflight. Preserve supervisor telemetry outside sealed policy-attempt inventories. Limit implementation validation to these new joins; do not reopen closed component suites.

## Execution sequence after training acceptance

**CPU first.** Authenticate the complete saved collection terminal, inventory, records and supervision receipts; the exact exposure; accepted tokenized plan; and accepted GPU03 training outputs. Reuse all four records, four SFT exposures, two selected DPO pairs, two unused strict pairs and two ties. No collection rerun or replacement examples.

Train the numerical control once on that exact exposure using the accepted 128-unit, seed-109 recipe with continuous optimizer state. `numeric_recipe(artificial=False)` selects the full architecture while data scope remains `artificial`, which the trainer permits. Retain both exports and reload evidence; deploy only the predeclared final DPO ranker. Do not use the older default epoch-based trainer or add a ninth arm.

Freeze all evaluation plans, checkpoint roles, source/model pins, namespaces and commands before evaluation responses. Initialize the campaign and run all five numerical policies over both worlds. The four acquisition controls use `llm_intervention_screen_policy.py`, its keyed decision-index bindings and 32 planning draws. The existing learned-ranker CLI can serve deterministic checkpoint inference; its computed legacy seed does not determine the ranker action. Tokenize new representative public envelopes, including bounded maximum-history/affordability shapes, without loading Qwen weights. Actual adaptive prompts retain the service's runtime token bounds.

**GPU second.** Run frozen, SFT and DPO sequentially, each in one fresh service reused across both worlds. Close each before starting the next. Use the same seven pinned Qwen files, revision `c1899de289a04d12100db370d81485cdf75e47ca`, accepted service/runtime closure and inference recipe. Frozen loads no adapter. Perform no optimizer updates, dataset preparation, numerical-control runs or private scoring in this phase.

Tiny adaptive simulator responses and SCM updates remain on host CPUs between LLM decisions. They cannot be prepared in advance without changing the episode. Timestamp and bound this work; no distributed simulator or uncharged response cache is needed.

**CPU last.** After all services terminate, authenticate every attempt and call the existing `finalize_campaign` once. Produce all 16 cell reports and 28 descriptive policy-pair comparisons. Reconcile original journals with actual response counts and resources; component-local zero-response finalization fields do not describe the whole campaign.

## Acceptance and reporting

Operational acceptance requires complete, sealable trajectories for all 16 cells; the correct three model identities/checkpoint roles; authenticated shared initialization; full generation-to-action and charge reconciliation; and clean process/GPU closure. Failed or unresolved cells remain in the report at the common acquisition cap of four, but a failure-containing report does not establish successful integration. Preserve failures without retries, replacement worlds or complete-case filtering.

Report first-crossing acquisition cost, initial-plus-crossing cost, actual charges, final error, attainment, policy stops and failures separately. Retain complete curves and the analytical structural-mean checks. Private scores never govern policy stopping or return to a policy. No LLM win or attainment threshold is an acceptance requirement. If a model only stops, report that its acquired-feedback-to-next-decision path was unexercised; do not force interventions or silently claim that coverage.

The outer acceptance links collection → exposure → tokenized plan → successful training → role-specific checkpoints → service identity → arm evidence → reports. Keep semantic plan digests distinct from artifact byte hashes. Record previous training costs as reused provenance, not newly incurred responses, and retain the current campaign's costs separately. All scientific acceptance/freeze flags remain false.

## Proposed resources

| Phase | Initial bounded request |
|---|---|
| CPU preparation, ranker and numerical arms | 2 CPUs, 7,680 MiB, 600 seconds, zero GPU |
| Three sequential LLM services | One A100 `3g.20gb` slice, 4 requested/max 8 allocated CPUs, 12 GiB, 600 seconds |
| CPU finalization, if separately scheduled | 2 CPUs, 7,680 MiB, 300 seconds, zero GPU |

The CPU memory proposal provides room above prior near-6-GiB batch peaks for ranker/subprocess work without another CPU. GPU14 measured approximately 3.26 GB reserved GPU memory. GPU03 measured 3.278 GB reserved and 95.05% of its 12 GiB host-memory limit. The latter includes runtime staging and constrains any wider campaign host-memory projection. Use accepted GPU03 measurements and the new CPU timings to finalize the projection. One sequential GPU slice is sufficient as the starting bound; no multi-GPU allocation is justified.

Proposed account is `ucb736_asc2`, CPU partition/QOS `acpu/cpu-normal`, GPU `aa100/gpu-testing`, with no requeue. Proposed exclusive output is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_01`. Reconcile actual current jobs, existing controller locks, journals, account and slice availability before any once-only submission.

Retain the 240-second first-verified-work limit, 90-second progress/telemetry guard, and cleanup margin inside the GPU cap. Size session admission for three services rather than copying the old two-session deadline assumption. Require positive CUDA work and advancing generation records together with independent owned-GPU-process evidence for every service. Record staging, model loading, inference, adaptive CPU work and final-output timestamps. Missing utilization percentages remain unknown, not proof of idleness or high utilization. Diagnose sustained CPU-only occupancy promptly and stop only owned work when safe, preserving evidence.

The proposed caps add at most **6,600 allocated CPU-seconds and 600 GPU-slice-seconds** if all three phases use the stated scheduler bounds. Recompute cumulative admission before submission; preserve the existing preparation allowance. Compare requested versus observed allocation, memory, throughput and elapsed time after this gate before considering scientific scale-up.

## Inspected implementation anchors

- `scripts/research/llm_intervention_campaign.py`: `prepare_campaign`, `run_policy`, `finalize_campaign`.
- `scripts/research/llm_intervention_paired_episode.py`: supplied plans, charged arms, shared-initial sealing and private finalization.
- `scripts/research/llm_intervention_training_collection.py`, `llm_intervention_training_exposure.py`, `llm_intervention_training_plan.py`: complete saved-batch lineage and matched exposure.
- `scripts/research/llm_intervention_ranker_training.py`, `llm_intervention_screen_policy.py`, `llm_intervention_policy_cli.py`: matched numerical training and public inference.
- `scripts/research/llm_intervention_model_service.py`: artificial adapter scope, identity and inference-only service.
- `scripts/research/qualify_llm_intervention_model_service.py`: staging-dependent command and historical fixed-request validator.
- `scripts/research/llm_intervention_service_runtime.py`, `llm_intervention_screen_reporting.py`: fixed old CPU fixtures and complete descriptive reporting.

This document records design only. Writing it performs no checks, fitting, descriptor construction, runtime qualification or submission.
