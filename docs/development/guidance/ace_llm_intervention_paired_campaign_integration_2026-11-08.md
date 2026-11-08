# Artificial paired-campaign integration

November 8, 2026. **CPU03 completed0:0, complete custody and saved validation passed. Independent acceptance complete. GPU01 completed0:0 as35388249, full custody and independent acceptance complete.** CPU01 remains a preserved failure before staging. The latest execution entry below supersedes the preceding snapshots. Routine bounded recovery and this artificial integration are covered by the current user authorization.

Training GPU03, job 35379434, is now independently accepted with zero required defects. It completed fresh four-step SFT and two-step DPO training, both exact checkpoint reloads, ten legal generations and clean shutdown. Its acceptance is in `results/ace_llm_intervention_training_gpu_20261108/acceptance03`. The two failed predecessors remain unchanged.

Current closed allocation is **10,368 CPU-seconds and 1,483 GPU-slice-seconds**. No ACE job remains active. Including the unchanged3,600CPU preparation allowance gives13,968/57,600CPU admission. Future submission reconciliation requires36terminaljobs and74journals. These are not whole-cycle process CPU.

## CPU finalization01 complete, final review pending

Job35389391 completed0:0 in61seconds,2CPU/7680MiB/zeroGPU, adding122allocatedCPU seconds. Complete2,219rawfiles/110,220,879bytes andall94transferpins match bothlocalcopies andtwo remoteinventories. FinalizationSHA`aa77436fbda2e3348b7eae08d33327c1fa5e33ae3e6602a4abd69930a53c6dfa`, summarySHA`790cb2ecbf6af7e5e63ad10e7e8056f7b69e75f68c185ad345004ce430c7d577`. All16cells/28pairs are retained withzero failed/unresolved cells andzero new responses/updates. Exactruntime,groupcleanup and scheduler completion are recorded. Independent final-result review is pending. Preserve the published finalization even if later review identifies a defect.

The supervisor took51.9811seconds and26.5349processCPU seconds. Child/cleanup CPU was1.414002seconds with163,819,520peakRSS bytes. The child resource elapsed field uses the original supervisor epoch and is not an isolated scoring duration. SlurmTotalCPU28.062seconds/batchMaxRSS7,862,160KiB overlap other measurements. Do not add them or infer memory headroom from the lower child-only peak.

## CPU finalization01 submitted once

Job35389391 was submitted at2026-11-08T08:51:13.343861Z. Packetcommit`380d1b3d939a64ffd0602014746df7304035cc6d`, source`12c7322e5046cd3e9b38cb59a1c01ed87054b71b`, packetSHA`417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c`. IntentSHA`dedd39caaf131af4d08fdf9d4750934d622fa8993f4568a6eb22c75bb8fbb7da`. The concretepacketreview has zero required findings. Both shared locks,35terminaljobs,72journals, exactoriginalroots/acceptances, account/partition/scratch and94transferpins passed before submission.

The request is2CPU,7680MiB,300seconds,zeroGPU,ucb736_asc2/acpu/cpu-normal,no-requeue. NewROOT`/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01` owns the launch and supervisor evidence. Original campaign paths remain unchanged. Startup/work/cleanup/hard limits are120/260/280/290seconds. Require full16-cell/28-pair reporting, zero new responses/updates, exactruntime/resources, ownedcleanup andcompletecustody before independent final-result acceptance. Never blindly repeat a published campaign finalization after an outer failure. Full-size scientific worlds remain ungenerated.

## GPU01 terminal custody

Job35388249 completed0:0 in136seconds on the requested4CPU/15GiB/one20GBslice. All1,834rawfiles and108,890,768bytes match exclusive external custody. All transferred source/input pins match. The three services each completed eight positive CUDA generations and closed cleanly. Independent NVIDIA process samples establish GPU memory use alongside generation progress. CUDA intervals are3.256/4.777/6.039seconds for frozen/SFT/DPO. These are not whole-job utilization percentages. No SM utilization percentage was recorded.

The allocation adds544CPU-seconds and136GPU-slice-seconds. Supervisor elapsed129.7967seconds, its process CPU and49.963905helper/service CPU seconds remain separate scopes. SlurmTotalCPU92.205seconds andbatchMaxRSS10,252,000KiB are overlapping measurements. All original adaptive responses are in the charged arm journals. Private score finalization has not run. Independent result acceptance is complete with zero required findings. AcceptanceSHA`26119972f7c3e61caf03d15bc72371bfe64dc118edd55d1e6bd26f917e994d0b`, reviewreceiptSHA`1147e74630d1c12b8519e7dc2446a2b112c517e4518aa95852c9f6f54ed52016`. The separate CPU finalizer packet is prepared but unsubmitted at`results/ace_llm_intervention_campaign_finalization_20261108/launch01`, SHA`417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c`. Its source is the previously reviewed12c7322e finalizer. Concrete packet review is pending. Terminal dispositionSHA`c3a754e54cac4fcc8e8959e14020e9fdaa772b659761bd9c28a84754de30db4e`. Reconcile35terminaljobs and72journals before further submission.

## GPU01 submitted once

Job **35388249** was submitted at **2026-11-08T08:30:13.531221Z**. Source commit`12c7322e5046cd3e9b38cb59a1c01ed87054b71b`, packet commit`07722b85059e47f8e1de27c4c9a0840d03aac420`, packet SHA-256`970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976`. Independent packet review found zero required fixes. Both locks,34terminaljobs,70journalpins, original CPU inputs/acceptance, account/partition/slice and scratch readiness reconciled. All89transferred pins passed readback and the original GPU byte-admission routine before the single submission. Remote intent SHA-256`830d77db4c57093d95b28e6c854e8a7d9432b76297bbf74e23c409c59ce53e35`.

The request is one A100`3g.20gb` slice, four CPUs with a reserved ceiling of eight allocated CPUs,15GiB and600seconds on`ucb736_asc2/aa100/gpu-testing`,without requeue. The new launch root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01`. Actual GPU output remains the frozen sibling`ace_llm_intervention_campaign_artificial_20261108_03/gpu03`. Original CPU inputs and three service contracts are unchanged. Launch custody is`results/ace_llm_intervention_campaign_gpu_20261108/launch01` and external`ACE-LLM-intervention-campaign-gpu-20261108-01`.

Frozen,SFT and DPO services run sequentially on both artificial worlds. No optimizer updates occur. Adaptive responses remain charged in their original arm journals. Require advancing positive generation records plus independent owned GPU-process memory, original startup/progress deadlines, clean closure, exact runtime/source/model/role identities, actual scheduler resources and full custody before accepting this phase. Private finalization remains separate and pending.

The new CPU-only finalizer has passed23fabricated tests and a bounded independent static review with zero required findings. Both test attempts, including the initial scheduler-parsing failure, are retained. It has not been allocated or executed. It will use the original finalization interface only after independently pinned GPU acceptance and whole-campaign custody. These preparations do not qualify a scientific screen.

## CPU03 independently accepted

Job **35387068 completed successfully** in 119 seconds on two CPUs with 7,680 MiB and no GPU. It used 238 allocated CPU-seconds. All ten CPU policy cells and 36 numerical decisions completed. The numerical ranker received the planned six training updates. Both ranker cells completed and no policy failed.

Complete custody contains **951 raw files and 89,451,811 bytes** in `cpu-terminal03` and its exclusive external mirror. Every input pin matches. The original saved-evidence validator passed, with receipt SHA-256 `4466242d97e36e1e9c9b72e1b8032d9a7b9a35b710b54af75cfbaab8a1e572dd`. Runtime, source, model identities and clean shutdown are recorded. No GPU service, private finalization or scientific world generation occurred. Independent result review is complete with zero required findings. The strict GPU handoff is `cpu-acceptance03/acceptance.json`, SHA-256 `11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f`. Its 467-file CPU inventory includes saved validation and excludes only the separately bound 402-file campaign. Original disposition and raw inventories remain unchanged.

The supervisor recorded 112.9761 seconds elapsed and 29.3458 process CPU-seconds. Its child and cleanup recorded 48.1176 CPU-seconds, 71.3199 seconds elapsed and 982,171,648 peak RSS bytes. Slurm recorded 77.591 TotalCPU seconds and 7,862,080 KiB batch MaxRSS. These overlapping measurements must not be added. The batch peak remains close to the 7,864,320 KiB limit. There is no recorded OOM and no claim of comfortable memory headroom or whole-cycle CPU cost.

Before another submission, reconcile 34 terminal jobs and 70 submission journals. The proposed GPU reservation is at most 4,800 CPU-seconds and 600 GPU-slice-seconds. Including previous allocations and the unchanged preparation allowance, this would bring admission to 18,102 CPU-seconds and 1,947 GPU-slice-seconds, within the existing ceilings. Independent CPU acceptance and fresh packet and resource checks are required first.

## CPU03 submitted after direct-origin correction

Job35387068 was submitted once at2026-11-08T08:06:30.500036Z. Source183ce3137213dcc1e4f33855bfab7b20dd307bb0 and packetcommit70d9ee060d10ed087ebf222e1ac3d5b9836b0fd1 are pushed. PacketSHA0533b4fad722652ec63ed12850d3a3d55d1f6c391e81f7280d36561aa9e5ac13 binds the reviewed correction. Distinct source and packet reviews have zero required findings. Both controller locks,33terminaljobs,68submissionjournals, account/partition, scratch readiness and all77transferpins matched. RemoteintentSHAe261328ea3278bee7e51761844cc155b32a3094e549da21de3ec66e8cd397dbc.

Request2CPU/7680MiB/600seconds,ucb736_asc2/acpu/cpu-normal,zeroGPU,no-requeue. ROOT `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03`, CPUout`cpu03`, futureGPUout`gpu03`. Repository/external custody use`cpu03`/`ACE-LLM-intervention-campaign-cpu-20261108-03`. Only the corrected wrapper,three new-path launch contracts and packet differ among the72inputs. Same artificial worlds,methods,training/checkpoints/runtime and bounded resources. CPU02 completed/failed trajectories remain unchanged and cannot be reused as accepted cells. Full successful execution,saved validation,custody and independent CPUacceptance remain necessary before GPUlaunch.

## CPU02 failure and direct module-origin correction

Job35385876 failed1:0 after116seconds on two CPUs, adding232allocatedCPU seconds and zeroGPU. Complete899rawfiles totaling87,742,308bytes and all original input pins are preserved in exclusive `cpu-terminal02` custody. The worker staged the runtime, fitted the matched numerical ranker for six updates and completed all eight world/policy cells for the four numerical acquisition controls. Both learned-ranker cells failed. No LLM service or private finalization ran.

Both failed numerical child calls captured a legal action after ranker inference, then failed the post-call provenance check. The wrapper used `getattr(module, '__file__')`, which can return a synthetic inherited label from a PyTorch namespace. Pinned PyTorch source confirms this behavior, but the old failure omitted the offending module, so its exact first offender is unknown. The failed arms remain failed even though an action was captured internally. The outer generic deadline/failure message does not establish a timeout.

The correction reads concrete module-dictionary origins, matching the established full runtime diagnostic, and names any rejected origin in future failure evidence. Three new fabricated tests pass without numerical imports, models or responses. They cover inherited/lazy labels, real outside or malformed paths, and retained version/extension/captured-project guards. Receipt fields now distinguish numerical weights from LLM weights. Changed source and all predecessor receipts are preserved separately. Independent correction review is complete with zero required findings. The corrected source SHA-256 is850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4. A fresh exclusive packet still precedes execution.

The scheduler reports56.983TotalCPU seconds and7,862,160KiB batchMaxRSS against a7,864,320KiB allocation. The worker child/cleanup receipt reports29.406548CPU seconds and977,424,384peakRSS bytes. These scopes overlap and must not be added. Memory is close to the scheduler limit, but the recorded failure is the provenance exception and cleanup confirms no owned group remains. Keep the same bounded CPU request for this correction, monitor it, and make no comfortable-memory-headroom or OOM attribution claim. Do not reuse failed trajectories as accepted cells, tune methods, alter private endpoints or restart CPU02.

## CPU02 submitted after format correction

Job **35385876** was submitted once at **2026-11-08T07:48:10.256910Z**. Source `1752684bd1dac4c42e3a84831ff4e6fccd05e02d`, packet commit `cfc1fc08ff1ccb35465016b578c6676c22f160c9`, packet SHA-256 `663f1aa887c838a922061a834a41455d8356e89cc6c56a90a9e534798314e6d5`. Distinct failure/correction review and the new packet delta review have zero required findings. Both locks,32terminal prior jobs,66journals, exact prior accounting, account/partition and scratch readiness, and all77transferpins matched. Remote submission intent is `d15ad0d9dd798f7eda54c81227ff5a882f02e738f4246e4c007b51a7f31da934`.

The resource request remains two CPUs,7,680MiB,600seconds,`ucb736_asc2/acpu/cpu-normal`,zeroGPU,no-requeue. The same hand-written systems, numerical recipe, accepted checkpoints and runtime are used. No CPU01 responses or numerical fit existed to reuse or select on. Five input files differ: the one-line CPU wrapper correction, three launch contracts with new exclusive paths, and the packet containing their pins. CPU01 is never restarted.

New repository custody is `results/ace_llm_intervention_campaign_artificial_20261108/cpu02`, with external `ACE-LLM-intervention-campaign-cpu-20261108-02`. Remote root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_02`, input directory `inputs`, CPU output `cpu02` and fixed future GPU output `gpu02`. Complete terminal custody, saved validation and independent CPU acceptance remain pending before a GPU successor.

## CPU01 failure and exact memory-format correction

Job35385156 failed1:0 after11seconds on two CPUs, adding22allocatedCPU seconds and zeroGPU. All86rawfiles totaling10,207,946bytes and all original input pins match exclusive terminal custody under `cpu-terminal01`. The original source, packet, output and failed attempt remain unchanged.

The captured `scontrol` record matches every required job, account, partition, QOS, node/task/CPU, wall-time and requeue field. Its exact allocation is `cpu=2,mem=7.50G,node=1,billing=2`. The new wrapper accepted the equivalent strings `7680M` and `7.5G` but rejected `7.50G`. This stopped execution before archive staging, numerical training, a worker launch or any intervention. It is a formatting defect, not excess resource use or a scientific result.

The correction adds only that exact equivalent memory literal. One new saved-record fixture passes the original allocation and rejects7.49G,7.51G and8G. It used0.189266childCPU seconds, excluding preparation/review. No model, fitting, response or submission occurred during the check. The old collector is retained, with a separately reviewed v2 that recognizes the same exact Slurm spelling. Distinct failure/correction review is complete with zero required findings. Failed-attempt disposition SHA-256 is `80d94efaf8b43c0684faef5e7ab5fc7dc285fde5f8fbb88b86e0f9be09093ad5`. Do not rerun CPU01 or amend its command, source, outputs or future-service contracts.

## CPU01 submitted

Job **35385156** was submitted once at **2026-11-08T07:37:56.979376Z**. Source commit `5a2d543f29eefde455f5b74c250fbe2d52b23ef6` and packet commit `f18947ce2e579b7f3e3d2b47f4d32d8671026b8d` are fixed. CPU packet SHA-256 is `e0fd09b926539c1a764894fb39e508d4aadd50de12aaecb7273bdd72ddcf9788`. The distinct concrete packet review found zero required defects. All 72 inputs and five additional launch objects matched remote bytes. Both controller locks, 31 terminal predecessor jobs, 64 predecessor journals, renewed account and CPU memory ratio, scratch stat/df and dedicated write/read/cleanup passed. Remote submission-intent SHA-256 is `dbb4955d2ed082d91dc36f964d5e7e1cf14b455c9b6e42cf6477684064f244b6`.

The request is two CPUs, 7,680 MiB and 600 seconds on `ucb736_asc2/acpu/cpu-normal`, with zero GPUs and no requeue. It performs the one matched numerical training and the ten CPU policy cells described below. The three future model-service commands are already fixed. No private finalization or scientific descriptors are authorized by this job.

Repository custody is `results/ace_llm_intervention_campaign_artificial_20261108/cpu01`, with exclusive external original `ACE-LLM-intervention-campaign-cpu-20261108-01`. Remote root is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_01`. Complete terminal custody, saved validation and independent CPU acceptance remain mandatory before any GPU successor. Never duplicate pending, running or ambiguous work.

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

## New integration components

The following new components connect the accepted training result to the proposed test. They do not constitute a complete launch packet.

- `llm_intervention_artificial_campaign.py` builds the exact two supplied systems and eight-policy roster. It returns detached plans, fixed numerical bindings and private analytical query means without drawing a response. A distinct source review found no required changes.
- `llm_intervention_campaign_launch.py` keeps the campaign command fixed while a later supervisor supplies the authenticated node-local runtime stage. Its separate review accepts this narrow consistency check. Archive execution, the Slurm allocation, inherited environment and whole-process supervision remain outer obligations.
- `llm_intervention_campaign_training_lineage.py` binds the accepted GPU03 run to the four-update SFT and six-update final adapters. Each service must use the original checkpoint receipt hash for its adapter receipt field. The outer acceptance hash is a separate link. A distinct review found no required changes, and one new byte-only admission passed on the accepted originals without loading tensors.
- `llm_intervention_campaign_cpu_phase.py` couples one matched numerical fit, fixed final-ranker deployment, complete campaign freezing, four fabricated tokenization shape probes and all five CPU controls. It preserves failed arms and prevents them from being treated as successful CPU admission. Its independent component review found no required fixes. A real captured-runtime wrapper remains necessary.
- `llm_intervention_campaign_service_evidence.py` joins variable-length trajectories across both systems to captured requests, generations, proposals and charged returns. Its separate static review found no required fixes. Failed and unresolved sessions remain evidence-only results. Token decoding and actual source/runtime/training/scheduler admission remain outer obligations.

The CPU shape probes use labelled fabricated zero-valued histories. They are not simulator observations or bounds on every possible adaptive prompt. Actual service calls retain the original token limits and must fail without truncation or repair if a prompt exceeds them.

New launch tests use mocked `exec`, lineage tests use fabricated pins, and CPU-phase tests use fake fitting, tokenization and policy calls. Their passes do not establish target execution. The builder's test-development failures and final eleven-method pass are retained in a retrospective transcript record. Historical test source-time hashes and CPU measurements were not recorded and must not be invented. No completed component suite is repeated.

The remaining critical work is a concrete captured-source CPU/GPU launcher, independent admission of retained CPU attempts, authenticated numerical-policy child commands, exact new service freezes and resource contracts, actual CPU preparation, then one bounded GPU campaign and private finalization. The final packet must bind every new interface and its dependencies. These additions do not alter any accepted artifact or the archived paper.

### Full launcher preparation

The new CPU launcher stages the accepted runtime once, authenticates the complete training batch and accepted adapters as bytes, fits the matched numerical control and runs the five CPU policies. Each adaptive numerical call uses the same staged interpreter and captured source with its own bounded deadline. It verifies its used numerical imports without rescanning the entire archive or importing a language model on every decision. Its saved validator reads evidence without fitting, loading tensors or computing private scores.

Independent CPU-wrapper review found one input/status interface defect before execution. The corrected adapter supplies both text and binary stdin and rejects a failed CLI return before publishing completion. Thirteen initial methods, two subsequent focused methods and two new interface methods passed. The final narrow independent recheck has zero required findings remaining. All source snapshots, logs and receipts are retained under `results/ace_llm_intervention_campaign_preparation_20261108/cpu_wrapper01` and `cpu_wrapper02`.

The GPU phase and outer launcher are implemented. Three required outer-review findings are resolved: the startup timer transfers to the original overall work deadline, unresolved owned cleanup keeps the controller alive until confirmed closure or its original hard deadline, and pinned separated output paths are checked before any claim or failure write. The narrow independent recheck has zero remaining required fixes. Fifteen distinct mocked phase methods passed, plus one affected cleanup-method rerun. The wrapper's four initial methods and three new or changed methods passed across preserved source versions. These are fabricated boundary checks, not real model execution or scheduler acceptance.

The current 15 GiB request supersedes the preserved 16 GiB prototype. No previous receipt is relabelled. A new committed CPU packet and concrete review still precede the actual CPU stage. Its successful execution, exclusive complete custody and separate acceptance must precede the GPU packet. The final GPU packet is not constructed by these preparations and no campaign job has yet been submitted.

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
| Three sequential LLM services | One A100 `3g.20gb` slice, 4 requested/max 8 allocated CPUs, 15 GiB, 600 seconds |
| CPU finalization, if separately scheduled | 2 CPUs, 7,680 MiB, 300 seconds, zero GPU |

The CPU memory proposal provides room above prior near-6-GiB batch peaks for ranker/subprocess work without another CPU. GPU14 measured approximately 3.26 GB reserved GPU memory. GPU03 measured 3.278 GB reserved and 95.05% of its 12 GiB host-memory limit. The latter includes runtime staging. The campaign also keeps a host SCM/controller alive beside each model process. Request 15 GiB host memory to provide margin above that measured near-limit run. The most recent accepted partition record limits memory to 3,840 MiB per requested CPU, so 15 GiB fits four CPUs while 16 GiB would exceed that ratio. Fresh partition admission remains required. This changes neither GPU count nor the CPU/GPU time reservation. Use the new CPU timings to check the projection before launch. One sequential GPU slice is sufficient as the starting bound; no multi-GPU allocation is justified.

Proposed account is `ucb736_asc2`, CPU partition/QOS `acpu/cpu-normal`, GPU `aa100/gpu-testing`, with no requeue. Proposed exclusive output is `/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_01`. Reconcile actual current jobs, existing controller locks, journals, account and slice availability before any once-only submission.

Retain the 240-second first-verified-work limit, 90-second progress/telemetry guard, and cleanup margin inside the GPU cap. Size session admission for three services rather than copying the old two-session deadline assumption. Saved GPU14 phases measured 19.73 and 18.00 seconds for its two four-generation services. The new maximum is eight decisions per service. Proposed per-service startup/total limits are 120/180 seconds, with client startup 125 seconds, request timeout 35 seconds and cleanup 5 seconds. These are conservative integration caps, not a promise of completion. Require positive CUDA work and advancing generation records together with independent owned-GPU-process evidence for every service. Record staging, model loading, inference, adaptive CPU work and final-output timestamps. Missing utilization percentages remain unknown, not proof of idleness or high utilization. Diagnose sustained CPU-only occupancy promptly and stop only owned work when safe, preserving evidence.

The proposed caps add at most **6,600 allocated CPU-seconds and 600 GPU-slice-seconds** if all three phases use the stated scheduler bounds. Recompute cumulative admission before submission; preserve the existing preparation allowance. Compare requested versus observed allocation, memory, throughput and elapsed time after this gate before considering scientific scale-up.

## Inspected implementation anchors

- `scripts/research/llm_intervention_campaign.py`: `prepare_campaign`, `run_policy`, `finalize_campaign`.
- `scripts/research/llm_intervention_paired_episode.py`: supplied plans, charged arms, shared-initial sealing and private finalization.
- `scripts/research/llm_intervention_training_collection.py`, `llm_intervention_training_exposure.py`, `llm_intervention_training_plan.py`: complete saved-batch lineage and matched exposure.
- `scripts/research/llm_intervention_ranker_training.py`, `llm_intervention_screen_policy.py`, `llm_intervention_policy_cli.py`: matched numerical training and public inference.
- `scripts/research/llm_intervention_model_service.py`: artificial adapter scope, identity and inference-only service.
- `scripts/research/qualify_llm_intervention_model_service.py`: staging-dependent command and historical fixed-request validator.
- `scripts/research/llm_intervention_service_runtime.py`, `llm_intervention_screen_reporting.py`: fixed old CPU fixtures and complete descriptive reporting.

## Boundary before the scientific screen

Even a successful two-world campaign will not qualify every proposed scientific prompt. The proposed screen has three nodes, eight actions, eight target queries and up to32 history rows, compared with the much smaller artificial inputs here. Before realizing scientific worlds, check representative full-size public envelopes with the exact tokenizer and context limits, then measure the training and inference resource projection. Do not truncate an adaptive prompt or change an intervention menu to rescue a scored run. The32/24 identities, optimization schedule, executable source and resource plan still need a separate scientific freeze.

The dated execution entries above record observed work. Earlier design sections remain prospective where their gates have not been completed.
