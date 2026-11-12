# ACE LLM and SCM research handoff

Prepared November 12, 2026 for Patrick Cooper and the next researcher or agent working in `/Users/pat/code/ACE`.

**The 24-system comparison is complete. The next task is to diagnose the learned intervention policy and design its successor.** There is no remaining reporting job or collector needed for this comparison. Large artifacts have verified copies on CURC. Reports, selected evidence and verification records are retained locally.

The current DPO training recipe did not improve intervention efficiency. Direct SCM planning performed better. This result does not end the broader LLM research route, but it prevents presenting this recipe as a demonstrated benefit to experimentalists.

This handoff reflects repository revision `f99d37659ad4bcaaa8e00210cb2b4386c0531fc2` and the completed November 10 evidence. It does not assert a fresh November 12 SSH connection, scheduler observation or archive readback. Dated execution entries below the latest ledger entry are historical, even when they describe a job as pending or a review as open.

## Research direction and publication status

Patrick directed that the delivery paper be archived and that ACE return to the broader LLM-based research route. The intended contribution is a language policy that uses an explicit structural causal model, or SCM, to choose useful interventions and reduce experimental cost beyond strong numerical policies. The LLM must contribute to the decision rather than only explain a numerical optimizer's choice.

The delivery paper remains a supporting component study. Its source revision is `661db7ac51d3dd4c1ce2c7075073b993ac6b8241`, tagged `archive/delivery-scm-2026-11-02`. The archive contains 42 source and report files. The older page-reviewed PDF is not an export of that final archived source revision. Preserve accepted A/B/C/confirmation evidence, failed attempts and required notices. Do not resume the old submission-only workflow or repeat its completed studies.

The broader paper is **not ready for submission** under Patrick's stated requirement. We still need an independently demonstrated contribution from the LLM, decisive controls, and a concrete experimentalist task with useful outcomes. Human submission approval remains required. The open manuscript at `/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex` was not changed during the latest comparison or this handoff.

Read the [research agenda](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_agenda_2026-11-02.md) for the mechanism, literature and submission conditions, and the [paper archive record](/Users/pat/code/ACE/docs/development/guidance/delivery_paper_archive_2026-11-02.md) for the preserved version. The agenda's original proposed counts, resources and `ucb736_asc1` account are superseded by the executed protocol and renewed account described below.

## What the completed comparison establishes

The screen compared eight intervention policies on 24 synthetic development systems. All 192 policy/system runs completed with evaluable reports. No failed or unresolved comparison cells were dropped. The systems crossed three known graphs, two coefficient regimes, two known noise levels and two cost menus. Each combination had one system. All policies used the same SCM learner within a system. There was no SCM-free arm.

| Policy | Mean capped total cost to target | Ever reached target out of 24 | Met target at final state out of 24 | Mean final NMSE |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 14.250 | 21 | 20 | 0.04113 |
| Menu coverage | 12.458 | 22 | 21 | 0.01974 |
| Local information | 11.958 | 23 | 18 | 0.01800 |
| Direct SCM target risk | 11.625 | 23 | 21 | 0.01722 |
| Learned numerical ranker | 13.583 | 21 | 20 | 0.03572 |
| Frozen LLM | 13.625 | 22 | 20 | 0.02008 |
| Supervised LLM | 14.625 | 21 | 19 | 0.02708 |
| DPO LLM | 15.167 | 19 | 18 | 0.03642 |

Lower cost and error are better. Cost includes eight units of initialization and the first privately evaluated crossing of NMSE 0.04. Nonattainment receives the common cap of 32. This is a **retrospective cost endpoint**, not actual autonomous stopping or money saved. Ten systems already met the target after initialization. Keep them in the analysis. All 28 paired comparisons and nine predeclared groups are available in the report data.

DPO had lower capped cost than target risk on two systems, tied on thirteen and had higher cost on nine. Against random, the counts were three lower, thirteen tied and eight higher. DPO had lower mean final error than random, so do not describe the result as a loss on every metric. These are descriptive development results without confirmatory significance claims.

The clearest behavioral finding is that DPO chose `set_x_neg`, setting X to −1, on all 576 acquired actions. The frozen model chose observation on 574 of 576 actions. The supervised model chose X = −1 on 481 of 523 actions. These counts exclude initialization and stop commands. They establish limited observed adaptation, not the cause of the behavior.

Seven policies spent all 768 available units across their 24 runs. The supervised model spent 721 and stopped early in 23 systems, including five that remained above the final error target. Total logical comparison charges were 6,097 cost units and 5,884 accepted responses. Lower spending without comparable accuracy is not an efficiency gain.

The [full results report](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md) contains methods, all policy results, cost definitions and interpretation. The [interactive comparison](/Users/pat/.cursor/projects/Users-pat-code-ACE/canvases/ACE-24-system-comparison.canvas.tsx) exposes every system, group, pair and saved curve. TypeScript and static data/display checks passed. Rendered pixel and interaction review was not completed.

## How the implemented mechanism works

At each step, the chooser receives the public target, causal graph, legal actions and costs, remaining budget, measured history, and SCM posterior means and covariances. It cannot access true mechanism parameters or private evaluation errors. Its output is a legal action identifier or stop command.

The executor charges the action and measures X, Y and Z. The SCM updates eligible nonintervened mechanisms. A clamped variable can still supply a known parent value to a downstream mechanism. The updated SCM state becomes the next chooser input.

The action menu contains observation, setting X to −1, 0 or 1, setting Y to −1, 0 or 1, and jointly setting X to −1 and Y to 1. All actions cost one in the uniform menu. Y and joint interventions cost two in the heterogeneous menu. The graphs are chains, forks and colliders with linear mechanisms and independent additive Gaussian structural noise.

The LLM is Qwen3-0.6B. Rank-eight LoRA adapters on q/v projections were trained offline on 32 separate training systems. Supervised fine-tuning used 160 updates from preassigned numerical teachers. Direct preference optimization, or DPO, used 119 updates from the fixed exposure rule applied to measured preference feedback. Preference labels compared replicated prediction-error reduction per acquisition-cost unit. All 3,584 collected preference comparisons were retained. Training collection charged 10,240 cost units for 9,145 accepted responses, separately from deployment costs.

The frozen, supervised and DPO arms used their prescribed weights throughout each development episode. **The SCM adapted online. The LLM weights did not.** Final checkpoints were fixed without development-score selection. The numerical ranker had matched example exposure, not equal architecture or compute.

The known graph, linear model, fixed action menu and constant public prose leave important proposed LLM roles untested. This screen did not evaluate translating varied scientific objectives into models, interpreting instruments, or proposing new intervention types. It also did not establish compressor or physical-laboratory transfer.

## Next research work

1. **Diagnose state sensitivity before training again.** Compare frozen, supervised and DPO command scores on saved training examples and separately declared artificial public states. Vary posterior uncertainty, target queries, costs and remaining budget. Check whether command rankings change appropriately and whether different inputs survive serialization and tokenization. Use the observed repeated action as the question to explain, not as permission to tune against the 24 development outcomes.
2. **Separate candidate explanations.** Examine teacher action frequencies, preference margins, chosen/rejected exposure, state-conditioned likelihoods and the scoring/decoding path. Training imbalance, objective mismatch and weak state sensitivity are hypotheses. Preserve evidence that contradicts them. Do not assert a cause from the action counts alone.
3. **Define a successor only after diagnosis.** Keep target-risk planning, the numerical ranker, frozen LLM and supervised LLM as strong controls. A change to representation, supervision or optimization needs an explicit testable rationale. Do not hard-code winners for particular nodes or regimes.
4. **Freeze fresh evaluation before scoring.** Declare new system namespaces, methods, model and runtime pins, endpoints, failure treatment and resource limits. Use artificial integration qualification before a scientific run. The completed 24 systems remain development evidence. They are not a fresh confirmation set for a revised recipe.
5. **Test a role that matters to experimentalists.** After a viable mechanism emerges, add varied objectives or a validated engineering environment with legal actuators, measured costs and a useful outcome. The compressor presentation is illustrative. A real executable compressor model and transfer evaluation remain future work.

The immediate deliverable should be a short diagnosis and a justified next-experiment proposal. Scaling the current DPO recipe or adding compute merely because budget remains is not the next step. No successor experiment was launched by the completed report or by this handoff.

Earlier October TabPFN experiments concern mechanism prediction on fixed histories, not an LLM intervention policy. Their conditional benefits, local harms and invalid language proposals remain preserved in the agenda's linked records. Do not merge those results with the new policy screen or recycle selected historical marketing numbers as confirmation.

## Evidence locations and storage rules

The authoritative local entry points are:

| Purpose | Location |
| --- | --- |
| Latest execution state and historical attempts | [Execution ledger](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md), newest entry first |
| Full reader report | [24-system results](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md) |
| Complete finalization evidence root | [Finalization directory](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108) |
| Accepted execution, runtime and accounting | [Execution acceptance](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/acceptance.json) and [review](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/review.md) |
| Original 242 reports and retrieval verification | [Report retrieval](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/report-retrieval01/handoff.json) |
| CSVs, display data, extraction and final prose | [Reporting directory](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01) and [receipt](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/receipt.json) |
| Independent numerical checks and their scope | [Numerical review](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/numerical-review01/review.md) |
| Storage amendment and backup history | [CURC custody guidance](/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_remote_custody_2026-11-09.md) |

Exclusive local copies are under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/`. The final report copy is `ACE-LLM-intervention-24-system-reporting-20261110-01`. The original report retrieval copy is `ACE-LLM-intervention-scientific-finalization-reports-20261110-01`. These are separate from the repository copies.

The full finalization archive is on CURC at `/projects/paco0228/ACE_evidence/ace-llm-finalization-20261110-01/raw.tar`. Its recorded size is 3,390,535,680 bytes and SHA256 is `413c38d990cd7f6a807d8280b8c44ba181da6b59c651faf614643d06f56ea3f9`. It preserves 107,284 original files. Source-before, source-after and archive-readback inventories matched. Separate upstream archives retain large runtime, model and training artifacts. The finalization archive alone is not the complete upstream runtime/model store.

Keep large artifacts on CURC and use the reviewed selective collector. It authenticates full remote membership and bytes, retains backup verification, and downloads the declared local subset. All 242 report payloads total 4,403,819 bytes, but local verification packages are larger. The former 85 GiB Mac free-space planning target is removed. Recheck actual local capacity before future transfers rather than treating this as a zero-space workflow.

The projects copies are verified research copies, not a claim of institutional backup service. Before relying on them in a new operation, check current access, retention and integrity. Preserve scratch originals, failures and interrupted transfers until their disposition is explicitly justified. Never overwrite an existing evidence directory or rerun a scientific job to repair a collection failure.

The finalizer execution review and the report arithmetic review have different scopes. Original intermediate flags such as `report_arithmetic_accepted: false` remain unchanged in historical receipts. The later numerical review and reporting receipt establish completion of that separate stage. Do not rewrite old receipts or treat a storage receipt as scientific qualification.

## Verification and resource accounting

Independent review reconstructed all 192 saved reports, 4,540 curve points, 28 paired comparisons and nine predeclared groups. It checked the CSVs and static canvas logic with 21,045 assertions and zero required fixes remaining. Saved evaluator errors were arithmetic inputs. This was not an independent rerun of private ground truth or model inference. Final prose differs from the reviewed draft only in recorded completion-status text and an evidence link.

The final reporting job was `35533604`, completed `0:0` on November 10. It used two CPUs for 1,596 seconds, or 3,192 allocated CPU seconds, and no GPU. Its batch peak RSS was about 7,677.7 MiB against a 7,680 MiB request. Preserve that near-limit observation when sizing future reporting work.

The accepted cycle reconciliation includes 49 jobs and 100 submission journals, including failures. It records 40,265 allocated CPU seconds and 4,458 A100 MIG slice seconds. A single 3,600-second preparation allowance brings CPU admission accounting to 43,865 of 57,600 seconds. GPU admission is 4,458 of 7,200 slice seconds. No reservation remained outstanding at closeout. A 20 GB MIG slice-hour is not a full A100 GPU-hour. Allocation, process CPU, wall time and reservations overlap and must not be added as separate costs. Whole-cycle CPU remains unknown because some development and review overhead is unmetered.

For new CURC work, use the `curc-access` skill and a fresh connection check. The adopted account is `ucb736_asc2`, replacing expired `ucb736_asc1`. Confirm the actual association, permitted resources, filesystem readiness and quota before submitting. Routine diagnosed operational recovery has blanket approval within the agreed scope and ceiling. Do not manufacture another approval request for a small correction, and do not infer an unlimited new research allocation from the renewal or this handoff.

Before any mutation, reconcile both `ace-delivery-controller.lock` and `ace-foundation-controller.lock` under the external evidence base, plus current submission journals and scheduler identities. Never duplicate a queued or running job, retry an ambiguous submission, cancel unrelated work, or launch old prepared packets as fresh experiments. Existing launch scripts are tied to their original exclusive directories and evidence.

Prepare downloads, data, tokenization and verification on CPUs where practical. Any new GPU stage needs measured memory and runtime justification, the smallest viable allocation, and at least two independent utilization signals alongside task progress. Preserve CPU preparation, model loading, GPU work and output timestamps. Diagnose stalled GPU work before repeating or scaling. No paid APIs, installations or unrelated environment changes are part of this handoff.

## Code entry points and safe restart

Use these modules to orient the next diagnosis. Actual executed bytes are the pinned source copies referenced by the scientific freeze and accepted packets, not simply whatever a future checkout contains.

- [World and charged response adapter](/Users/pat/code/ACE/scripts/research/llm_intervention_world.py), [SCM learner](/Users/pat/code/ACE/scripts/research/llm_intervention_scm.py), and [episode loop](/Users/pat/code/ACE/scripts/research/llm_intervention_episode.py).
- [Target-risk planner](/Users/pat/code/ACE/scripts/research/llm_intervention_target_risk.py), [screen policy wiring](/Users/pat/code/ACE/scripts/research/llm_intervention_screen_policy.py), and [language policy](/Users/pat/code/ACE/scripts/research/llm_intervention_language_policy.py).
- [Training engine](/Users/pat/code/ACE/scripts/research/llm_intervention_training_engine.py), [training exposure](/Users/pat/code/ACE/scripts/research/llm_intervention_training_exposure.py), and [training plan](/Users/pat/code/ACE/scripts/research/llm_intervention_training_plan.py).
- [Accepted finalizer collector source](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/terminal01/pull_terminal.py) and [remote custody helper](/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/terminal01/remote_custody.py). Read their source and pins before adapting them. Do not invoke their completed attempts again.

The frozen scientific source is `a9346022f145cc29439325b6f291597b2e6ec037`. The scientific freeze SHA256 is `15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d`. The final descriptive reporting receipt SHA256 is `11e6c378598dcbc739403b7a9f53c6ea86e850fdc0e0d4ef898aab8b35e58583`. These identify the completed study and its reporting, not a future recipe.

Start by reading the latest ledger entry and full results report. Inspect saved training supervision and the input/scoring path for the proposed sensitivity diagnosis. Record a concrete protocol before any new model execution. Preserve all completed tests and reviews without repeating them as busywork. Any new result needs its own source, inputs, costs and review scope.

At this handoff, `.DS_Store`, `presentations/ACE_theory_and_evidence/ACE_mechanisms_and_evidence_v9.pptx`, prior fixture directories and untracked `stream-attempt/archive.tar` files are preexisting unrelated or preserved material. Do not stage, remove or normalize them. The presentation is not the current scientific evidence record. Keep prose clear, use intervention consistently, avoid semicolon chains and em dashes, and distinguish a recorded experiment from an archived paper version.
