# Operations judgment of the five agenda proposals

Lens: principal investigator and research engineer who must execute the agenda in `/Users/pat/code/ACE` under the standing rules. Written November 12, 2026. Read-only: nothing under the repository was changed and nothing was executed beyond `wc`, `grep`, `sed`, `ls` and `sysctl`.

## 0. Facts I checked beyond the brief

These are the facts my judgments depend on. Each was read in a file or measured read-only.

| Fact | Source |
| --- | --- |
| The Mac has 18 CPU cores and 36 GiB of memory. | `sysctl hw.ncpu hw.memsize` |
| A vectorised re-implementation scored all 4,203 unique development states at 16,384 draws (plus five 256-draw estimates) in 499 s, single process. Reference scores for all 8 actions at all 4,371 decisions are already exported. | `w1/deployment-forensics-verify/v09.log`, `v09_reference_scores.py` (no pool), `decisions_verifier.csv` |
| Scratch replays already reproduce the saved artifacts: 560/560 target-risk decisions to 1.1e-16, saved curves to 1.2e-14, saved public states to 2.9e-14, NMSE at all 4,371 decision states to 2.2e-16. They are outside project custody. | `w1/deployment-forensics-verify.md` items 3-4, `w1/numerics-protocol.md` section 0 |
| The core mechanism modules total 4,255 lines across 16 files. | `wc -l scripts/research/llm_intervention_{training_engine,learning,language_policy,paired_episode,target_risk,scm,world,task,screen_protocol,screen_policy,cost_curves,episode_seal,screen_reporting,ranker,grammar,canonical_scoring}.py` |
| `run_episode` accepts only a `ProcessPolicy`, so every policy decision is an OS process today. | `llm_intervention_episode.py:69-79` |
| `choose_target_risk` rejects more than 256 draws and gives each action its own RNG stream (no common random numbers). | `llm_intervention_target_risk.py:102-103, 115-116` |
| `prepare_plan` allows only artificial or development scope, at most 8 policies, requires distinct arm measurement namespaces, and the task helper requires equal query weights. `cost_curves` also requires equal weights. | `llm_intervention_paired_episode.py:69-70, 77-78, 94-95, 114-117`; `llm_intervention_cost_curves.py:111-112` |
| The README still advertises the invalidated gain at lines 14, 114 and 116. | `grep README.md` |
| The learner is built on the world's graph, and the campaign assembler rejects a world graph that differs from the public design. A sparse-truth-on-supergraph class therefore needs either exact-zero coefficients in the world schema or a learner graph decoupled from the world graph in sealed modules. | `llm_intervention_paired_episode.py:85, 177`; `llm_intervention_cost_curves.py:119`; `llm_intervention_scientific_campaign.py:158-159` |
| The training engine admits two named recipes, has no scheduler or accumulation, no listwise loss for the LLM, one `supervised_action_id` per record, and the worker hard-codes one SFT pass. No `peft` or `accelerate`, LoRA is torch-only. | `w1/training.md` section 9; `w1/ledger-ops.md` (b) item 3 |
| GPU: `gpu-testing` has a 1 h wall and one 20 GB MIG slice per user, the slice is rejected under `gpu-normal`, resumed optimization across jobs is unqualified. A full A100-40GB bills 108.6 against 54.3 for the slice. `gpu-normal` caps jobs at 24 h, and an H200 route exists (`ah200`, `gpu-long`). | `w1/ledger-ops.md` (e) scheduler facts and (g); project CURC memory `reference_curc_ops.md` |
| `causalchamber` 0.2.8 declares Python >= 3.6, so it runs under the local Python 3.9. The Remote Lab client 1.0.0rc1 needs Python 3.11 (a second environment). pyCycle needs OpenMDAO >= 3.10, Python 3.9 compatibility unverified. | `w1/literature-benchmarks.md` lines 15, 40, 67 |
| paper-strategy E3 costs a run at 4,000 states while its data paragraph gives about 9,600 states. | `w2/paper-strategy.md` lines 235, 294, 685 |

## 1. Bottom line

Compute is not the binding constraint. The CURC account holds up to 450,000 SUs and the Mac has 18 cores, so every compute figure in all five proposals is affordable once Patrick sets a ceiling. The binding constraints are calendar, review bandwidth, engineering scope and Patrick's decisions. In November the scientific kernel was about 2.4 hours inside about 39 active hours, 21 of 49 jobs failed, half of the failed submissions were the custody harness rejecting a valid environment, and half of the 49,473 lines of new code were workers, services and validators. Any agenda that runs development work through that harness, or spends weeks on new families before the first decisive evidence, will not finish before any 2027 deadline.

So I rank experiments by decision value per engineering day. The cheapest decisive items are all CPU-only: promote the already validated scratch replays into one reviewed in-process bench, audit the endpoint on saved data, run a numerical learnability ladder, and run an LLM-independent headroom audit of at most three cheap classes. They decide whether a direct LLM policy is learnable in principle and whether any class admits an advantage test, before a single GPU hour is spent. The repaired LLM imitation gate comes next and is the only GPU-heavy item before Gate B.

## 2. Scores

| Designer | Score | One-line verdict |
| --- | ---: | --- |
| paper-strategy | 8.0 | Best spine against dates, kill rules that only narrow, LLM-independent admission with a parser condition. Gate A date and E3 cost are optimistic. |
| policy-repair | 7.5 | Most concretely costed and most of its unit costs check out. Spends weeks 1-5 in the class where it admits no advantage is possible, and E3 is low value per effort. |
| evaluation-statistics | 7.0 | Strongest measurement layer (CRN, Rao-Blackwellized endpoint, A/A and planted-effect validation, TOST). Not a full agenda and its iteration and seed counts are heavy. |
| advantage-theory | 6.0 | Best theory and a principled headroom atlas. Ten-control suite and 60-cell atlas are underestimated, and E10 drops two required controls. |
| experimentalist-task | 4.5 | Only proposal that treats submission requirement 4 as a design problem, with excellent hardware discipline. Largest engineering, most approvals, longest inferential chain, roughly double its own calendar. |

## 3. Cross-examination

### policy-repair

Cost realism. The bases check out where I could test them. 0.119 s per state at 16,384 draws is the logged 499 s over 4,203 states, single process, so labelling the 166k-state E4 corpus is about 5.5 CPU-hours, or roughly 20 to 30 minutes on 18 cores. Rank-16 LoRA on all seven projections of Qwen3-0.6B is 10.09M parameters, as stated. E1's planner already exists in scratch (`vtr.py`, 147 lines), so its real cost is custody promotion and review, not arithmetic.

Underestimates and hidden dependencies.
- E2 is a 500-line harness plus a donor-matched edit generator for 12 edit types, 16,400 forwards of 2,000 to 4,000 tokens at an unmeasured MPS speed. The census and four edit types carry most of the evidence.
- E3 needs the new engine (800 to 1,200 lines) before it can run, and its value is retrospective attribution on 128 in-sample states that every downstream experiment supersedes. The roster-order single-pass cells reproduce the old recipe on the old states, which is close to the "do not rerun" rule. Its own main risk says every cell may fail out of world.
- E5 runs of 0.34 to 1.4 h do not fit one `gpu-testing` slice job with checkpoint checks. It says so, but its staging still budgets in slice-hours.
- E8's 2 to 5 days per class assumes the in-process bench and CRN planner exist. Its classes (i) to (iii) and the privileged teacher for E9 are underspecified.

Spine. Weeks 1 to 5 sit almost entirely in the three-node known-graph class, where by its own statement the planner is the ceiling and no advantage can be shown. Up to 35 GPU-hours go to E5 to E7 before the advantage track (E8 to E10) produces a class. The diagnostic plus learnability paper it targets for ICML is realistic. The advantage branch is late.

Rules. D4 (probe on development prompts) selects an escalation arm in E7. Probes that drive method choices should use training-world states only, which E2 already proposes for its base-only probe.

Keep: E1, E4, lean E5 with gates G1 to G8, E6 as a trigger only, the E8 selection rule, and the theory items (trie-greedy lemma, state-blind collapse lemma, label-information bound, exact single-step gradient, Gibbs soft-target bound).

### advantage-theory

Strengths. T1 (zero value of adaptivity in the design-only subclass), T4 (equal-information ceiling), T5 (open-loop lower bound), T6 (design value of side information) and T7 (shuffled-feedback power) make class selection principled and independent of LLM performance. E5's synthetic policies of known capture are a cheap and strong guard on the analysis pipeline.

Underestimates and hidden dependencies.
- E3 promises ten controls in 2 to 3 engineer-weeks. A GO-CBED-style transformer with a 1x, 4x, 16x budget sweep is a project of its own, and the public GO-CBED code needs network, download and install approval. The law-aware spike-and-slab planner, the reliability-aware text planner, the rule parser and the semantic-feature policy are each new, reviewed code.
- E4's best class, C2 (sparse truth on a known supergraph), runs into a verified constraint: the learner is built on the world graph and the assembler rejects a differing graph (`paired_episode.py:85, 177`, `cost_curves.py:119`, `scientific_campaign.py:158-159`). It needs exact-zero coefficients in the world schema (one scalar coefficient SD today) or a decoupled learner graph in sealed modules. That is a medium change on the critical path.
- About 60 atlas cells at 48 systems each is more than class selection needs.
- E8 needs nonlinear mechanisms and a redefined estimand (medium). E9 has low payoff by its own statement.
- E7 says the recipe "comes from the training workstream", which leaves the hardest engineering item unspecified.

Conflicts with the standing rules.
- E10's arm list lacks the frozen LLM and the supervised LLM, which the brief requires as controls.
- E1's decision rules (iii) and (iv) set successor class parameters (n_init, boundary actions) from private truth of the 24 development systems. These are not LLM outcomes, but they are development-system outcomes. E1 should be descriptive, and class parameters should be set on fresh development-namespace systems in E4.

Calendar: 6 to 10 weeks to E7 and about 13 to E10, by its own estimate. Realistic for NeurIPS 2027 only if the control suite is cut.

### experimentalist-task

Strengths. It is the only proposal that designs for submission requirement 4. It has the best hardware discipline: a shared fixed decision rule so arms differ only in interventions, billed chamber time, pool-based replay with no interpolation, a digital-twin rehearsal before any spend, and spend released only after simulator evidence. Its numerical student with text embeddings (N9), anonymized and misleading text arms, and compute-matched rollout are good controls.

Cost realism. This is the most underestimated proposal.
- E1 (8 to 10 working days) needs a basis-function Bayesian learner. That breaks the closed-form `structural_moments` recursion that target risk relies on, so the planner needs Monte Carlo composition. It also needs state-dependent switching costs in the executor (medium per the numerics note), shared CRN namespaces, a truth-conditional evaluator, raised roster and world limits, and new sealer and cost-curve versions. I expect 3 to 5 weeks with review.
- E3 needs about 16 device classes with canonical forms, parameter ranges and units, plus 48 independently written descriptions frozen before any model run, in 5 to 7 days. This is weeks of authoring and a large surface of researcher degrees of freedom.
- E5 assumes 0.1 to 0.25 s per SFT step on prompts of about 1,500 tokens with no measurement. The only measured figure is about 0.6 s at about 3,000 tokens.
- E7 depends on pyCycle and OpenMDAO, unprototyped, with Python 3.9 compatibility unverified.

Spine. It builds three new families and a new generator before any evidence that the repaired training stack can produce a state-dependent policy at all. A positive E6 needs all of these to hold: headroom on W, semantic information in the generator, learnability, in-distribution gates, and zero-shot cross-family transfer. The chance of reaching E9 is small, so an early Remote Lab sign-up (E2) costs Patrick's time and terms acceptance (jointly owned data, use of his name for marketing, Swiss law) for little decision value. Its 9 to 10 weeks is closer to 16 to 20 in practice.

Keep: the wind-tunnel operating-point idea as a transfer test using the knowingly misspecified linear learner, the hardware protocol (twin, replay, spend gate), the shared decision rule, N9, and the misleading and anonymized text arms.

### evaluation-statistics

Strengths. E1 (endpoint non-favorability audit on saved data, under 1 CPU-hour) is the cheapest way to make an endpoint change defensible under the no-favorable-replacement rule. E2's A/A twin, planted-effect arm, CRN variance-reduction factor and evaluator unbiasedness check catch harness bugs before any policy is scored. T1 (Rao-Blackwellized evaluation) and T2 (threshold averaging) are useful theory.

Underestimates and hidden dependencies.
- E2 is effectively the shared bench plus evaluator plus analysis package. 1.5 to 3 weeks is fair, but every other proposal assumes the same component exists.
- E5 allows 3 iterations x 3 recipes x 3 seeds, so up to 27 LLM trainings plus students.
- E6 carries more than 15 arms with 5 seeds each for the LLM and the student. That is far beyond the 8-policy roster and a large inference load.
- ERB is closed form in linear-Gaussian classes with known laws, but text-hint classes may need Monte Carlo (flagged by the designer).

Gap. It brings no class and no recipe of its own. It is a measurement layer that the final agenda must adopt, not a spine.

### paper-strategy

Strengths. It lays out the best calendar against real dates. Its kill criteria (K1 to K6) only narrow the claim. The E1 admission rule is hashed before any LLM touches a new class and includes the condition that a rule parser must leave at least 10% of the value. It corrects the public record first, runs transfer through the misspecified linear learner on the wind-tunnel simulator (cheap), and puts hardware strictly after simulator evidence.

Underestimates and hidden dependencies.
- Gate A by December 5 needs, in about 2.5 weeks: K1 and K2 protocol modules, informed-prior learner support with new sealer and cost-curve versions, the CRN change, a truth-conditional evaluator, a rollout planner, the roster lift, three prose families and a parser.
- E3 costs a run at 4,000 states while its data is about 9,600 states. At its own 5,000 tokens per second and 1.3x for rank 16, a run is about 1.0 to 1.3 slice-hours, so it no longer fits a 1-hour `gpu-testing` job with checkpoint checks.
- E4 assumes 7B LoRA training fits and finishes on a 36 GB Mac. Unmeasured.
- E9 (delivery checkpoints) needs a Duo-authenticated CURC retrieval and belongs to another paper.
- Prose family D and the parser have no named author or effort budget. A weak parser inflates condition (b), and model-written prose risks being parser-solvable.

## 4. Evidence per unit of effort

| Rank | Item (source proposals) | Engineering and compute | What it decides |
| ---: | --- | --- | --- |
| 1 | Promote the validated scratch replays into one reviewed in-process bench with a fixed acceptance test (all) | 3 to 5 days, CPU minutes | Prerequisite for every numerical and closed-loop item. Removes the 48 s per episode process artefact. |
| 2 | Endpoint audit and descriptive value-gap on saved data (evaluation-statistics E1, advantage-theory E1 descriptive) | 1 to 3 days, under 2 CPU-hours | Makes the endpoint change non-favorable by construction and quantifies "no room in the completed class". |
| 3 | Numerical learnability ladder in the canonical class (policy-repair E4, paper-strategy N1/N2) | 1 to 1.5 weeks, under 15 CPU-hours local | Whether the planner's state-to-action mapping is learnable from public state at all, N*, and the matched numerical students needed everywhere. |
| 4 | LLM-independent headroom audit of at most three cheap classes plus instrument validation (paper-strategy E1, evaluation-statistics E2/E3, advantage-theory E4/E5, policy-repair E8) | 2 to 3 weeks, 20 to 100 CPU-hours local | Whether any advantage test exists. If none qualifies, the program becomes a diagnostic paper without further GPU spend. |
| 5 | Lean forensic scoring of the saved adapters (policy-repair E2, paper-strategy E2, evaluation-statistics E4a) | 3 to 4 days, 1 to 4 local GPU-hours | The anatomy section: decoding effect, position and identifier prior, posterior blindness. |
| 6 | Repaired LLM imitation gate with matched students and knockouts (policy-repair E5, paper-strategy E3) | 2 to 3 weeks including the engine, about 5 to 15 GPU-hours or local nights | Whether a 0.6B model can read an SCM state when supervision, exposure and interface are fixed. |
| 7 | Zero-shot text-reading probe in the admitted class (advantage-theory E6) | 2 to 3 days, 4 to 10 local hours | Whether text-reading has to be taught, and at which size. |
| 8 | Development pilot in the admitted class (paper-strategy E5, advantage-theory E7) | about 3 weeks, 5 to 15 GPU-hours | Promotion or futility. |
| 9 | Registered confirmation with contribution controls (all) | 2 to 3 weeks, 2 to 6 GPU-hours | The claim. |
| 10 | Wind-tunnel simulator transfer with the misspecified linear learner (paper-strategy E7) | 1.5 to 2 weeks, CPU plus 1 to 2 GPU-hours | Transfer beyond the generator. |

Low value per effort for this cycle: the 128-state attribution factorial, the ten-control suite, the misspecification class with a redefined estimand, the non-myopic DP class, the procedural device library, pyCycle, early Remote Lab use, delivery checkpoint diagnostics, and 7B LoRA training.

## 5. What the final agenda must include

1. A shared in-process bench as experiment zero. It promotes the scratch replay and CRN planner into reviewed project code: a callable-policy episode loop, a planner with more than 256 draws and common random numbers across candidates, a rollout planner, the conjugate learner, and a truth-conditional evaluator. Its acceptance test is fixed now: reproduce all 560 saved target-risk receipts, all saved CPU-arm public states and all 4,371 decision-state NMSEs from saved artifacts. Policy code may receive only the public envelope and may not import the evaluator or the private world (static check plus test). Reason: every proposal needs it, and the existing harness cannot express it (`episode.py:78`, `target_risk.py:102-103`, `paired_episode.py:94-95, 114-117`).
2. One batched decision memo to Patrick before work starts, with conditional pre-approved tranches, and a two-tier custody regime (development light, confirmation full). Reason: human-gated waits (23.5 h for one storage decision) and harness guards dominated the last cycle.
3. A measured local pilot as soon as local inference is approved: forward scoring speed and fidelity, plus about 50 LoRA steps on compact prompts. It decides whether training runs locally, on a full GPU (`gpu-normal` A100 or `ah200` H200 under `gpu-long`), or with a qualified resume on the slice. Reason: every GPU figure in the five proposals is unmeasured for compact prompts and MPS.
4. The endpoint non-favorability audit on saved data and a descriptive value-gap analysis of the completed class. Reason: they make the endpoint change and the "no headroom" claim defensible at almost no cost.
5. The numerical learnability ladder in the canonical class (coarse N0, converged ranker N1, permutation-equivariant N2, derived summaries N3, planner statistics N4 as an upper-bound control only) at 32 to 2,048 worlds on planner relabels. Reason: the highest decision value per day, CPU only, and it sets the LLM's data volume and representation without any LLM output.
6. The LLM-independent headroom audit of at most three classes that need only small or medium code changes, with the admission rule, the parser condition, the primary endpoint and instrument validation (A/A, planted effect, known-capture mixtures) hashed before any LLM touches a new class. A fresh-namespace replication of the completed class is the negative control. Reason: it is the only legitimate route to a new benchmark under the standing rules.
7. A lean forensic census of the saved adapters with a 99% fidelity gate on saved completions: 1,858 prompts x 3 weight sets, all legal-command scores, trie-greedy against whole-command argmax, menu and identifier permutations, posterior swaps and covariance scaling on about 240 to 300 declared artificial states, and a hidden-state probe. Reason: it is cheap, it is consensus across three proposals, and it supplies the anatomy section with direct evidence.
8. One repaired LLM imitation gate in the canonical class: one recipe (planner soft targets, compact rounded state, permuted single-token codes, argmax over legal codes, shuffled minibatches, rank-16 all-linear LoRA), three seeds, gates G1 to G8 as mandatory acceptance, matched N1 and N2 students on identical states, four leave-one-fix-out knockouts (codes, compact state, soft targets, volume), and one old-recipe anchor cell on the new engine. Reason: it tests the training stack and numeric-state reading before any text class, and the knockouts replace the 128-state factorial at a fraction of its cost.
9. A fixed ruling, before the development pilot, on which LLM roles count toward the submission condition (direct policy, compiler plus planner, adjudicator over planner candidates), and whether a benefit from information access counts. Reason: it decides which arm is primary and cannot be decided after results.
10. A development pilot and one registered confirmation in the admitted class. The confirmation keeps target risk, the converged numerical student as the ranker successor, the frozen LLM and the supervised LLM as controls, plus a parser written blind, a numerical student with text embeddings, and text-removed, misleading-text and shuffled-feedback ablations. It uses Holm across at most three primary contrasts, a TOST null branch, a failure ledger, an independent analyst and 10% re-execution.
11. Theory written in Phase 0: zero value of adaptivity in the design-only subclass with a small enumeration check, the equal-information ceiling and gap decomposition, the trie-greedy decoding lemma, the replicate tie-rule label-noise bound, the state-blind collapse lemma, and endpoint efficiency. Reason: these hold in every branch and cost no compute.
12. Public-record correction (README lines 14, 114 and 116, the arXiv preprint, the shared deck, the stale PDF) before any new submission or talk. This is an ask, not an experiment.

## 6. What must be dropped or changed

1. policy-repair E3 (2x2x2 factorial on the 128 training states, 16 LLM runs, 24 CPU runs): replace it with the knockouts in item 8 plus one old-recipe anchor cell. Reason: in-sample attribution of a collapse the brief already explains, near the rerun boundary, low value per GPU-hour.
2. policy-repair E2: cut the 12-edit donor-matched battery to four edit types on about 240 to 300 states. Run any probe that selects a method on training-world states only.
3. experimentalist-task E1/E3 basis-function learner and the 16-class semantic device generator: defer. Use the wind tunnel as a transfer test with the misspecified linear learner after confirmation. Reason: weeks of new learner and authoring work before any learnability evidence, and the basis learner breaks closed-form target-risk planning.
4. experimentalist-task E7 (pyCycle) and E10 (second unit): drop for this cycle. E2 (early Remote Lab calibration): defer until simulator transfer holds.
5. advantage-theory E3: build only the controls the admitted class needs. The in-house N2 student is the amortized numerical control on the critical path. Move GO-CBED (download and install approval) to a "should" for the final paper.
6. advantage-theory E8 and E9: drop for this cycle.
7. advantage-theory E1: make it descriptive. Set successor class parameters on fresh development-namespace systems, not on the 24 systems' private truth.
8. advantage-theory E10: add the frozen LLM, the supervised LLM and the ranker successor (standing-rule controls).
9. evaluation-statistics E5: at most 2 iterations x 2 recipes x 3 seeds. Five seeds only for the frozen recipe.
10. paper-strategy E3: recompute the cost at about 9,600 states (about 1.0 to 1.3 slice-hours per run) and plan for a full GPU or local runs. paper-strategy E4: escalate to the cached Qwen2.5-1.5B first. 7B stays frozen-probe only unless the pilot shows local 7B training is feasible.
11. paper-strategy E9 (delivery checkpoints): move out of this agenda into a separate optional track.
12. All proposals: no development iteration goes through the full CURC custody harness. Budget GPU in billing units of the queue actually used, not in slice-hours, whenever a run exceeds about 45 minutes. Replace the "22 to 35 core-hours for 100 to 150 systems" projection, which assumes the per-decision process model, with measured in-process costs after experiment zero.

## 7. What all five missed

1. No proposal budgets the shared in-process bench once, as a separately accepted component, even though every one depends on it. Validated scratch implementations already exist and only need promotion and review.
2. No proposal fixes a proportionate custody and review regime before work starts. Without one, the November ratio of overhead to science recurs in every experiment. I propose two tiers. Development: a committed protocol and namespace declaration, a code and environment hash per run, append-only output directories with file-hash manifests, a failure ledger, and one reviewer at gates only. Confirmation: full registration, blinded analysis, sealed outputs, an independent analyst and 10% re-execution. I also propose a cap on new non-test harness code before Gate B (about 3,000 lines) and a rule that every new guard ships with a fixture built from saved artifacts.
3. No proposal makes a measured throughput and fidelity pilot the first approved execution, and none states the QoS consequence plainly: training runs over about 45 minutes cannot use the 1-hour slice without a qualified resume.
4. No proposal asks for conditional pre-approval of tranches tied to gates, so each gate would wait on a new human decision.
5. Human text authoring and the blind parser are on the critical path of every text-channel class, but no proposal names an author, an effort budget or a schedule.
6. No proposal includes a leakage guard for in-process execution. Process isolation gave the old harness that property for free.
7. No proposal sets a dated ICML decision point with writing in parallel, or names the minimum paper that exists at each gate.
8. Existing exports (`decisions_verifier.csv` with 16,384-draw reference scores for all 4,371 decisions, the training-label check) are recomputed rather than promoted.

## 8. Recommended spine and calendar

- **Phase 0 (Nov 12 to 19, no GPU, mostly no approvals).** Send one decision memo. Build and accept the bench. Run the saved-data endpoint audit and the descriptive value-gap analysis. Write the theory. Run the local pilot as soon as inference is approved, then the lean forensic census.
- **Phase 1 (Nov 19 to about Dec 12, CPU only).** Two parallel tracks. (A) The numerical learnability ladder in the canonical class. (B) The headroom audit of at most three cheap classes: a fresh replication of the completed class, a prose-objective class with varied query laws, and partial measurement with large cost ratios, or prose mechanism knowledge if the informed-prior change is approved. Admission rule, parser condition, endpoint and instrument validation are hashed before any LLM sees a new class.
- **Gate A (about Dec 12).** Decide N*, class admission and the frozen endpoint. If no class qualifies, LLM-advantage work stops and the diagnostic, learnability and theory paper is the output.
- **Phase 2 (Dec 12 to about Jan 8, holidays included).** The repaired imitation gate in the canonical class, run locally or on a full GPU per the pilot.
- **Gate B (about Jan 8).** Promote the recipe, run one predeclared escalation (Qwen2.5-1.5B, or the compiler role if Patrick rules it counts), or stop direct-policy work. On the same date Patrick decides whether the diagnostic paper goes to ICML (abstract Jan 14, paper Jan 22).
- **Phase 3 (Jan 8 to about Feb 5).** The development pilot in the admitted class with predeclared roles and controls, at most two iterations.
- **Phase 4 (Feb to Mar).** One registered confirmation on the reserved namespace.
- **Phase 5 (Mar to Apr).** Wind-tunnel simulator transfer. Hardware only if transfer holds and Patrick approves spend, after a digital-twin rehearsal.
- **Target.** NeurIPS 2027 (estimated May) or TMLR for the positive branch. The canonical-class negative and every failed gate stay in the paper. No gate adds a benchmark.

## 9. Consolidated asks for Patrick (one memo)

1. Venue and date, and whether a diagnostic, learnability and theory paper may go to ICML 2027 on its own.
2. Which LLM roles count toward the submission condition, and whether a benefit from information access counts.
3. The two-tier custody regime, and acceptance of a pinned local runtime as the development runtime.
4. Local forward-only inference with Qwen3-0.6B and the two saved adapters, under a written protocol with a 99% fidelity gate.
5. Local LoRA training, or GPU tranches: tranche 1 for the imitation gate (about 5 to 15 GPU-hours on a full A100 or H200, or local nights), tranche 2 conditional on Gate B (about 10 to 15), tranche 3 conditional on promotion (about 5). Also: whether `gpu-testing` may still be used for scientific runs.
6. Training-scope and development-scope namespaces for synthetic corpora and headroom audits, and retirement of the 24 development systems from all future decisions.
7. Harness changes: common random numbers across arms, more than 8 arms and 64 worlds, non-equal query weights, informed-prior learner, and lifting the confirmation scope in the three modules.
8. Use of the cached Qwen2.5-1.5B-Instruct for escalation, and Qwen2.5-7B-Instruct for frozen probes only.
9. `pip install causalchamber==0.2.8` (or vendoring its numpy-only simulator file) for Phase 5 only.
10. A named human author for held-out prose, a parser author who does not develop the LLM, and an independent analyst for the confirmation (for example the coauthor).
11. Public-record correction before any new submission or talk.
12. Later and conditional: Remote Lab sign-up, terms acceptance and a spend cap of about 300 to 400 EUR, only after simulator transfer holds.
