# Integrity audit of the five agenda proposals

Judge lens: scientific integrity. I checked each proposal against the standing rules: no favorable replacement benchmark, no tuning on the 24 development outcomes, benchmark and endpoint choices justified without reference to LLM outcomes, an LLM that contributes rather than narrates, numerical controls that get the same information, and handicaps declared in advance. I also looked for forking paths, circular endpoints, leakage, unverifiable claims, outcomes that could not be published honestly, and how each proposal handles the public preprint and README.

Date: 12 November 2026. This was a reading-only exercise. Nothing under `/Users/pat/code/ACE` was modified, and nothing was run beyond read-only greps.

---

## 0. What I verified (14 targeted reads and greps)

I read the brief in full. Then I spot-checked the facts this audit depends on.

- **Public record.** arXiv 2602.02451 (v1 2 Feb 2026, v2 18 Jun 2026) claims a "70-71% improvement". `README.md` lines 14, 114 and 116 carry the same claim (w1/literature-policy.md:322; w1/literature-benchmarks.md:177). The preprint is linked to this project only by its identical title; nobody checked the author list.
- **Provenance of Finding A** (w1/numerics-protocol.md:178-204).
  - The -0.7% +/- 1.7% figure is a difference-in-differences of realized over working-model-expected design quality.
  - The plan-once and shared-schedule searches are greedy.
  - The exact Bayes-optimal sequential policy was never computed. The analyst labels "at most a few percent" an inference.
  - The "3 to 17 systems versus 67 to 340" comparison refers to the learner's public posterior risk, not to decision-level regret (numerics-protocol.md:241-249).
- **Forensic assets.** Both adapters hash-match the deployed pins. The base Qwen3-0.6B pin (`model_sha256 052d8c0b...`) has not been verified (w1/deployment-forensics-verify.md:48). There are 4,203 unique development states.
- **Namespaces.** `confirmation:[202611071000,202611072000)` was reserved under the old protocol, and using it requires a registered frozen design (w1/ledger-ops.md:100,110). No common random numbers exist across arms (`paired_episode.py:114-117`; w1/numerics-protocol.md:60,302).
- **Remote Lab.**
  - It is an external paid service, and the handoff scopes out paid APIs (w1/literature-benchmarks.md:60).
  - Terms v1.1 forbid automated sign-up.
  - Generated data are jointly owned, and either party may publish them.
  - The company may use the customer's name in marketing (literature-benchmarks.md:31).
- **Kriz et al.**
  - The Qwen3-0.6B check used 16 states x 16 rollouts, temperature 2 and entropy 0.02.
  - Rank-16 LoRA on seven projections is the configuration of the main Gemma-4-E4B model.
  - Alpha 32 does not appear in the notes (literature-policy.md:79).
- **Designer files.**
  - advantage-theory.md:224 (primary endpoint redraws noise), :420 ("any recipe is acceptable if it passes the gates"), :526-543 (E10 roster and hypotheses).
  - policy-repair.md:88, 210-220, 287 (planner-quality criteria and adoption decided on test worlds).
  - experimentalist-task.md:56, 92, 105, 133, 141, 192 (generic SCM, prompt contents, device library, failure treatment, W modifications, leakage risk).
  - evaluation-statistics.md:137 (non-favorability veto), :294, :435 (preprint "before submission").
  - paper-strategy.md:143-158 (parser and admission), :146 (P1 replay), :564-583 (public record).

---

## 1. Scores (integrity lens, 0 to 10)

| Designer | Score | Verdict |
| --- | ---: | --- |
| evaluation-statistics | 8.0 | The best measurement and registration discipline. Two flaws: the binding endpoint choice is made on development outcomes using a veto keyed to the old LLM arms, and in information classes the headline is a comparison with a planner that lacks the information. |
| paper-strategy | 7.5 | The only responsible handling of the public record. It requires a parser comparison for any full positive. Its primary endpoint replays designs with redrawn noise, and the parser's strength is a handicap the team controls. |
| policy-repair | 6.5 | Excellent tests of whether the model reads its state, plus parity with a numerical student. But planner-quality acceptance and adoption are decided on the test worlds, it says nothing about the preprint, and E10 has no independent review or seed policy. |
| advantage-theory | 6.0 | The best equal-information control suite and instrument calibration. But its confirmatory H1 compares only against controls that lack the text, the controls the standing rules require are missing from E10, the recipe search is unbounded, there are several classes with LLM stages, and it says nothing about the preprint. |
| experimentalist-task | 5.5 | Sound hierarchy of contrasts and hardware discipline. But "independently authored simulator" is false once the team extends it, the training library is written after reading the test simulator's source, core stages depend on a paid service, and the hardware primary analysis uses completed episodes only. |

---

## 2. Cross-cutting findings

### 2.1 Favorable replacement benchmark

All five proposals pass the basic test. Each moves LLM work out of the completed three-node class on grounds of Finding A, which was computed from numerical arms only. Each selects the next class by numerical criteria before any LLM touches it. Four residual risks remain.

- **Too many shots at a class.**
  - Advantage-theory may select two classes in E4 and also runs LLM stages in a curvature class (E8) and a budget class (E9).
  - E10 runs in "the selected class" (advantage-theory.md:519).
  - Unless one primary confirmatory class is fixed at the atlas gate, a team that sees development results in several classes can confirm in whichever looks best. That is a favorable replacement by another name.
- **Branch conditions inside selection rules.** Policy-repair's E8 rule picks the class with the largest H2 "if the hypothesized LLM contribution is semantic", and otherwise the largest max(H1, H3). The branch must be fixed before E8 runs, or the criterion can be chosen after the H values are seen.
- **Benchmark engineering before any LLM contact.** Examples:
  - experimentalist-task's levers (no initial measurement, budget 80, hatch as a decision variable) and its additions to W (fan coupling, AR(1) drift);
  - paper-strategy's held-out family C, produced by "a different template process";
  - evaluation-statistics' two rounds of class redesign.

  These are independent of the LLM and capped, so they are permissible. Every lever pulled must still be reported, and the claim must be scoped to the class as constructed.
- **Construct honesty.** A class in which the decision-relevant information exists only in prose is built to give language a channel. That is a legitimate test of a hypothesis, but three things follow:
  - the paper must say so;
  - the null in the completed class must stay in the main text;
  - the lead must rule in advance whether a benefit scoped this way meets submission condition 2 (paper-strategy ask 12; advantage-theory ask 7).

  Experimentalist-task never says that the November negative stays in the paper.

### 2.2 The 24 development outcomes and the saved November artifacts

- **Forensics on saved prompts are not tuning**, provided their outputs reach the method only through decision rules written before the forensics run. This covers the forward passes on the 1,698 deployment prompts in policy-repair E2, evaluation-statistics E4(a) and paper-strategy E2. Policy-repair already works this way: D1 to D5 are routed to fixed consequences, for example D4 decides the hidden-state action head and the compact interface. The other proposals should adopt the same form.
- **Policy-repair's retrospective "gate validity" (D5) is circular.** Its gate table already lists the November values ("Would it have caught the November adapters?"), so the thresholds were set with the outcomes in view. The paper may say the gates would have caught the collapse. It may not call that a validation. Prospective validation comes from E3 cells, some of which will collapse and some of which will not.
- **Evaluation-statistics E1 chooses the registered endpoint on the 24 development outcomes**, with a veto whenever an old LLM arm improves its rank (evaluation-statistics.md:137). The veto points in the conservative direction, but it is still a choice that depends on LLM outcomes. Worse, the old collapsed arms say nothing about whether an endpoint would favor a new policy. The binding choice should move to fresh calibration systems with numerical arms and criteria that ignore which arm wins (evaluation-statistics' own E2 has the machinery). E1 can then be published as a disclosure that the endpoint does not favor the LLM.
- **Advantage-theory E1, rules (iii) and (iv), use the 24 systems' private truth** to set the next design (initial-observation count, boundary actions in the menu). These rules do not depend on any LLM. The fresh-system atlas (E4) is still the cleaner place to make those decisions.

### 2.3 Endpoint validity and circularity

- **Regret under the planner's own score.** No proposal makes it the primary endpoint for an advantage claim. Policy-repair uses planner regret and working-model J/J_TR only for imitation claims and says so. That is legitimate, because imitation is what it claims.
- **Design quality with redrawn noise is not unbiased for adaptive policies.**
  - Advantage-theory's primary endpoint (advantage-theory.md:224) and paper-strategy's P1 (paper-strategy.md:146) score each realized action sequence by its expected error under the true parameters with freshly drawn noise.
  - For an adaptive policy, the design depends on the noise the policy actually saw. In general E[g(d(w), w')] is not E[g(d(w), w)].
  - Two biases follow. A targeted re-measurement after an outlier loses its value in replay, which biases against adaptation. Stopping after a lucky draw looks better in replay than it was.
  - The sign of the net bias is unknown, and its size is the noise-chasing term that advantage-theory itself lists as an uncertainty (line 642). Classes are being selected precisely for headroom in adaptation, so the bias is not negligible there.
  - Two endpoints are unbiased. One is the Rao-Blackwellized expectation, given the data actually observed, under the generating law (evaluation-statistics T1, by the tower property). The other is realized error against the truth, with replicates.
  - Redrawn-noise design quality can remain a secondary endpoint. It can be primary only after a registered calibration against realized error in the selected class.
- **The learner's posterior risk is biased** whenever the policy has information the learner's prior lacks, or the prior is shifted (evaluation-statistics T1(b)). Twelve of the 24 development systems are shifted. Posterior risk may serve as a stop rule and as a diagnostic. It must never be a confirmatory endpoint in an information class or a misspecified class.
- **Deferred endpoints.** Policy-repair's E10 leaves its primary endpoint to E8's power analysis. That is acceptable only if the endpoint is frozen at the E8 gate, before any LLM is trained in the class.

### 2.4 LLM contribution, narration and equal information

- **Planner scores.** All five keep per-action planner scores out of the primary LLM input. Good.
- **The equal-information comparison is the real gap.**
  - In an information class, the uninformed planner lacks the text by construction, and the headroom audit has already shown numerically that the text is valuable.
  - An LLM that beats only that planner has shown that it extracts some information from the text, which a rule parser might also do.
  - Advantage-theory's confirmatory H1 (line 536) is exactly that comparison. Its parser C7 and its semantic-feature student C9 are arms, but no hypothesis tests them.
  - Evaluation-statistics puts H1 against TR first in its gatekeeping order. In information classes its H2 must be co-primary. Its fallback label "beats myopic TR only" should read: "use of the information not shown to exceed a fixed-rule reading".
  - Two proposals have the right structure: paper-strategy (H1 and H2 both required) and experimentalist-task (C1 to C3 all required, including the numerical student given the same text embeddings).
- **Parser strength is a handicap the team controls.**
  - Paper-strategy admits a class only if the hand-written parser leaves at least 10% unrecovered on the held-out families (line 158). That rewards a weak parser.
  - Fix it with three things: a stated effort standard, an author who is not developing the LLM, and a second, learned non-LLM text model (N-bow) trained on the same families.
- **Experimentalist-task's prompt carries near-decision statistics.**
  - For each candidate it gives predicted responses with SDs and the probability of meeting the specification (line 92). These are close to the statistics a straddle or knowledge-gradient rule acts on.
  - N7 (a single rule) and N8 (a student on the prompt's features) help.
  - Add a small family of rules that use only the prompt, and report the share of decisions where the LLM deviates from them.
- **Hybrid roles are narrowed claims.**
  - This covers policy-repair's H-prior, advantage-theory's L3 compiler and paper-strategy's L-hyb.
  - Reported as narrowed claims, they are honest, but they do not meet condition 1, which asks for an LLM action policy.
  - Policy-repair's "non-LLM text prior (ridge regression on sentence embeddings)" is not free of LLMs if the embeddings come from a pretrained transformer. Add a bag-of-words prior.
  - Advantage-theory's L2 re-ranker sees the planner's scores, so it needs paper-strategy's rule: it must deviate from argmax-of-prompt on a declared share of decisions, and the deviations must help on average.
- **Controls the standing rules require.**
  - Advantage-theory's E10 roster (lines 526-533) lacks the frozen base LLM and the supervised-only LLM.
  - It also lacks three ablations that condition 3 needs: text removed, no pretraining, and no structure.
  - Policy-repair's E10 has the frozen LLM but no supervised-only checkpoint.

### 2.5 Leakage between training and evaluation

- **Namespaces.** All five declare disjoint namespaces. Paper-strategy reuses the confirmation range reserved under the old protocol. The registration must re-declare it for the new class.
- **Text as a side channel.** If the prose is generated from the private parameter draw, the text can become a numerical side channel. Paper-strategy's oracle-versus-parser comparison bounds how much the text carries, and policy-repair flags the risk. Require two things: the text generator is specified and frozen before the audit, and how much the text carries is measured by that parser gap.
- **Experimentalist-task's authoring order undermines its transfer claim.**
  - E1 reads the wind-tunnel source and transcribes its graph and forms.
  - E3, in parallel, writes a device library with power laws up to exponent 3, quadratic loss and DC motors (line 105).
  - The designer names the risk (line 192) but proposes no control.
  - The fix is either a library author who has not read W's mechanism code, or a written overlap audit frozen before E1 results. The claim must also be reworded as "a team-extended version of an independently authored simulator".
- **Isolation from private truth.** Several proposals move evaluation in-process and onto the Mac. That removes the isolation the custody harness provided, because an in-process evaluator shares memory with the policy and the private truth.
  - No proposal specifies a mechanical boundary.
  - Advantage-theory's leakage canary is a detection tool worth keeping: a policy that reads the private truth and should therefore exceed the ceiling.

### 2.6 Forking paths and reuse of test sets

- **Policy-repair selects on its 256 test worlds.**
  - N* is the smallest N that passes PQ-weak, and PQ is computed on the test worlds (line 88).
  - The DAgger procedure is "the smallest that passes PQ-weak and G8 on test worlds" (line 287).
  - E7 adopts the first escalation that passes E5's criteria, which include PQ on the test worlds.
  - Up to about seven configurations are therefore selected on the same test set, so a reported pass is optimistic.
- **Advantage-theory E7** says "Any recipe is acceptable if it passes the gates" (line 420), on development systems that are not refreshed between iterations.
- **Experimentalist-task.** The two rounds of levers in E1, and the choice of hardware design in E8 based on the E6 effect size, are declared in advance. They are acceptable if disclosed.
- **Evaluation-statistics has the best control:** a fresh validation shard for each iteration, a cap of three iterations, a failure ledger and equal tuning budgets.

### 2.7 Seeds, failures, nulls and clustering

- **Seeds.** Only evaluation-statistics specifies that every trained seed enters the confirmation (K = 5).
  - Policy-repair trains three seeds without saying which goes forward.
  - Paper-strategy freezes one set of weights after two seeds.
  - Experimentalist-task and advantage-theory leave it open.
  - Without a rule, a confirmed policy is a selected instance, and the claim covers that instance, not the recipe.
- **Failure treatment.**
  - Evaluation-statistics and paper-strategy score failures properly: no fallbacks, failed cells at the cap, no replacement systems.
  - Experimentalist-task's hardware primary uses completed episodes only. It reruns aborted episodes once, and scores aborts as loss 1 only in a sensitivity analysis (line 133).
  - If aborts correlate with arm, for example through latency, that primary is biased. It should be intention-to-treat.
  - Advantage-theory says failures are preserved but gives no scoring rule.
- **Null results need an equivalence margin.** Evaluation-statistics (TOST) and experimentalist-task (ratio CI within 0.90 to 1.10) have one. The others report a "null" without one, which cannot be published as evidence of absence.
- **States are nested in worlds.**
  - Policy-repair's CSC power calculation treats 300 states from 56 worlds as independent.
  - Paper-strategy's +/-6.3% bound treats 240 states from 24 worlds as independent.
  - The brief's standing rule is that independent systems are the units, so inference must be clustered by world.

### 2.8 The public preprint and README

What responsible handling looks like:

- The lead decides now. The README needs a banner and its table rows struck; this is internal and reversible.
- The preprint needs a decision on replacement or withdrawal, made with the coauthor, after confirming that 2602.02451 really is this project's paper.
- The v9 slide deck needs checking.
- Every new manuscript needs a section relating its results to the earlier public claims.
- No new ACE talk or submission should go out while the claim stands uncorrected.
- Editing the README is a change to public content, so it needs the lead's explicit go-ahead. No agent should make it unasked.

| Designer | Handling | Judgment |
| --- | --- | --- |
| paper-strategy | Section 7: a README banner now, an arXiv replacement or withdrawal with the coauthor's approval, a check of arXiv's version policy, the deck, and the stale PDF. No pre-audit numbers cited. No submission or talk until corrected. | Responsible. The only one that times the correction correctly. |
| evaluation-statistics | E10(e) includes a superseding statement; ask 10 says reconcile "before submission". | Partial. The timing is too late, because the README is live now. |
| experimentalist-task | Ask 10: reconcile before releasing new results or hardware data. Flags the Remote Lab's data-ownership and marketing terms. | Partial. No timing for the README and no disclosure section. |
| policy-repair | Absent. | Not responsible. |
| advantage-theory | Absent. | Not responsible. |

### 2.9 Misstated or unverified claims

- **Advantage-theory thesis.** It says any legitimate LLM advantage was "capped at roughly 2%", and that T4 "proves" this. That is premature: the Bayes sequential policy was never computed, and the analyst calls the bound an inference from greedy searches. It is a hypothesis for E1 and E2 to test.
- **Paper-strategy thesis.** It says "decision-level endpoints need 3 to 17" systems. The brief's figure is for public posterior risk, a model-internal quantity that is biased in the shifted systems.
- **Policy-repair E2.** It says "base Qwen3-0.6B ... all hash-matched", but only the adapters are verified. The fidelity gate covers this in practice, but the sentence is wrong.
- **Policy-repair E5.** It says "rank 16, alpha 32 on all seven projections, matching Kriz et al.'s main configuration". That main configuration is the Gemma-4-E4B model, and alpha 32 is not in the notes.
- **Evaluation-statistics.** The "8 to 96 times fewer systems" figure was computed with the learner's posterior, not with ERB. The designer caveats it. It should not be quoted until E1 recomputes it.
- **Policy-repair thesis.** It says the collapse was "an engineering failure, not a limit of small language models". That overreaches: a pass in one class shows "in this class, with this recipe".

---

## 3. Audit of each proposal

### policy-repair (6.5)

**Keep:**
- the precision-gated reference planner and the audit of what the old supervision could have taught (E1);
- fidelity-gated forensics with counterfactual edits rescored by the planner, invariance edits, and probes with grouped cross-validation (E2);
- interventional attribution of the collapse, with a numerical twin and a state-shuffled control (E3);
- the numerical learnability ladder, including a coarse baseline N0 (E4);
- the rules that parity with N2 at equal data is required and that a pass needing per-action planner statistics counts as narration;
- the mandatory gate suite G1 to G8;
- "do not claim an advantage in the three-node class";
- class changes made only by the E8 rule.

**Fix:**
- move every selection (N*, DAgger round, escalation) to the validation worlds, and score the test worlds once for the frozen configuration, or use a fresh test shard for each adoption;
- relabel D5 as a consistency check;
- cluster the CSC analysis by world;
- correct the claim about the base model's hash;
- resolve E8's semantic branch before E8 runs;
- freeze E10's endpoint and failure treatment at the E8 gate;
- add a seed policy, independent human review and the supervised-only control to E10;
- narrow the thesis;
- add the public-record ask.

**Publishability risk:** as written, a learnability claim whose test set was used for selection.

### advantage-theory (6.0)

**Keep:**
- E5's instrument calibration, with synthetic policies of known information capture and a leakage policy that should exceed the ceiling;
- the equal-information suite C1 to C10, including GO-CBED-style C8 with its rule for labelling an under-trained comparator;
- the atlas, with zero-dose anchors and dose monotonicity;
- dose, placebo and misleading-text strata, and an audit triggered whenever a gain exceeds the ceiling;
- "the policy model never writes its own evaluation texts";
- T7, which says the shuffled-feedback control has no power where adaptation is worth nothing.

**Fix:**
- make the equal-information controls (C7 parser plus C4, and C9) co-primary in E10, and require both;
- add the frozen base LLM, the supervised-only LLM, and the text-removed, no-pretraining and structure ablations to E10;
- cap and refresh E7 iterations;
- fix one primary class at the E4 gate, with E8 and E9 as secondary;
- label the 2% cap a hypothesis;
- validate or demote the redrawn-noise primary;
- rewrite "H1 failing: the claim narrows to development-only evidence" as "failed confirmation; no benefit claim";
- add the public-record ask.

**Publishability risk:** as written, a positive H1 would read as "the LLM uses text", when it shows only that text is valuable.

### experimentalist-task (5.5)

**Keep:**
- a shared, fixed decision rule, so that arms differ only in the interventions they choose;
- the numerical audit before any LLM runs, with the family dropped (never replaced) if it fails;
- the C1 to C3 Holm family, all required, including N9 with the same text embeddings;
- anonymized-text, misleading-text and shuffled-feedback arms;
- the hardware discipline: the task grid unseen until E9, replay only from recorded samples, chamber IDs not counted as systems, no reruns to improve a result, and a frozen manifest reviewed before any spend;
- its review of the Remote Lab terms.

**Fix:**
- word the claim as "team-extended" and declare every addition to W, with its documentary basis, before E1;
- control leakage from W into the training library, through a blind author or a frozen overlap audit;
- report N6 (the physics-informed planner) at headline level;
- make the hardware primary analysis intention-to-treat;
- add a seed policy;
- add a family of prompt-only rules;
- demote E2 and E9 from must until the lead rules whether the Remote Lab is compatible with "no paid APIs";
- keep the completed-class negative in the paper.

**Publishability risk:** "transfer to an independently authored simulator" would be inaccurate as worded.

### evaluation-statistics (8.0)

**Keep, almost entirely:**
- ERB with stated validity conditions and the counterexample showing that the learner's posterior risk can be gamed;
- harness validation (CRN marginals, A/A size, planted power, ERB calibration, agreement of the attribution, reliability of the stop rule);
- the class-qualification gate, with C0 as a negative control;
- a fresh validation shard per iteration, a cap of three, and equal tuning budgets;
- the registered confirmation, with blinding, K = 5, TOST, no extension after the fact, no dropped systems, and a failure ledger;
- the contribution-isolation family E7, including the identifiability argument for shuffled feedback (T5) and compute-matched TR;
- independent re-analysis and a 10% re-execution;
- retiring the 24 development systems.

**Fix:**
- move the binding endpoint choice off the development outcomes and drop the LLM-keyed veto as a selection criterion;
- in information classes, make H2 against the equal-information control co-primary, and relabel the fallback;
- correct the README now;
- recompute the 8 to 96 figure with ERB before quoting it.

### paper-strategy (7.5)

**Keep:**
- the public-record section;
- an admission rule hashed before any LLM sees K1 or K2;
- H1 and H2 both required, with N-emb and N-bow, F (the frozen LLM), and a supervised LLM control;
- the adjudicator's requirement to deviate beneficially from argmax-of-prompt;
- kill criteria in which every LLM-dependent kill only narrows the claim;
- the canonical class kept in the paper;
- the delivery study published separately;
- the 99% fidelity gate.

**Fix:**
- validate P1 or demote it in favour of ERB;
- set an effort standard and independent authorship for the parser, and keep family C away from the parser's author;
- avoid model-written held-out text unless the model is from another family and that is disclosed;
- correct the "3 to 17" misattribution;
- add a seed policy;
- cluster the probe intervals by world;
- schedule ask 12 before any LLM sees the admitted class.

---

## 4. What the final agenda must include

1. **The public record first.**
   - A README banner now, which needs the lead's approval.
   - A coauthor decision on replacing or withdrawing arXiv 2602.02451, after confirming it is this project's paper.
   - A check of the slide deck.
   - A section in every new manuscript relating it to the earlier public claims.
   - No new ACE talk or submission until this is done.
2. **Admission to a class that does not depend on the LLM**, frozen and externally timestamped before any LLM is trained, prompted or scored there.
   - One declared primary class for the headline claim.
   - All audited classes reported, including the failures and the completed three-node class.
   - At most two redesign rounds, using LLM-independent criteria only.
3. **A table of declared handicaps for each class**, written before the audit, with the numerical control that removes each handicap:
   - an oracle-informed planner;
   - a parser plus planner, written to an effort standard by someone not developing the LLM;
   - a bag-of-words text model;
   - a numerical student given frozen-LLM embeddings;
   - a GO-CBED-style amortized policy with a curve over training budget;
   - a rollout planner;
   - a compute-matched planner.

   All learned arms get the same data, and the same tuning budget on validation data only.
4. **Co-primary confirmatory hypotheses:** the LLM must beat the strongest planner without the side information, and also the strongest numerical control given the same information. A full positive requires both. The confirmation also includes:
   - the controls the standing rules require (target risk, the numerical student that replaces the ranker, the frozen base LLM, the supervised-only LLM);
   - text-removed, misleading-text, shuffled-feedback and state-permuted ablations;
   - a no-pretraining or embedding control;
   - dose and placebo strata;
   - an audit triggered whenever a gain exceeds the computed ceiling.
5. **Validate the instrument before any LLM is scored**, on fresh calibration systems with numerical arms only.
   - Checks: CRN marginals, an A/A test, planted-effect power, synthetic policies with known information capture, a leakage canary, and expected-versus-realized calibration.
   - The primary endpoint must be unbiased for an adaptive policy: either ERB, the expectation given the data actually observed, or realized error against the truth with replicates.
   - Redrawn-noise design quality is secondary unless it is validated.
6. **Development discipline.**
   - Disjoint, hashed namespaces.
   - A fresh validation shard for each iteration, with a cap of three iterations.
   - Every selection made on validation data.
   - Test worlds scored once for each frozen configuration.
   - A ledger of every attempt, with the count reported in the paper.
7. **Behavioural acceptance gates fixed in advance and applied before any rollout:**
   - legality;
   - concentration on a single action;
   - agreement better than a coarse-feature baseline;
   - held-out cross-entropy below that of a state-blind counter;
   - sensitivity to edits of the posterior, with invariance to menu order and identifiers;
   - stability across seeds.
8. **Fidelity-gated forensics and interventional attribution.**
   - Forensics run forward passes only and must reproduce at least 99% of the saved decisions; the base model's hash is checked as well.
   - Collapse attribution uses training-side factorials, with a numerical student given the same labels.
   - Forensic outputs reach the method only through rules written in advance.
   - Any causal claim about the collapse needs interventional support.
9. **Pre-registration with an external timestamp**, before the confirmation systems are generated. It fixes:
   - hashes, namespaces and endpoints;
   - failure treatment: failures scored at the cap, no replacement systems, aborted hardware episodes counted in the primary analysis;
   - the hypotheses and n;
   - a TOST margin for a null result;
   - the seed policy: all trained seeds are confirmed, and the registration says whether claims are about the recipe or the instance;
   - the analysis-code hash.

   Arm labels stay blinded, and a human re-analyses the results independently before any result is released.
10. **Narration controls.**
    - No per-action planner statistics in the primary input.
    - A family of argmax-of-prompt rules, plus a numerical student trained on exactly the prompt's features.
    - Any adjudicator must deviate beneficially from those rules.
    - Hybrid roles are reported as narrowed claims unless the lead rules otherwise before any LLM result.
11. **Response and compute accounting.** Simulated training responses, relabeling draws and compute are reported separately from deployment-time experimental cost. The "lower cost" claim refers to deployment responses only.

## 5. What must be dropped or changed

1. Advantage-theory E10. Replace H1 against controls without the text, as the sole superiority hypothesis, with co-primary H1 and H2 against C7 and C9. Add the frozen base LLM, the supervised-only LLM and the ablation arms.
2. Advantage-theory E7. Replace "any recipe that passes" with capped iterations on fresh development shards, recorded in a ledger.
3. Advantage-theory. Rewrite "H1 failing: claim narrows to development-only evidence" as "failed confirmation; no benefit claim". Relabel the 2% cap as a hypothesis. Fix one primary class before E7, with E8 and E9 as secondary.
4. Policy-repair E4 to E7. Selection and adoption on the test worlds become selection on validation only, with a single scoring of the test worlds or a fresh test shard.
5. Policy-repair. Relabel the D5 "gate validity" as a consistency check. Cluster the CSC analysis by world. Fix the claim about the base model's hash. Resolve the E8 branch in advance. Freeze E10's endpoint and failure treatment at the E8 gate. Add an independent human reviewer and a seed policy. Narrow the thesis.
6. Evaluation-statistics. Move the binding endpoint choice off the 24 development outcomes and drop the veto keyed to the old LLM arms as a selection criterion. Make H2 co-primary in information classes. Move the README and preprint action to now.
7. Paper-strategy. P1, the replay with 256 fresh noise draws, becomes secondary unless validated. Replace the "3 to 17" decision-level wording. The parser gets an effort standard, an independent author and a learned companion model.
8. Experimentalist-task.
   - Reword "independently authored" as "team-extended".
   - Control leakage from W into the training library.
   - Report N6 at headline level.
   - Make the hardware primary intention-to-treat.
   - Demote the Remote Lab stages until the ruling on "no paid APIs".
   - Add prompt-only rules.
9. All proposals. Hybrid or compiler success is never presented as meeting condition 1. Policy-repair's "non-LLM" text prior gets a bag-of-words version.
10. All proposals. Any stage that runs only after a positive result (hardware, a second family) is registered with the confirmation and reported whatever its outcome.

## 6. What all of them missed

1. **A mechanical boundary between private truth and policy code** once numerical policies run in-process and runs move to the Mac, outside the custody harness. It needs a process or API boundary, a hash of the public state for each decision, a leakage canary, and a provenance manifest (commit, environment hash, seeds) for every local run.
2. **A named human independent reviewer**, before the freeze. Agents write the policy, the evaluator and the analysis, so a review by another agent is not independent review (condition 5).
3. **A claim hierarchy and attempt ledger for the whole program**, covering classes, roles, model sizes, recipes, seeds and interface choices driven by the forensics, with the denominator reported in the paper. Holm correction within each experiment does not control a program that takes many shots at a positive result.
4. **A rule against deadline pressure:** venue deadlines may cut scope, but they never change a registered n, endpoint, gate or failure treatment. A confirmation that cannot reach its registered n is reported as incomplete.
5. **Confirmation that arXiv 2602.02451 is this project's preprint**, since its identity rests on the title, and a disclosure section covering the voided 70-71% claim, the hand-coded fills in Mission-1, and the eleven months of collapse.
6. **A lead ruling on whether the Remote Lab** (a paid external service with a real-time API) is permitted at all under "no paid APIs", before any sign-up, including the use of free credits. The ruling should also cover its terms on jointly owned data and marketing use of the customer's name.
7. **World-clustered inference for every analysis at the level of states**, and an equivalence margin for every test that can return a null.
8. **Retirement of the 24 development systems and the 32 November training worlds** from every evaluation role (only evaluation-statistics asks for this), and a fresh declaration of the reserved confirmation range for the new class.

## 7. Recommended spine

1. **Week 0.** The lead decides on the public record, on registration (an external timestamp and a named human reviewer) and on approvals. Nothing new about ACE is presented until the README is corrected.
2. **Stage 1, in parallel, with no LLM training.**
   - (a) Validate endpoints and the instrument on fresh calibration systems with numerical arms.
   - (b) Build a headroom atlas with a declared-handicap table. It ends in a hashed, externally timestamped admission decision naming one primary class.
   - (c) Run fidelity-gated forensics on the saved adapters, and a numerical learnability ladder on fresh training-namespace worlds.
3. **Stage 2.** In the completed three-node class, with fresh namespaces and training-side data only:
   - test whether a repaired recipe can imitate the planner, with knockout attribution;
   - train numerical students on identical data;
   - apply the mandatory behavioural gates.

   This establishes or refutes condition 1 and supports a diagnostic paper either way. It never claims an advantage in that class.
4. **Stage 3.** Development in the admitted primary class:
   - training distilled from a privileged teacher;
   - a fresh validation shard per iteration, at most three;
   - equal tuning budgets;
   - the LLM's role and the seed policy fixed by rules declared in advance.
5. **Stage 4.** One registered confirmation on fresh systems:
   - co-primary hypotheses against the uninformed planner and against the strongest equal-information numerical control;
   - the controls the standing rules require, and the ablation family;
   - dose and placebo strata, with an audit triggered whenever a gain exceeds the ceiling;
   - all trained seeds, intention-to-treat failure scoring and a TOST margin;
   - blinded analysis, then a human re-analysis.
6. **Stage 5.** A registered transfer family (the wind-tunnel simulator, described as team-extended), reported whatever the outcome. Hardware runs only after a lead ruling, with frozen policies, and is reported descriptively.
7. **Kill rules only narrow the claim.**
   - If no class is admitted, the paper is the theory, the atlas and the diagnostic.
   - If imitation fails, the paper is the diagnostic.
   - If the confirmation fails, it is reported as a failed confirmation, with no replacement class.
8. **Separate tracks.** The delivery study and the methodology results proceed separately and do not depend on any LLM result.

## 8. Asks the integrity lens adds

1. **The public record.**
   - Approve the README banner now.
   - Confirm the preprint's identity, then decide on replacement or withdrawal with the coauthor.
   - Check the v9 deck.
2. **Registration and review.** Choose a registration with an external timestamp, and name a human independent reviewer before any confirmation system is generated.
3. **The Remote Lab.**
   - Rule whether it counts as a paid API under the standing rules.
   - If it is allowed, accept or reject its data-ownership and marketing terms explicitly.
4. **What counts as success.** Rule before any LLM result in a new class whether these count toward conditions 1 and 2:
   - a benefit scoped to "information that exists only in prose";
   - a hybrid or compiler role.
5. **Development data.** Approve retiring the 24 development systems and the 32 November training worlds from all evaluation roles, and re-declaring the reserved confirmation range for the new class.
6. **The E3 anchor cell.** Rule that policy-repair's anchor cell (the original interface, single pass) is a control cell of a new factorial, not a rerun of the old pipeline.
