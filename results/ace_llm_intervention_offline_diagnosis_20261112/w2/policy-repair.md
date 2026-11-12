# Policy-repair agenda: making a learned intervention policy actually learn

Panel lens: machine-learning engineer responsible for getting a learned intervention policy to learn. Prepared 2026-11-12 from the verified brief (`../w2/brief.md`) and the `../w1/` notes. Read-only design work: nothing under `/Users/pat/code/ACE` was changed, no model was run, no network was used. Two small checks were run in scratch (arithmetic in `calc/calc.py`; a JSON lookup in the pinned tokenizer file). Facts are marked **[V]** (verified in the brief, the notes or a file) or **[A]** (my assumption, with the check that would settle it).

## 0. Bottom line

1. The saved evidence already explains most of the collapse in ML terms. The model saw 128 informative examples once, at batch size 1 and in a fixed order. Half of those labels came from teachers that ignore the SCM. Actions were read out by greedy decoding over a token trie whose branch pluralities spell `set_x_neg`. The state was 2–4k tokens of 17-digit floats. The DPO stage drifted the command bias and did not learn preferences. No check looked at behaviour. Two things are still missing: direct evidence about what the checkpoints read, and interventional evidence about which defect was sufficient. E1–E3 supply both, at small cost.
2. Before any language model is retrained, a numerical student must show that planner-quality decisions are learnable from the public state, with enough data and the right target (E4). If a learner with direct numeric access cannot do it, a 0.6B model reading text will not, and the direct-policy route should be abandoned for a hybrid.
3. The LLM recipe this agenda proposes (E5) changes every defect at once. The teacher is a single high-precision planner. Targets are listwise soft targets. The action is one neutral token, with codes and menu order permuted per example, and the decision is the argmax over legal codes. The state is a compact rendering that is sufficient for the planner. Training covers thousands of states from hundreds of worlds, in shuffled minibatches over several epochs, with LoRA on all linear layers. On-policy relabeling is added where needed. A mandatory gate suite runs before any rollout. Leave-one-fix-out ablations then attribute which change mattered.
4. In the known-graph linear-Gaussian class, distillation can at best match the planner. Finding A shows that no policy can beat it by more than a few percent. An advantage test has to wait until the policy passes the learnability gates and a world class has been chosen by criteria that use no LLM output (E8). The policy must then pass the same gates in that class (E9) before fresh systems are scored (E10).

## 1. Diagnosis in ML terms (what is already established)

| Defect | Evidence [V] | Consequence for the redesign |
| --- | --- | --- |
| Supervision quality | Four rotating teachers with identity hidden: 25% uniform random, 25% menu coverage (which emits `set_x_neg` at every initial state via an alphabetical tie-break), 25% local information, 25% target risk (half of its labels within 2 MC SE of the runner-up). Labels equal the high-draw planner optimum in 55–56 of 128 states. | One teacher, high-precision scores, graded targets (E1). |
| Data volume and exposure | 128 informative states plus 32 forced-stop examples. One unshuffled pass, batch 1, lr 1e-4, 149/160 SFT steps clipped. Prequential NLL 2.37 nats against 1.93 for a state-blind frequency counter. The model retained nothing about its own training states (label log-prob −2.168 before its own update, −2.177 at the end of SFT). | Far more states, shuffled minibatches, several epochs, held-out worlds (E4, E5). |
| Preference feedback | 4-replicate tie rule = t-test on 3 df at α = 0.139. About 70% of the 687 strict labels are at noise level, and they agree with the planner 68.7% of the time. That is about 0.107 bits per strict label. 119 pairs were used, with every step clipped. Both likelihoods fell. A state-blind model with token-factorized biases reproduces the DPO pass (r 0.80 and 0.93). | Drop measured-preference DPO in this class and use planner relabeling. On-policy states come from DAgger (E6). |
| State interface | Prompts are 1,990–4,160 tokens, 34–48% digits, one token per digit. History grows to 67% of the prompt. The 346 tokens nearest the answer are constant. The SFT adapter reads the budget counter, graph type and history length. No adapter shows posterior reading. | A compact rendering that is sufficient for the planner, with a probe and counterfactual tests (E2, E5). |
| Action interface and decoding | Multi-token commands with shared suffix tokens. Greedy trie decoding is not whole-command argmax: the `set` branch pools seven commands. Menu position, identity and the tie rule are confounded for `set_x_neg`. Stop was legal everywhere but labelled only at terminal states. | Single-token neutral codes permuted per example. Argmax over legal codes. No stop action in this class (E3, E5). |
| Acceptance | The gate checked legality, reload and repeatability only. The constant action was in files written by the training job (SFT 108/128, DPO 123/128 on training states). | A mandatory behavioural gate suite, validated retrospectively on the saved artifacts (E2). |
| Control | The exposure-matched numerical ranker also collapsed: two actions, 98.1% of score variance from action identity, losses at chance. | The failure is in data and exposure, not specific to language. The learned numerical control must be retrained to convergence (E3, E4). |

What is not established, and what this agenda settles: whether the base model already carries the `_neg` or first-listed prior; how much the trie decoder changed decisions; whether any checkpoint responds to the posterior at all; whether the hidden state encodes the posterior (a representation gap versus a knowing-doing gap); and which fix alone, on the same states, removes the constant action.

## 2. Design commitments and their LLM-independent justifications

### 2.1 Teacher and target (E1)

- **Teacher:** the frozen target-risk score: the expected reduction in posterior integrated risk per cost unit. It is re-implemented in-process with common random numbers across candidate actions and validated against the 560 saved receipts. A scratch re-implementation already reproduced them to 1.1e-16 [V, deployment-forensics-verify]. One teacher only.
- **Precision:** pick the smallest D in {1,024, 4,096, 16,384} draws at which two seeds agree on the argmax outside a declared near-tie band (relative gap < 1%) in at least 98% of states, and at which the regret of one seed's argmax under the other's scores is at most 0.005 in at least 95% of states.
- **Target forms:** S1 hard argmax; S2 listwise soft target p*(a|s) ∝ exp(−r(a|s)/τ), where r is relative regret; S3 regression on relative regret. τ is the largest value in {0.005, 0.01, 0.02, 0.05, 0.1} for which the mean over training-world states of E_{p*}[r] is at most 0.02. That bar is target risk's own deployed regret (0.013–0.016) plus a margin. Worst-case bound: E_{p*}[r] ≤ (K−1)τ/e, which is 0.013 at τ = 0.005 for K = 8 (section 5.6). Soft targets exist because near-ties are structural. In fork worlds with the uniform menu, `set_x_neg` and `set_xy` are exactly equivalent for Z, so a hard label there is a coin flip.
- **Supervision ceiling of the old process:** compute the argmax of the infinite-data predictor implied by the old four-teacher mixture, p_old(a|s) = ¼·U + ¼·δ_cov(s) + ¼·δ_LI(s) + ¼·δ_TR32(s), at all 697 saved training-world states, and score its regret under the reference. This gives the best policy the old label process could have taught with unlimited data.

### 2.2 State sampling (E4, E6)

- The training corpus has new training-scope world IDs, the same generator law (4 regimes × 3 graphs × 2 cost menus, balanced) and the same 8 + 24 protocol. Splits are by world: train, validation (256) and test (256). The world is the statistical unit.
- Each world gets three trajectories: (i) target risk with ε = 0.25 random actions, which adds perturbation coverage around the planner's own distribution; (ii) one of local information, uniform random or menu coverage; (iii) a stress trajectory that repeats one random action for k ∈ {4, 8, 12, 16, 24} steps and then follows target risk. Training histories never had a run longer than 10 [V], while DPO deployment histories had runs of 11 to 23.
- All decision states are kept (about 65 per world), covering every remaining-budget level from 1 to 24. The old data covered only 5, 6, 11, 12, 17, 18 and 24.
- On-policy states are added by DAgger in E6. Labels for those states cost no environment responses. Rollouts in synthetic training worlds do consume synthetic responses, and these are reported as training cost.

### 2.3 State interface levels and the equal-information rule

| Level | Content | LLM arm | Numerical twin |
| --- | --- | --- | --- |
| L0 | Original canonical JSON: full-precision posterior plus raw history | E5 ablation "−compact" | N1 (8,891-wide ranker features) |
| L1 | Compact text sufficient for the planner: graph, noise, posterior per head (means, SDs, correlations) rounded per E1, costs, remaining budget, task summary, history summarized as per-action counts and the last three actions | E5 main arm | N2 (permutation-equivariant per-action net on posterior features) |
| L2 | L1 plus non-action-specific derived summaries: per-query predictive SD, per-head eligible rows, current integrated risk | E7 escalation (a) | N3 |
| L3 | Per-action planner statistics | Upper-bound control only. Never a candidate, because the LLM would only narrate an argmax. | N4 (= planner) |

Justification [V, numerics-protocol §3]: target risk reads only the posterior summaries, menu, costs, budget and task. Raw history therefore carries no decision-relevant information beyond the posterior, and dropping it is not a favourable change chosen by looking at LLM results. Every feature shown to the LLM is also given to its numerical twin.

### 2.4 Action interface and decoding rule

- The eight acquisitions are mapped per example to a random permutation of the letters A–H. These are single tokens in the pinned Qwen3-0.6B vocabulary (ids 32–39; with a leading space, ids 356–479) [V, `tokenizer.json` lookup]. The menu order is also permuted per example, and each code sits next to its action description and cost.
- The decision is the argmax over the eight legal code tokens, renormalized over the legal set, with a keyed random tie-break. One forward pass per decision. There is no trie, no length bias (TWOSOME) and no alphabetical tie rule.
- **Stop is removed from the policy in this class.** Every numerical arm spends the full budget, and stop labels existed only at terminal states where the grammar forced stop. Episodes end at budget exhaustion. If a later class needs stopping, a shared public stop rule (posterior risk at or below tolerance) applies to all arms equally.

### 2.5 Optimization recipe (E5 default)

LoRA rank 16, alpha 32, dropout 0.05 on all seven linear projections (10,092,544 parameters, against 1,146,880 before [V by recomputation from the saved count]). This matches Kriz et al.'s main configuration. AdamW with lr from {5e-5, 2e-4} chosen on validation worlds, cosine schedule, 3% warmup, effective batch 32, two epochs, shuffled minibatches stratified by graph × cost × budget bin. Loss: soft-target cross-entropy over the legal code tokens. Gradient clip 1.0, with a predeclared rule that if more than 50% of steps are clipped the lr is halved; the old run clipped 268/279 steps. Early stopping on validation-world regret. Three seeds.

### 2.6 Feedback: offline, on-policy, preferences

- In this class the dense, near-noise-free teacher is the planner, and labels cost no environment responses. Measured single-response gains have reliability of about 0.15. The agenda therefore drops measured-preference DPO here.
- With a single-token action and known per-action scores, the on-policy RL objective E_{a∼π}[s(a|x)] has an exact gradient, Σ_a s(a|x)∇π(a|x). GRPO-style sampling adds variance and no information. The only on-policy ingredient that matters is the state distribution, which DAgger supplies (E6).
- Learning from realized outcomes is reserved for a class where the planner's model is incomplete (E9). A numerical student must receive identical feedback there.

### 2.7 Capacity ladder (predeclared, cheapest first)

LoRA rank 16 on all linear layers (E5). Then L2 summaries; then rank 32 with twice the data; then a linear action head on the decision-position hidden state, only if E2 shows the posterior is decodable there; then Qwen2.5-1.5B-Instruct, which is in the local cache but needs approval. Each step gets one seed. The first arm that passes is confirmed with two more seeds (E7).

### 2.8 Gate suite (mandatory before any rollout or test-world scoring)

| Gate | Criterion (validation worlds) | Would it have caught the November adapters? [V numbers] |
| --- | --- | --- |
| G1 legality | 100% legal outputs | No: both passed |
| G2 collapse alarm | Max share of any action ≤ max(0.6, reference-optimal share of that action on the same states + 0.15), and entropy of chosen actions ≥ 0.5 × entropy of the reference-optimal actions | **Yes.** SFT 108/128 = 0.84 and DPO 123/128 = 0.96 on training states. In deployment: frozen 0.997, DPO 1.0, SFT 0.92, ranker 0.755, all fail. Target risk (0.384) and local information (0.558) pass. The reference-optimal max share on the LLM arms' states is 0.30–0.35. |
| G3 agreement | Near-tie-tolerant agreement (chosen action within 5% relative regret of the best) ≥ coarse baseline N0 + 0.15 | **Yes.** SFT 36/128 = 0.28 and DPO 0.25, against the constant baseline 0.25, so 0.40 was required |
| G4 generalization | Held-out soft-target cross-entropy ≤ state-blind frequency model − 0.20 nats | **Yes.** SFT 2.37 against 1.93, so ≤ 1.73 was required |
| G5 counterfactual | Posterior-swap CSC ≥ 0.3 with 95% lower bound > 0.1, and total-variation change ≤ 0.10 under planner-invariant edits | Expected yes. Measured in E2. |
| G6 score structure | State × action interaction share of the policy's legal-set log-prob table ≥ 0.5 × the reference's own share | Likely: final adapter 10.0% interaction against 82.7% command main effect. Ranker 98.1% action identity. The reference share is computed in E2. |
| G7 seed stability | Across 3 seeds, SD of validation regret ≤ 0.03, and action-histogram total variation ≤ 0.15 | Not testable: one seed |
| G8 closed loop | Rollouts on 64 validation worlds: own-trajectory regret ≤ 0.20, and longest repeated-action run ≤ the planner's 99th percentile + 3 | Yes, by construction (DPO ran 24 identical actions) |

G2, G3 and G4 are computable from files the training job wrote (`sft-generation-NN.json`, `final-generation-NN.json`, the step losses), so any one of them would have blocked deployment. E2 makes this retrospective validation formal. The thresholds are fixed here, before they are computed on the planners. If a planner fails a gate, the gate may be revised once, with a written reason, before E5 starts.

**Planner-quality acceptance (PQ), test worlds, closed loop.** PQ-weak: mean relative regret ≤ 0.15, the 95% upper bound of the geometric-mean design-quality ratio J/J_TR ≤ 1.05, and regret ≤ N0 − 0.10 with a CI excluding zero. PQ-strong: regret ≤ 0.08 and J/J_TR ≤ 1.02. Anchors [V]: target risk 0.016 and 1.00; local information 0.154 and ×1.02; leave-one-out public static schedule ×1.208; coverage 0.441 and ×1.35; random 0.504 and ×1.42; ranker 0.575 and ×1.59; frozen 0.655 and ×1.72; SFT 0.665 and ×1.88; DPO 0.700 and ×2.10. **Parity:** |regret_LLM − regret_N2| ≤ 0.05 at equal data.

### 2.9 When to abandon the direct-policy route for a hybrid (predeclared)

- **A1, numerical impossibility:** no numerical student (N1, N2, or N3 with L2 summaries) reaches PQ-weak with 2,048 training worlds (E4). A learner with direct numeric access failed, so text access will not succeed at feasible scale.
- **A2, LLM-specific failure:** after E5 and the full E7 ladder, the LLM misses PQ-weak or parity, or fails G5. Or it passes only when given L3 per-action planner statistics, which is narration.
- **A3, closed-loop failure:** the state-level gates pass, but closed-loop PQ fails after E6.
- **A4, headroom location:** in the class E8 selects, the headroom sits in the information the planner consumes through its model (H2 ≥ 1.15 while H1 < 1.05 and H3 < 1.03). The natural LLM contribution is then a prior, not an action.

**Hybrid H-prior:** the LLM maps text (variable descriptions, units, domain notes, graph) to a numerical prior over mechanism coefficients, or to a distribution over graphs, and the numerical planner acts with it. The LLM changes the decision through the model, so it is not narrating. Training is supervised regression on the generating parameters of training worlds, with Gaussian NLL. Controls: the uninformed planner; the oracle-prior planner as an upper bound; and a non-LLM text prior (ridge regression on sentence embeddings of the same text). This hybrid is testable only in a class where the text carries information. The current class has one byte-identical prose payload, so it is not.

## 3. The agenda

### E1. Reference targets and supervision-quality audit (must)

- **Question:** What should a learned policy imitate? How precise must that target be? How much could the old supervision have taught, even if fitted perfectly?
- **Design:**
  - Implement the in-process common-random-numbers (CRN) target-risk planner, which must reproduce all 560 saved receipts.
  - Run the precision ladder (D ∈ {1,024, 4,096, 16,384} × 2 seeds) on the 697 saved training-world decision states plus the 300 E2 base states.
  - Compute the 16,384-draw reference on all 4,203 unique development states, for E2's regret computations only.
  - Rounding test on training-world states only, using the exact L1 rendering at 3 and 2 significant digits.
  - Choose τ by the rule in 2.1.
  - Old-supervision audit: regret of each of the 128 labels by teacher; the flip rate of the 687 strict pairs against the reference; bits per label; and the infinite-data ceiling argmax p_old (2.1).
  - Optional, as a separate ask: truth-conditional expected gains from the 32 private training-world descriptors.
- **Sample size:** all saved training-world states (697) and the 300 E2 base states. That is the whole available set; there is no sampling.
- **Endpoints:** precision gate; rounding criterion (top-1 agreement ≥ 98% outside near-ties and mean regret of the rounded argmax ≤ 0.005); τ; regret of the old-supervision ceiling.
- **Decision rule:**
  - Training proceeds only if the precision gate passes at D ≤ 16,384 and the rounding criterion passes at 3 digits. If it fails, L1 uses 4 digits. That choice is made here, with no LLM involved.
  - If the old-supervision ceiling regret is ≥ 0.30, the paper may state that the old label process could not have taught planner-quality decisions even with infinite data. If it is ≤ 0.15, the label process was adequate in the limit and the binding defects were volume, exposure and interface.
  - **Positive:** a precise, zero-response teacher. This justifies dropping measured preferences.
  - **Null:** irreducible multi-way ties. Report the tie fraction and use near-tie-tolerant agreement throughout.
  - **Negative:** the CRN planner disagrees with the saved receipts. Stop and fix before any training.
- **Prerequisites:** none beyond local CPU, with zero environment responses. A reviewed in-repo CRN planner; the scratch versions are not in project custody. The truth-conditional option needs approval.
- **Cost:** about 20 CPU-minutes in total. Basis [V]: a vectorised 16,384-draw re-implementation scored 4,203 states in 499 s on this Mac (`../w1/deployment-forensics-verify/v09.log`), 0.119 s per state. About 300 lines of code. 1–2 days.
- **Main risk:** the teacher is a myopic working-model planner, which is imperfect under the shifted prior. This is acceptable here because Finding A bounds any policy's edge over it at a few percent.

### E2. Forward-only forensics of the deployed policies (must)

- **Question:** Did the constant action come from a base-model prior, from supervised training, from DPO drift or from the decoder? Does any checkpoint read the posterior, and does its hidden state encode it?
- **Design:**
  - Checkpoints: the base Qwen3-0.6B, the SFT adapter (583b1659…) and the final adapter (6f540dfe…). All are on the Mac with matching hashes [V].
  - Prompts: all 160 training prompts and all 1,698 deployment prompts, using the exact saved token ids.
  - Per checkpoint and prompt: log-probabilities of all legal commands (prefix-shared scoring), branch distributions at the four trie decisions, a greedy re-decode, and hidden states at the decision position (layers 14 and 28).
  - **Fidelity gate:** the local re-decode must reproduce at least 99% of the 1,698 saved completions and the 322 saved training generations. Mismatches are allowed only where the branch margin is below 0.05 nats.
  - Analyses:
    - (a) Command / state / interaction decomposition of each 1,858 × 9 table.
    - (b) Decoding audit: how often trie-greedy differs from whole-command argmax and from length-normalized argmax, and the change in relative regret under E1's reference.
    - (c) Base prior: mass on the first menu entry and on the first listed intervention.
    - (d) Label tracking on the 128 training states (permutation test).
  - **Counterfactual battery:** 300 base states: all 128 nonterminal training states plus 172 deployment states, stratified by arm × decision block (0–3, 4–11, 12–23). Twelve edits per state:
    - C1 posterior-block swap from a donor matched on graph, cost menu, budget ±1 and history length (2 donors).
    - C2 covariance of the target-relevant head scaled ×0.25 and ×4.
    - C3 history-row permutation and history-value swap, which leave the planner unchanged.
    - C4 two random menu orders.
    - C5 identifiers renamed to neutral codes.
    - C6 budget ±3 with affordability unchanged.
    - C7 all floats rounded to 3 significant digits.
    - Each edited envelope is rescored by the reference planner.
  - **Probes:** ridge regression from hidden states to the 8-vector of planner relative regrets, with grouped 8-fold cross-validation over 56 worlds. Report the R² increment over a probe on coarse features (graph, cost, budget, decision index, per-action counts). A base-only probe on the L1 rendering of the 697 training-world states (no development states) tests whether a compact rendering makes the posterior decodable.
  - **Retrospective gates:** G2, G3, G4 and G6 (and G5 from the battery) on the saved SFT, DPO and frozen outputs and on the five numerical arms, with the thresholds of 2.8.
- **Sample size basis:** 300 states give power 0.82 to detect a counterfactual sensitivity coefficient (CSC) of 0.2 at two-sided α = 0.01 (Fisher z; 287 needed for 0.80). The 95% interval around zero is ±0.11. An observed CSC of 0.30 has a lower bound of 0.19, above G5's 0.1.
- **Endpoints:**
  - CSC: Spearman correlation across states between the planner's change in the score difference of the most-affected action pair and the checkpoint's change in log-odds for that pair (C1, C2).
  - Spurious sensitivity: mean total-variation change in the legal-set distribution under C3–C7.
  - The decomposition shares, decoding disagreement, probe R² increments and gate outcomes.
- **Decision rule (predeclared):**
  - D1, decoding contributed: whole-command argmax changes at least 10% of SFT deployment decisions.
  - D2, base prior contributed: the base model puts at least 0.5 of its `set`-subtree mass on `set_x_neg` on training prompts.
  - D3, posterior-blind: the CSC upper bound is below 0.1 for SFT and final, while spurious sensitivity to budget, history or menu order is at least 0.10.
  - D4, representation present: the probe increment is at least 0.10 with a lower bound above 0. Absent: the upper bound is below 0.05.
  - D5, gate validity: G2–G4 fail SFT and DPO and pass target risk and local information.
  - **Positive (D3 and D5):** the paper states, with direct evidence, that the deployed adapters did not read the posterior and that the collapse was detectable at training time.
  - **Null (CSC interval spans 0.1–0.3):** posterior use is undetermined; rely on E3.
  - **Negative (CSC ≥ 0.3 for SFT):** the adapters read the posterior, but a command prior and decoder dominated. The diagnosis is reframed as prior domination.
  - D4 absent means the representation is the bottleneck, which supports L1/L2. D4 present means a knowing-doing gap, which supports objective and readout fixes and enables escalation (c) in E7.
- **Prerequisites:** lead approval of local forward-only inference under a written protocol; local torch and transformers differ from the frozen runtime [V, ledger]. Alternatively, a CURC GPU ceiling of about 2 slice-hours. Also E1's planner, a reviewed scoring harness and an edit generator (about 500 lines).
- **Cost:**
  - About 16,400 forwards (5,574 for the tables plus 10,800 for the battery) on 2–4k-token prompts, plus about 700 short probe forwards.
  - Mac: 1.4–6.8 h, assuming 0.3–1.5 s per forward [A]; a 20-prompt pilot fixes the figure.
  - CURC: 0.55–1.8 slice-hours at the ledger's 0.12–0.4 s per forward [V]. The upper end exceeds the 0.76 h remainder.
  - Planner rescoring of 3,600 edited states: about 7 CPU-minutes.
  - Calendar: 3–5 days including the protocol.
- **Main risk:** local numerics may not reproduce near-tie decisions. Posterior swaps create inconsistent envelopes (posterior not matching history) that are out of distribution for every checkpoint. The C3 edits partly control for this.

### E3. Collapse attribution on the original 128 training states (must)

- **Question:** With the states held fixed, which repair alone removes the constant action: labels, interface or exposure? For the LLM and for the numerical student?
- **Design:**
  - States: the exact 128 nonterminal snapshot states the deployed models were trained on. No new responses.
  - Factors, 2 × 2 × 2:
    - Labels: S0 (original mixed hard labels) or S2 (E1 soft targets).
    - Interface: original (string ids, fixed order, full-precision JSON with raw history, trie-greedy decoding, whole-command loss) or repaired (per-example permuted single-token codes and menu order, L1 text, argmax over codes).
    - Exposure: single pass at batch 1 (seed 1 in roster order, seed 2 in random order) or converged (10 epochs, shuffled, batch 16, cosine lr 1e-4).
  - The LLM is held at the deployed capacity, rank 8 on q/v, so capacity is not confounded; capacity is tested in E5. 16 LLM runs.
  - Numerical student: original features (8,891-wide, ranker MLP) or repaired (equivariant per-action posterior net), crossed with the same label and exposure factors, × 3 seeds = 24 CPU runs.
  - **Anchor check:** the S0 / original / single-pass / roster-order cell should reproduce the collapse (at least 75% one action on the 128 states, against the saved 108/128). The numerical anchor should reproduce the saved ranker pattern (at least 90% of variance from action identity).
  - Controls: state-shuffled labels within graph × cost cells in the all-repaired cell; the constant plurality baseline; N0.
- **Sample size:** fixed by the question. All 128 training states; evaluation on the other 569 saved states of the same 32 worlds plus 32 new validation worlds × 24 states from the E4 corpus (about 1,340 states per run).
- **Endpoints:** maximum action share (collapse means ≥ 0.75); near-tie-tolerant agreement; mean relative regret; interaction share.
- **Decision rule:**
  - A factor is a sufficient repair if fixing it alone drops the max share below 0.6 and raises agreement by at least 0.15 over the constant, in both seeds. It is necessary if reverting it in the all-repaired cell restores collapse.
  - An order effect is confirmed if single-pass roster order collapses and single-pass random order does not.
  - The state-shuffled control must stay at or below N0. If it matches real labels, learning is coarse-feature only.
  - **Positive:** interventional attribution of the observed collapse.
  - **Null (collapse or N0-level regret in every cell):** 128 states cannot teach state dependence under any repair, so the collapse was overdetermined by data volume.
  - **Negative (the anchor does not collapse):** the collapse is fragile to runtime or order, which narrows every diagnostic claim. Reconcile with the saved logs before interpreting further.
- **Prerequisites:**
  - E1 targets and the E2 fidelity result.
  - The new reviewed training engine, about 800–1,200 lines and reused through E9: soft targets over codes or whole commands, minibatches, scheduler, all-linear LoRA, gate computation.
  - A GPU ceiling or approval for local training.
  - Possibly a lead ruling that this is not a rerun of the old pipeline: it is a new engine, no DPO, and no environment responses.
- **Cost:**
  - Original-interface cells: 2–4k-token prompts at 0.3–0.6 s per step. Repaired cells: about 600 tokens at 0.05–0.2 s per step [A, scaled from the 0.62 s batch-1 SFT update [V]].
  - Training: about 1 h. Evaluation: about 11k forwards, 0.5–1.5 h. Total about 1.5–3 GPU-slice-hours.
  - Numerical runs: CPU minutes.
  - Calendar: 4–6 days, mostly the engine.
- **Main risk:** every cell may be data-limited out of world, so attribution holds only in-sample. The original-interface cells are costly per step.

### E4. Numerical learnability ladder (must)

- **Question:** Can any numerical learner recover planner-quality decisions from the public state on held-out systems? With how much data, which representation and which target?
- **Design:**
  - Corpus as in 2.2, labelled by E1's reference at the gated D. Nested training sets N ∈ {32, 128, 512, 2,048} worlds; 256 validation and 256 test worlds.
  - Learners:
    - N0: coarse gradient-boosted trees on graph, cost menu, noise level, remaining budget, decision index, per-action history counts and last action. This is everything the SFT adapter was found to read.
    - N1: the ranker MLP on the 8,891-wide features.
    - N2: an equivariant per-action net on L1-equivalent posterior features.
    - N3: N2 plus L2 summaries.
    - N4: planner statistics, as an upper-bound control.
  - Targets S1, S2 and S3. Three seeds. Adam, minibatch 256, early stopping on validation loss.
  - Reference policies rolled out on the test worlds: target risk at 32 draws (as deployed), local information, coverage, random.
- **Sample size basis:**
  - 256 test worlds give a ±0.02 CI on mean regret if the world-level SD is at most 0.16 (216 needed at SD 0.15).
  - On log J ratios the half-width is ±0.021 for a coverage-like policy (SD 0.171, implied by the 24-system t of 8.6) and ±0.004 for a local-information-like policy (SD 0.035).
  - Training sizes span 64× around the old 32-world volume, enough to fit a learning curve and locate N*.
- **Endpoints:**
  - State-level regret on states from the ε-greedy target-risk trajectories (primary) and on stress states.
  - Near-tie-tolerant agreement; the action-identity share of score variance; numerical CSC under the E2 edits.
  - Closed-loop own-trajectory regret and J/J_TR, computed from the common post-initialization state with 4,000 draws, for the best configuration per learner × N.
- **Decision rule:** PQ-weak as in 2.8. N* is the smallest N at which N1 or N2 passes.
  - **Positive:** decisions are learnable from the public SCM state. The LLM gets N_LLM = min(max(N*, 128), 512) worlds and the representation level that passed. The paper states that the earlier collapse was a supervision and exposure failure, not an impossibility.
  - **Null (only N3 passes):** the LLM interface must carry L2, and the claim narrows to "reads an SCM summary".
  - **Negative (nothing passes at 2,048, or nothing beats N0 by 0.10):** abandon the direct route in this class (A1).
  - If N0 itself passes PQ-weak, the class does not discriminate posterior use, and only the G5 counterfactual tests can show posterior reading.
- **Prerequisites:**
  - Lead approval to generate a training-scope synthetic corpus with new world IDs (no development or confirmation IDs).
  - New code, about 1,000 lines: a learnability protocol module and an in-process rollout harness (the existing per-decision subprocess design costs about 48 s per episode [V]). The harness must reproduce saved training trajectories exactly before use. Also the equivariant student and the evaluation.
- **Cost:**
  - Labelling about 166k states × 0.119 s ≈ 5.5 CPU-hours at 16,384 draws, or about 1.4 h at 4,096, in parallel on the Mac's cores.
  - 168 small fits: 3–10 CPU-hours.
  - Closed-loop evaluation: 1–2 CPU-hours.
  - Zero GPU, zero CURC. Calendar: 5–7 days.
- **Main risk:** Finding A says reading the state is worth only 10–20% of final expected NMSE over a good graph-and-menu schedule, so N0 may approach the planner and weaken discrimination. The corpus generator is a second implementation of the protocol and must be checked against saved trajectories.

### E5. LLM listwise distillation with the repaired interface (must)

- **Question:** Given the data and representation at which a numerical student succeeded, can Qwen3-0.6B learn a state-dependent, planner-quality policy from the SCM in text? Which repairs are necessary?
- **Design:**
  - Data: the E4 corpus at N_LLM worlds × 24 states per world, sampled across all three trajectory types (512 × 24 = 12.3k states at the cap).
  - **Stage 1:** the A-full recipe of 2.3–2.5 × 3 seeds. The lr is chosen from two values on 128 worlds. Controls:
    - A label-permuted LLM: targets from another state in the same graph × cost × budget cell; 1 seed.
    - The frozen base model with the new interface, zero-shot.
    - N2 and N0 retrained on the identical states.
    - The N1 ranker architecture as the learned numerical control.
  - **Stage 2** (only if Stage 1 passes the gates): leave-one-fix-out, 1 seed each, with a second seed if the effect is below 2× the Stage-1 seed SD:
    - −codes: original ids, fixed order, trie-greedy decoding, listwise loss over whole-command log-probs.
    - −compact: L0 serialization, compared at matched steps.
    - −soft: hard labels.
    - −capacity: rank 8 on q/v.
    - −volume: 32 worlds. Compared with N2 at 32 worlds, this also measures any sample-efficiency effect of pretraining.
  - Gates G1–G8 run on validation worlds. Every 500 steps a monitor aborts the run if any action exceeds 60% of decisions on three consecutive checks.
- **Sample size:** about 12k states (comparable to Qiu et al.'s 6,240 examples and Paprika's 17,181 trajectories), 3 seeds (the minimum to estimate seed spread; Kriz et al. flag single seeds), 256 test worlds as in E4.
- **Endpoints:** closed-loop test-world regret, J/J_TR, parity with N2, the gate profile, and ablation deltas.
- **Decision rule:** pass = G1–G8 + PQ-weak + parity.
  - **Positive:** a small LLM reads an explicit SCM posterior and chooses planner-quality interventions on held-out systems, with leave-one-fix-out attribution. This meets submission requirement 1 for this class. It is not an advantage, because the planner is the ceiling here.
  - **Null:** G1–G4 pass, but G5 fails or regret falls in 0.15–0.30. The LLM learned a coarse schedule; go to E7.
  - **Negative:** G2–G4 fail while N2 passes. The language interface is the bottleneck; go to E7, and if E7 fails, A2.
  - A pass obtained only with L3 statistics does not count.
  - If the LLM beats N2 at 32 worlds by at least 0.05, report a pretraining sample-efficiency effect, not an advantage claim.
- **Prerequisites:**
  - E3 engine, E4 corpus and N*, E1 targets.
  - A GPU ceiling or local training approval. `gpu-testing` has a 1 h wall and resumed optimization is unqualified [V], so runs longer than about 45 min need a longer-wall QoS or the Mac.
- **Cost:**
  - A-full run: 24.6k example-steps at 0.05–0.2 s per step [A] = 0.34–1.4 h. Basis: the old SFT update took 0.62 s at batch 1 on 2–4k-token prompts with strict determinism and checkpointing; L1 prompts are about 600 tokens and minibatched. A pilot fixes the figure.
  - Stage 1: 1.6–6 h. Stage 2: 2–8 h; −compact uses about 5× longer prompts at half the steps.
  - Evaluation: about 110k single-forward decisions, 0.6–3 h.
  - Total about 4–17 GPU-slice-hours; 2–4× longer on the Mac [A]. Calendar: 6–9 days.
- **Main risk:** number reading by a 0.6B model may cap performance; LoRA results vary across seeds at this data scale. Critics may say L1 makes the task easier. The −compact ablation and E1's sufficiency argument, which does not depend on the LLM, answer that.

### E6. Closed-loop on-policy correction (should)

- **Question:** Does the policy stay planner-quality on its own state distribution, and does on-policy relabeling remove compounding error?
- **Design:**
  - Triggered if E5 passes at state level but closed-loop regret exceeds 0.15 or G8 fails. Otherwise run one confirmatory round.
  - Three DAgger rounds. Each round rolls out in 256 fresh training-scope worlds with planner-action mixing β = 0.5, 0.25 and 0, relabels every visited state with the reference, aggregates, and fine-tunes one epoch.
  - Alternative loss arm: the exact expected-score objective −Σ_a π(a|s)·s̃(a|s), plus entropy 0.01, on on-policy states.
  - Controls: the E5 checkpoint, and N2 with identical DAgger rounds.
- **Sample size:** about 6k new states per round, about 50% of the base corpus. DAgger usually settles in a few rounds.
- **Endpoints:** closed-loop test regret, J/J_TR, stress-state regret, longest repeated-action run.
- **Decision rule:** adopt the smallest on-policy procedure that passes PQ-weak and G8 on test worlds.
  - **Positive:** on-policy relabeling is reported as necessary, or as unnecessary if round 0 already passes.
  - **Negative:** both arms fail while state-level performance passes. This is A3, and the claim narrows to state-level imitation; go to the hybrid.
- **Prerequisites:** E5 state-level pass; the E4 rollout harness with the LLM in the loop.
- **Cost:** rollouts of 18.4k forwards (0.1–0.4 h); relabeling 18.4k × 0.119 s ≈ 0.6 CPU-hours; fine-tuning 1–5 h. Total 1.5–6 GPU-slice-hours. Calendar: 3–4 days.
- **Main risk:** aggregation over-weights unusual states and degrades the primary distribution.

### E7. Capacity and interface escalation, and the direct-versus-hybrid decision (should)

- **Question:** If the repaired 0.6B policy fails where the numerical student succeeds, does a predeclared escalation fix it, or should the LLM move to the hybrid role?
- **Design:**
  - Triggered by an E5 null or negative, or an E6 failure.
  - Order: (a) L2 text, with N3 as its twin; (b) rank 32 on all layers with 2× data; (c) a linear action head on the decision-position hidden state, only if E2 found D4 present; (d) Qwen2.5-1.5B-Instruct, cached, with approval.
  - One seed each; the first passing arm gets 2 more seeds.
  - The hybrid H-prior is specified now (2.9), to be used in E9.
- **Sample size:** E5's data and test worlds.
- **Endpoints:** E5's pass criteria.
- **Decision rule:**
  - The first escalation that passes becomes the frozen configuration. The paper reports it as the minimum capacity or representation needed.
  - Otherwise apply A1–A3. The direct-policy claim becomes a documented failure, with numerical students as positive controls, and the LLM arm in E9–E10 is H-prior.
  - (c) must be described as "LLM encoder plus linear action head".
- **Prerequisites:** E5 and E6 outcomes; approval of the 1.5B weights.
- **Cost:** (a)–(c) 0.4–2 slice-hours each; (d) 1–5 h. Total 3–12 GPU-slice-hours. Calendar: 3–6 days.
- **Main risk:** forking paths. The fixed order, one seed per arm and the single-confirmation rule control this.

### E8. LLM-independent headroom audit to select the advantage class (should; runs in parallel from week 1)

- **Question:** In which candidate class could any policy beat target-risk planning by a resolvable margin? Is the planner's decision mapping learnable there? Both are judged without running an LLM.
- **Design:**
  - Candidate classes:
    - (i) partial measurement with large cost ratios on 4–6 nodes;
    - (ii) informed priors from prose, with 0–2 initial observations and varied text;
    - (iii) wrong-known or unknown noise;
    - (iv) graph uncertainty, only if (i)–(iii) fail, because it is a large change.
  - Per class, 64 synthetic training-scope systems with: target risk, plan-once static, best public static schedule, a rollout planner, an oracle-prior planner, an uninformed planner and N2.
  - Criteria:
    - H1: J(plan-once)/J(sequential) ≥ 1.10.
    - H2: J(uninformed)/J(oracle prior) ≥ 1.15.
    - H3: J(target risk)/J(rollout) ≥ 1.05.
    - H4: at most 150 systems for 80% power on a 10% difference on the class's primary endpoint, with common random numbers.
    - H5: N2 reaches PQ-weak with at most 2,048 worlds.
- **Sample size basis:** with 64 systems, a 10% J difference has z ≈ 4.5 at SD 0.171 and about 22 at SD 0.035. A 5% difference has z ≈ 2.3 and 11.
- **Endpoints:** H1–H5 per class.
- **Decision rule (fixed now):**
  - Among classes passing H4 and H5, pick the largest H2 if the hypothesized LLM contribution is semantic; otherwise the largest max(H1, H3). Ties go to the lower code cost.
  - No class passes: the paper reports that no tested class offers resolvable headroom, and E9–E10 are not run.
  - The selection uses no LLM output, so it satisfies the standing rule on benchmark changes.
- **Prerequisites:** approval of protocol code per class; a pip install of `causalchamber` only if (ii) uses its simulators.
- **Cost:** CPU only, about 1–4 CPU-hours per class for the rollout planner plus E4-scale student fits. Code: 2–5 days per small or medium class. Calendar: 1–3 weeks.
- **Main risk:** the class with the most headroom is the costliest to code. The thresholds are judgement calls, fixed now.

### E9. Privileged-teacher training, gates and ablation arms in the selected class (should)

- **Question:** Does the frozen recipe transfer to the selected class? Do the arms that separate language pretraining, ACE-specific training, causal structure and extra computation pass the same gates?
- **Design:**
  - Teacher: the planner with privileged training-world information (oracle prior or true graph), or the rollout planner. The student sees only public text and data. Privileged distillation is followed by DAgger.
  - Arms:
    - the LLM direct policy, with the frozen recipe and only the lr re-chosen from two values;
    - H-prior;
    - N2 with identical data and rounds;
    - the frozen base LLM;
    - a text-ablated LLM (neutral names, no descriptions);
    - a structure-ablated LLM (graph withheld or permuted);
    - a non-LLM text prior feeding the planner;
    - a compute-matched planner (more draws or lookahead).
  - G1–G8 on validation worlds. Then freeze hashes, namespaces, endpoints and failure treatment.
- **Sample size:** E5 scale per arm; 256 validation worlds.
- **Endpoints:** the gate profile, and PQ relative to the privileged teacher.
- **Decision rule:**
  - Only arms that pass the gates enter E10.
  - If neither LLM arm passes, there is no advantage test, and the paper reports transfer failure.
  - If the text-ablated LLM equals the full LLM on validation, the semantic-contribution hypothesis is already falsified, and E10 is framed accordingly.
- **Prerequisites:** E8 selection; the E5–E7 recipe; a GPU ceiling.
- **Cost:** 6–20 GPU-slice-hours in total for the LLM and ablation arms; numerical arms on CPU. Calendar: 2–3 weeks.
- **Main risk:** the privileged teacher may be unattainable from public information. Prior leakage is possible if the text generator is not separated from the parameter draw.

### E10. Preregistered fresh-system advantage test (should)

- **Question:** On fresh systems, does the LLM arm (direct or hybrid) reach a reliable answer at lower experimental cost than strong numerical policies given the same information?
- **Design:**
  - Preregistered before any system is generated.
  - The frozen E9 arms plus target risk with the same information, the rollout planner, N2, the frozen LLM, N0 and the ablation arms.
  - Fresh IDs. Common random numbers across arms, which requires lifting the per-arm namespace rule at `paired_episode.py:114-117`, or replicate episodes per system.
  - A shared public stop rule. Primary endpoint and n from E8's H4 (at most 150 systems). Censoring and failure treatment fixed in the preregistration.
- **Sample size:** from H4.
- **Endpoints:** the preregistered primary; decision-level regret and realized error as secondaries.
- **Decision rule:**
  - Superiority over target risk with the same information at one-sided α = 0.05, with Holm correction across the predeclared contrasts. A non-inferiority margin for the hybrid against its own planner, fixed in the preregistration.
  - **Positive:** submission condition 2 is met for this class.
  - **Null:** a bounded negative, reported with its interval.
  - **Negative:** the claim narrows. There is no replacement benchmark.
- **Prerequisites:** E9 freeze; custody redesign (sharding); a compute ceiling. The ledger gives 22–35 core-hours and 1.9–2.9 slice-hours for 100–150 systems under the old process model.
- **Cost:** ledger scaling plus about 21.6k LLM decisions. Calendar: 6–10 days.
- **Main risk:** censoring or underpowering, and failure modes that appear only on fresh systems.

## 4. Sequence, gates and parallelism

- **Week 1:**
  - E1 starts at once; it needs no approval.
  - E2 starts after local inference is approved.
  - E4 corpus code and E8 class scoping start in parallel. Both are CPU-only and use no LLM.
  - **Gate A:** E1 precision and rounding gates pass. E2 fidelity reaches at least 99%, and D1–D5 are written up.
- **Weeks 2–3:**
  - E3, which needs the engine, runs alongside E4.
  - **Gate B:**
    - E4 gives PQ, N* and the representation level, or triggers A1. A1 stops all LLM direct-policy training in this class.
    - E3 gives the attribution, or shows the collapse was data-limited.
- **Weeks 3–5:**
  - E5 Stage 1, then the gates, then Stage 2. E6 if triggered; E7 if triggered.
  - **Gate C:** the direct-versus-hybrid decision (A2–A4).
- **Weeks 4–8:**
  - E8 finishes; its selection rule is fixed in this document.
  - E9, then preregistration, then E10.
- **Venue fit:**
  - ICML 2027 (abstract January 14, paper January 22) realistically holds E1–E6: the diagnostic plus learnability paper.
  - E8–E10 more plausibly target UAI 2027 (about February), NeurIPS 2027 or TMLR.
- **GPU staging:**
  - E2 needs 0–1.8 h; E3 needs 1.5–3 h.
  - E5–E7 need 8.5–35 h, with hard stops at gates. E5 Stage 1 is the only commitment before Gate C.
  - E9–E10 are decided after E8.

## 5. Supporting calculations

All computed with `calc/calc.py` (stdlib only) unless marked.

1. **CSC power (E2):** n = ((z_{1−α/2} + z_{power}) / atanh ρ)² + 3. For ρ = 0.2 and α = 0.01, n = 287 for power 0.80 and 320 for 0.85. At n = 300 the power is 0.821. For ρ = 0.3, n = 125. The 95% half-width near zero at n = 300 is 0.113. An observed 0.30 has a lower bound of 0.193.
2. **Test-world precision (E4, E5):** for a ±0.02 half-width on mean regret, n = (1.96·SD/0.02)²: 216 worlds at SD 0.15 and 384 at SD 0.20 [A on SD; measured on validation worlds]. Log J-ratio SDs implied by the 24-system design-quality t-values [V]: local information 0.035, coverage 0.171, random 0.235, ranker 0.264, DPO 0.466. The corresponding half-widths at n = 256 are 0.004, 0.021, 0.029, 0.032 and 0.057, against log 1.05 = 0.0488.
3. **Planner relabel cost:** 499 s / 4,203 states = 0.119 s per state at 16,384 draws (single process, this Mac [V]). Corpus labelling at N = 32, 128, 512 and 2,048 (about 65 states per world) takes 0.07, 0.27, 1.10 and 4.39 CPU-hours; validation plus test adds 1.10 CPU-hours.
4. **LLM steps (E5):** 512 × 24 × 2 = 24,576 steps → 0.34–1.37 h at 0.05–0.2 s per step. 256 × 24 × 3 = 18,432 → 0.26–1.02 h.
5. **Forward-only budget (E2):** 5,574 + 10,800 = 16,374 forwards → Mac 1.4–6.8 h at 0.3–1.5 s [A]; CURC 0.55–1.82 h at 0.12–0.4 s [V].
6. **Label noise:** P(|t₃| > 2) = 0.1393 (numerical integration; matches the notes). A strict label with flip rate 0.31 carries 1 − H₂(0.31) = 0.107 bits.
7. **Soft-target bound:** with r ≥ 0 and r_best = 0, Z ≥ 1, and r·e^{−r/τ} ≤ τ/e, so E_{p*}[r] ≤ (K−1)τ/e. For K = 8 this is 0.0129 at τ = 0.005, 0.0258 at 0.01 and 0.0515 at 0.02. The empirical rule in 2.1 will usually allow a larger τ than the worst case.
8. **LoRA sizes** (Qwen3-0.6B: hidden 1,024, q output 2,048, k/v output 1,024, MLP 3,072, 28 layers): rank 8 on q/v = 1,146,880, which equals the saved count [V]; rank 16 on all layers = 10,092,544; rank 32 on all layers = 20,185,088.
9. **Headroom audit (E8):** with 64 systems, z for a 10% J difference is 4.46 at SD 0.171 and 21.8 at SD 0.035. For 5% it is 2.28 and 11.2.
10. **Retrospective gate numbers [V]:**
    - G2 shares: SFT 0.844 and DPO 0.961 on training states; frozen 574/576, SFT 481/523 acquisitions, DPO 576/576 and ranker 435/576 in deployment.
    - Target risk 215/560 = 0.384 and local information 318/570 = 0.558 pass.
    - The planner-best max share on the LLM arms' own states is 0.30 (DPO), 0.32 (SFT) and 0.35 (frozen), so the G2 threshold is 0.6 for all three.
    - G3: 36/128 and 32/128 against the required 0.40.
    - G4: 2.37 against the required 1.73.
11. **Exact on-policy gradient:** for a single-token action and known per-action scores, ∇_θ E_{a∼π_θ}[s(a)] = Σ_a s(a)·∇_θ π_θ(a). Group sampling (GRPO) estimates this sum with variance and adds nothing. The state distribution is the only on-policy element.

## 6. Asks for the lead

1. Approve local forward-only inference on the Mac (base Qwen3-0.6B plus the two saved adapters) under a written protocol, accepting runtime differences with a ≥ 99% reproduction gate. Alternatively, set about 2 GPU-slice-hours on CURC for E2.
2. Approve generation of a training-scope synthetic learnability corpus with new world IDs only, and local-CPU planner relabeling (E1, E4, E6). Synthetic responses are reported as training cost.
3. Set a staged GPU ceiling (E3 about 3 h; E5–E7 up to about 35 h, with hard stops at the predeclared gates), or approve local LoRA training on the Mac. Approve a QoS with more than 1 h wall (or a full A100), since resumed optimization across jobs is unqualified.
4. Approve the new training-engine version and make the gate suite G1–G8 mandatory acceptance for every trained policy, replacing legality-only acceptance.
5. Approve redirecting feedback in the known-graph class from measured 4-replicate preferences and DPO to planner relabeling plus on-policy relabeling. This is a method redirect justified by the label-noise evidence.
6. Conditionally approve Qwen2.5-1.5B-Instruct (already cached) as E7 escalation (d). Rule now whether 7B is ever in scope.
7. Approve the E8 selection rule now, before any LLM result in a new class, and the protocol code for up to three candidate classes. Approve the `causalchamber` install only if class (ii) uses its simulators.
8. Optionally, approve the truth-conditional analysis using the 32 private training-world descriptors (E1).
9. Rule whether E3's single-pass cells count as "rerunning the old pipeline". They use a new engine, no DPO and no environment responses.
10. Set the venue and date, and whether a diagnostic-plus-learnability paper is acceptable if E8–E10 cannot finish or are negative.

## 7. What not to do

- Do not collect more 4-replicate measured preferences, and do not scale DPO in the known-graph class.
- Do not select recipes, temperatures, gates or interfaces on the 24 development systems. Development prompts are for E2 forensics only.
- Do not put per-action planner statistics in the LLM prompt as a candidate arm. They are an upper-bound control, reported with the argmax-of-prompt baseline.
- Do not keep greedy trie decoding over multi-token commands, fixed menu order, fixed identifiers or alphabetical tie-breaks.
- Do not deploy or roll out any checkpoint that has not passed G1–G8. Never train with a single unshuffled pass at batch 1.
- Do not add caps, diversity fills, collapse breakers or fallbacks. Provenance must stay 100% model-chosen.
- Do not treat the old exposure-matched ranker as the learned numerical control. Use converged N1 and N2.
- Do not claim an advantage in the 3-node class. Regret against the planner cannot show one, and Finding A bounds the room at a few percent.
- Do not use first-crossing cost on 24 systems as an endpoint for learnability.
- Do not escalate model size before the 0.6B repaired recipe has been gated, and never outside the predeclared E7 order.
- Do not change world class because the LLM failed. Classes are chosen only by the E8 rule.
- Do not rerun the 24-system screen or the old collection, training or campaign pipelines.

## 8. Biggest uncertainties

- Whether a 0.6B model can read rounded numeric posteriors from text well enough. E2's probes and E5 decide this, and the number-handling literature is pessimistic.
- Whether the 3-node class has enough posterior-driven state dependence beyond coarse features for the ladder to discriminate. N0 may approach the planner.
- The speed and fidelity of local MPS execution, and the true per-step training cost. These are unmeasured, and all GPU figures here are ranges with a pilot.
- Whether any class with resolvable headroom can be coded in time. In that class, does the LLM's contribution enter through the action (direct policy) or through the model (hybrid)?
- Whether the privileged or myopic teacher in the selected class is a good enough target. If not, learning from realized outcomes becomes necessary, and that is noisy.
- Variance across LoRA training seeds at about 10k states, and the training-cost accounting against deployment savings.

## 9. Theory items worth deriving

1. **Belief-state sufficiency.** In the known-graph linear-Gaussian known-noise class, the target-risk score and the Bayes-optimal sequential policy depend on the history only through the conjugate posterior (plus costs, budget and task). This justifies L1 without reference to the LLM, and it makes the C3 history edits a clean test of spurious dependence.
2. **Trie-greedy versus whole-command argmax.** Greedy picks the branch with the largest subtree mass M(b) = Σ_{c∈b} p(c). Derive the region where `observe_all` is the argmax command yet loses because p(observe) < M(set). Show that single-token codes remove the gap.
3. **State-blind collapse lemma.** A model whose output does not depend on state minimizes listwise cross-entropy at the average target p̄, and argmax decoding then gives a constant. With token-factorized logits and suffix tokens shared between X and Y, the argmax follows per-node pluralities, which explains `set_x_neg` from pooled `_neg` counts (32 + 10 against 29 + 10). This formalizes the 8-logit model.
4. **Label-information bound.** The 4-replicate, 2-SE rule gives α = 0.139, and a 31% flip rate leaves 0.107 bits per strict label. Contrast this with planner labels, whose flip probability is Φ(−Δ/σ_D) outside a declared tie band, and derive the D needed for at most 1% flips.
5. **Exact on-policy gradient for single-step known rewards** (section 5.11). Combined with DAgger's no-regret guarantee, this shows why RL adds nothing in this class beyond on-policy states.
6. **Regret-to-design-quality bridge.** Under local additivity of one-step risk reductions, which approximately holds here because plan-once ≈ sequential, bound J_π/J_TR − 1 by the gain-weighted mean relative regret. Calibrate it with local information: regret 0.154 corresponds to ×1.02. This gives the decision-level gates an outcome meaning.
7. **Gibbs soft-target regret bound** E_{p*}[r] ≤ (K−1)τ/e (section 5.7). It guarantees the training target is itself planner-quality.

## 10. Sources and checks

- Brief: `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w2/brief.md`.
- Notes read: `../w1/training-forensics.md` (all), `training-forensics-verify.md`, `deployment-forensics-verify.md`, `training.md`, `mechanism.md`, `numerics-protocol.md`, `ledger-ops.md` §(b) and §(g), `literature-policy.md` (targeted), `prior-evidence-core.md` (targeted).
- Verified here:
  - Single-letter codes are single tokens: `A`–`H` ids 32–39, `ĠA`–`ĠH` single ids, in `/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json` (JSON lookup, no model loaded).
  - The verifier's relabel timing, from `../w1/deployment-forensics-verify/v09.log`.
  - The LoRA parameter recount.
- Assumptions [A]: Mac MPS speed; per-step LoRA training cost with compact prompts; world-level SD of regret; N0's eventual performance; the planner's interaction share. Each is settled by a pilot or by the first measurement in the experiment that needs it.
