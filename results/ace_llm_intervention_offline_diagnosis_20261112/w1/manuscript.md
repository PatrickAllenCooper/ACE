# Notes: what the current ACE manuscript is, and what the broader paper would need

Label: `manuscript`. Prepared 2026-11-12. Read-only work. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was changed. No model, training, SLURM, SSH or launch script was run. Offline arithmetic on saved CSVs only.

Conventions: **V** = verified by me in the cited file. **I** = my inference. **D** = my own descriptive computation from saved files (post hoc, unregistered, not a manuscript claim).

Scratch scripts and outputs (all under `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/manuscript/`):
- `recompute_prospective.py` / `.out`: reproduces every prospective number in the manuscript from `paper/aistats_ace_2027/delivery_prospective_all_cells.csv`.
- `history_effect.py` / `.out`: paired balanced-versus-random collection effect on raw held-out MSE.
- `screen_power.py` / `.out`: paired differences and rough sample sizes for the 24-system LLM screen (`reporting01/data01/cells.csv`).
- `endpoint_robustness.py` / `.out`: policy ranking of the 24-system screen under alternative endpoints (`curves.csv`).

---

## 0. One-paragraph answer

The open manuscript `/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex` is a TMLR-format, anonymous, 26-page-class empirical paper titled **"From Experimental Histories to Causal Surrogates: Data Reuse, Optimization, and Error Propagation"**. It is about **post-collection refitting ("delivery") of SCM surrogates on a fixed paid history**, with known graph and measured intermediates. It contains **no LLM, no DPO, no acquisition policy and no experimental-cost result**. Its only statements about acquisition are four explicit disclaimers. It is byte-identical to the archived tag `archive/delivery-scm-2026-11-02` (commit `661db7ac5`). It was put on submission hold on November 2 and reclassified as a "supporting component study". To become the broader paper, every headline element is missing: the LLM policy method, a positive independent LLM contribution, strong-baseline policy comparisons on fresh systems, LLM-specific ablations, an experimentalist task, policy theory, policy related work and data figures. What can carry over is the evaluation discipline, three elementary propositions, the final-refit control arm, and several negative controls.

---

## 1. File anatomy (what lines 1 to 3019 contain) [V]

`paper.tex` has 4,559 lines, SHA256 `0c5527681a41eb7d3a0fa16b96f214f7fbe6e978afb63b2a45d0a9010482df70`.

- Lines 1-5: header comments ("TMLR manuscript; accepted prospective replay and both reports are integrated").
- Lines 6-3015: a generated **"EDITOR COMPANION BUNDLE"** of 16 `filecontents*` blocks, each preceded by a SHA256 comment. All 16 embedded hashes equal the standalone files on disk (I recomputed with `shasum`).
  - Style and bibliography boilerplate, lines 8-2117: `tmlr.sty` (8-215), `fancyhdr.sty` (217-674), `tmlr.bst` (676-2117).
  - `delivery_references.bib` (2119-2182): 11 entries.
  - `delivery_claims.tex` (2184-2221): 35 generated macros for confirmation, attribution (A) and physical (C).
  - `delivery_attribution_table.tex` (2223-2252), `delivery_history_table.tex` (2254-2279), `delivery_physical_table.tex` (2281-2327).
  - `delivery_theory.tex` (2329-2461): Section 8, three propositions.
  - `delivery_method_diagram.tex` (2463-2510): the only figure, a qualitative schematic.
  - `delivery_evaluation_design.tex` (2512-2530): Table 1.
  - `delivery_prospective_claims.tex` (2532-2556): 22 generated macros for the prospective study (B).
  - `delivery_prospective_primary_table.tex` (2558-2575), `..._sensitivity_table.tex` (2577-2602), `..._absolute_table.tex` (2604-2846), `..._descriptive_table.tex` (2848-3014).
- Lines 3016-3019: `\documentclass[10pt]{article}`, `\usepackage{tmlr}`, packages, `\newtheorem`.
- Line 3022: `\title`. Lines 3039-3061: abstract. Narrative body 3063-3701. Appendices A-G 3706-4557.
- Only five companions are `\input` (claims x2, diagram, design table, theory). The result tables are inlined as "EDITORIAL TABLE" copies that relabel arms as rolling/flat. The generated companions and machine-readable files keep the original labels `online`/`simpler`.

Counts [V]: 23 tables, 1 schematic figure, 0 data figures, 0 `\includegraphics`, 5 propositions (3 in Section 8, 2 in Appendix G), 11 references, about 5,700 words of main text.

Section map: 1 Introduction (3063) | 2 Setting and delivery procedure (3117) | 3 Related work (3184) | 4 Completed delivery confirmation (3236) | 5 Exploratory attribution (3272) | 6 Frozen prospective evaluation (3390) | 7 Physical measurements (3553) | 8 Conditions and limits of delivery, theory (input at 3626) | 9 Reproducibility and compute accounting (3628) | 10 Discussion and limitations (3647) | 11 Conclusion (3688) | App A implementation (3707) | B prospective systems (3849) | C physical split (3893) | D execution accounting (3934) | E provenance (3963) | F complete prospective reporting (3994) | G quantized targets (4489).

Git state [V]: `git log -- paper/aistats_ace_2027/paper.tex` last commit is `661db7ac5` (2026-11-02). `git diff archive/delivery-scm-2026-11-02 HEAD -- paper/` is empty. Working tree clean for `paper/`.

**Hazard [V]:** `paper/aistats_ace_2027/paper.pdf` (384,018 bytes, created 2026-09-10, 30 pages, untracked because `.gitignore` excludes `*.pdf`) is **not** a build of the current source. Its first page reads "ACE: Language-Model Priors and Lookahead Selection for Active Causal Mechanism Estimation", the pre-audit AISTATS draft whose ACE-versus-baseline numbers were later invalidated. `paper.log` and `paper.blg` are from that same old build (`aistats2026.sty`, `apalike.bst`, `references.bib`). The only page-reviewed PDF of the delivery paper is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-TMLR-submission-review-20261102-01/ACE_manuscript.pdf` (26 pages, SHA `1e44419f...`), and that export predates the final archived wording and mathematics changes (`delivery_paper_archive_2026-11-02.md` line 13).

---

## 2. (a) Title, abstract, contributions and quantitative claims

**Title** (paper.tex 3022): From Experimental Histories to Causal Surrogates: Data Reuse, Optimization, and Error Propagation.

**Abstract** (paper.tex 3039-3061, the authors' own text): "We assess post-collection refitting of structural causal model (SCM) surrogates with known graphs and measured intermediates. A prospective study uses twenty independent systems in each of two noise-disabled families, with five or thirty nodes. Its registered analysis weights balanced and matched-random collection histories equally within each system. At shared initialization zero, the pooled results favor delivery over a flat refit in both families and over rolling learning in the thirty-node family. Five-node delivery instead has 2.46 times the rolling error, a reversal whose cause remains undiagnosed. The flat comparison also depends strongly on collection history. For thirty-node balanced histories, the delivery/flat error ratio is 0.912 and delivery improves only 10 of 20 systems, compared with a ratio of 0.168 for matched-random histories. Across delivery, rolling and flat in the two families, median root-mean-square errors are about 0.5% to 3.4% of the training target's standard deviation. Analyses of recorded emulator runs favor long fits on retained histories. In a separate physical dataset, Fourier regression beats delivery in all eleven conditions. Conditional theory explains why eligible-row fitting, joint support and prediction through a learned chain must be considered separately. These completed comparisons show that the benefit of a final refit depends on the task, comparator and collection history. Individual fits can remain unstable across initializations, and training budgets differ across methods."

(The private TMLR worksheet `delivery_submission_form_draft_2026-11-02.md` carries the slightly older abstract of revision `2deb0b38`. Its last three sentences differ.)

**Stated contributions** (paper.tex 3093-3115):
1. Main evidence: a frozen prospective comparison across forty independently parameterized systems, all four registered contrasts and every system retained.
2. Context: two analyses of recorded emulator runs (repeated histories of one emulator, plus an exploratory scratch-fit matrix), and a physical case study where a Fourier predictor beats neural delivery in every condition.
3. "Elementary eligibility and propagation results". The text says they "are not new identification theorems".
4. Practical lesson: evaluate the delivered refit separately from the rolling learner, keep a simple direct or domain-informed control, score the complete prediction chain. "The evidence does not justify an automatic switch to a long SCM refit."

**Quantitative claims, tagged.** Status labels follow the manuscript's own Table 1 (paper.tex 2512-2530) and Section 6.3.

| # | Claim (exact values) | Status | Evidence source | My check |
|---|---|---|---|---|
| B1 | 5-node delivery/rolling NMSE ratio 2.45956, marginal 95% CI [1.78782, 3.38368], Holm p 3.29547e-05, gate **not met** (adverse) | Confirmatory, registered | `delivery_prospective_claims.tex`; `delivery_prospective_claim_index.json` (selectors into `acceptance.json`); paper.tex 3511 | Reproduced exactly from all-cells CSV |
| B2 | 5-node delivery/flat 0.356194 [0.282389, 0.449288], Holm p 6.62773e-08, met | Confirmatory | same; 3512 | Reproduced |
| B3 | 30-node delivery/rolling 0.274321 [0.158409, 0.47505], Holm p 0.000185887, met | Confirmatory | same; 3513 | Reproduced |
| B4 | 30-node delivery/flat 0.391027 [0.254548, 0.600681], Holm p 0.000205416, met | Confirmatory | same; 3514 | Reproduced |
| B5 | Design: 20 systems x 2 families, 2 histories of 400 responses each, 400 shared held-out root actions, init 0 fixed, 640 fits saved before held-out responses, gate = ratio <= 0.8 and upper CI < 1 and Holm p < 0.05 | Design fact | paper.tex 3399-3489 | 640 rows, arms delivery/online/simpler/ablation confirmed |
| B6 | History split, delivery/flat: balanced 0.793 (13/20) and 0.912 (10/20); random 0.160 (19/20) and 0.168 (18/20). Delivery/rolling: balanced 1.614 (4/20) and 0.695 (12/20); random 3.748 (3/20) and 0.108 (19/20) | Descriptive | paper.tex 3522-3534, 4449-4456; `delivery_prospective_history_specific.csv` | Reproduced |
| B7 | Median RMSE as % of training-target SD: 2.2, 1.6, 3.4 (5-node delivery, rolling, flat); 0.53, 0.85, 0.87 (30-node) | Descriptive | paper.tex 3536-3542 | Reproduced (2.178, 1.643, 3.378; 0.531, 0.851, 0.869) |
| B8 | Per-init ratios (Table at 4053-4058): 5 rolling 2.460/2.271/1.455, seed-avg 2.011; 5 flat 0.356/0.163/0.188, 0.222; 30 rolling 0.274/0.281/0.220, 0.257; 30 flat 0.391/0.360/0.264, 0.333 | Descriptive | paper.tex 4049-4066 | Reproduced |
| B9 | Worst/best delivery NMSE across 3 inits within a history: median 2.02 (5-node), 5.23 (30-node). System 5:08, matched random, init 1: delivery NMSE 0.02384, flat 0.04815, rolling 2.2844e-5, ratio 1043.55 | Descriptive | paper.tex 4068-4074 | Reproduced (2.018, 5.231; 0.023839, 0.048153). 90th percentiles 11.4 and 21.7, maxima 225 and 152 are not in the paper |
| B10 | Delivery/short-fit (100 updates): 0.00855424 (5-node), 0.0186544 (30-node). Floor hits 0 | Descriptive | paper.tex 4467-4482 | Floor count 0 confirmed |
| B11 | Budgets: rolling 400 x (20+100) = 48,000 updates per head vs delivery 30,000. Heads 3 or 25 vs 1. Params 13,123 vs 4,417 (5-node); 109,465-109,913 vs 4,609 (30-node). 30-node delivery 750,000 head-updates vs flat 30,000. Mean fit CPU s: 58.1/77.9/20.1 (5-node delivery/rolling/flat), 556.7/726.8/22.8 (30-node) | Descriptive | paper.tex 3430-3453 | Not recomputed |
| K1 | Confirmation: 12 histories of one deterministic emulator, geometric mean paired exact-level error ratio 0.183 [0.102, 0.327], exhaustive sign-flip p 0.00146484, 11/12 improve. History 124753321 worsens 0.196339 -> 0.395658 (ratio 2.015177) | Confirmatory but narrow: registered thresholds pass (`pairs12`, `ratio_le_0_8`, `ci_upper_lt1`, `p_lt_0_05`). Exposed grid that includes acquired inputs, one emulator, median of three outcome-scored initializations | `delivery_claims.tex`; `claim_index.json`; `results/delivery_final_history_20261006/{scores,independent_statistics}.json`; paper.tex 3236-3270 | `independent_statistics.json` gives 0.18271874, [0.10197035, 0.32741026], 0.00146484375. Score hash matches claim index |
| A1 | Attribution (480 fits, init 0, exposed grid): all-paid 30k SCM NMSE ratio 0.522 vs online, 0.551 vs selected flat, 0.423 vs matched-CPU flat | Exploratory | `delivery_claims.tex` A* macros; `claim_index.json` attribution block; paper.tex 3301-3338, Table 2 at 3342-3368 | Arithmetic consistent with Table 2 |
| A2 | Scratch factorial: 0.096 vs final-buffer long fit, 0.034 vs all-paid 100 epochs, 0.980 vs admitted long fit (10 histories improve, range 0.644-2.874), snapped ratio 0.469. SCM/flat at init 1 and 2: 0.458, 0.519. Matched-CPU flat (1.234) is worse than the 30k flat (0.948). 10.87 fit CPU-core-hours | Exploratory | same | Consistent |
| A3 | Worsening-history diagnostic: snapped error 0.397, local MSE 8.07e-5, chain MSE 2.86e-4, shift MSE 2.09e-4, 0.020% of true parent vectors outside the bounding box | Exploratory | `claim_index.json` ("scores/124753321.json, all_paid-scm-30000-i0"); paper.tex 3371-3388 | Not recomputed |
| C1 | Physical (`lt_malus_v1`, 11 conditions, 1,000 readings each, 772-815 train, 185-228 held out): delivery has lower row-weighted NMSE than rolling in 7, physics in 3, Fourier in 0. Four neural cells have NMSE > 1. Ten of eleven delivery/rolling bootstrap intervals include one. 0.053 CPU-core-hours | Descriptive. Fits sealed before held-out scoring, but "not independent-world tests" | `delivery_claims.tex` C* macros; `claim_index.json` physical block; paper.tex 3553-3624 | Win counts consistent with Table 4 |
| R1 | Accounting: 60,962 charged confirmation attempts, 57,636 persisted (12 x 4,803), 3,326 interrupted. Prospective chain 132,995 allocated CPU-seconds (36.9431 core-hours) over 59 one-CPU allocations, no GPU. 94.3683 of 150 core-hours reserved | Reporting fact | paper.tex 3640-3645, 3934-3961 | Not recomputed |
| T1 | Prop 8.1 Mechanism eligibility: under invariant additive noise, a policy that depends only on past experiments and independent randomness, and a retention rule with conditional mean-zero noise, regression on retained rows targets f_i on supported parents | Theory, elementary | `delivery_theory.tex` 8-45 (paper.tex 2337-2374) | Read |
| T2 | Counterexample: composing conditional means is not an interventional mean in a noisy nonlinear SCM (X = W + U, Y = X^2 gives 0 vs 1) | Theory | `delivery_theory.tex` 47-62 | Read |
| T3 | Prop 8.2 Deterministic DAG propagation: e <= (I - L)^(-1) eps = sum_k L^k eps | Theory | `delivery_theory.tex` 64-100 | Read |
| T4 | Prop 8.3 Root interventions need not constrain propagation: g_K(R,M) = R + M + K(M - R) agrees with f on all root interventions. With R = 0.25, delta = 0.01, K = 1000 the error is 10.01 inside [-1,1]^2. "An internal intervention can distinguish these mechanisms." | Theory | `delivery_theory.tex` 102-124 | Read |
| T5 | App G: quantization margin, Markov bound Pr(snap error) <= min{1, Pr(M <= tau) + R/tau^2}, counterexamples | Theory | paper.tex 4489-4557 | Read |

Scope strings in the indexes [V]: `claim_index.json` line 43 "12 histories of one emulator, median scored optimization performance, full exposed grid". Prospective index: "delivery recipe against root-target flat refit; supervision and compute differ, not pure architecture attribution", `supervision_and_compute_matched: false`.

---

## 3. (b) Limitations the manuscript itself declares

Section 10 "Scope of the evidence" (paper.tex 3668-3678) plus scattered statements:
1. Prospective results are conditional on two restricted, noise-disabled generator families and a shared initialization scheme (3669-3670). The families "are restricted generator families, not a representative sample of all SCMs" and the family comparison "does not isolate a scaling effect" (3408-3410).
2. Both emulator studies use an exposed grid from one emulator. The grid includes acquired inputs (3243-3245).
3. The confirmation summarizes three scored refits by their median, "not a deployable selection rule" (3250-3251, 3671-3672).
4. Attribution is exploratory, init 0 primary, not an additive decomposition of the online-to-delivery gain (3316-3318). Retention also changes input scaling (3283-3286).
5. Physical study: one mechanism, one apparatus, Fourier beats delivery in every condition, conditions are not independent systems, "not a test of DAG factorization" (3574-3576).
6. Unequal intermediate supervision, parameter counts and fitting work limit architecture claims (3676-3677). "Update counts alone are not compute equivalence" (3437).
7. **"None of these studies measures interventions saved by changing an acquisition policy."** (3677-3678)
8. The five-node reversal "remains undiagnosed" (3047, 3532-3534, 3656).
9. The flat advantage is concentrated on matched-random histories. The 30-node balanced result is "mixed, not evidence of equivalence" (3527).
10. Absolute errors are small and "describe prediction accuracy, not downstream decision value or saved interventions" (3541-3542).
11. Initialization can produce severe individual failures (3546-3551). Intervals condition on one shared initialization (3455-3459).
12. Intervals are marginal and parametric, not simultaneous or distribution-free. The 0.8 gate "is not 95% assurance of a 20% population gain" (3484-3486, 3500-3501).
13. The rolling comparator is retrospectively calibrated on the first 50 rows and "is not a streaming algorithm" (3423-3427).
14. Root-only histories do not identify responses to arbitrary internal interventions (3452-3453).
15. Theory does "not prove identification for arbitrary neural models or certify the fitted networks" (2333-2334). Residual and support diagnostics are "not certified instances" (2428-2429). All experiments use deterministic or noise-disabled systems, so the noisy-case eligibility result is never exercised (2381-2383).
16. Not a new fitting principle. GDumb, experience replay and warm-start precedents are acknowledged (3212-3222).
17. Process evidence is "author-reported", not established by the anonymous supplement (3633-3635). The executable archive is withheld because of named attribution (3637-3638). Whole-project and confirmation CPU are unknown (3642-3643).
18. Broader impact: the emulator's hand-coded scores "do not validate an operational system" and a fitted causal graph is not "a safety certificate" (3680-3686).
19. The sign-flip test assumes symmetry of paired log differences (3263-3264).

---

## 4. (c) Weaknesses and reviewer-style criticisms already on record

### 4.1 `ace_submission_audit_2026-11-01.md`
- Line 5: the submission "is not a demonstration that an LLM chooses more efficient interventions."
- Line 7: the October 24 statement that technical preparation was complete "was too broad". The reproduction ZIP is 361,670,105 bytes against TMLR's 100 MB anonymous supplement limit.
- Line 28: ratios are "not intervention counts or comparisons to random acquisition". "Supervision, parameterization and computation differ between SCM and flat."
- Line 30: neither emulator study "supplies twelve independent physical systems or a deployable initialization-selection rule". Physical: "one apparatus, not broad physical validation".
- Line 34, sharpest: "The principal review risks are audience interest beyond established scratch refitting, limited generator/baseline breadth, and the practical meaning of relative improvements when absolute errors are already small."
- Line 38: foundation exploration. Pretrained prediction was worse than a correct grammar in 5/6 systems, "all six language proposals were invalid", and "No pilot establishes improved acquisition efficiency or reliable LLM intervention selection."
- Line 40: marketing claims about fewer interventions or an LLM policy "must not be imported into this manuscript".
- Line 52: two missing direct precedents were found and added (Sussex et al. MCBO, Chao et al.).
- Line 58: main text extends onto page 13. Lines 68, 70: author-only information unconfirmed. Optional anonymous design specification and abstract streamlining left open.

### 4.2 `delivery_panel_revision_2026-11-02.md` and its synthesis `results/delivery_panel_revision_20261102/panel_synthesis.md`
Six models, two rounds. Round 1 had two "Leaning Reject" (Claude Opus 5.5, Muse). Final: five Leaning Accept, one Accept, and one reviewer still answered No on claims and evidence for the as-submitted text (`panel_synthesis.md` line 9; raw reports under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-TMLR-model-panel-review-20261102-01/`).

Dispositions R01-R17 (panel revision lines 19-35):
- R01 pooled estimand hides history heterogeneity.
- R02 absolute scale and practical interpretation.
- R03 restricted generator population.
- R04 unequal update, head, parameter and CPU budgets. No compute-matched flat control prospectively.
- R05 inference conditional on one initialization, with a severe individual failure (5:08, about 1,044x).
- R06 preserve registered inference.
- R07 attribution scope. The matched-CPU flat schedule is the weaker one.
- R08 undiagnosed reversal.
- R09 confirmation is one emulator, exposed grid, snapped endpoint, scored median.
- R10 paid-row arithmetic was unclear.
- R11, R12 physical sample units, uncertainty and nested controls. 10 of 11 delivery/rolling intervals contain one.
- R13 process evidence versus released evidence.
- R14 reader guide. R15 theory assumptions. R16 clarity and jargon. R17 obsolete claim-index annotations.
- Line 39: "Optional new experiments, checkpoint diagnoses, additional inference and acquisition-policy studies are deferred." None of O01-O06 was run.

Sharpest raw statements (round 1, `round1/claude-opus-5-5.md` and `round1/muse-wonder-1-0.md`):
- Claude line 41: "The delivery recipe appears not to be converged and depends strongly on the random seed."
- Claude line 256: the determining issue is "whether the delivery-versus-rolling contrast can be interpreted at all".
- Claude line 42: "one emulator, two noise-free synthetic families with root-only interventions, and one single-mechanism physical task".
- Claude line 139: "Disclaimers about claims the paper never sets up."
- Claude line 122: "The physical study mainly shows that the neural models extrapolate poorly."
- Claude lines 108-113: a simpler explanation for the five-node flat result. The flat model must learn to ignore an irrelevant root. Rolling also beats flat in 20/20 five-node systems (ratio 0.145).
- Claude lines 69-70: Prop 8.3 "describes almost exactly the five-node situation", yet no propagation evidence is given for the main experiment.
- Muse line 65: the pooled claim "conceals a collection-regime interaction", and the "frozen prospective" credibility claim "cannot be checked against any executable artifact in the supplement".
- AE note line 70-72: ten references then, none on warm-starting, exposure bias or causal experimental design. (Ash and Adams was added afterwards, giving 11.)

Experiments the panel asked for and that remain undone (`panel_synthesis.md` O01-O06; Claude items 2, 3, 7-13): teacher-forced versus free-running error per head from saved checkpoints, residual/shift decomposition, normal-direction sensitivity of the X3 head, training-loss curves, budget-matched or longer delivery fits, learning-rate decay, extra rolling seeds, ancestral-root flat control, classical controls in the prospective study, within-history validation as a deployable seed or model selector, a noisy variant, periodic neural inputs and a training-mean baseline for the physical study, anonymized registration evidence and code.

### 4.3 `delivery_submission_review_2026-10-24.md` and linked `reviews/delivery_integrated_scientific_review_2026-10-24.md`
- Line 3 (later banner): the upload-readiness assessment was superseded.
- Line 24: "No pure-architecture, equal-compute, foundation-model, acquisition-efficiency or universal scaling claim follows." Five-node delivery/rolling 2.45956 "is adverse and fails superiority".
- Line 32: the reproduction archive "is **not strictly anonymous**". Source-history and hash linkage remain possible.
- Line 36: whole-project, confirmation and preparation CPU and the interrupted confirmation's returned-response count "remain unknown".
- Integrated review: one required finding I1 (macro-checker scope, 35 versus 57 macros), closed. R7 (release, anonymity, redistribution) stayed an open gate.

### 4.4 `reviews/delivery_manuscript_readiness_review_2026-11-01.md`
Internal ACCEPT for the scoped contribution, zero required fixes. Strongest remaining concern, line 13: "limited generality". The results "cannot establish an architecture effect, graph-size scaling, or a prospective rule for when refitting will help."

---

## 5. (d) Does the manuscript say anything about an LLM intervention policy?

**No.** [V] In the manuscript body (lines 3016-4559) the terms LLM, language, DPO, Qwen, foundation, pretrain, fine-tun, preference, reinforcement, transformer, "experimental design" and "active learning" occur zero times. The only AI mention is the first-page footnote (3032-3035) disclosing AI-assisted planning, code, analysis and drafting.

Everything it says about acquisition or policies:
1. 3268-3270: the confirmation "establishes neither acquisition superiority, causal-factorization attribution, nor generalization over independently parameterized systems."
2. 3417-3418: the two-history design "tests dependence on the collection strategy without establishing acquisition superiority."
3. 3677-3678: "None of these studies measures interventions saved by changing an acquisition policy."
4. 4003-4004: the appendix tables add no "acquisition-superiority claim".
5. Prop 8.1 (`delivery_theory.tex` 12-14, paper.tex 2341-2343) assumes a generic adaptive policy: "The intervention policy is a function of previous experiments and auxiliary randomness independent of those experiments and the current noise vector. It chooses its action before the current exogenous noises are realized."
6. Related work (3198-3200) cites model-based causal Bayesian optimization as selecting interventions by propagated uncertainty.
7. Appendix A.1 (3757-3773) describes the historical acquisition loop that produced the recorded emulator histories: 200 iterations, four proposed probes per iteration, outcome-based mechanism choice, refreshes. It does not say what proposes the probes.

`claim_index.json` lines 49-55 list `removed_claims`: "acquisition superiority", "foundation-model benefit", "DPO optimality", "unrestricted causal identification", "generic MSE gain equals mutual information". The archive record (`delivery_paper_archive_2026-11-02.md` line 17) says the delivery study "is not evidence that an LLM selects better interventions, that policy fine-tuning improves acquisition, or that ACE saves experiments."

Historical context [V]: the pre-delivery draft preserved at `paper/aistats_ace_2027/archive/paper_before_delivery_20261006.tex` (custody note: "historical manuscript; unsupported claims retained for provenance only") was an LLM-policy paper with a Qwen2.5-1.5B proposer, lookahead and DPO. Its own first contribution was that DPO did not help. `paper/README.md` lines 34-36 say the delivery result "does not rehabilitate" older claims. The user memory note on the September metric audit records that all pre-audit ACE-versus-baseline numbers were cross-evaluator and invalid. Do not cite any number from that draft.

---

## 6. (e) What carries over, and what is entirely missing

### 6.1 Carry-over candidates

| Item | Where | Role in an LLM-policy paper | Caveat |
|---|---|---|---|
| Setting and notation: known DAG, paid history, eligible rows E_i, free-running prediction | paper.tex 3117-3143 | Problem setup for the SCM learner that every policy shares. The LLM screen's learner uses the same eligibility and the same compose-with-predicted-parents rule (`scripts/research/llm_intervention_scm.py` 82-96 and 121-130) [V] | Needs cost, target queries, stopping and policy added |
| Prop 8.1 eligibility under an adaptive policy | `delivery_theory.tex` 8-45 | Justifies updating nonintervened mechanisms from data chosen by any history-dependent chooser, including a frozen-weight LLM [I]. Also names the hazard of response-dependent retention | Elementary. Never exercised on noisy data in the manuscript |
| Prop 8.3 root-only support counterexample, and Prop 8.2 propagation | `delivery_theory.tex` 64-124 | Explains why intervention choice matters: support collapse leaves directions of a mechanism unconstrained, and only some interventions can resolve them. Directly relevant to a policy that emits one action 576 times [I]. In the screen's learner the feature vector is [1, parents] (`llm_intervention_scm.py` 87), so a constant clamp gives a rank-one design for the child mechanism [V code, I consequence] | Existence example, not a diagnosis of any measured failure |
| Counterexample on composed conditional means | `delivery_theory.tex` 47-62 | Warning for any nonlinear, noisy successor benchmark where an SCM planner composes means | Not triggered in the linear-Gaussian screen |
| Delivery as a predeclared final-refit arm | Sections 2, 6 | The agenda's "third comparison" (agenda 129-133) crosses policies with rolling and refit predictors. "If a benefit appears only after extra fitting, report it as a policy-plus-refit result." | Delivery lost to rolling in the five-node family, so the refit cannot be assumed helpful |
| Evaluation discipline: Table 1 design summary, systems as units, Eq. 2 estimand, Holm, three-part gate, all failures retained, charged versus persisted responses | 2512-2530, 3465-3489 | Template for the confirmatory policy study the agenda requires (agenda 159) | None |
| Collection-history dependence | 3522-3534 | Motivation that what is collected changes the delivered model | Descriptive, two non-adaptive designs, unregistered |
| Physical study and Fourier boundary | Section 7, App C | Negative control: a domain-informed function class beats neural delivery 11/11. Points to a possible LLM role (choose the function class) that is untested | Recorded data cannot support new adaptive interventions (agenda 147). October's six language proposals were all invalid |
| Negative controls and retained failures | throughout | Credibility. The agenda says they "warn against attributing a better final predictor to a better intervention policy" (line 17) | None |
| Related work on mechanism models, MCBO, warm starts, replay | 3184-3234 | One paragraph of a larger related-work section | 11 references only |
| Reusable infrastructure named by the agenda (line 25): intervention masks, response accounting, fixed-world descriptors, graph execution, learned-chain evaluation, failure preservation | code | Shared plumbing | None |

Low carry-over value: Appendix G quantization results (only matter for snapped targets), Sections 4-5 on the hand-coded communication emulator, provenance appendices D-E.

Extended related-work prose on Bayesian active causal discovery, in-context decision-making and LLM priors exists only in `paper/iclr_ace_2027/paper.tex` and the archived draft (`paper/README.md` 62-68). Every citation must be re-verified: the README (lines 102-103) records that a September 9 check corrected 5 entries and removed "10 fabricated or duplicate unused entries".

### 6.2 Contributions entirely missing from the manuscript

Mapped to the agenda's five submission conditions (agenda 161-167):
1. **LLM policy method.** No state serialization, action grammar, SFT-then-DPO recipe or frozen-weight deployment appears in the manuscript. It exists only in November guidance documents and code.
2. **Independent benefit over a strong numerical policy on fresh systems.** Absent, and the only executed evidence is adverse (DPO 15.167 vs target risk 11.625). No confirmatory policy study exists.
3. **Ablations separating language pretraining, ACE training, causal structure and compute.** Frozen, supervised, DPO and numerical-ranker arms exist only in the development screen. There was no SCM-free arm (handoff line 23), no graph ablation, no anonymized or misleading metadata arm, no shuffled-feedback arm, no model-size ablation, no compute-matched comparison.
4. **A concrete experimentalist task with a useful outcome, and transfer beyond the synthetic generator.** Absent. The compressor model is "a candidate, not yet an integrated ACE environment" (agenda 143).
5. Preserved failures and accounting exist as practice. Independent scientific review of the policy work does not.

Also missing: an experimental-cost endpoint in the paper, error-versus-cost curves, any data figure, a validated autonomous stopping rule, policy theory (the kappa + 2 epsilon + delta one-step decomposition and the amortization inequality appear only in agenda 45-53), LLM training and inference cost accounting, state-sensitivity evidence that the LLM reads the SCM state, related work on amortized design and LLM acquisition (DAD, Mazzaccara et al., ASIG, LLAMBO, LMABO, Model Discovery Agent, Gupta et al., agenda 31-37), and stronger external baselines (DAD, MCBO-style, exact linear-Gaussian design reference).

---

## 7. (f) Most attackable claims and the experiment that would defuse each

Cost classes: **C0** offline arithmetic on saved files. **C1** analysis of saved checkpoints, no new fits. **C2** new CPU fits on existing histories. **C3** new systems and responses, CPU. **C4** GPU LLM work.

### 7.1 Supporting component (the delivery manuscript)

| # | Claim | Attack | Defusing experiment | Class |
|---|---|---|---|---|
| S1 | Delivery beats flat in both families (0.356, 0.391) | Unequal supervision and compute (750,000 vs 30,000 head-updates, about 109k vs 4.6k parameters, 557 s vs 23 s). Concentrated on matched-random histories (30-node balanced 0.912, 10/20). Simpler explanation: graph-informed input selection. **D:** rolling beats flat 20/20 in the five-node family (ratio 0.145) but only 7/20 in the thirty-node family (1.43) | Flat control restricted to the target's root ancestors. Capacity- and compute-matched flat with a tuned schedule. Flat with intermediates as auxiliary outputs. Report history-specific contrasts as primary | C2 |
| S2 | Five-node reversal (2.46x worse than rolling), "undiagnosed" | The title promises error propagation, yet the main experiment has no propagation evidence. Possibly a convergence artifact: rolling got 48,000 updates per head against 30,000 | From the 640 saved checkpoints: teacher-forced vs free-running error per head, residual/shift decomposition, sensitivity of the X3 head normal to X2 = aX1 + b. Then budget-matched delivery (48k), learning-rate decay, loss curves | C1 then C2 |
| S3 | Intervals at one shared initialization | Seed sensitivity as large as the effects. **D:** worst/best 90th percentile 11.4 and 21.7, maxima 225 and 152. Rolling has one seed | Three or more seeds for every arm. Seed-averaged or ensembled delivery. Within-history block validation as a deployable selector | C2 |
| S4 | Generality | Noise-free, known graph, root-only actions, two hand-built generator families. Prop 8.1 never exercised | Noisy variants and internal-intervention histories. All-paid vs response-selected retention | C3 |
| S5 | Practical meaning | RMSE already 0.5% to 3.4% of SD. No decision endpoint | Learning curves at smaller budgets (25 to 400 responses). A decision-loss endpoint | C2/C3 |
| S6 | Confirmation 0.183 | One emulator, exposed grid with training inputs, snapped staircase endpoint, median of three scored seeds | Fixed-seed held-out re-evaluation. Release per-history values | C1 |
| S7 | Physical 7/11 vs rolling | 10/11 intervals contain one. Four neural cells worse than NMSE 1. Raw-angle encoding handicaps the neural arms | Training-mean baseline, periodic-input neural recipe | C2 |
| S8 | "Frozen prospective" process | Not checkable from the anonymous supplement. Executable archive withheld | Anonymized code and registration hashes | packaging |
| S9 | Novelty | Scratch refit vs warm start is known (Ash and Adams, GDumb) | Position as evaluation discipline, or fold into the broader paper | writing |

### 7.2 Headline claims the broader paper would have to make

| # | Needed claim | Attack, with current evidence | Defusing experiment | Class |
|---|---|---|---|---|
| H1 | The LLM policy lowers experimental cost against strong numerical policies | Current evidence is the opposite. DPO is worst on mean capped cost and emits one action 576 times. The agenda itself concedes "An LLM cannot beat an exact Bayes-optimal decision rule under the same correct model, information and objective" (line 49). With a known graph, linear-Gaussian mechanisms and an 8-action menu, the planner is near-optimal, so there is no headroom | First measure headroom: gap between random, coverage, target risk and an oracle or deep-lookahead planner. Then move to regimes where the numerical planner is handicapped: misspecified mechanism family, large or compositional action spaces, objectives and costs stated in prose, semantic priors that are sometimes wrong | C3 then C4 |
| H2 | The endpoint measures efficiency | **D:** 10/24 systems meet the 0.04 target at initialization (initial error median 0.044, quartiles 0.028 and 0.115), so 11 to 16 of 24 systems are ties in every one of the 28 pairings (`pairs.csv`). First crossing rewards volatile curves: local information has 20 relapses across 11 runs. Under a sustained-attainment endpoint the ranking changes (menu coverage 15.13, target risk 15.67, numerical ranker 15.71, supervised LLM 16.21, frozen and DPO LLM 17.00, local information 17.75, random 18.42) and every paired interval against target risk spans zero | Calibrate the threshold on training systems so that no system starts attained. Pre-register a sustained or area-under-curve endpoint. Report several endpoints | C0 now, C3 later |
| H3 | Policy differences are real | **D:** target risk leads menu coverage by 0.83 cost units (paired 95% interval -0.47 to +2.13) and local information by 0.33 (-2.29 to +2.96). Paired SD is 3.1 to 6.8 units, about 6 for the learned arms. Detecting a 1-unit gain over target risk at 80% power (two-sided 0.05, normal approximation, paired SD 6) needs roughly 280 systems. One system per design cell | Power analysis from development variance, more systems per cell, variance-reducing endpoint, common random numbers | C0, C3 |
| H4 | The SCM matters | No SCM-free arm, no graph ablation | LLM with history only, LLM with shuffled or wrong posterior, model trained without graph access | C4 |
| H5 | The LLM contributes beyond a small learned scorer | **V:** the non-language ranker also collapsed to two actions (`actions.csv`: set_x_neg 141, set_x_pos 435) and scored 13.583, close to the frozen LLM's 13.625. The collapse is not specific to language | State-sensitivity diagnostics on saved training examples and artificial states for all learned arms. Audit teacher action frequencies and preference margins (119 strict pairs used out of 3,584 comparisons) | C0/C4 |
| H6 | Training cost is worth it | Training collection cost 10,240 units before any deployment. A 0.6B model against a closed-form planner | Amortization break-even with measured c_llm, c_num and C_train | C0 |
| H7 | Useful to experimentalists | Three-variable linear toy, fixed menu, constant prose. Retrospective crossing is not autonomous stopping (the supervised LLM stopped early in 23 systems, 5 above target) | Benchmark ladder: nonlinear and misspecified synthetic, an independent interactive benchmark, an executable engineering model. A paid validation stopping rule for all arms | C3/C4 |
| H8 | Not a garden of forking paths | Four prior adverse LLM results in this project: the September audit, Mission-1 (ratio 1.0412, interval 0.8870-1.2221, 2/11 seeds, per agenda 19), six invalid language proposals, and this screen | Pre-registered confirmation on fresh namespaces, with every prior attempt reported | C3/C4 |

---

## 8. (g) Venue, deadline and formatting commitments on record

- **No venue or deadline is committed for the broader paper.** I found none in the November 12 handoff, the November 2 agenda or the November 10 results report. The only "deadline" hits in November guidance documents are job timeouts.
- `paper/README.md` (dated October 6, partly stale): line 4 "no venue or submission deadline is committed". Table line 46 lists AISTATS 2027 with deadline October 6, 2026 and "venue deferred until evidence gates pass". ICLR 2027 draft superseded. NeurIPS 2026 "Submitted, rejected". AISTATS is two-column with an 8-page main-body limit. ICLR was single-column with roughly 10 pages.
- `ace_submission_audit_2026-11-01.md`: TMLR "remains the recommended venue" for the delivery paper (line 5). AISTATS 2027's deadline "is closed" (line 46). TMLR allows at most 100 MB of anonymous supplement (line 7). Authorship cannot change after submission, human authors must verify AI-assisted work (line 48).
- `delivery_paper_archive_2026-11-02.md` line 3 and `delivery_submission_form_draft_2026-11-02.md` line 3: submission hold. The LLM mechanism and experimentalist utility are now prerequisites.
- Handoff line 17: the broader paper "is **not ready for submission**". "Human submission approval remains required."
- Formatting facts [V]: current source is TMLR style (`tmlr.sty` unmodified, 10pt single column), anonymous ("Anonymous authors"), unmarked first-page AI-assistance footnote, title without asterisk. The last page-checked export (November 2) has 26 pages with the conclusion ending and references starting on page 12. The November 1 version ran onto page 13, which the panel's AE note read as a "long" TMLR submission (more than 12 pages of main content). The archived source has had no fresh page review, so its length class is unverified.
- Style commitments: no semicolons or em dashes in visible narrative (panel revision R16, handoff line 124), "use intervention consistently", distinguish a recorded experiment from an archived paper version.
- Author worksheet (`delivery_submission_form_draft_2026-11-02.md` 26-31): two confirmed authors, no funding, no competing interests, no overlapping submission. Both must approve the exact revision.
- Agenda planning values, not commitments: 30-50 systems per family across at least three families "subject to that calculation" (line 159). "Revisit the title, abstract and figures only after the relevant evidence exists" (line 183).

---

## 9. Disagreements between documents (newest wins)

1. `paper/README.md` line 5 says "Stages A/B/C remain pending" and its build notes describe the AISTATS style and `tectonic`. The October 24, November 1 and November 2 documents show all stages complete and a TMLR-format source. README is stale.
2. October 24 review: technical preparation complete. November 1 audit: that was "too broad" because of the 100 MB limit. November 2: paper archived, no submission.
3. Page count: 25 pages on November 1, 26 on November 2, none checked for the archived source.
4. Agenda (November 2) proposed 16 initial plus up to 64 interventions, four regimes of six worlds, seven policies. The executed screen used 8 initial plus up to 24 cost units, 3 graphs x 2 coefficient regimes x 2 noise levels x 2 cost menus with one system each, eight policies. The handoff (line 19) says the executed protocol supersedes.
5. The handoff's action summary names the three LLM arms only. `actions.csv` also shows the numerical ranker used just two actions. Not a contradiction, but an omission that changes the reading.
6. `docs/ACE_evidence_and_public_claims_2026-10-08.md` cites the manuscript's confirmation section at `paper.tex:2776`. In the current file that line is a data row of the embedded absolute-error table (the confirmation section now starts at line 3236). The pointer is stale.
7. The October 8 public-claims note records separate historical PEV acquisition evidence (20 fresh systems: PEV 0.015430 vs random 0.043898, 19/20, p 0.00750 as a secondary comparison, while the primary PEV-versus-coverage comparison failed at p 0.05108). None of that is in the manuscript, and it is a numerical uncertainty policy, not an LLM.

---

## 10. Descriptive computations worth carrying forward [D]

All from saved files. None is a registered analysis.

**Prospective data, rolling versus flat directly (init 0):** five-node 0.145, rolling better in 20/20. Thirty-node 1.43, rolling better in 7/20.

**Best arm per system/history cell (init 0, of delivery, rolling, flat):** five-node: rolling 33, delivery 7, flat 0. Thirty-node: delivery 26, rolling 8, flat 6.

**Collection design effect on raw held-out MSE, balanced / matched-random, paired by system (held-out actions are shared across histories, paper.tex 3415):**

| Family | Arm | Ratio | 95% t interval | Balanced better |
|---|---|---|---|---|
| 5 | delivery | 4.79 | 3.15-7.31 | 0/20 |
| 5 | rolling | 11.13 | 6.66-18.60 | 0/20 |
| 5 | flat | 0.97 | 0.72-1.30 | 12/20 |
| 30 | delivery | 2.33 | 1.00-5.44 | 4/20 |
| 30 | rolling | 0.36 | 0.16-0.81 | 12/20 |
| 30 | flat | 0.43 | 0.28-0.65 | 18/20 |

Reading: the choice between two non-adaptive designs moves the same learner's error by up to an order of magnitude, the direction depends on the learner and family, and the coverage-style design lost to uniform random for both structured learners in every five-node system. Caveat: matched-random training matches the uniform held-out distribution more closely than an eleven-level grid does, so this is partly a train/test match effect.

**24-system screen, capped first-crossing cost, paired difference against target risk (policy minus target risk):** random +2.63 (0.33 to 4.92), menu coverage +0.83 (-0.47 to 2.13), local information +0.33 (-2.29 to 2.96), numerical ranker +1.96 (-0.84 to 4.76), frozen LLM +2.00 (-0.86 to 4.86), supervised LLM +3.00 (0.44 to 5.56), DPO LLM +3.54 (0.83 to 6.26). Unadjusted, descriptive.

**Same screen, 14 systems not already at target after initialization:** target risk 14.21, local information 14.79, menu coverage 15.64, numerical ranker 17.57, frozen LLM 17.64, random 18.71, supervised LLM 19.36, DPO LLM 20.29.

**Geometric-mean final error:** menu coverage 0.00721, target risk 0.00849, local information 0.00945, frozen LLM 0.01054, DPO LLM 0.01084, random 0.01093, supervised LLM 0.01152, numerical ranker 0.01276. Target risk is best on mean capped cost and mean final NMSE, menu coverage is best on geometric-mean final error and on sustained attainment.

**Action counts (`actions.csv`) [V]:** target risk: observe 215, set_x_neg 129, set_x_pos 76, set_y_pos 62, set_xy 31, set_y_neg 29, set_y_zero 16, set_x_zero 2. Local information: observe 318, set_x_neg 149, set_x_pos 67, set_y_pos 24, set_y_neg 7, set_xy 5. Numerical ranker: set_x_pos 435, set_x_neg 141. Frozen LLM: observe 574, set_x_zero 2. Supervised LLM: set_x_neg 481, set_x_pos 23, set_y_neg 11, observe 8. DPO LLM: set_x_neg 576.

---

## 11. Open questions

1. Which venue and date does the lead researcher want? Nothing is recorded. The answer decides length, format and whether a rigorous negative or diagnostic paper is acceptable.
2. Should the delivery study be folded into the broader paper as a supporting section, or submitted separately to TMLR and cited? The archive record calls it a "supporting component study" and agenda Stage 4 says to build the broader paper "with the delivery study as a supporting result".
3. Are checkpoint-only diagnostics on the 640 saved prospective fits permitted, given the instruction not to "repeat its completed studies"? They are new post hoc analyses, not repeats.
4. Is the five-node reversal a convergence or seed artifact or a propagation effect? Undiagnosed.
5. What is the oracle headroom on the screen's task family? I found no computation of the best achievable cost.
6. Does the numerical ranker's two-action behavior vary with state, and what were the teacher action frequencies in the 32 training systems?
7. Can code be released anonymously, given the attribution-bearing archive?
8. Must the earlier adverse LLM results (Mission-1, September audit, October language proposals) be reported in the broader paper? The agenda says Mission-1 "must remain part of the renewed project's evidence".
