# prior-evidence-core: what earlier ACE evidence is solid, what was refuted, what a new agenda must honor

Prepared 2026-11-12 by a read-only subagent. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was modified. No model, simulator, SLURM or SSH call was made.

Confidence tags used below: **[V-doc]** read in the cited document. **[V-data]** recomputed by me from saved files (scripts in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/prior-evidence-core/`). **[V-code]** read in source. **[INF]** my inference.

Abbreviations: ACE repo = `/Users/pat/code/ACE`. Runner = `/Users/pat/code/ACE-Runner`. Ledger = `ACE/docs/development/guidance/research_execution_2026-09-25.md`. G/ = `ACE/docs/development/guidance/`.

---

## 0. Five things the agenda writer should know first

1. **No experiment in either repository has ever shown an LLM proposer or policy adding measurable value over a non-LLM control.** Every fair test tied or lost (30-node audit, Mission-1 B vs C = 1.0412, Runner FM elicitation named = anonymized = 1.00, November 24-system screen).
2. **Every time an LLM policy's raw action distribution was observed without scaffolding it was near-constant**, across two model families, three codebases and eleven months (section E2). The November `set_x_neg` 576/576 result is the latest instance of a recurring failure, not a new one.
3. **The constant is repeatedly traceable to prompt surface features**: the first node name, the prompt's own few-shot example, or (hypothesis for November) menu position.
4. **In the old pipelines the apparent "strategy" was produced by hand-coded heuristics**, not the LLM: prompt hint naming the parents to intervene on, a 70% hard cap, collapse breakers, diversity fills. In Mission-1 only 22–31% of executed "LLM-arm" selections were genuine legal LLM proposals [V-data].
5. **The only acquisition signal that has replicated in this project is numerical, model-based variance or risk reduction on an explicit SCM** (PEV, Track B risk-pair selector, Runner PEV-style design, November target-risk planner). It is modest, conditional on topology, and involves no LLM.

---

## A. Prior results solid enough to cite (exact numbers, n, caveats)

| # | Result | Exact numbers | n / unit | Status and caveats | Source |
|---|---|---|---|---|---|
| A1 | **At ≥15 nodes the original ACE (Qwen2.5-1.5B + oracle lookahead + DPO) loses to Random; DPO hurts** | End-of-campaign non-root broad-range MSE per node. N=15: ACE 0.163, ACE-w/o-DPO 0.076, Random 0.059 (DPO hurts p<0.001; w/o-DPO vs Random p=0.16). N=30: 0.257 / 0.132 / 0.090 (p=0.009; p=0.20). N=50: 0.271 / 0.238 / 0.118 (both LM arms ~2× worse, p≤0.001). 30-node Table 2: ACE 0.149 (n=3) / 0.249 (expansion n=5), w/o-DPO 0.094 (n=12), Random 0.092, round-robin 0.078, max-var 0.090, "BOED" 0.129. Query-matched 30-node: ACE 0.245/0.249 vs 0.086/0.071/0.064/0.122 | 135 baseline cells re-run; Welch t, unpaired | **Solid negative.** Baselines here used the weak (16,)/50-epoch student, ACE the (64,64)/100-epoch one, so the conclusion only strengthens. One homogeneous family (`LargeScaleSCM`). | `G/metric_audit_2026-09-10.md:108-134`, `:157-162` [V-doc] |
| A2 | **The repo's "Bayesian OED" baseline is one-step oracle lookahead and does not beat Random** | 30-node 0.129 vs Random 0.092 | as A1 | Solid. Do not call it BOED. | same `:133`; `G/pev_design.md:22-25` [V-doc] |
| A3 | **Observational validation hides interventional failure** | Passive baselines ~0.02/node on observational validation, 4–8× higher and drifting under broad-range validation | as A1 | Solid methodology finding. | `metric_audit:130-132` [V-doc] |
| A4 | **Surviving 5-node result: a tie once the learner is matched** | Before matching: ACE 0.143 (0.127 paired seeds) vs Random 0.361, RR 0.336, MaxVar 0.356 ("2.5×", p<0.001); query-matched 0.155/0.153 vs 0.362/0.343/0.361. **After matching the student**: ACE 0.165 ± 0.032 vs round-robin 0.161 ± 0.023 (Welch p=0.85), Random 0.223 ± 0.050 (p=0.17). Per node: X3 collider ACE 0.155 vs RR 0.401 vs Random 0.560; X5 quadratic ACE 0.329 vs RR 0.072 vs Random 0.098 | Matched result is **3 seeds × 40 episodes, local, preliminary**, one fixed SCM (seeds are learner randomness only) | **Cite only as "no advantage after matching, with a per-node trade".** Unresolved asymmetries: (i) prompt names X3's parents, (ii) parent-balance scaffolding and hard cap, (iii) early stop, (iv) ACE uses 4–5× more total queries at matched episodes, (v) missing controls never run: ACE learner + random proposer, 5-node ACE-w/o-DPO, prompt without parent hint, (vi) the 10-seed × 171-episode CURC ladder that was to replace this table was never submitted. | `metric_audit:117-118,125-128,136-183`; `G/handoff_2026-09-25.md:87-88` [V-doc] |
| A5 | **PEV (numerical, no LLM) beats graph-matched coverage on two synthetic families** | Homogeneous 30-node: paired mean −0.01360, 95% CI [−0.02151, −0.00569], Holm p=0.00384, 20/20. Heterogeneous 30-node: −0.10919 [−0.18856, −0.02981], Holm p=0.00961, 16/20. PEV ≈ naive variance (p≈0.54, 0.90) | 40 fresh systems, 4 arms, 160 cells, exactly 2,000 samples each, frozen protocol | Solid within one hierarchical generator. K=3 ensemble, 20 epochs. **No IVR-formula claim** (ties naive variance). Observational loss unchanged. | `ACE/results/research_persistent_confirmation_v1/README.md`; Ledger `:92-96` [V-doc] |
| A6 | **PEV vs random on shifted mechanisms (secondary); primary vs coverage failed** | Means: random 0.043898, coverage 0.039791, PEV 0.015430, PEV-var 0.015871. Primary PEV−coverage −0.024360 [−0.048847, +0.000126], p=0.05108 **gate fails**. Secondary PEV−random −0.028468 [−0.048386, −0.008550], p=0.00750, 19/20 | 20 fresh systems, 80 cells | Cite both. Secondary, not multiplicity-adjusted. One system (seed 5013) drives the mean. | `ACE/results/research_pev_shift30_mean_confirmation/README.md`; `ACE/docs/ACE_evidence_and_public_claims_2026-10-08.md:37-56` [V-doc] |
| A7 | **Endpoint-coverage control did not beat PEV on fresh systems** | Endpoint coverage 0.019032, PEV 0.015347, PEV-var 0.017491; coverage−PEV +0.003685 [+0.000452, +0.006917], p=0.0276 | 20 fresh systems, 60 cells | The frozen hypothesis (coverage better) failed. Not a registered PEV win. A 3-system dev result had favored coverage and reversed. | `ACE/results/research_pev_endpoint_replication_v1/README.md` [V-doc] |
| A8 | **Track B posterior-risk pair selector (exact linear posterior, zero-call, cost-aware) vs fixed factorial-hub control** | Binary tree: 0.026942 vs 0.039722, ratio 0.6783, CI [−0.020726, −0.004834], 26/40 **pass**. Fanout: 0.013271 vs 0.015352, ratio 0.8645, CI [−0.004491, +0.000329], 25/40 **fail**. Random-recursive: 0.0227035 vs 0.0591759, ratio 0.38366, CI [−0.0548996, −0.0180453], 31/40 **pass**. Value optimization vs fixed-value: 0.025486 vs 0.032938, ratio 0.7738, 58/80 **pass** | 40/40/40/80 fresh systems, local CPU | Sequentially chosen topologies. Toy interaction motifs. Nearest numerical precedent for November target-risk planning [INF]. | Ledger `:276,280,289,339` [V-doc] |
| A9 | **Mission-1 (pre-registered): ACE's data ties Random and LHS; no component adds anything** | A 97.99% exact at 4,803 calls vs Random 97.97%, LHS 98.03%. B′ 95.17% at 1,103 vs 95.39%/95.38%. H1/H2 error ratios 1.0442, 1.0434, 1.0507, 1.0650, Holm p 0.469, all inconclusive. LLM (B vs C) **1.0412**, 98.75% CI 0.8870–1.2221, sign-flip p 0.440, 2/11. Selection 0.9922/0.9999/1.0467. Scorer 1.0235. Validation set 1.0094 | 11 paired seeds, 44 runs, one emulator (`tlam_mission1`, 390,625 designs) | Solid adverse result for that pipeline. Not equivalence (CI upper bounds 1.13–1.22 exceed the 1.05 non-inferiority bound). | `Runner/studies/2026-10-peter-baseline/package/stats_appendix.md:7-26` [V-doc] |
| A10 | **ACE's online delivered model is far worse than a refit of its own rows** | 82.37% vs 97.99% exact; error ratio 8.7155, 99.29% CI 6.6047–11.5008, Holm p 0.007, 0/11 | 11 seeds | The only significant Mission-1 result. A learner effect. | same `:26,57-64` [V-doc] |
| A11 | **Delivery confirmation** | Geometric mean snapped exact-level error ratio **0.182719**, 95% log-t CI [0.101970, 0.327410], exhaustive sign-flip p=0.00146484, 11/12 improve. History 124753321 worsens 0.196339 → 0.395658 (ratio 2.015177). Deployable init-0 ratio **0.204083**, median exact 97.40% vs 80.11% online, min 60.34%, 3/12 below 90% | 12 histories of **one** deterministic emulator; numerator is a median of 3 scored inits | Not acquisition, not 12 worlds, exposed grid, unequal fitting compute. | `ACE/results/delivery_final_history_20261006/{scores,independent_statistics}.json` [V-data]; `G/handoff_2026-09-25.md:3-28` |
| A12 | **Delivery prospective study (Stage B, registered)** | Delivery/online: 5-node **2.45956** [1.78782, 3.38368] Holm p 3.3e-05 **gate not met, adverse**; 30-node 0.274321 [0.158409, 0.47505] met. Delivery/flat: 5-node 0.356194 [0.282389, 0.449288] met; 30-node 0.391027 [0.254548, 0.600681] met. By history: 30-node balanced delivery/flat 0.912 (10/20), matched-random 0.168; 5-node balanced/online 1.614, random/online 3.748 | 20 independent systems per stratum, 640 fits, 32,000 charged responses, noise-disabled, root actions | Accepted. The 5-node reversal is **undiagnosed**. Init sensitivity: one delivery fit ~1,044× its rolling fit's error. | `ACE/results/delivery_prospective_reporting_20261024/terminal02/raw/primary/delivery_prospective_primary_table.tex`, `.../descriptive/history_specific.csv` [V-data]; `G/delivery_panel_revision_2026-11-02.md:9-15` |
| A13 | **Delivery attribution (Stage A, exploratory)** | All-paid SCM 30,000 epochs init 0 vs: online 0.522334, flat 0.550850, CPU-matched flat 0.423260, buffer-only long fit 0.095560, all-paid 100 epochs 0.033812, online-admitted long fit 0.980018 | 12 histories, 480 fits | Exploratory, exposed grid. Long fitting carries most of the gain. | `G/delivery_paper_execution_2026-10-06.md:68-77` [V-doc] |
| A14 | **Physical boundary (Causal Chambers lt_malus_v1)** | Delivery beats rolling in 7/11 conditions, relative-angle physics in 3/11, Fourier in 0/11 | 11 conditions of one apparatus | Not 11 worlds. Domain structure beats the generic neural recipe. | `G/delivery_paper_execution_2026-10-06.md:181-199` [V-doc] |
| A15 | **Runner arm-A delivery test (shipped LLM arm)** | r = 0.329, 95% CI 0.179–0.602, sign-flip p = 0.0029, 10/11; seed 7005 worse (72.09% vs 82.07%) | 11 seeds, 33 CPU fits | Pre-registered, superior. Learner change only. | `Runner/studies/2026-10-delivery-arm-a/RESULTS.md` [V-doc] |
| A16 | **Runner dev-only signals** | PEV-style SCM design / Random+SCM at 1,000 calls: `counter_uas` 0.68 (0.63–0.73, 10/10), `tlam_satellite` 0.62 (0.44–0.87, 8/10), `synthetic_scm` 0.81, `tlam_mission1` 0.91 (0.81–1.02), `missile_defense` 1.00. Runner lookahead without LLM 0.97–1.03 everywhere. Causal surrogate vs flat MLP 0.27/0.16/0.07 at 100/200/500 calls. **The one place an LLM output helped anything**: Qwen2.5-1.5B-Instruct elicited mechanism form (a bounded output link, a product term) added to a strong learner gives 0.49 at 500 calls (10/10), but named vs anonymized = 1.00, so the gain is form read from stated ranges, which a rule reproduces. In a weak linear-basis family names alone gave 0.86–0.87 (10/10), prompt-sensitive | 10 seeds, exploratory, exposed grid and seeds | **Dev only.** Shipped-LLM arm never ran (no T4 capacity). The LLM here is a prior compiler for the learner, not an intervention policy. | `Runner/studies/2026-10-deck-recreation/README.md`; `Runner/studies/2026-10-fm-scm-dev/README.md:62-100`; `Runner/docs/status.md:38-52` [V-doc] |
| A17 | **Input normalization moves results more than method choice** | 18/20 method × environment pairs improve, 15 significantly; only 3 of 15 ACE-vs-baseline contrasts p<0.05 on the normalized arm | 5 envs × 5 seeds × 2 arms | Solid, paired. | `Runner/docs/decisions/0004-normalize-inputs.md` [V-doc] |

---

## B. Refuted or retracted. Never lean on these

1. **Every ACE-vs-baseline number in the NeurIPS 2026, ICLR 2027 and AISTATS 2027 drafts.** "69%" at 5 nodes (0.61 vs 2.09–2.21), "3× at 30 nodes" (1.95 vs 5.86), scaling 0.05–0.14 vs 0.15–0.19, budget-fairness "76% / 43–50%". Four confounds: reward-shaped 0.2 root weight leaked into the metric, different validation distributions, baselines on a non-stationary simulator at N≥15, unmatched students. `metric_audit:1-67,140-162`.
2. **The paper's "matched learner" statement was false.** ACE (64,64)/100 epochs, ~4,600 parameters per mechanism. Baselines (16,)/50 epochs, ~50. `metric_audit:142-152`.
3. **The 5-node "2.5×" ACE win.** See A4.
4. **DPO as a contribution.** Hurts at every N ≥ 15. Five of five 30-node expansion seeds showed pathological DPO (two near chance for 64%/56% of updates, three collapsed to ~0 loss with extreme KL). `kl_from_ref` logged large negative values (−437 to −2375) in every run, a logging sign error or bug never resolved. `G/current_status.txt:141-168`; `Runner/docs/decisions/0003-policy-update-default-none.md:19-34`.
5. **"Seed expansion does not replicate".** On like-for-like metrics it does (2.21 ± 0.47 vs 1.95 ± 0.77, Welch p=0.64). `metric_audit:62-65,106`.
6. **Runner published win rates** (13/14, 17/19, 14/15 vs Random). Every run had DPO on, used the in-run metric that also selects experiments, and published bundles pass a "flattery gate" (commit only if ACE beats Random) with the linear baseline stripped and no record of rejected runs. `Runner/docs/decisions/0002-...md:17-36`, `0005-...md:13-28`, `code_health.md:339-349`.
7. **Customer-facing numbers.** July "177 runs / 5,650× / 99.85%", "ACE 142 vs NBF 177", September "53 / 102 / 1,583 calls", and the v9 deck's slide-12 values (500 and 1,400 come from the research repo's synthetic 30-node **PEV** study with no LLM; 29.26M is analytic). `Runner/HANDOFF-2026-10-peter-study.md:70-74`; `Runner/studies/2026-10-deck-recreation/README.md`.
8. **PEV's integrated-variance formula as the cause of anything.** It ties naive variance in every confirmation (exploratory p=0.0773 at best). Shift30 primary gate failed. Random-DAG graph-shift pilot failed its promotion gate (3 systems). Ledger `:112,140,142`.
9. **The "fixed topology" wording of the shift30 protocol.** There were 20 distinct DAGs. `G/erratum_shift_graph_provenance_2026-09-27.md`.
10. **The product-form "physics" control on Causal Chambers.** Misspecified. The correct control is intercept + cos²(relative angle). Ledger `:389-393`.
11. **Failed and stopped lanes:** semantic prior gate, modular transfer and repair gates, direct-prompt partial identification (8/8 schema failures), hand-written action-language parser (8/8 in-sample, 0/8 on later phrasing), NeuronBench state forecaster (5/12 cells vs gate of 8), joint-action capability (CI [−0.03150, +0.00043]). Ledger `:144-343`.
12. **The 20 September "OOM fixed" claim.** The canary OOM'd again at 9:14 h vs 4:56 h. The old GPU pipeline is not reliably runnable. `handoff_2026-09-25.md:144-225`.
13. **`FULL_ACE_COMPONENTS.md` (Jan 2026)** premise that the 5-node architecture "succeeded". Historical only.
14. **October TabPFN/foundation component results** are mechanism prediction on fixed histories, not policy evidence. All six language proposals were invalid and fell back to the numerical grammar. `G/ace_foundation_component_results_2026-10-09.md:18-24`.

---

## C. Fate of PEV

- **The CURC PEV ladder (`ACE/jobs/curc_submit_pev_ladder.sh`) was never run.** `handoff_2026-09-25.md:38` says "It has never been submitted". `G/research_portfolio_2026-09-25.md:215-221` found no PEV result directory on scratch and noted the script expands to 200 jobs and 10,440 requested CPU-core-hours, so a staged screen replaced it. No `results/pev_ladder/` exists locally, no commit touches that path, and no later document mentions it [V-data, read-only `git log`]. The user's memory note agrees.
- **Its three pre-registered criteria (`pev_design.md:150-157`) were therefore never tested as written**: PEV vs same-student Random at 30 nodes hetero with K=5/100 epochs, PEV not losing on LargeScaleSCM at 15/30/50, and PEV matching ACE's 0.14 at 5 nodes with ≤1/4 of the queries. The 5-node local preliminary (PEV 0.045 vs ACE 0.165, 3 seeds × 40 episodes, p≤0.001) is still the only PEV-vs-ACE number. On the persistent pilot the 5-node differences were small (0.6594 random, 0.6307 coverage, 0.6028 PEV, 0.5968 naive) and PEV did not beat coverage on every seed (Ledger `:80`).
- **What ran instead** was a new persistent-campaign harness (`ACE/scripts/research/persistent_scm.py`, sealed evaluator, exact 2,000-sample budgets, K=3, 20 epochs): results A5–A7 plus the failed random-DAG pilot. Net: PEV-type uncertainty acquisition beats random and ordinary coverage inside one hierarchical generator, is not separable from naive variance, and has no confirmed graph-generator or external transfer. The PEV lane was then stopped under its gates.
- **Relation to the November target-risk planner.** No code or documented lineage. No `ace_llm_intervention_*` document, review or module mentions PEV [V-data, grep]. Conceptually they are the same family: zero-environment-query, one-step, model-based expected variance reduction propagated through the SCM [INF]. Differences: PEV uses bootstrap MLP-ensemble disagreement, a mechanism-level objective summed over descendants, no cost term and a continuous value grid. Target risk uses an exact linear-Gaussian posterior with exact composed moments, a declared target-prediction risk over a public query law, division by action cost, 32 fantasy draws and a finite menu (`G/ace_llm_intervention_target_risk_2026-11-02.md`). The Track B `risk_pair` selector (`ACE/scripts/research/connected_acquisition.py`) is the closer numerical ancestor [INF].

---

## D. Mission-1, the earlier LLM proposer study

**Design** [V-doc: `Runner/studies/2026-10-peter-baseline/package/PREREGISTRATION.md`]. Frozen 2026-10-01. `tlam_mission1`, 8 root inputs × 5 levels = 390,625 designs, sealed full-grid truth. Every simulator call charged. One fixed refit learner for all methods (64-64 causal chain, 30,000 epochs, median of 3 inits). Arms: A = ACE as shipped (K=4 lookahead on the simulator, FIFO-50, 4,803 calls). B = fair data (train on all 4 probes, 1,103 calls). B′ = B with scale-free score. C = B with random proposals. Offline Random and LHS streams. Qwen2.5-1.5B @ 8faed76, DPO off (`policy_update=none`, teacher SFT pretrain still on). 11 paired seeds from a power rule (≥80% power for a true 30% error reduction). Verdict classes better / inferior / non-inferior / inconclusive, Holm, intersection-union.

**The 1.0412 ratio.** B vs C at 1,103 calls, n=11, mean log 0.0404, sd log 0.1749 (the noisiest test in the study), t p 0.462, sign-flip p 0.440, 98.75% CI 0.8870–1.2221, Holm p 1.000, 2/11 seeds favor the LLM, verdict inconclusive.

**Why it detected no LLM advantage.**
- Documented: all 8 inputs are roots, so the prompt carries no per-node loss signal (`code_health.md:478-482`). Every call already sets all 8 roots and returns every intermediate, so the design is full rank and adaptivity has no headroom. Mission-1 root-design acquisition was "retired as a superiority venue" (`Runner/docs/research/ace_fm_scm_program_2026-10.md:40-47,246`).
- **New, from the saved bundles [V-data]** (`.../prior-evidence-core/mission1_parse.py`, `mission1_example_check.py`, reading `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/host-mirror/results/{slot0,slot1,c0,c1}`):
  - The "LLM arm" was mostly not the LLM. Selected action per step by provenance, out of 2,200 steps per arm:

    | Arm | hand-coded `diverse` fill (empty LLM output) | prompt-example echo | genuine legal LLM proposal |
    |---|---:|---:|---:|
    | A | 1,521 (69.1%) | 196 (8.9%) | 482 (21.9%) |
    | B | 1,267 (57.6%) | 280 (12.7%) | 653 (29.7%) |
    | B′ | 1,240 (56.4%) | 282 (12.8%) | 678 (30.8%) |

    "Genuine" means the generation's first `DO` names one of the 8 legal inputs; the executed target equalled the generated target in all such rows. Arm A sums to 2,199: one row with an illegal generated target executed something other than the modal command. `diverse` is the Runner's coverage fill, used when the LLM proposal is invalid or repeats a target already used in that step (`Runner/ace/oracle.py:1503-1540`).

  - One command, `leo_dynamic_satellite_count = 200.0`, appears among the K=4 probes in 84.95% (A), 86.09% (B) and 87.18% (B′) of steps, against 9.68% in the random arm C (10% expected).
  - That command is the prompt's own first few-shot example. The prompt builder emits `DO <feature_names[1]> = <global midpoint>` = `DO leo_dynamic_satellite_count = 5000.0` (`Runner/ace/llm_policy.py:304-318`), the lenient parser clips it to the feature's upper bound 200.0, and it is returned whenever the generation has no legal command (`:380-388`). When the generation named an illegal target or contained no command, the executed command was this one in 196/197 (A), 280/280 (B), 282/282 (B′) selected rows.
  - Raw generations frequently named non-actionable intermediates with absurd values, for example `DO leo_link_score = 2.750000e+38`.
  - Among genuine legal proposals the generated target was still `leo_dynamic_satellite_count` in 32% (154/482, 211/653, 212/678) against 12.5% uniform.
  - The LLM's choice sets one coordinate. The other seven inputs of every call are randomized.
- So B vs C compared "random proposals" with "mostly hand-coded coverage fills plus a constant". The treatment was diluted to the point that 1.04 says little about what an LLM could contribute [INF]. It does establish that this pipeline's LLM contributed nothing.

**Execution disclosures and what they imply** (`stats_appendix.md:83-117`):
- B ran on GPU with the LLM, C on CPU without it. New comparisons need the same device and fitting implementation for every arm.
- LLM runs are not bit-reproducible on GPU. Two fresh same-seed runs diverged at step 2, one via a lenient parse and one via fallback. Seeds replicate designs, not trajectories. Save every raw run and compute from saved data. CUDA refits agree to about ±0.7 points, so use several inits.
- Raw LLM text was logged only for the selected candidate. New logs must bind completion, parsed action and executed command for every candidate, and report provenance shares as a treatment-fidelity check before interpreting any LLM contrast.
- Arm A's startup validation sets set intermediate nodes directly (1,000 of 4,803 calls), an action no baseline could take.
- Seed 7011 was added in a second batch after the n rule. Statistics were first computed before the descriptive CV phase finished. The package builder changed after the freeze (presentation only). None changed a verdict, all were disclosed.

---

## E. The original ACE LLM pipeline and the pattern of collapse

### E1. How it worked [V-code: `ACE/ace_experiments.py`]

- **Model.** `Qwen/Qwen2.5-1.5B` default (`:2222`), float32, **all weights trained** with Adam lr 1e-5 (`:2655`), no adapters.
- **What the LLM proposed.** Free text sampled at temperature 0.8, top-p 0.9, repetition penalty 1.3, 64 new tokens (`:774-783`), regex-parsed for `DO X<k> = <value>` in [−5, 5]. K=4 candidates per step, later 3.
- **Prompt** (`:627-753`). States the top-3 node losses, then **tells the model the answer**: "To fix {failing}, intervene on its parents: …" (`:709`). Adds "WARNING: X is over-sampled. AVOID X" when one node exceeds 30% (`:692-694`). Includes few-shot examples whose first command is `DO X2 = 1.5` (`:720`).
- **Parser defect, same as Runner.** If the generation has no command, the full text including the prompt is parsed and the first match is the prompt's example (`:796-802`). If that fails, a random parent of the failing node is substituted (`:804-816`). Parse rate is therefore 100% by construction.
- **Scoring.** Each candidate clones the student and trains it on **real environment data** (oracle lookahead, uncharged against the stated budget) (`:3109-3131`). Reward is the loss drop measured by `ScientificCritic`, which **queries the true mechanisms directly** with 500 fresh U(−4, 4) parent samples per node outside the executor (`:893-938`, call at `:927`). Score = 10 × Δloss (clipped) + node-importance bonus + 0.3 × diversity score (`:3133-3179`).
- **Hand-coded overrides after scoring.** Smart collapse breaker that injects a collider parent, or the same node at a sign-flipped value (`:3245-3341`). Mandatory diversity constraint above 60% (`:3343-3351`). Forced diversity every 10 steps (`:3353-3365`). ε-explore 0.10 (`:3367-3370`). **Hard cap: any node above 70% of recent actions is replaced by the least-sampled collider parent** (`:3374-3410`). Teacher fallback when nothing parses (`:3222-3240`).
- **DPO.** Online, every step, winner vs a strategically chosen loser, loss weight doubled for "novel beats collapsed" pairs (`:3412-3481`). Log-likelihood summed over prompt plus command tokens (`:1067-1124`). β = 0.1. Reference copy refreshed every 25 episodes. 100 supervised pretraining steps on a loss-guided teacher, repeated every 50 episodes, with the loss **not masked to the completion** (`:1233-1237`).
- **Student.** Per-mechanism (64,64) MLP, 100 epochs per step, reset every episode, 25 steps per episode.
- **Runner port.** Same design. Its historic DPO scores the **command string alone, with no state prompt** (`Runner/ace/dpo.py:287`), so by construction it can only learn a state-independent preference. `policy_update=none` still runs teacher SFT (`Runner/ace/config.py:278`).

### E2. Every recorded observation of collapse or constant behavior

| When | Setting | Observation | Source |
|---|---|---|---|
| 2026-01-03 | 5-node, Qwen2.5-1.5B + DPO | LLM generated X1 for 100% of candidates. DPO stuck at 0.693 | `G/guidance_doc.txt:666-669` |
| 2026-01-11 | same | "Total Policy Collapse: 100% of candidates targeted X1". Old prompt ended in "DO", biasing toward the first node name | `guidance_doc.txt:1290-1300`; `ace_experiments.py:631-632,1176` |
| 2026-01-12 | one day after the few-shot example `DO X2 = 1.5` was added (commit `080a910e8`) | Collapsed to "a single static value DO(X2 = 1.5)". Diagnosed then as a learning-dynamics trap. Nobody linked it to the prompt example [INF that it was the example] | `guidance_doc.txt:1240-1250`; read-only `git log -S` |
| 2026-01-19 | run_20260119_123852 | LLM generated X2 for 99.1% of candidates. 89.3% of steps zero reward. KL 0 → −2,300. "Policy collapsed to 69.4% X2, requiring constant hard-cap enforcement" | `guidance_doc.txt:1330-1385` |
| 2026-01-25 | 5-node multi-seed, N=5, the paper Table 1 era | **X2 share 68.7–69.2% in every seed, exactly under the 70% hard cap**. X1 30.5–30.9% (the cap's forced alternative). X4 0.17% (28 of 16,202). `X2 == 1.5` exactly in 969 of 11,185 X2 actions (5.98% of all rows); the next most common X2 value, rounded to two decimals, occurs 22 times. Parse rate 1.0. Rows are DPO-update steps only | [V-data] `.../prior-evidence-core/ace30_actions.py` on `ACE/results/ace/ace_multi_seed_20260125_115453`; its `ACE_SUMMARY.txt` says "Intervention concentration (X1+X2): 99.6-99.9%" |
| 2026-05 | 30-node batches | Gradient checkpointing silently zeroed the DPO gradient ("loss near random chance", "None of the inputs have requires_grad=True") | `ace_experiments.py:591-603` |
| ≤2026-08 | Runner, untrained LM | "collapses every candidate onto one feature (observed: thaad_battery_count 200/200)" | `Runner/ace/oracle.py:1465-1467`; `Runner/ace/llm_policy.py:460` |
| 2026-09-07 | 30-node expansion | DPO pathological in 5/5 seeds, bimodal: near chance or ~0 loss with extreme KL | `current_status.txt:141-168` |
| 2026-09-27 | BoxingGym Lotka canary, Qwen2.5-1.5B-Instruct | Descriptive and anonymous prompts returned the same choice, identical to the data-only selector | `ACE/results/research_boxing_lotka_open_proposal_dev_v1/README.md` |
| 2026-10-01 | Partial-identification canary | 8/8 fail the schema. 0/4 correct actions. Both null menus got `doZ1` instead of abstention | `ACE/results/research_partial_id_open_canary_20261001_v2/README.md` |
| 2026-10-01 | Mission-1 arms A/B/B′ | One command in 85–87% of steps, equal to the prompt example. See D | [V-data] |
| 2026-10-08 | Runner FM elicitation | Named = anonymized (1.00). Adding a format example changed the elicited spec and raised error to 57% | `Runner/studies/2026-10-fm-scm-dev/README.md:62-100` |
| 2026-10-09 | Foundation component screen | 6/6 language proposals invalid | `ace_foundation_component_results_2026-10-09.md:18-24` |
| 2026-11-02 | Qwen3-0.6B qualification | Unconstrained output `{"action_id":"a0}` rejected. After constrained decoding, `a0` generated. Two tiny random models both returned `a00` (weak) | Ledger `:3239,3288,3344` |
| 2026-11-10 | 24-system screen | DPO `set_x_neg` 576/576. Frozen `observe` 574/576. SFT `set_x_neg` 481/523 | `G/ace_llm_intervention_24_system_results_2026-11-10.md:34` |

### E3. The pattern [INF, built on the verified rows above]

1. **Unscaffolded LLM action choice has been near-constant every time it was measured.** 100% X1, 99.1% X2, 200/200 one feature, 85–87% one command, 100% / 99.7% / 92% in November.
2. **The constant tracks prompt surface, not state.** First node name when the prompt ended in "DO". The few-shot example in both codebases. In November the menu order is `("observe_all", "set_x_neg", "set_x_zero", …)` (`ACE/scripts/research/llm_intervention_screen_protocol.py:67`) [V-code]: the frozen model's constant is the **first** entry and the trained models' constant is the **second entry, the first intervention**. Whether the prompt lists the menu in that order, and whether teacher frequencies also favor `set_x_neg`, I did not check. This is a hypothesis for the state-sensitivity diagnostic.
3. **State dependence in the old pipeline was supplied by code.** The prompt computed the failing node and named its parents. The 69/31 X2/X1 split is the hard cap. The "prompt without the parent hint" control was never run.
4. **Preference training has sharpened collapse rather than cured it.** Old DPO runs end at loss ≈ 0 (0.0175–0.054 in four of five January seeds [V-data]) with runaway KL, or at chance. Runner's DPO was state-unconditional by construction. November's DPO moved a 92% mode to 100%.
5. **The November design is the first to remove every scaffold** (constrained menu, no breaker, no cap, no diversity fill, no oracle lookahead, frozen weights at deployment). It therefore shows the collapse cleanly. It is the same phenomenon, now measurable.

### E4. How November differs

Qwen3-0.6B with rank-8 LoRA on q/v instead of full fine-tuning of Qwen2.5-1.5B. Offline training on 32 separate systems with frozen deployment weights instead of online within-run DPO. Grammar-constrained choice from 8 legal actions plus stop instead of free text. Public state includes exact posterior means and covariances instead of three loss numbers and a hint. One charged executor and a private evaluator instead of oracle lookahead and a privileged critic. A shared exact linear-Gaussian learner for all arms. Cost-to-target endpoint with censoring. Strong numerical controls and a matched learned ranker. No hand-coded overrides.

---

## F. Controls and lessons any new experiment must honor

1. **One learner for every arm**: same architecture, epochs, device, fit schedule. (`metric_audit:140-162`)
2. **One evaluator, one validation distribution, one stationary system, with parity tests.** No reward-shaping weights in a reported metric. (`metric_audit:11-48`; `ACE/scripts/analysis/test_metric_parity.py`)
3. **Charge every environment call**, including lookahead probes, validation sets, refreshes, teacher and breaker calls. Compare at matched charged cost. (ADR-0005)
4. **Sealed evaluation the policy never sees.** No selection of experiments on the test set (Runner's `interv_val` was used three ways). No best-checkpoint endpoints: the 30-node margin once depended on checkpoint selection (`current_status.txt:135-138`).
5. **No selection on favorable bundles or seeds.** Keep every cell, report parked and failed runs, never replace seeds, keep a selection log. (ADR-0002)
6. **Parse only newly generated tokens. Bind completion → parsed action → executed command for every candidate.** A fallback or diversity fill is a separately labeled behavior, never an LLM choice. Report provenance shares. (`code_health.md:164-173`; D above)
7. **No privileged critic.** Rewards and lookahead must go through the charged executor. `--lookahead_on_student` alone does not remove oracle access. (`ace_experiments.py:865-938`)
8. **Define "frozen" by weight hashes and behavior**, not by a config flag. (`Runner/ace/config.py:278`)
9. **One explicit state-conditioned training objective with prompt tokens masked.** Runner DPO was command-only; research SFT trained on prompt tokens.
10. **Separate proposer from scorer from heuristics.** Cross LLM / random / numerical proposals with a common scorer and with random-of-K selection. Any hint that computes the answer must be given to numerical arms or removed.
11. **Same device across arms. Treat LLM seeds as independent replicates.** Use medians over several inits for GPU fits.
12. **Independent systems are the statistical unit.** One fixed SCM gives learner noise only. 12 histories of one emulator are not 12 worlds. One apparatus is not 11 worlds.
13. **Pre-register**: frozen protocol hash, pinned analysis code, blinded confirmation seeds, Holm, a verdict rule that includes inferior and non-inferior, n from development variance.
14. **Strong simple controls change rankings.** Naive variance tied PEV. Endpoint coverage beat PEV on dev. Correct physics erased a neural gap. A "random" arm must share the eligible action set.
15. **Development rankings reverse on fresh systems** (endpoint coverage, binary-tree risk selector). Freeze, then draw fresh systems. Never extend a seed grid to cross a threshold.
16. **Check headroom before testing acquisition.** Adaptive design had none on Mission-1. In November, 10 of 24 systems already met the target after initialization.
17. **Separate acquisition from final prediction** with a common refit. Report the deployable init, not a median of scored inits.
18. **Match the estimand.** A chain of conditional means is not a noisy interventional expectation.
19. **Check for prompt-surface artifacts**: permute menu order and identifiers, remove or vary examples, anonymize names. [INF from E]
20. **Operations.** SLURM `COMPLETED` is not a result. Never retry an ambiguous submission. Qualify the target runtime. The old `ace_experiments.py` GPU path leaks host RAM (unsolved).

---

## G. Reusable assets

**Environments and generators (ACE repo).** 5-node `GroundTruthSCM` (`ace_experiments.py:38-65`). `experiments/large_scale_scm.py` (frozen coefficients, `coeff_seed`). `experiments/heterogeneous_scm.py` (linear, quadratic, sin, tanh, abs, product). `experiments/shifted_mechanism_scm.py`, `experiments/random_order_shifted_scm.py`. `experiments/complex_scm.py` (15 nodes), `duffing_oscillators.py`, `phillips_curve.py`. Connected-motif, fanout, binary-tree and random-recursive joint-intervention simulators with exact posteriors (`scripts/research/connected_*.py`, `joint_interaction_information.py`). November three-node chain/fork/collider worlds (`scripts/research/llm_intervention_world.py`, `llm_intervention_screen_protocol.py`).

**External adapters with custody receipts.** BoxingGym Lotka–Volterra and signal localization (`scripts/research/boxing_*`). NeuronBench public controls. CausalMan fixed-path probe (53 public columns; multi-seed through its API is not independent SCMs). Causal Chambers `lt_malus_v1` archive (12 conditions × 1,000 rows, logged data only) with the correct physics control. Partial-identification 20-task suite with an arithmetic control that scores 20/20. Action-menu validators and a temporal action contract.

**Runner.** Environments `tlam_mission1` (390,625), `tlam_satellite` (5,625), `counter_uas` (15,625), `missile_defense` (15,625; its leaker mechanism reads a variable missing from its DAG), `synthetic_scm` (125). Sealed `ace/grid_eval.py::GridTruth`. Refit lens `scripts/analysis/refit_score.py`. Offline design streams. `scripts/analysis/study_stats.py` (paired log ratio, sign-flip, Holm, IUT, `choose_n`). Role-tagged append-only `observations.ndjson`. `ace/delivery.py`. `acecloud` orchestration. Azure registry deleted 2026-10-21; GPU runs need a rebuild or a Lambda A10.

**Datasets.** 44 Mission-1 bundles with every call and raw LLM text for selected actions (`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/host-mirror/results`). 12 delivery histories, 480 Stage A fits, 640 Stage B checkpoints over 40 worlds. 135 audit re-run cells (`ACE/results/audit_reruns`). PEV persistent results (`ACE/results/research_persistent_*`, `research_pev_*`). An incomplete LLM model-scale sweep (`ACE/results/curc_model_scale_sweep/aggregate.csv`: 31 of 63 cells present; zero-shot 30-node final per-node MSE 0.148 / 0.172 / 0.242 for 0.5B / 1.5B / 3B, n=5 each, pre-audit pipeline, descriptive only [V-data]).

**Infrastructure and protocols.** `persistent_scm.py` with `SealedEvaluator` and pre-query budget enforcement. `validate_cell.py`, `status.py`. Frozen protocol JSONs in `G/protocol_*`. ADR-0005 sealed-evaluation protocol. Delivery prospective registration and independent audit pattern. November charged executor, grammar-constrained decoding, target-risk planner, matched ranker, selective CURC collector. CURC account `ucb736_asc2`.

**Consumed seed namespaces to avoid** (non-exhaustive). 42/123/456/789/1011, 100–111, 200–202, 300–305, 400–402, 500–519, 700–719, 800–819, 900–939, 1000–1019, 1100–1179, 1300–1319, 1400–1439, 1600–1639, 1800–1805, 2000–2019, 2022–2026, 3000–3019, 4000–4019, 5000–5019, 6000–6019, 7001–7011, 9999, 94000–94005 (reserved). (`Runner/docs/research/ace_fm_scm_program_2026-10.md:195-196`; Ledger.)

---

## H. Disagreements between documents (newest wins)

- `Runner/docs/status.md` (2026-10-23) says Stage B is unexported so ADR-0006's revisit trigger "has not fired". The ACE Ledger (2026-10-24, `:3119`) reports the 5-node delivery/online contrast failing its superiority gate at 2.45956. By ADR-0006 §7 that triggers a review. No ADR-0007 exists.
- `handoff_2026-09-25.md` and the memory note say the PEV ladder was never submitted. The October public-claims document cites PEV confirmations. Both are true: those came from the later persistent-campaign harness, not the ladder.
- ADR-0003 (2026-09-10) reports DPO "not distinguishable at N=50 (p=0.29)" on the pre-audit metric. The 2026-09-17 audit outcome on one metric has both LM arms about 2× worse than Random at N=50.
- The November 2 agenda proposed seven policies, 16 + 64 responses and account `ucb736_asc1`. The executed screen used eight policies, 8 + 24 cost units and `ucb736_asc2` (`handoff_ace_llm_research_2026-11-12.md:19`).
- PEV is expanded as "Propagated Epistemic Variance" (`pev_design.md:1`) and "Propagated-Expected-Variance-reduction" (`handoff_2026-09-25.md:110`).

## I. Limits of this report

- Audit outcome numbers (A1–A4) were read from `metric_audit_2026-09-10.md`, not recomputed from `results/audit_reruns`.
- The attribution of the January `X2 = 1.5` mode and the 69/31 split to the prompt example and the hard cap is inference from code plus counts. January code may have differed in details from HEAD; the example and the full-text fallback are confirmed present by `git log -S` (added 2026-01-11 and 2025-12-29).
- Mission-1 provenance shares cover the selected candidate per step only. Raw text for unselected candidates was not logged.
- I did not inspect the November prompt rendering or teacher action frequencies.
