# ACE: public message and evidence against random experimentation

**Date:** October 8, 2026  
**Author:** Codex, acting as Patrick's AI research collaborator  
**Purpose:** An authored assessment of the proposed public tagline and the evidence for saying that ACE works better than random. This is an interpretation of recorded studies, not a new experiment or a statement authored by Patrick.

## Short answer

**Yes: ACE's historical uncertainty guided experimental policy has outperformed a specified random policy in controlled synthetic studies. The evidence supports that limited statement. It does not establish that ACE always chooses better experiments, that its particular scoring formula is uniquely responsible, or that the advantage generalizes to real laboratories.**

My suggested public tagline is:

> **ACE: Make every experiment count.**

An accompanying explanation could be:

> ACE is designed to learn from accumulated experimental results and help decide what to try next. In controlled computer simulations, an ACE experimental policy produced lower prediction error than random experiment selection. We are evaluating where those benefits hold and where simpler methods work as well or better.

“Make every experiment count” describes the design goal: preserve useful observations and use them in learning. It should not be presented as a guarantee that every individual experiment improves a model or that ACE is universally optimal.

## 1. What “better than random” means here

Three questions need separate answers:

1. **Experiment selection:** Does the policy choose a more useful sequence of experiments than a specified random policy, given the same number of environmental samples?
2. **Final model delivery:** Does fitting a final model from accumulated observations improve on the model produced during collection?
3. **General usefulness:** Does the complete system improve decisions in new domains, after considering computation, measurement cost and strong alternatives?

The historical PEV studies directly address the first question. The current delivery paper principally addresses the second. Neither alone establishes the third.

The direct random comparator was substantive: it used the same graph based eligible target set as PEV, selected targets randomly, and drew intervention values uniformly from the permitted range, −5 to +5. It also used the same persistent ensemble learner. Thus the comparison was not against an untrained model or deliberately irrelevant experiments. “Random” still denotes this particular policy; it is not a mathematical lower bound on performance. Target/value design and acquisition computation can differ even when sample budgets match. See the [policy implementations](/Users/pat/code/ACE/baselines.py:225) and [campaign runner](/Users/pat/code/ACE/scripts/research/persistent_scm.py:125).

## 2. Direct evidence against random

The clearest recorded comparison is the fresh shifted-mechanism confirmation: 20 synthetic systems, four policies, 80 completed cells, and exactly 2,000 environmental samples per policy per system. All four arms shared each system's graph and mechanisms. The endpoint was final held-out, noise-free feasible non-root mechanism prediction error; lower is better. It measures mechanism prediction, not the success rate of real-world decisions.

The recorded arithmetic mean errors were:

- Random experiment selection: **0.043898**.
- Systematic coverage: **0.039791**.
- PEV integrated variance reduction: **0.015430**.
- Simpler variance scoring: **0.015871**.

PEV had lower error than random on **19 of 20 systems**. The mean paired PEV-minus-random difference was **−0.028468**, with a reported 95% paired t interval of **[−0.048386, −0.008550]** and two-sided **p = 0.00750**. These are the study's recorded secondary comparison statistics, not a multiplicity-adjusted primary success claim. See the [completed study summary](/Users/pat/code/ACE/results/research_pev_shift30_mean_confirmation/README.md) and [frozen protocol](/Users/pat/code/ACE/docs/development/guidance/protocol_pev_shift30_mean_confirmation.json).

The two arithmetic mean errors imply approximately **64.85% lower mean error than that random policy**, calculated as `1 − 0.0154302049 / 0.0438980071`. This is a descriptive ratio of group means. It is not a paired geometric mean, a confidence bound on percentage improvement, or “65% greater accuracy.”

### Why this supports a limited claim

Fresh systems, matched environments, identical sample budgets and a held-out final endpoint make this meaningful evidence that the complete PEV policy/learner procedure used its experimental budget more effectively than the specified random procedure in this setting. All 20 systems contributed; no favorable subset or best checkpoint was selected.

The comparison against random was specified in advance as **secondary**. The primary comparison was PEV against systematic coverage, and it **did not pass**: mean difference −0.024360, 95% interval [−0.048847, +0.000126], p = 0.05108. The positive random comparison does not replace that failed primary gate. PEV versus simpler variance scoring was also unresolved: p = 0.62176. Failure to detect a difference is not proof of equivalence.

The original protocol inaccurately described fixed graph topology. A subsequent provenance audit established **20 distinct DAGs within one fixed hierarchical generator**, with matching graphs across the four arms of each system. That correction strengthens the description of within-family variation, but does not establish transfer to another graph generator. See the [preserved provenance correction](/Users/pat/code/ACE/docs/development/guidance/erratum_shift_graph_provenance_2026-09-27.md).

## 3. Other evidence and counterevidence

### Fresh homogeneous and heterogeneous confirmations

A separate frozen study used 40 fresh systems, 20 per synthetic family, with four policies and 2,000 samples per arm. PEV improved on the primary graph-matched coverage comparator in both families:

- Homogeneous: paired mean difference **−0.01360**, 95% interval [−0.02151, −0.00569], Holm-adjusted p **0.00384**.
- Heterogeneous: **−0.10919**, interval [−0.18856, −0.02981], Holm-adjusted p **0.00961**.

These strengthen the evidence for the complete uncertainty guided procedure under those synthetic conditions. They do not establish a special benefit from integrated variance reduction: comparisons with simpler variance scoring remained unresolved. Observational prediction losses were also similar among policies. I cite these as coverage comparisons, rather than inferring a random comparison from them. See the [confirmation summary](/Users/pat/code/ACE/results/research_persistent_confirmation_v1/README.md) and [registered primary contrasts](/Users/pat/code/ACE/docs/development/guidance/protocol_persistent_confirmation_v1.json).

### Stronger value controls changed the interpretation

After action logs showed that uncertainty policies often selected large intervention magnitudes, endpoint-valued controls were added. On three reused development systems, simple endpoint coverage beat PEV on all three, with mean errors **0.008802 versus 0.011167**. Endpoint random was worse, at **0.034605**. This showed that action magnitude alone was insufficient and that a stronger systematic comparator could change the ranking. These were post hoc development results. See the [endpoint control report](/Users/pat/code/ACE/results/research_pev_extreme_value_dev/README.md).

On 20 fresh systems, that development ranking reversed: endpoint coverage error was **0.019032**, PEV **0.015347**. The recorded coverage-minus-PEV difference was +0.003685, interval [+0.000452, +0.006917], p = 0.0276. The frozen hypothesis that endpoint coverage was better **failed**; the observed direction favored PEV. This supports competitiveness within the same generator, but should not be relabeled as a new successful directional hypothesis for PEV. See the [fresh endpoint replication](/Users/pat/code/ACE/results/research_pev_endpoint_replication_v1/README.md).

A three-system development study using a different sparse DAG generator found mixed rankings and **failed its promotion gate**. It did not justify a larger graph-shift confirmation. That result limits a broad generalization claim. See the [graph-shift report](/Users/pat/code/ACE/results/research_pev_random_dag_dev/README.md).

**My interpretation:** The record contains useful positive evidence and consequential negative controls. It supports a context dependent acquisition benefit, while leaving its precise cause and external applicability open. A random win by itself is too weak a basis for claiming that ACE is the best experimental design method.

## 4. What the current delivery paper establishes

The accepted delivery confirmation compares a final refit with unchanged online models using the **same collected histories**. Across 12 histories of one deterministic emulator, the geometric mean ratio of exact-level errors was **0.183**, with 95% interval **[0.102, 0.327]** and sign-flip p **0.00146484**. Eleven histories improved. History **124753321 worsened**, from 0.196339 online error to 0.395658 delivered median error, ratio **2.015177**; it remains in the result.

This supports a bundled final fitting procedure for those histories. The numerator is the **median of three scored optimization initializations**; it does not establish a deployable model-selection rule without test labels. Evaluation used a previously exposed grid that included acquired inputs, and fitting computation was unequal. The result is therefore neither an acquisition comparison against random nor an independent-domain or equal-compute demonstration. See the [same manuscript's confirmation section](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:2776), [generated numbers](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_claims.tex) and [claim index with scope and receipt hashes](/Users/pat/code/ACE/paper/aistats_ace_2027/claim_index.json).

The accepted physical prediction study supplies a further boundary: delivery beat rolling prediction in **7/11** conditions, physics regression in **3/11**, and Fourier regression in **0/11**. These are conditions of one apparatus, not 11 independent worlds. They prevent a general “ACE always outperforms simpler methods” claim.

The newer prospective study has completed its original 640 fits, evaluation and independent acceptance. Its supplemental exact-runtime numerical replay and final reporting remain unqualified at this writeup's cutoff. I use **no performance result from that study here**. Its randomized histories alone would not establish acquisition superiority: its registered primary contrasts concern delivered models versus specified model controls. See the [final integration contract](/Users/pat/code/ACE/docs/development/guidance/delivery_final_integration_2026-10-08.md) and [runtime recovery record](/Users/pat/code/ACE/docs/development/guidance/delivery_replay_runtime_recovery_2026-10-08.md).

## 5. Recommended public claims

**Use:**

> ACE: Make every experiment count.

> In controlled synthetic studies, an ACE uncertainty guided experimental policy achieved lower prediction error than a specified random policy at the same sample budget. Its benefits depend on the setting, and we compare it with systematic alternatives too.

For the current delivery research, use:

> ACE's delivery research studies how to build a better final model from observations already collected. We have observed substantial improvements in a specified emulator setting, alongside cases where delivery worsened or simpler physical models performed better.

**Avoid:** “ACE is proven better than random everywhere,” “65% more accurate,” “every experiment improves the model,” “the unique scoring formula caused the gain,” “equal-compute superiority,” or “proven laboratory decision improvement.” None follows from the evidence summarized here.

A stronger future general claim would require a separately frozen study on independently chosen tasks, a stated random policy, strong systematic controls, matched measurement budgets, explicit computation accounting, a deployable initialization rule and an independent held-out endpoint. All attempts and failures would need disposition. This describes the evidence needed for a broader claim; it is **not** a proposal to restart stopped acquisition agendas or add a fitted study to the present delivery paper.

## Evidence handling for this writeup

This assessment reads existing protocols, completed result summaries, source code and accepted delivery claim artifacts. A read-only extraction of the 80 shifted-mechanism cells confirmed the four final arithmetic means, 19/20 PEV-versus-random wins, 32 steps and 2,000 samples per cell, and matching system hashes within each four-arm comparison. The reported confidence intervals and p-values come from the existing completed study summary; no new hypothesis test was performed. Historical checksum and acceptance statements retain their original scope. This is not a new full custody audit, model replay or certification.

The original studies, frozen protocols, accepted attribution gate, manuscript and replay input packages were not changed to produce this assessment.
