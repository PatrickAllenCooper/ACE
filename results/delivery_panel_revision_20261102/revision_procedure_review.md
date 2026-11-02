# Revised procedure claims: bounded delta review

Reviewed 2026-11-02T16:16:06.461450+00:00, using the current `paper.tex` after the reported theory/counterexample relocation. **All reviewed numerical values match the existing procedure evidence. One required wording clarification and two optional clarifications remain.**

Scope: compare only the revised update budgets, current-row duplication, head/parameter counts, six mean fit-CPU values, archived 4,803-row accounting and physical row counts against the existing [procedure evidence](/Users/pat/code/ACE/results/delivery_panel_revision_20261102/procedure_evidence.md) and [JSON addendum](/Users/pat/code/ACE/results/delivery_panel_revision_20261102/procedure_evidence.json). No original receipts, frozen source, archives or experiment outputs were reopened. No experiments, model imports, replays, tests, custody audit, layout check, math/proof review, paper edit or commit was performed. The layout relocation and editorial label changes were used only to anchor findings to the current file.

## Required

**R1 — Remove unsupported chronology from the paid-row introduction.** [paper.tex:3728](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3728), the sentence beginning “Each independently seeded history starts with three observations. It then obtains…” turns the count decomposition into a chronological acquisition claim. The existing report establishes the five paid-row terms and their totals; it does not establish that the seed observations precede validation acquisition. This review therefore cannot endorse that added sequence. This is an evidence-scope issue, not a discrepancy in the 4,803 total; no alternative execution order is newly asserted here.

Suggested replacement: “Each independently seeded history includes three seed observations, 500 observational validation responses, and 500 interventional validation responses for each of five mechanisms.” Continue with the iteration/refresh description and existing equation. This preserves every verified term without inferring execution order from the order of addition. Evidence: JSON `/archived_history/equation`, `/archived_history/terms`, and report item 4.

## Optional

**O1 — Make the local consolidation budget's unit explicit.** [paper.tex:3743](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3743), “100 consolidation updates per iteration” can be mistaken for the complete learner's total number of optimizer calls. The surrounding report distinguishes per-head training, the fast phase and repeated consolidation use. Qualify this as the per-head consolidation budget and avoid presenting it as the entire acquisition-stage optimizer budget. This is an optional precision improvement: “consolidation” already distinguishes it from fast adaptation, and the main prospective paragraph correctly says “per head.” Do not apply the prospective 400-arrival total to the separate archived acquisition procedure. Evidence: report items 1 and 4 and JSON `/prospective/rolling`.

**O2 — Put the full CPU sample scope next to the six means.** [paper.tex:3425](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3425), “forty primary fits per arm and family” is correct, and nearby prose establishes initialization zero and the two histories. For a self-contained comparison, use “forty initialization-zero fits per arm and family (20 systems × 2 histories).” This avoids treating the 40 fits as 40 independent systems or mixing sensitivity/ablation fits into the means. Evidence: JSON `/prospective_primary_fit_cpu_addendum_20261102/scope` and `/groups`.

## Claims that agree with the evidence

- **Prospective optimizer budgets and batches:** [paper.tex:3410](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3410) correctly reports 30,000 full-batch delivery/flat updates per head versus `400(20+100)=48,000` rolling updates per head, with the calibration prefix included. “Up to 50 rows plus a second copy” accurately summarizes `min(t,50)+1` consolidation entries for arrivals `t=1..400`. The current query occurs twice in consolidation; fast adaptation uses one singleton current row. The duplication does not double the optimizer-step count or create a new paid response. The explicit warning that update counts do not equate compute is appropriate.
- **Heads and trainable parameters:** [paper.tex:3418](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3418) agrees: 3 versus 1 heads in the five-node family and 25 versus 1 in the thirty-node family; thirty-node delivery totals `25×30,000=750,000` head-updates. Structured/flat parameter counts are 13,123/4,417 and 109,465–109,913/4,609. These are model totals, not per-head counts. The complete-recipe and unequal-supervision qualifications prevent a misleading architecture-only interpretation.
- **All six means are correctly rounded to one decimal:** [paper.tex:3426](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3426) gives five-node delivery/rolling/flat **58.1 / 77.9 / 20.1 seconds** and thirty-node **556.7 / 726.8 / 22.8 seconds**. The evidence means are respectively **58.13621915535 / 77.865160465575 / 20.091165984325** and **556.7061086852 / 726.831553823625 / 22.76988371625**. Each group has 40 init0 fits; these are means per fit, not sums or per-head timers.
- **CPU timing scope is accurate:** [paper.tex:3428](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3428) says training loop plus bookkeeping, excluding *initial* runtime setup, model initialization, checkpoint writing and evaluation. That matches the report's receipt-level process timer. “Initial” matters: in-fit helper/runtime calls and tensor construction remain inside the timer. The paper does not call these optimizer-only, end-to-end, scheduler-allocated or wall-clock measurements. Explicitly spelling out pre-timer normalizer/optimizer initialization is unnecessary here.
- **Archived paid-row equation and reuse:** [paper.tex:3728](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3728) through the admitted-union paragraph correctly gives `3+500+5×500+200×4+40×25=4,803`, 200 selected and 600 unselected probes, `3+200+39×25=1,178` admitted rows, final buffer size 50, and reuse of eligible acquisition-side validation observations. Selecting or reusing a query adds no response. The formula is scoped to each retained history; it does not purport to replace the campaign's aggregate charged calls including discarded attempts. Only the introductory chronology needs the R1 clarification.
- **Physical counts and separate procedure:** [paper.tex:3530](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3530) correctly states 11 conditions, 1,000 rows each, 772–815 train and 185–228 test, with `white_64` excluded. Every condition-specific pair at [paper.tex:3884](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3884) matches the evidence: blue 128/255/64 = 790/210, 815/185, 793/207; green = 794/206, 789/211, 775/225; red = 783/217, 772/228, 774/226; white 128/255 = 787/213, 778/222. The physical appendix separately describes 100 rolling updates per arriving training row; the prospective 20-fast-plus-100-consolidation rule is not incorrectly transferred to it.

## Review input identities

These digests identify the manuscript and existing reports read for this delta review only. They are not a renewed validation of raw custody.

- [paper.tex](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex): SHA-256 `3d0273ff8ed563be52b9bcfdbd4abcab5621e5beb5d4651fc5c641634ce01106`.
- [procedure_evidence.md](/Users/pat/code/ACE/results/delivery_panel_revision_20261102/procedure_evidence.md): SHA-256 `eb2d2a4b9b617bb90ff7682d5a727f32e7761d6eb80a87f021f69920b8308eb3`.
- [procedure_evidence.json](/Users/pat/code/ACE/results/delivery_panel_revision_20261102/procedure_evidence.json): SHA-256 `c2d55bc0a5977c836ca83a22fb5f5e0864bf2074cf84d637d41e4f7919bc2d4f`.

Relevant existing JSON sections: `/prospective/rolling`, `/prospective/heads`, `/archived_history`, `/physical`, and `/prospective_primary_fit_cpu_addendum_20261102` (especially `/groups`, `/timing_definition`, `/parameter_summary` and `/current_observation_duplication`). Source identities and original receipt digests remain in those unchanged reports; this review relies on them as already established.

## Narrow disposition — November 2, 2026

All three requested wording edits are confirmed in the current manuscript:

- **R1 resolved:** [paper.tex:3731](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3731) now says “includes three seed observations,” removing the unsupported acquisition chronology.
- **O1 addressed:** [paper.tex:3746](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3746) now specifies “100 consolidation updates per head per iteration.”
- **O2 addressed:** [paper.tex:3427](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3427) now identifies “forty initialization-zero fits per arm and family,” followed by “20 systems × 2 histories.”

No findings remain open from this procedure delta review. This disposition checks only those three edits; no raw evidence was reopened and no broader review was repeated. Earlier review content is preserved.
