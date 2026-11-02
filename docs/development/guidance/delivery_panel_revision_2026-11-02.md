# Revision in response to the November 2 model panel

This revision improves the existing TMLR manuscript in place. The six models' two review rounds are internal feedback, not twelve independent reviews or a venue decision. No new experiment, model fit, response, or inferential test is part of this revision. Original studies, failed attempts, the November 1 packet, and generated scientific reports remain preserved.

## Scientific interpretation

The abstract, main results and conclusion now identify the registered equal-weight average over balanced and matched-random histories. The thirty-node balanced delivery/flat ratio is 0.912 with 10 of 20 systems improving. The matched-random ratio is 0.168 with 18 of 20 improving. The balanced result is described as mixed, not statistical equivalence. The full history-specific split against both controls appears beside the registered primary table.

Absolute scale is stated in the abstract and next to the main results. These are medians of sqrt(NMSE) over forty system/history cells per arm and family, not square roots of rounded median NMSEs or ratios of medians. The six primary arm/family medians range from approximately 0.5% to 3.4% of training target standard deviation.

The main text reports 48,000 rolling updates per head versus 30,000 full-batch delivery updates, including the current-row duplication and retrospective calibration. It gives 3 and 25 structured heads versus one flat head, parameter counts, and original mean fit process CPU by arm and family. These are complete recipes with unequal supervision and computation.

The primary intervals condition on one shared initialization scheme. A new descriptive table reports all three registered initialization contrasts and log-average summaries without extra tests or seed selection. System 5:08 under matched-random collection has initialization-one delivery NMSE 0.02384, about 1,044 times its single rolling fit's error. This failure and within-history sensitivity are explicit. The five-node prospective reversal remains undiagnosed.

## Panel disposition

- R01 pooled estimand and history heterogeneity: abstract, Section 6 and conclusion.
- R02 absolute scale and practical interpretation: abstract and Section 6.4.
- R03 restricted generator population: Sections 6.1 and 10, Appendix B.
- R04 unequal update, head, parameter and CPU budgets: Section 6.2.
- R05 conditional initialization and severe individual failure: Sections 6.2 and 6.4, Appendix F and its new descriptive table.
- R06 registered inference preserved: Equation 2, all four contrasts, twenty systems per family, marginal intervals, Holm adjustment, floor and gate unchanged.
- R07 attribution scope and weaker matched-CPU flat schedule: Section 5.
- R08 undiagnosed reversal separated from archived propagation example: Sections 6.4 and 10.
- R09 fixed-emulator exposed-grid confirmation and scored median: Section 4, with the interrupted continuation described in Appendix D.
- R10 paid-row arithmetic: Appendix A gives 3 + 500 + 5(500) + 200(4) + 40(25) = 4,803, alongside 1,178 admitted rows and the 50-row buffer.
- R11 physical sample and evaluation units: Section 7 and Appendix C give 1,000 readings per condition, condition-specific train/test counts and eight held-out blocks.
- R12 physical uncertainty and nested controls: ten of eleven delivery/rolling intervals contain one, with no equivalence claim. Fourier's basis contains the physics regressor. The 0/11 Fourier result is preserved.
- R13 process evidence versus released evidence: shorter Section 9 and Appendix E distinguish author-retained execution records from anonymous saved-report reconstruction.
- R14 reader guide and setup: existing mechanism diagram and evaluation-design table retained, with concise mechanism-family context in Section 6.1.
- R15 theory assumptions: the eligibility proof makes the pre-selection conditional mean explicit. Propagation and joint-support assumptions are unchanged. Quantization results, proofs and counterexamples are relocated intact to Appendix G.
- R16 clarity: shorter process and discussion prose, consistent rolling/flat display labels, readable captions, no semicolons or em dashes in visible narrative. Official style and AI-assistance disclosure remain.
- R17 obsolete agenda annotations: historical indexes remain unchanged. The new anonymous packet uses an explicitly identified publication projection of the historical claim index, omitting only obsolete agenda fields and preserving every empirical field and adverse boundary. The predecessor digest and projection scope are recorded.

Ash and Adams (2020), *On Warm-Starting Neural Network Training*, is added as the closest retraining precedent. The text does not claim that its explanation diagnoses this study's reversal.

Optional new experiments, checkpoint diagnoses, additional inference and acquisition-policy studies are deferred. The revision makes no pretraining, acquisition-efficiency or pure architecture claim. The all-paid/admitted continuous and snapped contrasts, weaker matched-CPU flat schedule, worsening archived history, five-node rolling loss and physical Fourier losses remain visible.

## Evidence and release scope

Numerical and procedure evidence, separate delta reviews, page checks and final packet pins are under `results/delivery_panel_revision_20261102`. The original prospective bundle remains pinned at `2cca8a3699bcee68adef1a7fe610df87a6518a39513f71b310e990d73183cd84`. Inline publication table copies change labels and captions while retaining original numeric entries. Original generated companions, all saved prospective CSVs and both original claim indexes remain unchanged in the repository.

Human authors still must approve the exact revised version and submission. The confirmed author order, affiliation, funding, competing-interest and overlap declarations are retained without asking again. No submission or public upload is authorized by this revision record.
