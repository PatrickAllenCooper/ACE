# Numerical revision review — 2026-11-02

**Disposition: PASS for the numerical revision. No incorrect displayed number, rounding error, changed original empirical table entry, or newly implied significance claim was found.** One optional wording clarification and the pre-existing seed-implementation evidence limit are recorded below.

This is a bounded delta review of the current manuscript after the layout follow-up. It reuses numerical_evidence.json and directly checks the relevant accepted CSV entries. It does not repeat the original acceptance/custody audit or reconstruct the original statistics. Budget/paid-row verification and page layout are outside this assignment. No manuscript/source changes were made.

Reviewed snapshot (UTC 2026-11-02T16:19:34.970274+00:00):

- Manuscript: /Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex
- Manuscript SHA256: 10b52481220442adaf51e1efd2cd7a22475c388d440007ae82873aa64832a787
- Existing evidence: /Users/pat/code/ACE/results/delivery_panel_revision_20261102/numerical_evidence.json
- Evidence SHA256: 863bf11b79aed35a36a21d9b7eb4c7285e9b24858f5faf57a17d4a3bfb5b03fd
- Git baseline for preservation comparison: c6ba2c663ff97f9594bdf296172da0fd7b264942
- Line references below apply to this snapshot, after the quantization/counterexample appendix relocation. This review does not re-review those proofs.

## Ratios, wins, and their interpretation

The abstract (lines 3032–3040), collection-history paragraph (3500–3511), and conclusion (3668–3676) agree with the existing evidence. The abstract's five-node delivery/rolling value 2.46 correctly rounds the pooled NMSE ratio 2.4595599000390007. Thirty-node balanced delivery/flat 0.912 and 10/20 wins, and matched-random 0.168, are correct. These are NMSE ratios; they are not RMSE ratios.

All eight history-specific numbers and counts match, in the manuscript's five-node / thirty-node order:

- Balanced delivery/flat: 0.793 / 0.912; 13/20 / 10/20 wins.
- Matched-random delivery/flat: 0.160 / 0.168; 19/20 / 18/20 wins.
- Balanced delivery/rolling: 1.614 / 0.695; 4/20 / 12/20 wins.
- Matched-random delivery/rolling: 3.748 / 0.108; 3/20 / 19/20 wins.

The exact thirty-node balanced delivery/flat value remains 0.912253569631413 (0.912254 in the retained descriptive table). Wins mean strict lower delivery NMSE in that history, among twenty systems, and all eight contrasts have zero ties. The text correctly treats these as descriptive paired comparisons, identifies the thirty-node balanced result as mixed, and does not infer equivalence or collection-strategy superiority. The claim that most of the pooled flat advantage occurs on matched-random histories is consistent with the equal-history log aggregation and the much smaller matched-random ratios.

The four primary pooled directions and original ratios remain unchanged. The 100-update-ablation statement agrees with the retained descriptive delivery/short-fit ratios 0.00855424 and 0.0186544, both below one. The primary-floor statement agrees with zero activations in the existing evidence.

## Absolute scale

The percentages at lines 3515–3520 round correctly from the recorded medians of sqrt(raw NMSE), using the forty system/history cells per arm and family:

- Five-node delivery: 2.1781350630% → 2.2%.
- Five-node rolling: 1.6425576821% → 1.6%.
- Five-node flat: 3.3781508213% → 3.4%.
- Thirty-node delivery: 0.5312700468% → 0.53%.
- Thirty-node rolling: 0.8509944511% → 0.85%.
- Thirty-node flat: 0.8693159404% → 0.87%.

The abstract's approximately 0.5%–3.4% range correctly summarizes these six main-arm/family medians at init0. The wording identifies delivery, rolling and flat, so does not include short-fit. It describes medians, not a bound on individual cells or later initializations. The denominator is each history's training-target SD. The forty descriptive cells do not become forty independent systems. The small distinction between median(sqrt(NMSE)) and sqrt(median NMSE), documented in the evidence, changes none of the displayed rounded percentages. The text also correctly declines to infer downstream decision value or saved interventions.

## Added descriptive initialization table and sensitivity text

All sixteen ratio entries in the new table, label tab:initialization-contrasts at lines 4016–4033, equal the existing evidence rounded to three decimal places. In column order init0, init1, init2, log average:

- Five-node / rolling: 2.460, 2.271, 1.455, 2.011.
- Five-node / flat: 0.356, 0.163, 0.188, 0.222.
- Thirty-node / rolling: 0.274, 0.281, 0.220, 0.257.
- Thirty-node / flat: 0.391, 0.360, 0.264, 0.333.

The seed-average ratios before rounding are 2.010793242838399, 0.22174426696863472, 0.25705347697941544, and 0.3334833827723291, respectively. These are exponentiated average log contrasts: average each delivery/flat arm's log NMSE over init0/1/2 within system/history, retain rolling's one init0 log NMSE, subtract control from delivery, average the two histories and then twenty systems equally, and exponentiate.

The accompanying paragraph and caption preserve init0 primary, correctly match delivery and flat initialization indices, explicitly reuse rolling's single recorded init0 fit, describe score averaging rather than prediction ensembling/seed selection, and add no interval or significance test. All four directions really persist at init0, init1, init2 and the seed average. The main-text direction claim at lines 3522–3523 is therefore correct.

**Optional clarity improvement, not a numerical error:** at lines 4008–4013 and the header at 4020, 'Log average' could be called 'Seed-averaged ratio', or the prose could add 'subtracts the control log mean and exponentiates the averaged contrast.' The caption already calls the displayed values NMSE ratios, but the prose does not explicitly state the final subtraction/exponentiation. The displayed numbers are ratios, not raw logarithms. No edit was made.

## Extreme cell and initialization meaning

The main-text value at lines 3524–3527 and appendix at 4035–4041 both agree with the existing evidence and direct accepted all_cells.csv lookups for system 5:08, matched_random:

- Delivery init1: 0.023838762668516332 → 0.0238 in the main text and 0.02384 in the appendix.
- Flat init1: 0.04815276162685173 → 0.04815.
- Rolling's single init0 fit: 2.284389589287386e-05 → 2.2844 × 10^-5.
- Delivery/rolling NMSE: 1043.5506614242986 → approximately 1,044 and 1043.55.

The normalized-RMSE interpretation is not substituted for the NMSE ratio. The extreme belongs to init1 sensitivity, while rolling has no init1 replicate. The caption and surrounding text make this clear and preserve the primary fit and history. The within-history worst/best delivery NMSE medians 2.02 and 5.23 correctly round 2.0182564526321 and 5.2308956821612345. These are descriptive medians across the forty system/history cells, not added hypothesis tests.

The registration/CSV interpretation at lines 3436–3442 is consistent with repeated delivery/flat init labels 0/1/2 and a single rolling/short-fit init0 fit per history. **Evidence limit retained:** the implementation claims about nonroot head seed r + topological position and flat head seed r + node count (lines 3858–3862) cannot be independently established from this numerical_evidence.json and score CSVs. The prior evidence explicitly distinguishes registered labels from resolved RNG seeds/initial weights. This review neither verifies nor refutes that implementation description; it must rely on the separately established methods/source record. The added caveat that identical seeds do not imply identical functions is appropriate.

## Original table preservation

Independent comparison of current rendered-source editorial blocks against their retained companion table files confirms **all 21 original empirical tables, comprising 320 data rows, retain their original data fields**. Numeric strings, intervals, p-values, counts, system identifiers, and registered-gate entries are unchanged. After only arm-name/header-spacing normalization, every original tabular body matches its companion source:

- delivery_attribution_table.tex: 1 table(s), 16 data rows; all fields unchanged except allowed arm labels.
- delivery_prospective_primary_table.tex: 1 table(s), 4 data rows; all fields unchanged except allowed arm labels.
- delivery_physical_table.tex: 2 table(s), 22 data rows; all fields unchanged except allowed arm labels.
- delivery_history_table.tex: 1 table(s), 12 data rows; all fields unchanged except allowed arm labels.
- delivery_prospective_sensitivity_table.tex: 1 table(s), 12 data rows; all fields unchanged except allowed arm labels.
- delivery_prospective_absolute_table.tex: 8 table(s), 160 data rows; all fields unchanged except allowed arm labels.
- delivery_prospective_descriptive_table.tex: 7 table(s), 94 data rows; all fields unchanged except allowed arm labels.

Observed tabular wording changes are online→rolling, Online→Rolling, simpler→flat, 'Marginal95%'→'Marginal 95%', and 'Ratio/init0'→'Ratio/init 0'. Caption wording/labels are display changes and are not being mistaken for altered result fields. The seven retained companion table files themselves match HEAD. Their original embedded filecontents blocks match those companion files. Both claim-macro files and their embedded definitions are unchanged; primary numeric macro expansion therefore remains unchanged. The evaluation-design table has no altered numeric tokens. There are 22 direct empirical tables in the current document body: the 21 preserved originals and exactly one added descriptive initialization-contrast table.

The layout relocation and historical confirmation disclaimer deletion do not change this numerical verdict. No paper compilation or page-layout judgment was attempted.

## Inputs and boundaries

The review reused the existing evidence rather than redoing aggregation. Direct CSV checks matched the history ratios, the already-recorded per-system paired NMSEs underlying wins, the recorded median middle cells, and the extreme-cell scores. The new table was compared with already-recorded primary/init1/init2/seed-average ratios. No medians, geometric aggregates, confidence intervals, p-values, or hypothesis tests were recalculated.

Accepted CSV inputs:

- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_history_specific.csv — SHA256 eb54e606666ac231de8d58479030c917d7bfdfeac656aface3996e8cf82d2fe4.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_all_cells.csv — SHA256 ba93de3a81f2b4a5ac52cecb9f4d27ddde51540276ea132a3a67aee0fe44bd45.

Original table source digests (recorded for this comparison; no custody chain was re-audited):

- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_attribution_table.tex — SHA256 47d6bf0a4628897b68582b7ace91dae77e1b31e2b7d9965caca35f0aaf2ceee6.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_primary_table.tex — SHA256 552f5344d2d721e645ce2b0ce27cd7ffa63f069274da47a265b1ea6d4c40a36e.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_physical_table.tex — SHA256 eb7135ef4ba9fb5267c8f9c18ea1011d5a8d51b492f0cd951e4f9bc4fdabe950.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_history_table.tex — SHA256 f236205aeebecf7e43123c23644eb6730c16677dfc0895521fce97bf4a25c35a.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_sensitivity_table.tex — SHA256 78ee2ef4ae9d754096d73dded29f72ebae2e2105c2f07f95672dfdaf5a064364.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_absolute_table.tex — SHA256 971ce8da8c389304c1cfe67f211ac979eeda325f35fcc3774ff2d9e248bbb44f.
- /Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_descriptive_table.tex — SHA256 14c7fb649b57728c2e694d0ccc7a0bd745c55df1cad2d269c2c45463748758dc.

Only results/delivery_panel_revision_20261102/revision_numbers_review.md was written. No other edits, model runs, reporters, new tests, lock operations, or commit.

## Narrow wording disposition — 2026-11-02T16:20:38.124791+00:00

**Optional clarification closed.** Inspected only the additional descriptive initialization paragraph and table in current paper.tex, beginning at line 4007 (manuscript SHA256 110dc7aa2d8e582dc254e23a47f34cb32cfe1b7dfd7eb56b28e5809afa679778). The column now reads “Seed-averaged ratio”; the prose explicitly subtracts the control log mean from the delivery log mean and exponentiates the averaged contrast. This correctly identifies the displayed quantity. Equal-weight averaging and subtraction are linear, so the stated order is equivalent to the reviewed aggregation. The single rolling fit, descriptive status, init0 primary rule, and exclusion of prediction ensembling or seed selection remain clear.

A direct comparison of the four displayed rows confirms that all sixteen ratio entries are unchanged from the reviewed table. No numerical summaries or statistics were recomputed.

Per the user's update, Hilbert separately verified the procedure RNG claims against the frozen model/oracle source. That separate verification is recorded here; this numerical reviewer did not repeat the source audit. The earlier CSV/JSON evidence limit describes this review's evidence boundary, not an outstanding request to duplicate Hilbert's work.

This addendum closes only the wording follow-up. It is not approval of a future final packet. Standing by for the distinct actual final-packet review once the new packet exists. Only this review note was appended; no other edits, model runs, new tests, or commit.
