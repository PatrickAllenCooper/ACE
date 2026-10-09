# Retention-aware SCM candidate selection: fresh pilot results

October 9, 2026. Exploratory development evidence; all four predeclared scenarios retained.

## What this changes

The flexible numerical control recovers the large advantage over a deliberately misspecified grammar. A pretrained model adds conditional value, especially when the downstream Y mechanism is missing from that grammar, but it is not a generally superior replacement. Training-only retention constraints reduce some local harm without guaranteeing private prediction improvement.

The study completed 216/216 cells: six fresh systems (93000–93005), four paired scenarios, nine methods. A separate same-source artificial integration completed 36/36 before scientific admission. No failed, replaced or filtered cell; no variance-floor activations. Saved-prediction reporting reconstructs 648 endpoint records, 96 reference contrasts, 72 direct contrasts and 384 local-harm records.

## Mechanism and comparison

The known graph is X → M → Y. Retained heads are fitted to pre-change data. Updated heads use a compact numerical grammar, a fixed non-pretrained RBF kernel ridge regressor, or TabPFN. The selector chooses a pair using composed Y error on five natural-M calibration rows, with predicted M fed to the Y head. Its local constraint admits an updated head only if it strictly improves measured-parent calibration MSE over the retained head. Its interval gate uses the retained function outside that head’s post-fit parent range. Ties favor retention.

Four selector ablations (raw, local, interval, combined), combined without PFN, and four fixed predictors (Grammar32, RBF24, PFN24, prechange24) are all reported. RBF24 and PFN24 share fitting eligibility; the RBF uses fixed gamma=1 and alpha=0.01. Grammar32 uses all 32 paid post-change rows; candidates use 24 fitting rows and reserve eight for calibration. This control tests a particular fixed numerical alternative, not pretraining as an isolated causal factor.

Across six seeds there are 960 shared training responses and 18,432 private probe responses. Each cell has 256 noise-disabled composed-Y root probes, 256 local-M probes and 256 local-Y probes. Local Y probes include extrapolation. NMSE uses the common full post-history eligible-label variance; no floors were activated. The calibration outcomes are noisy while private structural endpoints are noise-disabled. No private outcomes tune selection or refitting.

## All composed-Y results

Numbers below are geometric means of six within-system NMSE ratios; lower than one favors the candidate. Wins count strict ratio < 1. Full arithmetic means, every paired ratio, absolute errors, selections and support partitions are in the verified summary and interactive report. These are descriptive summaries, not population guarantees or extra hypothesis tests.

### null

- rbf24 / Grammar32: **5.816971**, 1/6 wins; arithmetic mean 8.364053.
- pfn24 / Grammar32: **3.876067**, 0/6 wins; arithmetic mean 4.533131.
- prechange24 / Grammar32: **1.966851**, 1/6 wins; arithmetic mean 2.194455.
- raw / Grammar32: **5.352064**, 0/6 wins; arithmetic mean 6.904939.
- local / Grammar32: **4.767695**, 0/6 wins; arithmetic mean 6.382963.
- interval / Grammar32: **3.819259**, 1/6 wins; arithmetic mean 5.552221.
- combined / Grammar32: **3.628198**, 1/6 wins; arithmetic mean 5.824375.
- combined_no_pfn / Grammar32: **3.636821**, 1/6 wins; arithmetic mean 5.650130.

Direct contrasts (wins / ties / losses):

- pfn24 / rbf24: **0.666338**, 3/0/3; arithmetic mean 1.309934.
- combined / raw: **0.677906**, 2/3/1; arithmetic mean 0.788062.
- combined / local: **0.760996**, 1/4/1; arithmetic mean 0.871443.
- combined / interval: **0.949974**, 1/4/1; arithmetic mean 0.995083.
- combined / combined_no_pfn: **0.997629**, 1/3/2; arithmetic mean 1.013300.
- combined / rbf24: **0.623726**, 3/0/3; arithmetic mean 0.912927.

### coefficient_M

- rbf24 / Grammar32: **4.804833**, 0/6 wins; arithmetic mean 5.370075.
- pfn24 / Grammar32: **2.699312**, 0/6 wins; arithmetic mean 3.602496.
- prechange24 / Grammar32: **209.463177**, 0/6 wins; arithmetic mean 388.339897.
- raw / Grammar32: **2.469386**, 0/6 wins; arithmetic mean 2.636954.
- local / Grammar32: **2.469386**, 0/6 wins; arithmetic mean 2.636954.
- interval / Grammar32: **2.375848**, 0/6 wins; arithmetic mean 2.540879.
- combined / Grammar32: **2.375848**, 0/6 wins; arithmetic mean 2.540879.
- combined_no_pfn / Grammar32: **2.060100**, 1/6 wins; arithmetic mean 2.326196.

Direct contrasts (wins / ties / losses):

- pfn24 / rbf24: **0.561791**, 5/0/1; arithmetic mean 0.715442.
- combined / raw: **0.962121**, 1/3/2; arithmetic mean 0.990293.
- combined / local: **0.962121**, 1/3/2; arithmetic mean 0.990293.
- combined / interval: **1.000000**, 0/6/0; arithmetic mean 1.000000.
- combined / combined_no_pfn: **1.153268**, 1/2/3; arithmetic mean 1.233417.
- combined / rbf24: **0.494470**, 4/0/2; arithmetic mean 0.634826.

### missing_M

- rbf24 / Grammar32: **0.014934**, 6/6 wins; arithmetic mean 0.015365.
- pfn24 / Grammar32: **0.025344**, 6/6 wins; arithmetic mean 0.028614.
- prechange24 / Grammar32: **1.005552**, 2/6 wins; arithmetic mean 1.014696.
- raw / Grammar32: **0.012249**, 6/6 wins; arithmetic mean 0.013178.
- local / Grammar32: **0.010585**, 6/6 wins; arithmetic mean 0.012856.
- interval / Grammar32: **0.012249**, 6/6 wins; arithmetic mean 0.013178.
- combined / Grammar32: **0.010585**, 6/6 wins; arithmetic mean 0.012856.
- combined_no_pfn / Grammar32: **0.007659**, 6/6 wins; arithmetic mean 0.008570.

Direct contrasts (wins / ties / losses):

- pfn24 / rbf24: **1.696999**, 1/0/5; arithmetic mean 1.994425.
- combined / raw: **0.864190**, 1/4/1; arithmetic mean 0.966918.
- combined / local: **1.000000**, 0/6/0; arithmetic mean 1.000000.
- combined / interval: **0.864190**, 1/4/1; arithmetic mean 0.966918.
- combined / combined_no_pfn: **1.382026**, 0/3/3; arithmetic mean 1.457598.
- combined / rbf24: **0.708793**, 5/0/1; arithmetic mean 0.886327.

### missing_Y

- rbf24 / Grammar32: **0.037581**, 6/6 wins; arithmetic mean 0.044425.
- pfn24 / Grammar32: **0.017741**, 6/6 wins; arithmetic mean 0.020260.
- prechange24 / Grammar32: **1.215121**, 2/6 wins; arithmetic mean 1.315069.
- raw / Grammar32: **0.005710**, 6/6 wins; arithmetic mean 0.011787.
- local / Grammar32: **0.005710**, 6/6 wins; arithmetic mean 0.011787.
- interval / Grammar32: **0.005710**, 6/6 wins; arithmetic mean 0.011787.
- combined / Grammar32: **0.005710**, 6/6 wins; arithmetic mean 0.011787.
- combined_no_pfn / Grammar32: **0.007005**, 6/6 wins; arithmetic mean 0.014257.

Direct contrasts (wins / ties / losses):

- pfn24 / rbf24: **0.472055**, 6/0/0; arithmetic mean 0.584868.
- combined / raw: **1.000000**, 0/6/0; arithmetic mean 1.000000.
- combined / local: **1.000000**, 0/6/0; arithmetic mean 1.000000.
- combined / interval: **1.000000**, 0/6/0; arithmetic mean 1.000000.
- combined / combined_no_pfn: **0.815129**, 3/3/0; arithmetic mean 0.876046.
- combined / rbf24: **0.151933**, 6/0/0; arithmetic mean 0.333099.

## Local harm and support

For the unchanged Y mechanism, combined selection has positive private local-MSE difference from retention in 4/6 null systems, 2/6 coefficient-M systems and 0/6 missing-M systems. PFN24 and RBF24 each worsen local Y in all six systems of each of those three scenarios. Combined reduces broad extrapolation harm but does not certify preservation: null mean local-Y difference remains +0.000350111. In the coefficient-M and missing-M scenarios its mean differences are −0.0000669172 and −0.0000170609. Positive means worse than the retained head at the same local probe inputs.

The post-fit M-parent interval is always [−1,1] and contains every planned root/local-M probe. Outside-M partitions are empty and undefined, not zero-error evidence. The direct extrapolation gate is exercised at Y. A preserved Y function at a fixed parent does not imply preserved composed output when the M head changes. Saved support diagnostics expose both local inside/outside errors and the range of predicted M inputs used for composed Y.

## Interpretation and methodological recommendation

1. **Separate flexibility from pretraining.** RBF24 beats PFN24 on five of six missing-M systems; PFN24 beats RBF24 on all six missing-Y systems. Both beat the inadequate grammar on every missing-family system. The earlier PFN/grammar advantage therefore cannot all be attributed to pretraining.
2. **Separate candidate quality from selector reliability.** Adding PFN to combined selection improves missing-Y in three systems and ties three; it harms missing-M in three and ties three. Null and coefficient-change effects are mixed. Even training-only choices can generalize poorly.
3. **Retain a simple reference when it is adequate.** Combined loses to Grammar32 in five of six null systems and all six coefficient-change systems. Do not deploy the selector as a universal replacement. Which family is adequate is known to this experiment’s designer, not discovered by the selector.
4. **Do not credit guards for identical forecasts.** All four selector ablations have identical composed-Y predictions/errors in missing-Y. That stratum’s large target gain does not identify a benefit of the local constraint or interval gate.
5. **Keep inference and acquisition separate.** This fixed-history experiment establishes no reduction in interventions, no real-world transfer and no ability to identify changed mechanisms.

A useful next design would measure selection reliability before claiming acquisition efficiency: fresh systems, a predeclared training-only decision rule, fixed validation-budget variants, and reporting of both final prediction error and unchanged-mechanism harm. Candidate-set expansion lowers the minimum observed calibration loss, but cannot by itself lower private risk. Overfitting five composed calibration rows is a plausible explanation here, not an identified cause. The calibration/structural-endpoint noise mismatch is another possible contributor. Never select a winning node-specific model or threshold from these private outcomes.

After that concern is resolved, an estimator × acquisition experiment can cross fixed estimators with random, coverage and uncertainty acquisition using identical legal action menus, charged budgets and independent stopping/evaluation. Its protocol must distinguish better prediction from genuinely fewer interventions. No new run is scheduled by this recommendation; remaining budget alone is not a scientific reason to execute.

## Source, custody and qualification

- Scientific source revision: `7cdacc338cc06a6e22a22a21604255dbcfbbd891` (nine source/protocol bindings).
- Scientific freeze SHA256: `ef1f1f0907ba4aecb243819d18a50016afb6733a8b7372ac39e8ef4f5125dd80`.
- Scientific terminal SHA256: `8ebe9573f2c2f172e6b3bb2da90286a9940848d7ae16289fd5f77c5081379711`.
- Verified scientific summary SHA256: `3d0903195419aa4d71b3e2e12bab178fd13735a337383a89300b9ad223eecfc4`.
- Artificial freeze SHA256: `fddc80b52b0e94619b47d6d16a13202e4edd818ca4f5ba11ad129b992fa860c4`.
- Artificial terminal SHA256: `1c37824d977c31891693021c7a89a9bf8c53c85dd44307041c23d13d403ee90e`.
- Verified artificial summary SHA256: `fdfc3536f53f20fc4ba024e755c5c3d154f81faf677da312e5cc20d2dfcbfab2`.
- Raw repository custody: `results/ace_foundation_retention_20261009/custody.json`, SHA256 `f38ed7d8646475309711c7852f163b34230b8aa9f6be29fe7e6ed677c5aad475`, 1,296 exact copies / 7,645,350 bytes. Exclusive external originals remain in `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-retention-{freeze,fixture,pilot}-20261009-*`.

Actual runtime is unchanged: Python3.11.15, NumPy2.4.6, torch2.14.1, transformers4.51.3, TabPFN2.2.1, scikit-learn1.6.1, SciPy1.17.1, safetensors0.8.0, tokenizers0.21.4 and huggingface-hub0.36.2. The fixed TabPFN-v2 checkpoint SHA256 is `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Finite repeat/partition checks passed on five fit-derived candidate inputs; they do not prove universal pointwise behavior. The reporter reconstructs saved prediction metrics and checks recorded selection rules; it does not refit or independently replay model calibration.

## Resources and disposition

Artificial qualification: 5.518750 child CPU seconds, 6.126281166 elapsed seconds, 747,831,296 peak RSS bytes; supervisor 0.015172 CPU seconds. Scientific pilot: 27.594859 child CPU seconds, 28.035219417 elapsed seconds, 782,155,776 peak RSS bytes; supervisor 0.029639 CPU seconds. Both executed once within respective 120/900-second caps; no GPU or new installation.

Measured model attempts across this development cycle now total 90.044439 child CPU seconds plus 0.108012 supervisor CPU seconds. Preparation, unit checks, reporting, reviews and Git are excluded and partly unmetered; this is not whole-sprint cost. Conservative reserved CPU is 7,920/28,800 seconds, GPU zero. Overlapping model/head/variant timings are not additive.

Accepted A/B/C, completed earlier pilots, failed single-use delivery replay06 and completed presentation v6 remain unchanged. No Stage B reporting or publication follows. The current cycle still stops and reconciles at 22:00 UTC. A distinct post-run review is pending when this draft is first written; its disposition will be appended without changing frozen study files.

## Completed independent numerical review

The distinct [post-run review](reviews/ace_foundation_retention_results_review_2026-10-09.md) found **zero required corrections**. It independently checked all 1,296 custody entries, the complete matrices and response chronology, all saved endpoint/paired/harm calculations, recorded selections/support and cost claims without loading models or rerunning the reporter. Its reviewed draft digest is preserved separately; this appended disposition changes no scientific result. The interactive artifact has a separate presentation-consistency review and is not covered by that numerical review.
