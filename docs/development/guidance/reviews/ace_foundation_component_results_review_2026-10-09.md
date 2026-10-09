# Foundation component pilot: independent results review

Date: October 9, 2026. **Disposition: 0 required corrections within the requested outcome/report scope.** The saved artifacts support 30 completed cells, the reported TabPFN composition comparison, and six language fallbacks. They do not support a benefit from valid language proposals or an acquisition-efficiency claim.

## Evidence and review boundary

Reviewed the [frozen protocol](/Users/pat/code/ACE/docs/development/guidance/ace_foundation_component_pilot_2026-10-09.md), the [freeze](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-component-freeze-20261009-01/freeze.json), and the pilot's [summary](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-component-pilot-20261009-01/descriptive_summary.json), [terminal](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-component-pilot-20261009-01/terminal.json), plan, completion, launch, startup and preflight records. Also read all 30 cell JSON files, six saved training/evaluation archives, 30 prediction arrays, six truth records and six raw language responses. Array decoding and arithmetic used Python's standard library; no scientific libraries, models, fitted estimators, research programs or tests were executed. Only this report was written.

Independently recomputed byte hashes:

- Freeze: `e7c5ac2f837fe8251f2eb2637137e152c02003b0f922a10076fd76c516f7ebad`.
- Terminal: `846723f540122ad74d645ef8221ad21350d5ff3b872cce75da9635ac77edced1`.
- Descriptive summary: `ebb3c03b07956b62b650098a22cb9328cc6823a895392c40889f0e50e6b703ad`.

The launch and worker startup records bind the actual freeze hash; the summary binds the actual terminal bytes. Worker, supervisor, protocol and reporter pins match both their current files and Git revision `68248cdd46248479f7452659291a6d617f4eff28`. Preflight's nine dependency versions, seven language-file hash records, protocol hash and configuration match the freeze. Both frozen compatibility-smoke terminal hashes and preparation-freeze hashes match their saved files; both smoke terminals report completion and exit 0. This checks recorded weight provenance, without rereading model weights or revalidating runtime behavior.

## Pairing, cells and saved errors

- The plan contains exactly seeds 91000–91005 crossed with polynomial, ExtraTrees, TabPFN v2, grammar and language: **30 unique planned cells**. All 30 are complete. Each cell JSON equals its terminal entry; terminal, completion and summary cell ledgers agree. No failed, interrupted, unattempted or omitted cell is hidden by aggregation.
- Each world has one shared saved training array of shape `(32, 3)` and one shared evaluation array of shape `(256, 3)`. Clamp labels are eight observational rows, twelve root interventions and twelve M interventions; saved intervention menus match the prescribed cycle. Every method records 20 eligible M labels and 32 eligible Y labels. Thus there are **192 shared training responses**, not 960 independent method-specific responses, and **1,536 common evaluation actions**, each scored by all five methods.
- Truth records preserve the prescribed family pairs: linear/quadratic, quadratic/tanh and tanh/linear, repeated with distinct seeds and coefficients. Saved evaluation targets agree with their recorded deterministic mechanisms within `2.22e-16`; evaluation roots remain in `[−1, 1]`. No new worlds or responses were generated during this review.
- All prediction arrays have shape `(256, 3)`, with columns M-local, Y-local and Y-composed, and contain finite values. TabPFN predictions are saved as float32; the other predictions are float64. Independently recomputed all **90 endpoint MSEs**, eligible-label population variances and NMSEs. They match the cell records within floating-point rounding; maximum absolute MSE discrepancy is `1.74e-18`.
- All reported NMSEs are finite and strictly positive. There are no normalization-floor activations; the smallest eligible training variance is approximately `0.149150`. All 12 prespecified comparisons include all six paired worlds, with no undefined ratios, substituted ratio floors or complete-case filtering. Their per-world ratios, arithmetic means and geometric means match independent arithmetic.

## TabPFN and language interpretation

Ratios are **candidate NMSE / grammar NMSE; lower is better**. For TabPFN's composed endpoint, the six ratios in seed order are:

- 91000: `1.8817425822`.
- 91001: `8.7928941375`.
- 91002: `0.3848225538`.
- 91003: `2.1204568912`.
- 91004: `4.5407213265`.
- 91005: `5.5629205745`.

Their geometric mean is **`2.6432300019251946`**, and their arithmetic mean is `3.880593010950594`. TabPFN has higher composed error in **5/6** worlds; seed 91002 is the exception. The geometric mean is not a ratio of pooled or arithmetic-mean errors. Its M-local and Y-local geometric ratios are `1.4504724442` and `6.3889699122`, respectively; Y-local is worse in all six worlds. These are descriptive results for this fixed numerical configuration and small development screen.

Language has **0/6 valid proposals and 6/6 grammar fallbacks**. Each raw response fails the frozen strict JSON interface; the saved invalidity and fallback flags agree. Five generations stop below the 64-token cap and one reaches it. For every world, language-selected families and metrics equal grammar's, and the entire saved language prediction file is byte-identical to grammar's prediction file. All language ratios and aggregates therefore equal **1 solely because of deterministic fallback**. There is no valid-proposal subset and no evidence of a valid language-selection benefit. All six language cells remain in the planned pipeline summary.

For context, composed-error geometric ratios for polynomial and ExtraTrees are `1.8290201948` and `12.9572040424`, respectively. This preserves the declared data-only comparisons; it does not establish that grammar is the strongest possible learner.

## Actual budget and limits on conclusions

The [terminal receipt](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ace-foundation-component-pilot-20261009-01/terminal.json) reports exit 0, no error, **13.9936005 seconds elapsed**, **14.526519 child CPU seconds**, and `0.013084` supervisor CPU seconds. Whole-child accounting includes startup, imports and loading, unlike the narrower completion record's `12.306831` seconds elapsed and `12.927080` CPU seconds after preflight. Both authoritative totals are below the frozen 1,800-second wall and child-CPU limits.

Recorded peak child RSS is **5,062,934,528 bytes (4.715 GiB)**. Launch, startup, preflight and freeze agree on one CPU thread and CPU-only execution; GPU seconds are recorded as zero. The local pilot protocol specifies no 3-GiB memory cap: that cap belongs to the separate delivery replay. Receipts support the configured thread/device limits, without independently measuring utilization. The freeze's 5,880 CPU-second combined reservation is explicitly a reservation, not measured preparation-plus-pilot consumption; this review does not convert it into an observed total. Accepted-study responses are recorded as zero; the 192 new synthetic training responses are separately accounted above.

The justified conclusion is that this exact TabPFN configuration did not improve composed prediction over the numerical grammar comparator on this six-world development screen, while the specified language interface produced no valid proposals. This is not confirmation, a general rejection of foundation models, causal-identification evidence, or an intervention/sample-efficiency result. The known fully observed graph, familiar mechanism families, shared fixed histories, noise-free evaluation and language-versus-grammar selection-information difference remain the frozen scope. No significance test, valid-only language aggregate or favorable rerun is needed to describe these outcomes faithfully.
