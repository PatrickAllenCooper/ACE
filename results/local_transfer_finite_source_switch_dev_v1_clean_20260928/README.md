# Finite-source scratch switch development (28 September 2026)

The [frozen protocol](../../docs/development/guidance/protocol_transfer_finite_source_switch_dev_2026-09-28.json) compares a source posterior against a broad scratch prior using only target observations. Four observations per node nominate a possible change; the later acquired observations must supply a predictive evidence increment greater than log(4) before the policy uses scratch. Otherwise it retains the fitted source posterior. Code revision: `41d1cc5` (full hash in `suite_complete.json`). This reuses the same 12 development systems, source acquisitions, uniform target prefixes, and 120/200/400 target budgets as the [parent screen](../local_transfer_finite_source_target_dev_v1_20260928/README.md).

The frozen development gate **fails**: 18 of 24 source-size/change-type/k/node-group comparisons at the 200-response budget pass, while six do not. The switch makes zero false switches on unchanged nodes in these systems and retains their warm-source MSE. Family-changed node MSE relative to scratch at 200 target responses is:

| Source examples/node | k=1 | k=3 | k=10 |
| --- | ---: | ---: | ---: |
| 16 | 1.267 | 0.870 | 1.760 |
| 64 | 1.572 | 0.920 | 1.483 |

All three k strata had to be at most 1.05. Changed-node switches at k=1/3/10 occur in 10/12, 35/36, 98/120 cases with 16 source examples, and 10/12, 34/36, 110/120 with 64. High switch counts can coexist with worse MSE because the remaining unswitched family changes are costly. For coefficient changes, k=1 and k=10 also miss the gate under at least one source-size arm. These are reused synthetic development systems, not a fresh confirmation.

All 144 cell receipts and 12,960 node rows validate. Warm and scratch MSE were independently recomputed and matched their corresponding parent cell metrics; source and parent hashes matched; exact uniform target counts were checked. A deterministic replay matched all cell and suite receipts. Source training responses (28,800 across both source-size arms) remain separately counted, and the 144 setting cells each use at most 400 target responses (57,600 across cells). The fitter receives no latent coefficient or changed-node label; those are used only by the evaluator. No CURC job or closed-model call was made.

The result supports the need for a better low-budget change detector or a less brittle fallback. It does not justify threshold tuning on these reused systems or neural training yet. A next test should isolate why the few missed family changes dominate MSE and freeze a remedy before evaluating new target systems.
