# Numerical evidence verification — 2026-11-02

Read-only arithmetic on accepted CSV/JSON; init0 remains primary. Numerical claims are confirmed, with one evidence limit: the exact per-head RNG rule is reported by the AE but is not recorded in the permitted score/receipt fields. No new tests or intervals were added.

## History-specific init0 ratios and wins

Each ratio is exp(mean of 20 paired delivery-minus-control log NMSEs); each win is strict lower delivery NMSE within that history. Source arm online means rolling; simpler means flat. All eight contrasts have zero ties.

- 5-node balanced: delivery/rolling **1.61407371247**, **4/20** wins; delivery/flat **0.792754840919**, **13/20** wins.
- 5-node matched random: delivery/rolling **3.74792976004**, **3/20** wins; delivery/flat **0.160042133221**, **19/20** wins.
- 30-node balanced: delivery/rolling **0.695011186954**, **12/20** wins; delivery/flat **0.912253569631**, **10/20** wins.
- 30-node matched random: delivery/rolling **0.108274745009**, **19/20** wins; delivery/flat **0.167609175651**, **18/20** wins.

30-node balanced delivery/flat is exactly **0.912253569631413** in the accepted history CSV, rounding to **0.912254**, with **10/20 wins**. It does not support a uniform or majority-win advantage in that history.

Primary pooling averages the two history log ratios within each system, then across 20 systems and exponentiates; pooled wins use the within-system mean log ratio below zero:

- 5-node delivery/rolling: 2.45955990004; 2/20 system wins.
- 5-node delivery/flat: 0.356194014354; 20/20 system wins.
- 30-node delivery/rolling: 0.274321269766; 17/20 system wins.
- 30-node delivery/flat: 0.39102694126; 17/20 system wins.

## Absolute scale and exact aggregation

For each family/arm, use all 40 **raw per-history init0 NMSEs** (20 systems × 2 histories), equally weighted. Sort these values: median NMSE = (N_(20)+N_(21))/2; median RMSE/training-targetSD = (sqrt(N_(20))+sqrt(N_(21)))/2. No within-system geometric averaging precedes these medians. Exact middle-cell values/selectors are retained in JSON.

Each cell satisfies NMSE = stored MSE / stored training-target variance and RMSE/training-targetSD = sqrt(NMSE). All 640 denominators are positive and no 1e-12 log floor activates. The stored variance convention was not reconstructed.

- 5-node delivery: median NMSE **0.000474944533445**; median RMSE/training-targetSD **0.0217813506302** = **2.1781351%**.
- 5-node rolling: median NMSE **0.000269805118466**; median RMSE/training-targetSD **0.0164255768208** = **1.6425577%**.
- 5-node flat: median NMSE **0.00114119104506**; median RMSE/training-targetSD **0.033781508213** = **3.3781508%**.
- 5-node short-fit: median NMSE **0.0519711534377**; median RMSE/training-targetSD **0.227971071509** = **22.797107%**.
- 30-node delivery: median NMSE **2.82260427965e-05**; median RMSE/training-targetSD **0.00531270046777** = **0.53127005%**.
- 30-node rolling: median NMSE **7.27232906825e-05**; median RMSE/training-targetSD **0.00850994451102** = **0.85099445%**.
- 30-node flat: median NMSE **7.55767252369e-05**; median RMSE/training-targetSD **0.00869315940375** = **0.86931594%**.
- 30-node short-fit: median NMSE **0.00145937847585**; median RMSE/training-targetSD **0.0381974183832** = **3.8197418%**.

The six delivery/rolling/flat medians span **0.531270047%–3.37815082%**, supporting approximately 0.5–3.4% of training-target SD. This is a range of six medians, not all errors, and excludes short-fit and later initializations. It is not percent of target mean/range or a claim of practical adequacy.

Because 40 is even, median(sqrt(NMSE)) differs slightly from sqrt(median NMSE). The latter, in delivery/rolling/flat order, is: 5-node 0.0217932221905, 0.0164257455985, 0.0337815192829; 30-node 0.00531281872423, 0.00852779518296, 0.00869348751865. Both support the approximate scale, but an exact median-RMSE statement should use the median of per-cell square roots.

## Initialization semantics and extreme cell

Registration and all 640 score rows verify delivery/flat init0,1,2 for every system/history, with one rolling and one short-fit init0 fit. Registration specifies “fixed0; init1/2 delivery/flat sensitivity separately, never deployment selection.” Descriptor/world seeds are 40 distinct values, separate from the repeated initialization labels.

The AE reports nonroot head seed = r + topological position. If accurate, this reuses seed values by position across systems/histories rather than independently drawing a seed per system. **The exact implementation is not independently verified here:** all 640 fit-receipt digests match the accepted seal, but receipts contain init labels and no resolved per-head RNG seeds or initial-weight fingerprints. Same seed labels do not prove identical tensors for differently shaped heads. Source code/checkpoints were outside this check.

For **5:08, matched_random**, delivery init1 NMSE is **0.023838762668516332**, flat init1 **0.048152761626851733**, and rolling's single init0 **2.2843895892873858e-05**. Delivery/rolling is **1043.55066142** (approximately 1044); delivery/flat is **0.495065326746**. Original score selectors are cells.140.metric, cells.141.metric, and cells.137.metric respectively. Normalized RMSEs are 0.1543980656, 0.2194373752, and 0.004779528836. This is an NMSE ratio in init1 sensitivity, not the primary init0 result or a matched rolling-init1 comparison.

## Descriptive initialization sensitivity

Seed-specific comparisons use delivery r versus flat r or rolling0; then the same equal-history/equal-system log aggregation:

- 5-node init1: delivery/rolling 2.27146941298 (3/20 system wins); delivery/flat 0.162731367811 (20/20 system wins).
- 5-node init2: delivery/rolling 1.4552511612 (6/20 system wins); delivery/flat 0.188104553569 (20/20 system wins).
- 30-node init1: delivery/rolling 0.281448282184 (19/20 system wins); delivery/flat 0.359875757764 (17/20 system wins).
- 30-node init2: delivery/rolling 0.219994773507 (18/20 system wins); delivery/flat 0.263550182662 (19/20 system wins).

Seed-averaged sensitivity first averages **log NMSE** across all three registered delivery/flat fits within each system/history, retains rolling single-fit log NMSE, subtracts arm means, averages equally across two histories and 20 systems, then exponentiates. This is a geometric score average, not a prediction ensemble or best-seed selection:

- 5-node delivery/rolling: **2.01079324284**; 5/20 system wins.
- 5-node delivery/flat: **0.221744266969**; 20/20 system wins.
- 30-node delivery/rolling: **0.257053476979**; 18/20 system wins.
- 30-node delivery/flat: **0.333483382772**; 19/20 system wins.

All four directions agree with init0: 5-node delivery is worse than rolling and better than flat; 30-node delivery is better than both. Init0 remains primary. No seed-averaged interval or hypothesis test was computed.

- Within-history delivery worst/best NMSE across init0/1/2, 5-node: median 2.01825645263; linearly interpolated 90th percentile 11.3818972706; 40 system/history cells.
- Within-history delivery worst/best NMSE across init0/1/2, 30-node: median 5.23089568216; linearly interpolated 90th percentile 21.6524671554; 40 system/history cells.

Forty history cells and three initialization labels must not be treated as additional independent systems. Scores do not diagnose the cause of initialization failures.

## Provenance and scope

The accepted report02 has accepted=true and no pending scientific review. Both companion index digests match its pins; indexed CSV/JSON hashes match loaded files. All 640 metric triples exactly match original accepted scores; init0 CSV is the 320-row init0 subset. All 8 history ratios, 4 primary ratios and 80 primary system log ratios reproduce to floating-point precision. These are consistency checks, not added hypothesis tests.

The descriptive index records scores_sha256=e5af266398968b64c5c22adf852eaec8ab81227a65e852a40679c5f9ab95890e. The original accepted score file checked here has digest 3ce9ee28f4c85b9b909c5e65c90b8d1d7eb39747d846ab5964cec1c36c0aded6. These are distinct file identities: this check establishes exact agreement of all 640 metric triples between the publication CSV and original accepted scores, not byte identity of the two score files.

Exact source paths and SHA256 digests:

- [all_cells](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_all_cells.csv)
  SHA256: ba93de3a81f2b4a5ac52cecb9f4d27ddde51540276ea132a3a67aee0fe44bd45
- [fixed_init0](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_fixed_init0.csv)
  SHA256: 7a7e267c32b0cfe49463032d8e9e107a6adab664c2f55a125429a08f1d85ec3c
- [history_specific](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_history_specific.csv)
  SHA256: eb54e606666ac231de8d58479030c917d7bfdfeac656aface3996e8cf82d2fe4
- [system_log_ratios](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_system_log_ratios.csv)
  SHA256: 9ced055f4bafde33ad82f8dbeeceaf6eca93157e3fa1225dab1ad42ee9f63353
- [supplement](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_supplement.json)
  SHA256: 58b8530c8b10f96009266aa838d2ce49b306ce29232efe5aa3e9d29405a1bdaa
- [output_index](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_output_index.json)
  SHA256: 022e78c2a7e862066265f16eb490d2b0df316d11884b3aa7f8dfac22b2ae9ada
- [claim_index](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_claim_index.json)
  SHA256: fdd5a3ee01708efbaaee7202d03c907cb6aede06dc1752ded7df5278d87f8deb
- [bundle](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_prospective_bundle.json)
  SHA256: 2cca8a3699bcee68adef1a7fe610df87a6518a39513f71b310e990d73183cd84
- [registration](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/registration.json)
  SHA256: e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5
- [original_scores](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/scores.json)
  SHA256: 3ce9ee28f4c85b9b909c5e65c90b8d1d7eb39747d846ab5964cec1c36c0aded6
- [original_acceptance](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/acceptance.json)
  SHA256: dda7a9a4414524ed74e6e8955718c4be93177e452f68dcc961984a521060a3cb
- [fit_seal](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/fit_seal.json)
  SHA256: cc167516d53775c5173cf4e1fb93e6ef3bd1b442d5d3c1ca2c4bce2adcdf536c
- [descriptor_manifest](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/descriptor_manifest.json)
  SHA256: d8efce939a5afda7594e842550a24b3d862d4c2db321457226116efe79cdeb49
- [report_acceptance](/Users/pat/code/ACE/results/delivery_prospective_reporting_20261024/terminal02/report_acceptance.json)
  SHA256: 86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10
- [ae_note](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-TMLR-model-panel-review-20261102-01/AE_discussion_note.md)
  SHA256: c71a066ea39a37e02e8eeb2086e5d76b3728f4aec92c7f4a24db53438facdc5d

The JSON additionally contains all 640 fit-receipt paths relative to the accepted raw root and their exact digests, per-system contrast values, median middle-cell locators, formulas, and caveats. The AE note is context only; numerical results derive from accepted CSV/JSON.

Only numerical_evidence.json and numerical_evidence.md were written. No model imports, reporters, refits, responses, new tests/intervals, frozen index/table edits, controller-lock changes, or git commit. Narrative/panel synthesis and budget accounting remain with their assigned agents.
