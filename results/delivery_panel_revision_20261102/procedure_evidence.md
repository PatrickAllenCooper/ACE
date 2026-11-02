# Procedure evidence for the November 2 paper revision

Verified 2026-11-02T16:01:36.149614+00:00. Read-only inspection of original frozen source, accepted JSON receipts, original paid-row logs and existing physical CSV archive. Only this file and `procedure_evidence.json` were written. No model imports, checkpoint deserialization, refits, tests, replays, new responses, paper edits, commits or controller-lock operations.

1. **Prospective rolling budget is 48,000 updates per nonroot head; delivery is 30,000.**

`U_roll,head = 400 * (max(5, int(0.2 * 100)) + 100) = 400 * (20 + 100) = 48,000 = 8,000 fast + 40,000 consolidation updates.`

The first 50 paid rows calibrate ranges before updates; replay then starts empty and processes all 400 arrivals, including the prefix. Each arrival is appended to the 50-entry buffer before the archived helper adds it again, so consolidation uses `min(t,50)+1` entries (at most 51 entries / 50 unique query IDs). Root-only clamps make nonroot masking vacuous. Adam persists per head. Delivery and flat each use 30,000 full-batch updates on the same 400 rows; the ablation uses 100. These are unequal total-compute recipes: batch sizes, head counts and intermediate supervision differ.

All **640/640 original fit receipts** match the accepted fit seal and the stated head/update/row configuration. All 80 training input/receipt pairs match the original acceptance hashes. Source: B_models lines 52–173, B_oracle lines 1245–1305, B_batch lines 342–353. No model was loaded to repeat these checks. The acceptance receipt’s separate “48,000 cached responses” means 32,000 training plus 16,000 evaluation responses across the study; it is unrelated to 48,000 optimizer steps per rolling head.

2. **Node counts and model counts must be distinguished.**

- Five-node systems: roots `X1,X4`; three structured heads `X2,X3,X5`; one flat head predicting `X3`. Aggregate updates over structured heads are 144,000 rolling and 90,000 delivery, versus 30,000 flat.
- Thirty-node systems: five roots `X1..X5`; **25 structured heads `X6..X30`**, versus **one** five-input flat head predicting `X30`. Aggregate updates are 1,200,000 rolling, 750,000 delivery and 30,000 flat. All nonroots are fit, including branches outside the target ancestry.
- Archived fixed emulator: 13 nodes = 8 roots + 5 modeled mechanisms, including `comm_cost`; flat has one eight-input head. Physical task: one two-input neural head per arm.

All 40 frozen descriptors and all accepted fit receipts agree on these counts. The prospective sample has 20 systems per size, two histories per system, primary initialization zero; 240 primary cells plus ablation and initialization sensitivity cells make 640 fits.

3. **Initialization seeds are reused across systems, not independently sampled for each system.**

`seed(n,r) = r + zero_based_index(order,n)` for a structured head; `seed(flat,r) = r + len(order)`. No system seed or history seed enters this formula. The five-node order is `X1,X4,X2,X3,X5`, so init0 seeds are `X2:2, X3:3, X5:4, flat:5`. Thirty-node init0 structured seeds are `5..29`, flat seed `30`. Delivery, rolling and ablation share structured initialization at `r=0`; only delivery and flat additionally use `r=1,2` sensitivity fits.

Within a shared input history, corresponding structured heads start with identical parameter initialization. Across systems, the same seed does not imply identical tensors when parent dimensionality changes; measured-parent normalizer buffers vary too. This conclusion comes from the authenticated initializer and MLP constructor, not regenerated tensors. Stage A uses a different offset convention (`flat` index 0; structured heads indices 1..5). Physical fitting resets seed 0 for each neural arm in every condition. Avoid implying independent initialization replications across worlds.

4. **The exact archived paid-row accounting is:**

`N_paid = 3 seed + 500 observational validation + (5 mechanisms * 500 interventional validation) + (200 iterations * 4 probes) + (floor(200/5) refreshes * 25 rows) = 3 + 500 + 2,500 + 800 + 1,000 = 4,803.`

The purported 2,300 subtotal is `500+800+1,000`; it also omits the three seed rows. Including the stated seeds gives 2,303. The substantive missing term is **2,500 interventional mechanism-validation calls**, constructed once as 500 independent parent interventions for each of the five nonroot mechanisms. The term is explicit in the frozen constructor and `_build_interventional_val`, and all 12 original logs contain precisely these five role counts and 4,803 unique query IDs. Each independent call journal also has 4,803 rows; every log/meta/call digest matches the original seal.

Of the 800 probes, 200 are selected and 600 remain unselected; selecting a paid probe and duplicating its training entry add no new response. Later all-paid refits reuse eligible paid validation rows. Their per-head eligible counts are 4,303 for `leo_link_score`, `meo_link_score`, `iftu_score`; 4,803 for `engagement_rate`, `comm_cost`; and 3,803 for flat. The frozen eligibility rule excludes a directly clamped response from its own mechanism and any nonroot-clamped row from flat.

The saved final buffer has 50 rows. The saved admitted union has `3+200+39*25=1,178` unique rows: the last 25-row refresh occurs after the final fit. These counts agree with all 12 authenticated saved attribution inputs; no online training was replayed.

**Preserve failure accounting:** accepted aggregate campaign calls are `60,962 = 12*4,803 + 3,326`, not just 57,636 retained-history rows. The 3,326 residual is computed from the accepted aggregate and was not decomposed into individual discarded attempts here. History `124753321` remains retained without rescue. The separate prospective pilot failure `33505661` (`FAILED`, `1:0`, missing `experiments` module) remains in the frozen evidence; it recorded zero new confirmation or development timing responses.

5. **Physical rows per condition are directly evidenced.**

The SHA-256-verified original ZIP, original fit seal and accepted physical receipt agree: each of the 11 evaluation conditions has **1,000 rows**, with 8 held-out action blocks. `white_64` is development-only. Train/test counts:
- `blue_128`: 790 train / 210 test.
- `blue_255`: 815 train / 185 test.
- `blue_64`: 793 train / 207 test.
- `green_128`: 794 train / 206 test.
- `green_255`: 789 train / 211 test.
- `green_64`: 775 train / 225 test.
- `red_128`: 783 train / 217 test.
- `red_255`: 772 train / 228 test.
- `red_64`: 774 train / 226 test.
- `white_128`: 787 train / 213 test.
- `white_255`: 778 train / 222 test.

Totals: 11,000 rows = 8,650 training + 2,350 held out. Split: `b_j=floor((theta_j+90)/30)`; held out iff `(b_1+2*b_2)%5==0`. Physical rolling is **100 updates per arriving training row**, totaling 77,200–81,500 updates per condition; it has no additional 20-update fast phase. Delivery uses 30,000 full-batch updates per condition. Both neural arms have one 4,417-parameter head.

**Evidence and limits.** No mismatches were found in the checks listed above. The JSON contains every inspected prospective fit receipt digest, frozen descriptor digest, training-input/receipt digest, all 12 history log/meta/call/input digests, per-condition archive-member digests, and the exact equations and terms. These checks establish source/receipt agreement, not a fresh runtime measurement. The brain lookup did not return within the bounded check and was stopped; no brain result was used.

Principal sources (all SHA-256 digests are of bytes read locally):
- B_acceptance: [acceptance.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/acceptance.json); SHA-256 `dda7a9a4414524ed74e6e8955718c4be93177e452f68dcc961984a521060a3cb`.
- B_registration: [registration.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/registration.json); SHA-256 `e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5`.
- B_fit_seal: [fit_seal.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/fit_seal.json); SHA-256 `cc167516d53775c5173cf4e1fb93e6ef3bd1b442d5d3c1ca2c4bce2adcdf536c`.
- B_models: [delivery_prospective_models.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/project/scripts/research/delivery_prospective_models.py:52); SHA-256 `044c7856c5e838be9e0bcfbf3dc981a61239003ba24a0433b74fa8a5b43e997f`.
- B_batch: [delivery_prospective_batch.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/project/scripts/research/delivery_prospective_batch.py:342); SHA-256 `40ca3837bdd0364d01fe1560ce5f964de29d4c55fde9ead32b47e8ff383db551`.
- B_oracle: [oracle.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/runner/ace/oracle.py:120); SHA-256 `4d48bef21c4d45a82dba4c0616614a434eff5e94d99a1a99d00f5193718fac56`.
- H_oracle: [oracle.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source/ace/oracle.py:571); SHA-256 `4d48bef21c4d45a82dba4c0616614a434eff5e94d99a1a99d00f5193718fac56`.
- H_config: [acquisition_config.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/protocol/acquisition_config.json); SHA-256 `d6595e9596fe6b461f279e787672838b9952cdeee511f2e1df1ae6d8a1b33212`.
- H_seal: [sealed.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/sealed.json); SHA-256 `4e43833c6baac28baee67d8189aa40418d6cabef0a887fea04798e7f1e3fd737`.
- H_terminal_acceptance: [terminal_audit.json](/Users/pat/code/ACE/results/delivery_final_history_20261006/terminal_audit.json); SHA-256 `7249abea4ed8e8b4c08983ffb7695b29ca09c07c899cdd8c67236a139c6d5c05`.
- A_adapter: [delivery_attribution.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-frozen-worker-custody-20261007-07/A/scripts/research/delivery_attribution.py:86); SHA-256 `c0f52b2027e4643ecd414bc1f714e6ad30350110f139127dc02678ced03de2e3`.
- A_protocol: [protocol.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-attribution-inputs-20261006-v3/protocol.json); SHA-256 `a454e59feeaf6b0570bc725816d95f101c3f770362ad563b4056fe48860ef169`.
- C_acceptance: [acceptance.json](/Users/pat/code/ACE/results/delivery_chambers_20261006/acceptance.json); SHA-256 `915526f09a947b274332adb40cf69f84973e4579b237394df18e5914b342a352`.
- C_fit_seal: [fit_seal.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-chambers-full-20261006/fit_seal.json); SHA-256 `2233a62117576662dcf94180752256af1dbfce2c811b77607a36b70eda334471`.
- C_worker: [chambers_delivery_validation.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-frozen-worker-custody-20261007-07/C/scripts/research/chambers_delivery_validation.py:61); SHA-256 `c83a65783dc033e4576fda1b33f9784c36e2735494a33b0a56dc7a5d6a3d188f`.
- C_archive: [lt_malus_v1.zip](/Users/pat/.cache/ace/causal_chambers/lt_malus_v1.zip); SHA-256 `584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9`.

6. **Panel follow-up: original prospective primary fit CPU and parameters (init0).**

Each value below summarizes **40 fits per family/arm = 20 systems × 2 histories**, using the original receipt-level `cpu_seconds`. Only primary init0 delivery, rolling (`online`) and flat (`simpler`) are included: 240 fits in total. These are fit process CPU values, excluding evaluation and outer runtime setup/qualification; they do not include sensitivity fits or the ablation. The timer starts after initial runtime imports, validation and model/optimizer/normalizer initialization, and ends inside `fit` before later checkpoint serialization. It includes training-loop tensor construction and in-fit bookkeeping, so it is not optimizer-only time, elapsed wall time or all-in study CPU.

- 5-node delivery: **2,325.449 CPU s total (0.6460 CPU h)**; mean 58.136 s/fit, median 55.031, range 54.434–84.290.
- 5-node rolling: **3,114.606 CPU s total (0.8652 CPU h)**; mean 77.865 s/fit, median 73.016, range 71.594–114.593.
- 5-node flat: **803.647 CPU s total (0.2232 CPU h)**; mean 20.091 s/fit, median 19.034, range 18.733–28.343.
- 30-node delivery: **22,268.244 CPU s total (6.1856 CPU h)**; mean 556.706 s/fit, median 478.917, range 452.950–956.023.
- 30-node rolling: **29,073.262 CPU s total (8.0759 CPU h)**; mean 726.832 s/fit, median 629.315, range 598.629–1,124.205.
- 30-node flat: **910.795 CPU s total (0.2530 CPU h)**; mean 22.770 s/fit, median 19.609, range 18.926–34.275.

**Trainable parameters:** five-node delivery/rolling each have 13,123 across 3 heads (`X2=4,353`, `X3=4,417`, `X5=4,353`); flat has 4,417. Thirty-node delivery/rolling each have **109,465–109,913 across 25 heads**, with 4,353–4,481 per head; flat has 4,609. Counts sum every nonroot head, including branches outside the target ancestry. The frozen MLP gives `P_head(d)=64d+4,289`; normalization buffers and Adam states are excluded.

**Buffer-count confirmation:** at prospective arrival `t=1..400`, the initially empty 50-entry deque contains `min(t,50)` rows after appending the current row. The helper then forms `list(buffer)+[current]`: exactly **`min(t,50)+1` consolidation entries**, with the current query represented twice and `min(t,50)` distinct query identities. At `t=1` this is two copies of one query; from `t=50` onward it is 51 entries for 50 queries. The 20 fast updates use only `X_t[-1:]`, a singleton current row; the duplicate is in the 100 consolidation batches. This does not add a paid response or optimizer step. The formula applies to prospective Stage B, not the physical rolling implementation.

Source identities are the previously inspected original receipts under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/fits/{5,30}-*/{balanced_varied_value,matched_random}/{delivery,online,simpler}-i0/receipt.json`; all 240 selected paths, original digest identities and extracted CPU/parameter values are recorded in the JSON addendum. Existing acceptance and fit-seal identities are reused; no custody-wide recheck was performed.

- B_models (130–132, 166–168 for timing; 101–114 for buffering): [delivery_prospective_models.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/project/scripts/research/delivery_prospective_models.py); previously verified SHA-256 `044c7856c5e838be9e0bcfbf3dc981a61239003ba24a0433b74fa8a5b43e997f`.
- B_oracle (134–148 for architecture; 1263 and 1291–1305 for duplication and updates): [oracle.py](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-final-release-bundle-20261006_45ebeb89/bundle/runner/ace/oracle.py); previously verified SHA-256 `4d48bef21c4d45a82dba4c0616614a434eff5e94d99a1a99d00f5193718fac56`.
- B_acceptance (original acceptance): [acceptance.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/acceptance.json); previously verified SHA-256 `dda7a9a4414524ed74e6e8955718c4be93177e452f68dcc961984a521060a3cb`.
- B_fit_seal (original receipt seal): [fit_seal.json](/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-prospective-accepted-raw-20261008T0414Z/fit_seal.json); previously verified SHA-256 `cc167516d53775c5173cf4e1fb93e6ef3bd1b442d5d3c1ca2c4bce2adcdf536c`.

Only this addendum was added. All earlier evidence and failure accounting remain intact; no model execution occurred.
