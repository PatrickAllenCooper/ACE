# Saved artificial timings: prospective resource sensitivities

This is a new read-only derivation, not a revised acceptance, scientific result, capacity certificate or allocation authorization. No project code was imported, no tests repeated, and no model/tokenizer/ranker/world/response/remote work ran. Only saved JSON timing/count fields and declared draft ceilings enter the calculations. All decisions, sessions, steps and generations in the named accepted runs are included; no outcomes selected a subset.

Final script: `attempt03/derive.py`, SHA256 `722a1f59b4574f4c7fc90610f1037db4b1552985d7145346ca0c6cdd9303a656`.
Receipt: `attempt03/derived/receipt.json`, SHA256 `b131faa64dcbf0e1e68838f831399e3d5e5b30fc0e1be29f40a73b4591ba5f72`.
Input manifest: `attempt03/derived/input-pins.json`, SHA256 `64af117ab6774f8f51646983ee1356c095d48dfb9a79cc886239c2bc3cec2bdf`.
757 exact inputs (1,584,223 bytes) were copied without modification under `attempt03/derived/inputs/` and rechecked after extraction. Attempt02 checked live originals; final attempt03 used its exact immutable input snapshot. The integrated draft captured here is SHA256 `f10373f9dda3cd6bd0f68afb9964119e29877289619be457d4a4e3cde1bb1ad3`; future edits do not change this receipt's scope. Separate accepted artifacts are bound for CPU03, persistent GPU campaign, finalization01, trainingCPU, 120-fixture token CPU and full-shape trainingGPU01.

All three new derivation attempts succeeded and remain original. Attempt01 was refined to add controller-per-call and point-query sensitivities, exact saved RSS, and the observed 2/3-node range. Attempt03 corrected stale narrative RSS rounding without changing numeric calculations. Final run: 0.173548s wall, 0.171303s child CPU, 25,329,664-byte peak RSS, exit0. Exact argv/stdout/stderr/resources are in `attempt03/run.json` and adjacent files. These are local JSON arithmetic costs, not workload costs. No failed derivation attempt occurred.

## CPU campaign

CPU03 outer112.976s included runtime staging37.381s. The five policy phases total45.724s; six ranker updates took1.276s. Remaining non-policy/non-ranker outer work65.976s is charged once in the scenarios.

| Method | Saved calls | Process mean / max seconds | Inner mean seconds | Policy phase seconds |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 8 | 0.576 / 0.710 | 0.377 | 5.589 |
| Menu coverage | 7 | 0.570 / 0.780 | 0.396 | 5.463 |
| Local information | 8 | 0.614 / 0.777 | 0.432 | 6.115 |
| Target risk | 7 | 0.586 / 0.650 | 0.403 | 6.183 |
| Learned ranker | 6 | 3.472 / 3.730 | 2.770 | 22.373 |

Every process receipt is joined to an inner child receipt by method and input hash. Process timing already includes per-call interpreter startup. The saved shapes have2 or3 nodes,3 acquisitions,2 queries and2–5 history rows. Prospective shapes have3 nodes,8 acquisitions,8 queries and up to32 rows. In particular, larger planning/state work is not empirically bounded by these observations.

For each method, compute576×observed per-call coefficient plus controller residual. Scaling residual by24/2 worlds yields3565.500s at means and4043.853s at observed maxima. Scaling residual by576/observed calls instead yields4076.076s and4554.430s. Both include65.976s once and61.254s for a constant-cost288/6 ranker-update sensitivity. This gives a component-based rationale for considering the6000s bounded attempt, but does not establish5940s work completion. Full-plan ranker initialization, larger state/query costs, and3584-pair validation are not timed at scale. Per-decision50/55s limits remain failure limits, not latency forecasts.

## Three GPU policy services

Saved total129.797s =70.326s in three sessions +59.471s outside them (including33.970s runtime staging). Session durations25.205/21.380/23.740s included generation wall sums3.268/4.795/6.058s. The rest must not be treated as all fixed startup: between-generation non-generation gaps averaged0.247/0.124/0.229s, with observed maxima0.665/0.319/0.667s. Subtracting these gaps separately leaves start-and-tail residuals20.208/15.719/16.077s. Gaps include controller/state/response-journal/transport work and world transitions.

For each service: start-and-tail residual +576×generation duration +575×gap duration. Add59.471s outside services once. Saved short-shape means give1473.232s overall; simultaneous observed maxima give2929.345s. Corresponding service maxima are795.588/896.527/1177.759s. The latter DPO sensitivity exceeds the1000s service cap; overall exceeds2700s. These simultaneous maxima are a sensitivity, not expected time or a proven worst case.

Substituting full-shape training-generation CUDA plus saved service host deltas gives1359.428s mean and2018.714s max sensitivities. Those CUDA observations are actually faster than some short-shape generations. Do not choose that more favorable case as if longer prompts dominate all runtime: completions, work placement and timing scopes differ. Keep both cases. A late600s reserve is not supported as a universal576-request capacity. The new24-world saved-service validation within the40s final margin has no separately matched timing here. No GPU model/count increase is implied; accepted peakreserved3.276GB and one20GB slice remain the memory evidence.

## Private finalizer

Saved outer51.981s included32.359s staging and reached its work marker at48.300s. Work marker to terminal was3.309721s UTC (including after-report validation/publication); marker to child-cleanup completion was3.360566s monotonic. It processed all16 cells,76 retained curve points and152 point-query predictions. Larger inputs will also increase custody/validation costs that these scenarios hold fixed.

Holding measured non-report overhead once: cell-linear192/16 scaling gives88.388s; point-linear4800/76 scaling gives257.706s, already near work260. The evaluator source loops over queries at every point. A point-query count sensitivity38400/152 gives884.811s if all report interval scales that way; that assumption overcharges fixed/other work but exposes the missing query-cost decomposition. None is a guarantee. The300s cap lacks matched-demand justification from this report timing alone; 8-query predictions, initialization8 vs2 and larger inventories are not measured in this accepted finalizer.

## Training/preparation and remaining gaps

TrainingCPU's65.282s outer included40.783s staging and20.174s child/cleanup; four-record tokenizer-load-plus-tokenization was0.648s. Full-size tokenCPU120 fixtures/888 command sequences took6.741s from tokenizer-load start to tokenization complete, after13.257s CPU imports. Scaling that combined component by160/120 (also1184/888 command-count ratio) gives8.988s. This overcounts fixed tokenizer load but omits the scientific3584-pair/custody/plan work. Exact saved peak scheduler RSS7677.891MiB leaves2.109MiB of7680; this is no meaningful demonstrated headroom. The earlier draft's rounded7677.844 differs slightly from the exact retained receipt7862160KiB used here.

Full-shape GPU01: four SFT wall times mean0.935/max1.220s, two DPO mean1.752/max1.839s; all ten generation CUDA mean0.520/max0.552s. One measured scoring forward is0.401216s CUDA. With160SFT,128DPO,768reference +2368legal forwards,322generations, the mixed wall/CUDA operation proxy is1799.516s at means and1866.462s at maxima. The previous whole-stage formula `(1866.462+2*118)*1.25=2628.078s` is reproduced; its first-generation coefficient happens also to be the maximum of all ten. This is whole-stage sensitivity, not2620s worker proof. Generation `elapsed_seconds` is cumulative journal time and was deliberately not misused as per-call wall time.

No equivalent full scientific realization/collection primitive timings appear in these accepted inputs.56 descriptors,9216 response attempts,10240 response-cost units and3584 pairs therefore do not acquire a measured300/900s guarantee from unrelated policy-call timing. The tokenizer stage's600s allowance likewise remains conditional on full-byte/plan/memory admission. Shared GPU use still requires actual startup, process/memory and work-progress evidence; no new allocation is requested here.

Declared total projection remains52440CPU/7001GPU-slice seconds, including11040/1601 closed and3600CPU preparation allowance. Remaining5160CPU/199GPU are budget arithmetic, not a remedy for any stage-specific deadline uncertainty. Preserve resource failures without altering data, exposure, worlds or endpoints.

## GPU outer review delta

R1/R2 closed on source `1a4ce8c4a6e91b819f3477ef857da37c2e4590cc0d9f6ef621cdd951038c0556`; tests `1b70742769218d7e1d84acd04d83c77aacd2a1ba8e8177a9d20ac8bd9ca4de6c`. Review SHA256 `a807eb6074630e71b98c9aa096433cc4f7d788219a4cd3378e2130043173ca3b` at `docs/development/guidance/reviews/ace_llm_intervention_scientific_campaign_gpu_outer_review_2026-11-08.md` retains initial and correction dispositions. Original3 new author-test receipts were inspected, never rerun. Delta custody: `/Users/pat/.codex/tmp/ace-scientific-campaign-gpu-outer-review-7uexqwrp/delta01`.
