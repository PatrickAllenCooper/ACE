# Final01 actual finalization review — 2026-11-08

**Required findings: 0. Artificial integration accepted; scientific qualification: false.** This review covers the new finalization execution and saved reporting arithmetic. Closed CPU/GPU/source/packet reviews are reused. No models, tensors, fitting, original finalizer, test suite, remote operation or new response was run by this reviewer. The existing pinned, standard-library-only report consistency function was reused; curve errors and pair differences were independently recomputed.

The single saved checker attempt passed. Its complete source, stdout, stderr, execution record and detailed arithmetic are retained in `results/ace_llm_intervention_campaign_finalization_20261108/acceptance01/attempt-01/`. No failed checker attempt occurred. Acceptance and the separate `review_receipt.json` sidecar are mirrored to `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-acceptance-20261108-01`.

Custody matches **2,219 raw files / 110,220,879 bytes**, both local copies and both saved remote inventories/readbacks. All **94** transfer pins match committed packet bytes, all **51** captured sources match source commit `12c7322e5046cd3e9b38cb59a1c01ed87054b71b`, and all **1,740** original common-root files remain unchanged. The accepted 729-file campaign prefix, original CPU/GPU evidence, eight attempt pins, newly sealed input provenance and exact final worker output inventory reconcile. Original `campaign/finalization.json` remains publication authority; its candidate has identical bytes. New reporting adds zero environment responses and zero training updates.

Job **35389391** and its steps completed **0:0**, with **2 CPUs, 7680 MiB, zero GPUs, 61 seconds**, `ucb736_asc2/acpu/cpu-normal`, five-minute limit, no requeue/restart. This adds **122 allocated CPU-seconds / 0 GPU-seconds**. Closed charges including failed predecessors are **10,368 CPU / 1,483 GPU-slice seconds**; with the 3,600 preparation CPU allowance, admission is **13,968 / 1,483**, within **57,600 / 7,200**. The reconciled extension is **36 jobs / 74 submission journals**. These allocation charges are separate from logical episode cost units and checker CPU time.

The inherited absolute 120/260/280/290-second deadlines, authenticated stage/handoff, isolated worker and owned process group reconcile. Staging ran from **08:51:56.805824 to 08:52:29.165020 UTC**; private work started **08:52:31.898105**, publication occurred **08:52:34.732418**, and outer completion **08:52:35.579056**. Observed Python 3.10.19/NumPy 2.2.5 paths remain in the staged runtime; captured project modules match input pins, mixed-path reports are empty, Torch/Transformers/SciPy are unloaded, CUDA is hidden and numerical thread limits are one. Cleanup records the owned group gone, return code zero and no errors. This verifies observed runtime evidence, not universal dependency closure.

**Resource observation:** Slurm batch MaxRSS is **7,862,160 KiB**, only **2,160 KiB below** the 7,864,320 KiB allocation. The successful bounded run does not establish comfortable memory headroom. The 163,819,520-byte worker/child peak is narrower evidence and must not replace batch MaxRSS. No repeat or scale-up is authorized by this review.

All **16 cells**, **76 saved curve points**, **28 policy pairs**, both strata and all eight methods are retained. True intervention means independently derived from the saved linear mechanisms are **[-1, 3]** for world 9101 and **[3.25, -0.75]** for world 9102. Equal-query normalized MSE uses scale 2 and threshold 0.04. Saved predictions reproduce every error, first crossing, capped endpoint and final attainment. All 28 differences, means, wins/losses/ties and the complete summary match; **0 failed/unresolved** cells. All accepted receipts join back to original saved return values and charges.

| Method | World 9101 first crossing (acquisition units) | Final NMSE 9101 | Final NMSE 9102 | Final attainment 9102 |
| --- | ---: | ---: | ---: | --- |
| uniform_random | 1 | 0.007367 | 0.759771 | no |
| menu_coverage | 4 | 0.021752 | 0.159304 | no |
| local_information | 4 | 0.014019 | 0.939080 | no |
| target_risk | 2 | 0.009704 | 0.011495 | yes |
| learned_ranker | 1 | 0.020073 | 0.005001 | yes |
| frozen_llm | 2 | 0.001291 | 0.482420 | no |
| sft_llm | 2 | 0.020498 | 0.009806 | yes |
| dpo_llm | never; capped at 4 | 0.101821 | 0.160977 | no |

**Adverse result retained:** every world-9102 first crossing is **0**, because shared initialization already meets the threshold. Five methods subsequently lose attainment; only target_risk, learned_ranker and sft_llm finish below threshold. World-9101 DPO never attains; the other seven finish below threshold. Overall 15 cells ever attain, but only 10 finish attained. No fixture tuning or rescue was performed.

Every cell actually spent **2 initial + 4 acquisition = 6 cost units**: **32 + 64 = 96** logical per-arm units across **92 accepted receipt entries** (32 initial, 60 acquisition). Shared initialization is logically charged per arm; these totals do not infer unique physical shared responses. First crossing is a retrospective endpoint, not an executed stopping rule or actual saved responses. The eight world-9102 zero-crossing endpoints contribute ties to every paired comparison; capped nonattainment is retained, not filtered out.

Each LLM's eight adaptive actions aggregate to **7 observe_all + 1 x_plus**. Equal action counts across different world/feedback namespaces do not establish identical noisy trajectories, equal predictions or benefit. The differing terminal errors demonstrate why these counts cannot support a scientific trained-policy comparison. This is artificial integration evidence only; no further GPU work or responses are requested.

Evidence SHA-256 pins:

- Packet: `417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c` (packet commit `380d1b3d939a64ffd0602014746df7304035cc6d`).
- Finalizer source: `88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f`.
- Original publication: `aa77436fbda2e3348b7eae08d33327c1fa5e33ae3e6602a4abd69930a53c6dfa`.
- Summary: `790cb2ecbf6af7e5e63ad10e7e8056f7b69e75f68c185ad345004ce430c7d577`.
- Independent checker: `22e005f23c436ae1e3f3b23d814bf2b88045c0643de795e5d9eb638038d9e510`.
