# Finalizer sizing — retained evidence, pending completed GPU02 records

This is a sizing preparation for entry `ec812c0688d04cc5239fe12bbcbb750a3492cfffb119b16d455559b24e674f37`, not adoption, execution qualification or an accepted timing disposition. Source review887050a3 has no required source correction; R1's sizing obligation remains open. At preparation time neither completed GPU02 `inventory_transport.json` nor its after counterpart was present in the repository or external terminal02 capsule. No active origin, archive, partial stream or private outcome payload was inspected. Final counts, timing scenarios and input pins await those complete records and final custody/actual GPU acceptance.

## Exact source work to size

Let C be the full CPU output excluding `worker/artifacts/campaign`, G the full GPU output, and A the complete campaign. Let each symbol denote both its file count and byte sum, evaluated separately from the completed full inventory.

| Phase | Required work |
|---|---|
| Parent authentication | C + G + A, complete byte/membership checks; flat inputs and original CPU/GPU input rosters reread |
| Runtime preparation | One exact archive stage, handoff and worker bootstrap |
| Worker authentication | C + G + A again, with the same full source/input/runtime/custody/private-decode gates |
| Reporting startup | Runtime-before, original campaign import, runtime-ready; no reporting until work-start |
| Original reporter | Eight disjoint policy-attempt inventories; complete attempt/candidate headers; up to192 arm receipt/identity checks, then up to192 full arm-file validations and sealed-input rereads; unchanged private sealing/reporting and24 world summaries/192 cells/28 comparisons |
| Worker validation | A again, including new reports; runtime-after, report-envelope and per-reported-cell cost-report reads |
| Parent validation/postcheck | A again including new reports, C and G again, then terminal/publication |

Therefore the outer full-inventory read total is **3C + 3G + 4A**, using pre-report A only as the starting size: the last two A passes also hash future report growth. The eight attempt inventories are disjoint subtrees, approximately one campaign-sized sweep, not eight full campaign sweeps. Their exact hash set excludes each attempt's two publication headers; those16 headers are read separately. `_load_attempt` additionally reads each present arm receipt. `_load_arm` then reads every arm file once (receipt, candidate and source evidence) and each sealed input once more. These are additional to the outer ten passes. `report_cost_curve` works from captured in-memory inputs; new report writes/reads and directory traversal remain costs.

The CPU exclusion suppresses file hashing, **not traversal**: the finalizer's `rglob` still visits the campaign subtree inside the CPU output on each C pass. Reporter `_inventory` has two tree walks; `_load_arm` also enumerates/stats the tree before its membership/read pass. File-count scaling must therefore include directory/stat overhead and cannot be byte-throughput-only.

Both process authentications additionally capture their flat finalizer roster once and reread each original CPU/GPU input file once. Across the path this is `2*F + 2*(Ncpu+1) + 2*(Ngpu+1)` file captures, plus four calls to `packet_destination` (each reads the packet and copied CPU packet), and fourteen explicit bundle/terminal/campaign-root/historical-marker reads. Small handoff/startup/terminal reads and finalizer evidence writes are separate. F is the future finalizer input roster; it must not be invented before a packet exists. The actual known CPU/GPU input counts/bytes can be extracted from inventory path names without opening their payloads.

## Closed timing evidence retained without rerun

- Failed GPU01 hit its240-second startup alarm during its first historical CPU inventory before staging; that scan did **not** complete.245 allocated seconds/6.755 process CPU seconds do not establish a complete-scan duration or identify the latency cause. This slow observation stays in the disposition; it is not converted to a240-second upper bound.
- Closed login-side CPU-root segments for35,751 files had saved progress brackets66.50–89.23 seconds and74.11–93.60 seconds. The brackets include adjacent work at their coarse1000-read edges and are observations of a different reader/environment, not compute-node benchmarks. Use the slower saved observation, with the new before/after records, rather than selecting the fastest pass.
- Accepted scientific CPU archive staging took48.0095 seconds wall/25.5322 process CPU. Earlier artificial finalization staging took32.3592 seconds. Artificial readiness at48.3002 includes staging plus other preparation; its difference15.9410 is not an isolated import measurement.
- Accepted artificial reporting-through-terminal was3.309721 seconds for16 cells/76 curve points/152 point-query predictions. At the screen's prospective4800 points/38400 point-query predictions, the **report component** of point-count and point-query scenarios is209.035 and836.140 seconds respectively. The earlier257.706/884.811 figures included48.6714 seconds of whole-run non-report overhead; do not add a fresh staging allowance to those totals without removing that old overhead. Graph dimension, byte volume, validation/sealing, output fsync and the separation of CPU arithmetic from filesystem delay do not scale identically. These scenarios are not runtime upper bounds or outcome-based selection.

## Candidate phase profile and remaining disposition

Single origin: preparation1200; reporting startup <=120 from worker authentication completion and absolute cap1320; worker work/validation2820; cleanup2850; parent validation/postchecks/publication hard3590; allocation3600 seconds,2CPU/7680MiB/0GPU/no-requeue. Latest possible startup leaves1500 seconds for work, followed by30 seconds to cleanup and740 seconds for parent postchecks/publication. Earlier startup creates additional work time; a scenario must show its joint schedule explicitly rather than simultaneously spending that slack twice.

After both complete records are pinned, retain all progress samples and derive per-root/group counts and intersecting1000-read timing brackets. Use max(before,after) and retain the older slow/failure evidence. Compare at least a direct scaling scenario and slower-I/O/higher-report-demand sensitivity, with explicit staging, startup, repeated validation, postcheck and publication/cleanup margins. Any multiplier or reserve is a sizing judgment, not measured demand or a universal bound. If a plausible high-latency sensitivity crosses a deadline, show it and its retained-failure consequence rather than rewriting the evidence or claiming guaranteed completion. The independent reviewer decides whether the evidence supports one bounded attempt.

No sustained probe will run on a login node. Prefer the completed operational verification evidence and accepted runtime/report timings. If evidence leaves a material phase-fit question unresolved, a compute-node probe would need separate coordination, an actual CPU allocation and job/journal/accounting extension; none is assumed here.

Parent's provisional actual GPU02 charge is6832CPU/1708GPU; provisional closed37073CPU/4458GPU +3600 preparation +7200 finalizer reservation =47873CPU, leaving9727 below57600. These supplied figures await the independent actual baseline pin; they are not a new accounting acceptance. The existing history remains48/98 before finalization and49/100 afterward, with no hidden qualification row.

`full_path_timing_supported`, `resource_admission_accepted`, `execution_qualified` and `scientific_qualified` remain false. Final sizing requires completed before/after and custody pins, accepted GPU02 core/supplement and baseline, plus independent timing review before operational adoption. No private result or model was decoded, no new science ran, and no allocation/lock/remote action occurred.
