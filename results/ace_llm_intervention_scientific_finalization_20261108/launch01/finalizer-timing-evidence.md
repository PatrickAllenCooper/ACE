# Finalizer sizing03: recommendation for one bounded CPU attempt

**Recommend the reviewed 2CPU/7680MiB/4200-second profile as an evidence-backed bounded attempt, subject to independent timing review and actual GPU02 acceptance/lineage.** No additional probe is recommended. This is an author sizing recommendation, not operational adoption, a completion guarantee or full-runtime qualification. `full_path_timing_supported` and all admission/qualification flags remain false pending the required independent disposition and actual GPU acceptance.

The entry is `8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f`, captured from commit `255b75fe37b28c302fb57821c0cc09a60a6cd3ea`; source review39ef3af6 closed the numeric delta with0 new required findings. The single-origin limits are preparation1500; reporting startup120 after worker authentication and absolute cap1620; work3320; cleanup3350; hard4190; allocation4200. Scientific reporting, full checks and failure retention remain unchanged.

## Completed evidence and full work counts

BEFORE2ca9bf1f and AFTER53b4fc60 have identical full file maps:99,039 files/2,507,157,901 bytes and3782 unique directories. The maps also match the pinned disposition and remote archive receipt; both sealed local-copy maps bind these exact metadata bytes. Receipt e59bca37, disposition df246a23 and seal66ddd22d are retained as exact input pins in the machine disposition. This is a local metadata association for sizing, not a repeated remote backup audit or GPU execution acceptance.

| Actual finalizer read group | Files / bytes | BEFORE seconds | AFTER seconds | Scenario charge |
|---|---:|---:|---:|---:|
| CPU output excluding campaign | 20,564 / 301,378,319 | 90.270 | 100.084 | **100.084** |
| GPU output | 13,262 / 159,768,470 | 304.743 | 58.029 | **304.743** |
| Campaign | 27,869 / 90,766,988 | 141.934 | 77.531 | **141.934** |

These are complete intersecting1000-read progress-bin totals, retaining the slower observation **for each group**. They include adjacent reads and traversal; group bins can overlap. They are deliberately conservative proxies, not isolated finalizer benchmark durations. Both full stderr histories remain saved, including the slower initial GPU-tree pass. The original observer did not record process CPU. The full transport UTC brackets693.446/490.538 seconds include queries, serialization and transport; they are not substituted for per-stage work.

Exact source multiplicity remains CPU×3, GPU×3, campaign×4:212,954 payload hashes/1,746,508,319 bytes before report growth. Two of the campaign passes include new reports. Eight **disjoint** attempt inventories and up to192 arm validations add a complete-valid-arm scenario of56,240 existing-payload reads, including sealed-input rereads and publication headers. Both parent/worker also capture the flat finalizer roster and reread original CPU/GPU inputs; the original inputs comprise97/110 files and20.92/38.85MB. CPU exclusion skips campaign hashing but still traverses its paths. Source formulas and the unchanged read sets are retained in sizing01/02; no membership or byte check is removed.

## Selected scenario, explicit allowances and joint schedule

The slower-group payload proxy is546.761 seconds per authentication. Twice that plus both original-input reread proxies is1142.648 seconds. The accepted scientific stage took48.010 seconds; its **scenario allowance is150**, not a claimed measured maximum. Preparation additionally reserves100 seconds for the future finalizer flat capture/metadata parsing and60 for directory traversal, scheduling and handoff variation.

Reporting uses the higher **836.140-second point-query sensitivity**, derived from the accepted3.309721-second artificial report interval at4800 points/38400 point-query predictions. The lower point-count case is209.035 seconds. Old257.706/884.811 totals included48.671 seconds of non-report overhead, removed before adding current allowances. Scaling all old report work by query count and then adding566.590 seconds of current reporter/worker-validation I/O proxies can double-count validation; that conservatism is retained. Neither extrapolation certifies3-node arithmetic or filesystem latency.

| Phase | Selected proxy plus explicit allowances | Phase window | Residual after allowances |
|---|---:|---:|---:|
| Preparation | 1142.648 +150 staging +100 capture/metadata +60 traversal/handoff = **1452.648s** | 1500s | **47.352s** |
| Reporting startup | **90s** | 120s relative, absolute1620 | **30s relative** |
| Worker reporting/validation | 836.140 +566.590 I/O +100 report growth/fsync +50 runtime/envelope +100 sealing/dimension variation = **1652.730s** | 1700s at latest startup | **47.270s** |
| Owned cleanup | **30s** | 30s after work limit | Full window assigned |
| Parent validation/postchecks/publication | 546.761 +100 report growth +50 report metadata +100 traversal/publication variation = **796.761s** | 840s after cleanup limit | **43.239s** |

The allowances are explicit operational judgments. Startup90 is substantially above the earlier artificial15.941-second non-staging readiness interval, though that interval is not an isolated import measurement. All-reported output growth has a source-count scenario of1973 files; its actual byte growth is unknown.100-second growth allowances are about ten times the simple existing campaign per-file timing projection (~10 seconds), but writes/fsync, byte size and new graph arithmetic need not follow that rate. The additional runtime/envelope and variation allowances keep those uncertainties visible. Natural cleanup is required; failure to close is retained as failure, never accepted as success.

The selected joint schedule is authentication1452.648, report start1542.648, worker end3195.378, cleanup3225.378, parent end**4022.140**:167.860 seconds before hard4190. Even with reporting starting at the absolute1620 boundary, the assigned worker/cleanup/postcheck durations end at**4099.491**, leaving90.509 seconds. No phase uses the same slack twice and no clock resets.

## Sensitivity, failure limits and admission boundary

The lower point-count case ends at3395.035 seconds. A harsher case adds25% to every already slower-group I/O proxy while retaining all allowances and the higher report component: preparation1738.310, worker1794.377 and parent933.452 seconds exceed their respective1500/1700/840 windows; its hypothetical full path is4586.139 seconds. The actual implementation would stop at the first exceeded phase, retain partial evidence/cleanup outcomes and enforce hard4190. The earlier compute-side240-second alarm during an **unfinished** CPU inventory is also preserved; it is neither a complete duration nor an upper bound, and its latency cause is unknown. These observations support a bounded attempt with an explicit failure limit, not assurance against all filesystem conditions. No automatic retry, truncation, dropped check or extra allocation follows from this recommendation.

Collector-reported accounting is48 jobs/98 journals,37073CPU/4458GPU; +3600 preparation +8400 finalizer reservation =**49073CPU**, leaving**8527** under57600. Finalization adds no GPU reservation. This arithmetic awaits the independently accepted actual baseline; no qualification row is hidden in the prospective49/100 history.

Before adoption: independently review this sizing disposition; bind actual GPU core acceptance, recovery supplement and baseline to the same sealed evidence; then supply explicit clock adoption and obtain concrete packet review. Keep the existing absolute failure/cleanup limits and all scientific/reporting/private-decode checks. No login probe or new CPU qualification allocation is required by this recommendation. No private payload/model, scientific function, remote operation, lock, allocation or test was executed during sizing.

## Retained derivation

`attempt01` compares the two completed observations and retains every progress sample:2.369458 child CPU seconds/2.428460 wall/678,772,736-byte peak RSS. `attempt02` joins the supplied metadata pins/seal and computes explicit scenarios:0.909846 child CPU/0.924139 wall/871,235,584-byte peak RSS. Both returned0; no failed derivation occurred. Source, arguments, stdout/stderr, exact inputs/pins, machine disposition and predecessors remain preserved. Other local metadata inspection/bookkeeping is not individually metered. This is saved-evidence analysis, not a runtime probe.
