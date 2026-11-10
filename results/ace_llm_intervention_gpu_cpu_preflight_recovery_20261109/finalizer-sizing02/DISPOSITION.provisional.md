# Finalizer sizing02 — completed BEFORE evidence only

**Provisional; no adoption or timing-support acceptance.** This extends preserved sizing01 for finalizer entry `ec812c0688d04cc5239fe12bbcbb750a3492cfffb119b16d455559b24e674f37`. GPU02 after observation, sealed custody, actual independent GPU acceptance/supplement and baseline remain pending. Source review887050a3 is closed; R1 sizing remains open.

The completed BEFORE transport is58,101,524 bytes, SHA `2ca9bf1fd7f73a6964d9a10e3e4b239329c4bd4b53b2723a0131ded9a0395d66`. Its recorded observation spans03:17:42.441784–03:29:15.887653 UTC on2026-11-10 (693.446 seconds including transport/query/serialization overhead). It inventories99,039 files/2,507,157,901 bytes. All171 progress samples and stderr are retained. The last171,000-read counter includes subsequent journal/upstream rereads; it is not a unique-file count. Original observer process CPU was not recorded.

| Finalizer read group | Files | Bytes | Saved contained / intersecting sample seconds |
|---|---:|---:|---:|
| CPU output, excluding campaign | 20,564 | 301,378,319 | 54.096 / 90.270 |
| GPU output | 13,262 | 159,768,470 | 262.401 / 304.743 |
| Campaign | 27,869 | 90,766,988 | 111.572 / 141.934 |
| Original CPU inputs | 97 | 20,919,713 | 0 / 5.812 |
| Original GPU inputs | 110 | 38,849,832 | 0 / 17.133 |

The original observer records every1000 reads. Intersecting-bin totals deliberately include adjacent files and directory work; separate groups' bins can overlap. They are coarse observed charges used for scenarios, not isolated finalizer timings or future bounds. Directory order is reconstructed using Path component ordering from the captured observer's root traversal. No payload is opened by this derivation.

The source's CPU×3/GPU×3/campaign×4 inventories imply212,954 payload hashes/1,746,508,319 bytes **before adding future report growth**. Reporting additionally scans eight disjoint attempts (27,850 files), up to192 arm receipts,20,954 arm files and7,228 sealed-input rereads. The complete-valid-arm scenario has56,240 extra existing-payload reads, including16 attempt headers. Original metadata/source recaptures, service checks and new reports add work. The preserved source formulas and private-decode boundary are detailed in sizing01; nothing is skipped or cached away.

## Conservative before-only phase worksheet

The profile remains one origin: preparation1200; report startup<=120 after worker authentication and absolute cap1320; work2820; cleanup2850; hard3590; allocation3600/2CPU/7680MiB/0GPU. At latest startup, worker time is1500 seconds; post-cleanup parent time is740 seconds.

| Partial scenario | Seconds | Time still unassigned within phase |
|---|---:|---:|
| Two complete authentication payload passes + accepted48.010-second stage | 1121.905 | 78.095 preparation |
| Above + two original CPU/GPU input reread proxies | 1167.794 | **32.206 preparation** |
| Report point-query sensitivity836.140 + extra reporter/worker-validation I/O proxies566.590 | 1402.730 | 97.270 at latest startup |
| Parent C+G+A existing-payload postchecks | 536.948 | 203.052 after cleanup |

These unassigned quantities are **not established margins**. Preparation still needs future finalizer flat-input capture and metadata checks/parsing; staging can be slower than its48-second accepted example. Work/postchecks still need runtime-after, directory/stat costs not isolated by this observer, newly written report bytes/fsync, report-envelope rereads and publication. The I/O proxies can overcharge adjacent work and the report sensitivity already scales some old validation costs; the worksheet exposes that conservatism rather than treating every term as a separately measured duration.

The lower point-count report component is209.035 seconds, versus836.140 for the point-query sensitivity. Both derive from the same retained3.309721-second artificial report interval. Their former257.706/884.811 whole-run figures included48.671 seconds of old non-report overhead, removed here before adding new stage/I/O allowances. Query count, graph dimension, arithmetic and filesystem work do not scale uniformly. Source-only all-reported output growth is1973 files (192×9 cell files,24×10 world files,5 campaign-level files); future byte growth remains unknown. No result values or cell statuses were decoded to choose a scenario.

**Disposition:** the slow GPU-tree read makes preparation1200 tight in the conservative two-BEFORE-pass worksheet. Keep this observation and the earlier incomplete240-second CPU scan; do not select only a faster after pass. Once AFTER is complete, compare the actual two-observation sequence and repeated-slower-observation sensitivity, then assign explicit residual-work reserves and show the joint schedule. This can justify one bounded attempt without proving a universal upper bound. If the evidence calls for a phase adjustment, propose it explicitly before adoption. Any timeout/failure remains retained under the original hard stop; no retry or extra probe is inferred.

Parent's provisional accounting37073CPU/4458GPU +3600 preparation +7200 proposed finalizer =47873CPU, leaving9727 under57600. Actual accepted baseline pins must replace that provisional attribution. No qualification allocation/history extension is included. No sustained login probe, remote action, lock, test/model/scientific execution or current-capsule mutation occurred. The new local metadata derivation ran once:1.193087 child CPU seconds,1.228502 wall,521,420,800-byte peak RSS; sources/output/transcripts retained in attempt01. Additional arithmetic/source-capture bookkeeping is not individually metered.
