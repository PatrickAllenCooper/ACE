# Consolidated downstream handoff — metadata deadline delta

Use implementation/ in this NEW preparation. It includes the independently closed D1/D2/D3 corrections from /tmp/ace-downstream-compact-parent-fix-j795b8og, plus only the new metadata-timeout diagnosis correction. The original17:10 preparation and every intermediate source/test/failure/review remain preserved. D1/D2/D3 review0 SHA23e5fdd54872cb7775c092ded23bc1ed6460166ab41cf07b2b01dc42eb169fa2 is captured in d1d2d3-closure-review/. No work was abandoned or awaiting an old API response.

## Exact metadata-only changes

GPU launcher48s remote/55s local becomes1200/1230. Finalizer launcher300/305 becomes1200/1230. Both reconcile full upstream bytes, so the same diagnosed metadata duration applies. Their read helper reports only phase, count and elapsed seconds to stderr every1000 successfully authenticated read operations. Counts include repeated reads, not distinct files. Existing source/path/link/race/size/hash checks are unchanged; progress never resets the absolute alarm.

Full terminal collectors also inventory and hash the complete growing history. GPU collector previously had a240s local observation timeout and no absolute remote observation alarm; it now uses1200 remote/1230 local. Finalizer collector300/305 becomes1200/1230. Their fingerprint helper emits the same count-only progress interval to stderr. Existing partial stdout/stderr handling, second observations, locks, claim gates, accounting, resources, allowed appends and failure retention are unchanged. Small live monitors remain45s/50s because they read bounded current status evidence rather than full history.

The parent's measured original CPU preflight failed before any submission during full-upstream readback. Parent reports45jobs/92journals, no CPU root/ACE queue, and completed36k-file backup974.6s as the bounded sizing basis. Captured CPU recovery controller/proof/reconciliation hashes are in diagnosis.json; source copied read-only into diagnosis-reference. This sidecar did not execute or modify that recovery.1200seconds is a finite operational bound, not a promise that larger or slower histories finish. A timeout remains a retained failure with no automatic retry.

No Slurm wall limit, requested/allocated resource profile, GPU startup/work/service deadline, scientific source, roster, input, method, endpoint or acceptance gate changed. Backend527d and authority02a7 remain closed. Builders/shared adapter/history helper/monitors are byte-identical to accepted D1/D2/D3; only four outer launcher/collector files changed. No packet, remote request, job, model or scientific code was run.

## New evidence

Three NEW fabricated methods passed once, no previous check repeated: (1) progress cadence at1000/2000 with source checks and no payload/path/stdout leakage, (2) absolute1200 alarm and1230 transport plus retained partial progress on timeout, and (3) protected reconciliation/resource/scientific/read-gate equality. Tests execute only captured small helpers and mocked transport on artificial local bytes; no main or actual signal timer was invoked. attempt01 stores exact sources, argv, stdout/stderr and receipt.0.174836childCPU/0.180929041elapsed/23,609,344RSSbytes. No new failed test in this delta.

## Exact new source pins

- `finalizer/launch_once.py`: `93369b72270d9563450e9ce5dbf18158001765e0457dbf0c38842d70bc989f7b`
- `finalizer/pull_terminal.py`: `35266b2b094a12960d6f7f6601f1a43e42c145911645047a970ae8fc38db70d0`
- `gpu/launch_once.py`: `4f60e336085322f13663d300e64f844031ac20cc9caaf070abcb15d5a302feaf`
- `gpu/pull_terminal.py`: `25877ff854f59523eed009b3f5c5cc00779f5b19b7c86da1a74454dad360c735`
- `test_metadata_deadline.py`: `8e158117130534f5996ed00d85f930201cb30fdc296c3354505ddb4059bf12c1`

Complete final source index and receipt pin: handoff.json. Narrow metadata-only independent review requested; no actual operational acceptance claimed.
