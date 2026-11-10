# Prospective operational clock interface

Use the six files under `implementation/` as one separately reviewed successor of finalizer-recovery-preparation01. The original builder CLI is retained with three additional mandatory metadata path/SHA pairs:

```
--finalizer-clock-adoption PATH --finalizer-clock-adoption-sha256 SHA256
--finalizer-timing-evidence PATH --finalizer-timing-evidence-sha256 SHA256
--finalizer-timing-review PATH --finalizer-timing-review-sha256 SHA256
```

These are future independent inputs, with no defaults. No invocation or actual adoption artifact is supplied here. Ordinary 32MiB metadata bounds apply. Evidence and review are captured opaque, nonempty bytes (`finalizer-timing-evidence.md`, `finalizer-timing-review.md` in the outer packet); the evidence may be a concise sizing document binding the retained raw timing receipts. The adoption is parsed as metadata. Both builder and launcher reauthenticate it before storage or submission. The collector reuses the same builder admission. Existing source-manifest and review inputs refer to the committed successor source; the original source authority remains independently checked.

Required adoption fields (extra explanatory fields permitted):

| Field | Required value or binding |
|---|---|
| schema/status/scope | `ace-finalizer-clock-adoption-v1` / `adopted` / `operational_finalization` |
| remote_root | Existing finalizer root, unchanged |
| freeze_sha256 | Original independent scientific freeze pin |
| source_commit, source_manifest_sha256, source_review_sha256 | New independently committed finalizer source and reviewed manifest/bytes |
| entry_sha256 | Exact successor finalizer entry pin in source-pins.json |
| predecessor_entry_sha256 | `b4d18f4be2320069ffc5f2aed2060efcfb8d49bfcc6e8b00f8099b91598f8745` |
| original_wall_seconds | 1200 |
| original_deadlines | `{startup:120, work:1140, cleanup:1170, hard:1190}` |
| resources | Exact existing account/partition/QoS/2CPU/7680MiB/0GPU/no-requeue with wall_seconds3600 |
| deadlines | `{preparation:1200, report_startup:120, report_startup_cap:1320, work:2820, cleanup:2850, hard:3590}` |
| cpu_acceptance_sha256 | Actual independent CPU campaign acceptance |
| gpu_acceptance_sha256, gpu_recovery_acceptance_sha256 | Actual independent GPU02 core acceptance and recovery supplement |
| baseline_sha256 | Actual accepted GPU02 48/98 baseline; no qualification job is assumed |
| timing_evidence_sha256, timing_review_sha256 | Exact independent evidence/review bytes supplied to this CLI |
| full_path_timing_supported | True only after independent full-path timing disposition |
| required_findings | Integer0 from that disposition; bool is rejected |
| scientific_inputs_unchanged | True |
| scientific_qualified, execution_qualified | Both false; adoption does not accept a future execution |

This record binds reviewed evidence rather than letting the builder infer adoption from timing numbers. Its source fields bind the entry, not a future packet, so no source/authority/packet hash cycle is introduced. No blanket-approved source authoring is treated as this future timing acceptance.

Inner schema becomes `ace-scientific-campaign-finalizer-phase-clock-v3`; outer becomes `ace-scientific-finalization-phase-clock-launch-v3`. The inner packet adds `clock_adoption_sha256`; captured flat files add `finalizer-clock-adoption.json`. Source names and all inherited source/input pins remain unchanged except this explicitly committed entry. The child's pre-private admission rechecks adoption bytes, profile, entry/commit/freeze/acceptance/baseline joins and affirmative timing disposition. Outer packet additionally binds the adoption pin; its existing input map captures the evidence/review. Original GPU acceptance remains exactly17 keys.

The original launcher/monitor/collector CLI remains unchanged. Scheduler and request verification now require `01:00:00`/60 minutes and elapsed<=3600; actual accounting still charges observed seconds*2. No current baseline, receipt, authority or packet was generated. Build/launch readiness still requires actual GPU02 acceptance and custody, independently accepted timing support, operational adoption, source review and concrete packet review.
