# Runtime-duration proposal and terminal10 review

Reviewer: Godel. The review addresses the preserved draft/source at preparation01. The later guidance addendum only records this review disposition; source bytes are unchanged.

The completed review covered:

- **Source and wall policy:** Inspected the supervisor/qualifier diff, new wall-policy tests, and affected 30-minute fixture. Confirmed exact frozen pairs `1800/00:30:00` and `3600/01:00:00`, scheduler equality with the selected cap, and child deadlines of 1750/3550 seconds measured from the original supervisor start. The 50-second allowance is relative to that start. CPU, 3 GiB, account/partition/QoS, accelerator exclusion, no-requeue, source, package, and numerical gates were unchanged by the diff.
- **Preparation lineage:** Matched the external preparation’s source, test, and proposal copies to repository files and verified recorded hashes. Read the saved four-test success report; no tests were rerun. Deadline wiring was inspected statically, not exercised.
- **Terminal custody:** Independently checked all 20 raw files, totaling 9,090,171 bytes, against the captured inventory. All nine frozen inputs matched launch10’s local and remote pin records. Checked receipt and proposal lineage hashes.
- **Runtime and numerical boundary:** Verified the initial qualification receipt’s freeze, source, interpreter, dependency, and inventory bindings: 28,577 objects and 7,453,922,966 file bytes. Its timestamp was 1412.833795 seconds after the wrapper’s start record. Neither pre-checkpoint nor full-replay receipt existed. The guard writes its receipt before forwarding the first load; saved checkpoint loading and numerical reproduction remain unqualified. The exact interrupted subphase remains unknown.
- **Termination and accounting:** Preserved records showed child `time_limit`, completed cleanup, and 1766 parent allocated CPU-seconds. Adding that to the frozen prior 120559 gives 122325 seconds. Reservation arithmetic is 90.368333 + 1.0 = 91.368333/150 core-hours; the historical baseline was not independently re-audited. Overlapping CPU and elapsed measures were not added. Doubling the observed first-gate interval is a planning allowance, not a measured completion estimate.

The review used local source and preserved metadata only: no edits, remote commands, installs, model imports, scientific execution, or outcome decoding. The consumed single 30-minute approval does not authorize the proposed single 60-minute attempt.

Required findings remaining: 0
