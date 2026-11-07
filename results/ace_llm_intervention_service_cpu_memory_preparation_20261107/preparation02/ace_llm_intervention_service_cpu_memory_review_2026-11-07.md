# CPU service failure and memory correction review

November 7, 2026. Independent read-only reviewer Beauvoir (01a1b7cb-f8ae-7790-a8c6-298f5f893b1c). Final disposition: zero required corrections.

The reviewer checked all 36 raw files (4,967,244 bytes) and all 40 custody files including collector records (4,994,314 bytes) against the remote inventory and exclusive external originals. All 28 transferred input pins and 17 committed source bindings match.

Job 35348012 ended FAILED 1:0 under ucb736_asc2 with two CPUs, 4 GiB, nine seconds and zero GPUs. The traceback reaches the allocation guard before staging or model-library imports. No service sessions, model loads, updates or environment responses occurred. This is complete custody of a failed resource contract, with no runtime or scientific qualification.

Observed allocation is 18 CPU-seconds. Closed allocations total 6,384 CPU-seconds and 836 GPU-slice-seconds. Admission including the unchanged 3,600-second preparation allowance is 9,984 / 57,600. Slurm process CPU of 0.160 seconds is an overlapping measure, not an additional charge.

The correction changes exactly three implementation lines: supervisor schema v2, memory_mib=3840 and required AllocTRES mem=3840M. Worker, model, runtime and deadline behavior are unchanged. Saved acpu partition evidence reports MaxMemPerCPU=3840. The new request fits that one-CPU limit. Prior diagnostic batch RSS was near 3 GiB, so this gives additional room, but actual service memory sufficiency remains unproven.

Reviewed supervisor SHA c827a53f0fe03204c447e8e7d406e6d8a2c9318b4d6cd5fdc82beb84b576b6b7. The new memory rejection method and affected allocation method passed in preserved logs. The reviewer did not rerun them. A fresh 600-second reservation would yield 10,584 / 57,600 admission. New-packet review and fresh scheduler gates remain required.

Review performed no edits, project-source execution, tests, remote calls, archive hashing or model loading. Original failed packet, collector, logs and receipts remain unchanged.
