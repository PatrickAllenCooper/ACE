# Campaign final01 concrete packet review — 2026-11-08

**Required findings: 0.** The concrete packet passes this bounded static integration review. Packet SHA-256: **`417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c`**.

Reviewed the new `build_packet.py`, `launch_once.py`, `monitor_final.py`, `pull_terminal.py`, and their concrete packet/job bindings in `results/ace_llm_intervention_campaign_finalization_20261108/launch01`. Reused the closed finalizer implementation review and actual GPU01 acceptance; neither was rerun. Only local byte comparisons, arithmetic, source inspection and syntax parsing were performed. No project validator, builder, launcher, monitor, collector, model, scoring, tests, remote action or commit was executed.

The captured finalizer matches source commit `12c7322e5046cd3e9b38cb59a1c01ed87054b71b` and reviewed SHA `88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f`. The exact source/input roster extends the unchanged GPU packet by only the finalizer and GPU packet/acceptance files. All **94 transfer pins** match the external mirror: **82 inputs + 11 root artifacts + metadata**.

Upstream byte pins match their original authorities:

- CPU packet: `0533b4fad722652ec63ed12850d3a3d55d1f6c391e81f7280d36561aa9e5ac13`.
- CPU acceptance: `11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f`.
- GPU packet: `970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976`.
- GPU acceptance: `26119972f7c3e61caf03d15bc72371bfe64dc118edd55d1e6bd26f917e994d0b`.
- GPU review receipt: `1147e74630d1c12b8519e7dc2446a2b112c517e4518aa95852c9f6f54ed52016`.

Original CPU/GPU input, output and campaign paths remain unchanged. The new root owns launch evidence and `finalization/` supervisor output. The accepted eight attempt pins supply the previously reviewed 16-cell/28-pair reporting contract. The builder's local admission path remaps only its validation view; saved packets and acceptance paths retain the original remote locations.

The job requests **2 CPUs, 7680 MiB, 300 seconds, zero GPUs**, `ucb736_asc2/acpu/cpu-normal`, and no requeue. Captured worker deadlines remain **120/260/280/290 seconds**. Actual predecessor charges recompute to **10,246 CPU / 1,483 GPU-slice seconds**; adding 3,600 preparation CPU-seconds and the new 600 CPU-second reservation yields **14,446 / 1,483**, within **57,600 / 7,200**. The exact predecessor extension contains **35 jobs / 72 pinned submission journals**.

Both coordinator locks cover readiness, transfers, readback and submission, with a second local claim check under lock. The launcher rechecks complete original roots, upstream acceptance, exact scheduler rows/journal pins, account/QOS, memory ratio and scratch readiness. All transferred pins and captured authentication are checked before its single `sbatch`; inherited `SBATCH_*` overrides are removed. The new root's submission intent is written after predecessor reconciliation, preserving the expected census. Exclusive journals and existing-attempt guards retain failed/ambiguous submissions without blind retry.

The monitor performs read-only scheduler/phase observations, flags missing startup progress and preserves campaign receipt authority after an outer failure. The collector requires terminal accounting and no related active jobs, holds both locks, retains complete launch/common roots including private and failure artifacts, checks remote inventory again after transfer, verifies the second local copy, and reconciles actual resources without granting acceptance. Existing custody directories block blind collector reuse.

New source SHA-256 pins:

- Builder: `4d708f7dbd157b55eabf46319921022235ae51bec2303b20b08a5cf9f270c632`.
- Launcher: `961c775413d9bfbfac7a66adf8f84a5c1e1a25b176ddcf7ec8e0d02d79019fae`.
- Monitor: `5b033b252a22afbd571629f073d77451855ca0d1abe73203a2250c975251b248`.
- Collector: `5201118fc81edf5ca5ef165b547bad6dac0fad1ba5527eabc1a58e7d66c9bdb7`.

This closes the new concrete packet review only. Remote readiness/readback, successful finalizer execution and independent final-result acceptance remain prospective; the five-minute budget is not an observed finalizer measurement. No additional review contract is introduced.
