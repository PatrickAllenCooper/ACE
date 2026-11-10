# CPU finalizer site readiness

Fresh metadata captured under both controller locks after coordination with Leibniz (no competing remote/alias operation). Observation2026-11-10T04:40:35.432627Z to04:40:36.053415Z on CURC. The original CPU readiness command/output format is reused with explicit acpu/QOS/profile/namespace gates; predecessor source is retained. No GPU readiness record is reused as CPU evidence.

Accountucb736_asc2 admits cpu-normal; acpu isUP and permits it. Both partition and QOS MaxWall are24hours, sufficient for4200seconds. MaxMemPerCPU is3840MiB:2CPU/7680MiB is exactly the limit, with no memory headroom claim. Profile is noGPU/no-requeue. These are current site limits, not an allocation or execution qualification.

No active ace_llm jobs were observed;48 terminal ACE parents report37073CPU seconds and4458GPU slice seconds. This is a scheduler observation only: full98-journal reconciliation and accepted-baseline authority remain separate launcher/build gates. Unrelated queued jobs, including35501725, were observed and untouched. Both original and recovery namespace bytes match independent pins. The prospective finalizer output root is absent. The existing workflow's exclusive scratch write/read/fsync/unlink/rmdir probe succeeded and was removed; no job/output root was created.

Projects quota reports74G available; statvfs reports78610694144available bytes. Scratch user quota reports6238G available; filesystem availability is separately retained. curc-quota returned0 with three TERM/tput formatting warnings preserved verbatim. Local and remote UTC clocks are recorded separately; cross-host time subtraction is not used. Full transport took33.312553958local wall seconds and0.011679child CPU seconds; remote metadata work reported0.620811086seconds.

No packet build, adoption, allocation, model/scientific work, or Git action occurred. Explicit actual GPU acceptance/recovery/baseline and clock adoption remain required. The later launcher performs fresh full reconciliation before submission.

Source/command/transport/raw evidence hashes and scoped facts follow; complete raw metadata output is retained separately.

```json
{
  "acceptance": false,
  "accepted_accounting_baseline": false,
  "account_association": true,
  "active_ace_jobs": 0,
  "build_invoked": false,
  "clock_scope": "Local and remote UTC clocks independently recorded; not a cross-host elapsed clock.",
  "command_sha256": "4158a47236234e0d5fd19ed47adc5af1d00605b2b48e4c1a7dad8a9b262e7643",
  "dedicated_scratch_probe_removed": true,
  "full_journal_reconciliation_performed": false,
  "held_both_locks": true,
  "local_return_utc": "2026-11-10T04:40:35.847154+00:00",
  "locks_released": true,
  "max_memory_per_cpu_mib": 3840,
  "memory_request_at_exact_per_cpu_maximum": true,
  "namespace_pins": {
    "/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json": "40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a",
    "/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace.json": "58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29"
  },
  "observed_cpu_seconds": 37073,
  "observed_gpu_slice_seconds": 4458,
  "observed_terminal_ace_jobs": 48,
  "output_root_absent": true,
  "partition_and_qos_max_wall_seconds": 86400,
  "partition_up": true,
  "profile": {
    "account": "ucb736_asc2",
    "cpus": 2,
    "gpus": 0,
    "memory_mib": 7680,
    "partition": "acpu",
    "qos": "cpu-normal",
    "requeue": false,
    "wall_seconds": 4200
  },
  "projects_statvfs_available_bytes": 78610694144,
  "quota_reported_projects_available": "74G",
  "quota_reported_scratch_available": "6238G",
  "quota_stderr": "curc-quota return0; retained three tput TERM warnings",
  "raw_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "raw_stdout_sha256": "08d0d5dfbe933d7530b4bac208415a48ab0eaa0c3380ee17dbf32dd6f8ea6eee",
  "remote_finished_utc": "2026-11-10T04:40:36.053415+00:00",
  "remote_started_utc": "2026-11-10T04:40:35.432627+00:00",
  "run_sha256": "20781f8a218b4ce210822f7cdeeb15525809df513d2ed997b322cc506e3be225",
  "scientific_execution": false,
  "source_sha256": "f207ec822caf8e3e406fe1cd82ecbb25ed06dae1bcb6be02bf7197624b0e2240",
  "status": "fresh_cpu_site_readiness_observed",
  "submission_invoked": false,
  "transport_sha256": "7149b987da595a3008d5efa39f59ae5a341f47bb01bceb0b1604cf3104366770"
}
```
