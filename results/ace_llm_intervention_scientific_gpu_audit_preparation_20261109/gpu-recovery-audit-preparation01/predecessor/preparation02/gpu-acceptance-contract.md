# Existing finalizer GPU consumer — documentation only

Exact17keys, from frozen28c5fc6 finalizer and captured downstream finalizer builder:
```
schema status scope gpu_packet_sha256 cpu_packet_sha256 cpu_acceptance_sha256
job_id allocation gpu_output campaign_root gpu_output_files campaign_files
bundle_sha256 attempt_pins accepted_checks terminal_pins scientific_qualified
```
schema=ace-scientific-campaign-gpu-acceptance-v1; status=development_gpu_phase_accepted; scope=development; scientific_qualified=false. No utility acceptance is implied.

All packet/CPU acceptance/job pins must bind actual original independent evidence. allocation is exactly accountucb736_asc2, partitionaa100, qosgpu-testing, requested_cpus4, allocated_cpus4, memory_mib15360, wall_seconds from the original packet (current concrete contract2880), gpus1, gpu_typea100_3g.20gb, stateCOMPLETED, exit_code0:0.

gpu_output and campaign_root are original logical remote paths. gpu_output_files and campaign_files are COMPLETE relative-name→SHA maps from full authenticated archived inventory, never merely the selected local subtree. bundle_sha256 equals accepted CPU's original opaque bundle pin. attempt_pins contains all8ordered-policy identities, preserving the5CPU pins and adding3GPU original attempt/candidate pins. accepted_checks is exactly {scheduler:true,runtime:true,cleanup:true,custody:true}, authored only by independent review. terminal_pins has exactly terminal.json and controller/phase/terminal.json, each matching gpu_output_files. CPU outputs/inputs and every campaign member outsidepolicy05..07 remain byte-identical to accepted CPU; no prior private finalization/retry directories.

This file is NOT an acceptance. The audit report does not create or assert accepted_checks. Actual independent review must close the explicitly remaining original gates before any such record is authored.
