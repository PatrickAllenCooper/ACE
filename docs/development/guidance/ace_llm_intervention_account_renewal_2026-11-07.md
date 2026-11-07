# Renewed CURC account for ACE

Patrick supplied CURC's approved renewal notice INC1508105 on November 7. The account for continued ACE work is **ucb736_asc2**, the successor to ucb736_asc1. The notice grants up to 450,000 SUs through November 6, 2027. The other two renewed allocations in his message are not selected for this project.

This renewal does not enlarge the research cycle. Its ceiling remains 57,600 CPU-seconds including GPU-host CPU and 7,200 GPU-slice-seconds. The 3,600-second local preparation allowance remains cumulative. Actual allocation, process CPU and wall time remain distinct quantities. No unused allowance authorizes unrelated work or automatic scientific scaling.

A live association query confirmed paco0228 has access to ucb736_asc2, with cpu-normal among its QOS values and no explicit zero MaxJobs field. Under both shared controller locks, the existing pending CPU diagnostic 35341862 was moved to that account at 17:55 UTC using Slurm's documented account update. No new job was submitted by this change, and its source, resource request and output path remained unchanged. The AssocMaxJobsLimit reason cleared. Unrelated jobs were left untouched.

The first migration preflight stopped before mutation because it compared the local pre-submission intent with the enriched remote intent. The original submission result independently pins the enriched remote intent. Read-only reconciliation confirmed that pin, the local declaration fields and absence of transition journals before the corrected migration. Both preflight records remain preserved.

The migrated job then failed before imports. Its frozen runtime guard still required ucb736_asc1. The parent missed this guard when changing the scheduler account. Slurm records FAILED/1:0, one CPU for 12 seconds and no GPU. Complete custody contains 10 remote files totaling 40,872 bytes, including both original submission journals, the account transition, allocation capture and logs. No diagnostic worker, model load, training or scientific response ran.

The new unsubmitted launch02 packet changes exactly one account literal in the probe. Its script uses the renewed account, new exclusive output and corresponding source digest. The import orders, deadlines, memory, CPU count and zero-GPU requirements are unchanged. Four new fabricated allocation checks passed, with no worker launch. The renewed account passes the guard, while the expired account, wrong CPU count and GPU allocation reject. These checks do not qualify actual imports.

Closed allocation is now 4,462 CPU-seconds and 836 GPU-slice-seconds. A new eight-minute, one-CPU diagnostic would make admission 8,542 CPU-seconds after the cumulative preparation allowance, within the existing ceiling. The independent account/packet/launcher review has zero required corrections. Submission still requires current queue/journal reconciliation, both controller locks, scratch readiness and independent remote byte verification. Failed or ambiguous submissions must be reconciled rather than retried blindly.

Original packets remain historical evidence and retain their original account. New allocation freezes must name ucb736_asc2 consistently in the script, runtime guard and reservation record. A pending account update alone does not change a pinned runtime guard. The account-migration operation follows the [Slurm scontrol reference](https://slurm.schedmd.com/scontrol.html).

The renewal requires Alpine acknowledgement in resulting publications. Its supplied wording is:

> This work utilized the Alpine high performance computing resource at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, Colorado State University, and the National Science Foundation (award 2201538).

The notice also retains the PI's reporting and user-eligibility responsibilities. This record does not submit a paper, amend the archived manuscript or grant a new resource budget.
