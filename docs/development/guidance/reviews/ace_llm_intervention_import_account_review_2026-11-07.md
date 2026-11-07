**Independent account-only CPU diagnostic review — November 7, 2026.**

**Required findings: 0.** The corrected account guard, launch packet and
unexecuted launcher pass this bounded static review. This removes the review
objection to this packet; it is not evidence of submission, successful imports,
current remote readiness, runtime qualification or authorization for GPU work.
Parent must commit the reviewed bytes and perform the fresh launcher
reconciliation/readiness/submission-journal steps before a single submission.

Reviewed [launch02 probe](/Users/pat/code/ACE/results/ace_llm_intervention_import_diagnostic_20261107/launch02/probe.py),
its job script, packet, four fabricated guard cases and receipt, and
[launch_once.py](/Users/pat/code/ACE/results/ace_llm_intervention_import_diagnostic_20261107/launch02/launch_once.py).
Compared against the retained failed job's source/script and the previously
closed diagnostic implementation review. Read the saved account-migration and
terminal records. No old test, new test, diagnostic, worker, model, project
import, remote operation, submission or commit was executed in this review.
Only this review document was written for the account task.

**Minimal change and bindings.** The 12,147-byte probe differs from preserved
`a885504774e34df3245d759a57075456d46970807393b8e022d325b2a54d9fec`
by exactly one literal substitution at line 205: allocation account
`ucb736_asc1` becomes `ucb736_asc2`. Verified exact byte equality after that one
replacement. Import orders, source capture, deadlines, owned-worker cleanup and
all other allocation checks are unchanged. The job script changes only its
account, job name, new output/probe directory and embedded probe pin. The source
bootstrap still hashes the bytes it compiles. Packet input pins match both files.
GPU10 service, frozen-manifest and packet pins are unchanged from the prior
packet; no GPU10 source or runtime was executed or requalified.

The account source is Patrick's supplied renewal `INC1508105`. The retained
association response admits `alpine|ucb736_asc2|paco0228` and includes
`cpu-normal`. Migration records show a successful account update on the same
pending job `35341862`, not another submission. These are saved observations;
this review did not query current remote association or queue state.

**Failure preservation and resource reconciliation.** Independently verified
all ten raw files and 40,872 bytes under `terminal01/remote/` against both the
disposition inventory and captured remote inventory, including empty stdout,
stderr, allocation response, unchanged source/script, original submission
records and account-transition records. The captured allocation has the renewed
account and every other fixed one-CPU condition. Stderr terminates at the old
account guard before the worker loop; there are no worker artifacts. This was
a pre-import guard failure, not a new import/runtime result.

Captured top-level accounting is `FAILED`, exit `1:0`, one allocated CPU for
12 seconds, `cpu=1,mem=3G,node=1`, with zero GPU allocation. Its `TotalCPU`
measurement is 0.166 seconds; that is distinct from 12 allocated CPU-seconds.
Batch/extern rows overlap the top-level allocation and must not be added again.
The thirteen prior closed-job rows independently sum to 4,450 allocated
CPU-seconds and 836 one-slice GPU-seconds. Adding this failure gives fourteen
closed jobs and 4,462 allocated CPU-seconds.

The new packet requests `ucb736_asc2`, `acpu/cpu-normal`, one node/task/CPU,
3 GiB, 480 seconds, no requeue and no GPU. Admission arithmetic is correct:
`4462 + 3600 cumulative preparation allowance + 480 new reservation = 8542`
CPU-seconds, below 57,600; GPU accounting stays `836 / 7200`. The failed job's
old 480-second reservation is released rather than counted again. Historical
allocation totals do not establish GPU utilization or scientific progress.

**Focused fabricated controls.** The retained receipt records exit zero,
0.068828 child CPU seconds and 0.07735208399999999 elapsed seconds, with empty
stderr. Source inspection and saved case artifacts agree: renewed account reaches
exactly one mocked `Popen` boundary; expired account, wrong CPU count and allocated
GPU each reject before it. The renewed process record contains `WorkerBoundary`
and no PID. No actual worker started. These four cases were inspected, not rerun,
and do not demonstrate actual scheduler or import behavior.

**Static launcher check.** Lines 20–27 bind probe, script, packet and launcher
bytes plus this review to the current commit before remote action, and record
commit/review/source pins. Lines 30–32 acquire both established controller locks
nonblockingly. Readiness and pre-submission reconciliation require no related
active job, fourteen terminal prior jobs, 4,462 allocated CPU-seconds, the same
836 GPU-slice-seconds, thirty prior submission journals and the renewed CPU
association. Journal hashes are retained; this count/aggregate gate is not an
independent re-audit of every historical journal's meaning.

The launcher creates an exclusive remote directory, checks scratch ownership
and write/fsync/readback/cleanup, and records free space/`df`. It transfers only
the three pinned inputs, authenticates them remotely, and checks them again
immediately before submission. Local/external and remote intents are exclusive
and durable before the single `sbatch --parsable` call. There is no retry loop.
Timeouts or unparseable submission outcomes leave a null job ID and require
reconciliation; a local intent or existing remote root/journal prevents blindly
repeating the launch. Transport loss can leave an uncertain remote outcome, so
the parent must reconcile scheduler and retained remote evidence instead of
relaunching. Lock release occurs in `finally`. Parsed the launcher and its three
embedded remote program bodies statically; none was executed.

The launch packet including the launcher contains **14 files / 30,240 bytes**,
byte-identical to the local custody copy at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-import-diagnostic-20261107-02`.
At inspection there were no controller, readiness, submission or launcher-failure
artifacts for launch02. The future launcher will add execution evidence; the
inventory above describes the reviewed preparation state.

SHA-256 pins, relative to `results/ace_llm_intervention_import_diagnostic_20261107/`:

- `launch02/probe.py`:
  `8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951`.
- `launch02/job.sbatch`:
  `9d9f230db70214ef85283e634dbc5e99b4ac84a3a58bc866a984370c08b98ca3`.
- `launch02/packet.json`:
  `546ffe5d314e790339b38bc86521b92992d56e481057a70a00a07780210a9057`.
- `launch02/launch_once.py`:
  `9b40cc31617e040e3ac0f3f355ef6531c4932e5a80d5636876cde3fc45f812a7`.
- `launch02/test_account_guard.py`:
  `d53211fa89bb21247b5ef7065c6ffbcd75e1d265ad8cdb0af8e45f7b7110aac7`.
- `launch02/account_guard_receipt.json`:
  `0a57bc410c6742300448d81ff53bfbbea1be4f7f53c178644dee3e68ccf6f400`.
- `launch02/account_guard_results.json`:
  `207027b2bcefce8049e1653eac72e5264b4e9f6d543450f1a6e3a1a1d61c5857`.
- `terminal01/disposition.json`:
  `2dffd7763d01f6f3c97bce50cb8e074cd6620400f2be5463f5914601dcb38b65`.
- `terminal01/inventory_transport.json`:
  `445cfd339ee613eba5fdf23a05bf36e33cfa613f9ae29d212c6bc8cb82580a0f`.
- Unchanged GPU10 service:
  `908496e344f97f718e0a5061728bf206cb67df291003ffd2ac85ba4c9cd650a1`.
- Unchanged GPU10 frozen manifest:
  `d33bc0f4a23b4a82f1ee8f433688e706a893e9127ef41b8abe970aab46878d37`.

The prior review's timing/cache, selected-source and uninstrumented dependency
capability-check limitations still apply. A retained diagnostic wrapper result
does not itself establish successful imports. Source/commit custody, fresh
readiness, actual submission reconciliation and subsequent outcome/accounting
remain the parent's responsibility. This account correction does not authorize
automatic GPU retry or any change to the scientific workload.
