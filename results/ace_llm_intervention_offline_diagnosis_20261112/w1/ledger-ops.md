# ledger-ops: operational reality of the November 2–12, 2026 ACE LLM-intervention cycle

Prepared November 12, 2026 by the `ledger-ops` reader. Read-only work. Nothing under `/Users/pat/code/ACE`, `/Users/pat/code/ACE-Runner` or `/Users/pat/ACE_Study_Results` was changed. No SSH, Slurm, model load or repository launch script was run. All numbers below were read from saved files or computed offline with `/usr/bin/python3 -I` from saved JSON/CSV. Scratch scripts are in `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/ledger-ops/` (`accounting.py`, `gen_stats.py`, `gen_stats.json`, `commit_times.txt`).

Conventions: **[V]** verified directly in a file, **[C]** computed by me from saved files, **[I]** my inference. Times are UTC unless marked local (Mountain, UTC−7). "Ledger" = `/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md` (1,513 lines, newest first). "Handoff" = `/Users/pat/code/ACE/docs/development/guidance/handoff_ace_llm_research_2026-11-12.md`.

Primary sources read in full:
- Handoff (124 lines), agenda `ace_llm_intervention_agenda_2026-11-02.md` (199), results `ace_llm_intervention_24_system_results_2026-11-10.md` (118), Ledger (all 1,513 lines).
- `ace_llm_intervention_account_renewal_2026-11-07.md`, `ace_llm_intervention_remote_custody_2026-11-09.md`, `ace_llm_intervention_scientific_campaign_2026-11-08.md`, `ace_llm_intervention_gpu_qualification_2026-11-02.md`, `ace_llm_intervention_staged_runtime_2026-11-07.md`, `ace_llm_intervention_screen_execution_design_2026-11-08.md`, `ace_llm_intervention_paired_campaign_integration_2026-11-08.md`, `ace_llm_intervention_scientific_contract_2026-11-08.md`, `ace_llm_intervention_screen_protocol_2026-11-07.md`, `ace_llm_intervention_scientific_collection_recovery_2026-11-08.md`, `ace_llm_intervention_scientific_mount_recovery_2026-11-08.md`, `delivery_continuation_authorization_2026-10-24.md`, `delivery_paper_archive_2026-11-02.md`, parts of `delivery_submission_form_draft_2026-11-01.md` and `ace_submission_audit_2026-11-01.md` (all under `/Users/pat/code/ACE/docs/development/guidance/`).
- `/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/{review.md,acceptance.json,baseline.json}`.
- `/Users/pat/.claude/skills/curc-access/SKILL.md` (read only, no connection made).

---

## (a) Timeline, November 2 to November 12

### Whole-cycle totals [V/C]

- Pivot commit: `Preserve delivery paper and define renewed LLM intervention agenda`, 2026-11-02 19:33 UTC. First Slurm job 20:03 UTC the same day (Ledger L557). Final report commit 2026-11-10 08:51 UTC. Handoff commit 2026-11-12 15:14 UTC. (read-only `git log`).
- Calendar span pivot to final report: **181.3 hours (7.55 days)**. 257 commits in that span [C].
- Work happened in **six sessions totalling about 39 hours** (commit-gap proxy, gaps over 2 h excluded) [C]: Nov 2 19:33–Nov 3 00:27 (4.9 h, 57 commits); Nov 4 15:58–16:40 (0.7 h, 7); Nov 7 16:19–22:43 (6.4 h, 49); Nov 8 04:16–17:47 (13.5 h, 98); Nov 9 17:16–23:16 (6.0 h, 20); Nov 10 01:19–08:51 (7.5 h, 26). No commits on Nov 3 (after 00:27), 5, 6 or 11.
- **50 submission attempts, 100 submission journals**: 49 allocated Slurm jobs plus 1 submission rejected by Slurm before a job ID (Nov 2 GPU01, wrong QoS, Ledger L616–618). Of the 49 jobs: **28 COMPLETED, 21 FAILED** [C from `baseline.json` `related_accounting`, 49 rows]. Of 21 GPU jobs, 12 failed.
- Allocated compute, all 49 jobs: **40,265 CPU-seconds (11.18 core-hours) and 4,458 GPU-slice-seconds (1.238 slice-hours)**. My independent sum of `AllocCPUS × ElapsedRaw` over the 49 rows reproduces both figures exactly [C]. Sum of job elapsed time is only **16,475 s = 4.58 hours**.
- Failed jobs consumed 7,155 CPU-s (17.8%) and 1,175 GPU-slice-s (26.4%) [C].

### Phase table [C from `baseline.json` rows; dates from Ledger entries]

| Phase | UTC span | Slurm jobs (failed) | Alloc CPU-s | GPU-slice-s |
| --- | --- | ---: | ---: | ---: |
| A. Contract primitives and runtime/numerics qualification (rows 1–12) | Nov 2 20:03 – Nov 3 00:16 | 12 (5) + 1 rejected | 3,634 | 632 |
| B. Persistent model service and runtime qualification (rows 13–27) | Nov 4 16:32 – Nov 8 04:26 | 15 (8) | 4,398 | 459 |
| C. Artificial training and 2-world paired campaign integration (rows 28–38) | Nov 8 04:55 – 10:16 | 11 (4) | 3,008 | 510 |
| D. Scientific chain (rows 39–49) | Nov 8 12:20 – Nov 10 06:59 | 11 (4) | 29,225 | 2,857 |
| Total | | 49 (21) | 40,265 | 4,458 |

Phases A–C (38 jobs, before any scientific system was drawn) cost 11,040 CPU-s and 1,601 GPU-slice-s, which matches the Ledger's pre-realization closed total (L407, L415).

### Scientific chain by stage [V for job data, C for spans from commit times and Ledger entries]

| Stage | Wall-clock span (UTC) | Jobs (failed) | Alloc CPU-s / GPU-s | Job elapsed | What failed |
| --- | --- | ---: | ---: | ---: | --- |
| Contract (freeze) | adopted Nov 8 12:19; successor adopted 12:54 | 0 | 0 | – | First freeze `57a3e6dd…` at source `f97cb08e` superseded after realization01 by candidate03 freeze `15602614…` at source `a9346022` (Ledger L379, L393). Candidate02 never adopted. |
| Realization (56 descriptors) | Nov 8 12:19 → 13:18 (1.0 h) | 2 (1) | 94 / 0 | 4 s + 43 s | 35402725: strict output-path equality guard, `/scratch/alpine` vs `/gpfs/alpine1/scratch` (L383–387). |
| Collection (32 training systems) | 13:24 → 14:53 (1.5 h) | 2 (1) | 1,852 / 0 | 126 s + 800 s | 35407428: 120 s startup deadline expired while staging the runtime (L357–359). Custody of the success took 1,191 s (L337). |
| Tokenization and plans | 15:09 → 17:00 (1.85 h) | 2 (1) | 1,203 / 0 | 129 s + 272 s | 35415334: startup 120 s precondition expired (staging finished at 119.97 s) (L303–311). |
| LLM training (160 SFT + 119 DPO) | 17:00 → 17:47 (0.8 h) | 1 (0) | 3,616 / 904 | 904 s | None. |
| Blocked on storage decision (gap between commits) | Nov 8 17:47 → Nov 9 17:16 (23.5 h) | 0 | 0 | – | Mac had about 23.9 GiB free; Patrick was asked for space, then authorized CURC custody on Nov 9 (L271, L245). The gap also spans a night, so not all of it is attributable to the decision. |
| CPU campaign (5 policies × 24 = 120 cells) | Nov 9 17:33 → 21:31 (4.0 h) | 1 (0) | 11,456 / 0 | 5,728 s | No job failure. Three operational failures: coordinator 48 s metadata alarm (L242), collector02 timeout at 1,770 s after 2,475 s (L220), first audit rejected directory ordering (L208). |
| GPU campaign (3 LLM policies × 24 = 72 cells) | Nov 9 21:37 → Nov 10 05:19 (7.7 h, including about 2 h interruption 23:16–01:19) | 2 (1) | 7,812 / 1,953 | 245 s + 1,708 s | 35498218: SIGALRM at the 240 s startup deadline while verifying the CPU evidence tree on the GPU node, before any model load (L170). Also collector exit 1 (L162), launcher stopped on a 45 s SCP timeout before sbatch (L120), three failed local audit attempts (L73), monitor hit 1 MiB limit (L93). |
| Finalization (192 cell reports) | Nov 10 05:19 → 08:16 (3.0 h) | 1 (0) | 3,192 / 0 | 1,596 s | No job failure. Launcher failed locally on a 42 MB packet exceeding a 32 MiB reader (L48); monitor exceeded 1 MiB (L28). |
| Reporting and numerical review | Nov 10 08:16 → 08:51 (0.6 h) | 0 | 0 (local) | – | None. |
| Handoff | Nov 12 15:14 | 0 | 0 | – | – |

Scientific phase total: 44.5 calendar hours, about 15.7–18.6 active hours [C], 11 Slurm jobs with 11,555 s (3.21 h) elapsed.

Queue wait was not a bottleneck [V]: every observed job started within about 20 s to a few minutes of submission (for example CPU campaign SubmitTime 10:50:46, StartTime 10:51:05 local, `results/ace_llm_intervention_scientific_campaign_cpu_20261108/watch01/observation-00/stdout.log`). Exceptions were external: a maintenance reservation that held GPU10 from Nov 4 (L950–954) and the expired account `ucb736_asc1` with `MaxJobs=0` on Nov 7 (L994–998).

---

## (b) Resource envelope, consumption, remainder and approval scope

### Envelope [V]

- Ceiling for "this first execution cycle": **16 CPU-core-hours including GPU-host CPUs = 57,600 CPU-s, and 2 GPU-hours = 7,200 GPU-slice-s**, plus one conservative 3,600 CPU-s local preparation allowance counted inside the CPU ceiling (Ledger L525; agenda L187 calls it "a planning ceiling, not an allocation").
- Accepted closing accounts (`acceptance01/review.md` L33; `baseline.json`): closed **40,265 CPU-s / 4,458 GPU-slice-s**; CPU admission **43,865 / 57,600** with the allowance; **remaining 13,735 CPU-s (3.82 core-h) and 2,742 GPU-slice-s (0.76 slice-h)**; outstanding reservation 0; 49 terminal jobs, 100 journals.
- A GPU "slice" is one `a100_3g.20gb` MIG slice. A slice-hour is not a full A100 GPU-hour (Handoff L103).
- Admission rule: closed actual allocations + every outstanding **maximum cap** including the candidate + the single allowance must stay under the ceiling (Ledger L897). So remaining headroom is tested against job caps, not expected actuals.
- The CURC account renewal is far larger than the cycle envelope: INC1508105 grants `ucb736_asc2` up to **450,000 SUs through November 6, 2027**, but "does not enlarge the research cycle" (`ace_llm_intervention_account_renewal_2026-11-07.md` L3–5). [C/I] Billing × elapsed hours over the 49 jobs is about 77 billing-hours, so the self-imposed envelope, not the account, is the binding limit. The SU-per-billing-unit exchange was not verified.

### What the remainder can and cannot fund [C/I]

Actual cost of the seven successful scientific jobs was 27,598 CPU-s and 2,612 GPU-slice-s. So the remaining envelope covers about **half the CPU** of one like-for-like repeat of the scientific chain. On GPU, a 2,700 s training cap would still be admitted against 2,742 s, but after a 904 s actual the remainder would be about 1,838 s. That is only 130 s above the 1,708 s the GPU campaign actually used and far below its previous 2,880 s cap, with nothing left for a failed attempt (the last one cost 245 s). A retrain plus a fresh 24-system screen does not fit without a new resource decision.

### Scope of the "blanket approval for routine diagnosed operational recovery" [V]

Origin: `delivery_continuation_authorization_2026-10-24.md` L3–9. Patrick's words as recorded: "Proceed, stop asking for approval for every small thing. I grant you blanket approval. Continue". Recorded scope there: routine corrected replay "within the existing 150 CPU-core-hour delivery envelope"; "Routine diagnosed recovery within the approved scope need not generate another permission question." Explicit exclusions there: "new fitted studies, paid APIs, GPU work, installations/unrelated environment changes, favorable rescues, omission of failures, unrelated-job changes, or public release/submission", and "blanket authorization cannot supply credentials."

Carried into this cycle with a new ceiling:
- Ledger L521: "Patrick has activated the goal: follow the research agenda until there is a concrete answer… This authorizes development and bounded qualification toward that question."
- Ledger L525: "A new campaign beyond this envelope requires a measured projection and a resource decision."
- Ledger L948: "Existing blanket authority remains in force for routine work within the original limits."
- Ledger L965: "covers a reviewed diagnosis and bounded correction within the original ceiling. No installation, environment modification, scientific fallback or favorable replacement is authorized."
- Contract L89: "Routine infrastructure recovery must preserve failed evidence and stay within the original budget and scientific rules."
- Mount recovery doc L19: "authorizes one specific corrected attempt… It is not automatic retry authority, a replacement-world rule or permission to skip a claimed or ambiguous attempt."
- Handoff L105: "Routine diagnosed operational recovery has blanket approval within the agreed scope and ceiling. Do not manufacture another approval request for a small correction, and do not infer an unlimited new research allocation from the renewal or this handoff."

What was actually done under it (precedents) [V]: a new exclusive attempt after an independently accepted failure; startup deadline 120 → 300 s for one stage inside the same allocation (L305, L351); metadata reader bound 32 → 64/128 MiB for named roles (L50, L263, L281); a same-packet transfer continuation (L113–124); moving exhaustive evidence verification off GPU time (L166); raising the finalizer wall cap from 1,200 s to 4,200 s through an explicit recorded "clock adoption" (L63). Each recovery still needed failure custody, failure acceptance, a reviewed amendment, a new exclusive root and one submission.

### What needs a new decision from Patrick [V unless marked]

1. Any campaign beyond 57,600 CPU-s / 7,200 GPU-slice-s (Ledger L525; agenda L187 "The full confirmation campaign needs a fresh measured projection").
2. [I] A successor scientific experiment itself. The Handoff frames the immediate deliverable as "a short diagnosis and a justified next-experiment proposal" and says no successor was launched (L68). The documents do not say whether a successor that fits the remainder needs explicit approval. Conservative reading: Patrick sees the proposal first.
3. Paid APIs, installations, environment changes (Handoff L109; Oct 24 authorization L9). Neither runtime has `peft` or `accelerate` (Ledger L543); LoRA is torch-only.
4. New model weights. Agenda L175 says "download any separately approved weights".
5. Credentials. Password plus Duo for every new SSH master (`SKILL.md` L8–18).
6. Account or association changes and contact with CURC (L996, L1057).
7. Storage policy changes and any deletion of evidence (remote custody doc L3; Handoff L93).
8. Restarting the paused automations `ace-publication-progress` and `ace-foundation-synthesis-progress` (archive doc L15).
9. Manuscript submission. "Patrick and the coauthor retain final submission approval" (agenda L169).
10. Use of the reserved confirmation namespace requires a registered frozen design (protocol L19–22).
11. Live Causal Chambers access "requires a separate arrangement" (agenda L147).

---

## (c) Standing process rules

### Rules that bind any scientific run [V]

- **Freeze before scoring.** Declare system namespaces, methods, model and runtime pins, endpoints, failure treatment and resource limits before any draw. Run an artificial integration qualification first (Handoff L65; agenda L181).
- **Three disjoint system sets, reserved before generation.** `policy_train` 202611070000–0031, `development` 202611070100–0123, and reserved but unenumerated `confirmation:[202611071000,202611072000)` (protocol L18–22; agenda L179). The completed 24 systems "are not a fresh confirmation set for a revised recipe" (Handoff L65).
- **No outcome-driven changes.** No checkpoint selection on development error, no replacement systems, no tuning on the 24 outcomes, all arms and failed cells retained at the common cap (contract L27, L83–89; Handoff L62–64).
- **Exclusive evidence directories.** Every attempt gets a new root in three places: `results/<stage>/<attemptNN>` in the repo, `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-…-NN`, and `/scratch/alpine/paco0228/ACE/results/<stage>_<date>_<NN>`. "Never overwrite an existing evidence directory or rerun a scientific job to repair a collection failure" (Handoff L93).
- **Once-only submission.** One `sbatch` per reviewed, committed, pushed and remote-matched packet, with local intent, remote intent and result journals; no requeue; never duplicate a queued or running job; never retry an ambiguous submission (Handoff L107; contract L77).
- **Controller locks.** Before any mutation, hold both `ace-delivery-controller.lock` and `ace-foundation-controller.lock` (they are files under `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/`, verified to exist) and reconcile all terminal jobs (now 49), all journals (now 100), queue and accounting (Handoff L107).
- **Custody and backup.** After each terminal stage one collector authenticates the full remote inventory, writes one archive under `/projects/paco0228/ACE_evidence/<stage>/raw.tar` with before/after/readback checks, downloads a declared subset to two local copies and seals them. Independent acceptance precedes the next stage. A storage receipt never grants scientific acceptance (remote custody doc L19–29).
- **GPU discipline.** CPU-first preparation, smallest viable allocation, measured memory and runtime justification, two independent work signals, first GPU work within 240 s, phase timestamps, diagnose before repeating (Handoff L109; contract L73).
- **Independent review.** Source review, concrete packet review and actual-result acceptance by a distinct reviewer, zero required findings, at each gate (pattern throughout the Ledger).
- **Accounting.** Allocation, process CPU, wall time and reservations are different quantities and are never added (Handoff L103).
- **Unrelated jobs.** Never touch `mono_s2s`, `foundati*`, `latent-proxy-dpo`, `ecp1-*` or other queues (memory `reference_curc_ops.md`; Ledger L122, L905).
- **Publication obligations.** Alpine acknowledgement text is required in resulting publications (account renewal doc L19–21). Prose style rule: no semicolon chains or em dashes, use "intervention" consistently (Handoff L124).

### What would not bind a cheap local offline diagnostic [I, my reading]

A read-only analysis of saved artifacts on the Mac makes no remote mutation and no allocation. So the following have nothing to act on: once-only sbatch journals, dual controller locks, scheduler reconciliation, CURC custody archives, GPU work signals and the Slurm envelope. Local CPU is not charged as allocation; the 3,600 s allowance is a fixed conservative figure and "whole-cycle CPU remains unknown" (Handoff L103).

What still binds it [V]:
- Existing evidence directories and receipts are immutable. New outputs go to new exclusive locations.
- "Record a concrete protocol before any new model execution" and "Any new result needs its own source, inputs, costs and review scope" (Handoff L122).
- Use the repeated action "as the question to explain, not as permission to tune against the 24 development outcomes" (Handoff L62).
- No installs, no paid APIs (Handoff L109).
- Artificial states used for sensitivity checks must be "separately declared" (Handoff L62).

---

## (d) Dominant overhead and failure sources

### Why the 21 jobs and 1 submission failed [V for each event, C/I for the grouping]

| Cause class | Count | Jobs |
| --- | ---: | --- |
| The verification harness rejecting a valid environment (allocation string format, path alias, provenance or version guard, device map, account literal, memory-release gate) | 11 | 35202931, 35341862, 35343957, 35345616, 35348012, 35349776, 35351262, 35378508, 35385156, 35385876, 35402725 |
| Fixed startup deadline too short for runtime staging or full evidence verification | 3 | 35407428, 35415334, 35498218 |
| Runtime environment (imports, MKL threading inheritance) | 2 | 35252306, 35348887 |
| Numerical recipe or backend (gradient equivalence, FP32 projection, long-context OOM, FlashAttention with padding) | 4 | 35208932, 35209701, 35209904, 35377234 |
| Model interface (unconstrained generation produced malformed JSON) | 1 | 35203022 |
| Scheduler configuration (MIG slice not allowed on `gpu-normal`) | 1 | Nov 2 GPU01, no job ID |

Half of all failed submissions were the custody and provenance harness's own guards. Only 5 of 22 concerned the model or numerics, and those were informative (they produced constrained decoding, decoder recomputation and the unpadded singleton recipe). **All four job failures after the scientific freeze were operational. None was a scientific-method failure.**

### Recurring patterns beyond job failures [V]

- **Fixed bounds versus a monotonically growing evidence tree.** Each stage's custody includes all predecessors: 30,839 files (collection) → 31,608 (token) → 36,623 (training) → 72,374 (CPU campaign) → 99,039 (GPU campaign) → 107,284 files / 3.31 GB (final). Size or time bounds set from earlier stages then failed: 32 MiB reader exceeded four times (33.85 MB seal L281; 72.4 MB seal L263–269; 33.8 MB receipt L208; 42.1 MB packet L48), monitors' 1 MiB limit twice (L93, L28), 48 s metadata alarm (L242), 1,770 s archive deadline (L220), 240 s on-GPU verification (L170), 45 s SCP (L120). The finalizer launcher's readiness check finished at 1,174.8 s of a 1,200 s bound (L35).
- **Runtime staging variance.** Every job stages an 8.44 GB, 49,966-object runtime archive to node-local scratch. Observed staging took 33–84 s in successes but was only 53% complete at 120 s in collection01 (collection recovery doc L9–11).
- **In-job verification dominates short jobs.** Tokenization: 272 s job, about 17 s of work after a 245 s startup (`…scientific_tokenization_20261108/acceptance02/source-runtime.json`). Finalizer: 1,596 s job, 294.4 s of reporting; 870.6 s parent authentication, with the CPU and GPU inventories each checked three times (`acceptance01/review.md` L15). Training: 904 s job, 291.5 s of training wall; 542 s of reload, scoring and generation checks (`…/fit/summary.json`).
- **Near-limit margins on successes.** CPU campaign ran 5,728 s against a 6,000 s limit (RunTime 01:35:28, TimeLimit 01:40:00). Batch MaxRSS was within about 2–3 MiB of the 7,680 MiB request in collection, CPU campaign and finalizer (7,677.7 MiB; `review.md` L19; costs.json). The contract's own sensitivity for the GPU campaign (2,929 s) exceeded its 2,880 s cap (contract L65); actual was 1,708 s.
- **Human-gated waits.** 23.5 h for the storage decision; Nov 4–7 maintenance hold and account expiry; SSH master expiry needing password plus Duo (L667, L137).
- **Cost of one diagnosed recovery.** Failure to resubmission took 40 min (realization), 31 min (collection), 71 min (tokenization) and 4.4 h (GPU campaign, including an interruption and a failed transfer) [C from Ledger timestamps].

### How much was science and how much operations [C, I for classification]

- **Time, scientific phase (44.5 calendar h, about 18.6 active h):** Slurm job elapsed 3.21 h (17% of active). Scientific kernel inside those jobs about 2.37 h (13% of active, 5% of calendar). Recorded local launcher, collector and transfer wall-clock 6.37 h (34% of active; 20 recorded operations, 5,651 s of it handling failures). The remaining roughly 9 h was packet building, reviews, audits and recovery design. The storage wait was 53% of calendar time.
- **Time, whole cycle (181 calendar h, about 39 active h):** Slurm elapsed 4.58 h (12% of active). Scientific kernel about 6% of active hours and 1.3% of calendar time.
- **Compute:** the seven successful scientific jobs used 68.5% of CPU-s and 58.6% of GPU-slice-s. Qualification before any scientific draw used 27.4% and 35.9%. Roughly 52% of all allocated CPU-s and 42–54% of GPU-slice-s were scientific kernel.
- **Code (filename classification of 173 `*llm_intervention*.py` files, 49,473 lines in `/Users/pat/code/ACE/scripts/research/`):** about 6,560 lines (13%) implement the mechanism and evaluation (world, SCM, episode, target risk, policies, ranker, training engine, exposure, cost curves, reporting); about 24,700 lines (50%) are workers, services, supervisors, qualifiers and evidence validators; 18,222 lines (37%) are tests. Launchers and collectors stored under `results/` are extra.
- **Artifacts:** 127 `results/ace_llm_intervention_*` directories, 16.5 GB; 543 external `ACE-LLM-*` capsules, 29.9 GB; 214 review documents dated Nov 2–9; 45 guidance documents. The scientific payload is 242 report files, 4.4 MB.

---

## (e) Measured unit costs

All on Qwen3-0.6B, rank-8 q/v LoRA, bf16 base with fp32 adapters, one `a100_3g.20gb` MIG slice, 4 host CPUs. Each is a single run (n = 1 per stage).

### LLM inference at deployment [C from 1,698 `generation-NNNN.json` files under `…/scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services/<policy>/worker/`; session data from `…/gpu/controller/phase/terminal.json`]

| Service | Decisions | Prompt tokens min / median / p90 / max | Generation time median / mean | Session wall per decision | Wall per 24-step episode |
| --- | ---: | --- | --- | ---: | ---: |
| frozen | 576 | 1,990 / 3,066 / 3,939 / 4,160 | 0.426 / 0.582 s | 0.78 s | 18.7 s |
| SFT | 546 | 1,990 / 2,933 / 3,679 / 3,954 | 0.538 / 0.596 s | 0.74 s | 16.8 s |
| DPO | 576 | 1,990 / 2,985 / 3,770 / 3,985 | 0.712 / 0.851 s | 1.19 s | 28.6 s |

- Generated tokens: 8 for `observe_all`, 9 for `set_*`, 6 for stop; 14,631 in total (matches `review.md` of GPU acceptance).
- Correlation between prompt tokens and generation time is 0.07–0.09. Latency is not explained by prompt length in this range. The DPO service was about 46% slower per decision; cause not recorded.
- Job level: 1,708 s for 1,698 decisions; first GPU work at 103.9 s; summed CUDA 1,147.9 s; service startup 14–19 s each.
- GPU memory: peak framework allocated 2.44 GB, reserved 3.26 GB; owned process 3,356 MiB. Host batch MaxRSS 10.25 GiB of 15 GiB.
- Context limit 8,192 tokens with 32 new tokens. Artificial stress maximum was 4,744 prompt tokens (Ledger L431).

### LLM training [C from 279 `step-NNNNN.complete.json` and `summary.json` under `…/scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/`]

- SFT update: median 0.595 s, mean 0.625 s; input 2,007–4,205 tokens (median 3,014).
- DPO update (two scored rows): median 0.994 s, mean 1.072 s; 4,025–7,342 input tokens.
- Sum of step wall 227.5 s; training wall 291.5 s for 279 updates (1.045 s per update including receipts).
- Whole job 904 s: about 35 s staging, first GPU work at 51.5 s, 291.5 s training, 542 s checkpoint reload/score/generation checks (3 base loads, 2 × 1,184 legal-command scores, 322 generations).
- GPU memory: peak allocated 2.66 GB, reserved 3.28 GB, owned process 3,404 MiB. This depends on decoder recomputation (11,144 recompute calls). Without it one SFT update at 4,341 tokens peaked at 16.57 GB and a 7,343-token forward exhausted the 19.62 GiB device (Ledger L865).
- Training prompts: 1,998–4,199 tokens; 481,651 stored tokens over 160 records; commands 6, 8 or 9 tokens (`…/scientific_tokenization_20261108/acceptance02/exposure-report.json`).
- Side observation [C]: DPO per-step loss was 0.705 mean in the first quarter of steps and 0.679 in the last quarter (ln 2 = 0.693). SFT loss fell from 0.482 to 0.214. Interpretation belongs to the training agents.

### Numerical policies [V totals from `…/scientific_campaign_cpu_20261108/acceptance01/costs.json`; C per-policy from `cpu-cell-counts.json` and 5-minute watcher snapshots in `watch01/`]

- Job: 5,728 s on 2 CPUs = 11,456 allocated CPU-s for 120 episodes and 2,673 decisions (one process per decision). Worker child CPU 2,701 s.
- Per episode: 47.7 s wall, **95.5 allocated CPU-s, 22.5 process CPU-s**. Per decision: about 2.1 s wall, 1.0 process CPU-s.
- Approximate wall per episode by policy: uniform random 33 s, menu coverage 28 s, local information 39 s, target risk 41 s (28–56), learned ranker 82 s. Per decision about 1.4–1.75 s for the four controls and 3.4 s for the ranker. These include interpreter startup per decision. In-process planning cost was never measured.
- So as implemented, an LLM episode on GPU (17–29 s) took less wall time than a numerical episode on CPU (28–82 s). That is an implementation artefact, not a compute-efficiency finding.

### Other stages [V]

- Collection: 800 s on 2 CPUs for 32 training systems, 9,145 responses, 10,240 cost units, 160 records, 3,584 pairs (25 s wall per training system).
- Finalizer: 294.4 s of reporting for 192 cells (12.3 s per system); 1,596 s job.
- Custody collectors: 451–2,765 s local wall each, about 25 ms per file at the end of the chain [C].

### Scheduler facts recorded live on 2026-11-08 17:00:51 UTC [V, `…/scientific_training_gpu_20261108/launch01/readiness_transport.json`]

- QoS `gpu-testing`: **MaxWall 01:00:00, MaxTRESPU `gres/gpu:a100_3g.20gb=1`**.
- The 20 GB MIG slice is rejected under `gpu-normal` (Slurm Error 7; Ledger L618–620). The QoS correction receipt cites CURC documentation that the slice is available only under `gpu-testing` "for workflow qualification".
- Partition `aa100`: MaxTime 7 days, MaxMemPerCPU 3,840 MiB, GPU types a100-40gb, a100_3g.20gb (6 slices), a100_80gb. Billing weights: CPU 0.5, a100_3g.20gb 54.3, a100-40gb 108.6, a100_80gb 130.4.
- Association QoS list for `ucb736_asc2`: compile, cpu-long, cpu-normal, cpu-testing, dtn, gpu-long, gpu-normal, gpu-testing, interactive, mem-long, mem-normal.

---

## (f) Recorded priorities, deadlines, venues, coauthor, API and model constraints

- **Submission condition** [V]: "Patrick has made this LLM mechanism and its demonstrated utility a condition of submission" (agenda L13). "The LLM must contribute to the decision rather than only explain a numerical optimizer's choice" (Handoff L13). The paper "is not ready for submission" (Handoff L17).
- **Negative results** [V]: "A negative result in the new research must narrow the claim or redirect the method, not trigger a favorable replacement benchmark" (agenda L13). "A narrow conditional gain can support a narrow paper, but the LLM cannot remain only a decorative element" (agenda L169).
- **Five submission requirements** [V]: agenda L161–167 (reproducible LLM policy; independently evaluated benefit over a strong numerical policy on fresh systems; ablations for pretraining, ACE training, causal structure and extra computation; a concrete experimentalist task with transfer; preserved failures and independent review).
- **Next priority** [V]: diagnose before training again; "Scaling the current DPO recipe or adding compute merely because budget remains is not the next step" (Handoff L62–68).
- **Deadline and venue** [V by search]: **no deadline and no target venue is recorded for the broader LLM paper** in the Handoff, agenda, Ledger or results report. For the archived delivery paper, TMLR was the recommended venue on November 1 and "AISTATS2027's main paper deadline was October 6, 2026 and is closed" (`ace_submission_audit_2026-11-01.md` L5, L46). The manuscript folder is still named `paper/aistats_ace_2027/`. Older folders `paper/iclr_ace_2027/` and `paper/neurips_ace_2026/` exist.
- **Coauthor** [V]: one coauthor, Patrick's advisor Alvaro Velasquez, final author, agreement to coauthorship "confirmed by Patrick"; "Approval of the final manuscript version by both authors: unconfirmed" (`delivery_submission_form_draft_2026-11-01.md` L40–61). "Patrick and the coauthor retain final submission approval" (agenda L169). Declared for the delivery paper: funding none, competing interests none.
- **Paid APIs and installs** [V]: "No paid APIs, installations or unrelated environment changes are part of this handoff" (Handoff L109). Same exclusions in Ledger L517 and the Oct 24 authorization L9.
- **Model choice** [V]: "Begin with an open-weight model whose license, weights, tokenizer and training implementation can be pinned. Choose a small viable model after profiling, rather than assuming a larger model is necessary" (agenda L85). Executed model: Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0 per its model card (Ledger L545). New weights need separate approval (agenda L175).
- **Compute culture** [V]: smallest viable GPU, no multi-GPU without measured need, no GPU idling during CPU work (agenda L187–191).
- **Acknowledgement** [V]: Alpine acknowledgement required (account renewal doc L19–21).
- **Related evidence that must stay visible** [V]: Mission-1 adverse result (LLM/random error ratio 1.0412, improvement in 2 of 11 seeds) and the delivery study's unfavourable regimes (agenda L17–19).

---

## (g) Projections

Assumptions for all three: same harness, same eight policies, same budget (8 initial + 24 cost units), Qwen3-0.6B on one 20 GB MIG slice, sequential single-job stages, unit costs from one run each, no change to the per-decision process model for numerical policies. Failure overhead from this cycle's scientific phase was +5.9% CPU and +9.4% GPU. Calendar estimates assume the observed cadence (about 6.5 commits per active hour by agents, 6–14 hour sessions) and Patrick available for Duo once per session.

Scaling model [C]: CPU policies 477 allocated CPU-s per system (239 s wall on 2 CPUs); LLM policies about 66 GPU-s per system for three arms plus about 120 s startup per job, with 4 host CPUs; finalizer about 2 × (1,300 s verification + 12.3 s per system) and growing with tree size; training chain (realization + collection + tokenization + training) 6,118 CPU-s and 904 GPU-s.

### 1. Offline diagnosis on saved artifacts

- **Compute:** local only, minutes of CPU. Zero CURC.
- **Calendar:** 0.5–1 day including a written protocol and write-up.
- **Why this is cheap [V]:** the needed artifacts are already on the Mac. Training: `plan.json` (tokenized prompts), 279 per-step loss and gradient records, `reference-scores.json`, `final-before-reload.json` and `*-reload-scores.json` (1,184 legal-command scores for all 160 training states), 322 saved generations, `initial/sft/final.adapter.pt` (4.6 MB each), all under `…/scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/`. Collection: 160 records, 3,584 pairs (687 strict, 2,897 tied, 119 selected). Deployment: every one of the 1,698 LLM requests with full prompt token IDs and completion, plus `reporting01/data01/*.csv` (192 cells, 4,540 curve points, action counts).
- **If new model scoring on declared artificial states is added:** [I] about 0.12–0.4 s per command forward on the slice. A grid of 300 states × 9 commands × 3 policies is about 8,100 forwards, roughly 1,000–3,200 GPU-s as singletons, which is most or all of the 2,742 s remainder. On CURC under the full process this is one CPU preparation job and one GPU job, 1–2 days. Local execution on the Mac is unmeasured and would not be the frozen runtime (local torch and transformers versions differ, Ledger L543).

### 2. Short GPU retraining of a revised recipe plus a 24 to 48 system screen

- **Compute, 24 systems with new training collection:** about 27,600 CPU-s (7.7 core-h) and 2,600 GPU-slice-s (0.73 slice-h) if every job succeeds; about 8.4 core-h and 0.8 slice-h with failure overhead.
- **Compute, 48 systems:** about 46,000–48,600 CPU-s (12.8–13.5 core-h) and about 4,300 GPU-slice-s (1.2 slice-h); about 14–15 core-h and 1.3 slice-h with overhead. The GPU campaign would need two jobs because of the 1 h QoS wall.
- **Longer training:** each extra SFT update costs about 0.6–0.9 s and each DPO update about 1.1–1.3 s. Ten times the current exposure (about 2,800 updates) is about 2,400–2,900 s, which no longer fits one `gpu-testing` job with the 542 s checkpoint checks. Resumed optimization across jobs is explicitly unqualified (Ledger L889).
- **Against the remainder (13,735 CPU-s, 2,742 GPU-slice-s):** does not fit. It needs a new envelope of roughly one more cycle (8–15 core-h, 0.8–1.3 slice-h).
- **Calendar:** 3–5 days. Day 1 freeze and artificial qualification of changed code (the artificial integration took about 6 h and 11 jobs this cycle). Day 2 collection, tokenization, training (5.5 h observed). Day 3 CPU and GPU campaigns with custody (4.0 h and 7.7 h observed). Day 4 finalization, report and numerical review (3.6 h observed). Add slack for human-gated events; one cost 23.5 h this cycle.

### 3. 100 to 150 system prospective confirmation (frozen policies, no retraining)

- **Compute, 100 systems:** about 80,000–85,000 CPU-s (22–24 core-h) and about 6,900–7,100 GPU-slice-s (1.9–2.0 slice-h).
- **Compute, 150 systems:** about 118,000–126,000 CPU-s (33–35 core-h) and about 10,000–10,400 GPU-slice-s (2.8–2.9 slice-h).
- Add 6–10% for failed attempts. Each extra LLM arm adds about 22 GPU-s and 88 host CPU-s per system. Each extra numerical arm adds about 60–80 CPU-s per system (about 160 for a ranker-type arm).
- This is 1.4–2.2 times the whole first-cycle CPU ceiling and 1.0–1.5 times its GPU ceiling. It is a small fraction of the 450,000 SU account [I].
- **Structural limits that force a redesign [V facts, I consequences]:** `gpu-testing` MaxWall 1 h and one MIG slice per user means 3–4 serial GPU jobs of at most about 40–45 systems each, or a full A100 under `gpu-normal` (twice the billing weight, and not the "smallest viable" type). The CPU campaign would be 6.6–9.9 h in one job, or four shards. The evidence tree would grow to roughly 400,000 files, so the collector's 1,770 s archive bound and the finalizer's three full-tree verification passes would fail unless custody is sharded or bounds are amended first.
- **Calendar:** 6–10 days. 1–2 days registration, sample-size calculation, freeze and qualification of the sharding changes; 3–5 days execution in about four shards, each a working session; 1–2 days finalization, report and independent review.

---

## Contradictions and state changes found

1. **Account.** `SKILL.md` L95 still says to use `--account=ucb736_asc1`. The newest documents adopt `ucb736_asc2`; `asc1` showed `MaxJobs=0` on Nov 7 (Handoff L105; account renewal doc). Newest wins. The skill also points to `$HOME/.codex/skills/curc-access/scripts/curc-access.sh`; an identical copy exists under `~/.claude/skills/curc-access/scripts/`.
2. **QoS intent versus practice.** Ledger L620 (Nov 2) says "Production scientific studies will use a supported production QOS." The Nov 8 contract (L47) put both scientific GPU stages on `gpu-testing`, and they ran there. No later document revisits this.
3. **Agenda versus execution.** The agenda proposed 16 initial + up to 64 interventions, seven policies and four regimes of six systems. Executed: 8 initial + 24 cost units, eight policies, 24 systems as 3 graphs × 2 coefficient regimes × 2 noise levels × 2 cost menus. The Handoff (L19) says the agenda's counts are superseded.
4. **Finalizer cap.** Contract says 1,200 s. Executed cap was 4,200 s after a recorded clock adoption. Actual 1,596 s, so the contract value would have failed.
5. **Local disk.** Ledger records 12.66 GB available on Nov 10 07:03 UTC (L30). A read-only `df` on Nov 12 shows about 197 GiB available on the Mac data volume, and `/Volumes/Aux` is mounted. The Handoff asks for a fresh capacity check before transfers.
6. **SSH socket.** `/tmp/defab-curc.sock` exists with mtime Nov 9 18:11 local. Liveness was not tested.
7. **Outside my scope but verified** [C from `reporting01/data01/actions.csv`]: the learned numerical ranker also collapsed to two actions (`set_x_pos` 435, `set_x_neg` 141 of 576). The Handoff's behavioural summary mentions only the three LLM arms.

## Open questions

- Does Patrick have a target venue or date for the broader paper? Nothing is recorded.
- Does a successor experiment inside the remaining 13,735 CPU-s / 2,742 GPU-s need explicit approval, and what new ceiling would Patrick set for a retrain-plus-screen (about 8–15 core-h) or a confirmation (about 25–40 core-h, 2–3 slice-h)?
- Is continued scientific use of `gpu-testing` acceptable to CURC, or should larger runs move to a full A100 under `gpu-normal`?
- May the custody design be changed (sharded archives, no repeated full-history verification inside jobs, in-process numerical policies)? These are operational, but each new piece of harness code has so far produced new guard failures.
- In-process latency of target-risk planning is unmeasured, so the agenda's compute break-even test (agenda L53) cannot be evaluated from current data.
- Scratch has a 90-day purge policy (remote custody doc L21). Nov 2 scratch originals would purge around the end of January 2027; the `/projects` archives are the retention copies and their current readability was not rechecked.
