# Supplemental replay failure and runtime recovery proposal

**Current state after October 8, 19:10 UTC:** Patrick approved the corrected
second preparation. Job33560677 installed the isolated environment and passed
pipcheck, then failed module provenance qualification. No successful runtime
receipt or numerical replay exists. The single further replay remains approved
and unused; no further installation is authorized. See the latest section below.
Earlier unapproved-proposal wording records the historical state.

## Latest corrected preparation33560677

Patrick's direct approval covers one corrected1CPU/3GiB/30minute preparation
and the existing single further1CPU/3GiB/15minute inference-only replay.
Reservation88.368333core-hours is within150. New06 freeze76b601ef preserves
reviewed proposal05 source; exclusive journals submitted33560677 exactly once.

The node resolved both base spellings to the same approved Python3.11 ELF,
with expected bytes. Venv/bootstrap/exact-six installation/pipcheck completed0.
The captured qualifier then rejected fileless module
`torch._C._dynamo.autograd_compiler`, after its unique metadata/actual import
version/origin checks. No qualification or full environment inventory receipt
was emitted. Do not treat completed installation as qualified numerical replay.

All15 terminal objects match independent remote hashes in exclusive external
custody delivery-runtime-preparation-job33560677-failed-20261008-01. Compact
repository disposition failed33560677/failure_disposition.json SHA
6de8dc6ccfe502b05cb9262a25e9fd5890c23610b02d676ef48e094f8ad52108.
Actual resources matched freeze; FAILED1:0,169allocatedseconds, TotalCPU65.454
seconds and batchMaxRSS642936KiB. Known allocated original/pilots/failedreplay/
preparations=119284seconds=33.13444444444445core-hours; not whole-sprintCPU.

Installed environment remains at
`/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02`.
No further installs, fits, updates, responses or replay submissions occurred.
Unused replay requires a NEW independently pinned source/runtime/input/resource
freeze, current imported dependency/provenance validation and successful runtime
qualification before any saved-checkpoint inference. Preserve both failed
preparations and failed original replay; do not weaken guards or blind retry.

Static upstream evidence explains a naming distinction to investigate in that
new verifier: PyTorch2.9.1 defines a module named autograd_compiler but attaches it
under compiled_autograd. See the authoritative
[binding initialization](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/dynamo/init.cpp)
and [module definition](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/dynamo/python_compiled_autograd.cpp).
The installed compiled_autograd.pyi is consistent with that attribute name.
This source evidence is not a successful live object-identity check; any derived
alias handling must authenticate the actual native parent/backing bytes and
reject substitutions. The older wrapper04 prototype also needs the independent
review's live-import/search-path checks; it is unsubmitted and unqualified.

## Validated state at October 8, 16:10 UTC and subsequent diagnostics

CURC transport was restored. The ACE controller acquired the exclusive external
project lock, reconciled the empty replay queue/journals and authenticated all
3,890 remote package members against independently pinned manifest f5415e24.
Scratch stat/df and dedicated write/read/cleanup passed. The three unchanged
launch03 inputs were transferred and independently authenticated. Exactly one
sbatch call submitted job **33539781** at 16:06 UTC. No original study was rerun.

Job33539781 FAILED1:0 after182 allocated seconds. Actual resources matched the
freeze: accountucb736_asc1, acpu/cpu-normal, one CPU, 3 GiB, fifteen minutes,
zero GPU. Its child stopped at the actual imported-dependency gate, before
checkpoint loading, scientific inference or performance reporting. There is no
successful replay receipt. All nine remote terminal objects match exclusive
local failed custody at:

`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-replay-job33539781-failed-20261008-01`.

The disposition receipt is
`results/delivery_prospective_replay_preparation_20261008/failed33539781/failure_disposition.json`,
SHA1b51319a72736476f6a713b91a0cb9baba3e4c296fcf8452baed67df827a7513.
External resume01/02 retain checks, journals, raw Slurm evidence and diagnostics.
The terminal scontrol lookup was already purged; its nonzero result is preserved.
The first running scontrol and final sacct independently bind the standalone
job identity. It is separate from the original forty-four-identity scheduler map.

## Runtime diagnosis and historical limit

The designated interpreter is Python3.10.19. NumPy distribution selection reports
2.2.6, but the actually imported module reports2.2.5 from that same environment's
site-packages. Both numpy-2.2.6.dist-info and numpy-2.2.5.dist-info are present.
The replay correctly rejects this mismatch. Four NumPy source/metadata snapshots
are separately hashed in private custody. Metadata-only preflight was insufficient.

Python3.10 is a separate pending failure: the unchanged supplemental verifier uses
stdlib tomllib, requiring Python>=3.11. The failed child did not reach that stage.
Read-only inspection of the existing ACE neuronbench_py311 environment shows
missing torch/pandas/SymPy/PyYAML and differing NumPy/SciPy metadata. Neither of
these inspected ACE environments satisfies the supplemental runtime contract.
No unrelated environment was changed or imported.

Original qualification/audit dependency reporting uses importlib.metadata.
Current module diagnostics do not establish the original run's imported module
bytes or when the environment diverged. Preserve original full acceptance and
all checkpoints unchanged. Do not relabel its recorded metadata as proof of
historical imported versions, or claim supplemental exact-runtime qualification.

## Supported development fixes, separate from failed frozen input

Development replay() now rejects Python<3.11 at function entry before resolving
its package root or reading replay artifacts. Importing authenticated verifier
source precedes that function-level check. Its version/origin rejection is
unchanged; the error now includes required/metadata/imported versions and paths.
Two new fabricated guard methods pass from unrelatedcwd. A distinct bounded
read-only failure/implementation review found no required remaining fixes.
Neither synthetic checks nor this source change qualify real target inference.

The frozen failed candidate, launch03 inputs, original worker guards, original
receipts and manuscript are unchanged. Any future adapter change needs a NEW
source/plan/manifest lineage; never overwrite the failed package or retry03.
Prior completed component tests/reviews were not repeated.

## Cost disposition

Count the one failed allocation once:182 CPU-seconds =0.050555556 allocated
core-hours. Slurm parent TotalCPU reports47.640 seconds; batch47.639 and
extern0.001 are overlapping identities within that total, not extra allocations.
Batch MaxRSS is826396KiB; wrapper sampled tree peak is674197504bytes and elapsed
174.705847324seconds. Sampling and scheduler memory scopes differ.

Original118680allocated seconds + actual126/119pilots +182failed replay =
119107seconds, or33.085277778allocated core-hours. Fit CPU/wall and confirmation
elapsed scopes overlap and are not added. Whole-sprint processCPU remains unknown.
The original86.868333 reservation plus the failed replay's existing0.25 remains
87.118333; do not add its actual cost again to that reservation.

Preflight01's manifest.sha256 sidecar omission, a system-Python3.6 monitor error,
and the purged-terminal scontrol prototype were metadata/control failures only.
They are preserved, with no additional allocation, fitted model or response.

## Historical proposal before Patrick's bounded recovery authorization

Recommended recovery is a NEW isolated ACE-only Python3.11 environment with:
torch2.9.1, NumPy2.2.6, SciPy1.15.3, pandas2.3.3, SymPy1.14.0 and PyYAML6.0.3.
It avoids repairing shared or historically used environments. Patrick must
explicitly authorize an exception to the existing no-install instruction and
one additional replay allocation. Alternatively, Patrick can identify an
existing matching interpreter; it must pass actual module/origin checks.

Proposed caps, to freeze before execution if approved:

- CPU-only environment preparation: one CPU, 3 GiB, thirty minutes (0.5 reserved
  core-hour). This is a bounded installation cap, not a measured completion ETA;
  a timeout/failure is retained and does not authorize retries or extra resources.
- One NEW supplemental inference-only replay: one CPU, 3 GiB, fifteen minutes
  (0.25 reserved core-hour), ucb736_asc1/acpu/cpu-normal, no GPU/fits/updates/new
  responses. Prior plus both proposed caps is87.868333 within the150 ceiling.

Before any new submission, independently authenticate Python/version/origins and
all six imported libraries, reject ambiguous duplicate metadata, bind the new
adapter/contract/predecessor/manifest/input/runtime/resource bytes, recheck scratch
and exclusive journals, and submit at most once into a NEW exclusive output.
No scheduler or environment changes are currently approved by this proposal.
If installation or runtime cannot satisfy the unchanged pins within the approved
caps, preserve failure and obtain a new concrete decision. Do not relax pins or
refit accepted artifacts.

After successful full supervised replay and final custody/accounting, run BOTH
original claims exporter and descriptive reporter; source-bind new outputs,
sync/compile the SAME manuscript and finish integrated scientific/proof/prose/
tables/page visuals/accounting/anonymity/redistribution/human approval. Those
scientific and submission gates remain open. No prospective performance claim
was generated in this resumption. No schedule or public artifact was changed.

## Current authorized recovery and failed preparation — October 8, 18:49 UTC

The coordinator verified Patrick's direct approval in thread
01a11c39-0c89-7152-8a56-432adf15402d, human turn
01a11c8f-d5ce-7172-a1ec-f4f1c136129c: “You may message the seven eligible project
agents and carry out the bounded recovery steps I approved in Jasper’s chat.”
The delivered ACE scope permits a bounded read-only search, one isolated CPU
preparation capped at 1 CPU/3 GiB/30 minutes, and one further inference-only
replay capped at 1 CPU/3 GiB/15 minutes. It does not permit retries after failure,
shared repairs, GPU, fits, responses, schedules or publication.

The search checked metadata in ten known conda prefixes and narrow Python3.11/
PyTorch2.9.1 module listings. None matched Python>=3.11 and all six pins; target
libraries were not imported in those shared prefixes. This is not a universal
CURC environment search. An existing ACE Python3.11.16 base and bundled seed
wheels were pinned. Source prototypes01–03 were preserved and never submitted.
The final preparation04 static review had zero required fixes.

An exclusive controller/output claim, fresh scratch stat/free-space/write-read-
cleanup, empty ACE queue and independently authenticated remote input/base/seed
bytes preceded exactly one sbatch. Job33556219 was submitted at18:45:18UTC with
1 CPU/3 GiB/30 minutes, ucb736_asc1/acpu/cpu-normal and zero GPU. Freeze SHA:
142bea17f79b8580302b3a1f10fa4fd186f3a6bed16562bd9dea31fc197aeb1f.
It FAILED1:0 at “base interpreter differs” before prefix creation or any command:
commands=[], environment absent, no installer, qualification or further replay.
All eleven terminal objects match exclusive local failed custody at
`/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-runtime-preparation-job33556219-failed-20261008-01`.
Disposition SHA93d41f48080f3c654274925f96047634ba565555f46949968af42aa7c46ea0dc.

The guard compared a node-resolved path to an unre-resolved login spelling.
Login bytes/path still match, but actual compute-node values were not recorded.
An alias mismatch is plausible; a byte mismatch cannot be excluded. Do not
assert a verified root cause or matching compute-node runtime.

Proposal05 canonicalizes BOTH paths on the execution node, records observed
paths/hashes, retains exact frozen binary plus running ELF hashes, and requires
Python3.11 before prefix creation. Qualification retains those guards. One NEW
focused diagnostic fixture passes; the distinct narrow review has zero required
fixes. This source is UNAUTHORIZED/UNSUBMITTED, not runtime qualification.

A NEW second preparation needs Patrick's decision. The concrete proposal uses
new output/environment02 and the same 1 CPU/3 GiB/30 minute cap, account and
unchanged six pins. An additional0.5 reservation yields88.368333 within150;
the already authorized further fifteen-minute replay remains unused. Proposal
SHA6c6c1b92a5b126cf492add23bd20cb3d9f3a27b01c7d75f9f47c73c1673d48cb.
Do not reinterpret a single-attempt cap as a cost-only limit or submit without
this new decision. Failure or runtime mismatch remains terminal.

Job33556219 adds eight allocated CPU-seconds, .002222222 core-hours. Parent
SlurmTotalCPU0.111 seconds, batchMaxRSS19312KiB, worker childCPU0.007215 seconds
and elapsed0.020526814 seconds have distinct scopes. Known original plus actual
pilots plus failed replay plus preparation allocation is119115 seconds,
33.0875 core-hours, not whole-sprint processCPU. Current87.868333 reservation
already includes the failed preparation and unused replay; actual costs are
not added to that reservation.

Private successor candidate03 passes byte/digest verification:3893 files,
5068 bindings, manifestb30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9,
source contract269bbc3f0f8d614177a3e181beb3ebd6e2eaa3fcc40026dba20b0048b998fe65.
It binds the failed-input predecessor manifest/source contract and updated
replay interface while preserving original models/responses. Build02 rejected
five changed mutable writing sources; build03 explicitly uses authenticated
inherited candidate01 bytes, with a source-resolution receipt and retained
partial build02. It is not current-writing or numerical-runtime qualification;
nothing transferred remotely or replayed, and no B reports were produced.

Evidence: results/delivery_runtime_recovery_20261008 and
reviews/delivery_runtime_preparation_review_2026-10-08.md. SAME manuscript,
accepted studies, failed33539781, schedules and public artifacts are unchanged.
Both reports, final source/report/writing lineage, integrated science/proofs/
prose/tables/page visuals/accounting/anonymity and human approval remain gated.
