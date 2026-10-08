# Supplemental replay failure and runtime recovery proposal

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

## Concrete decision required; nothing below is authorized or submitted

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
