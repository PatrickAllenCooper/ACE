# Bounded runtime preparation and failure review — October 8, 2026

Reviewer: Volta, agent01a11cbb-16ec-7072-a104-37b16455f398. Static local review;
the reviewer executed no installer, worker, scientific fixture, checkpoint or job.

## Preparation source review

Preparation01 had six required findings concerning fixed pins/caps, private
prefix/output custody, installer isolation, child-group termination/reaping,
captured-source execution and dependency/import provenance. Preparation02's four
findings covered Torch synthetic identity, output descriptor custody, complete
PYTHON-variable stripping and separate cleanup failures. Preparation03 retained
two closures: descriptor-bound freeze reads and explicit fileless-module
provenance. Preparation04 had one remaining bootstrap issue: captured code was
not canonical __main__. All were fixed in successor sources; earlier versions
remain in private custody. Final scoped static review: zero required fixes.

Focused mocked/source checks and a canonical-main sentinel support those guards
only. They do not establish compute-node interpreter identity or an installed
runtime. Prior scientific/component reviews and numerical replays were not repeated.

## Actual failure

All eleven remote terminal files match exclusive local custody. The running
allocation receipt and Slurm agree on 1 CPU, 3 GiB, 30 minutes, account
ucb736_asc1, acpu/cpu-normal and zero GPU. Job33556219 FAILED1:0 after eight
allocated seconds; parent TotalCPU was 0.111 seconds. Its receipt has commands=[]
and fails at "base interpreter differs", before prefix creation or installation.
No runtime qualification or additional replay occurred.

The old comparison resolves only its left operand and short-circuits before
hashing if paths differ. Compute-node observed values were not recorded. A path
alias mismatch is plausible, not established; login matches do not prove node
identity or exclude a byte mismatch.

## Proposal05, unsubmitted and unauthorized

The proposed guard strictly canonicalizes both named paths on the execution
node, records observed paths/hashes before rejection, and requires the frozen
base hash, executing ELF hash and Python3.11 before prefix creation. Qualification
retains both path and byte checks. One new focused alias/byte/ELF fixture passes.
Final narrow source/failure review: zero required fixes. This is not execution
qualification or permission to retry.

Another 1 CPU/3 GiB/30 minute preparation adds 0.5 reserved core-hours, yielding
88.368333 including the still-unused authorized replay allowance. A new human
decision is required. Do not relax pins, repair shared environments or refit.

Private successor candidate03 has byte verification for 3893 files/5068 bindings,
preserved failed-input lineage and updated replay-interface source. Its first
build rejected five subsequently changed working writing/source artifacts;
the next build explicitly uses authenticated predecessor bytes, with a separate
resolution receipt and retained failed partial build. It does not qualify
numerical replay, current writing, anonymity or public/submission readiness.

Evidence: results/delivery_runtime_recovery_20261008 and the current runtime
recovery guidance. Original accepted studies and failed33539781 remain intact.
