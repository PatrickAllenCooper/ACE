# Replay07 failure and proposed mount-alias correction — October 23, 2026

## Actual failure and custody

The single approved replay was submitted once as **34821094** and ended
`FAILED/1:0`. It used one CPU, 3 GiB, and 548 allocated CPU-seconds. The
supervisor recorded 520.554279 seconds elapsed and complete child cleanup.
Slurm TotalCPU was 141.010 seconds; sampled process-tree peak RSS was
556,085,248 bytes and Slurm batch MaxRSS was 608,872 KiB. These measurements
have different scopes and must not be added.

The qualifier rejected `exact VF source pin required` before writing a runtime
inventory or qualification receipt and before any checkpoint loading. There
was no fitting, optimizer update, new response, accepted supplemental replay
or Stage B reporting. Original study acceptance remains unchanged.

All 17 remote output files, 378,925 bytes, match the independently obtained
remote inventory in exclusive external custody
`delivery-vf-replay-terminal-20261023-07`. The repository copy is
`results/delivery_vf_replay_20261023/terminal07`. It includes original inputs,
submission journals, logs, terminal supervisor record and scheduler evidence.
The failed packet and all predecessors remain unchanged. Approval for this
attempt is consumed; another allocation requires a new explicit decision.

## Cause and bounded correction

The login node leaves the approved environment path under `/scratch/alpine`.
A read-only `readlink -f` query on the actual compute node, `c3cpu-e2-u1`,
resolves that same path under `/gpfs/alpine1/scratch`. The frozen VF checker
compared the supplied pin-map keys directly with a resolved path. Thus it
rejected the directory alias before comparing the source digest. An artificial
fixture reproduces this precise rejection using the unchanged failed07 source.

The new derived checker resolves pin-map paths to canonical files within the
approved resolved environment. It still requires the exact source and both
native backing files, their independent SHA256 pins, exact live Python code,
globals/builtins, object metadata and fresh-import reference continuity.
Duplicate canonical keys, symlink leaves, external files, altered code, wrong
digests and false live-code origins reject. The live code filename may use the
same directory alias only if it resolves to the authenticated source. Cheap
pin-shape/origin checks now precede expensive dependency imports and hashing.

This correction does not establish actual target execution, native execution
attestation, arbitrary same-process tamper resistance or scientific results.
The shared filesystem and same-user/root trust boundary are unchanged.

## Proposed resource decision

No new job is authorized or submitted. The concrete proposal is one
inference-only attempt at **1 CPU, 3 GiB, 30 minutes**,
`ucb736_asc1/acpu/cpu-normal`, zero accelerators, no requeue, no installs,
training, updates or new responses. Proposed output08 is a new exclusive
directory; it must not overwrite output07 or its source.

The longer cap is motivated by actual runtime verification, not model demand.
Attempt07 took 520.554279 supervisor seconds before the complete environment
inventory. A subsequent metadata-only count found 28,576 regular files and
one symlink, totaling 7,453,922,966 regular-file bytes. Qualification still
requires the complete inventory and its revalidation at the first checkpoint.
The original 20.43-second scientific audit therefore does not bound this
broader runtime gate. Thirty minutes is a bounded proposed allowance, not a
measured completion guarantee. The proposed supervisor deadline is 1,750
seconds with the existing cleanup logic; Slurm's hard cap is 1,800 seconds.
Memory stays at 3 GiB because observed peaks were below it.

The existing reservation remains **88.868333/150 core-hours**. A newly approved
30-minute attempt would add 0.5, giving **89.368333/150**. Known allocated CPU
time including failed07 is **119,961 seconds = 33.3225 core-hours**. Whole-sprint
process CPU and some development overhead remain unknown. Reserved, allocated,
process-CPU and elapsed scopes are distinct.

Before any newly approved submission, refresh shared transport, both controller
locks, scheduler/attempt journals, scratch readiness and all source/input pins;
claim a new output directory; freeze its actual inode, explicit authorization,
source/resource/manifest pins; independently check the final launcher and
transferred bytes; then submit exactly once with exclusive journals. Do not
blind-retry ambiguity or failure. Actual exact-runtime/full640-checkpoint replay,
240 primary cells, 48,000 cached responses, zero new updates/responses, cleanup,
scheduler success and full custody must precede both B reporters.

## New verification scope

Five focused artificial methods passed: four new alias methods and the affected
same-filename/changed-code regression. The unchanged frozen07 rejection was
reproduced separately in the same command. No real Torch or models were imported.
Two new allocation-parser methods then passed for the proposed 30-minute limit,
including rejection of the old 15-minute cap and two over-cap values. Earlier
completed source and scientific suites were not repeated.

The first focused command used 0.081343 child CPU seconds, 0.086072792 elapsed
seconds and 32,620,544 peak child RSS bytes. The resource-parser command used
0.048068 child CPU seconds, 0.059188 elapsed seconds and 28,999,680 peak child
RSS bytes. These exclude implementation, review, metadata queries and other
unmetered overhead. Preparations01/02 retain their original source/test scopes;
the qualifier's subsequent resource-limit literal change is bound separately.
Independent review disposition and final source pins are recorded with the
final preparation receipt, not inferred from these artificial checks.

Galileo's distinct review found one required exception-contract issue: missing
pin paths raised `FileNotFoundError` rather than the existing validation
`ValueError`. The correction and one new two-case method plus the exact
affected existing method pass; final static recheck has zero required findings.
That delta command used 0.071620 child CPU seconds, 0.084855125 elapsed seconds
and 32,210,944 peak child RSS bytes. No actual runtime or model was executed.
The complete reviewed source proposal and predecessor/test/review lineage are
in `results/delivery_vf_alias_preparation_20261023` and matching exclusive
external preparations01–04.
