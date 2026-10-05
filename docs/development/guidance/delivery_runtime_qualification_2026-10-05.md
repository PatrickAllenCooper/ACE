# Bounded normalization qualification: equivalent, insufficient paired saving

Patrick's standing instruction authorizes bounded existing-data local development
without another approval request. The validation freeze was committed/pushed as
f38f3a2a before fits. One independent900second watchdog, six CPU threads,8GiB
combined process-tree sampled RSS, at most six serial full fits. Accounting:
536.448419s stopped campaign +267.144395s prior development fitting/scoring +
1800s explicit preparation/test reserve +900s validation cap =3503.592814s,
below7200s. The1800s is a conservative reserve, not a measured historical total.

## Actual execution

Supervisor17210 and worker17230 started and were observed using about six CPU
cores. The20step exact fixture passed before full fits. Fixed order:
init0 baseline/candidate; init1 candidate/baseline; init2 baseline/candidate.
All six fits completed, each preserving six models,30000epochs, float32,
training-derived bounds, eligible rows, Adam.002, RNG order, and finite-loss
checks. Zero new emulator/API/GPU/scorer calls; no grid access.

Observed first-RSS-sample-to-finish window574.431231seconds; it excludes small
spawn/first-sample latency. The hard900second watchdog remained active. Peak
combined process-tree sampled RSS367820800bytes (~351MiB),1096samples.
Historical campaign artifacts remain unchanged, unsealed and unscored.

## Exact equivalence

All108 paired checkpoints matched: six checkpoints for each of six models and
three initializations. Digests include parameters/buffers, gradients, loss and
Adam state. Independent saved-tensor comparison verified all18 candidate models
bit-identical to their new baseline counterparts. The new baselines also match
all historical first-case saved weights bit-for-bit. This qualifies normalization
caching on this fixed history and tested runtime, not arbitrary data/runtime
universality. No benchmark accuracy was evaluated.

## Paired timing and frozen decision

- Init0: baseline96.206565s; candidate91.004618s; saving5.201947s.
- Init1: baseline95.132110s; candidate90.780096s; saving4.352013s.
- Init2: baseline104.120357s; candidate94.962393s; saving9.157964s.

Total baseline295.459031s; candidate276.747107s. Paired saving **18.711924s
(6.33%)**, below required **57.281752s**. **Frozen runtime qualification failed.**
The absolute candidate fit-time ceiling313.573436s passed, but both criteria
were prospectively required. Do not reinterpret this as qualification.

The new baseline itself is75.40s faster than the stopped campaign's370.86s
fit total. Hardware/load/thermal timing variation was not isolated; attributing
that cross-run difference to caching would exaggerate the engineering effect.
Alternating order helps but is only three paired repetitions on one history,
with symmetric checkpoint overhead. The old exact first-case gate duration was
unlogged; its aggregate536.45s proxy remains a limitation. No new full-case
acquisition timing was measured here, so two-hour twelve-case feasibility is
not established by the lower isolated fit time.

## Decision

Keep the confirmation stopped. Retain this modest, bit-identical optimization
as engineering evidence; do not ship it into the frozen campaign, tune it,
repeat fits, replace seeds, shorten training, reopen scores or restart. A further
route or material scope/resource decision needs its own prospective plan.
The original delivery-error claim remains untested; PR57 stays draft/unmerged.

Compact evidence: results/delivery_runtime_qualification_20261005/{approval,
frozen,audit,result,fits,fixture}.json. Full traces/weights/telemetry are retained
at /Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-runtime-qualification-20261005.
