# Budget-preserving delivery runtime investigation — proposal only

## Required saving

The old exact gate duration was not logged. Aggregate elapsed536.448419seconds
is a conservative proxy; the threshold is479.166667seconds. Required reduction
is **57.281752seconds (10.68% overall)**. With other costs held at165.593231seconds,
three delivery fits must fall from370.855188 to **313.573436seconds**, a **15.45%**
fit-time reduction. No runtime improvement has been measured.

## Source-supported engineering candidates

**Primary candidate: invariant input-normalization caching.** Delivery constructs
fixed full-batch x/y once and sets training-derived min/max buffers before its
30,000-epoch loop. The frozen surrogate forward repeatedly computes
`(x-in_lo)/clamp(in_hi-in_lo,min=1e-9)` before the network. Inputs and range
buffers do not require gradients and ranges never change in that loop. Computing
that exact float32 expression once, then using the unchanged network, loss,
optimizer and epoch loop could remove repeated work without changing the
mathematical learner. This applies only to delivery's fixed full-batch inputs;
online acquisition/replay inputs vary and must not be cached across changes.

Preserve normalization buffers in exported states for ordinary inference.
Preserve tensor operation order, dtype, RNG/model initialization order, finite
loss checks every epoch and all Adam updates. Module hooks, tensor layout/kernel
selection or autograd differences could invalidate exact equivalence; source
reasoning is not runtime proof. The source alone does not establish15.45% headroom.

**Smaller preserving candidates:** bind immutable callable references outside
Python loops and remove repeated nontraining metadata construction across the
three inits while retaining each receipt/custody check. Their possible saving is
unmeasured and likely secondary. Archiving, supervisor sampling and saving have
no isolated timing evidence supporting a57second reduction. Do not suppress
custody, per-step finiteness, telemetry or watchdogs to make a deadline.

**Not accepted as recipe-preserving shortcuts:** fewer epochs/inits/cases,
minibatches, changed learning rate/optimizer/schedule/dtype, fused or compiled
training without exact equivalence, altered online replay/selection, relaxed
finite checks, or18threads from three simultaneous six-thread fits. Splitting
six threads among concurrent models changes kernel/thread behavior and needs a
separate numerical/timing qualification; it is not the first route recommended.
GPU/API use, faster-host migration and history replacement are outside this plan.

## Finite validation request — not executed or authorized

Propose one **local CPU, six-thread,900second,8GiB sampled-RSS** engineering check,
using only preserved first-case rows. Zero emulator/API/GPU calls and no scorer
or grid access. This is a separate bounded development request, not a campaign
runtime expansion. No fits ran during preparation of this proposal.

First run short deterministic fixtures for exact normalized-input, loss,
gradient, parameter/buffer and optimizer-state equality. If any differ, stop.
Then at most six serial fits: baseline and caching candidate for inits0/1/2,
30,000epochs and all original eligible rows/architecture/Adam settings. Alternate
order prospectively: baseline/candidate, candidate/baseline, baseline/candidate.
Record checkpoints1/10/100/1000/10000/30000, exact tensor equality and wall/RSS
samples; keep timing instrumentation symmetric. Any numerical difference,
900second cap, custody mismatch or failure stops without retry or tuning.

Admission evidence requires exact saved weights/buffers at checkpoints and final
state for every model/init, identical call/row eligibility, and observed delivery
time consistent with the313.573436second target and at least57.281752seconds
paired saving. Machine contention and one-history variability limit inference;
this does not guarantee twelve histories within two hours. A qualified source
still needs a prospective freeze, explicit campaign/restart authorization and
actual first-case timing gate. No scoring-based choice or reopening partial
confirmation outcomes is permitted.

## Recommendation

**Keep the campaign stopped under the two-hour cap.** The normalization route is
plausible enough to propose the finite engineering check, but not sufficient to
promise a speedup or restart. If it cannot show exact equivalence and required
runtime headroom, retain the stop and request a material scope/resource decision.
The current2ecec441counter correction is technically ready; it alone does not
solve runtime or authorize another campaign. Registration, seeds,12histories,
3inits,30,000epochs,6threads,2h,8GiB and call/scorer limits remain unchanged.

Machine proposal: `results/runner_delivery_prospective_amendment_20261005/runtime_preserving_proposal.json`.
