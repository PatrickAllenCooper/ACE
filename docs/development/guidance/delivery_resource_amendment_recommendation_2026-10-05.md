# Owner decision: retain an unscored case or authorize a fresh campaign

This is a saved-evidence recommendation only. No fitting, acquisition, scoring,
grid access, restart, new seeds or historical artifact changes occurred.
The original timing stop and failed caching performance criterion remain valid.

## Accounting: measured versus reserved

Measured campaign elapsed536.448419s + prior development267.144395s + validation
sampling window574.431231s = **1378.024045s (22.97minutes)**. The validation window
excludes small spawn/first-sample latency. Add the explicit1800s preparation/test
reserve: **3178.024045s (52.97minutes)**. That reserve is not measured usage.
Charging the full900s validation cap instead of its observed window gives the
conservative accounted total **3503.592814s (58.39minutes)**. Only3696.407186s
(61.61minutes) remains inside the existing7200s aggregate envelope.

We use the conservative charged total for proposed caps. No fresh two-hour budget
is implicitly created. A resource amendment is required for either continuation.

## Option A — explicitly retain the immutable original case (recommended)

First seed27424209 was prospectively fixed, completed200steps,4803charged calls
and all three original30000epoch fits. Canonical rows, eligible masks, metadata,
configuration and original/refit weights passed independent custody checks.
Hashes remain unchanged; no campaign seal/evaluation scores exist. Eleven of
the original twelve seeds remain unstarted. The retained case was used only for
engineering equivalence/timing; no benchmark score selected it, and caching is
**not adopted**. Keep the original e9f811f learner, architecture/data-use/Adam/
epoch/init recipe for all remaining cases. The adapter's charged-counter and
exact-timing fixes are a separate prospective execution/custody change.

Retaining this case can be scientifically defensible without outcome-based
selection: include the predetermined case with every initialization, preserve
all failure history, retain original paid history and use all twelve endpoints
once complete. It does **not** satisfy the historical no-resume registration
without an explicit prospective amendment. No old approval is reused.

Projection: `1.2*11*536.448419 +300 =7381.119132s` (**123.02minutes** remaining),
using the observed aggregate duration as a conservative proxy for the unlogged
old gate duration. Minimum whole-minute stage cap124minutes; practical rounded
proposal **125minutes**. With the conservative prior account, projected total
10884.711946s (181.41minutes), or11003.592814s (183.39minutes) when charging the
whole125minute stage cap. Practical **185minute aggregate envelope** covers it.
Six CPU threads and8GiB sampled combined RSS remain unchanged.

The first new case, seed1726880744, must pass a timing-only admission with scores
sealed: `1.2*11*first_new_case_seconds +300 <=7500`, threshold **545.45seconds**.
The whole remaining stage has an independent7500s watchdog; aggregate ceiling
11100s includes the charged prior allowance. Stop on failure; no automatic
retries, exclusions, seed replacements or epoch reductions.

Maximum additional calls11*5203=57233; adding original4803 yields **62036**, below
existing62436 aggregate call ceiling. Each remaining case keeps its5203cap.
Score only after the immutable original case plus all eleven new cases pass
complete custody/model seals; no subset verdict. Preserve one-emulator/exposed-
grid scope, unequal online/delivery fitting cost and no acquisition-superiority
claim. Finite-case runtime variability remains unquantified;20% is not a proven
confidence bound or guaranteed admission.

## Option B — entirely fresh twelve-case campaign

Projection `1.2*12*536.448419 +300 =8024.857235s` (**133.75minutes**). Minimum
whole-minute stage134minutes; practical **135minutes**. With the conservative
prior account, projected11528.450049s (192.14minutes), or11603.592814s (193.39minutes)
charging the full135minute stage. Practical **195minute aggregate envelope**.
First-case admission threshold `(8100-300)/(1.2*12)` = **541.67seconds**.

A fresh campaign requires a separate prospective registration, independently
fixed/exclusion-audited seed rule and explicit owner authorization. Do not reuse
the charged first seed as a purportedly fresh history or silently replace the
old registration. No such seed set was generated here. It must also explicitly
amend the cumulative call budget:62436possible new calls +4803already charged =
**67239**, above current62436. This costs more without creating independent
physical systems; all histories still concern one deterministic emulator.

## Smallest defensible decision

Approve **Option A only**, if the owner accepts the explicit retention/no-resume
amendment:125minute remaining-stage cap,185minute conservative aggregate cap,
six threads/8GiB, unchanged learning source and original twelve ordered seeds,
existing62436aggregate call cap. Alternatively keep the campaign stopped. Neither
option is authorized by this recommendation. Cross-run baseline improvements
are not credited to caching; its paired runtime gate failed.

After approval, a continuation-capable adapter still needs outcome-blind
implementation/validation and a new freeze. Bind original registration,
execution, adapter and case-manifest hashes separately from prospective
counter/timing adapter identity; recheck dependencies/unused-seed registries and
live duplicate processes before starting. Existing mkdir/no-resume runner is
not secretly bypassed. No restart or score opening occurs during that work.

Machine-readable exact arithmetic, custody bindings and conditions:
results/runner_delivery_prospective_amendment_20261005/resource_amendment_recommendation.json.
