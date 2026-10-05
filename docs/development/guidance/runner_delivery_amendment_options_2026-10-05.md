# Prospective delivery adapter correction and runtime decision

## Technical correction

The new adapter counts only registered disjoint charged roles. `startup` and
`executed` are overlapping summaries checked separately against startup rows
and selected flags. Unknown, negative, noninteger or inconsistent counters fail
closed. Canonical row hashes, unique indices, call ceilings, configuration,
steps, receipt identity, eligible-row masks, original weights and byte-seal
checks remain in place. A future execution also records exact first-case elapsed
and admission projection; the stopped adapter did not persist that duration.

Thirteen tests pass, including actual first-case metadata with 4,803 charged rows,
3,003 startup calls and 200 selected executions, plus counter mutation tests.
The corrected verifier audits the preserved case in memory without writing a
campaign seal. It does not declare the twelve-case campaign complete. Original
case hashes, frozen adapter, campaign and execution receipt remain unchanged.
No acquisition, fitting, scoring, restart, replacement or resource change occurred.
New adapter identity invalidates the old approval binding; any future execution
requires a separately approved prospective freeze.

## Measured timing and smallest explicit choices

Original campaign elapsed: **536.448419 seconds**. Three delivery fits consumed
**370.855188 seconds** (115.84 / 118.85 / 136.17). The remaining **165.593231
seconds** combines acquisition, initialization, saving, preparation and supervisor
cost; it is not a separately instrumented acquisition measurement.

The original first-case gate permits 479.166667 seconds. Its failure remains
valid independently of the verifier bug. Using aggregate elapsed as a conservative
proxy gives `1.2 * 12 * 536.448419 + 300 = 8024.857235 seconds`, approximately
**133.75 minutes**. Historical exact gate duration was not persisted. Only one
case was measured; the 20% reserve is not a statistical runtime bound.

1. **Retain the approved two-hour cap:** keep the campaign stopped and unscored.
   A separately scoped engineering proposal could investigate runtime reductions
   on already exposed artifacts, but no additional fitting is authorized here.
2. **Review a prospective 135-minute cap:** this is the smallest practical rounded
   proposal above the measured formula (134 minutes is its mathematical whole-minute
   minimum). Keep six CPU threads, 8 GiB sampled RSS, twelve cases, three inits,
   30,000 epochs and original call ceilings. The timing formula would permit
   541.67 seconds for the first case. This is a proposal, not a guarantee or
   authorization. Any restart/history-reuse/seed policy must be expressly decided
   in a new prospective amendment; the stopped registration is not resumed or
   silently edited. No new seeds have been generated or substituted.

No performance rescue, changed sample size, altered fit recipe, shortened epochs,
GPU/API use, scoring of the partial case or automatic rerun is proposed as an
approved action. The original primary >=20% delivery-error claim remains untested.
One deterministic emulator and a previously exposed full grid do not establish
independent-system generalization or acquisition superiority. PR57 remains draft.

Evidence: `results/runner_delivery_prospective_amendment_20261005/receipt.json`
and `tests.txt`. The new freeze is for technical review only, execution disabled.
