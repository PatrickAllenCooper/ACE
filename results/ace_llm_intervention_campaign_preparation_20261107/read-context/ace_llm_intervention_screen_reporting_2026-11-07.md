# Descriptive roster cost reporting primitive (2026-11-07)

`scripts/research/llm_intervention_screen_reporting.py` provides the pure callable
`summarize_screen(*, roster, cells, report_pins)`. It consumes saved
`ace-offline-cost-curve-v1` JSON bytes. It does not import or run the cost reporter,
world, learner, model or response code, and does not read/write files. This is an
artificial/development reporting primitive, not a scientific protocol freeze,
provenance qualification or acceptance decision. The future 24-world × 8-policy
screen is not instantiated by this implementation or its tests.

## Exact input contract

All dictionaries have exactly the listed keys. The caller predeclares the roster
and supplies a separate, independently retained report-byte pin inventory.
Computing a fresh pin from whatever output happens to be present does not establish
independent provenance. The function checks equality; the caller establishes trust.

* `roster`: `scope` (`artificial` or `development`), `worlds`, `policy_ids`,
  `strata_fields`. There are 1–64 unique worlds and 1–8 unique policy IDs.
* Each world: `world_id` (nonnegative integer), `world_sha256`, `task_sha256`,
  `learner_sha256`, `shared_initial_sha256` (lowercase SHA-256 strings),
  `acquisition_cap_units` (positive integer), `initial_cost_units` (nonnegative
  integer), `scale`, `threshold` (finite positive numbers), `strata`. Initial plus
  acquisition cap is at most 1,000,000. Strata have exactly the 0–4 predeclared
  `strata_fields`, with nonempty string values. Combined strata, if wanted, must
  already exist as a predeclared label; no outcome-derived groups are constructed.
* `cells`: one exact `{world_id, policy_id, kind, raw_report, reason}` entry for
  **every** roster world × policy. `kind="report"` requires raw `bytes` (at most
  1 MiB) and `reason=None`. `kind="unresolved"` requires `raw_report=None` and an
  explicit nonempty reason. Missing/invalid upstream output requires a cell too.
* `report_pins`: one exact `{world_id, policy_id, sha256}` entry for every cell;
  a declared unresolved cell may have `sha256=None` or its previously known pin.
  Report cells require a valid independently supplied pin to become resolved.

Example invocation, using caller-supplied inventories only:

```python
summary = summarize_screen(roster=declared_roster,
                           cells=explicit_report_or_unresolved_cells,
                           report_pins=independent_report_byte_pins)
```

Omitted, extra or duplicate identities, unsupported scopes and malformed rosters
raise `RosterError`; there is no partial roster result. Inventory order is
irrelevant. Output order follows the declared world and policy order.

## Fail-closed validation and retained endpoints

Report size/type and SHA-256 equality are checked **before JSON decoding**. Strict
JSON rejects duplicate keys, nonfinite constants and overflowing number literals.
The reporter version, exact fields, typed counters, known/unknown returns,
initial/acquisition charges, receipt-pin inventory, accepted curve prefix,
inclusive threshold crossing and final endpoint must agree. World/task/learner/
shared-initial pins and cap/scale/threshold must match the roster. Full initial
cost must match its per-world declaration. Partial failed initialization retains
actual charged cost without manufacturing a comparable initial error.

Across individually consistent paired reports, overlapping initial receipt pins,
completed initial counts and any available shared-initial prediction/error point
must agree. Conflicting arms both become unresolved; no arm is chosen as authority.
These checks cannot authenticate the pinned source/receipts, prove the truth law,
verify physical cache reuse, or establish execution/source/split provenance.
Original bytes, upstream validation evidence and provenance remain caller-owned.

An invalid cell becomes an explicit unresolved row with expected/observed pin,
validation stage/reason and the common **acquisition cap**. Its attainment, final
error and actual charges are unknown (`None`), not false or zero. Operational or
evaluation failures retain their valid report's prefix attainment and charges but
also use the acquisition cap. A successful attained report uses its first crossing;
a successful nonattained report uses the cap. A crossing at cost zero stays zero.

`capped_total_cost_units = declared_initial_cost_units + capped_acquisition_cost_units`.
This comparable endpoint uses the **full declared** initial cost even for a failed
partial initialization. Actual initial/acquisition/total charges are retained
separately and can differ from the endpoint. All sums are **logical per-arm
charges**; they do not estimate deduplicated physical shared-initial/cache spending.

## Descriptive outputs and limits

All cells remain in `rows`. Every policy summary includes capped acquisition/total
vectors and means alongside attained, failed, unresolved and combined counts.
Failure-prefix attainment is separately counted. Actual charge vectors contain
unknowns; `all_world_sum` is absent (`None`) if any charge is unknown. An explicitly
named `reported_cells_subtotal` is only a partial charge accounting subtotal.

Every unordered policy pair includes per-world **left minus right** capped
acquisition and total differences and lower-cost wins, exact ties and losses.
Both differences agree because the declared initial cost is shared within each
world. Counts of failed/unresolved paired worlds accompany these comparisons;
cap ties are not claims of successful performance. Every predeclared stratum uses
all its worlds and all policies. There are no p-values, confidence intervals,
ratios, zero floors, outcome selections or complete-case performance summaries.

Final-error vectors and available/unavailable counts are always retained. The
all-world final-error mean is `None` if **any** world has no final error. A fully
evaluated failed operational prefix can supply a final error, with its failure
count still present; this does not certify successful episode completion.

The new tests hand-author small report schemas and stipulated scalar errors,
without generating worlds/truths/responses or calling the upstream reporter.
They cover a hand-calculated three-world/three-policy oracle, heterogeneous costs,
failed-prefix attainment, partial initialization, zeros/ties, paired conflicts,
byte/content tampering, ordering and explicit missing/unresolved cells. Their fake
receipt pins are fixture labels, not authentication evidence. The caller still
owns independent source/report custody, authentic upstream reports, prospective
roster/strata selection, and any later scientific acceptance or integration.
