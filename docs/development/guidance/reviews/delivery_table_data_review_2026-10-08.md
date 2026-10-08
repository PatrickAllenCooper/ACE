# Archived table-data verifier review

## Scope and findings

Goodall (`01a11a8c-e1df-7383-b728-cb55bad330f3`) performed one new bounded
static review of verify_delivery_table_data.py and its fabricated tests. The
reviewer read these two sources only; no package outcomes, model loading, tests,
inference, remote actions or file edits were performed by the reviewer.

Four initial required findings were corrected:

1. Rows after the bottom rule and unsupported table environments could escape
   inspection. The parser now rejects these, malformed boundaries and unexamined
   tabular blocks.
2. Column headers and units were discarded. All four generated tables now have
   independently specified header schemas, including CPU hours, snapped error
   and delivery/control ratio direction.
3. Individual authenticated receipts were not linked through original acceptance
   digests. Gate, summary, complete and physical acceptance now check their
   original input relationships, retaining projected versus original digests.
4. The prototype counter of267 was incorrect. Independent counting and dynamic
   checks give251 metric scalars plus12 numeric epoch fields,263 in total.
   History IDs are not counted as numeric measurements.

The first recheck identified one remaining digest-semantic defect: identity C
acceptance permitted different original and released digests. The final branch
rejects that case while allowing distinct projected acceptance digests. A new
negative fixture precedes the successful projection fixture. Final static
recheck reports zero required fixes; its closure covers the table-data verifier,
not execution or manuscript/package submission readiness.

Main also added twelve-history configuration checks and the verifier source
digest to its output. Nine focused methods pass, including omitted/duplicated/
reordered histories, wrong initialization/budget/CPU units, development-condition
inclusion, row-versus-block estimand confusion, interval/quantizer errors,
nonfinite metrics, hidden rows, changed labels and authenticated but inconsistent
receipt bindings. These fabricated checks do not establish scientific outcomes.

## Actual metadata check and disposition

Main froze final source, tests, independent candidate16 manifest and three
existing table files in a new exclusive attempt03. From an unrelated cwd, the
read-only standard-library verifier authenticated five A/C receipt snapshots,
checked original acceptance relationships and reconstructed all four displayed
tables. It checked16 attribution rows,12 retained histories and11 conditions
in each physical table; headers, row order and263 displayed numeric fields match.
No table data, accepted receipt, package or original worker was changed.

Attempts01/02 remain unaccepted prototypes with their frozen source, output,
timing and logs preserved. Attempt01's267 counter is not accepted evidence.
Attempt02 preceded the final identity-digest fix. Acceptance is a separate
hash-bound record referring to final03 execution, whose original record retains
its unaccepted-at-execution status. No predecessor was overwritten.

Final03 alone used0.032278 child CPU seconds,0.03579704096773639 elapsed seconds
and27590656 bytes peak child RSS. These exclude prototypes, development, tests,
reviews and checksum/packaging work; they are not a whole-sprint total. No models,
fits, optimizer updates, new responses or B outcomes were accessed.

The SAME manuscript gains a scoped provenance sentence and compiles natively;
all ten companion/style bindings still match. Captions, prose, graphical layout,
all final prospective tables and full integrated review remain open. The new
checker is outside existing candidate manifests and needs a new binding before
release inclusion. Neither this check nor a screen pass establishes historical
freeze timing, anonymous original training, redistribution or public readiness.

Evidence: results/delivery_release_preparation_20261007/table_data_verification_20261008.json.
