# Matched training exposure proposal

November 7, 2026. This is a pure prospective proposal for review. It is not an
adopted scientific schedule, first-screen freeze, training run or study admission.
No frozen helper, engine, protocol or ledger is changed.

`scripts/research/llm_intervention_training_exposure.py` exposes:

```python
proposal = propose_exposure(
    records={record_id: complete_raw_record_bytes, ...},
    record_pins={record_id: independent_raw_sha256, ...},
    supervision_receipt_pins={record_id: independent_receipt_sha256, ...},
    record_roster=(record_id_0, record_id_1, ...),
    scope="policy_train",  # "artificial" only for artificial fixtures
)
```

All three dictionaries must have exactly the unique, explicit roster membership.
Every record must be complete and independently pinned. The function calls the
existing `matched_views` once per record, checking scope, record/receipt binding,
public state, feasible commands and label alignment. The caller authenticates
the full study roster, original receipts, paid provenance, completeness of the
retained comparisons and pre-outcome choice of record identities. A linked
receipt hash alone does not prove receipt contents or execution. Replacing both
records and their claimed independent pins is outside this guarantee.

Record IDs must be stable predeclared identifiers, not hashes of outcomes or
label-bearing records. IDs are unique strings of 1–128 characters. Exact duplicate
raw records under aliases are rejected. Distinct independently admitted records
may share a public state; the caller establishes their separate provenance.

## Fixed rule

1. In explicit roster order, propose one SFT step for **every** record, including
   terminal, tied-only and empty-preference records.
2. After all SFT steps, in the same roster order, propose at most one DPO pair per
   record. Only strict comparisons are eligible. No strict pair means zero DPO
   exposure for that record.
3. For each comparison, sort the two canonical action-command strings. Commands
   are `{"action_id":"..."}` or `{"stop":true}`. Its selection key is SHA-256 of
   the UTF-8 canonical compact JSON array:

   ```text
   ["ace-matched-training-exposure-proposal-v1/unordered-command-pair",
    record_id, [lexically_first_command, lexically_second_command]]
   ```

   Choose the strict pair with the smallest lowercase hexadecimal digest; equal
   digests break lexically on the sorted command tuple. The domain is fixed in
   source, with no supplied seed or alternative-selection knob.

The hash excludes preference direction, pair position, record/receipt/envelope
hashes, supervised labels, error, gain, loss and private outcomes. Strict/tied
eligibility is necessarily supplied supervision; the rule does not claim that
eligibility is outcome-independent. Within the strict set, reversing directions,
reordering comparisons or changing input dictionary order leaves the selected
unordered comparison unchanged. Changing the explicit roster changes execution
order. Removing a pair, changing which pairs are strict, changing an ID or
changing available commands changes the admitted problem and is not an authorized
invariance. Such inputs require caller provenance, not a silent rescue.

## Output and engine handoff

The detached JSON-only result includes roster and its semantic digest, all raw
record/receipt pins, each record's world/envelope/task pins and action-index map,
SFT command/index, and one row for **every original comparison in original order**.
Rows include the original-preference semantic hash, unordered commands, selection
key, strict-only `pair_index` or null for a tie, numeric chosen/rejected indices,
language commands, `selected`/`unselected`/`tied` status and exposure count. Original
record bytes remain caller-owned and must be retained in full. Raw record pins
authenticate their serialization; semantic preference/roster digests have a
separate role and are not source or raw-byte authentication.

`proposal["schedule"]` contains only `stage`, `record_id`, `pair_index`, matching
the existing `prepare_training_plan` step schema. Convert its JSON list to a tuple
at that API boundary. `pair_index` indexes the strict-only list from `matched_views`
and engine `prepare_record`, **not** the full preferences list containing ties.
`numeric_schedule` projects to the exact same stage/record/pair sequence and adds
the numeric chosen/rejected indices; SFT has a null rejected index. The numerical
family must consume that schedule against the same matched records. There are
no epochs, repetitions, shuffling, batching, optimizer choices or loss feedback.
Matched example exposure does not imply equal parameters, FLOPs or wall time.

The caller separately prepares/tokenizes every original record and independently
pins each prepared record before calling `prepare_training_plan`. Raw-record pins
are not prepared-record pins. This module does not import or execute the training
engine; compatibility here is its step/index contract, not a qualification of
tokenization, real training, hardware or numerical-family optimization.

If no record has a strict pair, `status` is `untrainable_no_strict_pairs` and
`has_both_training_stages` is false. The diagnostic SFT schedule and every tie are
still returned; do not pass this proposal to an engine requiring both stages.
There is no fabricated pair, resampling, data repair or automatic alternative.
Otherwise status is `proposed`; independent review and caller adoption remain
required. Invalid or incomplete inputs raise `ExposureInputError` for the whole
proposal. Preserve rejected bytes and the failure; do not drop a roster member.

Exact ceilings: 256 records; 1,048,576 bytes per raw record; 67,108,864 raw bytes
total; 528 comparisons per record (all unordered pairs of at most 32 feasible
actions plus stop); at most 256 SFT and 256 DPO steps, 512 steps total. These are
input/proposal bounds, not token or compute budgets. The proposed 160-state,
3,584-retained-pair screen is not instantiated or verified here. If a later runner
authenticates 160 complete records, including 32 with no strict pairs, this rule
gives 160 SFT steps and at most 128 DPO steps. Actual counts and ties remain explicit.

## Artificial qualification and custody

Seven new methods use hand-authored one-node public states and supplied labels,
including a terminal stop. They check direction reversal, pair/menu/serialization
and dictionary permutations, roster order, exact cross-representation indices,
strict/tie/exposure accounting, no-strict status, a clearly labeled forced-hash
collision, rejected pins/membership/duplicates/labels/bounds, and purity. They
create no environment, scientific descriptor, policy response or model, and do
not call the training engine or older tests.

One focused invocation is retained under
`results/ace_llm_intervention_training_exposure_preparation_20261107` and exclusive
external `ACE-LLM-intervention-training-exposure-preparation-20261107-01`. Its source
manifest captures exact new source/test/guide and imported local dependency bytes;
the receipt captures command, environment overrides, stdout/stderr, retained input
bytes (including rejected fixtures), CPU and wall time. Source hashes belong to
that custody; the pure API performs no filesystem/source attestation. Caller
source/runtime qualification remains pending. Measured test-child CPU excludes
unmetered implementation, inspection and custody and creates no new allowance.
