"""Pure prospective matched exposure; no fitting, sampling or study admission."""
from __future__ import annotations

import hashlib
import json

from llm_intervention_contract import _unique_object, canonical, digest
from llm_intervention_supervision import matched_views


VERSION = "ace-matched-training-exposure-proposal-v1"
PAIR_DOMAIN = VERSION + "/unordered-command-pair"
MAX_RECORDS = 256
MAX_RECORD_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PAIRS_PER_RECORD = 33 * 32 // 2
MAX_STEPS = 2 * MAX_RECORDS


class ExposureInputError(ValueError):
    """Reject the complete proposal input; never silently filter its records."""


def _require(condition, message):
    if not condition:
        raise ExposureInputError(message)


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _command(action_id):
    return canonical({"stop": True} if action_id is None else {"action_id": action_id})


def _selection_sha256(record_id, commands):
    # Do not add record/envelope/receipt hashes, labels, scores, or pair indices.
    return digest([PAIR_DOMAIN, record_id, commands])


def propose_exposure(*, records, record_pins, supervision_receipt_pins, record_roster, scope):
    """Return a JSON-only proposal for identical SFT/DPO example exposure.

    records: exact dict of record ID -> complete raw matched-record UTF-8 bytes.
    Both pin maps have exactly the same IDs as the explicit ordered tuple roster.
    Caller must authenticate full membership, receipt provenance and predeclared,
    outcome-independent record IDs. This function verifies supplied byte pins,
    not their external trust or the contents of the supervision receipt itself.
    No input mutation, filesystem I/O, RNG, training-engine import or callbacks.
    """
    try:
        return _propose(records, record_pins, supervision_receipt_pins, record_roster, scope)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ExposureInputError(str(exc)) from exc


def _propose(records, record_pins, receipt_pins, roster, scope):
    _require(scope in ("artificial", "policy_train"), "training-only explicit scope required")
    _require(type(roster) is tuple and 1 <= len(roster) <= MAX_RECORDS
             and all(type(r) is str and 0 < len(r) <= 128 for r in roster), "bounded ordered record roster required")
    _require(len(set(roster)) == len(roster), "duplicate record identity in roster")
    for mapping in (records, record_pins, receipt_pins):
        _require(type(mapping) is dict and set(mapping) == set(roster), "exact roster membership required")
    _require(all(type(raw) is bytes and 0 < len(raw) <= MAX_RECORD_BYTES for raw in records.values())
             and sum(map(len, records.values())) <= MAX_TOTAL_BYTES, "bounded complete raw records required")
    for rid in roster:
        _require(_pin(record_pins[rid]) and _pin(receipt_pins[rid]), "independent record/receipt SHA-256 pins required")
        _require(hashlib.sha256(records[rid]).hexdigest() == record_pins[rid], "record bytes differ from pin: " + rid)
    _require(len(set(record_pins.values())) == len(roster), "duplicate raw record under different identities")

    details, sft, dpo, numeric_sft, numeric_dpo = [], [], [], [], []
    counts = {k: 0 for k in ("records", "retained_pairs", "strict_pairs", "selected_pairs", "unselected_pairs",
                             "tied_pairs", "records_without_strict_pairs", "sft_steps", "dpo_steps")}
    for rid in roster:
        def reject_constant(value):
            raise ExposureInputError("nonfinite matched-record JSON constant")
        raw = records[rid]
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=reject_constant)
        canonical(record)  # Reject overflowed float literals before the frozen adapter.
        _require(type(record) is dict and type(record.get("preferences")) is list
                 and len(record["preferences"]) <= MAX_PAIRS_PER_RECORD, "bounded complete preferences required")
        views = matched_views(raw, record_sha256=record_pins[rid],
                              supervision_receipt_sha256=receipt_pins[rid], allowed_scope=scope)
        alignment, numeric, language = views["alignment"], views["numeric"], views["language"]
        rows, strict_index = [], 0
        for index, preference in enumerate(record["preferences"]):
            commands = sorted((_command(preference["left"]), _command(preference["right"])))
            tied = preference["tie"]
            chosen_index, rejected_index = (None, None) if tied else numeric["pairs"][strict_index]
            chosen_command, rejected_command = (None, None) if tied else language["preference_completions"][strict_index]
            rows.append({"preference_index": index, "pair_index": None if tied else strict_index,
                "preference_sha256": digest(preference), "unordered_commands": commands,
                "selection_sha256": _selection_sha256(rid, commands),
                "status": "tied" if tied else "unselected", "exposure_count": 0,
                "numeric_chosen_index": chosen_index, "numeric_rejected_index": rejected_index,
                "chosen_command": chosen_command, "rejected_command": rejected_command})
            strict_index += int(not tied)
        strict = [row for row in rows if row["pair_index"] is not None]
        selected = min(strict, key=lambda row: (row["selection_sha256"], tuple(row["unordered_commands"]))) if strict else None
        if selected is not None:
            selected["status"], selected["exposure_count"] = "selected", 1
            step = {"stage": "dpo", "record_id": rid, "pair_index": selected["pair_index"]}
            dpo.append(step)
            numeric_dpo.append({**step, "chosen_index": selected["numeric_chosen_index"],
                                "rejected_index": selected["numeric_rejected_index"]})
        step = {"stage": "sft", "record_id": rid, "pair_index": None}
        sft.append(step)
        numeric_sft.append({**step, "chosen_index": numeric["chosen"], "rejected_index": None})
        details.append({"record_id": rid, "world_id": alignment["world_id"],
            "record_sha256": record_pins[rid], "supervision_receipt_sha256": receipt_pins[rid],
            "envelope_sha256": alignment["envelope_sha256"], "task_sha256": alignment["task_sha256"],
            "action_ids": alignment["action_ids"], "supervised_command": language["supervised_completion"],
            "numeric_supervised_index": numeric["chosen"], "sft_exposure_count": 1,
            "strict_pair_exposures": [row["exposure_count"] for row in strict],
            "dpo_exposure_count": int(selected is not None), "pairs": rows})
        counts["records"] += 1
        counts["retained_pairs"] += len(rows)
        counts["strict_pairs"] += len(strict)
        counts["selected_pairs"] += int(selected is not None)
        counts["unselected_pairs"] += len(strict)-int(selected is not None)
        counts["tied_pairs"] += len(rows)-len(strict)
        counts["records_without_strict_pairs"] += int(not strict)
    counts.update(sft_steps=len(sft), dpo_steps=len(dpo), sft_action_rows=len(sft), dpo_action_rows=2*len(dpo))
    return json.loads(canonical({"version": VERSION, "scope": scope, "proposal_only": True,
        "status": "proposed" if dpo else "untrainable_no_strict_pairs",
        "has_both_training_stages": bool(dpo), "record_roster": roster,
        "roster_sha256": digest(roster), "record_pins": record_pins, "supervision_receipt_pins": receipt_pins,
        "selection_domain": PAIR_DOMAIN, "selection_key_fields": ["fixed_domain", "record_id", "sorted_unordered_commands"],
        "selection_tiebreak": "lexical_sorted_unordered_commands", "records": details,
        "schedule": sft+dpo, "numeric_schedule": numeric_sft+numeric_dpo, "counts": counts,
        "bounds": {"records": MAX_RECORDS, "record_bytes": MAX_RECORD_BYTES, "total_record_bytes": MAX_TOTAL_BYTES,
                   "pairs_per_record": MAX_PAIRS_PER_RECORD, "sft_steps": MAX_RECORDS,
                   "dpo_steps": MAX_RECORDS, "total_steps": MAX_STEPS},
        "source_qualification": "pending_caller", "study_membership_qualified": False,
        "scientific_adoption": False, "equal_parameter_count_claimed": False, "equal_compute_claimed": False}))
