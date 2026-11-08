"""Bind identical public states and action labels to both learned policy arms.

This authenticates a record against caller-supplied pins and checks label
alignment. The future dataset runner must independently establish the receipt,
charged feedback, source/runtime closure and world separation. Matching hashes
alone are not proof that a scientific supervision campaign was executed.
"""
from __future__ import annotations

import hashlib
import json

from llm_intervention_contract import _unique_object, canonical, digest, integer
from llm_intervention_task import keys, numerical_features, task_prompt


VERSION = "ace-matched-intervention-supervision-v1"


def matched_views(raw, *, record_sha256, supervision_receipt_sha256, allowed_scope="policy_train"):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != record_sha256:
        raise ValueError("supervision record bytes differ from independent pin")
    if (allowed_scope not in ("policy_train", "artificial")
            or type(supervision_receipt_sha256) is not str or len(supervision_receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in supervision_receipt_sha256)):
        raise ValueError("training scope and independent supervision receipt pin required")
    def reject_constant(value):
        raise ValueError("nonfinite supervision JSON constant")
    record = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    keys(record, ("version", "scope", "world_id", "envelope", "supervised_action_id",
                  "preferences", "supervision_receipt_sha256"))
    if (record["version"] != VERSION or record["scope"] != allowed_scope
            or not integer(record["world_id"])
            or record["supervision_receipt_sha256"] != supervision_receipt_sha256
            or type(record["preferences"]) is not list):
        raise ValueError("supervision record identity or scope differs")
    ids, features, metadata = numerical_features(record["envelope"])
    chosen = record["supervised_action_id"]
    if chosen not in ids or chosen is not None and type(chosen) is not str:
        raise ValueError("supervised action is unavailable in its recorded state")
    def command(action_id):
        return canonical({"stop": True} if action_id is None else {"action_id": action_id})
    pairs, completions, ties, seen = [], [], [], set()
    for p in record["preferences"]:
        keys(p, ("left", "right", "tie", "chosen", "rejected"))
        if (any(x not in ids or x is not None and type(x) is not str for x in (p["left"], p["right"]))
                or p["left"] == p["right"] or type(p["tie"]) is not bool):
            raise ValueError("invalid available preference pair")
        key = frozenset((p["left"], p["right"]))
        if key in seen:
            raise ValueError("duplicate unordered preference pair")
        seen.add(key)
        if p["tie"]:
            if p["chosen"] is not None or p["rejected"] is not None:
                raise ValueError("tie must not carry a strict label")
            ties.append([p["left"], p["right"]])
        else:
            if (any(x is not None and type(x) is not str for x in (p["chosen"], p["rejected"]))
                    or p["chosen"] == p["rejected"] or frozenset((p["chosen"], p["rejected"])) != key):
                raise ValueError("strict labels differ from their compared actions")
            pairs.append((ids.index(p["chosen"]), ids.index(p["rejected"])))
            completions.append((command(p["chosen"]), command(p["rejected"])))
    return {
        "numeric": {"features": features, "chosen": ids.index(chosen), "pairs": tuple(pairs)},
        "language": {"prompt": task_prompt(record["envelope"]),
                     "supervised_completion": command(chosen), "preference_completions": tuple(completions)},
        "alignment": {"version": VERSION, "record_sha256": record_sha256,
            "scope": record["scope"], "world_id": record["world_id"],
            "supervision_receipt_sha256": supervision_receipt_sha256,
            "envelope_sha256": digest(record["envelope"]), "task_sha256": metadata["task_sha256"],
            "action_ids": ids, "supervised_action_id": chosen,
            "strict_preferences": len(pairs), "tied_pairs": ties,
            "numeric_encoding": metadata["encoding"], "feature_width": features.shape[1],
            "scientific_execution_provenance_qualified": False}}
