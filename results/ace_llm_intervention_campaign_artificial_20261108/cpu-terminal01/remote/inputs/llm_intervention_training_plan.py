"""Prepare complete collected batches for the supplied-model training engine.

CPU tokenization only. The outer runner authenticates source/runtime, tokenizer
and collection execution. Byte pins and this bridge do not establish admission.
"""
from __future__ import annotations

import hashlib
import json

from llm_intervention_contract import canonical, digest, _unique_object
from llm_intervention_training_exposure import propose_exposure
from llm_intervention_training_engine import (
    gpu09_numeric_recipe, prepare_record, prepare_training_plan, MAX_PARAMETERS,
)

VERSION = "ace-complete-collection-training-plan-v1"
COLLECTION_VERSION = "ace-supplied-training-collection-v1"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _read(raw, expected):
    _need(type(raw) is bytes and 0 < len(raw) <= 64 * 1024**2,
          "bounded captured JSON bytes required")
    _need(type(expected) is str and len(expected) == 64
          and hashlib.sha256(raw).hexdigest() == expected, "independent byte pin differs")
    def reject(value):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject)
    canonical(value)
    return value


def qwen_adapter_shapes():
    """Explicit qualified Qwen3-0.6B q/v architecture, not inferred from data."""
    return {f"model.layers.{i}.self_attn.{part}.{side}": shape
            for i in range(28) for part, width in (("q_proj", 2048), ("v_proj", 1024))
            for side, shape in (("A", [8, 1024]), ("B", [width, 8]))}


def prepare_complete_collection(*, terminal_raw, terminal_pin, inventory_raw,
        inventory_pin, collection_plan_raw, collection_plan_pin, records,
        receipts, expected_roster, expected_scope, tokenizer, tokenizer_pin,
        outer_provenance_pin, checkpoint_steps=()):
    """Authenticate the full supplied batch before the first tokenization call.

    Record and receipt dictionaries contain bytes keyed by the independently
    supplied ordered record IDs. No filesystem/model/environment callbacks.
    The output includes every record, tie, unused pair and exposure decision.
    The caller must independently admit the collection and tokenizer beforehand.
    """
    _need(expected_scope in ("artificial", "policy_train")
          and type(expected_roster) is tuple and 1 <= len(expected_roster) <= 256
          and all(type(x) is str and 0 < len(x) <= 128 for x in expected_roster)
          and len(set(expected_roster)) == len(expected_roster), "explicit complete roster required")
    _need(type(records) is dict and type(receipts) is dict
          and set(records) == set(receipts) == set(expected_roster), "full batch membership differs")
    _need(all(type(x) is bytes for x in [*records.values(), *receipts.values()])
          and sum(map(len, records.values())) + sum(map(len, receipts.values())) <= 64 * 1024**2,
          "bounded complete batch bytes required")
    terminal = _read(terminal_raw, terminal_pin)
    inventory = _read(inventory_raw, inventory_pin)
    collection = _read(collection_plan_raw, collection_plan_pin)
    _need(all(x.get("version") == COLLECTION_VERSION for x in (terminal, inventory, collection)),
          "collection schema differs")
    _need(terminal.get("status") == "complete" and terminal.get("exposure_ready") is True
          and terminal.get("accounting_complete") is True
          and terminal.get("partial_records_authorized_for_training") is False
          and terminal.get("complete_case_filtering_allowed") is False
          and terminal.get("policy_training_performed") is False
          and terminal.get("scientific_evaluation_performed") is False
          and terminal.get("accounting", {}).get("unknown_return_attempts") == 0,
          "incomplete, failed or previously evaluated collection")
    roster = list(expected_roster)
    _need(terminal.get("scope") == inventory.get("scope") == expected_scope
          and terminal.get("record_roster") == inventory.get("record_roster") == roster
          and terminal.get("inventory_sha256") == inventory_pin
          and inventory.get("collection_plan_sha256") == collection_plan_pin
          and terminal.get("planned_records") == collection.get("records")
          and terminal.get("world_roster") == collection.get("world_roster"),
          "terminal, inventory, original plan or independent roster differs")
    worlds = terminal.get("worlds")
    _need(type(worlds) is list and len(worlds) == len(collection["world_roster"])
          and [x.get("world_id") for x in worlds] == collection["world_roster"]
          and len(set(collection["world_roster"])) == len(worlds)
          and all(x.get("status") == "complete" for x in worlds), "incomplete world roster")
    accounting = terminal["accounting"]
    for world in worlds:
        account = world.get("accounting", {})
        _need(account.get("accounting_complete") is True
              and account.get("accounting_errors") == []
              and type(account.get("unknown_return_attempts")) is int
              and account["unknown_return_attempts"] == 0
              and account.get("reserved_response_attempts") == account.get("known_returns")
              == account.get("executor_accepted_responses"), "incomplete per-world accounting")
    fields = ("reserved_response_attempts", "known_returns", "executor_accepted_responses",
              "unknown_return_attempts", "total_reserved_cost_units")
    _need(all(type(accounting.get(k)) is int and accounting[k] >= 0
              and all(type(w.get("accounting", {}).get(k)) is int
                      and w["accounting"][k] >= 0 for w in worlds)
              and accounting[k] == sum(w["accounting"][k] for w in worlds) for k in fields)
          and accounting["reserved_response_attempts"] == accounting["known_returns"]
          == accounting["executor_accepted_responses"]
          and accounting["reserved_response_attempts"] <= collection["maximum_response_attempts"]
          and accounting["total_reserved_cost_units"] <= collection["maximum_reserved_cost_units"],
          "complete response accounting differs from world totals or fixed caps")
    planned, observed = collection["records"], inventory.get("records")
    _need(type(planned) is list and type(observed) is list
          and [x.get("record_id") for x in planned] == [x.get("record_id") for x in observed] == roster,
          "planned or observed record membership/order differs")
    record_pins, receipt_pins = inventory.get("record_pins"), inventory.get("supervision_receipt_pins")
    _need(type(record_pins) is dict and type(receipt_pins) is dict
          and set(record_pins) == set(receipt_pins) == set(roster), "incomplete pin maps")
    for original, row in zip(planned, observed):
        rid = original["record_id"]
        _need(all(row.get(k) == v for k, v in original.items())
              and row.get("scope") == expected_scope and row.get("status") == "complete"
              and row.get("record_sha256") == record_pins[rid]
              and row.get("supervision_receipt_sha256") == receipt_pins[rid],
              "partial or altered inventory row")
        record = _read(records[rid], record_pins[rid])
        receipt = _read(receipts[rid], receipt_pins[rid])
        _need(record.get("world_id") == receipt.get("world_id") == original["world_id"]
              and record.get("scope") == receipt.get("scope") == expected_scope
              and receipt.get("snapshot_index") == original["snapshot_index"]
              and receipt.get("candidate_ids") == original["candidate_ids"]
              and receipt.get("pair_count") == len(original["pairs"])
              and receipt.get("task_sha256") == digest(record["envelope"]["task"])
              and receipt.get("envelope_file_sha256") == hashlib.sha256(
                  (canonical(record["envelope"])+"\n").encode()).hexdigest()
              and [[p["left"], p["right"]] for p in record["preferences"]] == original["pairs"],
              "record/receipt differs from original complete pair roster")
    exposure = propose_exposure(records=records, record_pins=record_pins,
        supervision_receipt_pins=receipt_pins, record_roster=expected_roster, scope=expected_scope)
    _need(exposure["has_both_training_stages"], "no strict pairs, no training-plan rescue")
    recipe = gpu09_numeric_recipe()
    prepared = tuple(prepare_record(record_id=rid, raw=records[rid],
        record_sha256=record_pins[rid], supervision_receipt_sha256=receipt_pins[rid],
        scope=expected_scope, tokenizer=tokenizer, tokenizer_sha256=tokenizer_pin,
        numeric_recipe=recipe) for rid in expected_roster)
    plan = prepare_training_plan(records=prepared,
        record_pins={r["record_id"]: digest(r) for r in prepared},
        schedule=tuple(exposure["schedule"]), numeric_recipe=recipe, scope=expected_scope,
        execution_kind="qwen_gpu09", expected_adapter_shapes=qwen_adapter_shapes(),
        max_trainable_parameters=MAX_PARAMETERS, checkpoint_steps=checkpoint_steps,
        outer_provenance_receipt_sha256=outer_provenance_pin)
    return {"version": VERSION, "collection_terminal_sha256": terminal_pin,
        "collection_inventory_sha256": inventory_pin, "collection_plan_sha256": collection_plan_pin,
        "tokenizer_sha256": tokenizer_pin, "exposure": exposure, "exposure_sha256": digest(exposure),
        "training_plan": plan, "training_plan_sha256": digest(plan),
        "new_environment_responses": 0, "training_updates": 0,
        "outer_execution_qualified": False, "scientific_qualified": False}
