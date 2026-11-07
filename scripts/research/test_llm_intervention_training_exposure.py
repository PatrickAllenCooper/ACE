"""New hand-authored public-state fixtures only; no engine, model or world calls."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llm_intervention_contract import Action, Node, VERSION as CONTRACT_VERSION, canonical, digest
from llm_intervention_supervision import VERSION as RECORD_VERSION, matched_views
from llm_intervention_task import PredictionTask, make_envelope
import llm_intervention_training_exposure as exposure


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(value):
    return (canonical(value)+"\n").encode()


def command(a):
    return canonical({"stop": True} if a is None else {"action_id": a})


def pair(left, right, chosen=None, *, tie=False):
    return {"left": left, "right": right, "tie": tie, "chosen": None if tie else chosen,
            "rejected": None if tie else right if chosen == left else left}


def fixture(world=0, *, all_ties=False, terminal=False):
    graph = (Node("X", (), "unit", (-1., 1.)),)
    menu = tuple(Action(a, "intervene", (("X", x),), ("X",), 1) for a, x in (("a", -1.), ("b", 0.), ("c", 1.)))
    state = {"version": CONTRACT_VERSION, "objective": "handcrafted exposure fixture", "domain_notes": "",
        "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "scm": [{"node": "X", "terms": ["intercept"], "parameter_mean": [0.],
                 "parameter_covariance": [[1.]], "noise_variance": 1., "eligible_rows": 0}],
        "history": [], "spent_units": 0, "remaining_units": 0 if terminal else 3,
        "requests": 0, "returned_responses": 0, "terminal": terminal}
    preferences = [] if terminal else [pair("a", "b", "a"), pair("a", "c", tie=True),
        pair("a", None, None), pair("b", "c", "c"), pair("b", None, tie=True), pair("c", None, "c")]
    if all_ties:
        preferences = [dict(p, tie=True, chosen=None, rejected=None) for p in preferences]
    return {"version": RECORD_VERSION, "scope": "artificial", "world_id": world,
        "envelope": make_envelope(json.loads(canonical(state)), PredictionTask("X", (menu[0],), (1.,), 1., .04)),
        "supervised_action_id": None if terminal else "b", "preferences": preferences,
        "supervision_receipt_sha256": sha(("artificial receipt claim " + str(world)).encode())}


def inputs(entries, roster=None):
    records = {rid: raw(value) for rid, value in entries}
    return {"records": records, "record_pins": {rid: sha(value) for rid, value in records.items()},
        "supervision_receipt_pins": {rid: value["supervision_receipt_sha256"] for rid, value in entries},
        "record_roster": tuple(records) if roster is None else roster, "scope": "artificial"}


def selected(report):
    return {r["record_id"]: next((p for p in r["pairs"] if p["status"] == "selected"), None) for r in report["records"]}


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def retain(self, name, arguments, *, reject=None, artificial_patch=None):
        directory = self.directory/name
        directory.mkdir(exist_ok=False)
        inventory = []
        for i, (rid, value) in enumerate(arguments["records"].items()):
            filename = f"record-{i:03d}.bin"
            with (directory/filename).open("xb") as handle:
                handle.write(value)
            inventory.append({"record_id": rid, "file": filename, "sha256": sha(value)})
        metadata = {k: v for k, v in arguments.items() if k != "records"}
        metadata.update(records=inventory, record_roster_type=type(arguments["record_roster"]).__name__,
                        artificial_patch=artificial_patch, expected_rejection=reject)
        with (directory/"invocation.json").open("xb") as handle:
            handle.write(raw(metadata))
        before = deepcopy(arguments)
        try:
            result = exposure.propose_exposure(**arguments)
        except exposure.ExposureInputError as exc:
            with (directory/"rejection.json").open("xb") as handle:
                handle.write(raw({"exception_type": type(exc).__name__, "message": str(exc)}))
            if reject is None:
                raise
            self.assertIn(reject, str(exc))
            return None
        finally:
            self.assertEqual(arguments, before)
        with (directory/"proposal.json").open("xb") as handle:
            handle.write(raw(result))
        self.assertIsNone(reject, "invalid complete input was accepted")
        return result

    def assert_alignment(self, report, arguments):
        self.assertEqual([{k: s[k] for k in ("stage", "record_id", "pair_index")} for s in report["numeric_schedule"]], report["schedule"])
        for step in report["schedule"]:
            self.assertEqual(set(step), {"stage", "record_id", "pair_index"})
        for detail in report["records"]:
            rid = detail["record_id"]
            views = matched_views(arguments["records"][rid], record_sha256=arguments["record_pins"][rid],
                supervision_receipt_sha256=arguments["supervision_receipt_pins"][rid], allowed_scope="artificial")
            self.assertEqual(detail["supervised_command"], command(detail["action_ids"][detail["numeric_supervised_index"]]))
            self.assertEqual(detail["numeric_supervised_index"], views["numeric"]["chosen"])
            preferences = json.loads(arguments["records"][rid])["preferences"]
            for row in detail["pairs"]:
                self.assertEqual(row["preference_sha256"], digest(preferences[row["preference_index"]]))
                if row["status"] == "tied":
                    self.assertIsNone(row["pair_index"])
                    self.assertEqual(row["exposure_count"], 0)
                    self.assertIsNone(row["numeric_chosen_index"])
                    continue
                index = row["pair_index"]
                self.assertEqual([row["chosen_command"], row["rejected_command"]], list(views["language"]["preference_completions"][index]))
                self.assertEqual([row["numeric_chosen_index"], row["numeric_rejected_index"]], list(views["numeric"]["pairs"][index]))
                self.assertEqual(row["chosen_command"], command(detail["action_ids"][row["numeric_chosen_index"]]))
                self.assertEqual(row["rejected_command"], command(detail["action_ids"][row["numeric_rejected_index"]]))

    def test_exact_cross_representation_schedule_and_full_accounting(self):
        args = inputs([("z-active", fixture()), ("a-ties", fixture(1, all_ties=True)), ("m-stop", fixture(2, terminal=True))])
        result = self.retain("mixed", args)
        self.assert_alignment(result, args)
        self.assertEqual(result["schedule"][:3], [{"stage": "sft", "record_id": rid, "pair_index": None} for rid in args["record_roster"]])
        self.assertEqual(result["schedule"][3:], [{"stage": "dpo", "record_id": "z-active", "pair_index": selected(result)["z-active"]["pair_index"]}])
        self.assertEqual(result["counts"], {"records": 3, "retained_pairs": 12, "strict_pairs": 4,
            "selected_pairs": 1, "unselected_pairs": 3, "tied_pairs": 8, "records_without_strict_pairs": 2,
            "sft_steps": 3, "dpo_steps": 1, "sft_action_rows": 3, "dpo_action_rows": 2})
        self.assertEqual(result["record_pins"], args["record_pins"])
        self.assertEqual(result["supervision_receipt_pins"], args["supervision_receipt_pins"])
        for record in result["records"]:
            self.assertEqual(record["sft_exposure_count"], 1)
            for row in record["pairs"]:
                # Independent byte-layout oracle: exact declared domain and no other fields.
                key = json.dumps(["ace-matched-training-exposure-proposal-v1/unordered-command-pair",
                                  record["record_id"], row["unordered_commands"]], separators=(",", ":")).encode()
                self.assertEqual(row["selection_sha256"], sha(key))
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["scientific_adoption"])

    def test_direction_reversal_does_not_change_selected_comparison(self):
        original, reversed_record = fixture(), fixture()
        reversed_record["supervised_action_id"] = "a"
        for p in reversed_record["preferences"]:
            p["left"], p["right"] = p["right"], p["left"]
            if not p["tie"]:
                p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
        args0, args1 = inputs([("fixed-id", original)]), inputs([("fixed-id", reversed_record)])
        first, second = self.retain("forward", args0), self.retain("reversed", args1)
        a, b = selected(first)["fixed-id"], selected(second)["fixed-id"]
        self.assertEqual(a["unordered_commands"], b["unordered_commands"])
        self.assertEqual(a["selection_sha256"], b["selection_sha256"])
        self.assertEqual(a["numeric_chosen_index"], b["numeric_rejected_index"])
        self.assertEqual(a["numeric_rejected_index"], b["numeric_chosen_index"])
        self.assertNotEqual(first["record_pins"], second["record_pins"])
        self.assert_alignment(second, args1)

    def test_pair_input_permutations_and_explicit_roster_order(self):
        entries = [("second", fixture()), ("first", fixture(1))]
        args = inputs(entries)
        baseline = self.retain("baseline", args)
        permuted = inputs(list(reversed(entries)), roster=args["record_roster"])
        self.assertEqual(self.retain("mapping_permutation", permuted), baseline)
        changed = []
        for rid, record in deepcopy(entries):
            record["preferences"].reverse()
            record["envelope"]["state"]["menu"].reverse()
            for p in record["preferences"]:
                p["left"], p["right"] = p["right"], p["left"]
            changed.append((rid, record))
        changed_args = inputs(changed)
        for rid in changed_args["records"]:
            changed_args["records"][rid] = json.dumps(json.loads(changed_args["records"][rid]), indent=2).encode()
            changed_args["record_pins"][rid] = sha(changed_args["records"][rid])
        reordered = self.retain("pairs_menu_json_permutation", changed_args)
        for rid in args["record_roster"]:
            self.assertEqual(selected(baseline)[rid]["unordered_commands"], selected(reordered)[rid]["unordered_commands"])
            self.assertEqual(selected(baseline)[rid]["selection_sha256"], selected(reordered)[rid]["selection_sha256"])
        self.assert_alignment(reordered, changed_args)
        roster_changed = inputs(entries, roster=tuple(reversed(args["record_roster"])))
        ordered = self.retain("explicit_roster_reversed", roster_changed)
        self.assertEqual([s["record_id"] for s in ordered["schedule"]], ["first", "second", "first", "second"])

    def test_lexical_collision_tiebreak_excludes_tied_pairs(self):
        record = fixture()
        record["preferences"] = [pair("a", "b", tie=True), pair("a", None, None), pair("a", "c", "c")]
        with patch.object(exposure, "_selection_sha256", return_value="0"*64):
            result = self.retain("forced_hash_collision", inputs([("r", record)]),
                                 artificial_patch="_selection_sha256 returns 64 zeros; collision fixture only")
        self.assertEqual(selected(result)["r"]["unordered_commands"], [command("a"), command("c")])
        self.assertEqual([p["status"] for p in result["records"][0]["pairs"]], ["tied", "unselected", "selected"])

    def test_all_zero_strict_status_preserves_every_sft_and_tie(self):
        args = inputs([("ties", fixture(all_ties=True)), ("terminal", fixture(1, terminal=True))])
        result = self.retain("all_zero", args)
        self.assertEqual(result["status"], "untrainable_no_strict_pairs")
        self.assertFalse(result["has_both_training_stages"])
        self.assertEqual([s["stage"] for s in result["schedule"]], ["sft", "sft"])
        self.assertEqual(result["counts"]["dpo_steps"], 0)
        self.assertEqual(result["counts"]["tied_pairs"], 6)
        self.assertTrue(all(p["exposure_count"] == 0 for r in result["records"] for p in r["pairs"]))

    def test_invalid_pins_membership_duplicates_labels_and_bounds_reject(self):
        cases = {"altered_bytes": "record bytes differ", "wrong_record_pin": "record bytes differ",
            "bad_receipt_pin": "SHA-256 pins required", "wrong_receipt_pin": "identity or scope differs",
            "duplicate_roster": "duplicate record identity", "duplicate_raw": "duplicate raw record",
            "missing_record": "exact roster membership", "extra_pin": "exact roster membership",
            "duplicate_pair": "duplicate unordered preference", "tie_label": "tie must not carry",
            "unavailable_action": "invalid available preference", "partial_json": "Expecting",
            "duplicate_json_key": "duplicate JSON key", "bad_scope": "training-only",
            "foreign_record_scope": "identity or scope differs", "roster_type": "ordered record roster",
            "record_count": "bounded ordered record roster", "pair_count": "bounded complete preferences"}
        for case, message in cases.items():
            with self.subTest(case=case):
                args = inputs([("r", fixture())])
                if case == "altered_bytes": args["records"]["r"] += b" "
                elif case == "wrong_record_pin": args["record_pins"]["r"] = "0"*64
                elif case == "bad_receipt_pin": args["supervision_receipt_pins"]["r"] = "x"*64
                elif case == "wrong_receipt_pin": args["supervision_receipt_pins"]["r"] = "0"*64
                elif case == "duplicate_roster": args["record_roster"] = ("r", "r")
                elif case == "duplicate_raw":
                    for name in ("records", "record_pins", "supervision_receipt_pins"):
                        args[name]["alias"] = args[name]["r"]
                    args["record_roster"] = ("r", "alias")
                elif case == "missing_record": args["records"].clear()
                elif case == "extra_pin": args["record_pins"]["extra"] = "0"*64
                elif case == "bad_scope": args["scope"] = "confirmation"
                elif case == "roster_type": args["record_roster"] = ["r"]
                elif case == "record_count": args["record_roster"] = tuple("r"+str(i) for i in range(257))
                else:
                    record = fixture()
                    if case == "duplicate_pair":
                        p = record["preferences"][0]
                        record["preferences"].append(dict(p, left=p["right"], right=p["left"]))
                    if case == "tie_label": record["preferences"][1]["chosen"] = "a"
                    if case == "unavailable_action": record["preferences"][0]["left"] = "unknown"
                    if case == "foreign_record_scope": record["scope"] = "development"
                    if case == "pair_count": record["preferences"] = [record["preferences"][0]]*529
                    args = inputs([("r", record)])
                    if case == "partial_json": args["records"]["r"] = b'{"version":'
                    if case == "duplicate_json_key": args["records"]["r"] = args["records"]["r"].replace(b'{', b'{"scope":"artificial",', 1)
                    args["record_pins"]["r"] = sha(args["records"]["r"])
                self.retain(case, args, reject=message)

    def test_purity_exact_limits_and_each_roster_record_validated_once(self):
        args = inputs([("r", fixture())])
        py_before, np_before = random.getstate(), np.random.get_state()
        with patch.object(exposure, "matched_views", wraps=matched_views) as adapter:
            result = self.retain("pure", args)
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(random.getstate(), py_before)
        np_after = np.random.get_state()
        self.assertEqual(np_before[0], np_after[0])
        np.testing.assert_array_equal(np_before[1], np_after[1])
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertEqual(result["bounds"], {"records": 256, "record_bytes": 1048576, "total_record_bytes": 67108864,
            "pairs_per_record": 528, "sft_steps": 256, "dpo_steps": 256, "total_steps": 512})
        self.assertNotIn("llm_intervention_training_engine", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertFalse(result["equal_compute_claimed"])
        self.assertFalse(result["equal_parameter_count_claimed"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_llm_intervention_training_exposure.py NEW_FIXTURE_DIRECTORY")
    ExposureTests.root = Path(sys.argv.pop(1))
    ExposureTests.root.mkdir(exist_ok=False)
    unittest.main(verbosity=2)
