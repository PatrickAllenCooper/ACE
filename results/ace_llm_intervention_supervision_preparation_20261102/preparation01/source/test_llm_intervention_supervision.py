"""Artificial record alignment checks, no policy training or scientific worlds."""
import copy
import hashlib
import json
import unittest

from llm_intervention_contract import canonical
from llm_intervention_supervision import VERSION, matched_views
from test_llm_intervention_task import example


def record():
    return {"version": VERSION, "scope": "artificial", "world_id": 0,
        "envelope": example(), "supervised_action_id": "high",
        "preferences": [{"left": "low", "right": "high", "tie": False, "chosen": "high", "rejected": "low"},
                        {"left": "high", "right": None, "tie": True, "chosen": None, "rejected": None}],
        "supervision_receipt_sha256": "a"*64}


def views(r, **overrides):
    raw = canonical(r).encode()
    return matched_views(raw, **{"record_sha256": hashlib.sha256(raw).hexdigest(),
        "supervision_receipt_sha256": "a"*64, "allowed_scope": "artificial", **overrides})


class SupervisionTests(unittest.TestCase):
    def test_exact_state_action_alignment_and_ties_retained(self):
        r = record()
        result = views(r)
        ids = result["alignment"]["action_ids"]
        self.assertEqual(ids[result["numeric"]["chosen"]], "high")
        self.assertEqual(result["numeric"]["pairs"], ((0, 1),))
        self.assertEqual(result["language"]["preference_completions"],
                         (('{"action_id":"high"}', '{"action_id":"low"}'),))
        self.assertIn(canonical(r["envelope"]), result["language"]["prompt"])
        self.assertNotIn('supervision_receipt_sha256', result["language"]["prompt"])
        self.assertEqual(result["alignment"]["tied_pairs"], [["high", None]])
        self.assertEqual(result["language"]["supervised_completion"], '{"action_id":"high"}')

    def test_wrong_pins_scope_and_unavailable_labels_rejected(self):
        for changes in ({"record_sha256": "b"*64}, {"supervision_receipt_sha256": "b"*64},
                        {"allowed_scope": "policy_train"}, {"allowed_scope": "development"}):
            with self.assertRaises(ValueError):
                views(record(), **changes)
        for name in ("development", "confirmation"):
            r = record()
            r["scope"] = name
            with self.assertRaises(ValueError):
                views(r)
        r = record()
        r["envelope"]["state"]["remaining_units"] = 1
        with self.assertRaises(ValueError):
            views(r)

    def test_no_label_repair_duplicate_or_private_field(self):
        for mutation in (lambda r: r["preferences"].append(copy.deepcopy(r["preferences"][0])),
                         lambda r: r["preferences"][0].update(chosen=None),
                         lambda r: r["preferences"][1].update(chosen="high"),
                         lambda r: r["envelope"]["task"].update(private_score=1.)):
            r = record()
            mutation(r)
            with self.assertRaises(ValueError):
                views(r)
        r = record()
        raw = canonical(r).replace('"world_id":0', '"world_id":0,"world_id":1').encode()
        with self.assertRaises(ValueError):
            matched_views(raw, record_sha256=hashlib.sha256(raw).hexdigest(),
                          supervision_receipt_sha256="a"*64, allowed_scope="artificial")


if __name__ == "__main__":
    unittest.main()
