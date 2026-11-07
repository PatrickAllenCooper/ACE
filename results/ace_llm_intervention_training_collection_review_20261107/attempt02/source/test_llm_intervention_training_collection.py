"""NEW artificial collection fixtures; no existing suite, scientific draw or model."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from llm_intervention_contract import Action, Node, digest
from llm_intervention_replicated_branches import _save
from llm_intervention_screen_protocol import REQUIRED_SOURCE_PATHS, assembly_recipe
from llm_intervention_task import PredictionTask
import llm_intervention_training_collection as collection
from llm_intervention_training_dataset import Snapshot
from llm_intervention_training_exposure import propose_exposure
from llm_intervention_world import LinearMechanism, ResponseAdapter, WorldDescriptor


GRAPH = (Node("X", (), "unit", (-2., 2.)), Node("Y", ("X",), "unit"))
MENU = (Action("a", "intervene", (("X", 0.),), ("Y",), 1),
        Action("b", "intervene", (("X", 0.),), ("Y",), 2),
        Action("c", "intervene", (("X", 0.),), ("Y",), 1))
TASK = PredictionTask("Y", (Action("q", "intervene", (("X", 0.),), ("Y",), 1),), (1.,), 2., .04)
ROSTER = (17, 4)  # Deliberately unsorted; no input mapping determines execution order.


def read(path):
    return json.loads(path.read_text())


def fixture(output):
    # Two directly supplied hand-written constant laws. No generate_world call.
    worlds = {wid: WorldDescriptor("artificial", wid, GRAPH,
        (LinearMechanism("X", 0., (), 0.), LinearMechanism("Y", value, (("X", 0.),), 0.)))
        for wid, value in ((4, -3.), (17, 2.))}
    configs = {}
    for wid in reversed(ROSTER):
        configs[wid] = {"execution_authorized": False, "runtime_bindings_pending": collection.RUNTIME_BINDINGS,
            "learner_parameters": {"graph": GRAPH, "noise_variances": {"X": 1., "Y": 1.}, "prior_variance": 1.},
            "assembly_parameters": {"scope": "artificial", "graph": GRAPH, "menu": MENU, "task": TASK,
                "expected_task_sha256": digest(TASK.payload()), "initial_actions": ("a",), "total_budget": 3,
                "behavior_method": "menu_coverage", "snapshots": (
                    Snapshot(0, "menu_coverage", ("a", "b", "c")), Snapshot(2, "uniform_random", ())),
                "replicate_ids": ("r0", "r1"), "planning_draws": 2, "tie_multiplier": 2., "minimum_gain_gap": 0.,
                "structural_scores": True, "objective": "Predict this artificial target.", "domain_notes": "",
                "request_namespace": f"artificial-collection-{wid}"}}
    sources = {p: collection._pin(Path(__file__).parent/Path(p).name) for p in collection.SOURCE_PATHS}
    return dict(scope="artificial", world_roster=ROSTER, expected_roster_sha256=digest(ROSTER), worlds=worlds,
        world_pins={w: worlds[w].sha256 for w in ROSTER}, configurations=configs,
        configuration_pins={w: digest(collection.configuration_payload(configs[w])) for w in ROSTER},
        expected_source_pins=sources, output=output)


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(sys.argv[1])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def params(self, name="collection"):
        return fixture(self.directory/name)

    def repin(self, p, wid=4):
        p["configuration_pins"][wid] = digest(collection.configuration_payload(p["configurations"][wid]))

    def reject(self, p, match):
        with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no response permitted")) as respond:
            with self.assertRaisesRegex(ValueError, match):
                collection.collect_training_batch(**p)
            self.assertEqual(respond.call_count, 0)
        failure = read(p["output"]/"failure.json")
        self.assertFalse(failure["exposure_ready"])
        self.assertEqual(failure["accounting"]["reserved_response_attempts"], 0)
        return failure

    def test_two_world_complete_inventory_drives_exposure_and_exclusive_intent(self):
        p, observations = self.params(), []
        original = ResponseAdapter.__call__

        def audited(adapter, action, request):
            plan = read(p["output"]/"plan.json")
            self.assertEqual(plan["world_roster"], list(ROSTER))
            self.assertEqual(len(plan["records"]), 4)
            self.assertEqual(sum(len(r["pairs"]) for r in plan["records"]), 6)
            self.assertTrue((p["output"]/"input-001.json").exists())
            self.assertNotEqual(adapter._stream, "private_eval")
            observations.append(adapter._world.world_id)
            return original(adapter, action, request)

        with patch.object(ResponseAdapter, "__call__", audited):
            result = collection.collect_training_batch(**p)
        _save(self.directory/"response-order.json", observations)
        self.assertEqual([r["status"] for r in result["worlds"]], ["complete", "complete"])
        # Per world: initial=1, acquisition=2; six branches cost8, two scores cost2.
        self.assertEqual(observations, [17]*10+[4]*10)
        self.assertEqual(result["accounting"], {"total_reserved_cost_units": 26,
            "reserved_response_attempts": 20, "known_returns": 20,
            "executor_accepted_responses": 20, "unknown_return_attempts": 0})
        inventory = read(p["output"]/"inventory.json")
        expected_ids = [f"artificial/{w}/snapshot-{i:03d}" for w in ROSTER for i in range(2)]
        self.assertEqual(inventory["record_roster"], expected_ids)
        records = {r["record_id"]: (p["output"]/r["path"]/"record.json").read_bytes() for r in inventory["records"]}
        exposure = propose_exposure(records=records, record_pins=inventory["record_pins"],
            supervision_receipt_pins=inventory["supervision_receipt_pins"],
            record_roster=tuple(inventory["record_roster"]), scope="artificial")
        _save(self.directory/"exposure-proposal.json", exposure)
        self.assertEqual(exposure["counts"]["retained_pairs"], 6)
        self.assertEqual(exposure["counts"]["tied_pairs"], 2)
        self.assertEqual((exposure["counts"]["sft_steps"], exposure["counts"]["dpo_steps"]), (4, 2))
        before = {str(f): collection._pin(f) for f in p["output"].rglob("*") if f.is_file()}
        with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no repeat")):
            with self.assertRaises(FileExistsError):
                collection.collect_training_batch(**p)
        self.assertEqual(before, {str(f): collection._pin(f) for f in p["output"].rglob("*") if f.is_file()})

    def test_late_descriptor_config_and_source_mismatches_precede_every_response(self):
        for name in ("descriptor", "config", "source", "roster"):
            with self.subTest(name=name):
                p = self.params(name)
                if name == "descriptor":
                    p["world_pins"][4] = "0"*64
                elif name == "config":
                    p["configurations"][4]["assembly_parameters"]["planning_draws"] = 3
                elif name == "source":
                    p["expected_source_pins"][collection.SOURCE_PATHS[0]] = "0"*64
                else:
                    p["expected_roster_sha256"] = digest(tuple(reversed(ROSTER)))
                self.reject(p, "pin")

    def test_authenticated_but_invalid_late_recipe_fails_whole_preflight(self):
        cases = {"task": lambda a: a.update(expected_task_sha256="0"*64),
            "bounds": lambda a: a.update(total_budget=1000),
            "snapshot": lambda a: a.update(snapshots=(Snapshot(0, "menu_coverage", ("a",)),)),
            "teacher": lambda a: a.update(snapshots=(Snapshot(0, "invented", ("a", "b")),)),
            "replicates": lambda a: a.update(replicate_ids=("r0", "r0")),
            "extra": lambda a: a.update(temperature=.1)}
        for name, change in cases.items():
            with self.subTest(name=name):
                p = self.params(name)
                change(p["configurations"][4]["assembly_parameters"])
                self.repin(p)
                failure = self.reject(p, ".+")
                self.assertEqual([r["status"] for r in failure["worlds"]], ["unstarted", "failed"])
        p = self.params("learner")
        p["configurations"][4]["learner_parameters"]["noise_variances"]["Y"] = 0.
        self.repin(p)
        self.reject(p, "positive noise")

    def test_failure_keeps_complete_world_unknown_return_and_no_retry(self):
        p, original = self.params(), ResponseAdapter.__call__

        def interrupt(adapter, action, request):
            value = original(adapter, action, request)
            if adapter._world.world_id == 4 and adapter._stream == "measurement" and adapter._replicate == "trajectory":
                raise RuntimeError("artificial prepared outcome interrupted before return")
            return value

        with patch.object(ResponseAdapter, "__call__", interrupt):
            with self.assertRaisesRegex(RuntimeError, "before return"):
                collection.collect_training_batch(**p)
        failure = read(p["output"]/"failure.json")
        self.assertEqual([r["status"] for r in failure["worlds"]], ["complete", "failed"])
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 26)
        self.assertEqual(failure["accounting"]["unknown_return_attempts"], 1)
        self.assertEqual(failure["worlds"][1]["accounting"]["trajectory"]["unknown_return_reserved_cost"], 2)
        self.assertEqual(len(failure["partial_record_inventory"]), 4)
        self.assertEqual(sum(r["status"] == "complete" for r in failure["partial_record_inventory"]), 3)
        self.assertFalse((p["output"]/"inventory.json").exists())
        self.assertFalse(failure["partial_records_authorized_for_training"])
        with self.assertRaises(FileExistsError):
            collection.collect_training_batch(**p)

    def test_returned_invalid_response_keeps_failed_and_unstarted_roster(self):
        p, original = self.params(), ResponseAdapter.__call__

        def invalid(adapter, action, request):
            value = original(adapter, action, request)
            return {"Y": float("nan")} if adapter._stream == "train_score" else value

        with patch.object(ResponseAdapter, "__call__", invalid):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                collection.collect_training_batch(**p)
        failure = read(p["output"]/"failure.json")
        self.assertEqual([r["status"] for r in failure["worlds"]], ["failed", "unstarted"])
        self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 6)
        self.assertEqual(failure["accounting"]["known_returns"], 5)
        self.assertEqual(failure["accounting"]["executor_accepted_responses"], 4)
        self.assertEqual(failure["accounting"]["unknown_return_attempts"], 0)
        self.assertEqual(failure["worlds"][0]["accounting"]["scoring"]["returned_without_acceptance"], 1)
        self.assertEqual(failure["record_roster"], [r["record_id"] for r in failure["partial_record_inventory"]])
        self.assertFalse((p["output"]/"world-001.start.json").exists())
        raw = read(p["output"]/"world-000/snapshot-000/feedback/replicate-000.scoring.return-000.json")
        self.assertIn("nan", raw["raw_response_repr"])

    def test_roster_membership_and_scientific_scope_reject_without_execution(self):
        for name in ("missing", "extra", "duplicate", "scope", "policy_train"):
            p = self.params(name)
            if name == "missing":
                del p["configurations"][4]
            elif name == "extra":
                p["world_pins"][99] = "0"*64
            elif name == "duplicate":
                p["world_roster"] = (17, 17)
            else:
                p["scope"] = "development" if name == "scope" else "policy_train"
            _save(self.directory/(name+"-input.json"), {"scope": p["scope"], "roster": p["world_roster"],
                "world_pins": {str(k): v for k, v in p["world_pins"].items()},
                "configuration_ids": list(p["configurations"])})
            with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no responses")) as respond:
                with self.assertRaises(ValueError) as caught:
                    collection.collect_training_batch(**p)
                self.assertEqual(respond.call_count, 0)
            _save(self.directory/(name+"-rejection.json"), {"error": str(caught.exception)})

    def test_current_proposal_adoption_rejects_tuning_subsets_and_source_substitution(self):
        # Read existing prospective identities only; no scientific descriptor is constructed.
        proposal = read(Path(__file__).parent/"prospective-proposal.json")
        p = self.params()
        sources = p["expected_source_pins"]
        protocol_pins = {key: sources[key] for key in REQUIRED_SOURCE_PATHS}
        roster = tuple(r["world_id"] for r in proposal["training_worlds"])
        configs = {wid: assembly_recipe(proposal, world_id=wid, expected_source_pins=protocol_pins) for wid in roster}
        collection._adopt_recipes("policy_train", roster, configs, proposal, digest(proposal), sources)
        rejected = []
        for name in ("tuning", "subset", "source", "proposal", "pin"):
            changed, order, plan, pins, pin = deepcopy(configs), roster, deepcopy(proposal), dict(sources), digest(proposal)
            if name == "tuning":
                changed[roster[-1]]["assembly_parameters"]["planning_draws"] = 2
            elif name == "subset":
                order = roster[:-1]
            elif name == "source":
                pins[REQUIRED_SOURCE_PATHS[0]] = "0"*64
            elif name == "proposal":
                plan["training_worlds"][-1]["training"]["planning_draws"] = 2
                pin = digest(plan)
            else:
                pin = "0"*64
            with self.assertRaises(ValueError) as caught:
                collection._adopt_recipes("policy_train", order, changed, plan, pin, pins)
            rejected.append({"case": name, "error": str(caught.exception)})
        _save(self.directory/"recipe-admission.json", {"exact_current_recipes": len(configs), "rejected": rejected,
            "proposal_sha256": digest(proposal), "scientific_descriptors_created": 0, "responses": 0})


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *sys.argv[2:]], verbosity=2)
