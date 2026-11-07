"""NEW artificial collection fixtures; no existing suite, scientific draw or model."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
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


class CollectionCorrectionTests(unittest.TestCase):
    """Focused correction controls only; saved responses are copied, never redrawn."""
    @classmethod
    def setUpClass(cls):
        cls.root = Path(sys.argv[1])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def replay(self, **parameters):
        source = Path(os.environ["ACE_COLLECTION_REPLAY_INPUT"])/f"world-{ROSTER.index(parameters['world'].world_id):03d}"
        shutil.copytree(source, parameters["output"])
        _save(parameters["output"]/"replay-origin.json", {"source": str(source),
            "terminal_sha256": collection._pin(source/"terminal.json"), "new_responses": 0})
        return read(source/"terminal.json")

    def test_publication_candidates_exclusive_commit_and_interruption(self):
        original_save, original_link = collection._save, os.link
        value = {"status": "complete", "exposure_ready": True, "artificial_control": True}
        failure = {"status": "failed", "exposure_ready": False}
        for case in ("write", "fsync", "link", "postcommit", "existing", "success"):
            directory = self.directory/case
            directory.mkdir()

            def save(path, payload):
                if case == "write" and path.name == "terminal-candidate.json":
                    with path.open("x") as handle:
                        handle.write('{"status":')
                    raise OSError("injected write")
                if case == "fsync" and path.name == "terminal-candidate.json":
                    with patch.object(os, "fsync", side_effect=OSError("injected fsync")):
                        original_save(path, payload)
                else:
                    original_save(path, payload)

            def link(*args):
                if case == "link":
                    raise OSError("injected link")
                original_link(*args)
                if case == "postcommit":
                    raise KeyboardInterrupt("injected after commit")

            if case == "existing":
                (directory/"terminal.json").write_bytes(b'original exclusive terminal\n')
            with patch.object(collection, "_save", save), patch.object(os, "link", link):
                if case == "success":
                    collection._publish_terminal(directory, value)
                else:
                    with self.assertRaises((OSError, KeyboardInterrupt)):
                        collection._publish_terminal(directory, value)
                    collection._failure_receipt(directory, failure)
            if case in ("write", "fsync", "link"):
                self.assertFalse((directory/"terminal.json").exists())
                self.assertEqual(read(directory/"failure.json"), failure)
            elif case in ("existing", "postcommit"):
                self.assertFalse((directory/"failure.json").exists())
                self.assertEqual(read(directory/"publication-interruption.json")["status"],
                                 "publication_interrupted_reconciliation_required")
                if case == "existing":
                    self.assertEqual((directory/"terminal.json").read_bytes(), b'original exclusive terminal\n')
            else:
                self.assertTrue(os.path.samefile(directory/"terminal.json", directory/"terminal-candidate.json"))
                self.assertEqual(read(directory/"terminal.json"), value)
                self.assertFalse((directory/"failure.json").exists())

    def test_saved_collection_replay_publication_gate_without_responses(self):
        original_save, original_link = collection._save, os.link
        for case in ("candidate_fsync", "after_commit"):
            p = fixture(self.directory/case)

            def save(path, payload):
                if case == "candidate_fsync" and path.name == "terminal-candidate.json":
                    with patch.object(os, "fsync", side_effect=OSError("injected candidate fsync")):
                        original_save(path, payload)
                else:
                    original_save(path, payload)

            def link(*args):
                original_link(*args)
                if case == "after_commit":
                    raise KeyboardInterrupt("injected publication interruption")

            with patch.object(collection, "assemble_training_dataset", self.replay), \
                    patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no new responses")) as respond, \
                    patch.object(collection, "_save", save), patch.object(os, "link", link):
                with self.assertRaises((OSError, KeyboardInterrupt)):
                    collection.collect_training_batch(**p)
                self.assertEqual(respond.call_count, 0)
            self.assertTrue((p["output"]/"inventory.json").exists())  # Provisional evidence, gated by disposition.
            if case == "candidate_fsync":
                self.assertFalse((p["output"]/"terminal.json").exists())
                self.assertFalse(read(p["output"]/"failure.json")["exposure_ready"])
            else:
                self.assertFalse((p["output"]/"failure.json").exists())
                diagnostic = read(p["output"]/"publication-interruption.json")
                self.assertFalse(diagnostic["exposure_ready"])
                self.assertEqual(diagnostic["status"], "publication_interrupted_reconciliation_required")

    def test_unreadable_plan_and_reconstruction_errors_keep_primary_failure(self):
        original_accounting = collection._accounting
        for case in ("truncated_plan", "accounting_error"):
            p = fixture(self.directory/case)

            def fail(**parameters):
                self.replay(**parameters)
                plan = parameters["output"]/"plan.json"
                shutil.copyfile(plan, parameters["output"]/"plan-before-injection.json")
                plan.write_text('{"journal_plan":')
                raise OSError("original injected plan write failure")

            def accounting(directory, journals):
                if case == "accounting_error" and directory.name == "world-000":
                    raise OSError("injected reconstruction failure")
                return original_accounting(directory, journals)

            with patch.object(collection, "assemble_training_dataset", fail), \
                    patch.object(collection, "_accounting", accounting), \
                    patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no responses")) as respond:
                with self.assertRaisesRegex(OSError, "original injected plan write failure"):
                    collection.collect_training_batch(**p)
                self.assertEqual(respond.call_count, 0)
            failure = read(p["output"]/"failure.json")
            self.assertEqual(failure["exception_type"], "OSError")
            self.assertEqual(failure["message"], "original injected plan write failure")
            self.assertEqual([r["status"] for r in failure["worlds"]], ["failed", "unstarted"])
            self.assertFalse(failure["accounting_complete"])
            self.assertTrue(failure["accounting_totals_are_lower_bounds"])
            self.assertTrue(failure["worlds"][0]["accounting"]["counts_are_lower_bounds"])
            self.assertFalse(failure["exposure_ready"])
            self.assertFalse((p["output"]/"inventory.json").exists())
            self.assertEqual(failure["accounting"]["total_reserved_cost_units"], 13 if case == "truncated_plan" else 0)
            self.assertEqual(failure["accounting"]["known_returns"], 10 if case == "truncated_plan" else 0)
            self.assertEqual(len(failure["planned_records"]), 4)
            self.assertIn("inventory_error", failure)

    def test_late_artificial_action_and_query_name_bounds_precede_responses(self):
        for case in ("menu", "query"):
            p = fixture(self.directory/case)
            a = p["configurations"][4]["assembly_parameters"]
            if case == "menu":
                a["menu"] = (MENU[0], replace(MENU[1], action_id="x"*257), MENU[2])
                a["snapshots"] = (replace(a["snapshots"][0], candidates=("a", "x"*257, "c")), a["snapshots"][1])
            else:
                a["task"] = replace(TASK, queries=(replace(TASK.queries[0], action_id="x"*257),))
                a["expected_task_sha256"] = digest(a["task"].payload())
            p["configuration_pins"][4] = digest(collection.configuration_payload(p["configurations"][4]))
            with patch.object(ResponseAdapter, "__call__", side_effect=AssertionError("no responses")) as respond:
                with self.assertRaisesRegex(ValueError, "bounded immutable action menu"):
                    collection.collect_training_batch(**p)
                self.assertEqual(respond.call_count, 0)
            failure = read(p["output"]/"failure.json")
            self.assertEqual([r["status"] for r in failure["worlds"]], ["unstarted", "failed"])
            self.assertEqual(failure["accounting"]["reserved_response_attempts"], 0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *sys.argv[2:]], verbosity=2)
