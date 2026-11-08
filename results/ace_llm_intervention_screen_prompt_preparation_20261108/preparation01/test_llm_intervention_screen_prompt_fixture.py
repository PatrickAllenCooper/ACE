"""Run directly with unittest; bypass the repository's unrelated conftest.

ACE_SCREEN_PROMPT_FIXTURE_TEST_OUTPUT must name a new external directory. All
written packets and intentionally rejected inputs remain there after the run.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "tokenizers"}:
            raise AssertionError("model/tokenizer import forbidden in pure fixture tests: " + fullname)


sys.meta_path.insert(0, NoModelImports())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import llm_intervention_screen_prompt_fixture as fixture
import llm_intervention_world as world
from llm_intervention_contract import ChargedExecutor, canonical, digest, parse_completion
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import task_prompt, unpack_envelope


class PromptFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = Path(os.environ["ACE_SCREEN_PROMPT_FIXTURE_TEST_OUTPUT"])
        cls.output.mkdir(parents=True, exist_ok=False)
        cls.pins = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                    for p in fixture.REQUIRED_SOURCE_PATHS}
        with ExitStack() as stack:
            for owner, name in ((world, "generate_world"), (world, "keyed_seed"),
                    (world, "private_structural_means"), (world.WorldDescriptor, "__init__"),
                    (world.ResponseAdapter, "__init__"), (ChargedExecutor, "__init__"),
                    (LinearGaussianSCM, "update"), (LinearGaussianSCM, "simulate")):
                stack.enter_context(patch.object(owner, name,
                    side_effect=AssertionError("scientific execution forbidden: " + name)))
            cls.packet = fixture.build_fixtures(source_pins=cls.pins)
        (cls.output/"fixture-packet.json").write_text(canonical(cls.packet)+"\n")

    def test_complete_public_design_cross_product(self):
        rows = self.packet["fixtures"]
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({r["fixture_id"] for r in rows}), 120)
        self.assertEqual({(r["graph_shape"], r["cost_menu"], r["noise_variance"],
                           r["numeric_profile"], r["history_shape"]) for r in rows},
            {(g, c, n, p, s) for g in ("chain", "fork", "collider")
             for c in ("uniform", "heterogeneous") for n in (.25, 1.)
             for p in fixture.PROFILES for s in fixture.SHAPES})
        expected_parents = {"chain": [[], ["X"], ["Y"]], "fork": [[], ["X"], ["X"]],
                            "collider": [[], [], ["X", "Y"]]}
        for r in rows:
            state = r["envelope"]["state"]
            self.assertEqual([n["parents"] for n in state["graph"]], expected_parents[r["graph_shape"]])
            self.assertEqual([a["cost_units"] for a in state["menu"]],
                             [1]*8 if r["cost_menu"] == "uniform" else [1]*4+[2]*4)
            self.assertEqual(r["envelope"]["task"]["weights"], [.125]*8)
            self.assertEqual(len(r["envelope"]["task"]["queries"]), 8)

    def test_budget_history_clamps_and_eligibility(self):
        for row in self.packet["fixtures"]:
            state, _, graph, _, _ = unpack_envelope(row["envelope"])
            history = state["history"]
            self.assertLessEqual(len(history), 32)
            self.assertEqual([r["action_id"] for r in history[:8]], ["observe_all"]*8)
            self.assertEqual([r["request_id"] for r in history], list(range(len(history))))
            self.assertEqual(sum(r["cost_units"] for r in history), state["spent_units"])
            self.assertEqual(state["spent_units"]+state["remaining_units"], 32)
            for receipt in history:
                for name, value in receipt["targets"].items():
                    self.assertEqual(receipt["values"][name], value)
            for node, summary in zip(graph, state["scm"]):
                self.assertEqual(summary["eligible_rows"],
                                 sum(node.name not in r["targets"] for r in history))
            if row["history_shape"] == "full_history_terminal":
                self.assertEqual((len(history), state["remaining_units"], state["terminal"]), (32, 0, True))
            if row["history_shape"] == "mixed_actions":
                self.assertEqual({r["action_id"] for r in history}, {a["action_id"] for a in state["menu"]})

    def test_every_legal_command_and_stop(self):
        for row in self.packet["fixtures"]:
            state, _, _, menu, _ = unpack_envelope(row["envelope"])
            commands = row["legal_commands"]
            self.assertEqual(commands[-1], '{"stop":true}')
            legal = {a.action_id for a in menu if not state["terminal"]
                     and a.cost_units <= state["remaining_units"]}
            self.assertEqual({parse_completion(c, menu).action_id for c in commands}, legal | {None})
            if row["history_shape"] == "full_history_terminal":
                self.assertEqual(commands, ['{"stop":true}'])
            else:
                self.assertEqual(len(commands), 9)
            if row["history_shape"] == "last_all_actions":
                self.assertEqual(row["history_rows"], 31 if row["cost_menu"] == "uniform" else 30)

    def test_exact_untruncated_prompts_and_lengths(self):
        for row in self.packet["fixtures"]:
            self.assertEqual(row["prompt"], task_prompt(row["envelope"]))
            self.assertIn(canonical(row["envelope"]), row["prompt"])
            self.assertEqual(row["envelope_sha256"], digest(row["envelope"]))
            self.assertEqual(row["prompt_sha256"], hashlib.sha256(row["prompt"].encode()).hexdigest())
            self.assertEqual(row["prompt_utf8_bytes"], len(row["prompt"].encode()))
            self.assertEqual(row["envelope_utf8_bytes"], len(canonical(row["envelope"]).encode()))

    def test_numeric_spelling_profiles_and_dense_covariance(self):
        spellings = self.packet["numeric_spellings"]
        self.assertEqual(spellings["longest_corpus_characters"], max(map(len, spellings["corpus"])))
        for row in self.packet["fixtures"]:
            state = row["envelope"]["state"]
            if row["numeric_profile"] == "finite_edges":
                observed = {canonical(v) for r in state["history"] for v in r["values"].values()}
                self.assertTrue({canonical(v) for v in fixture.FINITE_VALUES} <= observed)
            else:
                for receipt in state["history"]:
                    for name, value in receipt["values"].items():
                        if name not in receipt["targets"]:
                            self.assertEqual(value, fixture.LONG_VALUE)
            for summary in state["scm"]:
                matrix = summary["parameter_covariance"]
                for i, values in enumerate(matrix):
                    self.assertGreater(values[i], sum(abs(v) for j, v in enumerate(values) if i != j))
                    self.assertTrue(all(v != 0 for v in values))

    def test_no_private_fields_or_execution_claims(self):
        for row in self.packet["fixtures"]:
            text = canonical(row["envelope"])
            for key in ('"world_id"', '"scope"', '"truth_law"', '"seed"', '"regime"',
                        '"descriptor_sha256"', '"private_evaluation"', '"supervised_command"'):
                self.assertNotIn(key, text)
        self.assertEqual(self.packet["environment_responses"], 0)
        self.assertEqual(self.packet["scientific_worlds_generated"], 0)
        self.assertFalse(self.packet["actual_tokenization_performed"])
        self.assertFalse(self.packet["universal_token_bound_claimed"])
        self.assertFalse(self.packet["scientific_execution_authorized"])
        self.assertFalse({"torch", "transformers", "tokenizers"}.intersection(sys.modules))
        qualification = self.packet["future_cpu_qualification"]
        self.assertEqual(qualification["arguments"], {"max_tokens": 8192, "max_new_tokens": 32})
        self.assertEqual(qualification["gpus"], 0)

    def test_invalid_history_and_covariance_are_rejected(self):
        bad = copy.deepcopy(self.packet["fixtures"][0]["envelope"])
        bad["state"]["history"][0]["request_id"] = 1
        (self.output/"rejected-history.json").write_text(canonical(bad)+"\n")
        with self.assertRaises(ValueError):
            task_prompt(bad)
        bad = copy.deepcopy(self.packet["fixtures"][0]["envelope"])
        bad["state"]["scm"][0]["parameter_covariance"][0][0] = -1.
        (self.output/"rejected-covariance.json").write_text(canonical(bad)+"\n")
        with self.assertRaises(ValueError):
            task_prompt(bad)

    def test_exclusive_packet_files_match_manifest(self):
        destination = self.output/"written-packet"
        manifest = fixture.write_packet(destination, self.packet)
        self.assertEqual(json.loads((destination/"manifest.json").read_bytes()), manifest)
        for row in manifest["fixtures"]:
            for key in ("envelope_file", "prompt_file"):
                record = row[key]
                raw = (destination/record["path"]).read_bytes()
                self.assertEqual(len(raw), record["utf8_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            self.assertEqual(row["envelope_file"]["utf8_bytes"], row["envelope_utf8_bytes"]+1)
        with self.assertRaises(FileExistsError):
            fixture.write_packet(destination, self.packet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
