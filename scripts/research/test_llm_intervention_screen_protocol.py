"""Only pure proposal/identity tests; no scientific draws or responses."""
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

from llm_intervention_contract import canonical, digest, validate_design
import llm_intervention_scm as scm
import llm_intervention_training_dataset as dataset
import llm_intervention_world as world
from llm_intervention_screen_protocol import (
    ARMS, COSTS, GRAPHS, METHODS, MILESTONES, REGIMES, REQUIRED_SOURCE_PATHS,
    assembly_recipe, build_proposal, public_design, validate_proposal,
)


def save(path, value, *, allow_nan=False):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=allow_nan)
        handle.write("\n")


class ScreenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["ACE_SCREEN_PROTOCOL_TEST_OUTPUT"])
        cls.root.mkdir(exist_ok=False)
        repo = Path(__file__).resolve().parents[2]
        cls.pins = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in REQUIRED_SOURCE_PATHS}
        cls.proposal = build_proposal(source_pins=cls.pins)
        save(cls.root/"prospective-proposal.json", cls.proposal)
        save(cls.root/"declared-source-pins.json", cls.pins)

    def setUp(self):
        self.directory = self.root/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def test_exact_identities_regime_graph_cost_and_assignment_balance(self):
        p = self.proposal
        self.assertEqual(validate_proposal(p, expected_source_pins=self.pins), digest(p))
        train, dev = p["training_worlds"], p["development_worlds"]
        self.assertEqual((len(train), len(dev)), (32, 24))
        self.assertEqual(len({(r["scope"], r["world_id"]) for r in train+dev}), 56)
        self.assertEqual(len({r["world_id"] for r in train+dev}), 56)
        self.assertTrue(all(not r["generated"] for r in train+dev))
        self.assertEqual(Counter(r["graph_stratum"] for r in train), {"chain": 12, "fork": 10, "collider": 10})
        self.assertEqual(Counter(r["cost_stratum"] for r in train), {c: 16 for c in COSTS})
        self.assertEqual(Counter(r["training"]["behavior_method"] for r in train), {m: 8 for m in METHODS})
        counts = []
        for regime, _, _ in REGIMES:
            subset = [r for r in train if r["regime"] == regime]
            self.assertEqual(len(subset), 8)
            self.assertEqual(Counter(r["training"]["behavior_method"] for r in subset), {m: 2 for m in METHODS})
            self.assertEqual(Counter(r["cost_stratum"] for r in subset), {c: 4 for c in COSTS})
            d = [r for r in dev if r["regime"] == regime]
            self.assertEqual(Counter((r["graph_stratum"], r["cost_stratum"]) for r in d),
                             {(g, c): 1 for g in GRAPHS for c in COSTS})
            for si in range(5):
                self.assertEqual(Counter(r["training"]["snapshots"][si]["teacher_method"] for r in subset), {m: 2 for m in METHODS})
        for cost in COSTS:
            self.assertEqual(Counter(r["training"]["behavior_method"] for r in train if r["cost_stratum"] == cost), {m: 4 for m in METHODS})
        for si in range(5):
            pairs = Counter((r["training"]["behavior_method"], r["training"]["snapshots"][si]["teacher_method"]) for r in train)
            self.assertEqual(pairs, {(b, t): 2 for b in METHODS for t in METHODS})
            counts.append({"snapshot": si, "behavior_teacher_counts": [[b, t, n] for (b, t), n in sorted(pairs.items())]})
        for rows in (train, dev):
            for pair in {r["stratum_pair"] for r in rows}:
                members = [r for r in rows if r["stratum_pair"] == pair]
                self.assertEqual(len(members), 2)
                self.assertEqual({r["cost_stratum"] for r in members}, set(COSTS))
                self.assertEqual(len({r["graph_stratum"] for r in members}), 1)
        save(self.directory/"balance-oracle.json", counts)

    def test_exhaustive_cost_crossing_feasibility_and_typed_interface_bounds(self):
        crossing = {m: set() for m in MILESTONES}
        crossing[0].add(0)
        # Exhaust the finite cost-state graph, not environment trajectories.
        for spent in range(24):
            for cost in (1, 2):
                after = spent+cost
                if after > 24:
                    continue
                for milestone in MILESTONES[1:]:
                    if spent < milestone <= after:
                        crossing[milestone].add(after)
        self.assertEqual(crossing, {0: {0}, 6: {6, 7}, 12: {12, 13}, 18: {18, 19}, 24: {24}})
        for threshold in MILESTONES[:-1]:
            self.assertGreaterEqual(min(24-c for c in crossing[threshold]), 2)
        self.assertEqual(self.proposal["budget_basis"]["crossing_upper_costs"], [max(crossing[m]) for m in MILESTONES])
        for row in self.proposal["training_worlds"]:
            recipe = assembly_recipe(self.proposal, world_id=row["world_id"], expected_source_pins=self.pins)
            params = recipe["assembly_parameters"]
            validate_design(params["graph"], params["menu"])
            params["task"].validate(params["graph"])
            self.assertEqual((len(params["menu"]), len(params["task"].queries)), (8, 8))
            self.assertEqual(params["task"].weights, (.125,)*8)
            self.assertEqual(sum(params["task"].weights), 1.)
            self.assertEqual(params["initial_actions"], ("observe_all",)*8)
            self.assertEqual(params["menu"][0].cost_units, 1)
            self.assertEqual(max(a.cost_units for a in params["menu"]), 1 if row["cost_stratum"] == "uniform" else 2)
            self.assertEqual([s.acquisition_cost_units for s in params["snapshots"]], list(MILESTONES))
            self.assertTrue(all(len(s.candidates) == 8 for s in params["snapshots"][:-1]))
            self.assertEqual(params["snapshots"][-1].candidates, ())
            self.assertEqual(params["total_budget"], 32)
            self.assertEqual(params["expected_task_sha256"], digest(params["task"].payload()))
            signature = set(inspect.signature(dataset.assemble_training_dataset).parameters)
            self.assertEqual(set(params) | set(recipe["runtime_bindings_pending"]), signature)
            self.assertFalse(set(params) & set(recipe["runtime_bindings_pending"]))
            self.assertFalse(recipe["execution_authorized"])
        save(self.directory/"cost-state-oracle.json", {str(k): sorted(v) for k, v in crossing.items()})

    def test_namespaces_pair_initials_and_exclude_development_feedback(self):
        addresses = set()
        for row in self.proposal["training_worlds"]:
            t = row["training"]
            for i, snapshot in enumerate(t["snapshots"]):
                self.assertEqual(len(snapshot["counterfactual_bindings"]), 4 if i < 4 else 0)
                self.assertEqual(len(snapshot["score_bindings"]), 4 if i < 4 else 0)
                for b, s in zip(snapshot["counterfactual_bindings"], snapshot["score_bindings"]):
                    self.assertEqual(b["request_namespace"], digest({"namespace": t["request_namespace"], "snapshot_index": i}))
                    self.assertNotEqual(b["request_namespace"], t["request_namespace"])
                    self.assertEqual((b["scope"], b["stream"], s["scope"], s["stream"]),
                                     ("policy_train", "measurement", "policy_train", "train_score"))
                    for binding in (b, s):
                        key = canonical(binding)
                        self.assertNotIn(key, addresses)
                        addresses.add(key)
            self.assertNotIn(canonical(row["initial_binding"]), addresses)
            self.assertNotIn(canonical(row["private_evaluation"]["binding"]), addresses)
        self.assertEqual(len(addresses), 32*4*4*2)
        for row in self.proposal["development_worlds"]:
            self.assertIsNone(row["training"])
            episodes = row["development_episodes"]
            self.assertEqual([e["arm"] for e in episodes], list(ARMS))
            self.assertTrue(all(e["initial_binding"] == row["initial_binding"] for e in episodes))
            self.assertEqual(len({canonical(e["measurement_binding"]) for e in episodes}), 8)
            for e in episodes:
                self.assertFalse(e["training_feedback_allowed"])
                self.assertNotIn(canonical(e["measurement_binding"]), addresses)
                self.assertNotEqual(e["measurement_binding"]["request_namespace"], row["initial_binding"]["request_namespace"])
            self.assertNotIn("train_score", canonical(row))
        reservation = self.proposal["confirmation_reservation"]
        self.assertEqual(reservation["identities"], [])
        self.assertIsNone(reservation["sample_size"])
        self.assertIsNone(reservation["families_and_laws"])
        self.assertFalse(reservation["generated"])
        for scope, wid in (("confirmation", reservation["world_id_start_inclusive"]), ("artificial", 0), ("policy_train", True)):
            with self.assertRaises(ValueError):
                public_design(self.proposal, scope=scope, world_id=wid, expected_source_pins=self.pins)
        with self.assertRaises(ValueError):
            assembly_recipe(self.proposal, world_id=self.proposal["development_worlds"][0]["world_id"], expected_source_pins=self.pins)
        save(self.directory/"namespace-counts.json", {"unique_training_feedback_bindings": len(addresses),
            "development_episodes": 192, "confirmation_descriptors": 0})

    def test_no_truth_or_response_draws_and_public_law_separation(self):
        python_state, numpy_state = repr(random.getstate()), repr(np.random.get_state())
        with ExitStack() as stack:
            for module, attr in ((world, "generate_world"), (world, "private_structural_means"),
                                 (world, "WorldDescriptor"), (world, "ResponseAdapter"),
                                 (dataset, "assemble_training_dataset"), (dataset, "ChargedExecutor"),
                                 (scm.LinearGaussianSCM, "__init__"), (random, "Random"),
                                 (np.random, "default_rng"), (np.random, "normal")):
                stack.enter_context(patch.object(module, attr, side_effect=AssertionError("no world/model/RNG/response work allowed")))
            proposal = build_proposal(source_pins=self.pins)
            validate_proposal(proposal, expected_source_pins=self.pins)
            first = proposal["training_worlds"][0]
            public = public_design(proposal, scope=first["scope"], world_id=first["world_id"], expected_source_pins=self.pins)
            assembly_recipe(proposal, world_id=first["world_id"], expected_source_pins=self.pins)
        self.assertEqual((repr(random.getstate()), repr(np.random.get_state())), (python_state, numpy_state))
        self.assertEqual(public, first["public"])
        for key in ("world_id", "regime", "seed", "coefficient_means", "realized_parameters", "truth_law", "descriptor_sha256"):
            self.assertNotIn(key, canonical(public))
        for matched, shifted in (("matched_low", "shifted_low"), ("matched_high", "shifted_high")):
            for graph in GRAPHS:
                a = next(r for r in proposal["development_worlds"] if r["regime"] == matched and r["graph_stratum"] == graph and r["cost_stratum"] == "uniform")
                b = next(r for r in proposal["development_worlds"] if r["regime"] == shifted and r["graph_stratum"] == graph and r["cost_stratum"] == "uniform")
                self.assertEqual(a["public"], b["public"])
                self.assertNotEqual(a["truth_law"]["coefficient_means"], b["truth_law"]["coefficient_means"])
        for row in proposal["training_worlds"]+proposal["development_worlds"]:
            law, prior = row["truth_law"], row["public"]["learner"]
            self.assertIsNone(law["realized_parameters"])
            self.assertIsNone(law["descriptor_sha256"])
            self.assertEqual(law["noise_variances"], list(prior["noise_variances"].values()))
            self.assertEqual((prior["prior_variance"], law["intercept_sd"], law["coefficient_sd"]), (1., 1., 1.))
        public["objective"] = "detached mutation"
        self.assertNotEqual(public, first["public"])
        save(self.directory/"no-execution-receipt.json", {"guarded_generators_executors_rngs_called": 0,
            "global_rng_states_unchanged": True, "public_matched_shifted_designs_equal": True})

    def test_malformed_and_source_pin_rejection_preserves_exact_bad_inputs(self):
        mutations = {
            "missing_identity": lambda p: p["training_worlds"].pop(),
            "duplicate_identity": lambda p: p["development_worlds"][1].update(world_id=p["development_worlds"][0]["world_id"]),
            "cost_above_cap": lambda p: p["training_worlds"][0]["public"]["menu"][0].update(cost_units=3),
            "missing_target": lambda p: p["training_worlds"][0]["public"]["task"]["queries"][0].update(measure=["X"]),
            "unequal_weights": lambda p: p["training_worlds"][0]["public"]["task"]["weights"].__setitem__(0, .2),
            "terminal_candidates": lambda p: p["training_worlds"][0]["training"]["snapshots"][4].update(candidates=["observe_all"]),
            "truth_leak": lambda p: p["training_worlds"][0]["public"].update(realized_coefficients=[3.]),
            "development_feedback": lambda p: p["development_worlds"][0].update(training=p["training_worlds"][0]["training"]),
            "claim_frozen": lambda p: p.update(scientific_protocol_frozen=True),
            "allocated_optimization": lambda p: p["optimization_schedule"].update(allocated=True),
            "nonfinite": lambda p: p["training_worlds"][0]["public"]["task"].update(scale=float("nan")),
            "extra_field": lambda p: p.update(unreviewed=1),
        }
        rejections = []
        for name, mutate in mutations.items():
            bad = copy.deepcopy(self.proposal)
            mutate(bad)
            save(self.directory/(name+".json"), bad, allow_nan=True)
            with self.subTest(name=name), self.assertRaises(ValueError) as error:
                validate_proposal(bad, expected_source_pins=self.pins)
            rejections.append({"case": name, "exception": str(error.exception)})
        pin_variants = [dict(self.pins), dict(self.pins), dict(self.pins)]
        pin_variants[0].pop(REQUIRED_SOURCE_PATHS[0])
        pin_variants[1]["unadmitted.py"] = "a"*64
        pin_variants[2][REQUIRED_SOURCE_PATHS[0]] = "Z"*64
        for i, pins in enumerate(pin_variants):
            save(self.directory/f"invalid-pins-{i}.json", pins)
            with self.assertRaises(ValueError):
                build_proposal(source_pins=pins)
        different_pins = dict(self.pins)
        different_pins[REQUIRED_SOURCE_PATHS[0]] = "0"*64
        with self.assertRaises(ValueError):
            validate_proposal(self.proposal, expected_source_pins=different_pins)
        save(self.directory/"rejections.json", rejections)

    def test_cost_totals_pending_optimization_and_no_scientific_admission(self):
        p = self.proposal
        train = p["training_worlds"]
        self.assertEqual(sum(r["training"]["expected_branch_cost_units"] for r in train), 5120)
        self.assertEqual(sum(r["training"]["expected_scoring_cost_units"] for r in train), 4096)
        self.assertEqual(32*32+5120+4096, p["budget_basis"]["training_total_response_cost_ceiling"])
        self.assertEqual(24*8*32, p["budget_basis"]["development_total_trajectory_cost_ceiling"])
        self.assertEqual(32*4*28, p["balance"]["training_nonterminal_pair_records_including_ties"])
        self.assertEqual(32*5, p["balance"]["training_supervised_records"])
        self.assertFalse(p["scientific_protocol_frozen"])
        self.assertFalse(p["scientific_execution_authorized"])
        self.assertFalse(p["provenance_qualified"])
        self.assertEqual((p["worlds_generated"], p["responses_generated"]), (0, 0))
        self.assertEqual(p["optimization_schedule"]["status"], "pending_parent_integration")
        self.assertFalse(p["optimization_schedule"]["allocated"])
        self.assertIsNone(p["optimization_schedule"]["settings"])
        self.assertEqual(len(p["remaining_required_decisions"]), 7)
        save(self.directory/"pending-decisions.json", p["remaining_required_decisions"])


if __name__ == "__main__":
    print("Interpreter:", sys.executable, flush=True)
    print("Version:", sys.version.replace("\n", " "), flush=True)
    print("Retained fixtures:", os.environ["ACE_SCREEN_PROTOCOL_TEST_OUTPUT"], flush=True)
    unittest.main(verbosity=2)
