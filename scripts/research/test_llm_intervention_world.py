"""New artificial-only world tests. Retain every journal, including failures."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

from llm_intervention_contract import Action, ChargedExecutor, Node, canonical, digest
from llm_intervention_replicated_branches import build_replicated_training_preferences
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask, make_envelope
from llm_intervention_world import (
    LinearMechanism, ResponseAdapter, STREAMS, WorldDescriptor, generate_world,
    keyed_seed, private_structural_means, training_callbacks,
)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.root / self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def chain(self, variances=(0., 0., 0.), world_id=17):
        graph = (Node("X", (), "u", (-10., 10.)),
                 Node("Y", ("X",), "u", (-10., 10.)),
                 Node("Z", ("Y",), "u", (-10., 10.)))
        return generate_world(scope="artificial", world_id=world_id, graph=graph,
            intercept_means=(1., 2., -4.), coefficient_means=((), (3.,), (2.,)),
            noise_variances=variances, intercept_sd=0., coefficient_sd=0.)

    def save_world(self, world, name="world.json"):
        with (self.directory / name).open("x") as handle:
            handle.write(canonical(world.private_payload()) + "\n")

    def run_adapter(self, name, world, menu, *, stream="measurement", replicate="r0",
                    namespace="fixture", structural=False, ids=None):
        output = self.directory / name
        output.mkdir(exist_ok=False)
        sequence = tuple(a.action_id for a in menu) if ids is None else ids
        budget = sum(next(a.cost_units for a in menu if a.action_id == aid) for aid in sequence)
        with ResponseAdapter(world, expected_sha256=world.sha256, menu=menu, stream=stream,
                replicate_id=replicate, request_namespace=namespace,
                executor_journal=output/"charged.jsonl", journal=output/"world.jsonl",
                max_requests=len(sequence), structural=structural) as adapter:
            with ChargedExecutor(world.graph, menu, budget, output/"charged.jsonl", adapter) as executor:
                receipts = [executor.execute_completion(canonical({"action_id": aid})) for aid in sequence]
        return receipts, output

    def test_chain_fork_collider_hand_solved_structural_means(self):
        chain = self.chain()
        self.save_world(chain)
        queries = (Action("obs", "observe", (), ("X", "Y", "Z"), 1),
                   Action("cut", "intervene", (("Y", 7.),), ("Z", "Y"), 2),
                   Action("root", "intervene", (("X", -2.),), ("Z",), 1),
                   Action("both", "intervene", (("X", -2.), ("Y", 7.)), ("Z",), 1))
        means = private_structural_means(chain, queries, expected_sha256=chain.sha256)
        self.assertEqual(means, ({"X": 1., "Y": 5., "Z": 6.}, {"Z": 10., "Y": 7.},
                                 {"Z": -12.}, {"Z": 10.}))
        fork_graph = (chain.graph[0], chain.graph[1], Node("Z", ("X",), "u"))
        fork = WorldDescriptor("artificial", 18, fork_graph,
            (chain.mechanisms[0], chain.mechanisms[1], LinearMechanism("Z", -4., (("X", -2.),), 0.)))
        self.save_world(fork, "fork.json")
        self.assertEqual(private_structural_means(fork, (queries[0], queries[1]), expected_sha256=fork.sha256),
                         ({"X": 1., "Y": 5., "Z": -6.}, {"Z": -6., "Y": 7.}))
        collider = WorldDescriptor("artificial", 19,
            (chain.graph[0], Node("Y", (), "u", (-10., 10.)), Node("Z", ("X", "Y"), "u")),
            (LinearMechanism("X", 1., (), 0.), LinearMechanism("Y", 2., (), 0.),
             LinearMechanism("Z", -4., (("X", 3.), ("Y", -2.)), 0.)))
        self.save_world(collider, "collider.json")
        self.assertEqual(private_structural_means(collider, (queries[0], queries[2]), expected_sha256=collider.sha256),
                         ({"X": 1., "Y": 2., "Z": -5.}, {"Z": -14.}))

    def test_clamp_cuts_incoming_equation_and_noise_but_not_other_noise(self):
        world = self.chain((4., 9., 16.))
        self.save_world(world)
        action = Action("cut", "intervene", (("Y", 7.),), ("X", "Y", "Z"), 3)
        rows, output = self.run_adapter("noisy", world, (action,), ids=("cut", "cut"))
        # Independent hand reconstruction from the specified keyed Gaussian law.
        for request, row in enumerate(rows):
            key = ("fixture", "r0", request, "cut", digest({"kind": "intervene", "targets": action.targets}))
            nx = random.Random(keyed_seed("artificial", 17, "measurement", *key, "X")).gauss(0., 1.)
            nz = random.Random(keyed_seed("artificial", 17, "measurement", *key, "Z")).gauss(0., 1.)
            self.assertEqual(row["values"], {"X": 1. + 2.*nx, "Y": 7., "Z": 10. + 4.*nz})
        self.assertNotEqual(rows[0]["values"]["Z"], rows[1]["values"]["Z"])
        changed = replace(world, mechanisms=(LinearMechanism("X", -900., (), 25.),
            LinearMechanism("Y", 400., (("X", -700.),), 81.), world.mechanisms[2]))
        other, _ = self.run_adapter("changed_upstream", changed, (action,), ids=("cut", "cut"))
        self.assertEqual([r["values"]["Z"] for r in rows], [r["values"]["Z"] for r in other])
        charged = records(output/"charged.jsonl")
        self.assertEqual(sum(r["cost_units"] for r in charged if r["event"] == "reserve"), 6)

    def test_partial_measurements_hide_internal_nodes_without_changing_draw(self):
        world = self.chain((1., 1., 1.))
        full = Action("obs", "observe", (), ("X", "Y", "Z"), 1)
        subset = replace(full, measure=("Z",))
        all_rows, _ = self.run_adapter("all", world, (full,))
        rows, _ = self.run_adapter("subset", world, (subset,))
        self.assertEqual(rows[0]["values"], {"Z": all_rows[0]["values"]["Z"]})
        learner = LinearGaussianSCM(world.graph, {n.name: 1. for n in world.graph})
        self.assertEqual(learner.update(rows[0]), ())  # no latent-parent inference claim

    def test_descriptor_serialization_immutable_and_public_allowlist(self):
        world = self.chain((1., 1., 1.))
        self.save_world(world)
        payload = world.private_payload()
        self.assertEqual(WorldDescriptor.from_private_payload(payload, expected_sha256=world.sha256), world)
        with self.assertRaises(FrozenInstanceError):
            world.world_id = 9
        with self.assertRaises(FrozenInstanceError):
            world.mechanisms[0].intercept = 2.
        public = world.public_design()
        self.assertEqual(set(public), {"graph"})
        public["graph"][0]["name"] = "changed"
        self.assertEqual(world.graph[0].name, "X")
        payload["mechanisms"][0]["intercept"] = 42.
        with self.assertRaises(ValueError):
            WorldDescriptor.from_private_payload(payload, expected_sha256=world.sha256)
        with self.assertRaises(ValueError):
            WorldDescriptor.from_private_payload(dict(world.private_payload(), extra=1), expected_sha256=world.sha256)
        menu = (Action("a", "intervene", (("X", 0.),), ("Y", "Z"), 1),)
        task = PredictionTask("Z", menu, (1.,), 1., .04)
        output = self.directory/"public"
        output.mkdir()
        with ResponseAdapter(world, expected_sha256=world.sha256, menu=menu, stream="measurement",
                replicate_id="r", request_namespace="public", executor_journal=output/"charged.jsonl",
                journal=output/"private.jsonl", max_requests=1) as adapter:
            with ChargedExecutor(world.graph, menu, 1, output/"charged.jsonl", adapter) as executor:
                learner = LinearGaussianSCM(world.graph, {n.name: 1. for n in world.graph})
                learner.update(executor.execute_completion('{"action_id":"a"}'))
                envelope = make_envelope(executor.state("fixture", learner.summaries()), task)
        encoded = canonical(envelope)
        for forbidden in ("descriptor_sha256", "world_id", "mechanisms", "private_eval", "train_score", world.sha256):
            self.assertNotIn(forbidden, encoded)

    def test_parameter_generation_topological_order_and_global_rng_isolation(self):
        original = self.chain((1., 1., 1.))
        params = dict(scope="artificial", world_id=23, graph=original.graph,
            intercept_means=(1., 2., -4.), coefficient_means=((), (3.,), (2.,)),
            noise_variances=(1., 2., 3.), intercept_sd=.5, coefficient_sd=.25)
        py_before, np_before = random.getstate(), np.random.get_state()
        a = generate_world(**params)
        for stream in STREAMS:
            keyed_seed("artificial", 23, stream, "unrelated", 9)
        b = generate_world(**params)
        self.save_world(a)
        self.assertEqual(a, b)
        expected = 1. + .5 * random.Random(keyed_seed("artificial", 23, "mechanism", "X", "intercept")).gauss(0., 1.)
        self.assertEqual(a.mechanisms[0].intercept, expected)
        action = Action("obs", "observe", (), ("Z",), 1)
        self.run_adapter("noise", a, (action,))
        private_structural_means(a, (action,), expected_sha256=a.sha256)
        self.assertEqual(random.getstate(), py_before)
        after = np.random.get_state()
        self.assertEqual(np_before[0], after[0])
        np.testing.assert_array_equal(np_before[1], after[1])
        self.assertEqual(np_before[2:], after[2:])
        with self.assertRaises(ValueError):
            generate_world(**dict(params, graph=tuple(reversed(original.graph))))
        # Reordering unrelated roots preserves their name-keyed mechanisms.
        roots = (Node("B", (), "u"), Node("A", (), "u"))
        roots_params = dict(params, graph=roots, intercept_means=(0., 0.),
                            coefficient_means=((), ()), noise_variances=(1., 1.))
        left = generate_world(**roots_params)
        right = generate_world(**dict(roots_params, graph=tuple(reversed(roots))))
        self.assertEqual({m.node: m for m in left.mechanisms}, {m.node: m for m in right.mechanisms})

    def test_stream_replicate_namespace_request_separation_and_order_independence(self):
        seeds = {keyed_seed("artificial", 17, s, "same", 0) for s in STREAMS}
        self.assertEqual(len(seeds), 6)
        self.assertNotEqual(keyed_seed("artificial", 17, "policy", "a", "bc"),
                            keyed_seed("artificial", 17, "policy", "ab", "c"))
        world = self.chain((1., 1., 1.))
        menu = (Action("obs", "observe", (), ("X", "Y", "Z"), 1),)
        baseline, _ = self.run_adapter("baseline", world, menu, ids=("obs", "obs"))
        alternatives = []
        for i, options in enumerate(({"stream": "initial"}, {"stream": "train_score"},
                                     {"replicate": "r1"}, {"namespace": "another"})):
            rows, _ = self.run_adapter("different" + str(i), world, menu, **options)
            alternatives.append(rows[0]["values"])
        means = private_structural_means(world, menu, expected_sha256=world.sha256)
        repeated, _ = self.run_adapter("repeat", world, menu, ids=("obs", "obs"))
        self.assertEqual(baseline, repeated)
        self.assertTrue(all(v != baseline[0]["values"] for v in alternatives))
        self.assertEqual(means, ({"X": 1., "Y": 5., "Z": 6.},))

    def test_invalid_descriptors_namespaces_and_action_memberships(self):
        world = self.chain()
        for scope in ("development", "confirmation", "invalid"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                keyed_seed(scope, 0, "train_score", "reject-before-draw")
        for scope, wid, stream, key in (("artificial", True, "initial", ()),
                ("artificial", 1, "unknown", ()), ("artificial", 1, "policy", (True,)),
                ("artificial", 1, "policy", ("",))):
            with self.subTest(key=key, wid=wid, stream=stream), self.assertRaises(ValueError):
                keyed_seed(scope, wid, stream, *key)
        bad = (replace(world.mechanisms[1], coefficients=(("Z", 1.),)),
               replace(world.mechanisms[1], node="unknown"))
        for mechanism in bad:
            with self.assertRaises(ValueError):
                replace(world, mechanisms=(world.mechanisms[0], mechanism, world.mechanisms[2]))
        with self.assertRaises(ValueError):
            replace(world, graph=world.graph + (world.graph[0],))
        with self.assertRaises(ValueError):
            LinearMechanism("X", 0., (), math.nan)
        with self.assertRaises(ValueError):
            replace(world, mechanisms=list(world.mechanisms))
        for action in (Action("bad", "observe", (), ("unknown",), 1),
                       Action("bad", "intervene", (("X", 11.),), ("Z",), 1)):
            with self.assertRaises(ValueError):
                private_structural_means(world, (action,), expected_sha256=world.sha256)
        with self.assertRaises(ValueError):
            private_structural_means(world, (Action("o", "observe", (), ("Z",), 1),), expected_sha256="wrong")

    def test_adapter_authentication_rejects_pin_unreserved_tampered_and_changed_action(self):
        world = self.chain()
        action = Action("o", "observe", (), ("Z",), 1)
        for case in ("pin", "unreserved", "tampered", "action"):
            output = self.directory/case
            output.mkdir()
            args = dict(expected_sha256="wrong" if case == "pin" else world.sha256,
                menu=(action,), stream="measurement", replicate_id="r", request_namespace="auth",
                executor_journal=output/"charged.jsonl", journal=output/"world.jsonl", max_requests=1)
            if case == "pin":
                with self.assertRaises(ValueError):
                    ResponseAdapter(world, **args)
            else:
                with ResponseAdapter(world, **args) as adapter:
                    def respond(a, request):
                        if case == "tampered":
                            # Preserve the genuine prefix before intentional artificial corruption.
                            raw = (output/"charged.jsonl").read_bytes()
                            (output/"before_corruption.jsonl").write_bytes(raw)
                            with (output/"charged.jsonl").open("ab") as handle:
                                handle.write(b'{"sha256":"wrong","previous_sha256":null}\n')
                        return adapter(replace(a, cost_units=2) if case == "action" else a, request)
                    with ChargedExecutor(world.graph, (action,), 1, output/"charged.jsonl", respond) as executor:
                        with self.assertRaises(ValueError):
                            if case == "unreserved":
                                adapter(action, 0)
                            else:
                                executor.execute_completion('{"action_id":"o"}')
            audit = records(output/"world.jsonl")
            self.assertEqual(audit[-1]["event"], "failure")
            self.assertFalse(any(r["event"] == "outcome" for r in audit))

    def test_exclusive_journal_duplicate_and_bound_failures_keep_reservations(self):
        world = self.chain()
        action = Action("o", "observe", (), ("Z",), 2)
        for case in ("duplicate", "limit"):
            output = self.directory/case
            output.mkdir()
            args = dict(expected_sha256=world.sha256, menu=(action,), stream="measurement",
                replicate_id="r", request_namespace="failure", executor_journal=output/"charged.jsonl",
                journal=output/"world.jsonl", max_requests=2 if case == "duplicate" else 1)
            with ResponseAdapter(world, **args) as adapter:
                with ChargedExecutor(world.graph, (action,), 4, output/"charged.jsonl", adapter) as executor:
                    executor.execute_completion('{"action_id":"o"}')
                    with self.assertRaises(ValueError):
                        if case == "duplicate":
                            adapter(action, 0)
                        else:
                            executor.execute_completion('{"action_id":"o"}')
            before = (output/"world.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                ResponseAdapter(world, **args)
            self.assertEqual((output/"world.jsonl").read_bytes(), before)
            self.assertEqual(records(output/"world.jsonl")[-1]["event"], "failure")
            reservations = [r for r in records(output/"charged.jsonl") if r["event"] == "reserve"]
            self.assertEqual(sum(r["cost_units"] for r in reservations), 2 if case == "duplicate" else 4)

    def test_nonfinite_world_outcome_is_preserved_as_charged_failure(self):
        world = self.chain()
        world = replace(world, mechanisms=(LinearMechanism("X", 1e308, (), 0.),
            LinearMechanism("Y", 1e308, (("X", 1e308),), 0.), world.mechanisms[2]))
        self.save_world(world)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_adapter("overflow", world, (Action("o", "observe", (), ("Z",), 7),))
        charged = records(self.directory/"overflow/charged.jsonl")
        audit = records(self.directory/"overflow/world.jsonl")
        self.assertEqual(charged[-1]["event"], "failure")
        self.assertEqual(charged[-1]["spent_units"], 7)
        self.assertEqual([r["event"] for r in audit], ["bind", "attempt", "mapping", "failure"])

    def test_training_callbacks_noisy_branches_and_charged_structural_scoring(self):
        world = self.chain((1., 1., 1.))
        self.save_world(world)
        menu = (Action("a", "intervene", (("Y", 0.),), ("Z",), 1),
                Action("b", "intervene", (("Y", 2.),), ("Z",), 2))
        queries = (Action("q0", "intervene", (("Y", 1.),), ("Z",), 3),
                   Action("q1", "intervene", (("Y", 3.),), ("Z",), 1))
        task = PredictionTask("Z", queries, (.25, .75), 2., .04)
        base = LinearGaussianSCM(world.graph, {n.name: 1. for n in world.graph})
        # Hand-derived means: -4 + 2*1 = -2; -4 + 2*3 = 2.
        for case, candidates in (("forward", ("a", "b")), ("reversed", ("b", "a"))):
            parent = self.directory/case
            parent.mkdir()
            audit = parent/"audit"
            audit.mkdir()
            output = parent/"driver"
            with ExitStack() as stack:
                branch, score = training_callbacks(stack, world, expected_sha256=world.sha256,
                    menu=menu, candidates=candidates, task=task, replicate_ids=("r0", "r1"),
                    branch_output=output, journal_directory=audit, request_namespace="training-snapshot-0",
                    structural_scores=True)
                result = build_replicated_training_preferences(scope="artificial", graph=world.graph,
                    menu=menu, candidates=candidates, learner=base, next_learner_request_id=7,
                    task=task, replicate_ids=("r0", "r1"), branch_respond=branch, score_respond=score,
                    tie_multiplier=2., minimum_gain_gap=0., output=output)
            self.assertEqual(result["feedback"]["scoring_truths"], [[-2., 2.], [-2., 2.]])
            self.assertEqual((result["charged_branch_cost"], result["charged_scoring_cost"]), (6, 8))
            self.assertEqual(result["environment_responses"], 8)
            self.assertEqual(result["scoring_learner_updates"], 0)
            self.assertFalse(result["provenance_qualified"])
            self.assertTrue(base.is_fresh())
            outcomes = {}
            for path in audit.glob("*.branch-*.world.jsonl"):
                events = records(path)
                mapping = next(r for r in events if r["event"] == "mapping")
                outcome = next(r for r in events if r["event"] == "outcome")
                rid, aid = events[0]["replicate_id"], mapping["action"]["action_id"]
                outcomes[rid, aid] = outcome["values"]["Z"]
                self.assertEqual(mapping["request_id"], 0)
                self.assertEqual(events[0]["stream"], "measurement")
            if case == "forward":
                forward = outcomes
            else:
                self.assertEqual(outcomes, forward)
        self.assertNotEqual(forward["r0", "a"], forward["r1", "a"])

    def test_noisy_training_scoring_and_explicit_mode_restrictions(self):
        world = self.chain((1., 1., 1.))
        menu = (Action("q", "intervene", (("Y", 1.),), ("Z",), 1),)
        noisy, _ = self.run_adapter("noisy", world, menu, stream="train_score")
        structural, _ = self.run_adapter("structural", world, menu, stream="train_score", structural=True)
        self.assertEqual(structural[0]["values"], {"Z": -2.})
        self.assertNotEqual(noisy[0]["values"], structural[0]["values"])
        for stream, structural_mode in (("measurement", True), ("private_eval", False), ("policy", False)):
            with self.assertRaises(ValueError):
                ResponseAdapter(world, expected_sha256=world.sha256, menu=menu, stream=stream,
                    replicate_id="r", request_namespace="invalid", executor_journal=self.directory/"none",
                    journal=self.directory/(stream + ".jsonl"), max_requests=1, structural=structural_mode)
        with ExitStack() as stack:
            with self.assertRaises(ValueError):
                training_callbacks(stack, world, expected_sha256=world.sha256, menu=menu,
                    candidates=("q", "unknown"), task=PredictionTask("Z", menu, (1.,), 1., .04),
                    replicate_ids=("r0", "r1"), branch_output=self.directory/"unused",
                    journal_directory=self.directory, request_namespace="invalid", structural_scores=True)


if __name__ == "__main__":
    # Explicit caller-owned custody; never delete failed fixture evidence.
    if len(sys.argv) < 2:
        raise SystemExit("usage: test_llm_intervention_world.py NEW_ARTIFACT_DIRECTORY [unittest selectors]")
    WorldTests.root = Path(sys.argv.pop(1))
    WorldTests.root.mkdir(exist_ok=False)
    unittest.main(verbosity=2)
