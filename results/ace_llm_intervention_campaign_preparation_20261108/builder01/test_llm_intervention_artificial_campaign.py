"""New, in-memory fabricated fixtures only; no execution or old test suites."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import unittest
from unittest.mock import patch

import numpy as np

import llm_intervention_artificial_campaign as builder
import llm_intervention_campaign as campaign
import llm_intervention_paired_episode as paired
import llm_intervention_screen_policy as screen
from llm_intervention_contract import VERSION as STATE_VERSION, canonical, digest
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import VERSION as TASK_VERSION, task_prompt


ROSTER = ("uniform_random", "menu_coverage", "local_information", "target_risk",
          "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm")
WORLD_IDS = (9101, 9102)


def fixture():
    """Paths and hashes are invented declarations, never opened or launched."""
    recipe = {"fixture": "inference-only", "temperature": 0.}
    services = {
        pid: {"identity": {
            "model_sha256": digest("fabricated-shared-model"),
            "source_sha256": digest("fabricated-shared-service-source"),
            "adapter_sha256": digest({"treatment": "frozen", "adapter": None})
                if pid == "frozen_llm" else digest(["fabricated-adapter", pid]),
            "numeric_recipe": copy.deepcopy(recipe), "numeric_recipe_sha256": digest(recipe)},
            "startup_seconds": 30., "cleanup_seconds": 5., "max_requests": 8}
        for pid in ROSTER[5:]}
    bindings = {(r["world_id"], r["policy_id"]): r for r in builder.numeric_policy_bindings()}
    specs = {}
    for wid in WORLD_IDS:
        specs[wid] = []
        for pid in ROSTER:
            command = ["/caller supplied/bin/launch", "--role", pid]
            if pid in ROSTER[:4]:
                row = bindings[wid, pid]
                command += ["--binding-json", canonical(row["binding"]),
                            "--binding-sha256", row["binding_sha256"]]
            elif pid == "learned_ranker":
                command += ["--checkpoint", "/caller supplied/final-dpo.json", "--world", str(wid)]
            specs[wid].append({"policy_id": pid,
                "kind": "persistent" if pid in services else "task_process", "command": command,
                "timeout_seconds": 20.,
                "identity_sha256": digest(services[pid]["identity"]) if pid in services else None,
                "policy_provenance_sha256": digest(["fabricated-provenance", pid])})
    return {"policy_specs_by_world": specs, "services": services,
            "outer_provenance_receipt_sha256": digest("fabricated-outer-receipt")}


def fabricated_envelope(plan, *, costly=False):
    """Hand-written public receipts and summaries; never sample or fit a learner."""
    graph, menu = plan["world"]["graph"], plan["menu"]
    history = [{"request_id": i, "action_id": "observe_all", "kind": "observe",
                "targets": {}, "values": {n["name"]: 0. for n in graph}, "cost_units": 1}
               for i in range(2)]
    if costly:
        history.append({"request_id": 2, "action_id": "x_plus", "kind": "intervene",
                        "targets": {"X": 1.},
                        "values": {n["name"]: 1. if n["name"] == "X" else 0. for n in graph},
                        "cost_units": menu[2]["cost_units"]})
    summaries = []
    for n in graph:
        d = 1 + len(n["parents"])
        summaries.append({"node": n["name"], "terms": ["intercept"] + n["parents"],
                          "parameter_mean": [0.] * d,
                          "parameter_covariance": [[float(i == j) for j in range(d)] for i in range(d)],
                          "noise_variance": .25, "eligible_rows": 2 + int(costly and n["name"] != "X")})
    spent = sum(r["cost_units"] for r in history)
    return {"version": TASK_VERSION, "task": copy.deepcopy(plan["task"]), "state": {
        "version": STATE_VERSION, "objective": plan["objective"], "domain_notes": plan["domain_notes"],
        "graph": copy.deepcopy(graph), "menu": copy.deepcopy(menu), "history": history, "scm": summaries,
        "spent_units": spent, "remaining_units": 6 - spent,
        "requests": len(history), "returned_responses": len(history), "terminal": False}}


class ArtificialCampaignTests(unittest.TestCase):
    def setUp(self):
        # Fail closed if any fixture accidentally crosses the pure preparation
        # boundary. Existing preparation may construct a fresh zero-prior SCM.
        forbidden = (
            "llm_intervention_world.generate_world", "llm_intervention_world._normal",
            "llm_intervention_world._evaluate", "llm_intervention_world.private_structural_means",
            "llm_intervention_world.ResponseAdapter.__init__",
            "llm_intervention_world.training_callbacks",
            "llm_intervention_contract.ChargedExecutor.__init__",
            "llm_intervention_scm.LinearGaussianSCM.update",
            "llm_intervention_screen_protocol.build_proposal",
            "llm_intervention_training_dataset.assemble_training_dataset",
            "llm_intervention_screen_policy.choose_public",
            "llm_intervention_screen_policy.choose_numerical",
            "llm_intervention_screen_policy.choose_target_risk",
            "llm_intervention_numerical_policy.choose",
            "llm_intervention_target_risk.choose_target_risk",
            "llm_intervention_campaign.initialize_campaign",
            "llm_intervention_campaign.run_policy", "llm_intervention_campaign.finalize_campaign",
            "llm_intervention_paired_episode.run_world_arm", "llm_intervention_paired_episode.run_episode",
            "llm_intervention_paired_episode.finalize_paired_world",
            "llm_intervention_persistent_policy.PersistentPolicy.__init__",
            "llm_intervention_task.TaskProcessPolicy.__init__", "subprocess.Popen",
            "builtins.open", "io.open", "pathlib.Path.mkdir", "os.mkdir",
        )
        # Resolve patch targets before blocking file access.
        for name in forbidden:
            guard = patch(name, side_effect=AssertionError("forbidden in pure fixture: " + name))
            guard.start()
            self.addCleanup(guard.stop)

    def build(self):
        return builder.build_artificial_campaign(**fixture())

    def test_exact_handwritten_laws_queries_and_declared_means(self):
        result = self.build()
        expected = (
            [("X", 0., []), ("Y", 1., [["X", 2.]])],
            [("X", 0., []), ("Y", -.5, [["X", 1.]]), ("Z", .25, [["Y", -2.]])],
        )
        means = [[1. + 2. * x for x in (-1., 1.)],
                 [.25 - 2. * (-.5 + x) for x in (-1., 1.)]]
        self.assertEqual(means, [[-1., 3.], [3.25, -.75]])
        for i, source in enumerate(result["campaign"]["sources"]):
            p, declared = source["plan"], result["private_query_means"][i]
            w = p["world"]
            self.assertEqual((w["scope"], w["world_id"]), ("artificial", WORLD_IDS[i]))
            self.assertEqual([(m["node"], m["intercept"], m["coefficients"]) for m in w["mechanisms"]], expected[i])
            names = [r[0] for r in expected[i]]
            self.assertEqual(w["graph"], [
                {"name": name, "parents": [] if j == 0 else [names[j-1]],
                 "unit": "dimensionless", "intervention_bounds": [-2., 2.] if j == 0 else None}
                for j, name in enumerate(names)])
            self.assertEqual([m["noise_variance"] for m in w["mechanisms"]], [.25] * len(names))
            self.assertEqual(p["learner"], {"graph": w["graph"],
                             "noise_variances": {n: .25 for n in names}, "prior_variance": 1.})
            self.assertEqual(p["menu"], [
                {"action_id": aid, "kind": "observe" if not targets else "intervene",
                 "targets": targets, "measure": names, "cost_units": cost}
                for aid, targets, cost in (("observe_all", [], 1), ("x_minus", [["X", -1.]], 1),
                                           ("x_plus", [["X", 1.]], 1 + i))])
            self.assertEqual(p["task"], {"target": names[-1], "queries": [
                {"action_id": "q_x_" + direction, "kind": "intervene", "targets": [["X", x]],
                 "measure": [names[-1]], "cost_units": 1}
                for direction, x in (("minus", -1.), ("plus", 1.))],
                "weights": [.5, .5], "scale": 2., "normalized_mse_tolerance": .04})
            self.assertEqual(declared, {"world_id": WORLD_IDS[i], "world_sha256": digest(w),
                "task_sha256": digest(p["task"]), "target": names[-1],
                "query_ids": ["q_x_minus", "q_x_plus"], "means": means[i]})

    def test_accepted_preparation_and_fresh_zero_priors(self):
        learners = []
        def fresh(*args, **kwargs):
            learner = LinearGaussianSCM(*args, **kwargs)
            learners.append(learner)
            return learner
        with patch.object(builder, "prepare_plan", wraps=paired.prepare_plan) as plans, \
             patch.object(builder, "prepare_campaign", wraps=campaign.prepare_campaign) as campaigns, \
             patch.object(paired, "LinearGaussianSCM", side_effect=fresh):
            result = self.build()
        self.assertEqual(plans.call_count, 2)
        self.assertEqual(campaigns.call_count, 1)
        self.assertTrue(learners)
        for learner in learners:
            self.assertTrue(learner.is_fresh())
            for head in learner.heads.values():
                self.assertTrue(np.array_equal(head.mean, np.zeros(len(head.mean))))
                self.assertTrue(np.array_equal(head.covariance, np.eye(len(head.mean))))
        self.assertEqual(campaign.validate_campaign(**{k: result[k] for k in (
            "campaign", "campaign_sha256", "source_plan_pins")}), result["campaign"])

    def test_complete_roster_phases_case_strata_and_charged_bounds(self):
        c = self.build()["campaign"]
        self.assertEqual(c["scope"], "artificial")
        self.assertEqual(c["policy_ids"], list(ROSTER))
        self.assertEqual(c["policy_phases"], {p: "cpu" if i < 5 else "gpu" for i, p in enumerate(ROSTER)})
        self.assertEqual(c["strata_fields"], ["case"])
        self.assertEqual(len({s["strata"]["case"] for s in c["sources"]}), 2)
        self.assertFalse(c["scientific_provenance_qualified"])
        self.assertFalse(c["retry_allowed"])
        initial_count = response_bound = cost_bound = decisions = cells = 0
        for source in c["sources"]:
            p = source["plan"]
            self.assertEqual(p["initial_actions"], ["observe_all", "observe_all"])
            self.assertEqual((p["initial_cost_units"], p["acquisition_cap_units"]), (2, 4))
            self.assertEqual([s["policy_id"] for s in p["policy_specs"]], list(ROSTER))
            self.assertFalse(p["scientific_provenance_qualified"])
            self.assertFalse(p["retry_allowed"])
            # Enumerate affordable action sequences, including stopping at any
            # point. No action is executed and no receipt is generated.
            def paths(left):
                yield (0, 0)
                for a in p["menu"]:
                    if a["cost_units"] <= left:
                        for count, cost in paths(left - a["cost_units"]):
                            yield count + 1, cost + a["cost_units"]
            possible = list(paths(4))
            self.assertEqual(max(n for n, _ in possible), 4)
            self.assertEqual(max(cost for _, cost in possible), 4)
            for spec in p["policy_specs"]:
                cells += 1
                initial_count += 2
                response_bound += 2 + max(n for n, _ in possible)
                cost_bound += 2 + max(cost for _, cost in possible)
                decisions += 4 * int(spec["kind"] == "persistent")
        self.assertEqual((cells, initial_count, response_bound, cost_bound, decisions), (16, 32, 96, 96, 24))
        self.assertEqual({pid: s["max_requests"] for pid, s in c["services"].items()}, dict.fromkeys(ROSTER[5:], 8))

    def test_paired_initialization_distinct_measurements_and_exact_seed_law(self):
        result = self.build()
        rows = result["numeric_policy_bindings"]
        self.assertEqual([(r["world_id"], r["policy_id"]) for r in rows],
                         [(wid, pid) for wid in WORLD_IDS for pid in ROSTER[:4]])
        all_namespaces = set()
        for source in result["campaign"]["sources"]:
            p, wid = source["plan"], source["plan"]["world"]["world_id"]
            def namespace(arm):
                return digest({"protocol": "ace-first-screen-proposal-20261107-v1", "scope": "artificial",
                               "world_id": wid, "purpose": "trajectory", "arm": arm})
            self.assertEqual(p["initial_binding"], {"request_namespace": namespace(None), "replicate_id": "trajectory"})
            self.assertEqual(p["measurement_bindings"], {
                pid: {"request_namespace": namespace(pid), "replicate_id": "trajectory"} for pid in ROSTER})
            all_namespaces.add(p["initial_binding"]["request_namespace"])
            all_namespaces.update(b["request_namespace"] for b in p["measurement_bindings"].values())
            for row in (r for r in rows if r["world_id"] == wid):
                b = screen.validate_binding(row["binding"], expected_sha256=row["binding_sha256"])
                self.assertEqual(b["seed_prefix"], ["artificial", wid, "policy", namespace(row["policy_id"]), "behavior"])
                self.assertEqual((b["initial_actions"], b["initial_cost_units"], b["total_budget"], b["planning_draws"]),
                                 (["observe_all", "observe_all"], 2, 6, 32))
                for index, costly in enumerate((False, True)):
                    seed, receipt = screen.decision_rng(fabricated_envelope(p, costly=costly), binding=b,
                        expected_binding_sha256=row["binding_sha256"], expected_decision_index=index)
                    key = ["ace-linear-gaussian-world-v1", "artificial", wid, "policy",
                           [namespace(row["policy_id"]), "behavior", index]]
                    expected = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).digest()
                    self.assertEqual(seed, int.from_bytes(expected, "big"))
                    self.assertEqual(receipt["canonical_key"], key)
                    self.assertEqual(receipt["decision_index"], index)
                    self.assertEqual(receipt["request_count"], 2 + index)
                    self.assertEqual(receipt["acquisition_cost_units"], p["menu"][2]["cost_units"] if costly else 0)
        self.assertEqual(len(all_namespaces), 18)

    def test_exact_caller_commands_and_provenance_are_frozen_detached(self):
        args = fixture()
        original = copy.deepcopy(args)
        r = builder.build_artificial_campaign(**args)
        self.assertEqual(args, original)
        self.assertEqual(r, builder.build_artificial_campaign(**args))
        for s in r["campaign"]["sources"]:
            p = s["plan"]
            self.assertEqual(p["policy_specs"], args["policy_specs_by_world"][p["world"]["world_id"]])
            self.assertEqual(p["outer_provenance_receipt_sha256"], args["outer_provenance_receipt_sha256"])
            self.assertEqual(s["plan_sha256"], digest(p))
        self.assertEqual(r["campaign_sha256"], digest(r["campaign"]))
        self.assertEqual(r["source_plan_pins"], [{"world_id": s["plan"]["world"]["world_id"], "sha256": digest(s["plan"])}
                                                for s in r["campaign"]["sources"]])
        frozen = canonical(r)
        args["policy_specs_by_world"][9101][0]["command"][0] = "/changed"
        args["services"]["sft_llm"]["identity"]["numeric_recipe"]["temperature"] = 1.
        self.assertEqual(canonical(r), frozen)
        r["campaign"]["sources"][0]["plan"]["policy_specs"][0]["command"][0] = "/tampered"
        with self.assertRaises(ValueError):
            campaign.validate_campaign(**{k: r[k] for k in ("campaign", "campaign_sha256", "source_plan_pins")})
        self.assertEqual(builder.numeric_policy_bindings(), self.build()["numeric_policy_bindings"])

    def test_public_prose_and_prompt_have_no_private_laws_or_identifiers(self):
        for s in self.build()["campaign"]["sources"]:
            p = s["plan"]
            self.assertEqual(p["objective"], "Predict the noise-disabled structural mean of " + p["task"]["target"]
                             + " under the supplied public query distribution.")
            self.assertEqual(p["domain_notes"],
                "Synthetic dimensionless variables with a known graph and hard interventions. "
                "All nodes are measured after each action. The learner uses public known "
                "noise variances and zero-mean independent Gaussian parameter priors. "
                "The task tolerance is a goal, not evidence of attainment.")
            prompt = task_prompt(fabricated_envelope(p))
            for forbidden in ("9101", "9102", "world_id", "seed", "regime", "private_query_means", "mechanisms",
                              "coefficients", "3.25", "-0.75", "Y =", "Z =", "sha256", "case", "provenance"):
                self.assertNotIn(forbidden, prompt)

    def test_no_io_rng_draws_policy_calls_or_evaluation(self):
        rng, nrng = random.getstate(), np.random.get_state()
        with patch("random.Random", side_effect=AssertionError("no random draws")), \
             patch("numpy.random.default_rng", side_effect=AssertionError("no random draws")):
            self.build()
        self.assertEqual(random.getstate(), rng)
        after = np.random.get_state()
        self.assertEqual(nrng[0], after[0])
        self.assertTrue(np.array_equal(nrng[1], after[1]))
        self.assertEqual(nrng[2:], after[2:])

    def test_reject_hostile_world_rosters_policy_specs_commands_and_pins(self):
        mutations = [
            ("missing world", lambda a: a["policy_specs_by_world"].pop(9102)),
            ("extra world", lambda a: a["policy_specs_by_world"].update({17: []})),
            ("string world ID", lambda a: a["policy_specs_by_world"].update({"9101": a["policy_specs_by_world"].pop(9101)})),
            ("float world ID", lambda a: a["policy_specs_by_world"].update({9101.: a["policy_specs_by_world"].pop(9101)})),
            ("missing arm", lambda a: a["policy_specs_by_world"][9101].pop()),
            ("ninth arm", lambda a: a["policy_specs_by_world"][9101].append(copy.deepcopy(a["policy_specs_by_world"][9101][0]))),
            ("reordered arms", lambda a: a["policy_specs_by_world"][9101].reverse()),
            ("changed persistent command", lambda a: a["policy_specs_by_world"][9102][5]["command"].append("different")),
        ]
        for key, value in (("kind", "persistent"), ("policy_id", "numerical_ranker"),
                           ("timeout_seconds", True), ("timeout_seconds", float("nan")), ("timeout_seconds", 0),
                           ("timeout_seconds", 3601), ("policy_provenance_sha256", "A" * 64),
                           ("identity_sha256", "a" * 64), ("private_means", [-1, 3]),
                           ("command", []), ("command", "shell text"), ("command", ["x\0y"]),
                           ("command", ["x" * 4097]), ("command", ["x"] * 65)):
            mutations.append((key + repr(value)[:40], lambda a, k=key, v=value: a["policy_specs_by_world"][9101][0].update({k: v})))
        for value in (None, True, "", "g" * 64, "a" * 63):
            mutations.append(("outer pin " + repr(value), lambda a, v=value: a.update(outer_provenance_receipt_sha256=v)))
        for key, value in (("policy_provenance_sha256", digest("changed")), ("timeout_seconds", 21.)):
            mutations.append(("cross-world " + key, lambda a, k=key, v=value: a["policy_specs_by_world"][9102][4].update({k: v})))
        for label, mutate in mutations:
            with self.subTest(label=label):
                args = fixture()
                mutate(args)
                with self.assertRaises(ValueError):
                    builder.build_artificial_campaign(**args)

    def test_reject_unbound_or_wrong_numeric_commands(self):
        for label in ("missing", "wrong pin", "wrong world", "wrong arm", "altered binding",
                      "duplicate", "equals option", "no prefix"):
            with self.subTest(label=label):
                args = fixture()
                spec = args["policy_specs_by_world"][9101][0]
                cmd = spec["command"]
                if label == "missing": del cmd[-4:]
                elif label == "wrong pin": cmd[-1] = digest("wrong")
                elif label == "wrong world": spec["command"] = copy.deepcopy(args["policy_specs_by_world"][9102][0]["command"])
                elif label == "wrong arm": spec["command"] = copy.deepcopy(args["policy_specs_by_world"][9101][1]["command"])
                elif label == "altered binding":
                    b = json.loads(cmd[-3])
                    b["planning_draws"] = 31
                    cmd[-3], cmd[-1] = canonical(b), digest(b)
                elif label == "duplicate": cmd[:0] = ["--binding-json", cmd[-3]]
                elif label == "equals option": cmd[:0] = ["--binding-sha256=" + cmd[-1]]
                elif label == "no prefix": del cmd[:-4]
                with self.assertRaises(ValueError):
                    builder.build_artificial_campaign(**args)

    def test_caller_runtime_prefixes_are_not_assumed(self):
        args = fixture()
        for wid, specs in args["policy_specs_by_world"].items():
            for spec in specs:
                prefix = ["/caller/gnu-env", "-i", "LANG=C", "--", "/caller/launch shim", spec["policy_id"]]
                if spec["policy_id"] in ROSTER[:4]:
                    spec["command"] = prefix + spec["command"][-4:]
                elif spec["policy_id"] == "learned_ranker":
                    spec["command"] = prefix + ["--world", str(wid)]
                else:
                    spec["command"] = prefix
        result = builder.build_artificial_campaign(**args)
        for source in result["campaign"]["sources"]:
            self.assertEqual(source["plan"]["policy_specs"],
                             args["policy_specs_by_world"][source["plan"]["world"]["world_id"]])

    def test_reject_service_roster_capacity_identity_and_private_text_injection(self):
        mutations = [
            ("missing service", lambda a: a["services"].pop("dpo_llm")),
            ("extra service", lambda a: a["services"].update(extra={})),
            ("recipe changed", lambda a: a["services"]["sft_llm"]["identity"]["numeric_recipe"].update(temperature=1.)),
            ("identity changed", lambda a: a["services"]["sft_llm"]["identity"].update(adapter_sha256=digest("wrong"))),
            ("persistent made CPU", lambda a: a["policy_specs_by_world"][9101][5].update(kind="task_process")),
            ("missing identity pin", lambda a: a["policy_specs_by_world"][9101][5].update(identity_sha256=None)),
            ("private text", lambda a: a["policy_specs_by_world"][9101][0].update(objective="Y = 1 + 2X")),
        ]
        for key, value in (("max_requests", 7), ("max_requests", 9), ("max_requests", True),
                           ("startup_seconds", float("inf")), ("cleanup_seconds", 0), ("private_score", 0.)):
            mutations.append((key + repr(value), lambda a, k=key, v=value: a["services"]["frozen_llm"].update({k: v})))
        for label, mutate in mutations:
            with self.subTest(label=label):
                args = fixture()
                mutate(args)
                with self.assertRaises(ValueError):
                    builder.build_artificial_campaign(**args)
        for keyword, value in (("scope", "development"), ("objective", "Y = 1 + 2X"),
                               ("world_ids", [17, 4]), ("acquisition_cap_units", 5), ("output", "/forbidden")):
            with self.subTest(keyword=keyword), self.assertRaises(TypeError):
                builder.build_artificial_campaign(**fixture(), **{keyword: value})


if __name__ == "__main__":
    unittest.main()
