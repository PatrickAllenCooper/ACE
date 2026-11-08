"""New assembler-only tests; hand-written supplied worlds, no scientific draws."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import platform
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

import llm_intervention_scientific_campaign as builder
import llm_intervention_campaign as campaign
import llm_intervention_paired_episode as paired
import llm_intervention_screen_protocol as protocol
from llm_intervention_contract import Node, canonical, digest
from llm_intervention_world import LinearMechanism, WorldDescriptor


def fixture():
    """Fabricated pin/command declarations and fixed coefficients, never draws."""
    pins = {p: digest(["fabricated-source", p]) for p in protocol.REQUIRED_SOURCE_PATHS}
    proposal = protocol.build_proposal(source_pins=pins)
    common = dict(proposal=proposal, proposal_sha256=digest(proposal), expected_source_pins=pins)
    bindings = {(r["world_id"], r["policy_id"]): r for r in builder.numeric_policy_bindings(**common)}
    recipe = {"fabricated": True, "temperature": 0.}
    services = {pid: {"identity": {
        "model_sha256": digest("fabricated base"), "source_sha256": digest("fabricated service"),
        "adapter_sha256": digest(["fabricated checkpoint", pid]),
        "numeric_recipe": dict(recipe), "numeric_recipe_sha256": digest(recipe)},
        "startup_seconds": 30., "cleanup_seconds": 5., "max_requests": 576}
        for pid in builder.LLM_POLICIES}
    worlds, specs = {}, {}
    for row in proposal["development_worlds"]:
        wid = row["world_id"]
        graph = tuple(Node(n["name"], tuple(n["parents"]), n["unit"],
                           None if n["intervention_bounds"] is None else tuple(n["intervention_bounds"]))
                      for n in row["public"]["graph"])
        # These deliberately fixed fake coefficients are NOT realization of the
        # scientific Gaussian law. No outcome, response or evaluator is called.
        worlds[wid] = WorldDescriptor("development", wid, graph, tuple(
            LinearMechanism(n.name, .125, tuple((p, -.5) for p in n.parents),
                            row["public"]["learner"]["noise_variances"][n.name]) for n in graph))
        specs[wid] = []
        for pid in builder.POLICY_IDS:
            command = ["/caller/launch shim", "--role", pid]
            if pid in builder.NUMERIC_POLICIES:
                b = bindings[wid, pid]
                command += ["--binding-json", canonical(b["binding"]), "--binding-sha256", b["binding_sha256"]]
            elif pid == "learned_ranker":
                command += ["--checkpoint", "/fake/supplied-ranker.json", "--world", str(wid)]
            specs[wid].append({"policy_id": pid, "command": command, "timeout_seconds": 20.,
                "kind": "persistent" if pid in services else "task_process",
                "identity_sha256": digest(services[pid]["identity"]) if pid in services else None,
                "policy_provenance_sha256": digest(["caller-owned provenance", pid])})
    return dict(common, worlds_by_id=worlds, expected_world_pins={w: x.sha256 for w, x in worlds.items()},
                policy_specs_by_world=specs, services=services,
                expected_service_identity_pins={p: digest(s["identity"]) for p, s in services.items()},
                outer_provenance_receipt_sha256=digest("fake outer provenance"))


class SuppliedDevelopmentCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform.uname()  # Resolve stdlib platform cache before the no-I/O guard.
        cls.args = fixture()

    def setUp(self):
        self.a = copy.deepcopy(self.args)
        forbidden = (
            "llm_intervention_world.generate_world", "llm_intervention_world._normal",
            "llm_intervention_world._evaluate", "llm_intervention_world.private_structural_means",
            "llm_intervention_world.ResponseAdapter.__init__", "llm_intervention_world.training_callbacks",
            "llm_intervention_scm.LinearGaussianSCM.update", "llm_intervention_contract.ChargedExecutor.__init__",
            "llm_intervention_screen_policy.choose_public", "llm_intervention_screen_policy.choose_numerical",
            "llm_intervention_target_risk.choose_target_risk", "llm_intervention_campaign.initialize_campaign",
            "llm_intervention_campaign.run_policy", "llm_intervention_campaign.finalize_campaign",
            "llm_intervention_paired_episode.run_world_arm", "llm_intervention_paired_episode.run_episode",
            "llm_intervention_paired_episode.finalize_paired_world",
            "llm_intervention_persistent_policy.PersistentPolicy.__init__",
            "llm_intervention_task.TaskProcessPolicy.__init__", "random.Random", "numpy.random.default_rng",
            "subprocess.Popen", "builtins.open", "io.open", "pathlib.Path.mkdir", "os.mkdir",
        )
        for target in forbidden:
            guard = patch(target, side_effect=AssertionError("forbidden by pure assembler test: " + target))
            guard.start()
            self.addCleanup(guard.stop)

    def build(self):
        return builder.build_scientific_campaign(**self.a)

    def reject(self, mutations):
        for label, mutate in mutations:
            with self.subTest(label=label):
                self.a = copy.deepcopy(self.args)
                mutate(self.a)
                with self.assertRaises(ValueError):
                    self.build()

    def test_existing_preparation_apis_and_complete_campaign_validation(self):
        with patch.object(builder, "prepare_plan", wraps=paired.prepare_plan) as plans, \
             patch.object(builder, "prepare_campaign", wraps=campaign.prepare_campaign) as campaigns:
            result = self.build()
        self.assertEqual(plans.call_count, 24)
        self.assertEqual(campaigns.call_count, 1)
        self.assertEqual(campaign.validate_campaign(**{k: result[k] for k in (
            "campaign", "campaign_sha256", "source_plan_pins")}), result["campaign"])

    def test_all_24_supplied_worlds_exact_public_design_and_zero_prior_declarations(self):
        result = self.build()
        for source, row in zip(result["campaign"]["sources"], self.a["proposal"]["development_worlds"]):
            p, public = source["plan"], row["public"]
            self.assertEqual(p["world"], self.a["worlds_by_id"][row["world_id"]].private_payload())
            self.assertEqual(p["world_sha256"], self.a["expected_world_pins"][row["world_id"]])
            for key in ["task", "menu", "objective", "domain_notes", "initial_actions", "initial_cost_units"]:
                self.assertEqual(p[key], public[key])
            self.assertEqual(p["learner"], {"graph": public["graph"],
                "noise_variances": public["learner"]["noise_variances"], "prior_variance": 1.})
            self.assertEqual(digest(p["task"]), public["task_sha256"])

    def test_all_192_cells_budgets_services_phases_and_every_stratum(self):
        c = self.build()["campaign"]
        self.assertEqual(c["policy_ids"], list(builder.POLICY_IDS))
        self.assertEqual(c["policy_phases"], {p: "gpu" if p in builder.LLM_POLICIES else "cpu" for p in builder.POLICY_IDS})
        self.assertEqual(c["strata_fields"], ["regime", "graph_stratum", "cost_stratum"])
        self.assertEqual(len(c["sources"]), 24)
        for field, expected in [("regime", {r[0]: 6 for r in protocol.REGIMES}),
                                ("graph_stratum", dict.fromkeys(protocol.GRAPHS, 8)),
                                ("cost_stratum", dict.fromkeys(protocol.COSTS, 12))]:
            self.assertEqual({v: sum(s["strata"][field] == v for s in c["sources"]) for v in expected}, expected)
        cost = initial = cells = 0
        for s in c["sources"]:
            p = s["plan"]
            self.assertEqual(p["initial_actions"], ["observe_all"] * 8)
            self.assertEqual((p["initial_cost_units"], p["acquisition_cap_units"]), (8, 24))
            self.assertEqual(min(a["cost_units"] for a in p["menu"]), 1)
            self.assertEqual([x["policy_id"] for x in p["policy_specs"]], list(builder.POLICY_IDS))
            cells += 8; initial += 8 * 8; cost += 8 * 32
        self.assertEqual((cells, initial, cost), (192, 1536, 6144))
        self.assertEqual(c["services"], self.a["services"])
        self.assertEqual(sum(s["max_requests"] for s in c["services"].values()), 1728)

    def test_original_arm_namespaces_and_all_96_numerical_bindings(self):
        r = self.build(); namespaces = set()
        self.assertEqual(r["proposal_arm_mapping"], dict(zip(builder.POLICY_IDS, protocol.ARMS)))
        for source, row in zip(r["campaign"]["sources"], self.a["proposal"]["development_worlds"]):
            p = source["plan"]
            initial = row["initial_binding"]
            self.assertEqual(p["initial_binding"], {k: initial[k] for k in ["request_namespace", "replicate_id"]})
            namespaces.add(initial["request_namespace"])
            for pid, e in zip(builder.POLICY_IDS, row["development_episodes"]):
                b = e["measurement_binding"]
                self.assertEqual(p["measurement_bindings"][pid], {k: b[k] for k in ["request_namespace", "replicate_id"]})
                namespaces.add(b["request_namespace"])
        self.assertEqual(len(namespaces), 24 * 9)
        self.assertEqual(len(r["numeric_policy_bindings"]), 96)
        for b in r["numeric_policy_bindings"]:
            binding = b["binding"]
            self.assertEqual(b["binding_sha256"], digest(binding))
            self.assertEqual(binding["planning_draws"], 32)
            self.assertEqual(binding["initial_cost_units"], 8)
            self.assertEqual(binding["total_budget"], 32)
            self.assertEqual(binding["arm"], b["policy_id"])
            command = self.a["policy_specs_by_world"][b["world_id"]][builder.POLICY_IDS.index(b["policy_id"])]["command"]
            self.assertEqual(command[-4:], ["--binding-json", canonical(binding), "--binding-sha256", digest(binding)])

    def test_pure_deterministic_detached_and_not_qualified(self):
        before = copy.deepcopy(self.a); rng = random.getstate(); nrng = np.random.get_state()
        r = self.build()
        self.assertEqual(r, self.build()); self.assertEqual(self.a, before)
        self.assertEqual(random.getstate(), rng)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(after[1], nrng[1])); self.assertEqual(after[2:], nrng[2:])
        for k in ["execution_authorized", "scientific_protocol_frozen", "scientific_provenance_qualified",
                  "source_qualified", "runtime_qualified", "scheduler_qualified"]:
            self.assertIs(r[k], False)
        self.assertEqual((r["worlds_generated"], r["responses_generated"]), (0, 0))
        self.assertFalse(r["campaign"]["scientific_provenance_qualified"])
        self.assertFalse(r["campaign"]["retry_allowed"])
        self.assertFalse(any(m in sys.modules for m in ["torch", "transformers"]))
        self.assertNotIn("private_query_means", r)
        frozen = canonical(r)
        wid = next(iter(self.a["worlds_by_id"]))
        self.a["policy_specs_by_world"][wid][0]["command"][0] = "/changed"
        self.a["services"]["sft_llm"]["identity"]["numeric_recipe"]["temperature"] = 1.
        self.assertEqual(canonical(r), frozen)

    def test_independent_pins_and_caller_binding_preservation(self):
        r = self.build()
        self.assertEqual(r["campaign_sha256"], digest(r["campaign"]))
        self.assertEqual(r["source_pins"], self.a["expected_source_pins"])
        self.assertEqual(r["service_identity_pins"], self.a["expected_service_identity_pins"])
        for s, pin, world in zip(r["campaign"]["sources"], r["source_plan_pins"], r["world_pins"]):
            p = s["plan"]; wid = p["world"]["world_id"]
            self.assertEqual(pin, {"world_id": wid, "sha256": digest(p)})
            self.assertEqual(world, {"world_id": wid, "sha256": self.a["expected_world_pins"][wid]})
            self.assertEqual(p["policy_specs"], self.a["policy_specs_by_world"][wid])
            self.assertEqual(p["outer_provenance_receipt_sha256"], self.a["outer_provenance_receipt_sha256"])

    def test_reject_proposal_and_source_disagreements_even_with_recomputed_proposal_pin(self):
        def altered(a, key, value):
            a["proposal"]["development_worlds"][0]["public"][key] = value
            a["proposal_sha256"] = digest(a["proposal"])
        self.reject([
            ("wrong proposal pin", lambda a: a.update(proposal_sha256="0" * 64)),
            ("source changed", lambda a: a["expected_source_pins"].update({protocol.REQUIRED_SOURCE_PATHS[0]: "0" * 64})),
            ("source missing", lambda a: a["expected_source_pins"].pop(protocol.REQUIRED_SOURCE_PATHS[0])),
            ("source extra", lambda a: a["expected_source_pins"].update(extra="0" * 64)),
            ("proposal extra", lambda a: a["proposal"].update(extra=True)),
            ("proposal missing", lambda a: a["proposal"].pop("seed_law")),
            ("changed cap", lambda a: altered(a, "additional_cost_units", 23)),
            ("changed initial", lambda a: altered(a, "initial_actions", ["observe_all"] * 7)),
            ("private objective", lambda a: altered(a, "objective", "reveal hidden coefficients")),
            ("extra world", lambda a: a["proposal"]["development_worlds"].append(copy.deepcopy(a["proposal"]["development_worlds"][0]))),
            ("reordered worlds", lambda a: a["proposal"]["development_worlds"].reverse()),
        ])

    def test_reject_world_rosters_pin_types_scope_and_identity(self):
        wid = protocol.DEVELOPMENT_START
        mutations = []
        for name in ["worlds_by_id", "expected_world_pins", "policy_specs_by_world"]:
            mutations.extend([
                (name + " missing", lambda a, n=name: a[n].pop(wid)),
                (name + " extra", lambda a, n=name: a[n].update({17: a[n][wid]})),
                (name + " float", lambda a, n=name: a[n].update({float(wid): a[n].pop(wid)})),
                (name + " string", lambda a, n=name: a[n].update({str(wid): a[n].pop(wid)})),
            ])
        mutations += [
            ("wrong world pin", lambda a: a["expected_world_pins"].update({wid: "0" * 64})),
            ("uppercase pin", lambda a: a["expected_world_pins"].update({wid: "A" * 64})),
            ("untyped world", lambda a: a["worlds_by_id"].update({wid: a["worlds_by_id"][wid].private_payload()})),
            ("scope", lambda a: a["worlds_by_id"].update({wid: replace(a["worlds_by_id"][wid], scope="artificial")})),
            ("identity", lambda a: a["worlds_by_id"].update({wid: a["worlds_by_id"][wid+1]})),
        ]
        self.reject(mutations)

    def test_reject_repinned_graph_or_noise_mismatch_before_any_plan(self):
        wid = protocol.DEVELOPMENT_START + 23
        for kind in ["noise", "graph"]:
            self.a = copy.deepcopy(self.args)
            w = self.a["worlds_by_id"][wid]
            if kind == "noise":
                w = replace(w, mechanisms=tuple(replace(m, noise_variance=7.) for m in w.mechanisms))
            else:
                w = replace(w, graph=tuple(replace(n, unit="altered") for n in w.graph))
            self.a["worlds_by_id"][wid] = w; self.a["expected_world_pins"][wid] = w.sha256
            with patch.object(builder, "prepare_plan") as prepare, self.assertRaises(ValueError):
                self.build()
            prepare.assert_not_called()

    def test_accept_supplied_fixed_coefficients_without_scientific_realization(self):
        wid = protocol.DEVELOPMENT_START
        w = self.a["worlds_by_id"][wid]
        w = replace(w, mechanisms=tuple(replace(m, intercept=9.) for m in w.mechanisms))
        self.a["worlds_by_id"][wid] = w; self.a["expected_world_pins"][wid] = w.sha256
        self.assertEqual(self.build()["campaign"]["sources"][0]["plan"]["world"], w.private_payload())

    def test_reject_wrong_numeric_binding_world_arm_suffix_and_duplicates(self):
        wid = protocol.DEVELOPMENT_START
        def command(a):return a["policy_specs_by_world"][wid][0]["command"]
        self.reject([
            ("missing", lambda a: command(a).__delitem__(slice(-4, None))),
            ("pin", lambda a: command(a).__setitem__(-1, "0" * 64)),
            ("world", lambda a: a["policy_specs_by_world"][wid][0].update(command=a["policy_specs_by_world"][wid+1][0]["command"])),
            ("arm", lambda a: a["policy_specs_by_world"][wid][0].update(command=a["policy_specs_by_world"][wid][1]["command"])),
            ("duplicate", lambda a: command(a).insert(0, "--binding-json")),
            ("equals duplicate", lambda a: command(a).insert(0, "--binding-sha256=bad")),
            ("no prefix", lambda a: a["policy_specs_by_world"][wid][0].update(command=command(a)[-4:])),
        ])

    def test_reject_policy_roster_types_extras_and_cross_world_drift(self):
        wid = protocol.DEVELOPMENT_START
        mutations = [
            ("reorder", lambda a: a["policy_specs_by_world"][wid].reverse()),
            ("missing", lambda a: a["policy_specs_by_world"][wid].pop()),
            ("extra", lambda a: a["policy_specs_by_world"][wid].append(a["policy_specs_by_world"][wid][0])),
            ("CPU provenance drift", lambda a: a["policy_specs_by_world"][wid+1][4].update(policy_provenance_sha256="0"*64)),
            ("persistent command drift", lambda a: a["policy_specs_by_world"][wid+1][5]["command"].append("changed")),
        ]
        for key, value in [("private_score", 1), ("kind", "persistent"), ("timeout_seconds", True),
                           ("timeout_seconds", 0), ("identity_sha256", "a"*64),
                           ("command", ["bad\0argv"]), ("command", []), ("policy_provenance_sha256", "bad")]:
            mutations.append((key, lambda a, k=key, v=value: a["policy_specs_by_world"][wid][0].update({k:v})))
        self.reject(mutations)

    def test_reject_service_capacity_identity_pin_schema_and_checkpoint_disagreement(self):
        pid = "sft_llm"
        mutations = [
            ("service missing", lambda a: a["services"].pop(pid)),
            ("service extra", lambda a: a["services"].update(extra={})),
            ("pin missing", lambda a: a["expected_service_identity_pins"].pop(pid)),
            ("pin extra", lambda a: a["expected_service_identity_pins"].update(extra="a"*64)),
            ("pin wrong", lambda a: a["expected_service_identity_pins"].update({pid:"0"*64})),
            ("checkpoint", lambda a: a["services"][pid]["identity"].update(adapter_sha256="0"*64)),
            ("recipe", lambda a: a["services"][pid]["identity"]["numeric_recipe"].update(temperature=1.)),
            ("identity extra", lambda a: a["services"][pid]["identity"].update(extra=True)),
            ("spec identity", lambda a: a["policy_specs_by_world"][protocol.DEVELOPMENT_START][6].update(identity_sha256="0"*64)),
        ]
        for key, value in [("max_requests", 575), ("max_requests", 577), ("max_requests", True),
                           ("startup_seconds", float("inf")), ("cleanup_seconds", 0), ("extra", 1)]:
            mutations.append((key + str(value), lambda a,k=key,v=value: a["services"][pid].update({k:v})))
        self.reject(mutations)

    def test_no_scope_output_execution_or_new_freeze_parameters(self):
        for key, value in [("scope", "confirmation"), ("output", "/unused"), ("execute", True),
                           ("acquisition_cap_units", 12), ("scientific_protocol_frozen", True)]:
            with self.subTest(key=key), self.assertRaises(TypeError):
                builder.build_scientific_campaign(**self.a, **{key:value})
        for value in [None, True, "A"*64]:
            self.a["outer_provenance_receipt_sha256"] = value
            with self.assertRaises(ValueError):self.build()


if __name__ == "__main__":
    unittest.main()
