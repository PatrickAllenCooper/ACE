"""New fabricated saved-file joins only; never starts workers or draws responses.

Each invocation retains its exclusive fixture directory, including failing cases.
Run directly with -B; no old suites, model/tensor loads, installs or GPU work.
"""
import copy
import os
from pathlib import Path
import resource
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import llm_intervention_campaign_service_evidence as evidence
import llm_intervention_campaign as campaign
import llm_intervention_paired_episode as paired
from llm_intervention_contract import VERSION as CONTRACT_VERSION, canonical, digest
from llm_intervention_persistent_policy import VERSION as PROTOCOL_VERSION
from llm_intervention_task import VERSION as TASK_VERSION, task_prompt

POLICIES = ["uniform_random", "menu_coverage", "local_information", "target_risk",
            "learned_ranker", "frozen_llm", "sft_llm", "dpo_llm"]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(paired._raw(value))


def inventory(path, excluded=()):
    return {str(p.relative_to(path)): paired._sha(p.read_bytes())
            for p in sorted(path.rglob("*")) if p.is_file() and str(p.relative_to(path)) not in excluded}


def journal(events):
    previous, lines = None, []
    for i, event in enumerate(events):
        row = dict(event, previous_sha256=previous, monotonic_ns=i+1)
        row["sha256"] = digest(row)
        previous = row["sha256"]
        lines.append(paired._raw(row))
    return b"".join(lines)


class SavedFixture:
    def __init__(self, root, sequences=(('act', 'stop'), ('stop',)), policy_id="frozen_llm"):
        self.root, self.pid = root, policy_id
        self.croot = root / "campaign"
        self.attempt = self.croot / f"policy-{POLICIES.index(policy_id):02d}"
        self.client = self.attempt / "service"
        self.worker = root / "service-workers" / policy_id
        self.client.mkdir(parents=True)
        self.worker.mkdir(parents=True)
        self.freeze_pin = digest(["byte hash supplied independently", policy_id])
        recipe = {"max_tokens": 8192, "max_new_tokens": 32}
        identities = {pid: {"model_sha256": digest("fake-model"), "source_sha256": digest("fake-source"),
                           "adapter_sha256": digest(["checkpoint, not outer acceptance", pid]),
                           "numeric_recipe": recipe, "numeric_recipe_sha256": digest(recipe)}
                      for pid in POLICIES[-3:]}
        self.identity = identities[policy_id]
        self.identity_pin = digest(self.identity)
        graph = [{"name": "X", "parents": [], "unit": "unit", "intervention_bounds": [-2., 2.]}]
        action = {"action_id": "observe_all", "kind": "observe", "targets": [], "measure": ["X"], "cost_units": 1}
        self.action = action
        specs = [{"policy_id": pid, "kind": "persistent" if pid in identities else "task_process",
                  "command": ["/never-executed/python", "/never-executed/shim", pid],
                  "timeout_seconds": 30., "identity_sha256": digest(identities[pid]) if pid in identities else None,
                  "policy_provenance_sha256": digest(pid)} for pid in POLICIES]
        sources = []
        for i, wid in enumerate((9101, 9102)):
            world = {"version": "ace-linear-gaussian-world-v1", "scope": "artificial", "world_id": wid,
                     "graph": graph, "mechanisms": [{"node": "X", "intercept": 0., "coefficients": [], "noise_variance": 1.}]}
            plan = {"version": paired.VERSION, "world": world, "world_sha256": digest(world),
                    "task": {"target": "X", "queries": [dict(action, action_id="query")], "weights": [1.],
                             "scale": 2.+i, "normalized_mse_tolerance": .04},
                    "menu": [action], "learner": {"graph": graph, "noise_variances": {"X": 1.}, "prior_variance": 1.},
                    "initial_actions": ["observe_all", "observe_all"], "initial_cost_units": 2,
                    "acquisition_cap_units": 4, "initial_binding": {"request_namespace": "initial", "replicate_id": "shared"},
                    "measurement_bindings": {pid: {"request_namespace": pid, "replicate_id": "measurement"} for pid in POLICIES},
                    "policy_specs": specs, "objective": "Fabricated public objective", "domain_notes": "Pure saved-file fixture",
                    "outer_provenance_receipt_sha256": digest("fixture-provenance"),
                    "scientific_provenance_qualified": False, "retry_allowed": False}
            sources.append({"plan": plan, "plan_sha256": digest(plan), "strata": {"case": str(i)}})
        self.campaign = {"version": campaign.VERSION, "scope": "artificial", "sources": sources,
                         "strata_fields": ["case"], "policy_ids": POLICIES,
                         "policy_phases": {pid: "gpu" if pid in identities else "cpu" for pid in POLICIES},
                         "services": {pid: {"identity": identity, "startup_seconds": 30., "cleanup_seconds": 2., "max_requests": 8}
                                      for pid, identity in identities.items()},
                         "scientific_provenance_qualified": False, "retry_allowed": False}
        self.campaign_pin = digest(self.campaign)
        self.source_pins = campaign._source_pins(sources)
        put(self.croot / "campaign.json", self.campaign)
        put(self.croot / "source-plan-pins.json", self.source_pins)
        settings = self.campaign["services"][policy_id]
        put(self.client / "launch.json", {"command": specs[POLICIES.index(policy_id)]["command"],
             "timeout_seconds": 30., "identity_sha256": self.identity_pin, "expected_identity": self.identity,
             **{k: settings[k] for k in ("startup_seconds", "cleanup_seconds", "max_requests")}})
        put(self.client / "process.json", {"pid": 123, "pgid": 123})
        put(self.client / "ready.json", {"status": "ready", "identity_sha256": self.identity_pin})
        put(self.client / "observed-handshake.json", {"type": "ready", "version": PROTOCOL_VERSION,
             "identity": self.identity, "identity_sha256": self.identity_pin})
        put(self.worker / "authenticated.json", {"freeze_sha256": self.freeze_pin, "identity": self.identity})
        for i, phase in enumerate(("model_cpu_load_start", "model_cpu_load_complete", "model_ready")):
            put(self.worker / f"phase-{i:03d}.json", {"phase": phase, "identity": self.identity, "gpu_allocated_bytes": 1024})
        self.ordinal, self.generation_count, self.failure = 0, 0, False
        self.arms = []
        for i, seq in enumerate(sequences):
            if seq is None:
                self.arms.append({"world_id": 9101+i, "status": "unstarted", "path": None,
                                  "receipt_sha256": None, "reason": "retained service failure; no restart"})
                continue
            self.make_arm(i, seq)
        cleanup = {"pgid": 123, "group_still_exists": False, "leader_returncode": 1 if self.failure else 0,
                   "term_sent": False, "kill_sent": False, "errors": []}
        if self.failure:
            close_name = "failure.json"
            close = {"status": "failed", "stage": "request_exchange", "cleanup": cleanup,
                     "completed_requests": self.ordinal, "last_completed_request_id": self.ordinal-1,
                     "retry_allowed": False, "message": "fixture worker failure"}
            put(self.worker / "failure.json", {"type": "RuntimeError", "message": "fixture worker failure",
                 "new_training_updates": 0, "environment_responses": 0, "retry_allowed": False,
                 "accepted_generation_records": sorted(p.name for p in self.worker.glob("generation-*.json")),
                 "available_raw_generation_records": sorted(p.name for p in self.worker.glob("raw-generation-*.json"))})
        else:
            close_name = "closed.json"
            close = {"status": "closed", "cleanup": cleanup, "completed_requests": self.ordinal,
                     "last_request_id": self.ordinal-1}
            put(self.worker / "terminal.json", {"status": "complete", "freeze_sha256": self.freeze_pin,
                 "generation_count": self.generation_count, "new_training_updates": 0, "environment_responses": 0})
        put(self.client / close_name, close)
        self.close_name = close_name
        self.reseal()

    def state(self, plan, history, terminal=False):
        n = len(history)
        return {"version": CONTRACT_VERSION, "objective": plan["objective"], "domain_notes": plan["domain_notes"],
                "graph": plan["world"]["graph"], "menu": plan["menu"], "history": copy.deepcopy(history),
                "scm": [{"node": "X", "terms": ["intercept"], "parameter_mean": [0.],
                         "parameter_covariance": [[1./(1+n)]], "noise_variance": 1., "eligible_rows": n}],
                "spent_units": n, "remaining_units": 6-n, "requests": n, "returned_responses": n, "terminal": terminal}

    def make_arm(self, i, sequence):
        plan = self.campaign["sources"][i]["plan"]
        directory = self.attempt / f"world-{i:02d}"
        ep = evidence._episode_plan(plan, self.pid)
        for name, value in (("paired-plan.json", plan), ("world.json", plan["world"]),
                            ("task.json", plan["task"]), ("learner.json", plan["learner"]), ("episode/plan.json", ep)):
            put(directory / name, value)
        before = self.ordinal
        put(self.client / f"task-binding-{i:04d}.json", {"binding_index": i, "task": plan["task"],
             "task_sha256": digest(plan["task"]),
             "previous_task_sha256": digest(self.campaign["sources"][i-1]["plan"]["task"]) if i else None,
             "last_completed_session_request_id": before-1, "worker_pid": 123, "model_reloaded": False})
        events = [{"event": "begin", "version": CONTRACT_VERSION, "budget_units": 6,
                   "graph": ep["graph"], "menu": ep["menu"]}]
        history = []
        proposals = 0

        def propose(completion, provenance, fail_response=False):
            nonlocal proposals
            pid = proposals
            proposals += 1
            events.append({"event": "proposal", "proposal_id": pid, "completion": completion, "provenance": provenance})
            if completion == '{"stop":true}':
                events.append({"event": "stop", "proposal_id": pid, "completion": completion, "spent_units": len(history)})
            else:
                rid = len(history)
                events.append({"event": "reserve", "proposal_id": pid, "request_id": rid, "action": self.action,
                               "completion": completion, "cost_units": 1})
                if fail_response:
                    events.append({"event": "failure", "request_id": rid, "response_returned_to_driver": None,
                                   "exception_type": "RuntimeError", "message": "fixture worker failure", "spent_units": rid+1})
                    return
                receipt = {"request_id": rid, "action_id": "observe_all", "kind": "observe", "targets": {},
                           "values": {"X": 0.}, "cost_units": 1}
                history.append(receipt)
                events.append({"event": "response", **receipt})
                put(directory / f"return-{rid:03d}.json", {"request_id": rid, "action": self.action,
                     "returned_to_wrapper": True, "response": {"python_type": "builtins.dict", "raw_response_repr": "{'X': 0.0}",
                     "raw_json_available": True, "raw_json": {"X": 0.}}})

        for _ in range(2):
            propose('{"action_id":"observe_all"}', {"kind": "shared_initial_schedule"})
        failed, response_failed = False, False
        for decision, mode in enumerate(sequence):
            state = self.state(plan, history)
            envelope = {"version": TASK_VERSION, "state": state, "task": plan["task"]}
            ordinal = self.ordinal
            put(self.client / f"request-{ordinal:04d}.episode.json", {
                "episode_directory": f"/saved/remote/campaign/{self.attempt.name}/world-{i:02d}/episode",
                "decision_index": decision, "session_request_id": ordinal, "task_sha256": digest(plan["task"]),
                "task_binding_index": i, "identity_sha256": self.identity_pin, "status": "planned_before_transmission",
                "scope": "client_only_episode_to_session_mapping"})
            put(self.client / f"request-{ordinal:04d}.input.json", {"type": "request", "request_id": ordinal,
                 "identity_sha256": self.identity_pin, "envelope": envelope, "envelope_sha256": digest(envelope)})
            put(self.worker / f"request-{ordinal:04d}.json", {"ordinal": ordinal, "envelope_sha256": digest(envelope),
                 "identity": self.identity, "tokens_available_at_request_start": False})
            put(self.worker / f"raw-generation-{ordinal:04d}.started.json", {"tokens_available": False, "status": "generation_invoked"})
            if mode == "fail":
                self.failure = failed = True
                events.append({"event": "driver_failure", "exception_type": "RuntimeError",
                               "message": "fixture worker failure", "spent_units": len(history)})
                break
            completion = '{"stop":true}' if mode == "stop" else '{"action_id":"observe_all"}'
            tokens = [10+ordinal, 20+ordinal, 30+ordinal, 151645]
            binary = struct.pack("<4q", *tokens)
            (self.worker / f"raw-generation-{ordinal:04d}.bin").write_bytes(binary)
            put(self.worker / f"raw-generation-{ordinal:04d}.json", {"status": "returned_unaccepted",
                 "tokens_available": True, "returned_type": "Tensor", "shape": [1, 4], "dtype": "torch.int64",
                 "numel": 4, "captured_elements": 4, "truncated": False,
                 "raw_sha256": paired._sha(binary), "token_prefix": tokens})
            record = {"version": "ace-shared-task-language-policy-v1", "envelope_sha256": digest(envelope),
                      "task_sha256": digest(plan["task"]), "prompt_sha256": paired._sha(task_prompt(envelope).encode()),
                      "prompt_tokens": 2, "longest_command_tokens": 4, "max_tokens": 8192, "max_new_tokens": 32,
                      "legal_commands": 2, "enable_thinking": False, "prompt_ids": tokens[:2], "sequence_ids": tokens,
                      "completion": completion, "new_training_updates": 0, "environment_callbacks_available": False}
            put(self.worker / f"generated-{ordinal:04d}.unaccepted.json", record)
            put(self.worker / f"generation-{ordinal:04d}.json", {**record, "ordinal": ordinal, "elapsed_seconds": .1,
                 "cuda_interval_ms": 1., "peak_gpu_allocated_bytes": 1024, "peak_gpu_reserved_bytes": 2048,
                 "tensor_versions_unchanged": True, "cross_request_cache_supplied": False})
            put(self.worker / f"phase-{ordinal+3:03d}.json", {"phase": "generation_complete", "ordinal": ordinal,
                 "envelope_sha256": digest(envelope)})
            wire = {"type": "completion", "request_id": ordinal, "identity_sha256": self.identity_pin,
                    "envelope_sha256": digest(envelope), "completion": completion}
            put(self.client / f"request-{ordinal:04d}.observed-response.json", wire)
            put(self.client / f"request-{ordinal:04d}.accepted.json", {"request_id": ordinal,
                 "response_sha256": digest(wire), "completion": completion})
            self.ordinal += 1
            self.generation_count += 1
            propose(completion, {"kind": "process_policy", "decision_index": decision, "public_state_sha256": digest(state)},
                    fail_response=mode == "response_fail")
            if mode == "response_fail":
                failed = response_failed = True
                break
        if not failed and sequence[-1] != "stop":
            propose('{"stop":true}', {"kind": "budget_cap"})
        raw_journal = journal(events)
        (directory / "episode/journal.jsonl").write_bytes(raw_journal)
        if failed:
            final_state = self.state(plan, history, True)
            if response_failed:
                final_state.update(spent_units=len(history)+1, remaining_units=5-len(history), requests=len(history)+1)
            disposition = {"status": "failed", "exception_type": "RuntimeError", "message": "fixture worker failure",
                           "initialized_responses": 2, "completed_policy_calls": self.ordinal-before,
                           "partial_state": final_state, "retry_allowed": False}
        else:
            disposition = {"status": "complete", "initialized_responses": 2, "policy_decisions": self.ordinal-before,
                           "final_state": self.state(plan, history, True), "scientific_evaluation_performed": False}
        put(directory / ("episode/failure.json" if failed else "episode/terminal.json"), disposition)
        put(directory / "execution-diagnostics.json", {"episode_returned_cleanly": not failed,
             "adapters_closed_cleanly": True, "errors": []})
        inputs = {"world.json": paired._raw(plan["world"]), "task.json": paired._raw(plan["task"]),
                  "learner.json": paired._raw(plan["learner"]), "episode-plan.json": paired._raw(ep),
                  "executor.jsonl": raw_journal, "episode-disposition.json": paired._raw(disposition)}
        inputs.update({p.name: p.read_bytes() for p in directory.glob("return-*.json")})
        inputs["outer-outcome.json"] = paired._raw({"version": paired.VERSION, "scope": "artificial", "world_id": 9101+i,
            "policy_id": self.pid, "status": "failed" if failed else "complete",
            "reason": "episode_failure:fixture" if failed else "policy_stop" if sequence[-1] == "stop" else "budget_exhausted",
            "episode_returned_cleanly": not failed, "adapters_closed_cleanly": True,
            "episode_disposition_sha256": paired._sha(inputs["episode-disposition.json"]),
            "outer_provenance_receipt_sha256": plan["outer_provenance_receipt_sha256"]})
        (directory / "sealed-inputs").mkdir()
        for name, raw in inputs.items():
            (directory / "sealed-inputs" / name).write_bytes(raw)
        receipt = {"version": paired.VERSION, "plan_sha256": digest(plan), "policy_id": self.pid,
                   "status": "raw_ready", "unresolved_reason": None,
                   "session": {"kind": "persistent", "identity_sha256": self.identity_pin, "pid": 123,
                               "request_before": before-1, "count_before": before,
                               "request_after": self.ordinal-1, "count_after": self.ordinal},
                   "input_sha256": {n: paired._sha(raw) for n, raw in inputs.items()},
                   "source_evidence_sha256": inventory(directory), "scientific_provenance_qualified": False,
                   "private_scores_computed": False, "new_training_updates": 0, "retry_allowed": False}
        put(directory / "arm-receipt.json", receipt)
        put(directory / "arm-receipt.candidate.json", receipt)
        self.arms.append({"world_id": 9101+i, "status": "failed" if failed else "completed", "path": directory.name,
                          "receipt_sha256": paired._sha(paired._raw(receipt)), "reason": "retained failure" if failed else None})

    def reseal(self):
        for arm in self.arms:
            if not arm["path"]:
                continue
            directory = self.attempt / arm["path"]
            receipt = evidence._read(directory / "arm-receipt.json")
            receipt["input_sha256"] = inventory(directory / "sealed-inputs")
            receipt["source_evidence_sha256"] = inventory(directory, ("arm-receipt.json", "arm-receipt.candidate.json"))
            put(directory / "arm-receipt.json", receipt)
            put(directory / "arm-receipt.candidate.json", receipt)
            arm["receipt_sha256"] = paired._sha(paired._raw(receipt))
        close_path = self.client / self.close_name
        attempt = {"version": campaign.VERSION, "campaign_sha256": self.campaign_pin, "source_plan_pins": self.source_pins,
                   "policy_id": self.pid, "phase": "gpu", "arms": self.arms,
                   "service": {"status": "recorded", "file": "service/"+self.close_name, "sha256": paired._sha(close_path.read_bytes())},
                   "errors": [], "evidence_sha256": inventory(self.attempt, ("attempt.json", "attempt.candidate.json")),
                   "retry_allowed": False, "private_scores_computed": False}
        put(self.attempt / "attempt.json", attempt)
        put(self.attempt / "attempt.candidate.json", attempt)
        self.args = {"campaign": self.campaign, "campaign_sha256": self.campaign_pin, "source_plan_pins": self.source_pins,
                     "campaign_root": self.croot, "policy_id": self.pid, "worker_output_dir": self.worker,
                     "freeze_sha256": self.freeze_pin, "identity_sha256": self.identity_pin,
                     "attempt_sha256": paired._sha(paired._raw(attempt))}

    def alter(self, where, name, change, reseal=True):
        path = getattr(self, where) / name
        original = path.read_bytes()
        backup = self.root / "mutation-originals" / where / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
        value = campaign._decode(original)
        change(value)
        put(path, value)
        if reseal:
            self.reseal()


class CampaignServiceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("ACE_CAMPAIGN_SERVICE_EVIDENCE_TEST_OUTPUT")
        cls.root = Path(explicit) if explicit else Path(tempfile.mkdtemp(prefix="ace-campaign-service-evidence-"))
        if explicit:
            cls.root.mkdir(exist_ok=False)
        print("Retained fabricated attempts:", cls.root, flush=True)

    def setUp(self):
        self.path = self.root / self._testMethodName
        self.path.mkdir()
        for target in ("llm_intervention_world.generate_world", "llm_intervention_world.ResponseAdapter.__call__",
                       "llm_intervention_world.private_structural_means", "llm_intervention_paired_episode.run_episode",
                       "llm_intervention_campaign.run_policy", "llm_intervention_campaign.finalize_campaign"):
            guard = patch(target, side_effect=AssertionError("pure saved files only"))
            guard.start()
            self.addCleanup(guard.stop)

    def fixture(self, name="case", **kwargs):
        return SavedFixture(self.path / name, **kwargs)

    def test_dynamic_three_requests_and_two_tasks_read_only(self):
        f = self.fixture()
        before = inventory(f.root)
        result = evidence.validate_campaign_service(**f.args, worker_evidence_sha256=inventory(f.worker))
        self.assertTrue(result["operational_success"])
        self.assertEqual(result["counts"]["accepted_requests"], 3)
        self.assertEqual(result["counts"]["response_attempts"], 5)
        self.assertEqual(result["counts"]["policy_stops"], 2)
        self.assertEqual(result["counts"]["feedback_to_next_decision"], 1)
        self.assertEqual([r["decision_index"] for r in result["joins"]], [0, 1, 0])
        self.assertEqual(result["joins"][0]["response_request_id"], 2)
        self.assertEqual(result["joins"][0]["charged_units"], 1)
        self.assertEqual(before, inventory(f.root))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_stop_only_and_eight_requests_need_no_repeat(self):
        for name, sequences, expected in (("stop", (("stop",), ("stop",)), 2),
                                           ("cap", (("act",)*4, ("act",)*4), 8)):
            f = self.fixture(name, sequences=sequences)
            r = evidence.validate_campaign_service(**f.args)
            self.assertEqual(r["counts"]["accepted_requests"], expected)
            self.assertEqual(r["feedback_path_exercised"], name == "cap")
            self.assertEqual(r["counts"]["budget_stops"], 2 if name == "cap" else 0)
            self.assertEqual(r["counts"]["charged_units"], 12 if name == "cap" else 4)
            self.assertTrue(r["operational_success"])

    def test_separate_policy_identity_for_frozen_sft_dpo(self):
        for pid in POLICIES[-3:]:
            f = self.fixture(pid, policy_id=pid)
            self.assertEqual(evidence.validate_campaign_service(**f.args)["policy_id"], pid)

    def test_retained_transport_failure_and_unstarted_world(self):
        f = self.fixture(sequences=(("act", "fail"), None))
        before = inventory(f.root)
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertEqual(r["status"], "evidence_only_failure_closure")
        self.assertEqual(r["counts"]["accepted_requests"], 1)
        self.assertEqual(r["counts"]["attempted_requests"], 2)
        self.assertEqual(r["counts"]["failed_cells"], 1)
        self.assertEqual(r["counts"]["unstarted_cells"], 1)
        self.assertFalse(r["counts_complete"])
        self.assertEqual(r["counts"]["charged_units"], 3)
        self.assertEqual(before, inventory(f.root))

    def test_reserved_response_failure_is_charged_despite_clean_service(self):
        f = self.fixture(sequences=(("response_fail",), ("stop",)))
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertTrue(r["clean_close"])
        self.assertTrue(r["counts_complete"])
        self.assertEqual(r["counts"]["charged_units"], 5)
        self.assertEqual(r["counts"]["returned_responses"], 4)
        self.assertEqual(r["joins"][0]["outcome"], "response_failure")
        self.assertEqual(r["joins"][0]["charged_units"], 1)

    def test_first_call_failure_preserves_zero_completed_requests(self):
        f = self.fixture(sequences=(("fail",), None))
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertEqual(r["counts"]["accepted_requests"], 0)
        self.assertEqual(r["counts"]["attempted_requests"], 1)
        self.assertEqual(r["counts"]["generations"], 0)
        self.assertEqual(r["counts"]["charged_units"], 2)

    def test_unresolved_episode_retains_rows_without_success_or_complete_costs(self):
        f = self.fixture()
        f.arms[0].update(status="failed", reason="retained sealing failure")
        f.alter("attempt", "world-00/arm-receipt.json",
                lambda v: v.update(status="unresolved", unresolved_reason="retained sealing failure"))
        before = inventory(f.root)
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertFalse(r["counts_complete"])
        self.assertEqual(r["counts"]["accepted_requests"], 3)
        self.assertIsNone(r["cells"][0]["charged_units"])
        self.assertEqual(r["joins"][0]["outcome"], "unresolved_executor_join")
        self.assertEqual(before, inventory(f.root))

    def test_wholly_unstarted_and_launch_failure_have_no_invented_work(self):
        for mode in ("not_started", "launch_failure"):
            f = self.fixture(mode, sequences=(None, None))
            # Move entire fabricated pre-mutation trees outside the attempt;
            # retain them, just as all other mutation controls retain originals.
            f.client.rename(f.root / "original-client")
            f.worker.rename(f.root / "original-worker")
            receipt = evidence._read(f.attempt / "attempt.json")
            if mode == "not_started":
                receipt["service"] = {"status": "not_started", "file": None, "sha256": None}
            else:
                put(f.client / "launch.json", evidence._read(f.root / "original-client/launch.json"))
                close = {"status": "failed", "stage": "launch", "cleanup": None, "completed_requests": 0,
                         "last_completed_request_id": -1, "retry_allowed": False}
                put(f.client / "failure.json", close)
                receipt["service"] = {"status": "recorded", "file": "service/failure.json",
                                      "sha256": paired._sha(paired._raw(close))}
            receipt["evidence_sha256"] = inventory(f.attempt, ("attempt.json", "attempt.candidate.json"))
            put(f.attempt / "attempt.json", receipt)
            put(f.attempt / "attempt.candidate.json", receipt)
            f.args["attempt_sha256"] = paired._sha(paired._raw(receipt))
            r = evidence.validate_campaign_service(**f.args)
            self.assertFalse(r["operational_success"])
            self.assertTrue(r["closure_confirmed"])
            self.assertEqual(r["counts"]["generations"], 0)
            self.assertEqual(r["counts"]["model_loads"], 0)

    def test_unaccepted_pending_generation_is_retained(self):
        f = self.fixture(sequences=(("act", "stop"), None))
        ordinal = 1
        # A complete model generation can precede client acceptance failure.
        accepted = f.client / f"request-{ordinal:04d}.accepted.json"
        accepted.rename(f.root / "original-accepted.json")
        f.worker.joinpath("terminal.json").rename(f.root / "original-terminal.json")
        f.client.joinpath("closed.json").rename(f.root / "original-closed.json")
        f.ordinal, f.failure, f.close_name = 1, True, "failure.json"
        put(f.client / "failure.json", {"status": "failed", "stage": "response_validation", "completed_requests": 1,
            "last_completed_request_id": 0, "cleanup": {"pgid": 123, "group_still_exists": False,
            "leader_returncode": 1, "errors": [], "term_sent": False, "kill_sent": False}})
        # Keep the unresolved executor evidence; no invented clean disposition.
        f.arms[0].update(status="failed", reason="client acceptance interrupted")
        f.alter("attempt", "world-00/arm-receipt.json", lambda v: (v.update(status="unresolved",
            unresolved_reason="client acceptance interrupted"), v["session"].update(count_after=1, request_after=0)))
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertEqual(r["counts"]["generations"], 2)
        self.assertEqual(r["counts"]["accepted_requests"], 1)
        self.assertEqual(r["counts"]["attempted_requests"], 2)

    def test_independent_pins_and_external_inventory(self):
        f = self.fixture()
        for key in ("campaign_sha256", "identity_sha256", "freeze_sha256", "attempt_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_campaign_service(**dict(f.args, **{key: "f"*64}))
        with self.assertRaises(ValueError):
            evidence.validate_campaign_service(**dict(f.args, source_plan_pins=list(reversed(f.source_pins))))
        with self.assertRaises(ValueError):
            evidence.validate_campaign_service(**f.args, worker_evidence_sha256={})
        f.alter("client", "request-0001.episode.json", lambda v: v.update(decision_index=8), reseal=False)
        with self.assertRaisesRegex(ValueError, "custody"):
            evidence.validate_campaign_service(**f.args)

    def test_resealed_cross_world_decision_session_and_history_attacks(self):
        variants = [
            ("client", "request-0002.episode.json", lambda v: v.update(decision_index=1)),
            ("client", "request-0001.episode.json", lambda v: v.update(session_request_id=40)),
            ("client", "request-0002.episode.json", lambda v: v.update(episode_directory=v["episode_directory"].replace("world-01", "world-00"))),
            ("client", "task-binding-0001.json", lambda v: v.update(previous_task_sha256="a"*64)),
            ("client", "task-binding-0001.json", lambda v: v.update(model_reloaded=True)),
            ("client", "request-0001.input.json", lambda v: v["envelope"]["state"]["history"][0]["values"].update(X=1.)),
            ("client", "request-0002.input.json", lambda v: v["envelope"]["task"].update(scale=17.)),
            ("client", "request-0000.accepted.json", lambda v: v.update(completion='{"stop":true}')),
            ("client", "request-0000.observed-response.json", lambda v: v.update(identity_sha256="a"*64)),
            ("worker", "request-0000.json", lambda v: v.update(envelope_sha256="a"*64)),
        ]
        for i, (where, name, mutate) in enumerate(variants):
            with self.subTest(i=i):
                f = self.fixture(str(i))
                f.alter(where, name, mutate)
                with self.assertRaises(ValueError):
                    evidence.validate_campaign_service(**f.args)

    def test_prompt_tokens_binary_identity_and_tensor_guards(self):
        variants = [
            ("generation-0000.json", lambda v: v.update(prompt_sha256="a"*64)),
            ("generation-0000.json", lambda v: v.update(sequence_ids=[99, 151645])),
            ("generation-0000.json", lambda v: v.update(tensor_versions_unchanged=False)),
            ("generation-0000.json", lambda v: v.update(cuda_interval_ms=0)),
            ("generation-0000.json", lambda v: v.update(cross_request_cache_supplied=True)),
            ("raw-generation-0000.json", lambda v: v.update(truncated=True)),
            ("raw-generation-0000.json", lambda v: v.update(dtype="torch.float64")),
            ("authenticated.json", lambda v: v.update(freeze_sha256="e"*64)),
            ("terminal.json", lambda v: v.update(generation_count=4)),
            ("terminal.json", lambda v: v.update(new_training_updates=1)),
        ]
        for i, (name, mutate) in enumerate(variants):
            with self.subTest(i=i):
                f = self.fixture(str(i))
                f.alter("worker", name, mutate)
                with self.assertRaises(ValueError):
                    evidence.validate_campaign_service(**f.args)
        f = self.fixture("binary-rehashed")
        binary = struct.pack("<4q", 90, 91, 92, 151645)
        (f.worker / "raw-generation-0000.bin").write_bytes(binary)
        f.alter("worker", "raw-generation-0000.json", lambda v: v.update(raw_sha256=paired._sha(binary)))
        with self.assertRaisesRegex(ValueError, "binary/token"):
            evidence.validate_campaign_service(**f.args)

    def test_exact_generation_membership_model_load_and_symlinks(self):
        for name in ("generation-0007.json", "raw-generation-0007.bin", "generated-0007.unaccepted.json"):
            f = self.fixture(name)
            (f.worker / name).write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                evidence.validate_campaign_service(**f.args)
        f = self.fixture("reload")
        put(f.worker / "phase-099.json", {"phase": "model_cpu_load_start"})
        with self.assertRaisesRegex(ValueError, "reloaded"):
            evidence.validate_campaign_service(**f.args)
        f = self.fixture("link")
        (f.worker / "linked").symlink_to(f.client, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            evidence.validate_campaign_service(**f.args)

    def test_cleanup_failure_never_becomes_success(self):
        f = self.fixture("forced")
        f.alter("client", "closed.json", lambda v: v["cleanup"].update(term_sent=True))
        r = evidence.validate_campaign_service(**f.args)
        self.assertFalse(r["operational_success"])
        self.assertFalse(r["clean_close"])
        f = self.fixture("leak")
        f.alter("client", "closed.json", lambda v: v["cleanup"].update(group_still_exists=True))
        with self.assertRaisesRegex(ValueError, "closure unconfirmed"):
            evidence.validate_campaign_service(**f.args)

    def test_original_sealed_return_and_executor_join(self):
        f = self.fixture("return")
        f.alter("attempt", "world-00/return-002.json", lambda v: v["response"]["raw_json"].update(X=7.))
        with self.assertRaisesRegex(ValueError, "return differs"):
            evidence.validate_campaign_service(**f.args)
        f = self.fixture("proposal")
        directory = f.attempt / "world-00"
        events = [campaign._decode(line) for line in (directory / "episode/journal.jsonl").read_bytes().splitlines()]
        for row in events:
            for key in ("sha256", "previous_sha256", "monotonic_ns"):
                row.pop(key)
            if row["event"] == "proposal" and row["provenance"]["kind"] == "process_policy":
                row["provenance"]["public_state_sha256"] = "a"*64
        raw = journal(events)
        (directory / "episode/journal.jsonl").write_bytes(raw)
        (directory / "sealed-inputs/executor.jsonl").write_bytes(raw)
        f.reseal()
        with self.assertRaisesRegex(ValueError, "proposal join"):
            evidence.validate_campaign_service(**f.args)


if __name__ == "__main__":
    start, cpu = time.monotonic(), time.process_time()
    result = unittest.main(exit=False)
    print(canonical({"test_elapsed_seconds": time.monotonic()-start, "test_process_cpu_seconds": time.process_time()-cpu,
                     "process_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "tests_run": result.result.testsRun, "successful": result.result.wasSuccessful(),
                     "scope": "new fabricated saved-file tests only; imports excluded from interval",
                     "new_environment_responses": 0, "model_loads": 0, "training_updates": 0, "gpu_allocations": 0}), flush=True)
    sys.exit(not result.result.wasSuccessful())
