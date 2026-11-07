"""Only handcrafted journal/capture fixtures; no executor or reporter runs."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import llm_intervention_episode_seal as seal
import llm_intervention_world as world_module


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encoded(obj):
    return (canonical(obj)+"\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def journal(events):
    previous, lines = None, []
    for i, event in enumerate(events):
        record = dict(event, previous_sha256=previous, monotonic_ns=100+i)
        previous = sha(canonical(record).encode())
        record["sha256"] = previous
        lines.append(encoded(record))
    return b"".join(lines)


def events_of(data):
    return [{k: v for k, v in json.loads(line).items() if k not in ("sha256", "previous_sha256", "monotonic_ns")}
            for line in data["inputs"]["executor.jsonl"].splitlines()]


def repin(data, name, raw):
    data["inputs"][name] = raw
    data["expected_sha256"][name] = sha(raw)


def change_disposition(data, mutate):
    d = json.loads(data["inputs"]["episode-disposition.json"])
    mutate(d)
    repin(data, "episode-disposition.json", encoded(d))
    outer = json.loads(data["inputs"]["outer-outcome.json"])
    outer["episode_disposition_sha256"] = data["expected_sha256"]["episode-disposition.json"]
    repin(data, "outer-outcome.json", encoded(outer))


def oracle_summaries(graph, learner, receipts):
    """Exact rational 1x1/2x2 inverses for the handcrafted fixtures only."""
    summaries = []
    rational = lambda x: Fraction(str(x))
    for node in graph:
        d = 1+len(node["parents"])
        assert d in (1, 2)
        rows = []
        for r in receipts:
            values = dict(r["values"], **r["targets"])
            if node["name"] not in r["targets"] and node["name"] in r["values"] and set(node["parents"]) <= set(values):
                rows.append(([Fraction(1)] + [rational(values[p]) for p in node["parents"]], rational(r["values"][node["name"]])))
        noise, prior = rational(learner["noise_variances"][node["name"]]), rational(learner["prior_variance"])
        precision = [[Fraction(int(i == j))/prior + sum((x[i]*x[j]/noise for x, _ in rows), Fraction(0)) for j in range(d)] for i in range(d)]
        rhs = [sum((x[i]*y/noise for x, y in rows), Fraction(0)) for i in range(d)]
        if d == 1:
            covariance = [[1/precision[0][0]]]
        else:
            a, b, c = precision[0][0], precision[0][1], precision[1][1]
            det = a*c-b*b
            covariance = [[c/det, -b/det], [-b/det, a/det]]
        mean = [sum(covariance[i][j]*rhs[j] for j in range(d)) for i in range(d)]
        summaries.append({"node": node["name"], "terms": ["intercept", *node["parents"]],
            "parameter_mean": list(map(float, mean)), "parameter_covariance": [list(map(float, row)) for row in covariance],
            "noise_variance": float(noise), "eligible_rows": len(rows)})
    return summaries


def fixture(mode="success", initial_count=1):
    """Stipulated two-node receipts with exact rational terminal summaries."""
    graph = [{"name": "X", "parents": [], "unit": "u", "intervention_bounds": [-1., 1.]},
             {"name": "Y", "parents": ["X"], "unit": "u", "intervention_bounds": None}]
    observe = {"action_id": "O", "kind": "observe", "targets": [], "measure": ["X", "Y"], "cost_units": 1}
    intervene = {"action_id": "D", "kind": "intervene", "targets": [["X", 1.]], "measure": ["X", "Y"], "cost_units": 2}
    menu = [observe, intervene]
    world = {"version": "ace-linear-gaussian-world-v1", "scope": "artificial", "world_id": 7,
        "graph": graph, "mechanisms": [{"node": "X", "intercept": 0., "coefficients": [], "noise_variance": 1.},
                                       {"node": "Y", "intercept": 0., "coefficients": [["X", 1.]], "noise_variance": 1.}]}
    task = {"target": "Y", "queries": [observe, intervene], "weights": [.5, .5], "scale": 2., "normalized_mse_tolerance": .25}
    learner = {"graph": graph, "noise_variances": {"X": 1., "Y": 1.}, "prior_variance": 1.}
    cap = 2 if mode == "budget" else 3
    plan = {"graph": graph, "menu": menu, "budget": initial_count+cap, "initial_actions": ["O"]*initial_count,
        "objective": "Artificial public objective", "domain_notes": "No private scores.",
        "policy_command": ["artificial-public-policy"], "policy_timeout_seconds": 1.}
    events = [{"event": "begin", "version": "ace-intervention-contract-v1", "budget_units": plan["budget"], "graph": graph, "menu": menu}]
    receipts, captures, calls, spent, proposals, initialized = [], {}, 0, 0, 0, 0
    failed, failed_message = False, "stipulated failure"

    def receipt(action, request):
        return {"request_id": request, "action_id": action["action_id"], "kind": action["kind"],
            "targets": dict(action["targets"]), "values": {"X": 1. if action["kind"] == "intervene" else .25, "Y": 1.5},
            "cost_units": action["cost_units"]}

    def proposal(action=None, kind="process_policy", completion=None):
        nonlocal proposals, calls
        if completion is None:
            completion = canonical({"stop": True} if action is None else {"action_id": action["action_id"]})
        provenance = {"kind": kind}
        if kind == "process_policy":
            provenance.update(decision_index=calls, public_state_sha256="c"*64)
            calls += 1
        events.append({"event": "proposal", "proposal_id": proposals, "completion": completion, "provenance": provenance})
        proposals += 1
        return completion

    def attempt(action, kind, outcome="accepted"):
        nonlocal spent, failed
        completion = proposal(action, kind)
        request = len(receipts)
        spent += action["cost_units"]
        events.append({"event": "reserve", "proposal_id": proposals-1, "request_id": request, "action": action,
                       "completion": completion, "cost_units": action["cost_units"]})
        r = receipt(action, request)
        if outcome != "unknown":
            response = {"X": 1.} if outcome == "invalid" else r["values"]
            raw = {"python_type": "builtins.dict", "raw_response_repr": repr(response), "raw_json_available": True, "raw_json": response}
            if outcome == "nonjson":
                raw = {"python_type": "builtins.object", "raw_response_repr": "<artificial object>", "raw_json_available": False}
            captures[f"return-{request:03d}.json"] = {"request_id": request, "action": action, "returned_to_wrapper": True, "response": raw}
        if outcome == "accepted":
            events.append({"event": "response", **r})
            receipts.append(r)
        else:
            events.append({"event": "failure", "request_id": request,
                "response_returned_to_driver": None if outcome in ("unknown", "unaccepted") else True,
                "exception_type": "RuntimeError", "message": failed_message, "spent_units": spent})
            failed = True

    shared = tuple(sha(encoded(receipt(observe, i))) for i in range(initial_count))
    for i in range(initial_count):
        outcome = "unknown" if mode == "partial_unknown" or (mode == "partial_accepted" and i == 1) else "accepted"
        attempt(observe, "shared_initial_schedule", outcome)
        if failed:
            break
        if mode == "learner_failure":
            events.append({"event": "driver_failure", "exception_type": "RuntimeError", "message": failed_message, "spent_units": spent})
            failed = True
            break
        initialized += 1
    if not failed:
        if mode == "reject":
            proposal(completion="{invalid")
            events.append({"event": "rejected", "proposal_id": proposals-1, "stage": "completion_parse",
                "exception_type": "RuntimeError", "reason": failed_message, "spent_units": spent, "retry_allowed": False})
            failed = True
        elif mode == "driver_failure":
            events.append({"event": "driver_failure", "exception_type": "RuntimeError", "message": failed_message, "spent_units": spent})
            failed = True
        else:
            attempt(intervene, "process_policy", mode if mode in ("unknown", "invalid", "unaccepted", "nonjson") else "accepted")
    if not failed:
        completion = proposal(kind="budget_cap" if mode == "budget" else "process_policy")
        events.append({"event": "stop", "proposal_id": proposals-1, "completion": completion, "spent_units": spent})
    applied = receipts[:-1] if mode == "learner_failure" else receipts
    summaries = oracle_summaries(graph, learner, applied)
    requests = sum(e["event"] == "reserve" for e in events)
    state = {"version": "ace-intervention-contract-v1", "objective": plan["objective"], "domain_notes": plan["domain_notes"],
        "graph": graph, "menu": menu, "history": receipts, "scm": summaries, "spent_units": spent,
        "remaining_units": plan["budget"]-spent, "requests": requests, "returned_responses": len(receipts), "terminal": True}
    disposition = ({"status": "failed", "exception_type": "RuntimeError", "message": failed_message,
        "initialized_responses": initialized, "completed_policy_calls": calls, "partial_state": state, "retry_allowed": False} if failed else
        {"status": "complete", "initialized_responses": initialized, "policy_decisions": calls, "final_state": state, "scientific_evaluation_performed": False})
    inputs = {"world.json": encoded(world), "task.json": encoded(task), "learner.json": encoded(learner),
        "episode-plan.json": encoded(plan), "executor.jsonl": journal(events), "episode-disposition.json": encoded(disposition),
        **{k: encoded(v) for k, v in captures.items()}}
    outer = {"version": "ace-paired-episode-v1", "scope": "artificial", "world_id": 7, "policy_id": "artificial-policy",
        "status": "failed" if failed or mode == "close_failure" else "complete",
        "reason": "episode_failure:RuntimeError" if failed else "adapter_close_failure:RuntimeError" if mode == "close_failure" else
                  "budget_exhausted" if mode == "budget" else "policy_stop",
        "episode_returned_cleanly": not failed, "adapters_closed_cleanly": mode != "close_failure",
        "episode_disposition_sha256": sha(inputs["episode-disposition.json"]), "outer_provenance_receipt_sha256": "a"*64}
    inputs["outer-outcome.json"] = encoded(outer)
    return {"inputs": inputs, "expected_sha256": {k: sha(v) for k, v in inputs.items()},
        "shared_initial_receipt_sha256": shared, "scope": "artificial", "world_id": 7,
        "policy_id": "artificial-policy", "acquisition_cap_units": cap}


class SealTests(unittest.TestCase):
    def retain(self, name, data, result=None, error=None):
        root = os.environ.get("ACE_EPISODE_SEAL_TEST_OUTPUT")
        if not root:
            return
        path = Path(root)/self._testMethodName/name
        path.mkdir(parents=True, exist_ok=False)
        (path/"input").mkdir()
        for k, v in data["inputs"].items():
            (path/"input"/k).write_bytes(v)
        (path/"arguments.json").write_bytes(encoded({k: v for k, v in data.items() if k != "inputs"}))
        if result is not None:
            (path/"bundle").mkdir()
            for k, v in result["bundle"].items():
                (path/"bundle"/k).write_bytes(v)
            (path/"result.json").write_bytes(encoded({k: v for k, v in result.items() if k != "bundle"}))
        if error is not None:
            (path/"rejection.json").write_bytes(encoded({"type": type(error).__name__, "message": str(error)}))

    def call(self, data, label):
        try:
            result = seal.seal_episode(**data)
        except Exception as exc:
            self.retain(label, data, error=exc)
            raise
        self.retain(label, data, result)
        return result

    def rejected(self, data, label):
        with self.assertRaises(seal.SealError) as caught:
            self.call(data, label)
        return caught.exception

    def test_success_schema_exact_receipt_pins_encoding_and_empty_initial(self):
        for mode, count in (("success", 1), ("budget", 1), ("success", 0), ("close_failure", 1)):
            data = fixture(mode, count)
            before = deepcopy(data)
            result = self.call(data, f"{mode}-initial-{count}")
            self.assertEqual(data, before)
            bundle = result["bundle"]
            self.assertEqual(set(bundle), {"world.json", "task.json", "learner.json", "plan.json", "shared-initial.json", "history.json", "terminal.json"}
                             | {f"receipt-{i:03d}.json" for i in range(count+1)})
            self.assertTrue(all(raw == encoded(json.loads(raw)) for raw in bundle.values()))
            self.assertEqual(result["bundle_sha256"], {k: sha(v) for k, v in bundle.items()})
            terminal = json.loads(bundle["terminal.json"])
            self.assertEqual(terminal["input_sha256"], {k: v for k, v in result["bundle_sha256"].items() if k != "terminal.json"})
            self.assertEqual((terminal["attempt_count"], terminal["accepted_count"], terminal["known_return_count"]), (count+1,)*3)
            self.assertEqual((terminal["initial_cost_units"], terminal["acquisition_cost_units"]), (count, 2))
            self.assertEqual(terminal["status"], "failed" if mode == "close_failure" else "complete")
            self.assertEqual(json.loads(bundle["shared-initial.json"])["receipt_sha256"], list(data["shared_initial_receipt_sha256"]))
            for i in range(count):
                self.assertEqual(sha(bundle[f"receipt-{i:03d}.json"]), data["shared_initial_receipt_sha256"][i])
            plan = json.loads(bundle["plan.json"])
            self.assertEqual(plan["version"], "ace-offline-cost-curve-v1")
            self.assertEqual(plan["acquisition_cap_units"], data["acquisition_cap_units"])
            self.assertEqual(plan["threshold"], .25)
            self.assertEqual(result["provenance"]["input_sha256"], data["expected_sha256"])
            self.assertFalse(result["provenance"]["scientific_acceptance"])
            self.assertFalse(result["provenance"]["private_scoring_performed"])

    def test_failures_remain_charged_with_exact_return_classification(self):
        cases = (("unknown", 1, "unknown", 1, 2), ("invalid", 1, "returned_invalid", 2, 2),
                 ("unaccepted", 1, "returned_unaccepted", 2, 2), ("partial_unknown", 2, "unknown", 0, 0),
                 ("partial_accepted", 2, "unknown", 1, 0))
        for mode, count, expected, known, acq in cases:
            result = self.call(fixture(mode, count), mode)
            history = json.loads(result["bundle"]["history.json"])["attempts"]
            terminal = json.loads(result["bundle"]["terminal.json"])
            self.assertEqual(history[-1]["outcome"], expected)
            self.assertEqual(terminal["status"], "failed")
            self.assertEqual(terminal["known_return_count"], known)
            self.assertEqual(terminal["acquisition_cost_units"], acq)
            self.assertEqual(terminal["initial_cost_units"], 2 if mode == "partial_accepted" else 1)
            if expected == "unknown":
                self.assertIsNone(history[-1]["evidence_file"])
            else:
                self.assertEqual(result["bundle"][history[-1]["evidence_file"]], encoded({"X": 1.} if mode == "invalid" else {"X": 1., "Y": 1.5}))
        for mode in ("reject", "driver_failure", "learner_failure"):
            result = self.call(fixture(mode, 2 if mode == "learner_failure" else 1), mode)
            terminal = json.loads(result["bundle"]["terminal.json"])
            self.assertEqual((terminal["status"], terminal["attempt_count"], terminal["accepted_count"]), ("failed", 1, 1))
        data = fixture("unknown")
        change_disposition(data, lambda d: d.update(partial_state={"state_unavailable": "RuntimeError: state unavailable", "accounting_source": "retained journal.jsonl"}))
        self.assertEqual(json.loads(self.call(data, "state-unavailable")["bundle"]["terminal.json"])["status"], "failed")

    def test_pin_inventory_preflight_and_scope_before_any_world_work(self):
        data = fixture()
        data["inputs"]["world.json"] = b"not JSON"
        with patch.object(seal, "_json", side_effect=AssertionError("decode before pin validation")):
            self.rejected(data, "pin-before-decode")
        for label, mutate in (("missing", lambda d: d["inputs"].pop("executor.jsonl")),
            ("extra", lambda d: repin(d, "adapter-outcome.json", b"{}\n")),
            ("oversize", lambda d: repin(d, "executor.jsonl", b" "*(seal.MAX_BYTES+1))),
            ("policy-train", lambda d: d.update(scope="policy_train")),
            ("confirmation", lambda d: d.update(scope="confirmation")),
            ("wrong-identity", lambda d: d.update(world_id=8)),
            ("missing-full-initial", lambda d: d.update(shared_initial_receipt_sha256=())),
            ("shared-pin", lambda d: d.update(shared_initial_receipt_sha256=("e"*64,)))):
            data = fixture(); mutate(data); self.rejected(data, label)
        data = fixture()
        for name in ("_normal", "generate_world", "private_structural_means", "_evaluate"):
            with patch.object(world_module, name, side_effect=AssertionError("world/RNG/scoring path called")):
                self.call(data, "no-"+name)

    def test_journal_proposal_order_hash_clamp_and_charge_rejections(self):
        base = fixture()
        mutations = [("proposal-id", lambda e: e[1].update(proposal_id=True)),
            ("initial-proposal", lambda e: e[1].update(completion='{"action_id":"D"}')),
            ("reserve-id", lambda e: e[2].update(request_id=True)),
            ("reserve-charge", lambda e: e[2].update(cost_units=2)),
            ("response-id", lambda e: e[3].update(request_id=1)),
            ("reserve-action", lambda e: e[5]["action"].update(cost_units=1)),
            ("clamp", lambda e: e[6]["values"].update(X=0.)),
            ("response-charge", lambda e: e[6].update(cost_units=1)),
            ("proposal-reserve-mismatch", lambda e: e[5].update(completion='{"action_id":"O"}')),
            ("order", lambda e: e.__setitem__(slice(2, 4), [e[3], e[2]])),
            ("extra-after-stop", lambda e: e.append(deepcopy(e[1]))),
            ("missing-terminal", lambda e: e.pop()),
            ("decision-index", lambda e: e[4]["provenance"].update(decision_index=4)),
            ("wrong-budget", lambda e: e[0].update(budget_units=3))]
        for label, mutate in mutations:
            data = deepcopy(base); events = events_of(data); mutate(events); repin(data, "executor.jsonl", journal(events))
            self.rejected(data, label)
        data = deepcopy(base)
        lines = data["inputs"]["executor.jsonl"].splitlines(); obj = json.loads(lines[2]); obj["sha256"] = "0"*64
        lines[2] = encoded(obj).rstrip(b"\n"); repin(data, "executor.jsonl", b"\n".join(lines)+b"\n")
        self.rejected(data, "hash-chain")
        data = fixture("unknown"); events = events_of(data); events.append(deepcopy(events[1])); repin(data, "executor.jsonl", journal(events))
        self.rejected(data, "continuation-after-failure")
        data = fixture("unknown"); events = events_of(data); events.pop(); repin(data, "executor.jsonl", journal(events))
        self.rejected(data, "unsealed-reservation")

    def test_terminal_and_source_consistency_fail_closed(self):
        for label, mutate in (("initial-counter", lambda d: d.update(initialized_responses=0)),
            ("decision-counter", lambda d: d.update(policy_decisions=1)),
            ("charge-counter", lambda d: d["final_state"].update(spent_units=2)),
            ("history", lambda d: d["final_state"]["history"].pop()),
            ("eligible-counter", lambda d: d["final_state"]["scm"][0].update(eligible_rows=0)),
            ("nonterminal-state", lambda d: d["final_state"].update(terminal=False))):
            data = fixture(); change_disposition(data, mutate); self.rejected(data, label)
        for label, mutate in (("false-close-success", lambda d: d.update(adapters_closed_cleanly=False)),
            ("false-episode-failure", lambda d: d.update(status="failed", reason="episode_failure:RuntimeError")),
            ("wrong-stop-reason", lambda d: d.update(reason="budget_exhausted")),
            ("original-pin", lambda d: d.update(episode_disposition_sha256="e"*64))):
            data = fixture(); outer = json.loads(data["inputs"]["outer-outcome.json"]); mutate(outer); repin(data, "outer-outcome.json", encoded(outer)); self.rejected(data, label)
        for label, name, mutate in (("budget", "episode-plan.json", lambda d: d.update(budget=8)),
            ("task-target", "task.json", lambda d: d.update(target="missing")),
            ("task-weights", "task.json", lambda d: d.update(weights=[.75, .25])),
            ("learner-noise", "learner.json", lambda d: d["noise_variances"].update(X=0.)),
            ("world-graph", "world.json", lambda d: d["graph"][1].update(parents=[]))):
            data = fixture(); obj = json.loads(data["inputs"][name]); mutate(obj); repin(data, name, encoded(obj)); self.rejected(data, label)

    def test_wrapper_evidence_no_adapter_outcome_inference_or_fabrication(self):
        self.rejected(fixture("nonjson"), "non-json")
        data = fixture("invalid")
        del data["inputs"]["return-001.json"]; del data["expected_sha256"]["return-001.json"]
        self.rejected(data, "known-driver-return-without-wrapper")
        for label, mutate in (("false-return", lambda r: r.update(returned_to_wrapper=False)),
            ("request", lambda r: r.update(request_id=True)),
            ("action", lambda r: r["action"].update(action_id="O")),
            ("contradict-accepted", lambda r: r["response"]["raw_json"].update(Y=4.))):
            data = fixture(); wrapper = json.loads(data["inputs"]["return-001.json"]); mutate(wrapper); repin(data, "return-001.json", encoded(wrapper)); self.rejected(data, label)
        data = fixture("unknown")
        # A prepared adapter record is not a callback wrapper and cannot establish return.
        repin(data, "return-001.json", encoded({"event": "outcome", "request_id": 1, "values": {"X": 1., "Y": 1.5}, "returned_subset_ready": True}))
        self.rejected(data, "prepared-outcome-is-not-return")
        data = fixture(); repin(data, "return-002.json", data["inputs"]["return-001.json"])
        self.rejected(data, "unreserved-return")


def atomic_acquisition_fixture(applied_count):
    data = fixture()
    events = events_of(data)[:-2]  # retained acquisition response, no subsequent stop proposal
    events.append({"event": "driver_failure", "exception_type": "RuntimeError", "message": "atomic update control", "spent_units": 3})
    repin(data, "executor.jsonl", journal(events))
    old = json.loads(data["inputs"]["episode-disposition.json"])
    state = old["final_state"]
    learner = json.loads(data["inputs"]["learner.json"])
    state["scm"] = oracle_summaries(state["graph"], learner, state["history"][:applied_count])
    d = {"status": "failed", "exception_type": "RuntimeError", "message": "atomic update control",
         "initialized_responses": 1, "completed_policy_calls": 1, "partial_state": state, "retry_allowed": False}
    repin(data, "episode-disposition.json", encoded(d))
    outer = json.loads(data["inputs"]["outer-outcome.json"])
    outer.update(status="failed", reason="episode_failure:RuntimeError", episode_returned_cleanly=False,
                 episode_disposition_sha256=data["expected_sha256"]["episode-disposition.json"])
    repin(data, "outer-outcome.json", encoded(outer))
    return data


class PosteriorCorrectionTests(unittest.TestCase):
    # Reuse artifact retention, never inherit or execute the six original tests.
    retain, call, rejected = SealTests.retain, SealTests.call, SealTests.rejected

    def test_positive_hand_oracle_and_pinned_prior_noise(self):
        data = fixture()
        state = json.loads(data["inputs"]["episode-disposition.json"])["final_state"]
        x, y = state["scm"]
        self.assertEqual((x["parameter_mean"], x["parameter_covariance"], x["eligible_rows"]), ([1/8], [[1/2]], 1))
        self.assertEqual(y["parameter_mean"], [123/148, 15/37])
        self.assertEqual(y["parameter_covariance"], [[33/74, -10/37], [-10/37, 24/37]])
        result = self.call(data, "rational-hand-oracle")
        check = result["provenance"]["posterior_reconciliation"]
        self.assertEqual(check["matching_accepted_prefix_lengths"], [2])
        self.assertEqual((check["atol"], check["rtol"]), (1e-10, 1e-8))
        self.assertTrue(result["provenance"]["posterior_replay_performed"])
        self.assertFalse(result["provenance"]["private_scoring_performed"])
        learner = json.loads(data["inputs"]["learner.json"])
        learner.update(prior_variance=2., noise_variances={"X": 3., "Y": 5.})
        repin(data, "learner.json", encoded(learner))
        expected = oracle_summaries(state["graph"], learner, state["history"])
        self.assertEqual(expected[0]["parameter_mean"], [1/10])
        self.assertEqual(expected[0]["parameter_covariance"], [[6/5]])
        self.assertEqual(expected[1]["parameter_mean"], [267/463, 150/463])
        self.assertEqual(expected[1]["parameter_covariance"], [[570/463, -200/463], [-200/463, 720/463]])
        change_disposition(data, lambda d: d["final_state"].update(scm=expected))
        self.call(data, "heterogeneous-noise-prior")
        learner["prior_variance"] = 3.
        repin(data, "learner.json", encoded(learner))
        self.rejected(data, "changed-pinned-prior-only")
        self.call(fixture(initial_count=0), "clamped-root-retains-prior")

    def test_changed_mean_entries_are_rejected(self):
        for node, width in ((0, 1), (1, 2)):
            for entry in range(width):
                data = fixture()
                def mutate(d):
                    d["final_state"]["scm"][node]["parameter_mean"][entry] += .01
                change_disposition(data, mutate)
                exc = self.rejected(data, f"mean-{node}-{entry}")
                self.assertIn("posterior", str(exc))

    def test_changed_covariance_entries_are_rejected(self):
        for node, width in ((0, 1), (1, 2)):
            for i in range(width):
                for j in range(width):
                    data = fixture()
                    def mutate(d):
                        d["final_state"]["scm"][node]["parameter_covariance"][i][j] += .01
                    change_disposition(data, mutate)
                    exc = self.rejected(data, f"covariance-{node}-{i}-{j}")
                    self.assertIn("posterior", str(exc))

    def test_atomic_prefixes_and_explicit_unavailable_failed_state(self):
        for applied in (1, 2):
            result = self.call(atomic_acquisition_fixture(applied), f"atomic-prefix-{applied}")
            self.assertEqual(json.loads(result["bundle"]["terminal.json"])["status"], "failed")
            check = result["provenance"]["posterior_reconciliation"]
            self.assertEqual(check["permitted_accepted_prefix_lengths"], [2, 1])
            self.assertEqual(check["matching_accepted_prefix_lengths"], [applied])
        data = atomic_acquisition_fixture(2)
        withheld = json.loads(atomic_acquisition_fixture(1)["inputs"]["episode-disposition.json"])["partial_state"]["scm"][1]["parameter_covariance"]
        change_disposition(data, lambda d: d["partial_state"]["scm"][1].update(parameter_covariance=withheld))
        self.rejected(data, "mixed-prefix-mean-covariance")
        data = atomic_acquisition_fixture(2)
        change_disposition(data, lambda d: d["partial_state"]["scm"][1].update(eligible_rows=0))
        self.rejected(data, "eligible-count-mismatch")
        result = self.call(fixture("learner_failure", 2), "initial-accepted-but-unapplied")
        self.assertEqual(result["provenance"]["posterior_reconciliation"]["matching_accepted_prefix_lengths"], [0])
        data = fixture("unknown")
        change_disposition(data, lambda d: d.update(partial_state={"state_unavailable": "RuntimeError: unavailable control", "accounting_source": "retained journal.jsonl"}))
        result = self.call(data, "explicit-unavailable-failure")
        self.assertEqual(json.loads(result["bundle"]["terminal.json"])["status"], "failed")
        self.assertFalse(result["provenance"]["posterior_replay_performed"])
        self.assertEqual(result["provenance"]["posterior_reconciliation"]["status"], "unavailable_failed_state")


if __name__ == "__main__":
    unittest.main()
