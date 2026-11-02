"""Isolated public-state and charged-action contract for the renewed ACE study.

This module knows no true mechanisms, evaluation scores or language models.
The runner must keep those objects out of the policy process. These API checks
do not sandbox malicious Python running in the executor's own process.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence


VERSION = "ace-intervention-contract-v1"
SCOPES = ("artificial", "policy_train", "development", "confirmation")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def stream_seed(scope: str, world: int, stream: str) -> int:
    """Stable named streams, independent of method order and evaluation calls."""
    if scope not in SCOPES or not integer(world) or stream not in (
        "mechanism", "initial", "measurement", "policy", "train_score", "private_eval"
    ):
        raise ValueError("invalid world or stream namespace")
    if stream == "train_score" and scope != "policy_train":
        raise ValueError("training feedback stream is unavailable in this scope")
    raw = canonical([VERSION, scope, world, stream]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...]
    unit: str
    intervention_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not isinstance(self.unit, str):
            raise ValueError("node name and unit required")
        if type(self.parents) is not tuple or len(set(self.parents)) != len(self.parents):
            raise ValueError("parents must be an immutable unique tuple")
        if self.intervention_bounds is not None:
            b = self.intervention_bounds
            if type(b) is not tuple or len(b) != 2 or not all(map(finite, b)) or b[0] > b[1]:
                raise ValueError("invalid intervention bounds")


@dataclass(frozen=True)
class Action:
    action_id: str
    kind: str
    targets: tuple[tuple[str, float], ...]
    measure: tuple[str, ...]
    cost_units: int

    def __post_init__(self):
        if not isinstance(self.action_id, str) or not self.action_id or self.kind not in ("intervene", "observe"):
            raise ValueError("invalid action identity or kind")
        if not integer(self.cost_units, 1):
            raise ValueError("each environment response has a positive integer cost")
        if type(self.targets) is not tuple or type(self.measure) is not tuple or not self.measure:
            raise ValueError("targets and measurements must be immutable tuples")
        if any(type(x) is not tuple or len(x) != 2 or not isinstance(x[0], str)
               or not finite(x[1]) for x in self.targets):
            raise ValueError("invalid target")
        if len({x[0] for x in self.targets}) != len(self.targets) or len(set(self.measure)) != len(self.measure):
            raise ValueError("duplicate target or measurement")
        if (self.kind == "intervene") != bool(self.targets):
            raise ValueError("interventions require targets and observations forbid them")


def validate_design(graph: tuple[Node, ...], menu: tuple[Action, ...]) -> None:
    if type(graph) is not tuple or not graph or type(menu) is not tuple or not menu:
        raise ValueError("nonempty immutable graph and menu required")
    seen = set()
    for node in graph:
        if node.name in seen or not set(node.parents) <= seen:
            raise ValueError("graph must have unique names in topological order")
        seen.add(node.name)
    if len({a.action_id for a in menu}) != len(menu):
        raise ValueError("duplicate menu action identity")
    nodes = {n.name: n for n in graph}
    for action in menu:
        if not set(action.measure) <= seen:
            raise ValueError("unknown measured variable")
        for name, value in action.targets:
            bounds = nodes[name].intervention_bounds if name in nodes else None
            if bounds is None or not bounds[0] <= value <= bounds[1]:
                raise ValueError("target is not manipulable at this value")


@dataclass(frozen=True)
class Decision:
    action_id: str | None
    stop: bool


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_completion(completion: str, menu: tuple[Action, ...]) -> Decision:
    """Accept exactly one generated JSON action, with no extraction or fallback."""
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    if not isinstance(completion, str) or len(completion) > 4096:
        raise ValueError("invalid completion size/type")
    obj = json.loads(completion, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    if type(obj) is not dict:
        raise ValueError("completion must be one object")
    if set(obj) == {"stop"} and obj["stop"] is True:
        return Decision(None, True)
    if set(obj) == {"action_id"} and type(obj["action_id"]) is str:
        if obj["action_id"] in {a.action_id for a in menu}:
            return Decision(obj["action_id"], False)
    raise ValueError("completion is outside the legal action grammar")


def parse_generated_sequence(prompt_ids: Sequence[int], sequence_ids: Sequence[int],
                             decode: Callable, menu: tuple[Action, ...]):
    """Decoder-only generation must return the exact input prefix plus new tokens."""
    prompt, sequence = list(prompt_ids), list(sequence_ids)
    if not prompt or sequence[:len(prompt)] != prompt or len(sequence) <= len(prompt):
        raise ValueError("missing or mismatched generation prefix/completion")
    completion = decode(sequence[len(prompt):])
    return parse_completion(completion, menu), completion


@dataclass(frozen=True)
class MechanismSummary:
    node: str
    terms: tuple[str, ...]
    parameter_mean: tuple[float, ...]
    parameter_covariance: tuple[tuple[float, ...], ...]
    noise_variance: float
    eligible_rows: int

    def __post_init__(self):
        k = len(self.terms)
        if (type(self.terms) is not tuple or type(self.parameter_mean) is not tuple
                or type(self.parameter_covariance) is not tuple or not k
                or any(not isinstance(x, str) for x in self.terms)
                or len(self.parameter_mean) != k or len(self.parameter_covariance) != k
                or not all(map(finite, self.parameter_mean))
                or any(type(row) is not tuple or len(row) != k or not all(map(finite, row))
                       for row in self.parameter_covariance)
                or not finite(self.noise_variance) or self.noise_variance < 0
                or not integer(self.eligible_rows)):
            raise ValueError("invalid fitted mechanism summary")
        # Positive semidefiniteness is the learner's qualification obligation.


class ChargedExecutor:
    """One attempt, one exclusive journal, reservation before every response.

    The callback is the trusted environment driver. Policies receive state()
    output only, never this object or its callback. Failed/ambiguous responses
    halt the attempt. There is deliberately no resume or automatic retry API.
    """

    def __init__(self, graph: tuple[Node, ...], menu: tuple[Action, ...], budget_units: int,
                 journal: Path, respond: Callable[[Action, int], Mapping[str, float]],
                 joint_feasible: Callable[[Action], bool] | None = None):
        validate_design(graph, menu)
        if not integer(budget_units, 1):
            raise ValueError("positive integer budget required")
        self.graph, self.menu, self.budget_units = graph, menu, budget_units
        self._respond, self._joint_feasible = respond, joint_feasible
        self._history, self._spent, self._requests, self._proposals = [], 0, 0, 0
        self._halted, self._previous = False, None
        self._file = Path(journal).open("x", encoding="utf-8")
        self._append({"event": "begin", "version": VERSION, "budget_units": budget_units,
                      "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu]})

    def _append(self, event):
        record = dict(event, previous_sha256=self._previous, monotonic_ns=time.monotonic_ns())
        record["sha256"] = digest(record)
        try:
            self._file.write(canonical(record) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._halted = True
            raise
        self._previous = record["sha256"]

    def _proposal(self, completion, provenance):
        proposal_id = self._proposals
        self._append({"event": "proposal", "proposal_id": proposal_id,
                      "completion": completion, "provenance": provenance})
        self._proposals += 1
        return proposal_id

    def _reject(self, proposal_id, stage, exc):
        self._halted = True
        self._append({"event": "rejected", "proposal_id": proposal_id, "stage": stage,
                      "exception_type": type(exc).__name__, "reason": str(exc),
                      "spent_units": self._spent, "retry_allowed": False})

    def execute_generated(self, prompt_ids, sequence_ids, decode):
        if self._halted:
            raise RuntimeError("attempt is terminal or ambiguous")
        proposal_id = self._proposal(None, {"kind": "decoder_only_sequence"})
        stage = "token_validation"
        try:
            for tokens in (prompt_ids, sequence_ids):
                if (type(tokens) not in (list, tuple) or not tokens or len(tokens) > 65536
                        or any(type(x) is not int or x < 0 for x in tokens)):
                    raise ValueError("bounded integer token lists required")
            prompt, sequence = list(prompt_ids), list(sequence_ids)
            self._append({"event": "generated_tokens", "proposal_id": proposal_id,
                          "prompt_ids": prompt, "sequence_ids": sequence})
            stage = "generation_prefix"
            if sequence[:len(prompt)] != prompt or len(sequence) <= len(prompt):
                raise ValueError("missing or mismatched generation prefix/completion")
            stage = "completion_decode"
            completion = decode(sequence[len(prompt):])
            if not isinstance(completion, str):
                raise ValueError("decoder did not return text")
            self._append({"event": "decoded_completion", "proposal_id": proposal_id,
                          "completion": completion})
            stage = "completion_parse"
            decision = parse_completion(completion, self.menu)
        except BaseException as exc:
            self._reject(proposal_id, stage, exc)
            raise
        return self._execute_proposal(decision, completion, proposal_id)

    def execute(self, decision: Decision, completion: str, *, provenance=None) -> dict | None:
        if self._halted:
            raise RuntimeError("attempt is terminal or ambiguous")
        proposal_id = self._proposal(completion, provenance or {"kind": "explicit_completion"})
        return self._execute_proposal(decision, completion, proposal_id)

    def _execute_proposal(self, decision, completion, proposal_id):
        stage = "completion_parse"
        try:
            # Independently bind the requested action to the actual completion.
            parsed = parse_completion(completion, self.menu)
            stage = "decision_binding"
            if parsed != decision:
                raise ValueError("decision does not match completion")
            if not decision.stop:
                action = next(a for a in self.menu if a.action_id == decision.action_id)
                stage = "joint_feasibility"
                if self._joint_feasible is not None and self._joint_feasible(action) is not True:
                    raise ValueError("joint feasibility rejected")
                stage = "budget"
                if self._spent + action.cost_units > self.budget_units:
                    raise ValueError("insufficient experimental budget")
        except BaseException as exc:
            self._reject(proposal_id, stage, exc)
            raise
        if decision.stop:
            self._append({"event": "stop", "proposal_id": proposal_id, "completion": completion,
                          "spent_units": self._spent})
            self._halted = True
            return None
        request_id = self._requests
        self._append({"event": "reserve", "proposal_id": proposal_id,
                      "request_id": request_id, "action": asdict(action),
                      "completion": completion, "cost_units": action.cost_units})
        self._spent += action.cost_units
        self._requests += 1
        returned = False
        try:
            response = self._respond(action, request_id)
            returned = True
            if not isinstance(response, Mapping) or set(response) != set(action.measure):
                raise ValueError("response must contain exactly the requested measurements")
            if not all(finite(v) for v in response.values()):
                raise ValueError("non-finite or nonnumeric measurement")
            record = {"request_id": request_id, "action_id": action.action_id,
                      "kind": action.kind, "targets": dict(action.targets),
                      "values": {k: float(response[k]) for k in action.measure},
                      "cost_units": action.cost_units}
            self._append({"event": "response", **record})
            self._history.append(record)
            return json.loads(canonical(record))
        except BaseException as exc:
            self._halted = True
            self._append({"event": "failure", "request_id": request_id,
                          "response_returned_to_driver": True if returned else None,
                          "exception_type": type(exc).__name__, "message": str(exc),
                          "spent_units": self._spent})
            raise

    def state(self, objective: str, summaries: tuple[MechanismSummary, ...],
              domain_notes: str = "") -> dict:
        if not isinstance(objective, str) or not objective or not isinstance(domain_notes, str):
            raise ValueError("objective and domain notes must be text")
        if type(summaries) is not tuple:
            raise ValueError("immutable mechanism summaries required")
        names = [s.node for s in summaries]
        if len(set(names)) != len(names) or not set(names) <= {n.name for n in self.graph}:
            raise ValueError("unknown or duplicated fitted mechanism")
        state = {"version": VERSION, "objective": objective, "domain_notes": domain_notes,
                 "graph": [asdict(n) for n in self.graph], "menu": [asdict(a) for a in self.menu],
                 "history": self._history, "scm": [asdict(s) for s in summaries],
                 "spent_units": self._spent, "remaining_units": self.budget_units - self._spent,
                 "requests": self._requests, "returned_responses": len(self._history),
                 "terminal": self._halted}
        # Roundtrip produces a detached, JSON-only public state.
        return json.loads(canonical(state))

    def close(self):
        if not self._file.closed:
            if not self._halted:
                self._append({"event": "closed", "spent_units": self._spent,
                              "requests": self._requests, "returned_responses": len(self._history)})
                self._halted = True
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None and not self._halted:
            self._halted = True
            self._append({"event": "driver_failure", "exception_type": exc_type.__name__,
                          "message": str(exc), "spent_units": self._spent})
        self.close()


def policy_prompt(public_state: dict) -> str:
    """The runner passes only the detached state from ChargedExecutor.state()."""
    required = {"version", "objective", "domain_notes", "graph", "menu", "history", "scm",
                "spent_units", "remaining_units", "requests", "returned_responses", "terminal"}
    if set(public_state) != required or public_state["version"] != VERSION:
        raise ValueError("unexpected public state fields or version")
    return ("Choose the next experimental intervention using only this supplied state. "
            "Each environment response costs the listed units. Repeating an intervention "
            "is a new paid measurement. Return exactly one JSON object containing action_id "
            "from the menu, or {\"stop\":true}. Do not invent observations or actions.\n"
            + canonical(public_state) + "\nDecision:\n")
