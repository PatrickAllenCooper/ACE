"""Bounded CPU-only linear-Gaussian worlds for a future intervention screen.

Everything here except public_design() is trusted-driver/private material.
Descriptor pins and executor journal checks authenticate bindings, not scientific
provenance or malicious same-user code. No scientific configuration is supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random

from llm_intervention_contract import (
    Action, Node, SCOPES, VERSION as CONTRACT_VERSION, canonical, digest,
    finite, integer, validate_design,
)


VERSION = "ace-linear-gaussian-world-v1"
STREAMS = ("mechanism", "initial", "measurement", "policy", "train_score", "private_eval")
MAX_NODES, MAX_ACTIONS, MAX_REQUESTS = 8, 64, 80
MAX_JOURNAL_BYTES = 8 * 1024 * 1024


def _name(value):
    return type(value) is str and 0 < len(value) <= 256


def _namespace(scope, world_id, stream):
    if (type(scope) is not str or scope not in SCOPES
            or not integer(world_id) or world_id >= 2**63
            or type(stream) is not str or stream not in STREAMS):
        raise ValueError("invalid world/stream namespace")
    if stream == "train_score" and scope not in ("artificial", "policy_train"):
        raise ValueError("train_score requires artificial or policy_train scope")


def keyed_seed(scope, world_id, stream, *key):
    """Domain-separated, order-independent keys; never mutates global RNG state.

    v1 is intentionally distinct from contract.stream_seed, whose train_score
    rejects artificial fixtures. All six namespaces use this one version here.
    Keys are bounded scalar strings/integers, not ambiguous joined strings.
    """
    _namespace(scope, world_id, stream)
    if len(key) > 16 or any(not (_name(x) or integer(x)) for x in key):
        raise ValueError("bounded string/nonnegative integer keys required")
    return int.from_bytes(hashlib.sha256(canonical([VERSION, scope, world_id, stream, key]).encode()).digest(), "big")


def _normal(scope, world_id, stream, *key):
    return random.Random(keyed_seed(scope, world_id, stream, *key)).gauss(0., 1.)


def _graph(graph):
    if type(graph) is not tuple or not 1 <= len(graph) <= MAX_NODES:
        raise ValueError("bounded immutable graph required")
    seen = set()
    for n in graph:
        if (type(n) is not Node or not _name(n.name) or type(n.unit) is not str
                or len(n.unit) > 256 or any(not _name(p) for p in n.parents)
                or n.name in seen or not set(n.parents) <= seen):
            raise ValueError("unique known DAG nodes in topological order required")
        seen.add(n.name)


def _menu(graph, menu):
    _graph(graph)
    if (type(menu) is not tuple or not 1 <= len(menu) <= MAX_ACTIONS
            or any(type(a) is not Action or not _name(a.action_id)
                   or any(not _name(n) for n in a.measure) for a in menu)):
        raise ValueError("bounded immutable action menu required")
    validate_design(graph, menu)


@dataclass(frozen=True)
class LinearMechanism:
    node: str
    intercept: float
    coefficients: tuple[tuple[str, float], ...]
    noise_variance: float

    def __post_init__(self):
        if (not _name(self.node) or not finite(self.intercept)
                or type(self.coefficients) is not tuple
                or any(type(c) is not tuple or len(c) != 2 or not _name(c[0])
                       or not finite(c[1]) for c in self.coefficients)
                or len({c[0] for c in self.coefficients}) != len(self.coefficients)
                or not finite(self.noise_variance) or self.noise_variance < 0):
            raise ValueError("finite immutable linear mechanism required")


@dataclass(frozen=True)
class WorldDescriptor:
    scope: str
    world_id: int
    graph: tuple[Node, ...]
    mechanisms: tuple[LinearMechanism, ...]

    def __post_init__(self):
        _namespace(self.scope, self.world_id, "mechanism")
        _graph(self.graph)
        if (type(self.mechanisms) is not tuple or len(self.mechanisms) != len(self.graph)
                or any(type(m) is not LinearMechanism or m.node != n.name
                       or tuple(p for p, _ in m.coefficients) != n.parents
                       for n, m in zip(self.graph, self.mechanisms))):
            raise ValueError("mechanisms must match graph nodes and ordered parents exactly")

    def private_payload(self):
        return json.loads(canonical({"version": VERSION, **asdict(self)}))

    @property
    def sha256(self):
        return digest(self.private_payload())

    def public_design(self):
        """Detached graph only: no seeds, truth, mechanism pins or evaluation."""
        return json.loads(canonical({"graph": [asdict(n) for n in self.graph]}))

    @classmethod
    def from_private_payload(cls, payload, *, expected_sha256):
        if (type(payload) is not dict or set(payload) != {
                "version", "scope", "world_id", "graph", "mechanisms"}
                or payload["version"] != VERSION or digest(payload) != expected_sha256):
            raise ValueError("private descriptor schema/pin mismatch")
        if type(payload["graph"]) is not list or type(payload["mechanisms"]) is not list:
            raise ValueError("descriptor arrays required")
        nodes, mechanisms = [], []
        for n in payload["graph"]:
            if (type(n) is not dict or set(n) != {"name", "parents", "unit", "intervention_bounds"}
                    or type(n["parents"]) is not list
                    or (n["intervention_bounds"] is not None and type(n["intervention_bounds"]) is not list)):
                raise ValueError("invalid serialized node")
            nodes.append(Node(n["name"], tuple(n["parents"]), n["unit"],
                              None if n["intervention_bounds"] is None else tuple(n["intervention_bounds"])))
        for m in payload["mechanisms"]:
            if (type(m) is not dict or set(m) != {"node", "intercept", "coefficients", "noise_variance"}
                    or type(m["coefficients"]) is not list
                    or any(type(c) is not list for c in m["coefficients"])):
                raise ValueError("invalid serialized mechanism")
            mechanisms.append(LinearMechanism(m["node"], m["intercept"],
                              tuple(tuple(c) for c in m["coefficients"]), m["noise_variance"]))
        return cls(payload["scope"], payload["world_id"], tuple(nodes), tuple(mechanisms))


def generate_world(*, scope, world_id, graph, intercept_means, coefficient_means,
                   noise_variances, intercept_sd, coefficient_sd):
    """Realize one supplied DAG, with explicit independent Gaussian parameter law.

    Tuples follow graph/parent order. No graph/menu/task/scope defaults, selection,
    rejection sampling, retries, nonlinear families or scientific presets exist.
    Zero SDs supply exact analytic mechanisms without drawing parameter noise.
    """
    _namespace(scope, world_id, "mechanism")
    _graph(graph)
    for values in (intercept_means, coefficient_means, noise_variances):
        if type(values) is not tuple or len(values) != len(graph):
            raise ValueError("one immutable parameter entry per graph node required")
    if (any(not finite(s) or s < 0 for s in (intercept_sd, coefficient_sd))
            or not all(finite(x) for x in intercept_means)
            or any(not finite(x) or x < 0 for x in noise_variances)
            or any(type(row) is not tuple or len(row) != len(n.parents)
                   or not all(finite(x) for x in row) for n, row in zip(graph, coefficient_means))):
        raise ValueError("invalid explicit parameter/noise law")
    mechanisms = []
    for n, intercept, means, variance in zip(graph, intercept_means, coefficient_means, noise_variances):
        if intercept_sd:
            intercept += intercept_sd * _normal(scope, world_id, "mechanism", n.name, "intercept")
        coefficients = tuple((p, mean + coefficient_sd * _normal(
            scope, world_id, "mechanism", n.name, "parent", p) if coefficient_sd else mean)
            for p, mean in zip(n.parents, means))
        mechanisms.append(LinearMechanism(n.name, intercept, coefficients, variance))
    return WorldDescriptor(scope, world_id, graph, tuple(mechanisms))


def _evaluate(world, action, noise=None):
    _menu(world.graph, (action,))
    values = dict(action.targets)
    for m in world.mechanisms:
        if m.node not in values:
            # A hard intervention replaces the entire equation, including noise.
            values[m.node] = m.intercept + sum(c * values[p] for p, c in m.coefficients)
            if noise is not None and m.noise_variance:
                values[m.node] += math.sqrt(m.noise_variance) * noise(m.node)
            if not finite(values[m.node]):
                raise ValueError("nonfinite generated response")
    return {name: float(values[name]) for name in action.measure}


def private_structural_means(world, queries, *, expected_sha256):
    """Private true interventional means, independent of learner or sampled rows.

    This is deterministic evaluation, not a paid response or policy stopping
    signal. No private_eval noise is needed for a linear structural mean. A
    future stochastic evaluator must use the separate private_eval namespace.
    """
    if type(world) is not WorldDescriptor or world.sha256 != expected_sha256:
        raise ValueError("private evaluator descriptor pin mismatch")
    _menu(world.graph, queries)
    return tuple(_evaluate(world, q) for q in queries)


class ResponseAdapter:
    """Trusted callback for an existing ChargedExecutor; never reserves charges.

    Its private exclusive journal binds each returned subset to a descriptor,
    exact durable reservation, stream and replicate. A consumed request cannot
    be retried in this adapter. Failed/ambiguous callbacks halt it. Use one
    adapter per executor/stream/replicate binding, kept outside policy processes.
    """

    def __init__(self, world, *, expected_sha256, menu, stream, replicate_id,
                 request_namespace, executor_journal, journal, max_requests,
                 structural=False):
        if type(world) is not WorldDescriptor:
            raise ValueError("immutable world descriptor required")
        _namespace(world.scope, world.world_id, stream)
        _menu(world.graph, menu)
        if (stream not in ("initial", "measurement", "train_score")
                or not _name(replicate_id) or not _name(request_namespace)
                or not integer(max_requests, 1) or max_requests > MAX_REQUESTS
                or type(structural) is not bool or (structural and stream != "train_score")):
            raise ValueError("invalid charged response binding")
        self._world, self._menu = world, menu
        self._stream, self._replicate, self._namespace = stream, replicate_id, request_namespace
        self._executor = Path(executor_journal).absolute()
        if self._executor == Path(journal).absolute():
            raise ValueError("adapter and reservation journals must differ")
        self._structural, self._limit = structural, max_requests
        self._seen, self._halted, self._previous = set(), False, None
        self._file = Path(journal).open("x", encoding="utf-8")
        try:
            self._append({"event": "bind", "version": VERSION, "scope": world.scope,
                "world_id": world.world_id, "descriptor_sha256": world.sha256,
                "expected_sha256": expected_sha256, "stream": stream,
                "replicate_id": replicate_id, "request_namespace": request_namespace,
                "executor_journal": str(self._executor), "menu": [asdict(a) for a in menu],
                "max_requests": max_requests, "structural": structural,
                "reservation_authority": "ChargedExecutor", "scientific_provenance_qualified": False})
            if world.sha256 != expected_sha256:
                raise ValueError("response descriptor pin mismatch")
        except BaseException as exc:
            self._fail(exc)
            self._file.close()
            raise

    def _append(self, event):
        record = dict(event, previous_sha256=self._previous,
                      utc=datetime.now(timezone.utc).isoformat())
        record["sha256"] = digest(record)
        try:
            self._file.write(canonical(record) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._halted = True
            raise
        self._previous = record["sha256"]

    def _fail(self, exc):
        self._halted = True
        self._append({"event": "failure", "exception_type": type(exc).__name__,
                      "message": str(exc), "retry_allowed": False})

    def _reservation(self, action, request_id):
        with self._executor.open("rb") as handle:
            raw = handle.read(MAX_JOURNAL_BYTES + 1)
        if len(raw) > MAX_JOURNAL_BYTES or not raw.endswith(b"\n"):
            raise ValueError("executor journal is oversized or incomplete")
        records, previous = [], None
        for line in raw.splitlines():
            record = json.loads(line)
            claimed = record.pop("sha256")
            if record["previous_sha256"] != previous or digest(record) != claimed:
                raise ValueError("executor journal hash chain mismatch")
            previous = claimed
            records.append(record)
        first, last = records[0], records[-1]
        if (first.get("event") != "begin" or first.get("version") != CONTRACT_VERSION
                or canonical(first.get("graph")) != canonical([asdict(n) for n in self._world.graph])
                or canonical(first.get("menu")) != canonical([asdict(a) for a in self._menu])
                or last.get("event") != "reserve" or type(last.get("request_id")) is not int
                or last["request_id"] != request_id
                or canonical(last.get("action")) != canonical(asdict(action))
                or last.get("cost_units") != action.cost_units):
            raise ValueError("callback does not match its current durable reservation")
        return previous

    def __call__(self, action, request_id):
        if self._halted or self._file.closed:
            raise RuntimeError("response adapter is terminal or ambiguous")
        try:
            self._append({"event": "attempt", "request_repr": repr(request_id),
                          "action_repr": repr(action)})
            if (type(action) is not Action or action not in self._menu
                    or not integer(request_id) or request_id >= MAX_REQUESTS
                    or request_id in self._seen or len(self._seen) >= self._limit):
                raise ValueError("invalid/repeated request or action outside exact menu")
            reservation = self._reservation(action, request_id)
            self._seen.add(request_id)
            # Measurement choices do not alter the underlying structural draw.
            intervention = digest({"kind": action.kind, "targets": action.targets})
            key = (self._namespace, self._replicate, request_id, action.action_id, intervention)
            self._append({"event": "mapping", "request_id": request_id,
                "action": asdict(action), "reservation_sha256": reservation,
                "noise_key": list(key), "stream": self._stream,
                "descriptor_sha256": self._world.sha256, "structural": self._structural})
            noise = None if self._structural else lambda node: _normal(
                self._world.scope, self._world.world_id, self._stream, *key, node)
            values = _evaluate(self._world, action, noise)
            self._append({"event": "outcome", "request_id": request_id, "values": values,
                          "reservation_sha256": reservation, "returned_subset_ready": True})
            # The executor alone certifies acceptance; crash after this durable
            # event but before return remains ambiguous at its reservation.
            return values
        except BaseException as exc:
            self._fail(exc)
            raise

    def close(self):
        if not self._file.closed:
            try:
                if not self._halted:
                    self._append({"event": "closed", "responses_prepared": len(self._seen)})
            finally:
                self._halted = True
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None and not self._halted:
                self._fail(exc)
        finally:
            self.close()


def training_callbacks(stack, world, *, expected_sha256, menu, candidates, task,
                       replicate_ids, branch_output, journal_directory,
                       request_namespace, structural_scores):
    """Bind the existing replicated_branches callback signatures and filenames.

    stack is caller-owned contextlib.ExitStack. branch_output must be the new
    driver's output path; journal_directory already exists and is private.
    Branches use noisy measurement; shared scoring uses independent train_score
    (noise-disabled privileged supervision only if explicitly requested).
    This helper neither creates a ChargedExecutor nor changes its accounting.
    """
    if world.scope not in ("artificial", "policy_train"):
        raise ValueError("training callbacks require artificial/policy_train worlds")
    _menu(world.graph, menu)
    task.validate(world.graph)
    if (type(candidates) is not tuple or not 2 <= len(candidates) <= MAX_ACTIONS
            or any(not _name(c) for c in candidates) or len(set(candidates)) != len(candidates)
            or not set(candidates) <= {a.action_id for a in menu}
            or type(replicate_ids) is not tuple or not 2 <= len(replicate_ids) <= MAX_REQUESTS
            or any(not _name(r) for r in replicate_ids) or len(set(replicate_ids)) != len(replicate_ids)
            or any(task.target not in q.measure for q in task.queries)
            or type(structural_scores) is not bool or not _name(request_namespace)):
        raise ValueError("invalid fixed training candidate/replicate/task contract")
    branch, score = {}, {}
    for ri, rid in enumerate(replicate_ids):
        for ci, candidate in enumerate(candidates):
            stem = f"replicate-{ri:03d}.branch-{ci:03d}"
            branch[rid, candidate] = stack.enter_context(ResponseAdapter(world,
                expected_sha256=expected_sha256, menu=menu, stream="measurement", replicate_id=rid,
                request_namespace=request_namespace, executor_journal=Path(branch_output)/(stem + ".jsonl"),
                journal=Path(journal_directory)/(stem + ".world.jsonl"), max_requests=1))
        stem = f"replicate-{ri:03d}.scoring"
        score[rid] = stack.enter_context(ResponseAdapter(world, expected_sha256=expected_sha256,
            menu=task.queries, stream="train_score", replicate_id=rid, request_namespace=request_namespace,
            executor_journal=Path(branch_output)/(stem + ".jsonl"),
            journal=Path(journal_directory)/(stem + ".world.jsonl"), max_requests=len(task.queries),
            structural=structural_scores))

    def branch_respond(replicate_id, candidate_id, action, journal_request_id):
        if (replicate_id, candidate_id) not in branch or action.action_id != candidate_id:
            raise ValueError("branch replicate/candidate/action membership mismatch")
        return branch[replicate_id, candidate_id](action, journal_request_id)

    def score_respond(replicate_id, action, journal_request_id):
        if replicate_id not in score:
            raise ValueError("scoring replicate membership mismatch")
        return score[replicate_id](action, journal_request_id)

    return branch_respond, score_respond
