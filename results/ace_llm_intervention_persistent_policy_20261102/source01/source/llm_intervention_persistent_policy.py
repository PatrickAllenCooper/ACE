"""Bounded JSONL transport for a policy initialized once in an owned process.

The CPU caller keeps environment execution, charged ledgers and private
evaluation. Only a validated public task envelope crosses this transport.
Handshake identity is an exact attestation: the future model initializer must
authenticate the actual artifacts and numerical setup it attests to. This is
process separation, not a sandbox against malicious same-UID programs.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from llm_intervention_contract import canonical, digest, parse_completion
from llm_intervention_episode import ProcessPolicy
from llm_intervention_task import PredictionTask, VERSION as TASK_VERSION, unpack_envelope


VERSION = "ace-persistent-public-policy-v1"


def _save(path, value):
    payload = (canonical(value) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate protocol key")
        result[key] = value
    return result


def _decode(raw):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=bad_constant)
    if type(value) is not dict:
        raise ValueError("protocol object required")
    return value


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("unexpected protocol fields")


def _identity(value):
    _keys(value, {"model_sha256", "source_sha256", "adapter_sha256", "numeric_recipe", "numeric_recipe_sha256"})
    for key in ("model_sha256", "source_sha256", "adapter_sha256", "numeric_recipe_sha256"):
        pin = value[key]
        if type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError("invalid identity pin")
    if type(value["numeric_recipe"]) is not dict or digest(value["numeric_recipe"]) != value["numeric_recipe_sha256"]:
        raise ValueError("numerical recipe differs from its pin")
    return _decode(canonical(value).encode())


def _request_id(value, previous):
    if type(value) is not int or not 0 <= value < 2**63 or value <= previous:
        raise ValueError("request IDs must be strictly increasing nonnegative integers")


def _envelope(value):
    detached = _decode(canonical(value).encode())
    state, _, _, menu, _ = unpack_envelope(detached)
    return detached, state, menu


def _completion(value, state, menu, limit):
    if type(value) is not str or not value or len(value.encode("utf-8")) > limit:
        raise ValueError("completion exceeds its byte bound or is not text")
    decision = parse_completion(value, menu)
    if not decision.stop:
        action = next(a for a in menu if a.action_id == decision.action_id)
        if state["terminal"] or action.cost_units > state["remaining_units"]:
            raise ValueError("completion is not an affordable active-state action")
    return value  # Never strip, extract, repair, or replace the completion.


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def _bounds(input_bytes, output_bytes, completion_bytes, max_requests):
    for value, cap in ((input_bytes, 16*1024*1024), (output_bytes, 1024*1024),
                       (completion_bytes, 4096), (max_requests, 100000)):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError("invalid transport bound")


def serve_policy(initialize, *, identity, input_stream=None, output_stream=None,
                 max_input_bytes=1024*1024, max_output_bytes=16384,
                 max_completion_bytes=4096, max_requests=256):
    """Initialize once, then serve sequential public envelopes until stdin EOF.

    initialize() returns decide(envelope)->exact completion text. Neither
    function receives an executor, ledger or private evaluator through this API.
    The owning client enforces deadlines and process cleanup. An invalid frame,
    decision or callback exception ends this worker; there is no retry loop.
    Streams are binary. Initialization/decision prints go to stderr.
    """
    _bounds(max_input_bytes, max_output_bytes, max_completion_bytes, max_requests)
    identity = _identity(identity)
    pin = digest(identity)
    inp = sys.stdin.buffer if input_stream is None else input_stream
    out = sys.stdout.buffer if output_stream is None else output_stream
    def emit(value):
        raw = (canonical(value)+"\n").encode()
        if len(raw) > max_output_bytes:
            raise ValueError("worker output line exceeds bound")
        out.write(raw)
        out.flush()
    with contextlib.redirect_stdout(sys.stderr):
        decide = initialize()
    if not callable(decide):
        raise ValueError("initializer must return a decision callback")
    emit({"type": "ready", "version": VERSION, "identity": identity, "identity_sha256": pin})
    previous, count = -1, 0
    while True:
        raw = inp.readline(max_input_bytes+1)
        if not raw:
            return
        if len(raw) > max_input_bytes or not raw.endswith(b"\n"):
            raise ValueError("worker input line exceeds bound or is unterminated")
        request = _decode(raw)
        _keys(request, {"type", "request_id", "envelope", "envelope_sha256", "identity_sha256"})
        if request["type"] != "request" or request["identity_sha256"] != pin or count >= max_requests:
            raise ValueError("worker request identity/type/count mismatch")
        _request_id(request["request_id"], previous)
        envelope, state, menu = _envelope(request["envelope"])
        if digest(envelope) != request["envelope_sha256"]:
            raise ValueError("worker envelope digest mismatch")
        previous, count = request["request_id"], count+1
        with contextlib.redirect_stdout(sys.stderr):
            completion = decide(envelope)
        completion = _completion(completion, state, menu, max_completion_bytes)
        emit({"type": "completion", "request_id": previous, "identity_sha256": pin,
              "envelope_sha256": request["envelope_sha256"], "completion": completion})


def cleanup_process_group(process, seconds, *, graceful=False):
    """Bounded cleanup of this Popen session only; always return evidence."""
    result = {"pgid": process.pid, "term_sent": False, "kill_sent": False, "errors": []}
    end = time.monotonic()+seconds
    def alive():
        process.poll()
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            result["errors"].append(type(exc).__name__+": "+str(exc))
            return True
    try:
        if process.stdin and not process.stdin.closed:
            try: process.stdin.close()
            except OSError as exc: result["errors"].append("stdin close: "+str(exc))
        if graceful:
            until = min(end, time.monotonic()+min(.2, seconds/4))
            while time.monotonic() < until and alive():
                time.sleep(.01)
        for sig, key, fraction in ((signal.SIGTERM, "term_sent", .5), (signal.SIGKILL, "kill_sent", 1.)):
            if not alive():
                break
            try:
                os.killpg(process.pid, sig)
                result[key] = True
            except ProcessLookupError:
                break
            except OSError as exc:
                result["errors"].append(type(exc).__name__+": "+str(exc))
            until = min(end, time.monotonic()+max(0., end-time.monotonic())*fraction)
            while time.monotonic() < until and alive():
                time.sleep(.01)
        result["group_still_exists"] = alive()
    except BaseException as exc:
        result["errors"].append(type(exc).__name__+": "+str(exc))
        result["group_still_exists"] = None
    process.poll()
    result["leader_returncode"] = process.returncode
    return result


class PersistentPolicy(ProcessPolicy):
    """One owned worker; call(envelope, request_id) returns exact completion.

    choose(state, directory, decision_index) adapts the existing ProcessPolicy
    interface when task=PredictionTask is supplied. Transport evidence lives in
    this client's exclusive output directory. Calls are sequential, not thread
    safe. Use a context manager or close(); any failure permanently closes the
    session. Inherited filesystem/environment access is not an OS sandbox.
    """
    def __init__(self, command, timeout_seconds, *, expected_identity, output, task=None,
                 startup_seconds=30., cleanup_seconds=2., max_input_bytes=1024*1024,
                 max_output_bytes=16384, max_completion_bytes=4096, max_stderr_bytes=65536,
                 max_requests=256):
        super().__init__(command, timeout_seconds)
        for value, name in ((startup_seconds, "startup"), (cleanup_seconds, "cleanup")):
            _positive(value, name)
        _bounds(max_input_bytes, max_output_bytes, max_completion_bytes, max_requests)
        if type(max_stderr_bytes) is not int or not 1 <= max_stderr_bytes <= 1024*1024:
            raise ValueError("invalid total stderr byte bound")
        if task is not None and not isinstance(task, PredictionTask):
            raise ValueError("choose adapter requires PredictionTask")
        self.identity = _identity(expected_identity)
        self.identity_pin = digest(self.identity)
        self.output, self.task = Path(output), task
        self.input_limit, self.output_limit, self.completion_limit = max_input_bytes, max_output_bytes, max_completion_bytes
        self.stderr_limit, self.max_requests, self.cleanup_seconds = max_stderr_bytes, max_requests, cleanup_seconds
        self.previous, self.count, self.closed, self.failed = -1, 0, False, False
        self.process, self.selector = None, None
        self.raw_files, self.bytes = {}, {"stdout": 0, "stderr": 0}
        self.stage, self.discarded_bytes = "launch", 0
        self.output.mkdir(exist_ok=False)
        try:
            _save(self.output/"launch.json", {"version": VERSION, "command": command,
                "expected_identity": self.identity, "identity_sha256": self.identity_pin,
                "timeout_seconds": timeout_seconds, "startup_seconds": startup_seconds,
                "cleanup_seconds": cleanup_seconds, "max_input_bytes": max_input_bytes,
                "max_output_bytes": max_output_bytes, "max_completion_bytes": max_completion_bytes,
                "max_stderr_bytes": max_stderr_bytes, "max_requests": max_requests,
                "provenance_qualified": False, "retry_allowed": False})
            for name in ("stdout", "stderr"):
                self.raw_files[name] = (self.output/(name+".bin")).open("xb")
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True, bufsize=0, cwd=self.output)
            _save(self.output/"process.json", {"pid": self.process.pid, "pgid": self.process.pid})
            self.selector = selectors.DefaultSelector()
            for name in ("stdin", "stdout", "stderr"):
                os.set_blocking(getattr(self.process, name).fileno(), False)
            for name in ("stdout", "stderr"):
                self.selector.register(getattr(self.process, name), selectors.EVENT_READ, name)
            self.stage = "handshake"
            hello = _decode(self._exchange(None, time.monotonic()+startup_seconds))
            _save(self.output/"observed-handshake.json", hello)
            _keys(hello, {"type", "version", "identity", "identity_sha256"})
            if (hello["type"] != "ready" or hello["version"] != VERSION
                    or canonical(_identity(hello["identity"])) != canonical(self.identity)
                    or hello["identity_sha256"] != self.identity_pin):
                raise ValueError("handshake identity differs from expected pins")
            _save(self.output/"ready.json", {"identity_sha256": self.identity_pin, "status": "ready"})
        except BaseException as exc:
            self._fail(exc)
            raise

    def _capture(self, name, data):
        cap = self.stderr_limit if name == "stderr" else (self.max_requests+1)*(self.output_limit+1)
        room = max(0, cap-self.bytes[name])
        retained = data[:room]
        if retained:
            self.raw_files[name].write(retained)
            self.raw_files[name].flush()
            os.fsync(self.raw_files[name].fileno())
            self.bytes[name] += len(retained)
        self.discarded_bytes += len(data)-len(retained)
        if len(data) > room:
            raise ValueError(name+" cumulative output bound exceeded; bounded prefix retained")

    def _exchange(self, payload, deadline):
        pending = memoryview(payload or b"")
        if payload is not None:
            # Reject output already waiting before a new request is sent.
            for key, _ in self.selector.select(0):
                if key.data == "stdout":
                    data = os.read(key.fd, self.output_limit+1)
                    self._capture("stdout", data)
                    raise ValueError("unsolicited stdout or worker exit between requests")
                if key.data == "stderr":
                    data = os.read(key.fd, 4096)
                    self._capture("stderr", data)
                    if not data: self.selector.unregister(key.fileobj)
            self.selector.register(self.process.stdin, selectors.EVENT_WRITE, "stdin")
        line = bytearray()
        try:
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("persistent policy deadline exceeded")
                events = self.selector.select(remaining)
                for key, _ in events:
                    if key.data == "stdin":
                        written = os.write(key.fd, pending)
                        pending = pending[written:]
                        if not pending: self.selector.unregister(key.fileobj)
                    elif key.data == "stderr":
                        data = os.read(key.fd, 4096)
                        self._capture("stderr", data)
                        if not data: self.selector.unregister(key.fileobj)
                    else:
                        data = os.read(key.fd, min(4096, self.output_limit+1-len(line)))
                        self._capture("stdout", data)
                        if not data:
                            raise EOFError("worker stdout closed before a complete response")
                        line.extend(data)
                        if len(line) > self.output_limit:
                            raise ValueError("worker output line exceeds bound; bounded prefix retained")
                        if b"\n" in line:
                            first, extra = bytes(line).split(b"\n", 1)
                            if pending or extra:
                                raise ValueError("unsolicited or multiple worker response lines")
                            return first
        finally:
            if payload is not None:
                try: self.selector.unregister(self.process.stdin)
                except KeyError: pass

    def _drain_stderr(self):
        if self.process is None or self.process.stderr.closed:
            return
        while True:
            try: data = os.read(self.process.stderr.fileno(), 4096)
            except BlockingIOError: return
            if not data: return
            self._capture("stderr", data)

    def _release(self):
        if self.selector: self.selector.close()
        if self.process:
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                if pipe and not pipe.closed: pipe.close()
        for handle in self.raw_files.values():
            if not handle.closed: handle.close()

    def _fail(self, exc):
        self.failed = self.closed = True
        errors = []
        failure = {"status": "failed", "stage": self.stage, "exception_type": type(exc).__name__,
            "message": str(exc), "last_completed_request_id": self.previous,
            "completed_requests": self.count, "retry_allowed": False, "provenance_qualified": False}
        try: _save(self.output/"failure-intent.json", failure)
        except BaseException as error: errors.append("intent: "+repr(error))
        cleanup = cleanup_process_group(self.process, self.cleanup_seconds) if self.process else None
        try: self._drain_stderr()
        except BaseException as error: errors.append("stderr: "+repr(error))
        try:
            _save(self.output/"failure.json", {**failure, "cleanup": cleanup,
                "captured_bytes": self.bytes, "discarded_observed_bytes": self.discarded_bytes,
                "persistence_errors": errors})
        finally:
            self._release()

    def call(self, public_envelope, request_id):
        if self.closed:
            raise RuntimeError("policy session is closed; no retry or restart")
        deadline = time.monotonic()+self.timeout
        prefix = f"request-{self.count:04d}"
        try:
            self.stage = "request_validation"
            _request_id(request_id, self.previous)
            if self.count >= self.max_requests:
                raise ValueError("session request bound exceeded")
            envelope, state, menu = _envelope(public_envelope)
            request = {"type": "request", "request_id": request_id, "identity_sha256": self.identity_pin,
                       "envelope": envelope, "envelope_sha256": digest(envelope)}
            raw = (canonical(request)+"\n").encode()
            if len(raw) > self.input_limit:
                raise ValueError("public request exceeds input byte bound")
            _save(self.output/(prefix+".input.json"), request)
            self.stage = "request_exchange"
            response = _decode(self._exchange(raw, deadline))
            _save(self.output/(prefix+".observed-response.json"), response)
            self.stage = "response_validation"
            _keys(response, {"type", "request_id", "identity_sha256", "envelope_sha256", "completion"})
            if (response["type"] != "completion" or type(response["request_id"]) is not int
                    or response["request_id"] != request_id or response["identity_sha256"] != self.identity_pin
                    or response["envelope_sha256"] != request["envelope_sha256"]):
                raise ValueError("completion response identity mismatch")
            completion = _completion(response["completion"], state, menu, self.completion_limit)
            if time.monotonic() > deadline:
                raise TimeoutError("persistent policy request validation exceeded deadline")
            _save(self.output/(prefix+".accepted.json"), {"request_id": request_id,
                "response_sha256": digest(response), "completion": completion})
            self.previous, self.count = request_id, self.count+1
            return completion
        except BaseException as exc:
            self._fail(exc)
            raise

    def choose(self, public_state, directory, decision_index):
        if self.task is None:
            exc = ValueError("choose requires a bound PredictionTask; use call for a public envelope")
            if not self.closed: self._fail(exc)
            raise exc
        try:
            envelope = {"version": TASK_VERSION, "state": public_state, "task": self.task.payload()}
            return self.call(envelope, decision_index)
        except BaseException as exc:
            if not self.closed: self._fail(exc)
            raise

    def close(self):
        if self.closed:
            return
        self.stage = "close"
        try:
            cleanup = cleanup_process_group(self.process, self.cleanup_seconds, graceful=True)
            _save(self.output/"close-cleanup.json", cleanup)
            self._drain_stderr()
            if cleanup["errors"] or cleanup["group_still_exists"] is not False or cleanup["leader_returncode"] != 0:
                raise RuntimeError("clean worker exit and owned process-group cleanup not confirmed")
            extra = os.read(self.process.stdout.fileno(), self.output_limit+1)
            if extra:
                self._capture("stdout", extra)
                raise ValueError("unsolicited worker output at close")
            _save(self.output/"closed.json", {"status": "closed", "cleanup": cleanup,
                "completed_requests": self.count, "last_request_id": self.previous,
                "captured_bytes": self.bytes, "provenance_qualified": False})
            self.closed = True
            self._release()
        except BaseException as exc:
            self._fail(exc)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.closed:
            if exc is None: self.close()
            else:
                self.stage = "caller_context_failure"
                self._fail(exc)
        return False
