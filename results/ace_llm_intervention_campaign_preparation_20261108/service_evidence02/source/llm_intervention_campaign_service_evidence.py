"""Read-only joins for one saved dynamic campaign service (no model execution).

Campaign/source pins are semantic digests; attempt/freeze pins are byte hashes.
The worker directory may be outside the sealed attempt. Its exact inventory is
returned for outer custody, and may optionally be independently supplied. This
does not qualify training acceptance, launch/runtime staging, scheduler telemetry,
GPU clearance, token decoding, shared initialization across policies or science.
In particular, an adapter's training-receipt pin denotes its checkpoint receipt,
not the independently accepted training run's outer acceptance receipt.
"""
from __future__ import annotations

from pathlib import Path
import re
import struct

import llm_intervention_campaign as campaign_api
import llm_intervention_episode_seal as seal
import llm_intervention_paired_episode as paired
from llm_intervention_contract import canonical, digest, finite, integer, parse_completion
from llm_intervention_persistent_policy import VERSION as PROTOCOL_VERSION, _completion
from llm_intervention_task import task_prompt, unpack_envelope

VERSION = "ace-campaign-service-evidence-v1"
MAX_REQUESTS = 8
MAX_FILES, MAX_BYTES = 4096, 128 * 1024 * 1024


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return campaign_api._decode(path.read_bytes())


def _same(left, right, message):
    _need(canonical(left) == canonical(right), message)


def _inventory(path):
    """Bound reads before reusing campaign custody checks; never follow links."""
    # System ancestors such as macOS /var may be links. The supplied evidence
    # root and every member beneath it must themselves be ordinary directories.
    _need(not path.is_symlink(), "symlink evidence root")
    if not path.exists():
        return {}
    _need(path.is_dir(), "evidence directory required")
    members = list(path.rglob("*"))
    _need(len(members) <= MAX_FILES and not any(p.is_symlink() for p in members),
          "evidence membership bound or symlink")
    files = [p for p in members if p.is_file()]
    _need(all(p.is_file() or p.is_dir() for p in members)
          and sum(p.stat().st_size for p in files) <= MAX_BYTES, "evidence byte/type bound")
    return {str(p.relative_to(path)): paired._sha(p.read_bytes()) for p in sorted(files)}


def _members(path, pattern, expression):
    result = {}
    for file in path.glob(pattern):
        match = re.fullmatch(expression, file.name)
        _need(match is not None and file.is_file(), "unexpected evidence member: " + file.name)
        ordinal = int(match[1])
        _need(ordinal < MAX_REQUESTS and ordinal not in result, "request membership exceeds bound")
        result[ordinal] = file
    return result


def _prefix(members, count, label):
    _need(set(members) == set(range(count)), label + " membership differs")


def _episode_plan(plan, policy_id):
    spec = next(s for s in plan["policy_specs"] if s["policy_id"] == policy_id)
    return {"graph": plan["world"]["graph"], "menu": plan["menu"],
            "budget": plan["initial_cost_units"] + plan["acquisition_cap_units"],
            "initial_actions": plan["initial_actions"], "objective": plan["objective"],
            "domain_notes": plan["domain_notes"], "policy_command": spec["command"],
            "policy_timeout_seconds": spec["timeout_seconds"]}


def _arm(path, arm, source, policy_id):
    """Authenticate raw and sealed copies, including failed/unresolved arms."""
    plan = source["plan"]
    result = {"world_id": arm["world_id"], "status": arm["status"], "reason": arm["reason"],
              "task_sha256": digest(plan["task"]), "requests": [],
              "response_attempts": None, "returned_responses": None, "charged_units": None,
              "acquisition_charged_units": None, "stop_reason": None,
              "acquisition_cap_units": plan["acquisition_cap_units"]}
    if arm["path"] is None:
        return result, None
    directory = path / arm["path"]
    receipt = _read(directory / "arm-receipt.json")
    inventory = _inventory(directory)
    _need((directory / "arm-receipt.candidate.json").read_bytes()
          == (directory / "arm-receipt.json").read_bytes(), "arm candidate differs")
    _same({n: p for n, p in inventory.items() if n not in
           ("arm-receipt.json", "arm-receipt.candidate.json")},
          receipt["source_evidence_sha256"], "arm source inventory differs")
    _same(_read(directory / "paired-plan.json"), plan, "arm paired plan differs")
    if receipt["status"] == "unresolved":
        _need(arm["status"] == "failed" and receipt["unresolved_reason"], "unresolved arm claims success")
        result["reason"] = receipt["unresolved_reason"]
        return result, {"receipt": receipt, "unresolved": True}
    receipt, inputs = paired._load_arm(directory, arm["receipt_sha256"], source["plan_sha256"], policy_id)
    for name, field in (("world.json", "world"), ("task.json", "task"), ("learner.json", "learner")):
        _need(inputs[name] == paired._raw(plan[field]) == (directory / name).read_bytes(),
              "saved arm source differs: " + name)
    ep = _episode_plan(plan, policy_id)
    _same(campaign_api._decode(inputs["episode-plan.json"]), ep, "episode plan differs")
    _need(inputs["episode-plan.json"] == (directory / "episode/plan.json").read_bytes()
          and inputs["executor.jsonl"] == (directory / "episode/journal.jsonl").read_bytes(),
          "sealed/original episode evidence differs")
    dispositions = [directory / "episode" / n for n in ("terminal.json", "failure.json")
                    if (directory / "episode" / n).exists()]
    _need(len(dispositions) == 1 and dispositions[0].read_bytes() == inputs["episode-disposition.json"],
          "original episode disposition differs")
    objects = {n: campaign_api._decode(b) for n, b in inputs.items() if n != "executor.jsonl"}
    for name in objects:
        if name.startswith("return-"):
            _need(inputs[name] == (directory / name).read_bytes(), "sealed/original return differs")
    _, world, _, menu = paired._validate(plan, source["plan_sha256"])
    chain = seal._journal(inputs["executor.jsonl"], ep, menu)
    # Local witnesses check captured returns only. Cross-policy shared-initial
    # custody remains the existing paired finalizer's responsibility.
    initial = tuple(paired._sha(paired._raw(r)) for r in chain["receipts"][:len(plan["initial_actions"])])
    seal._returns(objects, chain, initial)
    outer, _ = seal._disposition(objects, receipt["input_sha256"], ep, world.graph, chain,
                                 world.scope, world.world_id, policy_id)
    _need(outer["outer_provenance_receipt_sha256"] == plan["outer_provenance_receipt_sha256"],
          "outer provenance differs")
    diagnostics = _read(directory / "execution-diagnostics.json")
    for key in ("episode_returned_cleanly", "adapters_closed_cleanly"):
        _need(diagnostics[key] is outer[key], "arm execution diagnostics differ")
    _need((arm["status"] == "completed") == (outer["status"] == "complete"), "arm success contradicts episode")
    events = [campaign_api._decode(line) for line in inputs["executor.jsonl"].splitlines()]
    proposals, histories, history = [], [], []
    for event in events:
        if event["event"] == "proposal" and event["provenance"]["kind"] == "process_policy":
            proposals.append(event)
            histories.append(list(history))
        if event["event"] == "response":
            history.append({k: event[k] for k in seal.RECEIPT_FIELDS})
    result.update(response_attempts=len(chain["attempts"]), returned_responses=len(chain["receipts"]),
                  charged_units=chain["spent"], stop_reason=chain["stop_reason"],
                  acquisition_charged_units=sum(a["action"].cost_units for a in chain["attempts"]
                                                if a["phase"] == "acquisition"))
    return result, {"receipt": receipt, "unresolved": False, "plan": ep, "graph": world.graph,
                    "learner": plan["learner"], "chain": chain, "events": events,
                    "proposals": proposals, "histories": histories}


def _generation(ordinal, envelope, identity, members):
    """Check a completed generation, without tokenizer/tensor deserialization."""
    g = _read(members["generation"][ordinal])
    raw = _read(members["raw"][ordinal])
    started = _read(members["started"][ordinal])
    record = _read(members["generated"][ordinal])
    _need(set(record) == {"version", "envelope_sha256", "task_sha256", "prompt_sha256", "prompt_tokens",
              "longest_command_tokens", "max_tokens", "max_new_tokens", "legal_commands", "enable_thinking",
              "prompt_ids", "sequence_ids", "completion", "new_training_updates", "environment_callbacks_available"}
          and record["version"] == "ace-shared-task-language-policy-v1", "generation record schema differs")
    _same(started, {"tokens_available": False, "status": "generation_invoked"}, "raw generation start differs")
    _need(all(k in g and canonical(g[k]) == canonical(v) for k, v in record.items())
          and set(g) == set(record) | {"ordinal", "elapsed_seconds", "cuda_interval_ms",
              "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "tensor_versions_unchanged",
              "cross_request_cache_supplied"}, "unaccepted/generated record differs")
    state, _, _, menu, _ = unpack_envelope(envelope)
    recipe = identity["numeric_recipe"]
    _need(type(g["ordinal"]) is int and g["ordinal"] == ordinal
          and g["envelope_sha256"] == digest(envelope) and g["task_sha256"] == digest(envelope["task"])
          and g["prompt_sha256"] == paired._sha(task_prompt(envelope).encode()), "generation envelope/prompt join differs")
    prompt, sequence = g["prompt_ids"], g["sequence_ids"]
    _need(type(prompt) is list and type(sequence) is list and prompt
          and all(integer(t) and t < 151936 for t in sequence)
          and sequence[:len(prompt)] == prompt and sequence[-1] == 151645
          and all(integer(t) for t in prompt) and type(g["prompt_tokens"]) is int
          and g["prompt_tokens"] == len(prompt)
          and 1 < len(sequence)-len(prompt) <= recipe["max_new_tokens"]
          and len(prompt)+recipe["max_new_tokens"] <= recipe["max_tokens"], "generated token prefix/limits differ")
    _need(g["max_tokens"] == recipe["max_tokens"] and g["max_new_tokens"] == recipe["max_new_tokens"]
          and integer(g["longest_command_tokens"], 1)
          and len(sequence)-len(prompt) <= g["longest_command_tokens"] <= recipe["max_new_tokens"]
          and g["legal_commands"] == 1 + sum(a.cost_units <= state["remaining_units"] for a in menu)
          and g["enable_thinking"] is False, "generation token admission differs")
    _need(type(g["new_training_updates"]) is int and g["new_training_updates"] == 0
          and g["environment_callbacks_available"] is False and g["tensor_versions_unchanged"] is True
          and g["cross_request_cache_supplied"] is False, "inference-only/tensor invariants differ")
    _need(finite(g["cuda_interval_ms"]) and g["cuda_interval_ms"] > 0
          and finite(g["elapsed_seconds"]) and g["elapsed_seconds"] > 0
          and integer(g["peak_gpu_allocated_bytes"], 1)
          and integer(g["peak_gpu_reserved_bytes"], g["peak_gpu_allocated_bytes"]), "GPU work evidence missing")
    _completion(g["completion"], state, menu, 4096)
    n = len(sequence)
    _need(raw["status"] == "returned_unaccepted" and raw["tokens_available"] is True
          and raw["returned_type"] == "Tensor" and raw["dtype"] == "torch.int64"
          and canonical(raw["shape"]) == canonical([1, n]) and type(raw["numel"]) is int and raw["numel"] == n
          and type(raw["captured_elements"]) is int and raw["captured_elements"] == n
          and raw["truncated"] is False and raw["token_prefix"] == sequence, "raw token metadata differs")
    binary = members["binary"][ordinal].read_bytes()
    _need(paired._sha(binary) == raw["raw_sha256"] and binary == struct.pack("<" + "q"*n, *sequence),
          "raw binary/token lineage differs")
    return g


def validate_campaign_service(*, campaign, campaign_sha256, source_plan_pins, campaign_root,
                              policy_id, worker_output_dir, freeze_sha256, identity_sha256,
                              attempt_sha256, worker_evidence_sha256=None):
    """Validate one <=8-request service across the two supplied artificial worlds.

    All inputs are explicit. ``worker_evidence_sha256``, when supplied, is an
    independently retained relative-file -> byte-SHA256 inventory (not a digest).
    Without it the returned worker inventory is observed, not independently
    authenticated. This function writes nothing and never finalizes a campaign.

    Invalid custody or contradictory joins raise ValueError. Retained execution
    failures/unstarted/unresolved arms return operational_success=False; closure
    evidence never upgrades them to success. Counts with unknown histories are
    explicitly partial. The result covers this policy, not the full 16-cell gate.
    """
    for pin in (freeze_sha256, identity_sha256, attempt_sha256):
        _need(paired._pin(pin), "independent freeze/identity/attempt pins required")
    _need(not Path(campaign_root).is_symlink(), "symlink campaign root")
    c, root = campaign_api._root(campaign, campaign_sha256, source_plan_pins, campaign_root)
    _need(c["scope"] == "artificial" and len(c["sources"]) == 2 and len(c["policy_ids"]) == 8
          and len(c["services"]) == 3 and policy_id in c["services"]
          and c["policy_phases"][policy_id] == "gpu", "two-world/eight-arm/three-service artificial gate required")
    settings = c["services"][policy_id]
    identity = settings["identity"]
    _need(digest(identity) == identity_sha256
          and all(s["max_requests"] <= MAX_REQUESTS for s in c["services"].values()),
          "independent identity or campaign request limit differs")
    response_limit = len(c["policy_ids"]) * sum(len(s["plan"]["initial_actions"])
        + s["plan"]["acquisition_cap_units"] // min(a["cost_units"] for a in s["plan"]["menu"])
        for s in c["sources"])
    cost_limit = len(c["policy_ids"]) * sum(s["plan"]["initial_cost_units"]
        + s["plan"]["acquisition_cap_units"] for s in c["sources"])
    _need(response_limit <= 96 and cost_limit <= 96, "campaign response/cost limits exceed artificial gate")
    path = campaign_api._attempt(root, c, policy_id)
    worker = Path(worker_output_dir).absolute()
    attempt_inventory, worker_inventory = _inventory(path), _inventory(worker)
    if worker_evidence_sha256 is not None:
        _same(worker_inventory, worker_evidence_sha256, "independent worker inventory differs")
    attempt, closure = campaign_api._load_attempt(c, campaign_sha256, path, policy_id, attempt_sha256)
    _need(closure is not None or not worker_inventory, "unstarted service has worker evidence")
    client = path / "service"
    cells, contexts = [], []
    for arm, source in zip(attempt["arms"], c["sources"]):
        cell, context = _arm(path, arm, source, policy_id)
        cells.append(cell)
        contexts.append(context)
    failure_files = sorted(n for n in attempt_inventory if Path(n).name in ("failure.json", "failure-intent.json"))
    worker_failures = sorted(n for n in worker_inventory if "failure" in Path(n).name)
    issues = [{"world_id": a["world_id"], "status": a["status"], "reason": a["reason"]}
              for a in cells if a["status"] != "completed"]
    issues.extend({"attempt_error": e} for e in attempt["errors"])
    issues.extend({"failure_file": "attempt/" + n} for n in failure_files)
    issues.extend({"failure_file": "worker/" + n} for n in worker_failures)
    closed = campaign_api._decode(closure["raw"]) if closure else None
    cleanup = closed.get("cleanup") if closed else None
    clean_close = bool(closed and closed.get("status") == "closed" and cleanup
                       and cleanup["leader_returncode"] == 0 and cleanup.get("term_sent") is False
                       and cleanup.get("kill_sent") is False)
    if not clean_close:
        issues.append({"service": "not_started" if closure is None else "failure_or_forced_close"})
    count = closed["completed_requests"] if closed else 0
    _need(integer(count) and count <= settings["max_requests"], "completed request count exceeds limit")
    last = closed.get("last_request_id", closed.get("last_completed_request_id")) if closed else -1
    _need(type(last) is int and last == count-1, "session completed ID/count differs")
    families = {kind: _members(client, "request-*."+suffix, r"request-(\d{4})\."+re.escape(suffix))
                for kind, suffix in (("episode", "episode.json"), ("input", "input.json"),
                                     ("wire", "observed-response.json"), ("accepted", "accepted.json"))}
    _prefix(families["accepted"], count, "accepted request")
    pending = len(families["episode"])-count
    _need(pending in (0, 1) and (not pending or closed and closed["status"] == "failed"),
          "unexpected pending episode request")
    _prefix(families["episode"], count+pending, "episode request")
    for kind in ("input", "wire"):
        _need(set(range(count)) <= set(families[kind]) <= set(families["episode"]), kind+" membership differs")
    _need(set(families["wire"]) <= set(families["input"]), "response without input")
    members = {kind: _members(worker, pattern, expression) for kind, pattern, expression in (
        ("request", "request-*.json", r"request-(\d{4})\.json"),
        ("generation", "generation-*.json", r"generation-(\d{4})\.json"),
        ("generated", "generated-*", r"generated-(\d{4})\.unaccepted\.json"),
        ("started", "raw-generation-*.started.json", r"raw-generation-(\d{4})\.started\.json"),
        ("raw", "raw-generation-????.json", r"raw-generation-(\d{4})\.json"),
        ("binary", "raw-generation-*.bin", r"raw-generation-(\d{4})\.bin"))}
    for kind in members:
        _need(set(range(count)) <= set(members[kind]) <= set(families["input"]), kind+" membership differs")
        _prefix(members[kind], len(members[kind]), kind)
    known_worker = {p.name for family in members.values() for p in family.values()}
    _need(all(Path(n).parent == Path(".") and n in known_worker for n in worker_inventory
              if Path(n).name.startswith(("request-", "generation-", "generated-", "raw-generation-"))),
          "unrecognized or nested generation member")
    known_client = {p.name for family in families.values() for p in family.values()}
    _need(all(p.name in known_client for p in client.glob("request-*")), "unrecognized client request member")
    for child, parent in (("generation", "generated"), ("generated", "raw"),
                          ("raw", "started"), ("binary", "raw"), ("started", "request")):
        _need(set(members[child]) <= set(members[parent]), "orphan worker " + child)
    for ordinal, raw_path in members["raw"].items():
        raw = _read(raw_path)
        # Invalid returned tensors are legitimate failure evidence, but any
        # retained binary still has to match its recorded byte hash.
        if ordinal in members["binary"]:
            _need(paired._sha(members["binary"][ordinal].read_bytes()) == raw.get("raw_sha256"),
                  "raw failure binary hash differs")
        else:
            _need("raw_sha256" not in raw, "raw binary missing")
    if not issues:
        _need(all(len(v) == count for v in members.values()), "successful generation membership differs")
    auth_file = worker / "authenticated.json"
    if auth_file.exists():
        auth = _read(auth_file)
        _need(auth["freeze_sha256"] == freeze_sha256, "authenticated freeze differs")
        _same(auth["identity"], identity, "authenticated identity differs")
    else:
        _need(not members["request"] and count == 0 and issues, "authenticated startup missing")
    if (client / "ready.json").exists():
        _same(_read(client / "ready.json"), {"status": "ready", "identity_sha256": identity_sha256}, "client ready differs")
        _same(_read(client / "observed-handshake.json"), {"type": "ready", "version": PROTOCOL_VERSION,
              "identity": identity, "identity_sha256": identity_sha256}, "handshake identity differs")
    else:
        _need(count == 0 and issues, "client ready missing")
    phases = [_read(p) for p in sorted(worker.glob("phase-*.json"))]
    names = [p["phase"] for p in phases]
    loads = ("model_cpu_load_start", "model_cpu_load_complete", "model_ready")
    _need(all(names.count(n) <= 1 for n in loads), "model reloaded")
    if members["request"] or not issues:
        _need(all(names.count(n) == 1 for n in loads)
              and [names.index(n) for n in loads] == sorted(names.index(n) for n in loads), "one ordered model load required")
        ready = phases[names.index("model_ready")]
        _same(ready["identity"], identity, "loaded identity differs")
        _need(integer(ready["gpu_allocated_bytes"], 1), "model load GPU memory missing")
    terminal = _read(worker / "terminal.json") if (worker / "terminal.json").exists() else None
    if terminal:
        _need(terminal["status"] == "complete" and terminal["freeze_sha256"] == freeze_sha256
              and type(terminal["generation_count"]) is int
              and terminal["generation_count"] == len(members["generation"])
              and type(terminal["new_training_updates"]) is int and terminal["new_training_updates"] == 0
              and type(terminal["environment_responses"]) is int and terminal["environment_responses"] == 0,
              "worker terminal differs")
    elif not issues:
        raise ValueError("complete worker terminal missing")
    if (worker / "failure.json").exists():
        failure = _read(worker / "failure.json")
        _need(type(failure["new_training_updates"]) is int and failure["new_training_updates"] == 0
              and type(failure["environment_responses"]) is int and failure["environment_responses"] == 0
              and failure["retry_allowed"] is False, "worker failure violates inference scope")
        for field, expected in (("accepted_generation_records", sorted(p.name for p in members["generation"].values())),
                                ("available_raw_generation_records", sorted(p.name for kind in ("raw", "started")
                                                                            for p in members[kind].values()))):
            _same(failure[field], expected, "worker failure inventory differs: " + field)
    bindings = _members(client, "task-binding-*.json", r"task-binding-(\d{4})\.json")
    bound = [i for i, ctx in enumerate(contexts) if ctx is not None]
    _need(set(bindings) <= set(range(len(bound))), "extra task binding")
    offset, request_rows, prior_task, original_root = 0, [], None, None
    for binding_index, cell_index in enumerate(bound):
        ctx, cell, source = contexts[cell_index], cells[cell_index], c["sources"][cell_index]
        session = ctx["receipt"]["session"]
        _need(session["kind"] == "persistent" and session["identity_sha256"] == identity_sha256
              and type(session["count_before"]) is int and session["count_before"] == offset
              and type(session["request_before"]) is int and session["request_before"] == offset-1
              and integer(session["count_after"]) and offset <= session["count_after"] <= count
              and type(session["request_after"]) is int and session["request_after"] == session["count_after"]-1
              and cleanup and session["pid"] == cleanup["pgid"], "arm session range/process differs")
        if binding_index in bindings:
            b = _read(bindings[binding_index])
            _same(b, {"binding_index": binding_index, "task": source["plan"]["task"],
                     "task_sha256": cell["task_sha256"], "previous_task_sha256": prior_task,
                     "last_completed_session_request_id": offset-1, "worker_pid": session["pid"],
                     "model_reloaded": False}, "task rebinding differs")
        else:
            _need(ctx["unresolved"] and cell["status"] == "failed", "task binding missing")
        prior_task = cell["task_sha256"]
        end = session["count_after"]
        if ctx["unresolved"]:
            issues.append({"world_id": cell["world_id"], "join": "unresolved original episode"})
        else:
            _need(end-offset == len(ctx["proposals"]), "episode decisions/session range differs")
        ordinals = list(range(offset, end))
        if pending and end == count and cell_index == bound[-1]:
            ordinals.append(count)
        for decision_index, ordinal in enumerate(ordinals):
            mapping = _read(families["episode"][ordinal])
            episode_path = Path(mapping["episode_directory"])
            suffix = (path.name, f"world-{cell_index:02d}", "episode")
            _need(episode_path.is_absolute() and ".." not in episode_path.parts
                  and episode_path.parts[-3:] == suffix, "episode directory/world join differs")
            recorded_root = str(episode_path.parent.parent.parent)
            _need(original_root is None or original_root == recorded_root, "episode campaign roots differ")
            original_root = recorded_root
            _same(mapping, {"episode_directory": str(episode_path), "decision_index": decision_index,
                            "session_request_id": ordinal, "task_sha256": cell["task_sha256"],
                            "task_binding_index": binding_index, "identity_sha256": identity_sha256,
                            "status": "planned_before_transmission", "scope": "client_only_episode_to_session_mapping"},
                  "episode/session decision mapping differs")
            row = {"world_id": cell["world_id"], "task_sha256": cell["task_sha256"],
                   "decision_index": decision_index, "session_request_id": ordinal, "ordinal": ordinal,
                   "accepted": ordinal < count, "proposal_id": None, "response_request_id": None,
                   "charged_units": 0, "outcome": "failed_before_acceptance"}
            if ordinal in families["input"]:
                request = _read(families["input"][ordinal])
                envelope = request["envelope"]
                _same(request, {"type": "request", "request_id": ordinal, "identity_sha256": identity_sha256,
                                "envelope": envelope, "envelope_sha256": digest(envelope)}, "client request join differs")
                _same(envelope["task"], source["plan"]["task"], "public task differs from world plan")
                state = envelope["state"]
                task_prompt(envelope)  # public allowlist and arithmetic validation only
                row["envelope_sha256"] = digest(envelope)
                if not ctx["unresolved"]:
                    history = ctx["histories"][decision_index] if ordinal < end else ctx["chain"]["receipts"]
                    expected_state = {**state, "terminal": True}
                    prefix_chain = {"receipts": history, "attempts": history,
                                    "spent": sum(r["cost_units"] for r in history)}
                    _need(state["terminal"] is False, "request to terminal episode")
                    seal._state(expected_state, ctx["plan"], ctx["graph"], ctx["learner"], prefix_chain, [len(history)])
                if ordinal in members["request"]:
                    _same(_read(members["request"][ordinal]), {"ordinal": ordinal,
                          "envelope_sha256": digest(envelope), "identity": identity,
                          "tokens_available_at_request_start": False}, "worker request join differs")
                if ordinal in members["generation"]:
                    generation = _generation(ordinal, envelope, identity, members)
                    row["cuda_interval_ms"] = generation["cuda_interval_ms"]
                if ordinal < count:
                    wire, accepted = _read(families["wire"][ordinal]), _read(families["accepted"][ordinal])
                    _same(wire, {"type": "completion", "request_id": ordinal, "identity_sha256": identity_sha256,
                                 "envelope_sha256": digest(envelope), "completion": generation["completion"]},
                          "wire/generation completion differs")
                    _same(accepted, {"request_id": ordinal, "response_sha256": digest(wire),
                                     "completion": generation["completion"]}, "accepted completion differs")
                    if not ctx["unresolved"]:
                        proposal = ctx["proposals"][decision_index]
                        _need(proposal["completion"] == generation["completion"]
                              and proposal["provenance"]["public_state_sha256"] == digest(state), "executor proposal join differs")
                        row["proposal_id"] = proposal["proposal_id"]
                        reserve = next((e for e in ctx["events"] if e["event"] == "reserve"
                                        and e["proposal_id"] == proposal["proposal_id"]), None)
                        if reserve:
                            rid = reserve["request_id"]
                            row.update(response_request_id=rid, charged_units=reserve["cost_units"],
                                       outcome="response" if rid < len(ctx["chain"]["receipts"]) else "response_failure")
                        else:
                            _need(parse_completion(generation["completion"], unpack_envelope(envelope)[3]).stop,
                                  "accepted action has no charged reservation")
                            row["outcome"] = "policy_stop"
                    else:
                        row["outcome"] = "unresolved_executor_join"
            request_rows.append(row)
            cell["requests"].append(ordinal)
        offset = end
    _need(offset == count and len(request_rows) == count+pending, "unjoined session requests")
    complete_phases = [p for p in phases if p["phase"] == "generation_complete"]
    _need(len({p["ordinal"] for p in complete_phases}) == len(complete_phases), "duplicate generation phase")
    _need(set(range(count)) <= {p["ordinal"] for p in complete_phases} <= set(members["generation"]),
          "missing or orphan generation phase")
    for p in complete_phases:
        _need(p["envelope_sha256"] == _read(members["generation"][p["ordinal"]])["envelope_sha256"],
              "generation phase envelope differs")
    if not issues:
        _need({p["ordinal"] for p in complete_phases} == set(range(count)), "generation phase membership differs")
    # Recheck both trees: no mutation or file adoption during validation.
    _same(_inventory(path), attempt_inventory, "attempt changed during validation")
    _same(_inventory(worker), worker_inventory, "worker changed during validation")
    totals = {key: sum(cell[key] or 0 for cell in cells) for key in
              ("response_attempts", "returned_responses", "charged_units", "acquisition_charged_units")}
    feedback = sum(r["decision_index"] > 0 and "envelope_sha256" in r
                   and r["outcome"] != "unresolved_executor_join" for r in request_rows)
    return {"version": VERSION, "policy_id": policy_id, "campaign_sha256": campaign_sha256,
            "attempt_sha256": attempt_sha256, "freeze_sha256": freeze_sha256, "identity_sha256": identity_sha256,
            "status": "operational_success" if not issues else "evidence_only_failure_closure",
            "operational_success": not issues, "closure_confirmed": True, "clean_close": clean_close,
            "issues": issues, "cells": cells, "joins": request_rows,
            "counts": {"accepted_requests": count, "attempted_requests": count+pending,
                       "generations": len(members["generation"]), "model_loads": names.count("model_ready"),
                       "task_bindings": len(bindings), "policy_stops": sum(r["outcome"] == "policy_stop" for r in request_rows),
                       "budget_stops": sum(c["stop_reason"] == "budget_exhausted" for c in cells),
                       "failed_cells": sum(c["status"] == "failed" for c in cells),
                       "unstarted_cells": sum(c["status"] == "unstarted" for c in cells),
                       "feedback_to_next_decision": feedback, **totals},
            "counts_complete": all(c["charged_units"] is not None for c in cells),
            "feedback_path_exercised": feedback > 0,
            "cuda_interval_ms": sum(r.get("cuda_interval_ms", 0) for r in request_rows),
            "limits": {"worlds": 2, "campaign_cells": 16, "service_requests": settings["max_requests"],
                       "all_service_decisions": sum(s["max_requests"] for s in c["services"].values()),
                       "campaign_response_attempts": response_limit, "campaign_cost_units": cost_limit},
            "worker_evidence_sha256": worker_inventory,
            "worker_inventory_independently_pinned": worker_evidence_sha256 is not None,
            "recorded_campaign_root": original_root, "token_decoding_replayed": False,
            "scheduler_gpu_clearance_qualified": False, "training_acceptance_qualified": False,
            "shared_initial_across_policies_qualified": False, "scientific_acceptance": False,
            "validation_cost_scope": {"new_environment_responses": 0, "new_training_updates": 0,
                                      "model_loads": 0, "gpu_allocations": 0}}
