"""Artificial full-size screen inputs; no draws, responses or tokenization.

Build in an existing NumPy-capable CPU environment (no Torch/model imports):
  python -B scripts/research/llm_intervention_screen_prompt_fixture.py \
      --output /ABS/NEW/fixture-directory

Only public_design exports enter the envelopes. Histories and posterior numbers
are hand-constructed schema fixtures, NOT fitted posteriors or scientific data.
The output manifest supplies exact file inputs for a future authenticated CPU
worker calling llm_intervention_language_policy.prepare_task_tokens(envelope,
tokenizer, max_tokens=8192, max_new_tokens=32). That existing helper is also used
by training_engine.prepare_record and training_action_batch. Do not use a frozen
old campaign/preflight packet: its fixture roster is different. This builder
neither imports the tokenization helper (which imports Torch) nor launches work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from llm_intervention_contract import VERSION as STATE_VERSION, canonical, digest
from llm_intervention_screen_protocol import (
    COSTS, GRAPHS, REQUIRED_SOURCE_PATHS, build_proposal, public_design,
)
from llm_intervention_task import PredictionTask, action_from_dict, make_envelope, task_prompt


VERSION = "ace-screen-artificial-prompt-fixtures-v1"
# These exercise long round-trip spellings, NOT an exhaustive float/token bound.
LONG_VALUE = -1.2345678901234567e-100
FINITE_VALUES = (
    -sys.float_info.max, sys.float_info.max,
    -sys.float_info.min, sys.float_info.min,
    -float.fromhex("0x0.0000000000001p-1022"),
    float.fromhex("0x0.0000000000001p-1022"),
    -0.00012345678901234567, -1234567890123456.8,
    LONG_VALUE, 1.2345678901234567e+100, -0.0, 0.0,
)
PROFILES = ("long_spelling", "finite_edges")
SHAPES = ("initial", "last_training_milestone", "last_all_actions",
          "full_history_terminal", "mixed_actions")
LIMITATIONS = (
    "Artificial receipt values and posterior summaries are hand-constructed; no posterior fitting, "
    "world realization, environment response, teacher label or evaluation has occurred.",
    "Public validation checks dimensions, finite values, positive-semidefinite covariance and "
    "history accounting; it does not certify posterior consistency with these histories.",
    "The prospective Gaussian law has no finite scientific magnitude cap. Intervention bounds "
    "[-2,2] constrain clamps, not free observations or posterior means. Binary64 extremes are "
    "serialization probes, not plausible predictions or stable learner-update inputs.",
    "Long spellings are longest only within the declared finite value corpus. This finite fixture "
    "does not cover all real values, all accepted JSON numbers, all token segmentations, or all "
    "histories. Generic APIs also accept text/counters outside this proposal's fixed design.",
    "A valid 32-row history exhausts budget 32; all eight actions plus stop are instead tested "
    "at 31 rows for uniform cost and 30 rows for heterogeneous cost. Mixed histories exercise "
    "all action/clamp forms while respecting the eight initial observations and cost budget.",
    "Byte/character counts are not token counts or resource predictions. Every future real "
    "prompt still requires untruncated validation; fixture success is not universal admission.",
    "No context fit, training memory/throughput, scientific adoption or 32-train/24-dev "
    "generation is authorized or certified by this artifact.",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _schedule(public, shape):
    menu = public["menu"]
    maximum_cost = max(a["cost_units"] for a in menu)
    counts = {"initial": 8, "last_training_milestone": 26,
              "last_all_actions": 32 - maximum_cost, "full_history_terminal": 32}
    if shape in counts:
        return [menu[0]] * counts[shape]
    # Every menu action appears after the mandated initial observations. Fill
    # with the joint clamp to exercise its longer targets object, then observe
    # if needed to leave exactly two units (all actions remain affordable).
    actions = [menu[0]] * 8 + list(menu)
    spent = sum(a["cost_units"] for a in actions)
    while spent < 30:
        action = menu[-1] if spent + menu[-1]["cost_units"] <= 30 else menu[0]
        actions.append(action)
        spent += action["cost_units"]
    return actions


def _fixture(public, graph, cost, noise, profile, shape):
    actions = _schedule(public, shape)
    values = (LONG_VALUE,) if profile == "long_spelling" else FINITE_VALUES
    history, eligible = [], {n["name"]: 0 for n in public["graph"]}
    for i, action in enumerate(actions):
        targets = dict(action["targets"])
        observed = {name: values[(3*i+j) % len(values)] for j, name in enumerate(action["measure"])}
        observed.update(targets)
        history.append({"request_id": i, "action_id": action["action_id"],
                        "kind": action["kind"], "targets": targets,
                        "values": observed, "cost_units": action["cost_units"]})
        available = set(action["measure"]) | set(targets)
        for node in public["graph"]:
            name = node["name"]
            if name not in targets and name in action["measure"] and set(node["parents"]) <= available:
                eligible[name] += 1
    summaries = []
    for ni, node in enumerate(public["graph"]):
        width = 1 + len(node["parents"])
        # Dense, symmetric, strictly diagonally dominant SPD matrices with
        # long spellings. Avoid overflow/ill-conditioned eigensolver probes.
        covariance = [[0.9876543210987654 if i == j else -0.12345678901234567
                       for j in range(width)] for i in range(width)]
        summaries.append({"node": node["name"], "terms": ["intercept", *node["parents"]],
            "parameter_mean": [values[(3*ni+j) % len(values)] for j in range(width)],
            "parameter_covariance": covariance, "noise_variance": noise,
            "eligible_rows": eligible[node["name"]]})
    spent = sum(a["cost_units"] for a in actions)
    state = {"version": STATE_VERSION, "objective": public["objective"],
        "domain_notes": public["domain_notes"], "graph": public["graph"], "menu": public["menu"],
        "history": history, "scm": summaries, "spent_units": spent, "remaining_units": 32-spent,
        "requests": len(history), "returned_responses": len(history), "terminal": spent == 32}
    t = public["task"]
    task = PredictionTask(t["target"], tuple(action_from_dict(q) for q in t["queries"]),
                          tuple(t["weights"]), t["scale"], t["normalized_mse_tolerance"])
    envelope = make_envelope(json.loads(canonical(state)), task)
    prompt = task_prompt(envelope)
    commands = [canonical({"action_id": a["action_id"]})
                for a in sorted(public["menu"], key=lambda a: a["action_id"])
                if not state["terminal"] and a["cost_units"] <= state["remaining_units"]]
    commands.append(canonical({"stop": True}))
    return {"fixture_id": f"{graph}-{cost}-noise{noise:g}-{profile}-{shape}",
        "graph_shape": graph, "cost_menu": cost, "noise_variance": noise,
        "numeric_profile": profile, "history_shape": shape, "history_rows": len(history),
        "query_count": len(t["queries"]), "menu_count": len(public["menu"]),
        "spent_units": spent, "remaining_units": 32-spent, "legal_commands": commands,
        "envelope_sha256": digest(envelope), "envelope_utf8_bytes": len(canonical(envelope).encode()),
        "prompt_sha256": _sha(prompt.encode()), "prompt_utf8_bytes": len(prompt.encode()),
        "prompt_characters": len(prompt), "envelope": envelope, "prompt": prompt}


def build_fixtures(*, source_pins):
    """Pure API; NumPy is used only by the existing public-state validator.

    source_pins are declarations for the proposal API, not runtime admission.
    No world IDs or private proposal fields are exported in policy envelopes.
    """
    proposal = build_proposal(source_pins=source_pins)
    fixtures, seen = [], set()
    for row in proposal["development_worlds"]:
        graph, cost = row["graph_stratum"], row["cost_stratum"]
        # Deduplicate the shifted regime: its shift is private, and its public
        # text/design is identical. Both public noise variances remain covered.
        noise = row["public"]["learner"]["noise_variances"]["X"]
        if (graph, cost, noise) in seen:
            continue
        seen.add((graph, cost, noise))
        public = public_design(proposal, scope="development", world_id=row["world_id"],
                               expected_source_pins=source_pins)
        for profile in PROFILES:
            for shape in SHAPES:
                fixtures.append(_fixture(public, graph, cost, noise, profile, shape))
    corpus = sorted({canonical(v) for v in (*FINITE_VALUES, LONG_VALUE,
                                          0.9876543210987654, -0.12345678901234567)})
    longest = max(map(len, corpus))
    return {"version": VERSION, "purpose": "artificial full-size CPU capacity preflight inputs",
        "scientific_worlds_generated": 0, "environment_responses": 0, "training_updates": 0,
        "actual_tokenization_performed": False, "universal_token_bound_claimed": False,
        "scientific_execution_authorized": False, "limitations": list(LIMITATIONS),
        "declared_protocol_source_pins": dict(source_pins),
        "source_admission": "Declared/observed hashes only; authenticate future worker inputs and runtime separately.",
        "coverage": {"graph_shapes": list(GRAPHS), "cost_menus": list(COSTS),
            "noise_variances": [0.25, 1.0], "queries": 8, "menu_actions": 8,
            "maximum_history_rows": 32, "total_budget": 32, "initial_observations": 8,
            "profiles": list(PROFILES), "shapes": list(SHAPES), "fixture_count": len(fixtures)},
        "numeric_spellings": {"corpus": corpus, "longest_corpus_characters": longest,
            "longest_corpus_spellings": [s for s in corpus if len(s) == longest],
            "global_maximum_proven": False},
        "future_cpu_qualification": {
            "status": "required_not_run", "gpus": 0, "model_weights_required": False,
            "api": "llm_intervention_language_policy.prepare_task_tokens",
            "arguments": {"max_tokens": 8192, "max_new_tokens": 32},
            "chat_template": {"role": "user", "tokenize": True, "add_generation_prompt": True,
                              "enable_thinking": False},
            "required_inputs": ["Every manifest fixture envelope file and its independent byte pin",
                "Authenticated source/runtime including language_policy, task, grammar and training formatter",
                "Local pinned tokenizer files, tokenizer configuration/chat template and special token IDs",
                "Explicit exclusive CPU output path, CPU/memory/wall limits and deadline"],
            "required_evidence": ["Exact prompt IDs and every legal command+EOS token sequence",
                "Exact untruncated chat prefix length; command round trips and full 32-token reserve",
                "Match envelope/prompt hashes and complete legal-command roster to this manifest",
                "Record all fixtures, failures/overflows, tokenizer load and tokenization timestamps, CPU time and peak RSS"],
            "failure_rule": "Preserve failed inputs and attempts; never truncate, skip a failing fixture, "
                "change the study or context silently, or infer success from byte counts.",
            "scope": "Representative capacity only; real prompt checks and training resource qualification remain required."},
        "fixtures": fixtures}


def write_packet(output, packet):
    """Exclusive output; partial writes/failures are retained without retry."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for fixture in packet["fixtures"]:
        row = {k: v for k, v in fixture.items() if k not in ("envelope", "prompt")}
        for key, suffix, raw in (
            ("envelope_file", ".json", (canonical(fixture["envelope"])+"\n").encode()),
            ("prompt_file", ".prompt.txt", fixture["prompt"].encode()),
        ):
            name = fixture["fixture_id"]+suffix
            with (output/name).open("xb") as handle:
                handle.write(raw)
            row[key] = {"path": name, "sha256": _sha(raw), "utf8_bytes": len(raw)}
        rows.append(row)
    manifest = {**packet, "fixtures": rows}
    with (output/"manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical(manifest)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new exclusive CPU fixture directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    started = datetime.now(timezone.utc).isoformat()
    start, cpu_start = time.monotonic(), time.process_time()
    pins = {p: _sha((root/p).read_bytes()) for p in REQUIRED_SOURCE_PATHS}
    packet = build_fixtures(source_pins=pins)
    packet["builder_source_sha256"] = _sha(Path(__file__).read_bytes())
    packet["preparation"] = {"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic()-start, "cpu_seconds": time.process_time()-cpu_start,
        "python_version": sys.version, "model_loading": "not performed", "gpu_work": "not performed"}
    manifest = write_packet(args.output, packet)
    print(canonical({"output": str(args.output.resolve()), "fixtures": len(manifest["fixtures"]),
        "maximum_prompt_utf8_bytes": max(r["prompt_utf8_bytes"] for r in manifest["fixtures"]),
        "maximum_envelope_utf8_bytes": max(r["envelope_utf8_bytes"] for r in manifest["fixtures"]),
        "actual_tokenization_performed": False}))


if __name__ == "__main__":
    main()
