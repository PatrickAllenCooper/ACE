"""Trusted episode driver and charged, training-only preference construction.

Policies run as a separate process receiving detached public JSON on stdin.
They receive no environment callback, true coefficients or scoring outcomes.
This separation is not an OS sandbox against malicious same-user programs.
No scientific world generation, pretrained model or evaluation set lives here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from llm_intervention_contract import (
    Action, ChargedExecutor, canonical, digest, finite, integer, policy_prompt,
    validate_design,
)


def save(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")


def text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


class ProcessPolicy:
    def __init__(self, command: tuple[str, ...], timeout_seconds: float):
        if (type(command) is not tuple or not command
                or any(not isinstance(x, str) or not x for x in command)
                or not finite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("fixed command and positive timeout required")
        self.command, self.timeout = command, timeout_seconds

    def choose(self, public_state: dict, directory: Path, decision_index: int) -> str:
        prefix = f"decision-{decision_index:04d}"
        payload = canonical(public_state)
        # Persist exactly the public payload before starting the child.
        save(directory / (prefix + ".input.json"), public_state)
        started = time.monotonic()
        try:
            result = subprocess.run(self.command, input=payload, text=True,
                                    capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            save(directory / (prefix + ".process.json"), {
                "command": self.command, "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "status": "timeout", "stdout": text(exc.stdout), "stderr": text(exc.stderr),
                "elapsed_seconds": time.monotonic() - started})
            raise
        save(directory / (prefix + ".process.json"), {
            "command": self.command, "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "elapsed_seconds": time.monotonic() - started})
        if result.returncode:
            raise RuntimeError("policy process failed, preserved without fallback")
        if len(result.stdout) > 4096:
            raise ValueError("policy completion exceeds action limit")
        return result.stdout


def run_episode(*, graph, menu, budget, initial_actions, learner, respond,
                policy: ProcessPolicy, objective: str, output: Path, domain_notes=""):
    """Charge shared initialization and each chosen intervention in one journal.

    The policy's stop decision is recorded, never justified using private risk.
    The result has no scientific evaluation score. An independent evaluator may
    later read the sealed history under its own declared response accounting.
    """
    validate_design(graph, menu)
    if learner.graph != graph or not isinstance(policy, ProcessPolicy):
        raise ValueError("matching learner graph and process policy required")
    actions = {a.action_id: a for a in menu}
    if (type(initial_actions) is not tuple or not set(initial_actions) <= set(actions)
            or not integer(budget, 1)
            or sum(actions[x].cost_units for x in initial_actions) > budget):
        raise ValueError("invalid shared initial history schedule")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", {"graph": [asdict(n) for n in graph],
        "menu": [asdict(a) for a in menu], "budget": budget, "initial_actions": initial_actions,
        "objective": objective, "domain_notes": domain_notes, "policy_command": policy.command,
        "policy_timeout_seconds": policy.timeout})
    initialized = 0
    decisions = 0
    executor = None
    try:
        with ChargedExecutor(graph, menu, budget, output / "journal.jsonl", respond) as executor:
            for action_id in initial_actions:
                receipt = executor.execute_completion(canonical({"action_id": action_id}),
                                                      provenance={"kind": "shared_initial_schedule"})
                learner.update(receipt)
                initialized += 1
            while True:
                state = executor.state(objective, learner.summaries(), domain_notes)
                if state["terminal"]:
                    break
                if not any(a.cost_units <= state["remaining_units"] for a in menu):
                    executor.execute_completion('{"stop":true}', provenance={"kind": "budget_cap"})
                    break
                # Validate the complete public schema before exporting it.
                policy_prompt(state)
                completion = policy.choose(state, output, decisions)
                decisions += 1
                receipt = executor.execute_completion(completion, provenance={"kind": "process_policy",
                    "decision_index": decisions-1, "public_state_sha256": digest(state)})
                if receipt is not None:
                    learner.update(receipt)
            final = executor.state(objective, learner.summaries(), domain_notes)
        terminal = {"status": "complete", "initialized_responses": initialized,
                    "policy_decisions": decisions, "final_state": final,
                    "scientific_evaluation_performed": False}
        save(output / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        state = executor.state(objective, learner.summaries(), domain_notes) if executor else None
        save(output / "failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "initialized_responses": initialized,
            "completed_policy_calls": decisions, "partial_state": state, "retry_allowed": False})
        raise


def build_training_preferences(*, scope, graph, menu, candidates, learner,
        next_learner_request_id, branch_respond, scoring_menu, scoring_actions,
        score_respond, target, scale, tie_multiplier, minimum_gain_gap, output: Path):
    """One-step preferences from paid branch/scoring responses on training worlds.

    All branches fork the same pre-action learner. A fixed, independently drawn
    scoring set is paid once and shared across the branches. Score labels never
    update a branch. Gains are divided by each intervention's declared cost.
    Pairwise ties use a fixed standard-error heuristic, not a confidence claim.
    Callbacks own separate stream namespaces. This helper cannot prove their
    randomness or world separation, which the future frozen runner must bind.
    """
    if scope != "policy_train":
        raise ValueError("measured preference supervision is training-only")
    validate_design(graph, menu)
    validate_design(graph, scoring_menu)
    by_id = {a.action_id: a for a in menu}
    score_by_id = {a.action_id: a for a in scoring_menu}
    if (learner.graph != graph or type(candidates) is not tuple or len(candidates) < 2
            or len(set(candidates)) != len(candidates) or not set(candidates) <= set(by_id)
            or type(scoring_actions) is not tuple or len(scoring_actions) < 2
            or not set(scoring_actions) <= set(score_by_id)
            or any(target not in score_by_id[x].measure for x in scoring_actions)
            or not integer(next_learner_request_id) or not finite(scale) or scale <= 0
            or not finite(tie_multiplier) or tie_multiplier < 0
            or not finite(minimum_gain_gap) or minimum_gain_gap < 0):
        raise ValueError("invalid fixed preference design")
    output.mkdir(exist_ok=False)
    base_summaries = canonical([asdict(s) for s in learner.summaries()])
    plan = {"scope": scope, "candidates": candidates, "scoring_actions": scoring_actions,
        "graph": [asdict(n) for n in graph], "menu": [asdict(a) for a in menu],
        "scoring_menu": [asdict(a) for a in scoring_menu], "target": target, "scale": scale,
        "tie_multiplier": tie_multiplier, "minimum_gain_gap": minimum_gain_gap,
        "next_learner_request_id": next_learner_request_id,
        "base_summaries_sha256": hashlib.sha256(base_summaries.encode()).hexdigest(),
        "expected_branch_cost": sum(by_id[x].cost_units for x in candidates),
        "expected_scoring_cost": sum(score_by_id[x].cost_units for x in scoring_actions)}
    save(output / "plan.json", plan)
    journals = []
    try:
        branches = {}
        for index, action_id in enumerate(candidates):
            branch = learner.fork()
            filename = f"branch-{index:03d}.jsonl"
            journals.append(filename)
            with ChargedExecutor(graph, menu, by_id[action_id].cost_units, output / filename,
                    lambda action, request, key=action_id: branch_respond(key, action, request)) as exe:
                receipt = exe.execute_completion(canonical({"action_id": action_id}))
                # Journal-local IDs and learner-local IDs are distinct namespaces.
                # Preserve the explicit mapping. No old row is silently replaced.
                mapped = dict(receipt, request_id=next_learner_request_id)
                branch.update(mapped)
                save(output / f"branch-{index:03d}.mapping.json", {
                    "journal": filename, "source_request_id": receipt["request_id"],
                    "learner_request_id": next_learner_request_id, "action_id": action_id,
                    "post_summaries": [asdict(s) for s in branch.summaries()]})
            branches[action_id] = branch
        scoring = []
        journals.append("scoring.jsonl")
        with ChargedExecutor(graph, scoring_menu, plan["expected_scoring_cost"],
                             output / "scoring.jsonl", score_respond) as exe:
            for action_id in scoring_actions:
                scoring.append(exe.execute_completion(canonical({"action_id": action_id})))
        truth = np.array([r["values"][target] for r in scoring])
        eval_actions = [score_by_id[x] for x in scoring_actions]
        base_predictions = np.array([learner.predict_mean(a)[target] for a in eval_actions])
        base_losses = ((base_predictions-truth) / scale)**2
        predictions, losses, gains = {}, {}, {}
        for key in candidates:
            pred = np.array([branches[key].predict_mean(a)[target] for a in eval_actions])
            loss = ((pred-truth) / scale)**2
            gain = (base_losses-loss) / by_id[key].cost_units
            if not np.isfinite(loss).all() or not np.isfinite(gain).all():
                raise ValueError("nonfinite training feedback")
            predictions[key], losses[key], gains[key] = pred.tolist(), loss.tolist(), gain
        pairs = []
        for left, right in itertools.combinations(candidates, 2):
            differences = gains[left] - gains[right]
            mean = float(differences.mean())
            se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
            tie = abs(mean) <= minimum_gain_gap + tie_multiplier * se
            pairs.append({"left": left, "right": right, "mean_gain_difference": mean,
                "paired_standard_error": se, "tie": tie,
                "chosen": None if tie else (left if mean > 0 else right),
                "rejected": None if tie else (right if mean > 0 else left)})
        if canonical([asdict(s) for s in learner.summaries()]) != base_summaries:
            raise RuntimeError("base learner changed during isolated branches")
        result = {"status": "complete", "plan": plan, "journal_files": journals,
            "scoring_values": truth.tolist(), "base_predictions": base_predictions.tolist(),
            "base_losses": base_losses.tolist(), "predictions": predictions, "losses": losses,
            "gains_per_cost": {k: x.tolist() for k, x in gains.items()}, "pairs": pairs,
            "charged_branch_cost": plan["expected_branch_cost"],
            "charged_scoring_cost": plan["expected_scoring_cost"],
            "environment_responses": len(candidates)+len(scoring_actions),
            "base_learner_unchanged": True, "private_evaluation_used": False}
        save(output / "terminal.json", result)
        return result
    except BaseException as exc:
        save(output / "failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "message": str(exc), "journal_files": journals, "retry_allowed": False,
            "cost_instruction": "reconstruct reservations and returns from all retained journals"})
        raise
