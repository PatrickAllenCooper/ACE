"""Read-only loader: builds one pickle of all 160 saved supervision records.

Reads only JSON files under the saved training collection. Loads no model weights.
"""
import json
import pickle
import sys
from pathlib import Path

COLL = Path("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_collection_20261108/"
            "terminal02/remote/launch/collection/session/worker/task/collection")
OUT = Path(sys.argv[1])

proposal = json.loads((COLL / "supplied-proposal.json").read_text())
inventory = json.loads((COLL / "inventory.json").read_text())
plan = json.loads((COLL / "plan.json").read_text())
roster = inventory["record_roster"]
world_roster = plan["world_roster"]
meta = {w["world_id"]: w for w in proposal["training_worlds"]}

records, prefs = [], []
for wi, wid in enumerate(world_roster):
    wdir = COLL / f"world-{wi:03d}"
    wplan = json.loads((wdir / "plan.json").read_text())
    wterm = json.loads((wdir / "terminal.json").read_text())
    private = json.loads((wdir / "private-world.json").read_text())
    assert wplan["world_id"] == wid, (wplan["world_id"], wid)
    m = meta[wid]
    menu_costs = {a["action_id"]: a["cost_units"] for a in wplan["menu"]}
    parents = {n["name"]: tuple(n["parents"]) for n in wplan["graph"]}
    graph_kind = {((), ("X",), ("Y",)): "chain", ((), ("X",), ("X",)): "fork",
                  ((), (), ("X", "Y")): "collider"}[(parents["X"], parents["Y"], parents["Z"])]
    assert graph_kind == m["graph_stratum"]
    cost_kind = "heterogeneous" if max(menu_costs.values()) == 2 else "uniform"
    assert cost_kind == m["cost_stratum"]
    for si in range(5):
        sdir = wdir / f"snapshot-{si:03d}"
        record = json.loads((sdir / "record.json").read_text())
        teacher = json.loads((sdir / "teacher.json").read_text())
        selection = json.loads((sdir / "selection.json").read_text())
        receipt = json.loads((sdir / "receipt.json").read_text())
        alignment = json.loads((sdir / "alignment.json").read_text())
        language = json.loads((sdir / "language.json").read_text())
        rid = f"policy_train/{wid}/snapshot-{si:03d}"
        assert rid == roster[wi * 5 + si]
        state = record["envelope"]["state"]
        fb = None
        if (sdir / "feedback" / "terminal.json").exists():
            fb = json.loads((sdir / "feedback" / "terminal.json").read_text())["feedback"]
        rec = {
            "record_id": rid, "world_index": wi, "world_id": wid, "snapshot": si,
            "regime": m["regime"], "graph": graph_kind, "cost_menu": cost_kind,
            "noise_variance": wplan["initial_learner"]["initial_heads"]["X"]["noise_variance"],
            "behavior_method": wplan["behavior_method"],
            "teacher_method": teacher["method"],
            "teacher_decision": teacher["decision"],
            "teacher_metadata": teacher["metadata"],
            "supervised_action_id": record["supervised_action_id"],
            "supervised_completion": language["supervised_completion"],
            "prompt_chars": len(language["prompt"]),
            "candidates": selection["candidates"],
            "milestone": selection["acquisition_cost_units"],
            "actual_acquisition_cost_units": selection["actual_acquisition_cost_units"],
            "trajectory_request_count": selection["trajectory_request_count"],
            "remaining_units": state["remaining_units"], "spent_units": state["spent_units"],
            "terminal": state["terminal"],
            "history_actions": [h["action_id"] for h in state["history"]],
            "scm": state["scm"],
            "menu_costs": menu_costs,
            "alignment_action_ids": alignment["action_ids"],
            "strict_preferences": alignment["strict_preferences"],
            "n_preferences": len(record["preferences"]),
            "structural_scores": receipt["structural_scores"],
            "private_mechanisms": private["mechanisms"],
            "feedback": None,
        }
        if fb is not None:
            rec["feedback"] = {k: fb[k] for k in (
                "candidate_ids", "candidate_costs", "replicate_ids", "replicate_count",
                "base_predictions", "scoring_truths", "base_losses", "branch_losses",
                "per_replicate_gains_per_cost", "mean_gains_per_cost", "tie_multiplier",
                "minimum_gain_gap", "normalized_loss_reductions", "query_weights", "scale")}
            assert len(fb["pairs"]) == len(record["preferences"]) == 28
            for pi, (fp, rp) in enumerate(zip(fb["pairs"], record["preferences"])):
                assert fp["left"] == rp["left"] and fp["right"] == rp["right"]
                assert fp["tie"] == rp["tie"] and fp["chosen"] == rp["chosen"] and fp["rejected"] == rp["rejected"]
                prefs.append({
                    "record_id": rid, "world_index": wi, "world_id": wid, "snapshot": si,
                    "regime": m["regime"], "graph": graph_kind, "cost_menu": cost_kind,
                    "noise_variance": rec["noise_variance"],
                    "behavior_method": rec["behavior_method"], "teacher_method": rec["teacher_method"],
                    "preference_index": pi, "left": fp["left"], "right": fp["right"],
                    "tie": fp["tie"], "chosen": fp["chosen"], "rejected": fp["rejected"],
                    "mean_gain_difference": fp["mean_gain_difference"],
                    "paired_standard_error": fp["paired_standard_error"],
                    "tie_threshold": fp["tie_threshold"],
                    "replicate_differences": fp["replicate_differences"],
                    "left_cost": menu_costs[fp["left"]], "right_cost": menu_costs[fp["right"]],
                })
        else:
            assert record["preferences"] == []
        records.append(rec)

assert len(records) == 160, len(records)
print("records", len(records), "preference rows", len(prefs))
with OUT.open("wb") as fh:
    pickle.dump({"records": records, "prefs": prefs, "roster": roster}, fh)
print("wrote", OUT)
