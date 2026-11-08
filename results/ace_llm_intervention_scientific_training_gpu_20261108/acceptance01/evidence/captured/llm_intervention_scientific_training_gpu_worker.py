"""Pinned complete scientific CPU plan -> one supplied-runtime GPU training run.

Source only: no CLI, scheduler, admission, tokenization, world access or retries.
The outer caller authenticates this captured closure, the complete accepted CPU
preparation/plan, seven base files and runtime BEFORE run_training; it also owns
allocation, deadlines, telemetry, cleanup and independent whole-run acceptance.
prepared_sha256 is SHA256(canonical(prepared)+LF); plan_sha256 is semantic.
Saved tokens/labels are never selected again or truncated. Output is a new/empty
child of journal.output. Top-level imports and validate_outputs use stdlib only;
heavy imports are confined to the authenticated run_training call.

GPU lifecycle and evidence checks are adapted from the accepted full-shape
worker. Its pure scoring/disposal helpers are reused unchanged, without globals
patching or edits. That four/two acceptance does not qualify this 160+D run.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from llm_intervention_screen_training_gpu_worker import (
    PARAMETERS, SCREEN_NUMERIC_RECIPE, _canonical, _digest, _sha, _need, _finite,
    _pin, _clear_cuda_after_disposal, _check_disposal_memory, _pair_specs,
    scoring_layout, _score_difference, _legal_sequence, _expected_evidence_files,
)

VERSION = 'ace-scientific-training-gpu-worker-v1'
WORLD_IDS = tuple(range(202611070000, 202611070032))
ROSTER = tuple(f'policy_train/{wid}/snapshot-{i:03d}' for wid in WORLD_IDS for i in range(5))
MAX_JSON_BYTES = 256 * 1024**2


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'duplicate evidence JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: '+value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    _canonical(value)
    return value


def _read(path):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON_BYTES,
          'bounded ordinary evidence file required')
    with path.open('rb') as f:
        raw = f.read(MAX_JSON_BYTES+1)
    _need(len(raw) <= MAX_JSON_BYTES, 'evidence exceeded capture bound')
    return _decode(raw)


def _prepared_lineage(prepared):
    names = ('collection_terminal_sha256','collection_inventory_sha256','collection_plan_sha256',
             'exposure_sha256','tokenizer_sha256')
    _need(prepared.get('version') == 'ace-complete-collection-training-plan-v1'
          and all(_pin(prepared.get(k)) for k in names)
          and prepared.get('new_environment_responses') == 0 and prepared.get('training_updates') == 0
          and prepared.get('outer_execution_qualified') is False and prepared.get('scientific_qualified') is False,
          'accepted complete CPU preparation lineage required')
    return {k:prepared[k] for k in names}


def _plan(prepared, *, prepared_sha256, plan_sha256):
    """Check saved pins/shape/exposure without importing the engine or selecting data.

    The independent accepted CPU pin is authoritative for matched-view validation
    and tokenization. The unchanged engine additionally revalidates its complete
    plan before fitting. This gate does not confer CPU or scientific acceptance.
    """
    _need(_pin(prepared_sha256) and _pin(plan_sha256), 'independent prepared and plan pins required')
    _need(type(prepared) is dict and hashlib.sha256((_canonical(prepared)+'\n').encode()).hexdigest()
          == prepared_sha256, 'independent prepared artifact pin differs')
    _prepared_lineage(prepared)
    plan, exposure = prepared['training_plan'], prepared['exposure']
    _need(prepared['training_plan_sha256'] == plan_sha256 == _digest(plan)
          and prepared['exposure_sha256'] == _digest(exposure), 'plan/exposure pin differs')
    d = exposure['counts']['dpo_steps']
    _need(type(d) is int and 1 <= d <= 128 and exposure['has_both_training_stages'] is True,
          'D must be 1..128; no D=0 rescue')
    shapes = {f'model.layers.{i}.self_attn.{part}.{side}':shape for i in range(28)
              for part,width in (('q_proj',2048),('v_proj',1024))
              for side,shape in (('A',[8,1024]),('B',[width,8]))}
    _need(plan['scope'] == exposure['scope'] == 'policy_train'
          and plan['execution_kind'] == 'qwen_screen_singletons'
          and plan['numeric_recipe'] == SCREEN_NUMERIC_RECIPE
          and plan['expected_adapter_shapes'] == shapes
          and plan['declared_adapter_parameters'] == plan['max_trainable_parameters'] == PARAMETERS
          and plan['checkpoint_steps'] == [159,159+d] and plan['one_pass_sft'] is True
          and plan['provenance_qualified'] is False and _pin(plan['outer_provenance_receipt_sha256'])
          and (plan['sft_steps'],plan['dpo_steps'],plan['sft_rows'],plan['dpo_rows']) == (160,d,160,2*d),
          'scientific singleton recipe/checkpoint contract differs')
    records = plan['records']
    _need(type(records) is list and [r['record_id'] for r in records] == list(ROSTER)
          and exposure['record_roster'] == list(ROSTER) and exposure['roster_sha256'] == _digest(ROSTER)
          and set(plan['record_pins']) == set(exposure['record_pins']) == set(exposure['supervision_receipt_pins']) == set(ROSTER)
          and [r['record_id'] for r in exposure['records']] == list(ROSTER), 'exact 32-world/160-record roster required')
    schedule = plan['schedule']
    _need(schedule == exposure['schedule'] and len(schedule) == 160+d
          and schedule[:160] == [{'stage':'sft','record_id':rid,'pair_index':None} for rid in ROSTER],
          'fixed SFT-first exposure differs')
    expected_exposures, selected, stored, strict_count, tied_count = {}, [], 0, 0, 0
    raw_bytes = 0
    for i,(rec,detail) in enumerate(zip(records,exposure['records'])):
        rid = rec['record_id']; raw = rec['raw_record_utf8'].encode(); raw_bytes += len(raw)
        _need(0 < len(raw) <= 1024**2 and raw_bytes <= 64*1024**2
              and _digest(rec) == plan['record_pins'][rid]
              and hashlib.sha256(raw).hexdigest() == rec['record_sha256'] == exposure['record_pins'][rid] == detail['record_sha256']
              and _pin(rec['supervision_receipt_sha256'])
              and rec['supervision_receipt_sha256'] == exposure['supervision_receipt_pins'][rid] == detail['supervision_receipt_sha256'],
              'raw/prepared record or receipt pin differs')
        original = _decode(raw); state = original['envelope']['state']; terminal = i%5 == 4
        _need(original['scope'] == rec['scope'] == 'policy_train'
              and original['world_id'] == detail['world_id'] == WORLD_IDS[i//5]
              and state['terminal'] is terminal and rec['provenance_qualified'] is False,
              'record world/scope/terminal association differs')
        strict, tied = rec['strict_pairs'], rec['tied_pairs']
        _need(len(original['preferences']) == len(strict)+len(tied) == (0 if terminal else 28),
              'all original 3584 pairs must remain present')
        strict_count += len(strict); tied_count += len(tied)
        prompt, commands, meta = rec['prompt_ids'], rec['commands'], rec['token_metadata']
        _need(type(prompt) is list and prompt and len(prompt)+32 <= 8192
              and all(type(t) is int and t>=0 for t in prompt)
              and meta['prompt_tokens'] == len(prompt) and meta['max_tokens'] == 8192 and meta['max_new_tokens'] == 32
              and rec['pad_token_id'] == 151643 and rec['eos_token_id'] == 151645
              and rec['tokenizer_sha256'] == prepared['tokenizer_sha256'], 'complete saved token metadata differs')
        stop = _canonical({'stop':True})
        legal = {stop} if terminal else {stop,*(_canonical({'action_id':a['action_id']}) for a in state['menu']
                                               if a['cost_units'] <= state['remaining_units'])}
        _need(set(commands) == legal and len({tuple(ids) for ids in commands.values()}) == len(commands)
              and rec['supervised_command'] in legal and (not terminal or rec['supervised_command'] == stop)
              and detail['supervised_command'] == rec['supervised_command'], 'full legal command/label inventory differs')
        for command, tokens in commands.items():
            _need(type(tokens) is list and all(type(t) is int and t>=0 for t in tokens), 'integer command tokens required')
            _legal_sequence(rec,prompt+tokens,command)
        _need(all(type(pair) is list and len(pair)==2 and pair[0]!=pair[1]
                  and all(c in commands for c in pair) for pair in strict), 'ordered strict command pairs differ')
        weights = detail['strict_pair_exposures']
        _need(type(weights) is list and len(weights)==len(strict)
              and all(type(n) is int and n in (0,1) for n in weights)
              and sum(weights) == int(bool(strict)) == detail['dpo_exposure_count']
              and detail['sft_exposure_count'] == 1, 'at most one fixed strict-pair exposure required')
        if strict:
            selected.append({'stage':'dpo','record_id':rid,'pair_index':weights.index(1)})
        expected_exposures[rid] = {'sft':1,'dpo_pairs':weights,'tied_pairs_retained':len(tied)}
        stored += len(prompt)+sum(map(len,commands.values()))
    _need(schedule[160:] == selected and len(selected)==d and plan['exposures']==expected_exposures
          and plan['one_pass_dpo'] is all(n==1 for v in expected_exposures.values() for n in v['dpo_pairs'])
          and plan['stored_tokens']==stored<=4194304, 'fixed pair schedule/exposure or token total differs')
    expected_counts = {'records':160,'retained_pairs':3584,'strict_pairs':strict_count,'selected_pairs':d,
        'unselected_pairs':strict_count-d,'tied_pairs':tied_count,'records_without_strict_pairs':160-d,
        'sft_steps':160,'dpo_steps':d,'sft_action_rows':160,'dpo_action_rows':2*d}
    _need(exposure['counts']==expected_counts, 'complete retained/selected counts differ')
    return _decode(_canonical(plan))


def _check_training(root, plan, *, plan_sha256):
    """Validate engine JSON plus checkpoint bytes; never deserialize tensors."""
    root = Path(root)
    d, total = plan["dpo_steps"], 160+plan["dpo_steps"]
    recompute_min = 28*(160+2*d)
    _need(not (root / "failure.json").exists(), "retained engine failure")
    _need(_read(root / "plan.json") == plan, "engine saved a different plan")
    terminal = _read(root / "terminal.json")
    _need(terminal["status"] == "complete" and terminal["plan_sha256"] == plan_sha256
          and terminal["execution_kind"] == "qwen_screen_singletons" and terminal["completed_steps"] == total
          and terminal["sft_steps"] == 160 and terminal["dpo_steps"] == d
          and terminal["exposures"] == plan["exposures"]
          and terminal["one_pass_sft"] is True and terminal["one_pass_dpo"] is plan["one_pass_dpo"]
          and terminal["reference_scores_unchanged"] is True and terminal["reference_pair_count"] == d
          and terminal["provenance_qualified"] is False and terminal["retry_allowed"] is False
          and terminal["environment_responses"] == 0
          and terminal["outer_provenance_receipt_sha256"] == plan["outer_provenance_receipt_sha256"]
          and terminal["base_before"] == terminal["base_after"]
          and terminal["projection_before"] == terminal["projection_after"], "incomplete engine terminal")
    usage = terminal["checkpoint_usage"]
    _need(usage["enabled"] is True and usage["use_reentrant"] is False
          and usage["preserve_rng_state"] is True and usage["native_checkpointing"] is False
          and usage["recompute_calls"] >= recompute_min and usage["forward_calls"] >= recompute_min,
          "missing exact activation-checkpoint evidence")
    steps = []
    _need({p.name for p in root.glob("step-*.complete.json")} ==
          {f"step-{i:05d}.complete.json" for i in range(total)}, "step inventory differs")
    for index, scheduled in enumerate(plan["schedule"]):
        stem = f"step-{index:05d}"
        event = _read(root / (stem + ".complete.json"))
        start = _read(root / (stem + ".start.json"))
        returned = _read(root / (stem + ".update-returned.json"))
        attempted = _read(root / (stem + ".optimizer-attempt.json"))
        gradients = _read(root / (stem + ".gradients.json"))
        loss = _read(root / (stem + ".loss.json"))
        intent = _read(root / (stem + ".update-intent.json"))
        for receipt in (event, start, returned, attempted):
            _need(receipt["index"] == index and all(receipt[k] == v for k, v in scheduled.items()),
                  "engine step identity differs")
        _need(start["record_pin"] == plan["record_pins"][scheduled["record_id"]]
              and start["optimizer_step_started"] is False and start["optimizer_step_returned"] is False
              and attempted["optimizer_step_started"] is True and attempted["optimizer_step_returned"] is False
              and returned["optimizer_step_started"] is True and returned["optimizer_step_returned"] is True
              and event["optimizer_step_started"] is True and event["optimizer_step_returned"] is True,
              "incomplete optimizer attempt/return evidence")
        _need(loss["finite"] is True and _finite(event["loss"]) and loss["loss"] == event["loss"]
              and _finite(event["gradient_norm_before_clip"], positive=True)
              and _finite(event["gradient_norm_after_clip"], positive=True)
              and event["gradient_norm_after_clip"] <= 1.000001
              and intent["gradient_norm_before_clip"] == event["gradient_norm_before_clip"]
              and intent["gradient_norm_after_clip"] == event["gradient_norm_after_clip"]
              and intent["clip_limit"] == 1., "invalid loss or gradient clipping evidence")
        tensors = gradients["tensors"]
        _need(len(tensors) == 112 and {t["name"] for t in tensors} == set(plan["expected_adapter_shapes"])
              and all(t["present"] is True and t["finite"] is True
                      and t["shape"] == plan["expected_adapter_shapes"][t["name"]] for t in tensors)
              and type(event["checkpoint_recompute_delta"]) is int
              and gradients["checkpoint_recompute_delta"] == event["checkpoint_recompute_delta"]
                  >= (28 if scheduled["stage"] == "sft" else 56),
              "missing adapter gradients or recomputation")
        rec = next(r for r in plan["records"] if r["record_id"] == scheduled["record_id"])
        commands = [rec["supervised_command"]] if scheduled["stage"] == "sft" else rec["strict_pairs"][scheduled["pair_index"]]
        tokens = sum(len(rec["commands"][c]) for c in commands)
        _need(event["rows"] == len(commands) and event["action_tokens"] == tokens
              and event["input_tokens"] == len(commands) * len(rec["prompt_ids"]) + tokens,
              "training token/row exposure differs")
        _need(all(_finite(event[k], positive=True) for k in
                  ("cuda_elapsed_ms", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes")),
              "missing actual CUDA step/memory evidence")
        steps.append(event)
    _need(usage["recompute_calls"] >= sum(s["checkpoint_recompute_delta"] for s in steps),
          "terminal recomputation count is below completed-step deltas")
    checkpoints = {}
    for stem, count in (("initial", 0), ("sft", 160), ("final", total), ("step-00159", 160), (f"step-{total-1:05d}", total)):
        checkpoint = _read(root / (stem + ".checkpoint.json"))
        expected = {"adapter"} if stem == "initial" else {"adapter", "optimizer"}
        _need(checkpoint["completed_steps"] == count and set(checkpoint["files"]) == expected
              and checkpoint["provenance_qualified"] is False, "checkpoint metadata differs")
        for kind, pin in checkpoint["files"].items():
            path = root / (stem + "." + kind + ".pt")
            _need(path.is_file() and not path.is_symlink() and _sha(path) == pin, "checkpoint file pin differs")
        checkpoints[stem] = checkpoint
    _need(checkpoints["sft"]["adapter_signature"] == checkpoints["step-00159"]["adapter_signature"]
          and checkpoints["final"]["adapter_signature"] == checkpoints[f"step-{total-1:05d}"]["adapter_signature"]
          and len({checkpoints[s]["adapter_signature"] for s in ("initial", "sft", "final")}) == 3,
          "adapter did not change in both training stages")
    initial = _read(root / "initial-signatures.json")
    _need(initial["base"] == terminal["base_before"] and initial["projection"] == terminal["projection_before"]
          and initial["adapter"] == checkpoints["initial"]["adapter_signature"]
          and initial["adapter_parameters"] == PARAMETERS, "initial signature binding differs")
    reference = _read(root / "reference-snapshot.json")
    scores = _read(root / "reference-scores.json")
    pairs = _pair_specs(plan)
    _need(reference["signature"] == scores["signature"] == terminal["reference_signature"]
          and reference["reference_sha256"] == _sha(root / "reference.pt")
          and reference["sft_adapter_sha256"] == checkpoints["sft"]["files"]["adapter"]
          and reference["completed_sft_steps"] == 160 and reference["policy_optimizer_updates_after_snapshot"] == 0
          and set(scores) == {"signature", "scores", "unique_pair_score_calls", "decoder_forwards", "scoring_layout"}
          and type(scores["unique_pair_score_calls"]) is int and scores["unique_pair_score_calls"] == d
          and type(scores["decoder_forwards"]) is int and scores["decoder_forwards"] == 2*d
          and scores["scoring_layout"] == "ordered_unpadded_singletons"
          and set(scores["scores"]) == set(pairs)
          and all(type(v) is list and len(v) == 2 and all(_finite(x) for x in v) for v in scores["scores"].values()),
          "fixed post-SFT reference schema/binding differs")
    return terminal, checkpoints, steps, scores["scores"]

def validate_outputs(root, prepared, *, prepared_sha256, plan_sha256):
    """Independently check saved worker evidence, using stdlib only.

    root is the artifacts path passed to run_training (not journal.output).
    Raises on incomplete/failed/mutated evidence. The result is a worker evidence
    report, never runtime, scheduler-cleanup, scientific or whole-run acceptance.
    Tensor contents are byte-pinned; this function does not deserialize .pt files.
    """
    root, plan = Path(root), _plan(prepared, prepared_sha256=prepared_sha256, plan_sha256=plan_sha256)
    _need(root.is_dir() and not root.is_symlink() and not (root / "failure.json").exists(),
          "missing/failed worker artifacts")
    manifest = _read(root / "evidence-files.json")
    expected_files = _expected_evidence_files(plan)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _need(actual == expected_files, "evidence file inventory differs from pinned recipe")
    _need(type(manifest) is dict and set(manifest) == expected_files - {"evidence-files.json", "summary.json"},
          "evidence manifest inventory differs from pinned recipe")
    # Membership and byte integrity are separate checks: re-pinning an extra
    # attempt, generation, checkpoint or failure cannot extend the recipe.
    for name, pin in manifest.items():
        path = Path(name)
        _need(not path.is_absolute() and ".." not in path.parts and (root / path).is_file()
              and not any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)),
              "unsafe evidence path")
        _need(_sha(root / path) == pin, "worker evidence file changed: " + name)
    terminal, checkpoints, steps, reference_scores = _check_training(root / "training", plan, plan_sha256=plan_sha256)
    first = _read(root / "first-gpu-work.json")
    _need(first["training_updates"] == 0 and first["forward_count"] == 1
          and first["record_id"] == plan["records"][0]["record_id"] and first["device"] == "cuda:0"
          and _finite(first["cuda_interval_ms"], positive=True)
          and _finite(first["peak_gpu_allocated_bytes"], positive=True)
          and type(first["scores"]) is list and len(first["scores"]) == 1
          and all(_finite(x) for x in first["scores"]), "first real CUDA forward missing")
    arithmetic = _read(root / "arithmetic.json")
    _need(arithmetic["recipe"] == plan["numeric_recipe"] and arithmetic["strict"] is True
          and arithmetic["warn_only"] is False and arithmetic["float32_matmul"] == "highest"
          and arithmetic["allow_tf32"] is False and arithmetic["cublas_workspace_config"] == ":4096:8"
          and arithmetic["cpu_seed"] == arithmetic["cuda_seed"] == 109 and arithmetic["visible_gpus"] == 1
          and arithmetic["flash"] is True and all(arithmetic[k] is False for k in ("math", "efficient", "cudnn")),
          "screen singleton arithmetic flags differ")
    before = _read(root / "final-before-reload.json")
    order = {r["record_id"]: sorted(r["commands"]) for r in plan["records"]}
    _need(before["commands"] == order and set(before["scores"]) == set(order)
          and all(len(before["scores"][rid]) == len(cmds) for rid, cmds in order.items())
          and before["optimizer_updates"] == 0
          and before["scoring_layout"] == "ordered_unpadded_singletons"
          and type(before["decoder_forwards"]) is int
          and before["decoder_forwards"] == sum(map(len, order.values())), "final saved score roster differs")
    _score_difference(before["scores"], before["scores"])
    sequence = [*plan["records"], plan["records"][0]]
    generations, loads, reload_phases = [], [], []
    initial_load = _read(root / "initial-load.json")
    _need(initial_load["stage"] == "initial" and initial_load["fresh_base"] is True
          and initial_load["restoration"] is None, "initial base must not restore any adapter")
    loads.append(initial_load)
    for stage in ("sft", "final"):
        reload = _read(root / (stage + "-reload.json"))
        load = _read(root / (stage + "-load.json"))
        scored = _read(root / (stage + "-reload-scores.json"))
        expected = reference_scores if stage == "sft" else before["scores"]
        _need(scored["scoring_layout"] == "ordered_unpadded_singletons"
              and type(scored["decoder_forwards"]) is int
              and scored["decoder_forwards"] == sum(map(len, expected.values())),
              "reload must retain every ordered singleton decoder forward")
        _need(_score_difference(reload["scores"], expected) == 0 and reload["score_max_absolute_difference"] == 0
              and scored["scores"] == reload["scores"] and scored["expected_scores"] == expected
              and scored["max_absolute_difference"] == 0
              and reload["scores_exact"] is True and reload["optimizer_updates"] == 0
              and reload["adapter_sha256"] == checkpoints[stage]["files"]["adapter"]
              and reload["training_receipt_sha256"] == _sha(root / "training" / (stage + ".checkpoint.json"))
              and reload["fresh_base"] is True and reload["adapter_parameters"] == PARAMETERS,
              "fresh-base checkpoint score/restore binding differs")
        restoration = load["restoration"]
        _need(load["stage"] == stage and load["fresh_base"] is True
              and restoration["adapter_sha256"] == reload["adapter_sha256"]
              and restoration["training_receipt_sha256"] == reload["training_receipt_sha256"]
              and restoration["rank"] == 8 and restoration["alpha"] == 16.
              and restoration["adapter_parameters"] == PARAMETERS
              and len(restoration["projection_names"]) == 56
              and set(restoration["projection_names"]) == {n.rsplit(".", 1)[0] for n in plan["expected_adapter_shapes"]},
              "adapter restoration receipt differs")
        loads.append(load)
        signatures = reload["signatures_before"]
        _need(signatures == reload["signatures_after"] and signatures["base"] == terminal["base_before"]
              and signatures["projection"] == terminal["projection_before"]
              and signatures["adapter"] == checkpoints[stage]["adapter_signature"]
              and reload["tensor_versions_unchanged"] is True and reload["inference_parameters_frozen"] is True,
              "reload/inference tensor signatures differ")
        if stage == "sft":
            _need(signatures["reference"] == terminal["reference_signature"], "SFT restored reference differs")
        else:
            _need(before["signatures_before"] == before["signatures_after"]
                  and all(before["signatures_after"][k] == signatures[k] for k in ("base", "projection", "adapter")),
                  "final pre-reload signatures differ")
        stage_rows = []
        for index, rec in enumerate(sequence):
            row = _read(root / f"{stage}-generation-{index:02d}.json")
            raw = _read(root / f"{stage}-raw-generation-{index:02d}.json")
            _need(row["stage"] == stage and row["ordinal"] == index and row["record_id"] == rec["record_id"]
                  and row["record_pin"] == plan["record_pins"][rec["record_id"]]
                  and raw["sequence_ids"] == row["sequence_ids"] and raw["dtype"] == "torch.int64"
                  and raw["shape"] == [1, len(row["sequence_ids"])], "generation identity/raw capture differs")
            stop = _legal_sequence(rec, row["sequence_ids"], row["completion"])
            _need(row["stop"] is stop and row["legal_commands"] == len(rec["commands"])
                  and row["new_training_updates"] == 0 and row["environment_responses"] == 0
                  and row["tensor_versions_unchanged"] is True and row["cross_request_cache_supplied"] is False
                  and _finite(row["cuda_interval_ms"], positive=True)
                  and _finite(row["peak_gpu_allocated_bytes"], positive=True), "invalid frozen inference evidence")
            stage_rows.append(row)
        _need(stage_rows[0]["sequence_ids"] == stage_rows[-1]["sequence_ids"]
              and stage_rows[0]["completion"] == stage_rows[-1]["completion"], "first/intervening/first repeat differs")
        generations.extend(stage_rows)
        reload_phases.append([load, scored, *stage_rows, reload])
    for load in loads:
        projection = load["projection"]
        _need(projection["recipe"] == "fixed_row_fp32_action_reduction_v1"
              and projection["original_parameter_retained"] is True and projection["independent_storage"] is True
              and projection["original_dtype"] == "torch.bfloat16" and projection["projection_dtype"] == "torch.float32"
              and projection["new_trainable_parameters"] == 0 and projection["projection_rows_per_call"] == 1
              and _finite(load["wall_seconds"], positive=True), "canonical projection/load evidence differs")
    memory_phases = []
    for label in ("before-sft-reload", "before-final-reload", "after-final-inference"):
        memory = _read(root / (label + "-memory.json"))
        _check_disposal_memory(memory)
        memory_phases.append(memory)
    summary = _read(root / "summary.json")
    expected_summary = _summary_facts(prepared, terminal, checkpoints, steps, first, generations, root, prepared_sha256=prepared_sha256, plan_sha256=plan_sha256)
    _need(all(summary.get(k) == v for k, v in expected_summary.items()), "worker summary differs from saved evidence")
    _need(all(_finite(summary[k], positive=True) for k in ("elapsed_seconds", "process_cpu_seconds",
              "model_load_seconds", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "total_device_bytes")),
          "missing worker resource/timing evidence")
    _need(summary["model_load_seconds"] == sum(x["wall_seconds"] for x in loads)
          and summary["peak_gpu_allocated_bytes"] >= max([first["peak_gpu_allocated_bytes"]]
              + [s["peak_allocated_bytes"] for s in steps] + [g["peak_gpu_allocated_bytes"] for g in generations])
          and summary["peak_gpu_reserved_bytes"] >= max([first["peak_gpu_reserved_bytes"]]
              + [s["peak_reserved_bytes"] for s in steps] + [g["peak_gpu_reserved_bytes"] for g in generations]),
          "reported loading time or CUDA peak differs")
    clocks = [summary[k] for k in ("first_gpu_work_monotonic_seconds", "training_start_monotonic_seconds",
                                  "training_complete_monotonic_seconds", "inference_complete_monotonic_seconds")]
    _need(all(_finite(clock) for clock in clocks), "nonfinite summary phase clock")
    _need(_finite(first.get("monotonic_seconds")) and clocks[0] == first["monotonic_seconds"],
          "first GPU work clock differs from saved evidence")
    _need(clocks[0] <= clocks[1] < clocks[2] < clocks[3],
          "forward/training/inference phase ordering differs")
    # Engine receipts have no absolute clocks. Bound training completion by
    # the saved pre-reload score receipt, then reconstruct the stamped phases.
    phase_clocks = [initial_load.get("monotonic_seconds"), *clocks[:3]]
    phase_clocks.extend(row.get("monotonic_seconds") for row in
                        [before, memory_phases[0], *reload_phases[0], memory_phases[1], *reload_phases[1]])
    phase_clocks.extend((clocks[3], memory_phases[2].get("monotonic_seconds")))
    _need(all(_finite(clock) for clock in phase_clocks)
          and all(a <= b for a, b in zip(phase_clocks, phase_clocks[1:])),
          "summary phase bounds differ from saved clocks")
    return {**expected_summary, "status": "validated_worker_evidence", "evidence_files": len(manifest),
            "summary_sha256": _sha(root / "summary.json"), "tensor_files_deserialized": False,
            "runtime_and_cleanup_qualified": False}

def _summary_facts(prepared, terminal, checkpoints, steps, first, generations, root, *, prepared_sha256, plan_sha256):
    d = prepared["training_plan"]["dpo_steps"]
    return {"version": VERSION, "status": "worker_complete", "scope": "policy_train",
        "execution_kind": "qwen_screen_singletons",
        "prepared_sha256": prepared_sha256, "training_plan_sha256": plan_sha256,
        **_prepared_lineage(prepared),
        "engine_terminal_sha256": _sha(root / "training" / "terminal.json"),
        "initial_adapter_sha256": checkpoints["initial"]["files"]["adapter"],
        "sft_adapter_sha256": checkpoints["sft"]["files"]["adapter"],
        "final_adapter_sha256": checkpoints["final"]["files"]["adapter"],
        "reference_sha256": _sha(root / "training" / "reference.pt"),
        "base_signature": terminal["base_before"], "projection_signature": terminal["projection_before"],
        "reference_signature": terminal["reference_signature"], "first_gpu_work_sha256": _sha(root / "first-gpu-work.json"),
        "adapter_matrices": 112, "adapter_parameters": PARAMETERS, "seed": 109, "visible_gpus": 1,
        "records": 160, "sft_steps": 160, "dpo_steps": d, "training_updates": 160+d,
        "stored_tokens": prepared["training_plan"]["stored_tokens"],
        "training_input_tokens": sum(s["input_tokens"] for s in steps),
        "training_action_tokens": sum(s["action_tokens"] for s in steps),
        "sft_rows": 160, "dpo_rows": 2*d, "inference_training_updates": 0,
        "generation_count": len(generations), "legal_generation_count": len(generations),
        "stop_count": sum(g["stop"] for g in generations), "action_count": sum(not g["stop"] for g in generations),
        "prepared_legal_commands": sum(len(r["commands"]) for r in prepared["training_plan"]["records"]),
        "reference_pair_count": d, "reference_decoder_forwards": 2*d,
        "scoring_layout": "ordered_unpadded_singletons",
        "final_score_count": sum(len(r["commands"]) for r in prepared["training_plan"]["records"]),
        "fresh_base_load_count": 3, "prior_adapter_restored": False,
        "sft_reload_scores_exact": True, "final_reload_scores_exact": True, "same_state_repeatability": True,
        "sft_reload_score_max_difference": 0., "final_reload_score_max_difference": 0.,
        "inference_signatures_unchanged": True, "base_unchanged": True, "projection_unchanged": True,
        "checkpoint_recompute_calls": terminal["checkpoint_usage"]["recompute_calls"],
        "cuda_step_ms_total": sum(s["cuda_elapsed_ms"] for s in steps),
        "cuda_generation_ms_total": sum(g["cuda_interval_ms"] for g in generations),
        "cuda_first_forward_ms": first["cuda_interval_ms"], "training_wall_seconds": terminal["elapsed_seconds"],
        "environment_responses": 0, "scientific_responses": 0, "scientific_qualified": False,
        "whole_run_accepted": False, "outer_execution_qualified": False, "retry_allowed": False}

def run_training(prepared, model_directory, output_path, observer, journal, *, prepared_sha256, plan_sha256):
    """Run exactly 160 SFT then the fixed D DPO updates and two frozen reload checks.

    All heavy imports occur here, after parent-owned authenticated imports and
    model authentication. The already-armed deadline is neither reset nor caught
    as success. Evidence is exclusive-create; failures propagate without retry.
    """
    root = Path(output_path).absolute()
    journal_root = Path(journal.output).absolute()
    relative = root.relative_to(journal_root)
    _need(relative.parts and not root.is_symlink(), "artifacts must be a distinct child of journal.output")
    root.mkdir(exist_ok=True)
    _need(not list(root.iterdir()), "worker artifacts must be empty; no resume or overwrite")
    started, cpu = time.monotonic(), time.process_time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stage, model = "prepared_validation", None
    def save(name, value):
        journal.save((relative / name).as_posix(), value)
    def stamp():
        return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monotonic_seconds": time.monotonic(), "elapsed_seconds": time.monotonic() - started}
    try:
        plan = _plan(prepared, prepared_sha256=prepared_sha256, plan_sha256=plan_sha256)
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, GenerationConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        import llm_intervention_training_engine as engine
        from llm_intervention_learning import LoRALinear, install_adapters, pack_actions
        from llm_intervention_canonical_scoring import install_canonical_projection, sequence_log_probs_unpadded
        from llm_intervention_grammar import FiniteActionGrammar
        from llm_intervention_language_policy import restore_adapter

        _need(plan["numeric_recipe"] == engine.screen_numeric_recipe(), "engine recipe differs from accepted CPU plan")
        # Set before our first CUDA operation. Parent must also set this before
        # any earlier CUDA initialization; ServiceRuntime's closure is CPU-only.
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        _need(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "one BF16-capable visible CUDA GPU required")
        torch.manual_seed(109)
        torch.cuda.manual_seed_all(109)
        device_properties = torch.cuda.get_device_properties(0)
        names = tuple(f"model.layers.{i}.self_attn.{part}" for i in range(28) for part in ("q_proj", "v_proj"))
        pairs = _pair_specs(plan)
        records = plan["records"]
        command_order = {r["record_id"]: sorted(r["commands"]) for r in records}
        final_specs = {r["record_id"]: (r, command_order[r["record_id"]]) for r in records}
        grammars = {r["record_id"]: FiniteActionGrammar(tuple(r["prompt_ids"]),
            tuple(tuple(r["commands"][c]) for c in command_order[r["record_id"]]), r["eos_token_id"]) for r in records}
        for rec in records:
            for command, tokens in rec["commands"].items():
                _legal_sequence(rec, rec["prompt_ids"] + tokens, command)
        loads, memory_peaks, generations = [], [], []

        def memory():
            values = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            memory_peaks.append(values)
            return {"peak_gpu_allocated_bytes": values[0], "peak_gpu_reserved_bytes": values[1]}

        def clear_cuda(label):
            _clear_cuda_after_disposal(torch, label=label, save=save, stamp=stamp)

        def signatures(current):
            adapters = set(plan["expected_adapter_shapes"])
            return {"adapter": engine._signature((n, p) for n, p in current.named_parameters() if n in adapters),
                "base": engine._signature([(n, p) for n, p in current.named_parameters() if n not in adapters] +
                    [(n, b) for n, b in current.named_buffers() if "reference_" not in n]),
                "projection": engine._signature([("projection", current.lm_head.weight)]),
                "reference": engine._signature((n, b) for n, b in current.named_buffers() if "reference_" in n)}

        def versions(current):
            return {n: t._version for n, t in [*current.named_parameters(), *current.named_buffers()]}

        def scores(current, specs, *, mode="singletons"):
            # Validate the complete requested layout before the first forward.
            layouts = {key: scoring_layout(rec, commands, mode=mode)
                       for key, (rec, commands) in specs.items()}
            result = {}
            with torch.no_grad():
                for key, (rec, commands) in specs.items():
                    result[key] = []
                    for group in layouts[key]:
                        batch = pack_actions(rec["prompt_ids"], [rec["commands"][c] for c in group], rec["pad_token_id"], 8192)
                        value = sequence_log_probs_unpadded(current, batch, selected=True)
                        _need(value.device.type == "cuda" and value.dtype == torch.float32, "CUDA FP32 scores required")
                        result[key].extend(value.detach().cpu().tolist())
                    journal.event('score_record_complete', context=stage, record_id=rec['record_id'],
                                  score_key=str(key), decoder_forwards=len(result[key]))
            return result

        def load(stage_name, checkpoint=None):
            begin = time.monotonic()
            journal.event("model_load_start", checkpoint=stage_name)
            current = AutoModelForCausalLM.from_pretrained(str(model_directory), local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            _need(type(current) is Qwen3ForCausalLM and current.config._attn_implementation == "sdpa"
                  and not current.is_gradient_checkpointing
                  and not any(isinstance(m, LoRALinear) for m in current.modules())
                  and all(t.device.type == "cpu" for t in [*current.parameters(), *current.buffers()]),
                  "fresh unwrapped CPU Qwen3 base required")
            restoration = None
            if checkpoint is None:
                installed = install_adapters(current, rank=8, alpha=16., targets=("q_proj", "v_proj"))
                _need(set(installed) == set(names), "fresh adapter projection inventory differs")
            else:
                path = root / "training" / (stage_name + ".adapter.pt")
                raw = path.read_bytes()
                restoration = restore_adapter(current, raw, adapter_sha256=checkpoint["files"]["adapter"],
                    training_receipt_sha256=_sha(root / "training" / (stage_name + ".checkpoint.json")),
                    expected_projection_names=names, rank=8, alpha=16.)
                del raw
            current.to("cuda:0")
            projection = install_canonical_projection(current)
            current.eval()
            engine._inventory(current, plan)  # All exact flags, shapes, dtypes, seed, architecture.
            if checkpoint is None:
                _need(all(bool((m.B == 0).all()) for m in current.modules() if isinstance(m, LoRALinear)),
                      "fresh adapters must start with zero B; prior adapters forbidden")
            else:
                _need(not any(p.requires_grad or p.grad is not None for p in current.parameters()),
                      "reload inference must be frozen and gradient-free")
            torch.cuda.synchronize()
            receipt = {**stamp(), "stage": stage_name, "wall_seconds": time.monotonic() - begin,
                       "fresh_base": True, "projection": projection, "restoration": restoration, **memory()}
            loads.append(receipt)
            save(stage_name + "-load.json", receipt)
            journal.event("model_load_complete", checkpoint=stage_name, wall_seconds=receipt["wall_seconds"])
            return current

        def generate(current, rec, checkpoint, ordinal, baseline):
            grammar = grammars[rec["record_id"]]
            def allowed(batch_id, ids):
                _need(batch_id == 0 and ids.ndim == 1, "one unpadded prompt required")
                return list(grammar.allowed_next(ids.tolist()))
            def legal_scores(ids, logits):
                _need(ids.ndim == logits.ndim == 2 and ids.shape[0] == logits.shape[0] == 1,
                      "single-prompt generation logits required")
                legal = grammar.allowed_next(ids[0].tolist())
                _need(max(legal) < logits.shape[1] and bool(torch.isfinite(logits[0, list(legal)]).all()),
                      "nonfinite legal-command logits")
                return logits
            config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=32, pad_token_id=rec["pad_token_id"], eos_token_id=rec["eos_token_id"], use_cache=True)
            ids = torch.tensor([rec["prompt_ids"]], dtype=torch.long, device="cuda:0")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.monotonic()
            start.record()
            with torch.no_grad():
                result = current.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
                    prefix_allowed_tokens_fn=allowed, logits_processor=[legal_scores], use_model_defaults=False)
            end.record()
            torch.cuda.synchronize()
            raw = {"dtype": str(getattr(result, "dtype", None)), "shape": list(result.shape) if isinstance(result, torch.Tensor) else None,
                   "sequence_ids": result[0].detach().cpu().tolist() if isinstance(result, torch.Tensor) and result.ndim == 2 and result.shape[0] == 1 else None}
            save(f"{checkpoint}-raw-generation-{ordinal:02d}.json", raw)
            _need(type(result) is torch.Tensor and result.dtype == torch.long and result.ndim == 2 and result.shape[0] == 1,
                  "one integer generated sequence required")
            sequence = raw["sequence_ids"]
            grammar.validate_complete(sequence)
            suffix = sequence[len(rec["prompt_ids"]):]
            matches = [command for command, tokens in rec["commands"].items() if tokens == suffix]
            _need(len(matches) == 1, "generated tokens must map to exactly one prepared command")
            completion = matches[0]
            stop = _legal_sequence(rec, sequence, completion)
            unchanged = versions(current) == baseline
            row = {**stamp(), "stage": checkpoint, "ordinal": ordinal, "record_id": rec["record_id"],
                "record_pin": plan["record_pins"][rec["record_id"]], "sequence_ids": sequence,
                "completion": completion, "stop": stop, "legal_commands": len(rec["commands"]),
                "new_training_updates": 0, "environment_responses": 0, "cross_request_cache_supplied": False,
                "tensor_versions_unchanged": unchanged, "cuda_interval_ms": start.elapsed_time(end),
                "wall_seconds": time.monotonic() - began, **memory()}
            save(f"{checkpoint}-generation-{ordinal:02d}.json", row)
            _need(unchanged and _finite(row["cuda_interval_ms"], positive=True), "inference changed tensors or lacks CUDA work")
            journal.event("generation_complete", checkpoint=checkpoint, ordinal=ordinal, record_id=rec["record_id"])
            return row

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            save("arithmetic.json", {"recipe": plan["numeric_recipe"], "strict": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "float32_matmul": torch.get_float32_matmul_precision(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "cpu_seed": torch.initial_seed(),
                "cuda_seed": torch.cuda.initial_seed(), "visible_gpus": torch.cuda.device_count(),
                "flash": torch.backends.cuda.flash_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled(),
                "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(), "cudnn": torch.backends.cuda.cudnn_sdp_enabled()})
            stage = "initial_model_load"
            model = load("initial")
            stage = "first_gpu_forward"
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            first_scores = scores(model, {"first": (records[0], [records[0]["supervised_command"]])})["first"]
            end.record()
            torch.cuda.synchronize()
            first = {**stamp(), "record_id": records[0]["record_id"], "device": "cuda:0", "forward_count": 1,
                "training_updates": 0, "scores": first_scores, "cuda_interval_ms": start.elapsed_time(end), **memory()}
            save("first-gpu-work.json", first)
            _need(_finite(first["cuda_interval_ms"], positive=True) and first["peak_gpu_allocated_bytes"] > 0,
                  "first CUDA forward requires positive interval and memory evidence")
            journal.save("first-gpu-work.json", first)
            observer.observe("after-model-load")
            stage = "training"
            training_start = time.monotonic()
            journal.event("training_start", training_plan_sha256=plan_sha256, sft_steps=160, dpo_steps=plan["dpo_steps"])
            terminal = engine.fit(model=model, plan=plan, plan_sha256=plan_sha256, output=root / "training")
            training_complete = time.monotonic()
            journal.event("training_complete", completed_steps=terminal["completed_steps"], training_plan_sha256=plan_sha256)
            terminal, checkpoints, steps, reference_scores = _check_training(root / "training", plan, plan_sha256=plan_sha256)
            memory_peaks.extend((s["peak_allocated_bytes"], s["peak_reserved_bytes"]) for s in steps)
            stage = "final_scores_before_disposal"
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False)
            before_signatures = signatures(model)
            final_scores = scores(model, final_specs, mode="singletons")
            after_signatures = signatures(model)
            save("final-before-reload.json", {**stamp(), "commands": command_order, "scores": final_scores,
                "signatures_before": before_signatures, "signatures_after": after_signatures, "optimizer_updates": 0,
                "scoring_layout": "ordered_unpadded_singletons",
                "decoder_forwards": sum(map(len, final_scores.values()))})
            _need(before_signatures == after_signatures, "pre-reload scoring changed tensors")
            memory()
            model = None
            clear_cuda("before-sft-reload")
            for checkpoint in ("sft", "final"):
                stage = checkpoint + "_reload"
                model = load(checkpoint, checkpoints[checkpoint])
                baseline, signature_before = versions(model), signatures(model)
                specs = pairs if checkpoint == "sft" else final_specs
                mode = "reference_pair" if checkpoint == "sft" else "singletons"
                observed = scores(model, specs, mode=mode)
                expected = reference_scores if checkpoint == "sft" else final_scores
                difference = _score_difference(observed, expected)
                # Persist the actual discrepancy before any rejection, including
                # the chosen/rejected pair keys used by the engine reference.
                save(checkpoint + "-reload-scores.json", {**stamp(), "scores": observed,
                    "expected_scores": expected, "max_absolute_difference": difference,
                    "scoring_layout": "ordered_unpadded_singletons",
                    "decoder_forwards": sum(map(len, observed.values()))})
                _need(difference == 0, "fresh-base " + checkpoint + " scores differ exactly")
                _need(signature_before["base"] == terminal["base_before"]
                      and signature_before["projection"] == terminal["projection_before"]
                      and signature_before["adapter"] == checkpoints[checkpoint]["adapter_signature"],
                      "restored base/projection/adapter signatures differ")
                if checkpoint == "sft":
                    _need(signature_before["reference"] == terminal["reference_signature"], "SFT reference buffers differ")
                stage = checkpoint + "_inference"
                rows = [generate(model, rec, checkpoint, i, baseline) for i, rec in enumerate([*records, records[0]])]
                generations.extend(rows)
                signature_after = signatures(model)
                restore_pin = _sha(root / "training" / (checkpoint + ".adapter.pt"))
                receipt = {**stamp(), "scores": observed, "scores_exact": difference == 0,
                    "score_max_absolute_difference": difference, "optimizer_updates": 0, "fresh_base": True,
                    "adapter_sha256": restore_pin, "adapter_parameters": PARAMETERS,
                    "training_receipt_sha256": _sha(root / "training" / (checkpoint + ".checkpoint.json")),
                    "signatures_before": signature_before, "signatures_after": signature_after,
                    "tensor_versions_unchanged": versions(model) == baseline,
                    "inference_parameters_frozen": not any(p.requires_grad or p.grad is not None for p in model.parameters())}
                save(checkpoint + "-reload.json", receipt)
                _need(receipt["tensor_versions_unchanged"] and receipt["inference_parameters_frozen"]
                      and signature_after == signature_before and restore_pin == checkpoints[checkpoint]["files"]["adapter"]
                      and rows[0]["sequence_ids"] == rows[-1]["sequence_ids"], "inference/repeat/restore-file check failed")
                journal.event("checkpoint_inference_complete", checkpoint=checkpoint, generation_count=len(rows),
                              reload_scores_exact=True, new_training_updates=0)
                if checkpoint == "sft":
                    model = None
                    clear_cuda("before-final-reload")
            inference_complete = time.monotonic()
            # This observation is deliberately after the final *real* generate.
            observer.observe("after-requests")
            memory()
            model = None
            clear_cuda("after-final-inference")
        stage = "worker_evidence_validation"
        manifest = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
        save("evidence-files.json", manifest)
        summary = {**_summary_facts(prepared, terminal, checkpoints, steps, first, generations, root, prepared_sha256=prepared_sha256, plan_sha256=plan_sha256),
            "started_utc": started_utc, "completed_utc": stamp()["utc"], "elapsed_seconds": time.monotonic() - started,
            "process_cpu_seconds": time.process_time() - cpu, "model_load_seconds": sum(x["wall_seconds"] for x in loads),
            "peak_gpu_allocated_bytes": max(x[0] for x in memory_peaks),
            "peak_gpu_reserved_bytes": max(x[1] for x in memory_peaks),
            "device_name": device_properties.name, "total_device_bytes": device_properties.total_memory,
            "first_gpu_work_monotonic_seconds": first["monotonic_seconds"],
            "training_start_monotonic_seconds": training_start, "training_complete_monotonic_seconds": training_complete,
            "inference_complete_monotonic_seconds": inference_complete}
        save("summary.json", summary)
        validate_outputs(root, prepared, prepared_sha256=prepared_sha256, plan_sha256=plan_sha256)
        journal.save("worker-summary.json", summary)
        return summary
    except BaseException as exc:
        failure = {**stamp(), "status": "failed", "stage": stage, "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "retry_allowed": False,
            "whole_run_accepted": False, "environment_responses": 0,
            "update_outcome": "inspect_retained_engine_steps_and_checkpoints"}
        # Retain engine and generation evidence; never retry, remove or overwrite
        # partial artifacts. Parent also records the propagated failure.
        try:
            save("failure.json", failure)
        except BaseException:
            pass
        raise
