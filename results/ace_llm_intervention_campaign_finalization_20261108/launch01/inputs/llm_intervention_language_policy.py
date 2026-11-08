"""Shared-task LLM inference and adapter restoration primitives.

The caller must qualify source, model, tokenizer, training provenance, runtime
and resources. This module authenticates supplied adapter bytes and binds each
single unpadded prompt to its legal commands. It has no environment callback or
private evaluator. Reloading a pinned adapter is not proof of useful training.
"""
from __future__ import annotations

import hashlib
import io
import json
import math

import torch
from torch import nn

from llm_intervention_contract import canonical, digest, integer, parse_completion
from llm_intervention_grammar import build_action_grammar
from llm_intervention_learning import (LoRALinear, deterministic_scoring, install_adapters,
                                       pack_actions, snapshot_reference)
from llm_intervention_task import task_prompt, unpack_envelope


VERSION = "ace-shared-task-language-policy-v1"


def pin(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def restore_adapter(model, raw, *, adapter_sha256, training_receipt_sha256,
                    expected_projection_names, rank=8, alpha=16.):
    """Validate all tensors before installing q/v adapters on a CPU model.

    `expected_projection_names`, rank and alpha must come from an independently
    frozen architecture/training record. The caller authenticates the base
    model separately. The training receipt pin is lineage metadata, not a
    receipt verifier. No accelerator RNG or device is initialized here.
    """
    if (type(raw) is not bytes or not pin(adapter_sha256)
            or hashlib.sha256(raw).hexdigest() != adapter_sha256
            or not pin(training_receipt_sha256)):
        raise ValueError("independently pinned adapter bytes and training receipt required")
    if (type(rank) is not int or not 1 <= rank <= 256 or type(alpha) not in (int, float)
            or not math.isfinite(alpha) or alpha <= 0
            or type(expected_projection_names) is not tuple or not expected_projection_names
            or any(type(x) is not str or not x for x in expected_projection_names)
            or len(set(expected_projection_names)) != len(expected_projection_names)):
        raise ValueError("explicit bounded adapter architecture required")
    if (any(isinstance(m, LoRALinear) for m in model.modules())
            or any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers()))):
        raise ValueError("adapter restoration requires an unwrapped CPU base model")
    projections = {name: m for name, m in model.named_modules()
                   if name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj') and isinstance(m, nn.Linear)}
    if set(projections) != set(expected_projection_names):
        raise ValueError("base projection names differ from independently frozen architecture")
    tensors = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    expected = {name+suffix: shape for name, m in projections.items()
                for suffix, shape in (('.A', (rank, m.in_features)), ('.B', (m.out_features, rank)))}
    if type(tensors) is not dict or set(tensors) != set(expected):
        raise ValueError("adapter tensor keys differ from expected projections")
    for name, shape in expected.items():
        t = tensors[name]
        if (type(t) is not torch.Tensor or t.layout != torch.strided or t.dtype != torch.float32
                or t.device.type != 'cpu' or tuple(t.shape) != shape or not torch.isfinite(t).all()):
            raise ValueError("adapter tensor shape, dtype or values invalid: "+name)
    # Restore caller CPU randomness after constructors. No global manual_seed,
    # which could change CUDA/MPS/XPU randomness, is called.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(0).get_state())
        names = install_adapters(model, rank=rank, alpha=alpha)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in tensors:
                parameter.copy_(tensors[name])
    snapshot_reference(model)
    model.requires_grad_(False)
    model.eval()
    return {"version": VERSION, "adapter_sha256": adapter_sha256,
            "training_receipt_sha256": training_receipt_sha256,
            "projection_names": list(names), "rank": rank, "alpha": alpha,
            "adapter_parameters": sum(t.numel() for t in tensors.values()),
            "base_and_training_provenance_qualified": False}


def prepare_task_tokens(envelope, tokenizer, *, max_tokens, max_new_tokens):
    """Tokenize without truncation, admitting every affordable command and stop."""
    if (not integer(max_tokens, 2) or max_tokens > 131072
            or not integer(max_new_tokens, 1) or max_new_tokens >= max_tokens
            or not integer(tokenizer.eos_token_id) or not integer(tokenizer.pad_token_id)):
        raise ValueError("bounded explicit context and tokenizer special IDs required")
    detached = json.loads(canonical(envelope))
    state, task, _, menu, _ = unpack_envelope(detached)
    prompt = task_prompt(detached)
    prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    grammar = build_action_grammar(prompt_ids, menu,
        0 if state['terminal'] else state['remaining_units'],
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False), tokenizer.eos_token_id)
    longest = max(map(len, grammar.completion_ids))
    # Reserve the declared entire generation allowance, not only today's longest
    # command. The inference call uses this same max_new_tokens value.
    if longest > max_new_tokens or len(grammar.prompt_ids)+max_new_tokens > max_tokens:
        raise ValueError("public task or complete action exceeds frozen token limits")
    return grammar, {"version": VERSION, "envelope_sha256": digest(detached),
        "task_sha256": digest(task.payload()), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": len(grammar.prompt_ids), "longest_command_tokens": longest,
        "max_tokens": max_tokens, "max_new_tokens": max_new_tokens,
        "legal_commands": len(grammar.completion_ids), "enable_thinking": False}


def training_action_batch(envelope, tokenizer, completions, *, max_tokens, max_new_tokens):
    """Score precisely the same chat prefix and command+EOS used at deployment."""
    grammar, metadata = prepare_task_tokens(envelope, tokenizer,
        max_tokens=max_tokens, max_new_tokens=max_new_tokens)
    if type(completions) is not tuple or not completions or any(type(x) is not str for x in completions):
        raise ValueError("explicit nonempty tuple of matched action labels required")
    encoded = []
    legal_text = {tokenizer.decode(ids[:-1], skip_special_tokens=False) for ids in grammar.completion_ids}
    for completion in completions:
        if completion not in legal_text:
            raise ValueError("training completion text is not a canonical feasible command")
        tokens = tuple(tokenizer.encode(completion, add_special_tokens=False))+(tokenizer.eos_token_id,)
        if tokens not in grammar.completion_ids:
            raise ValueError("training completion is not a complete feasible deployment command")
        encoded.append(tokens)
    return pack_actions(grammar.prompt_ids, encoded, tokenizer.pad_token_id, max_tokens), metadata


def generate_task_decision(model, tokenizer, envelope, *, max_tokens, max_new_tokens):
    """Greedy single-prompt inference. Caller persists result before execution.

    This does not load weights, fit, charge an intervention or silently repair a
    rejected output. A separately qualified worker must supervise model lifetime,
    persist tokens and failures, and execute the validated returned command.
    """
    from transformers import GenerationConfig
    grammar, metadata = prepare_task_tokens(envelope, tokenizer,
        max_tokens=max_tokens, max_new_tokens=max_new_tokens)
    state, _, _, menu, _ = unpack_envelope(envelope)
    parameters = tuple(model.parameters())
    devices = {p.device for p in parameters}
    if not parameters or len(devices) != 1 or any(p.requires_grad for p in parameters):
        raise ValueError("one-device frozen inference model required")
    if (not integer(getattr(model.config, 'max_position_embeddings', None), 2)
            or max_tokens > model.config.max_position_embeddings):
        raise ValueError("declared context exceeds model configuration")
    def callback(batch_id, input_ids):
        if batch_id != 0 or input_ids.ndim != 1:
            raise ValueError("only one unpadded prompt is bound")
        return list(grammar.allowed_next(input_ids.tolist()))
    def check_legal_scores(input_ids, scores):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("single-prompt logits required")
        allowed = grammar.allowed_next(input_ids[0].tolist())
        if max(allowed) >= scores.shape[1] or not torch.isfinite(scores[0, list(allowed)]).all():
            raise ValueError("nonfinite legal-command score")
        return scores
    # A new explicit config avoids inheriting sampling, forced tokens or beam
    # settings from a model's stored generation configuration.
    config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, use_cache=True)
    ids = torch.tensor([grammar.prompt_ids], dtype=torch.long, device=parameters[0].device)
    with deterministic_scoring(model), torch.no_grad():
        sequences = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=config, prefix_allowed_tokens_fn=callback,
            logits_processor=[check_legal_scores], use_model_defaults=False)
    if (type(sequences) is not torch.Tensor or sequences.ndim != 2 or sequences.shape[0] != 1
            or sequences.dtype != torch.long):
        raise ValueError("expected one integer generated sequence")
    sequence = sequences[0].detach().cpu().tolist()
    grammar.validate_complete(sequence)
    completion = tokenizer.decode(sequence[len(grammar.prompt_ids):-1], skip_special_tokens=False)
    decision = parse_completion(completion, menu)
    if not decision.stop:
        a = next(a for a in menu if a.action_id == decision.action_id)
        if state['terminal'] or a.cost_units > state['remaining_units']:
            raise ValueError("generated intervention is unavailable")
    return completion, {**metadata, "prompt_ids": list(grammar.prompt_ids), "sequence_ids": sequence,
                        "completion": completion, "new_training_updates": 0,
                        "environment_callbacks_available": False}
