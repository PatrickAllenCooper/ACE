#!/usr/bin/env python3
"""
Validate that gradient checkpointing actually propagates gradients to the
policy LM parameters when inputs are integer token IDs.

Reproduces the failure mode from the May 2026 anon30 ACE batches:
  "None of the inputs have requires_grad=True. Gradients will be None"
which silently zeroed the DPO gradient.

Uses a tiny random Qwen2 model (few MB) so it runs on CPU without the real
Qwen2.5-1.5B download. The architecture (embedding -> checkpointed decoder
layers -> lm_head) is identical, so the gradient-flow behaviour is the same.

Pass criteria:
  - REENTRANT (default) path: at least one mid-stack decoder-layer parameter
    has a None or all-zero gradient (demonstrates the bug), AND emits the
    "requires_grad" warning. (This is the broken baseline.)
  - NON-REENTRANT path: ALL trainable parameters (including mid-stack decoder
    MLP weights) receive non-None, non-all-zero gradients. (This is the fix.)

Usage:
  python scripts/analysis/test_grad_checkpoint.py
"""
import contextlib
import sys
import warnings

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _nullcontext():
    yield


def build_tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def run_once(use_reentrant, enable_input_require_grads, no_grad=False):
    """Returns (n_params_with_grad, n_params_total, n_zero_or_none, warned, mid_bad).

    no_grad=True mimics the DPO reference-model forward (wrapped in
    torch.no_grad()); we expect a benign 'requires_grad' warning there and
    no gradients (by design).
    """
    model = build_tiny_qwen()
    model.train()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    model.config.use_cache = False
    if enable_input_require_grads:
        model.enable_input_require_grads()

    input_ids = torch.randint(0, 256, (1, 16))
    labels = input_ids.clone()

    warned = {"flag": False}
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ctx = torch.no_grad() if no_grad else _nullcontext()
        with ctx:
            out = model(input_ids=input_ids)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            logp = F.log_softmax(shift_logits, dim=-1)
            token_logp = torch.gather(
                logp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = token_logp.sum()
        if not no_grad:
            loss.backward()
        for w in wlist:
            if "requires_grad" in str(w.message):
                warned["flag"] = True

    if no_grad:
        return 0, 0, 0, warned["flag"], False

    n_total = 0
    n_with_grad = 0
    n_zero_or_none = 0
    # Focus on a mid-stack decoder MLP weight, the layer that failed before.
    mid_layer_bad = False
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is None:
            n_zero_or_none += 1
            if "layers.2.mlp" in name or "layers.1.mlp" in name:
                mid_layer_bad = True
        else:
            if torch.count_nonzero(p.grad).item() == 0:
                n_zero_or_none += 1
                if "layers.2.mlp" in name or "layers.1.mlp" in name:
                    mid_layer_bad = True
            else:
                n_with_grad += 1
    return n_with_grad, n_total, n_zero_or_none, warned["flag"], mid_layer_bad


def main():
    print("=== Gradient-checkpointing DPO-gradient-flow test ===\n")

    def report(tag, **kw):
        nwg, nt, nz, warned, mid_bad = run_once(**kw)
        print(f"    params with grad: {nwg}/{nt}, none/zero: {nz}, "
              f"requires_grad warning: {warned}, mid-stack MLP broken: {mid_bad}")
        return nwg, nt, nz, warned, mid_bad

    print("[1] POLICY forward, REENTRANT, NO enable_input_require_grads")
    print("    (the original a30r batch -- expected BROKEN: no grad flow)")
    r1 = report("a30r", use_reentrant=True, enable_input_require_grads=False)

    print()
    print("[2] POLICY forward, REENTRANT, WITH enable_input_require_grads")
    print("    (the a30d batch running now -- is it actually broken?)")
    r2 = report("a30d", use_reentrant=True, enable_input_require_grads=True)

    print()
    print("[3] POLICY forward, NON-REENTRANT, WITH enable_input_require_grads")
    print("    (the proposed fix -- expected: gradients everywhere)")
    r3 = report("fix", use_reentrant=False, enable_input_require_grads=True)

    print()
    print("[4] REFERENCE forward under torch.no_grad(), REENTRANT+enable")
    print("    (explains the warning source: ref model, harmless)")
    _, _, _, warned_ref, _ = run_once(
        use_reentrant=True, enable_input_require_grads=True, no_grad=True)
    print(f"    requires_grad warning under no_grad: {warned_ref}")

    print()
    print("=" * 70)
    nwg3, nt3, nz3, warned3, mid3 = r3
    fix_ok = (nwg3 == nt3) and (nz3 == 0) and (not mid3)
    a30d_ok = (r2[0] == r2[1]) and (r2[2] == 0) and (not r2[4])

    print(f"Original a30r (reentrant, no enable): "
          f"{'grads flow' if (r1[0]==r1[1] and r1[2]==0 and not r1[4]) else 'BROKEN (grads missing)'}")
    print(f"Running a30d  (reentrant, +enable):   "
          f"{'grads flow OK' if a30d_ok else 'BROKEN (grads missing)'}")
    print(f"Proposed fix  (non-reentrant,+enable):"
          f"{'grads flow OK' if fix_ok else 'BROKEN'}")
    print(f"Warning under no_grad (ref model):    "
          f"{'fires (benign)' if warned_ref else 'silent'}")
    print("=" * 70)

    if fix_ok:
        print("\nPASS: the non-reentrant fix propagates gradients to ALL params.")
        if a30d_ok:
            print("NOTE: the CURRENTLY RUNNING a30d config ALSO propagates "
                  "gradients in this test. The live-log warning is most likely "
                  "the benign ref-model no_grad warning -- the running jobs may "
                  "be LEARNING. Verify via loss trajectory before cancelling.")
        sys.exit(0)
    else:
        print("\nFAIL: non-reentrant path still missing gradients. Do NOT resubmit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
