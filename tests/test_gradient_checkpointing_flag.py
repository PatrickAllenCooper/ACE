"""
Unit tests for the --gradient_checkpointing and --policy_dtype flags.

Context (Alpine H200/RTX Pro 6000 expansion, model-scale sweep): gradient
checkpointing and bf16 were previously only auto-enabled by --large_scale
(>=50 nodes) or --anonymize_nodes (>=30 nodes with anonymisation), which
covers the paper's own scaling cells but not a DPO run on a *larger policy
model* at small graph scale (e.g. Qwen2.5-3B doing DPO at N=30, where the
graph is small but the policy itself no longer fits without checkpointing).
These flags let the model-scale sweep force either control on independent of
graph scale.
"""

import subprocess
import sys

import pytest


def _run_help():
    return subprocess.run(
        [sys.executable, "ace_experiments.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.unit
def test_gradient_checkpointing_flag_registered():
    result = _run_help()
    assert "--gradient_checkpointing" in result.stdout


@pytest.mark.unit
def test_policy_dtype_flag_registered_with_choices():
    result = _run_help()
    assert "--policy_dtype" in result.stdout
    assert "float32" in result.stdout
    assert "bfloat16" in result.stdout
    assert "float16" in result.stdout


@pytest.mark.unit
def test_policy_dtype_rejects_invalid_choice():
    result = subprocess.run(
        [sys.executable, "ace_experiments.py", "--custom",
         "--policy_dtype", "not_a_real_dtype"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "invalid choice" in result.stdout.lower()


@pytest.mark.unit
def test_gradient_checkpointing_forces_gc_flag_regardless_of_scale():
    """
    Reproduces the derivation logic from ace_experiments.py's main() so the
    "force on regardless of --large_scale/--anonymize_nodes" contract is
    checked without loading a real HuggingFace model.
    """
    def derive_gc_flag(large_scale, anonymize_nodes, no_dpo, gradient_checkpointing):
        ls = large_scale or 0
        return ((ls >= 50 or (ls >= 30 and anonymize_nodes)) and not no_dpo) or \
            bool(gradient_checkpointing)

    # Small graph, no auto-trigger -- only the explicit flag turns it on.
    assert derive_gc_flag(large_scale=30, anonymize_nodes=False, no_dpo=False,
                          gradient_checkpointing=False) is False
    assert derive_gc_flag(large_scale=30, anonymize_nodes=False, no_dpo=False,
                          gradient_checkpointing=True) is True

    # Auto-trigger paths (existing behaviour) remain unaffected.
    assert derive_gc_flag(large_scale=50, anonymize_nodes=False, no_dpo=False,
                          gradient_checkpointing=False) is True
    assert derive_gc_flag(large_scale=30, anonymize_nodes=True, no_dpo=False,
                          gradient_checkpointing=False) is True


@pytest.mark.unit
def test_policy_dtype_override_takes_precedence_over_scale_heuristic():
    import torch

    def derive_policy_dtype(large_scale, policy_dtype_override):
        ls = large_scale or 0
        if policy_dtype_override is not None:
            return {"float32": None, "bfloat16": torch.bfloat16,
                    "float16": torch.float16}[policy_dtype_override]
        return torch.bfloat16 if ls >= 50 else None

    # No override: legacy scale-based heuristic.
    assert derive_policy_dtype(large_scale=5, policy_dtype_override=None) is None
    assert derive_policy_dtype(large_scale=50, policy_dtype_override=None) is torch.bfloat16

    # Explicit override wins even when it contradicts the heuristic.
    assert derive_policy_dtype(large_scale=5, policy_dtype_override="bfloat16") is torch.bfloat16
    assert derive_policy_dtype(large_scale=50, policy_dtype_override="float32") is None
    assert derive_policy_dtype(large_scale=5, policy_dtype_override="float16") is torch.float16
