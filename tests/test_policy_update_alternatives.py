"""
Unit tests for --policy_update alternatives to DPO.

Context (NeurIPS 2026 review, reviewer d6tT): "The paper does not convincingly
show that DPO is needed... Simpler alternatives such as supervised learning on
the best candidate [or] a pairwise ranking loss... should be compared."

These tests exercise sft_best_loss_llm and ranking_loss_llm against a small
fake policy model that mimics HuggingFacePolicy's forward() signature
(returns (logits, input_ids) for a batch of texts), so the loss math can be
verified exactly without loading a real LLM.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FakeCausalLMPolicy(nn.Module):
    """
    Minimal stand-in for HuggingFacePolicy.forward(): maps a batch of texts
    to (logits, input_ids) with a learnable scalar so gradients can be
    checked, without tokenizing or loading any real model.
    """

    def __init__(self, vocab_size=4, seq_len=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # One learnable scalar biases the logit of "token 0" at every position;
        # increasing it makes the model more confident in the label sequence.
        self.bias = nn.Parameter(torch.tensor(0.0))
        # Deterministic per-text "identity" so winner/loser get distinguishable
        # (but reproducible) log-probabilities.
        self._text_offsets = {}

    def _offset_for(self, text):
        if text not in self._text_offsets:
            # Stable, content-derived offset (not random) so tests are deterministic.
            self._text_offsets[text] = float(sum(ord(c) for c in text) % 7) * 0.1
        return self._text_offsets[text]

    def forward(self, scm_state, target_text_list, node_losses=None, intervention_history=None):
        batch = len(target_text_list)
        input_ids = torch.zeros(batch, self.seq_len, dtype=torch.long)  # all-zero "labels"
        logits = torch.zeros(batch, self.seq_len, self.vocab_size)
        for i, text in enumerate(target_text_list):
            logits[i, :, 0] = self.bias + self._offset_for(text)
        return logits, input_ids


def _manual_seq_log_prob(logits, input_ids):
    """Reference implementation matching dpo_loss_llm's token log-prob sum."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)


# =============================================================================
# sft_best_loss_llm
# =============================================================================

@pytest.mark.unit
def test_sft_best_loss_matches_manual_nll():
    from ace_experiments import sft_best_loss_llm

    model = _FakeCausalLMPolicy()
    win_text = "DO X1 = 1.0"

    loss = sft_best_loss_llm(model, scm_state=None, win_text=win_text)

    logits, input_ids = model(None, [win_text])
    expected = -_manual_seq_log_prob(logits, input_ids).mean()

    assert torch.allclose(loss, expected)


@pytest.mark.unit
def test_sft_best_loss_ignores_loser_entirely():
    """SFT-on-best must not depend on any 'loser' text -- it never sees one."""
    from ace_experiments import sft_best_loss_llm
    import inspect

    sig = inspect.signature(sft_best_loss_llm)
    assert "lose_text" not in sig.parameters
    assert "loser_cmd" not in sig.parameters


@pytest.mark.unit
def test_sft_best_loss_is_differentiable():
    from ace_experiments import sft_best_loss_llm

    model = _FakeCausalLMPolicy()
    loss = sft_best_loss_llm(model, scm_state=None, win_text="DO X2 = 0.5")
    loss.backward()

    assert model.bias.grad is not None
    assert not torch.isnan(model.bias.grad)


@pytest.mark.unit
def test_sft_best_loss_decreases_as_confidence_increases():
    """Higher logit mass on the label sequence should mean lower NLL."""
    from ace_experiments import sft_best_loss_llm

    model = _FakeCausalLMPolicy()
    text = "DO X3 = 0.0"

    with torch.no_grad():
        model.bias.fill_(0.0)
    loss_low_conf = sft_best_loss_llm(model, scm_state=None, win_text=text)

    with torch.no_grad():
        model.bias.fill_(5.0)
    loss_high_conf = sft_best_loss_llm(model, scm_state=None, win_text=text)

    assert loss_high_conf.item() < loss_low_conf.item()


# =============================================================================
# ranking_loss_llm
# =============================================================================

@pytest.mark.unit
def test_ranking_loss_matches_manual_bradley_terry():
    from ace_experiments import ranking_loss_llm

    model = _FakeCausalLMPolicy()
    win_text, lose_text = "DO X1 = 2.0", "DO X1 = -2.0"
    beta = 0.1

    loss = ranking_loss_llm(model, scm_state=None, win_text=win_text, lose_text=lose_text, beta=beta)

    win_logits, win_ids = model(None, [win_text])
    lose_logits, lose_ids = model(None, [lose_text])
    win_lp = _manual_seq_log_prob(win_logits, win_ids)
    lose_lp = _manual_seq_log_prob(lose_logits, lose_ids)
    expected = -F.logsigmoid(beta * (win_lp - lose_lp)).mean()

    assert torch.allclose(loss, expected)


@pytest.mark.unit
def test_ranking_loss_has_no_reference_policy_term():
    """
    The defining difference from DPO: ranking_loss_llm must not accept (or
    require) a reference policy -- it operates on raw policy log-probs only.
    """
    from ace_experiments import ranking_loss_llm
    import inspect

    sig = inspect.signature(ranking_loss_llm)
    assert "ref_model" not in sig.parameters


@pytest.mark.unit
def test_ranking_loss_prefers_higher_win_confidence():
    """Loss should shrink as the winner's log-prob rises relative to loser's."""
    from ace_experiments import ranking_loss_llm

    win_text, lose_text = "DO X1 = 1.0", "DO X1 = -1.0"

    model = _FakeCausalLMPolicy()
    # Force the offsets so winner starts BELOW loser, then improve it.
    model._text_offsets[win_text] = 0.0
    model._text_offsets[lose_text] = 0.5
    loss_before = ranking_loss_llm(model, scm_state=None, win_text=win_text, lose_text=lose_text)

    model._text_offsets[win_text] = 2.0  # winner now clearly more likely
    loss_after = ranking_loss_llm(model, scm_state=None, win_text=win_text, lose_text=lose_text)

    assert loss_after.item() < loss_before.item()


@pytest.mark.unit
def test_ranking_loss_is_differentiable():
    from ace_experiments import ranking_loss_llm

    model = _FakeCausalLMPolicy()
    loss = ranking_loss_llm(model, scm_state=None, win_text="DO X1 = 1.0", lose_text="DO X2 = -1.0")
    loss.backward()

    assert model.bias.grad is not None
    assert not torch.isnan(model.bias.grad)


# =============================================================================
# --policy_update CLI wiring
# =============================================================================

@pytest.mark.unit
def test_ace_script_registers_policy_update_choices():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "ace_experiments.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert "--policy_update" in result.stdout
    assert "dpo" in result.stdout
    assert "sft_best" in result.stdout
    assert "ranking" in result.stdout


@pytest.mark.unit
def test_policy_update_rejects_invalid_choice():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "ace_experiments.py", "--custom", "--policy_update", "not_a_real_mode"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower() or "invalid choice" in result.stdout.lower()
