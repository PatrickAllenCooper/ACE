"""Small arithmetic/torch fixtures. No pretrained model or scientific worlds."""
import math
from types import SimpleNamespace
import unittest
import torch
from torch import nn

from llm_intervention_learning import (
    ActionBatch, LoRALinear, install_adapters, pack_actions, preference_loss,
    reference_policy, sequence_log_probs, snapshot_reference, supervised_loss,
)

torch.set_num_threads(1)


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.embed = nn.Embedding(9, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 9, bias=False)

    def forward(self, input_ids, attention_mask, use_cache):
        # Artificial causal dependence on ALL preceding tokens.
        x = (self.embed(input_ids) * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        return SimpleNamespace(logits=self.v_proj(torch.tanh(self.q_proj(x))))


class LearningTests(unittest.TestCase):
    def test_action_only_mask_and_no_silent_truncation(self):
        b = pack_actions([1, 2], [[3], [4, 5]], 0, 4)
        self.assertEqual(b.labels.tolist(), [[-100, -100, 3, -100], [-100, -100, 4, 5]])
        self.assertEqual(b.attention_mask.tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])
        for prompt, acts, maximum in (([1, 2], [[3, 4]], 3), ([], [[1]], 3), ([1], [[]], 3)):
            with self.assertRaises(ValueError):
                pack_actions(prompt, acts, 0, maximum)

    def test_independent_shifted_likelihood_oracle_and_state_conditioning(self):
        model = TinyCausalModel()
        b = pack_actions([1, 2], [[3, 4]], 0, 8)
        scores = sequence_log_probs(model, b)
        logits = model(b.input_ids, b.attention_mask, False).logits
        oracle = sum(logits[0, j-1, target] - torch.logsumexp(logits[0, j-1], 0)
                     for j, target in ((2, 3), (3, 4)))
        torch.testing.assert_close(scores[0], oracle)
        other = sequence_log_probs(model, pack_actions([6, 7], [[3, 4]], 0, 8))
        self.assertNotAlmostEqual(float(scores[0]), float(other[0]), places=5)

    def test_padding_does_not_change_action_scores(self):
        model = TinyCausalModel()
        single = sequence_log_probs(model, pack_actions([1, 2], [[3]], 0, 8))[0]
        batch = sequence_log_probs(model, pack_actions([1, 2], [[3], [4, 5, 6]], 0, 8))[0]
        torch.testing.assert_close(single, batch)

    def test_preference_reference_and_gradient_oracle(self):
        policy = torch.tensor([[-2., -4.]], requires_grad=True)
        reference = torch.tensor([[-3., -4.]], requires_grad=True)
        loss = preference_loss(policy, reference, beta=.5)
        self.assertAlmostEqual(float(loss), math.log1p(math.exp(-.5)), places=6)
        loss.backward()
        self.assertLess(float(policy.grad[0, 0]), 0.)
        self.assertGreater(float(policy.grad[0, 1]), 0.)
        self.assertIsNone(reference.grad)
        with self.assertRaises(ValueError):
            preference_loss(torch.tensor([[float("nan"), 1.]]), reference)

    def test_adapters_train_while_base_and_reference_stay_fixed(self):
        model = TinyCausalModel()
        b = pack_actions([1, 2], [[3, 4]], 0, 8)
        original = sequence_log_probs(model, b).detach().clone()
        names = install_adapters(model, rank=2, alpha=4.)
        self.assertEqual(names, ("q_proj", "v_proj"))
        torch.testing.assert_close(sequence_log_probs(model, b), original)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=.1)
        before = [p.detach().clone() for p in trainable]
        loss = supervised_loss(model, b)
        loss.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in trainable))
        optimizer.step()
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, trainable)))
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(p, frozen[name]))
        with reference_policy(model):
            torch.testing.assert_close(sequence_log_probs(model, b), original)
        self.assertEqual(snapshot_reference(model), 2)
        new_reference = sequence_log_probs(model, b).detach().clone()
        optimizer.zero_grad()
        supervised_loss(model, b).backward()
        optimizer.step()
        with reference_policy(model):
            torch.testing.assert_close(sequence_log_probs(model, b), new_reference)

    def test_reference_restores_after_exception_and_install_is_once_only(self):
        model = TinyCausalModel()
        install_adapters(model, rank=2)
        with self.assertRaises(RuntimeError):
            with reference_policy(model):
                raise RuntimeError("artificial interruption")
        self.assertFalse(any(x.reference for x in model.modules() if isinstance(x, LoRALinear)))
        with self.assertRaises(ValueError):
            install_adapters(model)

    def test_dropout_free_scores_preserve_gradients_and_module_modes(self):
        class DropoutModel(TinyCausalModel):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(.9)

            def forward(self, input_ids, attention_mask, use_cache):
                x = self.embed(input_ids).cumsum(dim=1)
                return SimpleNamespace(logits=self.v_proj(self.dropout(torch.tanh(self.q_proj(x)))))
        model = DropoutModel()
        install_adapters(model, rank=2)
        model.train()
        model.embed.eval()
        original_modes = [m.training for m in model.modules()]
        b = pack_actions([1, 2], [[3, 4]], 0, 8)
        policy = sequence_log_probs(model, b)
        repeated = sequence_log_probs(model, b)
        with reference_policy(model):
            reference = sequence_log_probs(model, b)
        torch.testing.assert_close(policy, repeated, rtol=0, atol=0)
        torch.testing.assert_close(policy, reference, rtol=0, atol=0)
        (-policy.mean()).backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in model.parameters() if p.requires_grad))
        self.assertEqual(original_modes, [m.training for m in model.modules()])

    def test_adapter_explicit_whole_model_cast(self):
        model = TinyCausalModel()
        install_adapters(model, rank=2)
        model.double()
        b = pack_actions([1], [[2]], 0, 8)
        self.assertTrue(torch.isfinite(sequence_log_probs(model, b)).all())


if __name__ == "__main__":
    unittest.main()
