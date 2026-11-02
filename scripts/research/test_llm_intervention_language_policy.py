"""Artificial adapter and task-decoding tests, no pretrained model or training."""
import copy
import hashlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from llm_intervention_contract import canonical
from llm_intervention_language_policy import (restore_adapter, prepare_task_tokens,
    training_action_batch, generate_task_decision)
from llm_intervention_learning import LoRALinear
from test_llm_intervention_task import example


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 2, bias=False)
        self.v_proj = nn.Linear(3, 4, bias=True)


def encoded(tensors):
    stream = io.BytesIO()
    torch.save(tensors, stream)
    raw = stream.getvalue()
    return raw, hashlib.sha256(raw).hexdigest()


def adapter():
    return {'q_proj.A': torch.ones(2, 3), 'q_proj.B': torch.full((2, 2), .5),
            'v_proj.A': torch.full((2, 3), .25), 'v_proj.B': torch.ones(4, 2)}


def load(model, tensors, **overrides):
    raw, sha = encoded(tensors)
    return restore_adapter(model, raw, **dict(adapter_sha256=sha, training_receipt_sha256='a'*64,
        expected_projection_names=('q_proj', 'v_proj'), rank=2, alpha=4., **overrides))


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c)+2 for c in text]
    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return ''.join(chr(i-2) for i in ids)
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return self.encode('USER:'+messages[0]['content']+'ASSISTANT:')


class FakeGenerator(nn.Module):
    def __init__(self, mode='good'):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(max_position_embeddings=20000)
        self.mode = mode
        self.calls = 0
    def generate(self, *, input_ids, attention_mask, generation_config,
                 prefix_allowed_tokens_fn, logits_processor, use_model_defaults):
        self.calls += 1
        assert not torch.is_grad_enabled() and not self.training
        assert generation_config.num_beams == 1 and not generation_config.do_sample
        assert use_model_defaults is False
        assert torch.all(attention_mask == 1)
        sequence = input_ids[0].tolist()
        for _ in range(generation_config.max_new_tokens):
            current = torch.tensor(sequence)
            allowed = prefix_allowed_tokens_fn(0, current)
            scores = torch.zeros(1, 512)
            if self.mode == 'nan':
                scores[0, allowed[0]] = float('nan')
            logits_processor[0](current[None], scores)
            sequence.append(min(allowed))
            if sequence[-1] == generation_config.eos_token_id:
                break
        if self.mode == 'bad_prefix':
            sequence[0] += 1
        if self.mode == 'incomplete':
            sequence.pop()
        return torch.tensor([sequence])


class LanguagePolicyTests(unittest.TestCase):
    def test_actual_configuration_helper_does_not_restore_model_defaults(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        class MergingFake(FakeGenerator):
            def __init__(self):
                super().__init__()
                self.generation_config = GenerationConfig(do_sample=True, num_beams=3,
                    forced_bos_token_id=77, forced_eos_token_id=78, suppress_tokens=[99],
                    repetition_penalty=1.8)
            def generate(self, *, generation_config, use_model_defaults=None, **kwargs):
                prepared, model_kwargs = GenerationMixin._prepare_generation_config(
                    self, generation_config, use_model_defaults=use_model_defaults)
                self.prepared = prepared
                assert model_kwargs == {}
                return super().generate(generation_config=prepared, use_model_defaults=use_model_defaults, **kwargs)
        model = MergingFake()
        inherited, _ = GenerationMixin._prepare_generation_config(model, GenerationConfig(), use_model_defaults=True)
        self.assertTrue(inherited.do_sample)
        self.assertEqual(inherited.num_beams, 3)
        generate_task_decision(model, Tokenizer(), example(), max_tokens=20000, max_new_tokens=32)
        self.assertFalse(model.prepared.do_sample)
        self.assertEqual(model.prepared.num_beams, 1)
        self.assertIsNone(model.prepared.forced_bos_token_id)
        self.assertIsNone(model.prepared.forced_eos_token_id)
        self.assertIsNone(model.prepared.suppress_tokens)
        self.assertEqual(model.prepared.repetition_penalty, 1.)
        self.assertTrue(model.generation_config.do_sample)

    def test_normalizing_tokenizer_cannot_repair_noncanonical_training_label(self):
        class NormalizingTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                if text.startswith(' '):
                    text = text.strip()
                return super().encode(text, add_special_tokens=add_special_tokens)
        with self.assertRaises(ValueError):
            training_action_batch(example(), NormalizingTokenizer(), (' {"action_id":"high"} ',),
                                  max_tokens=20000, max_new_tokens=32)

    def test_restore_exact_independent_linear_oracle_and_frozen_state(self):
        model = SmallModel()
        x = torch.tensor([[1., 2., 3.]])
        old = {n: p.detach().clone() for n, p in model.named_parameters()}
        q, v = model.q_proj(x).detach(), model.v_proj(x).detach()
        receipt = load(model, adapter())
        torch.testing.assert_close(model.q_proj(x), q+12.)
        torch.testing.assert_close(model.v_proj(x), v+6.)
        for n, p in old.items():
            module, suffix = n.split('.')
            torch.testing.assert_close(getattr(getattr(model, module).base, suffix), p)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)
        torch.testing.assert_close(model.q_proj.reference_A, model.q_proj.A)
        self.assertEqual(receipt['adapter_parameters'], 24)
        self.assertFalse(receipt['base_and_training_provenance_qualified'])

    def test_rejections_precede_model_mutation(self):
        variants = [lambda d: d.pop('q_proj.A'), lambda d: d.update(extra=torch.zeros(1)),
            lambda d: d.update({'q_proj.A': torch.ones(2, 4)}),
            lambda d: d.update({'q_proj.A': torch.ones(2, 3, dtype=torch.float64)}),
            lambda d: d['q_proj.A'].fill_(float('nan'))]
        for modify in variants:
            model = SmallModel()
            before = copy.deepcopy(model.state_dict())
            tensors = adapter(); modify(tensors)
            with self.assertRaises(ValueError):
                load(model, tensors)
            self.assertNotIsInstance(model.q_proj, LoRALinear)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            for name, value in before.items():
                torch.testing.assert_close(model.state_dict()[name], value)
        raw, sha = encoded(adapter())
        model = SmallModel()
        with patch('torch.load', side_effect=AssertionError('decode before pin')):
            with self.assertRaises(ValueError):
                restore_adapter(model, raw, adapter_sha256='0'*64, training_receipt_sha256='a'*64,
                    expected_projection_names=('q_proj', 'v_proj'), rank=2)
        with self.assertRaises(ValueError):
            restore_adapter(model, raw, adapter_sha256=sha, training_receipt_sha256='a'*64,
                expected_projection_names=('q_proj',), rank=2)

    def test_reload_preserves_cpu_rng_and_never_seeds_accelerators(self):
        model = SmallModel()
        before = torch.get_rng_state().clone()
        with patch('torch.manual_seed', side_effect=AssertionError('global seed')), \
             patch('torch.cuda.manual_seed_all', side_effect=AssertionError('cuda seed')), \
             patch('torch.mps.manual_seed', side_effect=AssertionError('mps seed')):
            load(model, adapter())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_same_chat_prefix_training_and_inference_and_all_legal_actions(self):
        tokenizer = Tokenizer(); e = example()
        grammar, meta = prepare_task_tokens(e, tokenizer, max_tokens=20000, max_new_tokens=32)
        decoded = {tokenizer.decode(x[:-1]) for x in grammar.completion_ids}
        self.assertEqual(decoded, {'{"action_id":"low"}', '{"action_id":"high"}', '{"stop":true}'})
        batch, training = training_action_batch(e, tokenizer,
            ('{"action_id":"high"}', '{"stop":true}'), max_tokens=20000, max_new_tokens=32)
        self.assertEqual(meta, training)
        n = len(grammar.prompt_ids)
        self.assertEqual(batch.input_ids[0, :n].tolist(), list(grammar.prompt_ids))
        self.assertTrue(torch.all(batch.labels[:, :n] == -100))
        self.assertEqual(batch.labels[0, batch.attention_mask[0].sum()-1].item(), 1)
        with self.assertRaises(ValueError):
            training_action_batch(e, tokenizer, ('{"action_id":"unknown"}',),
                                  max_tokens=20000, max_new_tokens=32)
        with self.assertRaises(ValueError):
            prepare_task_tokens(e, tokenizer, max_tokens=n+31, max_new_tokens=32)
        with self.assertRaises(ValueError):
            prepare_task_tokens(e, tokenizer, max_tokens=20000, max_new_tokens=2)

    def test_bound_generation_validates_prefix_eos_and_finite_scores(self):
        e = example(); before = canonical(e)
        model = FakeGenerator()
        completion, receipt = generate_task_decision(model, Tokenizer(), e, max_tokens=20000, max_new_tokens=32)
        self.assertEqual(completion, '{"action_id":"high"}')
        self.assertEqual(receipt['sequence_ids'][-1], 1)
        self.assertEqual(receipt['new_training_updates'], 0)
        self.assertTrue(model.training)
        self.assertEqual(canonical(e), before)
        for mode in ('bad_prefix', 'incomplete', 'nan'):
            with self.assertRaises(ValueError):
                generate_task_decision(FakeGenerator(mode), Tokenizer(), e, max_tokens=20000, max_new_tokens=32)
        small = FakeGenerator(); small.config.max_position_embeddings = 100
        with self.assertRaises(ValueError):
            generate_task_decision(small, Tokenizer(), e, max_tokens=20000, max_new_tokens=32)
        self.assertEqual(small.calls, 0)

    def test_terminal_or_budget_state_restricts_commands_before_any_execution(self):
        e = example(); e['state']['terminal'] = True
        completion, _ = generate_task_decision(FakeGenerator(), Tokenizer(), e, max_tokens=20000, max_new_tokens=32)
        self.assertEqual(completion, '{"stop":true}')
        e = example()
        e['state']['remaining_units'] = 1
        grammar, _ = prepare_task_tokens(e, Tokenizer(), max_tokens=20000, max_new_tokens=32)
        self.assertEqual({Tokenizer().decode(x[:-1]) for x in grammar.completion_ids},
                         {'{"action_id":"low"}', '{"stop":true}'})


if __name__ == '__main__':
    unittest.main()
