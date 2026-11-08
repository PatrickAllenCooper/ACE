"""Prospective singleton-layout CPU controls, never Qwen/runtime qualification.

Run this file directly with .venv_test and ACE_UNPADDED_TRAINING_TEST_OUTPUT
pointing to an absent directory. All fit artifacts (including intentional
failures) survive there. Imports reuse helper definitions, not old test cases.
The 6/8/9-token commands below are fabricated IDs, not tokenizer certification.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import llm_intervention_canonical_scoring as scoring
import llm_intervention_training_engine as engine
import llm_intervention_training_plan as bridge
from llm_intervention_contract import canonical, digest
from llm_intervention_learning import (ActionBatch, pack_actions, preference_loss,
    reference_policy, snapshot_reference)
import test_llm_intervention_training_engine as helpers
import test_llm_intervention_training_plan as bridge_helpers


class LengthTokenizer(helpers.Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return super().apply_chat_template(messages, **kwargs) + [7, 8, 2]

    def encode(self, text, **kwargs):
        token = self.commands[text]
        return [token] * {3: 7, 4: 8, 5: 5}[token]

    def decode(self, ids, **kwargs):
        return next(c for c in self.commands if list(ids) == self.encode(c))


class TinyDecoder(helpers.TinyModel):
    """Two real checkpointed nonlinear layers and 64 adapter parameters."""
    def __init__(self):
        super().__init__()
        self.model.layers = nn.ModuleList([nn.Tanh(), nn.SiLU()])
        with torch.no_grad():
            self.q_proj.B.fill_(.02)
            self.v_proj.B.fill_(.03)
            w = torch.arange(24, dtype=torch.float32).reshape(12, 2)
            self.lm_head.weight.copy_(torch.sin(w * .7))
        self.calls = []
        self.retain_logits = False

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
        assert input_ids.device.type == 'cpu' and use_cache is False
        self.calls.append({'ids': input_ids.detach().clone(),
            'mask': attention_mask.detach().clone(), 'reference': self.q_proj.reference,
            'positions': logits_to_keep.clone() if isinstance(logits_to_keep, torch.Tensor) else None,
            'grad_enabled': torch.is_grad_enabled()})
        x = torch.stack((torch.sin(input_ids.float()), torch.cos(input_ids.float())), -1)
        x = (x * attention_mask[..., None]).to(torch.bfloat16)
        h = self.q_proj(x.cumsum(1)) + self.v_proj(x)
        for layer in self.model.layers:
            h = layer(h)
        if isinstance(logits_to_keep, torch.Tensor):
            h = h[:, logits_to_keep]
        logits = self.lm_head(h)
        if self.retain_logits and logits.requires_grad:
            logits.retain_grad()
            self.calls[-1]['logits'] = logits
        return SimpleNamespace(logits=logits)


def command_batch(lengths=(8, 9)):
    # Long enough to catch accidentally forwarding only selected action positions.
    prompt = [1, 2, 7, 8, 6] * 9 + [2, 1]
    commands = [[3 + row] * (length - 1) + [9] for row, length in enumerate(lengths)]
    return pack_actions(prompt, commands, 0, 128)


def full_singleton_oracle(model, batch):
    """Independent shifted masked CE reduction over full vocabulary positions."""
    scores = []
    for row, length in enumerate(batch.attention_mask.sum(1).tolist()):
        logits = model(input_ids=batch.input_ids[row:row+1, :length],
            attention_mask=batch.attention_mask[row:row+1, :length], use_cache=False).logits
        labels = batch.labels[row:row+1, 1:length]
        mask = labels != -100
        values = F.log_softmax(logits[:, :-1], -1).gather(-1,
            labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        scores.append(values.masked_fill(~mask, 0).sum(1))
    return torch.cat(scores)


def make_records():
    result = []
    for index in range(2):
        old = helpers.prepared(index)
        result.append(engine.prepare_record(record_id=old['record_id'],
            raw=old['raw_record_utf8'].encode(), record_sha256=old['record_sha256'],
            supervision_receipt_sha256=old['supervision_receipt_sha256'], scope='artificial',
            tokenizer=LengthTokenizer(), tokenizer_sha256='2'*64,
            numeric_recipe=engine.screen_numeric_recipe()))
    return tuple(result)


def plan_arguments(records):
    return dict(records=records, record_pins={r['record_id']: digest(r) for r in records},
        schedule=tuple({'stage': stage, 'record_id': 'r'+str(i),
            'pair_index': None if stage == 'sft' else 0}
            for stage in ('sft', 'dpo') for i in range(2)),
        numeric_recipe=engine.screen_numeric_recipe(), scope='artificial',
        execution_kind='artificial_control', expected_adapter_shapes={name+suffix: shape
            for name in ('q_proj', 'v_proj') for suffix, shape in (('.A', [8, 2]), ('.B', [2, 8]))},
        max_trainable_parameters=64, checkpoint_steps=(), outer_provenance_receipt_sha256='3'*64)


class UnpaddedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(os.environ['ACE_UNPADDED_TRAINING_TEST_OUTPUT'])
        cls.root.mkdir(exist_ok=False)
        cls.records = make_records()

    def setUp(self):
        self.out = self.root/self._testMethodName
        self.out.mkdir(exist_ok=False)
        self.gpu_guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        self.gpu_guard.start()
        self.addCleanup(self.gpu_guard.stop)

    def test_unequal_both_orders_equal_and_stop_keep_all_inputs_eos_and_order(self):
        for lengths in ((8, 9), (9, 8), (8, 8), (6, 9)):
            for selected in (True, False):
                with self.subTest(lengths=lengths, selected=selected):
                    batch, model = command_batch(lengths), TinyDecoder().eval()
                    expected = full_singleton_oracle(copy.deepcopy(model), batch)
                    observed = scoring.sequence_log_probs_unpadded(model, batch, selected=selected)
                    # Full-position masked summation has a different FP32 reduction
                    # length. Mathematical agreement is not bitwise certification.
                    torch.testing.assert_close(observed, expected, rtol=3e-7, atol=1e-6)
                    self.assertEqual(len(model.calls), 2)
                    for row, call in enumerate(model.calls):
                        length = 47 + lengths[row]
                        self.assertEqual(call['ids'].shape, (1, length))
                        self.assertTrue(torch.equal(call['ids'][0], batch.input_ids[row, :length]))
                        self.assertTrue(torch.all(call['mask'] == 1))
                        self.assertEqual(call['ids'][0, -1].item(), 9)
                        if selected:
                            self.assertEqual(call['positions'].tolist(), list(range(46, length-1)))
                    reversed_batch = ActionBatch(*(x.flip(0) for x in
                        (batch.input_ids, batch.attention_mask, batch.labels)))
                    reverse = scoring.sequence_log_probs_unpadded(model, reversed_batch, selected=selected)
                    torch.testing.assert_close(reverse, observed.flip(0), rtol=0, atol=0)
                    other_path = scoring.sequence_log_probs_unpadded(model, batch, selected=not selected)
                    torch.testing.assert_close(other_path, observed, rtol=0, atol=0)

    def test_supervised_row_normalization_and_gradients_match_full_oracle(self):
        batch = command_batch()
        model, oracle = TinyDecoder().eval(), TinyDecoder().eval()
        loss = scoring.supervised_loss_unpadded(model, batch, selected=True)
        expected = -(full_singleton_oracle(oracle, batch)/torch.tensor([8, 9])).mean()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward(); expected.backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), oracle.named_parameters()):
            self.assertEqual(name, other_name)
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            else:
                self.assertIsNone(p.grad)

    def test_dpo_gradients_reach_both_completions_with_fixed_reference(self):
        for lengths in ((8, 9), (9, 8), (8, 8)):
            with self.subTest(lengths=lengths):
                model, batch = TinyDecoder().eval(), command_batch(lengths)
                snapshot_reference(model)
                refs = {n: b.clone() for n, b in model.named_buffers() if 'reference_' in n}
                with torch.no_grad(): model.q_proj.B.add_(.05)
                oracle = copy.deepcopy(model)
                with reference_policy(model):
                    reference = scoring.sequence_log_probs_unpadded(model, batch, selected=True)
                self.assertFalse(reference.requires_grad)
                model.calls.clear(); model.retain_logits = True
                policy = scoring.sequence_log_probs_unpadded(model, batch, selected=True)
                policy.retain_grad()
                # Even an accidentally grad-enabled supplied reference must be detached by DPO.
                supplied_reference = reference.clone().requires_grad_()
                loss = preference_loss(policy.reshape(1, 2), supplied_reference.reshape(1, 2))
                expected = preference_loss(full_singleton_oracle(oracle, batch).reshape(1, 2), reference.reshape(1, 2))
                loss.backward(); expected.backward()
                self.assertLess(policy.grad[0].item(), 0)
                self.assertGreater(policy.grad[1].item(), 0)
                self.assertIsNone(supplied_reference.grad)
                for call in model.calls:
                    self.assertGreater(call['logits'].grad.abs().sum().item(), 0)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, dict(oracle.named_parameters())[name].grad, rtol=0, atol=0)
                    else: self.assertIsNone(p.grad)
                with torch.no_grad(): model.v_proj.B.add_(.1)
                with reference_policy(model):
                    after = scoring.sequence_log_probs_unpadded(model, batch, selected=True)
                torch.testing.assert_close(after, reference, rtol=0, atol=0)
                for name, b in model.named_buffers():
                    if name in refs: torch.testing.assert_close(b, refs[name], rtol=0, atol=0)

    def test_bad_second_row_rejects_before_any_decoder_forward(self):
        for kind in ('attention_hole', 'left_padding', 'nonbinary', 'padding_label',
                     'label_identity', 'first_label', 'no_labels', 'negative_id', 'dtype'):
            with self.subTest(kind=kind):
                batch = command_batch((9, 8))
                ids, att, lab = [x.clone() for x in (batch.input_ids, batch.attention_mask, batch.labels)]
                if kind == 'attention_hole': att[1, 2] = 0
                elif kind == 'left_padding': att[1, 0] = 0
                elif kind == 'nonbinary': att[1, 2] = 2
                elif kind == 'padding_label': lab[1, -1] = ids[1, -1]
                elif kind == 'label_identity': lab[1, -2] = 3
                elif kind == 'first_label': lab[1, 0] = ids[1, 0]
                elif kind == 'no_labels': lab[1] = -100
                elif kind == 'negative_id': ids[1, 1] = -1
                elif kind == 'dtype': att = att.float()
                model = TinyDecoder()
                for function in (scoring.sequence_log_probs_unpadded, scoring.supervised_loss_unpadded):
                    with self.assertRaises(ValueError):
                        function(model, ActionBatch(ids, att, lab), selected=True)
                    self.assertEqual(model.calls, [])

    def test_explicit_recipe_modes_admit_and_mismatches_reject_before_fit(self):
        for mode, recipe in (('qwen_gpu09', engine.gpu09_numeric_recipe()),
                             ('qwen_screen_singletons', engine.screen_numeric_recipe())):
            args = dict(plan_arguments(self.records), execution_kind=mode, numeric_recipe=recipe)
            plan = engine.prepare_training_plan(**args)
            self.assertEqual(plan['execution_kind'], mode)
            wrong = engine.screen_numeric_recipe() if mode == 'qwen_gpu09' else engine.gpu09_numeric_recipe()
            with self.assertRaisesRegex(ValueError, 'disagree'):
                engine.prepare_training_plan(**dict(args, numeric_recipe=wrong))
            plan['numeric_recipe'] = wrong
            model = TinyDecoder()
            with patch.object(engine, '_inventory', side_effect=AssertionError('admitted invalid mode')):
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    engine.fit(model=model, plan=plan, plan_sha256=digest(plan), output=self.out/mode)
            self.assertFalse((self.out/mode).exists())
        with self.assertRaisesRegex(ValueError, 'disagree'):
            engine.prepare_training_plan(**dict(plan_arguments(self.records), execution_kind='implicit'))

    def test_four_real_steps_continuous_optimizer_reference_routes_and_reload(self):
        plan = engine.prepare_training_plan(**plan_arguments(self.records))
        model = TinyDecoder()
        with patch.object(engine, 'sequence_log_probs_canonical', side_effect=AssertionError('old route')), \
             patch.object(engine, 'supervised_loss_canonical', side_effect=AssertionError('old route')), \
             patch.object(torch.optim, 'AdamW', wraps=torch.optim.AdamW) as optimizer_factory:
            result = engine.fit(model=model, plan=plan, plan_sha256=digest(plan), output=self.out/'fit')
        self.assertEqual(optimizer_factory.call_count, 1)
        out = self.out/'fit'
        self.assertEqual((result['completed_steps'], result['sft_steps'], result['dpo_steps']), (4, 2, 2))
        self.assertEqual(result['base_before'], result['base_after'])
        self.assertEqual(result['projection_before'], result['projection_after'])
        self.assertFalse(result['provenance_qualified'])
        self.assertTrue(result['reference_scores_unchanged'])
        self.assertEqual(result['reference_pair_count'], 2)
        self.assertEqual(result['checkpoint_usage']['recompute_calls'], 12)
        self.assertEqual(result['checkpoint_usage']['forward_calls'], 12)
        events = [json.loads((out/f'step-{i:05d}.complete.json').read_text()) for i in range(4)]
        self.assertEqual([e['rows'] for e in events], [1, 1, 2, 2])
        self.assertEqual([e['checkpoint_recompute_delta'] for e in events], [2, 2, 4, 4])
        self.assertEqual([e['action_tokens'] for e in events], [8, 9, 17, 17])
        self.assertEqual([e['optimizer_step_returned'] for e in events], [True]*4)
        for stem, steps in (('sft', 2), ('final', 4)):
            state = torch.load(out/(stem+'.optimizer.pt'), weights_only=True)
            self.assertEqual(len(state['state']), 4)
            self.assertEqual({int(s['step']) for s in state['state'].values()}, {steps})
            self.assertTrue(all(s['exp_avg'].abs().sum() > 0 and s['exp_avg_sq'].sum() > 0
                                for s in state['state'].values()))
        report = json.loads((out/'reference-scores.json').read_text())
        self.assertEqual((report['unique_pair_score_calls'], report['decoder_forwards']), (2, 4))
        self.assertEqual(report['scoring_layout'], 'ordered_unpadded_singletons')
        self.assertNotIn('unique_pair_forwards', report)
        # SFT(2), initial reference(4), DPO(4), final reference(4) decoder calls.
        self.assertEqual([c['reference'] for c in model.calls], [False]*2+[True]*4+[False]*4+[True]*4)
        self.assertEqual([c['ids'].shape[1] for c in model.calls], [13, 14]+[13, 14]*6)
        self.assertTrue(all(c['ids'].shape[0] == 1 and torch.all(c['mask'] == 1) for c in model.calls))
        self.assertTrue(all(not c['grad_enabled'] for c in model.calls if c['reference']))
        sft = torch.load(out/'sft.adapter.pt', weights_only=True)
        references = torch.load(out/'reference.pt', weights_only=True)
        for name, value in references.items():
            torch.testing.assert_close(value, sft[name.replace('.reference_', '.')], rtol=0, atol=0)
        reload = TinyDecoder().eval()
        final = torch.load(out/'final.adapter.pt', weights_only=True)
        with torch.no_grad():
            for name, value in final.items(): dict(reload.named_parameters())[name].copy_(value)
            for name, value in references.items(): dict(reload.named_buffers())[name].copy_(value)
        for record in self.records:
            batch = pack_actions(record['prompt_ids'], [record['commands'][c] for c in record['strict_pairs'][0]], 0, 8192)
            with torch.no_grad():
                live = scoring.sequence_log_probs_unpadded(model, batch, selected=True)
                restored = scoring.sequence_log_probs_unpadded(reload, batch, selected=True)
            torch.testing.assert_close(live, restored, rtol=0, atol=0)
            with reference_policy(reload):
                ref = scoring.sequence_log_probs_unpadded(reload, batch, selected=True)
            torch.testing.assert_close(ref, torch.tensor(report['scores'][canonical([record['record_id'], 0])]), rtol=0, atol=0)

    def test_missing_one_singleton_recompute_rejects_before_dpo_update(self):
        # Under-report actual recomputation only for pairs; keep real scoring/backward.
        controls = []
        real_control = engine.DecoderCheckpointing
        def control(*args, **kwargs):
            value = real_control(*args, **kwargs); controls.append(value); return value
        def score(model, batch, **kwargs):
            value = scoring.sequence_log_probs_unpadded(model, batch, **kwargs)
            if value.requires_grad:
                def undercount(gradient):
                    controls[0].recompute_calls -= 2
                    return gradient
                value.register_hook(undercount)
            return value
        plan = engine.prepare_training_plan(**plan_arguments(self.records))
        with patch.object(engine, 'DecoderCheckpointing', control), patch.object(engine, 'sequence_log_probs_unpadded', score):
            with self.assertRaisesRegex(ValueError, 'missing recomputation'):
                engine.fit(model=TinyDecoder(), plan=plan, plan_sha256=digest(plan), output=self.out/'fit')
        failure = json.loads((self.out/'fit/failure.json').read_text())
        self.assertEqual(len(failure['completed_steps']), 2)
        self.assertFalse(failure['failed_attempt']['optimizer_step_started'])
        self.assertFalse((self.out/'fit/step-00002.optimizer-attempt.json').exists())
        self.assertFalse(controls[0].enabled)

    def test_collection_bridge_explicit_layout_and_unchanged_default(self):
        root = Path(__file__).resolve().parents[2]/'results/ace_llm_intervention_training_collection_preparation_20261107/attempt02/fixtures/test_two_world_complete_inventory_drives_exposure_and_exclusive_intent/collection'
        args = {}
        for field, name in (('terminal', 'terminal.json'), ('inventory', 'inventory.json'), ('collection_plan', 'plan.json')):
            raw = (root/name).read_bytes()
            args[field+'_raw'], args[field+'_pin'] = raw, hashlib.sha256(raw).hexdigest()
        inv = json.loads(args['inventory_raw'])
        args.update(records={r['record_id']: (root/r['path']/'record.json').read_bytes() for r in inv['records']},
            receipts={r['record_id']: (root/r['path']/'receipt.json').read_bytes() for r in inv['records']},
            expected_roster=tuple(inv['record_roster']), expected_scope='artificial', tokenizer_pin='a'*64,
            outer_provenance_pin='b'*64)
        for mode in (None, 'qwen_screen_singletons'):
            extra = {} if mode is None else {'execution_kind': mode}
            result = bridge.prepare_complete_collection(**args, tokenizer=bridge_helpers.Tokenizer(), **extra)
            plan = result['training_plan']
            self.assertEqual(plan['execution_kind'], mode or 'qwen_gpu09')
            self.assertEqual(plan['numeric_recipe'], engine.screen_numeric_recipe() if mode else engine.gpu09_numeric_recipe())
            self.assertEqual((plan['sft_steps'], plan['dpo_steps']), (4, 2))
            self.assertFalse(result['outer_execution_qualified'])
        tokenizer = bridge_helpers.Tokenizer()
        with self.assertRaisesRegex(ValueError, 'execution kind'):
            bridge.prepare_complete_collection(**args, tokenizer=tokenizer, execution_kind='artificial_control')
        self.assertEqual(tokenizer.calls, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
