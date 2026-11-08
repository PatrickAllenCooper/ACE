"""New bridge checks on saved artificial collection bytes, no new responses."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import unittest

import llm_intervention_training_plan as bridge


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def __init__(self):
        self.calls = 0
    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return [100, 101]  # Artificial tokenizer, never actual Qwen preparation.
    def encode(self, text, **kwargs):
        return [ord(c)+2 for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i-2) for i in ids)


def pin(raw):
    return hashlib.sha256(raw).hexdigest()


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]/'results/ace_llm_intervention_training_collection_preparation_20261107/attempt02/fixtures/test_two_world_complete_inventory_drives_exposure_and_exclusive_intent/collection'
        self.args = {}
        for field, name in [('terminal', 'terminal.json'), ('inventory', 'inventory.json'),
                            ('collection_plan', 'plan.json')]:
            raw = (root/name).read_bytes()
            self.args[field+'_raw'], self.args[field+'_pin'] = raw, pin(raw)
        inventory = json.loads(self.args['inventory_raw'])
        self.args.update(records={r['record_id']: (root/r['path']/'record.json').read_bytes()
                                  for r in inventory['records']},
            receipts={r['record_id']: (root/r['path']/'receipt.json').read_bytes()
                      for r in inventory['records']}, expected_roster=tuple(inventory['record_roster']),
            expected_scope='artificial', tokenizer=Tokenizer(), tokenizer_pin='a'*64,
            outer_provenance_pin='b'*64)

    def change(self, args, field, update):
        value = json.loads(args[field+'_raw']);update(value)
        raw = (bridge.canonical(value)+'\n').encode()
        args[field+'_raw'], args[field+'_pin'] = raw, pin(raw)

    def rejected_without_tokenization(self, args):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            bridge.prepare_complete_collection(**args)
        self.assertEqual(args['tokenizer'].calls, 0)

    def test_complete_saved_batch_preserves_all_examples_and_pairs(self):
        result = bridge.prepare_complete_collection(**self.args)
        plan, exposure = result['training_plan'], result['exposure']
        self.assertEqual((plan['sft_steps'], plan['dpo_steps']), (4, 2))
        self.assertEqual([s['pair_index'] for s in plan['schedule'][4:]], [0, 1])
        self.assertEqual(exposure['counts']['tied_pairs'], 2)
        self.assertEqual(exposure['counts']['unselected_pairs'], 2)
        self.assertEqual(exposure['counts']['records_without_strict_pairs'], 2)
        self.assertEqual([r['record_id'] for r in plan['records']], list(self.args['expected_roster']))
        self.assertEqual(plan['declared_adapter_parameters'], 1146880)
        self.assertEqual(len(plan['expected_adapter_shapes']), 112)
        self.assertEqual(sum(math.prod(s) for s in bridge.qwen_adapter_shapes().values()), 1146880)
        self.assertEqual(result['new_environment_responses'], 0)
        self.assertFalse(result['outer_execution_qualified'])
        self.assertEqual(self.args['tokenizer'].calls, 4)
        if os.environ.get('ACE_BRIDGE_TEST_OUTPUT'):
            out=Path(os.environ['ACE_BRIDGE_TEST_OUTPUT'])
            with (out/'artificial-prepared-plan.json').open('x') as f:
                f.write(bridge.canonical(result)+'\n')

    def test_partial_or_reordered_batch_rejects_before_tokenizer(self):
        for key in ['records', 'receipts', 'expected_roster']:
            with self.subTest(key=key):
                args=copy.deepcopy(self.args)
                if key == 'expected_roster':args[key]=tuple(reversed(args[key]))
                else:args[key].pop(args['expected_roster'][-1])
                self.rejected_without_tokenization(args)

    def test_failed_unknown_or_mismatched_world_accounting_rejects(self):
        changes=[lambda t:t.update(status='failed'), lambda t:t.update(exposure_ready=False),
            lambda t:t['accounting'].update(unknown_return_attempts=1),
            lambda t:t['accounting'].update(known_returns=False),
            lambda t:t['worlds'][0].update(world_id=999),
            lambda t:t['worlds'][0]['accounting'].update(known_returns=999)]
        for i, change in enumerate(changes):
            with self.subTest(i=i):
                args=copy.deepcopy(self.args);self.change(args, 'terminal', change)
                self.rejected_without_tokenization(args)

    def test_record_receipt_and_independent_pin_changes_reject(self):
        for mode in ['record', 'receipt', 'terminal_pin', 'inventory_binding']:
            with self.subTest(mode=mode):
                args=copy.deepcopy(self.args);rid=args['expected_roster'][0]
                if mode=='record':args['records'][rid]+=b' '
                elif mode=='receipt':args['receipts'][rid]+=b' '
                elif mode=='terminal_pin':args[mode]='0'*64
                else:self.change(args,'terminal',lambda t:t.update(inventory_sha256='0'*64))
                self.rejected_without_tokenization(args)

    def test_no_original_input_mutation(self):
        before=copy.deepcopy({k:v for k,v in self.args.items() if k!='tokenizer'})
        result=bridge.prepare_complete_collection(**self.args)
        result['exposure']['schedule'].clear()
        self.assertEqual(before, {k:v for k,v in self.args.items() if k!='tokenizer'})

    def test_rebound_bytes_cannot_hide_inconsistent_semantic_receipts(self):
        for mode in ['envelope', 'task']:
            with self.subTest(mode=mode):
                args=copy.deepcopy(self.args);rid=args['expected_roster'][0]
                rec=json.loads(args['records'][rid]);receipt=json.loads(args['receipts'][rid])
                if mode=='envelope':rec['envelope']['state']['objective'] += ' changed'
                else:
                    receipt['task_sha256']='0'*64
                    args['receipts'][rid]=(bridge.canonical(receipt)+'\n').encode()
                    rec['supervision_receipt_sha256']=pin(args['receipts'][rid])
                args['records'][rid]=(bridge.canonical(rec)+'\n').encode()
                def rebind(i):
                    i['record_pins'][rid]=pin(args['records'][rid])
                    i['supervision_receipt_pins'][rid]=pin(args['receipts'][rid])
                    i['records'][0].update(record_sha256=i['record_pins'][rid],
                        supervision_receipt_sha256=i['supervision_receipt_pins'][rid])
                self.change(args,'inventory',rebind)
                self.change(args,'terminal',lambda t:t.update(inventory_sha256=args['inventory_pin']))
                self.rejected_without_tokenization(args)

    def test_world_accounting_flags_and_balanced_count_errors_reject(self):
        def offset(t):
            t['worlds'][0]['accounting']['known_returns'] += 1
            t['worlds'][1]['accounting']['known_returns'] -= 1
        changes=[lambda t:t['worlds'][0]['accounting'].update(accounting_complete=False),
            lambda t:t['worlds'][0]['accounting'].update(accounting_errors=['missing journal']),offset]
        for i, change in enumerate(changes):
            with self.subTest(i=i):
                args=copy.deepcopy(self.args);self.change(args,'terminal',change)
                self.rejected_without_tokenization(args)
        # The unchanged complete artifact must still pass the added conditions.
        result=bridge.prepare_complete_collection(**self.args)
        self.assertEqual(result['training_plan']['sft_steps'],4)


if __name__ == '__main__':
    unittest.main()
