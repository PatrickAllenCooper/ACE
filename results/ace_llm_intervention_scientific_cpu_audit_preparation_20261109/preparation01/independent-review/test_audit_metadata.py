"""New artificial metadata checks only. No producer, files, model or remote calls."""
import hashlib
import json
from pathlib import Path
import types
import unittest

source = Path(__file__).with_name('audit_saved_cpu.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '6d56672582f8c41bdd01c64b446c1d1607316650c6357bec95a889c357ab7eb4'
aid = types.ModuleType('pinned_metadata_aid')
exec(compile(source, '<independent-pinned-metadata-aid>', 'exec'), aid.__dict__)


def journal(events):
    previous = None
    lines = []
    for i, event in enumerate(events):
        row = dict(event, previous_sha256=previous, monotonic_ns=10+i)
        row['sha256'] = aid.sha(aid.canonical(row))
        previous = row['sha256']
        lines.append(aid.canonical(row))
    return b'\n'.join(lines)+b'\n'


def events():
    result = [dict(event='begin', budget_units=32, graph=[], menu=[])]
    for i in range(9):
        cost = 1 if i < 8 else 2
        result += [
            dict(event='proposal', proposal_id=i, provenance={'kind': 'shared_initial_schedule' if i < 8 else 'text_worker'}, completion='fake'),
            dict(event='reserve', proposal_id=i, request_id=i, cost_units=cost),
            dict(event='response', request_id=i, cost_units=cost, action_id='observe_all' if i < 8 else 'intervene', values={'fake': 123.456}),
        ]
    return result+[dict(event='stop', proposal_id=9, spent_units=10)]


class NewMetadataChecks(unittest.TestCase):
    def test_opaque_lexical_spans_and_rejections(self):
        value = {'event': 'response', 'values': {'fake': ['}', '\\"', {'nested': [2, 3]}]}, 'cost_units': 2}
        raw = aid.canonical(value)
        pieces = aid.object_parts(raw)
        self.assertEqual(pieces['values'], aid.canonical(value['values']))
        self.assertEqual(aid.public(raw, {'event', 'cost_units'}), {'event': 'response', 'cost_units': 2})
        for invalid in (b'{"x":1,"x":2}', b'{"x":[1}}', b'{"x":1,}', b'{"x":1} trailing'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                aid.object_parts(invalid)

    def test_exact_charge_chain_and_failure_rejection(self):
        a = aid.Audit({})
        data = journal(events())
        a.b = lambda _: data
        a.h = lambda _: aid.sha(data)
        result = a.journal('fake/journal.jsonl')
        self.assertEqual((result['responses'], result['cost_units'], result['acquisition_cost_units']), (9, 10, 2))
        self.assertIs(result['values_decoded'], False)
        bad = events()
        bad[-2]['cost_units'] = 3
        for rows in (bad, events()[:-1], events()+[dict(event='closed', spent_units=10)]):
            data = journal(rows)
            with self.assertRaises(ValueError):
                a.journal('fake/journal.jsonl')

    def test_allocation_units_and_duplicate_gpu_fields(self):
        rows = [
            ['1', 'ace_llm_cpu', 'COMPLETED', '0:0', '2', '11', 'ucb736_asc2', 'cpu=2,mem=7680M,node=1'],
            ['2', 'ace_llm_gpu', 'COMPLETED', '0:0', '4', '13', 'ucb736_asc2', 'cpu=4,node=1,gres/gpu=1,gres/gpu:a100_3g.20gb=1'],
            ['3', 'ace_llm_cancel', 'CANCELLED', '0:0', '0', '0', 'ucb736_asc2', ''],
        ]
        self.assertEqual(aid.costs(rows), (74, 13))
        rows[1][-1] = 'cpu=4,node=1,gres/gpu=2,gres/gpu:a100_3g.20gb=1'
        with self.assertRaises(ValueError):
            aid.costs(rows)


if __name__ == '__main__':
    unittest.main(verbosity=2)
