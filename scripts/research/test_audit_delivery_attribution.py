import copy
import hashlib
import json
import unittest
from audit_delivery_attribution import eligible_indices, verify_receipt


class CustodyTests(unittest.TestCase):
    def setUp(self):
        self.data = {'meta': {'feature_names': ['x'], 'target_name': 'y', 'node_mlps': ['a', 'y'],
                              'causal_dag': {'x': [], 'a': ['x'], 'y': ['a']}},
                     'selection': {'all_paid': [0, 1, 2]},
                     'rows': [{'query_index': i, 'params': {'x': 1.}, 'intermediates': {'a': 2.},
                               'outcome': 3., 'interventions': do}
                              for i, do in enumerate([{}, {'a': 2.}, {'y': 3.}])]}
        self.cell = {'case': 'fixture', 'row_set': 'all_paid', 'arm': 'scm', 'epochs': 100, 'init': 0}
        self.reg = {'protocol_sha256': 'protocol', 'threads': 6, 'rss_bytes': 8192}
        heads = {head: {'eligible_rows': 2,
                       'query_indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                       'parameters_or_tree_nodes': 100, 'optimizer_updates': 100,
                       'fit_cpu_seconds': 1., 'fit_wall_seconds': .5}
                 for head, indices in [('a', [0, 2]), ('y', [0, 1])]}
        self.receipt = {**self.cell, 'complete': True, 'new_queries': 0, 'evaluation': None,
                        'protocol_sha256': 'protocol', 'input_sha256': 'input', 'threads': 6,
                        'peak_rss_bytes': 1024, 'heads': heads, 'fit_cpu_seconds': 2.,
                        'worker_wall_seconds': 1., 'matched_cpu_seconds': None}

    def check(self, receipt):
        verify_receipt(self.cell, receipt, self.reg, {'input_sha256': 'input'}, self.data)

    def test_independent_eligibility_masks(self):
        self.assertEqual(eligible_indices(self.data, 'all_paid', 'flat'), [0])
        self.assertEqual(eligible_indices(self.data, 'all_paid', 'a'), [0, 2])
        self.assertEqual(eligible_indices(self.data, 'all_paid', 'y'), [0, 1])
        self.check(self.receipt)

    def test_receipt_tampering_rejected(self):
        for key, wrong in [('epochs', 30000), ('input_sha256', 'other'), ('new_queries', 1),
                           ('threads', 8), ('peak_rss_bytes', 9000), ('fit_cpu_seconds', 1.)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({**self.receipt, key: wrong})
        receipt = copy.deepcopy(self.receipt)
        receipt['heads']['a']['query_indices_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.check(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt['heads']['y']['optimizer_updates'] = 99
        with self.assertRaises(ValueError):
            self.check(receipt)


if __name__ == '__main__':
    unittest.main()
