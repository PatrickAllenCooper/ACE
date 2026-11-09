"""New ordering regression only. No production or scientific execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).with_name('audit_saved_cpu.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '18d0255f4ae12ece4f3de86065c259626c880abceb2459ef9124b8a083560ba8'
spec = importlib.util.spec_from_file_location('saved_metadata_aid', source)
aid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aid)


class ObservationOrder(unittest.TestCase):
    def setUp(self):
        self.expected = {'files': {'launch/a.json': {'bytes': 2, 'sha256': 'a'*64}},
                         'directories': ['common', 'launch', 'training']}

    def test_root_order_is_not_directory_membership(self):
        observation = copy.deepcopy(self.expected)
        observation['directories'] = ['launch', 'training', 'common']
        observation['utc'] = 'artificial'
        original = copy.deepcopy(observation)
        self.assertEqual(self.expected, aid.normalized_observation_inventory(observation))
        self.assertEqual(observation, original)

    def test_duplicate_and_unsafe_names_still_reject(self):
        for names in (['common', 'launch', 'launch'], ['../launch', 'common']):
            observation = copy.deepcopy(self.expected)
            observation['directories'] = names
            with self.assertRaises(ValueError):
                aid.normalized_observation_inventory(observation)

    def test_missing_directory_and_changed_bytes_do_not_match(self):
        missing = copy.deepcopy(self.expected)
        missing['directories'].remove('training')
        changed = copy.deepcopy(self.expected)
        changed['files']['launch/a.json']['sha256'] = 'b'*64
        for observation in (missing, changed):
            with self.assertRaises(ValueError):
                aid.need(self.expected == aid.normalized_observation_inventory(observation),
                         'full original before/after archive inventory differs')


if __name__ == '__main__':
    unittest.main()
