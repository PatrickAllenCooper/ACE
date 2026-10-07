"""Reject cohort/import substitution and report all reconstructed numeric drift."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from replay_delivery_attribution_release import FROZEN_HISTORIES, compare, replay


class AttributionReplayTests(unittest.TestCase):
    def test_diagnostic_discrepancy_is_reported_and_nonfinite_rejected(self):
        expected = {'score': {'mse': 1.}, 'diagnostic': [2., None]}
        actual = {'score': {'mse': 1.}, 'diagnostic': [2.+1e-11, None]}
        self.assertGreater(compare(actual, expected, 'fit'), 0.)
        with self.assertRaises(ValueError): compare({'mse': float('nan')}, {'mse': 1.}, 'fit')
        with self.assertRaises(ValueError): compare({'mse': 1., 'extra': 0.}, {'mse': 1.}, 'fit')

    def make_root(self, base, histories):
        a = Path(base)/'A'; a.mkdir()
        (a/'protocol.json').write_text(json.dumps({'dependencies': {}}))
        for history in histories:
            folder = a/history; folder.mkdir(); (folder/'input.json').write_text('{}')
        return Path(base)

    def test_twelve_history_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            histories = (FROZEN_HISTORIES-{'983656467'}) | {'replacement'}
            root = self.make_root(td, histories)
            with patch('replay_delivery_attribution_release.verify', return_value={}):
                with self.assertRaisesRegex(ValueError, 'twelve-history membership'):
                    replay(root, 'trusted', trusted_pickles=True)

    def test_cached_learner_import_cannot_bypass_archived_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = self.make_root(td, FROZEN_HISTORIES)
            with patch('replay_delivery_attribution_release.verify', return_value={}), \
                    patch.dict(sys.modules, {'ace': types.ModuleType('ace')}):
                with self.assertRaisesRegex(ValueError, 'fresh process required'):
                    replay(root, 'trusted', trusted_pickles=True)


if __name__ == '__main__':
    unittest.main()
