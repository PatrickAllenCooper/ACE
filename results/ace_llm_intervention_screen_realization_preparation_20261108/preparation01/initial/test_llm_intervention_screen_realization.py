"""New hand-built/mock checks. Never draw a scientific parameter or response."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/research'))
import llm_intervention_screen_realization as r
from llm_intervention_contract import digest
from llm_intervention_screen_protocol import build_proposal, REQUIRED_SOURCE_PATHS
from llm_intervention_world import LinearMechanism, WorldDescriptor


class RealizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in REQUIRED_SOURCE_PATHS}
        cls.proposal = build_proposal(source_pins=cls.pins)
        cls.proposal_pin = digest(cls.proposal)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ace-realization-mock-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'output'
        self.kw = dict(proposal=self.proposal, proposal_sha256=self.proposal_pin,
            source_pins=self.pins, output=self.out,
            freeze_sha256='a'*64, namespace_admission_sha256='b'*64)

    @staticmethod
    def handmade(**kw):
        # Explicit fake coefficients, never the scientific keyed generator.
        return WorldDescriptor(kw['scope'], kw['world_id'], kw['graph'],
            tuple(LinearMechanism(n.name, 0., tuple((p, 1.) for p in n.parents), variance)
                  for n, variance in zip(kw['graph'], kw['noise_variances'])))

    def test_all_laws_order_and_namespaces_without_draws(self):
        with patch.object(r, 'generate_world', side_effect=AssertionError('draw forbidden')):
            recipes = r.realization_recipes(self.proposal,
                proposal_sha256=self.proposal_pin, source_pins=self.pins)
        self.assertEqual([x['world_id'] for x in recipes[:32]], list(range(202611070000,202611070032)))
        self.assertEqual([x['world_id'] for x in recipes[32:]], list(range(202611070100,202611070124)))
        self.assertEqual([x['scope'] for x in recipes], ['policy_train']*32+['development']*24)
        for recipe, row in zip(recipes, self.proposal['training_worlds']+self.proposal['development_worlds']):
            self.assertEqual(recipe['intercept_means'], (0.,)*3)
            self.assertEqual(recipe['intercept_sd'], 1.)
            self.assertEqual(recipe['coefficient_sd'], 1.)
            self.assertEqual(recipe['noise_variances'], tuple(row['truth_law']['noise_variances']))
            self.assertEqual(recipe['coefficient_means'], tuple(tuple(x) for x in row['truth_law']['coefficient_means']))

    def test_mock_complete_56_and_exclusive_second_call(self):
        with patch.object(r, 'generate_world', side_effect=self.handmade) as fake:
            result = r.realize_screen_worlds(**self.kw)
            self.assertEqual(fake.call_count, 56)
            self.assertEqual(result['worlds_generated'], 56)
            self.assertFalse(result['scientific_execution_qualified'])
            self.assertEqual(result['environment_responses'], 0)
            for entry in result['entries']:
                data = (self.out/entry['path']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['file_sha256'])
                self.assertEqual(digest(json.loads(data)), entry['descriptor_sha256'])
            self.assertEqual((self.out/'terminal.json').read_bytes(), (self.out/'terminal.candidate.json').read_bytes())
            with self.assertRaises(FileExistsError):
                r.realize_screen_worlds(**self.kw)
            self.assertEqual(fake.call_count, 56)

    def test_partial_failure_keeps_originals_and_unstarted_roster(self):
        seen = []
        def fake(**kw):
            seen.append(kw['world_id'])
            if len(seen) == 3:
                raise RuntimeError('artificial third-descriptor failure')
            return self.handmade(**kw)
        with patch.object(r, 'generate_world', side_effect=fake):
            with self.assertRaisesRegex(RuntimeError, 'third-descriptor'):
                r.realize_screen_worlds(**self.kw)
        failure = json.loads((self.out/'failure.json').read_bytes())
        self.assertEqual(len(seen), 3)
        self.assertEqual(failure['completed_descriptors'], 2)
        self.assertEqual([x['status'] for x in failure['entries']], ['complete']*2+['started']+['unstarted']*53)
        self.assertFalse((self.out/'terminal.json').exists())
        self.assertTrue((self.out/'world-002.intent.json').exists())
        with patch.object(r, 'generate_world', side_effect=AssertionError('retry forbidden')):
            with self.assertRaises(FileExistsError):
                r.realize_screen_worlds(**self.kw)

    def test_changed_proposal_source_and_missing_admission_reject_before_draw(self):
        cases=[]
        bad=copy.deepcopy(self.proposal)
        bad['training_worlds'][0]['truth_law']['intercept_sd']=2.
        cases.append(dict(self.kw, proposal=bad, proposal_sha256=digest(bad)))
        cases.append(dict(self.kw, proposal_sha256='0'*64))
        pins=dict(self.pins)
        pins[next(iter(pins))]='0'*64
        bad=build_proposal(source_pins=pins)
        cases.append(dict(self.kw, proposal=bad, proposal_sha256=digest(bad), source_pins=pins))
        cases.append(dict(self.kw, namespace_admission_sha256=None))
        with patch.object(r, 'generate_world', side_effect=AssertionError('draw forbidden')):
            for kwargs in cases:
                with self.assertRaises(ValueError):
                    r.realize_screen_worlds(**kwargs)
                self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
