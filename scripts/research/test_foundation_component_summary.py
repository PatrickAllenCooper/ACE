"""Check the frozen reporting rules on fabricated metrics only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('summary',Path(__file__).with_name('summarize_foundation_component.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Summary(unittest.TestCase):
    def fixture(self, change=None):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);p=Path(d.name)
        rows=[{'seed':s,'method':method,'status':'complete','metrics':{e:{'mse':1,'nmse':1 if method=='grammar' else 2} for e in m.ENDPOINTS}} for s in m.SEEDS for method in m.METHODS]
        if change:change(rows)
        (p/'terminal.json').write_text(json.dumps({'cells':rows}))
        (p/'plan.json').write_text(json.dumps({'mode':'pilot','cells':[{'seed':r['seed'],'method':r['method']} for r in rows]}))
        return m.summarize(p)
    def test_paired_ratios(self):
        x=self.fixture();self.assertEqual(len(x['comparisons']),12)
        for c in x['comparisons']:
            self.assertEqual(c['arithmetic_mean_paired_ratios'],2)
            self.assertAlmostEqual(c['geometric_mean_paired_ratios'],2)
    def test_failure_keeps_six_worlds_and_blocks_aggregate(self):
        x=self.fixture(lambda rows:rows[0].update(status='failed'))['comparisons'][0]
        self.assertEqual(len(x['worlds']),6);self.assertEqual(x['defined_worlds'],5)
        self.assertIsNone(x['arithmetic_mean_paired_ratios'])
        self.assertIsNone(x['geometric_mean_paired_ratios'])
    def test_zero_does_not_get_floor(self):
        def change(rows):rows[0]['metrics']['M_local']['nmse']=0
        x=self.fixture(change)['comparisons'][0]
        self.assertEqual(x['worlds'][0]['undefined_reason'],'zero_or_nonfinite_error')
        self.assertIsNone(x['geometric_mean_paired_ratios'])
    def test_fallback_is_not_valid_proposal(self):
        def change(rows):rows[4].update(proposal_valid=False,fallback='grammar')
        x=self.fixture(change)
        self.assertEqual(x['language_valid_count'],0);self.assertEqual(x['language_fallback_count'],1)
        self.assertEqual(x['comparisons'][9]['defined_worlds'],6)
    def test_missing_null_nonfinite_are_json_safe(self):
        for value in (None,float('nan'),float('inf')):
            def change(rows):rows[0]['metrics']['M_local']['nmse']=value
            x=self.fixture(change)
            self.assertIsNone(x['comparisons'][0]['arithmetic_mean_paired_ratios'])
            json.dumps(x,allow_nan=False)
        for container in (None,[],2):
            def broken(rows):rows[0]['metrics']=container
            self.assertIsNone(self.fixture(broken)['comparisons'][0]['arithmetic_mean_paired_ratios'])
            def endpoint(rows):rows[0]['metrics']['M_local']=container
            self.assertIsNone(self.fixture(endpoint)['comparisons'][0]['arithmetic_mean_paired_ratios'])
        def missing(rows):rows[0]['metrics']['M_local'].pop('nmse')
        x=self.fixture(missing)
        self.assertIsNone(x['comparisons'][0]['geometric_mean_paired_ratios'])
if __name__=='__main__':unittest.main()
