"""New report-runtime diagnostic/cap checks; fabricated, no scientific imports."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import run_delivery_prospective_reporting as report


class RuntimeDiagnostics(unittest.TestCase):
    def test_resource_limit_is_exact_and_deadline_leaves_cleanup_margin(self):
        reg = dict(wall_seconds=3600, rss_bytes=3*2**30, cpus=1,
                   account='ucb736_asc1', partition='acpu', qos='cpu-normal', slurm_wall_limit='01:00:00')
        report.resource_contract(reg)
        self.assertEqual(report.WALL_SECONDS-report.CHILD_SECONDS, 50)
        negatives = [('wall_seconds',1800),('wall_seconds',5400),('wall_seconds',3600.0),
                     ('slurm_wall_limit','00:30:00'),('slurm_wall_limit','01:30:00'),
                     ('rss_bytes',4*2**30),('cpus',2),('account','other'),('partition','gpu'),('qos','other')]
        for key,value in negatives:
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                report.resource_contract({**reg,key:value})

    def run_runtime(self, root, fail_inventory=False, wrong_version=False):
        prefix=root/'env';prefix.mkdir()
        values={};distributions={};calls=[]
        for name,version in [('numpy','2.2.6'),('scipy','1.15.3')]:
            p=prefix/name;p.mkdir();f=p/'__init__.py';f.write_text('fabricated')
            values[name]=types.SimpleNamespace(__file__=str(f),__version__='wrong' if wrong_version and name=='numpy' else version)
            distributions[name]=types.SimpleNamespace(version=version,locate_file=lambda name:prefix/name)
        def verify(*args):
            calls.append('verify_inventory')
            if fail_inventory:raise ValueError('fabricated inventory mismatch')
            return {'fake':{'sha256':'fake'}}
        qualifier=types.SimpleNamespace(start_diagnostics=lambda fd:calls.append('start'),
            stop_diagnostics=lambda:calls.append('stop'),
            diagnostic_phase=lambda fd,phase,**kw:calls.append(phase),verify_environment_inventory=verify)
        def imported(name):calls.append('import_'+name);return values[name]
        with patch.object(report.importlib,'import_module',side_effect=imported), \
             patch.object(report.importlib.metadata,'distribution',side_effect=lambda n:distributions[n]):
            try:result=report.statistics_runtime(root,prefix.resolve(),{'fabricated':True},qualifier)
            except BaseException:
                self.assertEqual(calls[-1],'stop')
                if fail_inventory:
                    self.assertEqual(calls,['start','report_inventory_started','verify_inventory','stop'])
                    self.assertFalse((root/'report_inventory_verified.json').exists())
                if wrong_version:self.assertNotIn('report_numpy_import_verified',calls)
                raise
        return result,calls

    def test_stage_order_and_diagnostic_only_inventory_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();result,calls=self.run_runtime(root)
            self.assertEqual(calls,['start','report_inventory_started','verify_inventory','report_inventory_verified',
                'report_numpy_import_started','import_numpy','report_numpy_import_verified',
                'report_scipy_import_started','import_scipy','report_scipy_import_verified','stop'])
            self.assertEqual({k:v['version'] for k,v in result[1].items()},{'numpy':'2.2.6','scipy':'1.15.3'})
            self.assertTrue(json.loads((root/'report_inventory_verified.json').read_text())['diagnostic_only'])
            self.assertFalse((root/'report_receipt.json').exists())

    def test_inventory_and_import_failure_keep_diagnostics_and_fail_closed(self):
        for inventory,version in [(True,False),(False,True)]:
            with self.subTest(inventory=inventory),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):self.run_runtime(Path(tmp).resolve(),inventory,version)

if __name__=='__main__':unittest.main()
