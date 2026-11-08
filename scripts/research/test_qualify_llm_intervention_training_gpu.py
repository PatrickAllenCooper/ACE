"""New GPU orchestration boundaries using saved metadata and fabricated work."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('training_gpu_checks',Path(__file__).with_name('qualify_llm_intervention_training_gpu.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
gpu=g.module('gpu_constants',ROOT/'scripts/research'/g.GPU,(ROOT/'scripts/research'/g.GPU).read_bytes())

class NewTrainingGPUBoundaries(unittest.TestCase):
    def test_accepted_cpu_lineage_cannot_be_replaced_or_dropped(self):
        old=json.loads((ROOT/'results/ace_llm_intervention_training_cpu_20261108/launch01/preparation.json').read_bytes())
        packet={'schema':g.VERSION,'files':{**old['files'],**{n:'a'*64 for n in g.EXTRAS},**g.INDEPENDENT},
                'source_names':sorted(set(old['source_names'])|g.EXTRAS),'resources':gpu.RESOURCES}
        g.packet_shape(packet,old,gpu)
        changes=[lambda p:p['files'].update({'prepared.json':'0'*64}),
                 lambda p:p['files'].update({'cpu-acceptance.json':'0'*64}),
                 lambda p:p['files'].update({'llm_intervention_training_engine.py':'0'*64}),
                 lambda p:p['files'].pop('record-3.json'),
                 lambda p:p['files'].update({'old-adapter.pt':'a'*64}),
                 lambda p:p['source_names'].remove(g.WORKER),
                 lambda p:p['resources'].update(wall_seconds=1200)]
        for change in changes:
            p=copy.deepcopy(packet);change(p)
            with self.assertRaises(ValueError):g.packet_shape(p,old,gpu)

    def test_live_gpu_receipt_requires_finite_positive_work_not_allocation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'first-gpu-work.json'
            self.assertFalse(g.positive_work(path))
            path.write_bytes(b'{"cuda_interval_ms":')
            self.assertFalse(g.positive_work(path))
            for row in [{'cuda_interval_ms':0,'peak_gpu_allocated_bytes':4000},
                        {'cuda_interval_ms':float('nan'),'peak_gpu_allocated_bytes':4000},
                        {'cuda_interval_ms':1,'peak_gpu_allocated_bytes':0},
                        {'cuda_interval_ms':True,'peak_gpu_allocated_bytes':4000}]:
                path.write_text(json.dumps(row));self.assertFalse(g.positive_work(path))
            path.write_text(json.dumps({'cuda_interval_ms':2.5,'peak_gpu_allocated_bytes':4000}))
            self.assertTrue(g.positive_work(path))

    def test_terminated_interrupted_or_live_group_never_qualifies(self):
        good={'leader_returncode':0,'group_still_exists':False,'term_sent':False,'kill_sent':False,'errors':[]}
        g.clean_success(good)
        for k,v in [('leader_returncode',124),('group_still_exists',True),('term_sent',True),
                    ('kill_sent',True),('errors',['telemetry failure'])]:
            with self.assertRaises(ValueError):g.clean_success({**good,k:v})

    def test_authentication_failure_kept_and_collision_does_not_touch_prior(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'exclusive'
            command=[sys.executable,'-I','-B',str(ROOT/'scripts/research'/g.ENTRY),
                     '--packet',str(Path(td)/'missing.json'),'--packet-sha256','a'*64,'--output',str(out)]
            first=subprocess.run(command,capture_output=True,timeout=10)
            self.assertNotEqual(first.returncode,0)
            raw=(out/'failure.json').read_bytes()
            self.assertEqual(json.loads(raw)['type'],'FileNotFoundError')
            second=subprocess.run(command,capture_output=True,timeout=10)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((out/'failure.json').read_bytes(),raw)
            self.assertEqual({p.name for p in out.iterdir()},{'failure.json'})


class ReviewCorrections(unittest.TestCase):
    def test_parent_validation_import_ignores_ambient_file_and_cleans_up(self):
        import importlib
        name='llm_intervention_gpu_mappings'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/(name+'.py')).write_text('raise AssertionError("ambient import executed")\n')
            data={name+'.py':b'VALUE=23\n'}
            from unittest.mock import patch
            def independent_check(*args):
                module=importlib.import_module(name)
                self.assertEqual(module.VALUE,23)
                with self.assertRaises(ValueError):g.CapturedSources(data,root)
                return {'validated':True}
            sys.path.insert(0,str(root))
            try:
                with patch.object(g,'_validate_worker',side_effect=independent_check):
                    result=g.validate_worker(root,{'source_names':list(data)},'pin',data,
                        None,None,None,None,None,None,None,None)
                self.assertEqual(result,{'validated':True})
            finally:sys.path.remove(str(root))
            self.assertNotIn(name,sys.modules)

    def test_startup_success_does_not_allow_stall_or_telemetry_loss(self):
        live={'returncode':0,'owned_rows':['41, GPU-fixed, 2048']}
        unknown={'returncode':None,'owned_rows':[]}
        watch=g.ProgressWatch(10,('initial',))
        watch.observe(70,live,('initial',))
        with self.assertRaisesRegex(TimeoutError,'no advancing'):watch.observe(101,live,('initial',))
        watch=g.ProgressWatch(10,('initial',))
        watch.observe(70,unknown,('initial','step1'))
        with self.assertRaisesRegex(TimeoutError,'uncertain'):watch.observe(101,unknown,('initial','step1','step2'))
        watch=g.ProgressWatch(10,('initial',))
        for t in (70,130,190):watch.observe(t,live,('initial',str(t)))

class SummaryBindingCorrection(unittest.TestCase):
    def test_matching_outer_edits_cannot_override_validated_fit(self):
        summary={'completed_steps':6,'checkpoint_pin':'a'*64}
        raw=json.dumps(summary).encode()
        checked={'summary_sha256':g.sha(raw)}
        g.consistent_summary({'summary':summary},summary,raw,checked)
        for changed in [{'completed_steps':5,'checkpoint_pin':'a'*64},
                        {'completed_steps':6,'checkpoint_pin':'b'*64}]:
            with self.assertRaises(ValueError):g.consistent_summary({'summary':changed},changed,raw,checked)
        with self.assertRaises(ValueError):g.consistent_summary({'summary':summary},summary,raw+b' ',checked)

class LiveReadCorrection(unittest.TestCase):
    def test_unstable_live_write_is_not_gpu_or_phase_evidence(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'phase-000.json';path.write_text('{}')
            with patch.object(g,'read_bounded',side_effect=ValueError('input changed while reading')):
                self.assertFalse(g.positive_work(path))
                self.assertEqual(g.progress_records(root),())

if __name__=='__main__':unittest.main()
