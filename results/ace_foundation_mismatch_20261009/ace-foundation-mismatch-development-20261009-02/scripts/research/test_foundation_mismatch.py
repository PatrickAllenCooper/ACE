"""Artificial boundary fixtures only; no scientific seed or pretrained model call."""
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import foundation_mismatch_pilot as w
import foundation_mixture_selection as sel
import summarize_foundation_mismatch as report
import supervise_foundation_mismatch as sup

class Linear:
    def fit(self,x,y):self.coef=np.linalg.lstsq(np.column_stack((np.ones(len(x)),x[:,0])),y,rcond=None)[0];return self
    def predict(self,x):return self.coef[0]+self.coef[1]*x[:,0]

def basis(x,f):
    x=np.asarray(x).reshape(-1)
    return np.column_stack((np.ones(len(x)),x,x*x)) if f=='quadratic' else np.column_stack((np.ones(len(x)),np.tanh(x) if f=='tanh' else x))

def eligible(train,clamp):return ((train[clamp!='M',0:1],train[clamp!='M',1]),(train[:,1:2],train[:,2]))

class Boundaries(unittest.TestCase):
    def setUp(self):
        w.np=np;w.sel=sel;w.base=types.SimpleNamespace(FAMILIES=('linear','quadratic','tanh'),basis=basis,eligible=eligible,make_model=lambda *args:Linear())
    def test_artificial_variants_and_shared_probes(self):
        spec=w.specification(123456,True);x=np.array([-.5,.5])
        np.testing.assert_allclose(w.truth(spec,'coefficient_M',0,x),.3+1.2*x)
        np.testing.assert_allclose(w.truth(spec,'missing_M',1,x),w.truth(spec,'null',1,x))
        np.testing.assert_allclose(w.truth(spec,'missing_Y',0,x),w.truth(spec,'null',0,x))
        np.testing.assert_allclose(w.truth(spec,'missing_Y',1,x),.2+1.1*np.sin(np.pi*x))
        p=w.private_probes(spec,'null',123456);q=w.private_probes(spec,'missing_M',123456)
        for a,b,col in zip(p,q,(0,0,1)):np.testing.assert_array_equal(a[:,col],b[:,col])
        with self.assertRaises(ValueError):w.specification(123456,False)
    def test_fitting_and_private_generation_barrier(self):
        spec=w.specification(123456,True);rows=w.history(spec,'null',123456,1)
        with patch.object(w,'private_probes',side_effect=AssertionError('private access')):
            heads,choices,errors,events=w.make_predictors(rows,None,Path('/unused'))
        self.assertEqual([h.fit_rows for h in heads['grammar32']],[20,32])
        self.assertEqual([h.fit_rows for h in heads['grammar24']],[15,24])
        self.assertEqual([h.fit_rows for h in heads['pfn24']],[15,24])
        self.assertEqual(set(errors),{'prechange24'})
        self.assertEqual(len(choices['terminal24']['candidates']),5)
        self.assertEqual(len(choices['mechanism24']['candidates']),25)
    def test_complete_artificial_saved_interface_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w.run(root,Path('/unused'),'fixture')
            c=json.loads((root/'complete.json').read_text())
            t={'status':'complete','mode':'fixture','cells':c['cells'],**dict.fromkeys(('child_cpu_s','supervisor_process_cpu_s','elapsed_s','peak_child_rss_bytes','gpu_seconds'),0)}
            s=report.verify(root,{'mode':'fixture'},t)
            self.assertEqual(len(s['cells']),24);self.assertEqual(s['response_accounting']['returned'],3232)
            seal=json.loads((root/'123456/null/selection_seal.json').read_text())
            self.assertFalse(seal['evaluation_generated'])
            path=root/'123456/null/grammar32_predictions.npy';a=np.load(path);a[0,0]+=1;np.save(path,a)
            with self.assertRaises(ValueError):report.verify(root,{'mode':'fixture'},t)
    def test_partial_attempt_and_unknown_response_returns(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w.run(root,Path('/unused'),'fixture')
            c=json.loads((root/'complete.json').read_text());(root/'complete.json').unlink()
            for variant in ('missing_M','missing_Y'):shutil.rmtree(root/'123456'/variant)
            (root/'123456/coefficient_M/evaluation.returned.json').unlink()
            for row in c['cells']:
                if row['variant']!='null':row['status']='interrupted'
            t={'status':'failed','reason':'wall_timeout','mode':'fixture','cells':c['cells'],**dict.fromkeys(('child_cpu_s','supervisor_process_cpu_s','elapsed_s','peak_child_rss_bytes','gpu_seconds'),0)}
            out=report.verify(root,{'mode':'fixture'},t)
            self.assertEqual(out['attempt_status'],'failed');self.assertEqual(len(out['cells']),24)
            self.assertEqual(sum(r['status']=='complete' for r in out['cells']),6)
            self.assertEqual(out['response_accounting']['private']['reserved_unknown_returns'],768)
            self.assertEqual(out['response_accounting']['private']['validated_returned'],768)
            self.assertEqual(out['response_accounting']['private']['unreserved_planned'],1536)
            self.assertTrue(all(not x['defined'] for x in out['comparisons'] if x['variant']!='null'))
            path=root/'123456/null/grammar32_predictions.npy';a=np.load(path);a[0,0]+=1;np.save(path,a)
            out=report.verify(root,{'mode':'fixture'},t)
            self.assertEqual(out['cells'][0]['status'],'invalid_record');self.assertTrue(out['verification_issues'])
            (root/'123456/prehistory.returned.json').unlink()
            out=report.verify(root,{'mode':'fixture'},t)
            self.assertEqual(out['cells'][5]['status'],'invalid_record')
            self.assertTrue(all(h['mse_difference'] is None for h in out['local_harm'] if h['variant']=='null'))
            t['cells'][0]=None
            out=report.verify(root,{'mode':'fixture'},t)
            self.assertEqual(len(out['cells']),24);self.assertEqual(out['cells'][0]['status'],'invalid_record')
            self.assertIsNone(out['raw_terminal_cells'][0])
    def test_checkpoint_metadata_rejected_before_source_execution(self):
        deps={n:w.importlib.metadata.version(n) for n in ('numpy','torch','transformers','tabpfn','scikit-learn','scipy','safetensors','tokenizers','huggingface-hub')}
        f={'dependencies':deps,'python_version':sys.version,'python':str(Path(sys.executable).absolute()),'checkpoint':'/unused','checkpoint_sha256':'0'*64}
        with patch.object(w,'sha',return_value='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'),patch.object(w,'load_source',side_effect=AssertionError('execution before checkpoint gate')):
            with self.assertRaises(ValueError):w.initialize(f)
    def test_preflight_deadline_before_spawn_and_owned_git_wait(self):
        with patch.object(sup.subprocess,'Popen',side_effect=AssertionError('late spawn')):
            with self.assertRaises(TimeoutError):sup.bounded_git(['unused'],time.monotonic()-1,[])
        with self.assertRaises(TimeoutError):sup.bounded_git([sys.executable,'-c','import time; time.sleep(10)'],time.monotonic()+.05,[])
    def test_failure_and_zero_pairs_not_filtered(self):
        cells=[{'seed':s,'variant':v,'method':m,'status':'complete','metrics':{e:{'mse':1.,'nmse':1.,'training_variance':1.} for e in report.ENDPOINTS}} for s in (1,2) for v in report.VARIANTS for m in report.METHODS]
        cells[1]['metrics']['Y_composed']['nmse']=0
        cells[7]['status']='failed'
        out=report.summarize(cells,(1,2))
        comp=next(c for c in out['comparisons'] if c['variant']=='null' and c['method']=='grammar24' and c['endpoint']=='Y_composed')
        self.assertFalse(comp['defined']);self.assertEqual(len(comp['pairs']),2)
        with self.assertRaises(ValueError):report.summarize(cells[:-1],(1,2))
        altered=copy.deepcopy(cells);altered[0]['metrics']['M_local']['training_variance']=2
        with self.assertRaises(ValueError):report.summarize(altered,(1,2))
    def test_preflight_failure_preserves_full_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);freeze=root/'bad.json';freeze.write_text('{}')
            args=types.SimpleNamespace(output=root/'attempt',mode='fixture',freeze=freeze,freeze_sha256='0'*64)
            self.assertEqual(sup.supervise(args),1)
            terminal=json.loads((args.output/'terminal.json').read_text())
            self.assertEqual(terminal['status'],'failed');self.assertIsNone(terminal['child_cpu_s'])
            self.assertEqual(len(terminal['cells']),24)
            self.assertEqual({c['status'] for c in terminal['cells']},{'unattempted'})
            self.assertFalse((args.output/'child.json').exists())
    def test_stage_rejects_missing_closure_and_late_start(self):
        f={'schema':'ace-mismatch-freeze-v1','mode':'fixture','sources':{}}
        with self.assertRaises(ValueError):sup.validate_stage(f,'fixture')
        f.update(sources=dict.fromkeys(sup.SOURCES,'x'),limit_seconds=120,stop_unix=sup.STOP_UNIX)
        with patch.object(sup.time,'time',return_value=sup.STOP_UNIX-100):
            with self.assertRaises(ValueError):sup.validate_stage(f,'fixture')
        self.assertEqual(datetime.datetime.fromtimestamp(sup.STOP_UNIX,datetime.timezone.utc).isoformat(),'2026-10-09T22:00:00+00:00')

if __name__=='__main__':unittest.main()
