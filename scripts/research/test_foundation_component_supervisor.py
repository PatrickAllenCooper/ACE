"""Bounded failure-path fixtures; no models or scientific worlds."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess,time,os,signal

spec=importlib.util.spec_from_file_location('supervisor',Path(__file__).with_name('supervise_foundation_component.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Supervision(unittest.TestCase):
    def run_fixture(self, worker, pin_ok=True, limit=2):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
        src=base/'worker.py';src.write_text(worker)
        freeze=base/'freeze.json';m.write(freeze,{'schema':'ace-component-freeze-v1','supervisor_sha256':m.sha(m.__file__),'source_sha256':m.sha(src)})
        args=argparse.Namespace(output=base/'out',mode='pilot',freeze=freeze,freeze_sha256=m.sha(freeze) if pin_ok else 'bad',worker=src,python=Path(sys.executable),checkpoint=base/'none',language_model=base/'none')
        prior=m.LIMITS['pilot'];m.LIMITS['pilot']=limit
        try:
            with patch.object(m,'validate_stage'):code=m.supervise(args)
        finally:m.LIMITS['pilot']=prior
        return code,json.loads((args.output/'terminal.json').read_text()),args.output
    def test_preflight_ledger(self):
        code,t,o=self.run_fixture('raise Exception("must not execute")',False)
        self.assertEqual(code,1);self.assertEqual(len(t['cells']),30)
        self.assertTrue(all(r['status']=='unattempted' for r in t['cells']))
        self.assertIsNone(t['child_cpu_s'])
    def test_timeout_retains_started(self):
        src='''import sys,json,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1])/'91000';p.mkdir()
(p/'polynomial.started.json').write_text('{}')
time.sleep(10)
'''
        code,t,o=self.run_fixture(src,limit=.4)
        self.assertEqual(t['reason'],'wall_timeout');self.assertEqual(code,1)
        self.assertEqual(t['cells'][0]['status'],'interrupted')
        self.assertGreater(t['peak_child_rss_bytes'],0)
        self.assertTrue(all(r['status']=='unattempted' for r in t['cells'][1:]))
    def test_failure_after_result(self):
        src='''import sys,json
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1])/'91000';p.mkdir()
(p/'polynomial.json').write_text(json.dumps({'seed':91000,'method':'polynomial','status':'complete'}))
raise RuntimeError('fixture failure')
'''
        code,t,o=self.run_fixture(src)
        self.assertEqual(t['reason'],'child_failed');self.assertEqual(t['cells'][0]['status'],'complete')
        self.assertGreaterEqual(t['child_cpu_s'],0)
    def test_json_nonfinite_before_publication(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json'
            with self.assertRaises(ValueError):m.write(p,{'bad':float('nan')})
            self.assertFalse(p.exists());self.assertFalse(Path(str(p)+'.pending').exists())
    def test_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';m.write(p,{'first':1})
            with self.assertRaises(FileExistsError):m.write(p,{'replacement':2})
            self.assertEqual(json.loads(p.read_text()),{'first':1})

    def test_captured_pin(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'freeze.json';p.write_text('{}');pin=m.sha(p);p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):m.captured(p,pin)
    def test_compatibility_cannot_score(self):
        with self.assertRaises(ValueError):m.validate_stage({'stage':'compatibility','allowed_modes':['pilot']},'pilot')
        with self.assertRaises(ValueError):m.validate_stage({'stage':'compatibility','allowed_modes':['tabpfn-smoke']},'pilot')
    def test_invalid_marker_not_success(self):
        src="import sys;from pathlib import Path;p=Path(sys.argv[sys.argv.index('--output')+1]);(p/'complete.json').write_text('{invalid')"
        code,t,o=self.run_fixture(src)
        self.assertEqual(code,1);self.assertEqual(t['reason'],'invalid_completion')
    def test_sigterm_reaps_child(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);src=b/'worker.py';src.write_text('import time;time.sleep(30)')
            f=b/'freeze.json';m.write(f,{'schema':'ace-component-freeze-v1','stage':'compatibility','allowed_modes':['tabpfn-smoke'],'supervisor_sha256':m.sha(m.__file__),'source_sha256':m.sha(src)})
            cmd=[sys.executable,m.__file__,'--output',str(b/'out'),'--python',sys.executable,'--worker',str(src),'--freeze',str(f),'--freeze-sha256',m.sha(f),'--checkpoint',str(b/'none'),'--language-model',str(b/'none'),'--lock',str(b/'lock'),'--mode','tabpfn-smoke']
            proc=subprocess.Popen(cmd)
            try:
                for _ in range(100):
                    if (b/'out/child.json').exists():break
                    time.sleep(.02)
                pid=json.loads((b/'out/child.json').read_text())['pid']
                proc.send_signal(signal.SIGTERM);self.assertEqual(proc.wait(timeout=5),1)
                t=json.loads((b/'out/terminal.json').read_text());self.assertEqual(t['status'],'failed')
                self.assertIn('controller signal',t['error'])
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()

    def test_cancel_during_popen_assignment(self):
        original=m.subprocess.Popen
        def interrupted(*a,**k):
            child=original(*a,**k);os.kill(os.getpid(),signal.SIGTERM);return child
        with patch.object(m.subprocess,'Popen',side_effect=interrupted):
            code,t,o=self.run_fixture('import time;time.sleep(10)')
        self.assertEqual(code,1);self.assertEqual(t['reason'],'cancelled')
    def test_cancel_during_reap_publication(self):
        original=m.os.wait4
        def interrupted(*a,**k):
            result=original(*a,**k)
            if result[0]:os.kill(os.getpid(),signal.SIGTERM)
            return result
        with patch.object(m.os,'wait4',side_effect=interrupted):
            code,t,o=self.run_fixture('pass')
        self.assertEqual(code,1);self.assertEqual(t['reason'],'cancelled')

if __name__=='__main__':unittest.main()
