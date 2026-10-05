import subprocess
import sys
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import delivery_option_a_continuation as c

class Continuation(unittest.TestCase):
    def test_watchdog_stops_owned_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'pid'
            code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(5)"
            result=c.guard.supervise([sys.executable,'-c',code,str(marker)],time.monotonic()+.4,8589934592,root/'log',interval=.01)
            self.assertEqual(result['status'],'time_limit')
            pid=marker.read_text().strip()
            state=subprocess.run(['ps','-p',pid,'-o','state='],capture_output=True,text=True).stdout.strip()
            self.assertTrue(not state or state.startswith('Z'))
    def test_exact_remaining_stage_gate(self):
        self.assertTrue(c.admission(545))
        self.assertFalse(c.admission(546))
        self.assertTrue(c.admission((7500-300)/13.2))
    def fixture(self,root):
        original=root/'original';(original/'cases/17').mkdir(parents=True);(original/'cases/17/weights').write_text('immutable')
        out=root/'out';(out/'cases/17').mkdir(parents=True);(out/'cases/17/weights').write_text('immutable')
        return out,{'original_output':str(original),'original_seeds':[17,19], 'remaining_seeds':[19], 'original_case_hashes':c.guard.inventory(original/'cases')}
    def test_failure_stops_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,f=self.fixture(Path(tmp))
            with patch.object(c,'validate',return_value=(f,{})),patch.object(c.guard,'approval_check'),patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],1)) as run:
                with self.assertRaises(RuntimeError):c.worker(out)
                self.assertEqual(run.call_count,1)
                self.assertFalse((out/'sealed.json').exists());self.assertFalse((out/'scores.json').exists())
    def test_duplicate_case_refused_before_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,f=self.fixture(Path(tmp));(out/'cases/19').mkdir()
            with patch.object(c,'validate',return_value=(f,{})),patch.object(c.subprocess,'run') as run:
                with self.assertRaises(ValueError):c.worker(out)
                run.assert_not_called()
    def test_first_timing_failure_stops_before_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,f=self.fixture(Path(tmp))
            with patch.object(c,'validate',return_value=(f,{})),patch.object(c.guard,'approval_check'),patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),patch.object(c.time,'monotonic',side_effect=[0,546]),patch.object(c.guard,'seal_cases') as seal:
                c.worker(out)
                self.assertEqual(c.guard.read(out/'stop.json')['status'],'timing_gate_failed');seal.assert_not_called()

if __name__=='__main__':unittest.main()
