"""Only two new R1/R2 methods; one owned local artificial child, never SSH/Slurm."""
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_CONTINUATION_DELTA_EVIDENCE']).resolve()

def load():
    m=types.ModuleType('artificial_review_delta');m.__file__=str(HERE/'continue_once.py')
    exec(compile((HERE/'continue_once.py').read_bytes(),m.__file__,'exec'),vars(m));return m

class ReviewDelta(unittest.TestCase):
    def setUp(self):
        self.m=load();self.root=EVIDENCE/self._testMethodName;self.root.mkdir()

    def test_sigterm_cleans_owned_group_retains_failure_and_restores_handlers(self):
        m=self.m;outputs=[]
        for n in ('local','mirror'):
            p=self.root/n;p.mkdir();outputs.append(p)
        pidfile=self.root/'artificial-child.json'
        child=('import os,json,time,sys\nfrom pathlib import Path\n'
               +'p=Path('+repr(str(pidfile))+')\n'
               +'with p.open("x") as f:json.dump({"pid":os.getpid(),"pgid":os.getpgrp(),"artificial":True},f)\n'
               +'print("ARTIFICIAL LOCAL CHILD EARLY PROGRESS",file=sys.stderr,flush=True)\ntime.sleep(12)\n')
        (self.root/'child-source.py').write_text(child)
        actual_popen=subprocess.Popen;spawned=[];inject_errors=[]
        def local_only(command,**kw):
            p=actual_popen([sys.executable,'-I','-B','-c',child],**kw);spawned.append(p);return p
        def terminate_after_start():
            deadline=time.monotonic()+5
            while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
            if not pidfile.exists():inject_errors.append('artificial child startup missing')
            time.sleep(.05)
            os.kill(os.getpid(),signal.SIGTERM)
        old=signal.getsignal(signal.SIGTERM)
        injector=threading.Thread(target=terminate_after_start,daemon=True)
        with mock.patch.object(m.subprocess,'Popen',side_effect=local_only):
            with self.assertRaises(m.TerminationRequested),m.termination_guard():
                injector.start();m.transport({'phase':'complete'},b'# ARTIFICIAL SOURCE, NEVER EXECUTED\n',outputs,members={})
        injector.join(timeout=2)
        self.assertEqual(signal.getsignal(signal.SIGTERM),old)
        self.assertEqual(inject_errors,[]);self.assertEqual(len(spawned),1)
        facts=json.loads(pidfile.read_text());self.assertEqual(facts['pid'],facts['pgid'])
        with self.assertRaises(ProcessLookupError):os.killpg(facts['pgid'],0)
        record=m.decode(m.read(outputs[0]/'complete.transport.json'))
        self.assertEqual(record['type'],'TerminationRequested');self.assertTrue(record['owned_process_group_closed'])
        self.assertTrue(record['stream_threads_closed']);self.assertTrue(record['ambiguous']);self.assertFalse(record['retry_allowed'])
        self.assertEqual(record,m.decode(m.read(outputs[1]/'complete.transport.json')))
        self.assertIn(b'ARTIFICIAL LOCAL CHILD EARLY PROGRESS',m.read(outputs[0]/'complete.stderr'))
        for p in spawned:
            for stream in (p.stdin,p.stdout,p.stderr):
                if stream is not None and not stream.closed:stream.close()
        # Group evidence cannot equate a departed leader or EPERM with absence.
        with mock.patch.object(m.os,'killpg',side_effect=[PermissionError('FABRICATED EPERM'),None,ProcessLookupError()]),mock.patch.object(m.time,'sleep'):
            probe=m.confirm_owned_group_closed(987654321)
        self.assertTrue(probe['closed']);self.assertEqual(probe['probes'],3)
        self.assertIn('PermissionError',probe['observations'][0]);self.assertEqual(probe['observations'][-1],'ESRCH')
        with mock.patch.object(m.os,'killpg',return_value=None):
            self.assertFalse(m.confirm_owned_group_closed(987654321,seconds=0)['closed'])
        primary=RuntimeError('FABRICATED ORIGINAL FAILURE');done=[]
        with m.cleanup_signals(primary):signal.raise_signal(signal.SIGTERM);done.append(True)
        self.assertEqual(done,[True]);self.assertEqual(signal.getsignal(signal.SIGTERM),old)
        with self.assertRaises(m.TerminationRequested):
            with m.cleanup_signals(None):signal.raise_signal(signal.SIGTERM);done.append(True)
        self.assertEqual(done,[True,True]);self.assertEqual(signal.getsignal(signal.SIGTERM),old)
        (self.root/'probe-facts.json').write_bytes(m.raw(probe))

    def test_failure_preservation_claims_follow_verified_phase_and_detach(self):
        m=self.m;s=m.preservation_state();exc=ValueError('FABRICATED FAILURE');records=[]
        records.append(m.failure_record(exc,s));self.assertIs(records[-1]['partial_preserved'],False)
        s['phase']='remote_preservation_requested_outcome_unknown'
        records.append(m.failure_record(exc,s));self.assertIsNone(records[-1]['partial_preserved'])
        s.update(phase='projects_verified_local_copies_in_progress',projects_readback=dict(path='/FAKE/projects/partial.original',bytes=2392064,sha256='a'*64,record_sha256='b'*64))
        s['local_readbacks']['/FAKE/local-a']=dict(bytes=2392064,sha256='a'*64)
        records.append(m.failure_record(exc,s));self.assertIsNone(records[-1]['partial_preserved'])
        s['local_readbacks']['/FAKE/local-b']=dict(bytes=2392064,sha256='a'*64)
        s.update(phase='projects_and_both_local_copies_verified',original_move='requested_outcome_unknown')
        records.append(m.failure_record(exc,s));self.assertIs(records[-1]['partial_preserved'],True)
        self.assertEqual(records[-1]['partial_preservation']['original_move'],'requested_outcome_unknown')
        saved=copy.deepcopy(records[-1]);s['local_readbacks'].clear();s['projects_readback']=None
        self.assertEqual(records[-1],saved)
        for i,record in enumerate(records):(self.root/('phase-%02d.json'%i)).write_bytes(m.raw(record))

if __name__=='__main__':unittest.main(verbosity=2)
