"""New H1/H2 fixtures only: mocked transfer and local isolated metadata bootstrap."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('custody_h1h2',HERE/'remote_custody.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
source=(HERE/'remote_custody.py').read_bytes();c.bind_source(source,hashlib.sha256(source).hexdigest())
ROOT=Path(os.environ['ACE_H1H2_FIXTURES'])

class BrokenFile:
    def __init__(self,events):self.events=events
    def close(self):self.events.append('buffered-close');raise OSError('fabricated buffered close failure')
class BrokenHandle:
    def __init__(self,events):self.events=events
    def __exit__(self,*args):self.events.append('underlying-close');raise OSError('fabricated handle close failure')

class Tests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/self._testMethodName;self.root.mkdir(mode=0o700)

    def test_primary_exception_survives_both_cleanup_errors_and_evidence_error(self):
        for evidence_failure in (False,True):
            out=self.root/str(evidence_failure);out.mkdir(mode=0o700);selected=out/'selected';selected.mkdir(mode=0o700)
            full={'files':{'namespace.json':{'bytes':0,'sha256':c.sha(b'')}},'directories':[]}
            events=[];primary=RuntimeError('original fabricated transport failure')
            def fail_transfer(request,receiver,helper_path,logs):
                receiver.fd=BrokenFile(events);receiver.handle=BrokenHandle(events);raise primary
            original_write=c.write_new
            def publish(path,data):
                if evidence_failure and Path(path).name=='receiver-cleanup-error.json':raise OSError('fabricated cleanup evidence publication failure')
                return original_write(path,data)
            with mock.patch.object(c,'transport',side_effect=fail_transfer),mock.patch.object(c,'write_new',side_effect=publish):
                with self.assertRaises(RuntimeError) as caught:
                    c.collect_backup(observation=full,roots={'namespace.json':'/scratch/alpine/paco0228/fabricated.json'},backup_root='/projects/paco0228/ACE_evidence/fabricated-h1',local_root=str(selected),helper_path='/never-used',log_dir=str(out/'logs'))
            self.assertIs(caught.exception,primary);self.assertEqual(events,['buffered-close','underlying-close'])
            failure=c.decode((out/'logs/failure.json').read_bytes());self.assertEqual(failure['message'],str(primary))
            if not evidence_failure:
                cleanup=c.decode((out/'logs/receiver-cleanup-error.json').read_bytes())
                self.assertEqual(cleanup['primary']['message'],str(primary));self.assertEqual(len(cleanup['cleanup_errors']),2)

    def test_cleanup_failure_after_otherwise_success_still_fails(self):
        logs=self.root/'logs';logs.mkdir(mode=0o700);events=[];receiver=c.Receiver({}, {}, lambda r:None)
        receiver.fd=BrokenFile(events);receiver.handle=BrokenHandle(events)
        with self.assertRaisesRegex(OSError,'buffered close failure'):c.close_receiver(receiver,logs,None)
        self.assertEqual(events,['buffered-close','underlying-close']);self.assertIsNone(receiver.fd);self.assertIsNone(receiver.handle)
        self.assertTrue((logs/'failure.json').exists());self.assertEqual(len(c.decode((logs/'receiver-cleanup-error.json').read_bytes())['cleanup_errors']),2)

    def test_actual_isolated_bootstrap_valid_altered_truncated_and_request_bound(self):
        tree=ast.parse(source);transport=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transport')
        joins=[n for n in ast.walk(transport) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='join' and isinstance(n.args[0],ast.List)]
        command=next(n.args[0] for n in joins if isinstance(n.args[0].elts[0],ast.Constant) and n.args[0].elts[0].value=='python3')
        flags=[ast.literal_eval(n) for n in command.elts[:4]];self.assertEqual(flags,['python3','-I','-B','-c'])
        stub=b'''import sys,json
MAX_METADATA=64
sys.stdout.buffer.write(b'EXECUTED\\n')
def bind_source(source,pin):pass
def decode(value):return json.loads(value)
def remote_main(request,out):out.write(json.dumps({'isolated':sys.flags.isolated,'optimize':sys.flags.optimize,'request':request}).encode())
'''
        digest=hashlib.sha256(stub).hexdigest();prefix=str(len(stub)).encode()+b'\n'
        ambient=self.root/'ambient';ambient.mkdir();marker=self.root/'ambient-executed'
        (ambient/'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("unexpected")\n')
        env=dict(os.environ,PYTHONOPTIMIZE='2',PYTHONPATH=str(ambient))
        cases=[('valid',prefix+stub+b'{"ok":true}',False,True),('altered',prefix+stub.replace(b'EXECUTED',b'ALTERED!')+b'{}',True,False),('truncated',prefix+stub[:-1],True,False),('request-bound',prefix+stub+b'x'*65,True,False)]
        for name,payload,optimize,success in cases:
            with self.subTest(name=name):
                out=self.root/name;out.mkdir();(out/'stdin.bin').write_bytes(payload)
                cmd=[sys.executable,*flags[1:3],*(['-O'] if optimize else []),'-c',c.BOOTSTRAP,digest]
                before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
                proc=subprocess.run(cmd,input=payload,cwd=str(ambient),env=env,capture_output=True,timeout=10)
                after=resource.getrusage(resource.RUSAGE_CHILDREN)
                (out/'stdout.bin').write_bytes(proc.stdout);(out/'stderr.bin').write_bytes(proc.stderr)
                (out/'receipt.json').write_text(json.dumps({'command':cmd,'returncode':proc.returncode,'wall_seconds':time.monotonic()-start,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'PYTHONOPTIMIZE':'2','PYTHONPATH':str(ambient),'local_only':True},indent=2)+'\n')
                self.assertEqual(proc.returncode==0,success);self.assertFalse(marker.exists())
                if success:
                    self.assertTrue(proc.stdout.startswith(b'EXECUTED\n'));value=json.loads(proc.stdout.split(b'\n',1)[1]);self.assertEqual(value,{'isolated':1,'optimize':0,'request':{'ok':True}})
                elif name=='request-bound':self.assertIn(b'request metadata bound',proc.stderr);self.assertEqual(proc.stdout,b'EXECUTED\n')
                else:
                    self.assertEqual(proc.stdout,b'');self.assertIn(b'independent helper source SHA differs' if name=='altered' else b'truncated helper source',proc.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
