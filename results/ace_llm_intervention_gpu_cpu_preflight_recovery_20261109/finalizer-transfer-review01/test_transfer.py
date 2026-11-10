"""New fabricated transfer cases only; no subprocess, SSH or scientific calls."""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'implementation/launch_once.py'
FIXTURES=Path(os.environ['ACE_TRANSFER_FIXTURES']).resolve()


def canonical(v):return (json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def need(ok,message):
    if not ok:raise ValueError(message)
def write(p,data):
    with Path(p).open('xb') as f:f.write(data)
def read(p,maximum=32*1024**2):
    value=Path(p).read_bytes();need(len(value)<=maximum,'test read bound');return value


class Clock:
    def __init__(self):self.value=0
    def monotonic(self):return self.value
    def sleep(self,seconds):self.value+=max(31,seconds)


class SynchronousThread:
    def __init__(self,*,target,args,daemon):self.target=target;self.args=args
    def start(self):self.target(*self.args)
    def join(self,timeout):pass
    def is_alive(self):return False


class TransferTests(unittest.TestCase):
    serial=0
    def fixture(self,*,success=False,cleanup_error=False,publication_error=False):
        type(self).serial+=1;root=FIXTURES/('%02d'%self.serial);root.mkdir()
        local=root/'local';external=root/'external';local.mkdir();external.mkdir()
        b=types.SimpleNamespace(LOCAL=local,EXT=external,need=need,raw=canonical,
            sha=lambda v:hashlib.sha256(v).hexdigest(),pin=lambda s:type(s) is str and re.fullmatch('[0-9a-f]{64}',s) is not None,
            utc=lambda:'2000-01-01T00:00:00+00:00',read=read,write=write)
        names={'TerminationRequested','termination_guard','cleanup_signals','confirm_owned_group_closed',
               'transfer_plan','scp_transfer','owned_scp_transfer'}
        tree=ast.parse(SOURCE.read_text())
        body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names
              or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('TRANSFER_') for t in n.targets)]
        clock=Clock();env={'b':b,'Path':Path,'os':os,'subprocess':subprocess,'threading':threading,
                         'contextlib':contextlib,'signal':signal,'sys':sys,'time':clock}
        exec(compile(ast.Module(body=body,type_ignores=[]),'new-fabricated-transfer-seam','exec'),env)
        calls=[];buffers={101:b'partial stdout\n',102:b'partial stderr\n'}
        proc=types.SimpleNamespace(pid=424242,returncode=0 if success else None,
            stdout=types.SimpleNamespace(fileno=lambda:101),stderr=types.SimpleNamespace(fileno=lambda:102))
        proc.poll=lambda:proc.returncode
        def wait(timeout):
            if proc.returncode is None:proc.returncode=-9
            return proc.returncode
        proc.wait=wait
        def spawn(command,**kwargs):
            calls.append(('spawn',command,kwargs));self.assertTrue(kwargs['start_new_session']);return proc
        def raw_read(fd,size):
            value=buffers.get(fd,b'');buffers[fd]=b'';return value
        def killpg(pgid,sig):
            self.assertEqual(pgid,proc.pid);calls.append(('killpg',pgid,sig))
            if cleanup_error:raise PermissionError('fabricated cleanup denial')
            if sig==0:raise ProcessLookupError('fabricated ESRCH')
        def save(name,value):
            for out in (local,external):write(out/name,canonical(value))
            if publication_error and name=='transfer-root.json':raise RuntimeError('fabricated publication failure')
        return root,env,calls,spawn,raw_read,killpg,save

    def run_fake(self,fixture):
        root,env,calls,spawn,raw_read,killpg,save=fixture
        with patch.object(subprocess,'Popen',spawn),patch.object(threading,'Thread',SynchronousThread),\
             patch.object(os,'read',raw_read),patch.object(os,'killpg',killpg):
            return env['scp_transfer']('transfer-root',['scp','fabricated-only'],
                {'group':'root','bytes':1,'timeout_seconds':120},save)

    def test_actual_size_cap_and_main_propagation(self):
        root,env,*_=self.fixture();limit=(900-120)*1024**2
        files={'root.json':{'bytes':limit,'sha256':'a'*64},'inputs/toy.py':{'bytes':1,'sha256':'b'*64}}
        plan=env['transfer_plan'](files)
        self.assertEqual(plan['root']['timeout_seconds'],900);self.assertEqual(plan['inputs']['timeout_seconds'],121)
        files['root.json']['bytes']+=1
        with self.assertRaises(ValueError):env['transfer_plan'](files)
        files['root.json']['bytes']=True
        with self.assertRaises(ValueError):env['transfer_plan'](files)
        tree=ast.parse(SOURCE.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        planned=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='transfer_plan')
        readiness=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='remote' and isinstance(n.args[0],ast.Constant) and n.args[0].value=='readiness_transport.json')
        scp=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='scp_transfer')
        self.assertLess(planned.lineno,readiness.lineno)
        self.assertEqual(ast.unparse(scp.args[2]),'planned_transfers[group]')
        verification=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='remote' and isinstance(n.args[0],ast.Constant) and n.args[0].value=='verification_transport.json')
        text=ast.unparse(verification.args[1]);self.assertIn('fresh=reconcile();checked=verify()',text)
        intent=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='save' and isinstance(n.args[0],ast.Constant) and n.args[0].value=='submission.intent.json')
        self.assertLess(scp.lineno,verification.lineno);self.assertLess(verification.lineno,intent.lineno)

    def test_timeout_preserves_partial_streams_and_closes_owned_group(self):
        f=self.fixture()
        with self.assertRaisesRegex(TimeoutError,'absolute planned SCP'):self.run_fake(f)
        root,env,calls,*_=f
        self.assertEqual(sum(x[0]=='spawn' for x in calls),1)
        for out in (root/'local',root/'external'):
            self.assertEqual((out/'transfer-root.stdout').read_bytes(),b'partial stdout\n')
            self.assertEqual((out/'transfer-root.stderr').read_bytes(),b'partial stderr\n')
            r=json.loads((out/'transfer-root.json').read_bytes())
            self.assertTrue(r['timeout']);self.assertTrue(r['ambiguous']);self.assertFalse(r['retry_allowed'])
            self.assertTrue(r['owned_process_group_closed']);self.assertEqual(r['owned_group_probe']['observations'][-1],'ESRCH')
        self.assertIn(('killpg',424242,signal.SIGKILL),calls)

    def test_cleanup_and_publication_errors_do_not_mask_primary(self):
        f=self.fixture(cleanup_error=True,publication_error=True)
        with self.assertRaisesRegex(TimeoutError,'absolute planned SCP'):self.run_fake(f)
        r=json.loads((f[0]/'local/transfer-root.json').read_bytes())
        self.assertEqual(r['type'],'TimeoutError');self.assertIn('local_group_cleanup_error',r)
        self.assertFalse(r['owned_process_group_closed']);self.assertTrue(r['ambiguous'])
        self.assertEqual(sum(x[0]=='spawn' for x in f[2]),1)

    def test_exited_leader_still_checks_group_before_success(self):
        f=self.fixture(success=True);r=self.run_fake(f)
        self.assertEqual(r['returncode'],0);self.assertFalse(r['ambiguous'])
        self.assertTrue(r['owned_process_group_closed']);self.assertTrue(r['remote_verification_pending'])
        self.assertIn(('killpg',424242,signal.SIGKILL),f[2])
        self.assertIn(('killpg',424242,0),f[2])


if __name__=='__main__':unittest.main(verbosity=2)
