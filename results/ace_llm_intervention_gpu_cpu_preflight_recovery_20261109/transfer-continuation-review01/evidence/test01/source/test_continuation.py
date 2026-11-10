"""New artificial continuation tests only. No SSH, real locks, scheduler or main."""
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_CONTINUATION_TEST_EVIDENCE']).resolve()

def load():
    m=types.ModuleType('artificial_continuation');m.__file__=str(HERE/'continue_once.py')
    exec(compile((HERE/'continue_once.py').read_bytes(),m.__file__,'exec'),vars(m))
    return m

class Continuation(unittest.TestCase):
    def setUp(self):
        self.m=load();self.base=EVIDENCE/self._testMethodName;self.base.mkdir(mode=0o700)
        # Deliberately fake bytes at the actual partial size; never actual evidence.
        self.partial=(b'ARTIFICIAL TRANSFER PREFIX ONLY\n'*100000)[:self.m.PARTIAL_BYTES]
        self.m.PARTIAL_SHA=self.m.sha(self.partial)

    def fixture(self,name):
        m=self.m;case=self.base/name;case.mkdir();root=case/'control';root.mkdir();(root/'inputs').mkdir();(root/'logs').mkdir()
        for n,data in [(m.PARTIAL,self.partial),('kept.json',b'FABRICATED KEPT\n'),('readiness.json',b'FABRICATED READINESS\n')]:m.write(root/n,data)
        obs=m.stat_map(root)
        complete={m.PARTIAL:self.partial+b'FABRICATED TAIL\n','inputs/missing.json':b'FABRICATED MISSING INPUT\n'}
        pins={n:m.sha(v) for n,v in complete.items()};pins['kept.json']=m.sha(b'FABRICATED KEPT\n')
        sizes={n:len(v) for n,v in complete.items()};sizes['kept.json']=len(b'FABRICATED KEPT\n')
        for n,v in obs['files'].items():v.update(expected_sha256=pins.get(n),matches_expected=n=='kept.json')
        obs.update(root=m.ROOT,missing=['inputs/missing.json'],mismatches=[m.PARTIAL],unexpected=[],changed_during_observation=[],submission_intent_exists=False,submission_result_exists=False,output_exists=False)
        project=case/'projects';project.mkdir();backup=project/'backup';evidence=root/m.OP
        return root,backup,evidence,obs,complete,pins,sizes

    def preserve_and_ack(self,root,backup,evidence,obs):
        m=self.m;r=m.preserve_partial(root,backup,evidence,obs,{'fabricated':True})
        copies={}
        for name in ('local-a','local-b'):
            p=root.parent/name;p.mkdir();m.write(p/'partial.original',self.partial)
            copies[str(p/'partial.original')]={'bytes':len(m.read(p/'partial.original',m.PARTIAL_SHA)),'sha256':m.PARTIAL_SHA}
        ack=dict(partial_sha256=m.PARTIAL_SHA,partial_bytes=m.PARTIAL_BYTES,local_copies=copies,preservation_sha256=r['record_sha256'])
        return r,ack

    def test_saved_join_and_fresh_quiescence_fail_closed(self):
        m=self.m;root,backup,evidence,obs,complete,pins,sizes=self.fixture('joins')
        proofs={'observation':obs,'quiescence':dict(root=m.ROOT,same_membership_and_stat_as_prior=True,partial_sha256=m.PARTIAL_SHA,fuser=dict(returncode=1,stdout='',stderr='')),
                'transfer':dict(returncode=None,timeout=True,ambiguous=True,retry_allowed=False),
                'failure':dict(retry_allowed=False,message='retained failed transfer-root.json'),
                'local_processes':dict(matches=[],returncode=0)}
        expected=m.validate_proofs(proofs,pins,sizes);self.assertEqual(set(expected),set(complete))
        for label,change in [('intent',lambda p:p['observation'].__setitem__('submission_intent_exists',True)),
                             ('second_partial',lambda p:p['observation']['mismatches'].append('kept.json')),
                             ('unknown_member',lambda p:p['observation']['unexpected'].append('unknown')),
                             ('inconclusive_fuser',lambda p:p['quiescence']['fuser'].__setitem__('stderr','usage')),
                             ('local_transfer_alive',lambda p:p['local_processes']['matches'].append('FAKE_SCP'))]:
            changed=copy.deepcopy(proofs);change(changed)
            (self.base/(label+'.json')).write_bytes(m.raw(changed))
            with self.subTest(case=label),self.assertRaises(ValueError):m.validate_proofs(changed,pins,sizes)
        for rc,out,err,ok in [(1,'','',True),(1,'','usage',False),(0,'PID123','',False)]:
            with mock.patch.object(m.subprocess,'run',return_value=types.SimpleNamespace(returncode=rc,stdout=out,stderr=err)) as run:
                if ok:m.quiescent(root,obs)
                else:
                    with self.assertRaises(ValueError):m.quiescent(root,obs)
                self.assertNotIn('--',run.call_args.args[0])
        self.assertFalse(evidence.exists());self.assertFalse(backup.exists())

    def test_preservation_precedes_move_and_missing_only_publication(self):
        m=self.m;root,backup,evidence,obs,complete,pins,sizes=self.fixture('positive')
        kept=m.stat_map(root)['files']['kept.json'];r,ack=self.preserve_and_ack(root,backup,evidence,obs)
        self.assertEqual(m.read(root/m.PARTIAL),self.partial)
        bad=copy.deepcopy(ack);bad['local_copies'].pop(next(iter(bad['local_copies'])))
        with self.assertRaises(ValueError):m.relocate_partial(root,evidence,backup,obs,bad)
        self.assertTrue((root/m.PARTIAL).exists())
        m.relocate_partial(root,evidence,backup,obs,ack)
        members={n:dict(bytes=len(v),sha256=m.sha(v)) for n,v in complete.items()}
        result=m.receive_missing(io.BytesIO(b''.join(complete[n] for n in sorted(complete))),root,evidence,members,obs)
        self.assertEqual(result['completed_members'],members)
        for n,v in complete.items():self.assertEqual(m.read(root/n),v)
        self.assertEqual(m.stat_map(root,excluded=(m.OP,))['files']['kept.json'],kept)
        self.assertEqual(m.read(backup/'partial.original'),self.partial)
        self.assertEqual(m.read(evidence/'partial.renamed'),self.partial)
        self.assertNotEqual((backup/'partial.original').stat().st_ino,(evidence/'partial.renamed').stat().st_ino)
        # A preservation write failure must leave the original partial in place.
        root,backup,evidence,obs,*_=self.fixture('backup-write-failure');original=m.write
        def fail(path,data):
            if Path(path)==backup/'partial.original':raise OSError('FABRICATED_PROJECT_WRITE_FAILURE')
            return original(path,data)
        with mock.patch.object(m,'write',side_effect=fail),self.assertRaisesRegex(OSError,'FABRICATED_PROJECT_WRITE_FAILURE'):
            m.preserve_partial(root,backup,evidence,obs,{'fabricated':True})
        self.assertEqual(m.read(root/m.PARTIAL),self.partial)
        self.assertTrue((backup/'intent.json').exists());self.assertFalse((evidence/'partial.renamed').exists())

    def test_corrupt_or_existing_destination_preserves_all_partial_evidence(self):
        m=self.m
        for label,preexisting in [('corrupt-stream',False),('existing-destination',True)]:
            root,backup,evidence,obs,complete,pins,sizes=self.fixture(label)
            _,ack=self.preserve_and_ack(root,backup,evidence,obs);m.relocate_partial(root,evidence,backup,obs,ack)
            members={n:dict(bytes=len(v),sha256=m.sha(v)) for n,v in complete.items()}
            if preexisting:m.write(root/m.PARTIAL,b'NEW UNEXPECTED FILE MUST SURVIVE')
            wire=b''.join(complete[n] for n in sorted(complete))
            with self.assertRaises(ValueError):m.receive_missing(io.BytesIO(wire[:-1]),root,evidence,members,obs)
            self.assertEqual(m.read(backup/'partial.original'),self.partial)
            self.assertEqual(m.read(evidence/'partial.renamed'),self.partial)
            self.assertFalse((root/'inputs/missing.json').exists())
            if preexisting:self.assertEqual(m.read(root/m.PARTIAL),b'NEW UNEXPECTED FILE MUST SURVIVE')
            else:self.assertFalse((root/m.PARTIAL).exists())
            self.assertFalse((root/'submission.intent.json').exists())

    def test_pinned_original_reconciliation_and_once_submission_ambiguity(self):
        m=self.m;original=(HERE/'references/original-launcher.py').read_bytes()
        common=m.common_from_source(original);self.assertIn('def reconcile():',common)
        with self.assertRaises(ValueError):m.common_from_source(original+b'\n# altered\n')
        collector=ast.parse((HERE/'references/original-collector.py').read_bytes())
        node=next(n for n in collector.body if isinstance(n,ast.FunctionDef) and n.name=='validate_remote_intent')
        scope=dict(need=m.need,sha=m.sha,raw=m.raw,decode=m.decode,pin=lambda x:isinstance(x,str) and len(x)==64)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<original pure intent comparison>','exec'),scope)
        for label,ambiguous in [('positive',False),('ambiguous',True)]:
            root,backup,evidence,obs,*_=self.fixture(label);r,ack=self.preserve_and_ack(root,backup,evidence,obs)
            m.write(evidence/'complete.json',m.raw({'fabricated_completion':True}));m.write(evidence/'pre-submission.json',m.raw({'fabricated_preflight':True}))
            lineage={'fabricated':True};intent={'existing_namespace_claim_sha256':'a'*64,'packet_commit':m.PACKET_COMMIT,'transfer_continuation':lineage}
            req=dict(phase='submit',context={'root':str(root)},projects_backup=str(backup),observation=obs,
                launcher_source=base64.b64encode(original).decode(),preservation_sha256=r['record_sha256'],lineage={'fabricated':True},
                completion_sha256=m.sha(m.read(evidence/'complete.json')),preflight_sha256=m.sha(m.read(evidence/'pre-submission.json')),
                submission_intent=intent,submission_lineage=lineage)
            stub='def reconcile():\n return {"fabricated_full_reconciliation":True}\ndef verify():\n return True\ndef failed_gpu_unchanged():\n return True\n'
            response=subprocess.TimeoutExpired('FABRICATED_SBATCH',25,output=b'FAKE PARTIAL STDOUT') if ambiguous else types.SimpleNamespace(returncode=0,stdout='99999999\n',stderr='')
            # Pure fake filesystem and stubbed scheduler only. No remote_entry production common executes.
            with mock.patch.object(m,'ROOT',str(root)),mock.patch.object(m,'PROJECT_PARENT',str(backup.parent)),mock.patch.object(m,'common_from_source',return_value=stub),mock.patch.object(m.subprocess,'run',side_effect=[response]) as run,mock.patch.object(m.signal,'alarm'),mock.patch.object(m.sys,'stdout',io.StringIO()):
                # Production backup prefix is retained even in this artificial case.
                newbackup=backup.with_name('gpu02-transfer-continuation-fake');backup.rename(newbackup);req['projects_backup']=str(newbackup)
                m.remote_entry(req,io.BytesIO())
                saved=m.decode(m.read(root/'submission.result.json'))
                self.assertEqual(saved['ambiguous'],ambiguous)
                if ambiguous:self.assertIsNone(saved['job_id']);self.assertEqual(saved['stdout'],'FAKE PARTIAL STDOUT')
                else:self.assertEqual(saved['job_id'],'99999999')
                scope['validate_remote_intent'](m.read(root/'submission.intent.json'),saved['intent_sha256'],intent,'a'*64)
                with self.assertRaises(ValueError):m.remote_entry(req,io.BytesIO())
                self.assertEqual(run.call_count,1)

    def test_transport_timeout_retains_primary_and_diagnostics_without_ssh(self):
        m=self.m;outputs=[]
        for n in ('local','mirror'):
            p=self.base/n;p.mkdir();outputs.append(p)
        stdout=self.base/'fake-stdout';stdout.write_bytes(b'')
        stderr=self.base/'fake-stderr';stderr.write_bytes(b'FABRICATED EARLY TRANSFER PROGRESS\n')
        proc=types.SimpleNamespace(pid=987654321,stdin=io.BytesIO(),stdout=stdout.open('rb'),stderr=stderr.open('rb'),poll=lambda:None,wait=lambda timeout:-9)
        original=m.write
        def failed_receipt(path,data):
            if str(path).endswith('.transport.json'):raise OSError('FABRICATED_RECEIPT_WRITE_FAILURE')
            return original(path,data)
        try:
            with mock.patch.object(m.subprocess,'Popen',return_value=proc) as popen,mock.patch.object(m.os,'killpg') as kill,mock.patch.object(m.time,'monotonic',side_effect=[0,631,632]),mock.patch.object(m,'write',side_effect=failed_receipt):
                with self.assertRaisesRegex(TimeoutError,'absolute continuation transport deadline'):
                    m.transport({'phase':'complete'},b'# ARTIFICIAL SOURCE ONLY\n',outputs,members={})
                self.assertEqual(shlex.split(popen.call_args.args[0][-1])[:3],['python3','-I','-B'])
                kill.assert_called_once_with(proc.pid,m.signal.SIGKILL)
            self.assertEqual((outputs[0]/'complete.stderr').read_bytes(),stderr.read_bytes())
            self.assertEqual((outputs[1]/'complete.stderr').read_bytes(),stderr.read_bytes())
            self.assertTrue((outputs[0]/'complete.request.json').exists())
        finally:proc.stdout.close();proc.stderr.close()

if __name__=='__main__':unittest.main(verbosity=2)
