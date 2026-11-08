"""New fabricated GPU collector transport integration; no remote, model or science."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import types
import unittest

HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_TRANSPORT_TEST_EVIDENCE']).resolve()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
C=load('gpu_collector_glue',HERE/'pull_terminal.py')
H=load('accepted_stream_helper',HERE/'stream_custody.py')
SOURCE=(HERE/'stream_custody.py').read_bytes()

class TransportIntegration(unittest.TestCase):
    def setUp(self):
        self.root=EVIDENCE/self._testMethodName;self.root.mkdir()
        self.ext=self.root/'external';self.local=self.root/'local'
        self.ext.mkdir();self.local.mkdir()
        source=self.root/'fabricated';source.mkdir()
        for label in ('launch','cpu','common','failed01'):
            p=source/label;p.mkdir();(p/'empty').mkdir();(p/'opaque.bin').write_bytes(b'fabricated\x00'+label.encode())
        for label in ('namespace.json','original-namespace.json'):(source/label).write_bytes(b'opaque fabricated claim')
        C.ROOT=str(source/'launch');C.CPU_ROOT=str(source/'cpu');C.COMMON_ROOT=str(source/'common');C.FAILED_ROOT=str(source/'failed01')
        C.REMOTE_NAMESPACE=str(source/'namespace.json');C.OLD_REMOTE_NAMESPACE=str(source/'original-namespace.json')
        self.before=dict(**C.inventory(source),root_present=True,cpu_present=True,common_present=True,failed_present=True,
            namespace_present=True,original_namespace_present=True,original_hardlink_groups=[],
            parent_accounting=['fabricated'],journal_pins={f'journal-{i}':'a'*64 for i in range(90)},upstream_pins={'opaque':'b'*64},
            upstream_errors=[],related_accounting=[[str(i+1),'fixture','FAILED' if i==0 else 'COMPLETED','0:0','2','4','fixture','cpu=4'] for i in range(44)])
        self.calls=[]
        self.helper=types.SimpleNamespace(request_check=H.request_check,archive_bound=H.archive_bound,
            capture=self.capture,materialize=H.materialize)
    def save(self,name,value):
        for dest in (self.ext,self.local):C.write(dest/name,C.raw(value))
    def capture(self,command,request,attempt,timeout):
        self.calls.append(command)
        self.assertEqual(command[:3],['bash',C.HELPER,'run'])
        self.assertEqual(shlex.split(command[3]),['python3','-B','-c',SOURCE.decode(),'emit','--request-sha256',C.sha(request)])
        self.assertEqual(timeout,1800)
        attempt.mkdir();C.write(attempt/'request.json',request);C.write(attempt/'command.json',C.raw(command))
        with (attempt/'archive.tar').open('xb') as f:emitted=H.emit(C.decode(request),f)
        C.write(attempt/'stderr.log',(json.dumps(emitted,sort_keys=True)+'\n').encode())
        receipt=dict(status='captured_pending_inventory_verification',returncode=0,acceptance=False,
            request_sha256=C.sha(request),command_sha256=C.sha(C.raw(command)),
            **{name:C.fingerprint(attempt/name,sole_link=True) for name in ('archive.tar','stderr.log')})
        C.write(attempt/'capture.json',C.raw(receipt));return receipt
    def transfer(self,after=None):
        return C.stream_transfer(self.before,observe=lambda _:copy.deepcopy(self.before if after is None else after),
            save=self.save,helper=self.helper,helper_source=SOURCE,external=self.ext,local=self.local,copy_file=shutil.copyfile)
    def test_complete_two_copies_and_full_archive_retained(self):
        actual,_=self.transfer()
        self.assertEqual(actual,dict(files=self.before['files'],directories=self.before['directories']))
        self.assertEqual(C.inventory(self.ext),C.inventory(self.local))
        self.assertEqual(len(self.calls),1)
        for name in actual['files']:
            self.assertNotEqual((self.ext/'remote'/name).stat().st_ino,(self.local/'remote'/name).stat().st_ino)
        self.assertNotEqual((self.ext/'stream-attempt/archive.tar').stat().st_ino,(self.local/'stream-attempt/archive.tar').stat().st_ino)
        self.assertFalse(C.decode((self.ext/'stream-transport.json').read_bytes())['acceptance'])
        self.assertEqual(set(C.stream_request(self.before)['roots']),{'launch','cpu','common','failed01'})
        self.assertEqual(len(self.before['journal_pins']),90)
        self.assertEqual(len(self.before['related_accounting']),44)
        self.assertIn('failed01/opaque.bin',actual['files'])
        # Stand-in checkpoint bytes are opaque, copied separately, never deserialized.
        self.assertEqual((self.ext/'remote/launch/opaque.bin').read_bytes(),b'fabricated\x00launch')
    def test_cpu_root_presence_or_journal_drift_preserves_archive(self):
        after=copy.deepcopy(self.before);after['cpu_present']=False
        with self.assertRaisesRegex(ValueError,'second remote'):self.transfer(after)
        self.assertEqual(C.inventory(self.ext/'stream-attempt'),C.inventory(self.local/'stream-attempt'))
        self.assertFalse((self.ext/'remote').exists());self.assertFalse((self.ext/'stream-transport.json').exists())
        after=copy.deepcopy(self.before);after['journal_pins']['journal-0']='c'*64
        with self.assertRaisesRegex(ValueError,'second remote'):C.compare_observations(self.before,after)
        after=copy.deepcopy(self.before);after['related_accounting'][0][4]='3'
        with self.assertRaisesRegex(ValueError,'second remote'):C.compare_observations(self.before,after)

    def test_partial_capture_failure_mirrors_exact_bytes_without_retry(self):
        def failed(command,request,attempt,timeout):
            self.calls.append(command);attempt.mkdir()
            for name,data in [('request.json',request),('command.json',C.raw(command)),('archive.tar',b'partial\x00archive'),('stderr.log',b'fabricated failure')]:C.write(attempt/name,data)
            raise RuntimeError('original fabricated capture failure')
        self.helper.capture=failed
        with self.assertRaisesRegex(RuntimeError,'original fabricated'):self.transfer()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(C.inventory(self.ext/'stream-attempt'),C.inventory(self.local/'stream-attempt'))
        self.assertFalse((self.ext/'remote').exists());self.assertFalse((self.ext/'stream-transport.json').exists())

    def test_materialization_failure_retains_archive_and_partial_raw(self):
        def failed(attempt,*args):
            (self.ext/'remote').mkdir()
            C.write(self.ext/'remote/partial-checkpoint.bin',b'fabricated partial checkpoint')
            C.write(attempt/'materialization-failure.json',C.raw(dict(acceptance=False,error='fixture')))
            raise RuntimeError('fabricated materialization failure')
        self.helper.materialize=failed
        with self.assertRaisesRegex(RuntimeError,'fabricated materialization'):self.transfer()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(C.inventory(self.ext/'stream-attempt'),C.inventory(self.local/'stream-attempt'))
        self.assertEqual((self.ext/'remote/partial-checkpoint.bin').read_bytes(),b'fabricated partial checkpoint')
        self.assertFalse((self.ext/'stream-transport.json').exists())
        self.assertFalse((self.local/'remote').exists())  # main retains best-effort partial mirror, unchanged.

    def test_standalone_commit_binding_fails_before_helper_execution(self):
        from unittest.mock import patch
        commit='1'*40;helper_path='sidecars/stream_custody.py';review_path='sidecars/stream_custody_review.md'
        review=(HERE/'stream_custody_review.md').read_bytes()
        def git(command,**kw):
            self.assertEqual(command[:3],['git','--no-replace-objects','show'])
            self.assertEqual(kw['cwd'],C.REPO)
            self.assertEqual(kw['timeout'],20)
            data={commit+':'+helper_path:SOURCE,commit+':'+review_path:review}[command[3]]
            return types.SimpleNamespace(returncode=0,stdout=data)
        with patch.object(C.subprocess,'run',side_effect=git):
            self.assertEqual(C.authenticate_stream_files(commit,helper_path,review_path),(SOURCE,review))
        with patch.object(C.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=b'wrong')):
            with self.assertRaisesRegex(ValueError,'full packet commit bytes'):C.authenticate_stream_files(commit,helper_path,review_path)
        with patch.object(C.subprocess,'run',side_effect=AssertionError('must not invoke git')):
            with self.assertRaisesRegex(ValueError,'safe explicit'):C.authenticate_stream_files(commit,'../bad.py',review_path)
        with patch.object(C,'read_metadata',return_value=b'untrusted'):
            with patch.object(C.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stdout=b'')):
                with self.assertRaisesRegex(ValueError,'full packet commit bytes'):C.authenticate_stream_files(commit,helper_path,review_path)
        with self.assertRaisesRegex(ValueError,'independently reviewed'):C.load_stream_helper(b'untrusted')

    def test_original_custody_accounting_gpu_and_admission_gates_unchanged(self):
        old_text=(HERE/'predecessors/gpu-collector/pull_terminal.py').read_text()
        new_text=(HERE/'pull_terminal.py').read_text()
        old=ast.parse(old_text);new=ast.parse(new_text)
        functions=lambda tree:{n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='main'}
        for name,value in functions(old).items():self.assertEqual(value,functions(new)[name],name)
        # The entire post-transfer authority/error/cost/44-job90-journal/partial-custody path is byte-identical.
        marker='        errors=[]\n        if inventory(cpu_custody)'
        self.assertEqual(old_text.split(marker,1)[1],new_text.split(marker,1)[1])
        for tree in (old,new):
            remote=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_OBSERVATION' for t in n.targets))
            if tree is old:original_remote=remote
            else:self.assertEqual(original_remote,remote)
        main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        lines={}
        for n in ast.walk(main):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('authenticate_stream_files','load_stream_helper','stream_transfer'):
                lines[n.func.id]=n.lineno
        self.assertLess(lines['authenticate_stream_files'],lines['load_stream_helper'])
        self.assertLess(lines['load_stream_helper'],lines['stream_transfer'])
        self.assertNotIn("command=['scp'",new_text)
        self.assertIn("for sub in ('coordinator','upstream-cpu-custody'):",new_text)
        self.assertIn("for sub in ('remote','coordinator','upstream-cpu-custody'):",new_text)  # retained failure mirroring
        self.assertIn("for sub in ('coordinator','namespaces'):",new_text)

if __name__=='__main__':unittest.main(verbosity=2)
