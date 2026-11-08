"""New fabricated TOKEN transport glue tests; no SSH, builder main or science."""
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
C=load('token_collector_glue',HERE/'pull_terminal.py')
H=load('accepted_stream_helper',HERE/'stream_custody.py')
SOURCE=(HERE/'stream_custody.py').read_bytes()

class TransportIntegration(unittest.TestCase):
    def setUp(self):
        self.root=EVIDENCE/self._testMethodName;self.root.mkdir()
        self.ext=self.root/'external';self.local=self.root/'local'
        self.ext.mkdir();self.local.mkdir()
        source=self.root/'fabricated';source.mkdir()
        for label in ('launch','common','failed01'):
            p=source/label;p.mkdir();(p/'empty').mkdir();(p/'opaque.bin').write_bytes(b'fabricated\x00'+label.encode())
        for label in ('namespace.json','original-namespace.json'):(source/label).write_bytes(b'opaque fabricated claim')
        C.ROOT=str(source/'launch');C.COMMON_ROOT=str(source/'common');C.FAILED_ROOT=str(source/'failed01')
        C.REMOTE_NAMESPACE=str(source/'namespace.json');C.OLD_REMOTE_NAMESPACE=str(source/'original-namespace.json')
        self.before=dict(**C.inventory(source),root_present=True,common_present=True,failed_present=True,
            namespace_present=True,original_namespace_present=True,original_hardlink_groups=[],
            parent_accounting=['fabricated'],journal_pins={'old':'a'*64},upstream_pins={'opaque':'b'*64},
            upstream_errors=[],related_accounting=[])
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
        for filename in ('launch_once.py','monitor_once.py'):
            self.assertEqual((HERE/filename).read_bytes(),(HERE/'predecessors'/filename).read_bytes())
        original=ast.parse((HERE/'predecessors/build_packet.py').read_text());new=ast.parse((HERE/'build_packet.py').read_text())
        # Root helper bundling is the only builder change; remove its single added loop.
        main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loops=[n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and
            any(isinstance(x,ast.Constant) and x.value=='stream_custody.py' for x in ast.walk(n.iter))]
        self.assertEqual(len(loops),1)
        for node in ast.walk(main):
            for _,value in ast.iter_fields(node):
                if isinstance(value,list) and loops[0] in value:value.remove(loops[0])
        self.assertEqual(ast.dump(original),ast.dump(new))
    def test_precreated_raw_directory_rejected_before_capture(self):
        (self.local/'remote').mkdir()
        with self.assertRaisesRegex(ValueError,'must remain absent'):self.transfer()
        self.assertEqual(self.calls,[]);self.assertTrue((self.local/'remote').is_dir())
    def test_after_journal_change_retains_archive_without_materialization(self):
        after=copy.deepcopy(self.before);after['journal_pins']['old']='c'*64
        with self.assertRaisesRegex(ValueError,'second remote'):self.transfer(after)
        self.assertEqual(C.inventory(self.ext/'stream-attempt'),C.inventory(self.local/'stream-attempt'))
        self.assertFalse((self.ext/'remote').exists());self.assertFalse((self.ext/'stream-transport.json').exists())
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

if __name__=='__main__':unittest.main(verbosity=2)
