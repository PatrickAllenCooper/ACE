"""Fabricated child-order checks; no actual libraries, checkpoints or runtime."""
import base64
import os
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'new_wrapper', Path(__file__).with_name('supervise_delivery_qualified_replay.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

QUALIFIER = r'''
import hashlib,json,sys,types
from pathlib import Path
def main():
 reg=json.loads(Path(sys.argv[sys.argv.index('--freeze')+1]).read_bytes())
 root=Path(reg['output'])
 if reg['fixture_case']=='qualification_failure':raise ValueError('fixture qualifier failed')
 inv=json.dumps({'prefix':reg['environment'],'files':[]}).encode()
 (root/'runtime_inventory.json').write_bytes(inv)
 evidence={'qualified':True,'freeze_sha256':sys.argv[-1],'runtime_inventory_sha256':hashlib.sha256(inv).hexdigest(),'dependencies':{'torch':{'imported':'fabricated'}}}
 (root/'runtime_qualification.json').write_text(json.dumps(evidence))
 if reg['fixture_case']=='inventory_changed':(root/'runtime_inventory.json').write_bytes(b'{}')
 (root/'qualification_finished.json').write_text('{}')
 torch=types.ModuleType('torch')
 def original_load(*args,**kwargs):
  if not (root/'qualification_finished.json').exists():raise ValueError('model loading before qualification')
  if not (root/'pre_checkpoint_runtime.json').exists():raise ValueError('model loading before live guard')
  (root/'model_load_attempt.json').write_text('{}')
  return 'fabricated checkpoint, no file accessed'
 torch.load=original_load;sys.modules['torch']=torch
 globals()['root']=root;globals()['case']=reg['fixture_case']
def dependency_records(reg,reject_cached=False):
 return {'torch':{'imported':'different' if case=='live_import_changed' else 'fabricated'}}
def search_path_check(prefix,base):
 if case=='external_search_path':raise ValueError('external current path')
def loaded_module_records(*args):return []
def verify_environment_inventory(*args):return {}
def bind_loaded_inventory(*args):pass
'''

CLI = r'''
import json,sys,torch
from pathlib import Path
assert vars(sys.modules['__main__']) is globals()
assert sys.modules['__main__'].__file__==__file__
assert torch.load(None)=='fabricated checkpoint, no file accessed'
Path(sys.argv[sys.argv.index('--receipt')+1]).write_text(json.dumps({'fabricated':True}))
'''


class ChildOrder(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve() / 'output'
            root.mkdir(mode=0o700)
            package = Path(td).resolve() / 'package'
            package.mkdir()
            source = {'replay_delivery_prospective_release.py': CLI.encode(),
                      'verify_delivery_release.py': b'# fabricated empty verifier\n'}
            for name, raw in source.items():
                (package / name).write_bytes(raw)
            (root / 'qualify_runtime.py').write_text(QUALIFIER)
            manifest = json.dumps({'files': [{'path': name, 'sha256': w.digest(raw)}
                                            for name, raw in source.items()]}).encode()
            (package / 'manifest.json').write_bytes(manifest)
            reg = {'output': str(root), 'output_inode': root.stat().st_ino,
                   'environment': str(Path(td) / 'fabricated_env'),
                   'approved_base_prefix': str(Path(td) / 'fabricated_base'),
                   'native_backing_pins': {}, 'fixture_case': case,
                   'qualification_worker_sha256': w.digest(QUALIFIER.encode())}
            guard = Path(__file__).with_name('delivery_replay_checkpoint_guard.py').read_bytes()
            reg['execution_source_pins']={'child_bootstrap.py': w.digest(w.CHILD_BOOTSTRAP.encode()),'checkpoint_guard.py': w.digest(guard),'qualify_runtime.py':w.digest(QUALIFIER.encode())}
            for name,raw in [('checkpoint_guard.py',guard),('child_bootstrap.py',w.CHILD_BOOTSTRAP.encode())]:
                (root/name).write_bytes(raw)
            freeze = json.dumps(reg).encode()
            (root / 'freeze.json').write_bytes(freeze)
            enc = lambda raw: base64.b64encode(raw).decode()
            # Darwin lacks /proc; only its pathname projection is emulated.
            # Descriptor reads/writes and the production bootstrap/guard execute;
            # this is not Linux custody or actual runtime qualification.
            projection="import pathlib; old=pathlib.Path.__new__; logical="+repr(str(root))+"; pathlib.Path.__new__=staticmethod(lambda cls,*args,**kwargs:old(cls,*[(logical+str(x).split('/proc/self/fd/'+"+repr('FD')+",1)[-1]) if str(x).startswith('/proc/self/fd/'+"+repr('FD')+") else x for x in args],**kwargs)); "
            fd=os.open(root,os.O_RDONLY)
            projection=projection.replace(repr('FD'),repr(str(fd)))
            loader=projection+"import base64,hashlib,sys; source,path,*args=sys.argv[1:]; raw=base64.b64decode(source); context=vars(sys.modules['__main__']); context.update(__name__='__main__',__file__=path,__executed_source_sha256__=hashlib.sha256(raw).hexdigest()); sys.argv=[path,*args];exec(compile(raw,path,'exec'),context)"
            try:
                result = subprocess.run([
                    sys.executable, '-I', '-B', '-c', loader,
                    enc(w.CHILD_BOOTSTRAP.encode()), str(root/'child_bootstrap.py'),
                    str(package), w.digest(manifest), str(fd), enc(CLI.encode()),
                    enc(source['verify_delivery_release.py']), enc(QUALIFIER.encode()),
                    enc(guard), enc(manifest), w.digest(freeze)],
                    capture_output=True,text=True,timeout=15,pass_fds=(fd,))
            finally:os.close(fd)
            return result.returncode, result.stderr, {f.name for f in root.iterdir()}

    def test_qualification_and_live_guard_before_loader(self):
        code, err, names = self.run_case('success')
        self.assertEqual(code, 0, err)
        self.assertTrue({'runtime_inventory.json', 'runtime_qualification.json',
                         'pre_checkpoint_runtime.json', 'model_load_attempt.json',
                         'replay_receipt.json'} <= names)

    def test_qualification_failure_prevents_loader(self):
        code, err, names = self.run_case('qualification_failure')
        self.assertNotEqual(code, 0)
        self.assertIn('fixture qualifier failed', err)
        self.assertNotIn('model_load_attempt.json', names)

    def test_inventory_change_prevents_loader(self):
        code, err, names = self.run_case('inventory_changed')
        self.assertNotEqual(code, 0)
        self.assertIn('runtime qualification/inventory differs', err)
        self.assertNotIn('model_load_attempt.json', names)

    def test_current_import_change_prevents_loader(self):
        code, err, names = self.run_case('live_import_changed')
        self.assertNotEqual(code, 0)
        self.assertIn('live dependencies changed', err)
        self.assertNotIn('model_load_attempt.json', names)

    def test_current_search_path_rejection_prevents_loader(self):
        code, err, names = self.run_case('external_search_path')
        self.assertNotEqual(code, 0)
        self.assertIn('external current path', err)
        self.assertNotIn('model_load_attempt.json', names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
