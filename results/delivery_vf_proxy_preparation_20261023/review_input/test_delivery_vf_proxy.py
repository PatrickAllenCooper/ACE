"""Fabricated VF provenance and first-loader checks; no torch/model imports."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q = load('qualify_delivery_replay_runtime')
g = load('delivery_replay_checkpoint_guard')
# Same proxy construction as upstream v2.9.1, with fabricated native functions.
SOURCE = '''import sys
import types
import torch

class VFModule(types.ModuleType):
    vf: types.ModuleType
    def __init__(self, name: str):
        super().__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, name: str) -> object:
        return getattr(self.vf, name)

sys.modules[__name__] = VFModule(__name__)
'''


class VFProxy(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.prefix = self.root/'environment'
        torchroot = self.prefix/'lib/python3.11/site-packages/torch'
        (torchroot/'lib').mkdir(parents=True)
        self.source = torchroot/'_VF.py'
        self.source.write_text(SOURCE)
        self.backing = torchroot/'_C.so'
        self.impl = torchroot/'lib/libtorch_python.so'
        for p in (self.backing,self.impl):p.write_bytes(b'fabricated native bytes, never loaded')
        self.torch = types.ModuleType('torch')
        self.torch.__file__ = str(torchroot/'__init__.py')
        Path(self.torch.__file__).write_text('# fabricated torch package\n')
        self.parent = types.ModuleType('torch._C')
        self.parent.__file__ = str(self.backing)
        self.parent._VariableFunctions = types.ModuleType('torch._C._VariableFunctions')
        self.torch._C = self.parent
        self.context = {'__name__':'torch._VF','__file__':str(self.source)}
        with patch.dict(sys.modules, {'torch':self.torch,'torch._C':self.parent}):
            old = sys.modules.get('torch._VF')
            try:
                exec(compile(SOURCE,str(self.source),'exec',dont_inherit=True),self.context)
                self.proxy = sys.modules['torch._VF']
            finally:
                if old is None:sys.modules.pop('torch._VF',None)
                else:sys.modules['torch._VF']=old
        self.torch._VF = self.proxy
        self.modules = {'torch':self.torch,'torch._C':self.parent,'torch._VF':self.proxy}
        self.native = {str(p):q.sha(p) for p in (self.backing,self.impl)}
        self.synthetic = {str(self.source):q.sha(self.source)}

    def check(self):
        with patch.dict(sys.modules,self.modules):
            return q.torch_vf_proxy(self.proxy,self.prefix,self.native,self.synthetic)

    def test_exact_proxy_binds_source_and_both_native_files(self):
        record=self.check()
        self.assertEqual(record['kind'],'python_native_proxy')
        self.assertEqual({w['backing_file'] for w in record['witnesses']},set(self.native)|set(self.synthetic))
        entries={p.relative_to(self.prefix).as_posix():{'kind':'file','sha256':q.sha(p)} for p in (self.source,self.backing,self.impl)}
        q.bind_loaded_inventory([record],self.prefix,entries)
        entries[self.source.relative_to(self.prefix).as_posix()]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'loaded backing differs'):
            q.bind_loaded_inventory([record],self.prefix,entries)

    def test_missing_extra_or_changed_pins_rejected(self):
        original=dict(self.synthetic)
        for pins in ({},{**original,'/other.py':'0'*64},{str(self.source):'0'*64}):
            with self.subTest(pins=pins):
                self.synthetic=pins
                with self.assertRaises(ValueError):self.check()
        self.synthetic=original
        self.impl.write_bytes(b'changed native implementation')
        with self.assertRaisesRegex(ValueError,'backing bytes'):self.check()

    def test_proxy_and_native_identity_substitution_rejected(self):
        self.torch._VF=types.ModuleType('torch._VF')
        with self.assertRaisesRegex(ValueError,'identity'):self.check()
        self.torch._VF=self.proxy
        self.proxy.vf=types.ModuleType('other native functions')
        with self.assertRaisesRegex(ValueError,'identity'):self.check()

    def test_same_filename_changed_code_rejected(self):
        alternate=compile('def altered(self, name):\n    return None\n',str(self.source),'exec')
        code=next(c for c in alternate.co_consts if isinstance(c,types.CodeType))
        self.context['VFModule'].__getattr__=types.FunctionType(code,self.context)
        with self.assertRaisesRegex(ValueError,'live method'):self.check()

    def test_class_closure_substitution_rejected(self):
        method=self.context['VFModule'].__init__
        method.__closure__[0].cell_contents=types.ModuleType
        with self.assertRaisesRegex(ValueError,'class closure'):self.check()

    def test_altered_globals_and_extra_class_behavior_rejected(self):
        self.context['torch']=types.ModuleType('torch')
        with self.assertRaisesRegex(ValueError,'globals'):self.check()
        self.context['torch']=self.torch
        self.context['VFModule'].__getattribute__=lambda self,name:None
        with self.assertRaisesRegex(ValueError,'extra behavior'):self.check()

    def test_changed_or_symlinked_source_rejected(self):
        self.source.write_text(SOURCE+'# changed\n')
        with self.assertRaisesRegex(ValueError,'backing bytes'):self.check()
        external=self.root/'external.py';external.write_text(SOURCE)
        self.source.unlink();self.source.symlink_to(external)
        with self.assertRaisesRegex(ValueError,'origin'):self.check()

    def test_inventory_dispatch_and_unknown_fileless_still_reject(self):
        with patch.object(sys,'modules',self.modules):
            records=q.loaded_module_records(self.prefix,self.root/'base',self.native,{},self.synthetic)
        self.assertEqual(records[-1]['module'],'torch._VF')
        with patch.object(sys,'modules',{'unexplained':types.ModuleType('unexplained')}):
            with self.assertRaisesRegex(ValueError,'unexplained fileless'):
                q.loaded_module_records(self.prefix,self.root/'base',self.native,{},self.synthetic)

    def test_first_loader_forwards_pins_and_blocks_bad_runtime(self):
        out=self.root/'out';out.mkdir()
        raw=json.dumps({'prefix':str(self.prefix),'files':[]}).encode()
        (out/'runtime_inventory.json').write_bytes(raw)
        (out/'runtime_qualification.json').write_text(json.dumps({'qualified':True,'runtime_inventory_sha256':q.hashlib.sha256(raw).hexdigest(),'dependencies':{}}))
        qualifier=Mock()
        qualifier.dependency_records.return_value={}
        qualifier.verify_environment_inventory.return_value={}
        qualifier.loaded_module_records.side_effect=ValueError('fixture VF changed')
        self.torch.load=Mock()
        reg={'environment':str(self.prefix),'approved_base_prefix':str(self.root/'base'),'native_backing_pins':self.native,'synthetic_backing_pins':self.synthetic}
        fd=os.open(out,os.O_RDONLY)
        try:
            with patch.dict(sys.modules,{'torch':self.torch}):
                original=self.torch.load
                g.install(qualifier,reg,fd,{'files':[]},out,{})
                with self.assertRaisesRegex(ValueError,'fixture VF changed'):self.torch.load(None)
                original.assert_not_called()
                self.assertEqual(qualifier.loaded_module_records.call_args.args[-1],self.synthetic)
        finally:os.close(fd)


if __name__=='__main__':unittest.main(verbosity=2)
