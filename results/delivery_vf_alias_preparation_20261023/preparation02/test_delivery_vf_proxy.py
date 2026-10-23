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
        binding=patch.object(q,'_VF_IMPORT_BINDING',None)
        binding.start();self.addCleanup(binding.stop)
        with patch.dict(sys.modules,self.modules):q.capture_vf_import_binding()

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

    def test_review_global_shadows_and_captured_builtins_rejected(self):
        for name in ('getattr','super'):
            self.context[name]=lambda *args:None
            with self.assertRaisesRegex(ValueError,'global/builtin'):self.check()
            del self.context[name]
        original=self.context['VFModule'].__getattr__
        actual_builtins=self.context['__builtins__']
        self.context['__builtins__']=dict(actual_builtins)
        self.context['VFModule'].__getattr__=types.FunctionType(original.__code__,self.context)
        # Restoring the global mapping cannot restore the function's captured one.
        self.context['__builtins__']=actual_builtins
        with self.assertRaisesRegex(ValueError,'captured builtins'):self.check()
        self.context['VFModule'].__getattr__=original
        self.assertFalse(self.check()['native_execution_attested'])

    def test_review_instance_override_rejected(self):
        self.proxy.relu=lambda *args:None
        with self.assertRaisesRegex(ValueError,'instance'):self.check()
        del self.proxy.relu
        self.check()

    def test_review_joint_reference_replacement_and_recapture_rejected(self):
        original=self.proxy.vf
        self.parent._VariableFunctions=self.proxy.vf=types.ModuleType('replacement')
        with self.assertRaisesRegex(ValueError,'fresh-import'):self.check()
        self.parent._VariableFunctions=self.proxy.vf=original
        with patch.dict(sys.modules,self.modules):
            with self.assertRaisesRegex(ValueError,'already captured'):q.capture_vf_import_binding()
        self.check()

    def test_review_exact_upstream_python_source_with_fabricated_native(self):
        reference=Path(__file__).resolve().parents[2]/'results/delivery_vf_proxy_preparation_20261023/references/pytorch_2_9_1_VF.py'
        raw=reference.read_bytes()
        self.assertEqual(q.hashlib.sha256(raw).hexdigest(),'5c87e1f29223cef1b4c9259c392733e04d3aec31684fef1391c9801b9a189fde')
        self.source.write_bytes(raw)
        context={'__name__':'torch._VF','__file__':str(self.source)}
        with patch.dict(sys.modules,self.modules):
            exec(compile(raw,str(self.source),'exec',dont_inherit=True),context)
            self.proxy=sys.modules['torch._VF']
        self.torch._VF=self.proxy;self.modules['torch._VF']=self.proxy
        self.context=context;self.synthetic={str(self.source):q.sha(self.source)}
        with patch.object(q,'_VF_IMPORT_BINDING',None),patch.dict(sys.modules,self.modules):
            q.capture_vf_import_binding()
            self.assertFalse(self.check()['native_execution_attested'])

    def alias_fixture(self):
        alias=self.root/'scratch-alias'
        alias.symlink_to(self.prefix,target_is_directory=True)
        source=alias/self.source.relative_to(self.prefix)
        context={'__name__':'torch._VF','__file__':str(source)}
        with patch.dict(sys.modules,self.modules):
            exec(compile(SOURCE,str(source),'exec',dont_inherit=True),context)
            self.proxy=sys.modules['torch._VF']
        self.torch._VF=self.proxy;self.modules['torch._VF']=self.proxy
        self.context=context
        self.parent.__file__=str(alias/self.backing.relative_to(self.prefix))
        self.native={str(alias/Path(p).relative_to(self.prefix)):pin for p,pin in self.native.items()}
        self.synthetic={str(source):q.sha(self.source)}
        q._VF_IMPORT_BINDING=None
        with patch.dict(sys.modules,self.modules):q.capture_vf_import_binding()
        return alias

    def test_site_alias_accepts_same_source_native_and_live_filename(self):
        alias=self.alias_fixture()
        record=self.check()
        self.assertEqual({w['backing_file'] for w in record['witnesses']},
                         {str(self.source),str(self.backing),str(self.impl)})
        self.assertFalse(record['native_execution_attested'])
        with patch.dict(sys.modules,self.modules):
            self.assertEqual(q.torch_vf_proxy(self.proxy,alias,self.native,self.synthetic),record)

    def test_site_alias_duplicate_wrong_digest_and_external_rejected(self):
        self.alias_fixture();original=dict(self.synthetic)
        extra=self.root/'external.py';extra.write_text(SOURCE)
        alternatives=[{**original,str(self.source):q.sha(self.source)},
                      {next(iter(original)):'0'*64},{str(extra):q.sha(extra)}]
        for pins in alternatives:
            with self.subTest(pins=pins):
                self.synthetic=pins
                with self.assertRaises(ValueError):self.check()
        self.synthetic=original
        self.native[str(self.backing)]=q.sha(self.backing)
        with self.assertRaisesRegex(ValueError,'duplicate canonical'):self.check()

    def test_site_alias_rejects_leaf_symlink_and_false_live_filename(self):
        self.alias_fixture()
        method=self.context['VFModule'].__getattr__
        alternate=self.prefix/'other.py';alternate.write_text(SOURCE)
        self.context['VFModule'].__getattr__=types.FunctionType(
            method.__code__.replace(co_filename=str(alternate)),self.context)
        with self.assertRaisesRegex(ValueError,'source origin'):self.check()
        self.context['VFModule'].__getattr__=method
        original_bytes=self.source.read_bytes();self.source.unlink()
        self.source.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError,'origin'):self.check()
        self.source.unlink();self.source.write_bytes(original_bytes)
        self.check()

    def test_site_alias_early_pin_check_needs_no_dependency_import(self):
        self.alias_fixture()
        with patch.object(q,'dependency_records',side_effect=AssertionError('imports forbidden')):
            source,pins=q.vf_source_pin(self.prefix,self.synthetic)
            self.assertEqual(source,self.source)
            self.assertEqual(pins,{str(self.source):q.sha(self.source)})
            for invalid in ({},[],{str(self.source):'invalid'}, {'relative.py':'0'*64}):
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                    q.vf_source_pin(self.prefix,invalid)


if __name__=='__main__':unittest.main(verbosity=2)
