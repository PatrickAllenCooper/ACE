"""New reader-only artificial boundaries. Never executes any coordinator main."""
import ast
import hashlib
import json
import os
from pathlib import Path
import types
import unittest

ROOT=Path(__file__).resolve().parent
IMPL=ROOT/'implementation'
FIXTURES=Path(os.environ['ROLE_BOUND_FIXTURES'])
MIB=1024**2
ACTUAL_RECEIPT_SIZE=33829045


def load_parts(relative, functions, constants=()):
    path=IMPL/relative
    tree=ast.parse(path.read_bytes())
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            # Explicitly exclude launcher sibling imports and any possible main.
            if isinstance(node,ast.Import) and any(x.name=='build_packet' for x in node.names):continue
            nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):nodes.append(node)
    module=types.ModuleType('captured_reader_fixture')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),vars(module))
    return module


def sparse(path,size):
    with path.open('xb') as f:
        f.seek(size-4);f.write(b'FAKE')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(MIB),b''):h.update(block)
    return {'bytes':size,'sha256':h.hexdigest()}


def raw(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()


class RoleBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURES.mkdir(mode=0o700)
        cls.large=FIXTURES/'fabricated-full-map.json';cls.large_info=sparse(cls.large,ACTUAL_RECEIPT_SIZE)
        cls.edge=FIXTURES/'fabricated-128mib.json';cls.edge_info=sparse(cls.edge,128*MIB)
        cls.over=FIXTURES/'fabricated-128mib-plus-one.json';cls.over_info=sparse(cls.over,128*MIB+1)
        cls.builders={stage:load_parts(stage+'/build_packet.py',{'need','sha','pin','read','metadata_maximum'},{'FULL_MAP_METADATA'}) for stage in ('gpu','finalizer')}
        cls.cli=load_parts('cli/assemble_gpu_cli.py',{'need','sha','capture','metadata_maximum'},{'FULL_MAP_METADATA','MIB'})
        cls.collectors={stage:load_parts(stage+'/pull_terminal.py',{'need','sha','pin','fingerprint','read_metadata','packet_member_maximum'},{'FULL_MAP_PACKET_MEMBERS'}) for stage in ('gpu','finalizer')}
        cls.adapters={stage:load_parts(stage+'/compact_inputs.py',{'need','sha','canonical','decode','pin','record','read_exact','capture_config','stored_config'},{'STORAGE_FILES'}) for stage in ('gpu','finalizer')}

    def test_builder_and_cli_named_maps_capture_exact_bytes_ordinary32_rejects(self):
        roles=set(self.builders['gpu'].FULL_MAP_METADATA)
        self.assertEqual(roles,set(self.builders['finalizer'].FULL_MAP_METADATA))
        self.assertEqual(roles,set(self.cli.FULL_MAP_METADATA))
        for name,module in list(self.builders.items())+[('cli',self.cli)]:
            with self.subTest(reader=name):
                reader=module.capture if name=='cli' else module.read
                for role in roles:self.assertEqual(module.metadata_maximum(role),128*MIB)
                self.assertEqual(module.metadata_maximum('site-readiness'),32*MIB)
                result=reader(self.large,self.large_info['sha256'],module.metadata_maximum('cpu-disposition'))
                self.assertEqual(len(result),ACTUAL_RECEIPT_SIZE);self.assertEqual(digest(result),self.large_info['sha256'])
                with self.assertRaises(ValueError):reader(self.large,self.large_info['sha256'],module.metadata_maximum('site-readiness'))
                with self.assertRaises(ValueError):reader(self.large,'0'*64,module.metadata_maximum('cpu-acceptance'))

    def test_collectors_read_supplied_maps_and_exact_named_packet_members(self):
        gpu=self.collectors['gpu']
        for n in gpu.FULL_MAP_PACKET_MEMBERS:self.assertEqual(gpu.packet_member_maximum(n),128*MIB)
        for n in ('build_packet.py','storage-config.json','unlisted/storage-receipt.json','ordinary.json'):
            self.assertEqual(gpu.packet_member_maximum(n),32*MIB)
        value=gpu.read_metadata(self.large,self.large_info['sha256'],gpu.packet_member_maximum('storage-receipt.json'))
        self.assertEqual(digest(value),self.large_info['sha256'])
        with self.assertRaises(ValueError):gpu.read_metadata(self.large,self.large_info['sha256'],gpu.packet_member_maximum('build_packet.py'))
        # Run ONLY each captured supplied-metadata comprehension, with fabricated
        # metadata filenames and reader globals; never execute main/locks/Git.
        folder=FIXTURES/'supplied';folder.mkdir()
        info=sparse(folder/'cpu-inventory.json',ACTUAL_RECEIPT_SIZE)
        (folder/'freeze.json').write_bytes(b'opaque-fabricated-small')
        pins={'cpu-inventory.json':info['sha256'],'freeze.json':digest(b'opaque-fabricated-small')}
        for stage,c in self.collectors.items():
            tree=ast.parse((IMPL/stage/'pull_terminal.py').read_bytes())
            main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            assign=next(n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='supplied' for t in n.targets))
            function=ast.FunctionDef(name='only_supplied_reader',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[assign,ast.Return(value=ast.Name(id='supplied',ctx=ast.Load()))],decorator_list=[])
            mod=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
            c.LAUNCH=folder;c.pins=pins;c.b=types.SimpleNamespace(METADATA=('cpu-inventory','freeze'),metadata_name=lambda n:n+'.json',metadata_maximum=self.builders[stage].metadata_maximum)
            exec(compile(mod,'<captured-new-supplied-reader-only>','exec'),vars(c))
            captured=c.only_supplied_reader()
            self.assertEqual(digest(captured['cpu-inventory']),info['sha256']);self.assertEqual(captured['freeze'],b'opaque-fabricated-small')

    def test_storage_capture_and_stored_config_named_roles_preserve_bound_and_bytes(self):
        folder=FIXTURES/'storage';folder.mkdir()
        adapter=self.adapters['gpu']
        members={};pins={}
        for role,name in adapter.STORAGE_FILES.items():
            path=folder/name
            if role in ('receipt','seal','after'):info=sparse(path,ACTUAL_RECEIPT_SIZE)
            else:path.write_bytes(b'opaque-fabricated-small');info={'bytes':path.stat().st_size,'sha256':digest(path.read_bytes())}
            members[role]={'path':str(path),'sha256':info['sha256']};pins[name]=info['sha256']
        config=raw({'schema':'ace-compact-downstream-inputs-v1','predecessor':'cpu','members':members})
        path=folder/'storage-config.json';path.write_bytes(config);pins[path.name]=digest(config)
        for stage,adapter in self.adapters.items():
            with self.subTest(adapter=stage):
                captured=adapter.capture_config(path,pins[path.name])
                stored=adapter.stored_config(folder,pins)
                self.assertEqual(captured,stored)
                for role in ('receipt','seal','after'):
                    name=adapter.STORAGE_FILES[role];self.assertEqual(len(stored[name]),ACTUAL_RECEIPT_SIZE);self.assertEqual(digest(stored[name]),pins[name])
                # Ordinary backend/source role retains its32MiB boundary even if
                # its independently pinned bytes also appear as a full-map role.
                bad=json.loads(config);bad['members']['backend']=dict(members['receipt'])
                badraw=raw(bad);badpath=folder/(stage+'-ordinary-too-large.json');badpath.write_bytes(badraw)
                with self.assertRaises(ValueError):adapter.capture_config(badpath,digest(badraw))

    def test_exact128_accepts_plus_one_rejects_and_internal_pins_match(self):
        for stage,module in self.builders.items():
            result=module.read(self.edge,self.edge_info['sha256'],module.metadata_maximum('cpu-acceptance'))
            self.assertEqual(len(result),128*MIB);self.assertEqual(digest(result),self.edge_info['sha256'])
            with self.assertRaises(ValueError):module.read(self.over,self.over_info['sha256'],module.metadata_maximum('cpu-acceptance'))
        with self.assertRaises(ValueError):self.cli.capture(self.over,self.over_info['sha256'],self.cli.metadata_maximum('gpu-inventory'))
        for c in self.collectors.values():
            with self.assertRaises(ValueError):c.read_metadata(self.over,self.over_info['sha256'],128*MIB)
        for a in self.adapters.values():
            with self.assertRaises(ValueError):a.read_exact(self.over,self.over_info,128*MIB)
        def constant(path,name):
            for n in ast.parse(path.read_bytes()).body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
            raise AssertionError(name)
        for stage in ('gpu','finalizer'):
            self.assertEqual(constant(IMPL/stage/'build_packet.py','COMPACT_ADAPTER_SHA'),digest((IMPL/stage/'compact_inputs.py').read_bytes()))
        self.assertEqual(constant(IMPL/'gpu/pull_terminal.py','BUILDER_SHA'),digest((IMPL/'gpu/build_packet.py').read_bytes()))
        knownraw=(IMPL/'cli/known-inputs.json').read_bytes();known=json.loads(knownraw)
        self.assertEqual(constant(IMPL/'cli/assemble_gpu_cli.py','KNOWN_SHA256'),digest(knownraw))
        for n,h in known['builder_sources'].items():self.assertEqual(h,digest((IMPL/'gpu'/n).read_bytes()))
        self.assertEqual(set(known['full_map_metadata']),set(self.builders['gpu'].FULL_MAP_METADATA))
        self.assertEqual(known['known_metadata'],json.loads((ROOT/'predecessor/cli/known-inputs.json').read_bytes())['known_metadata'])


if __name__=='__main__':unittest.main(verbosity=2)
