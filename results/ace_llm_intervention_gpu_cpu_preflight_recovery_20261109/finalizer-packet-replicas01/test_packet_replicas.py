"""New artificial metadata/opaque-byte cases only; fixtures are retained."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import types
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('toy_replica_adapter',HERE/'implementation/compact_inputs.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
FIXTURES=Path(os.environ['ACE_REPLICA_FIXTURES']).resolve()


class ReplicaTests(unittest.TestCase):
    serial=0

    def fixture(self):
        type(self).serial+=1
        top=FIXTURES/('%02d'%self.serial);top.mkdir()
        roots=[top/'repo-packet',top/'external-packet']
        protected=[top/'sealed-local',top/'sealed-external',top/'operations']
        for p in [*roots,*protected]:p.mkdir()
        payload={'toy.py':b'# inert supplied source; never executed\n',
                 'descriptor.json':b'opaque fabricated bytes, deliberately not JSON\n',
                 'adapter.pt':b'\x00toy opaque bytes\xff'}
        inner=c.canonical({'files':{n:c.sha(v) for n,v in payload.items()}})
        inputs={**payload,'packet.json':inner,'admission.json':b'{"toy":true}\n'}
        outer=c.canonical({'remote_root':'/scratch/toy/control','gpu_packet_sha256':c.sha(inner),
                          'new_input_files':{'inputs/'+n:c.sha(v) for n,v in inputs.items()}})
        files={'control/packet.json':{'bytes':len(outer),'sha256':c.sha(outer)},
               **{'control/inputs/'+n:{'bytes':len(v),'sha256':c.sha(v)} for n,v in inputs.items()}}
        after={'terminal_gate':True,'files':files,'directories':['control','control/inputs']}
        after_raw=c.canonical({'returncode':0,'stdout':c.canonical(after).decode()})
        for root in roots:
            (root/'inputs').mkdir();(root/'packet.json').write_bytes(outer)
            for n,v in inputs.items():(root/'inputs'/n).write_bytes(v)
        kwargs=dict(roots=[str(p) for p in roots],outer_sha256=c.sha(outer),inner_sha256=c.sha(inner),
                    after_raw=after_raw,after_sha256=c.sha(after_raw),control_root='/scratch/toy/control',
                    protected_roots=protected)
        (top/'fixture-context.json').write_bytes(c.canonical({k:v for k,v in kwargs.items()
            if k not in ('after_raw','protected_roots')}))
        (top/'after.json').write_bytes(after_raw)
        return top,roots,payload,kwargs

    def test_two_copies_detached_bytes_and_later_conflict(self):
        top,roots,payload,k=self.fixture();r=c.PacketReplicas(**k)
        for n,v in payload.items():
            self.assertEqual(r.capture('control/inputs/'+n,c.sha(v)),v)
        binding=r.binding();self.assertFalse(binding['acceptance']);self.assertFalse(binding['fallback_allowed'])
        self.assertEqual(binding['input_files'],len(payload)+2)
        binding['roots'][0]='untrusted mutation'
        self.assertEqual(r.binding()['roots'],k['roots'])
        # A same-size conflict is detected on recheck, never replaced or fetched.
        p=roots[1]/'inputs/adapter.pt';p.write_bytes(b'x'*len(payload['adapter.pt']))
        with self.assertRaises(ValueError):r.recheck()
        self.assertEqual(r.captured['adapter.pt'],payload['adapter.pt'])
        with self.assertRaises(ValueError):r.capture('control/inputs/adapter.pt',c.sha(payload['adapter.pt']))
        with self.assertRaises(ValueError):r.capture('control/inputs/adapter.pt','0'*64)

    def test_crossjoins_conflicts_membership_and_aliases_fail(self):
        cases=('after-pin','after-map','outer-inner','second-copy','missing','extra','symlink','same-root','protected-overlap')
        for case in cases:
            with self.subTest(case=case):
                top,roots,payload,k=self.fixture()
                if case=='after-pin':k['after_sha256']='0'*64
                elif case=='after-map':
                    t=c.decode(k['after_raw']);a=c.decode(t['stdout'])
                    a['files']['control/inputs/toy.py']['sha256']='0'*64
                    t['stdout']=c.canonical(a).decode();k['after_raw']=c.canonical(t);k['after_sha256']=c.sha(k['after_raw'])
                elif case=='outer-inner':
                    outer=c.decode((roots[0]/'packet.json').read_bytes());outer['new_input_files']['inputs/toy.py']='0'*64
                    raw=c.canonical(outer)
                    for p in roots:(p/'packet.json').write_bytes(raw)
                    t=c.decode(k['after_raw']);a=c.decode(t['stdout'])
                    a['files']['control/packet.json']={'bytes':len(raw),'sha256':c.sha(raw)}
                    t['stdout']=c.canonical(a).decode();k.update(outer_sha256=c.sha(raw),after_raw=c.canonical(t),after_sha256=c.sha(c.canonical(t)))
                elif case=='second-copy':(roots[1]/'packet.json').write_bytes(b'bad original outer')
                elif case=='missing':(roots[1]/'inputs/toy.py').rename(roots[1]/'missing-preserved.py')
                elif case=='extra':(roots[1]/'inputs/extra.json').write_bytes(b'{}')
                elif case=='symlink':
                    p=roots[1]/'inputs/toy.py';p.rename(roots[1]/'original-toy-preserved.py');p.symlink_to(roots[0]/'inputs/toy.py')
                elif case=='same-root':k['roots'][1]=k['roots'][0]
                elif case=='protected-overlap':k['protected_roots']=[top]
                with self.assertRaises((ValueError,FileNotFoundError)):c.PacketReplicas(**k)

    def test_builder_roundtrip_binding_keeps_storage_gate_and_recheck(self):
        top,roots,payload,k=self.fixture();events=[]
        class Evidence:
            def run(self,operation,action,details=None):
                events.append(operation)
                return action()
        gate=types.SimpleNamespace(capsules=tuple(k['protected_roots'][:2]),staging=k['protected_roots'][2],
                                  receipt_sha256='a'*64,helper_sha256='b'*64,ancestry=[])
        supplied={'storage-after.json':k['after_raw'],'storage-config.json':b'{}\n'}
        def ctx():return {'pins':{'gpu-launch-packet':k['outer_sha256'],'gpu-packet':k['inner_sha256'],
                                  'gpu-inventory-after':k['after_sha256']},'compact_inputs':gate,
                          'storage_supplied':supplied,'storage_operation_evidence':Evidence()}
        adapter=types.SimpleNamespace(PacketReplicas=c.PacketReplicas,stored_config=lambda local,pins:supplied,
                                      recheck_capsules=lambda gate:events.append('full-archive-capsules-recheck'))
        def configure(context,*args):
            events.append('original-full-storage-binding');return gate
        source=ast.parse((HERE/'implementation/build_packet.py').read_text())
        names={'configure_packet_replicas','recheck_storage','storage_packet_binding','configure_packet_storage'}
        selected=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        env={'compact_adapter':lambda:adapter,'GPU_ROOT':k['control_root'],'COMPACT_ADAPTER_SHA':'c'*64,
             'sha':c.sha,'need':c.need,'configure_storage':configure}
        exec(compile(selected,'new-builder-metadata-seam','exec'),env)
        original=ctx();env['configure_packet_replicas'](original,k['roots'])
        binding=env['storage_packet_binding'](original)
        packet={'gpu_terminal_custody':str(gate.capsules[0]),'gpu_terminal_mirror':str(gate.capsules[1]),'storage_binding':binding}
        fresh=ctx();env['configure_packet_storage'](fresh,top,{},packet);env['recheck_storage'](fresh)
        self.assertIn('original-full-storage-binding',events)
        self.assertIn('full-archive-capsules-recheck',events)
        self.assertIn('packet-replica-recheck',events)
        self.assertEqual(fresh['packet_replicas'].capture('control/inputs/toy.py',c.sha(payload['toy.py'])),payload['toy.py'])
        packet['storage_binding']['input_replica_binding']['after_sha256']='d'*64
        with self.assertRaises(ValueError):env['configure_packet_storage'](ctx(),top,{},packet)


if __name__=='__main__':unittest.main(verbosity=2)
