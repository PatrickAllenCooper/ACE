"""Only new artificial named-outer boundary checks; no main/module imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types
import unittest

HERE=Path(__file__).resolve().parent
IMPL=HERE/'implementation'
FIXTURES=Path(os.environ['ACE_OUTER_READER_FIXTURES']).resolve()
ACTUAL_BYTES=42123690

def extracted(path,names):
    body=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in body}==set(names)
    env=dict(Path=Path,hashlib=hashlib,os=os,re=re,stat=stat)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path)+'[pure reader definitions]','exec'),env)
    return env

class OuterReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins=json.loads((FIXTURES/'inventory.json').read_bytes())['files']

    def check_consumer(self,name):
        path=IMPL/name
        if name=='pull_terminal.py':
            env=extracted(path,{'need','sha','pin','fingerprint','read_metadata'})
            reader=env['read_metadata']
            spelling="LAUNCH / 'packet.json'"
        else:
            helper=extracted(IMPL/'build_packet.py',{'need','sha','pin','read'})
            reader=helper['read'];env={'b':types.SimpleNamespace(read=reader)}
            spelling="local / 'packet.json'"
        main=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and n.args and ast.unparse(n.args[0])==spelling
               and ((isinstance(n.func,ast.Attribute) and n.func.attr=='read') or (isinstance(n.func,ast.Name) and n.func.id=='read_metadata'))]
        self.assertEqual(len(calls),1)
        call=calls[0]
        self.assertEqual([k.arg for k in call.keywords],['maximum'])
        expr=compile(ast.Expression(call),str(path)+'[named outer read only]','eval')
        def invoke(directory,pin):
            return eval(expr,{**env,'local':directory,'LAUNCH':directory,'a':types.SimpleNamespace(packet_sha256=pin)})
        positive=FIXTURES/'actual-sized'
        value=invoke(positive,self.pins['actual-sized/packet.json']['sha256'])
        self.assertEqual(len(value),ACTUAL_BYTES)
        self.assertEqual(json.loads(value),{'synthetic':True,'scope':'no scientific data'})
        self.assertEqual(hashlib.sha256(value).hexdigest(),self.pins['actual-sized/packet.json']['sha256'])
        del value
        with self.assertRaisesRegex(ValueError,'bounded'):
            invoke(FIXTURES/'oversized',self.pins['oversized/packet.json']['sha256'])
        ordinary=FIXTURES/'ordinary/metadata.json'
        with self.assertRaisesRegex(ValueError,'bounded'):
            reader(ordinary,self.pins['ordinary/metadata.json']['sha256'])
        with self.assertRaises(ValueError):invoke(positive,'0'*64)

    def test_launcher_actual_size_and_bounds(self):self.check_consumer('launch_once.py')
    def test_monitor_actual_size_and_bounds(self):self.check_consumer('monitor_once.py')
    def test_collector_actual_size_and_bounds(self):self.check_consumer('pull_terminal.py')

    def test_successor_routing_keeps_scientific_construction_unchanged(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        def assigned(path,name):
            tree=ast.parse(path.read_bytes())
            n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
            return eval(compile(ast.Expression(n.value),'captured path expression','eval'),
                        {'REPO':Path('/Users/pat/code/ACE'),'BASE':Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')})
        predecessor=HERE/'predecessor'
        builder=IMPL/'build_packet.py';collector=IMPL/'pull_terminal.py'
        local=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/launch02')
        external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-finalization-20261108-02')
        self.assertEqual(assigned(builder,'LOCAL'),local);self.assertEqual(assigned(collector,'LAUNCH'),local)
        self.assertEqual(assigned(builder,'EXT'),external);self.assertEqual(assigned(collector,'LAUNCH_EXT'),external)
        self.assertEqual(assigned(builder,'ROOT'),assigned(predecessor/'build_packet.py','ROOT'))
        self.assertEqual(assigned(collector,'ROOT'),assigned(builder,'ROOT'))
        for name in ('read','make_packet','authenticate','clock_admission','job_script','budget'):
            self.assertEqual(functions(builder)[name],functions(predecessor/'build_packet.py')[name],name)
        for name in ('read_metadata','fingerprint','resources','accounting_extension','validate_remote_intent'):
            self.assertEqual(functions(collector)[name],functions(predecessor/'pull_terminal.py')[name],name)
        self.assertEqual((IMPL/'finalize_llm_intervention_scientific_campaign_cpu.py').read_bytes(),
                         (predecessor/'finalize_llm_intervention_scientific_campaign_cpu.py').read_bytes())
        # The passed size checks used these exact call ASTs/readers; only routing literals changed.
        for name in ('launch_once.py','monitor_once.py'):
            self.assertEqual((IMPL/name).read_bytes(),(HERE/'readers-only01'/name).read_bytes())
        self.assertEqual(functions(collector)['main'],functions(HERE/'readers-only01/pull_terminal.py')['main'])

if __name__=='__main__':unittest.main(verbosity=2)
