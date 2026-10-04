"""Deterministic safety fixtures; no emulator acquisition, fitting, or scoring."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import runner_delivery_confirmation as r

class Validation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_registered_protocol(self):
        reg,cfg=r.validate_protocol()
        self.assertEqual(len(reg['seeds']),12);self.assertEqual(cfg['proposer'],'random')
    def test_journal_ceiling_and_no_replacement(self):
        p=self.root/'calls';j=r.Journal(p,2);j.reserve();j.reserve()
        with self.assertRaises(RuntimeError):j.reserve()
        self.assertEqual(r.journal_count(p),2)
        with self.assertRaises(ValueError):r.Journal(p,2)
        p.write_text('{"attempt":2}\n')
        with self.assertRaises(ValueError):r.journal_count(p)
    def test_original_development_custody(self):
        bundle=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/host-mirror/results/slot0/ace_results/5f89033d')
        meta=r.read(bundle/'meta.json')
        rows=[json.loads(line) for line in (bundle/'observations.ndjson').read_text().splitlines()]
        ace=[row for row in rows if row['method']=='ace']
        self.assertEqual(len(ace),4803)
        self.assertEqual(meta['query_counts']['ace']['total'],len(ace))
        digest=hashlib.sha256(json.dumps(ace,sort_keys=True,allow_nan=False).encode()).hexdigest()
        receipt=r.read(Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-development-7001-20261003/init-0/receipt.json'))
        self.assertEqual(digest,receipt['source_rows_sha256'])
    def test_inventory_symlink(self):
        (self.root/'data').write_text('fixture');(self.root/'alias').symlink_to(self.root/'data')
        with self.assertRaises(ValueError):r.inventory(self.root)
    def test_approval_and_timing(self):
        p=self.root/'approval';r.write(p,{'approved':False})
        with self.assertRaises(ValueError):r.approval_check(p)
        reg,_=r.validate_protocol()
        self.assertTrue(r.timing_gate(400,reg));self.assertFalse(r.timing_gate(500,reg))
        with self.assertRaises(ValueError):r.launch(self.root/'never-created',p,sys.executable)
        self.assertFalse((self.root/'never-created').exists())
    def test_supervisor_success_failure_time_memory(self):
        for code,limit,sampler,expected in [('pass',2,lambda _:1,'complete'),
                 ('raise SystemExit(3)',2,lambda _:1,'failed'),
                 ('import time;time.sleep(5)',.1,lambda _:1,'time_limit'),
                 ('import time;time.sleep(5)',2,lambda _:200,'memory_limit')]:
            result=r.supervise([sys.executable,'-c',code],time.monotonic()+limit,100,self.root/'log',sampler=sampler,interval=.01)
            self.assertEqual(result['status'],expected)
    def test_watchdog_and_unrelated_process(self):
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])
        try:
            def slow(_):time.sleep(.25);return 1
            start=time.monotonic()
            result=r.supervise([sys.executable,'-c','import time;time.sleep(5)'],start+.05,100,self.root/'log',sampler=slow,interval=.01)
            self.assertEqual(result['status'],'time_limit');self.assertLess(time.monotonic()-start,1)
            self.assertIsNone(other.poll())
        finally:other.terminate();other.wait()
    def test_audit_hook_blocks_grid_and_network(self):
        for action in ["open('fixture.npz','wb')", "__import__('socket').socket().connect(('127.0.0.1',1))"]:
            code=f'import sys;sys.path.insert(0,{str(Path(r.__file__).parent)!r});import runner_delivery_confirmation as r;sys.addaudithook(r.no_grid_access);{action}'
            run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,cwd=self.root)
            self.assertNotEqual(run.returncode,0);self.assertIn('PermissionError',run.stderr)
        self.assertFalse((self.root/'fixture.npz').exists())
    def fixture(self):
        reg={'iterations':2,'seeds':[17],'delivery':{'inits':[0,1,2],'epochs':3},'resource_proposal':{'call_cap_per_case':2,'total_call_cap':2}}
        (self.root/'protocol').mkdir();r.write(self.root/'protocol/acquisition_config.json',{'proposer':'random'})
        case=self.root/'cases/17';(case/'online/mlps').mkdir(parents=True)
        rows=[{'method':'ace','role':'seed','query_index':i,'interventions':{} if i==0 else {'y':1}} for i in range(2)]
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
        r.write(case/'online/meta.json',{'total_steps':2,'has_baseline':False,'query_counts':{'ace':{'seed':2,'total':2}},'causal_dag':{'x':[],'y':['x']}})
        (case/'online/observations.ndjson').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (case/'online/mlp.pt').write_bytes(b'fixture');(case/'online/mlps/y.pt').write_bytes(b'fixture')
        j=r.Journal(case/'calls.jsonl',2);j.reserve();j.reserve()
        r.write(case/'complete.json',{'seed':17,'inits':[0,1,2],'calls':2,'rows_sha256':digest,'config':{'proposer':'random','seed':17}})
        for init in range(3):
            folder=case/f'init-{init}';folder.mkdir();(folder/'models.pt').write_bytes(b'fixture')
            r.write(folder/'receipt.json',{'source_rows_sha256':digest,'charged_rows':2,'custody_metadata_sha256':r.sha(case/'online/meta.json'),'eligible_rows':{'flat':1,'y':1},'epochs':3,'seed':init,'evaluation':None})
        return reg,case
    def test_seal_and_byte_mutation(self):
        reg,case=self.fixture();r.write(self.root/'sealed.json',r.seal_cases(self.root,reg));r.verify_seal(self.root,reg)
        (case/'online/mlp.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):r.verify_seal(self.root,reg)
    def test_semantic_parity_mutations(self):
        reg,case=self.fixture()
        p=case/'init-0/receipt.json';original=r.read(p)
        for field,value in [('charged_rows',1),('epochs',4),('seed',9),('eligible_rows',{'flat':2,'y':2}),('evaluation',{})]:
            r.write(p,{**original,field:value})
            with self.assertRaises(ValueError):r.seal_cases(self.root,reg)
        r.write(p,original)
        (self.root/'cases/18').mkdir()
        with self.assertRaises(ValueError):r.seal_cases(self.root,reg)
    def test_missing_seal_precedes_grid_access(self):
        r.write(self.root/'campaign.json',{'source_inventory':{}})
        with patch.object(r,'validate_protocol',return_value=({},{})),patch.object(r,'approval_check'),patch.object(sys,'addaudithook'),patch.object(r,'sha',side_effect=AssertionError('grid access')):
            with self.assertRaises(FileNotFoundError):r.evaluate_worker(self.root)

if __name__=='__main__':unittest.main()
