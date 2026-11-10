"""Two new fabricated CLI-boundary tests; no real study trees or producer main."""
from pathlib import Path
import importlib.util,json,tempfile,time,unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe',HERE/'probe_finalizer_inventory.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
REFERENCE=HERE/'reference-finalizer.py'

def fixture(root):
    cpu=root/'cpu';gpu=root/'gpu';campaign=cpu/'worker/artifacts/campaign'
    campaign.mkdir(parents=True);gpu.mkdir()
    for path,value in ((cpu/'payload.bin',b'cpu opaque'),(gpu/'payload.bin',b'gpu opaque'),
                       (campaign/'raw.bin',b'not JSON; no scientific payload')):
        path.write_bytes(value)
    pins=lambda tree:{str(x.relative_to(tree)):p.sha(x.read_bytes()) for x in tree.rglob('*') if x.is_file()}
    h='1'*64
    c=dict(schema='ace-scientific-campaign-cpu-acceptance-v1',status='development_cpu_phase_accepted',
        scientific_qualified=False,cpu_output_excluded_directory='worker/artifacts/campaign',
        cpu_packet_sha256=h,bundle_sha256=h,cpu_output=str(cpu),campaign_root=str(campaign),
        cpu_output_files={'payload.bin':p.sha((cpu/'payload.bin').read_bytes())},campaign_files=pins(campaign))
    cp=root/'cpu-acceptance.json';cp.write_bytes(p.raw(c));ch=p.sha(cp.read_bytes())
    g=dict(schema='ace-scientific-campaign-gpu-acceptance-v1',status='development_gpu_phase_accepted',
        scope='development',scientific_qualified=False,accepted_checks=dict(scheduler=True,runtime=True,cleanup=True,custody=True),
        cpu_acceptance_sha256=ch,cpu_packet_sha256=h,bundle_sha256=h,job_id='2',gpu_packet_sha256=h,
        gpu_output=str(gpu),campaign_root=str(campaign),gpu_output_files=pins(gpu),campaign_files=pins(campaign))
    gp=root/'gpu-acceptance.json';gp.write_bytes(p.raw(g));gh=p.sha(gp.read_bytes())
    s=dict(schema='ace-scientific-campaign-gpu-recovery-acceptance-v1',status='accepted',scope='development',
        required_findings=0,acceptance=True,historical_postcustody_verified=True,originals_preserved=True,
        scientific_qualified=False,gpu_acceptance_sha256=gh,cpu_acceptance_sha256=ch,gpu_packet_sha256=h,
        job_id='2',output_root=str(gpu),cpu_campaign_root=str(campaign))
    sp=root/'recovery.json';sp.write_bytes(p.raw(s))
    inputs={'reference_source':REFERENCE,'cpu_acceptance':cp,'gpu_acceptance':gp,'gpu_recovery_acceptance':sp}
    q=dict(schema=p.SCHEMA,probe_sha256=p.sha((HERE/'probe_finalizer_inventory.py').read_bytes()),soft_seconds=10,hard_seconds=15,
           inputs={k:dict(path=str(v),sha256=p.sha(v.read_bytes())) for k,v in inputs.items()})
    return q,gpu,sp

def invoke(root,q,name):
    request=root/(name+'-request.json');request.write_bytes(p.raw(q));out=root/name
    argv=['probe','--request',str(request),'--request-sha256',p.sha(request.read_bytes()),'--output',str(out)]
    with mock.patch('sys.argv',argv):rc=p.main()
    return rc,out

class Probe(unittest.TestCase):
    def test_full_membership_routing_acceptance_and_output_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();q,gpu,recovery=fixture(root)
            rc,out=invoke(root,q,'complete');self.assertEqual(rc,0)
            result=json.loads((out/'result.json').read_bytes());self.assertTrue(result['all_ten_file_rosters_verified'])
            self.assertEqual([r['role'] for r in result['phases']],['cpu','gpu','campaign','cpu','gpu','campaign','campaign','campaign','cpu','gpu'])
            self.assertTrue(all(r['files_read']==1 and r['bytes_read']>0 for r in result['phases']))
            self.assertFalse(result['acceptance']);self.assertFalse(result['runtime_qualified'])
            (gpu/'unexpected.bin').write_bytes(b'extra')
            rc,bad=invoke(root,q,'membership');self.assertEqual(rc,1)
            result=json.loads((bad/'result.json').read_bytes());self.assertEqual(len(result['phases']),2)
            self.assertFalse(result['phases'][1]['complete_membership_and_hashes_matched'])
            self.assertTrue((bad/'phase-01-mismatch-inventory.json').exists())
            self.assertFalse(result['metadata_recheck_performed'])
            # No accepted future status may be inferred; reject before claiming output.
            s=json.loads(recovery.read_bytes());s['historical_postcustody_verified']=False;recovery.write_bytes(p.raw(s))
            q['inputs']['gpu_recovery_acceptance']['sha256']=p.sha(recovery.read_bytes())
            with self.assertRaises(ValueError):invoke(root,q,'unaccepted')
            self.assertFalse((root/'unaccepted').exists())
            s['historical_postcustody_verified']=True;recovery.write_bytes(p.raw(s));q['inputs']['gpu_recovery_acceptance']['sha256']=p.sha(recovery.read_bytes())
            # Existing destinations must never receive failure files.
            request=root/'existing-request.json';request.write_bytes(p.raw(q))
            with mock.patch('sys.argv',['probe','--request',str(request),'--request-sha256',p.sha(request.read_bytes()),'--output',str(gpu)]),self.assertRaises(ValueError):p.main()
            self.assertFalse((gpu/'failure.json').exists())

    def test_real_alarm_retains_inflight_phase_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();q,_,_=fixture(root);q.update(soft_seconds=1,hard_seconds=6)
            factory=p.captured_inventory;reads=[]
            def delayed(source,pin):
                ns=factory(source,pin);read=ns['read']
                def slow(path,*args,**kwargs):
                    reads.append(str(path))
                    if len(reads)==2:time.sleep(2)
                    return read(path,*args,**kwargs)
                ns['read']=slow;return ns
            with mock.patch.object(p,'captured_inventory',side_effect=delayed):rc,out=invoke(root,q,'timeout')
            self.assertEqual(rc,1);self.assertEqual(len(reads),2)
            result=json.loads((out/'result.json').read_bytes());self.assertEqual(len(result['phases']),2)
            failed=result['phases'][-1]
            self.assertEqual(failed['failure']['type'],'TimeoutError');self.assertEqual(failed['files_read'],0)
            self.assertIsNotNone(failed['inflight_member']);self.assertFalse(result['all_ten_file_rosters_verified'])
            self.assertIn('phase_start',(out/'events.jsonl').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
