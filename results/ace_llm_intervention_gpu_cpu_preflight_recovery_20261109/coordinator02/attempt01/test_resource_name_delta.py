"""One new artificial R1 fixture; no remote, allocation, producer or scientific calls."""
import copy,importlib.util,json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu02_collector',ROOT/'authoring/pull_terminal.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class ResourceNameDelta(unittest.TestCase):
    def test_current_gpu02_accepts_and_predecessor_name_rejects(self):
        job='99999998';allocated='billing=58,cpu=4,gres/gpu:a100_3g.20gb=1,mem=15G,node=1'
        requested='billing=58,cpu=4,gres/gpu:a100_3g.20gb=1,mem=15360M,node=1'
        for name,expected in [('ace_llm_scientific_campaign_gpu02',True),('ace_llm_scientific_campaign_gpu01',False)]:
            with self.subTest(name=name):
                folder=ROOT/'fixtures'/name;folder.mkdir(parents=True,exist_ok=False)
                allocation={'fields':{'JobId':job,'JobName':name,'Requeue':'0','Restarts':'0','Account':'ucb736_asc2',
                                     'Partition':'aa100','QOS':'gpu-testing','TimeLimit':'00:48:00',
                                     'ReqTRES':requested,'AllocTRES':allocated}}
                payload=c.raw(allocation);(folder/'allocation.json').write_bytes(payload)
                obs={'parent_accounting':[job,name,'COMPLETED','0:0','4','100','ucb736_asc2',allocated,''],
                     'results':{'details':{'returncode':0,'stdout':job+'|aa100|gpu-testing|48|'+requested+'|paco0228|\n'},
                                'job':{'returncode':1,'stdout':'not needed; saved allocation is independently pinned'}}}
                files={'launch/gpu/allocation.json':{'bytes':len(payload),'sha256':c.sha(payload)}}
                (folder/'observation.json').write_bytes(c.raw(obs));(folder/'files.json').write_bytes(c.raw(files))
                result=c.resources(obs,job,folder,files);(folder/'result.json').write_bytes(c.raw(result))
                self.assertEqual(result['resource_contract_matched'],expected)
                self.assertTrue(result['no_requeue_verified']);self.assertEqual(result['allocation_source_sha256'],c.sha(payload))
                self.assertEqual(result['requested_tres']['mem'],'15360M');self.assertEqual(result['allocated_tres']['mem'],'15G')
                self.assertEqual((result['allocated_cpus'],result['gpu_count'],result['allocated_cpu_seconds'],result['gpu_slice_seconds']),(4,1,400,100))
                self.assertEqual(result['errors'],[] if expected else ['actual/requested GPU identity/resource/no-requeue/wall contract differs or unavailable'])

if __name__=='__main__':unittest.main(verbosity=2)
