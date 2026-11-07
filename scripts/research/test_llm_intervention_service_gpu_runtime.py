"""Artificial current GPU integration contracts. No runtime, model or GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import qualify_llm_intervention_model_service as gpu
import qualify_llm_intervention_service_cpu as cpu
import supervise_llm_intervention_checkpoint_gpu as helper
import test_llm_intervention_service_cpu as fixtures
import llm_intervention_model_service as service
import llm_intervention_service_runtime as runtime
import qualify_llm_intervention_runtime_metadata as diagnostic
import llm_intervention_gpu_mappings as mappings
import test_llm_intervention_gpu_mappings as map_fixtures


class GpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ace-gpu-v2-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {n:'a'*64 for n in set(cpu.INPUTS)|{
            'qualify_llm_intervention_model_service.py','supervise_llm_intervention_checkpoint_gpu.py',
            'cpu-preflight.json','cpu-acceptance.json'}}
        self.packet = {'schema':gpu.VERSION, 'files':self.files,
            'services':['frozen.json','adapter.json'], 'resources':dict(gpu.RESOURCES)}

    def test_exact_new_packet_roster_and_resource_types(self):
        self.assertEqual(len(self.files),30)
        gpu.packet_shape(self.packet,cpu.INPUTS)
        for change in (lambda p:p.update(schema='ace-persistent-model-integration-v1'),
            lambda p:p['files'].pop('cpu-acceptance.json'),
            lambda p:p['files'].update({'../surprise.py':'a'*64}),
            lambda p:p['files'].update({'probe.py':'bad'}),
            lambda p:p['resources'].update(gpus=True),
            lambda p:p['resources'].update(account='ucb736_asc1')):
            p=copy.deepcopy(self.packet);change(p)
            with self.assertRaises(ValueError):gpu.packet_shape(p,cpu.INPUTS)

    def test_cpu_acceptance_cannot_cover_different_service_inputs(self):
        preflight={'files':{n:'a'*64 for n in cpu.INPUTS}}
        accepted={'status':'cpu_service_preflight_accepted','job_id':'35350736',
            'preflight_sha256':'a'*64,'cpu_preflight_qualified':True,
            'model_service_qualified':False,'scientific_qualified':False}
        gpu.check_preflight(accepted,self.packet,preflight,cpu)
        for key,value in [('job_id','35348366'),('status','pending'),('preflight_sha256','b'*64),
                          ('cpu_preflight_qualified',False),('model_service_qualified',True)]:
            with self.assertRaises(ValueError):gpu.check_preflight({**accepted,key:value},self.packet,preflight,cpu)
        other=copy.deepcopy(preflight);other['files']['llm_intervention_model_service.py']='b'*64
        with self.assertRaises(ValueError):gpu.check_preflight(accepted,self.packet,other,cpu)

    def test_isolated_command_preserves_scheduler_gpu_visibility(self):
        c=gpu.service_command(Path('/node/runtime'),Path('/packet'),'frozen.json',self.files,
            Path('/out/session'),Path('/out/stage.json'),'b'*64,'pinned bootstrap')
        self.assertEqual(c[:5],('/node/runtime/bin/python','-I','-B','-c','pinned bootstrap'))
        self.assertNotIn('--verify-only',c)
        self.assertEqual(c[c.index('--runtime-stage-sha256')+1],'b'*64)
        original={'CUDA_VISIBLE_DEVICES':'MIG-fixture','PYTHONPATH':'bad','LD_PRELOAD':'bad','PYTHONHOME':'bad'}
        env=gpu.child_environment(Path('/node/runtime'),original)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'MIG-fixture')
        self.assertEqual(env['LD_LIBRARY_PATH'],'/node/runtime/lib')
        self.assertNotIn('PYTHONPATH',env)
        self.assertIn('PYTHONPATH',original)
        for v in ({},{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):gpu.child_environment(Path('/node/runtime'),v)

    def test_absolute_original_deadline_reserves_entire_session(self):
        gpu.require_session_time(100.,409.999)
        for now in (410.,500.,800.):
            with self.assertRaises(TimeoutError):gpu.require_session_time(100.,now)
        self.assertEqual(gpu.FIRST_GPU_SECONDS,240.)

    def test_renewed_gpu_allocation_and_no_requeue(self):
        base={'JobId':'123','UserId':'paco0228(348288)','Account':'ucb736_asc2','Partition':'aa100',
            'QOS':'gpu-testing','JobState':'RUNNING','NumNodes':'1','NumTasks':'1','TimeLimit':'00:10:00',
            'NumCPUs':'4','CPUs/Task':'4','Requeue':'0','Restarts':'0',
            'AllocTRES':'cpu=4,mem=12G,node=1,gres/gpu:a100_3g.20gb=1',
            'ReqTRES':'cpu=4,mem=12G,node=1,gres/gpu:a100_3g.20gb=1'}
        def call(v):return gpu.authenticate_allocation(' '.join(k+'='+v for k,v in v.items()),'123',gpu.RESOURCES,helper)
        self.assertEqual(call(base)['allocated_resources']['cpu'],'4')
        self.assertEqual(call({**base,'NumCPUs':'8','CPUs/Task':'8','AllocTRES':base['AllocTRES'].replace('cpu=4','cpu=8')})['allocated_resources']['cpu'],'8')
        for key,value in [('Account','ucb736_asc1'),('Requeue','1'),('Restarts','1'),
            ('NumCPUs','9'),('ReqTRES',base['ReqTRES'].replace('cpu=4','cpu=2')),
            ('AllocTRES',base['AllocTRES'].replace('20gb=1','20gb=2')),
            ('AllocTRES',base['AllocTRES'].replace('mem=12G','mem=24G'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):call({**base,key:value})

    def test_bounded_pin_and_link_rejection(self):
        p=self.root/'input';p.write_bytes(b'abc')
        self.assertEqual(gpu.bounded_input(p,gpu.sha(b'abc'),3),b'abc')
        for pin,maximum in [(gpu.sha(b'abc'),2),('0'*64,3),('bad',3)]:
            with self.assertRaises(ValueError):gpu.bounded_input(p,pin,maximum)
        link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(OSError):gpu.bounded_input(link,gpu.sha(b'abc'),3)

    def artificial_runtime(self):
        # Reuse only a fabricated record constructor, never an earlier test or model.
        f=fixtures.CpuPreflightTests();f.setUp();self.addCleanup(f.doCleanups)
        records=f.receipts()
        observation=records.pop('runtime-after-cpu-tokenization.json')
        records['runtime-stage-binding.json']['gpu_mapping_enabled']=True
        sample,inventory=map_fixtures.fixture()
        for phase in mappings.PHASES:
            raw=copy.deepcopy(observation)
            raw.update(process_maps=sample['process_maps'],mixed_native_paths=sample['mixed_native_paths'])
            _,proof=mappings.checked_view(raw,inventory,phase,prefix=f.stage['staged_prefix'],original_runtime=diagnostic.RUNTIME_SOURCE)
            records['runtime-'+phase+'.json']=raw
            records['gpu-mapping-inventory-'+phase+'.json']=inventory
            records['gpu-mapping-validation-'+phase+'.json']=proof
        records['terminal.json']={'runtime_stage_sha256':'c'*64}
        return f,records

    def check_runtime(self,f,records,index):
        directory=self.root/str(index);worker=directory/'worker';worker.mkdir(parents=True)
        for n,v in records.items():(worker/n).write_text(json.dumps(v))
        return gpu.validate_runtime(directory,f.freeze,f.files,f.stage,'c'*64,cpu,service,diagnostic,runtime,f.provenance)

    def test_each_gpu_observation_checked_against_stage_and_versions(self):
        f,records=self.artificial_runtime()
        self.assertTrue(self.check_runtime(f,records,'good')['actual_imports_qualified'])
        for i,phase in enumerate(('after-service-imports','after-model-load','after-requests')):
            bad=copy.deepcopy(records);bad['runtime-'+phase+'.json']['packages']['numpy']['version']='2.2.6'
            with self.subTest(phase=phase),self.assertRaises(ValueError):self.check_runtime(f,bad,i)
        bad=copy.deepcopy(records);bad.pop('runtime-after-model-load.json')
        with self.assertRaises(FileNotFoundError):self.check_runtime(f,bad,'missing')

    def test_model_and_project_bindings_remain_required_after_generation(self):
        f,records=self.artificial_runtime()
        changes=[lambda r:r['authenticated.json']['models'].pop('model.safetensors'),
            lambda r:r['terminal.json'].update(runtime_stage_sha256='d'*64),
            lambda r:r['runtime-after-requests.json']['captured_project_modules'].pop('llm_intervention_learning'),
            lambda r:r['runtime-after-model-load.json'].update(mixed_native_paths=['/untrusted/library.so'])]
        for i,change in enumerate(changes):
            bad=copy.deepcopy(records);change(bad)
            with self.assertRaises(ValueError):self.check_runtime(f,bad,i)


if __name__=='__main__':unittest.main(verbosity=2)
