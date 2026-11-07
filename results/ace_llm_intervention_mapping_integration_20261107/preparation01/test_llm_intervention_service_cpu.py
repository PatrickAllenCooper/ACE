"""Pure CPU-preflight envelope and saved-receipt checks, no model execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import llm_intervention_model_service as service
import llm_intervention_service_runtime as runtime
import qualify_llm_intervention_runtime_metadata as diagnostic
import qualify_llm_intervention_service_cpu as cpu


class CpuPreflightTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix="ace-service-cpu-check-")
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        needed=set(service.SOURCE_NAMES)|{"qualify_llm_intervention_service_cpu.py",
            "llm_intervention_runtime_archive.py","probe.py","frozen.json","adapter.json",
            "trained_adapter.pt","gpu09_terminal.json"}|cpu.FIXTURES
        self.files={name:"a"*64 for name in needed}
        self.files.update({"qualify_llm_intervention_runtime_metadata.py":cpu.DIAGNOSTIC_SHA,
            "llm_intervention_runtime_archive.py":diagnostic.HELPER_SHA,"probe.py":diagnostic.PROBE_SHA,
            "llm_intervention_gpu_mappings.py":runtime.GPU_MAPPINGS_SHA})
        self.freeze={"schema":service.VERSION,"scope":"artificial","model_directory":"/model",
            "model_files":{name:"b"*64 for name in service.MODEL_NAMES},
            "sources":{name:self.files[name] for name in service.SOURCE_NAMES},
            "runtime":copy.deepcopy(service.RUNTIME),"python_version":[3,10,19],"recipe":service.RECIPE,
            "treatment":"frozen","adapter":None,"limits":{"startup_seconds":180,
                "request_seconds":30,"idle_seconds":30,"total_seconds":240,"max_requests":4}}
        adapted=copy.deepcopy(self.freeze)
        adapted.update(treatment="artificial_adapter",adapter={"file":"trained_adapter.pt","sha256":"a"*64,
            "training_receipt":"gpu09_terminal.json","training_receipt_sha256":"a"*64,"training_scope":"artificial"})
        self.data={name:b"fabricated input" for name in needed}
        self.data.update({"frozen.json":json.dumps(self.freeze).encode(),"adapter.json":json.dumps(adapted).encode(),
            "gpu09_terminal.json":json.dumps({"status":"complete","adapter_sha256":"a"*64}).encode()})
        self.packet={"schema":cpu.VERSION,"files":self.files,"services":cpu.SERVICES,"resources":cpu.RESOURCES,
            "archive":{"path":"/original/archive","sha256":diagnostic.ARCHIVE_SHA},
            "manifest":{"path":"/original/manifest","sha256":diagnostic.MANIFEST_SHA},
            "runtime_source":diagnostic.RUNTIME_SOURCE}

    def test_renewed_cpu_allocation_and_drift(self):
        fields={"JobId":"123","JobState":"RUNNING","Account":"ucb736_asc2","Partition":"acpu",
            "QOS":"cpu-normal","NumCPUs":"2","NumTasks":"1","NumNodes":"1","TimeLimit":"00:10:00",
            "Requeue":"0","Restarts":"0","AllocTRES":"cpu=2,mem=6G,node=1,billing=1"}
        raw=lambda f:" ".join(k+"="+v for k,v in f.items())
        cpu.allocation_fields(raw(fields),"123")
        changes={"JobId":"124","JobState":"PENDING","Account":"ucb736_asc1","Partition":"aa100",
            "QOS":"gpu-testing","NumCPUs":"1","NumTasks":"2","NumNodes":"2","TimeLimit":"00:20:00",
            "Requeue":"1","Restarts":"1","AllocTRES":"cpu=1,mem=4G,node=1,gres/gpu=1"}
        for k,v in changes.items():
            with self.subTest(k=k),self.assertRaises(ValueError):
                cpu.allocation_fields(raw({**fields,k:v}),"123")
        with self.assertRaises(ValueError):
            cpu.allocation_fields(raw({**fields,"AllocTRES":"cpu=1,mem=3G,node=1"}),"123")

    def test_exact_input_scope_helpers_and_adapter_lineage(self):
        self.assertEqual(len(self.files),26)
        self.assertEqual(len(cpu.authenticate(self.root,self.packet,self.data,service,diagnostic)),2)
        mutations=[lambda p:p.update(services=list(reversed(cpu.SERVICES))),
            lambda p:p["resources"].update(gpus=1),lambda p:p["archive"].update(sha256="0"*64),
            lambda p:p["files"].update({"probe.py":"0"*64}),lambda p:p.update(runtime_source="/different")]
        for change in mutations:
            p=copy.deepcopy(self.packet);change(p)
            with self.assertRaises(ValueError):
                cpu.authenticate(self.root,p,self.data,service,diagnostic)
        bad=dict(self.data);bad.pop("stress.json")
        with self.assertRaises(ValueError):
            cpu.authenticate(self.root,self.packet,bad,service,diagnostic)
        bad=dict(self.data);bad["gpu09_terminal.json"]=b'{"status":"failed","adapter_sha256":"wrong"}'
        with self.assertRaises(ValueError):
            cpu.authenticate(self.root,self.packet,bad,service,diagnostic)

    def test_memory_limit_contract_rejects_old_or_expanded_allocation(self):
        self.assertEqual(cpu.RESOURCES['memory_mib'],6144)
        self.assertNotIn('memory_gib',cpu.RESOURCES)
        cpu.packet_shape(self.packet)
        old=copy.deepcopy(self.packet);old['schema']='ace-model-service-cpu-preflight-v1'
        with self.assertRaises(ValueError):cpu.packet_shape(old)
        old=copy.deepcopy(self.packet);old['resources'].pop('memory_mib');old['resources']['memory_gib']=4
        with self.assertRaises(ValueError):cpu.packet_shape(old)
        base='JobId=123 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:10:00 Requeue=0 Restarts=0 AllocTRES=cpu=2,mem=6G,node=1,billing=1'
        cpu.allocation_fields(base,'123')
        for bad in (base.replace('mem=6G','mem=4G'),
                    base.replace('NumCPUs=2','NumCPUs=1').replace('cpu=2,mem=6G','cpu=1,mem=6G'),
                    base.replace('mem=6G','mem=3G')):
            with self.assertRaises(ValueError):cpu.allocation_fields(bad,'123')
    def test_child_uses_staged_isolated_bootstrap_and_all_fixtures(self):
        command=cpu.child_command(Path('/stage/runtime'),self.root,"frozen.json",self.files,
            self.root/'session',self.root/'stage.json','c'*64,'captured bootstrap')
        self.assertEqual(command[:5],['/stage/runtime/bin/python','-I','-B','-c','captured bootstrap'])
        self.assertEqual(command[command.index('--runtime-stage-sha256')+1],'c'*64)
        self.assertIn('--verify-only',command)
        fixtures=[command[i+1] for i,x in enumerate(command) if x=='--fixture']
        self.assertEqual(fixtures,[name+'='+self.files[name] for name in sorted(cpu.FIXTURES)])

    def test_bounded_inputs_reject_size_pin_and_symlink_before_decode(self):
        path=self.root/'bounded';path.write_bytes(b'abc')
        self.assertEqual(cpu.bounded_input(path,cpu.sha(b'abc'),3),b'abc')
        with self.assertRaises(ValueError):cpu.bounded_input(path,cpu.sha(b'abc'),2)
        with self.assertRaises(ValueError):cpu.bounded_input(path,'0'*64,3)
        with self.assertRaises(ValueError):cpu.bounded_input(path,'bad',3)
        linked=self.root/'link';linked.symlink_to(path)
        with self.assertRaises(OSError):cpu.bounded_input(linked,cpu.sha(b'abc'),3)
        sparse=self.root/'large'
        with sparse.open('wb') as f:f.truncate(8*1024**3)
        with self.assertRaises(ValueError):cpu.bounded_input(sparse,'0'*64,128*1024)

    def test_packet_shape_rejects_bad_roster_before_entry_reads(self):
        cpu.packet_shape(self.packet)
        for mutate in (lambda p:p.update(files=[]),lambda p:p['files'].pop('stress.json'),
                       lambda p:p['files'].update({'../source.py':'a'*64}),
                       lambda p:p['files'].update({'probe.py':'unbounded'}),
                       lambda p:p['resources'].update(cpus=True)):
            bad=copy.deepcopy(self.packet);mutate(bad)
            with self.assertRaises(ValueError):cpu.packet_shape(bad)

    def receipts(self):
        prefix=self.root/'runtime'
        self.stage={'version':'ace-runtime-archive-v1','archive_sha256':service.RUNTIME['archive_sha256'],
            'manifest_sha256':service.RUNTIME['manifest_sha256'],'staged_prefix':str(prefix),
            'objects':49966,'file_bytes':8397575073,'model_loaded':False,'runtime_qualified':False,
            'cache_source_sha256':{n:'a'*64 for pair in diagnostic.CACHE_PAIRS for n in pair}}
        self.provenance={'sources':{},'expected_source':'artificial generated source',
            'expected_source_sha256':cpu.sha(b'artificial generated source')}
        temp=self.root/'service-tmp-123';generated=temp/'tmpfixture'
        baseline={'python_version':'3.10.19','executable':str(prefix/'bin/python'),'prefix':str(prefix),
            'base_prefix':str(prefix),'sys_path':[str(prefix/'lib/python3.10')],
            'packages':{n:{'loaded':False,'version':None} for n in ('numpy','scipy','torch','transformers')},
            'mixed_module_paths':[],'mixed_modules':{},'mixed_native_paths':[],
            'generated_provenance':{'present':False,'verified':False}}
        records={"terminal.json":{"status":"cpu_preflight_complete","freeze_sha256":"a"*64,
            "runtime_stage_sha256":"c"*64,"model_loaded":False,"adapter_deserialized":False,
            "explicit_cuda_api_called":False,"service_qualified":False,"scientific_execution_qualified":False,
            "new_training_updates":0,"environment_responses":0},
            "authenticated.json":{"freeze_sha256":"a"*64,"identity":service.identity(self.freeze),
                "runtime":service.RUNTIME,"models":{n:{"sha256":p,"bytes":3} for n,p in self.freeze["model_files"].items()}},
            "runtime-stage-binding.json":{"stage_sha256":"c"*64,"contract":service.RUNTIME,
                'stage':self.stage,'temporary_root':str(temp),'gpu_mapping_enabled':False},
            'runtime-before-imports.json':baseline,'runtime-provenance-sources.json':self.provenance,
            'cache-reconciliation.json':{'no_cache_modified':True,'pairs':[{
                'cache_path':str(prefix/c),'source_path':str(prefix/s),'cache_sha256':'a'*64,'source_sha256':'a'*64,
                'matches_stage_pins':True,'admissible_for_diagnosis':True,'cache_executed_by_comparison':False,
                'cpython_timestamp_reject_expected':True} for c,s in diagnostic.CACHE_PAIRS]}}
        distributions=[{'name':n,'version':v,'distribution_root':str(prefix),
            'metadata_path':str(prefix/(n+'-'+v+'.dist-info'))}
            for n,versions in service.RUNTIME['metadata_candidates'].items() for v in versions]
        records['runtime-metadata.json']={'distributions':distributions,
            'numpy_entries':[r for r in distributions if r['name']=='numpy'],
            'resolved_defaults':{r['name']:{'version':r['version'],'metadata_path':r['metadata_path']} for r in distributions}}
        snapshot={"packages":{k:{"version":v} for k,v in service.RUNTIME["imported_versions"].items()
                              if k in ('numpy','scipy','torch','transformers')},
            "additional_package_versions":{k:v for k,v in service.RUNTIME["imported_versions"].items()
                                            if k not in ('numpy','scipy','torch','transformers')},
            "generated_provenance":{"verified":True,'present':True,'temporary_root':str(temp),
                'directory':str(generated),'path':str(generated/(diagnostic.GENERATED_MODULE+'.py')),
                'source_content':self.provenance['expected_source'],'source_sha256':self.provenance['expected_source_sha256'],
                'expected_source_sha256':self.provenance['expected_source_sha256']},"captured_project_modules":{
                Path(n).stem:{"sha256":p} for n,p in self.freeze['sources'].items()
                if n!='llm_intervention_model_service.py'}}
        snapshot.update({k:copy.deepcopy(baseline[k]) for k in ('python_version','executable','prefix','base_prefix','mixed_native_paths')})
        snapshot['sys_path']=baseline['sys_path']+[str(generated)]
        snapshot['mixed_modules']={diagnostic.GENERATED_MODULE:snapshot['generated_provenance']['path']}
        snapshot['mixed_module_paths']=[snapshot['generated_provenance']['path']]
        for name in ('runtime-after-service-imports.json','runtime-after-cpu-tokenization.json'):
            records[name]=copy.deepcopy(snapshot)
        records['cpu-tokenization.json']={"model_weights_loaded":False,"adapter_deserialized":False,
            "explicit_cuda_api_called":False,"new_training_updates":0,"environment_responses":0,"fixtures":[{
                "name":n,"fixture_sha256":"a"*64,"envelope_sha256":"d"*64,"max_tokens":8192,"max_new_tokens":32,
                "enable_thinking":False,"prompt_ids":[1,2],"prompt_tokens":2,"completion_ids":[[3,151645]],
                "legal_commands":1,"longest_command_tokens":2} for n in sorted(cpu.FIXTURES)]}
        for i,name in enumerate(('service_cpu_import_start','service_cpu_import_complete',
                                 'cpu_tokenizer_load_start','cpu_tokenization_complete')):
            records[f'phase-{i:03d}.json']={"phase":name}
        return records

    def validate_records(self,records):
        out=self.root/'receipts';out.mkdir(exist_ok=True)
        for p in out.iterdir():p.unlink()
        for name,value in records.items():(out/name).write_text(json.dumps(value))
        return cpu.validate_session(out,'a'*64,'c'*64,self.freeze,self.files,service,{n:'d'*64 for n in cpu.FIXTURES},
            self.stage,diagnostic,runtime,self.provenance)

    def test_runtime_records_cannot_be_missing_or_contradictory(self):
        original=self.receipts()
        self.assertEqual(len(self.validate_records(original)),64)
        for name in ('runtime-before-imports.json','runtime-metadata.json','cache-reconciliation.json',
                     'runtime-provenance-sources.json'):
            bad=copy.deepcopy(original);bad.pop(name)
            with self.subTest(missing=name),self.assertRaises(FileNotFoundError):self.validate_records(bad)
        mutations=[lambda r:r['runtime-before-imports.json'].update(prefix='/wrong'),
            lambda r:r['runtime-after-service-imports.json'].update(mixed_native_paths=['/untrusted/lib.so']),
            lambda r:r['runtime-after-cpu-tokenization.json'].update(sys_path=['/wrong']),
            lambda r:r['runtime-stage-binding.json']['stage'].update(archive_sha256='0'*64),
            lambda r:r['runtime-provenance-sources.json'].update(expected_source='wrong'),
            lambda r:r['cache-reconciliation.json']['pairs'][0].update(matches_stage_pins=False),
            lambda r:r['runtime-metadata.json']['numpy_entries'].pop(),
            lambda r:r['runtime-after-service-imports.json']['additional_package_versions'].update(numpy='2.2.5'),
            lambda r:r['runtime-after-service-imports.json']['packages']['numpy'].update(version='wrong'),
            lambda r:r['runtime-after-cpu-tokenization.json']['generated_provenance'].update(source_content='wrong')]
        for i,change in enumerate(mutations):
            bad=copy.deepcopy(original);change(bad)
            with self.subTest(i=i),self.assertRaises(ValueError):self.validate_records(bad)

    def test_saved_receipts_accept_only_complete_cpu_scope(self):
        records=self.receipts()
        self.assertEqual(len(self.validate_records(records)),64)
        mutations=[lambda r:r['terminal.json'].update(model_loaded=True),
            lambda r:r['terminal.json'].update(runtime_stage_sha256='0'*64),
            lambda r:r['authenticated.json']['models'].pop('model.safetensors'),
            lambda r:r['runtime-after-service-imports.json']['packages']['numpy'].update(version='2.2.6'),
            lambda r:r['runtime-after-cpu-tokenization.json']['captured_project_modules'].pop('llm_intervention_learning'),
            lambda r:r['runtime-after-cpu-tokenization.json']['generated_provenance'].update(verified=False),
            lambda r:r['phase-003.json'].update(phase='model_ready'),
            lambda r:r.update({'failure.json':{'message':'failed'}})]
        for i,change in enumerate(mutations):
            bad=copy.deepcopy(records);change(bad)
            with self.subTest(i=i),self.assertRaises(ValueError):self.validate_records(bad)

    def test_saved_token_roster_and_count_corruptions_fail(self):
        records=self.receipts()
        for field,value in [('fixture_sha256','0'*64),('envelope_sha256','0'*64),('prompt_tokens',9000),
            ('completion_ids',[[3,0]]),('longest_command_tokens',33),('legal_commands',2),('enable_thinking',True)]:
            bad=copy.deepcopy(records);bad['cpu-tokenization.json']['fixtures'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate_records(bad)


if __name__=='__main__':unittest.main()
