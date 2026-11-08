"""New pure/mocked interface tests. No numerical libraries, fits or remote work."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock


ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('campaign_cpu_under_test',ROOT/'prepare_llm_intervention_campaign_cpu.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CampaignCPUTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'input'
        self.root.mkdir()
        self.service = m.module('test_service',ROOT/m.SERVICE,(ROOT/m.SERVICE).read_bytes())
        self.d = m.module('test_diagnostic',ROOT/m.DIAGNOSTIC,(ROOT/m.DIAGNOSTIC).read_bytes())
        self.training = m.module('test_training',ROOT/m.TRAINING,(ROOT/m.TRAINING).read_bytes())
        self.packet = dict(schema=m.VERSION,scope='artificial',record_roster=list(m.ROSTER),
            source_names=sorted(m.SOURCE_NAMES),files={n:'a'*64 for n in m.INPUT_NAMES},
            archive=dict(path='/archive/runtime.tar',sha256=self.d.ARCHIVE_SHA),
            manifest=dict(path='/archive/manifest.json',sha256=self.d.MANIFEST_SHA),
            runtime_source=self.d.RUNTIME_SOURCE,runtime=copy.deepcopy(self.service.RUNTIME),
            model_directory='/model',model_files=dict(m.MODEL_FILES),resources=dict(m.RESOURCES),
            persistent_specs={},services={},outer_provenance_receipt_sha256='b'*64,
            cpu_output=str(Path(self.temp.name)/'cpu01'))
        self.packet['files'].update(m.HELPER_PINS)
        self.packet['files'].update(self.training.COLLECTION_PINS)
        self.packet['files']['llm_intervention_gpu_mappings.py']=self.service.RUNTIME['gpu_mapping_source_sha256']
        self.data={m.LINEAGE:b'# lineage mocked only',m.LAUNCH:(ROOT/m.LAUNCH).read_bytes()}
        self.lineage={'roles':{role:dict(adapter_sha256=str(i)*64,checkpoint_sha256=str(i+2)*64)
                                for i,role in enumerate(('sft_llm','dpo_llm'),1)}}
        for role in m.LLM_POLICIES:
            adapter=None
            if role!='frozen_llm':
                stage='sft' if role=='sft_llm' else 'final'; r=self.lineage['roles'][role]
                adapter=dict(file=stage+'.adapter.pt',sha256=r['adapter_sha256'],
                    training_receipt=stage+'.checkpoint.json',training_receipt_sha256=r['checkpoint_sha256'],
                    training_scope='artificial')
            freeze=dict(schema=self.service.VERSION,scope='artificial',model_directory=self.packet['model_directory'],
                model_files=m.MODEL_FILES,sources={n:self.packet['files'][n] for n in self.service.SOURCE_NAMES},
                runtime=self.service.RUNTIME,python_version=[3,10,19],recipe=self.service.RECIPE,
                treatment='frozen' if adapter is None else 'artificial_adapter',adapter=adapter,limits=m.LIMITS)
            self.data[role+'.freeze.json']=m.canonical(freeze).encode()
            self.packet['files'][role+'.freeze.json']=m.sha(self.data[role+'.freeze.json'])
            identity=self.service.identity(freeze)
            self.packet['services'][role]=dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=8)
            gpu=Path(self.temp.name)/'gpu01'
            contract=dict(schema='ace-campaign-staged-launch-v1',policy_id=role,entry=str(self.root/m.SERVICE),
                entry_sha256=self.packet['files'][m.SERVICE],freeze=str(self.root/(role+'.freeze.json')),
                freeze_sha256=self.packet['files'][role+'.freeze.json'],worker_output=str(gpu/'services'/role/'worker'),
                launch_output=str(gpu/'services'/role/'launch'),binding_path=str(gpu/'stage-binding.json'),
                stage_path=str(gpu/'stage.json'),archive_sha256=self.d.ARCHIVE_SHA,manifest_sha256=self.d.MANIFEST_SHA,
                bootstrap=self.d.BOOTSTRAP,bootstrap_sha256=m.sha(self.d.BOOTSTRAP.encode()))
            self.data[role+'.launch.json']=m.canonical(contract).encode()
            self.packet['files'][role+'.launch.json']=m.sha(self.data[role+'.launch.json'])
            command=[str(Path(self.d.RUNTIME_SOURCE)/'bin/python'),'-I','-B','-c',self.d.BOOTSTRAP,
                str(self.root/m.LAUNCH),self.packet['files'][m.LAUNCH],'--contract',str(self.root/(role+'.launch.json')),
                '--contract-sha256',self.packet['files'][role+'.launch.json']]
            self.packet['persistent_specs'][role]=dict(policy_id=role,kind='persistent',command=command,
                timeout_seconds=35,identity_sha256=m.digest(identity),
                policy_provenance_sha256=self.packet['files'][role+'.freeze.json'])

    def admit(self):
        original=m.module
        def fake(name,path,raw):
            return types.SimpleNamespace(admit_training=lambda **kw:self.lineage) if Path(path).name==m.LINEAGE else original(name,path,raw)
        with patch.object(m,'module',side_effect=fake),patch.object(m,'training_inputs',return_value={}):
            return m.admit(self.root,self.packet,self.data,self.service,self.d)

    def test_packet_exact_roster_resources_and_model(self):
        m.packet_shape(self.packet,self.service,self.d,self.training)
        for key,value in [('resources',{**m.RESOURCES,'memory_mib':6144}),('model_files',{}),
                          ('source_names',sorted(m.SOURCE_NAMES-{m.ENTRY})),('cpu_output','relative')]:
            bad=copy.deepcopy(self.packet); bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.packet_shape(bad,self.service,self.d,self.training)
        bad=copy.deepcopy(self.packet); bad['files']['../elsewhere.py']='a'*64
        with self.assertRaises(ValueError): m.packet_shape(bad,self.service,self.d,self.training)

    def test_service_roles_checkpoint_receipts_bounds_and_exact_argv(self):
        self.assertEqual(self.admit(),self.lineage)
        original=copy.deepcopy(self.packet)
        self.packet['persistent_specs']['frozen_llm']['command'][0]='/usr/bin/python3'
        with self.assertRaisesRegex(ValueError,'argv'): self.admit()
        self.packet=original
        self.packet['services']['sft_llm']['startup_seconds']=120
        with self.assertRaisesRegex(ValueError,'client'): self.admit()
        self.packet['services']['sft_llm']['startup_seconds']=125
        freeze=m.decode(self.data['sft_llm.freeze.json'])
        freeze['adapter']['training_receipt_sha256']='9'*64
        self.data['sft_llm.freeze.json']=m.canonical(freeze).encode()
        with self.assertRaisesRegex(ValueError,'checkpoint role'): self.admit()

    def test_contract_overlap_and_future_gpu_hash_not_required(self):
        self.admit()
        self.assertNotIn('qualify_llm_intervention_campaign_gpu.py',m.INPUT_NAMES)
        contract=m.decode(self.data['dpo_llm.launch.json'])
        contract['worker_output']=self.packet['cpu_output']+'/bad'
        self.data['dpo_llm.launch.json']=m.canonical(contract).encode()
        with self.assertRaisesRegex(ValueError,'overlap'): self.admit()

    def test_captured_dependency_closures_and_pinned_helpers(self):
        for n,p in m.HELPER_PINS.items():
            if n!='probe.py': self.assertEqual(m.sha((ROOT/n).read_bytes()),p,n)
        for names in [m.SOURCE_NAMES,*m.NUMERIC_SOURCE_NAMES.values()]:
            for name in names:
                tree=ast.parse((ROOT/name).read_bytes())
                for node in ast.walk(tree):
                    imports=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                             [a.name for a in node.names] if isinstance(node,ast.Import) else [])
                    for imported in imports:
                        if imported.startswith(m.PROJECT_PREFIXES):
                            self.assertIn(imported+'.py',names,(name,imported))
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('torch',sys.modules)

    def test_capture_exclusive_save_and_symlink_rejection(self):
        path=self.root/'one.json'; p=m.save(path,{'a':1})
        self.assertEqual(m.decode(m.capture(path,p)),{'a':1})
        with self.assertRaises(FileExistsError): m.save(path,{'a':2})
        alias=self.root/'alias.json'; alias.symlink_to(path)
        with self.assertRaises(OSError): m.capture(alias,p)
        with self.assertRaises(ValueError): m.capture(path,'0'*64)

    def test_loader_blocks_cached_and_unapproved_project_imports(self):
        name='llm_intervention_fake'
        with patch.dict(sys.modules,{name:types.ModuleType(name)}):
            with self.assertRaisesRegex(ValueError,'cached'): m.CapturedSources({},self.root)
        loader=m.CapturedSources({name+'.py':b'value=17\n'},self.root)
        spec=loader.find_spec(name); mod=importlib.util.module_from_spec(spec); loader.exec_module(mod)
        self.assertEqual(mod.value,17); self.assertIs(loader.loaded[name],mod)
        for name in ('llm_intervention_other','prepare_llm_intervention_other','qualify_llm_intervention_other'):
            with self.assertRaises(ImportError): loader.find_spec(name)

    def test_numeric_binding_tail_and_same_prefix(self):
        b=m.numeric_binding(9101,'target_risk')
        args=['--binding-json',m.canonical(b),'--binding-sha256',m.digest(b)]
        m.numeric_tail('screen',args,self.packet)
        with self.assertRaises(ValueError): m.numeric_tail('screen',args+['--anything'],self.packet)
        b['world_id']=9103
        with self.assertRaises(ValueError): m.numeric_tail('screen',['--binding-json',m.canonical(b),'--binding-sha256',m.digest(b)],self.packet)
        prefix=m.numeric_prefix(self.root,self.packet,self.root/'packet.json','0'*64,'/stage','1'*64,480.,'screen',self.d.BOOTSTRAP)
        self.assertEqual(prefix[:5],['/stage/bin/python','-I','-B','-c',self.d.BOOTSTRAP])
        self.assertEqual(prefix[-1],'--')
        self.assertIn(str(self.root/m.ENTRY),prefix)

    def test_light_numeric_auth_reads_no_adapter_model_or_archive_and_allows_parent_alias(self):
        packet=self.packet
        packet_path=self.root/'packet.json'
        data={n:b'# captured\n' for n in m.INPUT_NAMES}
        for n in (m.TRAINING,m.DIAGNOSTIC,m.SERVICE): data[n]=(ROOT/n).read_bytes()
        data[m.ENTRY]=(ROOT/m.ENTRY).read_bytes()
        for n,b in data.items(): packet['files'][n]=m.sha(b)
        packet['files'].update({n:p for n,p in m.HELPER_PINS.items() if n not in (m.TRAINING,m.DIAGNOSTIC,m.SERVICE)})
        packet['files'].update(self.training.COLLECTION_PINS)
        packet['files']['llm_intervention_gpu_mappings.py']=self.service.RUNTIME['gpu_mapping_source_sha256']
        calls=[]
        def captured(path,p,maximum=0):
            calls.append(Path(path).name)
            return m.canonical(packet).encode() if Path(path)==packet_path else data[Path(path).name]
        # resolve differs, as it can under a CURC ancestor mount alias. The root
        # leaf itself remains a real directory and lexical contract paths survive.
        with patch.object(m,'capture',side_effect=captured),patch.object(m,'__file__',str(self.root/m.ENTRY)),\
             patch.object(Path,'resolve',return_value=Path('/different/mount')):
            root,p,source=m.numeric_authenticate(packet_path,'0'*64,'screen')
        self.assertEqual(root,self.root)
        self.assertEqual(set(source),m.NUMERIC_SOURCE_NAMES['screen']|{m.ENTRY})
        self.assertEqual(set(calls),m.NUMERIC_SOURCE_NAMES['screen']|{m.ENTRY,m.TRAINING,m.DIAGNOSTIC,m.SERVICE,'packet.json'})
        self.assertFalse(any(n.endswith('.pt') for n in calls))

    def test_allocation_exact_memory_and_zero_gpu(self):
        fields='JobId=123 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:10:00 Requeue=0 Restarts=0 AllocTRES=cpu=2,mem=7680M,node=1,billing=2'
        m.allocation_fields(fields,'123')
        for changed in (fields.replace('7680M','6G'),fields+',gres/gpu=1',fields.replace('Restarts=0','Restarts=1')):
            with self.assertRaises(ValueError): m.allocation_fields(changed,'123')

    def test_custom_child_480_deadline_and_cleanup_on_receipt_failure(self):
        proc=Mock(pid=123); proc.wait.return_value=0
        cleanup=dict(pid=123,pgid=123,kill_sent=False,group_still_exists=False,returncode=0,errors=[])
        clean=Mock(return_value=cleanup)
        with patch.object(m.subprocess,'Popen',return_value=proc) as popen,\
             patch.object(m.time,'monotonic',return_value=100.),patch.object(m.signal,'setitimer'),patch.object(m.signal,'signal'):
            result=m.supervise_child(['captured-python'],{},self.root,480.,580.,clean)
        proc.wait.assert_called_once_with(timeout=380.)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(result['cleanup'],cleanup); clean.assert_called_once_with(proc)
        for p in self.root.iterdir(): p.unlink()
        original=m.save
        def failing_save(path,value):
            if Path(path).name=='worker.started.json': raise OSError('receipt failure')
            return original(path,value)
        clean.reset_mock()
        with patch.object(m.subprocess,'Popen',return_value=proc),patch.object(m.time,'monotonic',return_value=100.),\
             patch.object(m.signal,'setitimer'),patch.object(m.signal,'signal'),patch.object(m,'save',side_effect=failing_save):
            with self.assertRaisesRegex(OSError,'receipt failure'):
                m.supervise_child(['captured-python'],{},self.root,480.,580.,clean)
        clean.assert_called_once_with(proc)
        self.assertTrue((self.root/'worker.cleanup.json').is_file())

    def test_timeout_retains_cleanup_and_prevents_success(self):
        proc=Mock(pid=123); proc.wait.side_effect=subprocess.TimeoutExpired('child',380)
        cleanup=dict(pid=123,pgid=123,kill_sent=True,group_still_exists=False,returncode=-9,errors=[])
        with patch.object(m.subprocess,'Popen',return_value=proc),patch.object(m.time,'monotonic',return_value=100.),\
             patch.object(m.signal,'setitimer'),patch.object(m.signal,'signal'):
            with self.assertRaises(subprocess.TimeoutExpired):
                m.supervise_child(['captured-python'],{},self.root,480.,580.,lambda p:cleanup)
        self.assertEqual(m.decode((self.root/'worker.cleanup.json').read_bytes()),cleanup)

    def test_saved_inventory_separates_evolving_campaign(self):
        out=m.paths(self.packet)['output']; campaign=m.paths(self.packet)['campaign']
        campaign.mkdir(parents=True); m.save(out/'supervisor.json',{'complete':True})
        m.save(campaign/'campaign.json',{'initial':True})
        before=m.inventory(out,('saved-validation','worker/artifacts/campaign'))
        (campaign/'policy-05').mkdir(); m.save(campaign/'policy-05/attempt.json',{'gpu':'later'})
        self.assertEqual(m.inventory(out,('saved-validation','worker/artifacts/campaign')),before)
        self.assertIn('policy-05/attempt.json',m.inventory(campaign))

    def test_main_exclusive_output_and_failed_auth_retained(self):
        packet_path=self.root/'packet.json'; p=m.save(packet_path,self.packet)
        args=['entry','--packet',str(packet_path),'--packet-sha256',p]
        with patch.object(sys,'argv',args),patch.object(m.signal,'setitimer'),patch.object(m.signal,'signal'),\
             patch.object(m,'authenticate',side_effect=ValueError('authentication failed')) as auth:
            with self.assertRaisesRegex(ValueError,'authentication failed'): m.main()
            prior=m.inventory(m.paths(self.packet)['output'])
            with self.assertRaises(FileExistsError): m.main()
            self.assertEqual(m.inventory(m.paths(self.packet)['output']),prior)
            self.assertEqual(auth.call_count,1)
        self.assertIn('failure.json',prior)


if __name__=='__main__':
    unittest.main()
