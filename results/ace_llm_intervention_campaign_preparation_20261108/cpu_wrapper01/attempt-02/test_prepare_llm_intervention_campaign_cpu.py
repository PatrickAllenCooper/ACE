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

    def test_saved_bundle_token_attempt_and_numeric_receipt_joins(self):
        """Synthetic JSON only; this fixture is not actual campaign coverage."""
        p=self.packet; loc=m.paths(p); loc['artifacts'].mkdir(parents=True)
        prefixes={k:m.numeric_prefix(self.root,p,self.root/'packet.json','0'*64,'/stage','1'*64,480.,k,self.d.BOOTSTRAP)
                  for k in ('screen','ranker')}
        ranker=dict(exports={'dpo':dict(sha256='2'*64,training_receipt_sha256='3'*64)})
        sources=[]; token_rows=[]
        for wid,target in ((9101,'Y'),(9102,'Z')):
            specs=[]
            for pid in m.NUMERIC_POLICIES:
                binding=m.numeric_binding(wid,pid)
                specs.append(dict(policy_id=pid,kind='task_process',command=prefixes['screen']+[
                    '--binding-json',m.canonical(binding),'--binding-sha256',m.digest(binding)],
                    timeout_seconds=30.,identity_sha256=None,
                    policy_provenance_sha256=m.digest({n:p['files'][n] for n in m.NUMERIC_SOURCE_NAMES['screen']})))
            specs.append(dict(policy_id='learned_ranker',kind='task_process',command=prefixes['ranker']+[
                '--method','learned_ranker','--seed','109','--draws','32','--checkpoint',
                str(loc['phase']/'ranker/dpo.ranker.pt'),'--checkpoint-sha256','2'*64,
                '--training-receipt-sha256','3'*64],timeout_seconds=30.,identity_sha256=None,policy_provenance_sha256='4'*64))
            specs.extend(p['persistent_specs'][pid] for pid in m.LLM_POLICIES)
            graph=[dict(name=target,parents=[],unit='synthetic',intervention_bounds=None)]
            plan=dict(world=dict(world_id=wid,scope='artificial'),learner=dict(graph=graph),menu=[],
                task=dict(target=target),objective='synthetic join test',domain_notes='No execution',
                initial_actions=['observe_all']*2,acquisition_cap_units=4,
                outer_provenance_receipt_sha256=p['outer_provenance_receipt_sha256'],policy_specs=specs)
            sources.append(dict(plan=plan,plan_sha256=m.digest(plan)))
            for count in (2,5):
                state=dict(graph=graph,menu=[],objective=plan['objective'],domain_notes=plan['domain_notes'],
                    requests=count,returned_responses=count,spent_units=count,remaining_units=6-count,terminal=False,
                    history=[dict(request_id=i,action_id='observe_all',kind='observe',targets={},values={target:0.},cost_units=1)
                             for i in range(count)],scm=[dict(node=target,terms=['intercept'],parameter_mean=[0.],
                        parameter_covariance=[[1.]],noise_variance=.25,eligible_rows=count)])
                envelope=dict(version='ace-public-prediction-task-v1',state=state,task=plan['task'])
                token_rows.append(dict(world_id=wid,fixture='fabricated_initial_shape' if count==2 else 'fabricated_late_affordability_shape',
                    environment_responses=0,envelope=envelope,envelope_sha256=m.digest(envelope),task_sha256=m.digest(plan['task']),
                    prompt_ids=[1],completion_ids=[[2,151645]],prompt_tokens=1,max_tokens=8192,max_new_tokens=32,
                    enable_thinking=False,legal_commands=1,longest_command_tokens=2,prompt_sha256='f'*64))
        campaign=dict(scope='artificial',policy_ids=list(m.CPU_POLICIES+m.LLM_POLICIES),
            policy_phases={pid:'cpu' if pid in m.CPU_POLICIES else 'gpu' for pid in m.CPU_POLICIES+m.LLM_POLICIES},
            services=p['services'],sources=sources)
        bundle=dict(campaign=campaign,campaign_sha256=m.digest(campaign),source_plan_pins=[
            dict(world_id=s['plan']['world']['world_id'],sha256=s['plan_sha256']) for s in sources])
        plans=m.validate_bundle(bundle,p,prefixes,ranker,'4'*64)
        tokens=dict(scope='fabricated shape probes only',rows=token_rows,environment_responses=0,no_all_adaptive_prompt_bound_claim=True)
        m.validate_tokens(tokens,plans)
        altered=copy.deepcopy(tokens); altered['rows'][0]['envelope']['state']['history'][0]['values']['Y']=1.
        with self.assertRaises(ValueError): m.validate_tokens(altered,plans)
        altered=copy.deepcopy(bundle); altered['campaign']['sources'][0]['plan']['policy_specs'][4]['command'][-1]='9'*64
        with self.assertRaises(ValueError): m.validate_bundle(altered,p,prefixes,ranker,'4'*64)
        loc['campaign'].mkdir(); loc['numeric'].mkdir()
        m.save(loc['campaign']/'campaign.json',campaign)
        m.save(loc['campaign']/'source-plan-pins.json',bundle['source_plan_pins'])
        m.save(loc['campaign']/'start.json',dict(retry_allowed=False))
        attempts={}
        for index,pid in enumerate(m.CPU_POLICIES):
            directory=loc['campaign']/f'policy-{index:02d}'; directory.mkdir(); arms=[]
            for i,source in enumerate(sources):
                arm=directory/f'world-{i:02d}'; arm.mkdir(); (arm/'episode').mkdir()
                envelope=token_rows[i*2]['envelope']; payload=m.canonical(envelope).encode()
                command=source['plan']['policy_specs'][index]['command']
                kind='ranker' if pid=='learned_ranker' else 'screen'; argv=command[command.index('--')+1:]
                process=dict(command=command,input_sha256=m.sha(payload),returncode=0,stdout='{"stop":true}\n',stderr='{}\n')
                m.save(arm/'episode/decision-0000.input.json',envelope)
                m.save(arm/'episode/decision-0000.process.json',process)
                m.save(arm/'execution-diagnostics.json',dict(episode_returned_cleanly=True,adapters_closed_cleanly=True))
                receipt=dict(status='raw_ready',plan_sha256=source['plan_sha256'],policy_id=pid,new_training_updates=0,
                    private_scores_computed=False,retry_allowed=False,scientific_provenance_qualified=False,
                    source_evidence_sha256=m.inventory(arm))
                arm_pin=m.save(arm/'arm-receipt.json',receipt); m.save(arm/'arm-receipt.candidate.json',receipt)
                arms.append(dict(world_id=source['plan']['world']['world_id'],status='completed',reason=None,
                    path=arm.name,receipt_sha256=arm_pin))
                cid=m.digest(dict(kind=kind,argv=argv,input_sha256=process['input_sha256']))
                call=loc['numeric']/cid; call.mkdir(); (call/'input.utf8').write_bytes(payload)
                (call/'stdout.log').write_text(process['stdout']); (call/'stderr.log').write_text(process['stderr'])
                source_pins={n:p['files'][n] for n in m.NUMERIC_SOURCE_NAMES[kind]|{m.ENTRY}}
                m.save(call/'command.json',dict(entry=kind,argv=argv,source_sha256=source_pins,packet_sha256='0'*64,stage_sha256='1'*64))
                m.save(call/'terminal.json',dict(status='complete',call_id=cid,packet_sha256='0'*64,
                    input_sha256=process['input_sha256'],stdout_sha256=m.sha(process['stdout'].encode()),
                    stderr_sha256=m.sha(process['stderr'].encode()),source_sha256=source_pins,
                    elapsed_seconds=.1,cpu_seconds=.05,retry_allowed=False))
                before=dict(prefix='/stage',python_version='3.10.19',pid=200+index*2+i,pgid=123,versions={},used_module_paths={},
                    standard_library_paths={'json':'/stage/lib/python3.10/json/__init__.py'},source_sha256={},
                    archive_rescanned=False,llm_model_or_adapter_loaded=False)
                used={'numpy':'/stage/lib/python3.10/site-packages/numpy/__init__.py'}
                versions={'numpy':p['runtime']['imported_versions']['numpy']}
                if kind=='ranker':
                    versions['torch']=p['runtime']['imported_versions']['torch']
                    used['torch']='/stage/lib/python3.10/site-packages/torch/__init__.py'
                after={**before,'versions':versions,'used_module_paths':used,
                       'source_sha256':{n:p['files'][n] for n in m.NUMERIC_SOURCE_NAMES[kind]}}
                m.save(call/'before.json',before); m.save(call/'after.json',after)
            attempt=dict(policy_id=pid,campaign_sha256=bundle['campaign_sha256'],phase='cpu',
                source_plan_pins=bundle['source_plan_pins'],errors=[],retry_allowed=False,private_scores_computed=False,
                service=dict(status='not_started',file=None,sha256=None),arms=arms,evidence_sha256=m.inventory(directory))
            attempts[pid]=m.save(directory/'attempt.json',attempt); m.save(directory/'attempt.candidate.json',attempt)
        processes=m.validate_attempts(loc['campaign'],bundle,attempts)
        self.assertEqual(len(processes),10)
        m.validate_numeric_calls(loc['numeric'],processes,p,'0'*64,dict(staged_prefix='/stage'),'1'*64,123)
        changed=copy.deepcopy(processes); changed[0]['stdout']='{"action_id":"bad"}\n'
        with self.assertRaises(ValueError):
            m.validate_numeric_calls(loc['numeric'],changed,p,'0'*64,dict(staged_prefix='/stage'),'1'*64,123)
        with self.assertRaisesRegex(ValueError,'stage observation'):
            m.validate_numeric_calls(loc['numeric'],processes,p,'0'*64,dict(staged_prefix='/stage'),'1'*64,999)
        (loc['campaign']/'policy-05').mkdir()
        with self.assertRaisesRegex(ValueError,'GPU attempts'):
            m.validate_attempts(loc['campaign'],bundle,attempts)

    def test_signal_is_not_ordinary_policy_exception_and_retains_failure(self):
        self.assertTrue(issubclass(m.CampaignInterrupted,BaseException))
        self.assertFalse(issubclass(m.CampaignInterrupted,Exception))
        packet_path=self.root/'packet.json'; p=m.save(packet_path,self.packet)
        handlers={}
        def register(sig,handler): handlers[sig]=handler
        def interrupt(*args): handlers[signal.SIGALRM](signal.SIGALRM,None)
        with patch.object(sys,'argv',['entry','--packet',str(packet_path),'--packet-sha256',p]),\
             patch.object(m.signal,'setitimer'),patch.object(m.signal,'signal',side_effect=register),\
             patch.object(m,'authenticate',side_effect=interrupt),patch.object(m,'parent') as parent:
            with self.assertRaises(m.CampaignInterrupted): m.main()
            parent.assert_not_called()
        failure=m.decode((m.paths(self.packet)['output']/'failure.json').read_bytes())
        self.assertEqual(failure['type'],'CampaignInterrupted')
        self.assertFalse(failure['retry_allowed'])


if __name__=='__main__':
    unittest.main()
