"""Fabricated glue controls only. No scientific response, tokenizer or model.

Run directly with ACE_SCIENTIFIC_PREPARATION_TEST_OUTPUT pointing to an absent
external directory. Synthetic policy_train-shaped dictionaries test binding;
they are never outputs of the scientific collector or accepted study evidence.
All engine/exposure/ranker entrypoints in pipeline tests are explicit doubles.
"""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import prepare_llm_intervention_scientific_training_cpu as p

REPO=Path(__file__).resolve().parents[2]
PROPOSAL=REPO/'results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'


def raw(v):return (p.canonical(v)+'\n').encode()
def command(a):return p.canonical({'stop':True} if a is None else {'action_id':a})


def fabricated():
    proposal=p.decode(PROPOSAL.read_bytes())
    sources={**proposal['source_pins'],'scripts/research/llm_intervention_training_collection.py':'a'*64}
    bindings={'root':'/fabricated/collection','files':{},'freeze_sha256':'b'*64,
        'realization_acceptance_sha256':'c'*64,'world_pins':{str(w):'d'*64 for w in p.WORLD_IDS},
        'configuration_pins':{str(w):'e'*64 for w in p.WORLD_IDS},'source_pins':sources,
        'namespaces':{str(w['world_id']):w['training']['request_namespace'] for w in proposal['training_worlds']}}
    intent={'proposal_sha256':p.PROPOSAL_SHA,'expected_source_pins':sources,'world_pins':bindings['world_pins'],
        'configuration_pins':bindings['configuration_pins'],'world_roster':list(p.WORLD_IDS),'expected_roster_sha256':p.digest(p.WORLD_IDS)}
    files={'intent.json':raw(intent)};planned=[];observed=[];records={};receipts={}
    for i,rid in enumerate(p.ROSTER):
        wi,si=i//5,i%5;wid=p.WORLD_IDS[wi];candidates=proposal['training_worlds'][wi]['training']['snapshots'][si]['candidates']
        pairs=[[a,b] for j,a in enumerate(candidates) for b in candidates[j+1:]]
        env={'state':{'terminal':si==4,'fixture_only':True},'task':{'fixture_only':True}}
        receipt={'candidate_ids':candidates,'pair_count':len(pairs),'world_id':wid,'scope':'policy_train','snapshot_index':si,
            'task_sha256':p.digest(env['task']),'envelope_file_sha256':p.sha(raw(env))}
        rr=raw(receipt);rec={'scope':'policy_train','world_id':wid,'envelope':env,'supervised_action_id':candidates[0] if candidates else None,
            'preferences':[{'left':a,'right':b,'tie':False,'chosen':a,'rejected':b} for a,b in pairs],
            'supervision_receipt_sha256':p.sha(rr)}
        records[rid]=raw(rec);receipts[rid]=rr
        name=f'world-{wi:03d}/snapshot-{si:03d}'
        row={'record_id':rid,'scope':'policy_train','world_id':wid,'snapshot_index':si,'path':name,'candidate_ids':candidates,'pairs':pairs}
        planned.append(row);observed.append({**row,'status':'complete','record_sha256':p.sha(records[rid]),'supervision_receipt_sha256':p.sha(rr)})
        files[name+'/record.json']=records[rid];files[name+'/receipt.json']=rr
    plan={'world_roster':list(p.WORLD_IDS),'intent_sha256':p.sha(files['intent.json']),'source_file_sha256':sources,
          'records':planned,'maximum_response_attempts':9216,'maximum_reserved_cost_units':10240}
    files['plan.json']=raw(plan)
    inv={'scope':'policy_train','record_roster':list(p.ROSTER),'records':observed,'record_pins':{k:p.sha(v) for k,v in records.items()},
         'supervision_receipt_pins':{k:p.sha(v) for k,v in receipts.items()},'collection_plan_sha256':p.sha(files['plan.json'])}
    files['inventory.json']=raw(inv)
    account={'reserved_response_attempts':288,'known_returns':288,'executor_accepted_responses':288,'unknown_return_attempts':0,
             'total_reserved_cost_units':320,'accounting_complete':True,'accounting_errors':[]}
    worlds=[{'world_id':w,'status':'complete','accounting':copy.deepcopy(account)} for w in p.WORLD_IDS]
    terminal={'status':'complete','exposure_ready':True,'accounting_complete':True,'scope':'policy_train','record_roster':list(p.ROSTER),
        'inventory_sha256':p.sha(files['inventory.json']),'world_roster':list(p.WORLD_IDS),'planned_records':planned,'worlds':worlds,
        'accounting':{k:v*32 for k,v in account.items() if type(v)is int},'partial_records_authorized_for_training':False,
        'complete_case_filtering_allowed':False,'policy_training_performed':False,'scientific_evaluation_performed':False}
    files['terminal.json']=raw(terminal);bindings['files']={k:p.sha(v) for k,v in files.items()}
    packet={'collection':bindings,'protocol_source_pins':proposal['source_pins'],'model_files':{n:'f'*64 for n in p.TOKENIZER_NAMES|{'model.safetensors'}}}
    acceptance={'schema':'ace-scientific-collection-acceptance-v1','status':'accepted','required_findings':0,'scope':'policy_train',
        'complete_custody_verified':True,'accounting_complete':True,'collection':copy.deepcopy(bindings)}
    return packet,proposal,acceptance,files


class Journal:
    def __init__(self,root):self.output=root;root.mkdir();self.events=[]
    def save(self,name,value):return p.write(self.output/name,value)
    def event(self,name,**values):self.events.append((name,values))


class TokenizerDouble:
    eos_token_id=151645;pad_token_id=151643
    def __init__(self,overflow=False):self.calls=0;self.overflow=overflow
    def tokens(self,text):
        self.calls+=1
        if self.overflow:raise ValueError('fabricated full prompt exceeds 8192; no truncation')
        return [len(text),self.eos_token_id]


class GlueDoubles:
    def __init__(self,batch,dpo=128):
        self.batch=batch;self.calls=[];self.loads=0;self.tokenizer=TokenizerDouble()
        rows=[r for i,r in enumerate(p.ROSTER) if i%5<4][:dpo]
        schedule=[{'stage':'sft','record_id':r,'pair_index':None} for r in p.ROSTER]
        schedule += [{'stage':'dpo','record_id':r,'pair_index':0} for r in rows]
        self.exposure={'counts':{'records':160,'retained_pairs':3584,'sft_steps':160,'dpo_steps':dpo},
            'has_both_training_stages':bool(dpo),'schedule':schedule,
            'numeric_schedule':[{**s,'chosen_index':0,'rejected_index':1 if s['stage']=='dpo' else None} for s in schedule]}
    def propose(self,**kwargs):
        assert kwargs['records'] is self.batch['records'] and kwargs['record_roster']==p.ROSTER and kwargs['scope']=='policy_train'
        self.calls.append('exposure');return copy.deepcopy(self.exposure)
    def factory(self):self.loads+=1;self.calls.append('tokenizer');return self.tokenizer
    def bridge(self,**kwargs):
        assert kwargs['execution_kind']=='qwen_screen_singletons' and kwargs['expected_scope']=='policy_train'
        assert kwargs['records'] is self.batch['records'] and kwargs['receipts'] is self.batch['receipts']
        assert kwargs['expected_roster']==p.ROSTER and kwargs['checkpoint_steps']==(159,159+self.exposure['counts']['dpo_steps'])
        self.calls.append('bridge');records=[]
        for rid in p.ROSTER:
            rec=p.decode(self.batch['records'][rid]);pairs=rec['preferences'];actions={x for r in pairs for x in (r['left'],r['right'])}|{None}
            commands={command(a):kwargs['tokenizer'].tokens(command(a)) for a in actions}
            records.append({'record_id':rid,'raw_record_utf8':self.batch['records'][rid].decode(),'record_sha256':self.batch['record_pins'][rid],
                'supervision_receipt_sha256':self.batch['receipt_pins'][rid],'tokenizer_sha256':kwargs['tokenizer_pin'],
                'pad_token_id':151643,'eos_token_id':151645,'prompt_ids':[1,2,3], 'commands':commands,
                'token_metadata':{'prompt_tokens':3,'max_tokens':8192,'max_new_tokens':32}})
        d=self.exposure['counts']['dpo_steps']
        plan={'records':records,'record_pins':{r['record_id']:p.digest(r) for r in records},'scope':'policy_train',
            'execution_kind':kwargs['execution_kind'],'numeric_recipe':{'scoring_path':'canonical_selected_unpadded','max_tokens':8192,'max_new_tokens':32},
            'schedule':self.exposure['schedule'],'sft_steps':160,'dpo_steps':d,'sft_rows':160,'dpo_rows':2*d,
            'checkpoint_steps':list(kwargs['checkpoint_steps']),'outer_provenance_receipt_sha256':kwargs['outer_provenance_pin'],
            'expected_adapter_shapes':{f'model.layers.{i}.self_attn.{part}.{side}':shape for i in range(28)
                for part,width in (('q_proj',2048),('v_proj',1024)) for side,shape in (('A',[8,1024]),('B',[width,8]))},
            'declared_adapter_parameters':1146880,'max_trainable_parameters':1146880,
            'stored_tokens':sum(3+sum(map(len,r['commands'].values())) for r in records),'provenance_qualified':False,'one_pass_dpo':False}
        return {'training_plan':plan,'training_plan_sha256':p.digest(plan),'exposure':copy.deepcopy(self.exposure),
                'exposure_sha256':p.digest(self.exposure),'tokenizer_sha256':kwargs['tokenizer_pin'],
                'new_environment_responses':0,'training_updates':0,'scientific_qualified':False,'outer_execution_qualified':False}
    def numeric_recipe(self,*,artificial):
        assert artificial is False
        return {'artificial_control':False,'hidden':128,'seed':109,'device':'cpu','dtype':'float32','optimizer':'Adam',
                'learning_rate':.001,'beta':.1,'optimizer_state':'continuous_across_sft_dpo'}
    def prepare_plan(self,**kwargs):
        self.calls.append('numerical');assert kwargs['exposure']==self.exposure and kwargs['exposure_sha256']==p.digest(self.exposure)
        assert kwargs['records'] is self.batch['records'] and kwargs['recipe']==self.numeric_recipe(artificial=False)
        return {'scope':'policy_train','record_roster':list(kwargs['record_roster']),
            'raw_records_utf8':{k:v.decode() for k,v in kwargs['records'].items()},'record_pins':kwargs['record_pins'],
            'supervision_receipt_pins':kwargs['supervision_receipt_pins'],'exposure':kwargs['exposure'],
            'exposure_sha256':kwargs['exposure_sha256'],'checkpoint_steps':list(kwargs['checkpoint_steps']),
            'outer_provenance_receipt_sha256':kwargs['outer_provenance_receipt_sha256'],'recipe':kwargs['recipe'],'scientific_provenance_qualified':False}


class ScientificPreparationGlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['ACE_SCIENTIFIC_PREPARATION_TEST_OUTPUT']);cls.root.mkdir(exist_ok=False)
    def setUp(self):
        self.out=self.root/self._testMethodName;self.out.mkdir()
        self.packet,self.proposal,self.acceptance,self.files=fabricated()
    def binding(self):
        return p.collection_binding(self.packet,self.proposal,self.acceptance,self.files.__getitem__)
    def run_pipeline(self,double,journal,bridge=None):
        return p.prepare_pipeline(self.packet,double.batch,'1'*64,journal,exposure_api=double.propose,
            bridge_api=bridge or double.bridge,ranker_api=double,tokenizer_factory=double.factory)

    def test_complete_160_3584_binding_and_same_two_plan_exposure(self):
        batch=self.binding();self.assertEqual(len(batch['records']),160)
        double=GlueDoubles(batch);journal=Journal(self.out/'worker');summary=self.run_pipeline(double,journal)
        self.assertEqual(summary['dpo_steps'],128);self.assertEqual(double.calls,['exposure','tokenizer','bridge','numerical'])
        llm=p.decode((journal.output/'llm-prepared.json').read_bytes());num=p.decode((journal.output/'numerical-plan.json').read_bytes())
        self.assertEqual(llm['training_plan']['checkpoint_steps'],[159,287]);self.assertEqual(num['exposure'],llm['exposure'])
        self.assertFalse(llm['training_plan']['one_pass_dpo']);self.assertEqual(double.tokenizer.calls,1184)
        self.assertEqual((journal.output/'llm-prepared.json').read_bytes(),(journal.output/'llm-prepared.candidate.json').read_bytes())

    def test_d_zero_retains_exposure_before_any_tokenizer_or_bridge(self):
        double=GlueDoubles(self.binding(),0);journal=Journal(self.out/'worker')
        with self.assertRaisesRegex(ValueError,'D=0'):self.run_pipeline(double,journal)
        self.assertEqual(double.calls,['exposure']);self.assertEqual(double.loads,0)
        self.assertTrue((journal.output/'exposure.json').exists());self.assertFalse((journal.output/'llm-prepared.json').exists())

    def test_partial_d_uses_exact_checkpoints_without_all_pair_exposure_requirement(self):
        double=GlueDoubles(self.binding(),7);journal=Journal(self.out/'worker');self.run_pipeline(double,journal)
        self.assertEqual(p.decode((journal.output/'numerical-plan.json').read_bytes())['checkpoint_steps'],[159,166])

    def test_acceptance_world_namespace_and_protocol_gates(self):
        for field in ('status','complete_custody_verified','accounting_complete','scope'):
            old=copy.deepcopy(self.acceptance)
            self.acceptance[field]=False if field.endswith('verified') or field=='accounting_complete' else 'wrong'
            with self.subTest(field=field),self.assertRaises(ValueError):self.binding()
            self.acceptance=old
        for modify in (lambda c:c['world_pins'].pop(str(p.WORLD_IDS[-1])),
                       lambda c:c['namespaces'].__setitem__(str(p.WORLD_IDS[0]),'foreign'),
                       lambda c:c['source_pins'].__setitem__(next(iter(self.proposal['source_pins'])),'0'*64)):
            original=copy.deepcopy(self.packet);modify(self.packet['collection']);self.acceptance['collection']=copy.deepcopy(self.packet['collection'])
            with self.assertRaises(ValueError):self.binding()
            self.packet=original;self.acceptance['collection']=copy.deepcopy(original['collection'])

    def test_missing_reordered_pairs_and_unknown_accounting_reject(self):
        for kind in ('missing_row','world_order','unknown','pair_order','path_escape'):
            files=copy.deepcopy(self.files)
            if kind in ('missing_row','pair_order','path_escape'):
                inv=p.decode(files['inventory.json'])
                if kind=='missing_row':inv['records'].pop()
                elif kind=='pair_order':inv['records'][0]['pairs'][0].reverse()
                else:inv['records'][0]['path']='../escape'
                files['inventory.json']=raw(inv);term=p.decode(files['terminal.json']);term['inventory_sha256']=p.sha(files['inventory.json']);files['terminal.json']=raw(term)
            else:
                term=p.decode(files['terminal.json'])
                if kind=='world_order':term['worlds'].reverse()
                else:term['worlds'][0]['accounting']['unknown_return_attempts']=1
                files['terminal.json']=raw(term)
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.collection_binding(self.packet,self.proposal,self.acceptance,files.__getitem__)

    def test_original_record_or_receipt_mutation_rejects_before_pipeline(self):
        for suffix in ('record.json','receipt.json'):
            name='world-000/snapshot-000/'+suffix;original=self.files[name];self.files[name]=original+b' '
            with self.assertRaisesRegex(ValueError,'byte pins'):self.binding()
            self.files[name]=original

    def test_tokenization_failure_preserves_exposure_and_original_collection(self):
        before=copy.deepcopy(self.files);double=GlueDoubles(self.binding());double.tokenizer.overflow=True;journal=Journal(self.out/'worker')
        with self.assertRaisesRegex(ValueError,'no truncation'):self.run_pipeline(double,journal)
        self.assertEqual(self.files,before);self.assertTrue((journal.output/'exposure.json').exists())
        self.assertFalse((journal.output/'llm-prepared.json').exists());self.assertNotIn('numerical',double.calls)

    def test_bridge_exposure_drift_preserves_candidate_without_success(self):
        double=GlueDoubles(self.binding());journal=Journal(self.out/'worker')
        def wrong(**kw):
            result=double.bridge(**kw);result['exposure']['schedule'][-1]['pair_index']=2;return result
        with self.assertRaisesRegex(ValueError,'different exposure'):self.run_pipeline(double,journal,wrong)
        self.assertTrue((journal.output/'llm-prepared.candidate.json').exists());self.assertFalse((journal.output/'llm-prepared.json').exists())
        self.assertNotIn('numerical',double.calls)

    def test_rebound_bad_tokens_or_numerical_binding_reject(self):
        double=GlueDoubles(self.binding());journal=Journal(self.out/'worker');self.run_pipeline(double,journal)
        llm=p.decode((journal.output/'llm-prepared.json').read_bytes());num=p.decode((journal.output/'numerical-plan.json').read_bytes())
        for kind in ('eos','prompt_limit','wrong_mode','numeric_raw','numeric_scope','numeric_checkpoint'):
            a,b=copy.deepcopy(llm),copy.deepcopy(num)
            if kind=='eos':next(iter(a['training_plan']['records'][0]['commands'].values()))[-1]=0
            if kind=='prompt_limit':a['training_plan']['records'][0]['prompt_ids']=[1]*8192
            if kind=='wrong_mode':a['training_plan']['execution_kind']='qwen_gpu09'
            if kind=='numeric_raw':b['raw_records_utf8'][p.ROSTER[0]]='{}'
            if kind=='numeric_scope':b['scope']='policy_dev'
            if kind=='numeric_checkpoint':b['checkpoint_steps']=[159,160]
            a['training_plan']['record_pins']={r['record_id']:p.digest(r) for r in a['training_plan']['records']};a['training_plan_sha256']=p.digest(a['training_plan'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.validate_plans(a,b,double.exposure,self.packet,double.batch,'1'*64)

    def test_exact_six_tokenizer_files_reject_weight_and_mutation(self):
        root=self.out/'tokenizer';root.mkdir();pins={}
        for n in p.TOKENIZER_NAMES:
            data=n.encode();(root/n).write_bytes(data);pins[n]=p.sha(data)
        packet={'model_directory':str(root),'tokenizer_files':pins}
        self.assertEqual(set(p.tokenizer_inventory(packet)),p.TOKENIZER_NAMES)
        (root/'model.safetensors').write_bytes(b'never load')
        with self.assertRaises(ValueError):p.tokenizer_inventory(packet)
        (root/'model.safetensors').unlink();(root/'tokenizer.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):p.tokenizer_inventory(packet)

    def test_tree_custody_rejects_extra_symlink_and_mutation(self):
        root=self.out/'collection';root.mkdir();(root/'journal.jsonl').write_bytes(b'fabricated\n');pins={'journal.jsonl':p.sha(b'fabricated\n')}
        self.assertEqual(p.tree_pins(root,pins)['files'],1)
        (root/'extra').write_bytes(b'forbidden')
        with self.assertRaises(ValueError):p.tree_pins(root,pins)
        (root/'extra').unlink();(root/'link').symlink_to(root/'journal.jsonl')
        with self.assertRaises(ValueError):p.tree_pins(root,pins)
        (root/'link').unlink();(root/'journal.jsonl').write_bytes(b'changed')
        with self.assertRaises(ValueError):p.tree_pins(root,pins)
        with self.assertRaises(ValueError):p.safe_relative('../escape')

    def test_dynamic_cpu_contract_and_no_accelerator_gate(self):
        resources={'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal','cpus':4,'memory_mib':12288,'wall_seconds':1800,'gpus':0,'requeue':False}
        limits={'startup':300,'work':1700,'cleanup':1750,'hard':1780};p.resource_contract(resources,limits)
        text='JobId=123 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=4 NumTasks=1 NumNodes=1 Requeue=0 Restarts=0 UserId=paco0228(123) TimeLimit=00:30:00 AllocTRES=cpu=4,mem=12G,node=1 ReqTRES=cpu=4,mem=12G,node=1'
        with patch.dict(os.environ,{},clear=True):
            p.allocation(text,'123',resources)
            for wrong in (text.replace('cpu=4','cpu=8'),text.replace('mem=12G','mem=6G'),text+',gres/gpu=0',text.replace('00:30:00','00:10:00')):
                with self.assertRaises(ValueError):p.allocation(wrong,'123',resources)
            with patch.dict(os.environ,{'SLURM_JOB_GPUS':'0'}):
                with self.assertRaises(ValueError):p.allocation(text,'123',resources)
        with self.assertRaises(ValueError):p.resource_contract(resources,{**limits,'cleanup':1800})

    def test_exclusive_publication_and_failure_preserve_original_terminal(self):
        out=self.out/'owned';out.mkdir();p.publish(out,{'status':'prepared_only'})
        before=(out/'terminal.json').read_bytes();p.failure(out,ValueError('interrupted after link'))
        self.assertEqual((out/'terminal.json').read_bytes(),before);self.assertFalse((out/'failure.json').exists())
        self.assertTrue((out/'publication-interruption.json').exists())
        with self.assertRaises(FileExistsError):p.publish(out,{'status':'overwrite'})

    def test_parent_passes_caller_deadlines_then_cleanup_before_validation(self):
        out=self.out/'parent';out.mkdir();scratch=self.out/'scratch';scratch.mkdir()
        resources={'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal','cpus':4,'memory_mib':12288,'wall_seconds':1800,'gpus':0,'requeue':False}
        limits={'startup':300,'work':1700,'cleanup':1750,'hard':1780}
        packet={'resources':resources,'deadlines':limits,'files':{p.ENTRY:'2'*64}}
        seen=[];finalizer=types.SimpleNamespace()
        def supervise(cmd,env,session,started,pin,kill_owned):
            self.assertEqual([getattr(finalizer,k.upper()) for k in limits],list(limits.values()))
            self.assertEqual(cmd[cmd.index('--source-commit')+1],'3'*40)
            self.assertEqual(cmd[cmd.index('--parent-started')+1],str(started))
            seen.append('cleanup');return {'group_still_exists':False,'returncode':0,'errors':[]}
        finalizer.supervise=supervise
        diagnostic=types.SimpleNamespace(events=lambda *a:lambda *a,**k:None,
            stage_runtime=lambda *a:{'staged_prefix':str(self.out/'runtime')},provenance_contract=lambda *a:{},BOOTSTRAP='captured-only-double')
        ctx={'packet':packet,'diagnostic':diagnostic,'base':types.SimpleNamespace(cpu_environment=lambda *a:{}),
             'finalizer':finalizer,'root':self.out,'data':{n:b'fixture' for n in ('llm_intervention_runtime_archive.py','probe.py','llm_intervention_service_runtime.py')}}
        import time
        args=types.SimpleNamespace(output=str(out),packet=str(self.out/'packet.json'),packet_sha256='1'*64,source_commit='3'*40,
            started=time.monotonic(),process_started=0.)
        allocation='JobId=123 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=4 NumTasks=1 NumNodes=1 Requeue=0 Restarts=0 UserId=paco0228(123) TimeLimit=00:30:00 AllocTRES=cpu=4,mem=12G,node=1 ReqTRES=cpu=4,mem=12G,node=1'
        def validate(*a):
            self.assertEqual(seen,['cleanup']);seen.append('validate');return '4'*64
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_SCRATCH':str(scratch)},clear=True), \
             patch.object(p.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=allocation,stderr='')), \
             patch.object(p,'load_module',return_value=types.SimpleNamespace(kill_owned=lambda *a:None)), \
             patch.object(p,'validate_worker',side_effect=validate),patch.object(p,'recheck',side_effect=lambda *a:seen.append('recheck')):
            p.parent(args,ctx)
        self.assertEqual(seen,['cleanup','validate','recheck'])
        terminal=p.decode((out/'terminal.json').read_bytes())
        self.assertFalse(terminal['training_execution_qualified']);self.assertEqual(terminal['environment_responses'],0)

    def test_postcheck_rejects_changed_packet_before_other_input_reads(self):
        path=self.out/'packet.json';path.write_bytes(b'original packet bytes')
        ctx={'packet':{},'packet_path':path,'packet_sha256':p.sha(path.read_bytes())}
        path.write_bytes(b'changed packet bytes')
        with self.assertRaisesRegex(ValueError,'independent byte pin'):
            p.recheck(ctx)

    def test_import_is_stdlib_only_and_source_closure_complete(self):
        self.assertFalse({'torch','numpy','scipy','transformers'}.intersection(sys.modules))
        tree=ast.parse((REPO/'scripts/research/llm_intervention_model_service.py').read_bytes())
        service_names=next(ast.literal_eval(n.value.args[0]) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_NAMES' for t in n.targets))
        sources=service_names|p.EXTRA_SOURCES
        for name in sources:
            for node in ast.walk(ast.parse((REPO/'scripts/research'/name).read_bytes())):
                deps=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                for dep in deps:
                    if dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_')):self.assertIn(dep+'.py',sources)
        for name,pin in p.HELPER_PINS.items():
            if name!='probe.py':self.assertEqual(p.sha((REPO/'scripts/research'/name).read_bytes()),pin)


if __name__=='__main__':unittest.main(verbosity=2)
