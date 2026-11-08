"""NEW hand-built metadata joins only. Never run a builder, model, remote command or science."""
import ast,copy,hashlib,importlib.util,json,os
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_HISTORY_TEST_EVIDENCE'])
spec=importlib.util.spec_from_file_location('new_gpu_history',HERE/'build_packet.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
def h(label):return hashlib.sha256(label.encode()).hexdigest()
def meta(label):return dict(bytes=1,sha256=h(label))
def row(job,seconds=1):return [job,'fixture','COMPLETED','0:0','3',str(seconds),'ucb736_asc2','cpu=3,mem=11520M,node=1']
def fixture():
    pins={n:h(n) for n in B.METADATA}
    pins.update({'freeze':B.FREEZE_SHA,'failed-token-packet':B.FAILED_TOKEN_PACKET_SHA,
        'failed-token-stage':B.FAILED_TOKEN_STAGE_SHA,'failed-token-result':B.FAILED_TOKEN_RESULT_SHA})
    oldrows=[row(str(1000+i)) for i in range(42)]
    old=dict(terminal_jobs=42,submission_journals=86,related_accounting=oldrows,
        journal_pins={f'/fixture/journal{i}':h(str(i)) for i in range(86)},closed_cpu_seconds=126,closed_gpu_slice_seconds=0)
    tokenrow=[B.FAILED_TOKEN_JOB,'ace_llm_scientific_token01','FAILED','1:0','3','129','ucb736_asc2','cpu=3,mem=11520M,node=1']
    result=dict(job_id=B.FAILED_TOKEN_JOB,returncode=0,ambiguous=False,intent_sha256=h('intent'))
    packet=dict(remote_root=B.FAILED_TOKEN_ROOT,source_commit=B.SOURCE_COMMIT,resources=B.TOKEN_RESOURCES,
        deadlines=B.TOKEN_ORIGINAL_DEADLINES,baseline_sha256=pins['previous-baseline'],new_input_files={
            'inputs/packet.json':pins['failed-token-stage'],'collection-acceptance.json':pins['collection-acceptance'],
            'namespace-claim.json':pins['namespace-claim'],'freeze.json':B.FREEZE_SHA})
    collection=dict(collection={'root':B.COLLECTION_ROOT+'/collection/session/worker/task/collection','files':{'fixture':h('collection')}})
    stage=dict(source_commit=B.SOURCE_COMMIT,resources=B.TOKEN_RESOURCES,deadlines=B.TOKEN_ORIGINAL_DEADLINES,
        model_directory=B.FAILED_TOKEN_ROOT+'/tokenizer',collection=collection['collection'],
        files={B.TOKEN_ENTRY:B.TOKEN_ENTRY_SHA},source_names=[B.TOKEN_ENTRY])
    files={'launch/'+n:dict(bytes=1,sha256=v) for n,v in packet['new_input_files'].items()}
    files.update({'launch/packet.json':dict(bytes=1,sha256=pins['failed-token-packet']),
        'launch/submission.intent.json':dict(bytes=1,sha256=result['intent_sha256']),
        'launch/submission.result.json':dict(bytes=1,sha256=pins['failed-token-result']),
        'launch/preparation/failure.json':meta('observed failure'),'common/fixture':meta('collection'),
        'failed01/fixture':meta('old collection failure')})
    journals={**old['journal_pins'],B.FAILED_TOKEN_ROOT+'/submission.intent.json':result['intent_sha256'],
        B.FAILED_TOKEN_ROOT+'/submission.result.json':pins['failed-token-result']}
    observed=dict(terminal_gate=True,files=files,directories=['launch','launch/inputs','launch/preparation','common','failed01'],
        original_hardlink_groups=[],journal_pins=journals,parent_accounting=tokenrow,upstream_pins={},upstream_errors={},
        related_accounting=oldrows+[tokenrow])
    transport=dict(returncode=0,stdout=json.dumps(observed),python_source='ROOT='+repr(B.FAILED_TOKEN_ROOT)+'\nJOB='+repr(B.FAILED_TOKEN_JOB)+'\n')
    disp=dict(job_id=B.FAILED_TOKEN_JOB,remote_root=B.FAILED_TOKEN_ROOT,packet_sha256=pins['failed-token-packet'],
        submission_result_sha256=pins['failed-token-result'],status='terminal_failed_outputs_preserved',
        custody_bytes_verified=True,current_lineage_verified=True,errors=[],input_pin_mismatches=[],acceptance=False,tokenization_accepted=False,
        remote_files=files,raw_files=len(files),raw_bytes=len(files),closed_cpu_seconds=513,closed_gpu_slice_seconds=0)
    base=dict(terminal_jobs=43,submission_journals=88,related_accounting=observed['related_accounting'],journal_pins=journals,
        input_pins=dict(predecessor=pins['previous-baseline'],acceptance=pins['token-failure-acceptance'],disposition=pins['token-failure-disposition']),
        closed_cpu_seconds=513,closed_gpu_slice_seconds=0,cpu_admission_seconds=4113,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600)
    a=dict(schema='ace-scientific-tokenization-failure-acceptance-v1',status='accepted_failed_before_token_worker',
        failure_accepted=True,tokenization_accepted=False,complete_custody_verified=True,required_findings=0,
        job_id=B.FAILED_TOKEN_JOB,packet_sha256=pins['failed-token-packet'],disposition_sha256=pins['token-failure-disposition'],
        freeze_sha256=B.FREEZE_SHA,source_commit=B.SOURCE_COMMIT,source_control_flow_and_traceback_verified=True,
        two_remote_inventories_matched=True,owned_worker_spawned=False,evidence_pins={
            'inventory_transport.json':pins['token-failure-inventory'],'inventory_after_transport.json':pins['token-failure-readback'],
            'custody-inventory.json':pins['token-failure-custody-inventory']})
    proof=dict(schema='ace-scientific-tokenization-no-learnedwork-proof-v1',status='verified_failed_before_token_worker',failed_job_id=B.FAILED_TOKEN_JOB,
        failure_acceptance_sha256=pins['token-failure-acceptance'],accounting_sha256=pins['token-failure-baseline'],
        packet_sha256=pins['failed-token-packet'],disposition_sha256=pins['token-failure-disposition'],freeze_sha256=B.FREEZE_SHA,
        source_commit=B.SOURCE_COMMIT,collection_acceptance_sha256=pins['collection-acceptance'],
        proof=dict(owned_worker_not_spawned=True,token_preparation_not_invoked=True))
    for v in (a,proof):
        for k in ('scientific_worlds_generated','environment_response_attempts','environment_responses','unknown_return_attempts','tokenizer_calls','model_loads','inference_calls','training_updates','policy_updates'):v[k]=0
    capsule={'remote/'+n:v for n,v in files.items()}
    capsule.update({n:dict(bytes=1,sha256=pins[key]) for n,key in [('disposition.json','token-failure-disposition'),('inventory_transport.json','token-failure-inventory'),('inventory_after_transport.json','token-failure-readback')]})
    m={'previous-baseline':old,'token-failure-baseline':base,'failed-token-packet':packet,'failed-token-stage':stage,
        'failed-token-result':result,'token-failure-acceptance':a,'token-no-learnedwork-proof':proof,
        'token-failure-disposition':disp,'token-failure-inventory':transport,'token-failure-readback':copy.deepcopy(transport),
        'token-failure-custody-inventory':dict(identical=True,copies={'/fixture/one':dict(files=capsule),'/fixture/two':dict(files=copy.deepcopy(capsule))}),
        'collection-acceptance':collection,'collection-disposition':dict(remote_files={'launch/fixture':meta('collection')}),
        'collection-failure-disposition':dict(remote_files={'launch/fixture':meta('old collection failure')}),
        'freeze':dict(stage_closure=dict(stages=dict(token=dict(source_names=[B.TOKEN_ENTRY]))))}
    return m,pins

class NewHistory(unittest.TestCase):
    def setUp(self):
        self.out=EVIDENCE/self._testMethodName;self.out.mkdir()
        self.m,self.pins=fixture()
        (self.out/'fixture.json').write_bytes(B.raw(dict(artificial=True,metadata=self.m,pins=self.pins)))
    def test_failed_token_complete_join_and_unknown_charge_rejection(self):
        result=B.failed_token_transition(self.m,self.pins)
        self.assertIn(B.FAILED_TOKEN_ROOT,result['preserved_roots'])
        for label,mutate in (
            ('unknown work',lambda m:m['token-no-learnedwork-proof'].__setitem__('unknown_return_attempts',None)),
            ('lost failure charge',lambda m:m['token-failure-baseline'].__setitem__('closed_cpu_seconds',126)),
            ('lost old journal',lambda m:m['token-failure-baseline']['journal_pins'].pop('/fixture/journal0')),
            ('wrong original root',lambda m:m['failed-token-packet'].__setitem__('remote_root',B.CPU_ROOT)),
            ('missing raw',lambda m:m['token-failure-custody-inventory']['copies']['/fixture/two']['files'].pop('remote/launch/preparation/failure.json'))):
            m=copy.deepcopy(self.m);mutate(m)
            with self.subTest(label=label),self.assertRaises((ValueError,KeyError)):B.failed_token_transition(m,self.pins)
    def test_accepted_token44_preserves_failed43_and_cost(self):
        old=self.m['token-failure-baseline'];job='99999999';new=row(job,400)
        result=dict(intent_sha256=h('new intent'));journals={**old['journal_pins'],B.CPU_ROOT+'/submission.intent.json':result['intent_sha256'],B.CPU_ROOT+'/submission.result.json':self.pins['cpu-result']}
        base=dict(terminal_jobs=44,submission_journals=90,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,
            input_pins=dict(predecessor=self.pins['token-failure-baseline'],acceptance=self.pins['cpu-acceptance'],disposition=self.pins['cpu-disposition']),
            related_accounting=old['related_accounting']+[new],journal_pins=journals,closed_cpu_seconds=1713,closed_gpu_slice_seconds=0,cpu_admission_seconds=5313)
        obs=dict(parent_accounting=new,journal_pins=journals)
        disp=dict(allocation=dict(resource_contract_matched=True,errors=[],allocated_cpus=3,elapsed_seconds=400,gpu_count=0,gpu_slice_seconds=0,allocated_cpu_seconds=1200),closed_cpu_seconds=1713,closed_gpu_slice_seconds=0)
        stage=dict(cpus=3,wall_seconds=600)
        B.accepted_token_transition(base,old,self.pins,job,result,obs,disp,stage)
        for label,mutate in [('dropped failure',lambda b:b['related_accounting'].pop(42)),('wrong predecessor',lambda b:b['input_pins'].__setitem__('predecessor',self.pins['previous-baseline'])),('zero failed charge',lambda b:b.__setitem__('closed_cpu_seconds',1326))]:
            changed=copy.deepcopy(base);mutate(changed)
            with self.subTest(label=label),self.assertRaises(ValueError):B.accepted_token_transition(changed,old,self.pins,job,result,obs,disp,stage)
        (self.out/'accepted-toy.json').write_bytes(B.raw(base))
    def test_token_science_equality_allows_only_declared_changes(self):
        old=self.m['failed-token-stage'];new=copy.deepcopy(old);new['deadlines']=B.TOKEN_DEADLINES;new['model_directory']=B.CPU_ROOT+'/tokenizer'
        B.token_science_equality(old,new)
        for field in ('files','collection'):
            wrong=copy.deepcopy(new);wrong[field]['changed']='fixture'
            with self.subTest(field=field),self.assertRaises(ValueError):B.token_science_equality(old,wrong)
        wrong=copy.deepcopy(new);wrong['deadlines']=dict(B.TOKEN_DEADLINES,work=551)
        with self.assertRaises(ValueError):B.token_science_equality(old,wrong)
    def test_producer_contract_and_frozen_training_code_alignment(self):
        producer=ast.parse((HERE/'predecessors/token02-build-interface.py').read_text())
        current=ast.parse((HERE/'build_packet.py').read_text());old=ast.parse((HERE/'predecessors/core/build_packet.py').read_text())
        funcs=lambda tree:{n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(funcs(current)['make_packet'],funcs(old)['make_packet'])
        self.assertEqual((HERE/'monitor_once.py').read_bytes(),(HERE/'predecessors/core/monitor_once.py').read_bytes())
        self.assertEqual((HERE/'launch_once.py').read_text().split("REMOTE_COMMON=r",1)[1],(HERE/'predecessors/core/launch_once.py').read_text().split("REMOTE_COMMON=r",1)[1])
        # Producer names and exact authority object are independently read from its shared source; no producer execution.
        producer_names=next(ast.literal_eval(n.value) for n in producer.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOKEN_RECOVERY_METADATA' for t in n.targets))
        self.assertEqual(set(producer_names),set(B.TOKEN_METADATA)-{'collection-failure-disposition'})
        fn=next(n for n in producer.body if isinstance(n,ast.FunctionDef) and n.name=='token_operational_binding')
        node=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
        aliases=dict(self.pins,baseline=self.pins['previous-baseline'])
        scope=dict(pins=aliases,FREEZE_SHA=B.FREEZE_SHA,SOURCE_COMMIT=B.SOURCE_COMMIT,ENTRY_SHA=B.TOKEN_ENTRY_SHA,
            FAILED_TOKEN_JOB=B.FAILED_TOKEN_JOB,FAILED_TOKEN_ROOT=B.FAILED_TOKEN_ROOT,ROOT=B.CPU_ROOT,
            ORIGINAL_CLAIM_SHA=B.ORIGINAL_CLAIM_SHA,ORIGINAL_DEADLINES=B.TOKEN_ORIGINAL_DEADLINES,DEADLINES=B.TOKEN_DEADLINES,RESOURCES=B.TOKEN_RESOURCES)
        expected=eval(compile(ast.Expression(node.value),'<producer-public-metadata-expression>','eval'),scope)
        m=copy.deepcopy(self.m);m['token-operational-amendment']=expected
        pins=dict(self.pins,**{'token-diagnosis':'d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63'})
        # Regenerate expected diagnosis member to match independently declared fixture pin.
        m['token-operational-amendment']['diagnosis_sha256']=pins['token-diagnosis']
        supplied={n:b'fixture independently supplied text' for n in B.TEXT_METADATA}
        B.token_operational_binding(supplied,m,pins)
        m['token-operational-amendment']['deadlines']=dict(B.TOKEN_DEADLINES,hard=591)
        with self.assertRaises(ValueError):B.token_operational_binding(supplied,m,pins)
        self.assertEqual(B.ROOT,'/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01')

if __name__=='__main__':unittest.main(verbosity=2)
