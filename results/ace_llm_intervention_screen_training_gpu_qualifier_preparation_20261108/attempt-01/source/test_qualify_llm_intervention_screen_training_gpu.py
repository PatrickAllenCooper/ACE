"""NEW stdlib toy qualifier controls: no actual acceptance, runtime or models.

Only isolated in-memory qualifier modules have their independent constants
replaced with fabricated file digests. Production qualifier bytes are untouched.
Every toy input/rejection is retained under ACE_SCREEN_QUALIFIER_TEST_OUTPUT.
"""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import types
import unittest

SOURCE=Path(__file__).with_name('qualify_llm_intervention_screen_training_gpu.py')
PREPARATION_SOURCES=(
    'llm_intervention_contract.py','llm_intervention_task.py','llm_intervention_numerical_policy.py',
    'llm_intervention_scm.py','llm_intervention_episode.py','llm_intervention_supervision.py',
    'llm_intervention_training_plan.py','llm_intervention_training_engine.py',
    'qualify_llm_intervention_checkpoint_gpu.py','llm_intervention_gradient_probe.py',
    'llm_intervention_learning.py','llm_intervention_action_scoring.py','llm_intervention_checkpointing.py',
    'llm_intervention_canonical_scoring.py','llm_intervention_precision.py','llm_intervention_language_policy.py',
    'llm_intervention_grammar.py','llm_intervention_training_exposure.py')
LONG='collider-uniform-noise0.25-long_spelling-last_all_actions'
TERMINAL='collider-uniform-noise0.25-long_spelling-full_history_terminal'

def raw(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(raw(value)[:-1]).hexdigest()
def sha(value):return hashlib.sha256(value).hexdigest()


def load_qualifier(source,path):
    allowed={'__future__','argparse','hashlib','importlib.abc','importlib.util','json','math','os','pathlib',
             'resource','signal','stat','subprocess','sys','time','traceback','types'}
    original=builtins.__import__
    def only_stdlib(name,*args,**kwargs):
        if name not in allowed:raise AssertionError('heavy/project import before authentication: '+name)
        return original(name,*args,**kwargs)
    module=types.ModuleType('isolated_toy_screen_qualifier')
    module.__file__=str(path)
    module.__dict__['__builtins__']={**vars(builtins),'__import__':only_stdlib}
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


class ToyPacket:
    """Invented source bytes and data, never a substitute for real CPU acceptance."""
    def __init__(self,root,source):
        self.root=root;root.mkdir(parents=True,exist_ok=False)
        self.g=load_qualifier(source,root/SOURCE.name);g=self.g
        self.executed=[];self.captured=[]
        self.data={n:b'# fabricated source, not an authenticated real helper\n' for n in g.SOURCE_NAMES}
        self.data[g.ENTRY]=source
        self.resources={'cpus':4,'max_allocated_cpus':8,'memory_mib':12288,'gpus':1,'wall_seconds':600}
        self.data[g.GPU]=('RESOURCES = '+repr(self.resources)+'\n').encode()
        self.data['llm_intervention_model_service.py']=b'''import json,hashlib
MODEL_NAMES={'model.fake','tokenizer.fake'}
def decode(raw):return json.loads(raw)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
'''
        for name in [g.CPU,g.DIAGNOSTIC,g.SUPERVISOR]:self.data[name]=b'TOY_HELPER=True\n'
        self.data['llm_intervention_runtime_archive.py']=b'TOY_ARCHIVE=True\n'
        self.capacity={'archive':{'path':'/toy/runtime.tar','sha256':'a'*64},
            'manifest':{'path':'/toy/manifest.json','sha256':'b'*64},'model_directory':'/toy/model',
            'model_files':{'model.fake':'c'*64,'tokenizer.fake':'d'*64},
            'runtime':{'python':'toy-python'},'runtime_source':{'sha256':'e'*64}}
        self.data['capacity-packet.json']=raw(self.capacity)
        commands={'{"action_id":"observe_all"}':[1,151645],'{"stop":true}':[2,151645]}
        envelopes=[{'state':{'terminal':False,'toy':'nonterminal'}},{'state':{'terminal':True,'toy':'terminal'}}]
        rows=[]
        for i,name in enumerate([LONG,TERMINAL]):
            rows.append({'fixture_id':name,'prompt_ids':[40+i,41+i],
                         'commands':commands if i==0 else {'{"stop":true}':[2,151645]},
                         'metadata':{'envelope_sha256':digest(envelopes[i]),'toy_metadata':i}})
        self.data['tokenization.json']=raw({'rows':rows})
        accepted={'job_id':'35391340','status':'accepted','required_findings':0,'capacity_qualified':True,
            'complete_custody_verified':True,'runtime_qualified_for_this_cpu_execution':True,
            'owned_process_cleanup_verified':True,'packet_sha256':sha(self.data['capacity-packet.json']),
            'tokenization_sha256':sha(self.data['tokenization.json']),
            'training_execution_qualified':False,'scientific_qualified':False}
        self.data['cpu-acceptance.json']=raw(accepted)
        records=[]
        for i in range(4):
            j=1 if i==2 else 0;row=rows[j]
            original=raw({'scope':'artificial','envelope':envelopes[j]}).decode()
            records.append({'record_id':'shape-'+str(i),'scope':'artificial','raw_record_utf8':original,
                'record_sha256':sha(original.encode()),'prompt_ids':row['prompt_ids'],
                'commands':row['commands'],'token_metadata':row['metadata'],
                'tokenizer_sha256':digest(self.capacity['model_files']),'pad_token_id':151643,'eos_token_id':151645})
        plan={'records':records,'execution_kind':'qwen_screen_singletons'}
        self.prepared={'schema':'ace-screen-training-shape-preparation-v1','scope':'artificial',
            'labels':'fabricated only','training_execution_qualified':False,'scientific_qualified':False,
            'environment_responses':0,'tokenizer_calls':0,'training_plan':plan,'training_plan_sha256':digest(plan),
            'capacity_acceptance_sha256':sha(self.data['cpu-acceptance.json']),
            'capacity_packet_sha256':sha(self.data['capacity-packet.json']),
            'tokenization_sha256':sha(self.data['tokenization.json']),
            'tokenizer_sha256':digest(self.capacity['model_files']),
            'source_commit':'a'*40,'source_sha256':{n:sha(self.data[n]) for n in PREPARATION_SOURCES}}
        self.data['prepared.json']=raw(self.prepared)
        # Explicit test-only module-local substitutions; never touch source or
        # actual acceptance files. The matching map must be replaced as well.
        for attr,name in [('GPU_SHA',g.GPU),('SUPERVISOR_SHA',g.SUPERVISOR),('CPU_SHA',g.CPU),
                          ('DIAGNOSTIC_SHA',g.DIAGNOSTIC),('CPU_ACCEPTANCE_SHA','cpu-acceptance.json'),
                          ('PREPARED_SHA','prepared.json'),('CAPACITY_PACKET_SHA','capacity-packet.json'),
                          ('TOKENS_SHA','tokenization.json')]:setattr(g,attr,sha(self.data[name]))
        g.PLAN_SHA=digest(plan)
        g.INDEPENDENT={name:sha(self.data[name]) for name in g.INDEPENDENT}
        self.packet={'schema':g.VERSION,'scope':'artificial','source_commit':'a'*40,
            'source_names':sorted(g.SOURCE_NAMES),'files':{n:sha(b) for n,b in self.data.items()},
            'resources':self.resources,**copy.deepcopy(self.capacity)}
        module=g.module;capture=g.capture
        def logged_module(name,path,data):
            self.executed.append(Path(path).name)
            return module(name,path,data)
        def logged_capture(path,pin,*args):
            result=capture(path,pin,*args);self.captured.append(Path(path).name);return result
        g.module=logged_module;g.capture=logged_capture
        self.flush()
        (root/'TOY-ONLY.json').write_bytes(raw({'actual_acceptance_qualified':False,'models':0,
            'scope':'in-process fabricated pins and helper bytes only','real_qualifier_source_sha256':sha(source)}))

    def flush(self):
        for n,b in self.data.items():(self.root/n).write_bytes(b)
        (self.root/'packet.json').write_bytes(raw(self.packet))

    def auth(self):
        self.flush()
        try:return self.g.authenticate(self.root/'packet.json',sha(raw(self.packet)))
        except BaseException as exc:
            (self.root/'rejection.json').write_bytes(raw({'exception':type(exc).__name__,'message':str(exc),
                'executed_toy_modules':self.executed,'captured_inputs':self.captured}))
            raise

    def repin(self,name,value,independent=False):
        self.data[name]=raw(value);self.packet['files'][name]=sha(self.data[name])
        if independent:
            attrs={'prepared.json':'PREPARED_SHA','cpu-acceptance.json':'CPU_ACCEPTANCE_SHA',
                   'capacity-packet.json':'CAPACITY_PACKET_SHA','tokenization.json':'TOKENS_SHA'}
            setattr(self.g,attrs[name],sha(self.data[name]));self.g.INDEPENDENT[name]=sha(self.data[name])
        self.flush()


class ScreenQualifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=SOURCE.read_bytes();cls.output=Path(os.environ['ACE_SCREEN_QUALIFIER_TEST_OUTPUT'])
        cls.output.mkdir(exist_ok=False)

    def toy(self,label):return ToyPacket(self.output/self._testMethodName/label,self.source)

    def test_unresolved_production_pins_fail_closed_but_explicit_toy_map_can_authenticate(self):
        toy=self.toy('pending');production=load_qualifier(self.source,toy.root/SOURCE.name)
        # Pending constants are intentional, not a test instruction to invent
        # live acceptance. Use a fabricated pin map to isolate this guard.
        production.INDEPENDENT=dict(toy.g.INDEPENDENT,**{'prepared.json':None})
        with self.assertRaisesRegex(ValueError,'unresolved'):
            production.packet_shape(toy.packet,types.SimpleNamespace(RESOURCES=toy.resources))
        toy=self.toy('toy_complete');context=toy.auth()
        self.assertEqual(context[4],toy.prepared)
        self.assertEqual(set(context[2]),set(toy.g.SOURCE_NAMES)|toy.g.EXTRA_DATA)

    def test_exact_source_data_membership_rejects_omissions_extras_and_source_order(self):
        cases=[('missing_worker',lambda t:t.packet['files'].pop(t.g.WORKER)),
               ('missing_gradient_probe',lambda t:t.packet['files'].pop('llm_intervention_gradient_probe.py')),
               ('extra_adapter',lambda t:t.packet['files'].update({'adapter.pt':'a'*64})),
               ('extra_data',lambda t:t.packet['files'].update({'extra.json':'a'*64})),
               ('missing_tokenization',lambda t:t.packet['files'].pop('tokenization.json')),
               ('source_order',lambda t:t.packet['source_names'].reverse()),
               ('scope',lambda t:t.packet.update(scope='policy_train'))]
        for label,change in cases:
            toy=self.toy(label);change(toy)
            with self.subTest(label=label),self.assertRaises(ValueError):toy.auth()
            self.assertEqual(toy.executed,[toy.g.GPU])

    def test_independent_helper_and_input_pins_reject_packet_repinning(self):
        for name in ['prepared.json','cpu-acceptance.json','capacity-packet.json','tokenization.json',
                     'qualify_llm_intervention_service_cpu.py','qualify_llm_intervention_runtime_metadata.py',
                     'supervise_llm_intervention_checkpoint_gpu.py']:
            toy=self.toy(name);toy.data[name]+=b'\n';toy.packet['files'][name]=sha(toy.data[name])
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'independent'):toy.auth()
            self.assertEqual(toy.executed,[toy.g.GPU])
        toy=self.toy('gpu_helper');toy.data[toy.g.GPU]+=b'\n';toy.packet['files'][toy.g.GPU]=sha(toy.data[toy.g.GPU])
        with self.assertRaises(ValueError):toy.auth()
        self.assertEqual(toy.executed,[])

    def test_all_input_bytes_are_captured_before_other_helpers_and_heavy_imports(self):
        toy=self.toy('changed_worker_bytes');toy.data[toy.g.WORKER]+=b'changed without packet pin\n'
        with self.assertRaisesRegex(ValueError,'input pin'):toy.auth()
        self.assertEqual(toy.executed,[toy.g.GPU])
        toy=self.toy('good_order');original=toy.g.module
        def checked(name,path,data):
            if Path(path).name!=toy.g.GPU:
                self.assertTrue(set(toy.packet['files'])<=set(toy.captured))
            return original(name,path,data)
        toy.g.module=checked;toy.auth()
        self.assertNotIn(toy.g.WORKER,toy.executed)
        self.assertNotIn('llm_intervention_service_runtime.py',toy.executed)

    def test_accepted_capacity_runtime_model_inputs_cannot_change(self):
        for key in ['archive','manifest','model_directory','model_files','runtime','runtime_source']:
            toy=self.toy(key);toy.packet[key]={'changed':'not accepted'} if isinstance(toy.packet[key],dict) else '/changed'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'accepted tokenizer/runtime/model'):toy.auth()

    def test_cpu_acceptance_must_be_affirmative_and_bound_to_actual_token_inputs(self):
        changes=[('job_id','another-job'),('status','pending'),('required_findings',1),
            ('capacity_qualified',False),('complete_custody_verified',False),
            ('runtime_qualified_for_this_cpu_execution',False),('owned_process_cleanup_verified',False),
            ('packet_sha256','0'*64),('tokenization_sha256','0'*64),
            ('training_execution_qualified',True),('scientific_qualified',True)]
        for key,value in changes:
            toy=self.toy(key);accepted=json.loads(toy.data['cpu-acceptance.json']);accepted[key]=value
            toy.repin('cpu-acceptance.json',accepted,independent=True)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'tokenizer acceptance'):toy.auth()

    def test_new_prepared_tokens_raw_envelopes_and_sources_cannot_drift(self):
        changes=[('prompt',lambda p:p['training_plan']['records'][0]['prompt_ids'].__setitem__(0,99)),
                 ('command',lambda p:p['training_plan']['records'][0]['commands']['{"stop":true}'].__setitem__(0,99)),
                 ('metadata',lambda p:p['training_plan']['records'][0]['token_metadata'].update(toy_metadata=99)),
                 ('identity',lambda p:p['training_plan']['records'][0].update(record_id='other')),
                 ('tokenizer',lambda p:p['training_plan']['records'][0].update(tokenizer_sha256='0'*64)),
                 ('eos',lambda p:p['training_plan']['records'][0].update(eos_token_id=1)),
                 ('source',lambda p:p['source_sha256'].update({'llm_intervention_training_engine.py':'0'*64}))]
        for label,change in changes:
            toy=self.toy(label);p=copy.deepcopy(toy.prepared);change(p)
            p['training_plan_sha256']=digest(p['training_plan']);toy.g.PLAN_SHA=p['training_plan_sha256']
            toy.repin('prepared.json',p,independent=True)
            with self.subTest(label=label),self.assertRaises(ValueError):toy.auth()
        toy=self.toy('raw_envelope');p=copy.deepcopy(toy.prepared);rec=p['training_plan']['records'][0]
        original=json.loads(rec['raw_record_utf8']);original['envelope']['state']['toy']='changed'
        rec['raw_record_utf8']=raw(original).decode();rec['record_sha256']=sha(rec['raw_record_utf8'].encode())
        p['training_plan_sha256']=digest(p['training_plan']);toy.g.PLAN_SHA=p['training_plan_sha256']
        toy.repin('prepared.json',p,independent=True)
        with self.assertRaisesRegex(ValueError,'association'):toy.auth()

    def test_prepared_source_inventory_cannot_be_emptied_or_extended_to_unrelated_helpers(self):
        for label,change in [('empty',lambda p,t:p.update(source_sha256={})),
                             ('missing',lambda p,t:p['source_sha256'].pop('llm_intervention_gradient_probe.py')),
                             ('extra_helper',lambda p,t:p['source_sha256'].update({t.g.GPU:t.packet['files'][t.g.GPU]}))]:
            toy=self.toy(label);p=copy.deepcopy(toy.prepared);change(p,toy)
            toy.repin('prepared.json',p,independent=True)
            with self.subTest(label=label),self.assertRaises(ValueError):toy.auth()

    def test_plan_pin_is_independent_even_after_prepared_container_is_repinned(self):
        toy=self.toy('plan');p=copy.deepcopy(toy.prepared)
        p['training_plan']['execution_kind']='qwen_gpu09';p['training_plan_sha256']=digest(p['training_plan'])
        toy.repin('prepared.json',p,independent=True)
        with self.assertRaisesRegex(ValueError,'preparation/plan'):toy.auth()

    def test_worker_calls_forward_both_independent_pins_and_model_auth_precedes_imports(self):
        tree=ast.parse(self.source)
        for fn_name,call_name in [('child','run_training'),('child','validate_outputs'),('_validate_worker','validate_outputs')]:
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn_name)
            calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and
                   ((isinstance(n.func,ast.Name) and n.func.id==call_name) or
                    (isinstance(n.func,ast.Attribute) and n.func.attr==call_name))]
            self.assertEqual(len(calls),1)
            kw={k.arg:ast.unparse(k.value) for k in calls[0].keywords}
            self.assertEqual(kw,{'prepared_sha256':'PREPARED_SHA','plan_sha256':'PLAN_SHA'})
        child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='child')
        assignments={n.targets[0].id:n for n in ast.walk(child) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        closure=next(n for n in ast.walk(child) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='import_cpu_closure')
        self.assertLess(assignments['models'].lineno,closure.lineno)
        self.assertIn('service.read_pinned',ast.unparse(assignments['models']))
        worker_import=next(n for n in ast.walk(child) if isinstance(n,ast.ImportFrom) and n.module=='llm_intervention_screen_training_gpu_worker')
        self.assertLess(closure.lineno,worker_import.lineno)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        auth=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='authenticate')
        for n in ast.walk(main):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['child','parent']:
                self.assertLess(auth.lineno,n.lineno)


if __name__=='__main__':unittest.main(verbosity=2)
