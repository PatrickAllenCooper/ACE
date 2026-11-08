"""New CPU coordinator joins only; imported stdlib authoring, never main/project/network."""
import ast,copy,hashlib,importlib.util,json,os,unittest,pathlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_CPU_HISTORY_EVIDENCE'])
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load('new_cpu_history',HERE/'build_packet.py')
C=load('new_cpu_collector',HERE/'pull_terminal.py')
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
        deadlines=B.TOKEN_ORIGINAL_DEADLINES,baseline_sha256=pins['collection-baseline'],new_input_files={
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
        input_pins=dict(predecessor=pins['collection-baseline'],acceptance=pins['token-failure-acceptance'],disposition=pins['token-failure-disposition']),
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
    m={'collection-baseline':old,'token-failure-baseline':base,'failed-token-packet':packet,'failed-token-stage':stage,
        'failed-token-result':result,'token-failure-acceptance':a,'token-no-learnedwork-proof':proof,
        'token-failure-disposition':disp,'token-failure-inventory':transport,'token-failure-readback':copy.deepcopy(transport),
        'token-failure-custody-inventory':dict(identical=True,copies={'/fixture/one':dict(files=capsule),'/fixture/two':dict(files=copy.deepcopy(capsule))}),
        'collection-acceptance':collection,'collection-disposition':dict(remote_files={'launch/fixture':meta('collection')}),
        'collection-failure-disposition':dict(remote_files={'launch/fixture':meta('old collection failure')}),
        'freeze':dict(stage_closure=dict(stages=dict(token=dict(source_names=[B.TOKEN_ENTRY]))))}
    return m,pins

def full_fixture():
 m,pins=fixture();pins['token-diagnosis']='d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63'
 supplied={n:b'fabricated independent review' for n in B.TEXT_METADATA}
 m['token-operational-amendment']=dict(schema='ace-scientific-tokenization-operational-amendment-v1',status='adopted',stage='tokenize',recovery_id='token02',
  freeze_sha256=B.FREEZE_SHA,source_commit=B.SOURCE_COMMIT,entry_sha256=B.TOKEN_ENTRY_SHA,failed_job_id=B.FAILED_TOKEN_JOB,
  failed_packet_sha256=pins['failed-token-packet'],failed_stage_packet_sha256=pins['failed-token-stage'],failed_root=B.FAILED_TOKEN_ROOT,recovery_root=B.TOKEN_ROOT,
  failure_acceptance_sha256=pins['token-failure-acceptance'],no_learnedwork_proof_sha256=pins['token-no-learnedwork-proof'],
  failure_disposition_sha256=pins['token-failure-disposition'],failure_inventory_sha256=pins['token-failure-inventory'],failure_readback_sha256=pins['token-failure-readback'],
  failure_custody_inventory_sha256=pins['token-failure-custody-inventory'],baseline_sha256=pins['token-failure-baseline'],collection_acceptance_sha256=pins['collection-acceptance'],
  collection_baseline_sha256=pins['collection-baseline'],namespace_admission_sha256=pins['namespace-admission'],namespace_claim_sha256=pins['namespace-claim'],
  original_namespace_claim_sha256=B.ORIGINAL_CLAIM_SHA,original_deadlines=B.TOKEN_ORIGINAL_DEADLINES,deadlines=B.TOKEN_DEADLINES,resources=B.TOKEN_RESOURCES,
  review_sha256=pins['token-operational-review'],guide_sha256=pins['token-operational-guide'],diagnosis_sha256=pins['token-diagnosis'],required_findings=0,
  scientific_inputs_unchanged=True,other_stages_unchanged=True,one_submission_only=True,automatic_retry_allowed=False,replacement_worlds_allowed=False)
 recovery=B.token_operational_binding(supplied,m,pins)
 m['cpu-packet']=copy.deepcopy(m['failed-token-stage']);m['cpu-packet'].update(deadlines=B.TOKEN_DEADLINES,model_directory=B.TOKEN_ROOT+'/tokenizer')
 outer=dict(remote_root=B.TOKEN_ROOT,source_commit=B.SOURCE_COMMIT,baseline_sha256=pins['token-failure-baseline'],
  resources=B.TOKEN_RESOURCES,deadlines=B.TOKEN_DEADLINES,token_recovery=recovery,new_input_files={
   'inputs/packet.json':pins['cpu-packet'],'baseline.json':pins['collection-baseline'],
   **{('failure-disposition.json' if n=='collection-failure-disposition' else B.metadata_name(n)):pins[n] for n in B.TOKEN_METADATA}})
 m['cpu-launch-packet']=outer;m['cpu-preparation-acceptance']=dict(job_id='90001',cpu_packet_sha256=pins['cpu-packet'])
 m['gpu-training-acceptance']=dict(job_id='90002');m['expected-training']=dict(collection_acceptance_sha256=pins['collection-acceptance'])
 result=dict(job_id='90001',returncode=0,ambiguous=False,intent_sha256=h('TOKEN02 intent'));m['cpu-result']=result
 old=m['token-failure-baseline'];newrow=row('90001',21)
 journals={**old['journal_pins'],B.TOKEN_ROOT+'/submission.intent.json':result['intent_sha256'],B.TOKEN_ROOT+'/submission.result.json':pins['cpu-result']}
 m['previous-baseline']=dict(terminal_jobs=44,submission_journals=90,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,
  cpu_cap_seconds=57600,gpu_cap_seconds=7200,input_pins=dict(predecessor=pins['token-failure-baseline'],acceptance=pins['cpu-preparation-acceptance'],disposition=pins['cpu-disposition']),
  related_accounting=old['related_accounting']+[newrow],journal_pins=journals,closed_cpu_seconds=576,closed_gpu_slice_seconds=0,cpu_admission_seconds=4176)
 files={'launch/'+n:dict(bytes=1,sha256=v) for n,v in {**outer['new_input_files'],'packet.json':pins['cpu-launch-packet'],
  'submission.intent.json':result['intent_sha256'],'submission.result.json':pins['cpu-result']}.items()}
 prior=json.loads(m['token-failure-inventory']['stdout'])
 for n,v in prior['files'].items():files[('opaque-failed/'+n[7:]) if n.startswith('launch/') else n]=v
 dirs=['launch','launch/inputs']+[('opaque-failed'+n[6:]) if n=='launch' or n.startswith('launch/') else n for n in prior['directories']]
 inv=dict(terminal_gate=True,root_present=True,parent_accounting=newrow,files=files,directories=dirs,
  journal_pins=journals,related_accounting=m['previous-baseline']['related_accounting'])
 m['cpu-inventory']=dict(returncode=0,stdout=json.dumps(inv),python_source='ROOT='+repr(B.TOKEN_ROOT)+'\nJOB="90001"\n')
 m['cpu-disposition']=dict(job_id='90001',remote_root=B.TOKEN_ROOT,packet_sha256=pins['cpu-launch-packet'],submission_result_sha256=pins['cpu-result'],
  custody_bytes_verified=True,current_lineage_verified=True,errors=[],input_pin_mismatches=[],exact_prior_accounting_matched=True,
  exact_prior_and_current_journals_matched=True,observed_related_jobs=44,observed_submission_journals=90,raw_files=len(files),raw_bytes=len(files),
  allocation=dict(resource_contract_matched=True,errors=[],allocated_cpus=3,elapsed_seconds=21,gpu_count=0,gpu_slice_seconds=0,allocated_cpu_seconds=63),
  closed_cpu_seconds=576,closed_gpu_slice_seconds=0)
 aliases={'collection-baseline':'previous-baseline','cpu-preparation-acceptance':'cpu-acceptance'}
 names=B.TOKEN_METADATA+('collection-baseline','collection-acceptance','collection-disposition','cpu-preparation-acceptance','cpu-packet','cpu-launch-packet','cpu-result','cpu-disposition','cpu-inventory')
 m['training-packet']=dict(new_input_files={**{B.metadata_name(aliases.get(n,n)):pins[n] for n in names},'baseline.json':pins['previous-baseline']},
  token_recovery=recovery,failed_token_root=B.FAILED_TOKEN_ROOT,failed_token_job_id=B.FAILED_TOKEN_JOB)
 def rename(n):return 'cpu'+n[6:] if n=='launch' or n.startswith('launch/') else 'different-failed'+n[13:] if n=='opaque-failed' or n.startswith('opaque-failed/') else n
 training={rename(n):v for n,v in files.items()};m['training-inventory']=dict(stdout=json.dumps(dict(files=training,directories=[rename(n) for n in dirs])))
 return supplied,m,pins,training

class NewCPUHistory(unittest.TestCase):
 def test_full_history_join_rejects_lost_failure_and_nested_authority_drift(self):
  supplied,m,pins,files=full_fixture()
  (EVIDENCE/'fabricated-history.json').write_bytes(B.raw(dict(metadata=m,pins=pins,training_files=files)))
  got=B.campaign_token_history(supplied,m,pins,files)
  self.assertIn(B.FAILED_TOKEN_ROOT+'/preparation/failure.json',got['upstream_pins'])
  self.assertIn(B.TOKEN_ROOT,got['preserved_directories'])
  for mutate in ('nested','cost','unknown','source','missing','empty-dir','capture'):
   mm=copy.deepcopy(m);ff=copy.deepcopy(files)
   if mutate=='nested':mm['training-packet']['token_recovery']['failure_acceptance_sha256']=h('wrong')
   elif mutate=='cost':mm['token-failure-baseline']['closed_cpu_seconds']-=387
   elif mutate=='unknown':mm['token-no-learnedwork-proof']['training_updates']=None
   elif mutate=='source':mm['cpu-packet']['files'][B.TOKEN_ENTRY]=h('drift')
   elif mutate=='missing':del ff['different-failed/preparation/failure.json']
   elif mutate=='capture':mm['training-packet']['new_input_files']['token-failure-baseline.json']=h('wrong')
   else:
    ti=json.loads(mm['training-inventory']['stdout']);ti['directories'].append('different-failed/added-empty');mm['training-inventory']['stdout']=json.dumps(ti)
   with self.subTest(mutate=mutate),self.assertRaises((ValueError,KeyError)):B.campaign_token_history(supplied,mm,pins,ff)
 def test_collector46_94_retains_failed43_and_rejects_count_only_substitute(self):
  _,m,pins,_=full_fixture();base=copy.deepcopy(m['previous-baseline'])
  train=['90002','fixture','COMPLETED','0:0','4','17','ucb736_asc2','cpu=4,mem=15360M,node=1,gres/gpu:a100_3g.20gb=1']
  base['related_accounting'].append(train);base['journal_pins'].update({C.TRAIN_ROOT+'/submission.intent.json':h('traini'),C.TRAIN_ROOT+'/submission.result.json':h('trainr')})
  current=['90003','fixture','FAILED','1:0','2','19','ucb736_asc2','cpu=2,mem=7680M,node=1']
  obs=dict(related_accounting=base['related_accounting']+[current],parent_accounting=current,
   journal_pins={**base['journal_pins'],C.ROOT+'/submission.intent.json':h('ci'),C.ROOT+'/submission.result.json':h('cr')})
  self.assertEqual(C.accounting_extension(obs,base,'90003',h('ci'),h('cr')),(True,True))
  self.assertEqual(C.cost(obs['related_accounting']),(682,17))
  bad=copy.deepcopy(obs);bad['related_accounting'][42][5]='0'
  self.assertFalse(C.accounting_extension(bad,base,'90003',h('ci'),h('cr'))[0])
  bad=copy.deepcopy(obs);bad['journal_pins'][B.FAILED_TOKEN_ROOT+'/submission.result.json']=h('changed')
  self.assertFalse(C.accounting_extension(bad,base,'90003',h('ci'),h('cr'))[1])
  roots={'arbitrary-old-label/packet.json':meta('original'),'other/packet.json':meta('other')}
  self.assertEqual(C.failed_token_label(roots,h('original')),'arbitrary-old-label')
  for v in ({'other/packet.json':meta('other')},{**roots,'duplicate/packet.json':meta('original')}):
   with self.assertRaises(ValueError):C.failed_token_label(v,h('original'))
 def test_frozen_child_monitor_and_double_launch_reconciliation_preserved(self):
  def tree(path):return ast.parse(path.read_text())
  def fn(path,name):return ast.dump(next(n for n in tree(path).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  self.assertEqual(fn(HERE/'build_packet.py','make_packet'),fn(HERE/'predecessor/build_packet.py','make_packet'))
  self.assertEqual((HERE/'monitor_once.py').read_bytes(),(HERE/'predecessor/monitor_once.py').read_bytes())
  self.assertEqual(B.RESOURCES,dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,requeue=False,wall_seconds=6000))
  self.assertEqual(B.DEADLINES,dict(startup=180,work=5940,hard=5980));self.assertEqual(C.BUILDER_SHA,h_bytes((HERE/'build_packet.py').read_bytes()))
  for n in ('build_packet.py','launch_once.py','monitor_once.py','pull_terminal.py'):tree(HERE/n)
  launch=tree(HERE/'launch_once.py');remote=next(ast.literal_eval(n.value) for n in launch.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets))
  ast.parse(remote);ast.parse(C.REMOTE_OBSERVATION)
  self.assertIn("C['preserved_directories']",remote);self.assertIn("C['original_inputs']",remote)
  self.assertIn('r=reconcile()', (HERE/'launch_once.py').read_text())
  self.assertIn('fresh=reconcile();verify()', (HERE/'launch_once.py').read_text())
  reconcile=next(n for n in ast.parse(remote).body if isinstance(n,ast.FunctionDef) and n.name=='reconcile')
  block=next(n for n in reconcile.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['base','expected_dirs'])
  code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'new-root-readback','exec')
  root=EVIDENCE/'fabricated-readback';root.mkdir();(root/'empty').mkdir();(root/'failure.json').write_bytes(b'original-failure')
  scope=dict(pathlib=pathlib,C=dict(preserved_directories={str(root):['empty']},original_inputs={str(root/'failure.json'):h_bytes(b'original-failure')}),need=B.need,sha=h_bytes,read=lambda path,**kwargs:Path(path).read_bytes())
  exec(code,scope)
  (root/'failure.json').write_bytes(b'changed')
  with self.assertRaises(ValueError):exec(code,scope)
  self.assertIn('failed_token_present',C.REMOTE_OBSERVATION)
  self.assertEqual(len(B.METADATA),len(set(B.METADATA)))
def h_bytes(v):return hashlib.sha256(v).hexdigest()
if __name__=='__main__':unittest.main()
