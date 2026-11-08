"""New proposed-recovery metadata tests; no main/project/tokenizer/remote execution."""
import ast,copy,importlib.util,json,os,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('proposed_token02',HERE/'build_packet.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
EVIDENCE=Path(os.environ['TOKEN02_FIXTURES']).resolve()
def fixture():
 pins={n:str(i+1).zfill(64) for i,n in enumerate(B.METADATA)}
 pins.update({'failed-token-packet':B.FAILED_TOKEN_PACKET_SHA,'failed-token-stage':B.FAILED_TOKEN_STAGE_SHA,'failed-token-result':B.FAILED_TOKEN_RESULT_SHA,'freeze':B.FREEZE_SHA,'token-diagnosis':'d4cb918b048b9fdc0dc843dc303ca4bbf436094b697fc8833e9b83114afbfb63'})
 # These are internally fabricated unit-boundary values; not actual receipts or byte-authenticated admission.
 oldrows=[[str(100+i),'ace_llm_fixture','COMPLETED','0:0','2','1','ucb736_asc2','cpu=2,node=1,mem=7680M'] for i in range(42)]
 oldjp={f'/fabricated/{i}':str(i).zfill(64) for i in range(86)}
 old=dict(terminal_jobs=42,submission_journals=86,related_accounting=oldrows,journal_pins=oldjp,closed_cpu_seconds=84,closed_gpu_slice_seconds=0)
 current=[B.FAILED_TOKEN_JOB,'ace_llm_scientific_token01','FAILED','1:0','3','129','ucb736_asc2','billing=3,cpu=3,mem=11520M,node=1']
 result=dict(job_id=B.FAILED_TOKEN_JOB,returncode=0,ambiguous=False,intent_sha256='a'*64)
 jp={**oldjp,B.FAILED_TOKEN_ROOT+'/submission.intent.json':result['intent_sha256'],B.FAILED_TOKEN_ROOT+'/submission.result.json':pins['failed-token-result']}
 base=dict(terminal_jobs=43,submission_journals=88,related_accounting=oldrows+[current],journal_pins=jp,closed_cpu_seconds=471,closed_gpu_slice_seconds=0,cpu_admission_seconds=4071,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,input_pins=dict(predecessor=pins['baseline'],acceptance=pins['token-failure-acceptance'],disposition=pins['token-failure-disposition']))
 c={'opaque_collection_binding':'fabricated'}
 source_names=[f'fixture_{i}.py' for i in range(28)]
 files={B.ENTRY:B.ENTRY_SHA,**{f'opaque_{i}':'b'*64 for i in range(33)}}
 stage=dict(source_commit=B.SOURCE_COMMIT,resources=B.RESOURCES,deadlines=B.ORIGINAL_DEADLINES,model_directory=B.FAILED_TOKEN_ROOT+'/tokenizer',collection=c,files=files,source_names=source_names)
 packet=dict(remote_root=B.FAILED_TOKEN_ROOT,source_commit=B.SOURCE_COMMIT,resources=B.RESOURCES,deadlines=B.ORIGINAL_DEADLINES,baseline_sha256=pins['baseline'],source_pins={f'source{i}':'c'*64 for i in range(32)},new_input_files={'inputs/packet.json':pins['failed-token-stage'],'collection-acceptance.json':pins['collection-acceptance'],'namespace-claim.json':pins['namespace-claim'],'freeze.json':B.FREEZE_SHA})
 rawfiles={f'launch/{n}':dict(bytes=1,sha256=h) for n,h in {**packet['new_input_files'],'packet.json':pins['failed-token-packet'],'submission.intent.json':result['intent_sha256'],'submission.result.json':pins['failed-token-result']}.items()}
 rawfiles.update({'launch/preparation/failure.json':dict(bytes=1,sha256='d'*64),'common/opaque.bin':dict(bytes=1,sha256='e'*64),'failed01/opaque.bin':dict(bytes=1,sha256='f'*64)})
 obs=dict(files=rawfiles,directories=['launch','launch/preparation','common','failed01'],original_hardlink_groups=[],journal_pins=jp,parent_accounting=current,related_accounting=oldrows+[current],upstream_pins={},upstream_errors={},terminal_gate=True)
 d=dict(job_id=B.FAILED_TOKEN_JOB,remote_root=B.FAILED_TOKEN_ROOT,packet_sha256=pins['failed-token-packet'],submission_result_sha256=pins['failed-token-result'],status='terminal_failed_outputs_preserved',custody_bytes_verified=True,current_lineage_verified=True,errors=[],input_pin_mismatches=[],acceptance=False,tokenization_accepted=False,remote_files=rawfiles,raw_files=len(rawfiles),raw_bytes=len(rawfiles),closed_cpu_seconds=471,closed_gpu_slice_seconds=0)
 zeros={k:0 for k in ('scientific_worlds_generated','environment_response_attempts','environment_responses','unknown_return_attempts','tokenizer_calls','model_loads','inference_calls','training_updates','policy_updates')}
 a=dict(schema='ace-scientific-tokenization-failure-acceptance-v1',status='accepted_failed_before_token_worker',failure_accepted=True,tokenization_accepted=False,complete_custody_verified=True,required_findings=0,job_id=B.FAILED_TOKEN_JOB,packet_sha256=pins['failed-token-packet'],disposition_sha256=pins['token-failure-disposition'],freeze_sha256=B.FREEZE_SHA,source_commit=B.SOURCE_COMMIT,source_control_flow_and_traceback_verified=True,two_remote_inventories_matched=True,owned_worker_spawned=False,evidence_pins={n:pins[k] for k,n in [('token-failure-inventory','inventory_transport.json'),('token-failure-readback','inventory_after_transport.json'),('token-failure-custody-inventory','custody-inventory.json')]},**zeros)
 proof=dict(schema='ace-scientific-tokenization-no-learnedwork-proof-v1',status='verified_failed_before_token_worker',failed_job_id=B.FAILED_TOKEN_JOB,failure_acceptance_sha256=pins['token-failure-acceptance'],accounting_sha256=pins['token-failure-baseline'],packet_sha256=pins['failed-token-packet'],disposition_sha256=pins['token-failure-disposition'],freeze_sha256=B.FREEZE_SHA,source_commit=B.SOURCE_COMMIT,collection_acceptance_sha256=pins['collection-acceptance'],proof=dict(owned_worker_not_spawned=True,token_preparation_not_invoked=True),**zeros)
 view=dict(files={**{'remote/'+n:v for n,v in rawfiles.items()},**{n:dict(bytes=1,sha256=pins[k]) for n,k in [('disposition.json','token-failure-disposition'),('inventory_transport.json','token-failure-inventory'),('inventory_after_transport.json','token-failure-readback')]}},directories=[])
 m={'baseline':old,'token-failure-baseline':base,'token-failure-acceptance':a,'token-no-learnedwork-proof':proof,'token-failure-disposition':d,'failed-token-packet':packet,'failed-token-stage':stage,'failed-token-result':result,'token-failure-inventory':dict(returncode=0,stdout=json.dumps(obs)),'token-failure-readback':dict(returncode=0,stdout=json.dumps(obs)),'token-failure-custody-inventory':dict(identical=True,copies={'/fabricated/a':view,'/fabricated/b':copy.deepcopy(view)}),'collection-acceptance':dict(collection=c),'collection-disposition':dict(remote_files={'launch/opaque.bin':dict(bytes=1,sha256='e'*64)}),'failure-disposition':dict(remote_files={'launch/opaque.bin':dict(bytes=1,sha256='f'*64)}),'freeze':dict(stage_closure=dict(stages=dict(token=dict(source_names=source_names))))}
 return m,pins

class Token02(unittest.TestCase):
 def test_startup_and_root_only_on_original_saved_metadata(self):
  old=json.loads((HERE/'predecessors/inputs/packet.json').read_text());new=copy.deepcopy(old)
  new['deadlines']=B.DEADLINES;new['model_directory']=B.ROOT+'/tokenizer'
  receipt=B.token_science_equality(old,new);self.assertTrue(receipt['scientific_inputs_unchanged'])
  self.assertEqual(len(old['source_names']),28);self.assertEqual(len(old['files']),34)
  for field,value in [('source_commit','0'*40),('resources',{**old['resources'],'cpus':4}),('collection',{})]:
   bad=copy.deepcopy(new);bad[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):B.token_science_equality(old,bad)
  bad=copy.deepcopy(new);bad['deadlines']={**B.DEADLINES,'work':551}
  with self.assertRaises(ValueError):B.token_science_equality(old,bad)
  (EVIDENCE/'equality.json').write_bytes(B.raw(receipt))
 def test_actual_boundary_schema_keeps_all43rows88journals_and_failed_charge(self):
  m,pins=fixture();result=B.failed_token_transition(m,pins)
  self.assertEqual(set(result['preserved_roots']),{B.FAILED_TOKEN_ROOT,B.COLLECTION_ROOT,B.FAILED_COLLECTION_ROOT})
  for case in ('cost','worker','proof','prior'):
   bad=copy.deepcopy(m)
   if case=='cost':bad['token-failure-baseline']['closed_cpu_seconds']-=387
   elif case=='worker':bad['token-failure-acceptance']['owned_worker_spawned']=True
   elif case=='proof':bad['token-no-learnedwork-proof']['tokenizer_calls']=None
   else:bad['token-failure-baseline']['related_accounting'][0][5]='2'
   with self.subTest(case=case),self.assertRaises(ValueError):B.failed_token_transition(bad,pins)
  (EVIDENCE/'fabricated-metadata.json').write_bytes(B.raw(dict(fabricated_only=True,metadata=m,pins=pins)))
 def test_exact_amendment_no_implicit_other_changes_or_diagnosis(self):
  m,pins=fixture();tree=ast.parse((HERE/'build_packet.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='token_operational_binding');a=next(n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
  # Only evaluate the isolated proposed metadata expression, no builder main or actual receipts.
  namespace={**vars(B),'pins':pins};exec(compile(ast.Module(body=[a],type_ignores=[]),'fabricated_amendment_fields','exec'),namespace)
  expected=namespace['expected'];m['token-operational-amendment']=expected
  supplied={k:b'fabricated independent review' for k in ('token-operational-review','token-operational-adoption-review','token-operational-guide')}
  B.token_operational_binding(supplied,m,pins)
  for key,value in [('source_commit','0'*40),('resources',{**B.RESOURCES,'wall_seconds':601}),('deadlines',{**B.DEADLINES,'cleanup':580}),('status','proposed')]:
   bad=copy.deepcopy(m);bad['token-operational-amendment'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):B.token_operational_binding(supplied,bad,pins)
  badpins=dict(pins,**{'token-diagnosis':'0'*64})
  with self.assertRaises(ValueError):B.token_operational_binding(supplied,m,badpins)
  (EVIDENCE/'fabricated-amendment-description.json').write_bytes(B.raw(dict(fabricated_only=True,not_actual_authority=True,example=expected)))
 def test_once_only_launcher_reconciles_full_failed_roots_before_both_actions(self):
  source=(HERE/'launch_once.py').read_text();tree=ast.parse(source)
  remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets))
  compile(remote,'embedded_observer','exec')
  self.assertIn("prior_receipts={path:verify_tree(dict(root=path,files=files,directories=C['prior_directories'][path])) for path,files in C['prior_roots'].items()}",remote)
  self.assertIn("baseline=m['token-failure-baseline']",source)
  self.assertIn("{C['failed_token_root']}",source)
  strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
  submit=[s for s in strings if "subprocess.run(['sbatch'" in s]
  self.assertEqual(len(submit),1);self.assertLess(submit[0].index('fresh=reconcile();verify()'),submit[0].index("write(root/'submission.intent.json'"))
  self.assertIn('ambiguous',submit[0]);self.assertIn("'ace-delivery-controller.lock','ace-foundation-controller.lock'",source)
  self.assertNotIn("'pull_terminal.py'",(HERE/'build_packet.py').read_text())
  for name in ('build_packet.py','monitor_once.py'):compile(ast.parse((HERE/name).read_text()),name,'exec')

 def test_preserved_empty_directories_are_bound_in_remote_preflight(self):
  import hashlib,stat
  tree=ast.parse((HERE/'launch_once.py').read_text())
  source=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets))
  parsed=ast.parse(source);function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='verify_tree')
  ns=dict(pathlib=__import__('pathlib'),os=os,stat=stat,hashlib=hashlib,need=B.need)
  exec(compile(ast.Module(body=[function],type_ignores=[]),'fabricated_directory_preflight','exec'),ns)
  root=EVIDENCE/'root';root.mkdir();(root/'empty').mkdir();(root/'opaque').write_bytes(b'fabricated')
  c=dict(root=str(root),files={'opaque':hashlib.sha256(b'fabricated').hexdigest()},directories=['empty'])
  self.assertTrue(ns['verify_tree'](c)['verified'])
  (root/'empty').rmdir()
  with self.assertRaisesRegex(ValueError,'directory roster'):ns['verify_tree'](c)
  m,pins=fixture();result=B.failed_token_transition(m,pins)
  self.assertEqual(result['preserved_directories'][B.FAILED_TOKEN_ROOT],['preparation'])

if __name__=='__main__':unittest.main(verbosity=2)
