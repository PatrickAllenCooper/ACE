"""New fabricated training coordinator metadata joins; no project/model execution."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,unittest
spec=importlib.util.spec_from_file_location('new_training_builder',Path(__file__).with_name('build_packet.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
PUBLIC=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_contract_20261108')
H=lambda text:hashlib.sha256(text.encode()).hexdigest()
class Metadata(unittest.TestCase):
 def fixture(self):
  paths={'cpu-packet-relative':'launch/inputs/packet.json',
   'cpu-prepared-relative':'launch/preparation/session/worker/artifacts/llm-prepared.json',
   'cpu-terminal-relative':'launch/preparation/session/worker/artifacts/terminal.json'}
  supplied={'freeze':(PUBLIC/'candidate03/scientific-freeze.json').read_bytes()}
  for n in ('freeze-review','adoption','namespace-admission','recovery-authority'):
   supplied[n]=(PUBLIC/'adoption02'/(n+('.md' if n=='freeze-review' else '.json'))).read_bytes()
  f=b.decode(supplied['freeze']);ns=b.decode(supplied['namespace-admission']);job='99999999'
  supplied['namespace-claim']=b.raw(dict(freeze_sha256=b.FIXED['freeze'],namespace_admission_sha256=b.FIXED['namespace-admission'],
   world_ids=ns['world_ids'],local_registry=str(b.LOCAL_REGISTRY),remote_registry=b.REMOTE_REGISTRY,
   recovery_authority_sha256=b.FIXED['recovery-authority'],original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA,
   retry_allowed=False,replacement_worlds=False))
  collection_binding=dict(freeze_sha256=b.FIXED['freeze'],files={n:H(n) for n in ('terminal.json','inventory.json','plan.json')})
  supplied['collection-acceptance']=b.raw(dict(schema='ace-scientific-collection-acceptance-v1',status='accepted',required_findings=0,
   scope='policy_train',complete_custody_verified=True,accounting_complete=True,collection=collection_binding))
  cpu=dict(schema='ace-scientific-training-preparation-cpu-v1',scope='policy_train',source_commit=b.SOURCE_COMMIT,
   runtime=f['runtime'],runtime_source=f['runtime_source'],model_files=f['model_files'],collection=collection_binding,
   files={'collection-acceptance.json':b.sha(supplied['collection-acceptance'])})
  for key in ('archive','manifest'):cpu[key]=dict(path=f['runtime_'+key+'_path'],sha256=f['runtime_'+key+'_sha256'])
  supplied['cpu-packet']=b.raw(cpu);cpu_pin=b.sha(supplied['cpu-packet'])
  plan=dict(dpo_steps=1,scope='policy_train',execution_kind='qwen_screen_singletons',sft_steps=160,sft_rows=160,dpo_rows=2,
   one_pass_sft=True,checkpoint_steps=[159,160],numeric_recipe=dict(seed=109,scoring_path='canonical_selected_unpadded'),
   records=[dict(record_id=n,prompt_ids=[1]) for n in b.ROSTER],outer_provenance_receipt_sha256=cpu_pin)
  exposure=dict(counts=dict(retained_pairs=3584,dpo_steps=1));pp=b.digest(plan);ep=b.digest(exposure)
  prepared=dict(training_plan=plan,training_plan_sha256=pp,exposure=exposure,exposure_sha256=ep,tokenizer_sha256=b.digest(f['model_files']),
   collection_terminal_sha256=H('terminal.json'),collection_inventory_sha256=H('inventory.json'),collection_plan_sha256=H('plan.json'))
  prepared_raw=(json.dumps(prepared,sort_keys=True,separators=(',',':'))+'\n').encode();prepared_pin=b.sha(prepared_raw)
  accepted=dict(schema='ace-scientific-training-preparation-acceptance-v1',status='accepted',scope='policy_train',required_findings=0,
   cpu_packet_sha256=cpu_pin,prepared_sha256=prepared_pin,plan_sha256=pp,freeze_sha256=b.FIXED['freeze'],namespace_admission_sha256=b.FIXED['namespace-admission'],
   complete_custody_verified=True,runtime_qualified_for_this_cpu_execution=True,owned_process_cleanup_verified=True,
   training_execution_qualified=False,scientific_qualified=False,job_id=job,numerical_plan_file_sha256=H('numfile'),numerical_plan_sha256=H('numplan'),exposure_sha256=ep)
  supplied['cpu-acceptance']=b.raw(accepted)
  terminal=dict(schema='ace-scientific-training-preparation-cpu-v1',status='cpu_preparation_complete',packet_sha256=cpu_pin,
   llm_prepared_file_sha256=prepared_pin,training_plan_sha256=pp,exposure_sha256=ep,numerical_plan_file_sha256=H('numfile'),numerical_plan_sha256=H('numplan'),
   records=160,retained_pairs=3584,sft_steps=160,dpo_steps=1,model_weights_loaded=False,explicit_cuda_api_called=False,
   training_execution_qualified=False,scientific_qualified=False,retry_allowed=False,inference_calls=0,environment_responses=0,training_updates=0)
  supplied['cpu-terminal']=b.raw(terminal)
  supplied['cpu-launch-packet']=b.raw(dict(remote_root=b.CPU_ROOT,source_commit=b.SOURCE_COMMIT,new_input_files={'inputs/packet.json':cpu_pin}))
  supplied['cpu-result']=b.raw(dict(job_id=job,returncode=0,ambiguous=False,intent_sha256=H('cpu intent')))
  rows=[[str(i),'ace_llm_toy','COMPLETED','0:0','1','1','ucb736_asc2','cpu=1,node=1'] for i in range(100,141)]
  journal={f'/toy/submission-{i}.json':H(str(i)) for i in range(84)}
  old=dict(terminal_jobs=41,submission_journals=84,outstanding_reservation=0,related_accounting=rows,journal_pins=journal,closed_cpu_seconds=41,closed_gpu_slice_seconds=0)
  supplied['previous-baseline']=b.raw(old)
  cpus=next(s['cpus'] for s in f['stages'] if s['stage']=='tokenize')
  row=[job,'ace_llm_toy_tokenization','COMPLETED','0:0',str(cpus),'1','ucb736_asc2',f'cpu={cpus},node=1']
  journals={**journal,b.CPU_ROOT+'/submission.intent.json':H('cpu intent'),b.CPU_ROOT+'/submission.result.json':b.sha(supplied['cpu-result'])}
  fs={paths['cpu-packet-relative']:cpu_pin,paths['cpu-prepared-relative']:prepared_pin,
   paths['cpu-terminal-relative']:b.sha(supplied['cpu-terminal']),'launch/packet.json':b.sha(supplied['cpu-launch-packet']),
   'launch/submission.intent.json':H('cpu intent'),'launch/submission.result.json':b.sha(supplied['cpu-result']),
   'launch/inputs/collection-acceptance.json':b.sha(supplied['collection-acceptance'])}
  files={n:dict(sha256=h,bytes=10) for n,h in fs.items()}
  observation=dict(files=files,parent_accounting=row,terminal_gate=True,root_present=True,journal_pins=journals)
  supplied['cpu-inventory']=b.raw(dict(returncode=0,stdout=json.dumps(observation),python_source=f'ROOT={b.CPU_ROOT!r}\nJOB={job!r}\n'))
  allocation=dict(resource_contract_matched=True,errors=[],allocated_cpus=cpus,elapsed_seconds=1,gpu_count=0,gpu_slice_seconds=0,allocated_cpu_seconds=cpus)
  disp=dict(job_id=job,remote_root=b.CPU_ROOT,packet_sha256=b.sha(supplied['cpu-launch-packet']),submission_result_sha256=b.sha(supplied['cpu-result']),
   custody_bytes_verified=True,current_lineage_verified=True,errors=[],exact_prior_accounting_matched=True,exact_prior_and_current_journals_matched=True,
   observed_related_jobs=42,observed_submission_journals=86,raw_files=len(files),raw_bytes=10*len(files),allocation=allocation,closed_cpu_seconds=41+cpus,closed_gpu_slice_seconds=0)
  supplied['cpu-disposition']=b.raw(disp)
  supplied['baseline']=b.raw(dict(terminal_jobs=42,submission_journals=86,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,
   input_pins=dict(predecessor=b.sha(supplied['previous-baseline']),acceptance=b.sha(supplied['cpu-acceptance']),disposition=b.sha(supplied['cpu-disposition'])),
   related_accounting=rows+[row],journal_pins=journals,closed_cpu_seconds=41+cpus,closed_gpu_slice_seconds=0,cpu_admission_seconds=3641+cpus))
  supplied['cpu-report']=b'ARTIFICIAL fixture report only\n';supplied['site-readiness']=b'ARTIFICIAL fixture readiness only\n'
  return supplied,prepared_raw,prepared_pin,pp,paths
 def test_saved_metadata_positive_no_disposition_files(self):
  args=self.fixture();ctx=b.authenticate(*args)
  self.assertEqual(ctx['dpo_steps'],1);self.assertEqual(len(ctx['upstream_pins']),7)
  self.assertNotIn('remote_files',b.decode(args[0]['cpu-disposition']))
 def test_reject_actual_joins_and_incomplete_acceptance(self):
  for name,change in (('cpu-acceptance',lambda v:v.pop('numerical_plan_sha256')),
                      ('cpu-acceptance',lambda v:v.update(runtime_qualified_for_this_cpu_execution=False)),
                      ('baseline',lambda v:v.update(terminal_jobs=41)),
                      ('baseline',lambda v:v['journal_pins'].pop(next(iter(v['journal_pins'])))),
                      ('cpu-inventory',lambda v:v.update(returncode=1)),
                      ('cpu-inventory',lambda v:v.update(python_source='ROOT="/different"\nJOB="99999999"\n')),
                      ('cpu-disposition',lambda v:v.update(raw_bytes=0))):
   args=list(self.fixture());v=b.decode(args[0][name]);change(v);args[0][name]=b.raw(v)
   with self.subTest(name=name,change=change),self.assertRaises((ValueError,KeyError)):b.authenticate(*args)
 def test_prompt_cap_rejects_without_truncation(self):
  args=list(self.fixture());v=b.decode(args[1]);v['training_plan']['records'][0]['prompt_ids']=[1]*4745
  v['training_plan_sha256']=b.digest(v['training_plan']);args[1]=(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
  args[2]=b.sha(args[1]);args[3]=v['training_plan_sha256'];a=b.decode(args[0]['cpu-acceptance']);a.update(prepared_sha256=args[2],plan_sha256=args[3]);args[0]['cpu-acceptance']=b.raw(a)
  with self.assertRaisesRegex(ValueError,'envelope'):b.authenticate(*args)
 def test_later_baseline_retains_failed_collection(self):
  args=list(self.fixture());supplied=args[0];old=b.decode(supplied['previous-baseline']);base=b.decode(supplied['baseline'])
  failed=['88888888','ace_llm_toy_failed_collection','FAILED','1:0','2','126','ucb736_asc2','cpu=2,node=1']
  old['related_accounting'].append(failed);old['terminal_jobs']+=1;old['submission_journals']+=2;old['closed_cpu_seconds']+=252
  additions={'/toy/failed/submission.intent.json':H('failed intent'),'/toy/failed/submission.result.json':H('failed result')}
  old['journal_pins'].update(additions);supplied['previous-baseline']=b.raw(old)
  base['related_accounting'].append(failed);base['terminal_jobs']+=1;base['submission_journals']+=2
  base['closed_cpu_seconds']+=252;base['cpu_admission_seconds']+=252;base['journal_pins'].update(additions)
  base['input_pins']['predecessor']=b.sha(supplied['previous-baseline'])
  disp=b.decode(supplied['cpu-disposition']);disp['observed_related_jobs']+=1;disp['observed_submission_journals']+=2;disp['closed_cpu_seconds']+=252
  supplied['cpu-disposition']=b.raw(disp);base['input_pins']['disposition']=b.sha(supplied['cpu-disposition']);supplied['baseline']=b.raw(base)
  transport=b.decode(supplied['cpu-inventory']);observation=b.decode(transport['stdout']);observation['journal_pins'].update(additions)
  transport['stdout']=json.dumps(observation);supplied['cpu-inventory']=b.raw(transport)
  checked=b.authenticate(*args)
  self.assertEqual(checked['metadata']['baseline']['terminal_jobs'],43)
  self.assertIn(failed,checked['metadata']['baseline']['related_accounting'])
  broken=copy.deepcopy(args);v=b.decode(broken[0]['baseline']);v['related_accounting'].remove(failed);broken[0]['baseline']=b.raw(v)
  with self.assertRaisesRegex(ValueError,'prior rows'):b.authenticate(*broken)
 def test_new_static_roster_and_remote_syntax(self):
  source=(PUBLIC/'candidate03/source/scripts/research'/b.ENTRY).read_bytes()
  self.assertEqual(len(b.literal(source,'SOURCE_NAMES')),29);self.assertEqual(len(b.literal(source,'PREPARATION_SOURCES')),18)
  self.assertEqual(b.literal(source,'RESOURCES'),b.RESOURCES);self.assertEqual(b.literal(source,'DEADLINES'),b.DEADLINES)
  for name in ('build_packet.py','launch_once.py','monitor_once.py'):
   tree=ast.parse(Path(__file__).with_name(name).read_bytes());compile(tree,name,'exec')
   for node in ast.walk(tree):
    if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.lstrip().startswith(('import pathlib,','need(root.parent','need(not os.path.lexists(root/')):
     compile(node.value,name+':embedded','exec')
if __name__=='__main__':unittest.main(verbosity=2)
