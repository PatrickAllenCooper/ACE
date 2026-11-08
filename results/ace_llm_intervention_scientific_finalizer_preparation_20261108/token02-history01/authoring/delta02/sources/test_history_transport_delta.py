"""New artificial history/custody integration only. Never project code or a main."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,shutil,types,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;EVIDENCE=Path(os.environ['ACE_FINALIZER_TEST_EVIDENCE']).resolve()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('finalizer_metadata_delta',HERE/'build_packet.py')
c=load('finalizer_collector_delta',HERE/'collector/pull_terminal.py')
h=load('cleared_stream_only',HERE/'collector/stream_custody.py')
H=lambda value:hashlib.sha256(str(value).encode()).hexdigest()
V=lambda value:dict(bytes=len(str(value)),sha256=H(value))

def history_fixture():
 rows=[[str(i),'ace_fixture','FAILED','1:0','2','1','ucb736_asc2','cpu=2,mem=7680M,node=1'] for i in range(1,42)]
 prev=dict(terminal_jobs=41,submission_journals=84,related_accounting=rows,journal_pins={f'/fixture/journal/{i}':H(i) for i in range(84)},
  closed_cpu_seconds=82,closed_gpu_slice_seconds=0,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,cpu_admission_seconds=3682)
 m={'collection-failure-baseline':prev};pins={'collection-failure-baseline':H('initial')};previous_pin=pins['collection-failure-baseline']
 for prefix,root,count,journals,cpus,gpus,wall in b.STAGES:
  if prefix=='token':
   row=[b.FAILED_TOKEN_JOB,'ace_fixture_failed_token','FAILED','1:0','3','11','ucb736_asc2','cpu=3,mem=11520M,node=1']
   prev=dict(prev,terminal_jobs=43,submission_journals=88,related_accounting=prev['related_accounting']+[row],
    journal_pins={**prev['journal_pins'],b.FAILED_TOKEN_ROOT+'/submission.intent.json':H('failed intent'),b.FAILED_TOKEN_ROOT+'/submission.result.json':H('failed result')},
    closed_cpu_seconds=prev['closed_cpu_seconds']+33,cpu_admission_seconds=prev['cpu_admission_seconds']+33)
   m['token-failure-baseline']=prev;pins['token-failure-baseline']=H('failure baseline');previous_pin=pins['token-failure-baseline']
  for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'):pins[prefix+'-'+kind]=H(prefix+kind)
  job=str(1000+count);row=[job,'ace_fixture_'+prefix,'COMPLETED','0:0',str(cpus),'10','ucb736_asc2',f'cpu={cpus},node=1,mem=1M'+(',gres/gpu=1,gres/gpu:a100_3g.20gb=1' if gpus else '')]
  result=dict(job_id=job,returncode=0,ambiguous=False,intent_sha256=H(prefix+'intent'))
  schemas=dict(collection='ace-scientific-collection-launch-v1',token='ace-scientific-tokenization-launch-v1',training='ace-scientific-training-gpu-launch-v1',cpu='ace-scientific-cpu-campaign-launch-v1',gpu='ace-scientific-campaign-gpu-launch-v1')
  packet=dict(schema=schemas[prefix],remote_root=root,source_commit=b.SOURCE_COMMIT,baseline_sha256=previous_pin,new_input_files={'inputs/public.json':H(prefix)},
   resources=dict(wall_seconds=wall,memory_mib=dict(collection=7680,token=11520,training=15360,cpu=7680,gpu=15360)[prefix],cpus=cpus,gpus=gpus,account='ucb736_asc2',partition='aa100' if gpus else 'acpu',qos='gpu-testing' if gpus else 'cpu-normal'))
  files={'launch/'+n:dict(bytes=1,sha256=v) for n,v in {**packet['new_input_files'],'packet.json':pins[prefix+'-launch-packet'],'submission.intent.json':result['intent_sha256'],'submission.result.json':pins[prefix+'-result']}.items()}
  jp={**prev['journal_pins'],root+'/submission.intent.json':result['intent_sha256'],root+'/submission.result.json':pins[prefix+'-result']}
  rows=prev['related_accounting']+[row];cpu,gpu=b.cost(rows)
  base=dict(prev,terminal_jobs=count,submission_journals=journals,related_accounting=rows,journal_pins=jp,closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,cpu_admission_seconds=cpu+3600,
   input_pins=dict(predecessor=previous_pin,acceptance=pins[prefix+'-acceptance'],disposition=pins[prefix+'-disposition']))
  obs=dict(terminal_gate=True,root_present=True,parent_accounting=row,related_accounting=rows,journal_pins=jp,files=files,directories=['launch','launch/inputs'])
  disp=dict(job_id=job,remote_root=root,packet_sha256=pins[prefix+'-launch-packet'],submission_result_sha256=pins[prefix+'-result'],custody_bytes_verified=True,current_lineage_verified=True,
   errors=[],input_pin_mismatches=[],exact_prior_accounting_matched=True,exact_prior_and_current_journals_matched=True,observed_related_jobs=count,observed_submission_journals=journals,raw_files=len(files),raw_bytes=len(files))
  m.update({prefix+'-acceptance':dict(job_id=job),prefix+'-launch-packet':packet,prefix+'-result':result,prefix+'-disposition':disp,prefix+'-inventory':dict(returncode=0,python_source='ROOT='+repr(root)+'\nJOB='+repr(job),stdout=json.dumps(obs)),prefix+'-baseline':base})
  prev=base;previous_pin=pins[prefix+'-baseline']
 return m,pins

class NewFinalizerHistory(unittest.TestCase):
 def setUp(self):self.out=EVIDENCE/self._testMethodName;self.out.mkdir()
 def test_failed43_splice_all_successors_and_exact_final_budget(self):
  m,p=history_fixture();(self.out/'fabricated-history.json').write_bytes(b.raw(dict(metadata=m,pins=p)))
  with patch.object(b,'failed_token_transition',return_value={}) as join:
   b.history(m,p);join.assert_called_once_with(m,p)
  self.assertEqual([s[2:4] for s in b.STAGES],[(42,86),(44,90),(45,92),(46,94),(47,96)])
  self.assertEqual(b.budget(m['gpu-baseline'])['candidate_cpu_reservation'],2400)
  changes=[lambda x:x['token-launch-packet'].__setitem__('baseline_sha256',p['collection-baseline']),
   lambda x:x['training-baseline']['related_accounting'].pop(42),lambda x:x['cpu-baseline']['journal_pins'].pop(b.FAILED_TOKEN_ROOT+'/submission.result.json'),
   lambda x:x['gpu-baseline'].__setitem__('closed_cpu_seconds',0)]
  for mutate in changes:
   wrong=copy.deepcopy(m);mutate(wrong)
   with patch.object(b,'failed_token_transition',return_value={}),self.assertRaises(ValueError):b.history(wrong,p)
 def test_new_producer_aliases_and_exact_recovery_object(self):
  pins={name:H(name) for name in b.METADATA};recovery={'independent':H('recovery')};m={}
  for stage in ('token','training','cpu','gpu'):
   m[stage+'-launch-packet']=dict(token_recovery=recovery,failed_token_root=b.FAILED_TOKEN_ROOT,failed_token_job_id=b.FAILED_TOKEN_JOB,new_input_files={b.metadata_name(n):pins[n] for n in b.TOKEN_METADATA})
  aliases={'training':{'cpu-packet.json':'token-packet'},'cpu':{'collection-baseline.json':'collection-baseline','collection-acceptance.json':'collection-acceptance','collection-disposition.json':'collection-disposition','cpu-packet.json':'token-packet','cpu-launch-packet.json':'token-launch-packet','cpu-result.json':'token-result','cpu-disposition.json':'token-disposition','cpu-inventory.json':'token-inventory'},'gpu':{'token-packet.json':'token-packet'}}
  for stage,mapping in aliases.items():m[stage+'-launch-packet']['new_input_files'].update({n:pins[k] for n,k in mapping.items()})
  b.recovery_joins(m,pins,recovery)
  for stage in ('token','training','cpu','gpu'):
   wrong=copy.deepcopy(m);wrong[stage+'-launch-packet']['token_recovery']={}
   with self.assertRaises(ValueError):b.recovery_joins(wrong,pins,recovery)
  wrong=copy.deepcopy(m);wrong['cpu-launch-packet']['new_input_files']['cpu-packet.json']=pins['cpu-packet']
  with self.assertRaises(ValueError):b.recovery_joins(wrong,pins,recovery)
 def test_full_root_history_and_empty_directory_retention(self):
  names=['collection','failed-collection','failed-token','token','training','cpu','gpu'];p={n+'-launch-packet':H(n) for n in names};p['failed-token-packet']=H('failed-token')
  roots={n:dict(files={'packet.json':dict(bytes=1,sha256=H(n)),'opaque.bin':V(n)},directories=['empty']) for n in names}
  def obs(which):
   return dict(files={n+'/'+k:v for n in which for k,v in roots[n]['files'].items()},directories=sorted([x for n in which for x in [n]+[n+'/'+d for d in roots[n]['directories']]]))
  failure=obs(names[:3]);observations={'collection':obs(names[:2])}
  for i,stage in enumerate(names[3:],3):observations[stage]=obs(names[:i+1])
  observations['gpu']['files']['cpu/cpu/worker/artifacts/campaign/policy-05/opaque.json']=V('new arm')
  observations['gpu']['directories'].append('cpu/cpu/worker/artifacts/campaign/policy-05')
  m={'token-failure-inventory':dict(stdout=json.dumps(failure)),'collection-failure-disposition':dict(packet_sha256=H('failed-collection'))}
  b.retained_root_history(m,p,observations)
  bad=copy.deepcopy(observations);bad['gpu']['directories'].remove('failed-token/empty')
  with self.assertRaises(ValueError):b.retained_root_history(m,p,bad)
  bad=copy.deepcopy(observations);bad['cpu']['files']['failed-token/opaque.bin']=V('changed')
  with self.assertRaises(ValueError):b.retained_root_history(m,p,bad)
 def test_whole_nested_failure_archive_required(self):
  view=dict(files={'archive.tar':V('archive'),'remote/launch/failure.json':V('original')},directories=['empty','remote','remote/launch'])
  seal=dict(identical=True,copies={'one':view,'two':copy.deepcopy(view)});hp=H('seal')
  ctx=dict(metadata={'token-failure-custody-inventory':seal},pins={'token-failure-custody-inventory':hp})
  files={'kept/'+n:v for n,v in view['files'].items()};files['kept/custody-inventory.json']=dict(bytes=1,sha256=hp)
  tree=dict(files=files,directories=['kept']+['kept/'+n for n in view['directories']]);b.nested_failure_capsule(ctx,tree)
  for name in ('kept/archive.tar','kept/custody-inventory.json'):
   bad=copy.deepcopy(tree);bad['files'].pop(name)
   with self.assertRaises(ValueError):b.nested_failure_capsule(ctx,bad)
 def test_finalizer_child_job_and_monitor_unchanged_by_history(self):
  def f(path,name):
   s=path.read_text();return ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
  for name in ('make_packet','job_script','gpu_authority'):
   self.assertEqual(f(HERE/'build_packet.py',name),f(HERE/'predecessor/build_packet.py',name))
  self.assertEqual((HERE/'monitor_once.py').read_bytes(),(HERE/'predecessor/monitor_once.py').read_bytes())
  self.assertEqual(b.ENTRY_SHA,'28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a')
 def test_collector48_98_failed_charge_preserves47_history(self):
  m,p=history_fixture();base=m['gpu-baseline'];job='88888888';row=[job,'ace_llm_scientific_finalization01','FAILED','1:0','2','17','ucb736_asc2','cpu=2,mem=7680M,node=1']
  observation=dict(parent_accounting=row,related_accounting=base['related_accounting']+[row],journal_pins={**base['journal_pins'],c.ROOT+'/submission.intent.json':H('intent'),c.ROOT+'/submission.result.json':H('result')})
  result=c.accounting_extension(observation,base,job,H('intent'),H('result'));self.assertEqual(result['allocated_cpu_seconds'],34);self.assertEqual(result['gpu_slice_seconds'],0)
  bad=copy.deepcopy(observation);bad['related_accounting'].pop(42)
  with self.assertRaises(ValueError):c.accounting_extension(bad,base,job,H('intent'),H('result'))
  (self.out/'actual-artificial-extension.json').write_bytes(c.raw(dict(observation=observation,result=result)))
 def test_only_private_surface_append_and_original_bytes_preserved(self):
  roots={'launch':c.ROOT,'cpu':b.CPU_ROOT,'failed-token01':b.FAILED_TOKEN_ROOT};expected={r:dict(files={'old.bin':H(r)},directories=['empty']) for label,r in roots.items() if label!='launch'}
  actual=dict(files={label+'/old.bin':dict(bytes=1,sha256=H(root)) for label,root in roots.items() if label!='launch'},directories=['launch','cpu','cpu/empty','failed-token01','failed-token01/empty'])
  added='cpu/cpu/worker/artifacts/campaign/private/summary.json';actual['files'][added]=V('opaque private bytes');actual['directories'].append('cpu/cpu/worker/artifacts/campaign/private')
  with patch.object(c,'ROOTS',roots):
   self.assertEqual(c.preserved_roots(actual,expected,b.CPU_ROOT),[])
   for name in ('cpu/old.bin','failed-token01/old.bin','cpu/undeclared.json'):
    bad=copy.deepcopy(actual);bad['files'][name]=V('changed')
    self.assertTrue(c.preserved_roots(bad,expected,b.CPU_ROOT))
 def test_stream_eight_roots_and_two_independent_private_copies(self):
  source=self.out/'artificial';source.mkdir();external=self.out/'external';local=self.out/'local';external.mkdir();local.mkdir()
  labels=('launch','gpu','cpu','training','token','common','failed01','failed-token01')
  roots={label:str(source/label) for label in labels}
  for label in labels:
   d=source/label;d.mkdir();(d/'empty').mkdir();(d/'opaque.bin').write_bytes(b'artificial opaque '+label.encode())
  for name in ('namespace.json','original-namespace.json'):(source/name).write_bytes(b'opaque artificial namespace')
  before=dict(**c.inventory(source),root_presence={n:True for n in labels},root_present=True,namespace_present=True,original_namespace_present=True,
   original_hardlink_groups=[],parent_accounting=[],journal_pins={},upstream_pins={},upstream_errors={},related_accounting=[])
  helper_source=(HERE/'collector/stream_custody.py').read_bytes();calls=[]
  def capture(command,request,attempt,timeout):
   calls.append(command);self.assertEqual(timeout,1800);attempt.mkdir();c.write(attempt/'request.json',request);c.write(attempt/'command.json',c.raw(command))
   with (attempt/'archive.tar').open('xb') as f:emitted=h.emit(c.decode(request),f)
   c.write(attempt/'stderr.log',(json.dumps(emitted,sort_keys=True)+'\n').encode())
   receipt=dict(status='captured_pending_inventory_verification',returncode=0,acceptance=False,request_sha256=c.sha(request),command_sha256=c.sha(c.raw(command)),**{n:c.fingerprint(attempt/n,sole_link=True) for n in ('archive.tar','stderr.log')})
   c.write(attempt/'capture.json',c.raw(receipt));return receipt
  def save(n,v):
   for d in (external,local):c.write(d/n,c.raw(v))
  helper=types.SimpleNamespace(request_check=h.request_check,archive_bound=h.archive_bound,capture=capture,materialize=h.materialize)
  with patch.multiple(c,ROOTS=roots,REMOTE_NAMESPACE=str(source/'namespace.json'),OLD_REMOTE_NAMESPACE=str(source/'original-namespace.json')):
   actual,_=c.stream_transfer(before,observe=lambda _:copy.deepcopy(before),save=save,helper=helper,helper_source=helper_source,external=external,local=local,copy_file=c.copy_file)
  self.assertEqual(len(calls),1);self.assertEqual(c.inventory(external),c.inventory(local))
  self.assertEqual(actual['files'],before['files'])
  self.assertFalse(os.path.samestat((external/'remote/cpu/opaque.bin').stat(),(local/'remote/cpu/opaque.bin').stat()))
 def test_finalizer_resource_profile_and_partial_retention(self):
  job='88888888';row=[job,'ace_llm_scientific_finalization01','FAILED','1:0','2','17','ucb736_asc2','cpu=2,mem=7680M,node=1']
  obs=dict(parent_accounting=row,results=dict(details=dict(returncode=0,stdout=job+'|acpu|cpu-normal|20|cpu=2,mem=7680M,node=1|paco0228|\n'),job=dict(returncode=0,stdout='JobId='+job+' Requeue=0 Restarts=0')))
  resource=c.resources(obs,job,self.out,{});self.assertTrue(resource['resource_contract_matched']);self.assertEqual(resource['allocated_cpu_seconds'],34)
  bad=copy.deepcopy(obs);bad['parent_accounting'][7]+=',gres/gpu=1'
  self.assertFalse(c.resources(bad,job,self.out,{})['resource_contract_matched'])
  source=self.out/'partial';source.mkdir();(source/'empty').mkdir();(source/'archive.partial').write_bytes(b'original failure partial')
  dest=self.out/'mirror';dest.mkdir();self.assertEqual(c.retain_partial(source,dest),[]);self.assertEqual(c.inventory(source),c.inventory(dest))

if __name__=='__main__':unittest.main(verbosity=2)
