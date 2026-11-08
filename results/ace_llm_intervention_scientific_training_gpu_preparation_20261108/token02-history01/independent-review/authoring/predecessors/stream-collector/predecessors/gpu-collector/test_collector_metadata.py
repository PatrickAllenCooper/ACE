"""New fabricated collector metadata tests, no main/remote/project/model execution."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest import mock
spec=importlib.util.spec_from_file_location('collector',Path(__file__).with_name('pull_terminal.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class Metadata(unittest.TestCase):
 def observation(self):
  job='90000000';t='cpu=4,node=1,mem=15G,gres/gpu=1,gres/gpu:a100_3g.20gb=1'
  return dict(parent_accounting=[job,'ace_llm_scientific_train01','COMPLETED','0:0','4','61','ucb736_asc2',t],
   results=dict(details=dict(returncode=0,stdout=f'{job}|aa100|gpu-testing|45|{t}|paco0228|\n'),
    job=dict(returncode=0,stdout=f'JobId={job} Requeue=0 Restarts=0')))
 def test_gpu_resource_accounting_and_uncertainty(self):
  v=self.observation();out=c.resources(v,'90000000',Path('/not-read'),{})
  self.assertTrue(out['resource_contract_matched']);self.assertEqual((out['allocated_cpu_seconds'],out['gpu_slice_seconds']),(244,61))
  for change in ('cpus','wall','wrong_mig','missing_tres','requeue'):
   w=copy.deepcopy(v)
   if change=='cpus':w['parent_accounting'][4]='8'
   if change=='wall':w['parent_accounting'][5]='2701'
   if change=='wrong_mig':w['parent_accounting'][7]=w['parent_accounting'][7].replace('a100_3g.20gb','a100_1g.5gb')
   if change=='missing_tres':w['parent_accounting'][7]=''
   if change=='requeue':w['results']['job']['stdout']='JobId=90000000 Requeue=1 Restarts=1'
   with self.subTest(change=change):
    result=c.resources(w,'90000000',Path('/not-read'),{});self.assertFalse(result['resource_contract_matched'])
    if change=='missing_tres':self.assertIsNone(result['gpu_slice_seconds'])
  self.assertEqual(c.cost([v['parent_accounting']]),(244,61))
  w=copy.deepcopy(v);w['parent_accounting'][7]=w['parent_accounting'][7].replace('gres/gpu=1','gres/gpu=2')
  self.assertIsNone(c.resources(w,'90000000',Path('/not-read'),{})['gpu_slice_seconds'])
  with self.assertRaises(ValueError):c.cost([w['parent_accounting']])
 def test_full44_rows90_journals_retains_failed_cost(self):
  v=self.observation();rows=[[str(100+i),'ace_llm_prior','FAILED' if i==0 else 'COMPLETED','1:0' if i==0 else '0:0','2','126' if i==0 else '1','ucb736_asc2','cpu=2,node=1,mem=7680M'] for i in range(43)]
  journals={f'/toy/prior-{i}.json':str(i) for i in range(88)};baseline=dict(related_accounting=rows,journal_pins=journals)
  v.update(related_accounting=rows+[v['parent_accounting']],journal_pins={**journals,c.ROOT+'/submission.intent.json':'i',c.ROOT+'/submission.result.json':'r'})
  self.assertEqual(c.accounting_extension(v,baseline,'90000000','i','r'),(True,True))
  self.assertEqual(c.cost(v['related_accounting']),(252+42*2+244,61))
  broken=copy.deepcopy(v);broken['related_accounting'][0][5]='0';self.assertFalse(c.accounting_extension(broken,baseline,'90000000','i','r')[0])
  broken=copy.deepcopy(v);broken['journal_pins']['/toy/prior-0.json']='replaced';self.assertFalse(c.accounting_extension(broken,baseline,'90000000','i','r')[1])
 def test_public_selector_never_decodes_scores_or_tensors(self):
  value=dict(scores={'SCORE_SENTINEL':['braces } , \\"',{'private':[1,2]}]},cuda_interval_ms=12.5,forward_count=1,device='cuda:0')
  real=c.decode
  def guarded(raw):
   self.assertNotIn('SCORE_SENTINEL',raw);return real(raw)
  with mock.patch.object(c,'decode',side_effect=guarded):
   self.assertEqual(c.public_fields(json.dumps(value).encode(),{'cuda_interval_ms','forward_count','device'}),dict(cuda_interval_ms=12.5,forward_count=1,device='cuda:0'))
  with self.assertRaises(ValueError):c.public_fields(b'{"pid":1,"pid":2}',{'pid'})
  # Every new test fixture remains in the retained attempt rather than auto-deleted.
  root=Path(os.environ['COLLECTOR_FIXTURES'])/'public';root.mkdir(parents=True)
  examples={'child-start.json':dict(pid=7,elapsed_seconds=2),
   'gpu-sample-0000.json':dict(returncode=0,owned_rows=['7, GPU-toy, 123'],elapsed_seconds=4),
   'session/worker/artifacts/first-gpu-work.json':value,
   'progress-0001.json':dict(progress=[['phase-1.json','toyhash']],progress_bound_seconds=180,telemetry_bound_seconds=90)}
  files={}
  for n,v in examples.items():
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));files['launch/training/'+n]=c.fingerprint(p)
  opaque=root/'adapter.safetensors';opaque.write_bytes(b'opaque fake bytes, not a tensor');files['launch/training/adapter.safetensors']=c.fingerprint(opaque)
  with mock.patch.object(c,'decode',side_effect=guarded):report=c.public_gpu_evidence(root,files)
  self.assertTrue(report['independent_signal_types_present']);self.assertEqual(report['owned_memory_mib_max'],123)
  self.assertIsNone(report['utilization_percent']);self.assertFalse(report['acceptance']);self.assertFalse(report['adapter_tensors_decoded'])
  missing=c.public_gpu_evidence(root,{});self.assertFalse(missing['independent_signal_types_present']);self.assertIsNone(missing['owned_memory_mib_max'])
 def test_exact_remote_intent_and_independent_copy_alias_rejection(self):
  local=dict(existing_namespace_claim_sha256='a'*64,packet_sha256='b'*64)
  remote={**local,'remote_utc':'toy','fresh_readiness':{}};data=c.raw(remote)
  c.validate_remote_intent(data,c.sha(data),local,'a'*64)
  remote['packet_sha256']='c'*64;data=c.raw(remote)
  with self.assertRaises(ValueError):c.validate_remote_intent(data,c.sha(data),local,'a'*64)
  root=Path(os.environ['COLLECTOR_FIXTURES'])/'links';root.mkdir();(root/'original').write_bytes(b'toy')
  os.link(root/'original',root/'alias')
  with self.assertRaises(ValueError):c.inventory(root)
if __name__=='__main__':unittest.main(verbosity=2)
