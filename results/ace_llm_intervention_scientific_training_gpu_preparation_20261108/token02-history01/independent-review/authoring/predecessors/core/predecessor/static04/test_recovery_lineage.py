"""Only new amendment/failed-cost joins. Future successes are fabricated metadata."""
from pathlib import Path
import copy,importlib.util,json,unittest
spec=importlib.util.spec_from_file_location('prior_fixture_only',Path(__file__).with_name('test_authoring_metadata.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
b=old.b;H=old.H;REPO=Path('/Users/pat/code/ACE')
class RecoveryLineage(unittest.TestCase):
 def fixture(self):
  supplied,prepared_raw,prepared_pin,plan_pin,paths=old.Metadata().fixture()
  locations={
   'operational-amendment':'results/ace_llm_intervention_scientific_collection_recovery_preparation_20261108/amendment01/operational-amendment.json',
   'collection-failure-acceptance':'results/ace_llm_intervention_scientific_collection_20261108/failure-acceptance01/acceptance.json',
   'no-response-proof':'results/ace_llm_intervention_scientific_collection_20261108/no-response-proof01/proof.json',
   'collection-failure-baseline':'results/ace_llm_intervention_scientific_collection_20261108/failure-acceptance01/accounting.json'}
  for name,path in locations.items():supplied[name]=(REPO/path).read_bytes()
  matches=list((REPO/'docs/development/guidance/reviews').glob('*collection02_amendment_review*'))
  assert len(matches)==1
  supplied['operational-adoption-review']=matches[0].read_bytes()
  authority=b.decode(supplied['operational-amendment']);failed=b.decode(supplied['collection-failure-baseline'])
  # Namespace fixture is public identity metadata only; adopt already-pinned actual claim bytes.
  supplied['namespace-claim']=b.read(b.LOCAL_REGISTRY,authority['namespace_claim_sha256'])
  cj='99999998';tj='99999999'
  collection=b.decode(supplied['collection-acceptance']);collection['collection']['root']=b.COLLECTION_ROOT+'/collection/session/worker/task/collection'
  collection_packet=dict(remote_root=b.COLLECTION_ROOT,source_commit=b.SOURCE_COMMIT,resources=authority['resources'],deadlines=authority['deadlines'],
   baseline_sha256=b.sha(supplied['collection-failure-baseline']),operational_amendment_sha256=b.sha(supplied['operational-amendment']),
   new_input_files={b.metadata_name(n):b.sha(supplied[n]) for n in ('freeze','namespace-admission','namespace-claim','operational-amendment','collection-failure-acceptance','no-response-proof')})
  collection_packet['new_input_files']['failure-baseline.json']=b.sha(supplied['collection-failure-baseline'])
  supplied['collection-packet']=b.raw(collection_packet)
  cr=dict(job_id=cj,returncode=0,ambiguous=False,intent_sha256=H('toy collection intent'));supplied['collection-result']=b.raw(cr)
  row=[cj,'ace_llm_scientific_collect02','COMPLETED','0:0','2','7','ucb736_asc2','cpu=2,node=1,mem=7680M']
  journals={**failed['journal_pins'],b.COLLECTION_ROOT+'/submission.intent.json':cr['intent_sha256'],b.COLLECTION_ROOT+'/submission.result.json':b.sha(supplied['collection-result'])}
  originals={'packet.json':b.sha(supplied['collection-packet']),'submission.intent.json':cr['intent_sha256'],
   'submission.result.json':b.sha(supplied['collection-result']),**collection_packet['new_input_files']}
  files={'launch/'+n:dict(bytes=10,sha256=h) for n,h in originals.items()}
  files.update({'launch/collection/session/worker/task/collection/'+n:dict(bytes=10,sha256=h) for n,h in collection['collection']['files'].items()})
  obs=dict(files=files,terminal_gate=True,root_present=True,parent_accounting=row,related_accounting=failed['related_accounting']+[row],journal_pins=journals)
  supplied['collection-inventory']=b.raw(dict(returncode=0,stdout=json.dumps(obs),python_source=f'ROOT={b.COLLECTION_ROOT!r}\nJOB={cj!r}\n'))
  cd=dict(job_id=cj,remote_root=b.COLLECTION_ROOT,packet_sha256=b.sha(supplied['collection-packet']),submission_result_sha256=b.sha(supplied['collection-result']),
   custody_bytes_verified=True,current_lineage_verified=True,errors=[],input_pin_mismatches=[],exact_prior_accounting_matched=True,
   exact_prior_and_current_journals_matched=True,observed_related_jobs=42,observed_submission_journals=86,raw_files=len(files),raw_bytes=10*len(files),
   allocation=dict(resource_contract_matched=True,errors=[],allocated_cpus=2,elapsed_seconds=7,allocated_cpu_seconds=14,gpu_count=0,gpu_slice_seconds=0),
   closed_cpu_seconds=failed['closed_cpu_seconds']+14,closed_gpu_slice_seconds=failed['closed_gpu_slice_seconds'])
  supplied['collection-disposition']=b.raw(cd)
  collection.update(job_id=cj,collection_packet_sha256=b.sha(supplied['collection-packet']),disposition_sha256=b.sha(supplied['collection-disposition']))
  supplied['collection-acceptance']=b.raw(collection)
  base=dict(terminal_jobs=42,submission_journals=86,outstanding_reservation=0,related_accounting=obs['related_accounting'],journal_pins=journals,
   preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,closed_cpu_seconds=cd['closed_cpu_seconds'],
   closed_gpu_slice_seconds=cd['closed_gpu_slice_seconds'],cpu_admission_seconds=cd['closed_cpu_seconds']+3600,
   input_pins=dict(predecessor=b.sha(supplied['collection-failure-baseline']),acceptance=b.sha(supplied['collection-acceptance']),disposition=b.sha(supplied['collection-disposition'])))
  supplied['previous-baseline']=b.raw(base)
  cpu=b.decode(supplied['cpu-packet']);cpu['collection']=collection['collection'];cpu['files']['collection-acceptance.json']=b.sha(supplied['collection-acceptance']);supplied['cpu-packet']=b.raw(cpu)
  prepared=b.decode(prepared_raw);prepared['training_plan']['outer_provenance_receipt_sha256']=b.sha(supplied['cpu-packet'])
  plan_pin=prepared['training_plan_sha256']=b.digest(prepared['training_plan']);prepared_raw=(json.dumps(prepared,sort_keys=True,separators=(',',':'))+'\n').encode();prepared_pin=b.sha(prepared_raw)
  a=b.decode(supplied['cpu-acceptance']);a.update(cpu_packet_sha256=b.sha(supplied['cpu-packet']),prepared_sha256=prepared_pin,plan_sha256=plan_pin);supplied['cpu-acceptance']=b.raw(a)
  terminal=b.decode(supplied['cpu-terminal']);terminal.update(packet_sha256=b.sha(supplied['cpu-packet']),llm_prepared_file_sha256=prepared_pin,training_plan_sha256=plan_pin);supplied['cpu-terminal']=b.raw(terminal)
  outer=b.decode(supplied['cpu-launch-packet']);outer.update(collection_job_id=cj,baseline_sha256=b.sha(supplied['previous-baseline']))
  for key,name in (('operational_amendment_sha256','operational-amendment'),('operational_adoption_review_sha256','operational-adoption-review'),
   ('collection_failure_acceptance_sha256','collection-failure-acceptance'),('no_response_proof_sha256','no-response-proof'),('collection_acceptance_sha256','collection-acceptance')):outer[key]=b.sha(supplied[name])
  names=('operational-amendment','operational-adoption-review','collection-failure-acceptance','no-response-proof','collection-failure-baseline',
   'collection-acceptance','collection-packet','collection-result','collection-disposition')
  outer['new_input_files']={b.metadata_name(n):b.sha(supplied[n]) for n in names}
  outer['new_input_files'].update({'inputs/packet.json':b.sha(supplied['cpu-packet']),'baseline.json':b.sha(supplied['previous-baseline'])})
  supplied['cpu-launch-packet']=b.raw(outer)
  tr=b.decode(supplied['cpu-result']);tokenrow=[tj,'ace_llm_scientific_token01','COMPLETED','0:0','3','1','ucb736_asc2','cpu=3,node=1,mem=11520M']
  tjournals={**journals,b.CPU_ROOT+'/submission.intent.json':tr['intent_sha256'],b.CPU_ROOT+'/submission.result.json':b.sha(supplied['cpu-result'])}
  tf={'launch/'+n:dict(bytes=10,sha256=h) for n,h in outer['new_input_files'].items()}
  tf.update({paths['cpu-prepared-relative']:dict(bytes=10,sha256=prepared_pin),paths['cpu-terminal-relative']:dict(bytes=10,sha256=b.sha(supplied['cpu-terminal'])),
   'launch/packet.json':dict(bytes=10,sha256=b.sha(supplied['cpu-launch-packet'])),'launch/submission.intent.json':dict(bytes=10,sha256=tr['intent_sha256']),
   'launch/submission.result.json':dict(bytes=10,sha256=b.sha(supplied['cpu-result'])),'launch/inputs/collection-acceptance.json':dict(bytes=10,sha256=b.sha(supplied['collection-acceptance']))})
  ti=b.decode(supplied['cpu-inventory']);ti['stdout']=json.dumps(dict(files=tf,parent_accounting=tokenrow,terminal_gate=True,root_present=True,journal_pins=tjournals));supplied['cpu-inventory']=b.raw(ti)
  td=b.decode(supplied['cpu-disposition']);td.update(packet_sha256=b.sha(supplied['cpu-launch-packet']),observed_related_jobs=43,observed_submission_journals=88,
   raw_files=len(tf),raw_bytes=10*len(tf),closed_cpu_seconds=base['closed_cpu_seconds']+3,closed_gpu_slice_seconds=base['closed_gpu_slice_seconds']);supplied['cpu-disposition']=b.raw(td)
  final=copy.deepcopy(base);final.update(terminal_jobs=43,submission_journals=88,related_accounting=base['related_accounting']+[tokenrow],journal_pins=tjournals,
   closed_cpu_seconds=base['closed_cpu_seconds']+3,cpu_admission_seconds=base['closed_cpu_seconds']+3603,
   input_pins=dict(predecessor=b.sha(supplied['previous-baseline']),acceptance=b.sha(supplied['cpu-acceptance']),disposition=b.sha(supplied['cpu-disposition'])))
  supplied['baseline']=b.raw(final)
  return supplied,prepared_raw,prepared_pin,plan_pin,paths
 def test_new_full_chain_positive_and_failure_cost_retained(self):
  args=self.fixture();ctx=b.authenticate(*args)
  self.assertEqual(ctx['metadata']['baseline']['terminal_jobs'],43)
  self.assertEqual(ctx['metadata']['baseline']['closed_cpu_seconds'],11386+14+3)
  self.assertEqual(ctx['metadata']['baseline']['closed_gpu_slice_seconds'],1601)
  self.assertTrue(any(r[0]=='35407428' for r in ctx['metadata']['baseline']['related_accounting']))
  self.assertIn(b.COLLECTION_ROOT+'/packet.json',ctx['upstream_pins'])
 def test_new_recovery_rejects_authority_and_capture_drift_before_prepared_decode(self):
  cases=[('operational-amendment',lambda v:v.update(other_stages_unchanged=False)),
   ('cpu-launch-packet',lambda v:v.update(operational_amendment_sha256=H('wrong'))),
   ('cpu-launch-packet',lambda v:v['new_input_files'].pop('no-response-proof.json')),
   ('collection-acceptance',lambda v:v.update(disposition_sha256=H('wrong')))]
  for name,change in cases:
   args=list(self.fixture());v=b.decode(args[0][name]);change(v);args[0][name]=b.raw(v)
   with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):b.authenticate(*args)
 def test_new_recovery_rejects_lost_failure_journals_cost_and_inventory(self):
  cases=[('previous-baseline',lambda v:v['related_accounting'].pop(0)),
   ('previous-baseline',lambda v:v['journal_pins'].pop(next(iter(v['journal_pins'])))),
   ('previous-baseline',lambda v:v.update(closed_cpu_seconds=1)),
   ('collection-inventory',lambda v:v.update(python_source='ROOT="/wrong"\nJOB="99999998"\n')),
   ('collection-disposition',lambda v:v.update(raw_bytes=0))]
  for name,change in cases:
   args=list(self.fixture());v=b.decode(args[0][name]);change(v);args[0][name]=b.raw(v)
   # Rebind independently supplied future references only so the new semantic gate is exercised.
   outer=b.decode(args[0]['cpu-launch-packet'])
   if name=='previous-baseline':outer['baseline_sha256']=b.sha(args[0][name]);outer['new_input_files']['baseline.json']=b.sha(args[0][name])
   args[0]['cpu-launch-packet']=b.raw(outer)
   with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):b.authenticate(*args)
if __name__=='__main__':unittest.main(verbosity=2)
