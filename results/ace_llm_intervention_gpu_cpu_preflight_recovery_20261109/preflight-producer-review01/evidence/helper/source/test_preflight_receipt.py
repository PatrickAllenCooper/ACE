"""NEW artificial receipt/context tests only. No file scans or scientific payloads."""
import copy,hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pure_preflight',HERE/'llm_intervention_cpu_preflight_receipt.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path(os.environ['PREFLIGHT_FIXTURES']).resolve()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def pin(label):return sha(('FABRICATED:'+label).encode())
def inventory(files):
 return dict(files=files,directories=sorted({'/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/')))}))
def fixture():
 reads={n:dict(bytes=3,sha256=pin(n)) for n in ('campaign.json','source-plan-pins.json','start.json')}
 for pi in range(5):
  for n in ('attempt.json','attempt.candidate.json'):reads[f'policy-{pi:02d}/'+n]=dict(bytes=3,sha256=pin(str(pi)+n))
  for wi in range(24):reads[f'policy-{pi:02d}/world-{wi:02d}/arm-receipt.json']=dict(bytes=3,sha256=pin(f'{pi}:{wi}'))
 cpu=inventory({'worker/artifacts/cpu-phase/campaign-bundle.json':dict(bytes=17,sha256=pin('opaque-bundle')),
                'worker/artifacts/cpu-phase/terminal.json':dict(bytes=23,sha256=pin('terminal'))})
 inv=dict(cpu_output=cpu,campaign=inventory(copy.deepcopy(reads)))
 # Explicit empty directory in the independently authenticated roster.
 inv['campaign']['directories']=sorted(inv['campaign']['directories']+['policy-00/empty-retained'])
 context=dict(schema='ace-gpu-cpu-preflight-context-v1',control_root='/scratch/FAKE_GPU02',gpu_output='/scratch/FAKE_GPU01/gpu',
  failed_gpu01_root='/scratch/FAKE_GPU01',cpu_roots=dict(cpu_output='/scratch/FAKE_CPU/cpu',campaign='/scratch/FAKE_CPU/cpu/worker/artifacts/campaign'),
  lineage={k:pin(k) for k in m.LINEAGE_KEYS},baseline=dict(sha256=pin('baseline'),terminal_jobs=47,submission_journals=96),
  proof_pins={k:pin(k) for k in m.PROOF_KEYS},source_files_sha256=pin('sources'),model_files_sha256=pin('models'),
  resources_sha256=pin('resources'),limits_sha256=pin('limits'),cpu_inventory_sha256=None,consumed_read_set_sha256=None)
 expected=dict(schema='ace-gpu-cpu-preflight-expected-v1',execution_context=context,execution_context_sha256=None,
  cpu_inventories=inv,consumed_read_set=reads,receipt_not_after_utc='2026-11-09T23:59:59.000001Z')
 receipt=dict(schema='ace-gpu-cpu-preflight-receipt-v1',phase='CPU-prelaunch-complete',execution_context_sha256=None,
  inventory_before=None,inventory_after=None,consumed_read_set=None,
  absent_paths=sorted(['/scratch/FAKE_GPU01/gpu']+[context['cpu_roots']['campaign']+f'/policy-{i:02d}/service' for i in range(5)]),
  started_utc='2026-11-01T01:00:00Z',finished_utc='2026-11-01T01:10:00Z')
 return rebind(receipt,expected)
def rebind(receipt,expected):
 context=expected['execution_context'];inv=expected['cpu_inventories'];reads=expected['consumed_read_set']
 full=dict(cpu_roots=context['cpu_roots'],cpu_output_excluded_directory='worker/artifacts/campaign',inventories=inv)
 fs=dict(canonical_sha256=sha(canonical(full)),file_count=sum(len(v['files']) for v in inv.values()),
         directory_count=sum(len(v['directories']) for v in inv.values()),total_bytes=sum(row['bytes'] for v in inv.values() for row in v['files'].values()))
 rs=dict(canonical_sha256=sha(canonical(reads)),file_count=len(reads),total_bytes=sum(v['bytes'] for v in reads.values()))
 context['cpu_inventory_sha256']=fs['canonical_sha256'];context['consumed_read_set_sha256']=rs['canonical_sha256']
 expected['execution_context_sha256']=receipt['execution_context_sha256']=sha(canonical(context))
 receipt.update(inventory_before=copy.deepcopy(fs),inventory_after=copy.deepcopy(fs),consumed_read_set=copy.deepcopy(rs))
 return receipt,expected

class PureReceiptTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=ROOT));self.receipt,self.expected=fixture()
  (self.root/'receipt.json').write_bytes(raw(self.receipt));(self.root/'expected.json').write_bytes(raw(self.expected))
 def check(self,receipt=None,expected=None):
  r=raw(self.receipt if receipt is None else receipt);e=raw(self.expected if expected is None else expected)
  return m.validate_receipt(r,sha(r),e,sha(e))
 def reject(self,label,r,e):
  (self.root/(label+'-receipt.json')).write_bytes(raw(r));(self.root/(label+'-expected.json')).write_bytes(raw(e))
  with self.subTest(case=label),self.assertRaises((ValueError,UnicodeError)):self.check(r,e)
 def test_detached_context_and_queue_delay_with_scoped_not_acceptance_return(self):
  original_r=raw(self.receipt);original_e=raw(self.expected)
  result=m.validate_receipt(original_r,sha(original_r),original_e,sha(original_e))
  self.assertEqual(result['consumed_read_set']['file_count'],133)
  self.assertTrue(result['full_external_postcustody_required']);self.assertFalse(result['acceptance'])
  self.assertFalse(result['continuous_immutability_proven']);self.assertFalse(result['gpu_execution_qualified'])
  # The producer finished eight days before the independent UTC bound: no queue TTL.
  self.assertEqual(result['finished_utc'],'2026-11-01T01:10:00Z')
  self.expected['execution_context']['lineage'].clear();result['absent_paths'].clear()
  again=m.validate_receipt(original_r,sha(original_r),original_e,sha(original_e))
  self.assertEqual(len(again['absent_paths']),6)
  with self.assertRaises(ValueError):m.validate_receipt(bytearray(original_r),sha(original_r),original_e,sha(original_e))
 def test_strict_byte_pins_keys_types_and_duplicate_nonfinite_json(self):
  r=raw(self.receipt);e=raw(self.expected)
  for label,rr,rp,ee,ep in [('bad_receipt_pin',r,'0'*64,e,sha(e)),('bad_expected_pin',r,sha(r),e,'0'*64),
   ('duplicate',r.rstrip()[:-1]+b',"phase":"CPU-prelaunch-complete"}',None,e,sha(e)),
   ('nonfinite',r.replace(b'"file_count": 133',b'"file_count": NaN'),None,e,sha(e))]:
   with self.subTest(case=label),self.assertRaises(ValueError):m.validate_receipt(rr,sha(rr) if rp is None else rp,ee,ep)
  for label,mutate in [('extra_future_packet',lambda r,e:e.update(packet_sha256=pin('future'))),
   ('missing_lineage',lambda r,e:e['execution_context']['lineage'].pop('recovery_authority_sha256')),
   ('float_baseline',lambda r,e:e['execution_context']['baseline'].update(terminal_jobs=47.0)),
   ('boolean_count',lambda r,e:r['consumed_read_set'].update(file_count=True)),
   ('monotonic_extra',lambda r,e:r.update(started_monotonic=1.0))]:
   rr=copy.deepcopy(self.receipt);ee=copy.deepcopy(self.expected);mutate(rr,ee);self.reject(label,rr,ee)
 def test_full_before_after_maps_directories_and_no_boolean_verification(self):
  for label,mutate in [('boolean_proof',lambda r,e:r.update(inventory_before={'verified':True})),
   ('missing_empty_directory',lambda r,e:e['cpu_inventories']['campaign']['directories'].remove('policy-00/empty-retained')),
   ('directory_dict',lambda r,e:e['cpu_inventories']['campaign'].update(directories={n:True for n in e['cpu_inventories']['campaign']['directories']})),
   ('after_count',lambda r,e:r['inventory_after'].update(directory_count=r['inventory_after']['directory_count']-1)),
   ('after_sha',lambda r,e:r['inventory_after'].update(canonical_sha256='0'*64)),
   ('CPU_service',lambda r,e:e['cpu_inventories']['campaign']['directories'].append('policy-00/service'))]:
   rr=copy.deepcopy(self.receipt);ee=copy.deepcopy(self.expected);mutate(rr,ee);self.reject(label,rr,ee)
  rr,ee=fixture();ee['cpu_inventories']['campaign']['directories']=sorted(ee['cpu_inventories']['campaign']['directories']+['policy-00/service'])
  rr,ee=rebind(rr,ee);self.reject('self_consistent_forbidden_service',rr,ee)
 def test_exact133_and_observed_size_class_and_16mib_readset_boundary(self):
  rr,ee=fixture();name='campaign.json'
  for total in (5548472,16*1024**2):
   ee['consumed_read_set'][name]['bytes']=total-sum(v['bytes'] for k,v in ee['consumed_read_set'].items() if k!=name)
   ee['cpu_inventories']['campaign']['files'][name]=copy.deepcopy(ee['consumed_read_set'][name]);rr,ee=rebind(rr,ee)
   self.assertEqual(self.check(rr,ee)['consumed_read_set']['total_bytes'],total)
  ee['consumed_read_set'][name]['bytes']+=1;ee['cpu_inventories']['campaign']['files'][name]=copy.deepcopy(ee['consumed_read_set'][name]);rr,ee=rebind(rr,ee)
  self.reject('16MiB_plus_one',rr,ee)
  for label,mutate in [('missing_arm',lambda e:e['consumed_read_set'].pop('policy-04/world-23/arm-receipt.json')),
   ('extra_member',lambda e:e['consumed_read_set'].update({'private/outcome.json':dict(bytes=1,sha256=pin('extra'))})),
   ('unrelated_pin',lambda e:e['consumed_read_set']['policy-00/attempt.json'].update(sha256=pin('unrelated')))]:
   rr,ee=fixture();mutate(ee);rr,ee=rebind(rr,ee);self.reject(label,rr,ee)
 def test_utc_absent_output_roots_and_execution_context_pin(self):
  cases=[('backwards',lambda r,e:r.update(started_utc='2026-11-02T01:00:00Z')),
   ('later_than_anchor',lambda r,e:r.update(finished_utc='2026-11-10T01:00:00Z')),
   ('offset',lambda r,e:r.update(finished_utc='2026-11-01T01:10:00+01:00')),
   ('invalid_utc',lambda r,e:r.update(finished_utc='2026-11-01T01:10:60Z')),
   ('absent_path_missing',lambda r,e:r['absent_paths'].pop()),
   ('new_output_relocation',lambda r,e:e['execution_context'].update(gpu_output='/scratch/FAKE_GPU02/gpu')),
   ('overlap_control',lambda r,e:e['execution_context'].update(control_root='/scratch/FAKE_CPU/cpu/child')),
   ('lineage_substitution',lambda r,e:e['execution_context']['lineage'].update(cpu_acceptance_sha256=pin('wrong')))]
  for label,mutate in cases:
   rr=copy.deepcopy(self.receipt);ee=copy.deepcopy(self.expected);mutate(rr,ee);self.reject(label,rr,ee)

if __name__=='__main__':
 ROOT.mkdir(mode=0o700)
 unittest.main(verbosity=2)
