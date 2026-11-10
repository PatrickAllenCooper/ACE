"""NEW fabricated metadata boundaries only; no producer or actual terminal calls."""
import contextlib,copy,hashlib,importlib.util,json,os,shutil,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu72_aid',HERE/'audit_saved_gpu72.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path(os.environ['AUDIT_FIXTURES']);ROOT.mkdir(exist_ok=False)
class Cases(unittest.TestCase):
 def test_compact_actual_collector_schema_opaque_members_and_128m_role(self):
  case=ROOT/'compact';case.mkdir();a=case/'copy1';b=case/'copy2';a.mkdir();b.mkdir()
  payload=b'{"artificial":true}\n';full=dict(files={'launch/gpu/terminal.json':dict(bytes=len(payload),sha256=m.sha(payload)),
      'launch/inputs/opaque.pt':dict(bytes=8,sha256='a'*64),'cpu/private.npy':dict(bytes=8,sha256='b'*64)},directories=['cpu','launch','launch/gpu','launch/inputs'])
  oracle=m.load_oracles()[0]['references/remote_custody4161.py'];local=m.selected(full,oracle)
  before=dict(files=full['files'],directories=list(reversed(full['directories'])),terminal_gate=True,original_hardlink_groups=[],parent_accounting=[],related_accounting=[],journal_pins={},upstream_pins={},upstream_errors={})
  omitted={k:v for k,v in full['files'].items() if k not in local['files']}
  receipt=dict(schema='ace-remote-custody-v1',full_inventory=full,local_inventory=local,full_backup_verified=True,full_local_copy=False,acceptance=False,retry_allowed=False,
    backup_kind='ustar',helper_sha256=m.BACKEND_SHA,selection_policy='current-launch-text-and-namespaces-v1',source_observation_sha256=m.sha(m.raw(before)),original_hardlink_groups=[],
    original_before_inventory_sha256=m.sha(m.raw(full)),original_after_inventory_sha256=m.sha(m.raw(full)),project_readback_inventory_sha256=m.sha(m.raw(full)),
    omitted_files=omitted,omitted_bytes=16,backup_archive_sha256='c'*64,backup_archive_path='/projects/toy/raw.tar')
  d=dict(schema='ace-scientific-campaign-gpu-terminal-custody-v1',job_id='1',errors=[],input_pin_mismatches=[],failure_marker_metadata={},custody_bytes_verified=True,current_lineage_verified=True,upstream_inputs_verified=True,
    complete_local_raw_copy=False,remote_files=full['files'],remote_directories=full['directories'],remote_custody_receipt_sha256=m.sha(m.raw(receipt)),backup_helper_sha256=m.BACKEND_SHA)
  data={'disposition.json':m.raw(d),'remote-custody.json':m.raw(receipt),'inventory_before_transport.json':m.raw(dict(returncode=0,stdout=json.dumps(before))),
        'inventory_after_transport.json':m.raw(dict(returncode=0,stdout=json.dumps(before))),'remote/launch/gpu/terminal.json':payload}
  for root in (a,b):
   for n,raw in data.items():(root/n).parent.mkdir(parents=True,exist_ok=True);(root/n).write_bytes(raw)
  seal=dict(copies={str(p):m.aid.inventory(p) for p in (a,b)},identical=True,excludes='custody-inventory.json itself',acceptance=False)
  for p in (a,b):(p/'custody-inventory.json').write_bytes(m.raw(seal))
  roles=dict(disposition='disposition.json',custody='custody-inventory.json',before='inventory_before_transport.json',after='inventory_after_transport.json',remote_custody='remote-custody.json')
  cfg=dict(schema='ace-prospective-gpu72-saved-audit-input-v1',job_id='1',helper_sha256=m.BACKEND_SHA,capsules=[str(a),str(b)],artifacts={k:dict(path=str(a/n),sha256=m.sha((a/n).read_bytes())) for k,n in roles.items()})
  view=m.Audit(cfg);view.custody();self.assertFalse(view.facts['custody']['full_local_copy']);self.assertEqual(view.h('launch/inputs/opaque.pt'),'a'*64)
  self.assertFalse(view.opaque['launch/inputs/opaque.pt']['locally_hashed'])
  with self.assertRaises(ValueError):view.b('launch/inputs/opaque.pt')
  bad=copy.deepcopy(cfg);bad['capsules']=[str(a),str(a)]
  with self.assertRaises(ValueError):m.Audit(bad).custody()
  (b/'remote/launch/gpu/terminal.json').write_bytes(b'changed')
  with self.assertRaises(ValueError):m.Audit(cfg).custody()
  # New reader integration: the reported actual receipt SIZE, wholly fabricated bytes.
  large=case/'full-custody-role.json'
  with large.open('xb') as f:f.write(b'{}');f.write(b' '*(33829045-2))
  view.c['artifacts']['large']=dict(path=str(large),sha256=hashlib.sha256(large.read_bytes()).hexdigest())
  self.assertEqual(len(view.artifact('large')),33829045)
  self.assertEqual(m.aid.decode(view.artifact('large')), {})

 def test_new_telemetry_original_clock_ownership_and_cleanup_boundaries(self):
  session=dict(policy_id='frozen_llm',registered_pid=123,positive_generations=2,started_monotonic=100.,first_verified_monotonic=110.,closed_monotonic=125.,error=None,process_group_closed=True,post_exit_gpu_empty=True)
  def row(now,n,first,last,owned):return dict(policy_id='frozen_llm',registered_pid=123,monotonic=now,positive_generations=n,
    gpu_process_sample=dict(returncode=0,owned_rows=owned),service_elapsed_seconds=now-100.,outer_elapsed_seconds=now-90.,
    first_verified_monotonic=first,last_gpu_monotonic=last,last_generation_monotonic=last,uncertainties=[] if owned else ['no current positive owned-process GPU observation'])
  telemetry=[row(101.,0,None,None,[]),row(110.,1,110.,110.,['123, GPU-toy, 100']),row(120.,2,110.,120.,['123, GPU-toy, 110'])]
  post=dict(registered_pid=123,process_group_closed=True,monotonic=124.,sample=dict(returncode=0,owned_rows=[]))
  facts=m.telemetry_facts(telemetry,session,post,90.,130.,150.,121.)
  self.assertEqual(facts['max_owned_memory_mib'],110.);self.assertIsNone(facts['utilization_percent'])
  # Cleanup may finish after service WORK deadline while before the original FINAL.
  for label,change in [('wrong_pid',lambda t:t[1]['gpu_process_sample'].update(owned_rows=['456, GPU-toy, 100'])),
      ('missing_signal',lambda t:t[1]['gpu_process_sample'].update(owned_rows=[])),
      ('future_progress',lambda t:t[2].update(last_generation_monotonic=121.)),
      ('late_work',lambda t:t[2].update(monotonic=122.,service_elapsed_seconds=22.,outer_elapsed_seconds=32.))]:
   with self.subTest(label=label):
    t=copy.deepcopy(telemetry);change(t)
    with self.assertRaises(ValueError):m.telemetry_facts(t,session,post,90.,130.,150.,121.)
  bad=copy.deepcopy(post);bad['sample']['owned_rows']=['123, GPU-toy, 1']
  with self.assertRaises(ValueError):m.telemetry_facts(telemetry,session,bad,90.,130.,150.,121.)

 def test_new_opaque_feedback_chain_and_generation_roster(self):
  history=b'[{"secret_value":[1e-100,"not decoded",{"hidden":-7}]}]'
  self.assertEqual(m.opaque_compact(b' { "history" : '+history+b', "literal" : " spaced value " } '),b'{"history":'+history+b',"literal":" spaced value "}')
  previous=None;lines=[]
  for payload in ({'event':'bind','world_id':1},{'event':'outcome','values':{'hidden':[1,2,3]}},{'event':'closed'}):
   value=dict(payload,previous_sha256=previous);value['sha256']=m.digest(value);previous=value['sha256'];lines.append(m.canonical(value))
  self.assertEqual(len(m.world_chain(b'\n'.join(lines)+b'\n')),3)
  bad=lines.copy();bad[1]=bad[1].replace(b'[1,2,3]',b'[1,2,4]')
  with self.assertRaises(ValueError):m.world_chain(b'\n'.join(bad)+b'\n')
  full={'w/generation-0000.json':{},'w/generation-0001.json':{}}
  self.assertEqual(set(m.rows_by_ordinal(full,'w/',r'generation-(\d{4})\.json',2)),{0,1})
  with self.assertRaises(ValueError):m.rows_by_ordinal({**full,'w/generation-0002.json':{}},'w/',r'generation-(\d{4})\.json',2)
  with self.assertRaises(ValueError):m.rows_by_ordinal({'w/generation-0001.json':{}},'w/',r'generation-(\d{4})\.json',2)

if __name__=='__main__':unittest.main(verbosity=2)
