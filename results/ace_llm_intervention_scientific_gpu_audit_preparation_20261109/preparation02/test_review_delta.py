"""Four NEW saved-metadata regressions, no actual evidence or producer execution."""
import copy,importlib.util,json,os,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu72_review_delta',HERE/'audit_saved_gpu72.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path(os.environ['GPU_AID_DELTA_FIXTURES']).resolve()


def compact_fixture(root,before_dirs=None,after_dirs=None,before_name='inventory_transport.json'):
 root.mkdir();a=root/'one';b=root/'two';a.mkdir();b.mkdir()
 data=b'{"fabricated":true}\n'
 full=dict(files={'launch/gpu/terminal.json':dict(bytes=len(data),sha256=m.sha(data))},directories=['launch','launch/gpu'])
 local=m.selected(full,m.load_oracles()[0]['references/remote_custody4161.py'])
 before=dict(files=full['files'],directories=['launch/gpu','launch'] if before_dirs is None else before_dirs,
  terminal_gate=True,original_hardlink_groups=[],parent_accounting=[],related_accounting=[],journal_pins={},upstream_pins={},upstream_errors={})
 after=copy.deepcopy(before);after['directories']=['launch','launch/gpu'] if after_dirs is None else after_dirs
 receipt=dict(schema='ace-remote-custody-v1',full_inventory=full,local_inventory=local,full_backup_verified=True,full_local_copy=False,
  acceptance=False,retry_allowed=False,backup_kind='ustar',helper_sha256=m.BACKEND_SHA,selection_policy='current-launch-text-and-namespaces-v1',
  source_observation_sha256=m.sha(m.raw(before)),original_hardlink_groups=[],original_before_inventory_sha256=m.sha(m.raw(full)),
  original_after_inventory_sha256=m.sha(m.raw(full)),project_readback_inventory_sha256=m.sha(m.raw(full)),omitted_files={},omitted_bytes=0,
  backup_archive_sha256='a'*64,backup_archive_path='/projects/FABRICATED/raw.tar')
 d=dict(schema='ace-scientific-campaign-gpu-terminal-custody-v1',job_id='1',errors=[],input_pin_mismatches=[],failure_marker_metadata={},
  custody_bytes_verified=True,current_lineage_verified=True,upstream_inputs_verified=True,complete_local_raw_copy=False,
  remote_files=full['files'],remote_directories=full['directories'],remote_custody_receipt_sha256=m.sha(m.raw(receipt)),backup_helper_sha256=m.BACKEND_SHA)
 files={'disposition.json':m.raw(d),'remote-custody.json':m.raw(receipt),before_name:m.raw(dict(returncode=0,stdout=json.dumps(before))),
  'inventory_after_transport.json':m.raw(dict(returncode=0,stdout=json.dumps(after))),'remote/launch/gpu/terminal.json':data}
 for folder in (a,b):
  for n,value in files.items():(folder/n).parent.mkdir(parents=True,exist_ok=True);(folder/n).write_bytes(value)
 seal=dict(copies={str(p):m.aid.inventory(p) for p in (a,b)},identical=True,excludes='custody-inventory.json itself',acceptance=False)
 for p in (a,b):(p/'custody-inventory.json').write_bytes(m.raw(seal))
 roles=dict(disposition='disposition.json',custody='custody-inventory.json',before=before_name,after='inventory_after_transport.json',remote_custody='remote-custody.json')
 cfg=dict(schema='ace-prospective-gpu72-saved-audit-input-v1',job_id='1',helper_sha256=m.BACKEND_SHA,capsules=[str(a),str(b)],
  artifacts={k:dict(path=str(a/n),sha256=m.sha((a/n).read_bytes())) for k,n in roles.items()})
 (root/'config.json').write_bytes(m.raw(cfg))
 return cfg,before,after


def ancestor_fixture():
 mapping={'launch':'cpu','cpu':'token','training':'training','common':'common','failed01':'failed01','failed-token01':'failed-token01',
  'namespace.json':'namespace.json','original-namespace.json':'original-namespace.json'}
 roots={n:'/scratch/FABRICATED/'+n for n in mapping}
 files={};dirs=[]
 for n in mapping:
  member=n if n.endswith('.json') else n+'/packet.json'
  files[member]=dict(bytes=len(n),sha256=m.sha(n.encode()))
  if not n.endswith('.json'):dirs.extend([n,n+'/empty'])
 previous=dict(roots=roots,full_inventory=dict(files=files,directories=sorted(dirs)))
 def rename(n):
  h,sep,t=n.partition('/');return mapping[h]+sep+t
 now=dict(roots={mapping[k]:v for k,v in roots.items()},full_inventory=dict(files={rename(n):v for n,v in files.items()},directories=sorted(rename(n) for n in dirs)))
 now['roots']['launch']='/scratch/FABRICATED/GPU'
 now['full_inventory']['files']['launch/packet.json']=dict(bytes=3,sha256=m.sha(b'GPU'))
 now['full_inventory']['directories'].append('launch')
 for i in (5,6,7):
  name=m.C+f'policy-{i:02d}/attempt.json';now['full_inventory']['files'][name]=dict(bytes=4,sha256=m.sha(b'fake'))
  now['full_inventory']['directories'].extend('/'.join(name.split('/')[:j]) for j in range(1,len(name.split('/'))))
 # These common parent directories already existed in the original CPU campaign.
 for directory in ('launch/cpu','launch/cpu/worker','launch/cpu/worker/artifacts','launch/cpu/worker/artifacts/campaign'):
  previous['full_inventory']['directories'].append(directory)
 previous['full_inventory']['directories']=sorted(set(previous['full_inventory']['directories']))
 now['full_inventory']['directories']=sorted(set(now['full_inventory']['directories']))
 return previous,now


class Delta(unittest.TestCase):
 def setUp(self):self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=ROOT))
 def test_r1_real_collector_roster_and_order_only_observation_delta(self):
  cfg,before,after=compact_fixture(self.root/'actual-roster')
  before_bytes=m.raw(before);view=m.Audit(cfg);view.custody()
  self.assertNotEqual(before['directories'],after['directories'])
  self.assertEqual(m.raw(view.before),before_bytes)
  self.assertEqual(view.receipt['source_observation_sha256'],m.sha(before_bytes))
  self.assertFalse((Path(cfg['capsules'][0])/'inventory_before_transport.json').exists())
  bad,_,_=compact_fixture(self.root/'invented-roster',before_name='inventory_before_transport.json')
  with self.assertRaises((KeyError,ValueError)):m.Audit(bad).custody()
 def test_r2_distinct_cpu_token_roots_full_file_and_empty_directory_preservation(self):
  old,now=ancestor_fixture();m.cpu_ancestor_inventory(old,now)
  self.assertNotEqual(old['full_inventory']['files']['launch/packet.json'],old['full_inventory']['files']['cpu/packet.json'])
  (self.root/'positive.json').write_bytes(m.raw(dict(previous=old,current=now)))
  changes=[('wrong_label',lambda v:v['full_inventory']['files'].update({'token/packet.json':v['full_inventory']['files']['cpu/packet.json']})),
   ('wrong_root',lambda v:v['roots'].update(token=v['roots']['cpu'])),
   ('lost_failure_file',lambda v:v['full_inventory']['files'].pop('failed-token01/packet.json')),
   ('lost_empty_directory',lambda v:v['full_inventory']['directories'].remove('common/empty')),
   ('changed_namespace',lambda v:v['full_inventory']['files']['namespace.json'].update(sha256='0'*64)),
   ('extra_ancestor_file',lambda v:v['full_inventory']['files'].update({'training/extra.json':dict(bytes=1,sha256='0'*64)})),
   ('extra_ancestor_directory',lambda v:v['full_inventory']['directories'].append('token/extra'))]
  for label,mutate in changes:
   value=copy.deepcopy(now);mutate(value);(self.root/(label+'.json')).write_bytes(m.raw(value))
   with self.subTest(case=label),self.assertRaises(ValueError):m.cpu_ancestor_inventory(old,value)
 def test_r3_original_directory_types_duplicates_and_cpu_disposition_normalization(self):
  bads=[{'launch':True,'launch/gpu':True},'launch',['launch','launch/gpu','launch/gpu']]
  for index,bad in enumerate(bads):
   for phase in ('before','after'):
    cfg,_,_=compact_fixture(self.root/(phase+str(index)),**{phase+'_dirs':bad})
    with self.subTest(phase=phase,bad=index),self.assertRaises(ValueError):m.Audit(cfg).custody()
   # Same boundary is called on original CPU disposition remote_directories.
   with self.subTest(cpu_disposition=index),self.assertRaises(ValueError):m.saved_inventory({},bad)
  dirs=['root/z','root'];saved=m.raw(dirs)
  self.assertEqual(m.saved_inventory({},dirs)['directories'],['root','root/z']);self.assertEqual(m.raw(dirs),saved)
 def test_r4_contiguous_unregistered_prefix_signals_and_original_clock_boundaries(self):
  session=dict(policy_id='frozen_llm',registered_pid=123,positive_generations=2,started_monotonic=100.,first_verified_monotonic=110.,closed_monotonic=125.,error=None,process_group_closed=True,post_exit_gpu_empty=True)
  def row(now,pid,n,first,last,sample):return dict(policy_id='frozen_llm',registered_pid=pid,monotonic=now,positive_generations=n,
   gpu_process_sample=sample,service_elapsed_seconds=now-100.,outer_elapsed_seconds=now-90.,first_verified_monotonic=first,
   last_gpu_monotonic=last,last_generation_monotonic=last,uncertainties=[] if sample is None or sample['owned_rows'] else ['no current positive owned-process GPU observation'])
  prefix=[row(t,None,0,None,None,None) for t in (101.,102.,103.)]
  telemetry=prefix+[row(104.,123,0,None,None,dict(returncode=0,owned_rows=[])),
   row(110.,123,1,110.,110.,dict(returncode=0,owned_rows=['123, GPU-fake, 100'])),
   row(120.,123,2,110.,120.,dict(returncode=0,owned_rows=['123, GPU-fake, 110']))]
  post=dict(registered_pid=123,process_group_closed=True,monotonic=124.,sample=dict(returncode=0,owned_rows=[]))
  facts=m.telemetry_facts(telemetry,session,post,90.,130.,150.,121.);self.assertEqual(facts['samples'],6)
  (self.root/'positive.json').write_bytes(m.raw(dict(telemetry=telemetry,session=session,post=post)))
  changes=[('positive_prefix',lambda t:t[1].update(positive_generations=1)),
   ('first_claim',lambda t:t[1].update(first_verified_monotonic=110.)),
   ('last_gpu_claim',lambda t:t[1].update(last_gpu_monotonic=102.)),
   ('last_progress_claim',lambda t:t[1].update(last_generation_monotonic=102.)),
   ('prefix_clock_reset',lambda t:t[1].update(outer_elapsed_seconds=0.)),
   ('registered_without_sample',lambda t:t[3].update(gpu_process_sample=None)),
   ('return_to_unregistered',lambda t:t.insert(4,row(105.,None,0,None,None,None))),
   ('wrong_owner',lambda t:t[4].update(registered_pid=456)),
   ('missing_after_work',lambda t:t[5].update(gpu_process_sample=None)),
   ('stale_progress',lambda t:t[5].update(last_generation_monotonic=29.))]
  for label,mutate in changes:
   value=copy.deepcopy(telemetry);mutate(value);(self.root/(label+'.json')).write_bytes(m.raw(value))
   with self.subTest(case=label),self.assertRaises(ValueError):m.telemetry_facts(value,session,post,90.,130.,150.,121.)
  late=copy.deepcopy(session);late.update(first_verified_monotonic=220.,closed_monotonic=230.)
  with self.assertRaises(ValueError):m.telemetry_facts(telemetry,late,post,90.,240.,250.,240.)

if __name__=='__main__':
 ROOT.mkdir(mode=0o700)
 unittest.main(verbosity=2)
