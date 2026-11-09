"""New fabricated compact coordinator wiring only. No transport, project or model execution."""
from pathlib import Path
import ast,copy,hashlib,shutil,tempfile,types,unittest
from unittest.mock import patch
import compact_inputs as C

HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/ace-downstream-compact-adapter-6r23szeq')
BACKEND=b'''from pathlib import Path
import hashlib,json
def bind_source(source,expected):
 assert hashlib.sha256(source).hexdigest()==expected
def verify_backup(*,receipt,observation,helper_path,log_dir):
 Path(log_dir).mkdir()
 return dict(full_backup_verified=True,full_inventory=receipt['full_inventory'],local_inventory=receipt['local_inventory'],receipt_sha256=hashlib.sha256((json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\\n').encode()).hexdigest(),backup_archive_sha256=receipt['backup_archive_sha256'],helper_sha256=receipt['helper_sha256'],acceptance=False)
def fetch_exact(**kwargs):
 raise AssertionError('no real or fake fetch permitted in bind fixture')
'''

def write(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def seal(a,b):
 view=C.selected_inventory(a);assert view==C.selected_inventory(b)
 raw=C.canonical(dict(copies={str(a):view,str(b):view},identical=True,acceptance=False,excludes='custody-inventory.json itself'))
 for p in (a,b):write(p/'custody-inventory.json',raw)
 return raw

class Wiring(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='fake-compact-wiring-');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name).resolve();self.source=HERE/'compact_inputs.py'
 def fixture(self):
  r=self.root;training=r/'training';mirror=r/'training-mirror'
  failed_view={'files':{'original.log':{'bytes':6,'sha256':C.sha(b'failed')}},'directories':[]}
  failed=dict(copies={'fake-first':failed_view,'fake-second':failed_view},identical=True,acceptance=False,excludes='custody-inventory.json itself')
  failed_raw=C.canonical(failed)
  for p in (training,mirror):
   write(p/'upstream-token01-custody/original.log',b'failed')
   write(p/'upstream-token01-custody/custody-inventory.json',failed_raw)
   write(p/'disposition.json',b'{}\n');write(p/'inventory_transport.json',b'{"transport": true}\n')
  train_seal=seal(training,mirror)
  ref=C.canonical(dict(path=str(training),custody_inventory_sha256=C.sha(train_seal),
   disposition_sha256=C.sha(b'{}\n'),raw_inventory_sha256=C.sha(b'{"transport": true}\n'),
   recursive_copy_created=False,original_preserved=True))
  a=r/'cpu';b=r/'cpu-mirror'
  for p in (a,b):write(p/'remote/launch/inputs/saved.json',b'opaque-no-private-decoding')
  selected=C.selected_inventory(a/'remote');full=copy.deepcopy(selected)
  full['files']['launch/inputs/checkpoint.pt']={'sha256':'a'*64,'bytes':128}
  obs=dict(**full,terminal_gate=True,parent_accounting=['123','COMPLETED'],related_accounting=[['123','COMPLETED']],journal_pins={})
  before=C.canonical(dict(returncode=0,stdout=C.canonical(obs).decode()));after=before
  receipt=dict(schema='ace-remote-custody-v1',full_backup_verified=True,full_local_copy=False,backup_kind='ustar',
   selection_policy='current-launch-text-and-namespaces-v1',acceptance=False,retry_allowed=False,
   helper_sha256=C.sha(BACKEND),roots={'launch':'/scratch/fake-cpu'},full_inventory=full,local_inventory=selected,
   local_root=str(a/'remote'),omitted_files={'launch/inputs/checkpoint.pt':full['files']['launch/inputs/checkpoint.pt']},
   omitted_bytes=128,backup_archive_sha256='b'*64)
  receipt_raw=C.canonical(receipt)
  disp=dict(custody_mode='curc_projects_full_backup_with_compact_local_evidence',complete_local_raw_copy=False,
   remote_custody_receipt_sha256=C.sha(receipt_raw),backup_helper_sha256=C.sha(BACKEND),remote_files=full['files'],remote_directories=full['directories'])
  for p in (a,b):
   for n,raw in {'remote-custody.json':receipt_raw,'disposition.json':C.canonical(disp),'inventory_transport.json':before,
     'inventory_after_transport.json':after,'predecessor-reference.json':ref}.items():write(p/n,raw)
  cpu_seal=seal(a,b)
  authority=dict(schema='ace-scientific-remote-custody-amendment-v1',status='adopted',scope='evidence_storage_only',
   freeze_sha256='f'*64,source_commit='c'*40,scientific_inputs_unchanged=True,originals_preserved=True,
   full_remote_verification_required=True,backup_kind='ustar',required_findings=0,review_sha256=C.sha(b'authority review'),backend_sha256=C.sha(BACKEND))
  supplied={'storage-authority.json':C.canonical(authority),'storage-review.md':b'authority review','remote_custody.py':BACKEND,
   'storage-backend-review.md':b'backend review','storage-receipt.json':receipt_raw,'storage-seal.json':cpu_seal,
   'storage-after.json':after,'storage-predecessor.json':ref}
  members={key:dict(path=str(r/name),sha256=C.sha(supplied[name])) for key,name in C.STORAGE_FILES.items()}
  members['backend'].update(source_commit='d'*40,repository_path='outer/remote_custody.py')
  members['backend_review']['repository_path']='outer/review.md'
  supplied['storage-config.json']=C.canonical(dict(schema='ace-compact-downstream-inputs-v1',predecessor='cpu',members=members))
  ctx=dict(metadata={'freeze':{'source_commit':'c'*40},'cpu-disposition':disp,
   'training-acceptance':{'custody_inventory_sha256':C.sha(train_seal)},'token-failure-custody-inventory':failed},
   pins={'freeze':'f'*64,'cpu-disposition':C.sha(C.canonical(disp)),'cpu-inventory':C.sha(before),
     'training-disposition':C.sha(b'{}\n'),'training-inventory':C.sha(b'{"transport": true}\n'),
     'token-failure-custody-inventory':C.sha(failed_raw)},observations={'cpu':obs})
  return ctx,supplied,a,b,training
 def bind(self,ctx,supplied,a,b,git=None):
  backend=supplied['remote_custody.py']
  original=tempfile.mkdtemp
  with patch.object(tempfile,'mkdtemp',side_effect=lambda **kwargs:original(prefix='operation-',dir=self.root)):
   return C.bind(ctx,supplied,predecessor='cpu',capsule=a,mirror=b,expected_roots={'launch':'/scratch/fake-cpu'},
    committed=git or (lambda commit,path:backend if path.endswith('.py') else b'backend review'),helper_path='FAKE-NO-REMOTE')
 def test_bind_complete_selected_capsules_and_original_failure_chain(self):
  ctx,s,a,b,training=self.fixture();gate=self.bind(ctx,s,a,b)
  self.assertEqual(gate.ancestry[0]['custody_inventory_sha256'],ctx['metadata']['training-acceptance']['custody_inventory_sha256'])
  self.assertTrue(gate.ancestry[0]['all_original_named_bytes_verified'])
  self.assertFalse((a/'upstream-training-custody').exists())
  data=gate.capture('launch/inputs/saved.json',gate.full['files']['launch/inputs/saved.json']['sha256'])
  self.assertEqual(gate.capture('launch/inputs/saved.json',C.sha(data)),data)
  C.recheck_capsules(gate)
 def test_wrong_source_commit_bytes_and_changed_ancestor_fail_closed(self):
  ctx,s,a,b,training=self.fixture()
  with self.assertRaisesRegex(ValueError,'commit bytes'):self.bind(ctx,s,a,b,git=lambda *_:b'wrong')
  (training/'upstream-token01-custody/original.log').write_bytes(b'CHANGED')
  with self.assertRaisesRegex(ValueError,'missing/changed'):self.bind(ctx,s,a,b)
 def test_parameterized_backend_requires_adopted_independent_pin(self):
  ctx,s,a,b,training=self.fixture()
  changed=BACKEND+b'\n# separately pinned fabricated backend revision\n'
  config=C.decode(s['storage-config.json']);config['members']['backend']['sha256']=C.sha(changed)
  s['remote_custody.py']=changed;s['storage-config.json']=C.canonical(config)
  with self.assertRaisesRegex(ValueError,'backend source'):self.bind(ctx,s,a,b)
 def test_outer_source_byte_and_metadata_contracts_stay_disjoint(self):
  for stage in ('gpu','finalizer'):
   old=ast.parse((ROOT/'predecessor'/stage/'build_packet.py').read_bytes());new=ast.parse((HERE/stage/'build_packet.py').read_bytes())
   funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
   of,nf=funcs(old),funcs(new)
   for name in ('authenticate','make_packet','coordinator_locks'):
    self.assertEqual(of[name],nf[name],(stage,name))
   if stage=='finalizer':self.assertEqual(of['job_script'],nf['job_script'])
   oldlaunch=ast.parse((ROOT/'predecessor'/stage/'launch_once.py').read_bytes())
   newlaunch=ast.parse((HERE/stage/'launch_once.py').read_bytes())
   constants=lambda tree:{n.targets[0].id:ast.dump(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
   self.assertEqual(constants(oldlaunch),constants(newlaunch)) # remote once-only protocol unchanged
   self.assertEqual((ROOT/'predecessor'/stage/'monitor_once.py').read_bytes(),(HERE/stage/'monitor_once.py').read_bytes())
   oldcollector=ast.parse((ROOT/'predecessor'/stage/('pull_terminal.py' if stage=='gpu' else 'collector/pull_terminal.py')).read_bytes())
   newcollector=ast.parse((HERE/stage/'pull_terminal.py').read_bytes())
   oc,nc=funcs(oldcollector),funcs(newcollector)
   for name in ('resources','compare_observations','accounting_extension','validate_remote_intent'):
    self.assertEqual(oc[name],nc[name],(stage,name))
   for name in ({'upstream_disagreements','public_gpu_evidence'} if stage=='gpu' else {'preserved_roots'}):
    if name in oc:self.assertEqual(oc[name],nc[name],(stage,name))
   self.assertEqual(constants(oldcollector)['REMOTE_OBSERVATION'],constants(newcollector)['REMOTE_OBSERVATION'])
   module=types.ModuleType('fake-coordinator');module.__file__=str(HERE/stage/'build_packet.py')
   exec(compile((HERE/stage/'build_packet.py').read_bytes(),module.__file__,'exec'),vars(module))
   self.assertEqual(module.compact_adapter().sha((HERE/stage/'compact_inputs.py').read_bytes()),module.COMPACT_ADAPTER_SHA)
   self.assertEqual(len(module.CPU_ACCEPTANCE_KEYS),16)

class Transfer(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='fake-compact-transfer-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
  self.a=self.root/'a';self.b=self.root/'b';self.a.mkdir();self.b.mkdir();self.saved={}
  self.payload={'launch/cpu/numeric-calls/call/input.utf8':b'opaque utf8 request', 'launch/inputs/a.json':b'opaque json'}
  dirs=sorted({str(p) for n in self.payload for p in Path(n).parents if str(p)!='.'})
  self.selected={'files':{n:{'bytes':len(v),'sha256':C.sha(v)} for n,v in self.payload.items()},'directories':dirs}
  self.full=copy.deepcopy(self.selected);self.full['files']['launch/inputs/a.pt']={'bytes':9,'sha256':'e'*64}
  self.before=dict(**self.full,terminal_gate=True);self.roots={'launch':'/scratch/fake'};self.calls=0;self.fail=False;self.wrong=False
 def normalized(self,obs):return obs
 def selected_inventory(self,obs):return self.selected,{}
 def collect_backup(self,**kw):
  self.calls+=1;Path(kw['log_dir']).mkdir();write(Path(kw['log_dir'])/'intent.json',b'{}')
  for n,data in self.payload.items():write(Path(kw['local_root'])/n,data)
  if self.fail:raise RuntimeError('fabricated partial backup')
  full=copy.deepcopy(self.full)
  if self.wrong:full['files']['launch/inputs/a.pt']['bytes']=8
  return dict(schema='ace-remote-custody-v1',full_backup_verified=True,acceptance=False,retry_allowed=False,
   full_local_copy=False,backup_kind='ustar',helper_sha256='f'*64,roots=self.roots,backup_root=kw['backup_root'],
   local_root=str(kw['local_root']),selection_policy='current-launch-text-and-namespaces-v1',full_inventory=full,local_inventory=self.selected,
   omitted_files={'launch/inputs/a.pt':self.full['files']['launch/inputs/a.pt']},omitted_bytes=9)
 def run_transfer(self):
  def save(n,v):
   self.saved[n]=v
   for root in (self.a,self.b):write(root/n,C.canonical(v))
  return C.collect_selected(backend=self,helper_sha256='f'*64,before=self.before,roots=self.roots,
   backup_root='/projects/paco0228/ACE_evidence/fake',external=self.a,local=self.b,helper_path='FAKE',
   save=save,observe=lambda n:copy.deepcopy(self.before),compare=lambda a,b:self.assertEqual(a,b),copy_file=shutil.copyfile)
 def test_complete_map_utf8_two_selected_copies_no_local_binary(self):
  actual,after,receipt=self.run_transfer();self.assertEqual(actual,self.full)
  self.assertEqual(C.selected_inventory(self.a/'remote'),C.selected_inventory(self.b/'remote'))
  self.assertFalse((self.b/'remote/launch/inputs/a.pt').exists());self.assertEqual(self.calls,1)
  self.assertFalse(receipt['full_local_copy']);self.assertFalse(receipt['acceptance'])
  for n in self.payload:self.assertNotEqual((self.a/'remote'/n).stat().st_ino,(self.b/'remote'/n).stat().st_ino)
 def test_existing_destination_and_fullmap_lie_reject(self):
  (self.b/'remote').mkdir()
  with self.assertRaisesRegex(ValueError,'absent'):self.run_transfer()
  self.assertEqual(self.calls,0);(self.b/'remote').rmdir();self.wrong=True
  with self.assertRaisesRegex(ValueError,'receipt binding'):self.run_transfer()
  self.assertEqual(self.calls,1);self.assertTrue((self.a/'backup-operations/intent.json').exists())
 def test_partial_failure_retained_and_cannot_retry(self):
  self.fail=True
  with self.assertRaises(RuntimeError):self.run_transfer()
  self.assertTrue((self.a/'backup-operations/intent.json').exists());self.assertNotIn('remote-custody.json',self.saved)
  with self.assertRaisesRegex(ValueError,'absent'):self.run_transfer()
  self.assertEqual(self.calls,1)

if __name__=='__main__':unittest.main(verbosity=2)
