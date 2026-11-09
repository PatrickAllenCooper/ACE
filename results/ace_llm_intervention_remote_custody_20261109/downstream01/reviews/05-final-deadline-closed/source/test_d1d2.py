"""Only new D1/D2 fabricated guards and evidence paths. No real backend or main."""
from pathlib import Path
import ast,contextlib,copy,os,tempfile,types,unittest
from unittest.mock import patch
import compact_inputs as C
import test_compact_wiring as F
HERE=Path(__file__).resolve().parent

class GuardTests(unittest.TestCase):
 def prefix(self,stage):
  tree=ast.parse((HERE/stage/'pull_terminal.py').read_bytes())
  main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  def configure(n):return any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='configure_packet_storage' for c in ast.walk(n))
  if stage=='gpu':
   outer=next(n for n in main.body if isinstance(n,ast.Try) and isinstance(n.body[0],ast.For))
   end=next(i for i,n in enumerate(outer.body) if configure(n));nodes=copy.deepcopy(outer.body[:end+1])
  else:
   outer=next(n for n in main.body if isinstance(n,ast.With) and 'coordinator_locks' in ast.unparse(n.items[0].context_expr))
   body=outer.body[1].body;end=next(i for i,n in enumerate(body) if configure(n))
   nodes=[ast.With(items=copy.deepcopy(outer.items),body=[copy.deepcopy(outer.body[0]),*copy.deepcopy(body[:end+1])])]
  return compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),stage+'-exact-collector-prefix','exec')
 def test_each_collector_lock_or_prior_attempt_rejects_before_backend(self):
  for stage in ('gpu','finalizer'):
   code=self.prefix(stage)
   for conflict in ('lock1','lock2','local','external','operations','none'):
    with self.subTest(stage=stage,conflict=conflict),tempfile.TemporaryDirectory(prefix='fake-collector-guards-') as tmp:
     root=Path(tmp).resolve();local=root/'local';external=root/'external';events=[];locks=[]
     for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):(root/name).write_bytes(b'')
     if conflict in ('local','external','operations'):
      {'local':local,'external':external,'operations':external.with_name(external.name+'-storage-collector-operations')}[conflict].mkdir()
     @contextlib.contextmanager
     def coordinator_locks():
      for index in (1,2):
       events.append('lock'+str(index))
       if conflict=='lock'+str(index):raise BlockingIOError('fake occupied controller lock')
      yield
     class Fcntl:
      LOCK_EX=1;LOCK_NB=2
      @staticmethod
      def flock(*_):
       key='lock'+str(1+len([v for v in events if v.startswith('lock')]))
       events.append(key)
       if conflict==key:raise BlockingIOError('fake occupied controller lock')
     def register(*_,**__):events.append('intent')
     def backend(*_,**__):events.append('backend');return object()
     b=types.SimpleNamespace(coordinator_locks=coordinator_locks,start_storage_operations=register,configure_packet_storage=backend)
     def write(path,data):
      with Path(path).open('xb') as f:f.write(data)
     env=dict(BASE=root,LOCAL=local,EXT=external,Path=Path,os=os,fcntl=Fcntl,need=C.need,
      locks=locks,claimed=[],write=write,__file__=str(HERE/stage/'pull_terminal.py'),self_bytes=b'source',review=b'review',
      b=b,ctx={'observations':{'cpu':{'files':{}}}},LAUNCH=root/'launch',pins={},packet={},
      a=types.SimpleNamespace(job_id='7',packet_sha256='a'*64,collector_sha256='b'*64,packet_commit='c'*40))
     try:
      if conflict=='none':exec(code,env);self.assertEqual(events,['lock1','lock2','intent','backend'])
      else:
       with self.assertRaises((ValueError,BlockingIOError)):exec(code,env)
       self.assertNotIn('intent',events);self.assertNotIn('backend',events)
     finally:
      for f in locks:f.close()

class EvidenceTests(unittest.TestCase):
 def fixture(self,backend):
  obj=F.Wiring();obj.setUp();self.addCleanup(obj.doCleanups)
  with patch.object(F,'BACKEND',backend):ctx,s,a,b,training=obj.fixture()
  owners=[obj.root/'owner-local',obj.root/'owner-external']
  for root in owners:root.mkdir()
  def publish(name,value):
   for root in owners:
    with (root/name).open('xb') as f:f.write(C.canonical(value))
  evidence=C.OperationEvidence(obj.root/'durable-operation-original',{'phase':'fabricated-caller','packet_sha256':'a'*64},publish)
  ctx['storage_operation_evidence']=evidence
  def bind():return evidence.run('binding',lambda:C.bind(ctx,s,predecessor='cpu',capsule=a,mirror=b,
   expected_roots={'launch':'/scratch/fake-cpu'},committed=lambda commit,path:backend if path.endswith('.py') else b'backend review',helper_path='FAKE'))
  return obj,ctx,evidence,owners,bind
 def check_reference(self,evidence,owners):
  ref=evidence.last_reference;raw=Path(ref['inventory_path']).read_bytes()
  self.assertEqual(C.sha(raw),ref['inventory_sha256']);manifest=C.decode(raw)
  self.assertEqual(manifest['inventory'],C.selected_inventory(evidence.root/'operations'))
  self.assertEqual(C.sha((evidence.root/'intent.json').read_bytes()),ref['intent_sha256'])
  name=next(p.name for p in owners[0].glob('storage-use-*.json') if C.decode(p.read_bytes())==ref)
  self.assertEqual((owners[0]/name).read_bytes(),(owners[1]/name).read_bytes())
  for p in owners:self.assertEqual(C.decode((p/'storage-use.intent.json').read_bytes()),evidence.intent_reference)
  return manifest
 def test_initial_verification_failure_has_registered_pinned_partial_records(self):
  backend=F.BACKEND.replace(b" Path(log_dir).mkdir()",b" Path(log_dir).mkdir();Path(log_dir,'partial.log').write_bytes(b'partial verification')\n assert (Path(log_dir).parents[1]/'intent.json').is_file() and (Path(log_dir).parents[2]/'owner-local/storage-use.intent.json').is_file()\n raise RuntimeError('fabricated initial verification failure')")
  obj,ctx,evidence,owners,bind=self.fixture(backend)
  with self.assertRaisesRegex(RuntimeError,'initial verification'):bind()
  self.assertNotIn('compact_inputs',ctx)
  manifest=self.check_reference(evidence,owners)
  self.assertEqual(manifest['status'],'failed');self.assertIn('verify-000/partial.log',manifest['inventory']['files'])
  self.assertEqual(len(list(evidence.root.glob('inventory-*.json'))),2)
 def test_later_fetch_failure_keeps_bytes_hashes_and_no_retry(self):
  backend=F.BACKEND.replace(b"def fetch_exact(**kwargs):\n raise AssertionError('no real or fake fetch permitted in bind fixture')",b"FETCHES=0\ndef fetch_exact(**kwargs):\n global FETCHES\n FETCHES+=1\n Path(kwargs['log_dir']).mkdir();Path(kwargs['log_dir'],'argv.json').write_bytes(b'fake argv')\n Path(kwargs['destination']).write_bytes(b'opaque partial bytes')\n raise RuntimeError('fabricated later fetch failure')")
  obj,ctx,evidence,owners,bind=self.fixture(backend);gate=bind()
  with self.assertRaisesRegex(RuntimeError,'later fetch'):gate.capture('launch/inputs/checkpoint.pt','a'*64)
  manifest=self.check_reference(evidence,owners)
  self.assertEqual(manifest['status'],'failed')
  self.assertEqual(manifest['inventory']['files']['fetch-000/checkpoint.pt']['sha256'],C.sha(b'opaque partial bytes'))
  self.assertIn('fetch-log-000/argv.json',manifest['inventory']['files'])
  with self.assertRaisesRegex(ValueError,'no implicit retry'):gate.capture('launch/inputs/checkpoint.pt','a'*64)
  self.assertEqual(gate.backend.FETCHES,1)
 def test_publication_failure_preserves_primary_and_original_manifest(self):
  with tempfile.TemporaryDirectory(prefix='fake-operation-secondary-') as tmp:
   root=Path(tmp).resolve();records=[]
   def publish(name,value):
    records.append(name)
    if name!='storage-use.intent.json':raise OSError('fabricated caller publication failure')
   e=C.OperationEvidence(root/'records',{'phase':'fake'},publish);primary=RuntimeError('primary failure')
   def fail():
    (e.root/'operations').mkdir();(e.root/'operations'/'partial.txt').write_bytes(b'partial')
    raise primary
   with self.assertRaises(RuntimeError) as caught:e.run('verify',fail)
   self.assertIs(caught.exception,primary)
   self.assertTrue((e.root/'reference-000.json').is_file());self.assertTrue(list(e.root.glob('evidence-failure-*.json')))
   ref=C.decode((e.root/'reference-000.json').read_bytes());self.assertEqual(C.sha(Path(ref['inventory_path']).read_bytes()),ref['inventory_sha256'])
 def test_only_transport_evidence_and_lock_scopes_change_protected_code(self):
  previous=HERE.parent/'predecessor'
  funcs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
  consts=lambda t:{n.targets[0].id:ast.dump(n.value) for n in ast.parse(t).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
  for stage in ('gpu','finalizer'):
   old=(previous/stage/'build_packet.py').read_text();new=(HERE/stage/'build_packet.py').read_text()
   for name in ('authenticate','make_packet','coordinator_locks'):self.assertEqual(funcs(old)[name],funcs(new)[name])
   self.assertEqual({k:v for k,v in consts(old).items() if k!='COMPACT_ADAPTER_SHA'},{k:v for k,v in consts(new).items() if k!='COMPACT_ADAPTER_SHA'})
   old=(previous/stage/'pull_terminal.py').read_text();new=(HERE/stage/'pull_terminal.py').read_text()
   for name in ('resources','accounting_extension','compare_observations','validate_remote_intent'):self.assertEqual(funcs(old)[name],funcs(new)[name])
   self.assertEqual(consts(old)['REMOTE_OBSERVATION'],consts(new)['REMOTE_OBSERVATION'])
   self.assertEqual((previous/stage/'monitor_once.py').read_bytes(),(HERE/stage/'monitor_once.py').read_bytes())
   tree=ast.parse((HERE/stage/'build_packet.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   text=ast.unparse(main)
   self.assertLess(text.index('exclusive new launch destinations required before storage use'),text.index('configure_storage(ctx'))
   self.assertIn('storage-preparation-intent.json',(HERE/stage/'build_packet.py').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
