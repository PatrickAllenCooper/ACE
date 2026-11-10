"""NEW B1-B3 artificial local regressions, mocked transport; no SSH/real locks."""
import base64,copy,hashlib,json,os,tempfile,types,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['BATCH_REVIEW_FIXTURES']).resolve()
sha=lambda v:hashlib.sha256(v).hexdigest()
def load():
 data=(HERE/'batch_gpu_audit.py').read_bytes();backend=(HERE/'dependencies/remote_custody.py').read_bytes()
 m=types.ModuleType('batch_b_review');m.__file__=str(HERE/'batch_gpu_audit.py');exec(compile(data,m.__file__,'exec'),vars(m))
 m.bind_sources(extension_source=data,extension_sha256=sha(data),backend_source=backend,backend_sha256=sha(backend))
 return m

def fixture(m,root):
 b=m._backend();C=m.CAMPAIGN
 def inv(files):return b.normalized(dict(files=files,directories=sorted({'/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/')))})))
 payload={C+'source-plan-pins.json':b'["FABRICATED"]\n'}
 payload.update({C+f'policy-00/world-{i:02d}/episode/plan.json':b'{"public_fixture":true}\n' for i in range(24)})
 cpu={k:None for k in m.CPU_KEYS};cpu.update(schema='ace-scientific-campaign-cpu-acceptance-v1',status='development_cpu_phase_accepted',
  scientific_qualified=False,campaign_root='/scratch/FAKE_CPU/cpu/worker/artifacts/campaign',campaign_files={n[len(C):]:sha(v) for n,v in payload.items()})
 payload['launch/inputs/cpu-acceptance.json']=b.raw(cpu)
 full=inv({n:dict(bytes=len(v),sha256=sha(v)) for n,v in payload.items()});local,omitted=b.selected_inventory(full)
 copies=[root/'capsule-a',root/'capsule-b']
 for p in copies:
  p.mkdir(mode=0o700);(p/'remote').mkdir(mode=0o700)
  for n in local['files']:
   q=p/'remote'/n;q.parent.mkdir(mode=0o700,parents=True,exist_ok=True);q.write_bytes(payload[n])
 receipt=dict(schema='ace-remote-custody-v1',full_backup_verified=True,full_local_copy=False,acceptance=False,retry_allowed=False,
  helper_sha256=m.BACKEND_SHA256,selection_policy='current-launch-text-and-namespaces-v1',full_inventory=full,local_inventory=local,
  local_root=str(copies[0]/'remote'),omitted_files=omitted,omitted_bytes=sum(v['bytes'] for v in omitted.values()),
  backup_root='/projects/paco0228/ACE_evidence/FAKE_B_REVIEW',backup_kind='ustar',
  backup_archive_path='/projects/paco0228/ACE_evidence/FAKE_B_REVIEW/raw.tar',backup_archive_sha256='a'*64,
  backup_archive_bytes=b.tar_size(full),roots={'cpu':'/scratch/FAKE_CPU'})
 for k in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256'):receipt[k]=sha(b.raw(full))
 for p in copies:(p/'remote-custody.json').write_bytes(b.raw(receipt))
 view=inv({'remote-custody.json':dict(bytes=len(b.raw(receipt)),sha256=sha(b.raw(receipt))),**{'remote/'+n:v for n,v in local['files'].items()}})
 seal=dict(identical=True,acceptance=False,excludes='custody-inventory.json itself',copies={str(p):copy.deepcopy(view) for p in copies})
 members=m.select_members(full);audit=(HERE/'references/audit_saved_gpu72.py').read_bytes();review=b'FABRICATED REVIEW ONLY; NEVER AUTHORIZES REAL WORK\n'
 args=dict(receipt=receipt,receipt_sha256=sha(b.raw(receipt)),custody_seal=seal,custody_seal_sha256=sha(b.raw(seal)),
  cpu_acceptance=cpu,cpu_acceptance_sha256=sha(b.raw(cpu)),gpu_audit_source=audit,gpu_audit_sha256=sha(audit),
  gpu_audit_review=review,gpu_audit_review_sha256=sha(review),members=members,members_sha256=sha(b.raw(members)),
  destination=root/'new-selected',log_dir=root/'new-log',helper_path='/NOT_INVOKED')
 return args,payload

def request(m,args):
 b=m._backend();q=dict(args,action='batch_gpu72_audit',helper_sha256=m.BACKEND_SHA256,extension_sha256=m._BOUND[1])
 q['destination']=str(q['destination']);q.pop('helper_path');q.pop('log_dir')
 for k in ('gpu_audit_source','gpu_audit_review'):q[k]=base64.b64encode(q[k]).decode()
 return b.decode(b.raw(q))

class Delta(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=ROOT));self.m=load();self.b=self.m._backend()
  self.args,self.payload=fixture(self.m,self.root)
 def test_b1_independent_corrected_source_pin_no_import_fixed_selection(self):
  m,b=self.m,self.b;q=request(m,self.args);m._selection(b,q)
  self.assertEqual(q['gpu_audit_sha256'],'e28392bb02adc2441c68be9279704cfec795f6af8d9d587a2bf2b41157318908')
  # A separately supplied, correctly pinned source is captured, never imported;
  # the batch's member selector remains fixed even if this source is arbitrary.
  toy=b'raise RuntimeError("MUST NEVER EXECUTE")\n';q['gpu_audit_source']=base64.b64encode(toy).decode();q['gpu_audit_sha256']=sha(toy)
  _,selected=m._selection(b,q);self.assertEqual(selected['files'],self.args['members'])
  cases=[('invalid_pin',lambda v:v.update(gpu_audit_sha256='Z'*64)),
   ('wrong_hash',lambda v:v.update(gpu_audit_sha256='0'*64)),
   ('wrong_review',lambda v:v.update(gpu_audit_review_sha256='0'*64)),
   ('empty_review',lambda v:v.update(gpu_audit_review='',gpu_audit_review_sha256=sha(b''))),
   ('oversized_source',lambda v:v.update(gpu_audit_source=base64.b64encode(b'x'*(256*1024+1)).decode(),gpu_audit_sha256=sha(b'x'*(256*1024+1))))]
  for label,change in cases:
   value=copy.deepcopy(q);change(value);(self.root/(label+'.json')).write_bytes(b.raw(value))
   with self.subTest(case=label),self.assertRaises(ValueError):m._selection(b,value)
  q['members']['cpu/private/outcome.json']=dict(bytes=1,sha256='0'*64);q['members_sha256']=sha(b.raw(q['members']))
  with self.assertRaises(ValueError):m._selection(b,q)
 def test_b2_mutated_caller_map_cannot_change_capacity_records_or_fetched_map(self):
  m,b=self.m,self.b;expected=copy.deepcopy(self.args['members']);original=m._selection;specialize=m._specialize;begin=b.begin;capacity=[]
  original_capsules={p:b.local_inventory(Path(p)) for p in self.args['custody_seal']['copies']}
  def selection(backend,req):
   result=original(backend,req)
   self.args['members'].clear()  # mutation immediately AFTER authenticated detach/validation
   return result
  def begin_observed(logs,req,extra):
   capacity.extend(extra);self.assertEqual(req['members'],expected)
   return begin(logs,req,extra)
  def fake_transport(req,receiver,helper,logs):
   self.args['members']['caller-injected.json']=dict(bytes=999,sha256='0'*64)
   full,sel=original(b,req);value=m._result(b,req,full,m._verification(b,req,full))
   frame=b.raw(dict(result=value,stream_files=sel['files']));data=len(frame).to_bytes(8,'big')+frame+b''.join(self.payload[n] for n in sorted(sel['files']))
   for i in range(0,len(data),211):receiver.feed(data[i:i+211])
   b.write_new(logs/'mock-transport.json',b.raw(dict(mocked=True,remote_actions=0)))
  def replacement(function,**bindings):
   if function is b.transport:return fake_transport
   return specialize(function,**bindings)
  with mock.patch.object(m,'_selection',selection),mock.patch.object(m,'_specialize',replacement),mock.patch.object(b,'begin',begin_observed):
   result=m._fetch_audit(**self.args)
  logs=self.args['log_dir'];dest=self.args['destination']
  self.assertEqual(b.decode((logs/'members.json').read_bytes()),expected)
  self.assertEqual(b.decode((logs/'fetched.json').read_bytes()),{n:dict(path=str(dest/n),sha256=v['sha256']) for n,v in expected.items()})
  self.assertEqual(capacity[0][2],sum(v['bytes'] for v in expected.values()))
  self.assertEqual(result['member_count'],len(expected));self.assertFalse(result['acceptance'])
  self.assertEqual(result['local_inventory']['files'],expected)
  self.assertIn('caller-injected.json',self.args['members'])
  for p,view in original_capsules.items():self.assertEqual(b.local_inventory(Path(p)),view)
 def test_b3_all_original_root_boundaries_reject_before_claim(self):
  m,b=self.m,self.b;roots=[Path(p) for p in self.args['custody_seal']['copies']]+[Path(self.args['receipt']['local_root'])]
  before={p:b.local_inventory(p) for p in roots};findings=[]
  for i,protected in enumerate(roots):
   for relation,p in [('equal',protected),('ancestor',protected.parent),('child',protected/'new-audit-child')]:
    for field in ('destination','log_dir'):
     args=dict(self.args);args[field]=p
     with self.subTest(root=i,relation=relation,field=field),mock.patch.object(b,'begin',side_effect=AssertionError('claim reached')) as begin:
      with self.assertRaises(ValueError) as exc:m._fetch_audit(**args)
      begin.assert_not_called();findings.append(dict(root=i,relation=relation,field=field,error=str(exc.exception)))
  alias=self.root/'capsule-alias';alias.symlink_to(roots[0],target_is_directory=True)
  args=dict(self.args,destination=alias/'new-audit-child')
  with mock.patch.object(b,'begin',side_effect=AssertionError('claim reached')) as begin:
   with self.assertRaises(ValueError):m._fetch_audit(**args)
   begin.assert_not_called()
  (self.root/'boundary-findings.json').write_bytes(b.raw(findings))
  for p,view in before.items():self.assertEqual(b.local_inventory(p),view)
  self.assertFalse(self.args['destination'].exists());self.assertFalse(self.args['log_dir'].exists())

if __name__=='__main__':
 ROOT.mkdir(mode=0o700)
 unittest.main(verbosity=2)
