"""NEW artificial selection/authority/framing fixtures. No SSH or real locks."""
import base64,copy,hashlib,json,os,tempfile,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FIXTURES=Path(os.environ['GPU_BATCH_FIXTURES']).resolve()
source=(ROOT/'batch_gpu_audit.py').read_bytes();backend_source=(ROOT/'dependencies/remote_custody.py').read_bytes()
m=types.ModuleType('new_gpu_batch');m.__file__=str(ROOT/'batch_gpu_audit.py')
exec(compile(source,m.__file__,'exec'),vars(m))
hash_bytes=lambda b:hashlib.sha256(b).hexdigest()
m.bind_sources(extension_source=source,extension_sha256=hash_bytes(source),backend_source=backend_source,backend_sha256=hash_bytes(backend_source))
b=m._backend();C=m.CAMPAIGN


def inventory(files):
 dirs=sorted({'/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/')))})
 return b.normalized(dict(files=files,directories=dirs))


def fixture():
 payload={C+'source-plan-pins.json':b'["FABRICATED"]\n'}
 for i in range(24):payload[C+f'policy-00/world-{i:02d}/episode/plan.json']=b'{"fake_public_plan":true}\n'
 cpu={k:None for k in m.CPU_KEYS}
 cpu.update(schema='ace-scientific-campaign-cpu-acceptance-v1',status='development_cpu_phase_accepted',scientific_qualified=False,
   campaign_root='/scratch/alpine/paco0228/FAKE_CPU/cpu/worker/artifacts/campaign',campaign_files={n[len(C):]:hash_bytes(v) for n,v in payload.items()})
 payload['launch/inputs/cpu-acceptance.json']=b.raw(cpu)
 for pi in (5,6,7):
  p=C+f'policy-{pi:02d}/';payload[p+'attempt.json']=b'{}\n'
  for n in ('launch','process','observed-handshake','ready','closed','close-cleanup','failure','failure-intent'):
   payload[p+'service/'+n+'.json']=b'{"fabricated":true}\n'
  for i in range(24):
   payload[p+f'service/task-binding-{i:04d}.json']=b'{}\n'
   for suffix in ('input','observed-response','accepted','episode'):payload[p+f'service/request-{i:04d}.{suffix}.json']=b'{}\n'
   w=p+f'world-{i:02d}/'
   for n in ('arm-receipt.json','execution-diagnostics.json','task.json','paired-plan.json','initial-world.jsonl','measurement-world.jsonl','episode/plan.json','episode/journal.jsonl','episode/terminal.json','episode/failure.json','return-000.json'):
    payload[w+n]=b'{"opaque_fake":true}\n'
   payload[w+'world.json']=b'NEVER SELECTED'
   payload[w+'sealed-inputs/task.json']=b'NEVER SELECTED'
   payload[w+'weights.pt']=b'NEVER SELECTED'
 payload[C+'private/world-00/summary.json']=b'NEVER SELECTED'
 payload['cpu/inputs/development-private.npy']=b'NEVER SELECTED'
 full=inventory({n:dict(bytes=len(v),sha256=hash_bytes(v)) for n,v in payload.items()})
 local,omitted=b.selected_inventory(full)
 receipt=dict(schema='ace-remote-custody-v1',full_backup_verified=True,full_local_copy=False,acceptance=False,retry_allowed=False,
  helper_sha256=m.BACKEND_SHA256,selection_policy='current-launch-text-and-namespaces-v1',full_inventory=full,local_inventory=local,
  omitted_files=omitted,omitted_bytes=sum(x['bytes'] for x in omitted.values()),backup_root='/projects/paco0228/ACE_evidence/FAKE_ONLY',
  backup_kind='ustar',backup_archive_path='/projects/paco0228/ACE_evidence/FAKE_ONLY/raw.tar',backup_archive_sha256='a'*64,
  backup_archive_bytes=b.tar_size(full),roots={'cpu':'/scratch/alpine/paco0228/FAKE_CPU'})
 for k in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256'):receipt[k]=hash_bytes(b.raw(full))
 view=inventory({'remote-custody.json':dict(bytes=len(b.raw(receipt)),sha256=hash_bytes(b.raw(receipt))),
                 **{'remote/'+n:v for n,v in local['files'].items()}})
 seal=dict(identical=True,acceptance=False,excludes='custody-inventory.json itself',copies={'/FAKE/one':view,'/FAKE/two':copy.deepcopy(view)})
 members=m.select_members(full);audit=(ROOT/'references/audit_saved_gpu72.py').read_bytes();review=b'FABRICATED REVIEW ONLY; NEVER ACTUAL AUTHORIZATION\n'
 req=dict(action='batch_gpu72_audit',receipt=receipt,receipt_sha256=hash_bytes(b.raw(receipt)),custody_seal=seal,
  custody_seal_sha256=hash_bytes(b.raw(seal)),cpu_acceptance=cpu,cpu_acceptance_sha256=hash_bytes(b.raw(cpu)),
  gpu_audit_source=base64.b64encode(audit).decode(),gpu_audit_sha256=hash_bytes(audit),
  gpu_audit_review=base64.b64encode(review).decode(),gpu_audit_review_sha256=hash_bytes(review),
  helper_sha256=m.BACKEND_SHA256,extension_sha256=hash_bytes(source),members=members,members_sha256=hash_bytes(b.raw(members)),destination='/FAKE/selected')
 return req,payload


class NewGpuBatchTests(unittest.TestCase):
 def setUp(self):
  self.path=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=FIXTURES))
  self.req,self.payload=fixture();(self.path/'request.json').write_bytes(b.raw(self.req))
 def test_exact_audit_candidate_roster_scopes_and_size_bounds(self):
  full=self.req['receipt']['full_inventory'];members=m.select_members(full)
  self.assertEqual(len(members),1204)
  self.assertEqual(sum('/world-' in n and n.endswith('/arm-receipt.json') for n in members),72)
  self.assertEqual(sum('/policy-00/' in n for n in members),24)
  self.assertFalse(any('/private/' in n or '/sealed-inputs/' in n or n.endswith(('world.json','.pt','.npy')) for n in members))
  # Independently evaluate the closed aid's documented predicate without importing it.
  expected={C+'source-plan-pins.json'}|{C+f'policy-00/world-{i:02d}/episode/plan.json' for i in range(24)}
  for n in full['files']:
   leaf=n.rsplit('/',1)[-1]
   if any(n.startswith(C+f'policy-{i:02d}/') for i in (5,6,7)) and (leaf in {'attempt.json','arm-receipt.json','execution-diagnostics.json','task.json','paired-plan.json'} or '/service/' in n or '/episode/' in n or leaf.endswith('-world.jsonl') or leaf.startswith('return-')) and n.endswith(('.json','.jsonl')) and '/sealed-inputs/' not in n:expected.add(n)
  self.assertEqual(set(members),expected)
  for bad in ('policy-05/private/task.json','policy-06/service/private/scores.json','policy-07/world-24/episode/plan.json','policy-05/world-00/episode/outcome.json'):
   files=dict(full['files']);files[C+bad]=dict(bytes=1,sha256='0'*64)
   with self.subTest(bad=bad),self.assertRaises(ValueError):m.select_members(inventory(files))
  files=dict(full['files']);name=next(iter(members));files[name]=dict(bytes=32*1024**2+1,sha256='0'*64)
  with self.assertRaises(ValueError):m.select_members(inventory(files))
  files=dict(full['files'])
  for name in list(members)[:17]:files[name]=dict(bytes=32*1024**2,sha256='0'*64)
  with self.assertRaises(ValueError):m.select_members(inventory(files))
 def test_independent_receipt_seal_cpu_audit_and_exact_map_joins(self):
  full,selection=m._selection(b,self.req);self.assertEqual(selection['files'],self.req['members'])
  mutations=[('wrong_receipt',lambda q:q.update(receipt_sha256='0'*64)),
    ('wrong_seal',lambda q:q['custody_seal']['copies']['/FAKE/one']['files']['remote-custody.json'].update(sha256='0'*64)),
    ('wrong_cpu',lambda q:q['cpu_acceptance'].update(status='unaccepted')),
    ('wrong_audit',lambda q:q.update(gpu_audit_sha256='0'*64)),
    ('wrong_review',lambda q:q.update(gpu_audit_review_sha256='0'*64)),
    ('omit_member',lambda q:q['members'].pop(next(iter(q['members'])))),
    ('add_private',lambda q:q['members'].update({C+'private/world-00/summary.json':q['receipt']['full_inventory']['files'][C+'private/world-00/summary.json']}))]
  for label,mutate in mutations:
   q=copy.deepcopy(self.req);mutate(q)
   # Re-pin changed containers: semantic joins must still reject unauthorized substitution.
   for key,pinkey in [('custody_seal','custody_seal_sha256'),('cpu_acceptance','cpu_acceptance_sha256'),('members','members_sha256')]:q[pinkey]=hash_bytes(b.raw(q[key]))
   (self.path/(label+'.json')).write_bytes(b.raw(q))
   with self.subTest(case=label),self.assertRaises(ValueError):m._selection(b,q)
 def test_new_audit_frame_identity_manifest_and_partial_payload_tamper(self):
  full,selection=m._selection(b,self.req);result=m._result(b,self.req,full,m._verification(b,self.req,full))
  frame=b.raw(dict(result=result,stream_files=selection['files']))
  payload=b''.join(self.payload[n] for n in sorted(selection['files']))
  valid=len(frame).to_bytes(8,'big')+frame+payload
  variants={'valid':valid,'truncated':valid[:-1],'extra':valid+b'X'}
  wrong=copy.deepcopy(result);wrong['cpu_acceptance_sha256']='0'*64
  f=b.raw(dict(result=wrong,stream_files=selection['files']));variants['wrong_authority']=len(f).to_bytes(8,'big')+f+payload
  files=dict(selection['files']);files.pop(next(iter(files)));f=b.raw(dict(result=result,stream_files=files));variants['wrong_manifest']=len(f).to_bytes(8,'big')+f+payload
  variants['wrong_payload']=valid[:-1]+b'X'
  for label,data in variants.items():
   with self.subTest(case=label):
    folder=self.path/label;folder.mkdir();(folder/'stdout.bin').write_bytes(data)
    dest=folder/'selected';dest.mkdir();paths=b.local_destinations(dest,selection)
    def validate(value):b.need(value==result,'batch identity changed')
    receiver=b.Receiver(selection['files'],paths,validate);error=None
    try:
     for i in range(0,len(data),137):receiver.feed(data[i:i+137])
     receiver.finish()
    except ValueError as exc:error=exc;(folder/'failure.txt').write_text(str(exc))
    finally:receiver.close()
    self.assertEqual(error is None,label=='valid')
    if label=='valid':self.assertEqual(b.local_inventory(dest),selection)
    else:self.assertTrue((folder/'stdout.bin').exists())

if __name__=='__main__':
 FIXTURES.mkdir(mode=0o700)
 unittest.main(verbosity=2)
