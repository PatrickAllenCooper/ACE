"""NEW fabricated control-selection tests; mocked transport, no real locks/SSH/audit."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def load():
    data=(ROOT/'batch_gpu_audit.py').read_bytes();backend=(ROOT/'dependencies/remote_custody.py').read_bytes()
    m=types.ModuleType('batch04_control_fixture');m.__file__=str(ROOT/'batch_gpu_audit.py')
    exec(compile(data,m.__file__,'exec'),vars(m))
    m.bind_sources(extension_source=data,extension_sha256=sha(data),backend_source=backend,backend_sha256=sha(backend))
    return m


def inventory(b,payload):
    files={n:dict(bytes=len(v),sha256=sha(v)) for n,v in payload.items()}
    return b.normalized(dict(files=files,directories=sorted({'/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/')))})))


def fixture(m,root):
    b=m._backend();C=m.CAMPAIGN;job='99999998';cpu_root='/scratch/FAKE_BATCH04_CPU'
    payload={C+'source-plan-pins.json':b'["FABRICATED_PUBLIC_METADATA"]\n'}
    payload.update({C+f'policy-00/world-{i:02d}/episode/plan.json':b'{"fabricated_public_plan":true}\n' for i in range(24)})
    cpu={k:None for k in m.CPU_KEYS}
    cpu.update(schema='ace-scientific-campaign-cpu-acceptance-v1',status='development_cpu_phase_accepted',
               scientific_qualified=False,campaign_root=cpu_root+'/cpu/worker/artifacts/campaign',
               campaign_files={n[len(C):]:sha(v) for n,v in payload.items()})
    payload['launch/inputs/cpu-acceptance.json']=b.raw(cpu)  # A matching old copy must never substitute for control.
    payload['launch/gpu/terminal.json']=b'{"fabricated_only":true}\n'
    inputs={'cpu-acceptance.json':b.raw(cpu),'cpu-packet.json':b'{"fabricated_cpu_packet":true}\n'}
    inputs.update({n:b.raw({'fabricated_role':n}) for n in [*m.SERVICES.values(),*m.RECOVERY_FILES.values()]})
    pfiles={n:sha(v) for n,v in inputs.items()}
    payload.update({'control/inputs/'+n:v for n,v in inputs.items()})
    paths=dict(cpu_inputs=cpu_root+'/inputs',cpu_output=cpu_root+'/cpu',campaign=cpu['campaign_root'],
               gpu_inputs=m.CONTROL_ROOT+'/inputs',gpu_output=m.GPU_OUTPUT)
    inner=dict(schema='ace-scientific-paired-campaign-gpu-cpu-history-v2',scope='development',paths=paths,
               cpu_acceptance_sha256=sha(b.raw(cpu)),cpu_packet_sha256=pfiles['cpu-packet.json'],
               files=pfiles,services=m.SERVICES,recovery_files=m.RECOVERY_FILES)
    payload['control/inputs/packet.json']=b.raw(inner)
    for n in ['gpu-failure-baseline.json','gpu-failure-acceptance.json','gpu-failure-seal.json','gpu-failure-disposition.json']:
        payload['control/'+n]=b.raw({'fabricated_role':n})
    lineage=dict(control_root=m.CONTROL_ROOT,failed_root=m.FAILED_GPU_ROOT,output_root=m.GPU_OUTPUT,failed_job_id='35498218',
                 historical_postcustody_required=True,authority_sha256=pfiles[m.RECOVERY_FILES['authority']],
                 preflight_receipt_sha256=pfiles[m.RECOVERY_FILES['receipt']],
                 failure_acceptance_sha256=sha(payload['control/gpu-failure-acceptance.json']),
                 failure_seal_sha256=sha(payload['control/gpu-failure-seal.json']),
                 failure_disposition_sha256=sha(payload['control/gpu-failure-disposition.json']))
    outer=dict(schema='ace-scientific-campaign-gpu-recovery-launch-v2',remote_root=m.CONTROL_ROOT,paths=paths,
               gpu_packet_sha256=sha(b.raw(inner)),cpu_packet_sha256=inner['cpu_packet_sha256'],
               cpu_acceptance_sha256=inner['cpu_acceptance_sha256'],prior_jobs=47,prior_journals=96,
               baseline_sha256=sha(payload['control/gpu-failure-baseline.json']),gpu_recovery=lineage,
               new_input_files={n[8:]:sha(v) for n,v in payload.items() if n.startswith('control/')})
    payload['control/packet.json']=b.raw(outer)
    payload['control/submission.intent.json']=b'{"fabricated_submission_intent":true}\n'
    result=dict(job_id=job,returncode=0,ambiguous=False,intent_sha256=sha(payload['control/submission.intent.json']))
    payload['control/submission.result.json']=b.raw(result)
    payload['control/logs/'+job+'.out']=b'FABRICATED STDOUT ONLY\n'
    payload['control/logs/'+job+'.err']=b''
    binding=dict(schema='ace-gpu02-audit-batch-control-v1',outer=outer,outer_sha256=sha(b.raw(outer)),
                 inner=inner,inner_sha256=sha(b.raw(inner)),result=result,result_sha256=sha(b.raw(result)))
    full=inventory(b,payload);local,omitted=b.selected_inventory(full)
    copies=[root/'capsule-a',root/'capsule-b']
    for p in copies:
        (p/'remote').mkdir(mode=0o700,parents=True)
        for n in local['files']:
            q=p/'remote'/n;q.parent.mkdir(mode=0o700,parents=True,exist_ok=True);q.write_bytes(payload[n])
    receipt=dict(schema='ace-remote-custody-v1',full_backup_verified=True,full_local_copy=False,acceptance=False,retry_allowed=False,
        helper_sha256=m.BACKEND_SHA256,selection_policy='current-launch-text-and-namespaces-v1',full_inventory=full,
        local_inventory=local,local_root=str(copies[0]/'remote'),omitted_files=omitted,omitted_bytes=sum(v['bytes'] for v in omitted.values()),
        backup_root='/projects/paco0228/ACE_evidence/FAKE_BATCH04',backup_kind='ustar',
        backup_archive_path='/projects/paco0228/ACE_evidence/FAKE_BATCH04/raw.tar',backup_archive_sha256='a'*64,
        backup_archive_bytes=b.tar_size(full),roots={'control':m.CONTROL_ROOT,'launch':m.FAILED_GPU_ROOT,'cpu':cpu_root})
    for key in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256'):
        receipt[key]=sha(b.raw(full))
    for p in copies:(p/'remote-custody.json').write_bytes(b.raw(receipt))
    seal_payload={'remote-custody.json':b.raw(receipt),**{'remote/'+n:payload[n] for n in local['files']}}
    view=inventory(b,seal_payload)
    seal=dict(identical=True,acceptance=False,copies={str(p):copy.deepcopy(view) for p in copies})
    binding_pin=sha(b.raw(binding));members=m.select_members(full,control_binding=binding,control_binding_sha256=binding_pin)
    aid=(ROOT/'references/aid-audit_saved_gpu72.py').read_bytes();review=b'FABRICATED TEST REVIEW; NO ACTUAL REVIEW OR EXECUTION AUTHORITY\n'
    args=dict(receipt=receipt,receipt_sha256=sha(b.raw(receipt)),custody_seal=seal,custody_seal_sha256=sha(b.raw(seal)),
        cpu_acceptance=cpu,cpu_acceptance_sha256=sha(b.raw(cpu)),gpu_audit_source=aid,gpu_audit_sha256=sha(aid),
        gpu_audit_review=review,gpu_audit_review_sha256=sha(review),members=members,members_sha256=sha(b.raw(members)),
        control_binding=binding,control_binding_sha256=binding_pin,destination=root/'new-payload',log_dir=root/'new-log',helper_path='/NEVER_INVOKED')
    return args,payload


def request(m,args):
    b=m._backend();q=dict(args,action='batch_gpu72_recovery_audit',helper_sha256=m.BACKEND_SHA256,extension_sha256=m._BOUND[1])
    q['destination']=str(q['destination']);q.pop('helper_path');q.pop('log_dir')
    for k in ('gpu_audit_source','gpu_audit_review'):q[k]=base64.b64encode(q[k]).decode()
    return b.decode(b.raw(q))


class ControlDelta(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'evidence/attempt01/fixtures'/self._testMethodName
        self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        self.m=load();self.b=self.m._backend();self.args,self.payload=fixture(self.m,self.root)

    def test_current_control_binding_rejects_old_copy_and_wrong_job(self):
        m,b=self.m,self.b;q=request(m,self.args);m._selection(b,q)
        full=q['receipt']['full_inventory'];binding=q['control_binding']
        (self.root/'positive-request.json').write_bytes(b.raw(q))
        cases=[('old_copy_only',lambda f,c:f['files'].pop('control/inputs/cpu-acceptance.json')),
               ('wrong_current_packet',lambda f,c:f['files']['control/inputs/packet.json'].__setitem__('sha256','0'*64)),
               ('missing_current_log',lambda f,c:f['files'].pop('control/logs/99999998.err')),
               ('foreign_control_log',lambda f,c:f['files'].__setitem__('control/logs/11111111.err',dict(bytes=0,sha256=sha(b'')))),
               ('old_control_root',lambda f,c:c['outer'].__setitem__('remote_root',m.FAILED_GPU_ROOT)),
               ('old_job',lambda f,c:c['result'].__setitem__('job_id','35498218'))]
        for name,change in cases:
            f=copy.deepcopy(full);c=copy.deepcopy(binding);change(f,c)
            # Re-pin deliberately changed metadata objects to reach semantic joins, not only independent hash rejection.
            for role,member in [('outer','control/packet.json'),('inner','control/inputs/packet.json'),('result','control/submission.result.json')]:
                if c[role]!=binding[role]:
                    c[role+'_sha256']=sha(b.raw(c[role]));f['files'][member]=dict(bytes=len(b.raw(c[role])),sha256=c[role+'_sha256'])
            (self.root/(name+'.json')).write_bytes(b.raw(dict(full=f,binding=c)))
            with self.subTest(case=name),self.assertRaises(ValueError):m._control_binding(b,f,c,sha(b.raw(c)))
        q['gpu_audit_review']='';q['gpu_audit_review_sha256']=sha(b'')
        with self.assertRaises(ValueError):m._selection(b,q)  # Pending actual review cannot be silently omitted.

    def test_exact_science_plus_six_control_union_and_combined_bounds(self):
        m,b=self.m,self.b;full=copy.deepcopy(self.args['receipt']['full_inventory']);binding=self.args['control_binding'];pin=self.args['control_binding_sha256']
        get=lambda f:m.select_members(f,control_binding=binding,control_binding_sha256=pin)
        selected=get(full)
        control={'control/inputs/cpu-packet.json',*('control/inputs/'+n for n in m.SERVICES.values()),'control/logs/99999998.out','control/logs/99999998.err'}
        self.assertEqual(set(selected),set(m.audit_names(full))|control);self.assertEqual(len(selected),31)
        self.assertFalse(any(n.startswith('launch/') for n in selected))
        frozen='control/inputs/frozen_llm.freeze.json';full['files'][frozen]['bytes']=b.LOCAL_MEMBER;get(full)
        full['files'][frozen]['bytes']+=1
        with self.assertRaises(ValueError):get(full)
        full=copy.deepcopy(self.args['receipt']['full_inventory'])
        for i in range(16):
            n=m.CAMPAIGN+f'policy-05/world-00/return-{i:03d}.json';full['files'][n]=dict(bytes=31*1024**2,sha256='c'*64)
            full['directories'] += [m.CAMPAIGN+'policy-05',m.CAMPAIGN+'policy-05/world-00']
        full['directories']=sorted(set(full['directories']));get(full)  # Scientific portion stays below512MiB.
        full['files'][frozen]['bytes']=31*1024**2
        (self.root/'combined-over512MiB.json').write_bytes(b.raw(full))
        with self.assertRaises(ValueError):get(full)  # Union, not two independently enlarged budgets.
        q=request(m,self.args);q['members']['control/inputs/private-extra.json']=dict(bytes=1,sha256='d'*64)
        q['members_sha256']=sha(b.raw(q['members']))
        with self.assertRaises(ValueError):m._selection(b,q)

    def test_mocked_control_payload_readback_and_detached_binding(self):
        m,b=self.m,self.b;original=m._selection;specialize=m._specialize
        expected_binding=copy.deepcopy(self.args['control_binding']);expected_members=copy.deepcopy(self.args['members'])
        before={p:b.local_inventory(Path(p)) for p in self.args['custody_seal']['copies']}
        def selection(backend,q):
            result=original(backend,q)
            self.args['control_binding']['result']['job_id']='CALLER_MUTATION_AFTER_DETACH'
            return result
        def transport(q,receiver,helper,logs):
            full,selected=original(b,q)
            value=m._result(b,q,full,m._verification(b,q,full))
            frame=b.raw(dict(result=value,stream_files=selected['files']))
            payload=len(frame).to_bytes(8,'big')+frame+b''.join(self.payload[n] for n in sorted(selected['files']))
            for i in range(0,len(payload),197):receiver.feed(payload[i:i+197])
            b.write_new(logs/'mock-transport.json',b.raw(dict(mocked=True,remote_actions=0,locks_acquired=0)))
        def replacement(function,**bindings):
            return transport if function is b.transport else specialize(function,**bindings)
        with mock.patch.object(m,'_selection',selection),mock.patch.object(m,'_specialize',replacement):
            result=m._fetch_audit(**self.args)  # Deliberately bypass real-lock public wrapper; transport replaced above.
        logs=self.args['log_dir'];dest=self.args['destination']
        self.assertEqual(b.decode((logs/'recovery-control-binding.json').read_bytes()),expected_binding)
        self.assertEqual(result['control_binding_sha256'],self.args['control_binding_sha256'])
        self.assertEqual(result['job_id'],'99999998');self.assertFalse(result['acceptance'])
        self.assertEqual(result['local_inventory']['files'],expected_members)
        fetched=b.decode((logs/'fetched.json').read_bytes())
        self.assertEqual(set(fetched),set(expected_members))
        for n in expected_members:self.assertEqual((dest/n).read_bytes(),self.payload[n])
        for p,inv in before.items():self.assertEqual(b.local_inventory(Path(p)),inv)
        self.assertEqual((logs/'gpu-audit-source.py').read_bytes(),self.args['gpu_audit_source'])
        self.assertEqual((logs/'gpu-audit-review.md').read_bytes(),self.args['gpu_audit_review'])


if __name__=='__main__':unittest.main(verbosity=2)
