"""Pinned opaque GPU72 audit materialization; never execution acceptance.

Derivative of unchanged final-reports-batch02. Full archive and exact selection
are checked before and during streaming. No network or locks at import.
"""
import base64
import contextlib
import fcntl
import re
import stat
import hashlib
import json
import os
from pathlib import Path
import types

BACKEND_SHA256 = '4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f'
CONTROL_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
FAILED_GPU_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
GPU_OUTPUT = FAILED_GPU_ROOT+'/gpu'
RECOVERY_FILES = {'authority':'gpu-recovery-authority.json','context':'cpu-history-expected.json',
                  'receipt':'cpu-history-preflight.json','read_set':'cpu-history-readset.json'}
SERVICES = {p:p+'.freeze.json' for p in ('frozen_llm','sft_llm','dpo_llm')}
CAMPAIGN = 'cpu/cpu/worker/artifacts/campaign/'
LOCK_BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
CPU_KEYS = {'schema','status','cpu_packet_sha256','job_id','allocation','cpu_output','campaign_root',
 'cpu_output_files','cpu_output_excluded_directory','campaign_files','bundle_path','bundle_sha256',
 'phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}
_BOUND = None


def audit_names(full):
    """Same prospective audit candidates, narrowed by exact public evidence paths.

    Unexpected candidates fail rather than silently disappear. Archive members
    outside the candidate predicate remain fully verified and are not emitted.
    """
    b = _backend()
    full = b.normalized(full)
    wanted = {CAMPAIGN+'source-plan-pins.json'}
    wanted.update(CAMPAIGN+f'policy-00/world-{i:02d}/episode/plan.json' for i in range(24))
    world = r'world-(?:0[0-9]|1[0-9]|2[0-3])/'
    arm = (r'(?:arm-receipt\.json|execution-diagnostics\.json|task\.json|paired-plan\.json|'
           r'(?:initial|measurement)-world\.jsonl|return-[0-9]{3}\.json|'
           r'episode/(?:plan\.json|journal\.jsonl|terminal\.json|failure\.json))')
    service = (r'service/(?:launch|process|observed-handshake|ready|failure-intent|failure|'
               r'close-cleanup|closed|task-binding-[0-9]{4}|'
               r'request-[0-9]{4}\.(?:input|observed-response|accepted|episode))\.json')
    allowed = re.compile(r'policy-0[567]/(?:attempt\.json|'+service+'|'+world+arm+')')
    for name in full['files']:
        if not any(name.startswith(CAMPAIGN+f'policy-{i:02d}/') for i in (5,6,7)):
            continue
        leaf = name.rsplit('/',1)[-1]
        candidate = (leaf in {'attempt.json','arm-receipt.json','execution-diagnostics.json','task.json','paired-plan.json'}
                     or '/service/' in name or '/episode/' in name or leaf.endswith('-world.jsonl') or leaf.startswith('return-'))
        if candidate and name.endswith(('.json','.jsonl')) and '/sealed-inputs/' not in name:
            b.need(allowed.fullmatch(name[len(CAMPAIGN):]) is not None,
                   'unexpected audit candidate path; no arbitrary/private subtree selection: '+name)
            wanted.add(name)
    b.need(wanted <= set(full['files']), 'required source pins/24 accepted public plans absent')
    return tuple(sorted(wanted))


def select_members(full_inventory, *, control_binding, control_binding_sha256):
    """Pure prospective map; caller must independently pin it before transfer."""
    b = _backend(); full = b.normalized(full_inventory)
    control = _control_binding(b, full, control_binding, control_binding_sha256)
    names = set(audit_names(full)) | set(control['members'])
    members = {name:full['files'][name] for name in sorted(names)}
    b.need(all(info['bytes'] <= b.LOCAL_MEMBER for info in members.values()), 'audit member over32MiB')
    b.need(sum(v['bytes'] for v in members.values()) <= b.LOCAL_TOTAL, 'audit batch over512MiB')
    return members


@contextlib.contextmanager
def coordinator_locks():
    """Acquire the two existing coordinator locks once, nonblocking; never create."""
    handles=[];primary=None
    try:
        for name in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
            path=LOCK_BASE/name
            _require(path==path.resolve(strict=True), 'lock alias forbidden')
            fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                st=os.fstat(fd)
                _require(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==os.geteuid()
                         and os.path.samestat(st,path.lstat()), 'owned ordinary stable lock required')
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                os.close(fd);raise
            handles.append(fd)
        yield
    except BaseException as exc:
        primary=exc;raise
    finally:
        cleanup=[]
        for fd in reversed(handles):
            for action in (lambda:fcntl.flock(fd,fcntl.LOCK_UN),lambda:os.close(fd)):
                try:action()
                except BaseException as exc:cleanup.append(exc)
        if cleanup and primary is None:raise cleanup[0]
        if cleanup and hasattr(primary,'add_note'):
            primary.add_note('Secondary coordinator-lock cleanup errors: '+repr(cleanup))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def bind_sources(*, extension_source, extension_sha256, backend_source, backend_sha256):
    """Caller loads this extension from captured, independently authenticated bytes."""
    global _BOUND
    sources = (extension_source, extension_sha256, backend_source, backend_sha256)
    for source, expected in ((extension_source, extension_sha256), (backend_source, backend_sha256)):
        _require(type(source) is bytes and 0 < len(source) <= 256 * 1024
                 and type(expected) is str and hashlib.sha256(source).hexdigest() == expected,
                 'captured independent source SHA required before execution')
    _require(backend_sha256 == BACKEND_SHA256, 'original accepted backend required')
    if _BOUND is not None:
        _require(_BOUND[:4] == sources, 'source binding cannot change')
        return
    backend = types.ModuleType('captured_original_remote_custody')
    backend.__file__ = '<captured-original-4161-backend>'
    exec(compile(backend_source, backend.__file__, 'exec'), backend.__dict__)
    backend.bind_source(backend_source, backend_sha256)
    _BOUND = (*sources, backend)


def _backend():
    _require(_BOUND is not None, 'authenticate and bind both sources first')
    return _BOUND[4]


def _pinned_object(b, obj, expected, role):
    payload=b.raw(obj)
    b.need(len(payload)<=128*1024**2 and b.pin(expected) and b.sha(payload)==expected,
           'independent canonical128MiB '+role+' pin differs')
    return payload


def _control_binding(b, full, binding, binding_sha256):
    """Pure exact-byte joins for current control. No audit or authority execution."""
    _pinned_object(b,binding,binding_sha256,'GPU02 control binding')
    b.need(type(binding) is dict and set(binding)=={'schema','outer','outer_sha256','inner','inner_sha256','result','result_sha256'}
           and binding['schema']=='ace-gpu02-audit-batch-control-v1', 'explicit recovery control binding required')
    for role,member in [('outer','control/packet.json'),('inner','control/inputs/packet.json'),
                        ('result','control/submission.result.json')]:
        payload=_pinned_object(b,binding[role],binding[role+'_sha256'],role+' control artifact')
        b.need(full['files'].get(member)==dict(bytes=len(payload),sha256=binding[role+'_sha256']),
               'current control artifact absent or different: '+member)
    outer,inner,result=(binding[k] for k in ('outer','inner','result'))
    b.need(outer['schema']=='ace-scientific-campaign-gpu-recovery-launch-v2'
           and inner['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2'
           and outer['remote_root']==CONTROL_ROOT and inner['scope']=='development'
           and outer['gpu_packet_sha256']==binding['inner_sha256']
           and outer['cpu_acceptance_sha256']==inner['cpu_acceptance_sha256']
           and outer['cpu_packet_sha256']==inner['cpu_packet_sha256'], 'current recovery packet/CPU identities differ')
    b.need(type(outer['prior_jobs']) is int and outer['prior_jobs']==47
           and type(outer['prior_journals']) is int and outer['prior_journals']==96,
           'current control must descend from failed47/96')
    paths=inner['paths']
    b.need(outer['paths']==paths and paths['gpu_inputs']==CONTROL_ROOT+'/inputs'
           and paths['gpu_output']==GPU_OUTPUT and paths['campaign']==paths['cpu_output']+'/worker/artifacts/campaign'
           and inner['services']==SERVICES and inner['recovery_files']==RECOVERY_FILES,
           'frozen output and exact three control freezes required')
    def captured(member,pin):
        b.need(b.pin(pin) and full['files'].get(member,{}).get('sha256')==pin,
               'current control captured pin differs: '+member)
    for name,pin in inner['files'].items():
        b.name_ok(name);captured('control/inputs/'+name,pin)
    for name,pin in outer['new_input_files'].items():
        b.name_ok(name);captured('control/'+name,pin)
    b.need(outer['new_input_files']['inputs/packet.json']==binding['inner_sha256'], 'outer inner-member pin differs')
    captured('control/inputs/cpu-acceptance.json',inner['cpu_acceptance_sha256'])
    captured('control/inputs/cpu-packet.json',inner['cpu_packet_sha256'])
    captured('control/gpu-failure-baseline.json',outer['baseline_sha256'])
    lineage=outer['gpu_recovery']
    b.need(lineage['control_root']==CONTROL_ROOT and lineage['failed_root']==FAILED_GPU_ROOT
           and lineage['output_root']==GPU_OUTPUT and lineage['failed_job_id']=='35498218'
           and lineage['historical_postcustody_required'] is True
           and lineage['authority_sha256']==inner['files'][RECOVERY_FILES['authority']]
           and lineage['preflight_receipt_sha256']==inner['files'][RECOVERY_FILES['receipt']],
           'explicit old-failure/new-control recovery binding differs')
    for field,name in [('failure_acceptance_sha256','gpu-failure-acceptance.json'),
                       ('failure_seal_sha256','gpu-failure-seal.json'),
                       ('failure_disposition_sha256','gpu-failure-disposition.json')]:
        captured('control/'+name,lineage[field])
    job=result['job_id']
    b.need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) is not None and job!='35498218'
           and type(result['returncode']) is int and result['returncode']==0 and result['ambiguous'] is False,
           'one unambiguous new submission identity required')
    captured('control/submission.intent.json',result['intent_sha256'])
    logs={'control/logs/'+job+suffix for suffix in ('.out','.err')}
    b.need({n for n in full['files'] if n.startswith('control/logs/')}==logs,
           'exact current-job stdout/stderr roster required')
    members={'control/inputs/cpu-packet.json',*('control/inputs/'+n for n in SERVICES.values()),*logs}
    b.need(members<=set(full['files']), 'required exact current control metadata/log member missing')
    return dict(job_id=job,members=tuple(sorted(members)),cpu_acceptance_sha256=inner['cpu_acceptance_sha256'],
                cpu_output=paths['cpu_output'],campaign_root=paths['campaign'])


def _selection(b, request):
    receipt=request['receipt']
    b.need(request['action']=='batch_gpu72_recovery_audit' and request['helper_sha256']==BACKEND_SHA256
           and request['extension_sha256']==_BOUND[1], 'bound audit operation differs')
    receipt_raw=_pinned_object(b,receipt,request['receipt_sha256'],'GPU custody receipt')
    full=b.validate_receipt(receipt)
    for key in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256'):
        b.need(receipt[key]==b.sha(b.raw(full)), 'full before/after/project archive inventory binding differs')
    seal=request['custody_seal'];_pinned_object(b,seal,request['custody_seal_sha256'],'GPU custody seal')
    b.need(seal['identical'] is True and seal['acceptance'] is False
           and type(seal['copies']) is dict and len(seal['copies'])==2, 'two original compact custody views required')
    views=list(seal['copies'].values());b.need(views[0]==views[1], 'sealed compact copies differ')
    view=b.normalized(views[0])
    b.need(view['files'].get('remote-custody.json')==dict(bytes=len(receipt_raw),sha256=request['receipt_sha256']),
           'seal does not bind original GPU receipt bytes')
    remote={'files':{n[7:]:v for n,v in view['files'].items() if n.startswith('remote/')},
            'directories':sorted(n[7:] for n in view['directories'] if n.startswith('remote/'))}
    b.need(remote==receipt['local_inventory'], 'seal selected subtree differs')
    b.need(receipt['roots'].get('control')==CONTROL_ROOT and receipt['roots'].get('launch')==FAILED_GPU_ROOT,
           'receipt must distinguish new control from frozen old launch')
    control=_control_binding(b,full,request['control_binding'],request['control_binding_sha256'])
    b.need(control['cpu_output']==receipt['roots']['cpu']+'/cpu', 'control packet CPU physical root differs')
    cpu=request['cpu_acceptance'];cpu_raw=_pinned_object(b,cpu,request['cpu_acceptance_sha256'],'accepted CPU')
    b.need(set(cpu)==CPU_KEYS and cpu['schema']=='ace-scientific-campaign-cpu-acceptance-v1'
           and cpu['status']=='development_cpu_phase_accepted' and cpu['scientific_qualified'] is False,
           'independently accepted CPU development record required; no GPU acceptance prerequisite')
    b.need(full['files'].get('control/inputs/cpu-acceptance.json')==dict(bytes=len(cpu_raw),sha256=request['cpu_acceptance_sha256']),
           'current GPU02 control captured accepted CPU bytes differ')
    b.need(control['cpu_acceptance_sha256']==request['cpu_acceptance_sha256'], 'control/independent CPU acceptance pin differs')
    b.need(cpu['campaign_root']==receipt['roots']['cpu']+'/cpu/worker/artifacts/campaign', 'accepted CPU campaign root differs')
    for name,h in cpu['campaign_files'].items():
        b.name_ok(name)
        b.need(full['files'].get(CAMPAIGN+name,{}).get('sha256')==h, 'accepted CPU campaign input changed')
    source=base64.b64decode(request['gpu_audit_source'],validate=True)
    review=base64.b64decode(request['gpu_audit_review'],validate=True)
    b.need(0<len(source)<=256*1024 and b.pin(request['gpu_audit_sha256'])
           and b.sha(source)==request['gpu_audit_sha256'],
           'exact independently reviewed GPU audit source required; never import it')
    b.need(0<len(review)<=128*1024 and review.strip() and b.pin(request['gpu_audit_review_sha256'])
           and b.sha(review)==request['gpu_audit_review_sha256'], 'independent GPU audit review bytes/pin required')
    members=request['members'];expected=select_members(full,control_binding=request['control_binding'],control_binding_sha256=request['control_binding_sha256'])
    b.need(type(members) is dict and members==expected, 'exact complete inventory-derived GPU audit selection differs')
    _pinned_object(b,members,request['members_sha256'],'member map')
    directories=sorted({'/'.join(n.split('/')[:i]) for n in members for i in range(1,len(n.split('/')))})
    return full,b.normalized({'files':members,'directories':directories})


def _result(b,request,full,verification):
    return dict(schema='ace-gpu72-recovery-audit-batch-transfer-v2',
        control_binding_sha256=request['control_binding_sha256'],
        job_id=request['control_binding']['result']['job_id'],
        inner_packet_sha256=request['control_binding']['inner_sha256'],outer_packet_sha256=request['control_binding']['outer_sha256'],
        control_root=CONTROL_ROOT,failed_gpu01_root=FAILED_GPU_ROOT,gpu_output=GPU_OUTPUT,
        receipt_sha256=request['receipt_sha256'],members_sha256=request['members_sha256'],
        extension_sha256=request['extension_sha256'],helper_sha256=BACKEND_SHA256,
        custody_seal_sha256=request['custody_seal_sha256'],cpu_acceptance_sha256=request['cpu_acceptance_sha256'],
        gpu_audit_sha256=request['gpu_audit_sha256'],gpu_audit_review_sha256=request['gpu_audit_review_sha256'],
        full_inventory_sha256=b.sha(b.raw(full)),verification_sha256=b.sha(b.raw(verification)),
        backup_archive_sha256=request['receipt']['backup_archive_sha256'],destination=request['destination'],
        member_count=len(request['members']),payload_bytes=sum(v['bytes'] for v in request['members'].values()),
        full_backup_verified=True,full_archive_local_copy=False,original_receipt_unchanged=True,
        acceptance=False,retry_allowed=False)


def _verification(b, request, full):
    return dict(full_backup_verified=True, full_inventory=full,
        local_inventory=request['receipt']['local_inventory'], receipt_sha256=request['receipt_sha256'],
        backup_archive_sha256=request['receipt']['backup_archive_sha256'],
        helper_sha256=BACKEND_SHA256, acceptance=False)


def _remote_emit(request, output):
    b = _backend()
    full, selection = _selection(b, request)
    # Pass1 includes original durable receipt/inventory and complete archive validation.
    verified = b.remote_verify(dict(receipt=request['receipt'], observation=full, helper_sha256=BACKEND_SHA256))
    b.need(verified == _verification(b, request, full), 'original full verification differs')
    result = _result(b, request, full, verified)
    framed = b.raw(dict(result=result, stream_files=selection['files']))
    b.need(len(framed) <= b.MAX_METADATA, 'bounded batch frame required')
    output.write(len(framed).to_bytes(8, 'big'))
    output.write(framed)
    # Pass2 validates EVERY member again while streaming only exact opaque audit metadata.
    actual = b.check_tar(request['receipt']['backup_archive_path'], full,
                         emit=output.write, wanted=selection['files'])
    b.need(actual == dict(bytes=request['receipt']['backup_archive_bytes'],
                          sha256=request['receipt']['backup_archive_sha256']), 'archive changed during streaming')
    output.flush()


def _specialize(function, **bindings):
    """Reuse captured code with isolated globals; never patch the original backend."""
    return types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                              function.__name__, function.__defaults__, function.__closure__)


BOOTSTRAP = r'''import sys,hashlib,json,base64
def need(ok,message):
 if not ok:raise ValueError(message)
line=sys.stdin.buffer.readline(32)
need(line.endswith(b'\n') and line[:-1].isdigit(),'bounded source bundle framing required')
n=int(line);need(0<n<=1048576,'source bundle size bound')
source=sys.stdin.buffer.read(n)
need(len(source)==n and hashlib.sha256(source).hexdigest()==sys.argv[1],'independent source bundle SHA differs')
def pairs(items):
 out={}
 for k,v in items:
  need(k not in out,'duplicate source bundle key');out[k]=v
 return out
bundle=json.loads(source,object_pairs_hook=pairs)
need(set(bundle)=={'extension','extension_sha256','backend','backend_sha256'},'source bundle keys differ')
extension=base64.b64decode(bundle['extension'],validate=True)
backend=base64.b64decode(bundle['backend'],validate=True)
for raw,pin in ((extension,bundle['extension_sha256']),(backend,bundle['backend_sha256'])):
 need(0<len(raw)<=262144 and hashlib.sha256(raw).hexdigest()==pin,'individual source SHA differs before exec')
need(bundle['backend_sha256']=='4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f','original backend SHA differs')
v={'__name__':'captured_batch_reports','__file__':'<captured-batch-reports>'}
exec(compile(extension,v['__file__'],'exec'),v)
v['bind_sources'](extension_source=extension,extension_sha256=bundle['extension_sha256'],
 backend_source=backend,backend_sha256=bundle['backend_sha256'])
b=v['_backend']();data=sys.stdin.buffer.read(b.MAX_METADATA+1)
need(len(data)<=b.MAX_METADATA,'bounded request metadata required')
v['_specialize'](b.remote_main,remote_emit=v['_remote_emit'])(b.decode(data),sys.stdout.buffer)
'''


def _bundle(b):
    extension, epin, backend, bpin = _BOUND[:4]
    value = b.raw(dict(extension=base64.b64encode(extension).decode(), extension_sha256=epin,
                       backend=base64.b64encode(backend).decode(), backend_sha256=bpin))
    b.need(len(value) <= 1024**2, 'bounded authenticated source bundle required')
    return value


@contextlib.contextmanager
def _transcript(b, logs):
    """Reuse backend owned-file cleanup without masking an earlier transfer error."""
    cleanup = logs/'transcript-cleanup'
    b.owned_dir(cleanup, True)
    owner = b.Receiver({}, {}, lambda result: None)
    primary = None
    duplicate = None
    try:
        handle = b.opened(logs/'stdout.bin', os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        fd = handle.__enter__()
        owner.handle = handle
        duplicate = os.dup(fd)
        owner.fd = os.fdopen(duplicate, 'wb')
        duplicate = None
        yield owner.fd
    except BaseException as exc:
        primary = exc
        raise
    finally:
        first_cleanup = None
        actions = []
        if owner.fd is not None:
            actions += [('transcript_flush', owner.fd.flush),
                        ('transcript_fsync', lambda: os.fsync(owner.fd.fileno()))]
        if duplicate is not None:
            actions.append(('unadopted_duplicate_close', lambda: os.close(duplicate)))
        for phase, action in actions:
            try:
                action()
            except BaseException as exc:
                if first_cleanup is None:
                    first_cleanup = exc
                owner.cleanup_errors.append(dict(phase=phase, type=type(exc).__name__, message=str(exc)))
        # Original close_receiver attempts BOTH buffered and underlying-handle
        # closes, publishes secondary errors, and preserves the supplied primary.
        b.close_receiver(owner, cleanup, primary if primary is not None else first_cleanup)
        if primary is None and first_cleanup is not None:
            raise first_cleanup


def fetch_audit(*, receipt, receipt_sha256, custody_seal, custody_seal_sha256,
                cpu_acceptance, cpu_acceptance_sha256, gpu_audit_source, gpu_audit_sha256,
                gpu_audit_review, gpu_audit_review_sha256, members, members_sha256,
                control_binding, control_binding_sha256, destination, helper_path, log_dir):
    """ONE exclusive batch under both existing locks. No GPU acceptance required.

    Caller pins canonical metadata bytes and independently reviewed audit source.
    Nothing executes the audit or grants execution/utility acceptance here.
    """
    with coordinator_locks():
        return _fetch_audit(receipt=receipt,receipt_sha256=receipt_sha256,custody_seal=custody_seal,
            custody_seal_sha256=custody_seal_sha256,cpu_acceptance=cpu_acceptance,
            cpu_acceptance_sha256=cpu_acceptance_sha256,gpu_audit_source=gpu_audit_source,
            gpu_audit_sha256=gpu_audit_sha256,gpu_audit_review=gpu_audit_review,
            gpu_audit_review_sha256=gpu_audit_review_sha256,members=members,members_sha256=members_sha256,
            control_binding=control_binding,control_binding_sha256=control_binding_sha256,
            destination=destination,helper_path=helper_path,log_dir=log_dir)


def _protect_original_roots(b,request,dest,logs):
    """No claim may overlap either original capsule or its selected evidence root."""
    copies=tuple(b.absolute(p) for p in request['custody_seal']['copies'])
    selected=b.absolute(request['receipt']['local_root'])
    b.need(len(copies)==2 and copies[0]!=copies[1], 'two distinct original capsule roots required')
    protected=(*copies,selected)
    for root in protected:
        b.need(root==root.resolve(strict=True) and root.is_dir(), 'canonical existing custody directory required')
        with b.opened(root,directory=True) as fd:
            b.need(os.path.samestat(os.fstat(fd),root.lstat()), 'original custody directory identity changed')
    b.need(selected in tuple(p/'remote' for p in copies), 'recorded selected root outside original capsule copies')
    for candidate in (dest,logs):
        b.need(candidate==candidate.resolve(strict=False), 'canonical new output/log spelling required')
        b.need(all(candidate!=root and candidate not in root.parents and root not in candidate.parents
                   for root in protected), 'output/log overlaps original compact custody or selected evidence')


def _fetch_audit(**args):
    b=_backend()
    request=dict(args,action='batch_gpu72_recovery_audit',helper_sha256=BACKEND_SHA256,extension_sha256=_BOUND[1])
    request['gpu_audit_source']=base64.b64encode(request['gpu_audit_source']).decode()
    request['gpu_audit_review']=base64.b64encode(request['gpu_audit_review']).decode()
    destination=request['destination']=str(request['destination'])
    helper_path=request.pop('helper_path');log_dir=request.pop('log_dir')
    request=b.decode(b.raw(request))  # detach all caller containers
    b.need(len(b.raw(request))<=b.MAX_METADATA,'unchanged256MiB combined request bound')
    full, selection = _selection(b, request)
    members=request['members']  # validated detached map, never the caller's mutable dictionary
    dest, logs = b.absolute(destination), b.absolute(log_dir)
    b.need(not os.path.lexists(dest) and not os.path.lexists(logs), 'exclusive new destinations required')
    b.need(dest != logs and dest not in logs.parents and logs not in dest.parents, 'disjoint destinations required')
    _protect_original_roots(b,request,dest,logs)
    bundle = _bundle(b)
    # Retain sources, framed stdout (including malformed/partial frames), request,
    # original receipt and operation records. Account on each actual filesystem.
    extra = 3*len(b.raw(request)) + 3*len(bundle) + b.MAX_METADATA + b.LOCAL_TOTAL + 1024**2
    logs = b.begin(logs, request, [('opaque_audit_payload', dest.parent,
                   sum(v['bytes'] for v in members.values())), ('additional_evidence', logs.parent, extra)])
    receiver = None
    primary = None
    try:
        b.write_new(logs/'extension.py', _BOUND[0])
        b.write_new(logs/'source-bundle.json', bundle)
        b.write_new(logs/'bootstrap.py', BOOTSTRAP.encode())
        b.write_new(logs/'original-receipt.json', b.raw(request['receipt']))
        b.write_new(logs/'members.json', b.raw(members))
        b.write_new(logs/'gpu-audit-source.py', base64.b64decode(request['gpu_audit_source']))
        b.write_new(logs/'gpu-audit-review.md', base64.b64decode(request['gpu_audit_review']))
        b.write_new(logs/'custody-seal.json', b.raw(request['custody_seal']))
        b.write_new(logs/'accepted-cpu.json', b.raw(request['cpu_acceptance']))
        b.write_new(logs/'recovery-control-binding.json', b.raw(request['control_binding']))
        # Do not inherit backend local_destinations' empty-directory adoption.
        # This exclusive mkdir is the actual claim, after the earlier precheck.
        b.owned_dir(dest, True)
        destinations = b.local_destinations(dest, selection)
        expected = _result(b, request, full, _verification(b, request, full))
        def validate(result):
            b.need(result == expected, 'batch verification identity differs')
        receiver = b.Receiver(selection['files'], destinations, validate)
        transport = _specialize(b.transport, BOOTSTRAP=BOOTSTRAP,
                                _BOUND_SOURCE=(bundle, b.sha(bundle)))
        with _transcript(b, logs) as transcript:
            class RetainedReceiver:
                allowed = receiver.allowed
                def feed(self, data):
                    transcript.write(data); transcript.flush()
                    return receiver.feed(data)
                def finish(self):
                    return receiver.finish()
            transport(request, RetainedReceiver(), helper_path, logs)
        result = receiver.finish()
        b.close_receiver(receiver, logs, None)
        receiver = None
        b.need(b.local_inventory(dest) == selection, 'opaque report readback differs')
        result = dict(result, local_inventory=selection, original_selected_inventory_modified=False,
                      source_bundle_sha256=b.sha(bundle), schema='ace-gpu72-recovery-audit-batch-retrieval-v2')
        b.write_new(logs/'fetched.json', b.raw({n:dict(path=str(dest/n),sha256=v['sha256']) for n,v in members.items()}))
        b.write_new(logs/'receipt.json', b.raw(result))
        return result
    except BaseException as exc:
        primary = exc
        b.safe_failure(logs, exc, partial_preserved=True, remote_outcome_ambiguous=True)
        raise
    finally:
        cleanup_error = None
        try:
            b.close_receiver(receiver, logs, primary)
        except BaseException as exc:
            cleanup_error = exc
            raise
        finally:
            try:
                b.write_new(logs/'operation-records.json', b.raw(b.local_inventory(logs)))
            except BaseException as secondary:
                first = primary if primary is not None else cleanup_error
                if first is None:
                    raise
                try:
                    b.write_new(logs/'evidence-publication-error.json', b.raw(dict(
                        primary_type=type(first).__name__, secondary_type=type(secondary).__name__,
                        secondary_message=str(secondary), acceptance=False)))
                except BaseException:
                    pass
