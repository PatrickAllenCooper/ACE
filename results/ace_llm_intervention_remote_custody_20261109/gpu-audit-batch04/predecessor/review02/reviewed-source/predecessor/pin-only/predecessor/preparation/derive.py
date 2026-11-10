from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'predecessor/batch_reports.py').read_text()
a=s.index('"""');z=s.index('"""',a+3)+3
s='''"""Pinned opaque GPU72 audit materialization; never execution acceptance.

Derivative of unchanged final-reports-batch02. Full archive and exact selection
are checked before and during streaming. No network or locks at import.
"""'''+s[z:]
s=s.replace('import contextlib\n','import contextlib\nimport fcntl\nimport re\nimport stat\n')
a=s.index("PREFIX =");z=s.index('\ndef _require',a)
s=s[:a]+'''CAMPAIGN = 'cpu/cpu/worker/artifacts/campaign/'
GPU_AUDIT_SHA256 = '9e3d5d00f43dbb478afdb3446cb1f5ea11cebb656a5aec4c834dd5dca7c32c75'
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
    arm = (r'(?:arm-receipt\\.json|execution-diagnostics\\.json|task\\.json|paired-plan\\.json|'
           r'(?:initial|measurement)-world\\.jsonl|return-[0-9]{3}\\.json|'
           r'episode/(?:plan\\.json|journal\\.jsonl|terminal\\.json|failure\\.json))')
    service = (r'service/(?:launch|process|observed-handshake|ready|failure-intent|failure|'
               r'close-cleanup|closed|task-binding-[0-9]{4}|'
               r'request-[0-9]{4}\\.(?:input|observed-response|accepted|episode))\\.json')
    allowed = re.compile(r'policy-0[567]/(?:attempt\\.json|'+service+'|'+world+arm+')')
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


def select_members(full_inventory):
    """Pure prospective map; caller must independently pin it before transfer."""
    b = _backend(); full = b.normalized(full_inventory)
    members = {name:full['files'][name] for name in audit_names(full)}
    b.need(all(info['bytes'] <= b.LOCAL_MEMBER for info in members.values()), 'audit member over32MiB')
    b.need(sum(v['bytes'] for v in members.values()) <= b.LOCAL_TOTAL, 'audit batch over512MiB')
    return members


@contextlib.contextmanager
def coordinator_locks():
    """Acquire the two existing coordinator locks once, nonblocking; never create."""
    handles=[]
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
    finally:
        for fd in reversed(handles):
            try:fcntl.flock(fd,fcntl.LOCK_UN)
            finally:os.close(fd)

''' + s[z:]
a=s.index('def _selection(');z=s.index('\ndef _verification',a)
s=s[:a]+'''def _pinned_object(b, obj, expected, role):
    payload=b.raw(obj)
    b.need(len(payload)<=128*1024**2 and b.pin(expected) and b.sha(payload)==expected,
           'independent canonical128MiB '+role+' pin differs')
    return payload


def _selection(b, request):
    receipt=request['receipt']
    b.need(request['action']=='batch_gpu72_audit' and request['helper_sha256']==BACKEND_SHA256
           and request['extension_sha256']==_BOUND[1], 'bound audit operation differs')
    receipt_raw=_pinned_object(b,receipt,request['receipt_sha256'],'GPU custody receipt')
    full=b.validate_receipt(receipt)
    for key in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256'):
        b.need(receipt[key]==b.sha(b.raw(full)), 'full before/after/project archive inventory binding differs')
    seal=request['custody_seal'];_pinned_object(b,seal,request['custody_seal_sha256'],'GPU custody seal')
    b.need(seal['identical'] is True and seal['complete_local_raw_copy'] is False
           and type(seal['copies']) is dict and len(seal['copies'])==2, 'two original compact custody views required')
    views=list(seal['copies'].values());b.need(views[0]==views[1], 'sealed compact copies differ')
    view=b.normalized(views[0])
    b.need(view['files'].get('remote-custody.json')==dict(bytes=len(receipt_raw),sha256=request['receipt_sha256']),
           'seal does not bind original GPU receipt bytes')
    remote={'files':{n[7:]:v for n,v in view['files'].items() if n.startswith('remote/')},
            'directories':sorted(n[7:] for n in view['directories'] if n.startswith('remote/'))}
    b.need(remote==receipt['local_inventory'], 'seal selected subtree differs')
    cpu=request['cpu_acceptance'];cpu_raw=_pinned_object(b,cpu,request['cpu_acceptance_sha256'],'accepted CPU')
    b.need(set(cpu)==CPU_KEYS and cpu['schema']=='ace-scientific-campaign-cpu-acceptance-v1'
           and cpu['status']=='development_cpu_phase_accepted' and cpu['scientific_qualified'] is False,
           'independently accepted CPU development record required; no GPU acceptance prerequisite')
    b.need(full['files'].get('launch/inputs/cpu-acceptance.json')==dict(bytes=len(cpu_raw),sha256=request['cpu_acceptance_sha256']),
           'GPU captured accepted CPU bytes differ')
    b.need(cpu['campaign_root']==receipt['roots']['cpu']+'/cpu/worker/artifacts/campaign', 'accepted CPU campaign root differs')
    for name,h in cpu['campaign_files'].items():
        b.name_ok(name)
        b.need(full['files'].get(CAMPAIGN+name,{}).get('sha256')==h, 'accepted CPU campaign input changed')
    source=base64.b64decode(request['gpu_audit_source'],validate=True)
    review=base64.b64decode(request['gpu_audit_review'],validate=True)
    b.need(0<len(source)<=256*1024 and b.sha(source)==request['gpu_audit_sha256']==GPU_AUDIT_SHA256,
           'exact independently reviewed GPU audit source required; never import it')
    b.need(0<len(review)<=128*1024 and review.strip() and b.pin(request['gpu_audit_review_sha256'])
           and b.sha(review)==request['gpu_audit_review_sha256'], 'independent GPU audit review bytes/pin required')
    members=request['members'];expected=select_members(full)
    b.need(type(members) is dict and members==expected, 'exact complete inventory-derived GPU audit selection differs')
    _pinned_object(b,members,request['members_sha256'],'member map')
    directories=sorted({'/'.join(n.split('/')[:i]) for n in members for i in range(1,len(n.split('/')))})
    return full,b.normalized({'files':members,'directories':directories})


def _result(b,request,full,verification):
    return dict(schema='ace-gpu72-audit-batch-transfer-v1',
        receipt_sha256=request['receipt_sha256'],members_sha256=request['members_sha256'],
        extension_sha256=request['extension_sha256'],helper_sha256=BACKEND_SHA256,
        custody_seal_sha256=request['custody_seal_sha256'],cpu_acceptance_sha256=request['cpu_acceptance_sha256'],
        gpu_audit_sha256=request['gpu_audit_sha256'],gpu_audit_review_sha256=request['gpu_audit_review_sha256'],
        full_inventory_sha256=b.sha(b.raw(full)),verification_sha256=b.sha(b.raw(verification)),
        backup_archive_sha256=request['receipt']['backup_archive_sha256'],destination=request['destination'],
        member_count=len(request['members']),payload_bytes=sum(v['bytes'] for v in request['members'].values()),
        full_backup_verified=True,full_archive_local_copy=False,original_receipt_unchanged=True,
        acceptance=False,retry_allowed=False)

''' + s[z:]
a=s.index('def fetch_reports(');z=s.index('    full, selection = _selection(b, request)',a)
s=s[:a]+'''def fetch_audit(*, receipt, receipt_sha256, custody_seal, custody_seal_sha256,
                cpu_acceptance, cpu_acceptance_sha256, gpu_audit_source, gpu_audit_sha256,
                gpu_audit_review, gpu_audit_review_sha256, members, members_sha256,
                destination, helper_path, log_dir):
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
            destination=destination,helper_path=helper_path,log_dir=log_dir)


def _fetch_audit(**args):
    b=_backend()
    request=dict(args,action='batch_gpu72_audit',helper_sha256=BACKEND_SHA256,extension_sha256=_BOUND[1])
    request['gpu_audit_source']=base64.b64encode(request['gpu_audit_source']).decode()
    request['gpu_audit_review']=base64.b64encode(request['gpu_audit_review']).decode()
    destination=request['destination']=str(request['destination'])
    helper_path=request.pop('helper_path');log_dir=request.pop('log_dir');members=request['members']
    request=b.decode(b.raw(request))  # detach all caller containers
    b.need(len(b.raw(request))<=b.MAX_METADATA,'unchanged256MiB combined request bound')
''' + s[z:]
s=s.replace("schema='ace-finalizer-report-batch-retrieval-v1'","schema='ace-gpu72-audit-batch-retrieval-v1'")
s=s.replace("('opaque_report_payload',", "('opaque_audit_payload',")
s=s.replace('exact opaque reports.','exact opaque audit metadata.')
s=s.replace("        b.write_new(logs/'members.json', b.raw(members))", "        b.write_new(logs/'members.json', b.raw(members))\n        b.write_new(logs/'gpu-audit-source.py', base64.b64decode(request['gpu_audit_source']))\n        b.write_new(logs/'gpu-audit-review.md', base64.b64decode(request['gpu_audit_review']))\n        b.write_new(logs/'custody-seal.json', b.raw(request['custody_seal']))\n        b.write_new(logs/'accepted-cpu.json', b.raw(request['cpu_acceptance']))")
s=s.replace("        b.write_new(logs/'receipt.json', b.raw(result))", "        b.write_new(logs/'fetched.json', b.raw({n:dict(path=str(dest/n),sha256=v['sha256']) for n,v in members.items()}))\n        b.write_new(logs/'receipt.json', b.raw(result))")
(r/'batch_gpu_audit.py').write_text(s)
