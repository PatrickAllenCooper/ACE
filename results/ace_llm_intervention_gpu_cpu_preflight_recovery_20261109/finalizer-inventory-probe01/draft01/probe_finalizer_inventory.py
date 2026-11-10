"""Read-only inventory timing; never a finalizer, scheduler, or acceptance writer."""
from pathlib import Path
import argparse,ast,hashlib,json,os,re,signal,stat,sys,time,types

SCHEMA='ace-finalizer-inventory-timing-request-v1'
REFERENCE_SHA='b4d18f4be2320069ffc5f2aed2060efcfb8d49bfcc6e8b00f8099b91598f8745'
FUNCTIONS={'need','sha','pin','relative','read','inventory','checked_inventory'}
ROLES={'reference_source','cpu_acceptance','gpu_acceptance','gpu_recovery_acceptance'}
PHASES=(('parent_auth_cpu','cpu'),('parent_auth_gpu','gpu'),('parent_auth_campaign','campaign'),
        ('worker_auth_cpu','cpu'),('worker_auth_gpu','gpu'),('worker_auth_campaign','campaign'),
        ('worker_report_inventory_proxy','campaign'),('parent_report_inventory_proxy','campaign'),
        ('parent_post_cpu','cpu'),('parent_post_gpu','gpu'))
UNMEASURED=('source/input authentication outside these inventories','runtime archive staging and guards/imports',
    'eight policy-attempt inventories and up to192 arm validations','private report evaluation/publication',
    'new private-report bytes in the two campaign postcheck passes','report semantic validation',
    'worker creation, process cleanup and scheduler/resource validation')

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def pin(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None
def raw(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def decode(value):
    def unique(items):
        result={}
        for k,v in items:need(k not in result,'duplicate metadata key');result[k]=v
        return result
    result=json.loads(value,object_pairs_hook=unique);raw(result);return result
def capture(path,expected,maximum):
    need(pin(expected),'independent byte pin required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'bounded regular metadata required')
        value=f.read(maximum+1);after=os.fstat(f.fileno())
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(len(value)<=maximum and ident(before)==ident(after) and sha(value)==expected,'captured metadata changed or pin differs')
    return value
def save(path,value):
    with Path(path).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())

def captured_inventory(source,expected):
    """Compile only seven exact function definitions, never the producer module."""
    need(expected==REFERENCE_SHA and sha(source)==expected,'exact reviewed inventory reference required')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    need(len(nodes)==len(FUNCTIONS) and not any(n.decorator_list for n in nodes),'exact plain function roster required')
    ns=dict(Path=Path,os=os,stat=stat,re=re,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'captured-finalizer-inventories','exec'),ns)
    return ns

def admit(data,pins,ns):
    """Only independently pinned acceptance metadata is decoded; payloads remain opaque."""
    cpu=decode(data['cpu_acceptance']);gpu=decode(data['gpu_acceptance']);recovery=decode(data['gpu_recovery_acceptance'])
    need(cpu['schema']=='ace-scientific-campaign-cpu-acceptance-v1' and cpu['status']=='development_cpu_phase_accepted'
         and cpu['scientific_qualified'] is False and cpu['cpu_output_excluded_directory']=='worker/artifacts/campaign',
         'independent actual CPU campaign acceptance required')
    need(gpu['schema']=='ace-scientific-campaign-gpu-acceptance-v1' and gpu['status']=='development_gpu_phase_accepted'
         and gpu['scope']=='development' and gpu['scientific_qualified'] is False
         and raw(gpu['accepted_checks'])==raw(dict(scheduler=True,runtime=True,cleanup=True,custody=True))
         and gpu['cpu_acceptance_sha256']==pins['cpu_acceptance'] and gpu['cpu_packet_sha256']==cpu['cpu_packet_sha256']
         and gpu['bundle_sha256']==cpu['bundle_sha256'],'independent actual complete GPU acceptance required')
    need(recovery['schema']=='ace-scientific-campaign-gpu-recovery-acceptance-v1' and recovery['status']=='accepted'
         and recovery['scope']=='development' and type(recovery['required_findings']) is int and recovery['required_findings']==0
         and recovery['acceptance'] is recovery['historical_postcustody_verified'] is recovery['originals_preserved'] is True
         and recovery['scientific_qualified'] is False and recovery['gpu_acceptance_sha256']==pins['gpu_acceptance']
         and recovery['cpu_acceptance_sha256']==pins['cpu_acceptance'] and recovery['gpu_packet_sha256']==gpu['gpu_packet_sha256']
         and recovery['job_id']==gpu['job_id'] and re.fullmatch('[1-9][0-9]*',gpu['job_id']),
         'future GPU acceptance is mandatory; preflight/aid/exit0 cannot substitute')
    roots=dict(cpu=cpu['cpu_output'],gpu=gpu['gpu_output'],campaign=gpu['campaign_root'])
    need(all(type(v) is str and Path(v).is_absolute() and str(Path(v))==v and '..' not in Path(v).parts for v in roots.values())
         and roots['campaign']==cpu['campaign_root']==str(Path(roots['cpu'])/'worker/artifacts/campaign')
         and recovery['output_root']==roots['gpu'] and recovery['cpu_campaign_root']==roots['campaign'], 'accepted roots differ')
    cp,gp=Path(roots['cpu']).resolve(),Path(roots['gpu']).resolve()
    need(cp!=gp and cp not in gp.parents and gp not in cp.parents,'CPU/GPU roots overlap')
    maps=dict(cpu=cpu['cpu_output_files'],gpu=gpu['gpu_output_files'],campaign=gpu['campaign_files'])
    for value in (*maps.values(),cpu['campaign_files']):
        need(type(value) is dict and value and len(value)<=262144
             and all(ns['relative'](n) and pin(h) for n,h in value.items()),'complete bounded accepted file roster required')
    need(all(maps['campaign'].get(n)==h for n,h in cpu['campaign_files'].items())
         and not any(n.startswith('worker/artifacts/campaign/') for n in maps['cpu'])
         and not any(n.split('/')[0] in {'private','finalization-intent.json','finalization.json','finalization.candidate.json'} for n in maps['campaign']),
         'accepted pre-report campaign required; no previous finalization')
    return roots,maps

def phases(ns,roots,maps,out,emit,deadline_at):
    """Time unchanged inventories; wrapper counts only completed opaque reads."""
    original_read=ns['read'];results=[];current={};observed=None
    def check_time():
        if time.monotonic()>=deadline_at:raise TimeoutError('absolute inventory measurement deadline')
    def counted_read(path,*args,**kwargs):
        check_time();current['inflight_member']=str(path)
        value=original_read(path,*args,**kwargs)
        current['files_read']+=1;current['bytes_read']+=len(value);current['inflight_member']=None
        if current['files_read']%1000==0:emit(dict(event='read_progress',**current))
        check_time();return value
    ns['read']=counted_read
    try:
        for index,(name,role) in enumerate(PHASES):
            current=dict(phase=name,role=role,files_read=0,bytes_read=0,inflight_member=None)
            wall=time.monotonic();cpu=time.process_time();observed=None;failure=None
            emit(dict(event='phase_start',**current,expected_files=len(maps[role])))
            def measured(root,**kwargs):
                nonlocal observed
                observed=ns['inventory'](root,**kwargs);return observed
            try:
                check_time()
                kwargs={'exclude':'worker/artifacts/campaign'} if role=='cpu' else {}
                ns['checked_inventory'](types.SimpleNamespace(inventory=measured),roots[role],maps[role],**kwargs)
                check_time()
            except BaseException as exc:
                failure=dict(type=type(exc).__name__,message=str(exc))
            record=dict(**current,wall_seconds=time.monotonic()-wall,process_cpu_seconds=time.process_time()-cpu,
                expected_files=len(maps[role]),expected_roster_sha256=sha(raw(maps[role])),
                observed_roster_sha256=None if observed is None else sha(raw(observed)),
                complete_membership_and_hashes_matched=failure is None,
                pre_report_campaign_proxy=name.endswith('_proxy'),failure=failure)
            if observed is not None and observed!=maps[role]:save(out/f'phase-{index:02d}-mismatch-inventory.json',observed)
            save(out/f'phase-{index:02d}.json',record);emit(dict(event='phase_end',**record));results.append(record)
            if failure is not None:break
    finally:ns['read']=original_read
    return results

def main():
    origin=time.monotonic();origin_cpu=time.process_time();previous=signal.getsignal(signal.SIGALRM)
    hard_at=origin+900;hard_mode=False;out=None;events=None;result=None
    def alarm(signum,frame):
        nonlocal hard_mode
        if hard_mode:os._exit(124)
        hard_mode=True;signal.setitimer(signal.ITIMER_REAL,max(.001,hard_at-time.monotonic()))
        raise TimeoutError('absolute probe deadline; retained partial evidence, no retry')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,900)
    try:
        ap=argparse.ArgumentParser(description=__doc__)
        for n in ('request','request-sha256','output'):ap.add_argument('--'+n,required=True)
        a=ap.parse_args();request_raw=capture(a.request,a.request_sha256,64*1024);q=decode(request_raw)
        need(set(q)=={'schema','probe_sha256','soft_seconds','hard_seconds','inputs'} and q['schema']==SCHEMA
             and type(q['soft_seconds']) is int and 1<=q['soft_seconds']<=880
             and type(q['hard_seconds']) is int and q['soft_seconds']+5<=q['hard_seconds']<=900
             and set(q['inputs'])==ROLES,'explicit bounded request and all actual acceptance inputs required')
        hard_at=origin+q['hard_seconds'];soft_at=origin+q['soft_seconds']
        need(time.monotonic()<soft_at,'request admission exhausted original deadline')
        signal.setitimer(signal.ITIMER_REAL,max(.001,soft_at-time.monotonic()))
        self_raw=capture(__file__,q['probe_sha256'],1024**2)
        pins={};data={}
        for role,spec in q['inputs'].items():
            need(set(spec)=={'path','sha256'} and type(spec['path']) is str and Path(spec['path']).is_absolute(), 'explicit absolute input path/pin required')
            pins[role]=spec['sha256'];data[role]=capture(spec['path'],pins[role],1024**2 if role=='reference_source' else 128*1024**2)
        ns=captured_inventory(data['reference_source'],pins['reference_source']);roots,maps=admit(data,pins,ns)
        out=Path(a.output).absolute();parent=out.parent.resolve(strict=True)
        need(parent==out.parent and not os.path.lexists(out),'canonical exclusive output required')
        protected=[Path(v).resolve() for v in roots.values()]+[Path(v['path']).resolve() for v in q['inputs'].values()]+[Path(a.request).resolve(),Path(__file__).resolve()]
        need(all(out!=p and out not in p.parents and p not in out.parents for p in protected),'measurement output overlaps original evidence')
        out.mkdir(mode=0o700)
        events=(out/'events.jsonl').open('xb')
        def emit(value):
            value={**value,'elapsed_seconds':time.monotonic()-origin,'process_cpu_seconds_total':time.process_time()-origin_cpu}
            events.write(raw(value));events.flush();os.fsync(events.fileno())
        emit(dict(event='admitted',request_sha256=a.request_sha256,probe_sha256=q['probe_sha256'],input_pins=pins,
                  roots=roots,physical_roots={k:str(Path(v).resolve()) for k,v in roots.items()},
                  soft_deadline_monotonic=soft_at,hard_deadline_monotonic=hard_at))
        for name,value in {'request.json':request_raw,'probe-source.py':self_raw,**{role+('.py' if role=='reference_source' else '.json'):v for role,v in data.items()}}.items():
            with (out/name).open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        records=phases(ns,roots,maps,out,emit,soft_at)
        complete=len(records)==len(PHASES) and all(r['complete_membership_and_hashes_matched'] for r in records)
        # Captured metadata must still match; failures never trigger a second measurement attempt.
        for role,spec in q['inputs'].items():capture(spec['path'],pins[role],1024**2 if role=='reference_source' else 128*1024**2)
        capture(__file__,q['probe_sha256'],1024**2);capture(a.request,a.request_sha256,64*1024)
        result=dict(status='inventory_measurement_complete' if complete else 'inventory_measurement_failed',
            request_sha256=a.request_sha256,probe_sha256=q['probe_sha256'],input_pins=pins,phases=records,
            all_ten_file_rosters_verified=complete,metadata_pins_unchanged=True,
            unmeasured=list(UNMEASURED),acceptance=False,runtime_qualified=False,scientific_qualified=False,
            retry_allowed=False,elapsed_seconds=time.monotonic()-origin,process_cpu_seconds=time.process_time()-origin_cpu)
        save(out/'result.json',result);return 0 if complete else 1
    except BaseException as exc:
        if out is not None and out.is_dir():
            # The previously armed original hard deadline also bounds failure publication.
            try:save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-origin,
                process_cpu_seconds=time.process_time()-origin_cpu,acceptance=False,runtime_qualified=False,retry_allowed=False))
            except BaseException as secondary:print('failure publication: '+repr(secondary),file=sys.stderr)
        raise
    finally:
        if events is not None:events.close()
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

if __name__=='__main__':sys.exit(main())
