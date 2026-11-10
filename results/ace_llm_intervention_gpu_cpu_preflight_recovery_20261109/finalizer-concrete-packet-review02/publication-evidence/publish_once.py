"""Authorized opaque byte-copy publication only; no producer or remote code."""
from pathlib import Path,PurePosixPath
import datetime,fcntl,hashlib,json,os,resource,stat,time
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
PREP=REPO/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
EVIDENCE=Path(__file__).resolve().parent
MIRROR=Path((EVIDENCE/'external-path.txt').read_text().strip())
AUTHOR=PREP/'finalizer-outer-reader128-preparation01'
OLD=(REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01',BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01')
NEW=(OLD[0].with_name('launch02'),BASE/'ACE-LLM-intervention-scientific-finalization-20261108-02')
OUTER='1a30c9bdfef0b81f98089c774efa5ff781ddc2b5bc8411c9d08e3b198adb87d3'
INNER='c71bafdf9f5a791c540c6a5a1a12ba0f2b84b9b221c2b018cbdebf9684728571'
REPLACEMENTS={'build_packet.py':'e46cd29633c9dcd83db174fb77ff5c7acf751553bfb46963129cdb1129a255cb',
 'launch_once.py':'4508ff44bd2a7c0d8de678bfc18d54b73cf6c2829dfe62835285e1a344e2991c',
 'monitor_once.py':'4d22fb95146f6e308fab021f35023a491909d03af49b9b5ea4d4bc5c618cf373'}
COLLECTOR='b1994191042e910923ffe23e504f2ecd9af6d5242ad6743a5537a2aa4e73e5c3'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(v):return hashlib.sha256(v).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
def capture(path,expected,load=False):
    p=Path(path);need(p==p.resolve(strict=True),'noncanonical source path')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_nlink==1 and a.st_size<=256*1024**2,'bounded sole-linked file required')
        h=hashlib.sha256();blocks=[]
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
            if load:blocks.append(block)
        b=os.fstat(f.fileno())
    need(identity(a)==identity(b)==identity(p.lstat()) and h.hexdigest()==expected,'file identity/hash differs: '+str(p))
    return dict(bytes=b.st_size,sha256=expected,identity=identity(b)),b''.join(blocks) if load else None
def write(path,data):
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def save(name,value):
    data=raw(value)
    for root in (EVIDENCE,MIRROR):write(root/name,data)
def roster(root):
    need(root==root.resolve(strict=True) and root.is_dir(),'canonical packet directory required')
    files=set();dirs=set()
    for p in root.rglob('*'):
        s=p.lstat();n=p.relative_to(root).as_posix()
        need(not p.is_symlink(),'packet alias forbidden')
        if stat.S_ISDIR(s.st_mode):dirs.add(n)
        else:need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'packet special/hardlinked member');files.add(n)
    return files,dirs
locks=[];created=[];started=time.monotonic();cpu=time.process_time();failure=None
try:
    save('invocation.intent.json',dict(utc=utc(),source_sha256=sha(Path(__file__).read_bytes()),
        scope='one metadata-only operational publication; three root sources and outer pins only',
        original_outer_sha256=OUTER,preserved_inner_sha256=INNER,old_roots=list(map(str,OLD)),new_roots=list(map(str,NEW)),retry_allowed=False))
    for n in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):
        p=BASE/n;need(p==p.resolve(),'lock alias');f=os.fdopen(os.open(p,os.O_RDWR|os.O_NOFOLLOW),'r+');locks.append(f)
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    need(not any(os.path.lexists(p) for p in NEW),'successor destination already exists')
    anchors={
      'source-review.md':(PREP/'finalizer-outer-reader128-review01/review.md','df74fcfaf09a0d23b47d956f162b0938d1156efb9bf4a795c32fd1f7560b5fd8'),
      'source-review-seal.json':(PREP/'finalizer-outer-reader128-review01/seal.json','2f2f8580d9b47eb7cc1c2f3c13b15d83aa8ee108ba9a223085f47f9c7bc77560'),
      'publication-plan.json':(AUTHOR/'publication-plan.json','d379c1c572fd3842fbf318419f47764fc571d09f4258d9ad842e801dcdcd37f4'),
      'original-packet-review.md':(PREP/'finalizer-concrete-packet-review01/review.md','4698740072ae466d813cbee8ae8454b1724db832a487cbb35c15b4b2d8bf3213'),
      'failed-launch-disposition.json':(OLD[0].parent/'launch-control01/failure-disposition.json','f4cff65cd1932670473d8f4a7538414c60d7ef106de735b98e7f7ab639e71f28')}
    anchor_records={}
    for name,(path,h) in anchors.items():
        info,data=capture(path,h,True);anchor_records[name]=dict(path=str(path),sha256=h,bytes=info['bytes'])
        for root in (EVIDENCE,MIRROR):write(root/name,data)
    replacement_bytes={}
    for n,h in {**REPLACEMENTS,'pull_terminal.py':COLLECTOR}.items():
        _,data=capture(AUTHOR/'implementation'/n,h,True)
        for root in (EVIDENCE,MIRROR):write(root/('successor-'+n),data)
        if n in REPLACEMENTS:replacement_bytes[n]=data
    _,old_raw=capture(OLD[0]/'packet.json',OUTER,True)
    _,copy_raw=capture(OLD[1]/'packet.json',OUTER,True);need(old_raw==copy_raw,'original outer copies differ')
    outer=json.loads(old_raw);need(raw(outer)==old_raw,'original canonical encoding differs')
    original={**outer['new_input_files'],'packet.json':OUTER};need(len(original)==223,'original223member contract differs')
    for n,h in original.items():
        parts=PurePosixPath(n).parts;need(not PurePosixPath(n).is_absolute() and parts and all(x not in ('','.','..') for x in parts),'unsafe member')
        need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'invalid member pin')
    need(original['inputs/packet.json']==INNER and all(n in original for n in REPLACEMENTS),'original inner/source binding differs')
    before={};member_info={}
    for root in OLD:
        names,dirs=roster(root);need(names==set(original) and dirs=={'inputs'},'original exact membership differs')
        before[str(root)]={}
        for n,h in sorted(original.items()):
            info,_=capture(root/n,h);before[str(root)][n]=info
            if n not in member_info:member_info[n]=dict(bytes=info['bytes'],sha256=h)
            else:need(member_info[n]['bytes']==info['bytes'] and info['identity'][:2]!=before[str(OLD[0])][n]['identity'][:2],'original mirrors not independent')
    successor={**outer,'new_input_files':{**outer['new_input_files'],**REPLACEMENTS}}
    new_raw=raw(successor);new_pin=sha(new_raw);new_expected={**successor['new_input_files'],'packet.json':new_pin}
    need({k for k in outer if outer[k]!=successor[k]}=={'new_input_files'},'non-source outer field changed')
    need({k for k in outer['new_input_files'] if outer['new_input_files'][k]!=successor['new_input_files'][k]}==set(REPLACEMENTS),'unexpected source replacement')
    save('authenticated-inputs.json',dict(anchors=anchor_records,source_pins=REPLACEMENTS,standalone_collector_sha256=COLLECTOR,
        original_members=member_info,original_outer_sha256=OUTER,successor_outer_sha256=new_pin,preserved_inner_sha256=INNER))
    for root in NEW:
        root.mkdir(mode=0o700);created.append(root);(root/'inputs').mkdir(mode=0o700)
    for n,h in sorted(original.items()):
        if n=='packet.json':value=new_raw
        elif n in replacement_bytes:value=replacement_bytes[n]
        else:_,value=capture(OLD[0]/n,h,True)
        need(sha(value)==new_expected[n],'publication value pin differs')
        for root in NEW:write(root/n,value)
    after={}
    for root in NEW:
        names,dirs=roster(root);need(names==set(original) and dirs=={'inputs'},'successor exact membership differs');after[str(root)]={}
        for n,h in sorted(new_expected.items()):
            info,_=capture(root/n,h);after[str(root)][n]=info
            need(info['identity'][:2] not in (before[str(OLD[0])][n]['identity'][:2],before[str(OLD[1])][n]['identity'][:2]),'successor aliases original')
            if root==NEW[1]:need(info['identity'][:2]!=after[str(NEW[0])][n]['identity'][:2],'successor copies not independent')
    for root in OLD:
        need(roster(root)==(set(original),{'inputs'}),'original membership changed')
        for n,info in before[str(root)].items():need(identity((root/n).lstat())==info['identity'],'original identity changed')
    changed={n for n in original if original[n]!=new_expected[n]}
    need(changed==set(REPLACEMENTS)|{'packet.json'},'unexpected changed member')
    save('packet-inventory.json',dict(files={n:{k:v for k,v in info.items() if k!='identity'} for n,info in after[str(NEW[0])].items()},directories=['inputs'],copies=list(map(str,NEW))))
    save('publication.json',dict(status='one_operational_metadata_publication_completed_pending_concrete_delta_review',utc=utc(),
        original_outer_sha256=OUTER,outer_sha256=new_pin,inner_sha256=INNER,changed_members=sorted(changed),
        unchanged_members=len(original)-len(changed),members_per_copy=len(original),bytes_per_copy=sum(v['bytes'] for v in after[str(NEW[0])].values()),
        old_roots=list(map(str,OLD)),new_roots=list(map(str,NEW)),anchors=anchor_records,source_pins=REPLACEMENTS,
        standalone_collector_sha256=COLLECTOR,original_packets_unchanged=True,all_inputs_and_job_storage_authority_bytes_unchanged=True,
        original_storage_build_proofs='preserved historical bytes only; no new storage/build execution claimed',
        builder_invoked=False,remote_actions=0,archive_verifications=0,scientific_executions=0,submission_invoked=False,
        new_concrete_review_required=True,parent_commit_push_match_required=True))
except BaseException as exc:
    failure=exc
    save('failure.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),created_roots=list(map(str,created)),retry_allowed=False))
    raise
finally:
    for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
    save('run.json',dict(utc=utc(),wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,returncode=0 if failure is None else 1,
        locks_released=True,builder_invoked=False,remote_actions=0,submission_invoked=False))
