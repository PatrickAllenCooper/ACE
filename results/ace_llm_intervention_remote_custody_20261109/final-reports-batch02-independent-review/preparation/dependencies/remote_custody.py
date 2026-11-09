"""Opaque remote archive custody only. Caller owns authorization/acceptance.

No SSH at import. Explicit source binding, exclusive operations, no retry/delete.
USTAR canonical headers/member checks follow vetted stream_custody fab01 semantics;
this standalone backend has explicit16GiB total/USTAR member bounds, no old helper mutation.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import time
import traceback

CHUNK=1024**2
MAX_TOTAL=16*1024**3
MAX_MEMBER=8*1024**3-1  # Standard USTAR size field; supplied8438650880-byte runtime fits.
MAX_MEMBERS=262144
MAX_METADATA=256*1024**2
LOCAL_MEMBER=32*1024**2
LOCAL_TOTAL=512*1024**2
TIMEOUT=1800
MAX_SOURCE_DEPTH=128  # Absolute components; at most129 owned source FDs, including '/'.
PROGRESS_MEMBERS=1000
CAPACITY_HEADROOM=64*1024**2
PROJECT_PARENT=Path('/projects/paco0228')
SCRATCH_PARENT=Path('/scratch/alpine/paco0228')
PROJECT_BASE=PROJECT_PARENT/'ACE_evidence'
TRANSFER_BASE=SCRATCH_PARENT/'ACE_custody_transfer'
TEXT_SUFFIXES={'.py','.sh','.md','.txt','.utf8','.log','.out','.err','.json','.jsonl','.yaml','.yml','.toml','.cfg','.ini','.csv','.tsv','.sbatch'}
BINARY_SUFFIXES={'.pt','.bin','.tar','.gz','.bz2','.xz','.zip','.npz','.npy','.safetensors','.pkl','.pickle','.so','.dylib','.whl'}
_BOUND_SOURCE=None


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def raw(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def pin(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def decode(data):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(value):raise ValueError('nonfinite JSON')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)


def bind_source(source,expected_sha256):
    global _BOUND_SOURCE
    need(type(source) is bytes and len(source)<=1024**2 and pin(expected_sha256) and sha(source)==expected_sha256,'independent helper source pin required')
    need(_BOUND_SOURCE is None or _BOUND_SOURCE==(source,expected_sha256),'helper binding cannot change')
    _BOUND_SOURCE=(source,expected_sha256)


def name_ok(name):
    need(type(name) is str and name and len(name.encode())<=255 and not name.startswith('/') and '\\' not in name,'safe bounded member name required')
    need(all(x not in ('','.','..') for x in name.split('/')) and all(ord(x)>=32 and ord(x)!=127 for x in name),'member traversal/control forbidden')
    return name


def normalized(observation):
    need(type(observation) is dict and type(observation.get('files')) is dict and type(observation.get('directories')) is list,'full original inventory required')
    files=observation['files'];dirs=observation['directories'];ds=set(dirs)
    need(len(files)+len(dirs)<=MAX_MEMBERS and len(ds)==len(dirs) and not set(files)&ds,'duplicate/colliding inventory members')
    for n in list(files)+dirs:
        name_ok(n)
        for i in range(1,len(n.split('/'))):need('/'.join(n.split('/')[:i]) in ds,'missing parent directory')
    for n,v in files.items():
        need(type(v) is dict and set(v)=={'bytes','sha256'} and type(v['bytes']) is int and 0<=v['bytes']<=MAX_MEMBER and pin(v['sha256']),'bounded member/pin required')
    need(sum(v['bytes'] for v in files.values())<=MAX_TOTAL,'explicit16GiB total bound')
    return {'files':dict(sorted(files.items())),'directories':sorted(dirs)}


def selected_inventory(full):
    files={};omitted={};total=0
    for n,v in sorted(full['files'].items()):
        suffixes=set(Path(n).suffixes)
        if (n.startswith('launch/') or n in ('namespace.json','original-namespace.json')) and Path(n).suffix.lower() in TEXT_SUFFIXES and not {s.lower() for s in suffixes}&BINARY_SUFFIXES and v['bytes']<=LOCAL_MEMBER:
            files[n]=v;total+=v['bytes']
        else:omitted[n]=v
    need(total<=LOCAL_TOTAL,'current-stage selected evidence exceeds512MiB; no partial subset')
    dirs={ '/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/'))) }
    return {'files':files,'directories':sorted(dirs)},omitted


def absolute(path):
    p=Path(path)
    need(p.is_absolute() and str(p)==str(path) and '..' not in p.parts,'absolute canonical spelling required')
    return p


@contextlib.contextmanager
def opened(path,flags=os.O_RDONLY,directory=False):
    """No symlinks at any component; exclusive writes use an owned parent FD."""
    p=absolute(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        leaf=os.open(p.name,flags|os.O_NOFOLLOW|os.O_NONBLOCK|(os.O_DIRECTORY if directory else 0),0o600,dir_fd=fd)
        try:yield leaf
        finally:os.close(leaf)
    finally:os.close(fd)


def identity(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)


def fingerprint(path,sole=False):
    with opened(path) as fd:
        a=os.fstat(fd);need(stat.S_ISREG(a.st_mode) and (not sole or a.st_nlink==1),'regular independent file required')
        h=hashlib.sha256()
        with os.fdopen(os.dup(fd),'rb') as f:
            for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
        need(identity(a)==identity(os.fstat(fd))==identity(Path(path).lstat()),'file changed while hashing')
    return {'bytes':a.st_size,'sha256':h.hexdigest()},identity(a)


def write_new(path,data):
    with opened(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL) as fd:
        with os.fdopen(os.dup(fd),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def safe_failure(root,exc,**extra):
    try:write_new(Path(root)/'failure.json',raw(dict(type=type(exc).__name__,message=str(exc),acceptance=False,retry_allowed=False,**extra)))
    except BaseException:pass  # Never replace the original error; raw transport stderr remains.


def owned_dir(path,exclusive=False):
    p=absolute(path)
    with opened(p.parent,directory=True):pass
    if exclusive:p.mkdir(mode=0o700,exist_ok=False)
    elif not os.path.lexists(p):p.mkdir(mode=0o700,exist_ok=False)
    with opened(p,directory=True) as fd:
        s=os.fstat(fd);need(s.st_uid==os.geteuid(),'directory owner differs')
        if exclusive or stat.S_IMODE(s.st_mode) in (0o700,0o2700):os.fchmod(fd,0o700)
        need(stat.S_IMODE(os.fstat(fd).st_mode)==0o700,'owned0700 directory required')
    return p


def roots_check(roots,full):
    need(type(roots) is dict and roots,'explicit scratch roots required')
    labels={n.split('/')[0] for n in list(full['files'])+full['directories']}
    need(set(roots)==labels,'roots/inventory membership differs')
    paths=[]
    for label,value in roots.items():
        need('/' not in name_ok(label),'top level labels only');p=absolute(value)
        need(p!=SCRATCH_PARENT and SCRATCH_PARENT in p.parents and TRANSFER_BASE not in (p,*p.parents),'original scratch-only source required')
        need(p==p.resolve(strict=True),'source root alias forbidden')
        with opened(p,directory=label in full['directories']):pass
        need(label in full['directories'] or label in full['files'],'root kind differs')
        paths.append(p)
    need(len(set(paths))==len(paths) and not any(p in q.parents for p in paths for q in paths if p!=q),'duplicate/overlapping source roots')


class Progress:
    """Sparse public counters only; never names, contents, hashes or outcomes."""
    def __init__(self,phase):
        self.phase=phase;self.started=time.monotonic();self.cpu=time.process_time()
        self.members=0;self.files=0;self.bytes=0

    def emit(self,state):
        value={'phase':self.phase,'state':state,'monotonic':time.monotonic(),
               'elapsed_seconds':time.monotonic()-self.started,'process_seconds':time.process_time()-self.cpu,
               'members':self.members,'files':self.files,'bytes':self.bytes}
        sys.stderr.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');sys.stderr.flush()

    def member(self,size=None):
        self.members+=1
        if size is not None:self.files+=1;self.bytes+=size
        if self.members%PROGRESS_MEMBERS==0:self.emit('progress')

    def __enter__(self):self.emit('start');return self
    def __exit__(self,kind,error,tb):
        try:self.emit('failed' if error is not None else 'complete')
        except BaseException:
            if error is None:raise
        return False


def _binding(s):return (s.st_dev,s.st_ino,s.st_mode)


class SourceCursor:
    """Bounded no-follow ancestry, with parent-entry checks before releasing FDs.

    Absolute ancestors remain open and are rechecked on close. Source directories
    use full identity; ancestors outside the source use inode/type/mode binding,
    allowing unrelated siblings to change. Cached FDs never substitute for a
    check that the original pathname is still attached to the same object.
    """
    def __init__(self,path,expected=None):
        p=absolute(path);parts=p.parts[1:]
        need(parts and len(parts)<=MAX_SOURCE_DEPTH,'source depth bound')
        self.frames=[];self.root_index=None
        try:
            fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:before=os.fstat(fd)
            except BaseException:
                try:os.close(fd)
                except BaseException:pass
                raise
            self.frames.append((fd,None,None,before,False))
            for i,part in enumerate(parts):
                leaf=i==len(parts)-1
                self.push(part,directory=None if leaf else True,full=leaf,expected=expected if leaf else None)
            self.root_index=len(self.frames)-1
        except BaseException:
            self.close(sys.exc_info()[1]);raise

    @property
    def fd(self):return self.frames[-1][0]
    @property
    def before(self):return self.frames[-1][3]

    def push(self,name,directory=None,full=True,expected=None):
        need(len(self.frames)<=MAX_SOURCE_DEPTH,'source depth bound')
        need(type(name) is str and name not in ('','.','..') and '/' not in name,'one source component required')
        parent=self.fd;before=os.stat(name,dir_fd=parent,follow_symlinks=False)
        isdir=stat.S_ISDIR(before.st_mode)
        need(isdir or stat.S_ISREG(before.st_mode),'ordinary source file/directory required; aliases forbidden')
        need(directory is None or isdir==directory,'source kind differs')
        if expected is not None:need(identity(before)==expected,'source changed before open')
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|(os.O_DIRECTORY if isdir else 0),dir_fd=parent)
        try:
            actual=os.fstat(fd)
            key=identity if full else _binding
            need(key(before)==key(actual),'source entry changed during open')
        except BaseException:
            try:os.close(fd)
            except BaseException:pass  # Preserve the primary open/identity failure.
            raise
        self.frames.append((fd,parent,name,before,full))

    def pop(self,primary=None):
        fd,parent,name,before,full=self.frames.pop();failure=None
        try:
            if primary is None:
                key=identity if full else _binding
                need(key(before)==key(os.fstat(fd)),'source descriptor changed')
                if parent is not None:
                    need(key(before)==key(os.stat(name,dir_fd=parent,follow_symlinks=False)),'source pathname detached or replaced')
        except BaseException as exc:failure=exc
        finally:
            try:os.close(fd)
            except BaseException as exc:
                if primary is None and failure is None:failure=exc
                elif hasattr(primary or failure,'add_note'):(primary or failure).add_note('secondary source FD close failure: '+type(exc).__name__)
        if primary is None and failure is not None:raise failure

    def close(self,primary=None):
        first=None
        while self.frames:
            try:self.pop(primary or first)
            except BaseException as exc:
                if first is None:first=exc
        if primary is None and first is not None:raise first

    @contextlib.contextmanager
    def child(self,name,**kwargs):
        self.push(name,**kwargs)
        try:yield self.fd
        finally:self.pop(sys.exc_info()[1])

    def parent(self,parts,identities,label):
        """Retain only the common directory prefix, preserving global tar order."""
        current=[f[2] for f in self.frames[self.root_index+1:]];common=0
        while common<min(len(current),len(parts)) and current[common]==parts[common]:common+=1
        while len(self.frames)>self.root_index+1+common:self.pop()
        for i in range(common,len(parts)):
            key=label+'/'+('/'.join(parts[:i+1]))
            self.push(parts[i],directory=True,expected=identities[key])

    def __enter__(self):return self
    def __exit__(self,kind,error,tb):self.close(error);return False


def scan(roots,sole=False):
    files={};dirs=[];ids={};links={}
    with Progress('source_scan') as progress:
        def visit(cursor,name):
            name_ok(name);s=cursor.before
            need(len(files)+len(dirs)<MAX_MEMBERS,'member count bound')
            if stat.S_ISDIR(s.st_mode):
                dirs.append(name);ids[name]=identity(s);progress.member()
                names=sorted(os.listdir(cursor.fd))
                need(len(names)<=MAX_MEMBERS-len(files)-len(dirs),'member count bound')
                for child in names:
                    name_ok(name+'/'+child)
                    with cursor.child(child):visit(cursor,name+'/'+child)
                need(sorted(os.listdir(cursor.fd))==names,'source directory membership changed')
            else:
                need(stat.S_ISREG(s.st_mode) and (not sole or s.st_nlink==1),'regular independent file required')
                h=hashlib.sha256()
                for b in iter(lambda:os.read(cursor.fd,CHUNK),b''):h.update(b)
                ident=identity(s);files[name]={'bytes':s.st_size,'sha256':h.hexdigest()};ids[name]=ident
                if ident[-1]>1:links.setdefault((ident[0],ident[1],ident[-1]),[]).append(name)
                progress.member(s.st_size)
        for label,value in sorted(roots.items()):
            name_ok(label)
            with SourceCursor(value) as cursor:visit(cursor,label)
    full=normalized({'files':files,'directories':dirs})
    groups=sorted([{'paths':sorted(v),'link_count':k[2]} for k,v in links.items()],key=lambda v:v['paths'])
    return full,ids,groups


def header(name,size=None):
    t=tarfile.TarInfo(name+'/' if size is None else name);t.type=tarfile.DIRTYPE if size is None else tarfile.REGTYPE
    t.size=0 if size is None else size;t.mode=0o700 if size is None else 0o600;t.uid=t.gid=t.mtime=0
    return t.tobuf(format=tarfile.USTAR_FORMAT,encoding='utf-8',errors='strict')


def tar_size(full):
    n=512*(len(full['directories'])+len(full['files']))+sum(((v['bytes']+511)//512)*512 for v in full['files'].values())+1024
    return ((n+10239)//10240)*10240


def source_path(roots,name):
    bits=name.split('/');return Path(roots[bits[0]]).joinpath(*bits[1:])


def make_tar(path,full,roots,identities):
    for n in full['directories']:header(n)
    for n,v in full['files'].items():header(n,v['bytes'])
    with Progress('archive_build') as progress,opened(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL) as fd,os.fdopen(os.dup(fd),'wb') as out:
        for n in full['directories']:out.write(header(n));progress.member()
        cursor=None;label=None
        try:
            for n,v in full['files'].items():
                bits=n.split('/')
                if bits[0]!=label:
                    if cursor is not None:cursor.close();cursor=None
                    label=bits[0];cursor=SourceCursor(roots[label],expected=identities[label])
                out.write(header(n,v['bytes']));h=hashlib.sha256();count=0
                cursor.parent(bits[1:-1],identities,label)
                handle=cursor.child(bits[-1],directory=False,expected=identities[n]) if len(bits)>1 else contextlib.nullcontext(cursor.fd)
                with handle as source_fd:
                    need(identity(os.fstat(source_fd))==identities[n],'source changed before archive')
                    while count<v['bytes']:
                        b=os.read(source_fd,min(CHUNK,v['bytes']-count));need(b,'truncated source');count+=len(b);h.update(b);out.write(b)
                    need(not os.read(source_fd,1),'source grew')
                    need(identity(os.fstat(source_fd))==identities[n],'source changed during archive')
                need(h.hexdigest()==v['sha256'],'source digest changed');out.write(b'\0'*((-count)%512));progress.member(count)
        finally:
            if cursor is not None:cursor.close(sys.exc_info()[1])
        out.write(b'\0'*(tar_size(full)-out.tell()));out.flush();os.fsync(out.fileno())


def check_tar(path,full,emit=None,wanted=()):
    """Canonical USTAR only; reject duplicate/extra/truncated headers and nonzero padding."""
    with Progress('archive_verify') as progress,opened(path) as fd,os.fdopen(os.dup(fd),'rb') as f:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size==tar_size(full),'independent exact archive size required')
        digest=hashlib.sha256();count=0
        def read(n):
            nonlocal count
            b=f.read(n);need(len(b)==n,'truncated archive');digest.update(b);count+=len(b);return b
        for n in full['directories']:need(read(512)==header(n),'directory header/order differs');progress.member()
        for n,v in full['files'].items():
            need(read(512)==header(n,v['bytes']),'file header/order differs');h=hashlib.sha256();left=v['bytes']
            while left:
                b=read(min(CHUNK,left));left-=len(b);h.update(b)
                if emit is not None and n in wanted:emit(b)
            need(h.hexdigest()==v['sha256'],'archive member digest differs: '+n)
            padding=(-v['bytes'])%512;need(read(padding)==b'\0'*padding,'archive payload padding differs');progress.member(v['bytes'])
        left=before.st_size-count;need(left>=1024 and read(left)==b'\0'*left and not f.read(1),'archive trailing members/padding differ')
        need(identity(before)==identity(os.fstat(fd))==identity(Path(path).lstat()),'archive changed during verification')
    return {'bytes':count,'sha256':digest.hexdigest()}


def copy_archive(src,dst,expected):
    with Progress('archive_copy') as progress,opened(src) as a,opened(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL) as b:
        before=os.fstat(a);need(before.st_nlink==1 and stat.S_ISREG(before.st_mode),'ordinary scratch archive required')
        need((before.st_dev,before.st_ino)!=(os.fstat(b).st_dev,os.fstat(b).st_ino),'archive alias forbidden')
        h=hashlib.sha256();size=0
        with os.fdopen(os.dup(a),'rb') as source,os.fdopen(os.dup(b),'wb') as target:
            for block in iter(lambda:source.read(CHUNK),b''):
                h.update(block);size+=len(block);target.write(block);progress.bytes=size
                if size%(64*CHUNK)==0:progress.emit('progress')
            target.flush();os.fsync(target.fileno())
        need(identity(before)==identity(os.fstat(a))==identity(Path(src).lstat()),'scratch archive changed during copy')
        need({'bytes':size,'sha256':h.hexdigest()}==expected,'project copy differs')
        progress.members=progress.files=1


def backup_location(root):
    p=absolute(root);need(p.parent==PROJECT_BASE and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',p.name),'exclusive projects backup child required')
    return p


def check_trusted_parent():
    need(PROJECT_PARENT==PROJECT_PARENT.resolve(strict=True),'trusted project parent alias')
    with opened(PROJECT_PARENT,directory=True) as fd:
        s=os.fstat(fd);need(s.st_uid==os.geteuid(),'trusted project parent owner differs')
    # The explicit parent may be2770. New children are always0700; no sameUID/admin guarantee.
    return {'path':str(PROJECT_PARENT),'uid':s.st_uid,'mode':oct(stat.S_IMODE(s.st_mode))}


def capacity_observation(items):
    """statvfs admission, not a reservation/quota or institutional-backup attestation.

    Sum obligations sharing st_dev; never count the same free pool twice.
    Items are(role,existing_parent,payload_bytes,metadata_budget_bytes).
    """
    rows=[]
    for role,parent,payload,metadata in items:
        p=absolute(parent)
        with opened(p,directory=True) as fd:device=os.fstat(fd).st_dev
        fs=os.statvfs(p)
        need(type(payload) is int and payload>=0 and type(metadata) is int and metadata>=0
             and fs.f_frsize>0 and fs.f_bavail>=0,'valid capacity observation required')
        rows.append({'role':role,'path':str(p),'device':device,'available_bytes':fs.f_bavail*fs.f_frsize,
                     'payload_bytes':payload,'metadata_budget_bytes':metadata,'headroom_bytes':CAPACITY_HEADROOM,
                     'required_bytes':payload+metadata+CAPACITY_HEADROOM,'observed_unix_seconds':time.time()})
    groups={}
    for row in rows:
        key=str(row['device']);g=groups.setdefault(key,{'required_bytes':0,'available_bytes':row['available_bytes']})
        g['required_bytes']+=row['required_bytes'];g['available_bytes']=min(g['available_bytes'],row['available_bytes'])
    return {'observations':rows,'filesystem_totals':groups,'admitted':all(g['available_bytes']>=g['required_bytes'] for g in groups.values()),
            'space_reserved':False,'institutional_backup_attested':False,'method':'statvfs.f_bavail*f_frsize'}


def require_capacity(value):
    need(value['admitted'],'insufficient storage capacity: '+raw(value).decode().strip())


def remote_collect(req):
    full=normalized(req['observation']);selected_inventory(full);roots=req['roots'];roots_check(roots,full);backup=backup_location(req['backup_root'])
    trusted=check_trusted_parent();owned_dir(PROJECT_BASE);owned_dir(TRANSFER_BASE)
    # Small exclusive intent/observation records precede payload; neither archive exists yet.
    archive_bytes=tar_size(full);metadata_budget=8*(len(raw(req))+len(raw(full)))+2*1024**2
    capacity=capacity_observation([('scratch_archive',TRANSFER_BASE,archive_bytes,metadata_budget),
                                   ('project_archive',PROJECT_BASE,archive_bytes,metadata_budget)])
    transfer=TRANSFER_BASE/backup.name;claimed=[]
    try:
        owned_dir(transfer,True);claimed.append(transfer);write_new(transfer/'intent.json',raw({'request':req,'capacity':capacity}));write_new(transfer/'capacity.json',raw(capacity))
        owned_dir(backup,True);claimed.append(backup);write_new(backup/'intent.json',raw({'request':req,'capacity':capacity}));write_new(backup/'capacity.json',raw(capacity))
        require_capacity(capacity)
        with Progress('before_scan'):before,ids,groups=scan(roots)
        need(before==full,'original observation no longer matches')
        if 'original_hardlink_groups' in req['observation']:need(groups==req['observation']['original_hardlink_groups'],'original hardlink metadata differs')
        archive=transfer/'raw.tar';make_tar(archive,full,roots,ids);a=check_tar(archive,full)
        with Progress('after_archive_scan'):after,after_ids,after_groups=scan(roots)
        need(after==before and ids==after_ids and groups==after_groups,'original tree changed across archive')
        project_recheck=capacity_observation([('project_before_copy',PROJECT_BASE,archive_bytes,metadata_budget)])
        write_new(backup/'capacity-before-copy.json',raw(project_recheck));require_capacity(project_recheck)
        copy_archive(archive,backup/'raw.tar',a);b=check_tar(backup/'raw.tar',full);need(a==b,'project readback archive differs')
        with Progress('after_project_scan'):final,final_ids,final_groups=scan(roots)
        need(final==full and final_ids==ids and final_groups==groups,'original tree changed across project copy')
        local,omitted=selected_inventory(full)
        receipt={'schema':'ace-remote-custody-v1','selection_policy':'current-launch-text-and-namespaces-v1','full_backup_verified':True,'full_inventory':full,'local_inventory':local,'local_root':req['local_root'],'omitted_files':omitted,'omitted_bytes':sum(v['bytes'] for v in omitted.values()),'full_local_copy':False,'backup_kind':'ustar','backup_root':str(backup),'backup_archive_path':str(backup/'raw.tar'),'backup_archive_sha256':b['sha256'],'backup_archive_bytes':b['bytes'],'scratch_transfer_root':str(transfer),'scratch_archive_path':str(archive),'roots':roots,'source_observation_sha256':sha(raw(req['observation'])),'helper_sha256':req['helper_sha256'],'original_hardlink_groups':groups,'trusted_project_parent':trusted,'capacity':{'remote_initial':capacity,'project_before_copy':project_recheck,'local':req.get('local_capacity')},'limits':{'total_source_bytes':MAX_TOTAL,'member_source_bytes':MAX_MEMBER,'local_member_bytes':LOCAL_MEMBER,'local_total_bytes':LOCAL_TOTAL},'original_before_inventory_sha256':sha(raw(before)),'original_after_inventory_sha256':sha(raw(final)),'project_readback_inventory_sha256':sha(raw(full)),'acceptance':False,'retry_allowed':False}
        write_new(backup/'inventory.json',raw(full));write_new(backup/'receipt.json',raw(receipt));write_new(transfer/'receipt.json',raw(receipt))
        return receipt
    except BaseException as exc:
        for p in claimed:safe_failure(p,exc,partial_preserved=True)
        raise


def validate_receipt(receipt):
    need(type(receipt) is dict and receipt.get('schema')=='ace-remote-custody-v1' and receipt.get('full_backup_verified') is True and receipt.get('full_local_copy') is False and receipt.get('acceptance') is False and receipt.get('retry_allowed') is False,'pinned transport-only receipt required')
    need(_BOUND_SOURCE is not None and receipt['helper_sha256']==_BOUND_SOURCE[1],'independent bound helper differs')
    need(receipt.get('selection_policy')=='current-launch-text-and-namespaces-v1','explicit local selection policy differs')
    full=normalized(receipt['full_inventory']);local,omitted=selected_inventory(full)
    need(receipt['local_inventory']==local and receipt['omitted_files']==omitted and receipt['omitted_bytes']==sum(v['bytes'] for v in omitted.values()),'exact selected/omitted partition differs')
    backup=backup_location(receipt['backup_root']);need(receipt['backup_kind']=='ustar' and receipt['backup_archive_path']==str(backup/'raw.tar') and pin(receipt['backup_archive_sha256']) and receipt['backup_archive_bytes']==tar_size(full),'backup archive identity differs')
    return full


def check_backup_record(r):
    for p in (PROJECT_BASE,Path(r['backup_root'])):
        with opened(p,directory=True) as fd:
            st=os.fstat(fd);need(st.st_uid==os.geteuid() and stat.S_IMODE(st.st_mode)==0o700,'owned0700 backup directory required')
    meta,_=fingerprint(Path(r['backup_root'])/'receipt.json',True)
    need(meta=={'bytes':len(raw(r)),'sha256':sha(raw(r))},'pinned durable receipt differs')
    meta,_=fingerprint(Path(r['backup_root'])/'inventory.json',True)
    need(meta=={'bytes':len(raw(r['full_inventory'])),'sha256':sha(raw(r['full_inventory']))},'durable full inventory differs')


def remote_verify(req):
    r=req['receipt'];full=validate_receipt(r);check_trusted_parent();check_backup_record(r)
    need(normalized(req['observation'])==full,'caller second observation differs')
    meta=check_tar(r['backup_archive_path'],full);need(meta=={'bytes':r['backup_archive_bytes'],'sha256':r['backup_archive_sha256']},'backup hash changed')
    return {'full_backup_verified':True,'full_inventory':full,'local_inventory':r['local_inventory'],'receipt_sha256':sha(raw(r)),'backup_archive_sha256':meta['sha256'],'helper_sha256':req['helper_sha256'],'acceptance':False}


def remote_fetch(req):
    r=req['receipt'];full=validate_receipt(r);check_trusted_parent();check_backup_record(r);n=name_ok(req['member'])
    need(n in full['files'] and req['expected_sha256']==full['files'][n]['sha256'],'explicit original inventory member pin required')
    meta=check_tar(r['backup_archive_path'],full);need(meta=={'bytes':r['backup_archive_bytes'],'sha256':r['backup_archive_sha256']},'backup hash changed')
    return {'verified':True,'member':n,'sha256':req['expected_sha256'],'bytes':full['files'][n]['bytes'],'destination':req['destination'],'backup_archive_sha256':meta['sha256'],'helper_sha256':req['helper_sha256'],'acceptance':False}


def remote_main(request,output):
    def expired(signum,frame):raise TimeoutError('bounded remote custody deadline')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(TIMEOUT-30)
    try:return remote_emit(request,output)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def remote_emit(request,output):
    need(request['helper_sha256']==_BOUND_SOURCE[1],'request/helper pin differs')
    action=request['action']
    if action=='collect':
        result=remote_collect(request);full=result['full_inventory'];wanted=result['local_inventory']['files'];archive=result['backup_archive_path']
    elif action=='verify':result=remote_verify(request);wanted={}
    elif action=='fetch':
        result=remote_fetch(request);full=request['receipt']['full_inventory'];wanted={request['member']:full['files'][request['member']]};archive=request['receipt']['backup_archive_path']
    else:raise ValueError('unknown operation')
    metadata=raw({'result':result,'stream_files':wanted});need(len(metadata)<=MAX_METADATA,'bounded result metadata')
    output.write(len(metadata).to_bytes(8,'big'));output.write(metadata)
    if wanted:
        actual=check_tar(archive,full,emit=output.write,wanted=wanted)
        need(actual['sha256']==result['backup_archive_sha256'],'archive changed during selected streaming')
    output.flush()


class Receiver:
    """Bounded framed metadata followed by exact opaque selected file bytes."""
    def __init__(self,allowed,destinations,validate):
        self.allowed=allowed;self.destinations=destinations;self.validate=validate
        self.prefix=bytearray();self.metadata=bytearray();self.length=None;self.result=None
        self.names=iter(sorted(allowed));self.current=None;self.fd=None;self.handle=None;self.cleanup_errors=[];self.left=0;self.digest=None;self.done=False

    def next_file(self):
        while True:
            self.current=next(self.names,None)
            if self.current is None:self.done=True;return
            p=self.destinations[self.current]
            self.handle=opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL);fd=self.handle.__enter__()
            self.fd=os.fdopen(os.dup(fd),'wb');self.left=self.allowed[self.current]['bytes'];self.digest=hashlib.sha256()
            if self.left:return
            self.fd.flush();os.fsync(self.fd.fileno());self.close()
            need(self.digest.hexdigest()==self.allowed[self.current]['sha256'],'empty file digest differs')

    def end_file(self):
        self.fd.flush();os.fsync(self.fd.fileno());self.close()
        need(self.digest.hexdigest()==self.allowed[self.current]['sha256'],'selected/fetched stream digest differs')
        self.next_file()

    def feed(self,data):
        view=memoryview(data)
        while view:
            if self.length is None:
                n=min(8-len(self.prefix),len(view));self.prefix.extend(view[:n]);view=view[n:]
                if len(self.prefix)==8:
                    self.length=int.from_bytes(self.prefix,'big');need(0<self.length<=MAX_METADATA,'bounded stream metadata required')
            elif self.result is None:
                n=min(self.length-len(self.metadata),len(view));self.metadata.extend(view[:n]);view=view[n:]
                if len(self.metadata)==self.length:
                    value=decode(bytes(self.metadata));need(set(value)=={'result','stream_files'} and value['stream_files']==self.allowed,'exact selected stream manifest required')
                    self.validate(value['result']);self.result=value['result'];self.next_file()
            else:
                need(not self.done,'extra stream bytes');n=min(self.left,len(view));b=view[:n];self.fd.write(b);self.digest.update(b);self.left-=n;view=view[n:]
                if not self.left:self.end_file()

    def finish(self):need(self.result is not None and self.done,'incomplete selected stream');return self.result

    def close(self):
        fileobj,self.fd=self.fd,None;handle,self.handle=self.handle,None
        first=None
        for phase,close in (('buffered_file',None if fileobj is None else fileobj.close),
                            ('underlying_handle',None if handle is None else lambda:handle.__exit__(None,None,None))):
            if close is None:continue
            try:close()
            except BaseException as exc:
                if first is None:first=exc
                self.cleanup_errors.append({'phase':phase,'type':type(exc).__name__,'message':str(exc)})
        if first is not None:raise first


def close_receiver(receiver,logs,primary):
    if receiver is None:return
    secondary=None
    try:receiver.close()
    except BaseException as exc:secondary=exc
    errors=receiver.cleanup_errors
    if not errors and secondary is None:return
    record={'primary':None if primary is None else {'type':type(primary).__name__,'message':str(primary)},
            'cleanup_errors':errors,'acceptance':False,'retry_allowed':False}
    if secondary is not None:record['secondary']={'type':type(secondary).__name__,'message':str(secondary)}
    try:write_new(Path(logs)/'receiver-cleanup-error.json',raw(record))
    except BaseException:pass  # Cleanup evidence failure must not replace the primary exception.
    if primary is None:
        error=secondary if secondary is not None else RuntimeError('receiver cleanup failed')
        safe_failure(logs,error,receiver_cleanup_failed=True)
        raise error


BOOTSTRAP=r"""import sys,hashlib
def require(ok,message):
 if not ok:raise ValueError(message)
line=sys.stdin.buffer.readline(32)
require(line.endswith(b'\n') and line[:-1].isdigit(),'bounded source length framing required')
n=int(line);require(0<n<=1048576,'source length bound')
s=sys.stdin.buffer.read(n)
require(len(s)==n,'truncated helper source')
require(hashlib.sha256(s).hexdigest()==sys.argv[1],'independent helper source SHA differs')
v={'__name__':'captured_remote_custody','__file__':'<independently-pinned-remote-custody>'}
exec(compile(s,v['__file__'],'exec'),v)
v['bind_source'](s,sys.argv[1])
b=sys.stdin.buffer.read(v['MAX_METADATA']+1)
require(len(b)<=v['MAX_METADATA'],'request metadata bound')
v['remote_main'](v['decode'](b),sys.stdout.buffer)
"""


def transport(request,receiver,helper_path,logs):
    """ONE SSH transport via existing authenticated helper; never retry on ambiguity."""
    source,source_pin=_BOUND_SOURCE;request_raw=raw(request);need(len(request_raw)<=MAX_METADATA,'bounded request metadata')
    helper=absolute(helper_path);need(helper.is_file() and not helper.is_symlink(),'explicit existing CURC transport helper required')
    command=['bash',str(helper),'run',shlex.join(['python3','-I','-B','-c',BOOTSTRAP,source_pin])]
    write_new(logs/'command.json',raw(command));write_new(logs/'request.bin',str(len(source)).encode()+b'\n'+source+request_raw)
    proc=None;start=time.monotonic();error=None;count=0;stderr_count=0;limit=8+MAX_METADATA+sum(v['bytes'] for v in receiver.allowed.values())
    try:
        with (logs/'request.bin').open('rb') as stdin,(logs/'stderr.bin').open('xb') as err:
            proc=subprocess.Popen(command,stdin=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout,selectors.EVENT_READ,'out');selector.register(proc.stderr,selectors.EVENT_READ,'err')
                while selector.get_map():
                    need(time.monotonic()-start<TIMEOUT,'bounded transport timeout; remote outcome ambiguous')
                    for key,event in selector.select(min(1,TIMEOUT-(time.monotonic()-start))):
                        block=os.read(key.fileobj.fileno(),CHUNK)
                        if not block:selector.unregister(key.fileobj);continue
                        if key.data=='err':
                            err.write(block);err.flush();stderr_count+=len(block);need(stderr_count<=64*1024**2,'stderr transport bound; partial retained')
                        else:count+=len(block);need(count<=limit,'stdout bound; partial retained');receiver.feed(block)
            rc=proc.wait(timeout=max(.01,TIMEOUT-(time.monotonic()-start)))
            need(rc==0,'remote transport nonzero; outcome ambiguous');receiver.finish()
    except BaseException as exc:
        error=exc
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except BaseException as secondary:
                try:write_new(logs/'owned-transport-cleanup-error.json',raw({'type':type(secondary).__name__,'message':str(secondary)}))
                except BaseException:pass
            try:proc.wait(timeout=5)
            except BaseException:pass
        raise
    finally:
        result={'command':command,'returncode':None if proc is None else proc.poll(),'wall_seconds':time.monotonic()-start,'stdout_bytes':count,'stderr_bytes':stderr_count,'remote_outcome_ambiguous':error is not None,'retry_allowed':False}
        try:write_new(logs/'transport.json',raw(result))
        except BaseException:
            if error is None:raise


def begin(log_dir,request,payloads=()):
    need(_BOUND_SOURCE is not None,'bind independent helper source before any operation')
    p=absolute(log_dir);request_raw=raw(request);need(len(request_raw)<=MAX_METADATA,'bounded operation metadata')
    metadata_budget=2*len(request_raw)+2*len(_BOUND_SOURCE[0])+MAX_METADATA+64*1024**2+2*1024**2
    # Account for request.json/request.bin, source, bounded response, stderr and receipts.
    items=[('local_operation_metadata',p.parent,0,metadata_budget)]
    items += [(role,Path(parent),size,0) for role,parent,size in payloads]
    local_capacity=capacity_observation(items);require_capacity(local_capacity)
    request['local_capacity']=local_capacity
    owned_dir(p,True)
    try:
        write_new(p/'capacity.json',raw(local_capacity))
        write_new(p/'helper.py',_BOUND_SOURCE[0]);write_new(p/'intent.json',raw({'request_sha256':sha(raw(request)),'helper_sha256':_BOUND_SOURCE[1],'operation':request['action'],'capacity':local_capacity,'retry_allowed':False}));write_new(p/'request.json',raw(request))
    except BaseException as exc:safe_failure(p,exc,remote_outcome_ambiguous=False);raise
    return p


def local_inventory(root):
    r=Path(root);full,ids,groups=scan({'local':str(r)},True)
    return {'files':{k[6:]:v for k,v in full['files'].items()},'directories':[k[6:] for k in full['directories'] if k!='local']}


def local_destinations(root,selection):
    r=absolute(root)
    need(not os.path.lexists(r) or (r.is_dir() and not r.is_symlink() and not any(r.iterdir())),'local selected root must be absent or empty')
    owned_dir(r,not r.exists())
    for n in selection['directories']:owned_dir(r/n,True)
    return {n:r/n for n in selection['files']}


def collect_backup(*,observation,roots,backup_root,local_root,helper_path,log_dir):
    need(_BOUND_SOURCE is not None,'bind independent helper source first')
    full=normalized(observation);selection,omitted=selected_inventory(full);backup_location(backup_root)
    local=absolute(local_root);logs_path=absolute(log_dir)
    need(local!=logs_path and local not in logs_path.parents and logs_path not in local.parents,'local subset and logs must be disjoint')
    request={'action':'collect','observation':observation,'roots':roots,'backup_root':str(backup_root),'local_root':str(local),'helper_sha256':_BOUND_SOURCE[1]}
    logs=begin(log_dir,request,[('local_selected_payload',local if local.exists() else local.parent,sum(v['bytes'] for v in selection['files'].values()))]);receiver=None;primary=None
    try:
        destinations=local_destinations(local,selection)
        def validate(result):
            need(validate_receipt(result)==full and result['roots']==roots and result['backup_root']==str(backup_root) and result['local_root']==str(local) and result['source_observation_sha256']==sha(raw(observation)),'collect receipt/request differs')
        receiver=Receiver(selection['files'],destinations,validate);transport(request,receiver,helper_path,logs);result=receiver.finish()
        need(local_inventory(local)==selection,'local selected readback differs')
        write_new(logs/'receipt.json',raw(result));return result
    except BaseException as exc:primary=exc;safe_failure(logs,exc,remote_outcome_ambiguous=True,partial_preserved=True);raise
    finally:close_receiver(receiver,logs,primary)


def verify_backup(*,receipt,observation,helper_path,log_dir):
    full=validate_receipt(receipt);need(normalized(observation)==full,'caller second observation differs')
    need(local_inventory(receipt['local_root'])==receipt['local_inventory'],'selected local custody changed')
    request={'action':'verify','receipt':receipt,'observation':observation,'helper_sha256':_BOUND_SOURCE[1]};logs=begin(log_dir,request)
    expected={'full_backup_verified':True,'full_inventory':full,'local_inventory':receipt['local_inventory'],'receipt_sha256':sha(raw(receipt)),'backup_archive_sha256':receipt['backup_archive_sha256'],'helper_sha256':_BOUND_SOURCE[1],'acceptance':False}
    def validate(result):need(result==expected,'remote verification identity differs')
    receiver=Receiver({}, {}, validate);primary=None
    try:
        transport(request,receiver,helper_path,logs);result=receiver.finish();need(local_inventory(receipt['local_root'])==receipt['local_inventory'],'selected local custody changed during verification')
        write_new(logs/'receipt.json',raw(result));return result
    except BaseException as exc:primary=exc;safe_failure(logs,exc,remote_outcome_ambiguous=True);raise
    finally:close_receiver(receiver,logs,primary)


def fetch_exact(*,receipt,member,expected_sha256,destination,helper_path,log_dir):
    full=validate_receipt(receipt);name_ok(member)
    need(member in full['files'] and pin(expected_sha256) and expected_sha256==full['files'][member]['sha256'],'exact independently pinned inventory member required')
    dest=absolute(destination);logs_path=absolute(log_dir)
    need(not os.path.lexists(dest) and dest.parent!=logs_path and dest.parent not in logs_path.parents and logs_path not in dest.parents,'new destination outside operation logs required')
    with opened(dest.parent,directory=True) as fd:
        st=os.fstat(fd);need(st.st_uid==os.geteuid() and stat.S_IMODE(st.st_mode)==0o700,'caller owned exclusive0700 staging directory required')
    request={'action':'fetch','receipt':receipt,'member':member,'expected_sha256':expected_sha256,'destination':str(dest),'helper_sha256':_BOUND_SOURCE[1]};logs=begin(log_dir,request,[('local_fetch_payload',dest.parent,full['files'][member]['bytes'])])
    expected={'verified':True,'member':member,'sha256':expected_sha256,'bytes':full['files'][member]['bytes'],'destination':str(dest),'backup_archive_sha256':receipt['backup_archive_sha256'],'helper_sha256':_BOUND_SOURCE[1],'acceptance':False}
    def validate(result):need(result==expected,'remote fetch identity differs')
    receiver=Receiver({member:full['files'][member]},{member:dest},validate);primary=None
    try:
        transport(request,receiver,helper_path,logs);result=receiver.finish();need(fingerprint(dest,True)[0]==full['files'][member],'fetched bytes differ on readback')
        write_new(logs/'receipt.json',raw(result));return result
    except BaseException as exc:primary=exc;safe_failure(logs,exc,remote_outcome_ambiguous=True,partial_preserved=True);raise
    finally:close_receiver(receiver,logs,primary)
