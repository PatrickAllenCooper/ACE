"""Opaque downstream input adapter for an explicitly pinned compact custody receipt.

No transport implementation, scientific imports, world/value decoding or acceptance.
Caller authenticates source/Git/review/amendment and original predecessor acceptance
before constructing this object. The independently pinned backend performs complete
remote archive verification and exact member fetches. Local evidence stays partial.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import stat


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate receipt key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite receipt value: '+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def relative(value):
    need(type(value) is str and value and not value.startswith('/') and
         '\\' not in value and '\x00' not in value and
         all(v not in ('', '.', '..') for v in value.split('/')),
         'normalized relative member required')
    return value


def record(value):
    need(type(value) is dict and set(value)=={'bytes','sha256'} and
         type(value['bytes']) is int and value['bytes']>=0 and pin(value['sha256']),
         'exact file size/hash record required')


def inventory_shape(view):
    need(type(view) is dict and set(view)=={'files','directories'} and
         type(view['files']) is dict and type(view['directories']) is list,
         'full file/directory inventory required')
    for name, info in view['files'].items():
        relative(name); record(info)
    dirs=view['directories']
    need(dirs==sorted(set(dirs)), 'sorted unique directories required')
    for name in dirs:
        relative(name)
    need(not set(dirs)&set(view['files']), 'file/directory collision')
    for name in list(view['files'])+dirs:
        parents=Path(name).parents
        need(all(str(p)=='.' or str(p) in dirs for p in parents), 'missing parent directory')


def read_exact(path, info, maximum):
    record(info)
    need(type(maximum) is int and maximum>=0 and info['bytes']<=maximum,
         'independent input byte limit exceeded')
    path=Path(path)
    need(not path.is_symlink(), 'input symlink forbidden')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and
             before.st_size==info['bytes'], 'independent exact regular input required')
        data=stream.read(maximum+1)
        after=os.fstat(stream.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(before)==identity(after)==identity(path.lstat()) and
         len(data)==info['bytes'] and sha(data)==info['sha256'], 'input bytes changed/differ')
    return data


def selected_inventory(root):
    root=Path(root)
    need(root.is_dir() and not root.is_symlink(), 'ordinary selected root required')
    files={};dirs=[]
    for parent, names, members in os.walk(root, followlinks=False):
        for name in names:
            p=Path(parent)/name
            need(not p.is_symlink(), 'selected directory symlink forbidden')
            dirs.append(p.relative_to(root).as_posix())
        for name in members:
            p=Path(parent)/name;info=p.lstat()
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1, 'selected regular sole-link file required')
            # Hash bounded chunks; no private JSON, tokens or tensors decoded.
            digest=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
            after=p.lstat()
            need((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                 'selected file changed while hashing')
            files[p.relative_to(root).as_posix()]={'bytes':info.st_size,'sha256':digest.hexdigest()}
    return {'files':files,'directories':sorted(dirs)}


class Inputs:
    """One admitted predecessor use. Fetch/log destinations are always exclusive."""
    def __init__(self, *, backend, receipt_raw, receipt_sha256, observation,
                 expected_roots, helper_sha256, local_roots, staging, helper_path):
        need(pin(receipt_sha256) and sha(receipt_raw)==receipt_sha256,
             'independently supplied receipt byte pin required')
        self.receipt=decode(receipt_raw)
        r=self.receipt
        need(canonical(r)==receipt_raw, 'exact canonical receipt bytes required')
        need(r['schema']=='ace-remote-custody-v1' and r['full_backup_verified'] is True and
             r['full_local_copy'] is False and r['backup_kind']=='ustar' and
             r['selection_policy']=='current-launch-text-and-namespaces-v1' and
             r['acceptance'] is False and r['retry_allowed'] is False,
             'explicit remote full / local selected custody required')
        need(pin(helper_sha256) and r['helper_sha256']==helper_sha256 and
             r['roots']==expected_roots, 'independent backend/root identity differs')
        self.full=r['full_inventory'];self.selected=r['local_inventory']
        inventory_shape(self.full);inventory_shape(self.selected)
        need(self.full=={'files':observation['files'],'directories':sorted(observation['directories'])},
             'complete original predecessor inventory differs')
        need(all(self.full['files'].get(n)==v for n,v in self.selected['files'].items()) and
             set(self.selected['directories'])<=set(self.full['directories']), 'selected view is not exact subset')
        omitted={n:v for n,v in self.full['files'].items() if n not in self.selected['files']}
        need(r['omitted_files']==omitted and r['omitted_bytes']==sum(v['bytes'] for v in omitted.values()),
             'selected/omitted partition differs')
        self.local_roots=tuple(Path(v).resolve(strict=True) for v in local_roots)
        need(self.local_roots and len(set(self.local_roots))==len(self.local_roots) and
             Path(r['local_root']) in self.local_roots, 'original receipt local root must be one selected mirror')
        need(all(a!=b and a not in b.parents and b not in a.parents
                 for i,a in enumerate(self.local_roots) for b in self.local_roots[i+1:]),
             'separate selected mirrors required')
        self.backend=backend;self.receipt_sha256=receipt_sha256;self.observation=observation
        self.helper_path=helper_path;self.helper_sha256=helper_sha256
        self.staging=Path(staging)
        need(self.staging.parent.resolve(strict=True)==self.staging.parent and
             not any(self.staging==p or p in self.staging.parents or self.staging in p.parents
                     for p in self.local_roots), 'canonical disjoint staging required')
        self.staging.mkdir(mode=0o700,exist_ok=False)
        self.used={};self.attempted=set();self.ordinal=0
        self.recheck()

    def recheck(self):
        # Backend verifies immutable receipt.local_root; second copy checked locally
        # against the exact same selected map. The receipt is never rewritten.
        need(sha(canonical(self.receipt))==self.receipt_sha256, 'captured receipt mutated')
        verification=self.backend.verify_backup(receipt=self.receipt,observation=self.observation,
            helper_path=self.helper_path,log_dir=self.staging/f'verify-{self.ordinal:03d}')
        self.ordinal+=1
        need(verification['full_backup_verified'] is True and verification['acceptance'] is False and
             verification['full_inventory']==self.full and verification['local_inventory']==self.selected and
             verification['receipt_sha256']==self.receipt_sha256 and
             verification['backup_archive_sha256']==self.receipt['backup_archive_sha256'] and
             verification['helper_sha256']==self.helper_sha256, 'fresh complete backup verification differs')
        for root in self.local_roots:
            need(selected_inventory(root)==self.selected, 'selected mirror changed/omitted/extra')
        for name in self.selected['files']:
            for i,a in enumerate(self.local_roots):
                for b in self.local_roots[i+1:]:
                    need(not os.path.samestat((a/name).stat(),(b/name).stat()), 'selected mirrors alias same inode')
        return verification

    def read(self, name, expected_sha256, maximum=256*1024**2):
        relative(name)
        need(name.startswith('launch/inputs/') and name.count('/')==2 and pin(expected_sha256),
             'only explicit flat inherited input members may be fetched')
        info=self.full['files'].get(name)
        need(info is not None and info['sha256']==expected_sha256, 'independent inherited input pin differs')
        need(type(maximum) is int and 0<=info['bytes']<=maximum, 'input exceeds unchanged byte bound')
        need(name not in self.attempted, 'input already attempted; no implicit retry')
        need(sha(canonical(self.receipt))==self.receipt_sha256, 'captured receipt mutated')
        self.attempted.add(name)
        if name in self.selected['files']:
            data=read_exact(self.local_roots[0]/name,info,maximum)
            for root in self.local_roots[1:]:
                need(read_exact(root/name,info,maximum)==data,'selected input copies differ')
        else:
            directory=self.staging/f'fetch-{len(self.used):03d}'
            directory.mkdir(mode=0o700,exist_ok=False)
            destination=directory/Path(name).name
            result=self.backend.fetch_exact(receipt=self.receipt,member=name,
                expected_sha256=expected_sha256,destination=destination,
                helper_path=self.helper_path,log_dir=self.staging/f'fetch-log-{len(self.used):03d}')
            need(result['verified'] is True and result['acceptance'] is False and result['member']==name and
                 result['sha256']==expected_sha256 and result['bytes']==info['bytes'] and
                 Path(result['destination'])==destination and result['helper_sha256']==self.helper_sha256 and
                 result['backup_archive_sha256']==self.receipt['backup_archive_sha256'],
                 'exact member transport evidence differs')
            data=read_exact(destination,info,maximum)
        self.used[name]=info
        return data
