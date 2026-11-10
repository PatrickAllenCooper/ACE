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



class OperationEvidence:
    """Durable caller-owned records; reference is published before any transport."""
    def __init__(self, root, identity, publish):
        self.root=Path(root);self.publish=publish;self.number=0;self.last_reference=None
        need(self.root.parent==self.root.parent.resolve(strict=True),'canonical durable evidence parent required')
        self.root.mkdir(mode=0o700,exist_ok=False)
        self.intent={'schema':'ace-compact-input-use-intent-v1','path':str(self.root),
            'identity':identity,'acceptance':False,'retry_allowed':False}
        self.intent_raw=canonical(self.intent)
        self._write('intent.json',self.intent_raw)
        self.intent_reference={'path':str(self.root),'intent_sha256':sha(self.intent_raw),
                               'acceptance':False,'retry_allowed':False}
        # If publication fails, no backend method has run. Original intent stays.
        self.publish('storage-use.intent.json',self.intent_reference)

    def _write(self,name,data):
        with (self.root/name).open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())

    def record(self, operation, error=None, details=None):
        ordinal=self.number;self.number+=1
        path=self.root/'operations'
        view=selected_inventory(path) if path.exists() else {'files':{},'directories':[]}
        manifest={'schema':'ace-compact-input-operation-inventory-v1',
            'intent_sha256':sha(self.intent_raw),'operation':operation,'ordinal':ordinal,
            'status':'completed' if error is None else 'failed',
            'error':None if error is None else {'type':type(error).__name__,'message':str(error)},
            'details':details,'operations_path':str(path),'inventory':view,'acceptance':False,'retry_allowed':False}
        data=canonical(manifest);name=f'inventory-{ordinal:03d}.json';self._write(name,data)
        reference={**self.intent_reference,'inventory_path':str(self.root/name),
                   'inventory_sha256':sha(data),'status':manifest['status'],'operation':operation}
        self._write(f'reference-{ordinal:03d}.json',canonical(reference))
        self.last_reference=reference
        # Always retain the original manifest/reference if caller publication fails.
        self.publish(f'storage-use-{ordinal:03d}.json',reference)
        return reference

    def run(self, operation, action, details=None):
        try:
            result=action()
        except BaseException as primary:
            try:self.record(operation,primary,details)
            except BaseException as secondary:
                try:self._write(f'evidence-failure-{self.number:03d}.json',canonical({
                    'primary':{'type':type(primary).__name__,'message':str(primary)},
                    'secondary':{'type':type(secondary).__name__,'message':str(secondary)},
                    'acceptance':False,'retry_allowed':False}))
                except BaseException:pass
            raise
        self.record(operation,details=details)
        return result


def recorded(operation):
    # Direct standalone Inputs tests may omit evidence; production bind cannot.
    def decorate(function):
        def run(self,*args,**kwargs):
            if self.evidence is None:return function(self,*args,**kwargs)
            details={'member':args[0],'expected_sha256':args[1]} if operation=='input-read' and len(args)>1 else None
            return self.evidence.run(operation,lambda:function(self,*args,**kwargs),details)
        return run
    return decorate


def observed_directories(values):
    need(type(values) is list and all(type(n) is str for n in values)
         and len(set(values))==len(values),'original unique directory list required')
    for n in values:relative(n)
    return sorted(values)


class Inputs:
    """One admitted predecessor use. Fetch/log destinations are always exclusive."""
    def __init__(self, *, backend, receipt_raw, receipt_sha256, observation,
                 expected_roots, helper_sha256, local_roots, staging, helper_path, evidence=None):
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
        need(expected_roots.get('control')=='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
             and expected_roots.get('launch')=='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01',
             'explicit GPU02 control / original output roots required')
        self.full=r['full_inventory'];self.selected=r['local_inventory']
        inventory_shape(self.full);inventory_shape(self.selected)
        need(self.full=={'files':observation['files'],'directories':observed_directories(observation['directories'])},
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
        self.evidence=evidence
        self.used={};self.captured={};self.attempted=set();self.ordinal=0
        self.recheck()

    @recorded("archive-verify")
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

    @recorded("input-read")
    def read(self, name, expected_sha256, maximum=256*1024**2):
        relative(name)
        need(name.startswith('control/inputs/') and name.count('/')==2 and pin(expected_sha256),
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
        self.used[name]=info;self.captured[name]=data
        return data

    def capture(self,name,expected_sha256,maximum=256*1024**2):
        # Explicit reuse of a successful captured byte object; failed attempts
        # never enter this cache and cannot trigger another transport.
        if name in self.captured:
            data=self.captured[name]
            need(sha(data)==expected_sha256 and len(data)<=maximum, 'captured input pin/bound differs')
            return data
        return self.read(name,expected_sha256,maximum)


class PacketReplicas:
    """Detached opaque bytes from two explicit original packet copies; no fallback.

    Full archive/capsule admission remains the caller's separate mandatory gate.
    Only the exact outer-pinned, AFTER-observed flat input set is readable here.
    """
    def __init__(self, *, roots, outer_sha256, inner_sha256, after_raw,
                 after_sha256, control_root, protected_roots):
        need(type(roots) in (list, tuple) and len(roots)==2,
             'two explicit packet replicas required')
        self.roots=tuple(Path(p) for p in roots)
        need(all(p.is_absolute() and p==p.resolve(strict=True) and p.is_dir()
                 and not p.is_symlink() for p in self.roots),
             'canonical ordinary packet replica roots required')
        protected=tuple(Path(p).resolve(strict=True) for p in protected_roots)
        overlap=lambda a,b:a==b or a in b.parents or b in a.parents
        need(not overlap(*self.roots) and
             all(not overlap(a,b) for a in self.roots for b in protected),
             'packet replicas must be disjoint from each other and custody/staging')
        need(all(pin(p) for p in (outer_sha256,inner_sha256,after_sha256))
             and type(after_raw) is bytes and sha(after_raw)==after_sha256,
             'independent replica/AFTER pins required')
        transport=decode(after_raw)
        need(transport['returncode']==0, 'successful original AFTER transport required')
        after=decode(transport['stdout'])
        need(after['terminal_gate'] is True and not after.get('observation_error'),
             'conclusive original AFTER observation required')
        self.outer_sha256=outer_sha256;self.inner_sha256=inner_sha256
        self.after_sha256=after_sha256;self.control_root=control_root
        self.outer_info=after['files']['control/packet.json']
        self.inner_info=after['files']['control/inputs/packet.json']
        record(self.outer_info);record(self.inner_info)
        need(self.outer_info['sha256']==outer_sha256 and
             self.inner_info['sha256']==inner_sha256, 'AFTER packet pins differ')
        self._dirs=self._directories()
        outer=decode(read_exact(self.roots[0]/'packet.json',self.outer_info,32*1024**2))
        inner=decode(read_exact(self.roots[0]/'inputs/packet.json',self.inner_info,32*1024**2))
        need(outer['remote_root']==control_root and outer['gpu_packet_sha256']==inner_sha256,
             'original control/inner packet binding differs')
        declared={n[len('inputs/'):]:h for n,h in outer['new_input_files'].items()
                  if n.startswith('inputs/')}
        need(declared and all(relative(n)==n and Path(n).name==n and pin(h)
                              for n,h in declared.items()), 'flat outer input map required')
        need(set(declared)==set(inner['files'])|{'packet.json','admission.json'}
             and declared['packet.json']==inner_sha256 and
             all(declared[n]==h for n,h in inner['files'].items()),
             'complete original inner/outer input map differs')
        self.members={n[len('control/inputs/'):]:v for n,v in after['files'].items()
                      if n.startswith('control/inputs/')}
        need(set(self.members)==set(declared), 'AFTER flat input membership differs')
        directories=observed_directories(after['directories'])
        need('control/inputs' in directories and
             not any(n.startswith('control/inputs/') for n in directories),
             'AFTER input directory membership differs')
        for n,v in self.members.items():
            record(v)
            need(v['sha256']==declared[n] and v['bytes']<=256*1024**2,
                 'outer/AFTER input pin or bound differs: '+n)
        self.requested=dict(inner['files']);self.captured={}
        self.recheck()

    def _directories(self):
        result=[]
        for root in self.roots:
            for path in (root,root/'inputs'):
                need(path==path.resolve(strict=True) and not path.is_symlink(),
                     'packet directory alias forbidden')
                s=path.lstat();need(stat.S_ISDIR(s.st_mode),'packet directory required')
                result.append((s.st_dev,s.st_ino,s.st_mtime_ns,s.st_ctime_ns,s.st_mode))
        need(result[0][:2]!=result[2][:2] and result[1][:2]!=result[3][:2],
             'independent packet directories required')
        return tuple(result)

    def recheck(self):
        need(self._directories()==self._dirs,'packet directory changed')
        copies=[]
        for root in self.roots:
            read_exact(root/'packet.json',self.outer_info,32*1024**2)
            need({p.name for p in (root/'inputs').iterdir()}==set(self.members),
                 'packet replica input membership changed')
            copies.append({n:read_exact(root/'inputs'/n,info,256*1024**2)
                           for n,info in sorted(self.members.items())})
        need(copies[0]==copies[1],'packet replica bytes differ')
        for name in ['packet.json',*('inputs/'+n for n in self.members)]:
            need(not os.path.samestat((self.roots[0]/name).stat(),(self.roots[1]/name).stat()),
                 'packet replicas must not alias files')
        need(self._directories()==self._dirs,'packet directory changed during capture')
        captured={n:copies[0][n] for n in self.requested}
        need(not self.captured or self.captured==captured,'captured packet input changed')
        self.captured=captured
        return self.binding()

    def binding(self):
        return {'schema':'ace-original-gpu-packet-replicas-v1',
                'roots':[str(p) for p in self.roots], 'control_root':self.control_root,
                'outer_sha256':self.outer_sha256,'inner_sha256':self.inner_sha256,
                'after_sha256':self.after_sha256,
                'input_members':decode(canonical(self.members)),
                'input_members_sha256':sha(canonical(self.members)),
                'input_files':len(self.members),
                'input_bytes':sum(v['bytes'] for v in self.members.values()),
                'acceptance':False,'fallback_allowed':False}

    def capture(self, member, expected_sha256, maximum=256*1024**2):
        relative(member)
        need(member.startswith('control/inputs/') and member.count('/')==2,
             'only exact control flat inputs may use packet replicas')
        name=Path(member).name
        need(name in self.requested and pin(expected_sha256) and
             self.requested[name]==expected_sha256,'independent requested replica pin differs')
        data=self.captured[name]
        need(type(maximum) is int and 0<=len(data)<=maximum and
             sha(data)==expected_sha256,'detached replica bytes differ or exceed bound')
        return data


# Outer-coordinator bindings only. These members never enter scientific children.
STORAGE_FILES={'authority':'storage-authority.json','authority_review':'storage-review.md',
    'backend':'remote_custody.py','backend_review':'storage-backend-review.md',
    'receipt':'storage-receipt.json','seal':'storage-seal.json',
    'after':'storage-after.json','reference':'storage-predecessor.json'}


def capture_config(path, expected_sha256):
    path=Path(path)
    data=read_exact(path,{'bytes':path.stat().st_size,'sha256':expected_sha256},128*1024)
    config=decode(data)
    need(config['schema']=='ace-compact-downstream-inputs-v1' and set(config['members'])==set(STORAGE_FILES),
         'explicit complete storage input configuration required')
    supplied={'storage-config.json':data}
    for key,name in STORAGE_FILES.items():
        entry=config['members'][key];p=Path(entry['path'])
        maximum=128*1024**2 if key in ('receipt','seal','after') else 32*1024**2
        need(pin(entry['sha256']) and p.stat().st_size<=maximum,'bounded independently pinned storage input')
        supplied[name]=read_exact(p,{'sha256':entry['sha256'],'bytes':p.stat().st_size},maximum)
    return supplied


def stored_config(directory, pins):
    directory=Path(directory);supplied={}
    for name in ['storage-config.json',*STORAGE_FILES.values()]:
        p=directory/name;maximum=128*1024**2 if name in ('storage-receipt.json','storage-seal.json','storage-after.json') else 32*1024**2
        need(name in pins and pin(pins[name]) and p.stat().st_size<=maximum,'pinned bounded outer storage member required')
        supplied[name]=read_exact(p,{'sha256':pins[name],'bytes':p.stat().st_size},maximum)
    return supplied


def sealed_reference(root, expected_seal_sha256, *, full_original=False):
    """Verify named original bytes in place; never recursively copy an ancestor.

    For the one legacy full training capsule, consume the authenticated flat seal
    roster and hash every named original file, without a filesystem tree walk.
    Later compact capsules are independently sealed selected views.
    """
    root=Path(root).resolve(strict=True);p=root/'custody-inventory.json'
    raw=read_exact(p,{'sha256':expected_seal_sha256,'bytes':p.stat().st_size},128*1024**2)
    seal=decode(raw);views=list(seal['copies'].values())
    need(seal['identical'] is True and len(views)==2 and views[0]==views[1] and
         seal['acceptance'] is False and seal['excludes']=='custody-inventory.json itself' and
         str(root) in seal['copies'],'original independently sealed reference required')
    view=seal['copies'][str(root)];inventory_shape(view)
    need(len(view['files'])<=262144 and len(view['directories'])<=262144,'bounded original capsule roster')
    if full_original:
        for name,info in view['files'].items():
            p=root/name;need(not p.is_symlink(),'original reference file symlink')
            # Stream opaque originals; do not allocate tensor/archive bytes.
            st=p.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size==info['bytes'],'original reference file missing/changed')
            digest=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
            after=p.lstat()
            need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
                 digest.hexdigest()==info['sha256'],'original referenced bytes changed')
        for name in view['directories']:
            p=root/name;need(p.is_dir() and not p.is_symlink(),'original referenced directory missing')
    else:
        actual=selected_inventory(root);del actual['files']['custody-inventory.json']
        need(actual==view,'compact reference seal differs')
    return root,view


def verify_ancestry(ctx, first_raw, predecessor):
    """Exactly GPU→CPU→training or CPU→training; retain every original failed byte."""
    m,pins=ctx['metadata'],ctx['pins'];ref=decode(first_raw);seen=set();chain=[]
    stages=['cpu','training'] if predecessor=='gpu' else ['training']
    for stage in stages:
        need(ref['recursive_copy_created'] is False and ref['original_preserved'] is True and
             ref['disposition_sha256']==pins[stage+'-disposition'] and
             ref['raw_inventory_sha256']==pins[stage+'-inventory'],'independent ancestor reference binding differs')
        expected=(m['training-acceptance']['custody_inventory_sha256'] if stage=='training'
                  else m['gpu-launch-packet']['new_input_files']['storage-seal.json'])
        need(ref['custody_inventory_sha256']==expected,'original ancestor seal pin differs')
        root=Path(ref['path']).resolve(strict=True);need(root not in seen,'ancestor reference cycle');seen.add(root)
        root,view=sealed_reference(root,expected,full_original=stage=='training')
        need(view['files']['disposition.json']['sha256']==pins[stage+'-disposition'] and
             view['files']['inventory_transport.json']['sha256']==pins[stage+'-inventory'],'ancestor capsule actual metadata differs')
        chain.append({'stage':stage,'path':str(root),'custody_inventory_sha256':expected,
                      'all_original_named_bytes_verified':stage=='training','recursive_copy_created':False})
        if stage=='cpu':
            item=view['files']['predecessor-reference.json']
            ref=decode(read_exact(root/'predecessor-reference.json',item,128*1024))
        else:
            wanted=pins['token-failure-custody-inventory'];matches=[n for n,v in view['files'].items()
                if Path(n).name=='custody-inventory.json' and v['sha256']==wanted]
            need(matches,'original failed TOKEN capsule/archive/receipts absent from training ancestor')
            failed=m['token-failure-custody-inventory'];oldviews=list(failed['copies'].values())
            need(failed['identical'] is True and len(oldviews)==2 and oldviews[0]==oldviews[1],'original failure views differ')
            for name in matches:
                prefix=name[:-len('custody-inventory.json')]
                observed={'files':{n[len(prefix):]:v for n,v in view['files'].items() if n.startswith(prefix) and n!=name},
                          'directories':sorted(n[len(prefix):] for n in view['directories'] if n.startswith(prefix) and n!=prefix.rstrip('/'))}
                need(observed==oldviews[0],'original complete failure evidence differs')
    return chain


def bind(ctx, supplied, *, predecessor, capsule, mirror, expected_roots, committed, helper_path):
    """Bind outer authority, two compact seals, original archive and exact references."""
    import types
    config=decode(supplied['storage-config.json'])
    need(config['schema']=='ace-compact-downstream-inputs-v1' and config['predecessor']==predecessor and
         set(config['members'])==set(STORAGE_FILES),'exact storage configuration differs')
    for key,name in STORAGE_FILES.items():
        need(sha(supplied[name])==config['members'][key]['sha256'],'independent storage member pin differs: '+name)
    authority=decode(supplied['storage-authority.json']);m,pins=ctx['metadata'],ctx['pins']
    need(authority['schema']=='ace-scientific-remote-custody-amendment-v1' and authority['status']=='adopted' and
         authority['scope']=='evidence_storage_only' and authority['freeze_sha256']==pins['freeze'] and
         authority['source_commit']==m['freeze']['source_commit'] and authority['scientific_inputs_unchanged'] is True and
         authority['originals_preserved'] is True and authority['full_remote_verification_required'] is True and
         authority['backup_kind']=='ustar' and authority['required_findings']==0 and type(authority['required_findings']) is int and
         authority['review_sha256']==sha(supplied['storage-review.md']) and supplied['storage-review.md'].strip(),
         'explicit adopted storage-only amendment/review required')
    backend_sha=config['members']['backend']['sha256']
    need(pin(backend_sha) and sha(supplied['remote_custody.py'])==backend_sha and authority['backend_sha256']==backend_sha and
         config['members']['backend']['sha256']==backend_sha and supplied['storage-backend-review.md'].strip(),
         'independent exact backend source/review required')
    backend_spec=config['members']['backend']
    for spec in (backend_spec,config['members']['backend_review']):
        relative(spec['repository_path'])
    need(committed(backend_spec['source_commit'],backend_spec['repository_path'])==supplied['remote_custody.py'],
         'captured backend full-commit bytes differ')
    review_spec=config['members']['backend_review']
    need(committed(backend_spec['source_commit'],review_spec['repository_path'])==supplied['storage-backend-review.md'],
         'captured backend review differs')
    backend=types.ModuleType('captured_remote_custody');backend.__file__='remote_custody.py'
    exec(compile(supplied['remote_custody.py'],'remote_custody.py','exec'),vars(backend))
    backend.bind_source(supplied['remote_custody.py'],backend_sha)
    seal_pin=sha(supplied['storage-seal.json']);capsules=tuple(Path(p).resolve(strict=True) for p in (capsule,mirror))
    need(capsules[0]!=capsules[1] and all(a not in b.parents for a in capsules for b in capsules if a!=b),
         'two disjoint selected capsules required')
    seal=decode(supplied['storage-seal.json'])
    need(set(seal['copies'])==set(map(str,capsules)),'independent selected seal roots differ')
    copies=[]
    for root in capsules:
        _,view=sealed_reference(root,seal_pin)
        for name,value in [('disposition.json',pins[predecessor+'-disposition']),
                           ('inventory_transport.json',pins[predecessor+'-inventory']),
                           ('inventory_after_transport.json',sha(supplied['storage-after.json'])),
                           ('remote-custody.json',sha(supplied['storage-receipt.json'])),
                           ('predecessor-reference.json',sha(supplied['storage-predecessor.json']))]:
            need(view['files'][name]['sha256']==value,'selected capsule metadata pin differs: '+name)
        copies.append(view)
    need(copies[0]==copies[1],'selected capsule copies differ')
    for name in copies[0]['files']:
        need(not os.path.samestat((capsules[0]/name).stat(),(capsules[1]/name).stat()),'selected capsule files alias')
    obs=ctx['observations'][predecessor];after=decode(supplied['storage-after.json'])
    need(after['returncode']==0,'successful original second observation required')
    after=decode(after['stdout'])
    for key in ('files','directories','parent_accounting','journal_pins','terminal_gate'):
        if key=='directories':
            need(observed_directories(after[key])==observed_directories(obs[key]),'original remote directory sets differ')
        else:need(after[key]==obs[key],'original remote before/after differs: '+key)
    need(sorted(after['related_accounting'])==sorted(obs['related_accounting']) and after['terminal_gate'] is True and
         not after.get('observation_error'),'conclusive original terminal observations required')
    disp=m[predecessor+'-disposition'];receipt_pin=sha(supplied['storage-receipt.json'])
    need(disp['custody_mode']=='curc_projects_full_backup_with_compact_local_evidence' and
         disp['complete_local_raw_copy'] is False and disp['remote_custody_receipt_sha256']==receipt_pin and
         disp['backup_helper_sha256']==backend_sha and disp['remote_files']==obs['files'] and
         sorted(disp['remote_directories'])==sorted(obs['directories']), 'explicit complete remote / partial local disposition required')
    evidence=ctx['storage_operation_evidence']
    parent=evidence.root
    need(evidence.last_reference is None,'fresh registered predecessor-use scope required')
    gate=Inputs(backend=backend,receipt_raw=supplied['storage-receipt.json'],receipt_sha256=receipt_pin,
                observation=obs,expected_roots=expected_roots,helper_sha256=backend_sha,
                local_roots=[p/'remote' for p in capsules],staging=parent/'operations',helper_path=helper_path,evidence=evidence)
    chain=verify_ancestry(ctx,supplied['storage-predecessor.json'],predecessor)
    gate.capsules=capsules;gate.capsule_seal=seal;gate.ancestry=chain;gate.supplied=supplied
    ctx['compact_inputs']=gate
    return gate


def recheck_capsules(gate):
    gate.recheck()
    for root in gate.capsules:
        actual=selected_inventory(root);del actual['files']['custody-inventory.json']
        need(actual==gate.capsule_seal['copies'][str(root)],'selected predecessor capsule changed during capture')


def collect_selected(*, backend, helper_sha256, before, roots, backup_root,
                     external, local, helper_path, save, observe, compare, copy_file):
    """Transport-only wrapper. Full immutable archive plus two selected views.

    Caller retains both locks, independent source/Git/adoption bindings, original
    input/resource/accounting gates and exception/failure publication. This never
    copies a predecessor capsule or claims that a selected view is a full tree.
    """
    import shutil
    external,local=Path(external),Path(local)
    need(external.is_dir() and local.is_dir() and external!=local and
         external not in local.parents and local not in external.parents,
         'two exclusive disjoint capsule roots required')
    need(not any(os.path.lexists(p/'remote') or os.path.lexists(p/'backup-operations')
                 for p in (external,local)), 'selected/operation destinations must be absent; no retry')
    expected={'files':before['files'],'directories':sorted(before['directories'])}
    inventory_shape(expected)
    selected,_=backend.selected_inventory(backend.normalized(before))
    inventory_shape(selected)
    selected_bytes=sum(v['bytes'] for v in selected['files'].values())
    # Conservatively charge both copies on each device, including bounded logs
    # and observation/capsule metadata. Existing references need no new storage.
    required=2*selected_bytes+32*len(canonical(before))+512*1024**2
    capacity={}
    for dest in (external,local):
        fs=os.statvfs(dest);available=fs.f_bavail*fs.f_frsize
        need(available>required,'insufficient aggregate selected-copy capacity')
        capacity[str(dest)]={'available_bytes':available,'required_increment_bytes':required,
                             'device':dest.stat().st_dev,'selected_bytes_per_copy':selected_bytes}
    save('local-capacity.json',capacity)
    receipt=backend.collect_backup(observation=before,roots=roots,backup_root=backup_root,
        local_root=external/'remote',helper_path=helper_path,log_dir=external/'backup-operations')
    need(receipt['schema']=='ace-remote-custody-v1' and receipt['full_backup_verified'] is True and
         receipt['acceptance'] is False and receipt['retry_allowed'] is False and
         receipt['full_local_copy'] is False and receipt['backup_kind']=='ustar' and
         receipt['helper_sha256']==helper_sha256 and receipt['roots']==roots and
         receipt['backup_root']==backup_root and receipt['local_root']==str(external/'remote') and
         receipt['selection_policy']=='current-launch-text-and-namespaces-v1' and
         receipt['full_inventory']==expected and receipt['local_inventory']==selected,
         'complete archive/selected receipt binding differs')
    omitted={n:v for n,v in expected['files'].items() if n not in selected['files']}
    need(all(expected['files'].get(n)==v for n,v in selected['files'].items()) and
         receipt['omitted_files']==omitted and receipt['omitted_bytes']==sum(v['bytes'] for v in omitted.values()) and
         selected_inventory(external/'remote')==selected,'exact selected/omitted partition differs')
    save('remote-custody.json',receipt)
    after=observe('inventory_after_transport.json')
    compare(before,after)  # Unchanged stage-specific scheduler/journal/root gates.
    for sub in ('remote','backup-operations'):
        shutil.copytree(external/sub,local/sub,copy_function=copy_file)
        need(selected_inventory(external/sub)==selected_inventory(local/sub),'independent selected/operation mirror differs')
        for name in selected_inventory(external/sub)['files']:
            need(not os.path.samestat((external/sub/name).stat(),(local/sub/name).stat()),'selected mirror shares inode')
    return expected,after,receipt
