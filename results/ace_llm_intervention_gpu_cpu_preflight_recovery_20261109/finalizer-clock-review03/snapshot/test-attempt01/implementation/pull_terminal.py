"""Once-only complete scientific finalizer custody. No private decode or acceptance.
Standalone coordinator, authenticated builder and cleared transport helper only.
"""
from pathlib import Path
import argparse,ast,datetime,fcntl,hashlib,json,os,re,shutil,stat,subprocess,time,types,shlex
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01'
LOCAL=LAUNCH.parent/'terminal01'
EXT=BASE/'ACE-LLM-intervention-scientific-finalization-terminal-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,wall_seconds=4200,gpus=0,requeue=False)

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(data):return hashlib.sha256(data).hexdigest()

def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def pin(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None

def decode(data):
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate metadata key');result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite metadata: '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)

def fingerprint(path,*,sole_link=False):
    path=Path(path)
    need(not path.is_symlink(),'symlink custody forbidden: '+str(path))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
        before=os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and (not sole_link or before.st_nlink==1),'regular independent custody file required')
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
        after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    need(identity(before)==identity(after)==identity(path.lstat()),'file changed during inventory')
    return dict(bytes=after.st_size,sha256=digest.hexdigest())

def read_metadata(path,expected=None,maximum=32*1024**2):
    meta=fingerprint(path,sole_link=True);need(type(maximum) is int and 0<=maximum<=128*1024**2 and meta['bytes']<=maximum,'bounded coordinator metadata required')
    data=Path(path).read_bytes()
    need(sha(data)==meta['sha256'],'metadata changed during read')
    if expected is not None:need(pin(expected) and sha(data)==expected,'independent pin differs: '+str(path))
    return data

def inventory(root):
    root=Path(root);need(root.is_dir() and not root.is_symlink(),'ordinary custody directory required')
    files={};directories=[]
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(),'custody symlink forbidden')
        name=path.relative_to(root).as_posix()
        if path.is_dir():directories.append(name)
        else:files[name]=fingerprint(path,sole_link=True)
    return dict(files=files,directories=directories)

def write(path,data):
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())

def normalize(rows):
    result=[]
    for row in rows:
        need(len(row)>=8 and not any(row[8:]) and row[0].isdigit() and row[4].isdigit() and row[5].isdigit(),
             'malformed allocation accounting row')
        pairs=[part.split('=',1) for part in row[7].split(',') if part]
        need(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'malformed allocation TRES')
        result.append([row[0],row[1].strip(),row[2].strip(),row[3].strip(),int(row[4]),int(row[5]),row[6].strip(),sorted(pairs)])
    need(len({r[0] for r in result})==len(result),'duplicate historical job')
    return sorted(result,key=lambda r:int(r[0]))

def tres(text):
    pairs=[p.split('=',1) for p in text.split(',') if p]
    need(all(len(p)==2 and all(p) for p in pairs) and len(dict(pairs))==len(pairs),'invalid/duplicate allocation TRES')
    return dict(pairs)

def gpu_count(t,cpus,elapsed):
    if not t:return 0 if cpus==elapsed==0 else None
    g={k:v for k,v in t.items() if k.startswith('gres/')}
    if any(not v.isdigit() or not (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()):return None
    typed=[int(v) for k,v in g.items() if k.startswith('gres/gpu:')]
    aggregate=int(g['gres/gpu']) if 'gres/gpu' in g else None
    if aggregate is not None and typed and aggregate!=sum(typed):return None
    return aggregate if aggregate is not None else sum(typed)

def cost(rows):
    cpu=gpu=0
    for row in normalize(rows):
        count=gpu_count(dict(row[7]),row[4],row[5]);need(count is not None,'unknown historical GPU charge')
        cpu+=row[4]*row[5];gpu+=count*row[5]
    return cpu,gpu

def validate_remote_intent(data, expected_pin, local_intent, claim_pin):
    need(sha(data)==expected_pin, 'authenticated remote intent bytes differ')
    remote=decode(data)
    additions={'remote_utc','fresh_readiness'}
    need(type(local_intent) is dict and not (set(local_intent)&additions)
         and type(remote) is dict and set(remote)==set(local_intent)|additions,
         'remote intent fields differ from local intent plus explicit additions')
    need(raw({k:remote[k] for k in local_intent})==raw(local_intent),
         'authenticated remote intent differs from original local intent')
    need(pin(claim_pin) and remote['existing_namespace_claim_sha256']==claim_pin,
         'authenticated remote intent namespace claim differs')
    need(type(remote['remote_utc']) is str and bool(remote['remote_utc'])
         and type(remote['fresh_readiness']) is dict, 'remote intent augmentation malformed')





REMOTE_OBSERVATION="import signal\ndef deadline(signum,frame):raise TimeoutError('bounded full terminal observation')\nsignal.signal(signal.SIGALRM,deadline);signal.alarm(1200)\n\nfrom pathlib import Path\nimport subprocess,hashlib,json,datetime,stat,os,sys,time\n_READ_COUNT=0;_READ_STARTED=time.monotonic()\nroot=Path(ROOT);namespace=Path(NAMESPACE);original_namespace=Path(OLD_NAMESPACE)\ndef need(ok,message):\n if not ok:raise ValueError(message)\ndef fingerprint(path):\n global _READ_COUNT\n need(not path.is_symlink(),'remote symlink forbidden')\n with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:\n  before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'nonregular remote file')\n  h=hashlib.sha256()\n  for block in iter(lambda:f.read(1024**2),b''):h.update(block)\n  after=os.fstat(f.fileno())\n identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)\n need(identity(before)==identity(after)==identity(path.lstat()),'remote file changed during capture')\n _READ_COUNT+=1\n if _READ_COUNT%1000==0:print(json.dumps(dict(phase='terminal_inventory_verification',files_read=_READ_COUNT,elapsed_seconds=time.monotonic()-_READ_STARTED)),file=sys.stderr,flush=True)\n return dict(bytes=after.st_size,sha256=h.hexdigest()),after\nv=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],\n root_present=root.exists(),root_presence={label:Path(path).exists() for label,path in ROOTS.items()},\n namespace_present=os.path.lexists(namespace),original_namespace_present=os.path.lexists(original_namespace),\n original_hardlink_groups=[],upstream_pins={},upstream_errors={})\ncommands={\n 'related_accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format',\n   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],\n 'steps':['sacct','-j',JOB,'--format',\n   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU','-n','-P'],\n 'details':['sacct','-X','-j',JOB,'--format','JobIDRaw,Partition,QOS,TimelimitRaw,ReqTRES%180,User','-n','-P'],\n 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],\n 'job':['scontrol','show','job',JOB,'-o']}\nfor name,command in commands.items():\n try:\n  p=subprocess.run(command,capture_output=True,text=True,timeout=20)\n  v['results'][name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)\n except subprocess.TimeoutExpired as exc:\n  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x\n  v['results'][name]=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr))\ntry:\n need(v['results']['related_accounting']['returncode']==0 and v['results']['queue']['returncode']==0,\n      'scheduler observation unavailable')\n rows=[l.split('|') for l in v['results']['related_accounting']['stdout'].splitlines() if '|ace_llm' in l]\n selected=[r for r in rows if r[0]==JOB]\n need(len(selected)==1 and selected[0][2].split()[0] in TERMINAL,'requested job not conclusively terminal')\n queue=[l.split('|') for l in v['results']['queue']['stdout'].splitlines() if l.strip()]\n need(all(len(r)==5 and r[0]!=JOB and not r[1].lower().startswith('ace') for r in queue),'current/related ACE job active')\n v['parent_accounting']=selected[0];v['related_accounting']=rows;v['terminal_gate']=True\n links={}\n for label,tree,present in ((label,Path(path),v['root_presence'][label]) for label,path in ROOTS.items()):\n  if not present:continue\n  need(tree.is_dir() and tree==tree.resolve(),'ordinary remote root required')\n  v['directories'].append(label)\n  for path in sorted(tree.rglob('*')):\n   need(not path.is_symlink(),'remote tree symlink forbidden')\n   name=label+'/'+path.relative_to(tree).as_posix()\n   if path.is_dir():v['directories'].append(name);continue\n   info,s=fingerprint(path);v['files'][name]=info\n   if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)\n for name,path,present in (('namespace.json',namespace,v['namespace_present']),\n                           ('original-namespace.json',original_namespace,v['original_namespace_present'])):\n  if not present:continue\n  need(path.parent==path.parent.resolve(),'registry parent alias')\n  info,s=fingerprint(path);v['files'][name]=info\n  if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)\n v['original_hardlink_groups']=sorted([dict(paths=sorted(paths),link_count=k[2]) for k,paths in links.items()],\n                                     key=lambda group:group['paths'])\n v['journal_pins']={str(p):fingerprint(p)[0]['sha256'] for d in root.parent.glob('ace_llm_intervention*')\n                    for p in d.glob('submission.*.json')}\n # Explicit original pinned paths, including24 supplied development descriptor bytes; never enumerate realization or decode payloads.\n for path in UPSTREAM_PINS:\n  try:v['upstream_pins'][path]=fingerprint(Path(path))[0]['sha256']\n  except TimeoutError:raise\n  except (OSError,ValueError) as exc:v['upstream_errors'][path]=dict(type=type(exc).__name__,message=str(exc))\n\nexcept BaseException as exc:\n v['observation_error']=dict(type=type(exc).__name__,message=str(exc))\nprint(json.dumps(v,sort_keys=True))\n"

def cpu_tres(t):
    return (t.get('cpu')=='2' and t.get('node')=='1' and t.get('mem') in ('7680M','7.5G','7.50G')
        and not any(k.startswith('gres/') for k in t))

def resources(observation,job,output,files):
    row=observation['parent_accounting'];normalized=normalize([row])[0];errors=[]
    cpus,elapsed=normalized[4:6];allocated=dict(normalized[7]);gpu=gpu_count(allocated,cpus,elapsed)
    details=observation['results']['details'];requested={};request_identity=False
    if details['returncode']==0:
        rows=[r.split('|') for r in details['stdout'].splitlines() if r.strip()]
        if len(rows)==1 and len(rows[0])>=6:
            r=rows[0]
            try:requested=tres(r[4])
            except ValueError as exc:errors.append(str(exc))
            request_identity=r[:4]==[job,'acpu','cpu-normal','70'] and r[5]=='paco0228'
    fields={};allocation_pin=files.get('launch/finalization/allocation.json',{}).get('sha256')
    if allocation_pin:
        try:
            record=decode(read_metadata(output/'allocation.json',allocation_pin))
            need(record['returncode']==0,'saved allocation command failed')
            fields=dict(re.findall(r'([^\s=]+)=(\S+)',record['stdout']))
        except (ValueError,KeyError,OSError) as exc:errors.append('saved allocation malformed: '+str(exc))
    if not fields and observation['results']['job']['returncode']==0:
        fields=dict(re.findall(r'([^\s=]+)=(\S+)',observation['results']['job']['stdout']))
    no_requeue=fields.get('JobId')==job and fields.get('Requeue')=='0' and fields.get('Restarts')=='0'
    matched=(request_identity and cpu_tres(requested) and cpu_tres(allocated) and no_requeue
        and normalized[1]=='ace_llm_scientific_finalization01' and normalized[6]=='ucb736_asc2'
        and cpus==2 and gpu==0 and 0<=elapsed<=4200)
    if not matched:errors.append('CPU finalizer identity/request/allocation/no-requeue/wall differs or unavailable')
    if gpu is None:errors.append('unknown accelerator charge; never infer zero')
    return dict(resource_contract_matched=matched,errors=errors,expected=RESOURCES,requested_tres=requested,
        allocated_tres=allocated,allocated_cpus=cpus,elapsed_seconds=elapsed,allocated_cpu_seconds=cpus*elapsed,
        gpu_count=gpu,gpu_slice_seconds=None if gpu is None else gpu*elapsed,
        scheduler_state=row[2],scheduler_exit=row[3],account=row[6],alloc_tres=row[7],
        no_requeue_verified=no_requeue,allocation_fields=fields,allocation_source_sha256=allocation_pin)

ROOTS={}
REMOTE_NAMESPACE=''
OLD_REMOTE_NAMESPACE=''




def compare_observations(before,after):
    for key in ('files','directories','root_presence','root_present','namespace_present','original_namespace_present',
                'original_hardlink_groups','parent_accounting','journal_pins','upstream_pins','upstream_errors'):
        need(before[key]==after[key],'second full observation differs: '+key)
    need(normalize(before['related_accounting'])==normalize(after['related_accounting']), 'second accounting differs')


def accounting_extension(obs,baseline,job,intent_pin,result_pin):
    prior=[r for r in obs['related_accounting'] if r[0]!=job];current=[r for r in obs['related_accounting'] if r[0]==job]
    need(baseline['terminal_jobs']==48 and baseline['submission_journals']==98 and len(prior)==48 and len(current)==1
         and normalize(prior)==normalize(baseline['related_accounting'])
         and normalize(current)==normalize([obs['parent_accounting']])
         and all(r[2].split()[0] in TERMINAL for r in obs['related_accounting']), 'exact48 predecessor rows plus actual49th required')
    expected={**baseline['journal_pins'],ROOT+'/submission.intent.json':intent_pin,ROOT+'/submission.result.json':result_pin}
    need(len(baseline['journal_pins'])==98 and len(expected)==100 and obs['journal_pins']==expected,'exact98 prior and100 final journal pins required')
    old=cost(baseline['related_accounting']);new=cost(obs['related_accounting']);delta=cost(current)
    need(old==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']) and new==(old[0]+delta[0],old[1]+delta[1]),'actual full history charge differs')
    return dict(closed_cpu_seconds=new[0],closed_gpu_slice_seconds=new[1],allocated_cpu_seconds=delta[0],gpu_slice_seconds=delta[1],
        cpu_admission_after_release=new[0]+3600,outstanding_reservation=0)


def preserved_roots(actual,expected_rosters,cpu_root):
    """Only the frozen private reporting surface may append; all original bytes remain."""
    errors=[]
    for label,root in ROOTS.items():
        if label=='launch':continue
        expected=expected_rosters[root];prefix=label+'/'
        files={n[len(prefix):]:v['sha256'] for n,v in actual['files'].items() if n.startswith(prefix)}
        dirs=sorted(n[len(prefix):] for n in actual['directories'] if n.startswith(prefix))
        if label not in actual['directories'] or any(files.get(n)!=h for n,h in expected['files'].items()):
            errors.append('original complete root missing/changed: '+label)
        additions=set(files)-set(expected['files']);new_dirs=set(dirs)-set(expected['directories'])
        if not set(expected['directories'])<=set(dirs):errors.append('original empty directory missing: '+label)
        if root==cpu_root:
            campaign='cpu/worker/artifacts/campaign/'
            allowed_file=lambda n:n.startswith(campaign+'private/') or n in {campaign+x for x in ('finalization-intent.json','finalization.candidate.json','finalization.json')}
            allowed_dir=lambda n:n==campaign+'private' or n.startswith(campaign+'private/')
            if not all(allowed_file(n) for n in additions) or not all(allowed_dir(n) for n in new_dirs):errors.append('write outside frozen finalizer reporting surface')
        elif additions or new_dirs:errors.append('unexpected upstream append: '+label)
    return errors


def committed(commit,relative,data):
    p=Path(relative)
    need(re.fullmatch('[0-9a-f]{40}',commit) and not p.is_absolute() and '..' not in p.parts,'safe full Git binding required')
    result=subprocess.run(['git','--no-replace-objects','show',commit+':'+relative],cwd=REPO,capture_output=True,timeout=30)
    need(result.returncode==0 and result.stdout==data,'committed exact bytes differ: '+relative)


def copy_file(src,dst):
    need(not Path(src).is_symlink(),'copy symlink forbidden')
    with Path(src).open('rb') as source,Path(dst).open('xb') as target:
        shutil.copyfileobj(source,target,1024**2);target.flush();os.fsync(target.fileno())
    return str(dst)


def retain_partial(source,dest):
    """Best-effort mirror of originals and partial bytes; existing files never overwritten."""
    errors=[]
    for path in sorted(source.rglob('*')):
        target=dest/path.relative_to(source)
        try:
            need(not path.is_symlink(),'partial evidence symlink')
            if path.is_dir():target.mkdir(mode=0o700,exist_ok=True)
            elif not os.path.lexists(target):copy_file(path,target)
            else:need(fingerprint(path)==fingerprint(target),'existing partial mirror differs')
        except BaseException as exc:errors.append(dict(path=str(path.relative_to(source)),type=type(exc).__name__,message=str(exc)))
    return errors


def main():
    global ROOTS,REMOTE_NAMESPACE,OLD_REMOTE_NAMESPACE
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('job-id','packet-sha256','submission-result-sha256','local-intent-sha256','packet-commit',
        'collector-sha256','collector-repository-path','collector-review','collector-review-sha256',
        'packet-review','packet-review-sha256','backup-root'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();need(re.fullmatch('[1-9][0-9]*',a.job_id),'explicit actual job required')
    self_bytes=read_metadata(Path(__file__),a.collector_sha256)
    committed(a.packet_commit,a.collector_repository_path,self_bytes)
    review=read_metadata(a.collector_review,a.collector_review_sha256);need(review.strip(),'independent collector review required')
    committed(a.packet_commit,str(Path(a.collector_review).absolute().relative_to(REPO)),review)
    packet_raw=read_metadata(LAUNCH/'packet.json',a.packet_sha256);packet=decode(packet_raw)
    pins={**packet['new_input_files'],'packet.json':a.packet_sha256}
    for name,h in pins.items():
        need(not Path(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/')) and pin(h),'safe pinned input roster required')
        data=(LAUNCH/name).read_bytes();need(sha(data)==h and fingerprint(LAUNCH/name,sole_link=True)==fingerprint(LAUNCH_EXT/name,sole_link=True),'exact input mirrors required')
        need(not os.path.samestat((LAUNCH/name).stat(),(LAUNCH_EXT/name).stat()),'input mirrors must be independent')
        committed(a.packet_commit,str((LAUNCH/name).relative_to(REPO)),data)
    packet_review=read_metadata(a.packet_review,a.packet_review_sha256);need(packet_review.strip(),'independent concrete packet review required')
    committed(a.packet_commit,str(Path(a.packet_review).absolute().relative_to(REPO)),packet_review)
    result_raw=read_metadata(LAUNCH/'submission.result.json',a.submission_result_sha256);result=decode(result_raw)
    intent_raw=read_metadata(LAUNCH/'submission.intent.json',a.local_intent_sha256);intent=decode(intent_raw)
    need(result['job_id']==a.job_id and result['returncode']==0 and result['ambiguous'] is False and pin(result['intent_sha256']), 'exact original unambiguous submission required')
    need(intent['packet_sha256']==a.packet_sha256 and intent['packet_commit']==a.packet_commit
         and intent['review_sha256']==a.packet_review_sha256 and intent['pins']==pins,'original intent/packet/review differs')
    # Only authenticated metadata coordinator code is loaded, never a production source or private payload.
    builder_source=read_metadata(LAUNCH/'build_packet.py',pins['build_packet.py'])
    b=types.ModuleType('pinned_finalizer_metadata');b.__file__=str(LAUNCH/'build_packet.py')
    exec(compile(builder_source,b.__file__,'exec'),vars(b))
    need(packet['schema']==b.OUTER_VERSION and packet['remote_root']==intent['remote_root']==ROOT
         and packet['source_commit']==intent['source_commit']==b.SOURCE_COMMIT
         and packet['resources']==intent['resources']==b.RESOURCES==RESOURCES
         and packet['deadlines']==intent['deadlines']==b.DEADLINES,'frozen finalizer profile differs')
    supplied={n:read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)],b.metadata_maximum(n)) for n in b.METADATA}
    ctx=b.authenticate(supplied);m=ctx['metadata'];base=m['gpu-baseline']
    claimed=[];started=time.monotonic();cpu=time.process_time()
    def save(name,value):
        for dest in claimed:write(dest/name,raw(value))
    with b.coordinator_locks():
        need(not any(os.path.lexists(p) for p in (EXT,LOCAL,EXT.with_name(EXT.name+'-storage-collector-operations'))),'exclusive terminal destinations; never retry')
        try:
            for dest in (EXT,LOCAL):
                dest.mkdir(mode=0o700);claimed.append(dest)
                write(dest/'pull_terminal.py',self_bytes);write(dest/'collector-review.md',review)
            b.start_storage_operations(ctx,EXT.with_name(EXT.name+'-storage-collector-operations'),(EXT,LOCAL),
                dict(phase='collector',job_id=a.job_id,packet_sha256=a.packet_sha256,
                     collector_sha256=a.collector_sha256,packet_commit=a.packet_commit))
            gate=b.configure_packet_storage(ctx,LAUNCH,pins,packet)
            need(intent['storage_binding']==packet['storage_binding'],'original storage intent differs')
            b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])
            need(packet['original_input_pins']==ctx['upstream_pins'] and packet['upstream_rosters']==ctx['upstream_rosters']
                 and packet['token_recovery']==intent['token_recovery']==ctx['token_recovery']
                 and packet['failed_token_root']==intent['failed_token_root']==b.FAILED_TOKEN_ROOT
                 and packet['failed_token_job_id']==intent['failed_token_job_id']==b.FAILED_TOKEN_JOB
                 and packet['baseline_sha256']==intent['baseline_sha256']==ctx['pins']['gpu-baseline']
                 and packet['gpu_acceptance_sha256']==intent['gpu_acceptance_sha256']==ctx['pins']['gpu-acceptance']
                 and intent['gpu_disposition_sha256']==ctx['pins']['gpu-disposition']
                 and intent['gpu_custody_inventory_sha256']==ctx['pins']['gpu-custody-inventory']
                 and packet['admission']==intent['admission']==b.budget(base)
                 and intent['existing_namespace_claim_sha256']==packet['namespace_claim_sha256']==ctx['pins']['namespace-claim']
                 and intent['namespace_admission_sha256']==ctx['pins']['namespace-admission']
                 and intent['recovery_authority_sha256']==ctx['pins']['recovery-authority']
                 and intent['original_namespace_claim_sha256']==b.ORIGINAL_CLAIM_SHA
                 and intent['new_namespace_claimed'] is intent['replacement_worlds'] is False
                 and packet['gpu_job_id']==intent['gpu_job_id']==m['gpu-acceptance']['job_id']
                 and packet['prior_jobs']==48 and packet['prior_journals']==98,'complete original history joins differ')
            need(packet['gpu_recovery']==intent['gpu_recovery']==ctx['gpu_recovery']
                 and packet['gpu_recovery_acceptance_sha256']==intent['gpu_recovery_acceptance_sha256']==ctx['pins']['gpu-recovery-acceptance']
                 and packet['operational_source_commit']==intent['operational_source_commit']==m['finalizer-source-manifest']['source_commit'],
                 'full GPU02/finalizer operational lineage differs')
            need(a.job_id not in {r[0] for r in base['related_accounting']},'new finalizer job required')
            backend=gate.backend;helper_source=gate.supplied['remote_custody.py'];helper_review=gate.supplied['storage-backend-review.md']
            ROOTS={'launch':ROOT,**{('gpu' if label=='launch' else 'gpu-control' if label=='control' else label):root for label,root in b.RAW_ROOTS.items()}}
            REMOTE_NAMESPACE=b.REMOTE_REGISTRY;OLD_REMOTE_NAMESPACE=b.ORIGINAL_REMOTE_REGISTRY
            source='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,JOB=a.job_id,NAMESPACE=REMOTE_NAMESPACE,
                OLD_NAMESPACE=OLD_REMOTE_NAMESPACE,ROOTS=ROOTS,TERMINAL=TERMINAL,UPSTREAM_PINS=ctx['upstream_pins']).items())+'\n'+REMOTE_OBSERVATION
            def observe(name):
                try:
                    p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=1230)
                    record=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,python_source=source,utc=utc())
                except subprocess.TimeoutExpired as exc:
                    txt=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else v
                    record=dict(returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr),python_source=source,utc=utc())
                save(name,record);need(record['returncode']==0,'failed original observation retained')
                obs=decode(record['stdout']);need(obs.get('terminal_gate') is True and not obs.get('observation_error'),'conclusive complete terminal observation required')
                return obs
            for dest in claimed:
                write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)
            save('collector-intent.json',dict(job_id=a.job_id,packet_sha256=a.packet_sha256,packet_commit=a.packet_commit,
                submission_result_sha256=a.submission_result_sha256,local_intent_sha256=a.local_intent_sha256,
                collector_sha256=a.collector_sha256,collector_review_sha256=a.collector_review_sha256,
                token_recovery=ctx['token_recovery'],gpu_custody_inventory_sha256=ctx['pins']['gpu-custody-inventory'],
                roots=ROOTS,backup_root=a.backup_root,backup_helper_sha256=gate.helper_sha256,
                storage_binding=packet['storage_binding'],complete_local_raw_copy=False,acceptance=False,retry_allowed=False))
            save('predecessor-reference.json',dict(path=str(ctx['custody_paths'][0]),
                custody_inventory_sha256=ctx['pins']['gpu-custody-inventory'],
                disposition_sha256=ctx['pins']['gpu-disposition'],raw_inventory_sha256=ctx['pins']['gpu-inventory'],
                complete_existing_capsule_authenticated=True,recursive_copy_created=False,original_preserved=True,
                custody_mode='curc_projects_full_backup_with_compact_local_evidence'))
            copies={}
            for label,original in [('coordinator-local',LAUNCH),('coordinator-external',LAUNCH_EXT)]:
                view=inventory(original);copies[label]=(Path(original),view)
                save(label+'-reference.json',dict(path=str(original),inventory=view,
                    recursive_copy_created=False,original_preserved=True))
            local_claims={}
            for name,path in [('local.namespace.json',b.LOCAL_REGISTRY),('original-local.namespace.json',b.ORIGINAL_LOCAL_REGISTRY)]:
                data=read_metadata(path);local_claims[name]=sha(data)
                for dest in claimed:write(dest/name,data)
            before=observe('inventory_transport.json')
            all_roots={**ROOTS,'namespace.json':REMOTE_NAMESPACE,'original-namespace.json':OLD_REMOTE_NAMESPACE}
            roots={label:path for label,path in all_roots.items() if label in before['directories'] or label in before['files']}
            actual,after,backup_receipt=b.compact_adapter().collect_selected(backend=backend,
                helper_sha256=gate.helper_sha256,before=before,roots=roots,backup_root=a.backup_root,
                external=EXT,local=LOCAL,helper_path=HELPER,save=save,observe=observe,
                compare=compare_observations,copy_file=copy_file)
            for label,(original,view) in copies.items():
                need(inventory(original)==view,'original coordinator changed')
            # Recheck both independent original GPU capsules, not only the chosen copied capsule.
            b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])
            errors=preserved_roots(actual,ctx['upstream_rosters'],b.CPU_ROOT)
            if before['upstream_pins']!=ctx['upstream_pins'] or before['upstream_errors']:errors.append('original upstream bytes changed')
            mismatches=[n for n,h in pins.items() if actual['files'].get('launch/'+n,{}).get('sha256')!=h]
            if mismatches:errors.append('original current packet/input pins differ')
            for name,h in [('submission.result.json',a.submission_result_sha256),('submission.intent.json',result['intent_sha256'])]:
                if actual['files'].get('launch/'+name,{}).get('sha256')!=h:errors.append('original current journal differs: '+name)
            try:validate_remote_intent(read_metadata(EXT/'remote/launch/submission.intent.json'),result['intent_sha256'],intent,ctx['pins']['namespace-claim'])
            except (ValueError,OSError,KeyError,TypeError) as exc:errors.append('remote/local intent differs: '+str(exc))
            for name,local_name,path,h in [('namespace.json','local.namespace.json',b.LOCAL_REGISTRY,ctx['pins']['namespace-claim']),
                ('original-namespace.json','original-local.namespace.json',b.ORIGINAL_LOCAL_REGISTRY,b.ORIGINAL_CLAIM_SHA)]:
                if actual['files'].get(name,{}).get('sha256')!=h or local_claims[local_name]!=h or fingerprint(path)['sha256']!=h:errors.append('separate namespace changed: '+name)
            accounting=None
            try:accounting=accounting_extension(before,base,a.job_id,result['intent_sha256'],a.submission_result_sha256)
            except (ValueError,KeyError,TypeError) as exc:errors.append('accounting/journals unresolved: '+str(exc))
            allocation=resources(before,a.job_id,EXT/'remote/launch/finalization',actual['files']);errors+=allocation['errors']
            if before['results']['details']!=after['results']['details']:errors.append('scheduler request evidence changed')
            steps=before['results']['steps'];rows=[r.split('|')[:8] for r in (steps.get('stdout') or '').splitlines() if r.startswith(a.job_id+'|')]
            if steps['returncode']!=0 or len(rows)!=1 or normalize(rows)!=normalize([before['parent_accounting']]):errors.append('step/parent accounting disagrees')
            if accounting:
                if accounting['closed_cpu_seconds']+3600>57600 or accounting['closed_gpu_slice_seconds']>7200:errors.append('actual cumulative resource cap exceeded')
                if (accounting['allocated_cpu_seconds'],accounting['gpu_slice_seconds'])!=(allocation['allocated_cpu_seconds'],allocation['gpu_slice_seconds']):errors.append('current allocation charge differs')
            names=('finalization/failure.json','finalization/worker/failure.json','finalization/worker/terminal.json',
                'finalization/worker.cleanup.json','finalization/worker.resources.json')
            markers={n:actual['files'].get('launch/'+n) for n in names}
            campaign='cpu/cpu/worker/artifacts/campaign/'
            markers.update({n:actual['files'].get(campaign+n) for n in ('finalization-intent.json','finalization.candidate.json','finalization.json')})
            failed=allocation['scheduler_state'].split()[0]!='COMPLETED' or allocation['scheduler_exit']!='0:0' or any(v is not None for n,v in markers.items() if n.endswith('failure.json'))
            status='terminal_failed_outputs_preserved' if failed else 'terminal_custody_review_pending'
            if errors:status='terminal_custody_with_binding_disagreements'
            disposition=dict(status=status,job_id=a.job_id,remote_root=ROOT,packet_sha256=a.packet_sha256,
                submission_result_sha256=a.submission_result_sha256,custody_bytes_verified=True,current_lineage_verified=not errors,
                errors=errors,input_pin_mismatches=mismatches,remote_files=actual['files'],remote_directories=actual['directories'],
                raw_files=len(actual['files']),raw_bytes=sum(v['bytes'] for v in actual['files'].values()),
                custody_mode='curc_projects_full_backup_with_compact_local_evidence',
                remote_custody_receipt_sha256=sha(raw(backup_receipt)),backup_helper_sha256=gate.helper_sha256,
                complete_local_raw_copy=False,local_raw_files=len(backup_receipt['local_inventory']['files']),
                local_raw_bytes=sum(v['bytes'] for v in backup_receipt['local_inventory']['files'].values()),
                original_hardlink_groups=before['original_hardlink_groups'],allocation=allocation,accounting=accounting,
                observed_related_jobs=len(before['related_accounting']),observed_submission_journals=len(before['journal_pins']),
                token_recovery=ctx['token_recovery'],original_input_pins=ctx['upstream_pins'],output_marker_metadata=markers,
                acceptance=False,scientific_qualified=False,runtime_qualified=False,cleanup_qualified=False,retry_allowed=False,
                private_outputs_decoded=False,collector_generated_responses=0,collector_training_updates=0,
                collector_process_cpu_seconds=time.process_time()-cpu,collector_wall_seconds=time.monotonic()-started,
                cost_scope='Local collector process only; actual allocation charges separate. Publication may precede failed cleanup.')
            save('disposition.json',disposition);need(not errors,'all raw outputs preserved; independent reconciliation required')
            views={str(p):inventory(p) for p in claimed};need(views[str(EXT)]==views[str(LOCAL)],'complete final mirrors differ')
            save('custody-inventory.json',dict(copies=views,identical=True,excludes='custody-inventory.json itself',acceptance=False))
            need(inventory(EXT)==inventory(LOCAL),'sealed capsule readback differs')
            print(json.dumps(dict(status=status,job_id=a.job_id,acceptance=False)))
        except BaseException as exc:
            mirror_errors=retain_partial(EXT,LOCAL) if len(claimed)==2 else []
            try:save('collector-failure.json',dict(type=type(exc).__name__,message=str(exc),partial_mirror_errors=mirror_errors,acceptance=False,retry_allowed=False))
            except BaseException:pass
            raise

if __name__=='__main__':main()
