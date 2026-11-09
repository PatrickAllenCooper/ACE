import signal
def deadline(signum,frame):raise TimeoutError('bounded full terminal observation')
signal.signal(signal.SIGALRM,deadline);signal.alarm(1200)

from pathlib import Path
import subprocess,hashlib,json,datetime,stat,os,sys,time
_READ_COUNT=0;_READ_STARTED=time.monotonic()
root=Path(ROOT);namespace=Path(NAMESPACE);original_namespace=Path(OLD_NAMESPACE)
def need(ok,message):
 if not ok:raise ValueError(message)
def fingerprint(path):
 global _READ_COUNT
 need(not path.is_symlink(),'remote symlink forbidden')
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'nonregular remote file')
  h=hashlib.sha256()
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
  after=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 need(identity(before)==identity(after)==identity(path.lstat()),'remote file changed during capture')
 _READ_COUNT+=1
 if _READ_COUNT%1000==0:print(json.dumps(dict(phase='terminal_inventory_verification',files_read=_READ_COUNT,elapsed_seconds=time.monotonic()-_READ_STARTED)),file=sys.stderr,flush=True)
 return dict(bytes=after.st_size,sha256=h.hexdigest()),after
v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],
 root_present=root.exists(),root_presence={label:Path(path).exists() for label,path in ROOTS.items()},
 namespace_present=os.path.lexists(namespace),original_namespace_present=os.path.lexists(original_namespace),
 original_hardlink_groups=[],upstream_pins={},upstream_errors={})
commands={
 'related_accounting':['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'],
 'steps':['sacct','-j',JOB,'--format',
   'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180,MaxRSS,TotalCPU','-n','-P'],
 'details':['sacct','-X','-j',JOB,'--format','JobIDRaw,Partition,QOS,TimelimitRaw,ReqTRES%180,User','-n','-P'],
 'queue':['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'],
 'job':['scontrol','show','job',JOB,'-o']}
for name,command in commands.items():
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=20)
  v['results'][name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired as exc:
  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
  v['results'][name]=dict(command=command,returncode=None,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr))
try:
 need(v['results']['related_accounting']['returncode']==0 and v['results']['queue']['returncode']==0,
      'scheduler observation unavailable')
 rows=[l.split('|') for l in v['results']['related_accounting']['stdout'].splitlines() if '|ace_llm' in l]
 selected=[r for r in rows if r[0]==JOB]
 need(len(selected)==1 and selected[0][2].split()[0] in TERMINAL,'requested job not conclusively terminal')
 queue=[l.split('|') for l in v['results']['queue']['stdout'].splitlines() if l.strip()]
 need(all(len(r)==5 and r[0]!=JOB and not r[1].lower().startswith('ace') for r in queue),'current/related ACE job active')
 v['parent_accounting']=selected[0];v['related_accounting']=rows;v['terminal_gate']=True
 links={}
 for label,tree,present in ((label,Path(path),v['root_presence'][label]) for label,path in ROOTS.items()):
  if not present:continue
  need(tree.is_dir() and tree==tree.resolve(),'ordinary remote root required')
  v['directories'].append(label)
  for path in sorted(tree.rglob('*')):
   need(not path.is_symlink(),'remote tree symlink forbidden')
   name=label+'/'+path.relative_to(tree).as_posix()
   if path.is_dir():v['directories'].append(name);continue
   info,s=fingerprint(path);v['files'][name]=info
   if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)
 for name,path,present in (('namespace.json',namespace,v['namespace_present']),
                           ('original-namespace.json',original_namespace,v['original_namespace_present'])):
  if not present:continue
  need(path.parent==path.parent.resolve(),'registry parent alias')
  info,s=fingerprint(path);v['files'][name]=info
  if s.st_nlink>1:links.setdefault((s.st_dev,s.st_ino,s.st_nlink),[]).append(name)
 v['original_hardlink_groups']=sorted([dict(paths=sorted(paths),link_count=k[2]) for k,paths in links.items()],
                                     key=lambda group:group['paths'])
 v['journal_pins']={str(p):fingerprint(p)[0]['sha256'] for d in root.parent.glob('ace_llm_intervention*')
                    for p in d.glob('submission.*.json')}
 # Explicit original pinned paths, including24 supplied development descriptor bytes; never enumerate realization or decode payloads.
 for path in UPSTREAM_PINS:
  try:v['upstream_pins'][path]=fingerprint(Path(path))[0]['sha256']
  except (OSError,ValueError) as exc:v['upstream_errors'][path]=dict(type=type(exc).__name__,message=str(exc))

except BaseException as exc:
 v['observation_error']=dict(type=type(exc).__name__,message=str(exc))
print(json.dumps(v,sort_keys=True))
