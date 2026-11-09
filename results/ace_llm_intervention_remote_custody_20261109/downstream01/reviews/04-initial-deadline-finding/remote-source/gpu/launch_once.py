
import pathlib,json,hashlib,subprocess,datetime,os,stat,re,signal,sys,time
os.umask(0o077)
def remote_deadline(signum,frame):raise TimeoutError('absolute remote coordinator deadline')
signal.signal(signal.SIGALRM,remote_deadline);signal.alarm(1200)
_READ_COUNT=0;_READ_STARTED=time.monotonic()
root=pathlib.Path(C['root']); registry=pathlib.Path(C['registry'])
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path,allow_hardlink=False):
 global _READ_COUNT
 path=pathlib.Path(path)
 need(path.absolute()==path.resolve(),'unresolved/symlink path '+str(path))
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and (allow_hardlink or s.st_nlink==1),'regular sole-linked input required')
  value=f.read(256*1024**2+1);after=os.fstat(f.fileno())
 identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink)
 need(identity(s)==identity(after)==identity(path.lstat()),'input changed during capture')
 need(len(value)<=256*1024**2,'oversize metadata')
 _READ_COUNT+=1
 if _READ_COUNT%1000==0:print(json.dumps(dict(phase='upstream_input_verification',files_read=_READ_COUNT,elapsed_seconds=time.monotonic()-_READ_STARTED)),file=sys.stderr,flush=True)
 return value
def write(path,value):
 with pathlib.Path(path).open('xb') as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
def run(command):
 try:p=subprocess.run(command,capture_output=True,text=True,timeout=20)
 except subprocess.TimeoutExpired as exc:
  txt=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else value
  raise RuntimeError(json.dumps(dict(command=command,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr)))) from exc
 need(p.returncode==0,json.dumps(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
 return p.stdout
def normalized(rows):
 result=[]
 for row in rows:
  need(len(row)>=8 and not any(row[8:]) and row[0].isdigit() and row[4].isdigit() and row[5].isdigit(),
       'malformed accounting row')
  pairs=[v.split('=',1) for v in row[7].split(',') if v]
  need(all(len(v)==2 for v in pairs) and len({v[0] for v in pairs})==len(pairs),'malformed allocation')
  result.append([row[0],row[1].strip(),row[2].strip(),row[3].strip(),int(row[4]),int(row[5]),row[6].strip(),sorted(pairs)])
 need(len({r[0] for r in result})==len(result),'duplicate job')
 return sorted(result,key=lambda r:int(r[0]))
def cost(rows):
    cpu=gpu=0
    for r in normalized(rows):
        t=dict(r[7]);need(t or r[4]==r[5]==0,'missing actual TRES; unknown cost')
        g={k:v for k,v in t.items() if k.startswith('gres/')}
        need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in g.items()),'unknown accelerator charge')
        typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
        n=int(g['gres/gpu']) if 'gres/gpu' in g else typed
        need(not ('gres/gpu' in g and any(k.startswith('gres/gpu:') for k in g)) or n==typed,'inconsistent GPU charge')
        cpu+=r[4]*r[5];gpu+=n*r[5]
    return cpu,gpu

def duration(s):
 if s in ('UNLIMITED','INFINITE'):return float('inf')
 days=0
 if '-' in s:
  d,s=s.split('-',1);days=int(d)
 fields=s.split(':');need(len(fields)==3 and all(v.isdigit() for v in fields),'unreadable live wall limit')
 h,m,sec=map(int,fields);return days*86400+h*3600+m*60+sec
def allows(value,name):return value=='ALL' or name in value.split(',')
def reconcile():
 queue=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 for line in queue.splitlines():
  fields=line.split('|');need(len(fields)==5,'malformed queue row')
  need(not fields[1].lower().startswith('ace'),'active ACE job; leave it untouched')
 accounting=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[line.split('|') for line in accounting.splitlines() if '|ace_llm' in line]
 need(len(rows)==C['baseline']['terminal_jobs'] and normalized(rows)==normalized(C['baseline']['related_accounting']),
      'exact historical accounting changed')
 need(all(r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),
      'nonterminal historical job')
 need(cost(rows)==(C['baseline']['closed_cpu_seconds'],C['baseline']['closed_gpu_slice_seconds']),'closed costs changed')
 need(C['baseline']['closed_cpu_seconds']+17520<=57600 and C['baseline']['closed_gpu_slice_seconds']+2880<=7200,'full remaining GPU/finalizer caps exceed budget')
 journals={str(p):sha(read(p)) for d in root.parent.glob('ace_llm_intervention*')
           for p in d.glob('submission.*.json')}
 need(len(journals)==C['baseline']['submission_journals'] and journals==C['baseline']['journal_pins'],
      'exact full prior journal inventory changed')
 partition=run(['scontrol','show','partition','aa100','-o'])
 fields=dict(x.split('=',1) for x in partition.split() if '=' in x)
 need(fields.get('PartitionName')=='aa100' and fields.get('State')=='UP' and
      fields.get('MaxMemPerCPU')=='3840' and duration(fields.get('MaxTime',''))>=2880,
      '4CPU/15360M/2880s partition feasibility changed')
 need(allows(fields.get('AllowAccounts',''),'ucb736_asc2') and allows(fields.get('AllowQos',''),'gpu-testing')
      and 'ucb736_asc2' not in fields.get('DenyAccounts','').split(',')
      and 'gpu-testing' not in fields.get('DenyQos','').split(','),'partition admission changed')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228',
            'format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[l.split('|') for l in assoc.splitlines() if l.strip()]
 need(len(ar)==1 and ar[0][:3]==['alpine','ucb736_asc2','paco0228'] and ar[0][3] in ('','aa100')
      and ar[0][4]!='0' and 'gpu-testing' in ar[0][5].split(','),'account GPU QoS differs')
 qos=run(['sacctmgr','-n','-P','show','qos','where','name=gpu-testing','format=Name,MaxWall,MaxTRESPU'])
 qr=[l.split('|') for l in qos.splitlines() if l.strip()]
 need(len(qr)==1 and qr[0][0]=='gpu-testing' and (qr[0][1]=='' or duration(qr[0][1])>=2880),
      'live QoS wall insufficient')
 limits=dict(part.split('=',1) for part in qr[0][2].split(',') if '=' in part)
 cap=limits.get('gres/gpu:a100_3g.20gb')
 need(cap is not None and cap.isdigit() and int(cap)>=1,'live QoS MIG cap unavailable/inadequate')
 gpu_types=run(['sinfo','-p','aa100','-h','-o','%G'])
 need('gpu:a100_3g.20gb:' in gpu_types,'required20GB MIG type unavailable')
 manifest=pathlib.Path(C['manifest']['path']);archive=pathlib.Path(C['archive']['path'])
 need(sha(read(manifest))==C['manifest']['sha256'],'accepted runtime manifest changed')
 st=archive.lstat()
 need(archive==archive.resolve() and stat.S_ISREG(st.st_mode) and st.st_size==8438650880,
      'accepted archive metadata changed')
 need(sha(read(registry))==C['namespace_claim_sha256'],'existing accepted namespace claim changed')
 need(sha(read(C['original_registry']))==C['original_namespace_claim_sha256'],'preserved original namespace changed')
 cpu_root=pathlib.Path(C['cpu_root'])
 need(cpu_root.is_dir() and cpu_root==cpu_root.resolve(),'original CPU root alias/missing')
 members=list(cpu_root.rglob('*'))
 need(len(members)<=262144 and not any(p.is_symlink() for p in members),'bounded ordinary CPU custody required')
 need({str(p.relative_to(cpu_root)) for p in members if p.is_file()}==set(C['cpu_roster']),'complete original CPU custody roster changed')
 for base,expected_dirs in C['history_directories'].items():
  tree=pathlib.Path(base);need(tree.is_dir() and not tree.is_symlink(),'original upstream root missing')
  entries=list(tree.rglob('*'));need(not any(p.is_symlink() for p in entries),'upstream symlink')
  need(sorted(p.relative_to(tree).as_posix() for p in entries if p.is_dir())==expected_dirs,'whole original upstream directories changed')
  expected={n[len(base)+1:]:h for n,h in C['cpu_inputs'].items() if n.startswith(base+'/')}
  need({p.relative_to(tree).as_posix():sha(read(p,allow_hardlink=True)) for p in entries if p.is_file()}==expected,'whole original upstream files changed')
 for path,h in C['cpu_inputs'].items():
  need(sha(read(path,allow_hardlink=True))==h,'accepted CPU input changed: '+path)
 model=pathlib.Path(C['model_directory'])
 need(model==model.resolve() and model.is_dir(),'frozen model directory alias')
 model_verified={}
 for name,expected in C['model_files'].items():
  path=model/name
  need(path.name==name and not path.is_symlink(),'ordinary exact model member required')
  h=hashlib.sha256()
  with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
   before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode),'regular model bytes required')
   for block in iter(lambda:f.read(1024**2),b''):h.update(block)
   after=os.fstat(f.fileno())
  need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
       and h.hexdigest()==expected,'frozen model bytes differ')
  model_verified[name]=expected
 return dict(utc=utc(),queue=queue,related_accounting=rows,journal_pins=journals,
             partition=partition,association=assoc,qos=qos,gpu_types=gpu_types,model_files=model_verified,manifest_sha256=C['manifest']['sha256'],
             archive_size=st.st_size,archive_byte_check='allocated stage authenticates full bytes before execution',
             existing_namespace_claim_sha256=C['namespace_claim_sha256'],
             original_namespace_claim_sha256=C['original_namespace_claim_sha256'],
             closed_cpu_seconds=C['baseline']['closed_cpu_seconds'],closed_gpu_slice_seconds=C['baseline']['closed_gpu_slice_seconds'],
             cpu_reserved_total=C['baseline']['closed_cpu_seconds']+15120,gpu_reserved_total=C['baseline']['closed_gpu_slice_seconds']+2880)
def verify():
 for name,h in C['pins'].items():need(sha(read(root/name))==h,'remote pin differs: '+name)
 return dict(utc=utc(),verified=C['pins'])
