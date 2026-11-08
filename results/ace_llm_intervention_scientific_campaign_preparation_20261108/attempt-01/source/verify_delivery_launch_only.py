"""Darwin launch-only verification; real manifest, explicit platform/qualifier stubs."""
import argparse,base64,ctypes,fcntl,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
PIN='b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9'
SIZE=2628435

def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

STUB=r'''
def main():
 import base64,hashlib,json,os,sys,time
 c=vars(sys.modules['__main__']);fd=__output_dir_fd__;a=json.loads(c['payload_raw'])['arguments'];raw=base64.b64decode(a[7])
 assert len(raw)==2628435 and hashlib.sha256(raw).hexdigest()=='b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9'
 assert not any(n.split('.')[0] in {'torch','numpy','scipy','pandas','sympy','yaml'} for n in sys.modules)
 assert globals()['__executed_source_sha256__']==hashlib.sha256(base64.b64decode(a[5])).hexdigest()
 assert os.fstat(fd).st_ino==c['reg']['output_inode']
 try:os.fstat(c['payload_fd'])
 except OSError as e:assert e.errno==9
 else:raise ValueError('child payload FD leaked')
 d={'stage':'qualification_entry_LAUNCH_ONLY_STUB','manifest_bytes':len(raw),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'output_fd':fd,'output_inode':os.fstat(fd).st_ino,'payload_fd_closed':True,'target_dependencies_imported':False,'scientific_payload_executed':False,'runtime_qualified':False,'new_fits':0,'new_responses':0,'cwd':os.getcwd(),'source_sha256':[hashlib.sha256(base64.b64decode(a[i])).hexdigest() for i in (3,4,5,6)],'pid':os.getpid(),'pgid':os.getpgrp()}
 mode=c['reg']['fixture_mode']
 if mode=='descendant':
  pid=os.fork()
  if pid==0:
   pipe=os.open('descendant-sentinel',os.O_WRONLY,dir_fd=fd);os.write(pipe,b'ready');time.sleep(30);os._exit(0)
  d['descendant_pid']=pid;time.sleep(.08)
 h=os.open('launch_only_receipt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
 with os.fdopen(h,'w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if mode=='child_error':raise ValueError('deliberate launch-only child error')
 raise SystemExit(0) # STOP before actual dependencies, guard, verifier or CLI.
'''

def prefix(root,fd,seals):
 return '''import fcntl,os,pathlib,sys
fcntl.F_SEAL_WRITE=8;fcntl.F_SEAL_GROW=4;fcntl.F_SEAL_SHRINK=2;fcntl.F_SEAL_SEAL=1;fcntl.F_GET_SEALS=1034
_actual_fcntl=fcntl.fcntl
fcntl.fcntl=lambda fd,op,*args: __FIXTURE_SEAL_MASK__ if op==1034 else _actual_fcntl(fd,op,*args)
_path_new=pathlib.Path.__new__
def _darwin_path(cls,*args,**kwargs):
 return _path_new(cls,*[(ROOT+str(x)[len(LOGICAL):]) if str(x).startswith(LOGICAL) else x for x in args],**kwargs)
pathlib.Path.__new__=staticmethod(_darwin_path)
def _deny_science(event,args):
 if event=='import' and args[0].split('.')[0] in {'torch','numpy','scipy','pandas','sympy','yaml'}:raise PermissionError('launch-only blocks target imports')
 if event=='open' and isinstance(args[0],str) and args[0].endswith(('.pt','.npz','.npy')):raise PermissionError('launch-only blocks checkpoints/response arrays')
 if event.startswith('socket.'):raise PermissionError('launch-only blocks network')
sys.addaudithook(_deny_science)
'''.replace('__FIXTURE_SEAL_MASK__',str(seals)).replace('ROOT',repr(str(root))).replace('LOGICAL',repr('/proc/self/fd/'+str(fd)))

def children():
 probe=subprocess.Popen(['ps','-axo','pid=,ppid='],stdout=subprocess.PIPE,text=True)
 try:raw=probe.communicate(timeout=2)[0]
 except BaseException as primary:
  cleanup=[]
  try:probe.kill()
  except ProcessLookupError:pass
  except BaseException as error:cleanup.append('kill: '+repr(error))
  try:probe.communicate(timeout=2)
  except BaseException as error:cleanup.append('reap: '+repr(error))
  if cleanup:primary.add_note('owned ps cleanup failures: '+repr(cleanup))
  raise
 return {int(r.split()[0]) for r in raw.splitlines() if len(r.split())==2 and int(r.split()[1])==os.getpid() and int(r.split()[0])!=probe.pid}
def rss(pid):
 raw=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True,timeout=2);rows=[tuple(map(int,r.split())) for r in raw.splitlines() if len(r.split())==3];tree={pid}
 for _ in rows:
  nxt=tree|{p for p,parent,_ in rows if parent in tree}
  if nxt==tree:break
  tree=nxt
 return sum(n*1024 for p,_,n in rows if p in tree)

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
 if sys.platform!='darwin' or not a.package.is_absolute() or not a.destination.is_absolute():raise ValueError('Darwin and absolute paths required')
 if a.manifest_sha256!=PIN:raise ValueError('independently pinned actual manifest required')
 a.destination.mkdir(mode=0o700);root=a.destination.resolve();package=a.package.resolve()
 w=load('wrapper',Path(__file__).with_name('supervise_delivery_qualified_replay.py'));s=load('supervisor',Path(__file__).with_name('delivery_replay_supervision.py'))
 raw=w.read(package/'manifest.json',PIN);assert len(raw)==SIZE;m=json.loads(raw);entries={e['path']:e for e in m['files']};assert len(entries)==len(m['files'])==3893
 cli=w.read(package/'replay_delivery_prospective_release.py',entries['replay_delivery_prospective_release.py']['sha256']);verifier=w.read(package/'verify_delivery_release.py',entries['verify_delivery_release.py']['sha256'])
 production={'qualify_runtime.py':Path(__file__).with_name('qualify_delivery_replay_runtime.py').read_bytes(),'checkpoint_guard.py':Path(__file__).with_name('delivery_replay_checkpoint_guard.py').read_bytes(),'child_bootstrap.py':w.CHILD_BOOTSTRAP.encode()}
 env=w.child_environment();fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);seal_calls=[];cases=[]
 def memfd(name,flags):
  h,path=tempfile.mkstemp(prefix='payload-',dir=root);os.unlink(path);seal_calls.append({'name':name,'flags':flags,'fd':h,'inode':os.fstat(h).st_ino});return h # real anonymous unlinked file, NOT memfd
 def fixture_fcntl(h,op,*values):seal_calls.append({'fd':h,'op':op,'values':list(values)});return 15
 libc=type('FixtureLibc',(),{'prctl':lambda *_:0})()
 with patch.object(w.os,'memfd_create',memfd,create=True),patch.object(w.os,'MFD_CLOEXEC',1,create=True),patch.object(w.os,'MFD_ALLOW_SEALING',2,create=True),patch.object(w.fcntl,'F_SEAL_WRITE',8,create=True),patch.object(w.fcntl,'F_SEAL_GROW',4,create=True),patch.object(w.fcntl,'F_SEAL_SHRINK',2,create=True),patch.object(w.fcntl,'F_SEAL_SEAL',1,create=True),patch.object(w.fcntl,'F_ADD_SEALS',1033,create=True),patch.object(w.fcntl,'F_GET_SEALS',1034,create=True),patch.object(w.fcntl,'fcntl',fixture_fcntl),patch.object(s.ctypes,'CDLL',return_value=libc),patch.object(s,'direct_children',side_effect=children):
  # Exact actual source capture and construction; NEVER launch this payload.
  cmd,payload,info=w.build_child_launch(package,PIN,fd,production,cli,verifier,raw,'0'*64,env)
  with os.fdopen(os.dup(payload),'rb') as f:captured=f.read()
  assert sha(captured)==info['sha256'];v=json.loads(captured)['arguments']
  for i,expected in [(3,cli),(4,verifier),(5,production['qualify_runtime.py']),(6,production['checkpoint_guard.py']),(7,raw)]:assert base64.b64decode(v[i])==expected
  write(root/'actual_capture.json',dict(info,actual_transport='Darwin anonymous unlinked regular file; Linux seals mocked',executed=False,loader_sha256=sha(cmd[4].encode()),argv_entries=len(cmd),descriptor_inode=os.fstat(payload).st_ino));os.close(payload)
  for mode in ['success','child_error','payload_changed','unsealed','missing_payload_fd','freeze_changed','descendant']:
   case=root/mode;case.mkdir(mode=0o700);outfd=os.open(case,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);sentinel=None;payload=None
   try:
    if mode=='descendant':os.mkfifo(case/'descendant-sentinel');sentinel=os.open(case/'descendant-sentinel',os.O_RDONLY|os.O_NONBLOCK)
    bootstrap=prefix(case,outfd,0 if mode=='unsealed' else 15).encode()+w.CHILD_BOOTSTRAP.encode();captures={**production,'child_bootstrap.py':bootstrap,'qualify_runtime.py':STUB.encode()}
    for name,source in captures.items():(case/name).write_bytes(source)
    reg={'output_inode':os.fstat(outfd).st_ino,'fixture_mode':mode,'qualification_worker_sha256':sha(STUB.encode()),'execution_source_pins':{'child_bootstrap.py':sha(bootstrap)}};freeze=(json.dumps(reg,indent=2)+'\n').encode();(case/'freeze.json').write_bytes(freeze)
    cmd,payload,info=w.build_child_launch(package,PIN,outfd,captures,cli,verifier,raw,sha(freeze),env);assert cmd[4]==w.CHILD_LOADER
    if mode=='payload_changed':os.write(payload,b'!');os.lseek(payload,0,os.SEEK_SET)
    if mode=='freeze_changed':(case/'freeze.json').write_bytes(b'{}')
    inherited=() if mode=='missing_payload_fd' else (payload,);start=time.monotonic();result=s.supervise(cmd,start+5,128*2**20,outfd,'child.log',env,rss,input_fds=inherited)
    oldfd=payload;os.close(payload);payload=None
    try:os.fstat(oldfd)
    except OSError as e:assert e.errno==9
    else:raise ValueError('parent payload FD leaked')
    receipt=(case/'launch_only_receipt.json').read_bytes() if (case/'launch_only_receipt.json').exists() else None
    if mode in ('success','descendant'):assert result['status']=='complete' and result['exit_code']==0 and result['cleanup']['complete'] is True and receipt,result
    else:
     assert result['exit_code']!=0 and result['cleanup']['complete'] is True,result
     if mode!='child_error':assert receipt is None
    assert not any((case/n).exists() for n in ['runtime_qualification.json','runtime_inventory.json','replay_receipt.json','pre_checkpoint_runtime.json'])
    descendant=None
    if mode=='descendant':
     ready=os.read(sentinel,4096);assert ready==b'ready' and os.read(sentinel,4096)==b''
     pid=json.loads(receipt)['descendant_pid'];ps=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,timeout=2);assert not ps.stdout.strip() or ps.stdout.strip().startswith('Z'),ps.stdout
     descendant={'pid':pid,'sentinel_ready':True,'writer_closed':True,'observed_stat':ps.stdout.strip(),'Linux_subreaping_proven':False}
    evidence={'mode':mode,'input':info,'actual_transport':'Darwin anonymous unlinked regular file; Linux seals mocked','command_shape':{'entries':len(cmd),'flags':cmd[1:4],'loader_sha256':sha(cmd[4].encode()),'source_argument_sha256':sha(cmd[5].encode()),'pass_fds':[outfd,*inherited]},'actual_supervision_result':result,'elapsed_seconds':time.monotonic()-start,'parent_payload_closed':True,'output_inode':os.fstat(outfd).st_ino,'descendant':descendant,'cwd':os.getcwd(),'actual_qualified':False};write(case/'evidence.json',evidence);cases.append(evidence)
   finally:
    if payload is not None:os.close(payload)
    if sentinel is not None:os.close(sentinel)
    os.close(outfd)
 os.close(fd)
 result={'scope':'Darwin actual-manifest launch-only; no allocation/science','manifest_sha256':PIN,'manifest_bytes':SIZE,'source_pins':{n:sha(Path(__file__).with_name(n).read_bytes()) for n in ['supervise_delivery_qualified_replay.py','delivery_replay_supervision.py','qualify_delivery_replay_runtime.py','delivery_replay_checkpoint_guard.py','verify_delivery_launch_only.py']},'cases':cases,'seal_calls_mocked':seal_calls,'substitutions':['Linux memfd/seals -> real anonymous unlinked file, mocked seal results','Linux prctl -> mock; directchildren/RSS -> actual Darwin ps','/proc outputpathname -> explicit Darwin projection','qualifier -> safe launch-only stub exiting before dependencies/guard/verifier/CLI','bootstrap -> declared platform preamble plus unchanged production body'],'covered':'actual source/input capture/shared constructor and loader/Popen/pass_fds/inherited byte identity and closure/error propagation/real process-group cleanup','remaining_unqualified':['actual Linux seals/subreaping/proc','Slurm/interpreter/allocation gates','six dependencies/native module origins','checkpoint/statistical replay and final scientific receipt acceptance'],'new_fits':0,'new_optimizer_updates':0,'new_responses':0,'new_allocation':False,'cwd':os.getcwd(),'python':sys.executable,'platform':sys.platform,'actual_runtime_qualified':False,'scientific_payload_executed':False};write(root/'receipt.json',result);print(json.dumps({'cases':len(cases),'manifest_bytes':SIZE,'manifest_sha256':PIN,'actual_runtime_qualified':False,'cwd':os.getcwd()}))

if __name__=='__main__':main()
