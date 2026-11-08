from pathlib import Path
import ast,copy,hashlib,json,subprocess,time,types
root=Path(__file__).resolve().parent
start=time.monotonic();cpu=time.process_time()
launch=ast.parse((root/'launch_once.py').read_bytes());monitor=ast.parse((root/'monitor_once.py').read_bytes())
def named(h):return isinstance(h.type,ast.Attribute) and h.type.attr=='TimeoutExpired'
lh=next(h for h in ast.walk(launch) if isinstance(h,ast.ExceptHandler) and named(h) and 'transfer-' in ast.unparse(h))
mh=next(h for h in ast.walk(monitor) if isinstance(h,ast.ExceptHandler) and named(h))
remote=next(n.value for n in ast.walk(monitor) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'observations=[]' in n.value)
rh=next(h for h in ast.walk(ast.parse(remote)) if isinstance(h,ast.ExceptHandler) and named(h))
count=0
for label,h in [('scp',lh),('local_monitor',mh),('remote_monitor',rh)]:
 for output,error in [(b'partial out',b'partial err'),('partial out','partial err'),(None,None)]:
  records=[];files={};exc=subprocess.TimeoutExpired(['fabricated'],1,output=output,stderr=error)
  env=dict(subprocess=subprocess,test_exc=exc,group='inputs',command=['fake'],observations=records,
   save=lambda n,v:records.append((n,v)),destination=Path('/fabricated'),HELPER='fake',job='1',
   a=types.SimpleNamespace(packet_sha256='a'*64,submission_result_sha256='b'*64),
   b=types.SimpleNamespace(write=lambda p,v:files.update({p.name:v}),raw=lambda v:json.dumps(v).encode()))
  body=ast.Try(body=[ast.Raise(exc=ast.Name(id='test_exc',ctx=ast.Load()),cause=None)],handlers=[copy.deepcopy(h)],orelse=[],finalbody=[])
  try:exec(compile(ast.fix_missing_locations(ast.Module(body=[body],type_ignores=[])),'<captured timeout handler>','exec'),env)
  except subprocess.TimeoutExpired as got:assert got is exc
  if label=='local_monitor':
   assert files['stdout.log']==(output if isinstance(output,bytes) else (output or '').encode())
   assert files['stderr.log']==(error if isinstance(error,bytes) else (error or '').encode())
   assert json.loads(files['failure.json'])['retry_submission'] is False
  else:
   row=records[0][1] if label=='scp' else records[0]
   assert row['stdout']==(output.decode() if isinstance(output,bytes) else output)
   assert row['stderr']==(error.decode() if isinstance(error,bytes) else error)
   assert row['timeout'] is True and row['command']==['fake']
  count+=1
result=dict(cases=count,scope='three newly corrected timeout handlers, bytes/text/absent output; isolated AST only, no main or remote command executed',elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,files={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['launch_once.py','monitor_once.py','check_timeout_delta.py']})
(root/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
