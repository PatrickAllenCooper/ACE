"""New fabricated metadata-only deadline/progress checks, no remote or main."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,os,pathlib,stat,subprocess,sys,tempfile,types,unittest
HERE=Path(__file__).resolve().parent

def remote_source(stage,kind,base=HERE):
 tree=ast.parse((base/stage/kind).read_bytes());name='REMOTE_COMMON' if kind=='launch_once.py' else 'REMOTE_OBSERVATION'
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)

def function(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
def need(ok,message):
 if not ok:raise ValueError(message)

class MetadataDeadline(unittest.TestCase):
 def test_progress_every_thousand_validated_reads_never_payload_or_path(self):
  with tempfile.TemporaryDirectory(prefix='fake-metadata-progress-') as tmp:
   root=Path(tmp).resolve();path=root/'OPAQUE-PATH-NOT-FOR-STDERR.json';payload=b'OPAQUE-PAYLOAD-NOT-FOR-STDERR';path.write_bytes(payload)
   alias=root/'alias';alias.symlink_to(path)
   for stage in ('gpu','finalizer'):
    for kind in ('launch_once.py','pull_terminal.py'):
     with self.subTest(stage=stage,kind=kind):
      tree=ast.parse(remote_source(stage,kind));name='read' if kind=='launch_once.py' else 'fingerprint'
      ns=dict(Path=Path,pathlib=pathlib,os=os,stat=stat,hashlib=hashlib,json=json,sys=sys,
       time=types.SimpleNamespace(monotonic=lambda:11.5),_READ_COUNT=999,_READ_STARTED=10)
      exec(compile(ast.Module(body=[function(tree,'need'),function(tree,name)],type_ignores=[]),'captured-metadata-read-only','exec'),ns)
      output=io.StringIO();stdout=io.StringIO()
      with contextlib.redirect_stderr(output),contextlib.redirect_stdout(stdout):
       result=ns[name](path);ns[name](path);ns['_READ_COUNT']=1999;ns[name](path)
       with self.assertRaises((ValueError,RuntimeError)):ns[name](alias)
      events=[json.loads(v) for v in output.getvalue().splitlines()]
      self.assertEqual([v['files_read'] for v in events],[1000,2000]);self.assertEqual(ns['_READ_COUNT'],2000)
      self.assertTrue(all(v['elapsed_seconds']==1.5 for v in events));self.assertEqual(stdout.getvalue(),'')
      self.assertNotIn(str(path),output.getvalue());self.assertNotIn(payload.decode(),output.getvalue())
      if kind=='launch_once.py':self.assertEqual(result,payload)
      else:self.assertEqual(result[0],{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
 def test_absolute_alarm_and_transport_bounds_keep_partial_progress(self):
  for stage in ('gpu','finalizer'):
   for kind in ('launch_once.py','pull_terminal.py'):
    with self.subTest(stage=stage,kind=kind):
     remote=ast.parse(remote_source(stage,kind))
     alarms=[n for n in ast.walk(remote) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='signal.alarm']
     self.assertEqual(len(alarms),1);self.assertEqual(ast.literal_eval(alarms[0].args[0]),1200)
     deadline=function(remote,'remote_deadline' if kind=='launch_once.py' else 'deadline')
     ns={};exec(compile(ast.Module(body=[deadline],type_ignores=[]),'captured-timeout-handler','exec'),ns)
     with self.assertRaises(TimeoutError):ns[deadline.name](None,None)
     tree=ast.parse((HERE/stage/kind).read_bytes());records=[];calls=[];fail=[False]
     def run(command,**kwargs):
      calls.append(kwargs)
      if fail[0]:raise subprocess.TimeoutExpired(command,1230,output=b'partial stdout',stderr=b'progress count 1000\n')
      return types.SimpleNamespace(returncode=0,stdout='{"terminal_gate":true}',stderr='progress count 1000\n')
     env=dict(subprocess=types.SimpleNamespace(run=run,TimeoutExpired=subprocess.TimeoutExpired),
      save=lambda n,v:records.append((n,v)),b=types.SimpleNamespace(decode=json.loads,utc=lambda:'FAKE',need=need),
      decode=json.loads,utc=lambda:'FAKE',need=need,HELPER='FAKE-NO-REMOTE',source='fake captured metadata source')
     names=['transport','remote'] if kind=='launch_once.py' else (['run_saved','observe'] if stage=='gpu' else ['observe'])
     nodes=[function(tree,name) for name in names]
     exec(compile(ast.Module(body=nodes,type_ignores=[]),'captured-local-transport-boundary','exec'),env)
     action=(lambda:env['remote']('fake.json','fake source')) if kind=='launch_once.py' else (lambda:env['observe']('fake.json'))
     self.assertTrue(action()['terminal_gate']);self.assertEqual(calls[-1]['timeout'],1230)
     fail[0]=True
     with self.assertRaises(ValueError):action()
     self.assertTrue(records[-1][1]['timeout']);self.assertEqual(records[-1][1]['stderr'],'progress count 1000\n')
     self.assertEqual(records[-1][1]['stdout'],'partial stdout');self.assertEqual(calls[-1]['timeout'],1230)
 def test_scientific_and_existing_source_gates_remain_unchanged(self):
  previous=HERE.parent/'predecessor'
  defs=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
  for stage in ('gpu','finalizer'):
   for name in ('build_packet.py','monitor_once.py','compact_inputs.py'):
    self.assertEqual((HERE/stage/name).read_bytes(),(previous/stage/name).read_bytes())
   for kind in ('launch_once.py','pull_terminal.py'):
    old=defs((previous/stage/kind).read_text());new=defs((HERE/stage/kind).read_text())
    self.assertEqual(set(old),set(new))
    for name in old:
     if name!='main':self.assertEqual(old[name],new[name],(stage,kind,name))
    old=defs(remote_source(stage,kind,previous));new=defs(remote_source(stage,kind))
    for name in old:
     if name not in ('read','fingerprint'):self.assertEqual(old[name],new[name],(stage,kind,name))
    read_name='read' if kind=='launch_once.py' else 'fingerprint'
    oldfn=function(ast.parse(remote_source(stage,kind,previous)),read_name)
    newfn=function(ast.parse(remote_source(stage,kind)),read_name)
    # Only count/progress statements are allowed around the unchanged authenticated read.
    def progress(n):
     return isinstance(n,ast.Global) or (isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='_READ_COUNT') or (isinstance(n,ast.If) and '_READ_COUNT' in ast.unparse(n.test))
    newfn.body=[n for n in newfn.body if not progress(n)]
    self.assertEqual(ast.dump(oldfn,include_attributes=False),ast.dump(newfn,include_attributes=False))

if __name__=='__main__':unittest.main(verbosity=2)
