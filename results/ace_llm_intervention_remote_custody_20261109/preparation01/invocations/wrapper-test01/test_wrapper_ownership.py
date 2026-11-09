import ast,fcntl,json,pathlib,tempfile,types,unittest,os
BASE=pathlib.Path(__file__).parent
class OwnershipTests(unittest.TestCase):
 def run_case(self,name,case):
  source=(BASE/name).read_text();tree=ast.parse(source);body=[n for n in tree.body if isinstance(n,ast.Try)];self.assertEqual(len(body),1)
  root=pathlib.Path(tempfile.mkdtemp(prefix='ace-wrapper-ownership-'));out=root/'output'
  for lock in ('ace-delivery-controller.lock','ace-foundation-controller.lock'):(root/lock).write_text('')
  if case!='owned':out.mkdir();(out/'sentinel').write_text('preserve')
  def write(path,data):
   with path.open('xb') as f:f.write(data)
  fc=fcntl
  if case=='lock':
   def fail(*a):raise BlockingIOError('simulated occupied controller lock')
   fc=types.SimpleNamespace(flock=fail,LOCK_EX=fcntl.LOCK_EX,LOCK_NB=fcntl.LOCK_NB,LOCK_UN=fcntl.LOCK_UN)
  env={'BASE':root,'out':out,'owned':False,'locks':[],'fcntl':fc,'write':write,'raw':lambda v:(json.dumps(v)+'\n').encode(),'Path':pathlib.Path,'os':os}
  expected=NameError if case=='owned' else BlockingIOError if case=='lock' else FileExistsError
  with self.assertRaises(expected):exec(compile(ast.Module(body=body,type_ignores=[]),name,'exec'),env)
  if case=='owned':self.assertTrue((out/'failure.json').is_file());self.assertTrue(json.loads((out/'failure.json').read_text())['partial_preserved'])
  else:self.assertEqual(sorted(p.name for p in out.iterdir()),['sentinel']);self.assertEqual((out/'sentinel').read_text(),'preserve')
  self.assertIn("'python3 -I -B -'",source);self.assertNotIn("'python3 -'",source)
 def test_existing_capsules_untouched(self):
  for name in ('backup_training_once.py','backup_static_inputs_once.py'):
   with self.subTest(name=name):self.run_case(name,'existing')
 def test_lock_failure_preserves_previous_capsule(self):
  for name in ('backup_training_once.py','backup_static_inputs_once.py'):
   with self.subTest(name=name):self.run_case(name,'lock')
 def test_owned_failure_is_retained(self):
  for name in ('backup_training_once.py','backup_static_inputs_once.py'):
   with self.subTest(name=name):self.run_case(name,'owned')
if __name__=='__main__':unittest.main()
