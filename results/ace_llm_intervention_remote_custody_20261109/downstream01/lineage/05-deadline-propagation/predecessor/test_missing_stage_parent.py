"""One new fabricated path fixture; no builder main, project or remote execution."""
from pathlib import Path
import ast,copy,os,tempfile,unittest
HERE=Path(__file__).resolve().parent

def need(ok,message):
 if not ok:raise ValueError(message)

class MissingParent(unittest.TestCase):
 def test_both_builders_create_stage_parent_keep_exclusive_leaves_and_reject_alias(self):
  for stage in ('gpu','finalizer'):
   source=(HERE/stage/'build_packet.py').read_bytes();tree=ast.parse(source)
   main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   scope=next(n for n in main.body if isinstance(n,ast.With))
   first=next(i for i,n in enumerate(scope.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='preparations' for t in n.targets))
   last=next(i for i,n in enumerate(scope.body[first:],first) if isinstance(n,ast.For) and ast.unparse(n.iter)=='preparations')
   code=compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(scope.body[first:last+1]),type_ignores=[])),stage+'-preparation-prefix','exec')
   for case in ('missing','existing-leaf','aliased-parent','file-parent'):
    with self.subTest(stage=stage,case=case),tempfile.TemporaryDirectory(prefix='fake-missing-stage-parent-') as temp:
     root=Path(temp).resolve();results=root/'repo/results';results.mkdir(parents=True);base=root/'base';base.mkdir()
     local=results/'new-stage/launch01';external=base/'new-stage-launch01'
     extprep=external.with_name(external.name+'-storage-build-preparation');localprep=local.parent/'storage-build-preparation01'
     if case=='existing-leaf':extprep.mkdir();(extprep/'original.txt').write_bytes(b'original')
     if case=='aliased-parent':
      sink=root/'alias-target';sink.mkdir();local.parent.symlink_to(sink,target_is_directory=True)
     if case=='file-parent':local.parent.write_bytes(b'not-directory')
     env=dict(LOCAL=local,EXT=external,need=need,os=os)
     if case=='missing':
      self.assertFalse(local.parent.exists());exec(code,env)
      self.assertTrue(extprep.is_dir());self.assertTrue(localprep.is_dir());self.assertFalse(local.exists());self.assertFalse(external.exists())
      (extprep/'original.txt').write_bytes(b'original')
      with self.assertRaisesRegex(ValueError,'never retry'):exec(code,env)
      self.assertEqual((extprep/'original.txt').read_bytes(),b'original')
     else:
      with self.assertRaises((ValueError,FileExistsError)):exec(code,env)
      if case=='existing-leaf':
       self.assertEqual((extprep/'original.txt').read_bytes(),b'original');self.assertFalse(local.parent.exists())
      else:self.assertFalse(extprep.exists())
      self.assertFalse(localprep.exists())

if __name__=='__main__':unittest.main(verbosity=2)
