"""One new fabricated nested-handler fixture; no signal, main or remote calls."""
from pathlib import Path
import ast,copy,unittest
HERE=Path(__file__).resolve().parent

class DeadlinePropagation(unittest.TestCase):
 def test_deadline_stops_next_input_and_retains_outer_failure_both_collectors(self):
  for stage in ('gpu','finalizer'):
   t=ast.parse((HERE/stage/'pull_terminal.py').read_bytes())
   source=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='REMOTE_OBSERVATION')
   t=ast.parse(source);deadline=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='deadline')
   outer=next(n for n in t.body if isinstance(n,ast.Try) and any(isinstance(h.type,ast.Name) and h.type.id=='BaseException' for h in n.handlers))
   loop=next(n for n in outer.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='UPSTREAM_PINS')
   captured=ast.Module(body=[copy.deepcopy(deadline),ast.Try(body=[copy.deepcopy(loop)],handlers=copy.deepcopy(outer.handlers),orelse=[],finalbody=[])],type_ignores=[])
   code=compile(ast.fix_missing_locations(captured),'captured-deadline-input-loop','exec')
   for failure in ('deadline','ordinary_oserror','ordinary_valueerror','none'):
    with self.subTest(stage=stage,failure=failure):
     calls=[];v=dict(terminal_gate=True,upstream_pins={},upstream_errors={})
     ns=dict(v=v,UPSTREAM_PINS={'first':'a'*64,'next':'b'*64},Path=Path)
     def fingerprint(path):
      calls.append(str(path))
      if str(path)=='first':
       if failure=='deadline':ns['deadline'](None,None)
       if failure=='ordinary_oserror':raise OSError('fabricated missing input')
       if failure=='ordinary_valueerror':raise ValueError('fabricated malformed input')
      return {'sha256':ns['UPSTREAM_PINS'][str(path)]},None
     ns['fingerprint']=fingerprint;exec(code,ns)
     if failure=='deadline':
      self.assertEqual(calls,['first']);self.assertEqual(v['upstream_errors'],{});self.assertEqual(v['upstream_pins'],{})
      self.assertEqual(v['observation_error'],{'type':'TimeoutError','message':'bounded full terminal observation'})
     else:
      self.assertEqual(calls,['first','next']);self.assertNotIn('observation_error',v)
      self.assertEqual(v['upstream_pins']['next'],'b'*64)
      self.assertEqual(set(v['upstream_errors']),set() if failure=='none' else {'first'})

if __name__=='__main__':unittest.main(verbosity=2)
