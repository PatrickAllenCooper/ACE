"""Only new interrupt-boundary cases, using fabricated metadata and isolated with AST."""
import ast
from pathlib import Path
import tempfile
import types
import unittest
import new_boundary_fixture as F

class Interrupt(BaseException):
    pass

class BoundaryTests(unittest.TestCase):
    def test_assignment_boundary_restores_on_baseexception(self):
        with tempfile.TemporaryDirectory(prefix='ace-entry-boundary-') as tmp:
            f=F.Fixture(tmp); h=f.capture(); original=f.module._load_attempt
            primary=Interrupt('injected immediately after hook assignment')
            class InterruptingModule(types.ModuleType):
                def __setattr__(self,name,value):
                    super().__setattr__(name,value)
                    if name=='_load_attempt' and value is not original and self.__dict__.get('armed'):
                        self.__dict__['armed']=False
                        raise primary
            f.module.__class__=InterruptingModule
            f.module.armed=True
            with self.assertRaises(Interrupt) as caught:
                with h.install(f.module):
                    self.fail('body must not be reached')
            self.assertIs(caught.exception,primary)
            self.assertIs(f.module._load_attempt,original)

    def test_actual_entry_with_body_exception_preserves_identity(self):
        source=Path(__file__).with_name('qualify_llm_intervention_scientific_campaign_gpu.py').read_bytes()
        tree=ast.parse(source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_controller')
        scopes=[n for n in ast.walk(run) if isinstance(n,ast.With) and ast.unparse(n.items[0].context_expr)=='historical.install(campaign)']
        self.assertEqual(len(scopes),1)
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('__enter__','__exit__') for n in ast.walk(run)))
        outer=next(n for n in run.body if isinstance(n,ast.Try))
        self.assertEqual(ast.unparse(outer.finalbody[0]),'sys.meta_path.remove(loader)')
        scope=scopes[0]; scope.body=ast.parse('raise primary').body
        module=ast.fix_missing_locations(ast.Module(body=[scope],type_ignores=[]))
        with tempfile.TemporaryDirectory(prefix='ace-entry-with-') as tmp:
            f=F.Fixture(tmp);h=f.capture();original=f.module._load_attempt;primary=Interrupt('body')
            with self.assertRaises(Interrupt) as caught:
                exec(compile(module,'<isolated-new-with-boundary>','exec'),dict(historical=h,campaign=f.module,primary=primary))
            self.assertIs(caught.exception,primary)
            self.assertIs(f.module._load_attempt,original)

if __name__=='__main__':
    unittest.main()
