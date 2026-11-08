"""New failure-unwind integration only, using fabricated parent/owner objects."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
import supervise_llm_intervention_scientific_cpu as stages
import prepare_llm_intervention_scientific_training_cpu as tokens

class EarlyCpuLifecycleTests(unittest.TestCase):
    def exercise(self,module,mode):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();events=[]
            args=types.SimpleNamespace(packet=str(root/'input/packet.json'),packet_sha256='a'*64,source_commit='a'*40,
                admission=str(root/'input/admission.json'),admission_sha256='b'*64,output=str(root/'output'),
                worker=False,handoff_sha256=None,parent_started=None,runtime_stage=None,runtime_stage_sha256=None)
            p=dict(mode=mode,output=args.output,resources={},deadlines=dict(startup=120,work=260,cleanup=280,hard=290),
                runtime_source=str(root/'runtime'),model_directory=str(root/'model'),archive=dict(path=str(root/'archive')),
                manifest=dict(path=str(root/'manifest')),collection=dict(root=str(root/'collection')))
            owner=types.SimpleNamespace(errors=[],hold_until_closed=lambda:events.append('hold'))
            owner.error=lambda *a:owner.errors.append(a)
            def parent(a,ctx):
                a.owned_child=owner
                raise ValueError('original failure')
            def timer(which,value):
                if value==0:
                    self.assertEqual(events[-1],'hold')
                    events.append('cancel')
            reader=module.B if module is stages else module
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(module.argparse.ArgumentParser,'parse_args',return_value=args))
                stack.enter_context(patch.object(reader,'read',return_value=json.dumps(p).encode()))
                stack.enter_context(patch.object(module,'resource_contract'))
                stack.enter_context(patch.object(module,'authenticate',return_value={}))
                stack.enter_context(patch.object(module,'parent',side_effect=parent))
                stack.enter_context(patch.object(module,'preserve_failure' if module is stages else 'failure',side_effect=OSError('receipt denied')))
                stack.enter_context(patch.object(module.signal,'signal'))
                stack.enter_context(patch.object(module.signal,'setitimer',side_effect=timer))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                with self.assertRaisesRegex(ValueError,'original failure'):module.main()
            self.assertEqual(events,['hold','cancel'])
            self.assertEqual(owner.errors[0][0],'failure_publication')

    def test_realization_collection_main_preserves_owned_handle(self):
        self.exercise(stages,'realize')
        self.exercise(stages,'collect')

    def test_token_preparation_main_preserves_owned_handle(self):
        self.exercise(tokens,'tokens')

    def test_only_parent_lifecycle_and_independent_helper_pin_change(self):
        here=Path(__file__).parent
        for module in (stages,tokens):
            self.assertEqual(module.HELPER_PINS[module.OWNED],'1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8')
            tree=ast.parse(Path(module.__file__).read_bytes())
            parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parent')
            calls=[ast.unparse(n.func) for n in ast.walk(parent) if isinstance(n,ast.Call)]
            self.assertIn('ownership.supervise',calls)
            self.assertNotIn('B.supervise',calls)
            self.assertNotIn('f.supervise',calls)
            self.assertNotIn('setattr',calls)
        self.assertNotIn('torch',__import__('sys').modules)
        self.assertNotIn('numpy',__import__('sys').modules)

if __name__=='__main__':unittest.main()
