"""New fabricated constant-constructor cases only; no actual GPU audit."""
from pathlib import Path
import hashlib
import types
import unittest

P = Path(__file__).resolve().parent / 'dependencies/audit_supplement.py'
data = P.read_bytes()
assert hashlib.sha256(data).hexdigest() == 'e2da9858fe137c653a96a3650ec0377f4d4d9721891dda73fe5727cddb292d13'
parser = types.ModuleType('focused_literal_parser')
parser.__file__ = str(P)
exec(compile(data, str(P), 'exec'), parser.__dict__)


class ConstructorCases(unittest.TestCase):
    def test_exact_frozen_construction_shape_without_source_execution(self):
        source = """raise AssertionError('fabricated source must not execute')
RESOURCES=dict(account='fixture',partition='fixture',qos='fixture',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,
 gpu_type='fixture',requeue=False)
LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)
ALIAS=RESOURCES
"""
        expected = dict(account='fixture', partition='fixture', qos='fixture',
                        requested_cpus=4, max_allocated_cpus=4, memory_mib=15360,
                        wall_seconds=2880, gpus=1, gpu_type='fixture', requeue=False)
        self.assertEqual(parser.constant(source, 'RESOURCES'), expected)
        self.assertEqual(parser.constant(source, 'ALIAS'), expected)
        self.assertEqual(parser.constant(source, 'LIMITS'),
                         dict(session_reserve_seconds=600, session_cap_seconds=1200,
                              work_seconds=2820, final_seconds=2860))

    def test_reject_nonliteral_or_nonkeyword_construction(self):
        cases = [
            "C=dict([('a',1)])",
            "C=dict(**{'a':1})",
            "C=dict(a=1,a=2)",
            "C=other(a=1)",
            "C=builtins.dict(a=1)",
            "N=1\nC=dict(a=N)",
            "C=dict(a=1+2)",
            "C=dict(a=side_effect())",
            "C=dict(a=dict(b=1))",
            "C=dict(a=[1,2])",
            "C=dict(a=(lambda:1))",
        ]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    parser.constant(source, 'C')

    def test_reject_shadowed_or_wildcard_constructor(self):
        prefixes = [
            "dict=7",
            "def dict(**kwargs): return kwargs",
            "async def dict(**kwargs): return kwargs",
            "class dict: pass",
            "import arbitrary as dict",
            "from arbitrary import dict",
            "from arbitrary import *",
        ]
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    parser.constant(prefix + '\nC=dict(a=1)', 'C')


if __name__ == '__main__':
    unittest.main(verbosity=2)
