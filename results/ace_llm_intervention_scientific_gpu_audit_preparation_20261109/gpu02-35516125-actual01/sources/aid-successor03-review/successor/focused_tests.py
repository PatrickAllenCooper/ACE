"""New fabricated dual-roster cases only; no actual audit or production execution."""
from pathlib import Path
import copy
import hashlib
import types
import unittest

path = Path(__file__).resolve().parent / 'dependencies/gpu_recovery_delta.py'
data = path.read_bytes()
assert hashlib.sha256(data).hexdigest() == 'e508f20eaec5f8a94c7638dc6183f5e46e9798918e8877a75c527fed6d4bb098'
mod = types.ModuleType('focused_roster_validator')
mod.__file__ = str(path)
exec(compile(data, str(path), 'exec'), mod.__dict__)


def fixture():
    old = {mod.ENTRY: mod.ORIGINAL_ENTRY, 'science.py': 'a' * 64, 'capture_only.py': 'b' * 64}
    declared = sorted([mod.ENTRY, 'science.py'])
    closure = dict(input_source_files=old, source_names=declared,
                   helpers_outside_source_names={'capture_only.py': 'b' * 64},
                   entrypoint=mod.ENTRY, entrypoint_sha256=mod.ORIGINAL_ENTRY,
                   loaded_work_source_roster=['science.py'],
                   loaded_work_source_pins={'science.py': 'a' * 64})
    packet = dict(files={**old, **mod.OPERATIONAL},
                  source_names=sorted(set(declared) | {mod.CHECKER, mod.HISTORY}))
    sources = {n: packet['files'][n] for n in packet['source_names']}
    authority = dict(operational_source_files=sources)
    context = dict(source_files_sha256=mod.digest(sources))
    manifest = dict(schema='ace-gpu02-operational-sources-v1', source_commit='1' * 40,
                    files={n: dict(repository_path='ops/' + n, sha256=h)
                           for n, h in mod.OPERATIONAL.items()})
    frozen = {'original/' + n: h for n, h in old.items()}
    return packet, closure, authority, context, manifest, frozen


class DualRosterCases(unittest.TestCase):
    def test_distinct_exact_capture_and_declared_rosters(self):
        args = fixture()
        result = mod.validate_sources(*args)
        self.assertEqual(result, args[2]['operational_source_files'])
        self.assertNotIn('capture_only.py', result)
        self.assertEqual(args[0]['files']['capture_only.py'], 'b' * 64)
        self.assertEqual(len(args[0]['files']), 5)
        self.assertEqual(len(result), 4)

    def test_reject_declared_or_captured_membership_drift(self):
        changes = [
            lambda x: x[0]['source_names'].append(mod.ENTRY),
            lambda x: x[0]['source_names'].remove('science.py'),
            lambda x: x[0]['source_names'].append('extra.py'),
            lambda x: x[0]['source_names'].append('capture_only.py'),
            lambda x: x[0]['source_names'].remove(mod.CHECKER),
            lambda x: x[0]['source_names'].reverse(),
            lambda x: x[0]['files'].update({'extra.py': 'c' * 64}),
            lambda x: x[0]['files'].pop('capture_only.py'),
            lambda x: x[0]['files'].pop('science.py'),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                args = fixture(); change(args)
                with self.assertRaises(ValueError):
                    mod.validate_sources(*args)

    def test_reject_original_operational_and_authority_pin_drift(self):
        changes = [
            lambda x: x[0]['files'].update({'capture_only.py': 'c' * 64}),
            lambda x: x[0]['files'].update({'science.py': 'c' * 64}),
            lambda x: x[0]['files'].update({mod.ENTRY: mod.ORIGINAL_ENTRY}),
            lambda x: x[0]['files'].update({mod.CHECKER: 'c' * 64}),
            lambda x: x[0]['files'].update({mod.HISTORY: 'c' * 64}),
            lambda x: x[5].update({'original/science.py': 'c' * 64}),
            lambda x: x[5].update({'duplicate/science.py': 'a' * 64}),
            lambda x: x[2]['operational_source_files'].update({'extra.py': 'c' * 64}),
            lambda x: x[3].update(source_files_sha256='c' * 64),
            lambda x: x[4]['files'][mod.CHECKER].update(sha256='c' * 64),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                args = fixture(); change(args)
                with self.assertRaises(ValueError):
                    mod.validate_sources(*args)

    def test_reject_inconsistent_frozen_relationships(self):
        changes = [
            lambda x: x[1]['source_names'].append('science.py'),
            lambda x: x[1]['source_names'].remove(mod.ENTRY),
            lambda x: x[1]['helpers_outside_source_names'].clear(),
            lambda x: x[1]['loaded_work_source_roster'].append(mod.ENTRY),
            lambda x: x[1]['loaded_work_source_pins'].update({'science.py': 'c' * 64}),
            lambda x: x[1].update(entrypoint_sha256='c' * 64),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                args = fixture(); change(args)
                with self.assertRaises(ValueError):
                    mod.validate_sources(*args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
