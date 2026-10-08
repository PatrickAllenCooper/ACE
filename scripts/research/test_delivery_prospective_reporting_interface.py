"""Optional reporting packaging only: fabricated custody, no report/replay runs.

Reuse fixture constructors, never run the older suites or import the reporter.
Placeholder checkpoint/prediction bytes establish no scientific qualification.
"""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_delivery_release as builder
import delivery_prospective_source_contract as successor
import extend_delivery_prospective_release_plan as planner
from test_delivery_prospective_release_assembly import fixture, put
from test_delivery_prospective_source_contract import add_predecessor
from verify_delivery_release import sha, verify


def reporting_fixture(root):
    args = fixture(root)
    inherited, old, _ = add_predecessor(args, root)
    here = Path(__file__).resolve().parent
    # Copies permit replacement tests without touching repository sources.
    for key, name in [('replay', 'replay_delivery_prospective_release.py'),
                      ('reporting', successor.REPORTING)]:
        target = root/name
        target.write_bytes((here/name).read_bytes())
        args[key] = target
    args['expected_reporting_sha256'] = sha(args['reporting'])
    return args, inherited, old


class ReportingInterfaceTests(unittest.TestCase):
    def assert_rejected_before_scores(self, args, message):
        original = planner.snapshot
        decoded = []
        def guarded(path, *a, **kw):
            if Path(path).name == 'scores.json':
                decoded.append(path)
                raise AssertionError('reporting rejection occurred after score decode')
            return original(path, *a, **kw)
        with patch.object(planner, 'snapshot', guarded):
            with self.assertRaisesRegex(ValueError, message):
                planner.extend(**args)
        self.assertEqual(decoded, [])
        # Some overlap cases point at an existing input; those tests separately
        # assert preservation. Normal rejected outputs must remain absent.
        return args

    def test_seventh_interface_package_bindings_relocation_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); root = base/'inputs'; root.mkdir()
            args, inherited, old = reporting_fixture(root)
            raw = args['reporting'].read_bytes()
            inventory = json.loads(args['inventory'].read_text())
            originals = {Path(s): sha(s) for e in inventory['files'] + inherited['files'] for s in e['sources']}
            result = planner.extend(**args)
            self.assertEqual(result['source_contract_transition']['interfaces'], 7)
            self.assertTrue(result['descriptive_reporting_included'])
            self.assertEqual(result['reporting_sha256'], args['expected_reporting_sha256'])
            plan = json.loads(args['out'].read_text())
            entry = next(e for e in plan['files'] if e['path'] == successor.REPORTING)
            self.assertEqual(entry['transform'], {'kind': 'identity'})
            self.assertEqual(entry['original_sha256'], args['expected_reporting_sha256'])
            self.assertEqual(Path(entry['sources'][0]).read_bytes(), raw)
            self.assertTrue(Path(entry['sources'][0]).is_relative_to(args['private_dir']))
            package = base/'package'
            built = builder.build(args['out'], package, base/'build-private.json')
            active = json.loads((package/successor.ACTIVE).read_text())
            interface = active['interfaces'][6]
            self.assertEqual(interface['path'], successor.REPORTING)
            self.assertEqual((package/successor.REPORTING).read_bytes(), raw)
            self.assertIn('separately authored', interface['implementation'])
            self.assertEqual(interface['runtime_records'][0]['path'], 'B/replay_contract.json')
            self.assertEqual([e['path'] for e in interface['import_records']],
                             ['replay_delivery_prospective_release.py', successor.VERIFIER])
            actual_imports = {node.module+'.py': [alias.name for alias in node.names]
                              for node in ast.parse(raw).body if isinstance(node, ast.ImportFrom)
                              and node.module in ('replay_delivery_prospective_release', 'verify_delivery_release')}
            self.assertEqual({r['path']: r['symbols'] for r in interface['import_records']}, actual_imports)
            self.assertEqual(interface['import_records'][1]['symbols'], ['relative'])
            for flag in ('anonymous_execution_qualified', 'import_qualified', 'reporting_execution_qualified',
                         'generated_report_qualified', 'B_numerical_replay_qualified'):
                self.assertIs(interface[flag], False)
            for flag in ('B_descriptive_reporting_qualified', 'B_numerical_replay_qualified',
                         'training_reproduction_available', 'public_release_approved', 'anonymity_review_complete'):
                self.assertIs(active[flag], False)
            self.assertIn('--manifest-sha256', interface['command'])
            self.assertIn('--replay-receipt', interface['command'])
            self.assertIn('--replay-sha256', interface['command'])
            self.assertEqual((result['new_fits'], result['new_responses']), (0, 0))
            manifest = json.loads((package/'manifest.json').read_text())
            for group in ('runtime_records', 'import_records'):
                for i, record in enumerate(interface[group]):
                    self.assertEqual(record['sha256'], sha(package/record['path']))
                    self.assertIn(dict(record=successor.ACTIVE, pointer=f'/interfaces/6/{group}/{i}/sha256',
                                       artifact=record['path'], digest='sha256'), manifest['bindings'])
            self.assertIn(dict(record=successor.ACTIVE, pointer='/interfaces/6/sha256',
                               artifact=successor.REPORTING, digest='sha256'), manifest['bindings'])
            self.assertEqual([(r['original_location'], r['original_sha256'], r['anonymous_execution_qualified'])
                              for r in active['bindings']],
                             [(r['original_location'], r['original_sha256'], False) for r in old['bindings']])
            self.assertEqual(originals, {p: sha(p) for p in originals})
            self.assertFalse(any(e['path'].endswith(('supplement.json', 'output_index.json', '.csv', '.tex'))
                                 for e in manifest['files']))
            moved = base/'relocated'; package.rename(moved)
            self.assertEqual(verify(moved, built['manifest_sha256'])['files_verified'], result['files'])
            (moved/successor.REPORTING).write_bytes(b'changed reporter\n')
            with self.assertRaisesRegex(ValueError, 'digest|changed|hash'):
                verify(moved, built['manifest_sha256'])

    def test_default_still_has_six_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, _, _ = reporting_fixture(root)
            del args['reporting']; del args['expected_reporting_sha256']
            result = planner.extend(**args)
            active = json.loads((args['private_dir']/'source_contract_B.json').read_text())
            self.assertEqual(result['source_contract_transition']['interfaces'], 6)
            self.assertFalse(result['descriptive_reporting_included'])
            self.assertNotIn(successor.REPORTING, {e['path'] for e in active['interfaces']})
            self.assertNotIn('B_descriptive_reporting_qualified', active)

    def test_optional_sources_captured_once_survive_replacement_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, _, _ = reporting_fixture(root)
            captured = {args[key]: args[key].read_bytes() for key in ('reporting', 'replay')}
            original_snapshot, original_open = planner.snapshot, os.open
            original_path_open = Path.open
            reads = {p: 0 for p in captured}; replaced = []
            def counted(path, *a, **kw):
                if Path(path) in reads: reads[Path(path)] += 1
                return original_open(path, *a, **kw)
            def reject_reopen(path, *a, **kw):
                if path in captured:
                    raise AssertionError('caller optional source reopened after capture')
                return original_path_open(path, *a, **kw)
            def replace_after_capture(path, *a, **kw):
                if Path(path).name == 'scores.json' and not replaced:
                    for source in captured:
                        source.unlink()
                        replacement = root/('changed-'+source.name)
                        replacement.write_bytes(b'# replacement must not be packaged\n')
                        source.symlink_to(replacement)
                    replaced.append(True)
                return original_snapshot(path, *a, **kw)
            with patch.object(planner.os, 'open', counted), patch.object(Path, 'open', reject_reopen), \
                    patch.object(planner, 'snapshot', replace_after_capture):
                planner.extend(**args)
            self.assertEqual(list(reads.values()), [1, 1])
            self.assertEqual(replaced, [True])
            plan = json.loads(args['out'].read_text())
            for source, raw in captured.items():
                entry = next(e for e in plan['files'] if e['path'] == source.name)
                self.assertEqual(Path(entry['sources'][0]).read_bytes(), raw)
            derivation = json.loads((args['private_dir']/'derivation.json').read_text())
            self.assertEqual(len(derivation['optional_interface_snapshots']), 2)
            self.assertEqual(next(e for e in derivation['optional_interface_snapshots'] if e['path'] == successor.REPORTING)['sha256'],
                             args['expected_reporting_sha256'])

    def test_required_pin_replay_and_predecessor_reject_before_scores(self):
        cases = ('no-pin', 'bad-pin', 'wrong-pin', 'no-replay', 'no-predecessor', 'pin-only')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, inherited, _ = reporting_fixture(root)
            for case in cases:
                with self.subTest(case=case):
                    changed = dict(args); plan = copy.deepcopy(inherited)
                    if case == 'no-pin': del changed['expected_reporting_sha256']
                    elif case == 'bad-pin': changed['expected_reporting_sha256'] = 'not a digest'
                    elif case == 'wrong-pin': changed['expected_reporting_sha256'] = '0'*64
                    elif case == 'no-replay': changed['replay'] = None
                    elif case == 'no-predecessor':
                        plan['files'] = [e for e in plan['files'] if e['path'] != successor.ACTIVE]
                    elif case == 'pin-only': del changed['reporting']
                    put(args['plan_file'], plan)
                    self.assert_rejected_before_scores(changed, 'reporting|explicit B replay')
                    self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())

    def test_missing_symlink_and_collision_reject_before_scores(self):
        cases = ('missing-reporting', 'missing-replay', 'reporting-symlink', 'replay-symlink',
                 'parent-symlink', 'collision', 'child-collision', 'predecessor-symlink', 'verifier-transform')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, inherited, _ = reporting_fixture(root)
            for case in cases:
                with self.subTest(case=case):
                    changed = dict(args); plan = copy.deepcopy(inherited)
                    if case.startswith('missing-'):
                        changed[case.removeprefix('missing-')] = root/'missing.py'
                    elif case in ('reporting-symlink', 'replay-symlink'):
                        key = case.removesuffix('-symlink'); link = root/(case+'.py')
                        link.symlink_to(args[key]); changed[key] = link
                    elif case == 'parent-symlink':
                        link = root/'alias-directory'; link.symlink_to(root, target_is_directory=True)
                        changed['reporting'] = link/args['reporting'].name
                    elif case in ('collision', 'child-collision'):
                        name = successor.REPORTING + ('/nested.py' if case == 'child-collision' else '')
                        plan['files'].append(dict(path=name, sources=[str(args['reporting'])],
                            original_sha256=args['expected_reporting_sha256'], role='collision', transform={'kind': 'identity'}))
                    elif case == 'predecessor-symlink':
                        entry = next(e for e in plan['files'] if e['path'] == successor.ACTIVE)
                        source = Path(entry['sources'][0]); link = root/'alias-predecessor'
                        link.symlink_to(source.parent, target_is_directory=True)
                        entry['sources'] = [str(link/source.name)]
                    elif case == 'verifier-transform':
                        entry = next(e for e in plan['files'] if e['path'] == successor.VERIFIER)
                        entry['transform'] = {'kind': 'project-json', 'keys': ['synthetic_interface']}
                    put(args['plan_file'], plan)
                    self.assert_rejected_before_scores(changed, 'missing|symlink|collision|identity')
                    self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())

    def test_optional_source_output_overlap_rejects_before_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, _, _ = reporting_fixture(root)
            alias_parent = root/'alias-parent'; alias_parent.mkdir()
            for key in ('reporting', 'replay'):
                for output in ('out', 'private_dir'):
                    for alias in (False, True):
                        with self.subTest(source=key, output=output, alias=alias):
                            changed = dict(args); changed[output] = args[key]/'future-output'
                            if alias: changed[key] = alias_parent/'..'/args[key].name
                            before = args[key].read_bytes()
                            self.assert_rejected_before_scores(changed, 'overlaps original custody')
                            self.assertEqual(args[key].read_bytes(), before)
                            self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())

    def test_original_acceptance_gate_is_first_even_with_reporting(self):
        with patch.object(planner.core, 'accepted_before_scores', side_effect=ValueError('original blocked')):
            with patch.object(planner, 'module_snapshot', side_effect=AssertionError('source captured before gate')):
                with self.assertRaisesRegex(ValueError, 'original blocked'):
                    planner.extend('plan', 'study', 'inv', 'srcinv', 'src', 'core', 'cc', 'a'*64,
                                   'private', 'out', reporting='report.py', expected_reporting_sha256='b'*64)

    def test_changed_import_closures_reject_with_fresh_matching_source_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()/'inputs'; root.mkdir()
            args, _, _ = reporting_fixture(root)
            original = {key: args[key].read_bytes() for key in ('reporting', 'replay')}
            variants = [
                ('reporting', 'from verify_delivery_release import relative',
                 'from unrecorded_reporting_dependency import relative'),
                ('reporting', 'import csv', 'import subprocess'),
                ('reporting', 'byte_integrity, gate)', 'byte_integrity, missing_symbol)'),
                ('replay', 'import numpy as np', 'import unsupported_array as np'),
                ('replay', 'from ace.oracle import MLPSurrogate',
                 'from unsupported_learner import MLPSurrogate'),
                ('replay', 'def gate(', 'def different_gate('),
            ]
            for key, before, after in variants:
                with self.subTest(key=key, after=after):
                    for name in original: args[name].write_bytes(original[name])
                    raw = original[key].decode(); self.assertIn(before, raw)
                    args[key].write_text(raw.replace(before, after))
                    args['expected_reporting_sha256'] = sha(args['reporting'])
                    self.assert_rejected_before_scores(args, 'unsupported captured|exports missing')
                    self.assertFalse(args['out'].exists()); self.assertFalse(args['private_dir'].exists())


if __name__ == '__main__':
    unittest.main()
