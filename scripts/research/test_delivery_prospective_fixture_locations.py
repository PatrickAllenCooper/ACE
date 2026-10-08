"""Source/AST fixtures only; never import or execute scientific fixture sources."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_delivery_prospective_fixture_locations as fixture


RAW = ("import os\nfrom pathlib import Path\n"
       "# UTF-8 offset fixture: \u03bb\n"
       "PROJECT = " + fixture.ORIGINAL['PROJECT'] + "\n"
       "SOURCE = " + fixture.ORIGINAL['SOURCE'] + "\n"
       "def preserved():\n    raise RuntimeError('NEVER EXECUTE SCIENTIFIC CODE')\n"
       "class Guards:\n    def test_charge(self):\n        assert 'charged before generation'\n"
       "    def test_holdout(self):\n        assert 'no heldout reads in fitting'\n").encode()


class FixtureLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_exact_independent_byte_oracle_and_unchanged_function_class_spans(self):
        derived = fixture.derive_model(RAW)
        # Byte oracle separate from the AST span algorithm.
        expected = RAW.replace(fixture.ORIGINAL['PROJECT'].encode(), fixture.REPLACEMENTS['PROJECT'].encode(), 1)
        expected = expected.replace(fixture.ORIGINAL['SOURCE'].encode(), fixture.REPLACEMENTS['SOURCE'].encode(), 1)
        self.assertEqual(derived, expected)
        before, after = ast.parse(RAW), ast.parse(derived)
        for a, b in zip(before.body, after.body):
            if isinstance(a, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.get_source_segment(RAW.decode(), a), ast.get_source_segment(derived.decode(), b))
        self.assertNotIn(b'/Users/pat/', derived)
        self.assertIn(b"os.environ['ACE_DELIVERY_FIXTURE_ROOT']", derived)

    def test_unicode_on_declaration_and_multiline_rhs(self):
        raw = RAW.replace(b'PROJECT = ', '\u03bb = 0; PROJECT = '.encode()).replace(
            ("SOURCE = " + fixture.ORIGINAL['SOURCE']).encode(),
            ("SOURCE = (\n    " + fixture.ORIGINAL['SOURCE'] + "\n)").encode())
        derived = fixture.derive_model(raw)
        self.assertIn('\u03bb = 0; PROJECT = '.encode(), derived)
        self.assertIn(("SOURCE = (\n    " + fixture.REPLACEMENTS['SOURCE'] + "\n)").encode(), derived)

    def test_unknown_duplicate_missing_or_nested_location_rejects(self):
        for raw in (
            RAW.replace(b'parents[2]', b'parents[3]'),
            RAW + b'PROJECT = Path("other")\n',
            RAW.replace(b'PROJECT = ', b'PROJECT: Path = '),
            RAW.replace(b'PROJECT = ', b'PROJECT = ALSO = '),
            RAW.replace(b'PROJECT = ', b'OTHER = '),
            RAW.replace(b'PROJECT = ', b'def nested():\n    PROJECT = '),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError): fixture.derive_model(raw)

    def make_closure(self):
        root = self.root/'closure'; root.mkdir()
        sources = {fixture.MODEL: RAW, fixture.IO: b'# unchanged IO source\nraise RuntimeError("NEVER EXECUTE")\n'}
        sources.update({f'scripts/research/fabricated_{i}.py': f'# fabricated project {i}\n'.encode() for i in range(22)})
        learners = {f'ace/fabricated_{i}.py': f'# fabricated learner {i}\n'.encode() for i in range(19)}
        record = {'source_revision': fixture.REVISION, 'registration_sha256': fixture.REGISTRATION,
                  'project_files': {}, 'learner_files': {}, 'outcomes_accessed': False}
        for folder, mapping, key in (('project', sources, 'project_files'), ('source', learners, 'learner_files')):
            for name, raw in mapping.items():
                target = root/folder/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
                record[key][name] = fixture.digest(raw)
        path = root/'source-closure.json'; path.write_text(json.dumps(record))
        return path, record, {name:fixture.digest(sources[name]) for name in (fixture.MODEL,fixture.IO)}

    def test_actual_preparer_with_fabricated_closure_no_source_execution(self):
        path, record, pins = self.make_closure(); out = self.root/'new'
        with patch.dict(fixture.PINS, pins):
            receipt = fixture.prepare(path,fixture.digest(path.read_bytes()),out)
        contract = json.loads((out/'contract.json').read_bytes())
        self.assertEqual(receipt['verified_original_project_files'],24)
        self.assertEqual(receipt['verified_original_learner_files'],19)
        self.assertEqual({f['path'] for f in contract['files']}, {
            'originals/test_delivery_prospective_models.py','originals/test_delivery_prospective_io.py',
            'fixtures/test_delivery_prospective_models.py','fixtures/test_delivery_prospective_io.py'})
        self.assertEqual((out/'originals/test_delivery_prospective_models.py').read_bytes(), RAW)
        self.assertEqual((out/'fixtures/test_delivery_prospective_models.py').read_bytes(), fixture.derive_model(RAW))
        self.assertEqual((out/'fixtures/test_delivery_prospective_io.py').read_bytes(), (path.parent/'project'/fixture.IO).read_bytes())
        for f in contract['files']: self.assertEqual(fixture.digest((out/f['path']).read_bytes()),f['sha256'])
        self.assertEqual(receipt['contract_sha256'],fixture.digest((out/'contract.json').read_bytes()))
        for name in ('fixtures_executed','original_training_reproduction_qualified','anonymous_execution_qualified','public_release_approved'):
            self.assertIs(contract[name],False)
        self.addCleanup(self.unseal, out)
        with patch.dict(fixture.PINS,pins), self.assertRaises(ValueError):
            fixture.prepare(path,fixture.digest(path.read_bytes()),out)

    @staticmethod
    def unseal(path):
        for folder in [path,*[p for p in path.rglob('*') if p.is_dir()]]: folder.chmod(0o700)

    def test_pin_metadata_source_and_output_fail_before_writes(self):
        path, original, pins = self.make_closure()
        out = self.root/'rejected'
        with patch.dict(fixture.PINS,pins):
            for pin in ('0'*64,'HEAD',fixture.digest(path.read_bytes()).upper()):
                with self.subTest(pin=pin), self.assertRaises(ValueError): fixture.prepare(path,pin,out)
                self.assertFalse(out.exists())
            variants=[]
            for key,val in (('source_revision','0'*40),('registration_sha256','0'*64),('outcomes_accessed',True)):
                v=copy.deepcopy(original);v[key]=val;variants.append(v)
            v=copy.deepcopy(original);v['project_files'].pop(fixture.IO);variants.append(v)
            v=copy.deepcopy(original);v['learner_files']['../escape']='0'*64;variants.append(v)
            for value in variants:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): fixture.prepare(path,fixture.digest(path.read_bytes()),out)
                self.assertFalse(out.exists())
            path.write_text(json.dumps(original))
            target=path.parent/'source/ace/fabricated_18.py';target.write_bytes(b'changed last closure object')
            with self.assertRaises(ValueError): fixture.prepare(path,fixture.digest(path.read_bytes()),out)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError): fixture.prepare(path,fixture.digest(path.read_bytes()),path.parent/'new')
            alias=self.root/'alias';alias.symlink_to(path.parent,target_is_directory=True)
            with self.assertRaises(ValueError): fixture.prepare(alias/path.name,fixture.digest(path.read_bytes()),out)


if __name__ == '__main__': unittest.main()
