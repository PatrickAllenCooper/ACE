"""Focused optional TOML metadata projection and downstream digest checks."""
import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
import importlib.util
from unittest.mock import patch

from build_delivery_release import build
from verify_delivery_release import project_toml_bytes, sha, identifying_bytes, verify
from extend_delivery_notice_plan import OMIT, outputs, REVIEWED, captured
from replay_delivery_prospective_release import byte_integrity


class NoticeProjectionTests(unittest.TestCase):
    def test_only_optional_fields_change_and_license_and_dependencies_remain(self):
        raw=b'''[tool.poetry]\nname="fixture"\nauthors=["Decisive AI Team"]\nhomepage="https://example.test/home"\nrepository="https://example.test/repo"\nlicense="MIT"\n[tool.poetry.dependencies]\npython=">=3.11"\nnumpy="2.2.6"\n'''
        transform={'kind':'project-toml','omit':OMIT}
        expected=copy.deepcopy(tomllib.loads(raw.decode()))
        for name in ('authors','homepage','repository'):del expected['tool']['poetry'][name]
        projected=project_toml_bytes(raw,transform)
        self.assertEqual(tomllib.loads(projected.decode()),expected)
        with self.assertRaises(ValueError):project_toml_bytes(raw,{'kind':'project-toml','omit':OMIT+['tool.poetry.license']})
        with self.assertRaises(ValueError):project_toml_bytes(raw.replace(b'license="MIT"',b'license="MIT"\nlicense="Other"'),transform)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); original=root/'original.toml';original.write_bytes(raw)
            self.assertTrue(identifying_bytes(original))
            derived_hash=__import__('hashlib').sha256(projected).hexdigest()
            record=root/'protocol.json';record.write_text(json.dumps({'source_sha256':sha(original),'derived_sha256':derived_hash}))
            plan={'status':'synthetic metadata test, no grant','files':[
                {'path':'source/runner/pyproject.toml','sources':[str(original)],'original_sha256':sha(original),'role':'derived-runtime-metadata','transform':transform},
                {'path':'F/protocol.json','sources':[str(record)],'original_sha256':sha(record),'role':'synthetic-original-protocol','transform':{'kind':'identity'}}],
                'bindings':[{'record':'F/protocol.json','pointer':'/source_sha256','artifact':'source/runner/pyproject.toml','digest':'original_sha256'},
                            {'record':'F/protocol.json','pointer':'/derived_sha256','artifact':'source/runner/pyproject.toml','digest':'sha256'}]}
            file=root/'plan.json';file.write_text(json.dumps(plan))
            result=build(file,root/'package',root/'private.json')
            self.assertEqual(original.read_bytes(),raw)
            self.assertEqual((root/'package/source/runner/pyproject.toml').read_bytes(),projected)
            self.assertFalse(identifying_bytes(root/'package/source/runner/pyproject.toml'))
            manifest=json.loads((root/'package/manifest.json').read_text())
            byte_integrity(root/'package',manifest)
            self.assertEqual(verify(root/'package',result['manifest_sha256'])['bindings_verified'],2)

    def test_multiline_or_missing_fields_fail_closed_without_losing_semantics(self):
        raw=b'''[tool.poetry]\nauthors=[\n"organization"\n]\nhomepage="example"\nrepository="example"\nlicense="MIT"\n'''
        with self.assertRaises(ValueError):project_toml_bytes(raw,{'kind':'project-toml','omit':OMIT})
        with self.assertRaises(ValueError):project_toml_bytes(b'[tool.poetry]\nlicense="MIT"\n',{'kind':'project-toml','omit':OMIT})

    def test_output_alias_and_both_overlap_directions_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();source=root/'source';source.mkdir()
            alias=root/'alias';alias.symlink_to(source,target_is_directory=True)
            for private,out in [(alias/'new',root/'plan'),(source/'..'/'source'/'new',root/'plan'),
                                (root/'plan'/'custody',root/'plan'),(root/'custody',root/'custody'/'plan')]:
                with self.subTest(private=private,out=out):
                    with self.assertRaises(ValueError):outputs(private,out,[source])
            self.assertFalse((source/'new').exists())

    def test_reviewed_notice_pin_rejects_changed_declaration(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'LICENSE';p.write_bytes(b'Apache License Version 2.0 invented declaration')
            with self.assertRaises(ValueError):captured(p,REVIEWED['LICENSE'])

    def test_imported_verifier_pin_does_not_follow_later_source_edits(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'verifier.py';raw=Path(__file__).with_name('verify_delivery_release.py').read_bytes()
            p.write_bytes(raw)
            spec=importlib.util.spec_from_file_location('isolated_notice_verifier',p)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            before=module.VERIFIER_SHA256
            p.write_bytes(raw+b'\n# later source edit\n')
            self.assertEqual(module.VERIFIER_SHA256,before)
            self.assertNotEqual(sha(p),before)
            # A race changing executable source before the import-time snapshot
            # must not authenticate the different implementation.
            p.write_bytes(raw)
            spec=importlib.util.spec_from_file_location('mismatched_notice_verifier',p)
            module=importlib.util.module_from_spec(spec)
            original=Path.read_bytes
            def different(path):
                value=original(path)
                return value+b'\nextra_executable_instruction = True\n' if path==p else value
            with patch.object(Path,'read_bytes',different):
                with self.assertRaisesRegex(ValueError,'executing implementation'):
                    spec.loader.exec_module(module)


if __name__=='__main__':unittest.main()
