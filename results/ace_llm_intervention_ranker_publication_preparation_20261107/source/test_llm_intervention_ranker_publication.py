"""New focused stdlib-only publication checks, never importing a model module."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(os.environ["ACE_NUMERIC_PUBLICATION_TEST_OUTPUT"])
SOURCE = Path(os.environ["ACE_NUMERIC_PUBLICATION_SOURCE"])
CONTRACT = Path(os.environ["ACE_NUMERIC_PUBLICATION_CONTRACT"])
namespace = {"os": os, "json": json}
for path, names in ((CONTRACT, {"canonical"}),
                    (SOURCE, {"_save", "_publish_terminal", "_failure_receipt"})):
    selected = [n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class TerminalPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.mkdir(exist_ok=False)

    def setUp(self):
        self.output = ROOT/self._testMethodName
        self.output.mkdir()
        self.value = {"status": "complete", "fixture": True}
        self.failure = {"status": "failed", "message": "injected"}

    def failed(self):
        namespace["_failure_receipt"](self.output, self.failure)

    def test_fsync_failure_retains_candidate_without_success(self):
        with patch.object(os, "fsync", side_effect=OSError("injected fsync")):
            with self.assertRaisesRegex(OSError, "fsync"):
                namespace["_publish_terminal"](self.output, self.value)
        self.failed()
        self.assertTrue((self.output/"terminal-candidate.json").exists())
        self.assertFalse((self.output/"terminal.json").exists())
        self.assertEqual(json.loads((self.output/"failure.json").read_bytes()), self.failure)

    def test_link_failure_retains_candidate_without_success(self):
        with patch.object(os, "link", side_effect=OSError("injected link")):
            with self.assertRaisesRegex(OSError, "link"):
                namespace["_publish_terminal"](self.output, self.value)
        self.failed()
        self.assertFalse((self.output/"terminal.json").exists())
        self.assertTrue((self.output/"failure.json").exists())

    def test_success_is_same_inode_with_no_postcommit_io(self):
        original, calls = os.link, []
        def commit(*args, **kwargs):
            calls.append("link")
            return original(*args, **kwargs)
        with patch.object(os, "link", commit):
            namespace["_publish_terminal"](self.output, self.value)
        self.assertEqual(calls, ["link"])
        self.assertTrue(os.path.samefile(self.output/"terminal-candidate.json", self.output/"terminal.json"))
        self.assertEqual(json.loads((self.output/"terminal.json").read_bytes()), self.value)
        function = next(n for n in ast.parse(SOURCE.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_publish_terminal")
        self.assertIsInstance(function.body[-1], ast.Expr)
        self.assertEqual(ast.unparse(function.body[-1].value.func), "os.link")
        self.assertFalse((self.output/"failure.json").exists())

    def test_existing_terminal_rejects_without_overwrite(self):
        existing = b'{"status":"existing-not-accepted"}\n'
        (self.output/"terminal.json").write_bytes(existing)
        with self.assertRaises(FileExistsError):
            namespace["_publish_terminal"](self.output, self.value)
        self.failed()
        self.assertEqual((self.output/"terminal.json").read_bytes(), existing)
        self.assertFalse((self.output/"failure.json").exists())
        self.assertEqual(json.loads((self.output/"publication-interruption.json").read_bytes())["status"],
                         "publication_interrupted_reconciliation_required")

    def test_interruption_after_commit_never_publishes_final_failure(self):
        original = os.link
        def commit_then_interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt("injected immediately after commit")
        with patch.object(os, "link", commit_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                namespace["_publish_terminal"](self.output, self.value)
        self.failed()
        self.assertTrue((self.output/"terminal.json").exists())
        self.assertFalse((self.output/"failure.json").exists())
        self.assertEqual(json.loads((self.output/"publication-interruption.json").read_bytes())["status"],
                         "publication_interrupted_reconciliation_required")


if __name__ == "__main__":
    unittest.main(verbosity=2)
