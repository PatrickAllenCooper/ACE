"""New service authentication tests using bytes and arithmetic modules only."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import llm_intervention_model_service as service


class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ace-model-service-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/"model"
        self.model.mkdir()
        def files(root, names):
            result = {}
            for name in names:
                raw = ("artificial bytes for " + name).encode()
                (root/name).write_bytes(raw)
                result[name] = hashlib.sha256(raw).hexdigest()
            return result
        self.freeze = {"schema": service.VERSION, "scope": "artificial",
            "model_directory": str(self.model), "model_files": files(self.model, service.MODEL_NAMES),
            "sources": files(self.root, service.SOURCE_NAMES), "runtime": service.RUNTIME,
            "python_version": [3, 10, 19], "recipe": service.RECIPE,
            "treatment": "frozen", "adapter": None,
            "limits": {"startup_seconds": 180, "request_seconds": 30,
                       "idle_seconds": 30, "total_seconds": 540, "max_requests": 16}}

    def packet(self):
        raw = (service.canonical(self.freeze)+"\n").encode()
        path = self.root/"freeze.json"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_all_inputs_are_authenticated_before_model_import(self):
        before = set(sys.modules)
        observed, captured, adapter, model_records = service.authenticate(*self.packet(), runtime=False)
        self.assertEqual(observed, self.freeze)
        self.assertEqual(set(captured), service.SOURCE_NAMES)
        self.assertEqual(set(model_records), service.MODEL_NAMES)
        self.assertIsNone(adapter)
        self.assertFalse({"torch", "transformers"} & (set(sys.modules)-before))
        self.assertEqual(service.identity(observed)["model_sha256"], service.digest(self.freeze["model_files"]))

    def test_source_model_and_manifest_corruption_fail(self):
        path, sha = self.packet()
        with self.assertRaises(ValueError):
            service.authenticate(path, "0"*64, runtime=False)
        for root, name in ((self.root, "llm_intervention_learning.py"), (self.model, "model.safetensors")):
            original = (root/name).read_bytes()
            (root/name).write_bytes(original+b"changed")
            with self.assertRaises(ValueError):
                service.authenticate(path, sha, runtime=False)
            (root/name).write_bytes(original)
        (self.model/"extra.json").write_text("{}")
        with self.assertRaises(ValueError):
            service.authenticate(path, sha, runtime=False)

    def test_duplicate_nonfinite_and_changed_recipe_are_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                service.decode(raw)
        for change in (lambda f: f["recipe"].update(thinking=True),
                       lambda f: f["limits"].update(max_requests=True),
                       lambda f: f["sources"].pop("llm_intervention_task.py"),
                       lambda f: f.update(scope="confirmation")):
            bad = copy.deepcopy(self.freeze)
            change(bad)
            with self.assertRaises(ValueError):
                service.validate_freeze(bad)

    def test_adapter_receipt_bytes_and_scope_must_match(self):
        adapter, receipt = b"not a real tensor", b"not a qualification receipt"
        (self.root/"adapter.pt").write_bytes(adapter)
        (self.root/"training.json").write_bytes(receipt)
        self.freeze.update(treatment="artificial_adapter", adapter={"file": "adapter.pt",
            "sha256": hashlib.sha256(adapter).hexdigest(), "training_receipt": "training.json",
            "training_receipt_sha256": hashlib.sha256(receipt).hexdigest(), "training_scope": "artificial"})
        _, _, raw, _ = service.authenticate(*self.packet(), runtime=False)
        self.assertEqual(raw, adapter)  # Authentication does not deserialize it.
        (self.root/"training.json").write_bytes(receipt+b"changed")
        with self.assertRaises(ValueError):
            service.authenticate(*self.packet(), runtime=False)
        self.freeze["scope"] = "development"
        with self.assertRaises(ValueError):
            service.validate_freeze(self.freeze)

    def test_captured_source_executes_authenticated_bytes_not_later_disk(self):
        name = "llm_intervention_arithmetic_fixture"
        path = self.root/(name+".py")
        path.write_text("answer = 999\n")
        loader = service.CapturedSources({name+".py": b"answer = 6 * 7\n"}, self.root)
        sys.meta_path.insert(0, loader)
        try:
            module = importlib.import_module(name)
            self.assertEqual(module.answer, 42)
            self.assertEqual(module.__file__, str(path))
            with self.assertRaises(ValueError):
                service.CapturedSources({name+".py": b"answer = 0"}, self.root)
            with self.assertRaises(ImportError):
                loader.find_spec("llm_intervention_unpinned_fixture")
        finally:
            sys.modules.pop(name, None)
            sys.meta_path.remove(loader)

    def test_symlink_and_runtime_mismatch_rejected(self):
        original = self.root/"llm_intervention_task.py"
        real = self.root/"actual.py"
        original.rename(real)
        original.symlink_to(real)
        with self.assertRaises(OSError):
            service.authenticate(*self.packet(), runtime=False)
        original.unlink()
        real.rename(original)
        with patch.object(service.importlib.metadata, "version", return_value="wrong"):
            with self.assertRaises(ValueError):
                service.authenticate(*self.packet())

    def test_exclusive_journal_and_ordinal_events(self):
        out = self.root/"output"
        journal = service.Journal(out)
        journal.event("before")
        journal.event("after", observed=42)
        self.assertEqual(json.loads((out/"phase-001.json").read_bytes())["observed"], 42)
        with self.assertRaises(FileExistsError):
            journal.save("phase-001.json", {})
        with self.assertRaises(FileExistsError):
            service.Journal(out)


if __name__ == "__main__":
    unittest.main()
