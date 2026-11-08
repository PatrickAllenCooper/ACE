"""Pure disposal controls: fake allocator/workspaces, no numeric/model imports."""
import ast
import builtins
import copy
from pathlib import Path
import types
import unittest


SOURCE = Path(__file__).with_name("llm_intervention_training_gpu_worker.py")
MIB = 1024 * 1024
OPERATION = "torch._C._cuda_clearCublasWorkspaces"


class FakeTorch:
    def __init__(self, *, live=0, cached=0, api="working", partial_failure=False):
        self.calls = []
        self.live, self.cached = live, cached
        self.workspace, self.reserved = 64 * MIB, 150 * MIB
        self.api, self.partial_failure = api, partial_failure
        self.cuda = self
        if api != "missing_binding":
            self._C = types.SimpleNamespace()
            if api != "missing_api":
                self._C._cuda_clearCublasWorkspaces = None if api == "noncallable" else self.clear

    def synchronize(self):
        self.calls.append("synchronize")

    def memory_allocated(self):
        self.calls.append("memory_allocated")
        return self.live + self.workspace

    def memory_reserved(self):
        self.calls.append("memory_reserved")
        return self.reserved

    def clear(self):
        self.calls.append("clear_workspaces")
        if self.api != "raising" or self.partial_failure:
            self.workspace = 0
        if self.api == "raising":
            raise RuntimeError("fake workspace release failed")

    def empty_cache(self):
        self.calls.append("empty_cache")
        self.reserved = self.live + self.workspace + self.cached


class WorkspaceCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            if name not in {"__future__", "gc", "hashlib", "json", "math", "os", "pathlib", "time", "traceback"}:
                raise AssertionError("non-stdlib import in pure cleanup test: " + name)
            return real_import(name, *args, **kwargs)

        cls.source = SOURCE.read_bytes()
        cls.worker = types.ModuleType("captured_workspace_cleanup_worker")
        cls.worker.__dict__["__builtins__"] = {**vars(builtins), "__import__": stdlib_only}
        exec(compile(cls.source, str(SOURCE), "exec"), cls.worker.__dict__)

    def dispose(self, fake):
        self.receipts = []
        original_gc = self.worker.gc
        self.worker.gc = types.SimpleNamespace(collect=lambda: fake.calls.append("gc.collect"))

        def save(name, evidence):
            fake.calls.append("save")
            self.receipts.append((name, copy.deepcopy(evidence)))

        try:
            self.worker._clear_cuda_after_disposal(fake, label="before-sft-reload", save=save,
                stamp=lambda: {"utc": "2026-11-08T00:00:00Z", "monotonic_seconds": 5., "elapsed_seconds": 1.})
        finally:
            self.worker.gc = original_gc

    @staticmethod
    def expected_calls(*, clear=True):
        return ["gc.collect", "synchronize", "memory_allocated", "memory_reserved",
                *(["clear_workspaces"] if clear else []), "synchronize", "empty_cache",
                "memory_allocated", "memory_reserved", "save"]

    @staticmethod
    def fabricated_memory():
        return {"cleanup_operation": OPERATION, "workspace_clear_calls": 1,
                "workspace_clear_succeeded": True, "before_allocated_bytes": 64 * MIB,
                "before_reserved_bytes": 150 * MIB, "allocated_bytes": 0, "reserved_bytes": 0}

    def test_workspace_release_once_in_order_and_before_after_receipt(self):
        fake = FakeTorch()
        self.dispose(fake)
        self.assertEqual(fake.calls, self.expected_calls())
        self.assertEqual(len(self.receipts), 1)
        name, receipt = self.receipts[0]
        self.assertEqual(name, "before-sft-reload-memory.json")
        self.assertEqual(receipt, {**self.fabricated_memory(), "utc": "2026-11-08T00:00:00Z",
                                  "monotonic_seconds": 5., "elapsed_seconds": 1.})
        self.worker._check_disposal_memory(receipt)

    def test_live_allocation_or_reserved_residue_fails_with_saved_measurements(self):
        for live, cached in ((MIB, 0), (0, MIB)):
            with self.subTest(live=live, cached=cached):
                fake = FakeTorch(live=live, cached=cached)
                with self.assertRaisesRegex(ValueError, "survived model disposal"):
                    self.dispose(fake)
                self.assertEqual(fake.calls, self.expected_calls())
                receipt = self.receipts[0][1]
                self.assertEqual(receipt["before_allocated_bytes"], 64 * MIB + live)
                self.assertEqual(receipt["before_reserved_bytes"], 150 * MIB)
                self.assertEqual((receipt["allocated_bytes"], receipt["reserved_bytes"]), (live, live + cached))
                self.assertTrue(receipt["workspace_clear_succeeded"])
                self.assertEqual(receipt["cleanup_error"]["exception_type"], "ValueError")

    def test_missing_or_noncallable_api_fails_closed_and_retains_cleanup(self):
        for api in ("missing_binding", "missing_api", "noncallable"):
            with self.subTest(api=api):
                fake = FakeTorch(api=api)
                # Even a zero-memory fake cannot bypass the required operation.
                fake.workspace = 0
                with self.assertRaisesRegex(ValueError, "API is not callable"):
                    self.dispose(fake)
                self.assertEqual(fake.calls, self.expected_calls(clear=False))
                receipt = self.receipts[0][1]
                self.assertEqual(receipt["workspace_clear_calls"], 0)
                self.assertFalse(receipt["workspace_clear_succeeded"])
                self.assertEqual((receipt["allocated_bytes"], receipt["reserved_bytes"]), (0, 0))
                self.assertEqual(receipt["cleanup_error"]["exception_type"], "ValueError")
                with self.assertRaises(ValueError):
                    self.worker._check_disposal_memory(receipt)

    def test_api_exception_is_propagated_once_even_if_memory_reaches_zero(self):
        for partial_failure in (False, True):
            with self.subTest(partial_failure=partial_failure):
                fake = FakeTorch(api="raising", partial_failure=partial_failure)
                with self.assertRaisesRegex(RuntimeError, "fake workspace release failed"):
                    self.dispose(fake)
                self.assertEqual(fake.calls, self.expected_calls())
                receipt = self.receipts[0][1]
                self.assertEqual(receipt["workspace_clear_calls"], 1)
                self.assertFalse(receipt["workspace_clear_succeeded"])
                self.assertEqual(receipt["before_allocated_bytes"], 64 * MIB)
                after = 0 if partial_failure else 64 * MIB
                self.assertEqual((receipt["allocated_bytes"], receipt["reserved_bytes"]), (after, after))
                self.assertEqual(receipt["cleanup_error"], {"exception_type": "RuntimeError",
                                 "message": "fake workspace release failed"})
                with self.assertRaises(ValueError):
                    self.worker._check_disposal_memory(receipt)

    def test_fabricated_memory_validator_requires_cleanup_metadata_and_exact_zero(self):
        valid = self.fabricated_memory()
        self.worker._check_disposal_memory(valid)
        for key in valid:
            with self.subTest(missing=key):
                changed = dict(valid)
                del changed[key]
                with self.assertRaises(ValueError):
                    self.worker._check_disposal_memory(changed)
        for key, value in (("cleanup_operation", "torch.cuda.empty_cache"),
                           ("workspace_clear_calls", 0), ("workspace_clear_calls", 2),
                           ("workspace_clear_calls", True), ("workspace_clear_succeeded", False),
                           ("workspace_clear_succeeded", 1), ("cleanup_error", {}),
                           ("before_allocated_bytes", -1), ("before_reserved_bytes", None),
                           ("allocated_bytes", False), ("reserved_bytes", 0.0),
                           ("allocated_bytes", 1), ("reserved_bytes", 1)):
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                self.worker._check_disposal_memory({**valid, key: value})

    def test_disposal_only_wiring_and_memory_validator_use(self):
        tree = ast.parse(self.source)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
        labels = []
        for node in ast.walk(run):
            for body in (getattr(node, "body", []), getattr(node, "orelse", []), getattr(node, "finalbody", [])):
                if not isinstance(body, list):  # Lambda/IfExp bodies are expressions.
                    continue
                for index, statement in enumerate(body):
                    if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                            and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "clear_cuda"):
                        self.assertGreater(index, 0)
                        self.assertEqual(ast.dump(body[index - 1]), ast.dump(ast.parse("model = None").body[0]))
                        labels.append(statement.value.args[0].value)
        self.assertCountEqual(labels, ["before-sft-reload", "before-final-reload", "after-final-inference"])
        wrapper = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == "clear_cuda")
        self.assertEqual([ast.dump(n) for n in wrapper.body], [ast.dump(n) for n in ast.parse(
            "_clear_cuda_after_disposal(torch, label=label, save=save, stamp=stamp)").body])
        validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_outputs")
        memory_loop = next(n for n in ast.walk(validator) if isinstance(n, ast.For)
                           and isinstance(n.target, ast.Name) and n.target.id == "label")
        self.assertEqual(ast.literal_eval(memory_loop.iter),
                         ("before-sft-reload", "before-final-reload", "after-final-inference"))
        self.assertEqual(ast.dump(memory_loop.body[1]), ast.dump(ast.parse("_check_disposal_memory(memory)").body[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
