"""Focused artificial layout controls; no numeric runtime or prior tests run."""
import ast
import builtins
import copy
from pathlib import Path
import types
import unittest


SOURCE = Path(__file__).with_name("llm_intervention_training_gpu_worker.py")


class ScoringLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imports = []
        real_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            cls.imports.append(name)
            if name not in {"__future__", "gc", "hashlib", "json", "math", "os", "pathlib", "time", "traceback"}:
                raise AssertionError("non-stdlib import in pure layout test: " + name)
            return real_import(name, *args, **kwargs)

        cls.worker = types.ModuleType("captured_scoring_layout_worker")
        cls.worker.__dict__["__builtins__"] = {**vars(builtins), "__import__": stdlib_only}
        cls.source = SOURCE.read_bytes()
        exec(compile(cls.source, str(SOURCE), "exec"), cls.worker.__dict__)

    @staticmethod
    def record():
        # Artificial token IDs; the diagnosed command lengths are 7/7/7/6.
        return {"prompt_ids": [90, 91, 92], "pad_token_id": 0, "commands": {
            '{"action_id":"a"}': [1, 2, 3, 4, 5, 6, 99],
            '{"action_id":"b"}': [1, 2, 3, 4, 5, 7, 99],
            '{"action_id":"c"}': [1, 2, 3, 4, 5, 8, 99],
            '{"stop":true}': [1, 2, 3, 4, 9, 99],
        }}

    def test_actual_7776_lengths_use_four_unpadded_singleton_calls(self):
        record = self.record()
        commands = list(record["commands"])
        self.assertEqual([len(record["commands"][c]) for c in commands], [7, 7, 7, 6])
        layout = self.worker.scoring_layout(record, commands, mode="singletons")
        self.assertEqual(layout, tuple((c,) for c in commands))
        # Shared prompt plus one command fills every row: no padding positions.
        for group in layout:
            widths = [len(record["prompt_ids"]) + len(record["commands"][c]) for c in group]
            self.assertEqual(sum(max(widths) - width for width in widths), 0)
        self.assertEqual(len(layout), 4)  # Equal-length actions must not regroup.
        self.assertEqual(self.worker.scoring_layout(record, commands[:1], mode="singletons"), ((commands[0],),))

    def test_supplied_order_and_identical_before_after_layout_are_preserved(self):
        record = self.record()
        a, b, c, stop = record["commands"]
        commands = [stop, c, a, b]
        original = copy.deepcopy((record, commands))
        before = self.worker.scoring_layout(record, commands, mode="singletons")
        reloaded = copy.deepcopy(record)
        reloaded["commands"] = dict(reversed(list(reloaded["commands"].items())))
        after = self.worker.scoring_layout(reloaded, tuple(commands), mode="singletons")
        self.assertEqual(before, ((stop,), (c,), (a,), (b,)))
        self.assertEqual(after, before)
        self.assertEqual((record, commands), original)
        commands.reverse()
        self.assertEqual(before, ((stop,), (c,), (a,), (b,)))

    def test_reference_pair_keeps_chosen_rejected_order_in_one_call(self):
        record = self.record()
        a, b, c, stop = record["commands"]
        self.assertEqual(self.worker.scoring_layout(record, [c, b], mode="reference_pair"), ((c, b),))
        self.assertEqual(self.worker.scoring_layout(record, [b, c], mode="reference_pair"), ((b, c),))
        # Positive equal lengths, including the smallest valid token sequence.
        self.assertEqual(self.worker.scoring_layout({"commands": {a: [0], b: [1]}},
                         (a, b), mode="reference_pair"), ((a, b),))
        for commands in ([], [a], [a, b, c], [a, stop], [stop, a]):
            with self.subTest(commands=commands), self.assertRaises(ValueError):
                self.worker.scoring_layout(record, commands, mode="reference_pair")
        for lengths in ((0, 0), (0, 1), (1, 0), (6, 7)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                self.worker.scoring_layout({"commands": {a: [1] * lengths[0], b: [2] * lengths[1]}},
                                           [a, b], mode="reference_pair")

    def test_invalid_modes_fail_closed(self):
        record = self.record()
        for mode in (None, "", "batch", "auto", "singleton", 1, True, [], {}):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "layout mode"):
                self.worker.scoring_layout(record, list(record["commands"]), mode=mode)

    def test_invalid_commands_and_token_tables_fail_closed(self):
        record = self.record()
        a, b, c, stop = record["commands"]
        invalid_commands = (None, [], (), a, {a}, {a: 1}, iter([a]), ["missing"], [a, "missing"],
                            [a, a], [a, b, a], [""], [1], [True], [[]], [a, None])
        for commands in invalid_commands:
            for mode in ("singletons", "reference_pair"):
                with self.subTest(commands=repr(commands), mode=mode), self.assertRaises(ValueError):
                    self.worker.scoring_layout(record, commands, mode=mode)
        for bad_record in (None, [], {}, {"commands": None}, {"commands": []}, {"commands": {}}):
            with self.subTest(record=bad_record), self.assertRaises(ValueError):
                self.worker.scoring_layout(bad_record, [a], mode="singletons")
        for tokens in (None, [], (), "1234567", [False], [1.0], [-1], [[1]], [1, None]):
            changed = copy.deepcopy(record)
            changed["commands"][a] = tokens
            for mode in ("singletons", "reference_pair"):
                with self.subTest(tokens=tokens, mode=mode), self.assertRaises(ValueError):
                    self.worker.scoring_layout(changed, [a, b], mode=mode)

    def test_scoring_call_sites_bind_explicit_layouts_and_keep_first_singleton(self):
        # Inspect wiring without importing/constructing torch, a model or backend.
        tree = ast.parse(self.source)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
        assignments = {n.targets[0].id: n.value for n in ast.walk(run)
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.unparse(assignments["final_scores"]),
                         "scores(model, final_specs, mode='singletons')")
        self.assertEqual(ast.unparse(assignments["specs"]), "pairs if checkpoint == 'sft' else final_specs")
        self.assertEqual(ast.unparse(assignments["mode"]),
                         "'reference_pair' if checkpoint == 'sft' else 'singletons'")
        self.assertEqual(ast.unparse(assignments["observed"]), "scores(model, specs, mode=mode)")
        self.assertEqual(ast.unparse(assignments["first_scores"]),
                         "scores(model, {'first': (records[0], [records[0]['supervised_command']])})['first']")
        score = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == "scores")
        self.assertEqual(score.args.kwonlyargs[0].arg, "mode")
        self.assertEqual(score.args.kw_defaults[0].value, "singletons")
        # The helper's validated groups, not the original multi-command roster,
        # feed the existing packer and canonical scorer; flatten in call order.
        self.assertEqual(ast.dump(score.body[0].value), ast.dump(ast.parse(
            "{key: scoring_layout(rec, commands, mode=mode) for key, (rec, commands) in specs.items()}",
            mode="eval").body))
        group_loop = next(n for n in ast.walk(score) if isinstance(n, ast.For)
                          and isinstance(n.target, ast.Name) and n.target.id == "group")
        self.assertEqual(ast.unparse(group_loop.iter), "layouts[key]")
        self.assertEqual(ast.unparse(group_loop.body[0].value),
                         "pack_actions(rec['prompt_ids'], [rec['commands'][c] for c in group], rec['pad_token_id'], 8192)")
        self.assertEqual(ast.unparse(group_loop.body[-1]), "result[key].extend(value.detach().cpu().tolist())")
        self.assertFalse(any(name.startswith(("torch", "numpy", "transformers", "llm_")) for name in self.imports))


if __name__ == "__main__":
    unittest.main(verbosity=2)
