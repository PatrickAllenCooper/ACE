"""New decoder integration and diagnostic fixtures, no pretrained model."""
import copy
import json
from pathlib import Path
import tempfile
import time
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm_intervention_contract import Action
from qualify_llm_intervention_gpu import build_decoding_fixture, prefix_callback, validate_freeze, SOURCES, MODELS, PACKAGES
from prepare_llm_intervention_decoding_cpu import check_processor
from supervise_llm_intervention_gpu import finish_failure


class CharacterTokenizer:
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))
    def decode(self, tokens, skip_special_tokens=False):
        assert not skip_special_tokens
        return "".join(map(chr, tokens))


class DecodingIntegrationTests(unittest.TestCase):
    def fixture(self):
        menu = (Action("a0", "intervene", (("X", -1.),), ("Y",), 1),
                Action("a1", "intervene", (("X", 1.),), ("Y",), 1))
        fixture = {"prompt_ids": [250, 251], "state": {"remaining_units": 1}}
        return menu, fixture

    def test_decoder_descriptor_cap_and_all_commands(self):
        menu, fixture = self.fixture()
        grammar, descriptor = build_decoding_fixture(CharacterTokenizer(), menu, fixture, 32)
        self.assertEqual(descriptor["commands"], ['{"action_id":"a0"}', '{"action_id":"a1"}', '{"stop":true}'])
        self.assertEqual(descriptor["longest_completion_tokens"], 19)
        for cap in (1, 18, True):
            with self.assertRaises(ValueError):
                build_decoding_fixture(CharacterTokenizer(), menu, fixture, cap)

    def test_actual_local_transformers_prefix_processor(self):
        menu, fixture = self.fixture()
        grammar, _ = build_decoding_fixture(CharacterTokenizer(), menu, fixture, 32)
        result = check_processor(grammar, 256)
        self.assertEqual(result["commands_checked"], 3)
        self.assertTrue(result["allowed_logits_preserved"])
        self.assertTrue(result["outside_tokens_negative_infinity"])
        with self.assertRaises(ValueError): check_processor(grammar, 128)

    def test_callback_rejects_wrong_prompt_batch_and_completed_sequence(self):
        import torch
        menu, fixture = self.fixture()
        grammar, _ = build_decoding_fixture(CharacterTokenizer(), menu, fixture, 32)
        callback = prefix_callback(grammar)
        for batch, ids in ((1, grammar.prompt_ids), (False, grammar.prompt_ids),
                           (0, (249, 251)), (0, grammar.prompt_ids+grammar.completion_ids[0])):
            with self.assertRaises(ValueError): callback(batch, torch.tensor(ids))

    def test_freeze_requires_new_representation_and_grammar_source(self):
        freeze = {"schema": "ace-llm-gpu-smoke-v2", "decoding": "finite-public-menu-v1",
            "sources": {n: "a"*64 for n in SOURCES}, "model_files": {n: "b"*64 for n in MODELS},
            "runtime": {n: "x" for n in PACKAGES}, "startup_seconds": 420, "worker_seconds": 840,
            "training": {"seed": 107, "rank": 8, "alpha": 16., "sft_steps": 8, "dpo_steps": 4,
                         "learning_rate": .0005, "beta": .1, "max_tokens": 512, "max_new_tokens": 32}}
        validate_freeze(freeze)
        for key, value in (("schema", "ace-llm-gpu-smoke-v1"), ("decoding", "unconstrained")):
            with self.assertRaises(ValueError): validate_freeze(dict(freeze, **{key: value}))
        altered = copy.deepcopy(freeze)
        del altered["sources"]["llm_intervention_grammar.py"]
        with self.assertRaises(ValueError): validate_freeze(altered)

    def test_failure_preserves_cleanup_and_missing_gpu_telemetry(self):
        process = SimpleNamespace(pid=123, returncode=1)
        cleanup = {"group_still_exists": False, "errors": []}
        unavailable = {"returncode": None, "owned_rows": [], "error": "unavailable"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("supervise_llm_intervention_gpu.arm_final_deadline"), \
                 patch("supervise_llm_intervention_gpu.terminate", return_value=cleanup), \
                 patch("supervise_llm_intervention_gpu.gpu_process_sample", return_value=unavailable):
                result = finish_failure(Path(tmp), {"status": "failed", "message": "original"},
                                        process, time.monotonic(), time.process_time())
            self.assertEqual(result["post_cleanup_gpu_sample"], unavailable)
            self.assertEqual(result["cleanup"], cleanup)
            self.assertEqual(json.loads((Path(tmp)/"supervisor_failure.json").read_text())["message"], "original")

    def test_isolated_cpu_entrypoint_rejects_before_sibling_or_framework_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            freeze = path/"invalid-freeze.json"
            freeze.write_text("{}")
            worker = Path(__file__).with_name("prepare_llm_intervention_decoding_cpu.py")
            result = subprocess.run([sys.executable, "-I", "-B", str(worker), "--freeze", str(freeze),
                "--freeze-sha256", "0"*64, "--output", str(path/"out")],
                cwd=tmp, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            failure = json.loads((path/"out/failure.json").read_text())
            self.assertEqual(failure["exception_type"], "ValueError")
            self.assertEqual(failure["message"], "freeze pin differs")
            self.assertFalse((path/"out/terminal.json").exists())


if __name__ == "__main__":
    unittest.main()
