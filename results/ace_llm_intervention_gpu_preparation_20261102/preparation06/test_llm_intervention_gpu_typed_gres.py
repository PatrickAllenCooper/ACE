"""Regression against the actual GPU02 scheduler record. No model loading."""
import json
from pathlib import Path
import unittest

from supervise_llm_intervention_gpu import authenticate_allocation
from test_llm_intervention_gpu_packet import RESOURCES


class TypedGresTests(unittest.TestCase):
    def test_actual_typed_only_and_optional_agreeing_aggregate(self):
        path = Path(__file__).resolve().parents[2] / (
            "results/ace_llm_intervention_gpu_runtime_20261102/gpu02/terminal/"
            "gpu02/scheduler_raw.json")
        saved = json.loads(path.read_bytes())
        self.assertEqual(saved["returncode"], 0)
        raw = saved["stdout"]
        allocation = authenticate_allocation(raw, saved["job_id"], RESOURCES)
        self.assertNotIn("gres/gpu", allocation["allocated_resources"])
        self.assertEqual(allocation["allocated_resources"]["gres/gpu:a100_3g.20gb"], "1")
        aggregate = raw.replace("AllocTRES=cpu=4", "AllocTRES=gres/gpu=1,cpu=4")
        self.assertEqual(authenticate_allocation(aggregate, saved["job_id"], RESOURCES)
                         ["allocated_resources"]["gres/gpu"], "1")
        for bad in (
            aggregate.replace("AllocTRES=gres/gpu=1", "AllocTRES=gres/gpu=2"),
            raw.replace("gres/gpu:a100_3g.20gb=1", "gres/gpu:a100_3g.20gb=2"),
            raw.replace(",gres/gpu:a100_3g.20gb=1", ""),
            raw.replace("AllocTRES=cpu=4", "AllocTRES=gres/gpu:a100_80gb=1,cpu=4"),
            raw.replace("AllocTRES=cpu=4", "AllocTRES=gres/gpu:a100_3g.20gb=1,cpu=4"),
            raw.replace("AllocTRES=cpu=4", "AllocTRES=gres/mps=1,cpu=4"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    authenticate_allocation(bad, saved["job_id"], RESOURCES)


if __name__ == "__main__":
    unittest.main()
