"""Pure fabricated GPU packet checks. No torch/model/GPU import or execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from qualify_llm_intervention_gpu import validate_freeze, SOURCES, MODELS, PACKAGES
from supervise_llm_intervention_gpu import (authenticate_allocation, scheduler_fields, gpu_process_sample,
                                          absolute_deadlines, terminate, group_alive, read_phases, finish_failure)

RESOURCES = {"account": "ucb736_asc1", "partition": "aa100", "qos": "gpu-normal",
             "requested_cpus": 4, "max_allocated_cpus": 8, "memory_mib": 12288,
             "wall_seconds": 900, "gpus": 1, "gpu_type": "a100_3g.20gb"}
RAW = ("JobId=123 UserId=paco0228(100) Account=ucb736_asc1 Partition=aa100 QOS=gpu-normal "
       "JobState=RUNNING NumNodes=1 NumTasks=1 NumCPUs=4 CPUs/Task=4 ReqB:S:C:T=0:0:*:* TimeLimit=00:15:00 "
       "AllocTRES=cpu=4,mem=12G,node=1,billing=28,gres/gpu=1,gres/gpu:a100_3g.20gb=1")


class PacketTests(unittest.TestCase):
    def test_full_allocated_resources_positive_and_units(self):
        result = authenticate_allocation(RAW, "123", RESOURCES)
        self.assertEqual(result["allocated_resources"]["cpu"], "4")
        self.assertEqual(scheduler_fields(RAW)["CPUs/Task"], "4")
        self.assertEqual(scheduler_fields(RAW)["ReqB:S:C:T"], "0:0:*:*")
        authenticate_allocation(RAW.replace("mem=12G", "mem=12288M"), "123", RESOURCES)
        authenticate_allocation(RAW.replace("NumCPUs=4", "NumCPUs=8").replace("cpu=4", "cpu=8"), "123", RESOURCES)

    def test_negative_allocation_variants(self):
        changes = [("JobId=123", "JobId=124"), ("paco0228(", "other("),
                   ("Account=ucb736_asc1", "Account=other"), ("aa100", "al40"),
                   ("gpu-normal", "gpu-long"), ("RUNNING", "PENDING"),
                   ("NumNodes=1", "NumNodes=2"), ("NumTasks=1", "NumTasks=2"),
                   ("CPUs/Task=4", "CPUs/Task=8"), ("00:15:00", "01:00:00"),
                   ("mem=12G", "mem=24G"), ("cpu=4", "cpu=12"),
                   ("gres/gpu=1", "gres/gpu=2"), ("a100_3g.20gb", "a100-40gb"),
                   ("AllocTRES", "ReqTRES")]
        for old, new in changes:
            with self.subTest(old=old), self.assertRaises(ValueError):
                authenticate_allocation(RAW.replace(old, new), "123", RESOURCES)
        for raw in (RAW+",gres/gpu:l40=1", RAW+",cpu=4", RAW+" JobId=123"):
            with self.assertRaises(ValueError):
                authenticate_allocation(raw, "123", RESOURCES)

    def test_fixed_worker_design_and_inventory(self):
        freeze = {"schema": "ace-llm-gpu-smoke-v1", "sources": {n: "a"*64 for n in SOURCES},
                  "model_files": {n: "b"*64 for n in MODELS}, "runtime": {n: "x" for n in PACKAGES},
                  "training": {"seed": 107, "rank": 8, "alpha": 16., "sft_steps": 8,
                    "dpo_steps": 4, "learning_rate": .0005, "beta": .1, "max_tokens": 512, "max_new_tokens": 32},
                  "startup_seconds": 420, "worker_seconds": 840}
        validate_freeze(freeze)
        for section in ("sources", "model_files", "runtime"):
            bad = copy.deepcopy(freeze)
            bad[section].pop(next(iter(bad[section])))
            with self.assertRaises(ValueError):
                validate_freeze(bad)
        for section, key in (("training", "sft_steps"), ("training", "max_tokens")):
            bad = copy.deepcopy(freeze)
            bad[section][key] += 1
            with self.assertRaises(ValueError):
                validate_freeze(bad)

    def test_gpu_sample_excludes_unrelated_processes(self):
        returned = SimpleNamespace(stdout="111, GPU-a, 8000\n222, MIG-b, N/A\n", stderr="", returncode=0)
        with patch("supervise_llm_intervention_gpu.subprocess.run", return_value=returned):
            result = gpu_process_sample(222)
        self.assertEqual(result["owned_rows"], ["222, MIG-b, N/A"])
        with patch("supervise_llm_intervention_gpu.subprocess.run", side_effect=OSError("unavailable")):
            result = gpu_process_sample(222)
        self.assertIsNone(result["returncode"])
        self.assertEqual(result["owned_rows"], [])

    def test_absolute_deadlines_include_supervisor_preparation(self):
        self.assertEqual(absolute_deadlines(0., 1.), {"supervisor_end": 870., "worker_end": 841., "startup_end": 421.})
        self.assertEqual(absolute_deadlines(0., 60.)["worker_end"], 845.)
        self.assertEqual(absolute_deadlines(0., 500.)["startup_end"], 845.)

    def test_cleanup_handles_missing_group_and_error_without_losing_failure(self):
        process = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        with patch("supervise_llm_intervention_gpu.os.killpg", side_effect=ProcessLookupError):
            result = terminate(process, time.monotonic()+1.)
        self.assertFalse(result["group_still_exists"])
        with patch("supervise_llm_intervention_gpu.os.killpg", side_effect=PermissionError("denied")):
            result = terminate(process, time.monotonic()+1.)
        self.assertIsNone(result["group_still_exists"])
        self.assertIn("PermissionError", result["errors"][0])

    def test_actual_owned_process_group_cleanup(self):
        for leader_exits in (False, True):
            code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);print('ready',flush=True)"
            if not leader_exits:
                code += ";time.sleep(30)"
            process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                if leader_exits:
                    process.wait(timeout=3)
                result = terminate(process, time.monotonic()+3.)
                print(json.dumps({"owned_cleanup_fixture": True, "leader_exits": leader_exits, **result}), flush=True)
                self.assertEqual(result["errors"], [])
                self.assertFalse(result["group_still_exists"])
                self.assertIsNotNone(process.poll())
            finally:
                terminate(process, time.monotonic()+2.)
                process.stdout.close()
                process.stderr.close()

    def test_final_phase_scan_reads_new_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/"phase-000.json").write_text('{"phase":"imports_complete"}')
            self.assertEqual(len(read_phases(path)), 1)
            (path/"phase-001.json").write_text('{"phase":"first_gpu_forward_complete","elapsed_seconds":10}')
            (path/"phase-002.json").write_text('{"phase":')
            phases = read_phases(path)
            self.assertEqual(len(phases), 2)
            self.assertEqual(phases[-1]["phase"], "first_gpu_forward_complete")

    def test_failure_receipt_error_still_cleans_and_keeps_deadline(self):
        process = SimpleNamespace(pid=123, returncode=-15)
        cleanup = {"group_still_exists": False, "errors": []}
        with patch("supervise_llm_intervention_gpu.arm_final_deadline") as arm, \
             patch("supervise_llm_intervention_gpu.terminate", return_value=cleanup) as stop, \
             patch("supervise_llm_intervention_gpu.save", side_effect=[OSError("full"), None]) as write:
            result = finish_failure(Path("/unused"), {"status": "failed", "message": "original"},
                                    process, time.monotonic(), time.process_time())
        arm.assert_called_once()
        stop.assert_called_once()
        self.assertEqual(write.call_count, 2)
        self.assertEqual(result["message"], "original")
        self.assertEqual(result["cleanup"], cleanup)
        self.assertIn("OSError", result["persistence_errors"][0])


if __name__ == "__main__":
    unittest.main()
