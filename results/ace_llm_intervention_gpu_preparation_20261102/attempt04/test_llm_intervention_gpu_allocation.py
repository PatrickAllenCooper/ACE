"""Focused allocation delta from actual Slurm task CPU rounding evidence."""
import unittest
from supervise_llm_intervention_gpu import authenticate_allocation, scheduler_fields
from test_llm_intervention_gpu_packet import RAW, RESOURCES

class AllocationDeltaTests(unittest.TestCase):
    def test_requested_cpus_distinct_from_bounded_actual_allocation(self):
        raw = RAW.replace("NumCPUs=4", "NumCPUs=8").replace("CPUs/Task=4", "CPUs/Task=8").replace("AllocTRES=cpu=4", "AllocTRES=cpu=8")
        self.assertEqual(authenticate_allocation(raw, "123", RESOURCES)["allocated_resources"]["cpu"], "8")
        self.assertEqual(scheduler_fields(raw+" MCS_label=N/A")["MCS_label"], "N/A")
        for old, new in [("ReqTRES=cpu=4", "ReqTRES=cpu=8"), ("CPUs/Task=8", "CPUs/Task=4"),
                         ("AllocTRES=cpu=8", "AllocTRES=cpu=16")]:
            with self.assertRaises(ValueError):
                authenticate_allocation(raw.replace(old, new), "123", RESOURCES)

if __name__ == "__main__":
    unittest.main()
