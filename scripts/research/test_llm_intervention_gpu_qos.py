"""Scheduler-diagnosed testing-QOS correction, no GPU allocation."""
import unittest
from supervise_llm_intervention_gpu import authenticate_allocation
from test_llm_intervention_gpu_packet import RAW, RESOURCES

class QOSTests(unittest.TestCase):
    def test_documented_testing_qos_and_wrong_normal_rejection(self):
        value = authenticate_allocation(RAW, "123", RESOURCES)
        self.assertEqual(value["fields"]["QOS"], "gpu-testing")
        for raw, resource in ((RAW.replace("gpu-testing", "gpu-normal"), RESOURCES),
                              (RAW, dict(RESOURCES, qos="gpu-normal"))):
            with self.assertRaises(ValueError):
                authenticate_allocation(raw, "123", resource)

if __name__ == "__main__":
    unittest.main()
