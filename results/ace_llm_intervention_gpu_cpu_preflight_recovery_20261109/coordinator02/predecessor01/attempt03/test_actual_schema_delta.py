"""One new fabricated actual-schema boundary. No actual records accepted or executed."""
import copy,importlib.util,pathlib,unittest
HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('new_schema_fixture',HERE/'test_recovery_coordinator.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
class SchemaDelta(unittest.TestCase):
 def test_actual_reconciliation_schema_preserves_failed_status_and_no_work(self):
  m,p=f.fixture();self.assertTrue(f.b.failed_gpu_history(m,p))
  for name,value in [('new_scientific_responses',1),('new_policy_attempts',1),('original_collector_returncode',0),
                     ('retroactive_collector_success',True),('required_findings',False),('closed_cpu_seconds',1071)]:
   with self.subTest(field=name):
    x=copy.deepcopy(m);x['gpu-failure-acceptance'][name]=value
    with self.assertRaises(ValueError):f.b.failed_gpu_history(x,p)
  x=copy.deepcopy(m);x['gpu-failure-acceptance']['input_pins']['original_failure_seal']['sha256']=f.H('changed')
  with self.assertRaises(ValueError):f.b.failed_gpu_history(x,p)
if __name__=='__main__':unittest.main(verbosity=2)
