import hashlib,importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('audit_saved_cpu.py')
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'bd3943ef7f1e817d831397fb338a4c641b0cae06c7d93a6de4f2cba490da637b'
s=importlib.util.spec_from_file_location('aid',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ContainerGuard(unittest.TestCase):
 def test_original_container_type(self):
  expected={'files':{},'directories':['common','launch']}
  for value in ({'common':'discarded','launch':'discarded'},'ab',('common','launch'),None):
   with self.assertRaises(ValueError):m.normalized_observation_inventory({'files':{},'directories':value})
  self.assertEqual(expected,m.normalized_observation_inventory({'files':{},'directories':['launch','common']}))
if __name__=='__main__':unittest.main()
