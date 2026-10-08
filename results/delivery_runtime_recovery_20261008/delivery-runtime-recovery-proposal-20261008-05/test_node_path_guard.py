import importlib.util,tempfile,sys,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('proposal',Path(__file__).parent/'prepare_environment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class NodePathGuard(unittest.TestCase):
 def test_alias_and_byte_mismatch_diagnostics(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td).resolve();real=root/'real';real.mkdir();binary=real/'python3.11';binary.write_bytes(b'fixture binary not executed');alias=root/'alias';alias.symlink_to(real,target_is_directory=True);request=alias/'python';request.symlink_to('python3.11')
   expected=alias/'python3.11';reg={'base_python':str(request),'base_python_resolved':str(expected),'base_python_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()};orig=m.sha
   with patch.object(m,'sha',side_effect=lambda p:reg['base_python_sha256'] if str(p)=='/proc/self/exe' else orig(p)):
    record=m.base_identity(reg);self.assertTrue(record['path_matches']);m.validate_base_identity(record)
    record['byte_match']=False
    with self.assertRaisesRegex(ValueError,'node_resolved_requested'):m.validate_base_identity(record)
    record['byte_match']=True;record['running_byte_match']=False
    with self.assertRaisesRegex(ValueError,'running_elf_sha256'):m.validate_base_identity(record)
if __name__=='__main__':unittest.main(verbosity=2)
