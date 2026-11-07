"""Pure diagnostics tests. No scientific runtime, model or remote execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATH = Path(__file__).with_name("qualify_llm_intervention_threading.py")
SPEC = importlib.util.spec_from_file_location("threading_diagnostic", PATH)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class DiagnosticTests(unittest.TestCase):
    def test_explicit_child_does_not_mutate_parent(self):
        original = {"MKL_THREADING_LAYER":"INTEL","MKL_SERVICE_FORCE_INTEL":"1",
                    "LD_PRELOAD":"bad","PYTHONPATH":"bad","CUDA_VISIBLE_DEVICES":"0","KEEP":"yes"}
        saved = dict(original)
        result = d.explicit_environment(original,Path("/copied"))
        self.assertEqual(original,saved)
        self.assertEqual(result["MKL_THREADING_LAYER"],"GNU")
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"],"")
        self.assertEqual(result["LD_LIBRARY_PATH"],"/copied/lib")
        self.assertEqual(result["KEEP"],"yes")
        for key in ("MKL_SERVICE_FORCE_INTEL","LD_PRELOAD","PYTHONPATH","PYTHONHOME"):
            self.assertNotIn(key,result)

    def test_c_environment_hidden_from_python_is_observed(self):
        code = '''import ctypes,importlib.util,json,os,sys
s=importlib.util.spec_from_file_location('d',sys.argv[1]);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
os.environ.pop('MKL_THREADING_LAYER',None)
c=ctypes.CDLL(None);c.setenv.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];c.setenv.restype=ctypes.c_int
assert c.setenv(b'MKL_THREADING_LAYER',b'INTEL',1)==0
before=d.environment();env=dict(os.environ);os.environ.clear();os.environ.update(env);after=d.environment()
assert before['python']['MKL_THREADING_LAYER'] is None and before['c']['MKL_THREADING_LAYER']=='INTEL'
assert after['python']['MKL_THREADING_LAYER'] is None and after['c']['MKL_THREADING_LAYER']=='INTEL'
print(json.dumps({'hidden_before':True,'hidden_after_clear_update':True}))
'''
        value = subprocess.check_output([sys.executable,"-I","-B","-c",code,str(PATH)],text=True)
        self.assertEqual(json.loads(value),{"hidden_before":True,"hidden_after_clear_update":True})

    def test_narrow_environment_allowlist(self):
        result = d.environment()
        self.assertEqual(set(result),{"python","c"})
        self.assertEqual(set(result["python"]),set(d.KEYS))
        self.assertEqual(set(result["c"]),set(d.KEYS))
        self.assertNotIn("HOME",result["c"])

    def test_allocation_and_negative_contracts(self):
        raw = "JobId=99 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:10:00 Requeue=0 Restarts=0 AllocTRES=billing=1,cpu=2,mem=6G,node=1"
        self.assertEqual(d.allocation(raw,"99")["NumCPUs"],"2")
        for old,new in (("cpu=2","cpu=1"),("NumCPUs=2","NumCPUs=4"),("mem=6G","mem=12G"),
                        ("ucb736_asc2","ucb736_asc1"),("Requeue=0","Requeue=1"),
                        ("node=1","node=1,gres/gpu:a100=1"),("00:10:00","00:20:00")):
            with self.subTest(new=new),self.assertRaises(ValueError):
                d.allocation(raw.replace(old,new),"99")

    def test_receipts_exclusive_and_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"receipt.json"
            self.assertEqual(len(d.save(path,{"observed":True})),64)
            with self.assertRaises(FileExistsError):
                d.save(path,{"changed":True})
            with self.assertRaises(ValueError):
                d.save(Path(temp)/"nonfinite.json",{"value":float("nan")})


if __name__=="__main__":
    unittest.main()
