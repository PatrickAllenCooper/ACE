"""Focused GNU handoff regression. Fresh disposable process, no model/runtime."""
from pathlib import Path
import subprocess
import sys
import unittest

SOURCE = Path(__file__).with_name("qualify_llm_intervention_model_service.py")


class ThreadingHandoffTests(unittest.TestCase):
    def run_case(self, body):
        code = '''import ctypes,importlib.util,json,os,subprocess,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('gpu',sys.argv[1]);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
libc=ctypes.CDLL(None);libc.setenv.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];libc.setenv.restype=ctypes.c_int
os.environ['CUDA_VISIBLE_DEVICES']='MIG-artificial'
for key,value in [('MKL_THREADING_LAYER','INTEL'),('MKL_SERVICE_FORCE_INTEL','1'),('LD_PRELOAD','bad-hidden-library')]:
 os.environ.pop(key,None);assert libc.setenv(key.encode(),value.encode(),1)==0
''' + body
        subprocess.run([sys.executable,"-I","-B","-c",code,str(SOURCE)],check=True,capture_output=True,text=True)

    def test_hidden_c_values_overwritten_before_inherited_exec(self):
        self.run_case('''
env=g.child_environment(Path('/artificial-prefix'),os.environ)
record=g.install_child_environment(env)
assert record['before']['python']['MKL_THREADING_LAYER'] is None
assert record['before']['c']['MKL_THREADING_LAYER']=='INTEL'
assert record['before']['c']['MKL_SERVICE_FORCE_INTEL']=='1'
expected={'MKL_THREADING_LAYER':'GNU','MKL_SERVICE_FORCE_INTEL':None,'CUDA_VISIBLE_DEVICES':'MIG-artificial'}
assert record['after']['c']==record['after']['python']==expected
child='import os,json;print(json.dumps({k:os.environ.get(k) for k in '+repr(tuple(expected))+"}))"
got=json.loads(subprocess.check_output([sys.executable,'-I','-B','-c',child],text=True,env=None))
assert got==expected
assert 'LD_PRELOAD' not in os.environ
''')

    def test_bad_environment_rejected_before_changes(self):
        self.run_case('''
env=g.child_environment(Path('/artificial-prefix'),os.environ)
for key,value in [('MKL_THREADING_LAYER','INTEL'),('MKL_SERVICE_FORCE_INTEL',''),('CUDA_VISIBLE_DEVICES',''),('LD_PRELOAD','bad')]:
 bad=dict(env);bad[key]=value
 try:g.install_child_environment(bad)
 except ValueError:pass
 else:raise AssertionError('bad environment accepted')
assert os.environ.get('MKL_THREADING_LAYER') is None
''')


if __name__=="__main__":
    unittest.main()
