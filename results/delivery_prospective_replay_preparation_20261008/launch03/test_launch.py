"""Fabricated launch boundary checks only. No real replay, model or scheduler."""
import base64,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('launch_under_test',HERE/'supervise_replay.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda b:hashlib.sha256(b).hexdigest()
DEPS={'torch':'2.9.1','numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','sympy':'1.14.0','PyYAML':'6.0.3'}
VALID={'full_supplemental_replay':True,'checkpoints_replayed':640,'primary_cells_recomputed':240,'cached_responses_checked':48000,'new_optimizer_updates':0,'new_responses':0,'runtime':DEPS,'integrity':{'manifest_sha256':'testpin','files_verified':3890,'bindings_verified':5065}}
class Tests(unittest.TestCase):
 def test_explicit_rejection_under_optimization(self):
  command=[sys.executable,'-I','-O','-c',"import sys;namespace={'__name__':'notmain','__file__':sys.argv[1]};exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),namespace);namespace['require'](False,'explicit rejection')",str(HERE/'supervise_replay.py')]
  r=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('explicit rejection',r.stderr)
 def test_scheduler_fields(self):
  reg={'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','rss_bytes':3*2**30}
  fields={'JobId':'123','Account':'ucb736_asc1','Partition':'acpu','QOS':'cpu-normal','NumNodes':'1','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1','TimeLimit':'00:15:00','MinMemoryNode':'3G','JobState':'RUNNING','AllocTRES':'cpu=1,mem=3G,node=1'}
  env={'SLURM_CPUS_PER_TASK':'1','SLURM_NTASKS':'1','SLURM_JOB_NUM_NODES':'1'}
  def run(f):
   with patch.dict(os.environ,env),patch.object(m.subprocess,'check_output',return_value=' '.join(k+'='+v for k,v in f.items())):return m.allocation('123',reg)
  self.assertEqual(run(fields)['verified_fields']['Account'],'ucb736_asc1')
  for k,v in [('Account','other'),('NumCPUs','2'),('NumTasks','2'),('NumNodes','2'),('TimeLimit','00:30:00'),('MinMemoryNode','4G'),('QOS','other'),('AllocTRES','cpu=1,gres/gpu=1'),('JobId','999'),('AllocTRES',''),('AllocTRES','Unknown'),('AllocTRES','cpu=1,node=1'),('AllocTRES','cpu=1,mem=4G,node=1'),('AllocTRES','cpu=1,mem=3G,node=1,cpu=1'),('AllocTRES','cpu=1,mem=3G,node=1,gres/gpu:a100=1')]:
   with self.subTest(k=k),self.assertRaises(ValueError):run(dict(fields,**{k:v}))
 def test_captured_cli_and_verifier(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cli=b"from verify_delivery_release import VALUE\nprint(VALUE)\n";verifier=b"VALUE='captured-authenticated'\n"
   (root/'replay_delivery_prospective_release.py').write_text("raise RuntimeError('substituted CLI')");(root/'verify_delivery_release.py').write_text("raise RuntimeError('substituted verifier')")
   r=subprocess.run([sys.executable,'-I','-B','-c',m.CHILD_BOOTSTRAP,str(root),'pin',str(root/'receipt'),base64.b64encode(cli).decode(),base64.b64encode(verifier).decode()],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'captured-authenticated')
 def test_receipt_rejections(self):
  reg={'manifest_sha256':'testpin','exact_dependencies':DEPS};self.assertEqual(m.validate_receipt(json.dumps(VALID).encode(),reg),sha(json.dumps(VALID).encode()))
  for k,v in [('checkpoints_replayed',639),('new_responses',1),('new_optimizer_updates',1),('full_supplemental_replay',False),('cached_responses_checked',True),('runtime',{'torch':'2.5.1'})]:
   with self.subTest(k=k),self.assertRaises(ValueError):m.validate_receipt(json.dumps(dict(VALID,**{k:v})).encode(),reg)
 def fixture(self,root,mode):
  pkg=root/'package';pkg.mkdir();entries=[]
  for n,b in [('replay_delivery_prospective_release.py',b'# harmless fabricated CLI'),('verify_delivery_release.py',b'# harmless fabricated verifier')]:
   (pkg/n).write_bytes(b);entries.append({'path':n,'sha256':sha(b)})
  entries.extend({'path':f'fabricated/{i}','sha256':'0'*64} for i in range(3888));manifest=json.dumps({'files':entries}).encode();(pkg/'manifest.json').write_bytes(manifest)
  receipt=dict(VALID,integrity=dict(VALID['integrity'],manifest_sha256=sha(manifest)))
  body="raise OSError('fabricated spawn failure')" if mode=='raise' else "return {'status':'complete','exit_code':0}" if mode=='missing' else "Path(command[7]).write_text('invalid JSON');return {'status':'complete','exit_code':0}" if mode=='malformed' else "Path(command[7]).write_text("+repr(json.dumps(receipt))+ ");return {'status':'complete','exit_code':0}"
  source=('from pathlib import Path\ndef supervise(command,*args,**kwargs):\n '+body+'\n').encode();(root/'supervisor.py').write_bytes(source)
  reg={'output':str(root),'package':str(pkg),'account':'ucb736_asc1','cpus':1,'rss_bytes':3*2**30,'wall_seconds':900,'new_fits':0,'new_responses':0,'manifest_sha256':sha(manifest),'supervision_worker_sha256':sha((HERE/'supervise_replay.py').read_bytes()),'original_supervisor':str(root/'supervisor.py'),'original_supervisor_sha256':sha(source),'exact_dependencies':DEPS}
  freeze=json.dumps(reg).encode();(root/'freeze.json').write_bytes(freeze);return freeze
 def call(self,root,freeze):
  with patch.object(sys,'argv',['wrapper','--freeze',str(root/'freeze.json'),'--freeze-sha256',sha(freeze)]),patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(m,'allocation',return_value={'verified_fields':{'Account':'ucb736_asc1'}}):return m.main()
 def test_terminal_failures_and_success(self):
  for mode in ['raise','missing','malformed','valid']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
    root=Path(td);freeze=self.fixture(root,mode)
    if mode=='valid':self.call(root,freeze)
    else:
     with self.assertRaises(SystemExit):self.call(root,freeze)
    d=json.loads((root/'replay_execution.json').read_bytes());self.assertEqual(d['status'],'complete' if mode=='valid' else 'wrapper_failure');self.assertTrue((root/'replay_started.json').is_file())
    if mode!='valid':self.assertTrue(d['failure_reason']);self.assertEqual(d['exit_code'],1)
 def test_existing_terminal_prevents_work(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);freeze=self.fixture(root,'valid');(root/'replay_execution.json').write_text('preserved sentinel')
   with self.assertRaises(ValueError):self.call(root,freeze)
   self.assertFalse((root/'replay_started.json').exists());self.assertEqual((root/'replay_execution.json').read_text(),'preserved sentinel')
if __name__=='__main__':unittest.main()
