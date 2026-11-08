from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,shutil,subprocess,time
repo=Path('/Users/pat/code/ACE')
ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-plan-preparation-20261108-02')
local=Path(__file__).parent/'attempt02'
ext.mkdir();local.mkdir();(ext/'source').mkdir();(ext/'fixtures').mkdir()
names=['llm_intervention_training_plan.py','test_llm_intervention_training_plan.py']
for n in names:shutil.copyfile(repo/'scripts/research'/n,ext/'source'/n)
shutil.copyfile(__file__,ext/'run_preparation.py')
locks=[]
try:
 for name in ['ace-delivery-controller.lock','ace-foundation-controller.lock']:
  f=open(ext.parent/name,'r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 command=[str(repo/'.venv_test/bin/python'),'-B','-m','unittest','-v','test_llm_intervention_training_plan.TrainingPlanTests.test_rebound_bytes_cannot_hide_inconsistent_semantic_receipts','test_llm_intervention_training_plan.TrainingPlanTests.test_world_accounting_flags_and_balanced_count_errors_reject']
 env=dict(os.environ,PYTHONPATH=str(repo/'scripts/research'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',ACE_BRIDGE_TEST_OUTPUT=str(ext/'fixtures'))
 start=time.monotonic();p=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=60)
 r=resource.getrusage(resource.RUSAGE_CHILDREN)
 value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'cwd':'/tmp','returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'child_cpu_seconds':r.ru_utime+r.ru_stime,'elapsed_seconds':time.monotonic()-start,'peak_child_rss_bytes':r.ru_maxrss,'source_sha256':{n:hashlib.sha256((ext/'source'/n).read_bytes()).hexdigest() for n in names},'scope':'Two new focused review regressions, five negative semantic cases and a positive unchanged-artifact check on saved artificial records. No model weights, CUDA, optimization, collection or scientific execution. Separate preparation cost, not whole cycle.'}
 (ext/'receipt.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
 print(json.dumps(value,indent=2))
finally:
 for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
 for p in ext.rglob('*'):
  target=local/p.relative_to(ext)
  if p.is_dir():target.mkdir(exist_ok=True)
  else:shutil.copyfile(p,target)
