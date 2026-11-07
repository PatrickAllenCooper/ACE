import datetime, hashlib, json, os, pathlib, resource, signal, subprocess, sys, time
root=pathlib.Path(__file__).resolve().parent
command=['/Users/pat/code/ACE/.venv_test/bin/python','-B',str(root/'source/test_llm_intervention_screen_policy.py')]
overrides={'ACE_SCREEN_POLICY_TEST_OUTPUT':str(root/'fixtures'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
env=dict(os.environ,**overrides)
env.pop('PYTHONPATH',None)
def write(name,value):
    with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
meta={'command':command,'cwd':'/tmp','environment_overrides':overrides,'pythonpath_removed':True,'wall_cap_seconds':60,'scope':'ONE invocation of nine NEW artificial public-envelope fixture methods, no environment/world generation or pretrained/target runtime imports','source_sha256':hashlib.sha256((root/'source/llm_intervention_screen_policy.py').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256((root/'source/test_llm_intervention_screen_policy.py').read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write('command.json',meta)
start=time.monotonic(); own=time.process_time(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
status='running'; error=None; child=None
try:
    with (root/'stdout.txt').open('x') as out,(root/'stderr.txt').open('x') as err:
        child=subprocess.Popen(command,cwd='/tmp',env=env,stdout=out,stderr=err,start_new_session=True)
        try:
            code=child.wait(timeout=60)
            status='passed' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            code=child.wait();status='timeout'
except BaseException as exc:
    error={'type':type(exc).__name__,'message':str(exc)};status='supervisor_failed'
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=None if child is None else child.returncode
after=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt={**meta,'status':status,'returncode':code,'exception':error,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'peak_child_rss_bytes_macos':after.ru_maxrss,'supervisor_cpu_seconds':time.process_time()-own,'cost_scope':'Command child and waited descendants. Peak child RSS is not summed simultaneous tree RSS. Excludes implementation/source preparation/review/custody work. No cumulative allowance reset.','new_environment_responses':0,'scientific_worlds_realized':0,'pretrained_models':0,'remote_actions':0,'new_allocations':0,'old_suites_repeated':0}
write('receipt.json',receipt)
print(json.dumps(receipt,indent=2))
print((root/'stderr.txt').read_text()[-12000:])
sys.exit(0 if status=='passed' else 1)
