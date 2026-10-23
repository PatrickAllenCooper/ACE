from pathlib import Path
import json,hashlib,subprocess,datetime
p=Path(__file__).parent
pin=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
review=p/'packet_review.md'
assert 'Required findings remaining: 0' in review.read_text()
pins=json.loads((p/'input_pins.json').read_text())
assert all(pin(p/n)==h for n,h in pins.items())
assert json.loads((p/'remote_input_pins.json').read_text())['pins']==pins
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/Users/pat/code/ACE',text=True).strip()
intent={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packet_commit':commit,'review_sha256':pin(review),'review_required_unresolved':0,'retry_allowed':False,'pins':pins,'authorization_sha256':pin(p/'authorization.json'),'predecessor_job':34824953,'single_attempt':True}
with (p/'submission_started.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n')
s=(p/'submit_remote_template.py').read_text().replace("intent=json.loads(os.environ['ACE_APPROVED_SUBMISSION_INTENT'])",'intent='+repr(intent))
with (p/'submit_remote_once.py').open('x') as f:f.write(s)
try:
 r=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','/scratch/alpine/paco0228/ACE/envs/neuronbench_py311/bin/python -I -B -'],input=s,text=True,capture_output=True,timeout=55)
 for n,raw in [('submission_transport.stdout',r.stdout),('submission_transport.stderr',r.stderr)]:
  with (p/n).open('x') as f:f.write(raw)
 record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'retry_allowed':False}
 with (p/'submission_transport.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 if r.returncode:raise RuntimeError('transport failed; reconcile remote journal, no automatic retry')
 result=json.loads(r.stdout)
 with (p/'submission_result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
except BaseException as exc:
 with (p/'local_submission_exception.json').open('x') as f:json.dump({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'type':type(exc).__name__,'message':str(exc),'retry_allowed':False},f)
 raise
