from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,time,traceback
REPO=Path('/Users/pat/code/ACE')
ROOT=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
LAUNCH=ROOT/'launch01'
OUT=ROOT/'watch01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-watch-20261109-01')
JOB='35498218'
PACKET='436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'
RESULT='7c484f72ecafe9188cf4a71cb7023844c46390eb200ffe7507123d473ba43b86'
raw=lambda v:(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 data=raw(value)
 for d in (OUT,EXT):
  with (d/name).open('xb') as f:f.write(data)
def main():
 assert sha(LAUNCH/'packet.json')==PACKET and sha(LAUNCH/'submission.result.json')==RESULT
 packet=json.loads((LAUNCH/'packet.json').read_bytes())
 assert sha(LAUNCH/'monitor_once.py')==packet['new_input_files']['monitor_once.py']
 EXT.mkdir(exist_ok=False)
 shutil.copyfile(Path(__file__),EXT/'watch.py')
 save('intent.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,packet_sha256=PACKET,result_sha256=RESULT,source_sha256=sha(Path(__file__)),deadline_seconds=3600,maximum_interval_seconds=60,read_only=True,submission=False))
 started=time.monotonic();index=0
 while time.monotonic()-started<3600:
  index+=1;tick=time.monotonic();name=f'observation-{index:04d}';out=OUT/name
  argv=[sys.executable,'-B',str(LAUNCH/'monitor_once.py'),'--job-id',JOB,'--packet-sha256',PACKET,'--submission-result-sha256',RESULT,'--packet-directory',str(LAUNCH),'--evidence-directory',str(out)]
  result=subprocess.run(argv,cwd=REPO,capture_output=True,text=True,timeout=60)
  if out.exists():shutil.copytree(out,EXT/name)
  if result.returncode:
   save('failure.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation=name,returncode=result.returncode,stderr=result.stderr[-8000:],submission_retry=False));raise RuntimeError('read-only observation failed; evidence retained')
  value=json.loads(result.stdout);rows=[o for o in value['observations'] if o['command'][0]=='sacct'];lines=rows[0].get('stdout','').splitlines();parents=[line.split('|') for line in lines if line.split('|')[0]==JOB];state=parents[0][2] if len(parents)==1 else None
  summary=dict(utc=value['utc'],job_id=JOB,state=state,elapsed_seconds=parents[0][5] if parents else None,public_telemetry=value['public_telemetry'],present=[k for k,v in value['evidence'].items() if v.get('present')],acceptance=False)
  print(json.dumps(summary,sort_keys=True),flush=True)
  if state and state.split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}:
   save('terminal-observation.json',dict(last=name,summary=summary,watcher_elapsed_seconds=time.monotonic()-started,acceptance=False));return
  remaining=60-(time.monotonic()-tick)
  while remaining>0:
   delay=min(remaining,45);time.sleep(delay);remaining-=delay
 save('deadline.json',dict(job_id=JOB,elapsed_seconds=time.monotonic()-started,acceptance=False,submission_retry=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();raise
