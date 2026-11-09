from pathlib import Path
import datetime,hashlib,json,subprocess,time
root=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108')
monitor=root/'launch01/monitor_once.py'
need_sha='cb63691494e51fd200b1d40105794126fc7cd1986e7521a1d667836225fd0012'
assert hashlib.sha256(monitor.read_bytes()).hexdigest()==need_sha
count_source=(root/'progress01-01/remote_source.py').read_bytes()
assert hashlib.sha256(count_source).hexdigest()=='f1d7a588ad1c33d4b767ec931591f6554ef688209fb9796cefc339c38ad08c3c'
watch=root/'watch01';last=None
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
for i in range(25):
 out=watch/('observation-%02d'%i)
 assert hashlib.sha256(monitor.read_bytes()).hexdigest()==need_sha
 cmd=['python3','-B',str(monitor),'--packet-sha256','b98873cae0e04221abcce02856ea7c610319f27f4c07e8275cc9bcfce1c2cd40','--submission-result-sha256','c4de71132a1b39edbee7d3ea4fbc9f0318efaf8da160fe0153baf93a08f00262','--packet-directory',str(root/'launch01'),'--evidence-directory',str(out)]
 proc=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 if proc.returncode:raise RuntimeError('read-only observation failed; original evidence retained: '+str(out))
 record=json.loads((out/'stdout.log').read_text());rows=record['observations'][1]['stdout'].splitlines()
 rows=[r.split('|') for r in rows if r.split('|')[0]=='35482135'];assert len(rows)==1
 state=rows[0][2].split()[0];message={'utc':record['utc'],'job_id':'35482135','state':state,'elapsed':rows[0][5]}
 if state not in terminal:
  c=subprocess.run(['bash','/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh','run','python3 -'],input=count_source,capture_output=True,timeout=45)
  transport={'returncode':c.returncode,'stdout':c.stdout.decode(),'stderr':c.stderr.decode(),'source_sha256':hashlib.sha256(count_source).hexdigest(),'acceptance':False}
  with (out/'counts-transport.json').open('x') as f:json.dump(transport,f,sort_keys=True,indent=2)
  if c.returncode:raise RuntimeError('counts observation failed; no job mutation')
  v=json.loads(c.stdout);message['receipt_counts']=[len(v['counts_only'][str(j)]['completed_receipt_indices']) for j in range(5)]
  message['completed_attempts']=[j for j in range(5) if v['counts_only'][str(j)]['attempt_receipt_present']]
 signature=(state,tuple(message.get('receipt_counts',[])))
 if signature!=last:print(json.dumps(message),flush=True);last=signature
 if state in terminal:
  with (watch/'terminal-observed.json').open('x') as f:json.dump(message,f,sort_keys=True,indent=2)
  break
 if i==24:raise RuntimeError('bounded watcher exhausted; no automatic retry or acceptance')
 for _ in range(6):time.sleep(50)
