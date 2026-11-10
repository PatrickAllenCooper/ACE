"""Independent saved-metadata gates only; never import project or audit sources."""
import base64, hashlib, json, os, re, stat, subprocess, traceback
from pathlib import Path, PurePosixPath
R=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
A=BASE/'ACE-LLM-gpu02-35516125-postcustody-01';OUT=Path(__file__).parent
AP=R/'results/ace_llm_intervention_scientific_gpu_audit_preparation_20261109/gpu02-35516125-actual01'
GP=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
PREP=R/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
pins={};facts={};checks=0
def need(ok,label):
 global checks
 checks+=1
 if not ok:raise ValueError(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(j):return json.dumps(j,sort_keys=True,separators=(',',':')).encode()
def digest(j):return sha(canonical(j))
def raw(j):return canonical(j)+b'\n'
def read(p,pin=None):
 p=Path(p);b=p.read_bytes();h=sha(b)
 if pin is not None:need(h==pin,'pin '+str(p))
 pins[str(p)]=dict(sha256=h,bytes=len(b));return b
def load(p,pin=None):return json.loads(read(p,pin))
def save(name,j):
 with (OUT/name).open('x') as f:json.dump(j,f,indent=2,sort_keys=True);f.write('\n')
def filehash(p):
 h=hashlib.sha256();count=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b);count+=len(b)
 return dict(bytes=count,sha256=h.hexdigest())
def gitblob(commit,path,expected):
 env=dict(os.environ,GIT_NO_REPLACE_OBJECTS='1')
 p=subprocess.run(['git','show',commit+':'+path],cwd=R,env=env,capture_output=True)
 need(p.returncode==0 and sha(p.stdout)==expected,'Git binding '+commit+':'+path)
 return dict(commit=commit,path=path,sha256=expected)
def normalized(rows):
 ids=[];ans=[]
 for row in rows:
  need(type(row) is list and len(row)==8,'accounting exact columns')
  need(row[0].isdigit() and row[4].isdigit() and row[5].isdigit(),'numeric job/CPU/time')
  ids.append(row[0]);tres=[x.split('=',1) for x in row[7].split(',')]
  need(all(len(x)==2 for x in tres) and len({x[0] for x in tres})==len(tres),'unique TRES')
  ans.append(row[:7]+[','.join(k+'='+v for k,v in sorted(tres))])
 need(len(ids)==len(set(ids)),'unique jobs');return sorted(ans)
def cost(rows):
 cpu=gpu=0
 for row in rows:
  seconds=int(row[5]);cpu+=int(row[4])*seconds;t=dict(x.split('=',1) for x in row[7].split(','))
  g=[int(v) for k,v in t.items() if k.startswith('gres/gpu:')]
  if not g and 'gres/gpu' in t:g=[int(t['gres/gpu'])]
  need(len(g)<=1,'unambiguous GPU allocation');gpu+=seconds*sum(g)
 return cpu,gpu
def relative_map(files,prefix):return {n[len(prefix):]:v['sha256'] for n,v in files.items() if n.startswith(prefix)}
try:
 c=load(A/'inputs/config.json','952dcdcb39f54330374552400118c040adb1a6f2fba0cb9777136f55379379a2')
 roles={k:load(v['path'],v['sha256']) for k,v in c['artifacts'].items() if not k.endswith('review')}
 for k,v in c['artifacts'].items():
  if k.endswith('review'):read(v['path'],v['sha256'])
 T=Path(c['capsules'][1]);TE=Path(c['capsules'][0])
 before=roles['before'];after=roles['after'];b=json.loads(before['stdout']);a=json.loads(after['stdout'])
 rec=roles['remote_custody'];full=rec['full_inventory']['files'];dirs=rec['full_inventory']['directories'];d=roles['disposition'];seal=roles['custody']
 need(before['returncode']==after['returncode']==0,'successful two observations')
 for observation in [before,after]:
  progress=[json.loads(line) for line in observation['stderr'].splitlines()]
  need(all(set(x)=={'phase','files_read','elapsed_seconds'} and x['phase']=='terminal_inventory_verification' and x['files_read']==1000*(i+1) and x['elapsed_seconds']>0 for i,x in enumerate(progress)),'only recorded fingerprint progress on stderr')
 need(b['files']==a['files']==full==d['remote_files'],'all complete raw maps exact')
 need(type(b['directories']) is list and type(a['directories']) is list and type(dirs) is list,'directory list types')
 need(sorted(b['directories'])==sorted(a['directories'])==sorted(dirs) and len(dirs)==len(set(dirs)),'unique equal directory roster')
 need(len(full)==99039 and sum(v['bytes'] for v in full.values())==2507157901,'actual raw total')
 need(rec['full_backup_verified'] is True and rec['full_local_copy'] is rec['acceptance'] is False,'truthful full remote/compact local')
 need(rec['original_before_inventory_sha256']==c['artifacts']['before']['sha256'] and rec['original_after_inventory_sha256']==c['artifacts']['after']['sha256'],'remote backup source observations')
 need(d['remote_custody_receipt_sha256']==c['artifacts']['remote_custody']['sha256'] and d['errors']==[] and d['input_pin_mismatches']==[],'collector final joins/no disagreements')
 need(d['acceptance'] is d['scientific_qualified'] is d['runtime_qualified'] is False,'collector not acceptance')
 need(d['job_id']=='35516125' and d['historical_postcustody_required'] is True,'actual recovery terminal')
 copies=seal['copies'];need(seal['identical'] is True and len(copies)==2,'two sealed capsules')
 inventories=list(copies.values());need(inventories[0]==inventories[1],'capsule same full selected inventory')
 # Read only current selected capsule bytes; do not traverse original ancestor archives.
 verify=[]
 for dirname,inventory in copies.items():
  root=Path(dirname);seen={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'custody-inventory.json'}
  need(seen==set(inventory['files']),'exact selected local membership '+dirname)
  for n,v in inventory['files'].items():
   p=root/n;need(not p.is_symlink() and filehash(p)==v,'selected capsule bytes '+n)
  verify.append(dict(root=dirname,files=len(seen),bytes=sum(v['bytes'] for v in inventory['files'].values())))
 need(read(T/'custody-inventory.json')==read(TE/'custody-inventory.json'),'seal mirror equality')
 facts['custody']=dict(raw_files=len(full),raw_bytes=sum(v['bytes'] for v in full.values()),directories=len(dirs),copies=verify,remote_archive_sha256=rec['backup_archive_sha256'],remote_archive_path=rec['backup_archive_path'],full_local_raw_copy=False)
 def member(n):
  row=full[n]
  if n in c['fetched']:
   x=c['fetched'][n];need(x['sha256']==row['sha256'],'batch member authority '+n);p=Path(x['path'])
  else:p=T/'remote'/n
  data=read(p,row['sha256']);need(len(data)==row['bytes'],'member bytes '+n);return data
 def j(n):return json.loads(member(n))
 p=roles['inner'];outer=roles['outer'];cpu=roles['cpu_acceptance'];old=j('control/inputs/cpu-packet.json');baseline=roles['baseline'];result=roles['result']
 # Independent complete historical map comparison (all content hashes, no payload decoding).
 fprev=roles['failed_remote_custody']['full_inventory'];mismatches=[n for n,v in fprev['files'].items() if full.get(n)!=v]
 need(not mismatches,'all original failed47 files unchanged')
 need(set(fprev['directories'])<=set(dirs),'all failed47 directories retained')
 old_launch={n:v for n,v in full.items() if n.startswith('launch/') and n not in fprev['files']}
 need(all(n.startswith('launch/gpu/') for n in old_launch),'only authorized old GPU output appended')
 cpuout=relative_map(full,'cpu/cpu/');ex=cpu['cpu_output_excluded_directory']
 cpuout={n:h for n,h in cpuout.items() if not (n==ex or n.startswith(ex+'/'))}
 need(cpuout==cpu['cpu_output_files'],'entire accepted CPU output outside campaign unchanged')
 camp=relative_map(full,'cpu/cpu/worker/artifacts/campaign/')
 need(all(camp.get(n)==h for n,h in cpu['campaign_files'].items()),'every accepted CPU campaign member unchanged')
 adds=set(camp)-set(cpu['campaign_files']);need(all(n.split('/')[0] in ['policy-05','policy-06','policy-07'] for n in adds),'only three original GPU policies appended')
 hist=roles['recovery_context'];facts['historical_postcustody']=dict(verified=True,failed_files=len(fprev['files']),failed_directories=len(fprev['directories']),cpu_output_files=len(cpuout),original_campaign_files=len(cpu['campaign_files']),campaign_files=len(camp),new_campaign_files=len(adds),authorized_old_root_new_files=len(old_launch))
 # Original packet/source values are fixed, never inferred from future output.
 need(p['cpu_acceptance_sha256']==c['artifacts']['cpu_acceptance']['sha256'] and p['cpu_packet_sha256']==cpu['cpu_packet_sha256'],'CPU authority joins')
 need(outer['gpu_packet_sha256']==c['artifacts']['inner']['sha256'] and outer['baseline_sha256']==c['artifacts']['baseline']['sha256'],'outer packet joins')
 need(outer['new_worlds']==outer['new_training_updates']==0 and outer['cells']==192,'unchanged scientific cell scope')
 for n,h in p['files'].items():need(full['control/inputs/'+n]['sha256']==h,'all actual worker input pins '+n)
 for n,h in outer['new_input_files'].items():need(full['control/'+n]['sha256']==h,'all actual outer input pins '+n)
 # Terminal accounting includes one actual added row and the exact prior journal map.
 rows=b['related_accounting'];current=[r for r in rows if r[0]=='35516125'];prior=[r for r in rows if r[0]!='35516125']
 need(len(rows)==48 and len(current)==1 and normalized(prior)==normalized(baseline['related_accounting']),'exact47+1 accounting')
 need(normalized(rows)==normalized(a['related_accounting']),'after accounting same')
 need(current[0][2:7]==['COMPLETED','0:0','4','1708','ucb736_asc2'],'current actual resource/time result')
 need(cost(prior)==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds'])==(30241,2750),'prior actual charge')
 need(cost(rows)==(37073,4458) and cost(rows)==(cost(prior)[0]+6832,cost(prior)[1]+1708),'actual accounting increment')
 journals=dict(baseline['journal_pins']);journals[outer['remote_root']+'/submission.intent.json']=result['intent_sha256'];journals[outer['remote_root']+'/submission.result.json']=c['artifacts']['result']['sha256']
 need(len(journals)==98 and journals==b['journal_pins']==a['journal_pins'],'exact96+2 journals')
 need(baseline['outstanding_reservation']==0 and baseline['preparation_allowance_cpu_seconds']==3600,'one prep allowance')
 need(d['closed_cpu_seconds']==37073 and d['closed_gpu_slice_seconds']==4458 and d['cpu_admission_after_release']==40673,'collector accounting agreement')
 facts['accounting']=dict(jobs=48,journals=98,elapsed_seconds=1708,allocated_cpu_seconds=6832,gpu_slice_seconds=1708,closed_cpu_seconds=37073,closed_gpu_slice_seconds=4458,preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=40673,outstanding_reservation=0)
 # Actual remote canonical intent: full equality of all local fields plus only two additions.
 local=roles['local_intent'];remote=roles['remote_intent'];need(set(remote)==set(local)|{'remote_utc','fresh_readiness'},'exact remote intent augmentation')
 need(all(remote[k]==v for k,v in local.items()),'every canonical local field unchanged remotely')
 need(result['job_id']=='35516125' and result['returncode']==0 and result['ambiguous'] is False and result['stdout']=='35516125\n' and result['stderr']=='','one unambiguous submission')
 need(result['intent_sha256']==c['artifacts']['remote_intent']['sha256']==sha(base64.b64decode(b['current_control_intent_base64'],validate=True))==sha(base64.b64decode(a['current_control_intent_base64'],validate=True)),'remote intent independently observed before/after')
 need(full['control/submission.result.json']['sha256']==c['artifacts']['result']['sha256'] and full['control/submission.intent.json']['sha256']==result['intent_sha256'],'canonical actual control journals')
 G=GP/'launch02/transfer-continuation01';tr=local['transfer_continuation'];cp='control/transfer-continuation01/'
 preserved=load(G/'preserve.stdout');record=raw(preserved['record']);partial=base64.b64decode(preserved['partial_base64'],validate=True)
 need(sha(record)==tr['preservation_sha256']==full[cp+'intent.json']['sha256']==full[cp+'projects-readback.json']['sha256'],'actual remote preservation record vs original returned bytes')
 need(len(partial)==2392064 and sha(partial)==full[cp+'partial.renamed']['sha256']=='7b8939e8509c6849203e0eb3daad0817aba743dc163fab1580de2505269cf66e','remote renamed original partial unchanged')
 for remote_name,local_name,key in [('local-readback-ack.json','partial-local-readback.json','local_readback_sha256'),('complete.json','completion.json','completion_sha256'),('pre-submission.json','pre-submission.json','preflight_sha256')]:
  need(sha(read(G/local_name))==full[cp+remote_name]['sha256']==tr[key],'actual remote continuation match '+remote_name)
 complete=load(G/'completion.json')
 need(complete['matching_original_members_unchanged'] is True,'original complete members unchanged')
 for n,row in complete['completed_members'].items():
  need(full['control/'+n]==row and full[cp+'staged/'+n]==row,'staged/completed/current file identity '+n)
 # New full-custody observation proves exact captured metadata member existence;
 # freshness record contents are a separately retained pending read when not locally materialized.
 pending=cp+'fresh-before-preservation.json'
 facts['transfer_control']=dict(remote_intent_all_local_fields_equal=True,independently_observed_remote_intent=True,preserved_partial_bytes=len(partial),preserved_partial_sha256=sha(partial),all_staged_completed_members_equal=True,completed_members=len(complete['completed_members']),fresh_before_preservation=full[pending],fresh_before_preservation_content_review_pending=True,prior_independent_review_sha256='ddb4977accdd649019bc05edef9b19f7f7b4e79ea895611071f2ba24bc442ab7')
 # Git bindings are actual committed byte identities, no project callables invoked.
 git=[];manifest=roles['operational_manifest']
 for n,v in manifest['files'].items():git.append(gitblob(manifest['source_commit'],v['repository_path'],p['files'][n]))
 git.append(gitblob(local['packet_commit'],'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02/packet.json',c['artifacts']['outer']['sha256']))
 intent=load(T/'collector-intent.json');git.append(gitblob(intent['packet_commit'],intent['collector_repository_path'],intent['collector_sha256']))
 git.append(gitblob(tr['driver_commit'],'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/transfer-continuation01/continue_once.py',tr['driver_sha256']))
 for n,h in p['files'].items():
  if not n.endswith('.py') or n in manifest['files'] or n=='probe.py':continue
  git.append(gitblob(local['source_commit'],'scripts/research/'+n,h))
 facts['git']=git
 # Aid provenance and final sealed package, reusing closed source reviews.
 report=load(A/'audit-attempt04/report.json','eb0f711af3f98e997acec67cd56690c9b68146c112bd052412770f7bd4cd1060')
 need(report['status']=='saved_checks_complete_independent_review_required' and report['errors']==[] and report['missing_exact_fetches']=={},'complete saved checker')
 need(report['completed_checks']==['custody','recovery_fetch_requirements','packet','accounting','runtime','terminals','fetch_requirements','services'] and len(report['completed_cells'])==72,'all checker stages/cells')
 need(report['acceptance'] is report['scientific_qualified'] is report['runtime_qualified'] is False,'checker remains evidence only')
 finalseal=load(A/'final-copy-receipt.json','3465f27d525f4b5d5b6297349a11f8de00b33b0be28a691175f20b26995f5950')
 need(read(AP/'final-copy-receipt.json')==read(A/'final-copy-receipt.json'),'final aid seal mirror')
 for root in [A,AP]:
  for n,row in finalseal['files'].items():need(filehash(root/n)==row,'sealed actual aid byte '+str(root/n))
 facts['aid']=dict(report_sha256=sha(read(A/'audit-attempt04/report.json')),final_seal_sha256=sha(read(A/'final-copy-receipt.json')),files_per_copy=finalseal['files_per_copy'],all_original_attempts_preserved=True,acceptance=False)
 facts['checks']=checks;facts['acceptance']=False
 save('facts.json',facts);save('read-pins.json',pins)
 print(json.dumps({k:v for k,v in facts.items() if k not in ['git']},sort_keys=True))
except BaseException:
 save('partial-facts.json',facts);save('partial-read-pins.json',pins)
 with (OUT/'failure.txt').open('x') as f:f.write(traceback.format_exc())
 raise
