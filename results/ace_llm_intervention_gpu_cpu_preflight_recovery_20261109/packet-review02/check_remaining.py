"""Continue only pending checks; retain initial multiline-shlex checker failure unchanged."""
from pathlib import Path
import hashlib,json,shlex,stat,os,subprocess,time,traceback,collections,re
REPO=Path('/Users/pat/code/ACE');P=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02'
E=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-20261109-02')
R=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
def j(p):return json.loads(Path(p).read_bytes())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def fp(p):
 p=Path(p);need(not p.is_symlink() and p==p.resolve(strict=True),'ordinary canonical member')
 with p.open('rb') as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_nlink==1,'regular independent file');h=hashlib.sha256()
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
  z=os.fstat(f.fileno())
 sig=lambda s:(s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 need(sig(a)==sig(z)==sig(p.lstat()),'changed while read')
 return {'bytes':a.st_size,'sha256':h.hexdigest()}
def inventory(p):
 return {'files':{q.relative_to(p).as_posix():fp(q) for q in sorted(p.rglob('*')) if not q.is_dir()},
         'directories':[q.relative_to(p).as_posix() for q in sorted(p.rglob('*')) if q.is_dir()]}
checks=[];facts={};start=time.monotonic()
try:
 prior=j(R/'check01.json');outer=j(P/'packet.json');inner=j(P/'inputs/packet.json');cpu=j(P/'cpu-packet.json');new=outer['remote_root'];old=inner['paths']['gpu_output']
 text=(P/'job.sbatch').read_text();args=shlex.split(text.split('\nexec ',1)[1])
 entry='qualify_llm_intervention_scientific_campaign_gpu.py'
 need(args[:4]==[cpu['runtime_source']+'/bin/python','-I','-B','-c'],'exact isolated runtime bootstrap')
 need(args[5:]==[new+'/inputs/'+entry,inner['files'][entry],'--packet',new+'/inputs/packet.json','--packet-sha256',outer['gpu_packet_sha256'],
      '--admission',new+'/inputs/admission.json','--admission-sha256',outer['new_input_files']['inputs/admission.json'],'--output',old],'multiline exact argv bindings')
 need(args[4] in (P/'inputs/qualify_llm_intervention_runtime_metadata.py').read_text() or 'import' in args[4], 'bootstrap is source text')
 # Bootstrap equality is checked by comparing old packet command, without importing either source.
 oldscript=(P.parent/'launch01/job.sbatch').read_text();oldargs=shlex.split(oldscript.split('\nexec ',1)[1])
 need(args[:5]==oldargs[:5],'same accepted isolated runtime bootstrap')
 checks.append('exact multiline sbatch bootstrap and new entry/packet/admission/frozen output argv')
 facts['sbatch_directives']=[l for l in text.splitlines() if l.startswith('#SBATCH')]
 need(all(not (P/n).exists() and not (E/n).exists() for n in ('submission.intent.json','submission.result.json','controller.json','launcher_failure.json')),'packet has submission/controller markers')
 checks.append('no local submission/controller markers at review')
 ref=j(P/'storage-preparation-reference.json');intentref=j(P/'storage-preparation-intent.json');root=Path(ref['path'])
 need(sha(Path(ref['inventory_path']).read_bytes())==ref['inventory_sha256'],'captured storage reference digest')
 need(sha((root/'intent.json').read_bytes())==ref['intent_sha256']==intentref['intent_sha256'],'storage intent join')
 intent=j(root/'intent.json');need(intent['identity']['coordinator_sha256']==outer['new_input_files']['build_packet.py']
     and intent['identity']['phase']=='build' and set(intent['identity']['intended_packet_roots'])=={str(P),str(E)},'exact build/source/destinations')
 allinv=sorted(root.glob('inventory-*.json'));need(allinv,'saved operation records')
 counts=collections.Counter();last=None
 for p in allinv:
  x=j(p);n=x['ordinal'];rr=j(root/f'reference-{n:03d}.json')
  need(x['status']=='completed' and x['error'] is None and x['intent_sha256']==ref['intent_sha256'],'failed/inconsistent storage operation')
  need(rr['inventory_sha256']==sha(p.read_bytes()) and rr['inventory_path']==str(p) and rr['status']=='completed','operation reference binding')
  counts[x['operation']]+=1;last=x
 need(counts=={'binding':1,'input-read':95,'archive-verify':3},'exact completed operation sequence')
 op=Path(last['operations_path']);need(inventory(op)==last['inventory'],'complete retained storage operations changed')
 ops=[]
 for p in sorted(op.glob('*/transport.json')):
  x=j(p);need(x['returncode']==0 and x['remote_outcome_ambiguous'] is False,'failed/ambiguous archive transport')
  parent=p.parent;capacity=j(parent/'capacity.json');need(capacity['admitted'] is True,'capacity admission missing')
  for v in capacity['filesystem_totals'].values():need(v['available_bytes']>=v['required_bytes'],'insufficient captured capacity')
  saved=j(parent/'receipt.json')
  if parent.name.startswith('verify-'):
   need(saved['full_backup_verified'] is True and saved['receipt_sha256']==outer['storage_binding']['predecessor_remote_custody_sha256'],'archive verification original receipt join')
  else:
   need(parent.name in ('fetch-log-028','fetch-log-091'),'unexpected opaque artifact fetch')
  ops.append({'name':parent.name,'transport_sha256':sha(p.read_bytes()),'receipt_sha256':sha((parent/'receipt.json').read_bytes()),
              'wall_seconds':x['wall_seconds'],'capacity':capacity['filesystem_totals']})
 need(len(ops)==5,'three archive verifies plus two exact omitted adapters')
 facts['storage_operations']={'intent_sha256':ref['intent_sha256'],'captured_reference':ref,'final_inventory_path':str(allinv[-1]),
      'final_inventory_sha256':sha(allinv[-1].read_bytes()),'counts':dict(counts),'transports':ops,'all_retained_operation_files':len(last['inventory']['files']),
      'all_retained_operation_bytes':sum(v['bytes'] for v in last['inventory']['files'].values())}
 checks.append('all99 completed storage operations, three verification transports and two opaque adapter fetches; final record inventory and capacity')
 site=(P/'site-readiness.md').read_text();s=json.loads(site.split('```json\n',1)[1].split('\n```',1)[0])
 facts['saved_site_headroom']={k:v for k,v in s.items() if k.startswith('/') or k in ('utc','scratch_probe','scratch_write_read_cleanup')}
 facts['saved_site_command_summaries']={k:{z:v.get(z) for z in ('returncode','stdout')} for k,v in s.get('commands',{}).items() if k in ('queue','partition','association','qos')}
 commit='bb3a3d29d715cf718503013838cd5fc450e09053';cp='results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/coordinator02/authoring/pull_terminal.py';rp='results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/coordinator-review02/review.md'
 for path,h in [(cp,'3697547ddace45fda812c069c08d2d299e75b61fe9db717807a5379645771c27'),(rp,'2d75c6ef17bc1b4b807efe2e3ce76505dfbf4261380a19f99bd251b764ce2523')]:
  expected=(REPO/path).read_bytes();actual=subprocess.run(['git','--no-replace-objects','show',commit+':'+path],cwd=REPO,capture_output=True,check=True).stdout
  need(sha(expected)==h and actual==expected,'collector/review full Git source provenance')
 facts['collector']={'source_commit':commit,'repository_path':cp,'collector_sha256':'3697547ddace45fda812c069c08d2d299e75b61fe9db717807a5379645771c27','review_path':rp,'review_sha256':'2d75c6ef17bc1b4b807efe2e3ce76505dfbf4261380a19f99bd251b764ce2523'}
 checks.append('collector3697547 and closed review2d75 both match explicit bb3a full Git commit')
 # Final immutable readback after the review, not a repeat of old qualification.
 need(inventory(P)==inventory(E)==prior['facts']['packet_inventory'],'packet changed during concrete review')
 checks.append('both original packet copies unchanged at final readback')
 out={'status':'remaining_checks_passed','checks':checks,'facts':facts,'wall_seconds':time.monotonic()-start,'actual_execution':False,'remote_actions':0,'tests_run':0,
      'corrected_checker_issue':'check01 selected only first line of valid multiline shell-quoted bootstrap; this continuation parses the whole exec tail. Original check01/source preserved.'}
except BaseException as e:
 out={'status':'checker_failure_preserved','checks':checks,'facts':facts,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'wall_seconds':time.monotonic()-start}
with (R/'check02.json').open('x') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='facts'},indent=2))
