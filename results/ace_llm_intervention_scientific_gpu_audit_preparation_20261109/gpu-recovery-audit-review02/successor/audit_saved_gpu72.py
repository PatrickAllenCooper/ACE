"""Prospective read-only stdlib GPU72 saved-evidence aid. NEVER acceptance.

Only pinned pure audit/receipt metadata dependencies execute. Frozen producer/collector/backend
sources are text/AST oracles, never imported. No remote, model or tensor calls.
"""
import argparse,collections,hashlib,json,math,os,re,resource,stat,struct,time,traceback,types
from pathlib import Path,PurePosixPath

HERE=Path(__file__).resolve().parent
AID_SHA='6d56672582f8c41bdd01c64b446c1d1607316650c6357bec95a889c357ab7eb4'
SUPPLEMENT_SHA='e2da9858fe137c653a96a3650ec0377f4d4d9721891dda73fe5727cddb292d13'
ORACLES_SHA='b4dd283396a6da3f8f90d5e6ea2bd8587a5dc55c91d67eea56ab3e7756103c84'
RECOVERY_SHA='1d72953472d00eda134546697f256bc0104cb0e0750edef080d5896e66347f6d'
PREFLIGHT_SHA='b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8'
BACKEND_SHA='4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f'
SOURCE='a9346022f145cc29439325b6f291597b2e6ec037'
FREEZE='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
ENTRY='qualify_llm_intervention_scientific_campaign_gpu.py'
POLICIES=('uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm')
LLMS=POLICIES[5:];WORLDS=list(range(202611070100,202611070124))
G='launch/gpu/';CPU='cpu/cpu/';C=CPU+'worker/artifacts/campaign/'
REMAINING=(
 'Independent original scheduler/packet/collector-source Git commits, both namespace locks and authentic original SSH/project readback: a local aid cannot attest these.',
 'Independent accepted CPU120 and training ancestors, their complete preserved failures and actual source/runtime qualification; this aid reuses their supplied independent acceptance and inventory pins, not raw scientific reruns.',
 'Original public task/SCM-state allowlists, numeric learner semantics, exact prompt rendering/tokenizer decode and completion parser predicates remain frozen-validator/manual gates. This aid checks opaque input/state/history byte joins and recorded token metadata; it does not execute those producers.',
 'Full original GPU native-device classification/major/minor/loader provenance predicates and actual owned cleanup authority must be reviewed against pinned original runtime records. Saved consistency checks do not establish fresh process absence.',
 'Read every error/missing member and preserved checker attempt; only independent review may author the EXISTING17-key GPU acceptance. No success flag here is acceptance or utility qualification.',
)

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return sha(canonical(v))
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def positive(v):return finite(v) and v>0

def dependency(name,pin):
 p=HERE/'dependencies'/name;data=p.read_bytes();need(sha(data)==pin,'pure audit dependency pin differs')
 m=types.ModuleType('pinned_'+p.stem);m.__file__=str(p);exec(compile(data,str(p),'exec'),vars(m));return m

aid=dependency('audit_saved_cpu.py',AID_SHA)
supp=dependency('audit_supplement.py',SUPPLEMENT_SHA)
recovery=dependency('gpu_recovery_delta.py',RECOVERY_SHA)
preflight=dependency('llm_intervention_cpu_preflight_receipt.py',PREFLIGHT_SHA)

def opaque_compact(data):
 """Remove only JSON whitespace outside strings; never interpret payload values.
 Frozen writers sort all keys. Digest joins, not this function, establish that
 these saved spans match their producer's canonical bytes. No numeric parsing.
 """
 out=bytearray();quoted=escaped=False
 for c in data:
  if quoted:
   out.append(c)
   if escaped:escaped=False
   elif c==92:escaped=True
   elif c==34:quoted=False
  elif c==34:quoted=True;out.append(c)
  elif c not in b' \r\n\t':out.append(c)
 need(not quoted and not escaped,'unfinished opaque JSON string')
 return bytes(out)

def public(data,keys):return aid.public(data,set(keys))
def parts(data):return aid.object_parts(data)
def load_oracles():
 manifest=aid.decode(aid.file_bytes(HERE/'oracle-pins.json',ORACLES_SHA))
 need(manifest['source_commit']==SOURCE,'oracle source commit')
 sources={}
 for name,pin in manifest['pins'].items():
  p=HERE/name if name.startswith('references/') else HERE/'oracles'/name
  sources[name]=aid.file_bytes(p,pin,maximum=1024**2)
 return sources,manifest['pins']

def selected(full,source):
 suffixes=supp.constant(source,'TEXT_SUFFIXES');binary=supp.constant(source,'BINARY_SUFFIXES');files={}
 for n,v in full['files'].items():
  p=PurePosixPath(n)
  if (n.startswith('launch/') or n in ('namespace.json','original-namespace.json')) and p.suffix.lower() in suffixes and not {s.lower() for s in p.suffixes}&binary and v['bytes']<=32*1024**2:files[n]=v
 need(sum(v['bytes'] for v in files.values())<=512*1024**2,'complete selected local evidence exceeds bound')
 directories=sorted({'/'.join(n.split('/')[:i]) for n in files for i in range(1,len(n.split('/')))})
 return dict(files=dict(sorted(files.items())),directories=directories)

def rows_by_ordinal(full,prefix,expression,count):
 found={}
 for name in full:
  if not name.startswith(prefix):continue
  rel=name[len(prefix):];match=re.fullmatch(expression,rel)
  if match:
   i=int(match[1]);need(i not in found and i<count,'extra/duplicate generation ordinal');found[i]=name
 need(set(found)==set(range(count)),'missing/noncontiguous generation family')
 return found

def saved_inventory(files,directories):
 """Validate the original directory container, then make a semantic comparison view.
 Never mutate an authenticated observation or change its original digest.
 """
 need(type(directories) is list,'original directory inventory must be a list')
 need(all(type(n) is str for n in directories),'directory names must be strings')
 value=dict(files=files,directories=sorted(directories))
 aid.validate_inventory(value)
 return value

def cpu_ancestor_inventory(previous,current):
 """Exact original CPU roots/pins, with only the existing three GPU append roots."""
 mapping={'launch':'cpu','cpu':'token','training':'training','common':'common',
          'failed01':'failed01','failed-token01':'failed-token01',
          'namespace.json':'namespace.json','original-namespace.json':'original-namespace.json'}
 need(type(previous['roots']) is dict and type(current['roots']) is dict
      and set(previous['roots'])==set(mapping)
      and set(current['roots'])==set(mapping.values())|{'launch','control'},'exact original ancestor root labels required')
 for old,new in mapping.items():
  path=previous['roots'][old]
  need(type(path) is str and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
       and current['roots'][new]==path,'original ancestor physical root changed: '+old)
 old=saved_inventory(previous['full_inventory']['files'],previous['full_inventory']['directories'])
 now=saved_inventory(current['full_inventory']['files'],current['full_inventory']['directories'])
 def mapped(name):
  head,sep,tail=name.partition('/')
  need(head in mapping,'ancestor member outside original root map')
  return mapping[head]+sep+tail
 files={mapped(n):v for n,v in old['files'].items()};dirs={mapped(n) for n in old['directories']}
 need(all(now['files'].get(n)==v for n,v in files.items()),'whole original ancestor/failure bytes changed')
 need(dirs<=set(now['directories']),'original ancestor directory missing')
 append={C+f'policy-{i:02d}' for i in (5,6,7)}
 def allowed(name):
  return name=='control' or name.startswith('control/') or name=='launch' or name.startswith('launch/') or any(name==a or name.startswith(a+'/') for a in append)
 need(all(allowed(n) for n in set(now['files'])-set(files))
      and all(allowed(n) for n in set(now['directories'])-dirs),'ancestor changed outside three-policy append/current launch')

def telemetry_facts(telemetry,session,post,origin,work,final,session_deadline):
 pid=session['registered_pid'];count=session['positive_generations']
 need(type(pid) is int and pid>0 and type(count) is int and 1<=count<=576,'owned PID/count required')
 need(telemetry and session['error'] is None and session['process_group_closed'] is session['post_exit_gpu_empty'] is True,'failed/unclosed session')
 start=session['started_monotonic'];first=session['first_verified_monotonic'];closed=session['closed_monotonic']
 need(all(finite(v) for v in (origin,start,first,closed,session_deadline)) and origin<=start<=first<closed<final
      and first-start<120 and first<work,'service absolute startup/close deadline differs')
 memory=[];previous=None;first_seen=None;registered=False
 for i,t in enumerate(telemetry):
  now=t['monotonic'];n=t['positive_generations'];sample=t['gpu_process_sample']
  need(finite(now) and start<=now<=closed and type(n) is int and 0<=n<=count and t['registered_pid'] in (None,pid)
       and t['policy_id']==session['policy_id'] and abs(t['service_elapsed_seconds']-(now-start))<1e-6
       and abs(t['outer_elapsed_seconds']-(now-origin))<1e-6,'telemetry clock/owner differs')
  if previous:need(now>previous['monotonic'] and n>=previous['positive_generations'] and (previous['first_verified_monotonic'] is None or now-previous['monotonic']<90),'missing/stale periodic observation')
  if sample is None:
   need(not registered and t['registered_pid'] is None and n==0
        and all(t[k] is None for k in ('first_verified_monotonic','last_gpu_monotonic','last_generation_monotonic'))
        and t['uncertainties']==[],'only contiguous unregistered zero-progress prefix can omit samples')
  else:
   need(t['registered_pid']==pid,'sample requires original registered process')
   registered=True
   need(sample['returncode']==0 and type(sample['owned_rows']) is list,'NVIDIA observation failed')
   for line in sample['owned_rows']:
    cols=[v.strip() for v in line.split(',')]
    need(len(cols)==3 and cols[0]==str(pid) and cols[1] and positive(float(cols[2])),'wrong-owner or nonpositive GPU memory')
    memory.append(float(cols[2]))
   allowed=['no current positive owned-process GPU observation']
   need(not t['uncertainties'] or (t['uncertainties']==allowed and not sample['owned_rows']),'unexpected GPU telemetry uncertainty')
  if t['first_verified_monotonic'] is not None:
   need(t['first_verified_monotonic']==first and finite(t['last_gpu_monotonic']) and finite(t['last_generation_monotonic'])
        and first<=now and 0<=now-t['last_gpu_monotonic']<90 and 0<=now-t['last_generation_monotonic']<90,'verified work freshness differs')
   if first_seen is None:
    need(n>0 and sample is not None and memory,'first verification lacks both work signals');first_seen=now
  previous=t
 need(first_seen==first and telemetry[-1]['positive_generations']==count and telemetry[-1]['monotonic']<session_deadline and memory,'missing/late first/final work observation')
 need(post['registered_pid']==pid and post['process_group_closed'] is True and post['sample']['returncode']==0
      and post['sample']['owned_rows']==[] and start<=post['monotonic']<=closed,'dedicated post-exit clearance differs')
 return dict(samples=len(telemetry),max_owned_memory_mib=max(memory),first_verified_seconds=first-origin,
             session_seconds=closed-start,utilization_percent=None,signal_types=['owned NVIDIA process memory','positive CUDA generation/progress'])

class Audit(aid.Audit):
 def __init__(self,cfg):
  super().__init__(cfg);self.oracles,self.oracle_pins=load_oracles();self.completed=[];self.cells=[]
  for filename,name in [('host-entry04.py',recovery.ENTRY),('history-adapter04.py',recovery.HISTORY),('preflight-checker.py',recovery.CHECKER)]:
   aid.file_bytes(HERE/'recovery-references'/filename,recovery.OPERATIONAL[name],maximum=1024**2)
 def j(self,n):
  need(PurePosixPath(n).name not in {'world.json','paired-plan.json','campaign.json','campaign-bundle.json'}
       and not PurePosixPath(n).name.startswith('return-'),'opaque science payload decode forbidden')
  return aid.decode(self.b(n))
 def artifact(self,key,maximum=128*1024**2):
  r=self.c['artifacts'][key];need(set(r)=={'path','sha256'},'independent path/pin pair required')
  return aid.file_bytes(Path(r['path']),r['sha256'],maximum)
 def custody(self):
  need(self.c['schema']=='ace-prospective-gpu72-recovery-saved-audit-input-v2' and re.fullmatch('[1-9][0-9]*',self.c['job_id']), 'actual job/config required')
  need(self.c['helper_sha256']==BACKEND_SHA,'independently pinned traversal backend required')
  self.d=aid.decode(self.artifact('disposition'));seal=aid.decode(self.artifact('custody'))
  observations=[];inventories=[]
  for key in ('before','after'):
   transport=aid.decode(self.artifact(key));need(transport['returncode']==0,'inventory transport failed')
   v=aid.decode(transport['stdout']);need(v['terminal_gate'] is True and v.get('observation_error') is None,'incomplete terminal observation')
   inventories.append(saved_inventory(v['files'],v['directories']))
   observations.append(v)
  self.before,after=observations;self.after=after
  for key in ('files','original_hardlink_groups','parent_accounting','related_accounting','journal_pins','upstream_pins','upstream_errors'):
   need(self.before[key]==after[key],'original second observation differs: '+key)
  need(inventories[0]==inventories[1],'original second directory inventory differs')
  for key in ('root_present','failed_gpu_root_present','cpu_present','training_present','token_present',
              'failed_token_present','common_present','failed_present','namespace_present','original_namespace_present',
              'current_control_intent_base64'):
   need(self.before[key]==after[key],'second recovery observation differs: '+key)
  need(self.before['results']['details']==after['results']['details'],'second saved scheduler request differs')
  receipt=aid.decode(self.artifact('remote_custody'));self.receipt=receipt
  full=receipt['full_inventory'];local=receipt['local_inventory'];aid.validate_inventory(full);aid.validate_inventory(local)
  need(local==selected(full,self.oracles['references/remote_custody4161.py']),'exact current-launch text selection differs')
  need(full==inventories[0],'full archive/original inventory differs')
  need(receipt['schema']=='ace-remote-custody-v1' and receipt['full_backup_verified'] is True
       and receipt['full_local_copy'] is receipt['acceptance'] is receipt['retry_allowed'] is False
       and receipt['backup_kind']=='ustar' and receipt['helper_sha256']==BACKEND_SHA
       and receipt['selection_policy']=='current-launch-text-and-namespaces-v1','backup identity/scope differs')
  need(receipt['source_observation_sha256']==sha(raw(self.before)) and receipt['original_hardlink_groups']==self.before['original_hardlink_groups']
       and all(receipt[k]==sha(raw(full)) for k in ('original_before_inventory_sha256','original_after_inventory_sha256','project_readback_inventory_sha256')),'archive before/after readback binding')
  omitted={n:v for n,v in full['files'].items() if n not in local['files']}
  need(all(full['files'].get(n)==v for n,v in local['files'].items()) and receipt['omitted_files']==omitted
       and receipt['omitted_bytes']==sum(v['bytes'] for v in omitted.values()),'selected/omitted partition')
  verify=aid.decode(self.artifact('archive_verification')) if 'archive_verification' in self.c['artifacts'] else None
  if verify is not None:
   need(verify['full_backup_verified'] is True and verify['full_inventory']==full and verify['local_inventory']==local
        and verify['receipt_sha256']==sha(raw(receipt)) and verify['backup_archive_sha256']==receipt['backup_archive_sha256']
        and verify['helper_sha256']==BACKEND_SHA,'saved independent second archive verification differs')
  copies=[Path(p) for p in self.c['capsules']];need(len(copies)==2 and copies[0]!=copies[1] and all(str(p)==str(p.resolve()) for p in copies) and not os.path.samestat(copies[0].stat(),copies[1].stat()) and seal['identical'] is True,'two original compact copies required')
  views=[]
  for p in copies:
   v=aid.inventory(p);need(v['files'].pop('custody-inventory.json')['sha256']==self.c['artifacts']['custody']['sha256'],'original seal differs')
   need(seal['copies'].get(str(p))==v and aid.inventory(p/'remote')==local,'whole compact copy/selected inventory differs');views.append(v)
  need(views[0]==views[1],'independent copies differ')
  for role,member in [('disposition','disposition.json'),('before','inventory_transport.json'),('after','inventory_after_transport.json'),('remote_custody','remote-custody.json')]:
   need(views[0]['files'].get(member,{}).get('sha256')==self.c['artifacts'][role]['sha256'],'original sealed metadata role/pin differs')
  d=self.d
  self.facts['retained_collector_findings']=dict(errors=d.get('errors'),failure_marker_metadata=d.get('failure_marker_metadata'),input_pin_mismatches=d.get('input_pin_mismatches'))
  need(d['schema']=='ace-scientific-campaign-gpu-terminal-custody-v1' and d['job_id']==self.c['job_id']
       and d['errors']==d['input_pin_mismatches']==[] and d['failure_marker_metadata']=={}
       and d['custody_bytes_verified'] is d['current_lineage_verified'] is d['upstream_inputs_verified'] is True
       and d['complete_local_raw_copy'] is False and d['remote_files']==full['files'] and d['remote_directories']==full['directories']
       and d['remote_custody_receipt_sha256']==sha(raw(receipt)) and d['backup_helper_sha256']==BACKEND_SHA,'collector errors/failed outputs retained; success checks blocked')
  self.root=copies[0]/'remote';self.full=full['files'];self.local=local['files']
  self.facts['custody']=dict(raw_files=len(self.full),raw_bytes=sum(v['bytes'] for v in self.full.values()),local_files=len(self.local),
     full_local_copy=False,remote_archive_sha256=receipt['backup_archive_sha256'],archive_path=receipt['backup_archive_path'],separate_saved_archive_verification=verify is not None)
 def packet(self):
  self.p=aid.decode(self.artifact('inner'));self.outer=aid.decode(self.artifact('outer'));self.packet_pin=self.c['artifacts']['inner']['sha256']
  self.cpu=aid.decode(self.artifact('cpu_acceptance'));self.old=self.j('control/inputs/cpu-packet.json')
  p=self.p;builder=self.oracles['references/gpu_builder.py'];resources=supp.constant(builder,'RESOURCES');limits=supp.constant(builder,'LIMITS')
  need(p['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2' and p['scope']=='development' and p['freeze_sha256']==FREEZE
       and p['resources']==resources and p['limits']==limits and self.outer['source_commit']==SOURCE
       and self.outer['schema']=='ace-scientific-campaign-gpu-recovery-launch-v2'
       and self.outer['gpu_packet_sha256']==self.packet_pin
       and self.outer['baseline_sha256']==self.c['artifacts']['baseline']['sha256']
       and p['recovery_files']==recovery.RECOVERY_FILES,'explicit recovery with frozen scientific GPU contract')
  for key,member in [('outer','control/packet.json'),('inner','control/inputs/packet.json'),('admission','control/inputs/admission.json'),('result','control/submission.result.json'),('remote_intent','control/submission.intent.json'),('cpu_acceptance','control/inputs/cpu-acceptance.json'),('baseline','control/gpu-failure-baseline.json'),('freeze','control/freeze.json'),('training_acceptance','control/training-acceptance.json')]:self.h(member,self.c['artifacts'][key]['sha256'])
  freeze=aid.decode(self.artifact('freeze'));need(sha(self.artifact('freeze'))==FREEZE and freeze['source_commit']==SOURCE,'frozen source metadata')
  closure=freeze['stage_closure']['stages']['GPUcampaign']
  authority=aid.decode(self.artifact('recovery_authority'));context=aid.decode(self.artifact('recovery_context'))['execution_context']
  manifest=aid.decode(self.artifact('operational_manifest'))
  recovery.validate_sources(p,closure,authority,context,manifest,freeze['source_pins'])
  need(self.outer['operational_source_commit']==manifest['source_commit'],'outer operational Git identity differs')
  for name in p['source_names']:
   if name in self.oracle_pins and name!=ENTRY:need(p['files'][name]==self.oracle_pins[name],'original scientific oracle differs')
  for role,key in [('authority','recovery_authority'),('context','recovery_context'),('receipt','preflight_receipt'),('read_set','readset')]:
   need(p['files'][p['recovery_files'][role]]==self.c['artifacts'][key]['sha256'],'independent recovery role differs')
  for n,h in p['files'].items():self.h('control/inputs/'+aid.name(n),h)
  need(self.subtree('control/inputs/')==dict(p['files'],**{'packet.json':self.packet_pin,'admission.json':self.c['artifacts']['admission']['sha256']}),'complete GPU inputs membership differs')
  for n,h in self.outer['new_input_files'].items():self.h('control/'+aid.name(n),h)
  root=recovery.CONTROL;cpu_root=supp.constant(builder,'CPU_ROOT')
  need(p['paths']==dict(cpu_inputs=cpu_root+'/inputs',cpu_output=cpu_root+'/cpu',campaign=cpu_root+'/cpu/worker/artifacts/campaign',gpu_inputs=root+'/inputs',gpu_output=recovery.OUTPUT)
       and self.d['remote_root']==self.outer['remote_root']==root and self.d['output_root']==recovery.OUTPUT and self.cpu['campaign_root']==p['paths']['campaign'],'frozen original endpoint/output roots')
  admission=aid.decode(self.artifact('admission'))
  need(admission['packet_sha256']==self.packet_pin and admission['resources']==resources and admission['limits']==limits
       and admission['freeze_sha256']==FREEZE and admission['accounting_sha256']==self.c['artifacts']['baseline']['sha256'],'original admission binding')
  need(set(self.cpu)==set(supp.constant(builder,'CPU_ACCEPTANCE_KEYS')) and self.cpu['schema']=='ace-scientific-campaign-cpu-acceptance-v1'
       and self.cpu['status']=='development_cpu_phase_accepted' and self.cpu['scientific_qualified'] is False
       and self.cpu['cpu_packet_sha256']==p['cpu_packet_sha256'] and p['cpu_acceptance_sha256']==self.c['artifacts']['cpu_acceptance']['sha256'],'independent accepted CPU authority differs')
  self.h('control/inputs/cpu-packet.json',p['cpu_packet_sha256'])
  for n,h in self.old['files'].items():need(p['files'].get(n)==h,'original CPU input changed');self.h('cpu/inputs/'+n,h)
  need({n:h for n,h in self.subtree(CPU).items() if not n.startswith('worker/artifacts/campaign/')}==self.cpu['cpu_output_files'],'original CPU output changed')
  need({n:h for n,h in self.subtree(C).items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}}==self.cpu['campaign_files'],'CPU campaign changed outside three GPU attempts')
  self.h(CPU+self.cpu['bundle_path'],self.cpu['bundle_sha256'])
  previous=aid.decode(self.artifact('cpu_remote_custody'));previous_d=aid.decode(self.artifact('cpu_disposition'))
  self.h('control/cpu-disposition.json',self.c['artifacts']['cpu_disposition']['sha256'])
  need(previous_d['job_id']==self.cpu['job_id'] and previous_d['remote_custody_receipt_sha256']==sha(raw(previous))
       and saved_inventory(previous_d['remote_files'],previous_d['remote_directories'])==previous['full_inventory'],'independent accepted CPU receipt/disposition join')
  need(previous['full_backup_verified'] is True and previous['full_local_copy'] is False,'accepted original CPU archive required')
  cpu_ancestor_inventory(previous,self.receipt)
  for n,h in [('namespace.json',self.c['artifacts']['namespace_claim']['sha256']),('original-namespace.json',self.c['artifacts']['original_namespace_claim']['sha256'])]:self.h(n,h)
  need(self.d['cpu_acceptance_sha256']==p['cpu_acceptance_sha256'],'collector CPU acceptance join')
  recovery.recovery(self,preflight,aid.decode,raw,aid.validate_inventory)
  self.facts['packet']=dict(source_commit=SOURCE,source_files=len(p['source_names']),inputs=len(p['files']),inner_sha256=self.packet_pin,reused_cpu_acceptance=True)
 def recovery_fetch_requirements(self):
  p=aid.decode(self.artifact('inner'))
  need(p['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2','new packet schema required')
  wanted={'control/inputs/cpu-packet.json'}
  need(set(p['services'])==set(LLMS),'exact three service freezes required')
  wanted.update('control/inputs/'+aid.name(p['services'][policy]) for policy in LLMS)
  logs={'control/logs/'+self.c['job_id']+suffix for suffix in ('.out','.err')}
  wanted.update(logs)
  need({n for n in self.full if n.startswith('control/logs/')}==logs,'exact current-job stdout/stderr custody required')
  for n in sorted(wanted):
   need(n in self.full,'required control metadata/log member absent: '+n)
   need(self.full[n]['bytes']<=32*1024**2,'control member exceeds bounded exact materialization; separate reviewed path needed')
   if n not in self.local and n not in self.c.get('fetched',{}):self.missing[n]=self.full[n]
  need(sum(self.full[n]['bytes'] for n in wanted)<=512*1024**2,'control fetch aggregate exceeds bound')
  self.facts['control_selection']={'required_members':{n:self.full[n] for n in sorted(wanted)},
      'collector_selects_control':False,'batch03_selects_control':False,'control_log_content_interpreted':False}
  need(not self.missing,'new control metadata/log exact fetches required; batch03 is insufficient')
  for n in sorted(logs):self.b(n)  # Hash opaque log bytes; do not parse outcomes.
 def fetch_requirements(self):
  wanted={C+'source-plan-pins.json'}
  for i in range(24):wanted.add(C+f'policy-00/world-{i:02d}/episode/plan.json')
  for n in self.full:
   if any(n.startswith(C+f'policy-{i:02d}/') for i in (5,6,7)):
    rel=PurePosixPath(n)
    if rel.name in {'attempt.json','arm-receipt.json','execution-diagnostics.json','task.json','paired-plan.json'} or '/service/' in n or '/episode/' in n or rel.name.endswith('-world.jsonl') or rel.name.startswith('return-'):
     if n.endswith(('.json','.jsonl')) and '/sealed-inputs/' not in n:wanted.add(n)
  for n in sorted(wanted):
   need(n in self.full,'required raw metadata member absent: '+n)
   if n not in self.local and n not in self.c.get('fetched',{}):self.missing[n]=self.full[n]
  need(not self.missing,'pin-bound omitted metadata fetches required; see report exact map')
 def accounting(self):
  baseline=aid.decode(self.artifact('baseline'));job=self.c['job_id'];obs=self.before;rows=obs['related_accounting']
  prior=[r for r in rows if r[0]!=job];current=[r for r in rows if r[0]==job]
  need(baseline['terminal_jobs']==len(prior)==47 and len(rows)==48 and len(current)==1
       and baseline['submission_journals']==len(baseline['journal_pins'])==96 and aid.normalize(prior)==aid.normalize(baseline['related_accounting']),'exact47+1 actual history differs')
  need(all(r[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'} for r in aid.normalize(rows)),'active/nonterminal history')
  r=aid.normalize(current)[0];resources=self.p['resources'];elapsed=r[5];t=r[7]
  need(r==aid.normalize([obs['parent_accounting']])[0] and r[2:4]==['COMPLETED','0:0'] and r[1]=='ace_llm_scientific_campaign_gpu02' and r[4]==4 and 0<elapsed<=resources['wall_seconds']
       and r[6]=='ucb736_asc2' and t.get('cpu')=='4' and t.get('node')=='1' and t.get('mem') in ('15G','15.0G','15360M')
       and t.get('gres/gpu:a100_3g.20gb')=='1' and all(k in ('gres/gpu','gres/gpu:a100_3g.20gb') for k in t if k.startswith('gres/'))
       and t.get('gres/gpu','1')=='1','actual4CPU15GiB/one20GB allocation differs')
  details=obs['results']['details'];need(details['returncode']==0,'saved scheduler details unavailable')
  detail=[line.split('|') for line in details['stdout'].splitlines() if line.strip()]
  need(len(detail)==1 and detail[0][:4]==[job,'aa100','gpu-testing',str(resources['wall_seconds']//60)] and detail[0][5]=='paco0228','requested wall/owner/account differs')
  requested=dict(v.split('=',1) for v in detail[0][4].split(',') if v)
  need(requested.get('cpu')=='4' and requested.get('mem') in ('15G','15.0G','15360M') and requested.get('gres/gpu:a100_3g.20gb')=='1','requested GPU resources differ')
  observed_allocation=allocation_facts(self.j(G+'allocation.json'),self.j(G+'scheduler_raw.json'),job,resources)
  result=aid.decode(self.artifact('result'));local=aid.decode(self.artifact('local_intent'));remote=aid.decode(self.artifact('remote_intent'))
  need(result['job_id']==job and result['returncode']==0 and result['intent_sha256']==self.c['artifacts']['remote_intent']['sha256'],'once-only submission result')
  need(set(remote)==set(local)|{'remote_utc','fresh_readiness'} and {k:remote[k] for k in local}==local,'original intent semantic join')
  journals=dict(baseline['journal_pins']);journals[self.d['remote_root']+'/submission.intent.json']=result['intent_sha256'];journals[self.d['remote_root']+'/submission.result.json']=self.c['artifacts']['result']['sha256']
  need(len(journals)==98 and obs['journal_pins']==journals,'full96+2 journals differ')
  recovery.history_extension(baseline,rows,obs['journal_pins'],job,result['intent_sha256'],self.c['artifacts']['result']['sha256'],aid.normalize)
  old=aid.costs(prior);new=aid.costs(rows)
  need(old==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']) and new==(old[0]+4*elapsed,old[1]+elapsed)
       and baseline['outstanding_reservation']==0 and baseline['preparation_allowance_cpu_seconds']==3600,'actual accounting extension differs')
  need(new[0]+3600<=57600 and new[1]<=7200 and self.d['closed_cpu_seconds']==new[0] and self.d['closed_gpu_slice_seconds']==new[1]
       and self.d['cpu_admission_after_release']==new[0]+3600,'disposition costs/caps differ')
  self.facts['accounting']=dict(requested_and_original_running=observed_allocation,steps=obs['results']['steps'],jobs=48,journals=98,elapsed_seconds=elapsed,allocated_cpu_seconds=4*elapsed,gpu_slice_seconds=elapsed,closed_cpu_seconds=new[0],closed_gpu_seconds=new[1],preparation_cpu_allowance=3600,utilization_inferred=False)
 def runtime(self):
  self.stage=self.j(G+'stage.json');self.stage_pin=self.h(G+'stage.json');self.handoff=self.j(G+'handoff.json')
  stage=self.stage;h=self.handoff;contract=self.old['runtime'];prefix=supp.remote_path(stage['staged_prefix'])
  need(h['packet_sha256']==self.packet_pin and h['job_id']==self.c['job_id'] and h['stage_sha256']==self.stage_pin
       and h['binding_sha256']==self.h(G+'stage-binding.json') and h['allocation_sha256']==self.h(G+'allocation.json'),'runtime handoff bindings')
  need(stage['archive_sha256']==contract['archive_sha256'] and stage['manifest_sha256']==contract['manifest_sha256']
       and stage['objects']==49966 and stage['file_bytes']==8397575073 and stage['model_loaded'] is stage['runtime_qualified'] is False,'original staged runtime pins/shape')
  diagnostic=self.oracles['qualify_llm_intervention_runtime_metadata.py'];pairs=supp.constant(diagnostic,'CACHE_PAIRS')
  generated=supp.constant(diagnostic,'GENERATED_MODULE');expected_prov=supp.constant(diagnostic,'PROVENANCE_SOURCES')
  extra=supp.constant(self.oracles['llm_intervention_service_runtime.py'],'EXTRA_PACKAGES')
  provenance=self.j(G+'independent-provenance-sources.json')
  need(set(provenance['sources'])==set(expected_prov) and sha(provenance['expected_source'].encode())==provenance['expected_source_sha256'],'generated source capture')
  for n,(rel,pin) in expected_prov.items():need(provenance['sources'][n]['sha256']==pin and provenance['sources'][n]['path']==str(prefix/'lib/python3.10/site-packages'/rel),'independent provenance source pin/path')
  need(set(stage['cache_source_sha256'])=={n for pair in pairs for n in pair},'cache/source stage roster')
  records=[]
  for pid in (None,*LLMS):
   base=G+'controller/runtime/' if pid is None else G+'services/'+pid+'/worker/'
   bound=self.j(base+'runtime-stage-binding.json');before=self.j(base+'runtime-before-imports.json')
   need(bound['stage']==stage and bound['stage_sha256']==self.stage_pin and bound['contract']==contract
        and bound['gpu_mapping_enabled'] is (pid is not None),'controller/worker runtime stage policy')
   supp.check_prefix(prefix,before,generated);need(before['python_version']=='3.10.19' and not before['generated_provenance']['present'],'before-imports snapshot')
   need(self.j(base+'runtime-provenance-sources.json')==provenance,'independent saved provenance source mismatch')
   metadata=self.j(base+'runtime-metadata.json');candidates=collections.defaultdict(list)
   for row in metadata['distributions']:
    candidates[re.sub('[-_.]+','-',row['name']).lower()].append(row['version'])
    need(supp.inside(row['metadata_path'],prefix) and supp.inside(row['distribution_root'],prefix),'mixed metadata paths')
   need(all(sorted(candidates[n])==v for n,v in contract['metadata_candidates'].items()) and sorted(r['version'] for r in metadata['numpy_entries'])==['2.2.5','2.2.6'],'original duplicate NumPy/all metadata candidates')
   need(all(supp.inside(r['metadata_path'],prefix) for r in metadata['resolved_defaults'].values()),'mixed resolved metadata')
   cache=self.j(base+'cache-reconciliation.json');need(cache['no_cache_modified'] is True and len(cache['pairs'])==len(pairs),'cache changed/missing')
   for row,(c,s) in zip(cache['pairs'],pairs):
    need(row['cache_path']==str(prefix/c) and row['source_path']==str(prefix/s) and row['cache_sha256']==stage['cache_source_sha256'][c]
         and row['source_sha256']==stage['cache_source_sha256'][s] and row['matches_stage_pins'] is row['admissible_for_diagnosis'] is True
         and row['cache_executed_by_comparison'] is False and (row.get('cpython_timestamp_reject_expected') is True or row.get('code_equivalent') is True),'cache reconciliation')
   phases=('after-service-imports','after-campaign-host') if pid is None else ('after-service-imports','after-model-load','after-requests')
   freeze=None if pid is None else self.j('control/inputs/'+self.p['services'][pid])
   expected={PurePosixPath(n).stem:self.p['files'][n] for n in self.p['source_names'] if n!=ENTRY} if pid is None else {PurePosixPath(n).stem:v for n,v in freeze['sources'].items() if n!='llm_intervention_model_service.py'}
   original_input=self.p['paths']['gpu_inputs'] if pid is None else self.p['paths']['cpu_inputs']
   for phase in phases:
    snapshot=self.j(base+'runtime-'+phase+'.json');view=dict(snapshot)
    mapped=[]
    for line in snapshot['process_maps'].splitlines():
     fields=line.split(None,5)
     if len(fields)==6 and fields[5].startswith('/'):
      p=fields[5].removesuffix(' (deleted)')
      if supp.inside(p,self.old['runtime_source']) or not supp.inside(p,prefix) and not any(supp.inside(p,r) for r in ('/lib','/lib64','/usr/lib','/usr/lib64')):mapped.append((p,fields[1],line))
    need(sorted(set(p for p,_,_ in mapped))==snapshot['mixed_native_paths'],'raw native-map rejection list differs')
    if mapped:
     need(pid is not None and phase in ('after-model-load','after-requests'),'device maps before model/inside controller')
     inv=self.j(base+'gpu-mapping-inventory-'+phase+'.json');val=self.j(base+'gpu-mapping-validation-'+phase+'.json')
     need(val['device_inventory']==inv and val['original_mixed_native_paths']==snapshot['mixed_native_paths']
          and val['original_snapshot_preserved'] is True and val['bound_runtime_prefix']==str(prefix)
          and val['bound_original_runtime']==self.old['runtime_source'] and sha(inv['proc_devices'].encode())==inv['proc_devices_sha256']
          and {r['line'] for r in val['classified_rows']}=={r[2] for r in mapped},'saved GPU classification/raw maps binding')
     for p,permissions,line in mapped:
      need(p in inv['devices'] and re.fullmatch(r'[r-][w-]-s',permissions) and (p in ('/dev/nvidiactl','/dev/nvidia-uvm','/dev/zero') or re.fullmatch('/dev/nvidia[0-9]+',p)),'unexpected native mapping')
      device=inv['devices'][p];need(device['uid']==0 and stat.S_ISCHR(device['mode']),'device ownership/type')
     view['mixed_native_paths']=[]
    supp.check_prefix(prefix,view,generated,before['sys_path'])
    need(snapshot['python_version']=='3.10.19' and set(snapshot['packages'])=={'numpy','scipy','torch','transformers'}
         and set(snapshot['additional_package_versions'])==set(extra),'runtime package roster')
    versions={k:v['version'] for k,v in snapshot['packages'].items()};versions.update(snapshot['additional_package_versions']);proof=snapshot['generated_provenance']
    need(versions==contract['imported_versions'] and all(v['loaded'] for v in snapshot['packages'].values()) and proof['verified'] is True
         and proof['source_content']==provenance['expected_source'] and proof['source_sha256']==proof['expected_source_sha256']==provenance['expected_source_sha256']
         and proof['temporary_root']==bound['temporary_root'],'actual imports/generated provenance')
    need(all(supp.inside(p,prefix) or p==proof['path'] for p in snapshot['module_paths'].values()),'mixed Python module origins')
    projects=snapshot['captured_project_modules']
    need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'} and all(projects[n]['sha256']==pin for n,pin in expected.items()),'captured complete project closure')
    for n,r in projects.items():
     filename=PurePosixPath(r['path']).name
     need(filename in self.p['files'] and r['sha256']==self.p['files'][filename] and str(PurePosixPath(r['path']).parent)==original_input,'captured project input path/hash')
    records.append({'member':base+'runtime-'+phase+'.json','sha256':self.h(base+'runtime-'+phase+'.json'),'device_mapping_rows':len(mapped)})
  self.facts['runtime']=dict(stage_sha256=self.stage_pin,snapshots=records,original_validator_executed=False,full_device_predicate_review_remaining=True)
 def terminals(self):
  self.t=self.j(G+'terminal.json');self.phase=self.j(G+'controller/phase/terminal.json');start=self.j(G+'controller/phase/start.json')
  t=self.t;phase=self.phase;origin=self.handoff['started_monotonic'];limits=self.p['limits'];self.origin=origin
  need(t['schema']==self.p['schema'] and t['status']=='development_gpu_campaign_complete' and t['packet_sha256']==self.packet_pin
       and t['cpu_acceptance_sha256']==self.p['cpu_acceptance_sha256'] and t['cpu_packet_sha256']==self.p['cpu_packet_sha256']
       and t['admission_sha256']==self.c['artifacts']['admission']['sha256'] and t['stage_sha256']==self.stage_pin
       and t['new_training_updates']==0 and t['private_scores_computed'] is t['scientific_qualified'] is False
       and t['limits']==limits and t['resources']==self.p['resources'] and 0<t['elapsed_seconds']<limits['final_seconds'],'outer successful inference-only terminal')
  need(phase['status']=='gpu_phase_complete' and phase['failure'] is None and phase['errors']==phase['unstarted_not_recorded']==[]
       and phase['signals_restored'] is True and phase['cleanup_handoff'] is None and phase['new_training_updates']==phase['cpu_control_policies_run']==0
       and phase['private_scores_computed'] is phase['retry_allowed'] is phase['scientific_acceptance'] is False,'whole phase failure/cleanup')
  need(phase['campaign_sha256']==t['campaign_sha256']==start['campaign_sha256'] and phase['attempt_pins']==t['attempt_pins'] and set(t['attempt_pins'])==set(LLMS)
       and t['combined_attempt_pins']=={**self.cpu['attempt_pins'],**t['attempt_pins']} and set(t['combined_attempt_pins'])==set(POLICIES),'all8 attempt authority')
  need(start['outer_started']==origin and start['limits']==limits and start['work_deadline']==origin+limits['work_seconds']
       and start['final_deadline']==origin+limits['final_seconds'] and start['first_global_seconds']==240 and start['first_service_seconds']==120 and start['freshness_seconds']==90
       and origin<=start['phase_entered']<start['work_deadline'] and start['work_deadline']-start['phase_entered']>=3*limits['session_reserve_seconds'],'inherited absolute phase deadlines')
  need([s['policy_id'] for s in phase['sessions']]==list(LLMS) and len({s['registered_pid'] for s in phase['sessions']})==3,'three fixed original sessions')
  need(phase['sessions'][0]['first_verified_monotonic']-origin<240,'first global GPU work later than240')
  for prior,next_session in zip(phase['sessions'],phase['sessions'][1:]):need(prior['closed_monotonic']<=next_session['started_monotonic'],'overlapping/reloaded sessions')
  recovery.recovery_terminal(self)
  self.facts['terminal']=dict(combined_attempt_pins=t['combined_attempt_pins'],outer_seconds=t['elapsed_seconds'],scientific_qualified=False,training_updates=0)
 def cell(self,base,arm,index,policy,identity,offset,pid):
  path=base+f'world-{index:02d}/';receipt=self.j(path+'arm-receipt.json');self.h(path+'arm-receipt.json',arm['receipt_sha256']);self.h(path+'arm-receipt.candidate.json',arm['receipt_sha256'])
  need(arm['status']=='completed' and arm['reason'] is None and arm['path']==f'world-{index:02d}' and arm['world_id']==WORLDS[index],'all24 original cells, no filtering')
  need(receipt['status']=='raw_ready' and receipt['unresolved_reason'] is None and receipt['policy_id']==policy and receipt['plan_sha256']==self.plan_pins[index]['sha256']
       and receipt['new_training_updates']==0 and receipt['private_scores_computed'] is receipt['retry_allowed'] is receipt['scientific_provenance_qualified'] is False
       and receipt['source_evidence_sha256']==self.subtree(path,('arm-receipt.json','arm-receipt.candidate.json')),'arm publication/evidence failed')
  for n,h in receipt['input_sha256'].items():self.h(path+'sealed-inputs/'+aid.name(n),h)
  for n in ('world.json','task.json','learner.json','paired-plan.json'):
   expected=self.cpu['campaign_files'][f'policy-00/world-{index:02d}/'+n];self.h(path+n,expected)
   if n!='paired-plan.json':self.h(path+'sealed-inputs/'+n,expected)
  ep=self.j(path+'episode/plan.json');old_ep=self.j(C+f'policy-00/world-{index:02d}/episode/plan.json')
  spec=self.old['persistent_specs'][policy]
  need({k:v for k,v in ep.items() if k not in ('policy_command','policy_timeout_seconds')}=={k:v for k,v in old_ep.items() if k not in ('policy_command','policy_timeout_seconds')}
       and ep['policy_command']==spec['command'] and ep['policy_timeout_seconds']==spec['timeout_seconds'],'original public episode menu/plan/endpoint differs')
  for original,sealed in [('episode/plan.json','episode-plan.json'),('episode/journal.jsonl','executor.jsonl'),('episode/terminal.json','episode-disposition.json')]:self.h(path+original,receipt['input_sha256'][sealed])
  diag=self.j(path+'execution-diagnostics.json');need(diag['errors']==[] and diag['episode_returned_cleanly'] is diag['adapters_closed_cleanly'] is True,'episode cleanup failure')
  counts=self.journal(path+'episode/journal.jsonl');events=[parts(line) for line in self.b(path+'episode/journal.jsonl').splitlines()]
  proposals=[];reserves={};history=[];contexts=[];menu={a['action_id']:a for a in ep['menu']}
  for event in events:
   kind=aid.decode(event['event'])
   if kind=='proposal':
    proposal=public(b'{'+b','.join(canonical(k)+b':'+v for k,v in event.items())+b'}',('proposal_id','provenance','completion'))
    if proposal['provenance']['kind']=='process_policy':contexts.append((proposal,list(history)));proposals.append(proposal)
   elif kind=='reserve':
    r={k:aid.decode(event[k]) for k in ('request_id','proposal_id','cost_units','action')}
    need(r['action']==menu[r['action']['action_id']] and r['cost_units']==r['action']['cost_units'],'charged action/menu disagreement');reserves[r['proposal_id']]=r
   elif kind=='response':history.append(b'{'+b','.join(canonical(k)+b':'+event[k] for k in sorted(('action_id','cost_units','kind','request_id','targets','values')))+b'}')
  terminal_parts=parts(self.b(path+'episode/terminal.json'));top={k:aid.decode(terminal_parts[k]) for k in ('status','initialized_responses','policy_decisions','scientific_evaluation_performed')}
  state=public(terminal_parts['final_state'],('spent_units','remaining_units','requests','returned_responses','terminal'))
  need(top['status']=='complete' and top['initialized_responses']==8 and top['policy_decisions']==len(contexts) and top['scientific_evaluation_performed'] is False
       and state==dict(spent_units=counts['cost_units'],remaining_units=32-counts['cost_units'],requests=counts['responses'],returned_responses=counts['responses'],terminal=True),'terminal exact charge/decision totals')
  session=receipt['session'];end=offset+len(contexts)
  need(session==dict(kind='persistent',identity_sha256=identity,count_before=offset,request_before=offset-1,count_after=end,request_after=end-1,pid=pid),'cell session range/owned PID')
  for n in receipt['input_sha256']:
   if n.startswith('return-'):self.h(path+n,receipt['input_sha256'][n])
  for rid in range(8):self.h(path+f'return-{rid:03d}.json',self.cpu['campaign_files'][f'policy-00/world-{index:02d}/return-{rid:03d}.json'])
  task=self.j(path+'task.json');binding=self.j(base+f'service/task-binding-{index:04d}.json')
  need(binding['binding_index']==index and binding['task']==task and binding['task_sha256']==digest(task)
       and binding['previous_task_sha256']==(None if index==0 else self.cells[-1]['task_sha256']) and binding['last_completed_session_request_id']==offset-1 and binding['worker_pid']==pid and binding['model_reloaded'] is False,'original task rebinding')
  contexts=[dict(ordinal=offset+i,world_id=WORLDS[index],index=index,decision=i,proposal=p,history=rows,reserve=reserves.get(p['proposal_id']),task=task,menu=ep['menu']) for i,(p,rows) in enumerate(contexts)]
  self.worlds(path,policy,index,events,counts)
  cell=dict(policy_id=policy,world_id=WORLDS[index],task_sha256=digest(task),policy_decisions=len(contexts),**counts)
  self.cells.append(cell);return end,contexts
 def services(self):
  self.plan_pins=self.j(C+'source-plan-pins.json')
  need([r['world_id'] for r in self.plan_pins]==WORLDS,'24 ordered supplied worlds')
  tops={n[len(C):].split('/')[0] for n in list(self.full)+self.receipt['full_inventory']['directories'] if n.startswith(C)}
  need(tops=={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}' for i in range(8)},'extra attempt/finalization directory')
  sessions=[];total_generations=0
  training=aid.decode(self.artifact('training_acceptance'))
  need(training['schema']=='ace-scientific-training-gpu-acceptance-v1' and training['status']=='scientific_training_gpu_accepted'
       and training['required_findings']==0 and training['training_gpu_qualified'] is training['complete_custody_verified'] is True
       and training['scientific_utility_qualified'] is False and training['source_commit']==SOURCE and training['scientific_freeze_sha256']==FREEZE,'independent accepted training lineage')
  self.h('control/inputs/gpu-training-acceptance.json',self.c['artifacts']['training_acceptance']['sha256'])
  for pi,policy in enumerate(LLMS,5):
   base=C+f'policy-{pi:02d}/';worker=G+'services/'+policy+'/worker/';client=base+'service/'
   attempt=self.j(base+'attempt.json');pin=self.t['attempt_pins'][policy]
   self.h(base+'attempt.json',pin);self.h(base+'attempt.candidate.json',pin)
   need(attempt['policy_id']==policy and attempt['phase']=='gpu' and attempt['campaign_sha256']==self.t['campaign_sha256']
        and attempt['source_plan_pins']==self.plan_pins and attempt['errors']==[] and attempt['retry_allowed'] is attempt['private_scores_computed'] is False
        and len(attempt['arms'])==24 and attempt['evidence_sha256']==self.subtree(base,('attempt.json','attempt.candidate.json')),'whole original GPU attempt failure/membership')
   freeze=self.j('control/inputs/'+self.p['services'][policy]);auth=self.j(worker+'authenticated.json');identity=auth['identity'];ip=digest(identity)
   expected_identity=dict(model_sha256=digest(freeze['model_files']),source_sha256=digest(freeze['sources']),
      adapter_sha256=freeze['adapter']['sha256'] if freeze['adapter'] else digest(dict(treatment='frozen',adapter=None)),
      numeric_recipe=freeze['recipe'],numeric_recipe_sha256=digest(freeze['recipe']))
   need(identity==expected_identity==self.old['services'][policy]['identity'] and ip==self.old['service_identity_pins'][policy]
        and auth['freeze_sha256']==self.p['files'][self.p['services'][policy]] and auth['runtime']==self.old['runtime']
        and set(auth['models'])==set(freeze['model_files']) and all(v['sha256']==freeze['model_files'][n] and type(v['bytes']) is int and v['bytes']>0 for n,v in auth['models'].items())
        and freeze['scope']=='development' and freeze['limits']==dict(startup_seconds=120,request_seconds=30,idle_seconds=30,total_seconds=1200,max_requests=576),'actual authenticated model/source/identity/limits')
   if policy=='frozen_llm':need(freeze['adapter'] is None and worker+'adapter_restoration.json' not in self.full,'frozen adapter injected')
   else:
    role='sft' if policy=='sft_llm' else 'final';adapter=freeze['adapter'];restored=self.j(worker+'adapter_restoration.json')
    need(adapter['sha256']==training[role+'_adapter_sha256'] and adapter['training_receipt_sha256']==training[role+'_checkpoint_sha256']
         and restored['adapter_sha256']==adapter['sha256'] and restored['training_receipt_sha256']==adapter['training_receipt_sha256'],'accepted checkpoint role/restoration')
    self.h('control/inputs/'+adapter['file'],adapter['sha256']);self.h('control/inputs/'+adapter['training_receipt'],adapter['training_receipt_sha256'])
   session=self.phase['sessions'][pi-5];process=self.j(client+'process.json');close=self.j(client+'closed.json');cleanup=self.j(client+'close-cleanup.json')
   launch=self.j(G+'services/'+policy+'/launch/execution.json');terminal=self.j(worker+'terminal.json');n=close['completed_requests'];pid=session['registered_pid']
   need(type(n) is int and 24<=n<=576 and close['status']=='closed' and close['last_request_id']==n-1 and close['cleanup']==cleanup
        and cleanup['errors']==[] and cleanup['group_still_exists'] is False and cleanup['leader_returncode']==0 and cleanup['term_sent'] is cleanup['kill_sent'] is False
        and cleanup['pgid']==pid==process['pid']==process['pgid']==launch['pid'],'three natural owned sessions/cleanup')
   need(attempt['service']==dict(status='recorded',file='service/closed.json',sha256=self.h(client+'closed.json')),'original attempt/closure authority')
   client_launch=self.j(client+'launch.json');spec=self.old['persistent_specs'][policy];settings=self.old['services'][policy]
   need(client_launch['command']==spec['command'] and client_launch['timeout_seconds']==spec['timeout_seconds'] and client_launch['identity_sha256']==ip
        and client_launch['expected_identity']==identity and all(client_launch[k]==settings[k] for k in ('startup_seconds','cleanup_seconds','max_requests')),'original frozen service command/settings')
   need(launch['contract_sha256']==self.p['files'][self.p['contracts'][policy]] and launch['stage_sha256']==self.stage_pin
        and launch['stage_binding_sha256']==self.handoff['binding_sha256'],'launch source/stage/contract')
   need(terminal['status']=='complete' and terminal['generation_count']==n and terminal['new_training_updates']==terminal['environment_responses']==0
        and terminal['runtime_stage_sha256']==self.stage_pin and terminal['freeze_sha256']==auth['freeze_sha256'],'inference-only worker terminal')
   need(self.j(client+'ready.json')==dict(status='ready',identity_sha256=ip),'client ready')
   hand=self.j(client+'observed-handshake.json');need(hand['type']=='ready' and hand['identity']==identity and hand['identity_sha256']==ip,'original identity handshake')
   offset=0;contexts=[]
   for wi,arm in enumerate(attempt['arms']):
    offset,rows=self.cell(base,arm,wi,policy,ip,offset,pid);contexts.extend(rows)
   need(offset==len(contexts)==n,'unjoined requests/cells')
   families={kind:rows_by_ordinal(self.full,worker,pattern,n) for kind,pattern in dict(request=r'request-(\d{4})\.json',generation=r'generation-(\d{4})\.json',generated=r'generated-(\d{4})\.unaccepted\.json',started=r'raw-generation-(\d{4})\.started\.json',raw=r'raw-generation-(\d{4})\.json',binary=r'raw-generation-(\d{4})\.bin').items()}
   known={name for family in families.values() for name in family.values()}
   need(all(name in known for name in self.full if name.startswith(worker) and PurePosixPath(name).name.startswith(('request-','generation-','generated-','raw-generation-'))),'extra/nested generation artifact')
   clients={kind:rows_by_ordinal(self.full,client,expression,n) for kind,expression in dict(input=r'request-(\d{4})\.input\.json',mapping=r'request-(\d{4})\.episode\.json',wire=r'request-(\d{4})\.observed-response\.json',accepted=r'request-(\d{4})\.accepted\.json').items()}
   known_client={name for family in clients.values() for name in family.values()}
   need(all(name in known_client for name in self.full if name.startswith(client) and PurePosixPath(name).name.startswith('request-')),'unknown client request member')
   rows_by_ordinal(self.full,client,r'task-binding-(\d{4})\.json',24)
   evidence=self.j(G+'controller/'+policy+'-evidence.json');validation=self.t['service_validation'][policy]
   need(evidence['operational_success'] is evidence['counts_complete'] is evidence['clean_close'] is evidence['closure_confirmed'] is True
        and evidence['issues']==[] and evidence['worker_evidence_sha256']==validation['worker_inventory']==self.subtree(worker)
        and validation['attempt_sha256']==evidence['attempt_sha256']==pin and validation['evidence_sha256']==self.h(G+'controller/'+policy+'-evidence.json')
        and evidence['freeze_sha256']==auth['freeze_sha256'] and evidence['identity_sha256']==ip and len(evidence['joins'])==n,'saved validator/complete worker inventory')
   generated=[];policy_stops=0
   for ordinal,ctx in enumerate(contexts):
    self.generation(worker,client,families,clients,ordinal,ctx,identity,session,evidence)
    g=self.j(families['generation'][ordinal]);generated.append(g);policy_stops+=ctx['reserve'] is None
   need(set(session['generation_evidence'])=={str(i) for i in range(n)} and session['positive_generations']==n,'all generation session pins')
   cuda=sum(r['cuda_interval_ms'] for r in generated)
   need(cuda==session['cuda_interval_ms']==evidence['cuda_interval_ms'],'CUDA interval exact sum')
   phases=[self.j(k) for k in sorted(self.full) if k.startswith(worker+'phase-') and PurePosixPath(k).parent==PurePosixPath(worker)]
   names=[p['phase'] for p in phases];load=('model_cpu_load_start','model_cpu_load_complete','model_ready')
   need(all(names.count(k)==1 for k in load) and [names.index(k) for k in load]==sorted(names.index(k) for k in load),'one ordered model load per service')
   ready=phases[names.index('model_ready')];need(ready['identity']==identity and positive(ready['gpu_allocated_bytes']),'model-ready identity/memory')
   complete=[p for p in phases if p['phase']=='generation_complete']
   need(len(complete)==n and {p['ordinal'] for p in complete}==set(range(n)) and all(p['envelope_sha256']==generated[p['ordinal']]['envelope_sha256'] for p in complete),'generation phase roster')
   telemetry=[self.j(k) for k in sorted(self.full) if k.startswith(G+'controller/phase/'+policy+'/telemetry-') and k.endswith('.json')]
   limits=self.p['limits'];work=self.origin+limits['work_seconds'];end=min(session['started_monotonic']+limits['session_cap_seconds'],work-(7-pi)*limits['session_reserve_seconds'])
   facts=telemetry_facts(telemetry,session,self.j(G+'controller/phase/'+policy+'/post-exit-gpu.json'),self.origin,work,self.origin+limits['final_seconds'],end)
   cells=self.cells[-24:];totals=dict(response_attempts=sum(c['responses'] for c in cells),returned_responses=sum(c['responses'] for c in cells),charged_units=sum(c['cost_units'] for c in cells),acquisition_charged_units=sum(c['acquisition_cost_units'] for c in cells))
   need(evidence['limits']==dict(worlds=24,campaign_cells=192,service_requests=576,all_service_decisions=1728,campaign_response_attempts=6144,campaign_cost_units=6144),'scientific saved-validator limits differ')
   counts=evidence['counts'];need(all(counts[k]==v for k,v in totals.items()) and counts['accepted_requests']==counts['attempted_requests']==counts['generations']==n
        and counts['task_bindings']==24 and counts['model_loads']==1 and counts['failed_cells']==counts['unstarted_cells']==0 and counts['policy_stops']==policy_stops and counts['budget_stops']==sum(c['cost_units']==32 for c in cells) and counts['budget_stops']+policy_stops==24
        and counts['feedback_to_next_decision']==sum(c['policy_decisions']-1 for c in cells),'saved aggregate counts disagree with all24 cells')
   sessions.append(dict(policy_id=policy,framework_device=ready['device'],cuda_build=ready['cuda_build'],framework_total_device_memory_bytes=None,generations=n,cells=24,cuda_interval_ms=cuda,generated_tokens=sum(len(g['sequence_ids'])-len(g['prompt_ids']) for g in generated),
       peak_framework_allocated_bytes=max(g['peak_gpu_allocated_bytes'] for g in generated),peak_framework_reserved_bytes=max(g['peak_gpu_reserved_bytes'] for g in generated),**facts,**totals));total_generations+=n
  need(len(self.cells)==72 and self.phase['phase_counts']['services_started']==3 and self.phase['phase_counts']['positive_generations']==total_generations,'exact72 cells/3 sessions/phase work totals')
  self.facts['sessions']=sessions;self.facts['campaign']=dict(cells=72,declared_cells=192,generations=total_generations,shared_initial_charged_responses=576,
      total_responses=sum(c['responses'] for c in self.cells),total_cost_units=sum(c['cost_units'] for c in self.cells),new_training_updates=0,utility_evaluated=False)
 def generation(self,worker,client,families,clients,i,ctx,identity,session,evidence):
  g=self.j(families['generation'][i]);request=self.j(families['request'][i]);unaccepted=self.j(families['generated'][i]);rg=self.j(families['raw'][i])
  need(self.j(families['started'][i])==dict(tokens_available=False,status='generation_invoked'),'raw invocation marker')
  need(set(unaccepted)==set(('version','envelope_sha256','task_sha256','prompt_sha256','prompt_tokens','longest_command_tokens','max_tokens','max_new_tokens','legal_commands','enable_thinking','prompt_ids','sequence_ids','completion','new_training_updates','environment_callbacks_available'))
       and unaccepted['version']=='ace-shared-task-language-policy-v1','exact generation publication schema')
  need(set(g)==set(unaccepted)|{'ordinal','elapsed_seconds','cuda_interval_ms','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','tensor_versions_unchanged','cross_request_cache_supplied'}
       and all(g[k]==v for k,v in unaccepted.items()),'unaccepted/accepted generation record differs')
  inp=parts(self.b(clients['input'][i]));env=parts(inp['envelope']);state=parts(env['state']);envelope_pin=sha(opaque_compact(inp['envelope']))
  recipe=identity['numeric_recipe'];seq=g['sequence_ids'];prompt=g['prompt_ids']
  need(type(seq) is list and type(prompt) is list and prompt and all(type(t) is int and 0<=t<151936 for t in seq)
       and seq[:len(prompt)]==prompt and seq[-1]==151645 and len(prompt)==g['prompt_tokens']
       and 1<len(seq)-len(prompt)<=g['longest_command_tokens']<=recipe['max_new_tokens'] and len(prompt)+recipe['max_new_tokens']<=recipe['max_tokens']
       and g['max_tokens']==recipe['max_tokens'] and g['max_new_tokens']==recipe['max_new_tokens'],'recorded token prefix/limits')
  need(g['ordinal']==i and positive(g['cuda_interval_ms']) and positive(g['elapsed_seconds']) and type(g['peak_gpu_allocated_bytes']) is int and g['peak_gpu_allocated_bytes']>0
       and type(g['peak_gpu_reserved_bytes']) is int and g['peak_gpu_reserved_bytes']>=g['peak_gpu_allocated_bytes'] and g['new_training_updates']==0
       and g['tensor_versions_unchanged'] is True and g['cross_request_cache_supplied'] is g['environment_callbacks_available'] is g['enable_thinking'] is False,'independent CUDA/memory inference invariants')
  need(g['envelope_sha256']==envelope_pin==aid.decode(inp['envelope_sha256']) and g['task_sha256']==digest(ctx['task'])==sha(opaque_compact(env['task']))
       and aid.decode(inp['request_id'])==i and aid.decode(inp['identity_sha256'])==digest(identity)
       and request==dict(ordinal=i,envelope_sha256=envelope_pin,identity=identity,tokens_available_at_request_start=False),'exact saved request/input/task/model joins')
  need(opaque_compact(state['history'])==b'['+b','.join(ctx['history'])+b']' and aid.decode(state['requests'])==len(ctx['history'])
       and aid.decode(state['terminal']) is False and sha(opaque_compact(env['state']))==ctx['proposal']['provenance']['public_state_sha256'],'opaque response feedback/next-state hash')
  remaining=aid.decode(state['remaining_units']);need(type(remaining) is int and 0<=remaining<=24 and g['legal_commands']==1+sum(a['cost_units']<=remaining for a in ctx['menu']),'recorded feasible command count')
  wire=self.j(clients['wire'][i]);accepted=self.j(clients['accepted'][i]);mapping=self.j(clients['mapping'][i]);ip=digest(identity)
  need(wire==dict(type='completion',request_id=i,identity_sha256=ip,envelope_sha256=envelope_pin,completion=g['completion'])
       and accepted==dict(request_id=i,response_sha256=digest(wire),completion=g['completion']) and ctx['proposal']['completion']==g['completion'],'wire/accepted/executor completion join')
  need(mapping==dict(episode_directory=self.p['paths']['campaign']+f'/policy-{POLICIES.index(session["policy_id"]):02d}/world-{ctx["index"]:02d}/episode',decision_index=ctx['decision'],session_request_id=i,
       task_sha256=g['task_sha256'],task_binding_index=ctx['index'],identity_sha256=ip,status='planned_before_transmission',scope='client_only_episode_to_session_mapping'),'exact episode/session mapping')
  need(session['generation_evidence'][str(i)]==dict(sha256=self.h(families['generation'][i]),cuda_interval_ms=g['cuda_interval_ms'],peak_gpu_allocated_bytes=g['peak_gpu_allocated_bytes']),'session generation original pin')
  need(rg['status']=='returned_unaccepted' and rg['tokens_available'] is True and rg['returned_type']=='Tensor' and rg['dtype']=='torch.int64'
       and rg['shape']==[1,len(seq)] and rg['numel']==rg['captured_elements']==len(seq) and rg['truncated'] is False and rg['token_prefix']==seq,'raw saved tensor metadata')
  self.h(families['binary'][i],rg['raw_sha256']);need(self.full[families['binary'][i]]['bytes']==8*len(seq) and sha(struct.pack('<'+'q'*len(seq),*seq))==rg['raw_sha256'],'opaque raw int64 byte length/metadata hash')
  join=evidence['joins'][i];reserve=ctx['reserve']
  need(join['ordinal']==join['session_request_id']==i and join['world_id']==ctx['world_id'] and join['decision_index']==ctx['decision']
       and join['task_sha256']==g['task_sha256'] and join['envelope_sha256']==envelope_pin and join['cuda_interval_ms']==g['cuda_interval_ms']
       and join['proposal_id']==ctx['proposal']['proposal_id'] and join['accepted'] is True,'original independent saved-evidence join')
  command=aid.decode(g['completion'])
  if reserve is None:need(command=={'stop':True} and join['outcome']=='policy_stop' and join['charged_units']==0 and join['response_request_id'] is None,'uncharged action without stop')
  else:need(command=={'action_id':reserve['action']['action_id']} and join['outcome']=='response' and join['charged_units']==reserve['cost_units'] and join['response_request_id']==reserve['request_id'],'generation charge/response mismatch')

 def worlds(self,path,policy,index,events,counts):
  plan_raw=self.b(path+'paired-plan.json');plan=public(plan_raw,('world_sha256','initial_binding','measurement_bindings'))
  need(sha(opaque_compact(plan_raw))==self.plan_pins[index]['sha256'],'original opaque paired-plan semantic pin')
  reserves={aid.decode(e['request_id']):e for e in events if aid.decode(e['event'])=='reserve'}
  responses={aid.decode(e['request_id']):e for e in events if aid.decode(e['event'])=='response'}
  for filename,stream,binding,ids in [('initial-world.jsonl','initial',plan['initial_binding'],list(range(8))),
       ('measurement-world.jsonl','measurement',plan['measurement_bindings'][policy],list(range(8,counts['responses'])))]:
   rows=world_chain(self.b(path+filename));header={k:aid.decode(v) for k,v in rows[0].items()}
   need(header['event']=='bind' and header['scope']=='development' and header['world_id']==WORLDS[index]
        and header['descriptor_sha256']==header['expected_sha256']==plan['world_sha256'] and header['stream']==stream
        and header['structural'] is False and header['replicate_id']==binding['replicate_id'] and header['request_namespace']==binding['request_namespace']
        and header['reservation_authority']=='ChargedExecutor','original world identity/stream/arm namespace')
   need([aid.decode(r['event']) for r in rows]==['bind']+['attempt','mapping','outcome']*len(ids)+['closed'],'whole world journal event membership/failure')
   mappings=[r for r in rows if aid.decode(r['event'])=='mapping'];outcomes=[r for r in rows if aid.decode(r['event'])=='outcome']
   need([aid.decode(r['request_id']) for r in mappings]==[aid.decode(r['request_id']) for r in outcomes]==ids
        and aid.decode(rows[-1]['responses_prepared'])==len(ids),'all response reservations/outcomes')
   for rid,mp,oc in zip(ids,mappings,outcomes):
    action=aid.decode(reserves[rid]['action'])
    need(aid.decode(mp['descriptor_sha256'])==plan['world_sha256'] and aid.decode(mp['stream'])==stream and aid.decode(mp['structural']) is False
         and aid.decode(mp['noise_key'])==[binding['request_namespace'],binding['replicate_id'],rid,action['action_id'],digest({'kind':action['kind'],'targets':action['targets']})],'original response namespace/action noise-key')
    need(aid.decode(mp['reservation_sha256'])==aid.decode(oc['reservation_sha256'])==aid.decode(reserves[rid]['sha256'])
         and aid.decode(oc['returned_subset_ready']) is True and oc['values']==responses[rid]['values'],'opaque reservation/world/returned value join')
    returned=parts(self.b(path+f'return-{rid:03d}.json'));response=parts(returned['response'])
    need(aid.decode(returned['request_id'])==rid and aid.decode(returned['returned_to_wrapper']) is True
         and aid.decode(response['raw_json_available']) is True and opaque_compact(response['raw_json'])==responses[rid]['values'],'opaque original return bytes')

def allocation_facts(record,raw_record,job,resources):
 need(set(record)=={'fields','allocated_resources','requested_resources'} and raw_record['returncode']==0
      and raw_record['job_id']==job and raw_record['command']==['scontrol','show','job','-o',job],'original scientific allocation record schema')
 fields=dict(re.findall(r'([^\s=]+)=(\S+)',raw_record['stdout']))
 need(record['fields']==fields,'raw/validated allocation fields differ')
 wall=resources['wall_seconds']
 expected=dict(JobId=job,Account='ucb736_asc2',Partition='aa100',QOS='gpu-testing',JobState='RUNNING',NumCPUs='4',NumNodes='1',NumTasks='1',Requeue='0',Restarts='0',TimeLimit=f'{wall//3600:02d}:{wall%3600//60:02d}:{wall%60:02d}')
 expected['CPUs/Task']='4'
 need(all(fields.get(k)==v for k,v in expected.items()) and fields.get('UserId','').startswith('paco0228('),'original exact running allocation')
 for field,key in [('AllocTRES','allocated_resources'),('ReqTRES','requested_resources')]:
  pairs=[v.split('=',1) for v in fields[field].split(',')];need(all(len(p)==2 for p in pairs) and len(dict(pairs))==len(pairs),'duplicate/invalid TRES')
  tres=dict(pairs)
  need(tres==record[key] and tres.get('cpu')=='4' and tres.get('node')=='1' and tres.get('mem') in ('15G','15.0G','15.00G','15360M')
       and tres.get('gres/gpu:a100_3g.20gb')=='1' and tres.get('gres/gpu','1')=='1'
       and all(k in ('gres/gpu','gres/gpu:a100_3g.20gb') for k in tres if k.startswith('gres/')),'actual/requested exact CPU15GiB/one20GBMIG')
 return record

def world_chain(data):
 need(data.endswith(b'\n'),'partial world journal');previous=None;result=[]
 for line in data.splitlines():
  p=parts(line);unsigned=b'{'+b','.join(canonical(k)+b':'+p[k] for k in sorted(p) if k!='sha256')+b'}'
  need(line==b'{'+b','.join(canonical(k)+b':'+p[k] for k in sorted(p))+b'}' and aid.decode(p['previous_sha256'])==previous and aid.decode(p['sha256'])==sha(unsigned),'world hash chain')
  previous=aid.decode(p['sha256']);result.append(p)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--self-sha256',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();source=aid.file_bytes(Path(__file__).resolve(),args.self_sha256,maximum=1024**2)
 config_raw=aid.file_bytes(Path(args.config),args.config_sha256);cfg=aid.decode(config_raw);out=Path(args.output)
 need(out.is_absolute() and str(out)==str(out.resolve()),'exclusive canonical output required')
 inputs=[Path(p) for p in cfg['capsules']]+[Path(v['path']) for v in cfg['artifacts'].values()]+[Path(v['path']) for v in cfg.get('fetched',{}).values()]
 need(all(out!=p and out not in p.parents and p not in out.parents for p in inputs),'output overlaps original evidence')
 out.mkdir(mode=0o700,exist_ok=False);(out/'audit_saved_gpu72.py').write_bytes(source);(out/'config.json').write_bytes(config_raw)
 start=time.monotonic();cpu=time.process_time();view=None
 report=dict(schema='ace-gpu72-recovery-saved-audit-report-v2',acceptance=False,scientific_qualified=False,runtime_qualified=False,
      source_sha256=sha(source),config_sha256=sha(config_raw),dependency_sha256={'aid':AID_SHA,'supplement':SUPPLEMENT_SHA,'recovery':RECOVERY_SHA,'preflight_metadata_checker':PREFLIGHT_SHA},oracle_manifest_sha256=ORACLES_SHA,
      facts={},errors=[],completed_checks=[],missing_exact_fetches={},remaining_independent_gates=list(REMAINING),new_responses=0,new_updates=0,remote_actions=0,models_or_tokenizers_executed=0,all_checker_attempts_must_be_preserved=True)
 try:
  view=Audit(cfg)
  for name in ('custody','recovery_fetch_requirements','packet','accounting','runtime','terminals','fetch_requirements','services'):
   getattr(view,name)();report['completed_checks'].append(name)
  report['status']='saved_checks_complete_independent_review_required'
 except BaseException as exc:
  report['status']='saved_checks_incomplete';report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
  (out/'failure.traceback.txt').write_text(traceback.format_exc())
 finally:
  if view is not None:
   report['facts']=view.facts;report['missing_exact_fetches']=view.missing;report['opaque_archived_members']=view.opaque;report['completed_cells']=view.cells
  report.update(process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-start,maxrss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (out/'report.json').write_bytes(raw(report))
 print(json.dumps({'status':report['status'],'acceptance':False,'errors':len(report['errors']),'missing_exact_fetches':len(report['missing_exact_fetches']),'report_sha256':sha(raw(report))}))
 return 0 if not report['errors'] else 1

if __name__=='__main__':raise SystemExit(main())
