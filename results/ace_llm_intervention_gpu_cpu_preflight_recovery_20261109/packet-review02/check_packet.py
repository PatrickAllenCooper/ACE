"""Independent read-only concrete metadata/hash review. No packet module imports or remote calls."""
from pathlib import Path
import ast,datetime,hashlib,json,os,re,shlex,stat,subprocess,time,traceback
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
P=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch02'
E=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261109-02'
OLD=P.parent/'launch01'
R=Path(__file__).resolve().parent
NEWROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
OLDROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
ANCHORS={'packet.json':'25fb597c6f475fa0bbfe924062bc439bb197ca0b646cecdae2be1a080010d684',
 'inputs/packet.json':'7d37d1ed4aca9b6471ec2e9a54984ada29c0bc7b36a59e03f10aea51b9998205',
 'inputs/admission.json':'945dec5bd002c8ecd860b9cc701bc2dbecfe181a86bbe295a89671ab707c13a6'}
SOURCES={'build_packet.py':'85691a9c7d52e40b334d9e42ea2da67762a955953c115de8a7d58be385935c5a',
 'launch_once.py':'7bcd8c3db86446d92e954011a2351ebd773018b92fd7879607f8c2e9a9c9fe93',
 'monitor_once.py':'e9d5273f3eb000de9d73f2dc7bb3ee55e1fdc4585fce3e34123afd9bd9248903',
 'compact_inputs.py':'a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245',
 'cpu_history_contract.py':'3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'}
BINDINGS={'gpu-recovery-authority.json':'b3630d400649ecd186e35232b24e9020e5bced7ad616379c7eaf5eef84498603',
 'operational-source-manifest.json':'7eceb8f0fb61696d5c42024a981f5e97dde683e58f2e24427b30a9e7703470f8',
 'gpu-failure-acceptance.json':'cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0',
 'gpu-failure-baseline.json':'cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef',
 'gpu-failure-seal.json':'8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0',
 'gpu-failure-disposition.json':'545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507',
 'gpu-failure-packet.json':'436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7',
 'preflight-review.md':'bf069359dc24b1785e700d69131984e46ad8c66a09adb6f997e74abf0d3c1caa',
 'gpu-recovery-review.md':'bf069359dc24b1785e700d69131984e46ad8c66a09adb6f997e74abf0d3c1caa',
 'operational-source-review.md':'77ab1d1d58572703711ebe37040613ab8f45448a0db14238b11cdd9ae570335a',
 'cpu-history-readset.json':'9be8223034cadedb0a3ff885236f923519b81f8c98bd46a7ffbcd2dafd9812b0',
 'cpu-history-preflight.json':'ab9d0ac13e2cbb17cc7c32c9ac0ba07a6e6a988e890402c02757807a519f1421'}
checks=[];facts={};readpins={}
def need(ok,what):
 if not ok:raise ValueError(what)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def decode(raw):
 def unique(items):
  d={}
  for k,v in items:need(k not in d,'duplicate JSON');d[k]=v
  return d
 v=json.loads(raw,object_pairs_hook=unique);json.dumps(v,allow_nan=False);return v
def read(p):
 b=Path(p).read_bytes();readpins[str(p)]=sha(b);return b
def j(p):return decode(read(p))
def fp(p):
 p=Path(p);need(p==p.resolve(strict=True),'unresolved/aliased packet member '+str(p))
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'independent regular member '+str(p));h=hashlib.sha256()
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
  t=os.fstat(f.fileno())
 sig=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink)
 need(sig(s)==sig(t)==sig(p.lstat()),'member changed during read '+str(p))
 return {'bytes':s.st_size,'sha256':h.hexdigest()},(s.st_dev,s.st_ino)
def tree(root):
 need(root==root.resolve(strict=True) and root.is_dir(),'ordinary packet root')
 files={};ids={};dirs=[]
 for p in sorted(root.rglob('*')):
  need(not p.is_symlink(),'packet symlink');n=p.relative_to(root).as_posix()
  if p.is_dir():dirs.append(n)
  else:files[n],ids[n]=fp(p)
 return {'files':files,'directories':dirs},ids
def git(commit,path):
 need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'full Git commit required')
 out=subprocess.run(['git','--no-replace-objects','show',commit+':'+path],cwd=REPO,capture_output=True,check=True)
 return out.stdout
def passed(name):checks.append(name)

def review():
 first,ids=tree(P);other,eids=tree(E)
 need(first==other and all(ids[n]!=eids[n] for n in ids),'full independent packet copies differ')
 outer=j(P/'packet.json');inner=j(P/'inputs/packet.json');admit=j(P/'inputs/admission.json')
 want={**outer['new_input_files'],'packet.json':ANCHORS['packet.json']}
 need(set(first['files'])==set(want) and first['directories']==['inputs'],'exact complete packet membership differs')
 for n,h in want.items():need(first['files'][n]['sha256']==h,'captured packet member differs '+n)
 for n,h in {**ANCHORS,**SOURCES,**BINDINGS}.items():need(first['files'][n]['sha256']==h,'independent anchor differs '+n)
 need(outer['gpu_packet_sha256']==ANCHORS['inputs/packet.json']==admit['packet_sha256'],'inner/outer/admission pin differs')
 passed('all original concrete packet members and two independent copies')
 facts['packet_inventory']=first;facts['packet_files']=len(want);facts['packet_bytes']=sum(v['bytes'] for v in first['files'].values())
 original=j(OLD/'inputs/packet.json');cpu=j(P/'cpu-packet.json');oldouter=j(OLD/'packet.json')
 need(sha(read(OLD/'inputs/packet.json'))=='85d2607d94114109819ca5e8150c5849c13f1bbf4b37765d3f7ee2ff64e77c58','original GPU inner pin')
 need(sha(read(OLD/'packet.json'))==BINDINGS['gpu-failure-packet.json'],'original GPU outer pin')
 need(sha(read(P/'cpu-packet.json'))==outer['cpu_packet_sha256']==inner['cpu_packet_sha256']=='ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344','original CPU packet')
 need(len(cpu['files'])==95,'95 original CPU files required')
 for n,h in cpu['files'].items():
  need(inner['files'][n]==h==first['files']['inputs/'+n]['sha256']==fp(OLD/'inputs'/n)[0]['sha256'],'original CPU input altered '+n)
 need(inner['files'].keys()=={n[7:] for n in first['files'] if n.startswith('inputs/') and n not in ('inputs/packet.json','inputs/admission.json')},'child source/data membership')
 newfiles={'llm_intervention_cpu_preflight_receipt.py','llm_intervention_cpu_history_admission.py',*inner['recovery_files'].values()}
 need(set(inner['files'])-set(original['files'])==newfiles and not set(original['files'])-set(inner['files']),'only six additions required')
 changed=[n for n,h in original['files'].items() if inner['files'][n]!=h]
 entry='qualify_llm_intervention_scientific_campaign_gpu.py';need(changed==[entry],'only operational entry replaced')
 for key in ('contracts','services','lineage_files','scope','limits','resources','cpu_acceptance_sha256','cpu_packet_sha256','freeze_sha256','namespace_admission_sha256'):
  need(inner[key]==original[key],'frozen contract changed '+key)
 need(outer['original_cpu_inputs']==oldouter['original_cpu_inputs'] and outer['history_directories']==oldouter['history_directories']
      and outer['token_recovery']==oldouter['token_recovery'],'original complete history changed')
 passed('all95 CPU inputs opaque-hash preserved; only entry and six permitted additions; original services/history unchanged')
 facts.update(cpu_original_files=95,original_gpu_files=len(original['files']),new_gpu_files=len(inner['files']),only_changed_original_input=entry)
 authority=j(P/'gpu-recovery-authority.json');context=j(P/'cpu-history-expected.json');receipt=j(P/'cpu-history-preflight.json');manifest=j(P/'operational-source-manifest.json')
 for role,n in inner['recovery_files'].items():need(read(P/n)==read(P/'inputs'/n),'duplicate recovery input differs '+role)
 need(first['files']['cpu-history-expected.json']['sha256'].startswith('e3c8b4be'),'reviewed actual expected pin prefix')
 need(digest(context['execution_context'])==context['execution_context_sha256']==receipt['execution_context_sha256']==outer['gpu_recovery']['execution_context_sha256'],'semantic execution context joins')
 need(inner['paths']==outer['paths'] and inner['paths']['gpu_inputs']==NEWROOT+'/inputs' and inner['paths']['gpu_output']==OLDROOT+'/gpu'
      and {k:v for k,v in inner['paths'].items() if k!='gpu_inputs'}=={k:v for k,v in original['paths'].items() if k!='gpu_inputs'},'newcontrol/frozenoutput paths')
 need(authority['control_root']==outer['remote_root']==NEWROOT and authority['allowed_new_output']==admit['output']==OLDROOT+'/gpu','control/output authority')
 need(authority['baseline']['sha256']==outer['baseline_sha256']==admit['accounting_sha256']==BINDINGS['gpu-failure-baseline.json'],'47/96 accounting admission join')
 sourcepins={n:inner['files'][n] for n in inner['source_names']}
 need(len(sourcepins)==56 and sourcepins==authority['operational_source_files'] and digest(sourcepins)==context['execution_context']['source_files_sha256'],'complete56-source operational map')
 need(manifest['source_commit']==outer['operational_source_commit']=='6296e5ceb538ae3957728a2ad9567d8d0ae611ce','operational source commit')
 for n,info in manifest['files'].items():need(read(P/'inputs'/n)==git(manifest['source_commit'],info['repository_path']) and sourcepins[n]==info['sha256'],'operational captured Git bytes '+n)
 need(outer['source_commit']==original['files'].get('unused',outer['source_commit'])=='a9346022f145cc29439325b6f291597b2e6ec037','scientific source revision')
 passed('authority/context/actual preflight/reviews and committed operational source identity')
 facts['actual_expected_sha256']=first['files']['cpu-history-expected.json']['sha256'];facts['operational_source_commit']=manifest['source_commit']
 expectedres=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
 need(outer['resources']==inner['resources']==admit['resources']==expectedres,'exact frozen resources')
 need(outer['limits']==inner['limits']==admit['limits']==dict(final_seconds=2860,session_cap_seconds=1200,session_reserve_seconds=600,work_seconds=2820),'exact frozen limits')
 base=j(P/'gpu-failure-baseline.json');fail=j(P/'gpu-failure-acceptance.json')
 need((base['terminal_jobs'],base['submission_journals'],len(base['related_accounting']),len(base['journal_pins']),base['outstanding_reservation'])==(47,96,47,96,0),'actual closed47/96 baseline')
 need((base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'],base['cpu_admission_seconds'])==(30241,2750,33841),'actual accepted charges')
 need(outer['admission']==dict(closed_cpu_seconds=30241,closed_gpu_slice_seconds=2750,preparation_allowance_cpu_seconds=3600,candidate_cpu_reservation=11520,candidate_gpu_reservation=2880,total_cpu_seconds=45361,total_gpu_seconds=5630,remaining_finalizer_cpu_cap=2400,total_with_finalizer_cpu_seconds=47761,ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200),'candidate/residual budget arithmetic')
 need(fail['original_collector_returncode']==1 and not fail['retroactive_collector_success'] and outer['gpu_recovery']['original_collector_status']=='terminal_custody_with_binding_disagreements','original failed attempt retained')
 need(outer['prior_jobs']==47 and outer['prior_journals']==96 and outer['worlds']==24 and outer['cells']==192 and len(outer['policy_ids'])==8 and outer['gpu_policy_ids']==['frozen_llm','sft_llm','dpo_llm'],'fixed scientific campaign scope')
 need(outer['submitted'] is outer['execution_qualified'] is outer['scientific_qualified'] is False and outer['new_training_updates']==outer['new_worlds']==0,'no execution claim in packet')
 sbatch=read(P/'job.sbatch').decode();flags=[l for l in sbatch.splitlines() if l.startswith('#SBATCH')]
 for flag in ('--job-name=ace_llm_scientific_campaign_gpu02','--account=ucb736_asc2','--partition=aa100','--qos=gpu-testing','--cpus-per-task=4','--mem=15360M','--gres=gpu:a100_3g.20gb:1','--time=00:48:00','--no-requeue'):
  need(flags.count('#SBATCH '+flag)==1,'SBATCH differs '+flag)
 command=shlex.split(next(l[5:] for l in sbatch.splitlines() if l.startswith('exec ')))
 need(command[:4]==[cpu['runtime_source']+'/bin/python','-I','-B','-c'],'isolated staged-runtime bootstrap')
 need(command[5:]==[NEWROOT+'/inputs/'+entry,inner['files'][entry],'--packet',NEWROOT+'/inputs/packet.json','--packet-sha256',ANCHORS['inputs/packet.json'],'--admission',NEWROOT+'/inputs/admission.json','--admission-sha256',ANCHORS['inputs/admission.json'],'--output',OLDROOT+'/gpu'],'exact launched entry/packet/admission/output argv')
 passed('resource/clock/scientific scope and exact sbatch bootstrap argument bindings')
 facts['admission']=outer['admission'];facts['sbatch_directives']=flags
 readiness=read(P/'site-readiness.md').decode();saved=json.loads(readiness.split('```json\n',1)[1].split('\n```',1)[0])
 facts['saved_site_headroom']={k:v for k,v in saved.items() if k in ('utc','scratch_probe','scratch_write_read_cleanup','/projects/paco0228','/scratch/alpine/paco0228/ACE/results')}
 facts['saved_site_commands']={k:{x:v.get(x) for x in ('returncode','stdout') if x in v} for k,v in saved.get('commands',{}).items() if k in ('queue','partition','association','qos')}
 # Reuse closed collector source review; independently verify source/review in explicit actual Git commit.
 commit='bb3a3d29d715cf718503013838cd5fc450e09053'
 collector_path='results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/coordinator02/authoring/pull_terminal.py'
 review_path='results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/coordinator-review02/review.md'
 collector=read(REPO/collector_path);review=read(REPO/review_path)
 need(sha(collector)=='3697547ddace45fda812c069c08d2d299e75b61fe9db717807a5379645771c27' and sha(review).startswith('2d75'),'reviewed collector pins')
 need(git(commit,collector_path)==collector and git(commit,review_path)==review,'collector/review committed bytes')
 facts['collector_provenance']={'source_commit':commit,'repository_path':collector_path,'collector_sha256':sha(collector),'review_path':review_path,'review_sha256':sha(review),'not_in_child_roster':True}
 passed('standalone collector and closed R1 review independently bound to full bb3a source commit')
 return first,outer

start=time.monotonic()
try:
 initial,outer=review()
 (R/'packet-inventory.json').write_text(json.dumps(initial,sort_keys=True,indent=2)+'\n')
 outcome={'status':'packet_checks_passed','checks':checks,'facts':facts,'read_pins':readpins,'wall_seconds':time.monotonic()-start,
          'packet_code_executed':False,'models_or_private_outcomes_decoded':False,'remote_actions':0,'tests_run':0}
except BaseException as e:
 outcome={'status':'checker_failure_preserved','checks':checks,'facts':facts,'read_pins':readpins,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'wall_seconds':time.monotonic()-start}
with (R/'check01.json').open('x') as f:json.dump(outcome,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in outcome.items() if k not in ('facts','read_pins')},indent=2))
