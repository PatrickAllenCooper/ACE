from pathlib import Path
import ast,hashlib,json,os,stat,tempfile,shlex,shutil,datetime
REPO=Path('/Users/pat/code/ACE');BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
L=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/launch02';E=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-02';OLD=L.parent/'launch01'
R=Path(tempfile.mkdtemp(prefix='ace-token02-concrete-review-',dir='/tmp')).resolve();Path('/tmp/ace-token02-concrete-review-current.txt').write_text(str(R)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
checks=[]
def check(v,msg):
 if not v:raise ValueError(msg)
 checks.append(msg)
def inv(root):
 files={};dirs=[]
 for p in sorted(root.rglob('*')):
  s=p.lstat();check(not stat.S_ISLNK(s.st_mode),'no symlink '+p.relative_to(root).as_posix())
  if p.is_dir():dirs.append(p.relative_to(root).as_posix());continue
  check(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'independent regular '+p.relative_to(root).as_posix())
  files[p.relative_to(root).as_posix()]=dict(bytes=s.st_size,sha256=sha(p.read_bytes()))
 return dict(files=files,directories=dirs)
try:
 outerpin='69aaaccdf9f20b11665f4708638db9ba621df4c9881befc1d4bb4225a46f912f';innerpin='cab514d59959dd64c080d1414bd1dea6a47da59a93c55d3d23fe3b2fc338ee1c'
 check(sha((L/'packet.json').read_bytes())==outerpin,'independent outer pin')
 outer=j(L/'packet.json');inner=j(L/'inputs/packet.json');oldinner=j(OLD/'inputs/packet.json');freeze=j(L/'freeze.json')
 check(sha((L/'inputs/packet.json').read_bytes())==innerpin,'independent inner pin')
 left,right=inv(L),inv(E);check(left==right,'complete two-copy bytes and directories')
 members={**outer['new_input_files'],'packet.json':outerpin}
 check(set(left['files'])==set(members) and set(left['directories'])=={'inputs','tokenizer'},'exact complete packet membership; no controller/intent/result')
 for n,h in members.items():
  check(left['files'][n]['sha256']==h,'outer member pin '+n)
  a,b=(L/n).stat(),(E/n).stat();check((a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino),'distinct copy inode '+n)
 root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02';oldroot=root[:-2]+'01'
 check(outer['remote_root']==root and outer['local_root']==str(L) and outer['external_root']==str(E),'exact successor roots')
 check(outer['source_commit']==inner['source_commit']==freeze['source_commit']=='a9346022f145cc29439325b6f291597b2e6ec037','unchanged full source commit')
 check(sha((L/'freeze.json').read_bytes())=='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d','unchanged science freeze')
 diff=[]
 def walk(a,b,path=''):
  if type(a) is not type(b):diff.append(path);return
  if isinstance(a,dict):
   if set(a)!=set(b):diff.append(path+'.keys');return
   for k in a:walk(a[k],b[k],path+'.'+k if path else k)
  elif a!=b:diff.append(path)
 walk(oldinner,inner)
 check(sorted(diff)==['deadlines.startup','model_directory'],'exact two-field child delta')
 check(inner['model_directory']==root+'/tokenizer' and oldinner['model_directory']==oldroot+'/tokenizer','tokenizer-only successor path')
 resources=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,wall_seconds=600,gpus=0,requeue=False)
 deadlines=dict(startup=300,work=550,cleanup=570,hard=590)
 check(outer['resources']==inner['resources']==resources and outer['deadlines']==inner['deadlines']==deadlines,'exact CPU resources/deadlines')
 check(inner['files']==oldinner['files'] and inner['collection']==oldinner['collection'] and inner['protocol_source_pins']==oldinner['protocol_source_pins'],'complete original source/collection/protocol immutable')
 closure=freeze['stage_closure']['stages']['token']
 check(len(outer['source_pins'])==32 and len(inner['files'])==34 and len(inner['source_names'])==28 and inner['source_names']==closure['source_names'],'exact32 captured Python files28 loaded names34 inputs')
 for n,h in inner['files'].items():check(sha((L/'inputs'/n).read_bytes())==h and (L/'inputs'/n).read_bytes()==(OLD/'inputs'/n).read_bytes(),'unchanged inner bytes '+n)
 for name,h in outer['source_pins'].items():
  check(freeze['source_pins'][name]==h,'frozen source pin '+name)
  check(left['files']['inputs/'+Path(name).name]['sha256']==h,'captured source '+name)
 check({n for n in left['files'] if n.startswith('tokenizer/')}=={'tokenizer/'+n for n in inner['tokenizer_files']},'six tokenizer-only files exact')
 for n,h in inner['tokenizer_files'].items():check(left['files']['tokenizer/'+n]['sha256']==h,'tokenizer byte pin '+n)
 reviewed={'build_packet.py':'c1bf0ae82b3fe004d76ffa151bf8181a0921bfc2b8f9120bcb16f44b72cb4fbc','launch_once.py':'6a002d2bb584dafe1dd169d5b64dc6fd06a1f1f597f08a95855dff4ecea64a60','monitor_once.py':'6c5b7aac6114321708a47b056ecfe869df6d2105652c03348c342c684e560649','stream_custody.py':'fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a','stream_custody_review.md':'5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371'}
 for n,h in reviewed.items():check(members[n]==h,'closed reviewed outer source '+n)
 reviews=REPO/'docs/development/guidance/reviews'
 reviewpins={n:sha((reviews/n).read_bytes()) for n in ['ace_llm_intervention_scientific_token02_core_review_2026-11-08.md','ace_llm_intervention_scientific_token02_collector_review_2026-11-08.md']}
 check(reviewpins['ace_llm_intervention_scientific_token02_core_review_2026-11-08.md']=='0aac35495b759f5291397991303f98c6c2f957ba82f67c6371f6bdfc0125b0e3','closed core review pin')
 check(reviewpins['ace_llm_intervention_scientific_token02_collector_review_2026-11-08.md']=='39bbebe31a4feca11945032310ef47d37ef31513d78d1b92ad2ec2e5621b9413','closed collector review pin')
 collector=REPO/'results/ace_llm_intervention_scientific_tokenization_recovery_preparation_20261108/collector02/authoring/pull_terminal.py'
 check(sha(collector.read_bytes())=='68d19c2037fa54968483e718d64a11d3eb1f8d7aec9ae09219b96ecd3613c7a0' and 'pull_terminal.py' not in members,'exact standalone collector outside packet roster')
 prep=collector.parents[2]
 prep_pins={n:sha((prep/n/'preparation.json').read_bytes()) for n in ['coordinator02','collector02']}
 check(prep_pins['coordinator02']=='167419770cf1ce9c87e713249659f5a80da4d1c896648dde82e11e605892b527' and prep_pins['collector02'].startswith('ae0845'),'preserved coordinator/collector preparation receipt pins')
 auth=j(L/'token-operational-amendment.json');a=j(L/'token-failure-acceptance.json');proof=j(L/'token-no-learnedwork-proof.json');base=j(L/'token-failure-baseline.json');prior=j(L/'baseline.json');result=j(L/'failed-token-result.json')
 expected={'token-operational-amendment.json':'a77f63dab7f495bee1c00f81027d9cc6c5743b2c3c9df9341901bbe54a019c84','token-operational-adoption-review.md':'56f139e529a4b4e4b474dd564a8d84c0f9fd39db4a26aaef284fafa7631640ea','token-failure-acceptance.json':'bde58f25694c63b9f77ca427a2940582bb06f46178bd5fd111c7f5f70a49d9dd','token-failure-baseline.json':'7a80a2300afde1ef71f8b303815b51fc2bb347515b78c40d972b4ec10138daf5','token-no-learnedwork-proof.json':'14c1872765315c9162b1182290c4db631fa99e7878b81fff72b8c5f25e1b606a'}
 for n,h in expected.items():check(members[n]==h,'independent actual authority pin '+n)
 aliases={'freeze_sha256':'freeze.json','failure_acceptance_sha256':'token-failure-acceptance.json','baseline_sha256':'token-failure-baseline.json','no_learnedwork_proof_sha256':'token-no-learnedwork-proof.json','failure_disposition_sha256':'token-failure-disposition.json','failure_inventory_sha256':'token-failure-inventory.json','failure_readback_sha256':'token-failure-readback.json','failure_custody_inventory_sha256':'token-failure-custody-inventory.json','collection_acceptance_sha256':'collection-acceptance.json','collection_baseline_sha256':'baseline.json','namespace_admission_sha256':'namespace-admission.json','namespace_claim_sha256':'namespace-claim.json','review_sha256':'token-operational-review.md','guide_sha256':'token-operational-guide.md','diagnosis_sha256':'token-diagnosis.json','failed_packet_sha256':'failed-token-packet.json','failed_stage_packet_sha256':'failed-token-stage.json'}
 check(all(auth[k]==members[n] for k,n in aliases.items()),'complete actual amendment member joins')
 check(auth['status']=='adopted' and auth['stage']=='tokenize' and auth['recovery_root']==root and auth['failed_root']==oldroot and auth['resources']==resources and auth['deadlines']==deadlines and auth['original_deadlines']==oldinner['deadlines'] and auth['scientific_inputs_unchanged'] is auth['other_stages_unchanged'] is auth['one_submission_only'] is True and auth['automatic_retry_allowed'] is auth['replacement_worlds_allowed'] is False,'only independently adopted operational recovery')
 check(a['status']=='accepted_failed_before_token_worker' and a['required_findings']==0 and a['complete_custody_verified'] is True and a['tokenization_accepted'] is False and a['owned_worker_spawned'] is False,'independent failure acceptance reused without raw audit')
 check(proof['failure_acceptance_sha256']==members['token-failure-acceptance.json'] and proof['accounting_sha256']==members['token-failure-baseline.json'] and proof['proof']['token_preparation_not_invoked'] is proof['proof']['owned_worker_not_spawned'] is True,'independent no-learnedwork proof joins')
 check(all(v[k]==0 and type(v[k]) is int for v in (a,proof) for k in ['environment_responses','unknown_return_attempts','tokenizer_calls','model_loads','inference_calls','training_updates','policy_updates']),'known zero failed work retained')
 check(prior['terminal_jobs']==42 and prior['submission_journals']==86 and base['terminal_jobs']==43 and base['submission_journals']==88,'exact42/86 to failed43/88')
 oldrows={r[0]:r for r in prior['related_accounting']};rows={r[0]:r for r in base['related_accounting']}
 check(len(rows)==43 and set(rows)-set(oldrows)=={'35415334'} and all(rows[k]==v for k,v in oldrows.items()),'every prior allocation row preserved')
 row=rows['35415334'];check(row[2:6]==['FAILED','1:0','3','129'],'exact failed actual387CPU row')
 check(base['journal_pins']=={**prior['journal_pins'],oldroot+'/submission.intent.json':result['intent_sha256'],oldroot+'/submission.result.json':members['failed-token-result.json']},'all86 prior plus two failed journals')
 cpu=sum(int(r[4])*int(r[5]) for r in rows.values());gpu=0
 for row in rows.values():
  tres=dict(x.split('=',1) for x in row[7].split(',') if x);generic=tres.get('gres/gpu');typed=[int(v) for k,v in tres.items() if k.startswith('gres/gpu:')]
  count=int(generic) if generic is not None else sum(typed)
  check(generic is None or not typed or int(generic)==sum(typed),'no double-counted GPU TRES '+row[0]);gpu+=count*int(row[5])
 check((cpu,gpu)==(13373,1601)==(base['closed_cpu_seconds'],base['closed_gpu_slice_seconds']),'actual complete historical CPU/GPU totals')
 check(base['outstanding_reservation']==0 and base['preparation_allowance_cpu_seconds']==3600 and outer['baseline_sha256']==members['token-failure-baseline.json'],'released reservation actual baseline binding')
 check(outer['admission']==dict(candidate_cpu_reservation=1800,candidate_gpu_reservation=0,ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,closed_cpu_seconds=13373,closed_gpu_slice_seconds=1601,preparation_allowance_cpu_seconds=3600,prior_jobs=43,prior_journals=88,total_cpu_seconds=18773,total_gpu_seconds=1601),'exact new candidate admission')
 remaining=freeze['stages'][2:];projection=(cpu+3600+sum(s['cpus']*s['wall_seconds'] for s in remaining),gpu+sum(s['gpus']*s['wall_seconds'] for s in remaining));check(projection==(55493,7181),'all remaining frozen caps within ceilings')
 first=j(L/'token-failure-inventory.json');last=j(L/'token-failure-readback.json');first=jv=json.loads(first['stdout']);last=json.loads(last['stdout']);disp=j(L/'token-failure-disposition.json');custody=j(L/'token-failure-custody-inventory.json')
 check(first['files']==last['files']==disp['remote_files'] and first['directories']==last['directories'],'accepted failure inventory metadata joins; raw audit not repeated')
 check(first['journal_pins']==base['journal_pins'] and first['parent_accounting'][0]=='35415334','accepted failure scheduler/journal metadata joins')
 for n,key in [('inventory_transport.json','token-failure-inventory.json'),('inventory_after_transport.json','token-failure-readback.json'),('custody-inventory.json','token-failure-custody-inventory.json')]:check(a['evidence_pins'][n]==members[key],'independent failure custody evidence '+n)
 readback=outer['failed_token_custody_readback'];check(readback==dict(files=len(first['files']),bytes=sum(v['bytes'] for v in first['files'].values()),files_sha256=digest({n:v['sha256'] for n,v in first['files'].items()})),'builder failed-custody readback matches accepted full map')
 check(custody['identical'] is True and len(custody['copies'])==2,'accepted two-copy failure inventory retained')
 collection=j(L/'collection-acceptance.json');check(inner['collection']==collection['collection'] and outer['collection_acceptance_sha256']==members['collection-acceptance.json'],'all original collection inputs and method identities retained')
 for suffix,h in [('.namespace.json',auth['original_namespace_claim_sha256']),('.namespace-recovery02.json',auth['namespace_claim_sha256'])]:check(sha((BASE/('ACE-LLM-intervention-scientific-screen-20261107'+suffix)).read_bytes())==h,'local unchanged registry '+suffix)
 batch=(L/'job.sbatch').read_text();directives=[s for s in batch.splitlines() if s.startswith('#SBATCH')]
 for s in ['--account=ucb736_asc2','--partition=acpu','--qos=cpu-normal','--cpus-per-task=3','--mem=11520M','--time=00:10:00','--nodes=1','--ntasks=1','--no-requeue']:check('#SBATCH '+s in directives,'Slurm '+s)
 check(not any('--gres' in s or '--gpu' in s for s in directives) and "export CUDA_VISIBLE_DEVICES=''" in batch,'no GPU request/visibility')
 command=shlex.split(batch.split('\nexec ',1)[1]);bootstrap=next(ast.literal_eval(n.value) for n in ast.parse((L/'inputs/qualify_llm_intervention_runtime_metadata.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
 entry='prepare_llm_intervention_scientific_training_cpu.py'
 check(command==[inner['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,root+'/inputs/'+entry,inner['files'][entry],'--packet',root+'/inputs/packet.json','--packet-sha256',innerpin,'--source-commit',inner['source_commit'],'--output',root+'/preparation'],'exact pinned bootstrap/entry/packet/root argv')
 check(inv(L)==left and inv(E)==right,'packet copies unchanged through review')
 (R/'packet-inventory.json').write_text(json.dumps(left,indent=2,sort_keys=True)+'\n')
 result=dict(status='review_complete',required_findings=0,packet_sha256=outerpin,inner_sha256=innerpin,packet_files=len(members),packet_bytes=sum(v['bytes'] for v in left['files'].values()),source_python_files=32,loaded_source_names=28,inner_inputs=34,child_changed_fields=diff,resources=resources,deadlines=deadlines,closed_cpu_seconds=cpu,closed_gpu_seconds=gpu,admission_cpu_seconds=18773,remaining_cap_projection=projection,reviewed_sources=reviewed,closed_review_pins=reviewpins,preparation_pins=prep_pins,standalone_collector=dict(path=str(collector),sha256=sha(collector.read_bytes()),eventual_packet_commit_required=True),authority_pins=expected,checks=checks,target_code_executed=False,tests_run=0,remote_actions=0,raw_audit_repeated=False,scientific_payloads_decoded=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (R/'metadata-check.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','reviewed_sources','closed_review_pins','authority_pins','standalone_collector']},indent=2))
except BaseException as exc:
 (R/'check-error.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc),checks_completed=len(checks)),indent=2)+'\n');raise
finally:shutil.copyfile(__file__,R/'review-check.py');print('EVIDENCE',R)
