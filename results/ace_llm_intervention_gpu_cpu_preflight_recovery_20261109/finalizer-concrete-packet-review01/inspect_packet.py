"""Independent saved-metadata packet inspection; no producer imports or execution."""
import ast, hashlib, json, os, resource, shlex, stat, time, traceback
from pathlib import Path

REPO=Path('/Users/pat/code/ACE')
PREP=REPO/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
HERE=Path(__file__).resolve().parent
LOCAL=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-finalization-20261108-01')
OUTER='1a30c9bdfef0b81f98089c774efa5ff781ddc2b5bc8411c9d08e3b198adb87d3'
INNER='c71bafdf9f5a791c540c6a5a1a12ba0f2b84b9b221c2b018cbdebf9684728571'
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'
ENTRY_SHA='8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
SOURCES={'build_packet.py':'c2890210a996d75070aea7df64696562218c9a22f1fe2fecee0e12847518a418',
 'compact_inputs.py':'d810468058111c02466c3579ee4311ad0b14e5d78297f43f8773919c40bbea0e',
 'launch_once.py':'aaf4f431576b657b8adbe09d9c1f02fea5cb7d8f7ca6608009ef49052822bdb4',
 'monitor_once.py':'8acbf213cba75c22c5ea83fdb997c198f80feaf071b6aff91b0eff6646e681c1'}
PINS={'gpu-acceptance.json':'510ceadd2bdf831fd43c27567dfa799df91efb8f42aa11141f14d294870534c6',
 'gpu-recovery-acceptance.json':'1c0329023225c0f97e63ef0291743695ea930af85ec8f790e9dae902155afe2a',
 'gpu-baseline.json':'40e9ec4c91bf174214e8788306b3900ba5f92ec32d3a01a024bc2b0558f50897',
 'gpu-review.md':'c618a210593852fa0c9247c120f5d9a02524cd7a30047ec0c0840b3cbab5ebdf',
 'finalizer-clock-adoption.json':'4ad691f031caa48b47c5fbfc337a777738170a73dc02c391d36f6d94d77fb10f',
 'finalizer-timing-evidence.md':'be3fe18e35c5a6f5af7d2b59e3099e79660e522b42fbe0ae443f7fd64da401fd',
 'finalizer-timing-review.md':'af0aff21407e4b1e9d983d8a26810f869ee15bc1395d822ffcf24ecea8bf8530',
 'finalizer-source-review.md':'39ef3af657fd64e8174d6c0b462fdb1519c1978ee3ea9d5288f11c998d4a5bc9',
 'freeze.json':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'}
report={'scope':'concrete packet only; closed components reused','checks':[], 'facts':{},'errors':[],
 'acceptance':False,'scientific_qualified':False,'remote_actions':0,'tests_rerun':0,'producer_execution':False}
cache={}
def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def check(ok,label):
 if not ok:raise ValueError(label)
 report['checks'].append(label)
def info(path):
 path=Path(path)
 if str(path) in cache:return cache[str(path)]
 s=path.lstat();check(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'ordinary single-link file: '+str(path))
 h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   b=f.read(1024**2)
   if not b:break
   h.update(b)
 after=path.lstat();check((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'stable read: '+str(path))
 result={'bytes':s.st_size,'sha256':h.hexdigest()};cache[str(path)]=result;return result
def j(path):return json.loads(Path(path).read_bytes())
def manifest(root):
 check(root==root.resolve() and root.is_dir(),'canonical root '+str(root))
 result={}
 for p in sorted(root.rglob('*')):
  check(not p.is_symlink(),'no link '+str(p))
  if p.is_file():result[str(p.relative_to(root))]=info(p)
 return result
def main():
 outer=j(LOCAL/'packet.json');inner=j(LOCAL/'inputs/packet.json')
 check(info(LOCAL/'packet.json')['sha256']==OUTER and info(LOCAL/'inputs/packet.json')['sha256']==INNER,'supplied outer/inner pins')
 expected={**outer['new_input_files'],'packet.json':OUTER};copies=[manifest(LOCAL),manifest(EXT)]
 check(copies[0]==copies[1] and set(copies[0])==set(expected),'two complete exact packet copies')
 check(all(v['sha256']==expected[n] for n,v in copies[0].items()),'all outer member pins')
 check(all(not os.path.samestat((LOCAL/n).stat(),(EXT/n).stat()) for n in expected),'independent packet inodes')
 check(set(n[7:] for n in expected if n.startswith('inputs/'))==set(inner['files'])|{'packet.json'},'exact inner membership')
 check(all(expected['inputs/'+n]==h for n,h in inner['files'].items()),'all inner pins')
 for n,h in {**SOURCES,**PINS}.items():check(expected[n]==h,'closed component/pin '+n)
 check(inner['files'][ENTRY]==ENTRY_SHA,'operational entry8a26')
 collector=PREP/'finalizer-transfer-preparation01/implementation/pull_terminal.py'
 check(info(collector)['sha256']=='c58738e9fc85ff640b96eb6686f92893ab360056013dedcd4c1317baf0462112','standalone closed collector')
 command=j(PREP/'finalizer-build-command03/command.json');argv=command['argv'];args=dict(zip(argv[4::2],argv[5::2]))
 check(len(argv[4:])%2==0 and len(args)==len(argv[4:])//2,'literal unique command arguments')
 for flag,h in args.items():
  if flag.endswith('-sha256'):
   role=flag[2:-7];path=Path(args[flag[:-7]])
   check(info(path)['sha256']==h,'independent command input '+role)
   names=[role+'.json',role+'.md'];matches=[n for n in names if n in expected]
   check(len(matches)==1 and expected[matches[0]]==h,'captured command role '+role)
 check(info(PREP/'finalizer-build-command03/command.json')['sha256']=='d8637a67030f406b9ef8404a7221320b38f3ea19c57fd94519d95b0d6d6a264f','command pin')
 run=j(PREP/'finalizer-build-command03/run.json');handoff=j(PREP/'finalizer-build-command03/build-handoff.json')
 check(run['returncode']==0 and run['submission_invoked'] is False and run['automatic_retry'] is False,'one completed nonsubmission build')
 for n in ['stdout.log','stderr.log']:check(info(PREP/'finalizer-build-command03'/n)['sha256']==run[n+'_sha256'],'build transcript '+n)
 resources=dict(account='ucb736_asc2',cpus=2,gpus=0,memory_mib=7680,partition='acpu',qos='cpu-normal',requeue=False,wall_seconds=4200)
 deadlines=dict(cleanup=3350,hard=4190,preparation=1500,report_startup=120,report_startup_cap=1620,work=3320)
 adoption=j(LOCAL/'finalizer-clock-adoption.json');base=j(LOCAL/'gpu-baseline.json');gpu=j(LOCAL/'gpu-acceptance.json');recovery=j(LOCAL/'gpu-recovery-acceptance.json')
 check(outer['resources']==inner['resources']==adoption['resources']==resources and outer['deadlines']==adoption['deadlines']==deadlines,'adopted profile propagation')
 for key,name in [('gpu_acceptance_sha256','gpu-acceptance.json'),('gpu_recovery_acceptance_sha256','gpu-recovery-acceptance.json'),('baseline_sha256','gpu-baseline.json')]:check(outer[key]==adoption[key]==expected[name],'adopted actual lineage '+key)
 check(inner['clock_adoption_sha256']==outer['clock_adoption_sha256']==expected['finalizer-clock-adoption.json'],'inner clock adoption')
 for key,name in [('timing_evidence_sha256','finalizer-timing-evidence.md'),('timing_review_sha256','finalizer-timing-review.md'),('source_manifest_sha256','finalizer-source-manifest.json'),('source_review_sha256','finalizer-source-review.md')]:check(adoption[key]==expected[name],'adoption evidence '+key)
 check(adoption['status']=='adopted' and adoption['full_path_timing_supported'] is True and adoption['required_findings']==0 and adoption['entry_sha256']==ENTRY_SHA,'affirmative scoped adoption')
 check(outer['source_commit']==inner['source_commit']=='a9346022f145cc29439325b6f291597b2e6ec037' and outer['operational_source_commit']==inner['operational_source_commit']==adoption['source_commit']=='255b75fe37b28c302fb57821c0cc09a60a6cd3ea','frozen versus operational source identity')
 check(gpu['status']=='development_gpu_phase_accepted' and len(gpu)==17 and gpu['job_id']=='35516125' and recovery['acceptance'] is True and recovery['required_findings']==0 and recovery['historical_postcustody_verified'] is True,'actual core17 and affirmative recovery')
 check(base['terminal_jobs']==outer['prior_jobs']==48 and base['submission_journals']==outer['prior_journals']==98 and len(base['related_accounting'])==48 and len(base['journal_pins'])==98 and base['outstanding_reservation']==0,'accepted48/98 released baseline')
 check((base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'])==(37073,4458) and outer['admission']==dict(candidate_cpu_reservation=8400,candidate_gpu_reservation=0,ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,closed_cpu_seconds=37073,closed_gpu_slice_seconds=4458,preparation_allowance_cpu_seconds=3600,total_cpu_seconds=49073,total_gpu_seconds=4458),'full accounting and single allowance')
 check((outer['worlds'],outer['cells'],outer['comparisons'])==(24,192,28) and len(outer['policy_ids'])==8 and outer['new_worlds']==outer['new_responses']==outer['new_training_updates']==0,'fixed reporting scope')
 check(outer['scientific_qualified'] is outer['execution_qualified'] is outer['submitted'] is False,'no future execution qualification')
 gp=j(LOCAL/'gpu-packet.json');freeze=j(LOCAL/'freeze.json');closure=freeze['stage_closure']['stages']['finalizer']
 extras={ENTRY,'qualify_llm_intervention_campaign_gpu.py','gpu-packet.json','gpu-acceptance.json','gpu-recovery-acceptance.json','gpu-success-baseline.json','gpu-failure-acceptance.json','gpu-failure-baseline.json','finalizer-clock-adoption.json'}
 check(set(inner['files'])==set(gp['files'])|extras and all(inner['files'][n]==h for n,h in gp['files'].items()),'all108 original GPU inputs unchanged plus exact9')
 check(inner['source_names']==sorted(set(gp['source_names'])|{ENTRY,'qualify_llm_intervention_campaign_gpu.py'}),'exact inherited source roster')
 check(all(inner['files'][n]==h for n,h in closure['helper_pins'].items()),'all frozen finalizer helper pins')
 for n,h in closure['input_source_files'].items():
  if n not in {ENTRY,'qualify_llm_intervention_scientific_campaign_gpu.py'}:check(inner['files'][n]==h,'frozen source '+n)
 for n in extras- {ENTRY,'qualify_llm_intervention_campaign_gpu.py'}:
  rootname='gpu-baseline.json' if n=='gpu-success-baseline.json' else n
  check(inner['files'][n]==expected[rootname],'captured child metadata '+n)
 config=j(LOCAL/'storage-config.json');binding=outer['storage_binding'];receipt=j(LOCAL/'storage-receipt.json');after=json.loads(j(LOCAL/'storage-after.json')['stdout'])
 names=dict(authority='storage-authority.json',authority_review='storage-review.md',backend='remote_custody.py',backend_review='storage-backend-review.md',receipt='storage-receipt.json',seal='storage-seal.json',after='storage-after.json',reference='storage-predecessor.json')
 check(set(config['members'])==set(names) and config['predecessor']=='gpu','exact storage8 roles')
 for k,n in names.items():check(config['members'][k]['sha256']==expected[n] and info(Path(config['members'][k]['path']))['sha256']==expected[n],'storage member origin '+k)
 check(binding['backend_sha256']==expected['remote_custody.py']=='4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f' and binding['compact_adapter_sha256']==SOURCES['compact_inputs.py'] and binding['complete_local_raw_copy'] is False,'closed compact backend/adapter')
 check(receipt['full_inventory']['files']==after['files'] and receipt['full_inventory']['directories']==sorted(after['directories']) and receipt['full_backup_verified'] is True and receipt['acceptance'] is False,'sealed full remote map association')
 upstream={}
 for label,physical in receipt['roots'].items():
  if label.endswith('.json'):continue
  prefix=label+'/';files={n[len(prefix):]:v['sha256'] for n,v in after['files'].items() if n.startswith(prefix)};dirs=sorted(n[len(prefix):] for n in after['directories'] if n.startswith(prefix))
  check(outer['upstream_rosters'][physical]==dict(files=files,directories=dirs),'full upstream roster '+label)
  upstream.update({physical+'/'+n:h for n,h in files.items()})
 check(outer['original_input_pins']==upstream,'complete historical input pins')
 check(after['files']['namespace.json']['sha256']==outer['namespace_claim_sha256']==expected['namespace-claim.json'],'preserved recovery namespace')
 rep=binding['input_replica_binding'];check(rep['after_sha256']==expected['gpu-inventory-after.json'] and rep['outer_sha256']==expected['gpu-launch-packet.json'] and rep['inner_sha256']==expected['gpu-packet.json'],'independent replica pins')
 members={n[len('control/inputs/'):]:v for n,v in after['files'].items() if n.startswith('control/inputs/')}
 check(rep['input_members']==members and rep['input_members_sha256']==sha(canonical(members)) and rep['input_files']==110 and rep['input_bytes']==38849832,'all110 original flat members inclpacket/admission')
 check(rep['roots']==[args['--gpu-packet-replica'],args['--gpu-packet-replica-mirror']] and rep['fallback_allowed'] is rep['acceptance'] is False,'explicit replica roots no fallback')
 for root in map(Path,rep['roots']):
  check(info(root/'packet.json')['sha256']==rep['outer_sha256'],'replica original outer '+str(root))
  check(set(x.name for x in (root/'inputs').iterdir())==set(members),'replica exact input membership '+str(root))
  for n,v in members.items():check(info(root/'inputs'/n)==v,'replica member '+str(root/n))
 groups={}
 for label in ('root','inputs'):
  selected={n:v for n,v in copies[0].items() if n.startswith('inputs/')==(label=='inputs')};size=sum(v['bytes'] for v in selected.values());limit=120+(size+1024**2-1)//1024**2
  check(limit<=900,'bounded actual transfer group '+label);groups[label]=dict(files=len(selected),bytes=size,timeout_seconds=limit)
 check(groups=={'root':{'files':105,'bytes':738753006,'timeout_seconds':825},'inputs':{'files':118,'bytes':44009413,'timeout_seconds':162}},'actual transfer sizes and deadlines')
 job=(LOCAL/'job.sbatch').read_text();prefix,commandtext=job.split('exec ',1);cmd=shlex.split(commandtext)
 bootstrap=next(ast.literal_eval(n.value) for n in ast.parse((LOCAL/'inputs/qualify_llm_intervention_runtime_metadata.py').read_bytes()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
 check(cmd==[j(LOCAL/'cpu-packet.json')['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,ENTRY_SHA,'--packet',ROOT+'/inputs/packet.json','--packet-sha256',INNER,'--output',ROOT+'/finalization'],'exact captured-byte bootstrap argv')
 directives=[line for line in prefix.splitlines() if line.startswith('#SBATCH')]
 check(directives==['#SBATCH --job-name=ace_llm_scientific_finalization01','#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M','#SBATCH --time=01:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err'],'exact CPU-only scheduler directives')
 refs=[j(LOCAL/'storage-preparation-reference.json'),handoff['storage_final_reference']]
 operation_records=[]
 for ref in refs:
  ip=Path(ref['inventory_path']);check(info(ip)['sha256']==ref['inventory_sha256'],'pinned durable storage reference '+ip.name);record=j(ip)
  check(record['status']=='completed' and record['error'] is None and record['acceptance'] is False and record['intent_sha256']==ref['intent_sha256'],'completed storage reference '+ip.name)
  op=Path(record['operations_path'])
  for n,v in record['inventory']['files'].items():check(info(op/n)==v,'operation original '+n)
  operation_records.append({'path':str(ip),**info(ip),'operation':record['operation'],'status':record['status']})
 final=j(Path(refs[-1]['inventory_path']));op=Path(final['operations_path']);verification=[]
 for q in sorted(op.glob('verify-*/receipt.json')):
  v=j(q);check(v['full_backup_verified'] is True and v['acceptance'] is False and v['helper_sha256']==binding['backend_sha256'] and v['receipt_sha256']==expected['storage-receipt.json'],'saved complete archive verification '+q.parent.name)
  verification.append({'path':str(q),**info(q)})
 check(len(verification)==3,'three existing saved archive verifications; no new verification run')
 report['facts'].update(packet_files=len(expected),packet_bytes=sum(v['bytes'] for v in copies[0].values()),inner_inputs=len(inner['files']),source_names=len(inner['source_names']),groups=groups,admission=outer['admission'],storage_references=operation_records,archive_verifications=verification,build_resources=run,collector_pin=info(collector),all_object_pins=cache)

started=time.monotonic();cpu=time.process_time()
try:main()
except BaseException as exc:
 report['errors'].append({'type':type(exc).__name__,'message':str(exc)});(HERE/'failure.traceback.txt').write_text(traceback.format_exc())
finally:
 report.update(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__).read_bytes()))
 with (HERE/'checks.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'checks':len(report['checks']),'errors':report['errors'],'wall_seconds':report['wall_seconds']}))
raise SystemExit(bool(report['errors']))
