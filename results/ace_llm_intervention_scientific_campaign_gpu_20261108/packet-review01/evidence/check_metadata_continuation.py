"""Independent read-only packet bytes/metadata joins; no project imports or execution."""
from pathlib import Path
import ast,collections,hashlib,json,os,stat,subprocess,traceback
W=Path(__file__).parent
R=Path('/Users/pat/code/ACE'); BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
ST=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'; P=ST/'launch01'; E=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261108-01'
OPS=E.with_name(E.name+'-storage-build-preparation')/'records'
checks=[]; facts={};pins={}
def h(b):return hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def hp(p):
 t=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):t.update(b)
 return t.hexdigest()
def ck(name,ok,detail=None):
 checks.append(dict(check=name,passed=bool(ok),detail=detail))
 if not ok: print('CHECK FAILED',name,detail)
def inv(root):
 d=[];f={};special=[]
 for p in sorted(root.rglob('*')):
  s=p.lstat();n=p.relative_to(root).as_posix()
  if stat.S_ISDIR(s.st_mode):d.append(n)
  elif stat.S_ISREG(s.st_mode):f[n]={'sha256':hp(p),'bytes':s.st_size}
  else:special.append(n)
 return dict(files=f,directories=d,special=special)
def const(p,k):
 for n in ast.parse(p.read_bytes()).body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   if any(isinstance(t,ast.Name) and t.id==k for t in targets):return ast.literal_eval(n.value.args[0]) if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='frozenset' and len(n.value.args)==1 else ast.literal_eval(n.value)
 raise ValueError(k)
def git_blobs(specs):
 raw=subprocess.run(['git','cat-file','--batch'],cwd=R,input=('\n'.join(specs)+'\n').encode(),capture_output=True,check=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).stdout
 out={};pos=0
 for spec in specs:
  end=raw.index(b'\n',pos);header=raw[pos:end].split();assert len(header)==3 and header[1]==b'blob',header
  size=int(header[2]);start=end+1;out[spec]=raw[start:start+size];pos=start+size+1
 assert pos==len(raw)
 return out
try:
 outer=j(P/'packet.json');inner=j(P/'inputs/packet.json');adm=j(P/'inputs/admission.json');freeze=j(P/'freeze.json');base=j(P/'cpu-baseline.json');old=j(P/'cpu-packet.json');receipt=j(P/'storage-receipt.json');full=receipt['full_inventory'];sc=outer['source_commit']
 corepins={'compact_inputs.py':'a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245','remote_custody.py':'4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f'}
 rows=base['related_accounting'];gput=0
 for r in rows:
  tr=dict(t.split('=',1) for t in r[7].split(',')); typed=sum(int(v) for k,v in tr.items() if k.startswith('gres/gpu:')); generic=int(tr.get('gres/gpu','0'))
  ck('typed GPU count coherent '+r[0],not generic or not typed or generic==typed)
  gput+=int(r[5])*(generic or typed)
 cput=sum(int(r[4])*int(r[5]) for r in rows);a=outer['admission']
 ck('corrected review parser: unique46 terminal rows exact94 journals closed29261CPU2505GPU',len(rows)==len({r[0] for r in rows})==46 and len(base['journal_pins'])==94 and gput==2505==base['closed_gpu_slice_seconds'] and cput==29261 and base['outstanding_reservation']==0)
 ck('corrected review parser: GPU5385 CPU44381 finalizer46781',a['total_gpu_seconds']==gput+2880==5385 and a['total_cpu_seconds']==cput+3600+11520==44381 and a['total_with_finalizer_cpu_seconds']==cput+3600+11520+2400==46781)
 facts['check01_resolution']='Two failed arithmetic checks were reviewer-only generic gres/gpu parsing; rows use typed gres/gpu:a100_3g.20gb. The stopped literal reader required handling the observed frozenset literal. No packet bytes changed. Passed check01 checks were not repeated.'
 facts['actual_accounting']={'closed_cpu_seconds':cput,'closed_gpu_seconds':gput,'terminal_jobs':len(rows),'journals':len(base['journal_pins'])}
 train=j(P/'training-acceptance.json');service_source=P/'inputs/llm_intervention_model_service.py';service_roster=const(service_source,'SOURCE_NAMES');recipe=const(service_source,'RECIPE')
 for pid in outer['gpu_policy_ids']:
  s=j(P/'inputs'/f'{pid}.freeze.json');contract=j(P/'inputs'/f'{pid}.launch.json');role='sft' if pid=='sft_llm' else 'final';ad=None if pid=='frozen_llm' else dict(file=role+'.adapter.pt',sha256=train[role+'_adapter_sha256'],training_receipt=role+'.checkpoint.json',training_receipt_sha256=train[role+'_checkpoint_sha256'],training_scope='policy_train')
  ck(pid+' unchanged accepted service model runtime recipe identity',s['scope']=='development' and s['runtime']==freeze['runtime'] and s['model_files']==freeze['model_files'] and s['model_directory']==freeze['model_directory'] and s['recipe']==recipe and s['recipe']['new_training_updates']==0 and s['adapter']==ad and s['sources']=={n:inner['files'][n] for n in service_roster} and s['limits']=={'startup_seconds':120,'request_seconds':30,'idle_seconds':30,'total_seconds':1200,'max_requests':576})
  ck(pid+' preserved CPU persistent launch contracts',inner['files'][pid+'.freeze.json']==old['files'][pid+'.freeze.json'] and inner['files'][pid+'.launch.json']==old['files'][pid+'.launch.json'] and contract['freeze_sha256']==inner['files'][pid+'.freeze.json'] and contract['entry_sha256']==inner['files']['llm_intervention_model_service.py'] and contract['archive_sha256']==freeze['runtime_archive_sha256'] and contract['manifest_sha256']==freeze['runtime_manifest_sha256'] and contract['bootstrap_sha256']==h(contract['bootstrap'].encode()))
  if ad:ck(pid+' opaque adapter and checkpoint accepted pins',inner['files'][ad['file']]==ad['sha256'] and inner['files'][ad['training_receipt']]==ad['training_receipt_sha256'])
 ck('all24 descriptors inherited by bytes without decoding',all(inner['files'][f'descriptor-{i:03d}.json']==old['files'][f'descriptor-{i:03d}.json']==full['files'][f'launch/inputs/descriptor-{i:03d}.json']['sha256'] for i in range(24)))
 cfg=j(P/'storage-config.json')['members'];sb=outer['storage_binding'];authority=j(P/'storage-authority.json')
 capture={'after':'storage-after.json','authority':'storage-authority.json','authority_review':'storage-review.md','backend':'remote_custody.py','backend_review':'storage-backend-review.md','receipt':'storage-receipt.json','reference':'storage-predecessor.json','seal':'storage-seal.json'}
 ck('storage config all independent path pins and captured bytes',all(hp(Path(cfg[k]['path']))==cfg[k]['sha256']==hp(P/n) for k,n in capture.items()))
 ck('storage binding final4161 review and original archive full scope',sb['backend_sha256']==corepins['remote_custody.py']==authority['backend_sha256']==receipt['helper_sha256'] and sb['compact_adapter_sha256']==corepins['compact_inputs.py'] and sb['storage_config_sha256']==hp(P/'storage-config.json') and sb['predecessor_remote_custody_sha256']==hp(P/'storage-receipt.json') and receipt['full_backup_verified'] is True and receipt['full_local_copy'] is sb['complete_local_raw_copy'] is False and receipt['backup_archive_sha256']=='52c14412b007825067387eabf4e09f14394804f7e76f654792b9a999b3fd71e3' and authority['freeze_sha256']==hp(P/'freeze.json') and authority['source_commit']==sc and authority['scope']=='evidence_storage_only')
 bg=git_blobs([cfg['backend']['source_commit']+':'+cfg['backend']['repository_path']]) if 'repository_path' in cfg['backend'] else None
 if bg:ck('backend exact separately committed source',h(next(iter(bg.values())))==cfg['backend']['sha256'])
 command=j(ST/'build-command01/command.json');br=j(ST/'build-command01/run01/result.json');bo=j(ST/'build-command01/run01/stdout.log')
 ck('concrete command sources and metadata pins',all(hp(P/n)==v for n,v in command['builder_sources'].items()) and all(hp(Path(v['path']))==v['sha256'] for v in command['metadata'].values()) and hp(ST/'build-command01/command.json')=='fd12a8a9087d087d20b08d0a7543ad0513e49b4176732f55c91576dafb253e40')
 ck('single successful metadata builder result exact emitted pins',br['returncode']==0 and bo['packet_sha256']==hp(P/'packet.json') and bo['gpu_packet_sha256']==hp(P/'inputs/packet.json') and bo['admission_sha256']==hp(P/'inputs/admission.json') and bo['submitted'] is bo['scientific_qualified'] is False)
 facts['build_result']=br
 intentpin=hp(OPS/'intent.json');opcounts=collections.Counter();finalinv=j(OPS/'inventory-098.json');operationinv=inv(OPS/'operations')
 ck('final098 complete operation byte directory membership',not operationinv['special'] and finalinv['inventory']=={k:operationinv[k] for k in ('files','directories')} and hp(OPS/'inventory-098.json')=='43eb2016d5ea87e7eaf6324b2d6220156fae15d859ccc1be8567d2c09844de33')
 errors=[]
 for i in range(99):
  name=f'storage-use-{i:03d}.json';r=j(ST/'storage-build-preparation01'/name);f=Path(r['inventory_path']);x=j(f);opcounts[x['operation']]+=1
  if not (f==OPS/f'inventory-{i:03d}.json' and hp(f)==r['inventory_sha256'] and r['intent_sha256']==x['intent_sha256']==intentpin and x['ordinal']==i and x['status']==r['status']=='completed' and x['error'] is None and r['acceptance'] is r['retry_allowed'] is False and (ST/'storage-build-preparation01'/name).read_bytes()==(OPS.parent/name).read_bytes()):errors.append(i)
 ck('all99 completed operation references pins and both publication copies',not errors,errors)
 ck('packet captured097 and final output098 accurately joined',j(P/'storage-preparation-reference.json')==j(ST/'storage-build-preparation01/storage-use-097.json') and bo['storage_operations']==j(ST/'storage-build-preparation01/storage-use-098.json'))
 for n in ['verify-000','verify-001','verify-002']:
  rr=j(OPS/'operations'/n/'receipt.json');tt=j(OPS/'operations'/n/'transport.json');facts[n]={'receipt':rr,'transport_keys':list(tt),'transport_returncode':tt.get('returncode')}
  pins[str(OPS/'operations'/n/'receipt.json')]=hp(OPS/'operations'/n/'receipt.json')
 facts['storage_operations']={'counts':dict(opcounts),'total':99,'full_operation_files':len(operationinv['files']),'full_operation_bytes':sum(v['bytes'] for v in operationinv['files'].values()),'final_inventory_sha256':hp(OPS/'inventory-098.json'),'packet_capture_inventory_sha256':hp(OPS/'inventory-097.json')}
 for p in [ST/'build-command01/command.json',*sorted((ST/'build-command01/run01').iterdir()),OPS/'intent.json',OPS/'inventory-097.json',OPS/'inventory-098.json']:
  pins[str(p)]=hp(p)
except BaseException:
 checks.append({'check':'reviewer metadata script completion','passed':False,'detail':traceback.format_exc()});print(traceback.format_exc())
finally:
 result={'checks':checks,'failed_checks':[c for c in checks if not c['passed']],'facts':facts,'scope':'Local metadata, opaque bytes, AST literals, and Git blob reads only. No project imports, scientific execution, models, tests, remote access or mutations.'}
 (W/'checks02.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(W/'pins02.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'checks':len(checks),'failed_checks':result['failed_checks'],'facts':{k:v for k,v in facts.items() if not k.startswith('verify-')},'workspace':str(W)},indent=2))
