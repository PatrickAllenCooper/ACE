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
   if any(isinstance(t,ast.Name) and t.id==k for t in targets):return ast.literal_eval(n.value)
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
 outer=j(P/'packet.json');inner=j(P/'inputs/packet.json');adm=j(P/'inputs/admission.json');freeze=j(P/'freeze.json');base=j(P/'cpu-baseline.json');old=j(P/'cpu-packet.json');receipt=j(P/'storage-receipt.json')
 known={'packet.json':'436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7','inputs/packet.json':'85d2607d94114109819ca5e8150c5849c13f1bbf4b37765d3f7ee2ff64e77c58','inputs/admission.json':'330d7cc1902d923df2f1c83861822a8f44d5329b71729ed4e123b86c3ce87689','cpu-acceptance.json':'3c5ce7375d0c64292553ded6ad3078e634745eb7996d83fd102e344eeab3f063','cpu-baseline.json':'c21f43194ff61b30c0322acff99c20e71158d980a00a34a1685b87eac7f27125','cpu-review.md':'396ef13d95db6f31d2d8708af55ced0c666e02c0add10ed664c75c33f6c7d653','freeze.json':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d','storage-receipt.json':'f0f8a52c7f823b5a07041faf531382afa4d7c0364452413bbb17bd916dfc38e6','storage-seal.json':'24f97438d6787079514e1a40dd4ab18401e7028207913d21490af87b69ca72a4'}
 lp=inv(P);ep=inv(E);ck('two packet mirrors exact regular membership and bytes',lp==ep and not lp['special'])
 ck('outer complete 177 input pins plus packet',set(lp['files'])==set(outer['new_input_files'])|{'packet.json'} and all(lp['files'][n]['sha256']==v for n,v in outer['new_input_files'].items()))
 ck('independently supplied packet and accepted pins',all(lp['files'][n]['sha256']==v for n,v in known.items()))
 pins.update({str(P/n):v['sha256'] for n,v in lp['files'].items()})
 facts['packet_mirrors']={'roots':[str(P),str(E)],'files':len(lp['files']),'bytes_each':sum(v['bytes'] for v in lp['files'].values()),'directories':lp['directories']}
 (W/'packet-inventory.json').write_text(json.dumps(lp,indent=2,sort_keys=True)+'\n')
 ck('inner complete input membership and pin map',set(n.removeprefix('inputs/') for n in lp['files'] if n.startswith('inputs/'))==set(inner['files'])|{'packet.json','admission.json'} and all(lp['files']['inputs/'+n]['sha256']==v for n,v in inner['files'].items()))
 ck('original CPU input bytes all unchanged',all(inner['files'].get(n)==v for n,v in old['files'].items()))
 closure=freeze['stage_closure']['stages']['GPUcampaign'];sc=outer['source_commit'];entry=closure['entrypoint']
 ck('entry and full captured source roster match frozen closure',inner['source_names']==closure['source_names'] and all(inner['files'][n]==v for n,v in closure['input_source_files'].items()) and inner['files'][entry]==closure['entrypoint_sha256']=='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1')
 specs=[sc+':'+n for n in freeze['source_pins']];blobs=git_blobs(specs)
 ck('all 67 frozen Git source objects match',len(blobs)==67 and all(h(blobs[sc+':'+n])==v for n,v in freeze['source_pins'].items()))
 ck('captured 56 source inputs bind unique Git origins',all(len([origin for origin,pin in freeze['source_pins'].items() if Path(origin).name==n and pin==v])==1 for n,v in closure['input_source_files'].items()))
 extra=set(closure['input_source_files'])-set(old['files']);facts['source_rosters']={'frozen_git_sources':len(blobs),'loaded_work_sources':len(inner['source_names']),'input_source_files':len(closure['input_source_files']),'added_gpu_source_names':sorted(extra),'cpu_inherited_files':len(old['files']),'gpu_inner_files':len(inner['files'])}
 ck('exact inherited inputs plus GPU sources and two binding records',set(inner['files'])==set(old['files'])|extra|{'cpu-packet.json','cpu-acceptance.json'})
 ck('CPU inner and acceptance duplicated without rewriting',inner['files']['cpu-packet.json']==hp(P/'cpu-packet.json') and inner['files']['cpu-acceptance.json']==hp(P/'cpu-acceptance.json'))
 reviewroot=R/'results/ace_llm_intervention_remote_custody_20261109/downstream-fullmap128-independent-review01'
 priorpins=j(reviewroot/'source-pins.json');facts['prior_review_pin_schema']=list(priorpins) if isinstance(priorpins,dict) else type(priorpins).__name__
 # Reviewed pins independently supplied in the request/history; full source is not re-reviewed.
 corepins={'build_packet.py':'44606908e80df1480bb80c465630a9ea96086ede67a40798f1e03827f0f808d5','launch_once.py':'8b3d723d4625b6b16a06564e11280932e83e86a788c807bbae3958161c9b8bc4','monitor_once.py':'220ba06cf4d3bb2ef40d09e7d41b59300a3ba93163d5ab73bdbf45f2adc24b32','cpu_history_contract.py':'3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025','compact_inputs.py':'a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245','remote_custody.py':'4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f'}
 ck('reviewed coordinators and backend exact captured bytes',all(hp(P/n)==v for n,v in corepins.items()) and hp(reviewroot/'review.md')=='18c3ef244ece5fb5d8ec55a326c15a248bb410fddd59aa4524a460755c1b6604')
 pins[str(reviewroot/'review.md')]=hp(reviewroot/'review.md');pins[str(reviewroot/'source-pins.json')]=hp(reviewroot/'source-pins.json')
 ca=R/'results/ace_llm_intervention_scientific_campaign_cpu_20261108/acceptance01';commit='9d85974ae97ee281e2b233939a04883b56c2640c';names={'cpu-acceptance.json':'acceptance.json','cpu-baseline.json':'baseline.json','cpu-review.md':'review.md'}
 gs=git_blobs([commit+':'+str((ca/n).relative_to(R)) for n in names.values()])
 ck('accepted CPU acceptance review baseline equal committed bytes',all((P/k).read_bytes()==(ca/n).read_bytes()==gs[commit+':'+str((ca/n).relative_to(R))] for k,n in names.items()))
 rows=base['related_accounting'];jp=base['journal_pins'];cput=sum(int(r[4])*int(r[5]) for r in rows);gput=0
 for r in rows:
  tr=dict(t.split('=',1) for t in r[7].split(','));gput+=int(r[5])*int(tr.get('gres/gpu','0'))
 ck('46 unique terminal rows 94 exact accepted journal pins totals',len(rows)==len({r[0] for r in rows})==46 and len(jp)==94 and cput==base['closed_cpu_seconds']==29261 and gput==base['closed_gpu_slice_seconds']==2505 and base['outstanding_reservation']==0)
 ck('admission resource and deadline joins',outer['resources']==inner['resources']==adm['resources'] and outer['limits']==inner['limits']==adm['limits'] and adm['packet_sha256']==hp(P/'inputs/packet.json') and adm['accounting_sha256']==outer['baseline_sha256']==hp(P/'cpu-baseline.json') and adm['site_readiness_sha256']==hp(P/'site-readiness.md'))
 stage=next(s for s in freeze['stages'] if s['stage']=='gpu_campaign');res=outer['resources'];a=outer['admission']
 ck('exact frozen GPU allocation limits paths',all(res[k]==stage[k] for k in ['account','partition','qos','memory_mib','gpus','gpu_type','wall_seconds','requeue']) and res['requested_cpus']==res['max_allocated_cpus']==stage['cpus']==4 and outer['remote_root']==stage['remote_root'] and outer['limits']==stage['deadlines'])
 ck('candidate and remaining-finalizer budget',a['candidate_cpu_reservation']==4*2880==11520 and a['candidate_gpu_reservation']==2880 and a['total_cpu_seconds']==cput+3600+11520==44381 and a['total_gpu_seconds']==gput+2880==5385 and a['remaining_finalizer_cpu_cap']==2400 and a['total_with_finalizer_cpu_seconds']==46781<=57600 and a['total_gpu_seconds']<=7200)
 facts['accounting']={'rows':len(rows),'journals':len(jp),'closed_cpu_seconds':cput,'closed_gpu_seconds':gput,'preparation_cpu_allowance':3600,**a}
 full=receipt['full_inventory'];roots=receipt['roots'];up={}
 for n,v in full['files'].items():
  root,sep,tail=n.partition('/')
  if sep:up[roots[root]+'/'+tail]=v['sha256']
 ck('entire original CPU and historical raw source maps inherited',outer['original_cpu_inputs']==up and len(up)==72372)
 ck('exact five historical directory rosters',all(d==sorted(n.removeprefix(next(k for k,v in roots.items() if v==root)+'/') for n in full['directories'] if n.startswith(next(k for k,v in roots.items() if v==root)+'/')) for root,d in outer['history_directories'].items()) and len(outer['history_directories'])==5)
 cpuaccept=j(P/'cpu-acceptance.json');cpuprefix='launch/cpu/';camp='launch/cpu/worker/artifacts/campaign/'
 ck('CPU output and campaign maps from full archive inventory',cpuaccept['campaign_files']=={n.removeprefix(camp):v['sha256'] for n,v in full['files'].items() if n.startswith(camp)} and cpuaccept['cpu_output_files']=={n.removeprefix(cpuprefix):v['sha256'] for n,v in full['files'].items() if n.startswith(cpuprefix) and not n.startswith(camp)})
 ck('all94 journal map entries present and identical where rooted in full history',all(up[path]==pin for path,pin in jp.items() if path in up))
 facts['journal_pins_in_this_full_history']=sum(p in up for p in jp)
 ck('namespace active and original exact full map pins',full['files']['namespace.json']['sha256']==outer['namespace_claim_sha256']==hp(P/'namespace-claim.json') and full['files']['original-namespace.json']['sha256']=='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29')
 ck('frozen scientific scope no new training/worlds/outcome claim',outer['worlds']==len(freeze['development_world_ids'])==24 and outer['cells']==freeze['endpoints']['cells']==192 and outer['policy_ids']==freeze['policies'] and outer['gpu_policy_ids']==freeze['policies'][5:] and outer['new_training_updates']==outer['new_worlds']==0 and outer['scientific_qualified'] is outer['execution_qualified'] is outer['submitted'] is False and inner['scope']=='development')
 train=j(P/'training-acceptance.json');service_source=P/'inputs/llm_intervention_model_service.py';service_roster=const(service_source,'SOURCE_NAMES');recipe=const(service_source,'RECIPE')
 for pid in outer['gpu_policy_ids']:
  s=j(P/'inputs'/f'{pid}.freeze.json');contract=j(P/'inputs'/f'{pid}.launch.json');role='sft' if pid=='sft_llm' else 'final';ad=None if pid=='frozen_llm' else dict(file=role+'.adapter.pt',sha256=train[role+'_adapter_sha256'],training_receipt=role+'.checkpoint.json',training_receipt_sha256=train[role+'_checkpoint_sha256'],training_scope='policy_train')
  ck(pid+' unchanged accepted service model runtime recipe identity',s['scope']=='development' and s['runtime']==freeze['runtime'] and s['model_files']==freeze['model_files'] and s['model_directory']==freeze['model_directory'] and s['recipe']==recipe and s['recipe']['new_training_updates']==0 and s['adapter']==ad and s['sources']=={n:inner['files'][n] for n in service_roster} and s['limits']=={'startup_seconds':120,'request_seconds':30,'idle_seconds':30,'total_seconds':1200,'max_requests':576})
  ck(pid+' preserved CPU persistent launch contracts',inner['files'][pid+'.freeze.json']==old['files'][pid+'.freeze.json'] and inner['files'][pid+'.launch.json']==old['files'][pid+'.launch.json'] and contract['freeze_sha256']==inner['files'][pid+'.freeze.json'] and contract['entry_sha256']==inner['files']['llm_intervention_model_service.py'] and contract['archive_sha256']==freeze['runtime_archive_sha256'] and contract['manifest_sha256']==freeze['runtime_manifest_sha256'] and contract['bootstrap_sha256']==h(contract['bootstrap'].encode()))
  if ad:ck(pid+' opaque adapter and checkpoint accepted pins',inner['files'][ad['file']]==ad['sha256'] and inner['files'][ad['training_receipt']]==ad['training_receipt_sha256'])
 ck('all24 descriptors inherited by bytes without decoding',all(inner['files'][f'descriptor-{i:03d}.json']==old['files'][f'descriptor-{i:03d}.json']==full['files'][f'launch/inputs/descriptor-{i:03d}.json']['sha256'] for i in range(24)))
 cfg=j(P/'storage-config.json');sb=outer['storage_binding'];authority=j(P/'storage-authority.json')
 capture={'after':'storage-after.json','authority':'storage-authority.json','authority_review':'storage-review.md','backend':'remote_custody.py','backend_review':'storage-backend-review.md','receipt':'storage-receipt.json','reference':'storage-predecessor.json','seal':'storage-seal.json'}
 ck('storage config all independent path pins and captured bytes',all(hp(Path(cfg[k]['path']))==cfg[k]['sha256']==hp(P/n) for k,n in capture.items()))
 ck('storage binding final4161 review and original archive full scope',sb['backend_sha256']==corepins['remote_custody.py']==authority['backend_sha256']==receipt['helper_sha256'] and sb['compact_adapter_sha256']==corepins['compact_inputs.py'] and sb['storage_config_sha256']==hp(P/'storage-config.json') and sb['predecessor_remote_custody_sha256']==hp(P/'storage-receipt.json') and receipt['full_backup_verified'] is True and receipt['full_local_copy'] is sb['complete_local_raw_copy'] is False and receipt['backup_archive_sha256']=='52c14412b007825067387eabf4e09f14394804f7e76f654792b9a999b3fd71e3' and authority['freeze_sha256']==hp(P/'freeze.json') and authority['source_commit']==sc and authority['scope']=='evidence_storage_only')
 bg=git_blobs([cfg['backend']['source_commit']+':'+cfg['backend']['repo_relative_path']]) if 'repo_relative_path' in cfg['backend'] else None
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
 (W/'checks01.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(W/'pins01.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'checks':len(checks),'failed_checks':result['failed_checks'],'facts':{k:v for k,v in facts.items() if not k.startswith('verify-')},'workspace':str(W)},indent=2))
