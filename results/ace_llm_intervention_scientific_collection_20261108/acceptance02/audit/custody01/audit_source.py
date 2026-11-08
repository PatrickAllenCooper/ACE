from audit_common import *
HERE=Path(__file__).resolve().parent
OUT=HERE/'custody01';OUT.mkdir(exist_ok=False)
result={'acceptance':False,'status':'incomplete'}
start=time.monotonic();cpu=time.process_time()
try:
 pins={'disposition.json':'9ab4b1f7fd8aff0e29ec3a8dc2849483d27a3da4071b397d60b95ee46b4e79a9',
 'custody-inventory.json':'9f9abed853fb3d3d935f5a848abef743ab09f07c731ee2b5ea5c4960a68ac00b',
 'inventory_transport.json':'0e28a057408e2a372fdddb42f45a8e9a6abc535518a4cf2feea23a8c3b0c8cb3',
 'inventory_after_transport.json':'69518e41d31d732a063e8ce82ec79e8346e119f42857d3612be46acef85c5d5c'}
 for n,h in pins.items():need(sha(raw(LOCAL/n))==h,'independent custody pin '+n)
 inv=inventory(LOCAL);ext=inventory(EXT);need(inv==ext,'complete two-copy capsule differs')
 sealed=read(LOCAL/'custody-inventory.json');check={**inv,'files':{k:v for k,v in inv['files'].items() if k!='custody-inventory.json'}}
 need(sealed['identical'] is True and sealed['acceptance'] is False and len(sealed['copies'])==2,'custody seal flags')
 need(all(v==check for v in sealed['copies'].values()),'sealed capsule differs')
 for n in inv['files']:
  need((LOCAL/n).stat().st_ino!=(EXT/n).stat().st_ino or (LOCAL/n).stat().st_dev!=(EXT/n).stat().st_dev,'copies share inode')
 observations=[]
 for name in ('inventory_transport.json','inventory_after_transport.json'):
  receipt=read(LOCAL/name);need(receipt['returncode']==0,'observation transport failed')
  v=decode(receipt['stdout'].encode());need(v['terminal_gate'] is True and not v.get('observation_error'),'terminal gate')
  constants={}
  for node in ast.parse(receipt['python_source']).body:
   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
    try:constants[node.targets[0].id]=ast.literal_eval(node.value)
    except (ValueError,TypeError):pass
  need(constants['JOB']==JOB,'wrong observed job');observations.append(v)
 before,after=observations
 for k in ('files','directories','journal_pins','parent_accounting','original_hardlink_groups','upstream_pins','upstream_errors'):
  need(before[k]==after[k],'observations disagree '+k)
 need(normalize(before['related_accounting'])==normalize(after['related_accounting']),'accounting readback')
 remote=inventory(LOCAL/'remote');need(remote==dict(files=before['files'],directories=before['directories']),'complete raw mirror')
 need(len(remote['files'])==30839 and sum(v['bytes'] for v in remote['files'].values())==175574845,'reported raw count/bytes')
 d=read(LOCAL/'disposition.json');need(d['remote_files']==remote['files'] and d['remote_directories']==remote['directories'],'disposition map')
 need(d['status']=='terminal_custody_review_pending' and d['errors']==d['input_pin_mismatches']==[],'collector conflicts')
 for k in ('custody_bytes_verified','current_lineage_verified','exact_prior_accounting_matched','exact_prior_and_current_journals_matched','upstream_training_inputs_verified','failed_collection_custody_verified'):need(d[k] is True,'collector '+k)
 for k in ('acceptance','collection_accepted','scientific_qualified','retry_allowed'):need(d[k] is False,'collector false '+k)
 launch=LOCAL/'remote/launch';packet=read(launch/'packet.json')
 need(sha(raw(launch/'packet.json'))==PACKET and packet['source_commit']==SOURCE,'outer source/packet')
 allpins={**packet['new_input_files'],'packet.json':PACKET};need(len(allpins)==92,'92 transferred objects')
 for n,h in allpins.items():
  b=raw(launch/n);need(sha(b)==h,'input pin '+n)
  need(git(COMMIT,'results/ace_llm_intervention_scientific_collection_20261108/launch02/'+n)==b,'committed packet byte '+n)
  for label in ('local-launch','external-launch'):need(raw(LOCAL/'coordinator'/label/n)==b,'coordinator byte '+n)
 for rel,h in packet['source_pins'].items():
  b=raw(launch/'inputs'/Path(rel).name);need(sha(b)==h and git(SOURCE,rel)==b,'frozen source byte '+rel)
 need(sha(raw(launch/'failure-baseline.json'))==BASELINE,'predecessor41/84 pin')
 base=read(launch/'failure-baseline.json');need(base['terminal_jobs']==41 and base['submission_journals']==84,'predecessor counts')
 ri=read(launch/'submission.intent.json');rr=read(launch/'submission.result.json')
 need(sha(raw(launch/'submission.intent.json'))==INTENT==rr['intent_sha256'],'actual remote intent')
 need(sha(raw(launch/'submission.result.json'))=='669fc7544078afbc532813a71bca4644743dd9172aef3f0458c6f1bf5397bb46','actual result pin')
 need(rr['job_id']==JOB and rr['returncode']==0 and rr['ambiguous'] is False and rr['stdout'].strip()==JOB,'once submission receipt')
 need(ri['packet_commit']==COMMIT and ri['packet_sha256']==PACKET and ri['source_commit']==SOURCE and ri['pins']==allpins,'intent exact source/inputs')
 for label in ('local-launch','external-launch'):
  li=read(LOCAL/'coordinator'/label/'submission.intent.json');lr=read(LOCAL/'coordinator'/label/'submission.result.json')
  need(lr==rr and all(ri[k]==v for k,v in li.items()),'local/remote intent comparison')
  need(set(ri)-set(li)=={'remote_utc','fresh_readiness'},'explicit remote augmentations')
 review='docs/development/guidance/reviews/ace_llm_intervention_scientific_collection02_packet_review_2026-11-08.md'
 need(sha(git(COMMIT,review))==ri['review_sha256'],'committed independent packet review')
 need(sha(raw(LOCAL/'pull_terminal.py'))==allpins['pull_terminal.py']=='5d2bdfdfb47d1d728c5d3d5823f322cd378804979de1dd084a77f7e5ddf0974e','collector exact source')
 # All failed bytes remain; no repeated failure audit/interpretation.
 fd=read(launch/'failure-disposition.json');failed={n.removeprefix('failed01/'):v for n,v in remote['files'].items() if n.startswith('failed01/')}
 need(len(failed)==146 and failed==fd['remote_files'],'whole failed01 raw map unchanged')
 need(sha(raw(launch/'operational-amendment.json'))=='fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855','operational authority')
 old=decode(read(launch/'realization-inventory.json')['stdout'].encode());upstream=before['upstream_pins']
 need(len(upstream)==36 and before['upstream_errors']=={},'accepted realization input set')
 upstream_root=packet['remote_root'].replace('scientific_collection_20261108_02','scientific_realization_20261108_02')
 for path,h in upstream.items():need(old['files']['launch/'+path.removeprefix(upstream_root+'/')]['sha256']==h,'accepted realization byte')
 for n,h in [('namespace.json',packet['namespace_claim_sha256']),('original-namespace.json','58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29')]:need(remote['files'][n]['sha256']==h,'immutable namespace')
 need(sha(raw(LOCAL/'namespaces/local.namespace.json'))==packet['namespace_claim_sha256'],'local namespace')
 rows=normalize(before['related_accounting']);prior=[r for r in rows if r[0]!=JOB];current=[r for r in rows if r[0]==JOB]
 need(len(rows)==42 and len(prior)==41 and prior==normalize(base['related_accounting']) and len(current)==1,'all predecessor allocation rows')
 row=current[0];need(row[:7]==[JOB,'ace_llm_scientific_collect02','COMPLETED','0:0','2','800','ucb736_asc2'],'actual allocation row')
 tres=dict(s.split('=',1) for s in row[7].split(','));need(tres=={'billing':'2','cpu':'2','mem':'7.50G','node':'1'},'no GPU/resource shape')
 journals={**base['journal_pins'],packet['remote_root']+'/submission.intent.json':INTENT,packet['remote_root']+'/submission.result.json':sha(raw(launch/'submission.result.json'))}
 need(len(journals)==86 and before['journal_pins']==journals,'all predecessor/current journals')
 need(all(r[2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL') for r in rows),'all rows terminal')
 cpu_total=sum(int(r[4])*int(r[5]) for r in rows);gpu_total=0
 for r in rows:
  tr=dict(x.split('=',1) for x in r[7].split(','));gp=int(tr.get('gres/gpu',sum(int(v) for k,v in tr.items() if k.startswith('gres/gpu:'))));gpu_total+=gp*int(r[5])
 need(cpu_total==base['closed_cpu_seconds']+1600==12986 and gpu_total==base['closed_gpu_slice_seconds']==1601,'exact cumulative actual cost')
 need(base['outstanding_reservation']==0 and base['preparation_allowance_cpu_seconds']==3600,'baseline allowance')
 for k in ('related_accounting','steps','details','queue','job'):need(before['results'][k]['returncode']==0,'accounting command failed '+k)
 need(not before['results']['queue']['stdout'].strip(),'no live reservation')
 need(d['allocation']['resource_contract_matched'] is True and d['allocation']['errors']==[],'resource observations')
 need(d['closed_cpu_seconds']==cpu_total and d['closed_gpu_slice_seconds']==gpu_total and d['cpu_admission_after_release']==cpu_total+3600,'collector cost independently derived')
 result.update(status='verified_saved_custody_and_accounting',pins=pins,raw_files=len(remote['files']),raw_bytes=sum(v['bytes'] for v in remote['files'].values()),capsule_files=len(inv['files']),source_files=len(packet['source_pins']),transferred_objects=len(allpins),related_accounting=rows,journal_pins=journals,terminal_jobs=42,submission_journals=86,closed_cpu_seconds=cpu_total,closed_gpu_slice_seconds=gpu_total,cpu_admission_after_release=cpu_total+3600,outstanding_reservation=0,current_allocation_row=row,packet_sha256=PACKET,packet_commit=COMMIT,source_commit=SOURCE)
 save(OUT/'complete-capsule-map.json',inv)
except BaseException as exc:
 result.update(error=type(exc).__name__+': '+str(exc));raise
finally:
 result.update(process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-start)
 save(OUT/'audit.json',result)
