"""Remaining native-device and actual loader/cleanup metadata review, stdlib only."""
import hashlib,json,re,stat,traceback
from pathlib import Path,PurePosixPath
OUT=Path(__file__).parent;A=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-gpu02-35516125-postcustody-01')
pins={};checks=0
def need(v,m):
 global checks
 checks+=1
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p,pin=None):
 b=Path(p).read_bytes()
 if pin is not None:need(sha(b)==pin,'pin '+str(p))
 pins[str(p)]=dict(bytes=len(b),sha256=sha(b));return b
def save(n,j):
 with (OUT/n).open('x') as f:json.dump(j,f,sort_keys=True,indent=2);f.write('\n')
def inside(path,parent):
 p,q=PurePosixPath(path),PurePosixPath(parent)
 return p==q or q in p.parents
try:
 c=json.loads(read(A/'inputs/config.json','952dcdcb39f54330374552400118c040adb1a6f2fba0cb9777136f55379379a2'));T=Path(c['capsules'][1])
 receipt=json.loads(read(c['artifacts']['remote_custody']['path'],c['artifacts']['remote_custody']['sha256']));full=receipt['full_inventory']['files']
 def get(n):
  path=Path(c['fetched'][n]['path']) if n in c['fetched'] else T/'remote'/n
  return json.loads(read(path,full[n]['sha256']))
 p=json.loads(read(c['artifacts']['inner']['path'],c['artifacts']['inner']['sha256']));old=get('control/inputs/cpu-packet.json');G='launch/gpu/'
 stage=get(G+'stage.json');prefix=stage['staged_prefix'];original=old['runtime_source'];runtime=old['runtime']
 need(prefix=='/scratch/local/35516125/ace-scientific-campaign-gpu-35516125/runtime' and not inside(prefix,original) and not inside(original,prefix),'disjoint fixed node-local runtime')
 need(runtime['gpu_mapping_source_sha256']==p['files']['llm_intervention_gpu_mappings.py']=='ad1c6854e720473a800a6aeb8cb5f5530033abb2a4c8a18c6af5417063aaa856','frozen mapping source pin')
 read(T/'remote/launch/inputs/llm_intervention_gpu_mappings.py',runtime['gpu_mapping_source_sha256'])
 records=[];policies=['frozen_llm','sft_llm','dpo_llm']
 for policy in [None]+policies:
  root=G+'controller/runtime/' if policy is None else G+'services/'+policy+'/worker/'
  bound=get(root+'runtime-stage-binding.json');initial=get(root+'runtime-before-imports.json');prov=get(root+'runtime-provenance-sources.json')
  need(bound['stage']==stage and bound['contract']==runtime and bound['stage_sha256']==full[G+'stage.json']['sha256'],'actual bound stage and runtime')
  need(initial['python_version']=='3.10.19' and initial['prefix']==prefix and not initial['generated_provenance']['present'],'initial staged Python')
  phases=['after-service-imports','after-campaign-host'] if policy is None else ['after-service-imports','after-model-load','after-requests']
  freeze=None if policy is None else get('control/inputs/'+p['services'][policy])
  source_map={PurePosixPath(n).stem:p['files'][n] for n in p['source_names'] if n!='qualify_llm_intervention_scientific_campaign_gpu.py'} if policy is None else {PurePosixPath(n).stem:h for n,h in freeze['sources'].items() if n!='llm_intervention_model_service.py'}
  for phase in phases:
   s=get(root+'runtime-'+phase+'.json');projects=s['captured_project_modules']
   need(s['prefix']==s['base_prefix']==prefix and inside(s['executable'],prefix) and s['sys_path']==initial['sys_path']+[s['generated_provenance']['directory']],'actual isolated runtime roots/path')
   need(set(source_map)<=set(projects)<=set(source_map)|{'__main__','__mp_main__'},'exact permitted project roster')
   input_root=p['paths']['gpu_inputs'] if policy is None else p['paths']['cpu_inputs']
   for name,row in projects.items():
    need(row['sha256']==p['files'][PurePosixPath(row['path']).name] and str(PurePosixPath(row['path']).parent)==input_root,'actual captured source origin/pin')
    if name in source_map:need(row['sha256']==source_map[name] and row['loader']=='authenticated_captured_bytes','captured-byte loader provenance')
    else:need(row['loader']=='outer_authenticated_entrypoint','entrypoint alias provenance')
   versions={k:v['version'] for k,v in s['packages'].items()};versions.update(s['additional_package_versions'])
   need(versions==runtime['imported_versions'] and s['python_version']=='3.10.19','actual package versions')
   proof=s['generated_provenance']
   need(proof['verified'] is True and proof['source_content']==prov['expected_source'] and sha(proof['source_content'].encode())==proof['source_sha256']==proof['expected_source_sha256']==prov['expected_source_sha256'],'generated module exact source')
   need(proof['temporary_root']==bound['temporary_root'] and inside(proof['path'],bound['temporary_root']),'owned generated module path')
   need(s['mixed_modules']=={'_remote_module_non_scriptable':proof['path']} and s['mixed_module_paths']==[proof['path']] and not any(v.get('version_error') for v in s['packages'].values()),'exact original generated-module exception')
   need(all(inside(path,prefix) or path==proof['path'] for path in s['module_paths'].values()),'no mixed module path except authenticated generated source')
   rows=[]
   for line in s['process_maps'].splitlines():
    fields=line.split(None,5);need(5<=len(fields)<=6,'raw map columns')
    address=re.fullmatch(r'([0-9a-f]+)-([0-9a-f]+)',fields[0]);need(address is not None and int(address[1],16)<int(address[2],16),'map address range')
    need(re.fullmatch(r'[r-][w-][x-][ps]',fields[1]) is not None and re.fullmatch('[0-9a-f]+',fields[2]) is not None and re.fullmatch('[0-9a-f]+:[0-9a-f]+',fields[3]) is not None and fields[4].isdigit(),'map fields')
    if len(fields)<6 or not fields[5].startswith('/'):continue
    path=fields[5];deleted=path.endswith(' (deleted)');path=path[:-10] if deleted else path
    major,minor=(int(x,16) for x in fields[3].split(':'))
    rows.append(dict(line=line,path=path,deleted=deleted,permissions=fields[1],offset=int(fields[2],16),dev_major=major,dev_minor=minor,inode=int(fields[4])))
   rejected=sorted({r['path'] for r in rows if inside(r['path'],original) or not inside(r['path'],prefix) and not any(inside(r['path'],x) for x in ['/lib','/lib64','/usr/lib','/usr/lib64'])})
   need(rejected==s['mixed_native_paths'],'all raw native paths classified')
   if not rejected:
    records.append(dict(policy=policy,phase=phase,snapshot_sha256=full[root+'runtime-'+phase+'.json']['sha256'],device_rows=0));continue
   need(policy is not None and phase in ['after-model-load','after-requests'],'GPU devices only after model load')
   specials={'/dev/nvidiactl','/dev/nvidia-uvm','/dev/zero'};indexed=set(rejected)-specials
   need(len(indexed)==1 and len(rejected)==4,'exact4 device paths');index=next(iter(indexed));m=re.fullmatch(r'/dev/nvidia(0|[1-9][0-9]{0,2})',index)
   need(m is not None and int(m[1])<254,'canonical physical NVIDIA index')
   inv=get(root+'gpu-mapping-inventory-'+phase+'.json');v=get(root+'gpu-mapping-validation-'+phase+'.json')
   need(set(inv)=={'schema','devices','proc_devices','proc_devices_sha256'} and inv['schema']=='ace-gpu-device-map-observation-v2' and set(inv['devices'])==set(rejected),'device inventory exact schema/roster')
   need(sha(inv['proc_devices'].encode('ascii'))==inv['proc_devices_sha256'] and len(inv['proc_devices'])<=65536,'kernel registry byte pin')
   registry={};section=None
   for line in inv['proc_devices'].splitlines():
    if line in ['Character devices:','Block devices:']:section=line;continue
    if section!='Character devices:' or not line.strip():continue
    parts=line.split();need(len(parts)==2 and parts[0].isdigit() and 0<int(parts[0])<2**20,'character registry entry')
    if parts[1] in ['nvidia','nvidia-uvm']:
     need(parts[1] not in registry,'unique required driver');registry[parts[1]]=int(parts[0])
   need(set(registry)=={'nvidia','nvidia-uvm'} and registry['nvidia']==195,'exact frontend and UVM registry')
   expected={index:(195,int(m[1])),'/dev/nvidiactl':(195,255),'/dev/nvidia-uvm':(registry['nvidia-uvm'],0),'/dev/zero':(1,5)}
   for path,row in inv['devices'].items():
    need(set(row)=={'uid','mode','inode','dev_major','dev_minor','rdev_major','rdev_minor'} and all(type(x) is int and x>=0 for x in row.values()),'device exact integer stat')
    need(row['uid']==0 and stat.S_ISCHR(row['mode']) and row['inode']>0 and (row['rdev_major'],row['rdev_minor'])==expected[path],'device owner/type/rdev')
   selected=[r for r in rows if r['path'] in set(rejected)]
   need({r['path'] for r in selected}==set(rejected),'complete selected device maps')
   for row in selected:
    device=inv['devices'][row['path']]
    need(re.fullmatch('[r-][w-]-s',row['permissions']) is not None and row['inode']>0,'non-executable shared map')
    if row['path']=='/dev/zero':need(row['deleted'] and row['permissions']=='rw-s' and row['offset']==0 and (row['dev_major'],row['dev_minor'])==(0,1),'exact anonymous shared-memory backing')
    else:need(not row['deleted'] and (row['dev_major'],row['dev_minor'],row['inode'])==(device['dev_major'],device['dev_minor'],device['inode']),'actual mapped device inode identity')
   expected_v=dict(schema=inv['schema'],phase=phase,classified_rows=selected,original_mixed_native_paths=rejected,device_inventory=inv,bound_runtime_prefix=prefix,bound_original_runtime=original,original_snapshot_preserved=True,gpu_assignment_proved=False,scope='Non-executable device/shared mappings only. Other runtime guards still required.')
   need(v==expected_v,'full original classification evidence exact')
   records.append(dict(policy=policy,phase=phase,snapshot_sha256=full[root+'runtime-'+phase+'.json']['sha256'],device_rows=len(selected),physical_node=index,physical_minor=int(m[1]),uvm_major=registry['nvidia-uvm']))
 phase=get(G+'controller/phase/terminal.json');cleanup=[]
 for i,s in enumerate(phase['sessions'],5):
  pid=s['policy_id'];client='cpu/cpu/worker/artifacts/campaign/'+f'policy-{i:02d}/service/'
  closed=get(client+'closed.json');clean=get(client+'close-cleanup.json');process=get(client+'process.json');post=get(G+'controller/phase/'+pid+'/post-exit-gpu.json')
  need(s['error'] is None and s['process_group_closed'] is s['post_exit_gpu_empty'] is True,'session close authority')
  need(closed['cleanup']==clean and clean['errors']==[] and clean['group_still_exists'] is False and clean['leader_returncode']==0 and clean['term_sent'] is clean['kill_sent'] is False,'original natural owned process closure')
  need(clean['pgid']==s['registered_pid']==process['pid']==process['pgid'],'owned group identity')
  cleanup.append(dict(policy_id=pid,pid=s['registered_pid'],cleanup_sha256=full[client+'close-cleanup.json']['sha256'],post_exit_sha256=full[G+'controller/phase/'+pid+'/post-exit-gpu.json']['sha256'],post_exit_record=post))
 command=get(G+'controller-command.json');handoff=get(G+'handoff.json');entry=p['files']['qualify_llm_intervention_scientific_campaign_gpu.py']
 need(command['same_pid_exec'] is True and command['command'][:4]==[prefix+'/bin/python','-I','-B','-c'] and entry in command['command'],'actual isolated pinned controller bootstrap')
 need(handoff['job_id']=='35516125' and handoff['packet_sha256']==c['artifacts']['inner']['sha256'],'original controller handoff')
 save('runtime-facts.json',dict(checks=checks,records=records,owned_cleanup=cleanup,accepted=False,scope='Saved native map/stat/driver/loader provenance and owned cleanup metadata only; no runtime functions executed or live process claims'))
 save('runtime-read-pins.json',pins);print(json.dumps(dict(checks=checks,snapshots=len(records),full_device_snapshots=sum(bool(x['device_rows']) for x in records),owned_sessions=len(cleanup))))
except BaseException:
 save('runtime-partial-pins.json',pins)
 with (OUT/'runtime-failure.txt').open('x') as f:f.write(traceback.format_exc())
 raise
