from pathlib import Path
import json,hashlib,datetime
p=Path(__file__).parent;r=p/'raw';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda n:json.loads((r/n).read_text())
inv=json.loads((p/'remote_inventory.json').read_text());custody=json.loads((p/'custody_verification.json').read_text())
assert custody['job_id']==34864950 and custody['parent_state']=='COMPLETED' and custody['frozen_inputs_verified']==17
assert sha(r/'report_freeze.json')=='256887bcbcbd989e0bab93655b68f371c0615d648a83d816ffeddac1f0a65686'
freeze=read('report_freeze.json');execution=read('report_execution.json');receipt=read('report_receipt.json');runtime=read('report_runtime_verified.json')
assert execution['status']=='complete' and execution['job_id']=='34864950' and execution['freeze_sha256']==sha(r/'report_freeze.json')
assert execution['child']['status']=='complete' and execution['child']['exit_code']==0 and execution['child']['cleanup']['complete'] is True
assert execution['report_receipt_sha256']==sha(r/'report_receipt.json') and receipt['status']=='complete' and receipt['freeze_sha256']==sha(r/'report_freeze.json')
for k in ['source_pins','metadata_pins','original_input_pins']:assert receipt[k]==freeze[k]
assert receipt['metadata_pins']['replay_acceptance.json']=='57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232'
assert receipt['metadata_pins']['replay_receipt.json']=='d824cbbe125ae7ed50222298bba227c84ab4384685c4b625f761801e28f5b98b'
assert runtime['inventory_sha256']==receipt['metadata_pins']['runtime_inventory.json']=='27ca032d9106b9684138b3e9363a667557fcd80788629b221c07e0eb9c2801c0'
assert runtime['environment_objects']==28577
assert {k:v['version'] for k,v in receipt['statistics_dependencies'].items()}=={'numpy':'2.2.6','scipy':'1.15.3'}
assert runtime['dependencies']==receipt['statistics_dependencies'] and receipt['loaded_statistics_files']
for k in ['new_fits','new_responses','new_optimizer_updates']:assert receipt[k]==freeze[k]==0
assert receipt['counts']=={'cells':640,'fixed_init0':320,'system_log_ratios':80,'history_specific_ratios':8,'short_fit_ablation':2,'primary_floor_counts':4}
assert len(receipt['outputs'])==13
for n,h in receipt['outputs'].items():assert sha(r/n)==h,n
assert {f.relative_to(r).as_posix() for d in ['primary','descriptive'] for f in (r/d).iterdir()}==set(receipt['outputs'])
rows=[line.split('|') for line in inv['commands']['sacct']['stdout'].splitlines()];parent=[v for v in rows if v[:2]==['34864950','34864950']];assert len(parent)==1
v=parent[0];assert v[2:4]==['COMPLETED','0:0'];assert v[9]=='1' and v[13:16]==['ucb736_asc1','acpu','cpu-normal'];assert v[12]=='cpu=1,mem=3G,node=1';assert int(v[6])==int(v[7])<=1800
assert not (r/'launcher_failure.json').exists()
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':34864950,'actual_execution_checks_pass':True,'independent_scientific_output_review_pending':True,'freeze_sha256':sha(r/'report_freeze.json'),'execution_sha256':sha(r/'report_execution.json'),'report_receipt_sha256':sha(r/'report_receipt.json'),'runtime_verified_sha256':sha(r/'report_runtime_verified.json'),'remote_inventory_sha256':sha(p/'remote_inventory.json'),'custody_sha256':sha(p/'custody_verification.json'),'primary_index_sha256':sha(r/'primary/claim_index.json'),'descriptive_index_sha256':sha(r/'descriptive/output_index.json'),'counts':receipt['counts'],'new_fits':0,'new_optimizer_updates':0,'new_responses':0,'allocation_seconds':int(v[7]),'known_cumulative_allocated_seconds':128690+int(v[7]),'known_cumulative_allocated_core_hours':(128690+int(v[7]))/3600,'reservation_core_hours':93.36833333333334,'slurm_total_cpu':v[8],'slurm_max_rss_by_step':{row[1]:row[11] for row in rows},'child_process_cpu_seconds':receipt['process_cpu_seconds'],'child_elapsed_seconds':receipt['elapsed_seconds'],'child_peak_rss_kib':receipt['peak_self_rss_kib'],'wrapper_elapsed_seconds':execution['elapsed_seconds'],'measurements_overlap_not_additive':True,'whole_sprint_cpu_unknown':True}
with (p/'execution_verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
