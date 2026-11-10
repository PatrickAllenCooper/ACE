import json,pathlib,hashlib
W=pathlib.Path(__file__).parent;R=pathlib.Path('/Users/pat/code/ACE');G=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108';L=G/'launch02';O=L/'transfer-continuation01';S=pathlib.Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-20261109-02/transfer-continuation01/storage-operations/operations')
def j(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def ck(n,b):checks.append(dict(check=n,passed=bool(b)))
p=j(L/'packet.json');ctx=j(O/'preserve.request.json')['context'];cpu=json.loads(j(L/'cpu-inventory.json')['stdout']);failed=json.loads(j(L/'gpu-failure-inventory.json')['stdout']);old=j(L/'cpu-packet.json');backup=j(L/'storage-receipt.json')
ck('complete_context_cpu_history_roster',ctx['cpu_inputs']==p['original_cpu_inputs'] and len(ctx['cpu_inputs'])==72372 and ctx['history_directories']==p['history_directories'])
ck('full_cpu_launch_roster_and_directories',ctx['cpu_roster']=={n.removeprefix('launch/'):v['sha256'] for n,v in cpu['files'].items() if n.startswith('launch/')} and ctx['cpu_directories']==sorted(n[7:] for n in cpu['directories'] if n.startswith('launch/')))
ck('original_failed_gpu_roster',ctx['failed_gpu_inventory']==dict(directories=sorted(n[7:] for n in failed['directories'] if n.startswith('launch/')),files={n.removeprefix('launch/'):v for n,v in failed['files'].items() if n.startswith('launch/')}))
ck('runtime_model_identity',all(ctx[k]==old[k] for k in ['archive','manifest','model_directory','model_files']))
for f in ['verify-000','verify-001','fetch-log-028','fetch-log-091']:
 t=j(S/f/'transport.json');v=j(S/f/'receipt.json')
 ck(f+'_saved_transport',t['returncode']==0 and t['remote_outcome_ambiguous'] is False and t['retry_allowed'] is False)
 ck(f+'_accepted_backup_binding',v['acceptance'] is False and v['helper_sha256']==p['storage_binding']['backend_sha256'] and v['backup_archive_sha256']==backup['backup_archive_sha256'])
 if f.startswith('verify'):
  ck(f+'_full_archive_map',v['full_backup_verified'] is True and v['full_inventory']==backup['full_inventory'] and v['receipt_sha256']==p['storage_binding']['predecessor_remote_custody_sha256']==sha(L/'storage-receipt.json'))
 else:
  ck(f+'_opaque_input_pin',v['verified'] is True and v['sha256']==p['new_input_files'][v['member'].removeprefix('launch/')] and v['bytes']==4622303 and v['member'] in ['launch/inputs/final.adapter.pt','launch/inputs/sft.adapter.pt'])
report=dict(checks=checks,failed_checks=[x for x in checks if not x['passed']],scope='Remaining packet-specific historical-map/storage-receipt joins only. Original private/model payloads not decoded; no project callable execution.')
(W/'remaining-joins-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
