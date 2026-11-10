"""One independent saved-metadata comparison; no project imports or callable execution."""
from pathlib import Path, PurePosixPath
import ast, datetime, hashlib, json
W=Path(__file__).parent
R=Path('/Users/pat/code/ACE')
P=R/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
O=P/'actual-preflight01/objects'
checks=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return sha(canonical(v))
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
def decode(b):
 v=json.loads(b,object_pairs_hook=unique);canonical(v);return v
def check(label,condition,**evidence):
 checks.append(dict(check=label,passed=bool(condition),**evidence))
def iso(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def metadata_inventory(root):
 return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(root.rglob('*')) if p.is_file()}
expected_pins={'cpu-history-expected.json':'e3c8b4be09558e5f9d182c4e2264d5f51a2497457546020d57fb6a855543b8fa789'}
# Explicit externally supplied pins; the context value below corrects no input artifact.
expected_pins['cpu-history-expected.json']='e3c8b4be09558e5f9d182c4e2264d5f51a2497457546020d57a855543b8fa789'
expected_pins.update({'cpu-history-preflight.json':'ab9d0ac13e2cbb17cc7c32c9ac0ba07a6e6a988e890402c02757807a519f1421','assembly-record.json':'108fbf4996ad35f624cb75e89474d68cb595e93b9c3b3e15b8bff4701419a634','assemble_cpu_preflight.py':'eb5894f449c30d3da5c57e4772117cd7f63835822cbd00484506d598e89cd6c7','source-review.md':'9c486e51ed6db5fc5cdfc27af74e9a3926556dc33edffbf09fbae19677b4b52d','llm_intervention_cpu_preflight_receipt.py':'b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8'})
for n,h in expected_pins.items():check('output/source pin '+n,sha((O/n).read_bytes())==h,sha256=sha((O/n).read_bytes()))
e=decode((O/'cpu-history-expected.json').read_bytes());t=decode((O/'cpu-history-preflight.json').read_bytes());a=decode((O/'assembly-record.json').read_bytes());k=decode((O/'checker-result.json').read_bytes());inputs=decode((O/'input-pins.json').read_bytes());ctx=e['execution_context']
data={};input_checks={}
for role,rec in inputs.items():
 b=Path(rec['path']).read_bytes();h=sha(b);input_checks[role]=dict(bytes=len(b),sha256=h,matches=h==rec['sha256'] and len(b)==rec['bytes']);data[role]=decode(b)
check('all twelve independently supplied input bytes',len(inputs)==12 and all(x['matches'] for x in input_checks.values()),inputs=input_checks)
check('assembly record binds all twelve original pins',a['input_pins']=={n:r['sha256'] for n,r in inputs.items()})
auth=data['authority'];fail=data['failure_acceptance'];proof=data['failure_proof'];base=data['baseline'];cpu=data['cpu_acceptance'];gp=data['original_gpu_packet'];cp=data['original_cpu_packet'];rs=data['readset'];backup=data['remote_custody'];seal=data['custody']
# Read literal field sets only; never import or run the supplied checker.
constants={}
for n in ast.parse((O/'llm_intervention_cpu_preflight_receipt.py').read_bytes()).body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_KEYS'):
  constants[n.targets[0].id]=ast.literal_eval(n.value)
check('exact concrete context and receipt field sets',set(e)==constants['EXPECTED_KEYS'] and set(t)==constants['RECEIPT_KEYS'] and set(ctx)==constants['CONTEXT_KEYS'] and set(ctx['lineage'])==constants['LINEAGE_KEYS'] and set(ctx['proof_pins'])==constants['PROOF_KEYS'] and set(ctx['cpu_roots'])==constants['ROOT_KEYS'])
check('exact metadata schemas',e['schema']=='ace-gpu-cpu-preflight-expected-v1' and ctx['schema']=='ace-gpu-cpu-preflight-context-v1' and t['schema']=='ace-gpu-cpu-preflight-receipt-v1' and t['phase']=='CPU-prelaunch-complete' and a['schema']=='ace-gpu-cpu-preflight-assembly-v1')
check('independent failure and accounting lineage reused',fail['status']=='accepted_failed_prepolicy_attempt' and fail['required_findings']==0 and fail['no_policy_work_proof_sha256']==inputs['failure_proof']['sha256'] and fail['input_pins']['original_failure_seal']['sha256']==inputs['custody']['sha256'] and fail['input_pins']['remote_custody']['sha256']==inputs['remote_custody']['sha256'] and base['input_pins']['acceptance']==inputs['failure_acceptance']['sha256'] and proof['before']['sha256']==inputs['before_observation']['sha256'] and proof['after']['sha256']==inputs['after_observation']['sha256'] and (base['terminal_jobs'],base['submission_journals'],base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'],base['outstanding_reservation'])==(47,96,30241,2750,0))
check('failure semantics never rewritten',fail['original_collector_returncode']==a['original_collector_returncode']==1 and a['retroactive_collector_success'] is False and all(type(proof[n]) is int and proof[n]==0 for n in ['new_policy_attempts','new_scientific_responses','new_training_updates','new_world_realizations']))
obs=[]
for role in ['before_observation','after_observation']:
 rec=data[role];v=decode(rec['stdout'].encode());obs.append(v)
 check(role+' transport success',rec['returncode']==0)
check('seal contains exact original observations and backup receipt in both copies',len(seal['copies'])==2 and all(all(inv['files'][n]['sha256']==inputs[role]['sha256'] for n,role in [('inventory_transport.json','before_observation'),('inventory_after_transport.json','after_observation'),('remote-custody.json','remote_custody')]) and inv['files']['pull_terminal.py']['sha256']==auth['historical_verifier_sha256'] for inv in seal['copies'].values()))
before,after=obs
check('complete before after backup file maps equal',before['files']==after['files']==backup['full_inventory']['files'],files=len(before['files']))
check('complete unique directory rosters equal',all(type(v['directories']) is list and len(v['directories'])==len(set(v['directories'])) for v in [before,after,backup['full_inventory']]) and sorted(before['directories'])==sorted(after['directories'])==backup['full_inventory']['directories'],directories=len(before['directories']))
backup_pin=sha((json.dumps(backup['full_inventory'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
check('archive and independent readback inventory joins',backup['full_backup_verified'] is True and backup['full_local_copy'] is False and backup['acceptance'] is False and backup_pin==backup['original_before_inventory_sha256']==backup['original_after_inventory_sha256']==backup['project_readback_inventory_sha256'] and a['archive']=={n:backup[n] for n in ['backup_archive_bytes','backup_archive_path','backup_archive_sha256']},full_inventory_sha256=backup_pin)
# Invert each proposed subset member's absolute location into the complete sealed tree.
root_alias=backup['roots']['cpu'];cpu_root=ctx['cpu_roots']['cpu_output'];campaign_root=ctx['cpu_roots']['campaign'];rel=str(PurePosixPath(cpu_root).relative_to(PurePosixPath(root_alias)));prefix='cpu/'+rel+'/'
cprefix=prefix+'worker/artifacts/campaign/'
full=before['files'];dirs=before['directories']
derived_campaign={'files':{n[len(cprefix):]:v for n,v in full.items() if n.startswith(cprefix)},'directories':sorted(n[len(cprefix):] for n in dirs if n.startswith(cprefix))}
derived_cpu={'files':{n[len(prefix):]:v for n,v in full.items() if n.startswith(prefix) and not n.startswith(cprefix)},'directories':sorted(n[len(prefix):] for n in dirs if n.startswith(prefix) and n!=cprefix[:-1] and not n.startswith(cprefix))}
check('full CPU output and campaign partition exact',e['cpu_inventories']=={'campaign':derived_campaign,'cpu_output':derived_cpu},prefix=prefix,campaign_prefix=cprefix,counts={n:dict(files=len(v['files']),directories=len(v['directories']),bytes=sum(x['bytes'] for x in v['files'].values())) for n,v in e['cpu_inventories'].items()})
check('both full hash maps exactly match independent CPU acceptance',all({n:v['sha256'] for n,v in e['cpu_inventories'][role]['files'].items()}==cpu[key] for role,key in [('cpu_output','cpu_output_files'),('campaign','campaign_files')]))
check('root bundle terminal and original packet bindings',ctx['cpu_roots']=={'cpu_output':cpu['cpu_output'],'campaign':cpu['campaign_root']} and cpu['cpu_output']==cp['cpu_output'] and cpu['cpu_packet_sha256']==gp['cpu_packet_sha256']==inputs['original_cpu_packet']['sha256'] and full['launch/inputs/packet.json']['sha256']==inputs['original_gpu_packet']['sha256'] and full['launch/inputs/cpu-packet.json']['sha256']==inputs['original_cpu_packet']['sha256'] and all(derived_cpu['files'][cpu[n]]['sha256']==cpu[n.replace('_path','_sha256')] for n in ['bundle_path','phase_terminal_path']))
roster={'campaign.json','source-plan-pins.json','start.json'}|{f'policy-{i:02d}/{n}' for i in range(5) for n in ['attempt.json','attempt.candidate.json']}|{f'policy-{i:02d}/world-{j:02d}/arm-receipt.json' for i in range(5) for j in range(24)}
readmap={n:{'bytes':v['bytes'],'sha256':v['sha256']} for n,v in rs['files'].items()}
check('all133 saved read-set byte metadata joins',set(readmap)==roster and e['consumed_read_set']==readmap and rs['cpu_acceptance_sha256']==inputs['cpu_acceptance']['sha256'] and rs['campaign_root']==campaign_root and all(derived_campaign['files'][n]==v and len(rs['files'][n]['utf8'].encode())==v['bytes'] and sha(rs['files'][n]['utf8'].encode())==v['sha256'] for n,v in readmap.items()),files=len(readmap),bytes=sum(v['bytes'] for v in readmap.values()))
check('original five accepted policy attempt pins',all(derived_campaign['files'][f'policy-{i:02d}/'+n]['sha256']==cpu['attempt_pins'][pid] for i,pid in enumerate(['uniform_random','menu_coverage','local_information','target_risk','learned_ranker']) for n in ['attempt.json','attempt.candidate.json']))
allmembers=set(full)|set(dirs);absent=['launch/gpu']+[cprefix+f'policy-{i:02d}/service' for i in range(5)]
check('saved exclusive output and CPU service absences',all(not any(n==p or n.startswith(p+'/') for n in allmembers) for p in absent) and t['absent_paths']==sorted([auth['allowed_new_output']]+[campaign_root+f'/policy-{i:02d}/service' for i in range(5)]))
lineage={'checker_sha256':'b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8','cpu_acceptance_sha256':inputs['cpu_acceptance']['sha256'],'failed_gpu01_acceptance_sha256':inputs['failure_acceptance']['sha256'],'namespace_sha256':auth['namespace_sha256'],'original_namespace_sha256':auth['original_namespace_sha256'],'recovery_authority_sha256':inputs['authority']['sha256'],'source_sha256':auth['historical_verifier_sha256']}
check('all context authority and namespace joins',ctx['lineage']==lineage and ctx['proof_pins']==auth['historical_proof_pins'] and ctx['control_root']==auth['control_root'] and ctx['failed_gpu01_root']==auth['failed_gpu01_root']==backup['roots']['launch'] and ctx['gpu_output']==auth['allowed_new_output']==ctx['failed_gpu01_root']+'/gpu' and ctx['baseline']==auth['baseline']=={'sha256':inputs['baseline']['sha256'],'terminal_jobs':47,'submission_journals':96} and full['namespace.json']['sha256']==auth['namespace_sha256'] and full['original-namespace.json']['sha256']==auth['original_namespace_sha256'])
objects={'source_files_sha256':auth['operational_source_files'],'model_files_sha256':cp['model_files'],'resources_sha256':gp['resources'],'limits_sha256':gp['limits']}
check('exact source model resource and limits digests',all(ctx[n]==digest(v) for n,v in objects.items()),digests={n:digest(v) for n,v in objects.items()},model_files=len(cp['model_files']))
full_object={'cpu_roots':ctx['cpu_roots'],'cpu_output_excluded_directory':'worker/artifacts/campaign','inventories':e['cpu_inventories']}
summary={'canonical_sha256':digest(full_object),'file_count':sum(len(v['files']) for v in e['cpu_inventories'].values()),'directory_count':sum(len(v['directories']) for v in e['cpu_inventories'].values()),'total_bytes':sum(x['bytes'] for v in e['cpu_inventories'].values() for x in v['files'].values())}
reads={'canonical_sha256':digest(readmap),'file_count':len(readmap),'total_bytes':sum(v['bytes'] for v in readmap.values())}
check('all computed context inventory and readset summaries',ctx['cpu_inventory_sha256']==summary['canonical_sha256'] and ctx['consumed_read_set_sha256']==reads['canonical_sha256'] and t['inventory_before']==t['inventory_after']==k['cpu_inventory']==summary and t['consumed_read_set']==k['consumed_read_set']==reads,inventory=summary,readset=reads)
contextpin=digest(ctx)
check('context digest and output byte pin joins',contextpin==e['execution_context_sha256']==t['execution_context_sha256']==a['execution_context_sha256']==k['execution_context_sha256'] and a['expected_sha256']==k['expected_sha256']==expected_pins['cpu-history-expected.json'] and a['receipt_sha256']==k['receipt_sha256']==expected_pins['cpu-history-preflight.json'],context_sha256=contextpin)
times={'before_scan':before['utc'],'before_transport':data['before_observation']['utc'],'after_scan':after['utc'],'after_transport':data['after_observation']['utc']}
check('saved timestamps not assembly time',a['saved_times']==times and t['started_utc']==times['before_scan'] and t['finished_utc']==times['after_transport'] and e['receipt_not_after_utc']==k['receipt_not_after_utc']==auth['receipt_not_after_utc'] and iso(times['before_scan'])<=iso(times['before_transport'])<=iso(times['after_scan'])<=iso(times['after_transport'])<=iso(auth['receipt_not_after_utc'])<=iso(a['assembled_utc']),times=times,authority_upper=auth['receipt_not_after_utc'],assembled_utc=a['assembled_utc'])
check('false qualification and historical-only scope',all(a[n] is False for n in ['acceptance','continuous_immutability_proven','fresh_remote_scan_performed','gpu_execution_qualified','scientific_qualified','retroactive_collector_success']) and all(k[n] is False for n in ['acceptance','continuous_immutability_proven','gpu_execution_qualified']) and a['full_external_postcustody_required'] is True and k['full_external_postcustody_required'] is True)
run=decode((P/'actual-preflight01/result.json').read_bytes());check('single saved metadata assembly result',run['returncode']==0 and iso(run['started_utc'])<=iso(a['assembled_utc'])<=iso(run['finished_utc']) and run['child_cpu_seconds']==5.809118 and abs(run['elapsed_seconds']-5.82741775)<1e-8,child_cpu_seconds=run['child_cpu_seconds'],elapsed_seconds=run['elapsed_seconds'],peak_child_rss_bytes=run['peak_child_rss_bytes'])
local=metadata_inventory(P/'actual-preflight01');external=metadata_inventory(Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-gpu-cpu-preflight-actual-20261109-01'))
check('both actual metadata copies identical',local==external,files=len(local),bytes=sum(v['bytes'] for v in local.values()))
declared=decode((P/'actual-preflight01/inventory.json').read_bytes());check('actual package inventory complete and correct',declared=={n:v for n,v in local.items() if n!='inventory.json'})
result={'scope':'independent pure saved-metadata comparison; no project calls','checks':checks,'required_findings':[x['check'] for x in checks if not x['passed']],'tests_run':0,'source_functions_called':0,'remote_actions':0,'actual_scientific_values_interpreted':False}
(W/'actual-metadata-checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'checks':len(checks),'passed':sum(x['passed'] for x in checks),'failures':result['required_findings'],'report':str(W/'actual-metadata-checks.json')},indent=2))
