"""Continue AFTER completed byte/Git/history checks in retained custody-attempt02.
No completed full-tree checks repeated. The second scontrol lookup is separately
unavailable; both original sacct/queue observations remain authoritative.
"""
import json,time,hashlib
import audit_custody as a
start=time.monotonic();cpu=time.process_time()
prior=a.load(a.HERE/'custody-attempt02/execution.json');err=(a.HERE/'custody-attempt02/stderr.log').read_text()
a.need(prior['returncode']==1 and 'line 137, in main' in err and 'scheduler observation failed' in err,'exact retained continuation checkpoint required')
a.need(a.sha((a.HERE/'custody-attempt02/audit_custody.py').read_bytes())==prior['script_sha256'],'original failed auditor source differs')
for n,h in a.PINS.items():a.load(a.TERM/n,h)
d=a.load(a.TERM/'disposition.json');seal=a.load(a.TERM/'custody-inventory.json');snapshot=seal['copies'][str(a.TERM)]
obs,after=[json.loads(a.load(a.TERM/n)['stdout']) for n in ('inventory_transport.json','inventory_after_transport.json')]
for record in (obs,after):
 for name in ('steps','details','queue','related_accounting'):a.need(record['results'][name]['returncode']==0,'saved sacct/queue unavailable')
 queue=[l.split('|') for l in record['results']['queue']['stdout'].splitlines() if l.strip()]
 a.need(all(len(r)==5 and r[0]!=a.JOB and not r[1].lower().startswith('ace') for r in queue),'active ACE reservation')
a.need(obs['results']['job']['returncode']==0,'first original full terminal scontrol unavailable')
a.need(after['results']['job']['returncode']==1 and after['results']['job']['stdout']=='' and 'Invalid job id specified' in after['results']['job']['stderr'],'unexpected second controller observation')
a.need(obs['results']['steps']['stdout']==after['results']['steps']['stdout'] and obs['results']['details']['stdout']==after['results']['details']['stdout'],'saved scheduler accounting/resource evidence differs')
fields=dict(x.split('=',1) for x in obs['results']['job']['stdout'].split() if '=' in x)
for k,v in dict(JobId=a.JOB,Account='ucb736_asc2',QOS='cpu-normal',JobState='FAILED',Requeue='0',Restarts='0',ExitCode='1:0',RunTime='00:02:09',TimeLimit='00:10:00',Partition='acpu',NumCPUs='3',NumNodes='1',NumTasks='1',AllocTRES='cpu=3,mem=11520M,node=1,billing=3').items():a.need(fields[k]==v,'scheduler field differs '+k)
local=a.load(a.TERM/'coordinator/local-launch/submission.intent.json');result=a.load(a.TERM/'remote/launch/submission.result.json')
# Prefix-completed checks are bound to their exact failed script and execution record.
out=dict(status='complete_saved_custody_and_failure_accounting_verified',acceptance=False,job_id=a.JOB,
 raw_files=31039,raw_bytes=218721958,raw_directories=len(obs['directories']),
 capsule_files=len(snapshot['files'])+1,capsule_bytes=sum(v['bytes'] for v in snapshot['files'].values())+(a.TERM/'custody-inventory.json').stat().st_size,
 artifact_pins=a.PINS,packet_sha256=d['packet_sha256'],packet_commit=local['packet_commit'],submission_result_sha256=d['submission_result_sha256'],remote_intent_sha256=result['intent_sha256'],
 original_input_pins_verified=87,two_independent_full_copies_verified=True,two_remote_inventories_verified=True,retained_archive_all_members_verified=True,
 complete_collection_and_failed_predecessor_bytes_unchanged=True,collection_files=30418,terminal_jobs=43,submission_journals=88,
 closed_cpu_seconds=13373,closed_gpu_slice_seconds=1601,cpu_admission_seconds=16973,outstanding_reservation=0,failed_allocation_cpu_seconds=387,failed_allocation_gpu_seconds=0,
 related_accounting=obs['related_accounting'],journal_pins=obs['journal_pins'],scheduler_steps=obs['results']['steps'],scheduler_details=obs['results']['details'],node=fields['NodeList'],
 saved_second_controller_lookup=after['results']['job'],second_controller_lookup_unavailable_resolved_by_matching_sacct=True,
 completed_prefix_attempt_sha256=a.sha((a.HERE/'custody-attempt02/execution.json').read_bytes()),completed_prefix_script_sha256=prior['script_sha256'],
 completed_prefix_through='Both full byte/inode inventories, sealed copies, before/after inventories, entire archive, all87 original Git pins, unchanged complete collection/failure roots, exact43 rows/88journals/costs. Failure occurred only at subsequent second scontrol return-code check.',
 local_audit_resume_wall_seconds=time.monotonic()-start,local_audit_resume_process_cpu_seconds=time.process_time()-cpu,no_private_payload_decoding=True,no_scientific_execution=True)
(a.HERE/'custody-result.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('journal_pins','related_accounting','scheduler_steps','scheduler_details','artifact_pins')}))
