from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'predecessors/gpu-launch_once.py').read_text()
s=s.replace('scientific GPU campaign launcher; accepted CPU campaign', 'scientific finalizer launcher; accepted complete GPU campaign')
s=s.replace("packet['limits']==b.LIMITS", "packet['deadlines']==b.DEADLINES")
s=s.replace("b.read(local/(b.metadata_name(name)),\n            pins[b.metadata_name(name)])", "b.read(local/(b.metadata_name(name)),\n            pins[b.metadata_name(name)],256*1024**2)")
s=s.replace("cpu_data=b.admit_cpu_bytes(ctx,Path(packet['cpu_remote_custody']))", "gpu_data=b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])")
s=s.replace("ctx['pins']['cpu-baseline']", "ctx['pins']['gpu-baseline']")
s=s.replace("packet['cpu_job_id']==m['cpu-acceptance']['job_id']", "packet['gpu_job_id']==m['gpu-acceptance']['job_id'] and packet['gpu_acceptance_sha256']==ctx['pins']['gpu-acceptance']")
s=s.replace('b.make_packet(ctx,cpu_data,supplied)', 'b.make_packet(ctx,gpu_data,supplied)')
s=s.replace("expected,data,_=", "expected,data,bootstrap=")
s=s.replace("packet['gpu_packet_sha256']==b.sha(b.raw(expected))", "packet['finalizer_packet_sha256']==b.sha(b.raw(expected))")
s=s.replace("        b.need(b.read(b.LOCAL/'inputs/admission.json',pins['inputs/admission.json'])==b.raw(b.admission(ctx,stage)), 'stage admission differs')", """        b.need(b.read(local/'job.sbatch',pins['job.sbatch'])==b.job_script(ctx,stage,bootstrap),'exact finalizer job/bootstrap differs')
        b.need(packet['admission']==b.budget(m['gpu-baseline']) and packet['prior_jobs']==46 and packet['prior_journals']==94
               and packet['worlds']==24 and packet['cells']==192 and packet['comparisons']==28
               and packet['paths']==stage['paths'],'actual finalizer admission/reporting envelope differs')""")
s=s.replace("packet['original_cpu_inputs']", "packet['original_input_pins']")
a=s.index('        context=dict(');z=s.index("        common=",a)
s=s[:a]+"""        b.need(packet['upstream_rosters']==ctx['upstream_rosters'], 'complete original custody rosters differ')
        context=dict(root=b.ROOT,registry=b.REMOTE_REGISTRY,pins=pins,baseline=m['gpu-baseline'],
            archive=m['cpu-packet']['archive'],manifest=m['cpu-packet']['manifest'],namespace_claim_sha256=claim_pin,
            original_inputs=ctx['upstream_pins'],upstream_rosters=ctx['upstream_rosters'],
            original_registry=b.ORIGINAL_REMOTE_REGISTRY,original_namespace_claim_sha256=b.ORIGINAL_CLAIM_SHA)
"""+s[z:]
s=s.replace('timeout=55)', 'timeout=305)')
s=s.replace("ace_llm_intervention_scientific_campaign_gpu*", "ace_llm_intervention_scientific_finalization*")
s=s.replace('existing campaign GPU root', 'existing scientific finalization root')
s=s.replace('ACE scientific GPU campaign readiness', 'ACE scientific finalization readiness')
s=s.replace('>=4*1024**3', '>=10*1024**3')
s=s.replace("cpu_acceptance_sha256=ctx['pins']['cpu-acceptance'],", "gpu_acceptance_sha256=ctx['pins']['gpu-acceptance'],\n            gpu_disposition_sha256=ctx['pins']['gpu-disposition'],gpu_custody_inventory_sha256=ctx['pins']['gpu-custody-inventory'],")
s=s.replace("cpu_job_id=m['cpu-acceptance']['job_id']", "gpu_job_id=m['gpu-acceptance']['job_id']")
s=s.replace('resources=b.RESOURCES,limits=b.LIMITS', 'resources=b.RESOURCES,deadlines=b.DEADLINES')
s=s.replace("root/'gpu'", "root/'finalization'").replace('existing GPU campaign attempt','existing finalization attempt')
# Reuse CPU live account/partition/QoS branch, retain robust GPU predecessor cost parser.
a=s.index(" partition=run(");z=s.index(" manifest=",a)
cs=(p/'predecessors/cpu-launch_once.py').read_text();ca=cs.index(" partition=run(");cz=cs.index(" manifest=",ca)
s=s[:a]+cs[ca:cz].replace('6000','1200')+s[z:]
s=s.replace("signal.alarm(48)","signal.alarm(300)")
s=s.replace("+17520<=57600 and C['baseline']['closed_gpu_slice_seconds']+2880<=7200", "+6000<=57600 and C['baseline']['closed_gpu_slice_seconds']<=7200")
s=s.replace('full remaining GPU/finalizer caps', 'full finalizer cap')
a=s.index(" cpu_root=pathlib.Path(");z=s.index(" return dict(utc=utc(),queue=queue",a)
s=s[:a]+""" for original,expected in C['upstream_rosters'].items():
  tree=pathlib.Path(original)
  need(tree.is_dir() and tree==tree.resolve(),'original root alias/missing')
  members=list(tree.rglob('*'))
  need(len(members)<=262144 and not any(p.is_symlink() for p in members),'bounded ordinary complete custody required')
  need({str(p.relative_to(tree)) for p in members if p.is_file()}==set(expected['files'])
       and sorted(str(p.relative_to(tree)) for p in members if p.is_dir())==expected['directories'],
       'complete original file/directory roster changed')
  need(all(p.is_file() or p.is_dir() for p in members),'special upstream file forbidden')
 for path,h in C['original_inputs'].items():
  need(sha(read(path,allow_hardlink=True))==h,'original accepted byte changed: '+path)
"""+s[z:]
s=s.replace('qos=qos,gpu_types=gpu_types,model_files=model_verified,manifest_sha256=', 'qos=qos,manifest_sha256=')
s=s.replace("+15120,gpu_reserved_total=C['baseline']['closed_gpu_slice_seconds']+2880", "+6000,gpu_reserved_total=C['baseline']['closed_gpu_slice_seconds']")
(p/'launch_once.py').write_text(s)
# Hash-only monitor; no report/private payload parsing.
s=(p/'predecessors/gpu-monitor_once.py').read_text()
s=s.replace('scientific GPU campaign snapshot', 'scientific finalization snapshot')
s=s.replace("p['limits']==b.LIMITS", "p['deadlines']==b.DEADLINES")
a=s.index("paths=['gpu/");z=s.index("\n'''",a)
s=s[:a]+"""paths=['finalization/start.json','finalization/attempt.json','finalization/failure.json','finalization/allocation.json',
 'finalization/stage.json','finalization/handoff.json','finalization/worker.started.json','finalization/worker.cleanup.json',
 'finalization/worker.resources.json','finalization/worker/start.json','finalization/worker/work-start.json',
 'finalization/worker/runtime-before.json','finalization/worker/runtime-ready.json','finalization/worker/runtime-after.json',
 'finalization/worker/terminal.json','finalization/worker/attempt.json','finalization/worker/failure.json']
evidence={name:mark(root/name) for name in paths}
# Publication may precede a failing outer cleanup: presence is never acceptance or permission to retry.
campaign=Path(CAMPAIGN)
for name in ('finalization-intent.json','finalization.candidate.json','finalization.json'):
 evidence['campaign/'+name]=mark(campaign/name)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
 observations=observations,evidence=evidence,acceptance=False,scientific_qualified=False,
 note='Bounded opaque marker hashes only. Preserve failed/partial reports and all original campaign outputs; terminal custody and independent audit required.')))
"""+s[z:]
s=s.replace("'\\n'+REMOTE", "'\\nCAMPAIGN='+repr(b.CPU_ROOT+'/cpu/worker/artifacts/campaign')+'\\n'+REMOTE")
# Same two locks around the one read-only remote observation, no claim or submission.
s=s.replace("        run=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=50)",
"        with b.coordinator_locks():\n            run=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=50)")
s=s.replace('exact reviewed GPU campaign packet', 'exact reviewed scientific finalizer packet')
(p/'monitor_once.py').write_text(s)
