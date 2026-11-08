from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'predecessors/gpu-build_packet.py').read_text()
s=s[:s.index('def admit_cpu_bytes(')]
s=s.replace('Prospective GPU development campaign coordinator, byte-only CPU custody admission.', 'Prospective scientific finalizer coordinator, byte-only complete GPU custody admission.')
s=s.replace("LOCAL=REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch01'", "LOCAL=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'")
s=s.replace("EXT=BASE/'ACE-LLM-intervention-scientific-campaign-gpu-20261108-01'", "EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01'")
s=s.replace("ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'", "ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'\nGPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'")
s=s.replace("ENTRY='qualify_llm_intervention_scientific_campaign_gpu.py'", "ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'\nOLD_GPU_ENTRY='qualify_llm_intervention_campaign_gpu.py'")
s=s.replace("ENTRY_SHA='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'", "ENTRY_SHA='28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a'")
s=s.replace("VERSION='ace-scientific-paired-campaign-gpu-v1'", "VERSION='ace-scientific-campaign-finalizer-cpu-v1'")
s=s.replace("OUTER_VERSION='ace-scientific-campaign-gpu-launch-v1'", "OUTER_VERSION='ace-scientific-finalization-launch-v1'")
a=s.index('RESOURCES=dict(');z=s.index('POLICIES=',a)
s=s[:a]+"""RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,
 memory_mib=7680,wall_seconds=1200,gpus=0,requeue=False)
DEADLINES=dict(startup=120,work=1140,cleanup=1170,hard=1190)
GPU_RESOURCES=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,
 max_allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',requeue=False)
GPU_LIMITS=dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)
RAW_ROOTS=dict(launch=GPU_ROOT,cpu=CPU_ROOT,training=TRAIN_ROOT,token=TOKEN_ROOT,common=COLLECTION_ROOT,
 failed01='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01')
GPU_ACCEPTANCE_KEYS={'schema','status','scope','gpu_packet_sha256','cpu_packet_sha256','cpu_acceptance_sha256',
 'job_id','allocation','gpu_output','campaign_root','gpu_output_files','campaign_files','bundle_sha256',
 'attempt_pins','accepted_checks','terminal_pins','scientific_qualified'}
"""+s[z:]
s=s.replace("('cpu',CPU_ROOT,45,92,2,0,6000))", "('cpu',CPU_ROOT,45,92,2,0,6000),('gpu',GPU_ROOT,46,94,4,1,2880))")
s=s.replace("'site-readiness','cpu-review'}", "'site-readiness','cpu-review','gpu-review'}")
s=s.replace("'cpu-packet','cpu-phase-terminal')", "'cpu-packet','cpu-phase-terminal','gpu-review','gpu-packet','gpu-inventory-after','gpu-custody-inventory')")
s=s.replace("cpu='ace-scientific-cpu-campaign-launch-v1')", "cpu='ace-scientific-cpu-campaign-launch-v1',gpu='ace-scientific-campaign-gpu-launch-v1')")
s=s.replace('training=15360,cpu=7680)', 'training=15360,cpu=7680,gpu=15360)')
s=s.replace("s=stages['gpu_campaign']", "s=stages['finalize']")
s=s.replace("s['deadlines']==LIMITS", "s['deadlines']==DEADLINES")
s=s.replace("s['cpus']==4 and s['memory_mib']==15360 and s['wall_seconds']==2880 and s['gpus']==1 and s['gpu_type']=='a100_3g.20gb'", "s['cpus']==2 and s['memory_mib']==7680 and s['wall_seconds']==1200 and s['gpus']==0 and s['gpu_type'] is None")
s=s.replace("s['partition']=='aa100' and s['qos']=='gpu-testing'", "s['partition']=='acpu' and s['qos']=='cpu-normal'")
s=s.replace("'frozen GPU campaign profile differs'", "'frozen scientific finalizer profile differs'")
s=s.replace("base=m['cpu-baseline'];remaining=[stages['gpu_campaign'],stages['finalize']]", "base=m['gpu-baseline'];remaining=[stages['finalize']]\n    gpu_authority(m,pins)\n    # The GPU producer captures every predecessor metadata argument byte unchanged.\n    for name in tuple(FIXED)+('namespace-claim','cpu-review','cpu-packet','cpu-phase-terminal')+tuple(\n        prefix+'-'+kind for prefix,*_ in STAGES[:-1] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline')):\n        need(m['gpu-launch-packet']['new_input_files'][metadata_name(name)]==pins[name], 'original GPU lineage capture differs: '+name)")
(p/'build_packet.py').write_text(s)
