from pathlib import Path
import json,hashlib,shutil,subprocess,datetime,sys
repo=Path('/Users/pat/code/ACE');sys.path.insert(0,str(repo/'scripts/research'))
from qualify_llm_intervention_task_gpu import SOURCES,RECIPE,validate_freeze
base=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-task-gpu-20261102-05');dest=base/'input'
if list(dest.iterdir()):raise RuntimeError('exclusive inputs already populated')
fixtures=base.parent/'ACE-LLM-intervention-context-fixtures-20261102-01'
gpu04=repo/'results/ace_llm_intervention_gpu_runtime_20261102/gpu04'
cpu01=repo/'results/ace_llm_intervention_task_runtime_20261102/cpu01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((cpu01/'input/freeze.json').read_bytes())
for n in sorted(SOURCES):shutil.copyfile(repo/'scripts/research'/n,dest/n)
for n in ['case0.json','case2.json','case3.json','stress.json','profile.json','prepared_tokens.json']:shutil.copyfile(fixtures/n,dest/n)
for p,n in [(gpu04/'terminal/gpu04/worker/adapter.pt','accepted_adapter.pt'),(gpu04/'terminal/gpu04/worker/terminal.json','gpu04_terminal.json'),(gpu04/'qualification_acceptance.json','gpu04_acceptance.json'),(cpu01/'cpu_acceptance.json','cpu01_acceptance.json')]:shutil.copyfile(p,dest/n)
review=repo/'docs/development/guidance/reviews/ace_llm_intervention_task_gpu_review_2026-11-02.md'
shutil.copyfile(review,dest/'implementation_review.md')
assets={p.name:sha(p) for p in dest.iterdir() if p.suffix!='.py' and p.name!='implementation_review.md'}
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
freeze={'schema':'ace-shared-task-gpu-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':revision,'sources':{n:sha(dest/n) for n in sorted(SOURCES)},'model_files':prior['model_files'],'model_path':prior['model_path'],'model_revision':prior['model_revision'],'python_version':prior['python_version'],'runtime':prior['runtime'],'assets':assets,'recipe':RECIPE,'startup_seconds':420,'worker_seconds':840,'resources':{'account':'ucb736_asc1','partition':'aa100','qos':'gpu-testing','requested_cpus':4,'max_allocated_cpus':8,'memory_mib':12288,'wall_seconds':900,'gpus':1,'gpu_type':'a100_3g.20gb'},'scope':'new shared-task long-context artificial runtime qualification, no science or environment response','resource_basis':'smallest compatible A10020GB testing slice, prior actual base fit3718MiB NVIDIA process memory,3.30GB framework peak, new4341/7343prompt memory unknown bounded measured test','conservative_cpu_reservation_seconds':7200,'conservative_gpu_reservation_seconds':900,'cycle_reserved_cpu_seconds':37200,'cycle_reserved_gpu_seconds':3600,'cpu_ceiling_seconds':57600,'gpu_ceiling_seconds':7200}
validate_freeze(freeze)
(dest/'freeze.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n');pin=sha(dest/'freeze.json')
remote='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_task_gpu_20261102_05'
old=(gpu04/'input/gpu04.sbatch').read_text();old=old.replace('ace_llm_gpu04','ace_llm_tgpu05').replace('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_gpu_20261102_04',remote).replace('supervise_llm_intervention_gpu.py','supervise_llm_intervention_task_gpu.py').replace('ae98aa3627d8738eef31c2307a5f9731c2de643ec5825a697e15215ba70d077b',pin).replace('/gpu04','/gpu05')
(dest/'gpu05.sbatch').write_text(old)
(dest/'source-pins.sha256').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(dest.iterdir())))
manifest={'freeze_sha256':pin,'files':{p.name:sha(p) for p in sorted(dest.iterdir())},'remote_root':remote,'source_revision':revision}
(base/'input_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
shutil.copyfile('/tmp/ace_build_task_gpu05.py',base/'build_packet.py')
print(json.dumps({'freeze':pin,'source_revision':revision,'files':len(manifest['files']),'assets':len(assets)}))
