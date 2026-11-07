"""Prepare one exclusive diagnostic packet from independently tested source bytes."""
from pathlib import Path
import datetime,hashlib,json,shlex,sys
repo=Path('/Users/pat/code/ACE')
local=Path(__file__).parent
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-runtime-metadata-20261107-01')
remote='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_runtime_metadata_20261107_01'
prior='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification/packed'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(name,raw):
 for root in [local,external]:
  with (root/name).open('xb') as f:f.write(raw)
def j(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
source=(repo/'scripts/research/qualify_llm_intervention_runtime_metadata.py').read_bytes()
assert len(sys.argv)==2 and sha(source)==sys.argv[1],'independent tested source pin required'
save('qualify_llm_intervention_runtime_metadata.py',source)
pins={'self':sha(source),'helper':sha((local/'llm_intervention_runtime_archive.py').read_bytes()),'probe':sha((local/'probe.py').read_bytes())}
resources={'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal','cpus':1,'memory_gib':3,'wall_seconds':600,'gpus':0,'requeue':False}
limits={'supervisor_seconds':580,'child_success_seconds':180,'child_outer_seconds':185,'cleanup_seconds':5,'stage_peak_rss_bytes':512*1024**2,'worker_peak_rss_bytes':1536*1024**2}
diag={'version':'ace-runtime-metadata-diagnostic-v1','sources':{k:{'path':remote+'/'+n,'sha256':pins[k]} for k,n in [('self','qualify_llm_intervention_runtime_metadata.py'),('helper','llm_intervention_runtime_archive.py'),('probe','probe.py')]},'archive':{'path':prior+'/runtime.tar','sha256':'6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98'},'manifest':{'path':prior+'/manifest.json','sha256':'b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a'},'runtime_source':'/projects/paco0228/miniconda3/envs/ace','output':remote+'/qualification','resources':resources,'limits':limits,'actual_versions':{'python':'3.10.19','numpy':'2.2.5','scipy':'1.15.3','torch':'2.9.1+cu128','transformers':'4.57.3'},'numpy_metadata_versions':['2.2.5','2.2.6']}
draw=j(diag);save('diagnostic.json',draw)
bootstrap='import hashlib,sys\np=sys.argv[1]; raw=open(p,"rb").read(); expected=sys.argv[2]\nif hashlib.sha256(raw).hexdigest()!=expected: raise ValueError("diagnostic source pin differs")\nsys.argv=[p]+sys.argv[3:];exec(compile(raw,p,"exec"),{"__name__":"__main__","__file__":p})'
cmd=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',bootstrap,diag['sources']['self']['path'],pins['self'],'--packet',remote+'/diagnostic.json','--packet-sha256',sha(draw),'--self-sha256',pins['self'],'--helper-sha256',pins['helper'],'--probe-sha256',pins['probe']]
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_meta01','#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=1','#SBATCH --mem=3G','#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+remote+'/logs/%j.out','#SBATCH --error='+remote+'/logs/%j.err','set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=','exec '+shlex.join(cmd),''])
save('job.sbatch',script.encode())
packet={'schema':'ace_runtime_metadata_diagnostic_launch_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':remote,'new_input_files':{n:sha((local/n).read_bytes()) for n in ['qualify_llm_intervention_runtime_metadata.py','llm_intervention_runtime_archive.py','probe.py','diagnostic.json','job.sbatch']},'resources':resources,'limits':limits,'admission':{'closed_actual_cpu_seconds':6215,'closed_actual_gpu_seconds':836,'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':600,'total_cpu_seconds':10415,'total_gpu_seconds':836,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},'predecessor_job':'35343957','predecessor_results_acceptance_sha256':'3b364ccb0360bf4b5edc392e697008e6afcb38901605642e668efbb2711bbe8f','runtime_archive':diag['archive'],'runtime_manifest':diag['manifest'],'purpose':'Observe unchanged copied runtime metadata, actual package versions/origins and deterministic arithmetic. Not model service or scientific qualification.','source_environment_changed':False,'installs':False,'metadata_or_cache_removed':False,'scientific_generation':False,'automatic_gpu_retry':False,'sizing':'Prior extraction/verification35.303541seconds, CPU import workerRSS700504KiB. Stage512MiB and worker1536MiB are conservative diagnostic process limits within3GiB allocation, not measured peak predictions. Prior batch3,143,400KiB near cap retained, cgroup/file-cache attribution unknown. Ten minute cap covers stage and original180second child target with cleanup.'}
save('packet.json',j(packet))
print(json.dumps({'packet_sha256':sha(j(packet)),'diagnostic_sha256':sha(draw),'source_sha256':pins['self']},indent=2))
