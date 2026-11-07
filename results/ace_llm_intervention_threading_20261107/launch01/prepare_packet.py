"""Freeze CPU-only threading observations. No runtime/model execution."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=Path(__file__).parent
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-threading-20261107-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_threading_20261107_01'
OLD=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch02'
BASE=sys.argv[1]
assert len(BASE)==40 and all(c in '0123456789abcdef' for c in BASE)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(name,raw):
 for root in (EXT,LOCAL):
  with (root/name).open('xb') as f:f.write(raw)
preflight=json.loads((OLD/'preflight.json').read_bytes())
data={n:(OLD/n).read_bytes() for n in preflight['files']}
assert all(sha(data[n])==h for n,h in preflight['files'].items())
data['cpu-preflight.json']=(OLD/'preflight.json').read_bytes()
assert sha(data['cpu-preflight.json'])=='f728963e1bedfce94646afacc7619671ba94237a7b34bbd6f3b624f43bde38e6'
for n in ('qualify_llm_intervention_threading.py','qualify_llm_intervention_model_service.py'):
 path=REPO/'scripts/research'/n;data[n]=path.read_bytes()
 assert data[n]==subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)
spec=importlib.util.spec_from_file_location('threading_packet',REPO/'scripts/research/qualify_llm_intervention_threading.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert sha(data['qualify_llm_intervention_service_cpu.py'])==m.CPU_SHA
assert sha(data['qualify_llm_intervention_model_service.py'])==m.GPU_SHA
assert len(data)==28
packet={'schema':m.VERSION,'files':{n:sha(b) for n,b in sorted(data.items())},'resources':m.RESOURCES}
EXT.mkdir()
for n,b in data.items():write(n,b)
write('diagnostic.json',encoded(packet))
dspec=importlib.util.spec_from_file_location('diagnostic_packet',OLD/'qualify_llm_intervention_runtime_metadata.py')
d=importlib.util.module_from_spec(dspec);dspec.loader.exec_module(d)
command=[d.RUNTIME_SOURCE+'/bin/python','-I','-B','-c',d.BOOTSTRAP,ROOT+'/qualify_llm_intervention_threading.py',sha(data['qualify_llm_intervention_threading.py']),'--packet',ROOT+'/diagnostic.json','--packet-sha256',sha(encoded(packet)),'--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_thread01','#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=6G','#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1','export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=','exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_threading_cpu_launch_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':BASE,'remote_root':ROOT,'resources':m.RESOURCES,'new_input_files':{**packet['files'],'diagnostic.json':sha(encoded(packet)),'job.sbatch':sha(script)},'runtime_archive':preflight['archive'],'runtime_manifest':preflight['manifest'],'predecessor_gpu_job':'35348887','predecessor_cpu_job':'35348366','predecessor_cpu_preflight_sha256':sha(data['cpu-preflight.json']),'admission':{'closed_actual_cpu_seconds':6670,'closed_actual_gpu_seconds':886,'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':1200,'total_cpu_seconds':11470,'total_gpu_seconds':886,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},'purpose':'CPU-only paired observation of inherited C threading versus explicit GNU in fresh torch-first children after original-runtime numerical parent imports. Preserve both outcomes. No automatic GPU retry.','sizing':'One prior CPU stage54seconds, entire service CPU preflight86seconds, near3.75GiB batchRSS. Two CPUs support6GiB under3840MiB per CPU. 580second absolute supervisor includes single archive stage and two185second parent caps plus cleanup. Each parent child90seconds. Numerical work single-threaded. NoGPU.','model_weights_loaded':False,'adapter_deserialized':False,'scientific_generation':False,'installs':False,'source_environment_changed':False,'model_service_qualified':False}
write('packet.json',encoded(outer))
for n in ('prepare_packet.py','launch_once.py','monitor_once.py','pull_terminal.py'):
 with (EXT/n).open('xb') as f:f.write((LOCAL/n).read_bytes())
print(json.dumps({'source_commit':BASE,'packet_sha256':sha(encoded(outer)),'diagnostic_sha256':sha(encoded(packet)),'input_count':len(data),'transfer_count':len(outer['new_input_files'])+1,'submitted':False},indent=2))
