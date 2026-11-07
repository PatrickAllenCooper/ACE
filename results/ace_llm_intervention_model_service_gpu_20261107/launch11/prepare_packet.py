"""Prepare a new reviewed GPU-service packet from accepted CPU inputs. No models."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_model_service_gpu_20261107/launch11'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-model-service-gpu-20261107-11')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_11'
BASE='355760fe227c58dae0bc9d875b1a9d90177371be'
CPU=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch02'
ACCEPT=REPO/'results/ace_llm_intervention_service_cpu_20261107/acceptance02/acceptance.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(name,raw):
 for root in (LOCAL,EXT):
  with (root/name).open('xb') as f:f.write(raw)
def load(name):
 p=REPO/'scripts/research'/name;raw=p.read_bytes()
 assert raw==subprocess.check_output(['git','show',BASE+':'+str(p.relative_to(REPO))],cwd=REPO)
 spec=importlib.util.spec_from_file_location('prepare_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
gpu=load('qualify_llm_intervention_model_service.py');cpu=load('qualify_llm_intervention_service_cpu.py');d=load('qualify_llm_intervention_runtime_metadata.py')
preflight=json.loads((CPU/'preflight.json').read_bytes());data={n:(CPU/n).read_bytes() for n in cpu.INPUTS}
assert all(sha(raw)==preflight['files'][n] for n,raw in data.items())
for name in ('qualify_llm_intervention_model_service.py','supervise_llm_intervention_checkpoint_gpu.py'):
 p=REPO/'scripts/research'/name;data[name]=p.read_bytes();assert data[name]==subprocess.check_output(['git','show',BASE+':'+str(p.relative_to(REPO))],cwd=REPO)
data['cpu-preflight.json']=(CPU/'preflight.json').read_bytes();data['cpu-acceptance.json']=ACCEPT.read_bytes()
assert sha(data['cpu-acceptance.json'])=='f4f0fea9f5c3d77720d365256583f7cebd7aa823777db11a1f98f32398df22f4'
packet={'schema':gpu.VERSION,'services':['frozen.json','adapter.json'],'resources':gpu.RESOURCES,'files':{n:sha(v) for n,v in sorted(data.items())}}
gpu.packet_shape(packet,cpu.INPUTS);gpu.check_preflight(json.loads(data['cpu-acceptance.json']),packet,preflight,cpu)
for n,v in data.items():write(n,v)
write('integration.json',encoded(packet))
command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',d.BOOTSTRAP,
 ROOT+'/qualify_llm_intervention_model_service.py',sha(data['qualify_llm_intervention_model_service.py']),
 '--packet',ROOT+'/integration.json','--packet-sha256',sha(encoded(packet)),'--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_service11','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=4','#SBATCH --mem=12G','#SBATCH --gres=gpu:a100_3g.20gb:1',
 '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
 '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_service_gpu_launch_v2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':gpu.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'integration.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_cpu_job':'35348366','cpu_acceptance_sha256':sha(data['cpu-acceptance.json']),
 'implementation_preparation_sha256':'a5cc35de698552208b20dba4c015ebf18a3a7e6add7f32139defc83bcb26234e',
 'runtime_archive':preflight['archive'],'runtime_manifest':preflight['manifest'],
 'admission':{'closed_actual_cpu_seconds':6470,'closed_actual_gpu_seconds':836,
 'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':4800,'candidate_gpu_reservation':600,
 'total_cpu_seconds':14870,'total_gpu_seconds':1436,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'sizing':'GPU09: one20GBslice, peak4.0168GBallocated/4.8444GBreserved and independent4900MiBownedprocess,126seconds total. CPUpreflight35348366:54.113seconds node-local stage, both full import/tokenization processes under13seconds. Host12GiB and fourrequested/maxeightactualCPUs retain GPU09 recipe.580second hard deadline includes all staging/cleanup, first ownedGPUgeneration within240seconds. Node-local archive extraction cannot reuse another node local scratch. No speedup guarantee.',
 'purpose':'Two persistent saved-policy sessions, each four artificial public requests with separated repeat. No training or environment callbacks.',
 'new_training_updates':0,'new_environment_responses':0,'scientific_worlds':0,'installs':False,
 'source_environment_changed':False,'static_packet_authentication_passed':True,'actual_gpu_qualification':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'integration_sha256':sha(encoded(packet)),'inputs':len(data),'transfer_files':len(outer['new_input_files'])+1,'submitted':False}))
