"""Prepare one new committed-source GPU training packet, without submission."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_training_gpu_20261108/launch03'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-gpu-20261108-03')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_gpu_20261108_03'
BASE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
PREVIOUS=REPO/'results/ace_llm_intervention_training_cpu_20261108/launch01'
CPUROOT=REPO/'results/ace_llm_intervention_training_cpu_20261108'
def sha(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def write(n,b):
 for r in [EXT,LOCAL]:
  with (r/n).open('xb') as f:f.write(b)
def committed(path):
 raw=path.read_bytes()
 if subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)!=raw:raise ValueError('uncommitted source '+str(path))
 return raw
entry=REPO/'scripts/research/qualify_llm_intervention_training_gpu.py'
spec=importlib.util.spec_from_file_location('packet_training_gpu',entry)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
gpu=g.module('packet_gpu',REPO/'scripts/research'/g.GPU,committed(REPO/'scripts/research'/g.GPU))
old=json.loads((PREVIOUS/'preparation.json').read_bytes())
data={n:(PREVIOUS/n).read_bytes() for n in old['files']}
assert all(sha(b)==old['files'][n] for n,b in data.items())
for n in g.EXTRAS:data[n]=committed(REPO/'scripts/research'/n)
data['cpu-preparation.json']=(PREVIOUS/'preparation.json').read_bytes()
data['cpu-acceptance.json']=committed(CPUROOT/'acceptance01/acceptance.json')
data['prepared.json']=committed(CPUROOT/'terminal01/remote/qualification/session/worker/artifacts/prepared.json')
packet={'schema':g.VERSION,'files':{n:sha(b) for n,b in sorted(data.items())},
 'source_names':sorted(set(old['source_names'])|g.EXTRAS),'resources':gpu.RESOURCES}
g.packet_shape(packet,old,gpu)
EXT.mkdir(mode=0o700,exist_ok=False)
for n,b in data.items():write(n,b)
write('training.json',encoded(packet))
d=g.module('packet_diagnostic',LOCAL/'qualify_llm_intervention_runtime_metadata.py',data['qualify_llm_intervention_runtime_metadata.py'])
command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',d.BOOTSTRAP,
 ROOT+'/'+g.ENTRY,sha(data[g.ENTRY]),'--packet',ROOT+'/training.json','--packet-sha256',sha(encoded(packet)),
 '--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_traingpu03','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=4','#SBATCH --mem=12G','#SBATCH --gres=gpu:a100_3g.20gb:1',
 '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
 '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8',
 'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_training_gpu_launch_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':gpu.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'training.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_cpu_job':'35374034','predecessor_cpu_acceptance_sha256':g.CPU_ACCEPTANCE_SHA,
 'prepared_file_sha256':g.PREPARED_SHA,'cpu_preparation_sha256':g.CPU_PACKET_SHA,
 'runtime_archive':old['archive'],'runtime_manifest':old['manifest'],
 'admission':{'closed_actual_cpu_seconds':8778,'closed_actual_gpu_seconds':1239,
  'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':4800,'candidate_gpu_reservation':600,
  'total_cpu_seconds':17178,'total_gpu_seconds':1839,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'purpose':'Fresh seed109 adapters, four SFT and two DPO updates on the accepted artificial token batch, independent checkpoint reload/inference and runtime/GPU/scheduler qualification. No new responses or scientific worlds.',
 'sizing':'GPU09:126seconds, six updates including7343tokencontext,4.017GB allocated/4.844GBreserved GPU peak. GPU14:82seconds copiedruntime integration. Newpromptmax586, one20GBslice4CPUs12GiB600seconds, startup240/worker540/parent550/final580seconds.',
 'predecessor_failed_job':'35378508','predecessor_outer_sha256':'793c5b5588865a196354849972799d892bf2dc74855245177c1598fe3c7ddd5d',
 'correction':'Explicit synchronized cuBLAS workspace release between model lifetimes, measured before and after with zero-memory gate retained. Singleton final scoring and paired SFT scoring retained. Training engine/backend/plan unchanged. Fresh adapters, never resumed.',
 'fresh_adapter_required':True,'scientific_generation':False,'new_environment_responses':0,'installs':False,
 'actual_runtime_qualification':False,'training_qualified':False}
write('packet.json',encoded(outer))
for n in ['prepare_packet.py','launch_once.py','monitor_once.py','pull_terminal.py']:
 with (EXT/n).open('xb') as f:f.write((LOCAL/n).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'training_sha256':sha(encoded(packet)),
 'inputs':len(data),'sources':len(packet['source_names']),'transfer_inputs':len(outer['new_input_files'])+1,'submitted':False},indent=2))
