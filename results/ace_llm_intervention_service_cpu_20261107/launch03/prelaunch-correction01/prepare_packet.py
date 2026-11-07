"""Prepare one new CPU service preflight packet. No runtime/model execution."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch03'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-service-cpu-20261107-03')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_03'
BASE='8f3c0c5ad2321bc1f0d8f4f3b4239b085ae56ff3'
OLD=REPO/'results/ace_llm_intervention_model_service_gpu_20261104/launch10/input'
PREVIOUS=REPO/'results/ace_llm_intervention_runtime_metadata_20261107/launch02'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(name,raw):
 for root in (EXT,LOCAL):
  with (root/name).open('xb') as f:f.write(raw)
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def module(name):
 path=REPO/'scripts/research'/name
 raw=path.read_bytes()
 assert subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)==raw
 spec=importlib.util.spec_from_file_location('prepare_'+path.stem,path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
service=module('llm_intervention_model_service.py')
cpu=module('qualify_llm_intervention_service_cpu.py')
diagnostic=module('qualify_llm_intervention_runtime_metadata.py')
data={}
for name in sorted(service.SOURCE_NAMES|{'qualify_llm_intervention_service_cpu.py','llm_intervention_runtime_archive.py'}):
 path=REPO/'scripts/research'/name
 data[name]=path.read_bytes()
 assert subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)==data[name]
data['probe.py']=(PREVIOUS/'probe.py').read_bytes()
assert sha(data['probe.py'])==diagnostic.PROBE_SHA
assert sha(data['llm_intervention_runtime_archive.py'])==diagnostic.HELPER_SHA
old_packet=json.loads((OLD/'packet.json').read_bytes())
for name in sorted(cpu.FIXTURES|{'trained_adapter.pt','gpu09_terminal.json'}):
 data[name]=(OLD/name).read_bytes()
 assert sha(data[name])==old_packet['files'][name]
for name in cpu.SERVICES:
 f=json.loads((OLD/name).read_bytes())
 assert f['recipe']==service.RECIPE and f['scope']=='artificial'
 f.update(schema=service.VERSION,runtime=service.RUNTIME,
          sources={n:sha(data[n]) for n in sorted(service.SOURCE_NAMES)})
 service.validate_freeze(f)
 data[name]=encoded(f)
assert set(data)==cpu.INPUTS and len(data)==26
prior=json.loads((PREVIOUS/'packet.json').read_bytes())
packet={'schema':cpu.VERSION,'files':{n:sha(raw) for n,raw in sorted(data.items())},
 'services':cpu.SERVICES,'resources':cpu.RESOURCES,'archive':prior['runtime_archive'],
 'manifest':prior['runtime_manifest'],'runtime_source':diagnostic.RUNTIME_SOURCE}
cpu.authenticate(LOCAL,packet,data,service,diagnostic)
for name,raw in data.items():write(name,raw)
write('preflight.json',encoded(packet))
command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',diagnostic.BOOTSTRAP,
 ROOT+'/qualify_llm_intervention_service_cpu.py',sha(data['qualify_llm_intervention_service_cpu.py']),
 '--packet',ROOT+'/preflight.json','--packet-sha256',sha(encoded(packet)),'--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_servicecpu03','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=2','#SBATCH --mem=6G','#SBATCH --time=00:10:00','#SBATCH --no-requeue',
 '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_service_cpu_launch_v3','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':cpu.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'preflight.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_diagnostic_job':'35347001',
 'predecessor_acceptance_sha256':'6e557150edb12f0db45c3b0baa9c72eeb63d763af769c062d3c8a556b633f578',
 'source_preparation_receipt_sha256':'f107b1f788dd44807afbc77683239519dfa7219c568b73bd22a962f4f431aa41',
 'predecessor_cpu_job':'35349416','predecessor_cpu_acceptance_sha256':'f4f0fea9f5c3d77720d365256583f7cebd7aa823777db11a1f98f32398df22f4',
 'predecessor_failed_gpu_job':'35349776',
 'runtime_archive':packet['archive'],'runtime_manifest':packet['manifest'],
 'admission':{'closed_actual_cpu_seconds':7072,'closed_actual_gpu_seconds':941,
   'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':1200,
   'total_cpu_seconds':11872,'total_gpu_seconds':941,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'purpose':'Authenticate copied-runtime service imports and tokenize all three public artificial fixtures in frozen and saved-adapter configurations. No model weight loading or inference.',
 'sizing':'The accepted two-CPU six-GiB threading diagnostic35349555 used91seconds. Prior service preflight35348366 used86seconds ononeCPU and near-limit3840MiB batchRSS. Request2CPU6GiB600seconds for26-input revisedclosure,16servicesources.580second supervisor includes one archived-runtime stage and two180/185second children. Observed batchRSS is near sixGiB and does not establish memoryheadroom. NoGPU or weights loaded.',
 'model_weights_loaded':False,'adapter_deserialized':False,'scientific_generation':False,'installs':False,
 'source_environment_changed':False,'automatic_gpu_retry':False,
 'static_packet_authentication_passed':True,'actual_runtime_qualification':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'preflight_sha256':sha(encoded(packet)),
 'inputs':len(data),'transfer_inputs_including_outer':len(outer['new_input_files'])+1,'submitted':False},indent=2))
