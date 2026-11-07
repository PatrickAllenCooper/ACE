"""Prepare one new CPU service preflight packet. No runtime/model execution."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch02'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-service-cpu-20261107-02')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_02'
BASE='cddb7eb73da0ec110f13412e1587cfa8ed271147'
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
assert set(data)==cpu.INPUTS and len(data)==25
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
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_servicecpu02','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=1','#SBATCH --mem=3840M','#SBATCH --time=00:10:00','#SBATCH --no-requeue',
 '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_service_cpu_launch_v2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':cpu.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'preflight.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_diagnostic_job':'35347001',
 'predecessor_acceptance_sha256':'6e557150edb12f0db45c3b0baa9c72eeb63d763af769c062d3c8a556b633f578',
 'source_preparation_receipt_sha256':'74a3553fc5b5e1ab4ecc5dd74f490c36398015d00406baffbecd39cb7d7c214e',
 'resource_correction_receipt_sha256':'ac9bae87445df65375df55b0adfa1151b3d14bb95cc09309a4cf24e273b1635b',
 'predecessor_failed_job':'35348012',
 'runtime_archive':packet['archive'],'runtime_manifest':packet['manifest'],
 'admission':{'closed_actual_cpu_seconds':6384,'closed_actual_gpu_seconds':836,
   'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':600,
   'total_cpu_seconds':10584,'total_gpu_seconds':836,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'purpose':'Authenticate copied-runtime service imports and tokenize all three public artificial fixtures in frozen and saved-adapter configurations. No model weight loading or inference.',
 'sizing':'Accepted diagnostic used79allocatedCPU seconds,49.671118seconds staging,17.700921worker seconds,706961408workerRSS bytes. BatchMaxRSS3143420KiB was near3GiB, and live acpu MaxMemPerCPU3840MiB caused4GiB to allocate2CPUs. This corrected one-CPU preflight requests3840MiB. Actual service memory sufficiency remains unproven.580second absolute supervisor includes one stage and both fixed180/185second children plus cleanup. No GPU.',
 'model_weights_loaded':False,'adapter_deserialized':False,'scientific_generation':False,'installs':False,
 'source_environment_changed':False,'automatic_gpu_retry':False,
 'static_packet_authentication_passed':True,'actual_runtime_qualification':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'preflight_sha256':sha(encoded(packet)),
 'inputs':len(data),'transfer_inputs_including_outer':len(outer['new_input_files'])+1,'submitted':False},indent=2))
