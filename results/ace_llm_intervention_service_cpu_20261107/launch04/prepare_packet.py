"""Prepare one new CPU service preflight packet. No runtime/model execution."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch04'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-service-cpu-20261107-04')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_service_cpu_20261107_04'
BASE='0a8d25f7842687f4b41d83145dfdeb19caee26ae'
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
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_servicecpu04','#SBATCH --account=ucb736_asc2',
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
 'source_preparation_receipt_sha256':'dc541c7b550975b3720f091f0f26100c1be5374253a7ca213e3bacfc2263fd28',
 'predecessor_cpu_job':'35350736','predecessor_cpu_acceptance_sha256':'996be82b0387f9efe82a471a7ff735eee44ee5929c0e7ae1906d5eb3666d1f76',
 'predecessor_failed_gpu_job':'35351262',
 'runtime_archive':packet['archive'],'runtime_manifest':packet['manifest'],
 'admission':{'closed_actual_cpu_seconds':7550,'closed_actual_gpu_seconds':1009,
   'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':1200,
   'total_cpu_seconds':12350,'total_gpu_seconds':1009,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'purpose':'Authenticate copied-runtime service imports and tokenize all three public artificial fixtures in frozen and saved-adapter configurations. No model weight loading or inference.',
 'sizing':'AcceptedCPU03job35350736 used2CPU6GiB103seconds,58.590secondnode-localstage andtwofreshfullservice/tokenization sessions. Requestsame2CPU6GiB600seconds fornew26-input16-sourceclosure.580secondabsolute supervisor and180/185secondchildren unchanged. BatchRSS6289300KiB wasnear6GiB, so no comfortableheadroomclaim. NoGPU/weights/adapterdeserialize/newresponses.',
 'model_weights_loaded':False,'adapter_deserialized':False,'scientific_generation':False,'installs':False,
 'source_environment_changed':False,'automatic_gpu_retry':False,
 'static_packet_authentication_passed':True,'actual_runtime_qualification':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'preflight_sha256':sha(encoded(packet)),
 'inputs':len(data),'transfer_inputs_including_outer':len(outer['new_input_files'])+1,'submitted':False},indent=2))
