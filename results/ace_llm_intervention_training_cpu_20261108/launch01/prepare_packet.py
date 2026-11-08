"""Prepare a new bounded actual-tokenizer packet from committed sources."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_training_cpu_20261108/launch01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-cpu-20261108-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_training_cpu_20261108_01'
BASE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
COLLECTION=REPO/'results/ace_llm_intervention_training_collection_preparation_20261107/attempt02/fixtures/test_two_world_complete_inventory_drives_exposure_and_exclusive_intent/collection'
PREVIOUS=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch04'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(n,raw):
 for root in (EXT,LOCAL):
  with (root/n).open('xb') as f:f.write(raw)
def module(name):
 path=REPO/'scripts/research'/name
 raw=path.read_bytes()
 assert subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)==raw
 spec=importlib.util.spec_from_file_location('packet_'+path.stem,path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
p=module('prepare_llm_intervention_training_cpu.py');s=module('llm_intervention_model_service.py')
c=module('qualify_llm_intervention_service_cpu.py');d=module('qualify_llm_intervention_runtime_metadata.py')
data={}
for n in s.SOURCE_NAMES|p.EXTRA_SOURCES|{'llm_intervention_runtime_archive.py'}:
 path=REPO/'scripts/research'/n;data[n]=path.read_bytes()
 assert subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(REPO))],cwd=REPO)==data[n]
data['expectations.json']=(REPO/'scripts/research/fixtures/llm_intervention_training_cpu_expectations.json').read_bytes()
assert sha(data['expectations.json'])==p.EXPECTATIONS_SHA
assert subprocess.check_output(['git','show',BASE+':scripts/research/fixtures/llm_intervention_training_cpu_expectations.json'],cwd=REPO)==data['expectations.json']
data['probe.py']=(PREVIOUS/'probe.py').read_bytes()
for n,old in [('collection-terminal.json','terminal.json'),('collection-inventory.json','inventory.json'),('collection-plan.json','plan.json')]:
 data[n]=(COLLECTION/old).read_bytes();assert sha(data[n])==p.COLLECTION_PINS[n]
i=json.loads(data['collection-inventory.json'])
for ordinal,row in enumerate(i['records']):
 for kind in ['record','receipt']:data[f'{kind}-{ordinal}.json']=(COLLECTION/row['path']/(kind+'.json')).read_bytes()
prior=json.loads((PREVIOUS/'preflight.json').read_bytes());model=json.loads((PREVIOUS/'frozen.json').read_bytes())
packet={'schema':p.VERSION,'files':{n:sha(b) for n,b in sorted(data.items())},'source_names':sorted(s.SOURCE_NAMES|p.EXTRA_SOURCES),
 'scope':'artificial','record_roster':list(p.ROSTER),'archive':prior['archive'],'manifest':prior['manifest'],
 'runtime_source':d.RUNTIME_SOURCE,'resources':c.RESOURCES,'runtime':s.RUNTIME,
 'model_directory':model['model_directory'],'model_files':model['model_files']}
p.packet_shape(packet,s,c,d)
EXT.mkdir(mode=0o700,exist_ok=False)
for n,b in data.items():write(n,b)
write('preparation.json',encoded(packet))
command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',d.BOOTSTRAP,
 ROOT+'/'+p.ENTRY,sha(data[p.ENTRY]),'--packet',ROOT+'/preparation.json','--packet-sha256',sha(encoded(packet)),
 '--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_traincpu01','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=2','#SBATCH --mem=6G','#SBATCH --time=00:10:00','#SBATCH --no-requeue',
 '#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=',
 'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_training_cpu_launch_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':c.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'preparation.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_cpu_job':'35353861','predecessor_cpu_acceptance_sha256':'110b44c20a08251501326c74fda9eff7edbcace18118a37be3257e67d5579e14',
 'predecessor_gpu_job':'35372844','predecessor_gpu_acceptance_sha256':'e1fca5005c2a709964955f83b02b27a12374b77882325cc2ada94a532e747340',
 'runtime_archive':packet['archive'],'runtime_manifest':packet['manifest'],
 'admission':{'closed_actual_cpu_seconds':8032,'closed_actual_gpu_seconds':1091,
   'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':1200,
   'total_cpu_seconds':12832,'total_gpu_seconds':1091,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'purpose':'Real Qwen tokenizer prepares all four saved artificial records for four SFT and two DPO updates. No model loading, optimizer, CUDA, new responses or scientific worlds.',
 'sizing':'CPU04 used2CPU6GiB77seconds,39.082secondstage plus two fresh import/tokenizer sessions. This job stages once and imports one expanded 24-source closure, tokenizes four saved records and validates the complete plan. Same2CPU6GiB600secondcap,580supervisor and180/185child. Prior6288928KiB batchRSS was near6GiB, no headroom claim.',
 'model_weights_loaded':False,'new_environment_responses':0,'scientific_generation':False,'installs':False,
 'actual_runtime_qualification':False,'training_qualified':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'preparation_sha256':sha(encoded(packet)),
 'inputs':len(data),'sources':len(packet['source_names']),'transfer_inputs':len(outer['new_input_files'])+1,'submitted':False},indent=2))
