"""Prepare a new reviewed GPU-service packet from accepted CPU inputs. No models."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,subprocess,sys
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_model_service_gpu_20261107/launch13'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-model-service-gpu-20261107-13')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_model_service_gpu_20261107_13'
BASE=subprocess.check_output(['git','rev-parse','--verify',sys.argv[1]+'^{commit}'],cwd=REPO,text=True).strip()
assert len(BASE)==40 and all(c in '0123456789abcdef' for c in BASE)
CPU=REPO/'results/ace_llm_intervention_service_cpu_20261107/launch03'
ACCEPT=REPO/'results/ace_llm_intervention_service_cpu_20261107/acceptance03/acceptance.json'
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
assert sha(data['cpu-acceptance.json'])=='996be82b0387f9efe82a471a7ff735eee44ee5929c0e7ae1906d5eb3666d1f76'
packet={'schema':gpu.VERSION,'services':['frozen.json','adapter.json'],'resources':gpu.RESOURCES,'files':{n:sha(v) for n,v in sorted(data.items())}}
gpu.packet_shape(packet,cpu.INPUTS);gpu.check_preflight(json.loads(data['cpu-acceptance.json']),packet,preflight,cpu)
for n,v in data.items():write(n,v)
write('integration.json',encoded(packet))
command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',d.BOOTSTRAP,
 ROOT+'/qualify_llm_intervention_model_service.py',sha(data['qualify_llm_intervention_model_service.py']),
 '--packet',ROOT+'/integration.json','--packet-sha256',sha(encoded(packet)),'--output',ROOT+'/qualification']
script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_service13','#SBATCH --account=ucb736_asc2',
 '#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task=4','#SBATCH --mem=12G','#SBATCH --gres=gpu:a100_3g.20gb:1',
 '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
 '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
 'exec '+' '.join(map(shlex.quote,command)),'']).encode()
write('job.sbatch',script)
outer={'schema':'ace_service_gpu_launch_v4','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_commit':BASE,'resources':gpu.RESOURCES,'remote_root':ROOT,
 'new_input_files':{**packet['files'],'integration.json':sha(encoded(packet)),'job.sbatch':sha(script)},
 'predecessor_cpu_job':'35350736','cpu_acceptance_sha256':sha(data['cpu-acceptance.json']),
 'threading_diagnostic_job':'35349555', 'failed_predecessor_gpu_job':'35349776', 'threading_diagnostic_acceptance_sha256':sha((REPO/'results/ace_llm_intervention_threading_20261107/acceptance01/acceptance.json').read_bytes()), 'implementation_preparation_sha256':sha((REPO/'results/ace_llm_intervention_service_threading_preparation_20261107/preparation01/preparation.json').read_bytes()),
 'mapping_integration_preparation_sha256':'f107b1f788dd44807afbc77683239519dfa7219c568b73bd22a962f4f431aa41',
 'runtime_archive':preflight['archive'],'runtime_manifest':preflight['manifest'],
 'admission':{'closed_actual_cpu_seconds':7278,'closed_actual_gpu_seconds':941,
 'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':4800,'candidate_gpu_reservation':600,
 'total_cpu_seconds':15678,'total_gpu_seconds':1541,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
 'sizing':'Same measuredGPU09 single20GBslice:4.0168GBallocated/4.8444GBreserved,independent4900MiBownedprocess,126seconds. GPU12loadedmodel in11.606childseconds,1.815GBallocated and3302MiBownedprocess before mappingguardfailure. CPU03completed26-input16-sourceclosure,58.590secondnode-localstage,103secondwholeCPUjob. Requests4CPU/max8,12GiB600seconds,240secondfirstverifiedgeneration deadline,560operational/580cleanup absolute limits. GPU12memoryresidency is not inference qualification. Newdeviceclassification preservesrawmaps; no broadpath exemption. No speedup or memoryheadroom claim. Archivepacking/tokenizationdoneonCPU; node-localstagecannotreuse anothernode scratch.',
 'purpose':'Two persistent saved-policy sessions, each four artificial public requests with separated repeat. No training or environment callbacks.',
 'new_training_updates':0,'new_environment_responses':0,'scientific_worlds':0,'installs':False,
 'source_environment_changed':False,'static_packet_authentication_passed':True,'actual_gpu_qualification':False}
write('packet.json',encoded(outer))
with (EXT/'prepare_packet.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print(json.dumps({'packet_sha256':sha(encoded(outer)),'integration_sha256':sha(encoded(packet)),'inputs':len(data),'transfer_files':len(outer['new_input_files'])+1,'submitted':False}))
