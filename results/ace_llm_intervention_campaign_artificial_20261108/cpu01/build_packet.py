"""Construct one unsubmitted CPU packet from committed source and accepted bytes."""
from pathlib import Path
import hashlib,json,subprocess,types,shlex
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_campaign_artificial_20261108/cpu01'
EXT=BASE/'ACE-LLM-intervention-campaign-cpu-20261108-01'
REMOTE='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_01'
INPUT=REMOTE+'/inputs'; CPUOUT=REMOTE+'/cpu01'; GPUOUT=REMOTE+'/gpu01'
REV=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def git(name):return subprocess.check_output(['git','show',REV+':scripts/research/'+name],cwd=REPO)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def module(name,b):
 m=types.ModuleType(name);m.__file__=INPUT+'/'+name+'.py';exec(compile(b,m.__file__,'exec'),m.__dict__);return m
entry='prepare_llm_intervention_campaign_cpu.py'
cpu=module(entry[:-3],git(entry))
data={n:git(n) for n in cpu.SOURCE_NAMES}
prior=REPO/'results/ace_llm_intervention_training_gpu_20261108/launch03'
terminal=REPO/'results/ace_llm_intervention_training_gpu_20261108/terminal03/remote'
fit=terminal/'qualification/session/worker/artifacts/fit/training'
for n in ['probe.py','llm_intervention_runtime_archive.py','collection-plan.json','collection-inventory.json',
          'collection-terminal.json',*[f'{k}-{i}.json' for k in ('record','receipt') for i in range(4)]]:
 data[n]=(prior/n).read_bytes()
data['prepared.json']=(terminal/'prepared.json').read_bytes()
data['gpu03-acceptance.json']=(REPO/'results/ace_llm_intervention_training_gpu_20261108/acceptance03/acceptance.json').read_bytes()
data['engine-terminal.json']=(fit/'terminal.json').read_bytes()
for n in ['sft.adapter.pt','final.adapter.pt','sft.checkpoint.json','final.checkpoint.json']:data[n]=(fit/n).read_bytes()
service=module('llm_intervention_model_service',data[cpu.SERVICE])
diagnostic=module('qualify_llm_intervention_runtime_metadata',data[cpu.DIAGNOSTIC])
training=module('prepare_llm_intervention_training_cpu',data[cpu.TRAINING])
lineage=module('llm_intervention_campaign_training_lineage',data[cpu.LINEAGE]).admit_training(**cpu.training_inputs(data))
old=json.loads((prior/'cpu-preparation.json').read_bytes())
intent={'scope':'two supplied artificial worlds, eight fixed policies, CPU phase only',
 'source_commit':REV,'world_ids':[9101,9102],'policies':[ *cpu.CPU_POLICIES,*cpu.LLM_POLICIES],
 'gpu03_acceptance_sha256':sha(data['gpu03-acceptance.json']),
 'design_sha256':sha((REPO/'docs/development/guidance/ace_llm_intervention_paired_campaign_integration_2026-11-08.md').read_bytes()),
 'cpu_output':CPUOUT,'future_gpu_output':GPUOUT,'scientific_worlds_generated':False,'no_utility_claim':True}
packet={k:old[k] for k in ('archive','manifest','runtime_source','runtime','model_directory','model_files')}
packet.update(schema=cpu.VERSION,scope='artificial',record_roster=list(cpu.ROSTER),source_names=sorted(cpu.SOURCE_NAMES),
 resources=cpu.RESOURCES,cpu_output=CPUOUT,outer_provenance_receipt_sha256=sha(raw(intent)),persistent_specs={},services={})
for role in cpu.LLM_POLICIES:
 adapter=None
 if role!='frozen_llm':
  stage='sft' if role=='sft_llm' else 'final';r=lineage['roles'][role]
  adapter=dict(file=stage+'.adapter.pt',sha256=r['adapter_sha256'],training_receipt=stage+'.checkpoint.json',
      training_receipt_sha256=r['checkpoint_sha256'],training_scope='artificial')
 f=dict(schema=service.VERSION,scope='artificial',model_directory=old['model_directory'],model_files=old['model_files'],
     sources={n:sha(data[n]) for n in service.SOURCE_NAMES},runtime=service.RUNTIME,python_version=[3,10,19],
     recipe=service.RECIPE,treatment='frozen' if adapter is None else 'artificial_adapter',adapter=adapter,limits=cpu.LIMITS)
 service.validate_freeze(f);data[role+'.freeze.json']=raw(f)
 c=dict(schema='ace-campaign-staged-launch-v1',policy_id=role,entry=INPUT+'/'+cpu.SERVICE,entry_sha256=sha(data[cpu.SERVICE]),
     freeze=INPUT+'/'+role+'.freeze.json',freeze_sha256=sha(data[role+'.freeze.json']),
     worker_output=GPUOUT+'/services/'+role+'/worker',launch_output=GPUOUT+'/services/'+role+'/launch',
     binding_path=GPUOUT+'/stage-binding.json',stage_path=GPUOUT+'/stage.json',
     archive_sha256=diagnostic.ARCHIVE_SHA,manifest_sha256=diagnostic.MANIFEST_SHA,
     bootstrap=diagnostic.BOOTSTRAP,bootstrap_sha256=sha(diagnostic.BOOTSTRAP.encode()))
 data[role+'.launch.json']=raw(c);identity=service.identity(f)
 command=[old['runtime_source']+'/bin/python','-I','-B','-c',diagnostic.BOOTSTRAP,INPUT+'/'+cpu.LAUNCH,
     sha(data[cpu.LAUNCH]),'--contract',INPUT+'/'+role+'.launch.json','--contract-sha256',sha(data[role+'.launch.json'])]
 packet['persistent_specs'][role]=dict(policy_id=role,kind='persistent',command=command,timeout_seconds=35,
     identity_sha256=service.digest(identity),policy_provenance_sha256=sha(data[role+'.freeze.json']))
 packet['services'][role]=dict(identity=identity,startup_seconds=125,cleanup_seconds=5,max_requests=8)
packet['files']={n:sha(b) for n,b in sorted(data.items())}
cpu.packet_shape(packet,service,diagnostic,training)
cpu.admit(Path(INPUT),packet,data,service,diagnostic)
assert set(data)==cpu.INPUT_NAMES
data['packet.json']=raw(packet)
cmd=[old['runtime_source']+'/bin/python','-I','-B','-c',diagnostic.BOOTSTRAP,INPUT+'/'+entry,sha(data[entry]),
     '--packet',INPUT+'/packet.json','--packet-sha256',sha(data['packet.json']),'--output',CPUOUT]
job=('''#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_cpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output='''+REMOTE+'''/logs/%j.out
#SBATCH --error='''+REMOTE+'''/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec '''+shlex.join(cmd)+'\n').encode()
meta=dict(source_commit=REV,remote_root=REMOTE,input_directory=INPUT,cpu_output=CPUOUT,
 cpu_packet_sha256=sha(data['packet.json']),input_files={n:sha(b) for n,b in data.items()},job_sha256=sha(job),
 resources=cpu.RESOURCES,scope='artificial CPU integration, no GPU/private finalization',
 admission=dict(closed_cpu_seconds=9210,closed_gpu_slice_seconds=1347,cumulative_preparation_cpu_allowance=3600,
                new_cpu_reservation=1200,cpu_admission=14010,cpu_ceiling=57600,gpu_slice_ceiling=7200),
 expected_prior_jobs=31,expected_prior_submission_journals=64)
for dest in [EXT,LOCAL]:
 dest.mkdir(parents=True,exist_ok=False);(dest/'inputs').mkdir()
 for n,b in data.items():(dest/'inputs'/n).write_bytes(b)
 (dest/'job.sbatch').write_bytes(job);(dest/'packet-metadata.json').write_bytes(raw(meta))
 (dest/'intent-provenance.json').write_bytes(raw(intent));(dest/'build_packet.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'source_commit':REV,'files':len(data),'packet_sha256':sha(data['packet.json']),'local':str(LOCAL),'external':str(EXT)}))
