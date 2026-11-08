"""Build an exclusive frozen full-shape GPU qualification packet. Never submit."""
from pathlib import Path
import argparse, datetime, hashlib, json, shlex, subprocess, types

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_screen_training_gpu_20261108/launch01'
EXT = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-training-gpu-20261108-01')
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_training_gpu_20261108_01'
PREPARATION = REPO/'results/ace_llm_intervention_screen_training_shape_20261108/preparation01/artifact'
CAPACITY = REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108'
ENTRY = 'qualify_llm_intervention_screen_training_gpu.py'

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def module(name,path,raw):
    result=types.ModuleType(name); result.__file__=str(path)
    exec(compile(raw,str(path),'exec'),vars(result)); return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-commit','entry-sha256','worker-review','worker-review-sha256',
                'supervisor-review','supervisor-review-sha256','input-review','input-review-sha256'):
        p.add_argument('--'+key,required=True)
    args=p.parse_args()
    rev=args.source_commit
    need(len(rev)==40 and all(c in '0123456789abcdef' for c in rev), 'full reviewed commit required')
    def committed(path):
        return subprocess.check_output(['git','--no-replace-objects','show',rev+':'+path],cwd=REPO)
    source=committed('scripts/research/'+ENTRY)
    need(sha(source)==args.entry_sha256,'reviewed qualifier source differs')
    g=module('screen_qualification_build',REPO/'scripts/research'/ENTRY,source)
    data={name:committed('scripts/research/'+name) for name in g.SOURCE_NAMES}
    data['llm_intervention_runtime_archive.py']=committed('scripts/research/llm_intervention_runtime_archive.py')
    data['prepared.json']=(PREPARATION/'prepared.json').read_bytes()
    data['cpu-acceptance.json']=(CAPACITY/'acceptance01/acceptance.json').read_bytes()
    for name,source_name in (('capacity-packet.json','capacity-packet.json'),('tokenization.json','tokenization.json')):
        data[name]=(PREPARATION/source_name).read_bytes()
    need(all(sha(data[n])==pin for n,pin in g.INDEPENDENT.items()),'independent preparation/helper pins differ')
    capacity=json.loads(data['capacity-packet.json'])
    gpu=module('shape_gpu_build',REPO/'scripts/research'/g.GPU,data[g.GPU])
    service=module('shape_service_build',REPO/'scripts/research/llm_intervention_model_service.py',data['llm_intervention_model_service.py'])
    packet={**{k:capacity[k] for k in ('archive','manifest','model_directory','model_files','runtime','runtime_source')},
        'schema':g.VERSION,'scope':'artificial','source_commit':rev,'resources':gpu.RESOURCES,
        'files':{n:sha(raw) for n,raw in sorted(data.items())},'source_names':sorted(g.SOURCE_NAMES)}
    g.packet_shape(packet,gpu)
    prepared=json.loads(data['prepared.json'])
    g.validate_prepared(prepared,packet,data,service)
    worker=module('shape_worker_build',REPO/'scripts/research'/g.WORKER,data[g.WORKER])
    worker._plan(prepared,prepared_sha256=g.PREPARED_SHA,plan_sha256=g.PLAN_SHA)
    reviews={}
    for name,path,pin in (('worker-review.md',args.worker_review,args.worker_review_sha256),
                          ('supervisor-review.md',args.supervisor_review,args.supervisor_review_sha256),
                          ('input-review.md',args.input_review,args.input_review_sha256)):
        absolute=Path(path).absolute(); raw=committed(str(absolute.relative_to(REPO)))
        need(sha(raw)==pin and raw==absolute.read_bytes(),'review differs from committed independent pin')
        reviews[name]=raw
    LOCAL.mkdir(parents=True,mode=0o700,exist_ok=False)
    EXT.mkdir(mode=0o700,exist_ok=False)
    def write(name,raw):
        for root in (LOCAL,EXT):
            with (root/name).open('xb') as f:f.write(raw)
    write('build_packet.py',Path(__file__).read_bytes())
    for name,path in [('launch_once.py','/tmp/ace-launch-screen-training-gpu01-20261108.py'),
                      ('monitor_once.py','/tmp/ace-screen-training-gpu01-monitor_once.py')]:
        write(name,Path(path).read_bytes())
    for name,raw in {**data,**reviews}.items():write(name,raw)
    training_raw=encoded(packet); write('training.json',training_raw)
    d=module('shape_diagnostic_build',LOCAL/g.DIAGNOSTIC,data[g.DIAGNOSTIC])
    command=['/projects/paco0228/miniconda3/envs/ace/bin/python','-I','-B','-c',d.BOOTSTRAP,
             ROOT+'/'+ENTRY,sha(source),'--packet',ROOT+'/training.json',
             '--packet-sha256',sha(training_raw),'--output',ROOT+'/qualification']
    script='\n'.join(['#!/bin/bash','#SBATCH --job-name=ace_llm_shape01','#SBATCH --account=ucb736_asc2',
        '#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=4','#SBATCH --mem=12G','#SBATCH --gres=gpu:a100_3g.20gb:1',
        '#SBATCH --time=00:10:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out',
        '#SBATCH --error='+ROOT+'/logs/%j.err','set -euo pipefail',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL',
        'exec '+' '.join(map(shlex.quote,command)),'']).encode()
    write('job.sbatch',script)
    outer={'schema':'ace_screen_training_gpu_launch_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit':rev,'resources':gpu.RESOURCES,'remote_root':ROOT,
        'new_input_files':{**packet['files'],**{n:sha(raw) for n,raw in reviews.items()},
                           'training.json':sha(training_raw),'job.sbatch':sha(script)},
        'capacity_job':'35391340','capacity_acceptance_sha256':g.CPU_ACCEPTANCE_SHA,
        'prepared_sha256':g.PREPARED_SHA,'training_plan_sha256':g.PLAN_SHA,
        'runtime_archive':packet['archive'],'runtime_manifest':packet['manifest'],
        'admission':{'closed_actual_cpu_seconds':10568,'closed_actual_gpu_seconds':1483,
            'cumulative_cpu_preparation_allowance':3600,'candidate_cpu_reservation':4800,'candidate_gpu_reservation':600,
            'total_cpu_seconds':18968,'total_gpu_seconds':2083,'ceiling_cpu_seconds':57600,'ceiling_gpu_seconds':7200},
        'sizing':('One20GB A100 slice,4requested CPUs,max8charged,12GiB,600seconds. '
                  'Prior accepted training03 took108seconds for4SFT2DPO andthree model loads. '
                  'Earlier GPU09 tested7343tokencontext within20GB. New4638/4744token prompts are savedCPU artifacts. '
                  'This bounded smoke measures the new two-singleton DPO graph memory and full-context cost; '
                  'it does not assert full-study throughput. First verifiedGPUwork240s,worker540s,parent550s,final580s.'),
        'purpose':'Artificial full-context fourSFT/twoDPO,8/9 and9/8 pairs, freshSFT/final reload andtenlegalgenerations.',
        'fresh_adapter_required':True,'scientific_generation':False,'new_environment_responses':0,
        'tokenizer_calls_on_gpu':0,'installs':False,'training_qualified':False,'submitted':False}
    write('packet.json',encoded(outer))
    print(json.dumps({'packet_sha256':sha(encoded(outer)),'training_sha256':sha(training_raw),
        'inputs':len(data),'source_files':len(g.SOURCE_NAMES),'transfer_objects':len(outer['new_input_files'])+1,
        'submitted':False},indent=2))

if __name__=='__main__':main()
