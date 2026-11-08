#!/usr/bin/env python3
"""Build one unsubmitted GPU01 packet from independently accepted CPU03 custody.

Future invocation only; no remote commands, stages, models or submissions here.
Required CLI inputs are existing receipts with independently supplied byte pins.
No new review/acceptance schema is introduced. The CPU acceptance stays exact.
The copied GPU wrapper's existing byte-only CPU gate is reused for local custody;
only the validation view of its roots is mapped locally, never the saved packet,
acceptance, service commands or launch contracts. No project/model worker runs.
"""
from pathlib import Path
import argparse, hashlib, json, os, shlex, subprocess, types

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL=REPO/'results/ace_llm_intervention_campaign_gpu_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-campaign-gpu-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01'
CPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
INPUT=CPUROOT+'/inputs'; CPUOUT=CPUROOT+'/cpu03'; GPUOUT=CPUROOT+'/gpu03'
CPUPACKET='0533b4fad722652ec63ed12850d3a3d55d1f6c391e81f7280d36561aa9e5ac13'
CPUSOURCE='183ce3137213dcc1e4f33855bfab7b20dd307bb0'
CPUCOMMIT='70d9ee060d10ed087ebf222e1ac3d5b9836b0fd1'
CPUJOB='35387068'
ENTRY='qualify_llm_intervention_campaign_gpu.py'
PHASE='llm_intervention_campaign_gpu_phase.py'
REVIEWED={ENTRY:'58f64d3eea63ae56cc7d14d9188c5d1f9e6e4126fc5b4512abc9758006a803bc',
          PHASE:'c0783ad85ad49dd5a9bcb147420c66fa74fa86484a04bc6951443cb23f3552ad'}
RESOURCES={'account':'ucb736_asc2','partition':'aa100','qos':'gpu-testing','requested_cpus':4,
           'max_allocated_cpus':8,'memory_mib':15360,'wall_seconds':600,'gpus':1,'gpu_type':'a100_3g.20gb','requeue':False}

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def pinned(path,pin):
 b=Path(path).read_bytes();need(len(pin)==64 and sha(b)==pin,'independent pin differs: '+str(path));return b
def git(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
def module(name,b,path):
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(b,str(path),'exec'),m.__dict__);return m

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cpu-terminal-root','cpu-acceptance','cpu-acceptance-sha256','cpu-disposition','cpu-disposition-sha256',
           'cpu-review-receipt','cpu-review-receipt-sha256'):
  p.add_argument('--'+n,required=True)
 p.add_argument('--source-commit',default='HEAD')
 p.add_argument('--launcher',default='/tmp/ace-launch-campaign-gpu01-20261108.py')
 a=p.parse_args()
 rev=subprocess.check_output(['git','rev-parse',a.source_commit+'^{commit}'],cwd=REPO,text=True).strip()
 source=git(rev,'scripts/research/'+ENTRY)
 need(sha(source)==REVIEWED[ENTRY],'existing accepted GPU wrapper source changed')
 gpu=module('gpu_packet_byte_admission',source,ROOT+'/inputs/'+ENTRY)
 need(gpu.RESOURCES==RESOURCES and len(gpu.EXTRA_SOURCES)==5,'fixed GPU resource/source closure changed')
 terminal=Path(a.cpu_terminal_root).resolve(strict=True)
 acceptance_raw=pinned(a.cpu_acceptance,a.cpu_acceptance_sha256)
 disposition_raw=pinned(a.cpu_disposition,a.cpu_disposition_sha256)
 review_raw=pinned(a.cpu_review_receipt,a.cpu_review_receipt_sha256)
 acceptance=gpu.decode(acceptance_raw);disposition=gpu.decode(disposition_raw)
 old_raw=gpu.read(terminal/'inputs/packet.json',CPUPACKET);old=gpu.decode(old_raw)
 cpu_rel='results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/'
 need(git(CPUCOMMIT,cpu_rel+'packet.json')==old_raw,'CPU03 packet commit differs')
 data={}
 for n,h in old['files'].items():
  need(Path(n).name==n,'plain CPU input filename required')
  data[n]=gpu.read(terminal/'inputs'/n,h)
  need(git(CPUCOMMIT,cpu_rel+n)==data[n],'committed CPU input differs: '+n)
 for n in old['source_names']:
  need(git(CPUSOURCE,'scripts/research/'+n)==data[n],'CPU source commit differs: '+n)
 for n in gpu.EXTRA_SOURCES:
  b=git(rev,'scripts/research/'+n)
  need(n not in data or data[n]==b,'GPU addition would replace CPU input')
  data[n]=b
 for n,h in REVIEWED.items():need(sha(data[n])==h,'accepted GPU component source changed: '+n)
 for n,h in ((gpu.GPU_HELPER,gpu.GPU_HELPER_SHA),(gpu.PROCESS_HELPER,gpu.PROCESS_HELPER_SHA),(gpu.DIAGNOSTIC,gpu.DIAGNOSTIC_SHA)):
  need(sha(data[n])==h,'fixed GPU helper differs: '+n)
 data.update({'cpu-packet.json':old_raw,'cpu-acceptance.json':acceptance_raw})
 paths={'cpu_inputs':INPUT,'cpu_output':CPUOUT,'campaign':CPUOUT+'/worker/artifacts/campaign','gpu_output':GPUOUT}
 packet={'schema':gpu.VERSION,'files':{n:sha(b) for n,b in sorted(data.items())},
  'source_names':sorted(set(old['source_names'])|gpu.EXTRA_SOURCES),'cpu_packet_sha256':CPUPACKET,
  'cpu_acceptance_sha256':sha(acceptance_raw),'paths':paths,'resources':RESOURCES,
  'services':{pid:pid+'.freeze.json' for pid in gpu.POLICIES[5:]},
  'contracts':{pid:pid+'.launch.json' for pid in gpu.POLICIES[5:]},
  'lineage_files':dict(acceptance_raw='gpu03-acceptance.json',prepared_raw='prepared.json',engine_terminal_raw='engine-terminal.json',
      sft_checkpoint_raw='sft.checkpoint.json',sft_adapter_raw='sft.adapter.pt',final_checkpoint_raw='final.checkpoint.json',final_adapter_raw='final.adapter.pt')}
 need(old['cpu_output']==CPUOUT and acceptance['cpu_output']==CPUOUT and acceptance['campaign_root']==paths['campaign']
      and acceptance['job_id']==CPUJOB,'CPU03 original identity differs')
 need(gpu.inventory(terminal/'inputs')==dict(old['files'],**{'packet.json':CPUPACKET}),'original CPU input roster differs')
 localpaths={**paths,'cpu_output':str(terminal/'cpu03'),'campaign':str(terminal/'cpu03/worker/artifacts/campaign')}
 bundle=gpu.admit_cpu({**acceptance,'cpu_output':localpaths['cpu_output'],'campaign_root':localpaths['campaign']},
                     {**packet,'paths':localpaths},old)
 # Admit complete retained CPU root, not merely selected successful files.
 need(gpu.inventory(terminal)=={n:v['sha256'] for n,v in disposition['files'].items()},'complete terminal custody differs')
 need(disposition['job_id']==CPUJOB and disposition['scheduler_state']=='COMPLETED' and disposition['scheduler_exit']=='0:0'
      and disposition['resource_contract_matched'] is True and disposition['accounting_errors']==[]
      and disposition['input_pin_mismatches']==[] and disposition['observed_account']=='ucb736_asc2'
      and disposition['observed_cpus']==2 and disposition['observed_gpu_count']==0,'CPU disposition not independently admissible')
 elapsed=disposition['observed_elapsed_seconds']
 need(type(elapsed) is int and 0<elapsed<=600,'CPU03 elapsed accounting missing')
 cpu_cost=2*elapsed
 need(disposition['allocated_cpu_seconds']==cpu_cost and disposition['gpu_slice_seconds']==0
      and 9464+cpu_cost==9702 and disposition['known_closed_cpu_seconds']==9702
      and disposition['cpu_admission_after_release']==13302 and disposition['gpu_admission_seconds']==1347,'CPU03 accounting differs')
 admission=dict(closed_cpu_seconds=9464+cpu_cost,closed_gpu_slice_seconds=1347,cumulative_preparation_cpu_allowance=3600,
                cpu03_job_id=CPUJOB,cpu03_allocated_cpu_seconds=cpu_cost,new_cpu_reservation=4800,new_gpu_slice_reservation=600,
                cpu_admission=9464+cpu_cost+3600+4800,gpu_slice_admission=1347+600,cpu_ceiling=57600,gpu_slice_ceiling=7200)
 cpu_intent_raw=gpu.read(terminal/'submission.intent.json');cpu_intent=gpu.decode(cpu_intent_raw)
 cpu_result_raw=gpu.read(terminal/'submission.result.json');cpu_result=gpu.decode(cpu_result_raw)
 need(cpu_result['job_id']==CPUJOB and cpu_result['intent_sha256']==sha(cpu_intent_raw)
      and cpu_intent['cpu_packet_sha256']==CPUPACKET and cpu_intent['source_commit']==CPUSOURCE
      and cpu_intent['packet_commit']==CPUCOMMIT,'retained CPU03 submission lineage differs')
 prior_rows=[r[:8] for r in cpu_intent['prior_accounting']]
 need(len(prior_rows)==33 and len(cpu_intent['prior_journals'])==68,'CPU03 predecessor census differs')
 prior_rows.append([CPUJOB,'ace_llm_campaign_cpu03','COMPLETED','0:0','2',str(elapsed),'ucb736_asc2',disposition['observed_alloc_tres']])
 journals={**cpu_intent['prior_journals'],CPUROOT+'/submission.intent.json':sha(cpu_intent_raw),CPUROOT+'/submission.result.json':sha(cpu_result_raw)}
 need(len(journals)==70,'GPU predecessor journals differ')
 diagnostic=module('packet_diagnostic',data[gpu.DIAGNOSTIC],INPUT+'/'+gpu.DIAGNOSTIC)
 for pid in gpu.POLICIES[5:]:
  contract=gpu.decode(data[pid+'.launch.json'])
  expected={'entry':INPUT+'/llm_intervention_model_service.py','freeze':INPUT+'/'+pid+'.freeze.json',
   'worker_output':GPUOUT+'/services/'+pid+'/worker','launch_output':GPUOUT+'/services/'+pid+'/launch',
   'binding_path':GPUOUT+'/stage-binding.json','stage_path':GPUOUT+'/stage.json'}
  need(all(contract[k]==v for k,v in expected.items()),'frozen CPU03 service paths differ; never rewrite them')
 data['packet.json']=raw(packet)
 command=[old['runtime_source']+'/bin/python','-I','-B','-c',diagnostic.BOOTSTRAP,ROOT+'/inputs/'+ENTRY,sha(data[ENTRY]),
          '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(data['packet.json']),'--output',GPUOUT]
 job=('''#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_gpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360M
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output='''+ROOT+'''/logs/%j.out
#SBATCH --error='''+ROOT+'''/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
# Preserve scheduler-provided GPU visibility.
exec '''+shlex.join(command)+'\n').encode()
 provenance=dict(source_commit=rev,cpu_source_commit=CPUSOURCE,cpu_packet_commit=CPUCOMMIT,cpu_job_id=CPUJOB,
  cpu_acceptance={'path':str(Path(a.cpu_acceptance).absolute()),'sha256':sha(acceptance_raw)},
  cpu_disposition={'path':str(Path(a.cpu_disposition).absolute()),'sha256':sha(disposition_raw)},
  cpu_review_receipt={'path':str(Path(a.cpu_review_receipt).absolute()),'sha256':sha(review_raw)},
  terminal_root=str(terminal),original_contract_paths=paths,admission=admission,
  resource_rationale='One sequential 20GB slice: prior ~3.278GB GPU reserved; 15GiB host margin over prior 95.05% of 12GiB. Four requested CPUs fit 3840MiB/CPU; reserve maximum eight CPUs for 600s.',
  guards={'first_gpu_seconds':240,'per_service_startup_seconds':120,'progress_freshness_seconds':90,'work_seconds':550,'final_seconds':580,
          'signals':'positive CUDA generations/progress plus independent owned NVIDIA process memory; absent telemetry remains uncertainty'},
  final_obligations='Retain full launch/GPU/campaign custody; reconcile allocation, GPU/memory/progress/throughput/elapsed evidence; independent GPU acceptance and separate CPU finalization. Exit 0 never qualifies.',
  scientific_qualified=False,gpu_executed=False)
 artifacts={'job.sbatch':job,'intent-provenance.json':raw(provenance),'cpu-disposition.json':disposition_raw,
            'cpu-review-receipt.json':review_raw,'build_packet.py':Path(__file__).read_bytes(),'launch_once.py':Path(a.launcher).read_bytes()}
 for label,name in [('gpu-wrapper-review.md','reviews/ace_llm_intervention_campaign_gpu_wrapper_review_2026-11-08.md'),
                    ('gpu-phase-review.md','reviews/ace_llm_intervention_campaign_gpu_phase_review_2026-11-08.md'),
                    ('integration-guidance.md','ace_llm_intervention_paired_campaign_integration_2026-11-08.md')]:
  artifacts[label]=git(rev,'docs/development/guidance/'+name)
 meta=dict(source_commit=rev,remote_root=ROOT,input_directory=ROOT+'/inputs',cpu_input_directory=INPUT,gpu_output=GPUOUT,
  gpu_packet_sha256=sha(data['packet.json']),cpu_packet_sha256=CPUPACKET,cpu_acceptance_sha256=sha(acceptance_raw),
  cpu_disposition_sha256=sha(disposition_raw),cpu_review_receipt_sha256=sha(review_raw),
  input_files={n:sha(b) for n,b in data.items()},root_files={n:sha(b) for n,b in artifacts.items()},
  resources=RESOURCES,admission=admission,expected_prior_jobs=34,expected_prior_submission_journals=70,
  prior_accounting=sorted(prior_rows,key=lambda r:int(r[0])),prior_journals=journals)
 artifacts['packet-metadata.json']=raw(meta)
 for dest in (EXT,LOCAL):need(not dest.exists() and not dest.is_symlink(),'exclusive packet destination exists: '+str(dest))
 for dest in (EXT,LOCAL):
  dest.mkdir(parents=True,exist_ok=False);(dest/'inputs').mkdir()
  for n,b in {**artifacts,**{'inputs/'+n:b for n,b in data.items()}}.items():
   with (dest/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 print(json.dumps({'local':str(LOCAL),'external':str(EXT),'packet_sha256':meta['gpu_packet_sha256'],
                   'admission':admission,'status':'unsubmitted; concrete packet review required'},indent=2))

if __name__=='__main__':main()
