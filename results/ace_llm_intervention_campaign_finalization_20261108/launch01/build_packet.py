#!/usr/bin/env python3
"""Future CPU final01 packet preparation; never stages, scores or submits.

Run only after independently accepted GPU01 terminal custody exists. Required
pins come from the independent reviewer, never from this builder. The separate
review receipt is retained verbatim without imposing a new review schema.
Defaults use terminal01/{disposition.json,remote/{launch,common}} and acceptance01.
After building, obtain a concrete packet review and commit before launch_once.py.
"""
from pathlib import Path
import argparse, hashlib, json, os, shlex, subprocess, types

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
GPU=REPO/'results/ace_llm_intervention_campaign_gpu_20261108'
LOCAL=REPO/'results/ace_llm_intervention_campaign_finalization_20261108/launch01'
EXT=BASE/'ACE-LLM-intervention-campaign-finalization-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
GPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01'
CPUROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
COMMIT='12c7322e5046cd3e9b38cb59a1c01ed87054b71b'
GPUCOMMIT='07722b85059e47f8e1de27c4c9a0840d03aac420'
GPUPIN='970dc22bdeab1d8cdeded84dfe4f42c58c186cfd77fc9763a9273fbc4a0a3976'
CPUPIN='0533b4fad722652ec63ed12850d3a3d55d1f6c391e81f7280d36561aa9e5ac13'
CPUACCEPT='11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f'
ENTRY='finalize_llm_intervention_campaign_cpu.py'
SOURCEPIN='88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f'
JOB='35388249'
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,wall_seconds=300,gpus=0,requeue=False)

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def git(rev,name):return subprocess.check_output(['git','show',rev+':'+name],cwd=REPO)
def module(name,b,path):
 m=types.ModuleType(name);m.__file__=str(path);exec(compile(b,str(path),'exec'),m.__dict__);return m

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--gpu-terminal-root',default=str(GPU/'terminal01'))
 p.add_argument('--gpu-acceptance',default=str(GPU/'acceptance01/acceptance.json'))
 p.add_argument('--gpu-disposition',default=str(GPU/'terminal01/disposition.json'))
 p.add_argument('--gpu-review-receipt',required=True)
 for n in ('gpu-acceptance-sha256','gpu-disposition-sha256','gpu-review-receipt-sha256'):p.add_argument('--'+n,required=True)
 p.add_argument('--launcher',default='/tmp/ace-launch-campaign-final01-20261108.py')
 p.add_argument('--monitor',default='/tmp/ace-monitor-campaign-final01-20261108.py')
 p.add_argument('--terminal-puller',default='/tmp/ace-pull-campaign-final01-20261108.py')
 a=p.parse_args()
 source=git(COMMIT,'scripts/research/'+ENTRY);need(sha(source)==SOURCEPIN,'reviewed finalizer changed')
 f=module('final01_byte_checks',source,ROOT+'/inputs/'+ENTRY)
 need(f.RESOURCES==RESOURCES,'fixed resource contract differs')
 terminal=Path(a.gpu_terminal_root).absolute();remote=terminal/'remote'
 ga_raw=f.read(a.gpu_acceptance,a.gpu_acceptance_sha256)
 d_raw=f.read(a.gpu_disposition,a.gpu_disposition_sha256)
 review_raw=f.read(a.gpu_review_receipt,a.gpu_review_receipt_sha256)
 need(bool(review_raw.strip()),'independent review receipt empty')
 ga=f.decode(ga_raw);d=f.decode(d_raw)
 gp_raw=f.read(remote/'launch/inputs/packet.json',GPUPIN);gp=f.decode(gp_raw)
 rel='results/ace_llm_intervention_campaign_gpu_20261108/launch01/'
 need(git(GPUCOMMIT,rel+'inputs/packet.json')==gp_raw,'committed GPU packet differs')
 data={}
 for n,h in gp['files'].items():
  need(Path(n).name==n and f.pin(h),'flat pinned GPU input required')
  data[n]=f.read(remote/'launch/inputs'/n,h)
  need(git(GPUCOMMIT,rel+'inputs/'+n)==data[n],'committed GPU input differs: '+n)
 need(sha(data['cpu-packet.json'])==CPUPIN and sha(data['cpu-acceptance.json'])==CPUACCEPT,'CPU predecessor changed')
 need(sha(data[f.GPU_ENTRY])==f.GPU_SHA,'reviewed GPU wrapper changed')
 api=module('final01_gpu_byte_checks',data[f.GPU_ENTRY],remote/'launch/inputs'/f.GPU_ENTRY)
 need(api.inventory(remote)=={n:v['sha256'] for n,v in d['files'].items()},'complete raw custody differs')
 need(all((remote/n).stat().st_size==v['bytes'] for n,v in d['files'].items()),'raw custody sizes differ')
 need(api.inventory(remote/'launch/inputs')==dict(gp['files'],**{'packet.json':GPUPIN}),'GPU input inventory differs')
 old=f.decode(data['cpu-packet.json']);ca=f.decode(data['cpu-acceptance.json'])
 need(api.inventory(remote/'common/inputs')==dict(old['files'],**{'packet.json':CPUPIN}),'CPU input inventory differs')
 paths=dict(cpu_inputs=CPUROOT+'/inputs',cpu_output=CPUROOT+'/cpu03',campaign=CPUROOT+'/cpu03/worker/artifacts/campaign',
            gpu_inputs=GPUROOT+'/inputs',gpu_output=CPUROOT+'/gpu03',output=ROOT+'/finalization')
 need(gp['paths']=={k:paths[k] for k in ('cpu_inputs','cpu_output','campaign','gpu_output')},'original GPU paths differ')
 data.update({ENTRY:source,'gpu-packet.json':gp_raw,'gpu-acceptance.json':ga_raw})
 packet=dict(schema=f.VERSION,scope='artificial',files={n:sha(b) for n,b in sorted(data.items())},
  source_names=sorted(set(gp['source_names'])|{ENTRY}),paths=paths,resources=RESOURCES,
  cpu_packet_sha256=CPUPIN,gpu_packet_sha256=GPUPIN,cpu_acceptance_sha256=CPUACCEPT,gpu_acceptance_sha256=sha(ga_raw))
 need(set(packet)==f.PACKET_KEYS and set(ga)==f.GPU_ACCEPTANCE_KEYS and len(ca)==16,'exact existing packet/acceptance contracts required')
 # Validation view only. Never rewrite saved packets, acceptances or prior bytes.
 lp={**paths,'cpu_output':str(remote/'common/cpu03'),'gpu_output':str(remote/'common/gpu03'),
     'campaign':str(remote/'common/cpu03/worker/artifacts/campaign')}
 need(ca['cpu_output']==paths['cpu_output']==old['cpu_output'] and ca['campaign_root']==ga['campaign_root']==paths['campaign']
      and ga['gpu_output']==paths['gpu_output'] and ga['job_id']==JOB,'original accepted identities differ')
 view={**data,'cpu-acceptance.json':raw({**ca,'cpu_output':lp['cpu_output'],'campaign_root':lp['campaign']}),
       'gpu-acceptance.json':raw({**ga,'gpu_output':lp['gpu_output'],'campaign_root':lp['campaign']})}
 f.admit_evidence({**packet,'paths':lp},{**old,'cpu_output':lp['cpu_output']},gp,view,api)
 need(d['job_id']==JOB and d['scheduler_state']=='COMPLETED' and d['scheduler_exit']=='0:0'
      and d['account']=='ucb736_asc2' and d['gpu_count']==1 and d['resource_contract_matched'] is True
      and d['accounting_errors']==[] and d['input_pin_mismatches']==[],'terminal disposition not admissible')
 cpus,elapsed=d['allocated_cpus'],d['elapsed_seconds']
 need(type(cpus) is int and cpus==ga['allocation']['allocated_cpus'] and type(elapsed) is int and 0<elapsed<=600,'actual GPU allocation missing')
 need(d['allocated_cpu_seconds']==cpus*elapsed and d['gpu_slice_seconds']==elapsed
      and d['closed_cpu_seconds']==9702+cpus*elapsed and d['closed_gpu_slice_seconds']==1347+elapsed
      and d['cumulative_preparation_cpu_allowance']==3600 and d['cpu_admission_after_release']==9702+cpus*elapsed+3600,'terminal accounting differs')
 transport=f.decode(f.read(terminal/'inventory_transport.json'))
 need(transport['returncode']==0,'terminal collection failed')
 observed=f.decode(transport['stdout']);r=observed['parent_accounting']
 need(observed['files']==d['files'] and observed['results']['accounting']['returncode']==0,'terminal evidence differs')
 matches=[l.split('|') for l in observed['results']['accounting']['stdout'].splitlines() if l.startswith(JOB+'|')]
 need(matches==[r] and r[:9]==[JOB,JOB,'ace_llm_campaign_gpu01',d['scheduler_state'],d['scheduler_exit'],str(cpus),str(elapsed),d['account'],d['alloc_tres']],
      'terminal actual scheduler row differs')
 intent_raw=f.read(remote/'launch/submission.intent.json');intent=f.decode(intent_raw)
 result_raw=f.read(remote/'launch/submission.result.json');result=f.decode(result_raw)
 need(result['job_id']==JOB and result['returncode']==0 and result['intent_sha256']==sha(intent_raw)
      and intent['gpu_packet_sha256']==GPUPIN and intent['packet_commit']==GPUCOMMIT and intent['remote_root']==GPUROOT,'GPU submission lineage differs')
 for n,h in intent['pins'].items():
  need(f.relative(n),'unsafe original launch member');f.read(remote/'launch'/n,h)
 rows=[x[:8] for x in intent['prior_accounting']]
 need(len(rows)==34 and len(intent['prior_journals'])==70 and intent['closed_cpu_seconds']==9702
      and intent['closed_gpu_slice_seconds']==1347,'GPU predecessor census differs')
 rows.append([r[0],r[2],r[3],r[4],r[5],r[6],r[7],r[8]])
 journals={**intent['prior_journals'],GPUROOT+'/submission.intent.json':sha(intent_raw),GPUROOT+'/submission.result.json':sha(result_raw)}
 need(len(journals)==72 and len({r[0] for r in rows})==35,'finalizer predecessor census differs')
 admission=dict(closed_cpu_seconds=9702+cpus*elapsed,closed_gpu_slice_seconds=1347+elapsed,
  cumulative_preparation_cpu_allowance=3600,new_cpu_reservation=600,new_gpu_slice_reservation=0,
  cpu_admission=9702+cpus*elapsed+3600+600,gpu_slice_admission=1347+elapsed,cpu_ceiling=57600,gpu_slice_ceiling=7200)
 need(admission['cpu_admission']<=57600 and admission['gpu_slice_admission']<=7200,'cumulative cap exceeded')
 need((cpus,elapsed,admission['closed_cpu_seconds'],admission['closed_gpu_slice_seconds'],admission['cpu_admission'])
      ==(4,136,10246,1483,14446),'independently pinned terminal evidence differs from confirmed GPU01 accounting')
 diagnostic=module('final01_bootstrap',data['qualify_llm_intervention_runtime_metadata.py'],ROOT+'/inputs/qualify_llm_intervention_runtime_metadata.py')
 data['packet.json']=raw(packet)
 command=[old['runtime_source']+'/bin/python','-I','-B','-c',diagnostic.BOOTSTRAP,ROOT+'/inputs/'+ENTRY,SOURCEPIN,
          '--packet',ROOT+'/inputs/packet.json','--packet-sha256',sha(data['packet.json']),'--output',paths['output']]
 job=('''#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_final01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output='''+ROOT+'''/logs/%j.out
#SBATCH --error='''+ROOT+'''/logs/%j.err
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
exec '''+shlex.join(command)+'\n').encode()
 provenance=dict(source_commit=COMMIT,gpu_packet_commit=GPUCOMMIT,gpu_job_id=JOB,terminal_root=str(terminal),
  gpu_acceptance={'path':str(Path(a.gpu_acceptance).absolute()),'sha256':sha(ga_raw)},
  gpu_disposition={'path':str(Path(a.gpu_disposition).absolute()),'sha256':sha(d_raw)},
  gpu_review_receipt={'path':str(Path(a.gpu_review_receipt).absolute()),'sha256':sha(review_raw)},admission=admission,
  resource_rationale='CPU03 same-runtime staging/worker: 119s, peak 982MB (parent-reported). Finalizer 2 CPUs, 7680 MiB, 300s is a bounded cap; actual finalizer cost unmeasured. Reserve 600 allocated CPU seconds.',
  deadlines=dict(startup=120,work=260,cleanup=280,publication=290),
  final_obligations='Check early startup and periodically; retain complete custody and reconcile actual resources. Independent final review required; no blind retry.',
  scientific_qualified=False,finalizer_executed=False)
 artifacts={'job.sbatch':job,'intent-provenance.json':raw(provenance),'gpu-disposition.json':d_raw,'gpu-review-receipt.json':review_raw,
  'gpu-submission.intent.json':intent_raw,'gpu-submission.result.json':result_raw,
  'build_packet.py':Path(__file__).read_bytes(),'launch_once.py':Path(a.launcher).read_bytes(),
  'monitor_final.py':Path(a.monitor).read_bytes(),'pull_terminal.py':Path(a.terminal_puller).read_bytes(),
  'finalizer-review.md':git(COMMIT,'docs/development/guidance/reviews/ace_llm_intervention_campaign_finalizer_review_2026-11-08.md')}
 meta=dict(source_commit=COMMIT,remote_root=ROOT,input_directory=ROOT+'/inputs',output=paths['output'],
  finalizer_packet_sha256=sha(data['packet.json']),cpu_packet_sha256=CPUPIN,gpu_packet_sha256=GPUPIN,
  cpu_acceptance_sha256=CPUACCEPT,gpu_acceptance_sha256=sha(ga_raw),gpu_disposition_sha256=sha(d_raw),gpu_review_receipt_sha256=sha(review_raw),
  input_files={n:sha(b) for n,b in data.items()},root_files={n:sha(b) for n,b in artifacts.items()},resources=RESOURCES,admission=admission,
  expected_prior_jobs=35,expected_prior_submission_journals=72,prior_accounting=sorted(rows,key=lambda r:int(r[0])),prior_journals=journals)
 artifacts['packet-metadata.json']=raw(meta)
 for dest in (EXT,LOCAL):need(not dest.exists() and not dest.is_symlink(),'exclusive destination exists: '+str(dest))
 for dest in (EXT,LOCAL):
  dest.mkdir(parents=True,exist_ok=False);(dest/'inputs').mkdir()
  for n,b in {**artifacts,**{'inputs/'+n:b for n,b in data.items()}}.items():
   with (dest/n).open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 print(json.dumps(dict(local=str(LOCAL),external=str(EXT),packet_sha256=meta['finalizer_packet_sha256'],admission=admission,
                       status='unsubmitted; independent concrete packet review and commit required'),indent=2))

if __name__=='__main__':main()
