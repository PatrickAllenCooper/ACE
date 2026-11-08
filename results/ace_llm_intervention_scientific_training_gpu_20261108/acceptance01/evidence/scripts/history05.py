from pathlib import Path
import json,hashlib,types,sys,time,traceback,resource
OUT=Path(__file__).resolve().parents[1];T=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01');L=T/'remote/launch';start=time.monotonic()
def need(b,m):
 if not b:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(n,v):
 with (OUT/'attempts'/n).open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def module(name,p,pin):
 b=p.read_bytes();need(sha(b)==pin,'independently pinned module '+name);m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(b,str(p),'exec'),vars(m));return m
try:
 aid=module('saved_audit_aid',Path('/tmp/ace-gputrain35420219-audit-prep-zqkcv893/audit_saved_training.py'),'bf378f64ca534438d6060e7b4d4125420d539df45470b1771ea54f665dcbd11f')
 path=Path('/private/tmp/ace-cpu-campaign-next-argument-map-71cvg3m6/argument-map.json');need(sha(path.read_bytes())=='8fdf6b80f4f8d98797ee360f1f4f8a7d4fedf2a5ef2704767cf5eefed0a25421','prior independently pinned argument map');mapping=read(path)
 with aid.closed_validators() as mods:
  cpu=mods['cpu_consumer'];a=read(OUT/'authority.json');outer,result,disp,obs=aid.terminal_identity(outer_raw=(L/'packet.json').read_bytes(),outer_sha256='33fa6064bd224a64736bf466623bd3ae9399c81022dac6023c689515764a9ec0',result_raw=(L/'submission.result.json').read_bytes(),disposition_raw=(T/'disposition.json').read_bytes(),disposition_sha256=a['pins']['disposition.json'],inventory_raw=(T/'inventory_transport.json').read_bytes(),inventory_sha256=a['pins']['inventory_transport.json'])
  expected=aid.expected_training(cpu=cpu,outer=outer,token_acceptance_raw=(L/'cpu-acceptance.json').read_bytes(),freeze_raw=(L/'freeze.json').read_bytes(),freeze_sha256='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d',prepared_sha256=outer['prepared_sha256'])
  need(expected==read(Path('/private/tmp/ace-cpu-campaign-next-argument-map-71cvg3m6/expected-training.json')),'independent expected record agrees with original parent input map')
  supplied={}
  for n,v in mapping['metadata'].items():
   if v['path'] is not None:supplied[n]=aid.read_pinned(v['path'],v['sha256'])
  need(len(supplied)==43,'43 original resolved inputs')
  supplied['training-inventory']=(T/'inventory_transport.json').read_bytes();supplied['training-disposition']=(T/'disposition.json').read_bytes()
  pins={n:sha(b) for n,b in supplied.items()};m={n:cpu.decode(b) for n,b in supplied.items() if n not in cpu.TEXT_METADATA}
  # This is only independently known submission identity, not an acceptance object.
  m['gpu-training-acceptance']={'job_id':result['job_id']}
  history=cpu.campaign_token_history(supplied,m,pins,obs['files'])
  need(history['token_recovery']==outer['token_recovery'],'nested exact recovery')
  for n,h in history['upstream_pins'].items():need(obs['upstream_pins'][n]==h,'original transport upstream pin '+n)
  prior=m['previous-baseline'];before=[r for r in obs['related_accounting'] if r[0]!=aid.JOB]
  need(cpu.normalized(before)==cpu.normalized(prior['related_accounting']) and prior['terminal_jobs']==44 and prior['submission_journals']==90,'all previous jobs unchanged')
  journals={**prior['journal_pins'],aid.ROOT+'/submission.intent.json':result['intent_sha256'],aid.ROOT+'/submission.result.json':aid.RESULT_SHA}
  need(journals==obs['journal_pins'] and len(journals)==92,'45/92 complete journal transition')
  need(cpu.gpu_cost(obs['related_accounting'])==(17805,2505) and cpu.gpu_cost(prior['related_accounting'])==(14189,1601),'actual charges')
  intent=read(L/'submission.intent.json');need(intent['packet_commit']==aid.PACKET_COMMIT and intent['packet_sha256']==sha((L/'packet.json').read_bytes()) and intent['blind_retry_forbidden'] is True and intent['new_namespace_claimed'] is False and intent['replacement_worlds'] is False,'once-only immutable namespace intent')
  collector=read(T/'collector-intent.json');need(collector['local_intent_sha256']==result['intent_sha256'] and collector['job_id']==aid.JOB and collector['submission_result_sha256']==aid.RESULT_SHA,'local/remote exact intent result')
  for n,pin in [('namespace.json','40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a'),('original-namespace.json','58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29')]:need(sha((T/'remote'/n).read_bytes())==pin,'remote original claim')
  for n,pin in [('local.namespace.json','40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a'),('original-local.namespace.json','58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29')]:need(sha((T/'namespaces'/n).read_bytes())==pin,'local original claim')
  prepared=read(L/'inputs/prepared.json');records=prepared['training_plan']['records'];freeze=read(L/'freeze.json')
  need([r['record_id'] for r in records]==[f'policy_train/{w}/snapshot-{j:03d}' for w in freeze['training_world_ids'] for j in range(5)],'exact32x5 original policy_train roster')
  need(all(r['scope']=='policy_train' for r in records) and prepared['training_plan']['sft_steps']==160 and prepared['training_plan']['dpo_steps']==119,'exact accepted exposure')
  need(sum(len(r['strict_pairs'])+len(r['tied_pairs']) for r in records)==3584,'all3584pairs retained')
  planfile=T/'remote/cpu/preparation/session/worker/artifacts/numerical-plan.json';need(sha(planfile.read_bytes())==m['cpu-preparation-acceptance']['numerical_plan_file_sha256'],'matched numerical plan byte pin')
  need(cpu.digest(read(planfile))==m['cpu-preparation-acceptance']['numerical_plan_sha256'],'matched numerical plan semantic pin')
  need(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','safetensors','tokenizers')),'saved metadata only')
  save('history05.json',dict(status='verified_original_lineage_and_cpu_consumer_history',expected_training=expected,metadata_pins=pins,token_recovery=history['token_recovery'],upstream_pin_count=len(history['upstream_pins']),preserved_root_count=len(history['preserved_directories']),terminal_jobs=45,submission_journals=92,closed_cpu_seconds=17805,closed_gpu_slice_seconds=2505,cpu_admission_seconds=21405,environment_responses=0,retained_pairs=3584,policy_train_worlds=32,records=160,steps=279,sft_steps=160,dpo_steps=119,full_cpu_admit_called=False,full_cpu_admit_pending='Fresh actual CPU site readiness and independent future concrete CPU packet remain parent-owned; no readiness text substituted.',wall_seconds=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  print('history and strict CPU consumer metadata lineage verified')
except BaseException as e:
 save('history05-failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
