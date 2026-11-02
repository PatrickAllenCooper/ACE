from pathlib import Path
import sys,json,hashlib,time,resource,shutil
sys.path.insert(0,'/Users/pat/code/ACE/scripts/research')
from transformers import AutoTokenizer
from llm_intervention_language_policy import prepare_task_tokens,training_action_batch
root=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-context-fixtures-20261102-01')
start=time.monotonic();cpu=time.process_time()
tok=AutoTokenizer.from_pretrained('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca',local_files_only=True,trust_remote_code=False)
prepared={}
for name in ('case0.json','case2.json','case3.json','stress.json'):
 envelope=json.loads((root/name).read_bytes());g,_=prepare_task_tokens(envelope,tok,max_tokens=8192,max_new_tokens=32)
 commands=tuple(tok.decode(x[:-1],skip_special_tokens=False) for x in g.completion_ids)
 commands=tuple(x for x in commands if 'action_id' in x)[:2]
 b,_=training_action_batch(envelope,tok,commands,max_tokens=8192,max_new_tokens=32)
 prepared[name]={'prompt_ids':list(g.prompt_ids),'completion_ids':[list(x) for x in g.completion_ids],'input_ids':b.input_ids.tolist(),'attention_mask':b.attention_mask.tolist(),'labels':b.labels.tolist()}
with (root/'prepared_tokens.json').open('x') as f:json.dump(prepared,f,separators=(',',':'));f.write('\n')
r={'scope':'tokens and masks only, no model load or optimization','prepared_tokens_sha256':hashlib.sha256((root/'prepared_tokens.json').read_bytes()).hexdigest(),'process_cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with (root/'token_receipt.json').open('x') as f:json.dump(r,f,indent=2)
shutil.copyfile('/tmp/ace_prepare_context_tokens.py',root/'prepare_tokens.py')
print(json.dumps(r))
