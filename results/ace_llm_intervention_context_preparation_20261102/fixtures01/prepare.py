from pathlib import Path
import sys,json,hashlib,time,resource,shutil
sys.path.insert(0,'/Users/pat/code/ACE/scripts/research')
from transformers import AutoTokenizer
from llm_intervention_language_policy import prepare_task_tokens
root=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-context-fixtures-20261102-01');root.mkdir()
repo=Path('/Users/pat/code/ACE');source=repo/'results/ace_llm_intervention_task_runtime_20261102/cpu01/input'
start=time.monotonic();cpu=time.process_time()
model=Path('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
profiles={}
for name in ['case0.json','case2.json','case3.json']:
 shutil.copyfile(source/name,root/name)
 e=json.loads((root/name).read_bytes());g,m=prepare_task_tokens(e,tok,max_tokens=8192,max_new_tokens=32);profiles[name]=m
stress=json.loads((source/'case3.json').read_bytes())
# A repeated neutral note changes only public text length, not any numerical state.
stress['state']['domain_notes']='Artificial context length qualification. ' * 600
raw=json.dumps(stress,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
(root/'stress.json').write_text(raw)
g,m=prepare_task_tokens(stress,tok,max_tokens=8192,max_new_tokens=32);profiles['stress.json']=m
report={'scope':'CPU tokenizer-only artificial context fixture preparation. No model weights, updates, environment callbacks or scientific world.','profiles':profiles,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json')},'model_revision':model.name,'local_transformers':'4.51.3','process_cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(root/'profile.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
shutil.copyfile('/tmp/ace_prepare_context_fixtures.py',root/'prepare.py')
print(json.dumps(report))
