from pathlib import Path
import json,datetime,hashlib
root=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02/collection')
items={}
for n in ['stage.json','session/worker/work-start.json','session/worker/runtime-before.json','session/worker/runtime-before_collection.json','session/worker/terminal.json','session/worker/failure.json','session/worker/task/start.json','session/worker/task/terminal.json','session/worker/task/failure.json']:
 p=root/n;assert not p.is_symlink()
 items[n]=dict(present=p.is_file())
 if p.is_file():
  b=p.read_bytes();items[n].update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  if n.endswith('work-start.json'):items[n]['metadata']=json.loads(b)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),markers=items,acceptance=False)))
