from pathlib import Path
import json,datetime,hashlib
root=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02/collection')
paths=sorted(root.glob('phase-[0-9][0-9][0-9][0-9].json'))
items=[]
for p in paths[-4:]:
 assert not p.is_symlink()
 b=p.read_bytes();assert len(b)<131072
 items.append(dict(path=p.name,sha256=hashlib.sha256(b).hexdigest(),phase=json.loads(b)))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase_count=len(paths),latest=items,acceptance=False)))
