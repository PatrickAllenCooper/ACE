from pathlib import Path
import hashlib,json
r=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01');rows={}
for n in ['logs/35402725.err','logs/35402725.out']:
 p=r/n
 if p.is_file():
  assert not p.is_symlink() and p.stat().st_size<=131072
  b=p.read_bytes();rows[n]=dict(sha256=hashlib.sha256(b).hexdigest(),text=b.decode(errors='replace'))
print(json.dumps(rows))
