from pathlib import Path
import hashlib,json
r=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01')
result={}
for n in ('preparation/failure.json','preparation/attempt.json','logs/35415334.err'):
 p=r/n
 if p.is_file() and not p.is_symlink():
  b=p.read_bytes()
  if len(b)>131072:raise ValueError('bounded diagnostic only')
  result[n]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode()}
print(json.dumps({'files':result,'scope':'partial diagnostic, not complete custody'}))
