from pathlib import Path
import os,json
paths=['/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01','/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03','/scratch/alpine/paco0228/ACE/results']
print(json.dumps(dict(host=os.uname().nodename,paths=[dict(path=p,resolve=str(Path(p).resolve()),exists=Path(p).exists(),symlink=Path(p).is_symlink()) for p in paths])))
