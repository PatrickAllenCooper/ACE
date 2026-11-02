import ast,json,sys
from pathlib import Path
from prepare_llm_intervention_task_cpu import validate_allocation
root=Path(sys.argv[1]);raw=(root/'prior_allocation.txt').read_text()
for p in (root/'source').glob('*.py'):ast.parse(p.read_text())
assert validate_allocation(raw,'35204167')['allocated']['cpu']=='2'
negatives=[raw.replace('cpu-normal','gpu-testing'),raw.replace('TimeLimit=00:10:00','TimeLimit=00:15:00'),raw.replace('AllocTRES=cpu=2','AllocTRES=gres/gpu=1,cpu=2'),raw.replace('NumCPUs=2','NumCPUs=3'),raw.replace('Requeue=0','Requeue=1'),raw.replace('JobState=RUNNING','JobState=COMPLETED')]
for changed in negatives:
 try:validate_allocation(changed,'35204167')
 except ValueError:pass
 else:raise AssertionError('invalid allocation accepted')
print(json.dumps({'historical_metadata_positive':1,'fabricated_negative_allocations':len(negatives),'live_scheduler_claim':False}))
