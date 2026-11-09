import pathlib,json,datetime,signal
signal.alarm(40)
r=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01/cpu/worker/artifacts/campaign')
out={}
for i in range(5):
 d=r/('policy-%02d'%i)
 if d.is_symlink():raise ValueError('symlink')
 rows=[]
 for j in range(24):
  a=d/('world-%02d'%j)/'arm-receipt.json'
  if a.is_symlink():raise ValueError('symlink')
  if a.is_file():rows.append(j)
 out[str(i)]={'completed_receipt_indices':rows,'attempt_receipt_present':(d/'attempt.json').is_file()}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'35482135','counts_only':out,'acceptance':False,'performance_decoded':False}))
