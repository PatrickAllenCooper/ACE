from pathlib import Path
import subprocess,json,datetime
root=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02/collection/session/worker/task/collection')
records=sum((root/f'world-{i:03d}/snapshot-{j:03d}/record.json').is_file() for i in range(32) for j in range(5))
receipts=sum((root/f'world-{i:03d}/snapshot-{j:03d}/receipt.json').is_file() for i in range(32) for j in range(5))
p=subprocess.run(['sstat','-j','35409614.batch','--format=JobID,AveCPU,AveRSS,MaxRSS','-n','-P'],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),saved_record_files=records,saved_receipt_files=receipts,sstat=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),acceptance=False,method='bounded160-path existence check only, no outcome decoding')))
