from pathlib import Path
import json,datetime
w=Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01/training/session/worker/artifacts')
f=w/'fit'
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':len(list((f/'training').glob('step-*.complete.json'))),'sft_generation_records':len(list(f.glob('sft-generation-*.json'))),'final_generation_records':len(list(f.glob('final-generation-*.json'))),'sft_reload_record':(f/'sft-reload.json').is_file(),'final_reload_record':(f/'final-reload.json').is_file(),'worker_terminal':(w/'terminal.json').is_file(),'worker_failure':(w/'failure.json').is_file(),'acceptance':False}))
