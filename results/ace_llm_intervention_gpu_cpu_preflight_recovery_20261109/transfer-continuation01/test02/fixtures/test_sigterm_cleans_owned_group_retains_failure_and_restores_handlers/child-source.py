import os,json,time,sys
from pathlib import Path
p=Path('/Users/pat/code/ACE/results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109/transfer-continuation01/test02/fixtures/test_sigterm_cleans_owned_group_retains_failure_and_restores_handlers/artificial-child.json')
with p.open("x") as f:json.dump({"pid":os.getpid(),"pgid":os.getpgrp(),"artificial":True},f)
print("ARTIFICIAL LOCAL CHILD EARLY PROGRESS",file=sys.stderr,flush=True)
time.sleep(12)
