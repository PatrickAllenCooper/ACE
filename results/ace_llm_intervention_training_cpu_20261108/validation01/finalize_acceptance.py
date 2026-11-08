"""Bind a completed distinct result review to the retained CPU qualification."""
from pathlib import Path
import datetime,hashlib,json
repo=Path('/Users/pat/code/ACE');base=repo/'results/ace_llm_intervention_training_cpu_20261108'
review=repo/'docs/development/guidance/reviews/ace_llm_intervention_training_cpu_results01_review_2026-11-08.md'
text=review.read_text()
assert '35374034' in text and ('0 required' in text.lower() or 'zero required' in text.lower())
validation=base/'validation01/saved_validation.json';saved=json.loads(validation.read_bytes())
assert saved['status']=='saved_byte_checks_passed_independent_review_pending' and saved['job_id']=='35374034'
assert saved['allocated_cpu_seconds']==154 and saved['gpu_slice_seconds']==0
local=base/'acceptance01';ext=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-cpu-acceptance-20261108-01')
for p in [local,ext]:p.mkdir(mode=0o700,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r={k:v for k,v in saved.items() if k not in ['scope','supervisor','next_gate','status','actual_training_plan_qualified','utc']}
r.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='artificial_complete_batch_cpu_preparation_accepted',
 actual_training_plan_qualified=True,model_weights_loaded=False,explicit_cuda_api_called=False,
 required_remaining=0,review_sha256=sha(review),saved_validation_sha256=sha(validation),
 disposition_sha256=sha(base/'terminal01/disposition.json'),
 qualification_scope='Complete saved artificial batch with actual Qwen tokenizer in authenticated staged runtime, source/input closure, supervised CPU execution and scheduler completion only.',
 training_execution_qualified=False,scientific_worlds=0,no_outstanding_candidate_reservation=True,
 supervisor_elapsed_seconds=saved['supervisor']['elapsed_seconds'],supervisor_cpu_seconds=saved['supervisor']['supervisor_cpu_seconds'],
 child_and_cleanup_cpu_seconds=saved['supervisor']['session']['child_and_cleanup_cpu_seconds'],
 child_elapsed_including_cleanup_seconds=saved['supervisor']['session']['elapsed_including_cleanup_seconds'],
 child_peak_rss_bytes=saved['supervisor']['session']['cumulative_child_peak_rss_bytes'],
 scheduler_total_cpu_seconds=45.042,scheduler_batch_maxrss_kib=6289292,memory_headroom_qualified=False,
 runtime_contract_scope='Prospective copied runtime only, not historical freeze qualification',
 next_required_gate='Separately reviewed integrated fresh-adapter GPU training and checkpoint inference qualification, then paired campaign qualification before scientific generation.')
raw=(json.dumps(r,sort_keys=True,indent=2)+'\n').encode()
for root in [local,ext]:
 (root/'acceptance.json').write_bytes(raw);(root/'review.md').write_bytes(review.read_bytes());(root/'finalize_acceptance.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'acceptance_sha256':hashlib.sha256(raw).hexdigest(),'status':r['status'],'training_execution_qualified':False,'scientific_qualified':False},indent=2))
