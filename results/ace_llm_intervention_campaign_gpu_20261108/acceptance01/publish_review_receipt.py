"""Seal the completed local saved-evidence review; no execution revalidation."""
from pathlib import Path
import datetime
import hashlib
import json
import os

ROOT = Path('/Users/pat/code/ACE')
LOCAL = ROOT/'results/ace_llm_intervention_campaign_gpu_20261108/acceptance01'
EXTERNAL = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-gpu-acceptance-20261108-01')
REVIEW = ROOT/'docs/development/guidance/reviews/ace_llm_intervention_campaign_gpu_execution01_review_2026-11-08.md'
def sha(b): return hashlib.sha256(b).hexdigest()
def raw(v): return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def put(path,b):
    with path.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())

acceptance=(LOCAL/'acceptance.json').read_bytes()
checks=json.loads((LOCAL/'evidence-checks.json').read_bytes())
assert sha(acceptance)==checks['acceptance_sha256']=='26119972f7c3e61caf03d15bc72371bfe64dc118edd55d1e6bd26f917e994d0b'
assert checks['status']=='independent_saved_evidence_checks_passed' and checks['required_findings']==0
assert len(json.loads(acceptance))==17
assert sha((LOCAL/'review_saved_evidence.py').read_bytes())==checks['review_source_sha256']
review=REVIEW.read_bytes()
put(LOCAL/'review.md',review)
for p in sorted(LOCAL.iterdir()):
    assert p.is_file() and not p.is_symlink()
    target=EXTERNAL/p.name
    if target.exists(): assert target.read_bytes()==p.read_bytes()
    else: put(target,p.read_bytes())
files={p.name:sha(p.read_bytes()) for p in sorted(LOCAL.iterdir()) if p.is_file()}
receipt={
    'status':'artificial_gpu_phase_accepted','required_findings':0,'scientific_qualified':False,
    'job_id':'35388249','acceptance_schema_fields':17,'acceptance_sha256':sha(acceptance),
    'review_path':str(REVIEW),'review_sha256':sha(review),
    'review_source_sha256':checks['review_source_sha256'],
    'receipt_publisher_sha256':sha(Path(__file__).read_bytes()),
    'evidence_checks_sha256':files['evidence-checks.json'],
    'artifact_sha256':files,
    'custody_path':str(ROOT/'results/ace_llm_intervention_campaign_gpu_20261108/terminal01'),
    'external_custody_path':'/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-gpu-terminal-20261108-01',
    'acceptance_path':str(LOCAL/'acceptance.json'),'external_acceptance_path':str(EXTERNAL/'acceptance.json'),
    'source_commit':checks['source_commit'],'submitted_packet_commit':checks['packet_commit'],
    'remote_submission_intent_sha256':checks['remote_intent_sha256'],
    'raw_files':checks['raw_files'],'raw_bytes':checks['raw_bytes'],
    'original_cpu_raw_files_preserved':checks['original_cpu_raw_files_preserved'],
    'gpu_output_files':checks['gpu_output_files'],'campaign_files':checks['campaign_files'],
    'combined_attempt_pins':json.loads(acceptance)['attempt_pins'],
    'counts':checks['counts'],'complete_journal_chains':checks['complete_journal_chains'],
    'runtime_snapshots':checks['runtime_snapshots'],'sessions':checks['sessions'],
    'scheduler':checks['scheduler'],'related_closed_jobs_after':35,'related_submission_journals_after':72,
    'census_scope':checks['census_scope'],
    'cuda_interval_ms':checks['cuda_interval_ms'],'cuda_intervals_are_percent_utilization':False,
    'hardware_utilization_percentage':None,
    'new_training_updates':0,'private_scores_computed':False,
    'reviewer_executed_project_validator_or_tests':False,
    'reviewer_executed_models_tensor_deserialization_or_tokenizer':False,
    'reviewer_executed_remote_actions_or_private_scoring':False,
    'reviewer_decoded_private_bundle':False,
    'successful_check_process_cpu_seconds':checks['review_process_cpu_seconds'],
    'review_cost_scope':'Successful saved-evidence pass only; earlier inspection and reviewer schema corrections excluded.',
    'reviewer_schema_corrections_retained':3,
    'acceptance_scope':'Observed artificial GPU01 execution, scheduler, runtime, cleanup and custody. Prior CPU and implementation reviews reused. Independent CPU finalization remains pending.',
    'resource_followup':'Startup dominates this small run. Before repetition/scaling assess CPU pre-staging and tighter bounded resources; CUDA intervals do not measure percent utilization.',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
b=raw(receipt)
for root in (LOCAL,EXTERNAL): put(root/'review_receipt.json',b)
assert {p.name:sha(p.read_bytes()) for p in LOCAL.iterdir()}=={p.name:sha(p.read_bytes()) for p in EXTERNAL.iterdir()}
print(raw({'acceptance_sha256':sha(acceptance),'review_sha256':sha(review),'review_receipt_sha256':sha(b),'review_source_sha256':checks['review_source_sha256'],'required_findings':0,'mirrors_identical':True}).decode())
