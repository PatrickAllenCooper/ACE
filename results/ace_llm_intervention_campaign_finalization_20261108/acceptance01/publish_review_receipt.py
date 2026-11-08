"""Publish this bounded review's acceptance and separate metadata, without remote actions."""
from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('/Users/pat/code/ACE');A=B/'results/ace_llm_intervention_campaign_finalization_20261108/acceptance01';T=A.parent/'terminal01';L=T/'remote/launch';C=T/'remote/common/cpu03/worker/artifacts/campaign'
E=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-acceptance-20261108-01')
D=B/'docs/development/guidance/reviews/ace_llm_intervention_campaign_finalization_execution01_review_2026-11-08.md'
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x') as f: f.write(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def inventory(root):return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()}
checks=j(A/'attempt-01/checks.json');execution=j(A/'attempt-01/execution.json');pkt=j(L/'inputs/packet.json');receipt=j(C/'finalization.json')
assert checks['status']=='passed' and execution['returncode']==0
assert checks['source_sha256']==sha(A/'check_saved_evidence.py')==sha(A/'attempt-01/check_saved_evidence.py')
assert not (A/'acceptance.json').exists() and not (E/'acceptance.json').exists()
shutil.copy2(D,A/'review.md')
acceptance={
 'schema':'ace-artificial-campaign-finalization-acceptance-v1','status':'artificial_finalization_accepted','scope':'artificial','artificial_qualified':True,'scientific_qualified':False,'job_id':'35389391',
 'accepted_checks':{k:True for k in ['source','runtime','cleanup','scheduler','custody','original_input_preservation','saved_curve_arithmetic','all_pair_arithmetic','charges_and_failures']},
 'allocation':{'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal','requested_cpus':2,'allocated_cpus':2,'memory_mib':7680,'gpus':0,'wall_seconds':300,'elapsed_seconds':61,'state':'COMPLETED','exit_code':'0:0'},
 'finalizer_packet_sha256':sha(L/'inputs/packet.json'),'finalizer_source_sha256':sha(L/'inputs/finalize_llm_intervention_campaign_cpu.py'),
 'cpu_acceptance_sha256':pkt['cpu_acceptance_sha256'],'gpu_acceptance_sha256':pkt['gpu_acceptance_sha256'],
 'campaign_root':pkt['paths']['campaign'],'attempt_pins':receipt['attempt_pins'],'finalization_sha256':sha(C/'finalization.json'),'summary_sha256':sha(C/'private/summary.json'),
 'cell_count':16,'policy_comparisons':28,'failed_or_unresolved_cells':0,'new_environment_responses':0,'new_training_updates':0,'retry_allowed':False,
 'logical_actual_cost_units':{'initial':32,'acquisition':64,'total':96},
 'attainment':{'ever_attained_cells':15,'final_attained_cells':10,'world_9102_all_first_crossings_zero':True,'first_crossing_is_retrospective':True,'actual_saved_responses_demonstrated':False}}
write(A/'acceptance.json',acceptance)
evidence={}
for p in sorted(T.rglob('*')):
    if not p.is_file():continue
    # Raw bytes are transitively bound by disposition; additionally pin all new finalizer evidence and reports.
    if p.parent==T or p.is_relative_to(L/'finalization') or (p.is_relative_to(C/'private') and p.name in ['cost-report.json','summary.json','report-pins.json']) or p in [L/'submission.intent.json',L/'submission.result.json',C/'finalization.json',C/'finalization.candidate.json']:
        evidence[str(p.relative_to(T))]={'bytes':p.stat().st_size,'sha256':sha(p)}
sidecar={'schema':'ace-independent-review-receipt-v1','scope':'new final01 actual execution and saved evidence arithmetic only','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'required_findings':[],'required_findings_count':0,
 'artificial_qualified':True,'scientific_qualified':False,'acceptance_sha256':sha(A/'acceptance.json'),'review_path':str(D),'review_sha256':sha(D),'checker_source_sha256':sha(A/'check_saved_evidence.py'),'publisher_source_sha256':sha(Path(__file__)),
 'checks_sha256':sha(A/'attempt-01/checks.json'),'checker_attempts':[{'path':'attempt-01','returncode':0,'complete_files':inventory(A/'attempt-01')}],'failed_checker_attempts':0,
 'reused_cpu_gpu_reviews':True,'original_finalizer_rerun':False,'models_or_tensors_loaded_by_reviewer':False,'remote_actions':False,
 'terminal_root':str(T),'terminal_external_mirror':'/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-terminal-20261108-01','raw_files':2219,'raw_bytes':110220879,'transfer_pins':94,'custody_copies_verified':2,'remote_inventories_matched':2,
 'allocation_cost':{'new_cpu_seconds':122,'new_gpu_slice_seconds':0,'closed_cpu_seconds':10368,'closed_gpu_slice_seconds':1483,'preparation_cpu_allowance':3600,'cpu_admission':13968,'terminal_jobs':36,'submission_journals':74},
 'batch_max_rss_kib':7862160,'batch_memory_headroom_kib':2160,'memory_interpretation':'Successful bounded run; batch MaxRSS near limit. Worker-only RSS does not establish headroom.',
 'checker_cpu_seconds':checks['process_cpu_seconds'],'checker_cpu_scope':'One local standard-library checker process; excludes git child processes, inspection, writing and transfer. Not whole-cycle CPU.',
 'interpretation':['All sixteen cells actually spend six logical cost units; first crossing is retrospective and shows no actual saved responses.','All world-9102 initial errors meet threshold; only target_risk, learned_ranker and sft_llm finish attained.','World-9101 DPO never attains; cap four retained.','All methods, twenty-eight pairs and both strata retained; no favorable filtering.','Matching seven observe_all and one x_plus per LLM does not prove identical noisy trajectories or scientific benefit.'],
 'evidence_pins':evidence}
write(A/'review_receipt.json',sidecar)
# Preserve previously mirrored attempts; never replace divergent bytes.
for p in sorted(A.rglob('*')):
    q=E/p.relative_to(A)
    if p.is_dir():q.mkdir(exist_ok=True)
    elif q.exists():assert sha(q)==sha(p),str(q)
    else:shutil.copy2(p,q)
assert inventory(A)==inventory(E)
print(json.dumps({'review_sha256':sha(D),'acceptance_sha256':sha(A/'acceptance.json'),'review_receipt_sha256':sha(A/'review_receipt.json'),'mirrored_files':len(inventory(A))},indent=2))
