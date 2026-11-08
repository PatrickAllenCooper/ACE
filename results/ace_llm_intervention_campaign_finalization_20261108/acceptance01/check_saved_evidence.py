"""Bounded independent final01 custody/runtime and saved-arithmetic review.
No models, tensor decoding, numerical fitting, tests, remote calls or finalizer.
The sole reused code is the pinned, stdlib-only pure screen-report consistency helper.
"""
from pathlib import Path
import hashlib,json,math,itertools,types,sys,subprocess,time,collections
BASE=Path('/Users/pat/code/ACE')
ROOT=BASE/'results/ace_llm_intervention_campaign_finalization_20261108'
T=ROOT/'terminal01'; R=T/'remote'; L=R/'launch'; F=L/'finalization'
C=R/'common/cpu03/worker/artifacts/campaign'; P=C/'private'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-finalization-terminal-20261108-01')
OUT=Path(sys.argv[1]); checks=[]; started=time.process_time()
def j(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(v,label):
    if not v:raise AssertionError(label)
def done(label): checks.append(label);print('PASS',label,flush=True)
def inv(root):
    result={}
    for p in sorted(root.rglob('*')):
        ck(not p.is_symlink(),f'symlink {p}')
        if p.is_file(): result[p.relative_to(root).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    return result
def close(a,b,label):ck(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label)
packetpin='417bd889465198c0276fcc2a2ebf55168e21efad278c7978a34ed9360761bf6c'
sourcepin='88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f'
summarypin='790cb2ecbf6af7e5e63ad10e7e8056f7b69e75f68c185ad345004ce430c7d577'
receiptpin='aa77436fbda2e3348b7eae08d33327c1fa5e33ae3e6602a4abd69930a53c6dfa'
d=j(T/'disposition.json'); actual=inv(R)
ck(actual==d['files'],'complete local raw inventory')
ck((len(actual),sum(x['bytes'] for x in actual.values()))==(2219,110220879),'raw totals')
ck(inv(EXT)==inv(T),'complete second local custody copy, including transport metadata')
transports=[]
for name in ('inventory_transport.json','inventory_after_transport.json'):
    t=j(T/name);ck(t['returncode']==0,'transport success');v=json.loads(t['stdout']);transports.append(v)
    ck(v['files']==actual,'remote inventory/readback matches all raw files')
done('2219 files / 110220879 bytes; both copies and both remote inventories match')
meta=j(L/'packet-metadata.json');pkt=j(L/'inputs/packet.json');intent=j(L/'submission.intent.json')
ck(sha(L/'inputs/packet.json')==packetpin,'independent packet pin')
ck(sha(L/'inputs/finalize_llm_intervention_campaign_cpu.py')==sourcepin,'independent finalizer pin')
ck(sha(L/'submission.intent.json')=='dedd39caaf131af4d08fdf9d4750934d622fa8993f4568a6eb22c75bb8fbb7da','independent submission intent')
ck(len(intent['pins'])==94,'94 transferred pins')
for name,pin in intent['pins'].items():ck(sha(L/name)==pin,'transfer pin '+name)
for name,pin in pkt['files'].items():ck(sha(L/'inputs'/name)==pin,'input '+name)
ck(meta['source_commit']==intent['source_commit']=='12c7322e5046cd3e9b38cb59a1c01ed87054b71b','source commit')
for name in pkt['source_names']:
    raw=subprocess.check_output(['git','show',meta['source_commit']+':scripts/research/'+name],cwd=BASE)
    ck(hashlib.sha256(raw).hexdigest()==pkt['files'][name],'captured source commit '+name)
ck(intent['packet_commit']=='380d1b3d939a64ffd0602014746df7304035cc6d','packet commit')
for name,pin in intent['pins'].items():
    raw=subprocess.check_output(['git','show',intent['packet_commit']+':'+str((ROOT/'launch01'/name).relative_to(BASE))],cwd=BASE)
    ck(hashlib.sha256(raw).hexdigest()==pin,'committed packet '+name)
for name,key in [('cpu-acceptance.json','cpu_acceptance_sha256'),('gpu-acceptance.json','gpu_acceptance_sha256')]:
    ck(sha(L/'inputs'/name)==pkt[key]==meta[key],'upstream acceptance '+name)
ck(pkt['gpu_acceptance_sha256']=='26119972f7c3e61caf03d15bc72371bfe64dc118edd55d1e6bd26f917e994d0b','independent GPU acceptance')
ck(pkt['cpu_acceptance_sha256']=='11bb11f75715d232cc8b74af81dbc6826272bb982769c054353f16def204817f','independent CPU acceptance')
prior=j(BASE/'results/ace_llm_intervention_campaign_gpu_20261108/terminal01/disposition.json')
oldcommon={k:v for k,v in prior['files'].items() if k.startswith('common/')}
for name,v in oldcommon.items():ck(actual[name]==v,'immutable original common bytes '+name)
ga=j(L/'inputs/gpu-acceptance.json')
for name,pin in ga['campaign_files'].items():ck(sha(C/name)==pin,'accepted campaign bytes '+name)
ck(len(ga['campaign_files'])==729,'729 accepted campaign files')
done('94 committed packet pins, 51 captured sources, upstream acceptance and all original common bytes preserved')
sched=transports[-1]['results'];rows=[x.split('|') for x in sched['accounting']['stdout'].splitlines()]
ck(all(x[3:5]==['COMPLETED','0:0'] for x in rows),'all Slurm steps terminal success')
ck(rows[0][0]=='35389391' and rows[0][5:9]==['2','61','ucb736_asc2','billing=2,cpu=2,mem=7.50G,node=1'],'actual parent allocation')
ck(sched['details']['stdout'].strip()=='acpu|cpu-normal|5|billing=2,cpu=2,mem=7.50G,node=1|paco0228','actual requested limits')
ck('Requeue=0 Restarts=0' in sched['job']['stdout'] and 'gres/gpu' not in sched['job']['stdout'],'no GPU/requeue/restart')
ck(not any('|ace_llm' in x for x in sched['queue']['stdout'].splitlines()),'no related active queue')
ck(len(meta['prior_accounting'])==35 and len(meta['prior_journals'])==72,'predecessor census')
ck(intent['prior_accounting']==meta['prior_accounting'] and intent['prior_journals']==meta['prior_journals'],'readiness exact prior census')
cpu=sum(int(x[4])*int(x[5]) for x in meta['prior_accounting'])
gpu=sum(int(x[5])*sum(int(t.split('=')[1]) for t in x[7].split(',') if t.startswith('gres/gpu:')) for x in meta['prior_accounting'])
ck((cpu,gpu)==(10246,1483),'prior charges including failed jobs')
ck((d['closed_cpu_seconds'],d['closed_gpu_slice_seconds'],d['cpu_admission_after_release'])==(cpu+122,gpu,cpu+122+3600),'closed/admission costs')
ck(d['accounting_errors']==[] and d['input_pin_mismatches']==[],'collector discrepancies')
submission=j(L/'submission.result.json');ck(submission['job_id']=='35389391' and submission['stdout']=='35389391\n' and submission['returncode']==0,'single job result')
start=j(F/'start.json');attempt=j(F/'attempt.json');worker=j(F/'worker.started.json');cleanup=j(F/'worker.cleanup.json');stage=j(F/'stage.json');handoff=j(F/'handoff.json')
ck(start['resources']==pkt['resources']=={'account':'ucb736_asc2','cpus':2,'gpus':0,'memory_mib':7680,'partition':'acpu','qos':'cpu-normal','requeue':False,'wall_seconds':300},'resource contract')
for k,v in {'startup':120,'work':260,'cleanup':280,'hard':290}.items():close(start['deadlines'][k]-start['started_monotonic'],v,'absolute deadline '+k)
ck(worker['started_monotonic']==handoff['started_monotonic']==start['started_monotonic'],'inherited start')
ck(worker['pid']==worker['pgid']==cleanup['pid']==cleanup['pgid'],'owned process group')
ck(cleanup['already_gone'] and not cleanup['group_still_exists'] and cleanup['errors']==[] and cleanup['returncode']==0,'clean owned worker')
ck(attempt['status']=='completed' and attempt['error'] is None and attempt['elapsed_seconds']<290,'outer terminal within deadline')
ck(j(F/'worker/attempt.json')['status']=='completed','worker terminal')
ck(handoff['stage_sha256']==sha(F/'stage.json') and worker['command'][-1]==sha(F/'handoff.json'),'stage handoff pins')
ck(sourcepin in worker['command'] and packetpin in worker['command'] and worker['command'][1:3]==['-I','-B'],'isolated authenticated worker')
workstart=j(F/'worker/work-start.json');ck(workstart['private_finalization_calls']==1 and workstart['monotonic']<start['deadlines']['startup'],'one finalization call before deadline')
cpustage=j(R/'common/cpu03/stage.json')
for key in ['archive_sha256','manifest_sha256','objects','file_bytes','cache_source_sha256']:ck(stage[key]==cpustage[key],'reused authenticated runtime '+key)
ck(stage['model_loaded'] is False,'no model loaded by stage')
for name in ['runtime-before.json','runtime-ready.json','runtime-after.json']:
    rt=j(F/'worker'/name);s=rt['snapshot'];prefix=stage['staged_prefix']
    ck(s['prefix']==s['base_prefix']==prefix and s['executable']==prefix+'/bin/python','runtime prefix '+name)
    ck(s['python_version']=='3.10.19' and all(x.startswith(prefix+'/') for x in s['sys_path']),'runtime paths')
    ck(not s['mixed_modules'] and not s['mixed_module_paths'] and not s['mixed_native_paths'],'no observed mixed paths')
    ck(all(not s['packages'][x]['loaded'] for x in ['torch','transformers','scipy']),'no model packages')
    ck(rt['environment']['CUDA_VISIBLE_DEVICES']=='','CUDA hidden')
    for mod,v in rt['captured_project_modules'].items():
        name=Path(v['path']).name;ck(v['loader']=='authenticated_captured_bytes' and v['sha256']==pkt['files'][name],'captured runtime module '+mod)
        ck(v['path']==meta['input_directory']+'/'+name,'captured input location')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:ck(rt['environment'][key]=='1','thread bound')
phases=[j(p) for p in sorted(F.glob('phase-*.json'))]
ck(phases[0]['phase']=='stage_start' and phases[-1]['phase']=='stage_complete','staging complete')
ck(phases[-1]['elapsed_seconds']<120,'staging startup bounded')
done('new Slurm/absolute deadlines/owned cleanup/runtime; 122 CPU seconds, zero GPU; census 36 jobs / 74 journals')
ck(sha(C/'finalization.json')==sha(C/'finalization.candidate.json')==receiptpin,'one original finalization publication')
ck(sha(P/'summary.json')==summarypin,'independent summary pin')
receipt=j(C/'finalization.json');terminal=j(F/'worker/terminal.json');report=terminal['report'];s=j(P/'summary.json')
ck(receipt['attempt_pins']==ga['attempt_pins'] and len(receipt['attempt_pins'])==8,'eight accepted attempt pins')
for i,p in enumerate(s['roster']['policy_ids']):ck(sha(C/f'policy-{i:02d}/attempt.json')==receipt['attempt_pins'][p],'attempt pin '+p)
ck(report['campaign_files']=={k:v['sha256'] for k,v in inv(C).items()},'exact worker campaign output claim')
ck((report['cell_count'],report['descriptive_policy_comparisons'],report['failed_or_unresolved_cells'])==(16,28,0),'reported counts')
for v in [receipt,report]:ck(v['new_environment_responses']==v['new_training_updates']==0 and not v['scientific_acceptance'],'no new responses/updates/science')
ck(report['finalization_sha256']==receiptpin and report['summary_sha256']==summarypin,'worker output pins')
ck(terminal['status']=='descriptive_finalization_complete' and not terminal['retry_allowed'],'terminal disposition')
policies=s['roster']['policy_ids']; worlds=s['roster']['worlds'];cells=[];curves=[];computed={};actioncounts=collections.defaultdict(collections.Counter)
for wi,w in enumerate(worlds):
    ck(j(P/f'world-{wi:02d}/sealing-errors.json')=={},'no omitted sealing failures')
    for pi,p in enumerate(policies):
        a=P/f'world-{wi:02d}/arm-{pi:02d}'; original=C/f'policy-{pi:02d}/world-{wi:02d}/sealed-inputs'
        r=j(a/'cost-report.json');prov=j(a/'sealing-provenance.json');hist=j(a/'history.json')['attempts'];world=j(a/'world.json');task=j(a/'task.json')
        ck(prov['bundle_sha256']==r['input_sha256'],'report bundle separation')
        for name,pin in r['input_sha256'].items():ck(sha(a/name)==pin,'bundle pin '+name)
        for name,pin in prov['input_sha256'].items():ck(sha(original/name)==pin,'original sealing pin '+name)
        ck(prov['scientific_acceptance'] is False and prov['sealed_status']=='complete','sealed outcome')
        truth=[]
        for query in task['queries']:
            means={}; intervention=dict(query['targets'])
            for mechanism in world['mechanisms']:
                node=mechanism['node'];means[node]=intervention[node] if node in intervention else mechanism['intercept']+math.fsum(coef*means[parent] for parent,coef in mechanism['coefficients'])
            truth.append(means[task['target']])
        ck(truth==([-1.0,3.0] if wi==0 else [3.25,-.75]),'true query means')
        ck(task['weights']==[.5,.5] and task['scale']==r['scale']==2 and task['normalized_mse_tolerance']==r['threshold']==.04,'fixed metric')
        ck(all(x['outcome']=='accepted' for x in hist),'all actual histories retained/accepted')
        initial=[x for x in hist if x['phase']=='initial'];acq=[x for x in hist if x['phase']=='acquisition']
        ck(len(initial)==2 and len(initial)+len(acq)==len(hist),'complete history phases')
        for entry in hist:
            rr=j(a/entry['evidence_file']);raw=j(original/f"return-{entry['request_id']:03d}.json")
            ck(rr['values']==raw['response']['raw_json'] and raw['returned_to_wrapper'],'saved raw response equality')
            ck(rr['action_id']==entry['action_id']==raw['action']['action_id'] and rr['cost_units']==entry['cost_units']==raw['action']['cost_units'],'action charges')
        charges=r['charges'];expected={'accepted_responses':len(hist),'reserved_attempts':len(hist),'known_returns':len(hist),'unknown_returns':0,'actual_total_cost_units':sum(x['cost_units'] for x in hist),'actual_initial_cost_units':sum(x['cost_units'] for x in initial),'actual_acquisition_cost_units':sum(x['cost_units'] for x in acq),'known_return_cost_units':sum(x['cost_units'] for x in hist)}
        for key,v in expected.items():ck(charges[key]==v,'charge '+key)
        ck([expected[x] for x in ['actual_initial_cost_units','actual_acquisition_cost_units','actual_total_cost_units']]==[2,4,6],'full actual spend')
        for phase,entries in [('initial',initial),('acquisition',acq)]:ck(charges['by_phase'][phase]=={'attempts':len(entries),'cost_units':sum(x['cost_units'] for x in entries)},'phase charge')
        for outcome in ['accepted','returned_invalid','returned_unaccepted','unknown']:ck(charges['by_outcome'][outcome]==({'attempts':len(hist),'cost_units':6} if outcome=='accepted' else {'attempts':0,'cost_units':0}),'all outcomes charged')
        curve=r['curve'];ck(len(curve)==1+len(acq),'all prefixes retained');running=0;errors=[]
        for k,point in enumerate(curve):
            if k:running+=acq[k-1]['cost_units'];ck(point['request_id']==acq[k-1]['request_id'] and point['action_id']==acq[k-1]['action_id'],'curve/receipt alignment')
            ck(point['acquisition_cost_units']==running and point['accepted_receipts']==2+k,'curve prefix cost/count')
            error=math.fsum(((pred-true)/task['scale'])**2 for pred,true in zip(point['predictions'],truth))/len(truth)
            ck(len(point['predictions'])==len(truth),'all query predictions');close(error,point['normalized_mse'],'recomputed saved NMSE');errors.append(error)
        first=next((x['acquisition_cost_units'] for x,e in zip(curve,errors) if e<=r['threshold']),None);cap=first if first is not None else 4
        ck(r['first_crossing_cost_units']==first and r['capped_cost_units']==cap and r['attained']==(first is not None) and r['final_attained']==(errors[-1]<=.04),'first versus final attainment')
        ck(r['capped_cost_reason']==('first_crossing' if first is not None else 'nonattainment'),'nonattainment retained');close(r['final_error'],errors[-1],'final error')
        ck(r['operational_status']=='complete' and r['report_status']=='reported' and r['evaluation_error'] is None and r['curve_complete'],'curve status')
        ck(r['new_environment_responses']==r['policy_updates']==0 and not r['private_scores_authorize_policy_stop'],'private retrospective only')
        row=next(x for x in s['rows'] if (x['world_id'],x['policy_id'])==(w['world_id'],p))
        ck(row['charges']==charges and not row['failed_or_unresolved'],'row charges/failures')
        ck(row['first_crossing_acquisition_cost_units']==first and row['first_crossing_total_cost_units']==(first+2 if first is not None else None) and row['capped_acquisition_cost_units']==cap and row['capped_total_cost_units']==cap+2,'summary endpoints')
        actioncounts[p].update(x['action_id'] for x in acq)
        computed[w['world_id'],p]={'cap':cap,'first':first,'final_error':errors[-1],'final_attained':errors[-1]<=.04}
        curves.append({'world_id':w['world_id'],'policy_id':p,'true_query_means':truth,'recomputed_errors':errors,'first_crossing_acquisition':first,'capped_acquisition':cap,'final_attained':errors[-1]<=.04,'charges':expected})
        cells.append({'world_id':w['world_id'],'policy_id':p,'kind':'report','raw_report':(a/'cost-report.json').read_bytes(),'reason':None})
ck(len(curves)==16,'all 16 curves')
ck([computed[9101,p]['first'] for p in policies]==[1,4,4,2,1,2,2,None],'world9101 adverse DPO nonattainment')
ck(all(computed[9102,p]['first']==0 for p in policies),'world9102 all initially attained')
ck([p for p in policies if computed[9102,p]['final_attained']]==['target_risk','learned_ranker','sft_llm'],'world9102 only three terminal attainments')
ck(all(actioncounts[p]=={'observe_all':7,'x_plus':1} for p in ['frozen_llm','sft_llm','dpo_llm']),'LLM action counts only, no trajectory equivalence inference')
ck(sum(x['charges']['accepted_responses'] for x in curves)==92,'92 logical response receipts')
ck(sum(x['charges']['actual_total_cost_units'] for x in curves)==96,'96 logical total cost units')
done('16 saved prediction curves independently recomputed; all receipt charges; adverse final attainment and DPO nonattainment retained')
ck(len(s['policy_pairs'])==28,'28 policy pairs')
for pair,(left,right) in zip(s['policy_pairs'],itertools.combinations(policies,2)):
    differences=[computed[w['world_id'],left]['cap']-computed[w['world_id'],right]['cap'] for w in worlds]
    ck((pair['left'],pair['right'])==(left,right),'pair roster/order')
    ck(pair['acquisition_cost_differences']==pair['total_cost_differences']==differences,'all-world pair differences')
    ck(pair['mean_acquisition_cost_difference']==pair['mean_total_cost_difference']==sum(differences)/2,'pair mean')
    ck(pair['left_wins_lower_cost']==sum(x<0 for x in differences) and pair['left_losses_higher_cost']==sum(x>0 for x in differences) and pair['ties']==sum(x==0 for x in differences),'pair wins/losses/ties')
    ck(pair['worlds_with_failed_cell']==pair['worlds_with_unresolved_cell']==0,'pair failure counts')
helper=L/'inputs/llm_intervention_screen_reporting.py'
ck(sha(helper)=='4c324e9d11ce8fc77f796e22a1c4b81234219bf4d108d9922c2497e484b63c8c','pure reporting source pin')
module=types.ModuleType('independent_saved_screen_consistency');exec(compile(helper.read_bytes(),str(helper),'exec'),module.__dict__)
recomputed=module.summarize_screen(roster=s['roster'],cells=cells,report_pins=j(P/'report-pins.json'))
ck(recomputed==s,'exact full saved summary, all policies/pairs/strata via existing pure consistency helper')
ck(not any(name in sys.modules for name in ['numpy','torch','scipy','transformers']),'review uses no numerical/model libraries')
done('28 independent pair calculations and exact full summary/strata consistency, no favorable filtering')
result={'status':'passed','checks':checks,'curves':curves,'pairs':s['policy_pairs'],'action_counts':dict(actioncounts),'raw_files':len(actual),'raw_bytes':sum(x['bytes'] for x in actual.values()),'original_common_files_preserved':len(oldcommon),'curve_points':sum(len(x['recomputed_errors']) for x in curves),'logical_accepted_responses':92,'logical_actual_initial_cost_units':32,'logical_actual_acquisition_cost_units':64,'logical_actual_total_cost_units':96,'closed_cpu_seconds':10368,'closed_gpu_slice_seconds':1483,'cpu_admission_with_preparation_allowance':13968,'terminal_jobs':36,'submission_journals':74,'batch_max_rss_kib':int(rows[1][9][:-1]),'staging_start_utc':phases[0]['utc'],'staging_complete_utc':phases[-1]['utc'],'finalization_work_start':workstart,'outer_attempt':{k:v for k,v in attempt.items() if k!='result'},'process_cpu_seconds':time.process_time()-started,'source_sha256':sha(Path(__file__))}
(OUT/'checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['curves','pairs','action_counts','outer_attempt','finalization_work_start']},indent=2))
