#!/usr/bin/env python3
"""Read saved JSON timing/count fields only. No project imports or computation replay.
Every consulted byte is snapshotted and pinned; output directory must be absent.
Prospective counts are explicitly declared inputs, not measured scientific results.
"""
import argparse, datetime, hashlib, json, pathlib, statistics
P=argparse.ArgumentParser();P.add_argument('--repo',type=pathlib.Path,required=True);P.add_argument('--output',type=pathlib.Path,required=True);a=P.parse_args()
a.output.mkdir(exist_ok=False); pins={}
def sha(b): return hashlib.sha256(b).hexdigest()
def read(rel,decode=True):
    p=a.repo/rel; b=p.read_bytes(); pin={'sha256':sha(b),'bytes':len(b)}
    if rel in pins: assert pins[rel]==pin, 'input changed during read'
    else:
        dest=a.output/'inputs'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(b)
    pins[rel]=pin
    return json.loads(b) if decode else b

def files(root,pat): return [str(p.relative_to(a.repo)) for p in sorted((a.repo/root).glob(pat))]
def rows(root,pat): return [read(p) for p in files(root,pat)]
def stats(x):
    assert x
    return dict(n=len(x),sum=sum(x),min=min(x),mean=statistics.mean(x),max=max(x))
def phases(root):
    return [{k:d[k] for k in ('phase','elapsed_seconds','ordinal','policy_id','training_updates','fixture_count') if k in d} for d in rows(root,'phase-*.json')]
def at(ph,name,policy=None):
    x=[d['elapsed_seconds'] for d in ph if d['phase']==name and (policy is None or d.get('policy_id')==policy)]
    assert len(x)==1,(name,policy,x);return x[0]
def stage_span(root):
    ph=phases(root);return at(ph,'stage_complete')-at(ph,'stage_start')
def save(name,obj):
    with (a.output/name).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
R='results/'
C=R+'ace_llm_intervention_campaign_artificial_20261108/cpu-terminal03/remote/cpu03'
G=R+'ace_llm_intervention_campaign_gpu_20261108/terminal01/remote/common/gpu03'
F=R+'ace_llm_intervention_campaign_finalization_20261108/terminal01/remote/launch/finalization'
T=R+'ace_llm_intervention_training_cpu_20261108/terminal01/remote/qualification'
K=R+'ace_llm_intervention_screen_prompt_cpu_20261108/terminal01/remote/launch/cpu01'
H=R+'ace_llm_intervention_screen_training_gpu_20261108/terminal01/remote/qualification'
accept_paths={
'cpu03':'ace_llm_intervention_campaign_artificial_20261108/cpu-acceptance03/acceptance.json',
'campaign_gpu':'ace_llm_intervention_campaign_gpu_20261108/acceptance01/acceptance.json',
'finalizer':'ace_llm_intervention_campaign_finalization_20261108/acceptance01/acceptance.json',
'training_cpu':'ace_llm_intervention_training_cpu_20261108/acceptance01/acceptance.json',
'prompt_cpu':'ace_llm_intervention_screen_prompt_cpu_20261108/acceptance01/acceptance.json',
'shape_gpu':'ace_llm_intervention_screen_training_gpu_20261108/acceptance01/acceptance.json'}
accept={k:read(R+p) for k,p in accept_paths.items()}
for k,d in accept.items(): assert 'accepted' in d['status'],k
contract='docs/development/guidance/ace_llm_intervention_scientific_contract_2026-11-08.md'
read(contract,False)
read(R+'ace_llm_intervention_screen_training_gpu_20261108/resource-projection01/guide.md',False)
read(R+'ace_llm_intervention_screen_training_gpu_20261108/resource-projection01/preparation.json')
# Operation ceilings from the draft, no data-dependent selection.
counts=dict(training_worlds=32,development_worlds=24,sft_records=160,all_pairs=3584,
    max_dpo=128,cpu_policies=5,cpu_decisions_per_policy=576,cpu_cells=120,
    llm_policies=3,llm_decisions_per_policy=576,all_cells=192,paired_contrasts=28,
    final_curve_points_max=4800,training_reference_forwards_max=768,
    training_legal_score_forwards=2368,training_generations=322,
    training_commands_max=1184,collection_response_attempts_max=9216,
    collection_response_cost_units_max=10240)
# CPU decision measurements, joined to inner child by input pin AND method.
art=C+'/worker/artifacts'; cp=phases(art); sup=read(C+'/supervisor.json')
methods=['uniform_random','menu_coverage','local_information','target_risk','learned_ranker']
inner={m:[] for m in methods}
for p in files(art,'numeric-calls/*/terminal.json'):
    d=read(p); cmd=read(str(pathlib.PurePosixPath(p).with_name('command.json')))
    argv=cmd['argv']
    m='learned_ranker' if cmd['entry']=='ranker' else json.loads(argv[argv.index('--binding-json')+1])['arm']
    assert d['status']=='complete';inner[m].append(d)
by={}; shape=[]
for i,m in enumerate(methods):
    paths=files(art,f'campaign/policy-{i:02d}/world-*/episode/decision-*.process.json')
    ds=[read(p) for p in paths];assert all(d['returncode']==0 for d in ds)
    assert sorted(d['input_sha256'] for d in ds)==sorted(d['input_sha256'] for d in inner[m])
    for p in paths:
        inp=read(p.replace('.process.json','.input.json'))
        shape.append((len(inp['state']['graph']),len(inp['state']['menu']),len(inp['state']['history']),len(inp['task']['queries'])))
    span=at(cp,'cpu_policy_complete',m)-at(cp,'cpu_policy_start',m)
    q=stats([d['elapsed_seconds'] for d in ds]);residual=span-q['sum'];assert residual>=0
    by[m]={'process_wall_seconds':q,'inner_wall_seconds':stats([d['elapsed_seconds'] for d in inner[m]]),
       'inner_cpu_seconds':stats([d['cpu_seconds'] for d in inner[m]]),'policy_phase_wall_seconds':span,
       'controller_residual_wall_seconds':residual,
       'scenario_576_mean_calls_plus_12x_residual_seconds':576*q['mean']+12*residual,
       'scenario_576_observed_max_calls_plus_12x_residual_seconds':576*q['max']+12*residual}
assert sum(v['process_wall_seconds']['n'] for v in by.values())==36
ranker=at(cp,'cpu_ranker_training_complete')-at(cp,'cpu_ranker_training_start')
policy_total=sum(v['policy_phase_wall_seconds'] for v in by.values())
fixed=sup['elapsed_seconds']-policy_total-ranker
cpu={'outer_wall_seconds':sup['elapsed_seconds'],'runtime_stage_wall_seconds':stage_span(C),
    'worker_resources':read(C+'/worker.resources.json'),'methods':by,
    'ranker_6_updates_phase_seconds':ranker,'ranker_288_updates_constant_per_update_sensitivity_seconds':ranker*288/6,
    'non_policy_non_ranker_outer_residual_seconds':fixed,
    'observed_public_shapes':{'nodes':sorted({x[0] for x in shape}),'menu':sorted({x[1] for x in shape}),
       'history_rows_min':min(x[2] for x in shape),'history_rows_max':max(x[2] for x in shape),'queries':sorted({x[3] for x in shape})}}
for mode in ['mean','observed_max']:
    cpu['scenario_'+mode+'_seconds']=fixed+ranker*288/6+sum(v[f'scenario_576_{mode}_calls_plus_12x_residual_seconds'] for v in by.values())
cpu['scenario_warning']='Process duration includes process startup. Residual scales by 24/2 worlds, not 576/calls; response-count-dependent residual may grow faster. All observed min/mean/max are sample statistics, not bounds. Ranker extrapolation from six updates lacks full-plan/cache/checkpoint validation cost.'
# Long-context GPU training: all six steps and all ten generations, no favorable subset.
fit=H+'/session/worker/artifacts/fit'; steps=rows(fit,'training/step-*.complete.json');assert len(steps)==6
stepstats={s:stats([d['wall_seconds'] for d in steps if d['stage']==s]) for s in ('sft','dpo')}
gens=rows(fit,'sft-generation-*.json')+rows(fit,'final-generation-*.json');assert len(gens)==10
cuda=stats([d['cuda_interval_ms']/1000 for d in gens]);first=read(fit+'/first-gpu-work.json')['cuda_interval_ms']/1000
hp=phases(H+'/session/worker/artifacts')
training={'step_wall_seconds':stepstats,'all_generation_cuda_seconds':cuda,'one_scoring_forward_cuda_seconds':first,
  'outer_measurements':accept['shape_gpu']['measurements'],
  'training_phase_wall_seconds':at(hp,'training_complete')-at(hp,'training_start'),
  'note':'Generation elapsed_seconds is cumulative journal elapsed, NOT per-generation wall time. CUDA timing is used explicitly; host overhead is unmeasured by it.'}
for coeff in ('mean','max'):
    training['mixed_wall_cuda_operation_proxy_'+coeff+'_seconds']=160*stepstats['sft'][coeff]+128*stepstats['dpo'][coeff]+(768+2368)*first+322*cuda[coeff]
training['prior_published_operation_proxy_seconds']=160*stepstats['sft']['max']+128*stepstats['dpo']['max']+(768+2368)*first+322*gens[0]['cuda_interval_ms']/1000
training['prior_published_whole_stage_sensitivity_seconds']=(training['prior_published_operation_proxy_seconds']+2*accept['shape_gpu']['allocation']['elapsed_seconds'])*1.25
training['all_observed_max_generation_whole_stage_same_margin_sensitivity_seconds']=(training['mixed_wall_cuda_operation_proxy_max_seconds']+2*accept['shape_gpu']['allocation']['elapsed_seconds'])*1.25
# Persistent GPU service split: exact session - sum generations - inter-completion gaps.
gpu={}; total_session=0
for m in ('frozen_llm','sft_llm','dpo_llm'):
    sm=read(G+'/controller/phase/'+m+'/summary.json');wp=G+'/services/'+m+'/worker'
    gs=rows(wp,'generation-*.json');assert len(gs)==8
    ph=phases(wp);complete={d['ordinal']:d['elapsed_seconds'] for d in ph if d['phase']=='generation_complete'}
    gs=sorted(gs,key=lambda d:d['ordinal']);assert list(complete)==list(range(8))
    gaps=[complete[i]-complete[i-1]-gs[i]['elapsed_seconds'] for i in range(1,8)];assert min(gaps)>=0
    duration=sm['closed_monotonic']-sm['started_monotonic'];total_session+=duration
    wall=stats([d['elapsed_seconds'] for d in gs]);gapstats=stats(gaps)
    fixedtail=duration-wall['sum']-gapstats['sum'];assert fixedtail>0
    out={'session_wall_seconds':duration,'generation_wall_seconds':wall,'generation_cuda_seconds':stats([d['cuda_interval_ms']/1000 for d in gs]),
      'non_generation_total_seconds':duration-wall['sum'],'between_generation_non_generation_gap_seconds':gapstats,
      'start_and_tail_residual_seconds':fixedtail,'model_ready_worker_offset_seconds':at(ph,'model_ready'),
      'prompt_tokens':stats([d['prompt_tokens'] for d in gs])}
    for mode in ('mean','max'):
        out['scenario_576_same_short_shape_'+mode+'_seconds']=fixedtail+576*wall[mode]+575*gapstats[mode]
        out['scenario_576_long_shape_cuda_plus_saved_host_delta_'+mode+'_seconds']=fixedtail+576*(cuda[mode]+max(d['elapsed_seconds']-d['cuda_interval_ms']/1000 for d in gs))+575*gapstats[mode]
    gpu[m]=out
outer=read(G+'/terminal.json')['elapsed_seconds']
gpu['outer_wall_seconds']=outer;gpu['stage_seconds']=stage_span(G);gpu['outside_three_sessions_seconds']=outer-total_session
for mode in ('mean','max'):
    gpu['scenario_all_services_short_'+mode+'_seconds']=outer-total_session+sum(gpu[m]['scenario_576_same_short_shape_'+mode+'_seconds'] for m in ('frozen_llm','sft_llm','dpo_llm'))
    gpu['scenario_all_services_long_'+mode+'_seconds']=outer-total_session+sum(gpu[m]['scenario_576_long_shape_cuda_plus_saved_host_delta_'+mode+'_seconds'] for m in ('frozen_llm','sft_llm','dpo_llm'))
gpu['warning']='Completion gaps include controller transport, public-state construction, response journals and world transitions; they are NOT pure fixed startup. Short-shape samples cannot bound 3-node 8-menu 32-row 8-query requests. Long-shape CUDA is a different saved worker with the same recipe, not a measured persistent-service duration. No throughput guarantee or universal max. Reserve600/cap1000 may bind before total2700.'
# Finalizer distinguishes one-time readiness/staging from actual report interval.
hand=read(F+'/handoff.json');start=read(F+'/worker/work-start.json');term=read(F+'/worker/terminal.json');resource=read(F+'/worker.resources.json');attempt=read(F+'/attempt.json')
work=(datetime.datetime.fromisoformat(term['completed_utc'])-datetime.datetime.fromisoformat(start['utc'])).total_seconds()
ready=start['monotonic']-hand['started_monotonic'];reportroot=R+'ace_llm_intervention_campaign_finalization_20261108/terminal01/remote/common/cpu03/worker/artifacts/campaign'
curves=rows(reportroot,'private/world-*/arm-*/cost-report.json');assert len(curves)==16
points=sum(len(d['curve']) for d in curves);assert term['report']['cell_count']==len(curves)
final={'outer_wall_seconds':attempt['elapsed_seconds'],'runtime_stage_seconds':stage_span(F),'readiness_offset_seconds':ready,
    'report_through_terminal_utc_seconds':work,'work_start_through_child_cleanup_monotonic_seconds':resource['elapsed_seconds']-ready,
    'after_terminal_to_outer_end_seconds':attempt['elapsed_seconds']-ready-work,
    'cells':len(curves),'curve_points':points,'curve_points_per_cell':stats([len(d['curve']) for d in curves]),
    'cell_count_linear_scenario_seconds':attempt['elapsed_seconds']-work+work*192/len(curves),
    'curve_point_linear_scenario_seconds':attempt['elapsed_seconds']-work+work*4800/points,
    'warning':'UTC report interval includes after-report validation/publication. Monotonic cleanup interval is a cross-check, not pure evaluator time. 4800 points means192*(initial+24 acquisitions); reports use only actually retained points. 8 vs2 queries,3 vs2 nodes and larger input inventories are unmeasured multiplicative costs; curve-point estimate is not a bound.'}
# Preparation stages. Timer origins differ; never subtract unrelated worker clocks.
tp=phases(T+'/session/worker/artifacts');tt=read(T+'/terminal.json');kp=phases(K+'/worker/artifacts');ka=accept['prompt_cpu']
load_token=at(kp,'cpu_tokenization_complete')-at(kp,'cpu_tokenizer_load_start')
prep={'training_cpu_4_record_outer_wall_seconds':tt['elapsed_seconds'],'training_cpu_runtime_stage_seconds':stage_span(T),
    'training_cpu_session':tt['session'],'training_cpu_import_seconds':at(tp,'service_cpu_import_complete')-at(tp,'service_cpu_import_start'),
    'training_cpu_load_plus_tokenization_seconds':at(tp,'cpu_tokenization_complete')-at(tp,'cpu_tokenizer_load_start'),
    'prompt_cpu_120_fixture_load_plus_tokenization_seconds':load_token,
    'prompt_cpu_import_seconds':at(kp,'service_cpu_import_complete')-at(kp,'service_cpu_import_start'),
    'prompt_cpu_worker_resources':read(K+'/worker.resources.json'),
    'accepted_token_counts':{k:ka[k] for k in ('fixtures','command_sequences','max_prompt_tokens','max_command_plus_eos_tokens','max_prompt_plus_reserve','fits','overflows','failures')},
    'load_plus_tokenization_160_record_linear_sensitivity_seconds':load_token*160/ka['fixtures'],
    'warning':'Combined load+tokenization scaling overcounts fixed tokenizer load but omits complete3584-pair verification, larger input custody and plan assembly. Accepted120-fixture maximum4744 is representative only. Scheduler RSS7677.844MiB/7680 gives effectively no proven headroom; retained CPU allocation is a bounded attempt, not measured full-stage memory capacity.'}
# Absence is explicit: no scientific collection primitive timers in these accepted inputs.
limits=['No source import, test, tokenizer, model, fit, world realization, response or remote action was run.',
 'All operation ceilings are draft constants, not scientific observed counts or admission authorization.',
 'No accepted measurement here separates full scientific realization56 descriptors or collection9216 responses/3584pairs. CPU numerical policy-call timings are not collection-branch throughputs;300/900 caps remain unsupported by equivalent-demand observations.',
 '2-node3-menu2-query artificial decisions with short histories do not bound3-node8-menu8-query32-row work. No arbitrary shape multiplier is certified.',
 'Mean/max scenarios describe sample coefficients; max is not a finite upper bound on future floats, timings or token lengths.',
 'No uncertainty is cured by private outcomes, discarding failures, retries, altered science or scaling whole staging-heavy jobs.',
 'Public-state/query/report counts only are extracted. Decision identities, values, losses, tokens and outcomes are not emitted in derived receipt. Original bytes are retained solely as pinned evidence.',
 'The prior published training equation uses first generation CUDA coefficient; all-ten mean/max sensitivities are additionally reported. No fitted pilot was performed.']
result={'schema':'ace-saved-artificial-timing-projection-v1','scientific_qualified':False,'throughput_guaranteed':False,'resource_admission_authorized':False,
    'counts_source':contract,'prospective_counts':counts,'accepted_input_status':{k:d['status'] for k,d in accept.items()},
    'cpu_campaign':cpu,'training_gpu':training,'gpu_campaign':gpu,'finalization':final,'cpu_preparation':prep,'limitations':limits,
    'budget':{'closed_cpu_seconds':11040,'closed_gpu_seconds':1601,'preparation_allowance_cpu_seconds':3600,
       'future_cpu_seconds':37800,'future_gpu_seconds':5400,'projected_total_cpu_seconds':52440,'projected_total_gpu_seconds':7001,
       'remaining_cpu_seconds':5160,'remaining_gpu_seconds':199},'script_sha256':sha(pathlib.Path(__file__).read_bytes())}
# Verify all sources unchanged after extraction, including the draft being edited elsewhere.
for rel,pin in pins.items():assert sha((a.repo/rel).read_bytes())==pin['sha256'], 'changed input '+rel
save('input-pins.json',pins); result['input_pins_sha256']=sha((a.output/'input-pins.json').read_bytes());save('receipt.json',result)
print(json.dumps({'receipt_sha256':sha((a.output/'receipt.json').read_bytes()),'input_files':len(pins),
 'cpu_campaign_mean_seconds':cpu['scenario_mean_seconds'],'cpu_campaign_sample_max_seconds':cpu['scenario_observed_max_seconds'],
 'gpu_campaign_mean_long_seconds':gpu['scenario_all_services_long_mean_seconds'],'gpu_campaign_max_long_seconds':gpu['scenario_all_services_long_max_seconds'],
 'finalizer_point_scenario_seconds':final['curve_point_linear_scenario_seconds'],'training_all_max_margin_seconds':training['all_observed_max_generation_whole_stage_same_margin_sensitivity_seconds']},sort_keys=True))
