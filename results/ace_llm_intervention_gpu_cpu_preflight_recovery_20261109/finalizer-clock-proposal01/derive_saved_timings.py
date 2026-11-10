"""Read saved public timing/custody metadata only. No producer imports or replay."""
from pathlib import Path
import ast,hashlib,json,datetime
ROOT=Path('/Users/pat/code/ACE');OUT=Path(__file__).resolve().parent
pins={}
def load(relative):
    p=ROOT/relative;b=p.read_bytes()
    pins[relative]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
    return json.loads(b)
prefix='results/ace_llm_intervention_scientific_campaign_gpu_20261108/'
failure=load(prefix+'failure-acceptance01/resource-reconciliation.json')
baseline=load(prefix+'failure-acceptance01/baseline.json')
accepted=load('results/ace_llm_intervention_scientific_resource_preparation_20261108/preparation01/saved-timing-analysis/attempt03/derived/receipt.json')
stage_prefix='results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/'
stage_start=load(stage_prefix+'phase-0000.json');stage_end=load(stage_prefix+'phase-0101.json')
observations={}
for name in ('inventory_transport.json','inventory_after_transport.json'):
    t=load(prefix+'terminal01/'+name);o=json.loads(t['stdout'])
    records=[json.loads(line) for line in t['stderr'].splitlines() if line.startswith('{')]
    samples={v['files_read']:v['elapsed_seconds'] for v in records if v.get('phase')=='terminal_inventory_verification'}
    assert len(samples)==len(records) and sorted(samples)==list(samples)
    parsed=ast.parse(t['python_source'])
    loop=next(n for n in ast.walk(parsed) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
              and [getattr(x,'id',None) for x in n.target.elts]==['label','tree','present'] and isinstance(n.iter,ast.Tuple))
    labels=[ast.literal_eval(n.elts[0]) for n in loop.iter.elts]
    sizes={label:sum(n.startswith(label+'/') for n in o['files']) for label in labels}
    offset=sum(sizes[label] for label in labels[:labels.index('cpu')]);count=sizes['cpu']
    # At most1000 adjacent reads are included at each edge. These are bounds on
    # this saved segment, not future runtime or the different finalizer reader.
    samples={0:0.0,**samples}
    begin=max(k for k in samples if k<=offset);end=min(k for k in samples if k>=offset+count)
    inner_begin=min(k for k in samples if k>=offset);inner_end=max(k for k in samples if k<=offset+count)
    observations[name]={
        'transport_utc':t['utc'],'observation_utc':o['utc'],'root_order':labels,'root_file_counts':sizes,
        'full_raw_files':len(o['files']),'progress_samples':len(records),
        'last_progress_files':max(samples),'last_progress_elapsed_seconds':samples[max(samples)],
        'cpu_root_read_count':count,'cpu_root_counter_interval':[offset+1,offset+count],
        'cpu_root_saved_segment_seconds_lower':samples[inner_end]-samples[inner_begin],
        'cpu_root_saved_segment_seconds_upper':samples[end]-samples[begin],
        'outer_sample_counters':[begin,end],
        'counter_scope':'root walk followed by separately pinned file/journal rereads; last sample is not a unique-file count or final completion timestamp'}
assert baseline['closed_cpu_seconds']==30241 and baseline['preparation_allowance_cpu_seconds']==3600
available=57600-30241-3600-4*2880
result={
    'schema':'ace-finalizer-clock-saved-evidence-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'new read-only timing derivation, not custody re-audit or workload replay',
    'gpu01_failure':{'job_id':failure['job_id'],'startup_alarm_seconds':failure['lifecycle']['startup_alarm_seconds'],
        'elapsed_seconds':failure['observed']['elapsed_seconds'],'allocated_cpu_seconds':failure['observed']['allocated_cpu_seconds'],
        'process_total_cpu_seconds':failure['observed']['process_total_cpu_seconds'],
        'no_complete_first_inventory_timing':True,'latency_cause_identified':False},
    'saved_login_inventory_observations':observations,
    'accepted_scientific_cpu_runtime_staging':{'elapsed_seconds':stage_end['elapsed_seconds']-stage_start['elapsed_seconds'],
        'process_cpu_seconds':stage_end['process_cpu_seconds']-stage_start['process_cpu_seconds'],
        'start_utc':stage_start['utc'],'end_utc':stage_end['utc']},
    'accepted_artificial_finalization':accepted['finalization'],
    'conservative_budget':{'closed_cpu_seconds':30241,'preparation_allowance_cpu_seconds':3600,
        'gpu02_max_cpu_reservation':11520,'available_cpu_seconds_for_finalizer_and_any_new_cpu_job':available,
        'existing_1200_cap_cpu_reservation':2400,'existing_cap_remaining_cpu_seconds':available-2400,
        'illustrative_4800_cap_cpu_reservation':9600,'illustrative_900_second_2cpu_measurement_reservation':1800,
        'illustrative_combined_remaining_cpu_seconds':available-9600-1800,
        'closed_gpu_seconds':2750,'gpu02_max_gpu_reservation':2880,'gpu_remaining_after_max':7200-2750-2880},
    'all_input_pins':pins,'allocation_authorized':False,'resource_admission_accepted':False,
    'scientific_qualified':False,'throughput_bound':False,
    'limits':['Login metadata reads are not a compute-node benchmark; the failed compute read did not complete.',
              'No GPU02 output inventory or actual report-demand count is supplied; those remain future.',
              '885-second reporting value is an old sensitivity, not a measured upper bound.',
              'No runtime staging, source execution, model or private outcome was performed.']}
(OUT/'saved-timing-evidence.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ('gpu01_failure','saved_login_inventory_observations','accepted_scientific_cpu_runtime_staging','conservative_budget')},indent=2))
