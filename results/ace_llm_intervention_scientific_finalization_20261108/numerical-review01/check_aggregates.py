"""Independent aggregation/export check over the already pinned saved reports."""
from pathlib import Path
import collections, csv, hashlib, itertools, json, math, resource, time, traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RET=ROOT/'report-retrieval01'
PRIVATE=RET/'opaque/cpu/cpu/worker/artifacts/campaign/private'
DISPLAY=ROOT/'reporting01/data01'
POLS=['uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm']
LABELS=['Uniform random','Menu coverage','Local information','Target risk','Numerical ranker','Frozen LLM','Supervised LLM','DPO LLM']
report={'checks':0,'errors':[],'scope':'saved report aggregates, strata, all pairs and extraction outputs',
 'source_imports':0,'models_loaded':0,'new_scores':0,'tests_rerun':0,'remote_actions':0}
def sha(b):return hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def check(ok,label):
    report['checks']+=1
    if not ok:report['errors'].append(label)
def eq(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a)==type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
def mean(v):return math.fsum(v)/len(v)

def main():
    first=j(HERE/'cells-checks.json');assert not first['errors']
    report['cell_check_sha256']=sha((HERE/'cells-checks.json').read_bytes())
    cells=first['cells'];by={(c['world_id'],c['policy_id']):c for c in cells};worldids=list(range(202611070100,202611070124))
    s=j(PRIVATE/'summary.json');assert sha((PRIVATE/'summary.json').read_bytes())==first['summary_sha256']
    def source_policy(p,ids):
        cs=[by[w,p] for w in ids];av=[c['capped_acquisition_cost_units'] for c in cs];tv=[8+x for x in av];ev=[c['final_error'] for c in cs]
        charges={}
        for phase in ('acquisition','initial','total'):
            v=[c['charges']['actual_'+phase+'_cost_units'] for c in cs]
            charges[phase]={'vector':v,'available_cells':len(cs),'reported_cells_subtotal':sum(v),'all_world_sum':sum(v)}
        return {'policy_id':p,'world_ids':ids,'world_count':len(ids),'failed':sum(c['failed'] for c in cs),'unresolved':0,
          'failed_or_unresolved':sum(c['failed'] for c in cs),'attained':sum(c['attained'] for c in cs),
          'attained_without_failure':sum(c['attained'] and not c['failed'] for c in cs),
          'attained_on_failed_prefix':sum(c['attained'] and c['failed'] for c in cs),
          'not_attained':sum(not c['attained'] for c in cs),'attainment_unresolved':0,
          'capped_acquisition_cost_vector':av,'capped_acquisition_cost_sum':sum(av),'mean_capped_acquisition_cost':mean(av),
          'declared_initial_cost_vector':[8]*len(cs),'capped_total_cost_vector':tv,'capped_total_cost_sum':sum(tv),'mean_capped_total_cost':mean(tv),
          'final_error_vector':ev,'available_final_errors':len(cs),'unavailable_final_errors':0,'all_world_mean_final_error':mean(ev),
          'logical_per_arm_actual_charges':charges}
    def source_pair(left,right,ids):
        diffs=[by[w,left]['capped_acquisition_cost_units']-by[w,right]['capped_acquisition_cost_units'] for w in ids]
        return dict(left=left,right=right,world_ids=ids,difference_definition='left_minus_right_capped_cost',
          acquisition_cost_differences=diffs,total_cost_differences=diffs,mean_acquisition_cost_difference=mean(diffs),mean_total_cost_difference=mean(diffs),
          left_wins_lower_cost=sum(v<0 for v in diffs),ties=diffs.count(0),left_losses_higher_cost=sum(v>0 for v in diffs),
          worlds_with_failed_cell=sum(by[w,left]['failed'] or by[w,right]['failed'] for w in ids),worlds_with_unresolved_cell=0,
          comparison_scope='all roster worlds, including failed/unresolved cap endpoints; descriptive, not evidence of success')
    def check_summary(x,ids,label):
        check(eq(x['policy_summaries'],[source_policy(p,ids) for p in POLS]),label+' all8 policy fields/vectors')
        check(eq(x['policy_pairs'],[source_pair(a,b,ids) for a,b in itertools.combinations(POLS,2)]),label+' all28 pair fields/vectors')
    check_summary(s,worldids,'full roster')
    groups=[('all','All 24 systems',worldids)]
    for field,values in [('regime',['matched_low','matched_high','shifted_low','shifted_high']),('graph_stratum',['chain','fork','collider']),('cost_stratum',['uniform','heterogeneous'])]:
        for value in values:
            ids=[w for w in worldids if by[w,POLS[0]]['strata'][field]==value]
            groups.append((field+'/'+value,field+': '+value,ids))
    check(len(s['strata'])==9,'exact9 declared stratified summaries')
    for actual,(key,label,ids) in zip(s['strata'],groups[1:]):
        f,v=key.split('/');check(actual['field']==f and actual['value']==v and actual['world_ids']==ids,'stratum exact members '+key)
        check(len(ids)=={'regime':6,'graph_stratum':8,'cost_stratum':12}[f],'stratum size '+key)
        check_summary(actual,ids,key)
    for wi,w in enumerate(worldids):
        ws=j(PRIVATE/f'world-{wi:02d}/summary.json')
        check(ws['cell_count']==8 and ws['strata']==[] and ws['roster']['policy_ids']==POLS and len(ws['roster']['worlds'])==1 and ws['roster']['worlds'][0]['world_id']==w,'world summary roster '+str(w))
        check_summary(ws,[w],'world '+str(w))
    def policy_display(p,ids):
        x=source_policy(p,ids);cs=[by[w,p] for w in ids]
        return dict(policy_id=p,label=LABELS[POLS.index(p)],world_count=len(ids),mean_capped_total_cost=x['mean_capped_total_cost'],
          mean_capped_acquisition_cost=x['mean_capped_acquisition_cost'],ever_attained=x['attained'],final_attained=sum(c['final_attained'] for c in cs),
          mean_final_error=x['all_world_mean_final_error'],failed=x['failed'],unresolved=0,actual_total_cost=x['logical_per_arm_actual_charges']['total']['all_world_sum'],
          stopped=sum(c['termination_reason']=='policy_stop' for c in cs))
    def pair_display(a,b,ids):
        x=source_pair(a,b,ids)
        return dict(left=a,right=b,mean_difference=x['mean_total_cost_difference'],wins=x['left_wins_lower_cost'],ties=x['ties'],losses=x['left_losses_higher_cost'],world_ids=ids,differences=x['total_cost_differences'])
    dp=[policy_display(p,worldids) for p in POLS]
    pairs=[pair_display(a,b,worldids) for a,b in itertools.combinations(POLS,2)]
    dg=[dict(id=k,label=label,world_ids=ids,policies=[policy_display(p,ids) for p in POLS],pairs=[pair_display(a,b,ids) for a,b in itertools.combinations(POLS,2)]) for k,label,ids in groups]
    dc=[dict(world_id=c['world_id'],policy_id=c['policy_id'],**c['strata'],capped_total_cost=c['capped_total_cost_units'],capped_acquisition_cost=c['capped_acquisition_cost_units'],
        final_error=c['final_error'],ever_attained=c['attained'],final_attained=c['final_attained'],failed=c['failed'],unresolved=False,
        actual_total_cost=c['charges']['actual_total_cost_units'],termination_reason=c['termination_reason'],first_crossing_total=c['first_crossing_total_cost_units']) for c in cells]
    dcurves=[dict(world_id=p['world_id'],policy_id=p['policy_id'],total_cost_units=8+p['acquisition_cost_units'],acquisition_cost_units=p['acquisition_cost_units'],
       normalized_mse=p['normalized_mse'],action_id=p['action_id'],accepted_receipts=p['accepted_receipts'],**{'prediction_'+str(i):v for i,v in enumerate(p['predictions'])}) for p in first['curves']]
    actions=[];action_maps={};legal={'observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy'}
    for p in POLS:
        counter=collections.Counter(a for c in cells if c['policy_id']==p for a in c['actions'])
        check(set(counter)<=legal,'fixed action ids '+p);action_maps[p]=dict(counter)
        actions.extend(dict(policy_id=p,action_id=a,accepted_acquisitions=n) for a,n in sorted(counter.items()))
    desired={'scope':'24 synthetic development systems; descriptive, not confirmation','source_summary_sha256':first['summary_sha256'],
      'source_members_sha256':'012e5fc3f4994ff46b1776b2f50054084e080d946c4ad0780ae0b3dcde803ff4','finalizer_acceptance_sha256':first['acceptance_sha256'],
      'threshold':.04,'initial_cost':8,'total_cap':32,'worlds':[dict(id=w,index=i,**by[w,POLS[0]]['strata']) for i,w in enumerate(worldids)],
      'groups':dg,'cells':dc,'curves':dcurves,'actions':actions}
    actual=j(DISPLAY/'display-data.json');check(eq(actual,desired),'entire independent display-data reconstruction')
    deriv=j(DISPLAY/'derivation.json');extract=ROOT/'reporting01/extract_display_data.py'
    check(sha(extract.read_bytes())==deriv['script_sha256'],'saved extraction source pin (not executed)')
    check(deriv['source_members']==first['member_pins'],'extraction all242 member pins')
    output_pins={}
    for n,v in deriv['output_files'].items():
        b=(DISPLAY/n).read_bytes();check(sha(b)==v['sha256'] and len(b)==v['bytes'],'derived output pin '+n);output_pins[n]=v
    def parse_csv(n,expected):
        with (DISPLAY/n).open(newline='') as f:actual=list(csv.DictReader(f))
        check(len(actual)==len(expected),'CSV row count '+n)
        for i,(a,e) in enumerate(zip(actual,expected)):
            check(a.keys()==e.keys(),'CSV fields '+n+'/'+str(i))
            good=True
            for k,v in e.items():
                observed=a[k]
                if v is None:good &= observed==''
                elif isinstance(v,bool):good &= observed==str(v)
                elif isinstance(v,int):good &= int(observed)==v
                elif isinstance(v,float):good &= eq(float(observed),v)
                elif isinstance(v,(list,dict)):good &= eq(json.loads(observed),v)
                else:good &= observed==v
            check(good,'CSV values '+n+'/'+str(i))
    for n,data in [('policies.csv',dp),('cells.csv',dc),('curves.csv',dcurves),('pairs.csv',pairs),('strata.csv',[dict(stratum=g['label'],**p) for g in dg[1:] for p in g['policies']]),('actions.csv',actions)]:parse_csv(n,data)
    dpo={}
    for other in ('target_risk','uniform_random','frozen_llm','sft_llm','learned_ranker'):
        dpo[other]=pair_display('dpo_llm',other,worldids)
    diagnostic={'initial_attained_worlds':[x['world_id'] for x in first['initial'] if x['attained']],
      'total_logical_cost':sum(c['charges']['actual_total_cost_units'] for c in cells),'accepted_responses':sum(c['charges']['accepted_responses'] for c in cells),
      'sft_stops_above_final_target':sum(c['policy_id']=='sft_llm' and c['termination_reason']=='policy_stop' and not c['final_attained'] for c in cells),
      'actions':action_maps,'dpo_paired_contrasts':dpo,
      'regime_costs':{g['id']:{p['policy_id']:p['mean_capped_total_cost'] for p in g['policies']} for g in dg if g['id'].startswith('regime/')},
      'curve_points':len(dcurves),'rows':len(dc),'all_group_pairs':len(dg)*28,'all_world_summary_pairs':24*28}
    check(len(diagnostic['initial_attained_worlds'])==10 and diagnostic['total_logical_cost']==6097 and diagnostic['accepted_responses']==5884 and diagnostic['sft_stops_above_final_target']==5,'draft diagnostic counts')
    check(action_maps['dpo_llm']=={'set_x_neg':576} and action_maps['frozen_llm']=={'observe_all':574,'set_x_zero':2} and action_maps['sft_llm']['set_x_neg']==481 and sum(action_maps['sft_llm'].values())==523,'draft LLM action claims')
    expected_dpo={'target_risk':(2,13,9),'uniform_random':(3,13,8),'frozen_llm':(5,11,8),'sft_llm':(4,13,7),'learned_ranker':(2,14,8)}
    check(all((dpo[p]['wins'],dpo[p]['ties'],dpo[p]['losses'])==v for p,v in expected_dpo.items()),'draft all5 DPO contrasts')
    check(all(v['dpo_llm']>v['target_risk'] and v['dpo_llm']>v['frozen_llm'] for v in diagnostic['regime_costs'].values()),'draft all4 regime inequalities')
    report.update(diagnostics=diagnostic,policy_table=dp,all_group_summaries=dg,output_pins=output_pins,
       extraction_source_sha256=sha(extract.read_bytes()),derivation_sha256=sha((DISPLAY/'derivation.json').read_bytes()),
       float_comparison={'relative_tolerance':1e-12,'absolute_tolerance':1e-14,'independent_mean':'math.fsum / count'})

start=time.monotonic();cpu=time.process_time()
try:main()
except BaseException as e:
    report['errors'].append({'exception':type(e).__name__,'message':str(e)})
    (HERE/'aggregates-failure.traceback.txt').write_text(traceback.format_exc())
finally:
    report.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__).read_bytes()))
    with (HERE/'aggregate-checks.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:report.get(k) for k in ('checks','errors','diagnostics')}))
raise SystemExit(bool(report['errors']))
