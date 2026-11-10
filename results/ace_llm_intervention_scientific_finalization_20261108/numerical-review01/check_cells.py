"""Independent saved JSON arithmetic only: no producer imports or world replay."""
from pathlib import Path
import collections, hashlib, json, math, os, resource, stat, time, traceback

HERE=Path(__file__).resolve().parent
RET=HERE.parent/'report-retrieval01'
PREFIX='cpu/cpu/worker/artifacts/campaign/private/'
POLICIES=['uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm']
ANCHORS={'handoff.json':'f95d744d29ad1fa4200b8b654e087cbcf5d7768013580b140c987fe6ebff8db9',
 'operation/receipt.json':'52f9cee616a67dec67a1ed6218b958dc4cf01399e1e8511edd4396a5f8c57e5b',
 'copy-receipt.json':'276213ae040aa5699b47699efe9c289aee23e49d0b60fd16f7ca713acee0e0c5'}
report={'scope':'all saved report bytes and per-cell arithmetic','checks':0,'errors':[],
 'models_loaded':0,'new_scores':0,'source_imports':0,'remote_actions':0,'old_tests_rerun':0}
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key '+k)
        d[k]=v
    return d
def decode(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def read(p,pin=None):
    before=p.lstat();assert stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=128*1024**2
    b=p.read_bytes();after=p.lstat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    if pin:assert sha(b)==pin
    return b
def check(ok,label):
    report['checks']+=1
    if not ok:report['errors'].append(label)
def number(v):return type(v) in (int,float) and math.isfinite(v)
def main():
    captured={n:read(RET/n,h) for n,h in ANCHORS.items()}
    hand=decode(captured['handoff.json']);receipt=decode(captured['operation/receipt.json']);copy=decode(captured['copy-receipt.json'])
    ext=Path(hand['external'])
    for n,b in captured.items():assert b==read(ext/n,ANCHORS[n])
    acceptance_pin='8ae40bd7952694271be0890590972250a5d8777f6856261b983b330fd999cb97'
    acceptance=decode(read(RET/'inputs/acceptance.json',acceptance_pin))
    assert acceptance['status']=='development_finalization_execution_accepted' and acceptance['job_id']=='35533604'
    assert hand['accepted_finalizer_sha256']==receipt['finalizer_acceptance_sha256']==acceptance_pin
    assert receipt['full_backup_verified'] and receipt['member_count']==242 and receipt['payload_bytes']==4403819
    members_raw=read(RET/'inputs/report-members.json',hand['members_sha256']);members=decode(members_raw)
    paths_raw=read(RET/'report-paths.json',hand['output_inventory_sha256']);paths=decode(paths_raw)['files']
    expected={PREFIX+'summary.json',PREFIX+'report-pins.json'}
    for w in range(24):
        expected|={PREFIX+f'world-{w:02d}/'+n for n in ('summary.json','report-pins.json')}
        expected|={PREFIX+f'world-{w:02d}/arm-{p:02d}/cost-report.json' for p in range(8)}
    assert set(members)==set(paths)==expected==set(receipt['local_inventory']['files'])
    objects={};byte_map={}
    for n in sorted(expected):
        v=members[n];p=RET/'opaque'/n;q=ext/'opaque'/n;b=read(p,v['sha256'])
        assert read(q,v['sha256'])==b and len(b)==v['bytes'] and not os.path.samestat(p.stat(),q.stat())
        assert paths[n]==dict(v,repo=str(p),external=str(q))
        assert receipt['local_inventory']['files'][n]==copy['files']['opaque/'+n]==v
        objects[n]=decode(b);byte_map[n]=dict(v)
    assert sum(v['bytes'] for v in members.values())==4403819
    s=objects[PREFIX+'summary.json'];assert acceptance['summary_sha256']==members[PREFIX+'summary.json']['sha256']
    roster=s['roster'];worlds=roster['worlds']
    assert roster['policy_ids']==POLICIES and roster['scope']=='development'
    assert roster['strata_fields']==['regime','graph_stratum','cost_stratum'] and len(worlds)==24
    assert [w['world_id'] for w in worlds]==list(range(202611070100,202611070124))
    assert len(s['rows'])==192 and len(s['policy_pairs'])==28
    allpins=objects[PREFIX+'report-pins.json'];assert len(allpins)==192
    pinindex={(r['world_id'],r['policy_id']):r['sha256'] for r in allpins};assert len(pinindex)==192
    rows={(r['world_id'],r['policy_id']):r for r in s['rows']};assert len(rows)==192
    cells=[];curve_count=0;initial=[];curves=[]
    regimes=['matched_low','matched_high','shifted_low','shifted_high'];graphs=['chain','fork','collider'];costs=['uniform','heterogeneous']
    for wi,w in enumerate(worlds):
        wid=w['world_id'];strata={'regime':regimes[wi//6],'graph_stratum':graphs[(wi%6)//2],'cost_stratum':costs[wi%2]}
        check(w['strata']==strata,'predeclared strata '+str(wid))
        check((w['initial_cost_units'],w['acquisition_cap_units'],w['scale'],w['threshold'])==(8,24,2.,.04),'fixed endpoints '+str(wid))
        ws=objects[PREFIX+f'world-{wi:02d}/summary.json'];wp=objects[PREFIX+f'world-{wi:02d}/report-pins.json']
        check(wp==[x for x in allpins if x['world_id']==wid],'world report pin subset '+str(wid))
        initial0=None
        for pi,policy in enumerate(POLICIES):
            label=f'{wid}/{policy}';n=PREFIX+f'world-{wi:02d}/arm-{pi:02d}/cost-report.json';r=objects[n];row=rows[wid,policy]
            check(r['world_id']==wid and r['policy_id']==policy and r['scope']=='development','report identity '+label)
            check(pinindex[wid,policy]==members[n]['sha256']==row['expected_report_sha256']==row['observed_report_sha256'],'report joins '+label)
            check(row['report_bytes']==members[n]['bytes'],'report size '+label)
            check(row['strata']==strata,'row stratum '+label)
            check(r['scale']==2. and r['threshold']==.04 and r['acquisition_cap_units']==24,'report endpoint law '+label)
            for k,f in [('world_sha256','world.json'),('task_sha256','task.json'),('learner_sha256','learner.json'),('shared_initial_sha256','shared-initial.json')]:
                check(r['input_sha256'][f]==w[k]==row['expected_inputs'][k],'paired input '+label+'/'+k)
            check(r['private_evaluator_only'] and not r['private_scores_authorize_policy_stop'] and r['new_environment_responses']==r['policy_updates']==0,'private saved-only scope '+label)
            c=r['charges'];curve=r['curve'];curve_count+=len(curve)
            check(all(number(p['normalized_mse']) and p['normalized_mse']>=0 and len(p['predictions'])==8 and all(number(x) for x in p['predictions']) for p in curve),'finite error/prediction curves '+label)
            check(c['actual_initial_cost_units']+c['actual_acquisition_cost_units']==c['actual_total_cost_units'],'charge total '+label)
            for group in ('by_phase','by_outcome'):
                check(sum(v['attempts'] for v in c[group].values())==c['reserved_attempts'],'attempt sum '+label+'/'+group)
                check(sum(v['cost_units'] for v in c[group].values())==c['actual_total_cost_units'],'cost sum '+label+'/'+group)
            check(c['by_phase']['initial']=={'attempts':8,'cost_units':8} and c['actual_initial_cost_units']==8,'shared initial charges '+label)
            check(c['known_returns']==c['reserved_attempts']-c['unknown_returns'] and c['accepted_responses']==c['by_outcome']['accepted']['attempts'],'return totals '+label)
            check(c['known_return_cost_units']==c['actual_total_cost_units']-c['by_outcome']['unknown']['cost_units'],'known charge total '+label)
            check(c['by_phase']['acquisition']['cost_units']==c['actual_acquisition_cost_units'],'acquisition charge phase '+label)
            # No failing outcomes were observed: still derive failed/capped status rather than assume attainment.
            check(r['operational_status']=='complete' and r['report_status']=='reported' and r['evaluation_error'] is None and r['curve_complete'],'complete/evaluable '+label)
            check(c['accepted_responses']==c['reserved_attempts']==c['known_returns'] and c['unknown_returns']==0,'all returns accepted '+label)
            check(len(curve)==1+c['accepted_responses']-8,'curve point count '+label)
            check(curve[0]['kind']=='shared_initial' and curve[0]['acquisition_cost_units']==0 and curve[0]['accepted_receipts']==8 and curve[0]['action_id'] is None and curve[0]['request_id'] is None,'initial curve point '+label)
            initial_evidence={'point':curve[0],'pins':[r['input_sha256'][f'receipt-{i:03d}.json'] for i in range(8)]}
            if initial0 is None:initial0=initial_evidence;initial.append({'world_id':wid,'error':curve[0]['normalized_mse'],'attained':curve[0]['normalized_mse']<=.04,'strata':strata})
            check(initial_evidence==initial0,'exact shared initial state/pins '+label)
            prev=0;actions=[]
            for i,p in enumerate(curve[1:],1):
                action=p['action_id'];cost=2 if strata['cost_stratum']=='heterogeneous' and (action.startswith('set_y_') or action=='set_xy') else 1
                check(p['acquisition_cost_units']==prev+cost and p['accepted_receipts']==8+i and p['request_id']==7+i and p['kind']=='accepted_acquisition','curve order/cost '+label+'/'+str(i))
                prev=p['acquisition_cost_units'];actions.append(action)
            check(prev==c['actual_acquisition_cost_units']<=24,'curve final cost '+label)
            crossing=next((p['acquisition_cost_units'] for p in curve if p['normalized_mse']<=.04),None)
            final=curve[-1]['normalized_mse'];failed=r['operational_status']=='failed' or r['evaluation_error'] is not None
            cap=24 if failed or crossing is None else crossing
            reason='operational_failure' if r['operational_status']=='failed' else 'evaluation_failure' if r['evaluation_error'] is not None else 'nonattainment' if crossing is None else 'first_crossing'
            expected_fields={'attained':crossing is not None,'final_attained':final<=.04,'final_error':final,'first_crossing_cost_units':crossing,'capped_cost_units':cap,'capped_cost_reason':reason}
            check(all(r[k]==v for k,v in expected_fields.items()),'recomputed cell endpoints '+label)
            expected_row={'attained':crossing is not None,'final_attained':final<=.04,'final_error':final,'first_crossing_acquisition_cost_units':crossing,'first_crossing_total_cost_units':None if crossing is None else 8+crossing,'capped_acquisition_cost_units':cap,'capped_total_cost_units':8+cap,'capped_cost_reason':reason,'failed':failed,'failed_or_unresolved':failed,'cell_status':'reported','declared_initial_cost_units':8,'charges':c}
            check(all(row[k]==v for k,v in expected_row.items()),'campaign row endpoints '+label)
            wr=next(x for x in ws['rows'] if x['policy_id']==policy)
            check({k:v for k,v in wr.items() if k!='strata'}=={k:v for k,v in row.items() if k!='strata'} and wr['strata']=={},'world/campaign rows '+label)
            cell=dict(world_id=wid,policy_id=policy,strata=strata,**expected_row,termination_reason=r['termination_reason'],actions=actions,curve_points=len(curve),report_sha256=members[n]['sha256'])
            cells.append(cell)
            curves.extend(dict(world_id=wid,policy_id=policy,**p) for p in curve)
    policies=[]
    for p in POLICIES:
        cs=[c for c in cells if c['policy_id']==p]
        policies.append({'policy_id':p,'worlds':len(cs),'mean_capped_total_cost':sum(c['capped_total_cost_units'] for c in cs)/24,
          'mean_capped_acquisition_cost':sum(c['capped_acquisition_cost_units'] for c in cs)/24,'ever_attained':sum(c['attained'] for c in cs),
          'final_attained':sum(c['final_attained'] for c in cs),'crossed_then_final_missed':sum(c['attained'] and not c['final_attained'] for c in cs),
          'mean_final_error':math.fsum(c['final_error'] for c in cs)/24,'actual_total_paid':sum(c['charges']['actual_total_cost_units'] for c in cs),
          'accepted_responses':sum(c['charges']['accepted_responses'] for c in cs),'failed':sum(c['failed'] for c in cs),
          'stops':sum(c['termination_reason']=='policy_stop' for c in cs)})
    report.update(input_pins=ANCHORS,member_pins=members,acceptance_sha256=acceptance_pin,summary_sha256=sha(read(RET/'opaque'/PREFIX/'summary.json')),
       cells=cells,curves=curves,policies=policies,initial=initial,curve_points=curve_count,
       limitations=['Saved normalized errors are inputs; no world/SCM replay or new ground-truth evaluation.',
       'Charge counters describe logical per-arm reservations, not physical caching or CPU/GPU time.'])

start=time.monotonic();cpu=time.process_time()
try:main()
except BaseException as e:
    report['errors'].append({'exception':type(e).__name__,'message':str(e)})
    (HERE/'cells-failure.traceback.txt').write_text(traceback.format_exc())
finally:
    report.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__).read_bytes()))
    with (HERE/'cells-checks.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:report.get(k) for k in ('checks','errors','policies','curve_points')}))
raise SystemExit(bool(report['errors']))
