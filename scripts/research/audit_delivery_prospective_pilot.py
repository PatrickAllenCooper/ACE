"""Accept a complete target-runtime pilot without opening numerical test losses."""
import argparse
import math
from pathlib import Path
from runner_delivery_confirmation import read,write,sha,utc


def audit(out,source,acceptance):
    out=Path(out);source=Path(source)
    p=read(out/'registration.json');projection=read(out/'projection.json')
    if projection['registration_sha256']!=sha(out/'registration.json'):
        raise ValueError('pilot registration changed')
    for name,h in p['worker_hashes'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('pilot worker changed')
    for name,h in p['source_hashes'].items():
        if sha(source/name)!=h:raise ValueError('original source changed')
    for key in ('dependencies','source_hashes','generator_hashes','worker_hashes'):
        if projection[key]!=p[key]:raise ValueError('projection provenance mismatch')
    if (not projection['complete'] or projection['test_losses_computed'] or
            projection['confirmation_responses_evaluated']!=0 or projection['development_responses']!=800 or
            p['development_coefficient_seed']!=0 or p['threads']!=1 or p['wall_seconds']!=900):
        raise ValueError('pilot scope/allowance mismatch')
    for phase in ('execution.json','qualification_execution.json'):
        r=read(out/phase)
        if r['status']!='complete' or r['exit_code']!=0 or r['peak_tree_rss_bytes']>3*2**30:
            raise ValueError('target-runtime qualification/execution failed')
    if read(out/'started.json')['registration_sha256']!=sha(out/'registration.json'):
        raise ValueError('pilot start changed')
    hashes={file:sha(out/file) for file in ('registration.json','projection.json','execution.json',
        'qualification_execution.json','qualification.log','started.json','supervisor_started.json','work.log')}
    recomputed={}
    for size in ('5','30'):
        data=read(out/size/'development_input.json')
        if len(data['rows'])!=400 or [r['query_index'] for r in data['rows']]!=list(range(1,401)):
            raise ValueError('development row set changed')
        expected_heads=3 if size=='5' else 25
        hashes[size+'/development_input.json']=sha(out/size/'development_input.json')
        if data['target']!=('X3' if size=='5' else 'X30'):raise ValueError('target changed')
        for arm in ('delivery','simpler','ablation','online'):
            cost=read(out/size/(arm+'_cost.json'))
            if cost['input_sha256']!=sha(out/size/'development_input.json') or cost['models_sha256']!=sha(out/size/(arm+'_models.pt')):
                raise ValueError('development input/model artifact changed')
            if {k:v for k,v in cost.items() if k not in ('input_sha256','models_sha256')}!=projection['timings'][size][arm]:
                raise ValueError('timing record changed')
            if (not cost['complete'] or cost['evaluation_responses_read']!=0 or cost['new_queries_in_fit']!=0 or
                    not math.isfinite(cost['cpu_seconds']) or cost['cpu_seconds']<=0 or
                    len(cost['heads'])!=(1 if arm=='simpler' else expected_heads)):
                raise ValueError('incomplete/invalid measured cell')
            updates={'delivery':2000,'simpler':2000,'ablation':100,'online':1200}[arm]
            if any(h['updates']!=updates or h['eligible_rows']!=400 for h in cost['heads'].values()):
                raise ValueError('development update/eligibility counts changed')
            for file in (arm+'_cost.json',arm+'_models.pt'):hashes[size+'/'+file]=sha(out/size/file)
        v=projection['timings'][size]
        recomputed[size]=40*(3*(v['delivery']['cpu_seconds']+v['simpler']['cpu_seconds'])*15
                            +v['ablation']['cpu_seconds']+v['online']['cpu_seconds']*40)
        if not math.isclose(recomputed[size],projection['strata'][size]['raw_full_fit_cpu_seconds'],rel_tol=1e-12):
            raise ValueError('stratum projection does not reproduce')
    estimate=(3*sum(recomputed.values())+640*5+5000+900)/3600
    if not math.isclose(estimate,projection['estimated_full_cpu_core_hours'],rel_tol=1e-12):
        raise ValueError('full640fit projection does not reproduce')
    if estimate>150 or not projection['within_cap'] or projection['matrix_fits']!=640:
        raise ValueError('full Stage B resource gate exceeds authorization')
    result={'at':utc(),'full_acceptance':True,'out':str(out),'receipt_hashes':hashes,
        'estimated_full_cpu_core_hours':estimate,'matrix_fits':640,'source_hashes':p['source_hashes'],
        'generator_hashes':p['generator_hashes'],'worker_hashes':p['worker_hashes'],'dependencies':p['dependencies'],
        'independent_system_outcomes_read':False,'test_losses_read':False,
        'scope':'development-only target-runtime resource acceptance; scientific protocol freeze still required'}
    acceptance=Path(acceptance)
    if acceptance.exists():
        old=read(acceptance)
        if {k:v for k,v in old.items() if k!='at'}!={k:v for k,v in result.items() if k!='at'}:
            raise ValueError('existing pilot acceptance differs')
        return old
    write(acceptance,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('out','source','acceptance'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();r=audit(a.out,a.source,a.acceptance)
    print({'full_acceptance':r['full_acceptance'],'projected_cpu_core_hours':r['estimated_full_cpu_core_hours']})
