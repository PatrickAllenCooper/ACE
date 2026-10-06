"""Independent read-only Stage B custody, checkpoint replay and statistics audit.

No fitting, simulator calls or outcome-based selection. Audit copied results
without rewriting their canonical remote paths. Acceptance requires the pinned
target runtime; a custody-only summary cannot substitute for full acceptance.
"""
import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time

from runner_delivery_confirmation import read,sha,utc,journal_count,no_network

ARMS=(('delivery',0),('online',0),('simpler',0),('ablation',0),
      ('delivery',1),('simpler',1),('delivery',2),('simpler',2))
HISTORIES={'balanced_varied_value':'balanced','matched_random':'random'}
WORLD_IDS=[f'{size}:{i:02d}' for size in (5,30) for i in range(20)]
# Fixed before any prospective outcomes. CPU float32 kernels can differ by
# roundoff across machines; score reconstruction always uses the original
# cached predictions, never substitutes a different replay into the endpoint.
REPLAY_RTOL=1e-6
REPLAY_ATOL=1e-7


def match(actual,expected,label):
    """Compare complete nested objects; do not ignore an unreported field."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError(label+' keys changed')
        for key in expected:match(actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError(label+' length changed')
        for i,(a,b) in enumerate(zip(actual,expected)):match(a,b,label+'/'+str(i))
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if isinstance(actual,bool) or not isinstance(actual,(float,int)) or not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(label+' nonfinite or invalid number')
        if not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-12):raise ValueError(label+' numeric discrepancy')
    elif actual!=expected:raise ValueError(label+' changed')


def canonical_cells(reg,inputs,protocol_hash):
    if reg['worlds']!=WORLD_IDS or reg['histories']!=HISTORIES or reg['cells']!=[list(c) for c in ARMS]:
        raise ValueError('registered world/history/init matrix changed')
    base=Path(reg['output']);result=[]
    if not base.is_absolute() or '..' in base.parts:raise ValueError('invalid canonical output')
    if set(inputs)!={case+'/'+history for case in WORLD_IDS for history in HISTORIES}:
        raise ValueError('missing shared history; no complete-case filtering')
    for case in WORLD_IDS:
        for history in HISTORIES:
            for arm,init in ARMS:
                result.append({'index':len(result),'case':case,'graph_size':int(case.split(':')[0]),
                    'history':history,'arm':arm,'init':init,
                    'input_dir':str(base/'training'/case.replace(':','-')/history),
                    'out':str(base/'fits'/case.replace(':','-')/history/(arm+'-i'+str(init))),
                    'binding':{'protocol_sha256':protocol_hash,'input_sha256':inputs[case+'/'+history],
                        'kernel_sha256':reg['worker_hashes']['delivery_prospective_models.py'],
                        'source_hashes':reg['source_hashes'],'dependencies':reg['dependencies']}})
    return result


def history_receipt(directory,spec,actions,evaluation):
    """Verify charged cached responses and exact action order; never simulate."""
    import numpy as np
    from delivery_prospective_design import action_block,reserved
    directory=Path(directory);data=read(directory/'input.json');r=read(directory/'receipt.json')
    roots=[n for n in spec['order'] if not spec['parents'][n]]
    if (set(data)!= {'order','parents','roots','target','rows'} or data['order']!=spec['order'] or
            data['parents']!=spec['parents'] or data['roots']!=roots or
            data['target']!=('X3' if spec['size']==5 else 'X30') or len(actions)!=400 or len(data['rows'])!=400):
        raise ValueError('cached graph/target/row membership changed')
    if (not r['complete'] or r['evaluation']!=evaluation or r['rows']!=400 or r['charged_responses']!=400 or
            journal_count(directory/'queries.ndjson')!=400 or r['input_sha256']!=sha(directory/'input.json') or
            r['journal_sha256']!=sha(directory/'queries.ndjson') or r['source_world_coefficients_in_input'] or
            r['noise']!='disabled in both collection and evaluation'):
        raise ValueError('response charge/custody receipt changed')
    for i,(row,action) in enumerate(zip(data['rows'],actions),1):
        if (set(row)!= {'query_index','clamps','node_values'} or row['query_index']!=i or row['clamps']!=action or
                set(row['node_values'])!=set(spec['order']) or not np.isfinite(list(row['node_values'].values())).all() or
                any(row['node_values'][n]!=action[n] for n in roots) or
                reserved(action_block([action[n] for n in roots]))!=evaluation):
            raise ValueError('response/action identity or heldout partition changed')
    return data


def replay_predictions(data,states,MLP,torch,arm):
    """Independently traverse the saved graph using only root clamps as inputs."""
    import numpy as np
    models={n:MLP.from_state_dict(s).eval() for n,s in states.items()}
    expected={'flat'} if arm=='simpler' else {n for n in data['order'] if data['parents'][n]}
    if set(models)!=expected:raise ValueError('checkpoint head membership changed')
    values={n:np.array([r['clamps'][n] for r in data['rows']]) for n in data['roots']}
    with torch.no_grad():
        nodes=[data['target']] if arm=='simpler' else [n for n in data['order'] if n not in values]
        for node in nodes:
            parents=data['roots'] if arm=='simpler' else data['parents'][node]
            x=torch.tensor(np.stack([values[n] for n in parents],axis=1),dtype=torch.float32)
            values[node]=models['flat' if arm=='simpler' else node](x).numpy()
    if any(v.shape!=(len(data['rows']),) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('invalid replayed predictions')
    return values,models


def calibration_ranges(data):
    """Reconstruct only the shared paid prefix; never consult test outcomes."""
    result={}
    for node in data['order']:
        if not data['parents'][node]:continue
        lo=[];hi=[]
        for parent in data['parents'][node]:
            values=[r['node_values'][parent] for r in data['rows'][:50]]
            a,b=(-3.,3.) if parent in data['roots'] else (min(values),max(values))
            lo.append(a);hi.append(b if b>a else a+1.)
        result[node]={'lo':lo,'hi':hi}
    result['flat']={'lo':[-3.]*len(data['roots']),'hi':[3.]*len(data['roots'])}
    return result


def primary_statistics(rows):
    """Recompute log-ratio units, t intervals and Holm without the scorer."""
    import numpy as np
    from scipy.stats import t
    errors={}
    for r in rows:
        key=(r['graph_size'],r['system_id'],r['history'],r['arm'])
        if r['init']!=0 or r['status']!='complete' or key in errors or not math.isfinite(r['nmse']) or r['nmse']<0:
            raise ValueError('invalid/duplicate primary row')
        errors[key]=r['nmse']
    expected={(int(c.split(':')[0]),c,h,a) for c in WORLD_IDS for h in HISTORIES for a in ('delivery','online','simpler')}
    if set(errors)!=expected:raise ValueError('full240primary matrix required')
    contrasts=[];floor=1e-12
    for size in (5,30):
        for control in ('online','simpler'):
            logs={};activations={}
            for case in (c for c in WORLD_IDS if c.startswith(str(size)+':')):
                history_logs=[]
                for history in sorted(HISTORIES):
                    d=errors[size,case,history,'delivery'];c=errors[size,case,history,control]
                    for arm,error in (('delivery',d),(control,c)):
                        activations[arm]=activations.get(arm,0)+int(error<floor)
                    history_logs.append(math.log(max(d,floor))-math.log(max(c,floor)))
                logs[case]=sum(history_logs)/2
            v=np.array(list(logs.values()));mean=float(v.mean());sd=float(v.std(ddof=1));se=sd/math.sqrt(20)
            p=float(2*t.sf(abs(mean/se),19)) if se else (1. if mean==0 else 0.)
            half=float(t.ppf(.975,19)*se)
            contrasts.append({'graph_size':size,'control':control,'n_systems':20,'log_mean':mean,'log_sd':sd,
                'ratio':math.exp(mean),'ci95_marginal':[math.exp(mean-half),math.exp(mean+half)],'p_raw':p,
                'system_log_ratios':logs,'floor_activations':activations})
    running=0.
    for rank,index in enumerate(sorted(range(4),key=lambda i:contrasts[i]['p_raw'])):
        c=contrasts[index];running=max(running,min(1.,(4-rank)*c['p_raw']))
        c['p_holm_four_tests']=running
        c['superiority']=c['ratio']<=.8 and c['ci95_marginal'][1]<1 and running<.05
    return {'complete':True,'n_primary_cells':240,'primary_init':0,'floor':floor,
        'independent_unit':'system; equal-weight mean of two history log ratios',
        'multiple_testing':'Holm across two controls in each of two strata','contrasts':contrasts}


def secondary_summary(cell_metrics):
    """Descriptive complete-matrix ablation/history/init sensitivity, no tests.

    Every history pair is first aggregated within its system. No deployment
    initialization is selected and no extra superiority claims are produced.
    """
    errors={}
    for c,error in cell_metrics:
        key=(c['case'],c['history'],c['arm'],c['init'])
        if key in errors or not math.isfinite(error) or error<0:raise ValueError('invalid secondary matrix')
        errors[key]=error
    expected={(c,h,a,i) for c in WORLD_IDS for h in HISTORIES for a,i in ARMS}
    if set(errors)!=expected:raise ValueError('full640secondary matrix required')
    result={};floor=1e-12
    def log_error(case,history,arm,init):return math.log(max(errors[case,history,arm,init],floor))
    for size in (5,30):
        worlds=[c for c in WORLD_IDS if c.startswith(str(size)+':')];stratum={}
        differences={c:sum(log_error(c,h,'delivery',0)-log_error(c,h,'ablation',0) for h in HISTORIES)/2 for c in worlds}
        stratum['delivery_vs_short_fit']={'ratio':math.exp(sum(differences.values())/20),'system_log_ratios':differences}
        stratum['history_specific_primary_ratios']={}
        for history in HISTORIES:
            stratum['history_specific_primary_ratios'][history]={control:math.exp(sum(
                log_error(c,history,'delivery',0)-log_error(c,history,control,0) for c in worlds)/20)
                for control in ('online','simpler')}
        sensitivity={}
        for arm in ('delivery','simpler'):
            by_init={}
            for init in (0,1,2):
                world_logs={c:sum(log_error(c,h,arm,init) for h in HISTORIES)/2 for c in worlds}
                ratio_logs={c:sum(log_error(c,h,arm,init)-log_error(c,h,arm,0) for h in HISTORIES)/2 for c in worlds}
                by_init[str(init)]={'geometric_nmse':math.exp(sum(world_logs.values())/20),
                    'ratio_to_fixed_init0':math.exp(sum(ratio_logs.values())/20),
                    'systems_better_than_init0':sum(v<0 for v in ratio_logs.values()),
                    'system_log_nmse':world_logs,'floor_activations':sum(errors[c,h,arm,init]<floor for c in worlds for h in HISTORIES)}
            sensitivity[arm]=by_init
        stratum['initialization_sensitivity']=sensitivity;result[str(size)]=stratum
    return {'scope':'descriptive only; no added significance tests or scored model-selection rule',
        'primary_deployment_init':0,'systems_per_stratum':20,'histories_per_system':2,'floor':floor,'strata':result}


def audit(out,project,source,acceptance):
    import numpy as np
    from delivery_prospective_models import dependencies,runtime
    out,project,source,acceptance=map(lambda v:Path(v).resolve(),(out,project,source,acceptance))
    sys.addaudithook(no_network);cpu0=time.process_time();start=time.monotonic()
    p=read(out/'registration.json');complete=read(out/'complete.json');protocol_hash=sha(out/'registration.json')
    if (p['matrix_fits']!=640 or p['primary_cells']!=240 or p['new_response_ceiling']!=48000 or
            complete['n_fits']!=640 or complete['n_primary_cells']!=240 or complete['charged_responses']!=48000 or
            complete['registration_sha256']!=protocol_hash or complete['account']!='ucb736_asc1' or
            complete['scores_sha256']!=sha(out/'scores.json') or complete['fit_seal_sha256']!=sha(out/'fit_seal.json')):
        raise ValueError('full study completion binding changed')
    for file,key in (('descriptor_manifest.json','descriptor_manifest_sha256'),('attribution_gate.json','gate_sha256'),
                     ('pilot_acceptance.json','pilot_acceptance_sha256'),('pilot_projection.json','pilot_projection_sha256'),
                     ('historical_pilot_failure.json','historical_pilot_failure_sha256')):
        if sha(out/file)!=p[key]:raise ValueError('protocol receipt changed')
    if p['gate_sha256']!='a81d0ac51965f71123ecc915764cf18c8166537a1d69f1053d096dd36ed3310f':
        raise ValueError('immutable attribution gate changed')
    for name,h in p['worker_hashes'].items():
        if sha(project/'scripts/research'/name)!=h:raise ValueError('frozen worker changed')
    for name,h in p['source_hashes'].items():
        if sha(source/name)!=h:raise ValueError('original learner changed')
    for name,h in p['generator_hashes'].items():
        if sha(project/name)!=h:raise ValueError('generator changed')
    if (sha(project/'source_commit_receipt.json')!=p['source_commit_receipt_sha256'] or
            read(project/'source_commit_receipt.json')['source_revision']!=p['source_revision'] or dependencies()!=p['dependencies']):
        raise ValueError('committed source/runtime custody changed')
    committed=read(project/'source_commit_receipt.json')
    for name,h in committed['files'].items():
        if sha(project/name)!=h:raise ValueError('committed file changed')
    # Execute the exact frozen runtime implementation, even for copied results.
    from delivery_prospective_batch import load_descriptors,allocation_plan
    for module in ('delivery_prospective_models','delivery_prospective_batch','delivery_prospective_design'):
        if sha(sys.modules[module].__file__)!=p['worker_hashes'][module+'.py']:
            raise ValueError('audit imported a different frozen helper')
    if allocation_plan(read(out/'pilot_projection.json'))!=p['resources']:raise ValueError('resource plan changed')
    if not read(out/'pilot_acceptance.json')['full_acceptance']:raise ValueError('pilot was not accepted')
    phase_hashes={}
    for phase,key in (('qualification','qualification'),('collect','collection'),('evaluate','evaluation')):
        filename=phase+'_execution.json';r=read(out/filename)
        if (r['status']!='complete' or r['exit_code']!=0 or r['registration_sha256']!=protocol_hash or
                r['account']!='ucb736_asc1' or r['peak_tree_rss_bytes']>p['resources']['rss_bytes'] or
                r['elapsed_seconds']>p['resources'][key+'_wall_seconds']):
            raise ValueError('phase execution/resource acceptance failed')
        phase_hashes[filename]=sha(out/filename)
    qualified=read(out/'qualification_complete.json')
    if (qualified['registration_sha256']!=protocol_hash or qualified['dependencies']!=p['dependencies'] or
            qualified['confirmation_responses_evaluated']!=0 or
            qualified['execution_sha256']!=phase_hashes['qualification_execution.json'] or
            qualified['log_sha256']!=sha(out/'qualification.log')):
        raise ValueError('target-runtime qualification binding changed')
    descriptors=load_descriptors(out/'descriptors',read(out/'descriptor_manifest.json'))
    collection=read(out/'collection_complete.json');matrix=read(out/'matrix.json');seal=read(out/'fit_seal.json')
    if (collection['registration_sha256']!=protocol_hash or collection['matrix_sha256']!=sha(out/'matrix.json') or
            collection['charged_responses']!=32000 or matrix['registration_sha256']!=protocol_hash or
            seal['registration_sha256']!=protocol_hash or seal['matrix_sha256']!=sha(out/'matrix.json')):
        raise ValueError('shared training/matrix/fit seal changed')
    training={};input_hashes={};receipt_hashes={}
    for case in WORLD_IDS:
        spec,actions=descriptors[case]
        for history,strategy in HISTORIES.items():
            key=case+'/'+history;directory=out/'training'/case.replace(':','-')/history
            training[key]=history_receipt(directory,spec,actions[strategy],False)
            files={n:sha(directory/n) for n in ('input.json','queries.ndjson','receipt.json')}
            if collection['artifacts'].get(key)!=files:raise ValueError('training custody changed')
            input_hashes[key]=files['input.json'];receipt_hashes[key]=files
    if set(collection['artifacts'])!=set(input_hashes):raise ValueError('unregistered/missing history')
    cells=canonical_cells(p,input_hashes,protocol_hash)
    if matrix['cells']!=cells or set(seal['artifacts'])!={c['out'] for c in cells}:raise ValueError('full640membership changed')
    score=read(out/'scores.json')
    if (set(score['cells'])!={str(i) for i in range(640)} or score['fit_seal_sha256']!=sha(out/'fit_seal.json') or
            score['new_training_responses']!=32000 or score['new_shared_evaluation_responses']!=16000 or score['scope']!=p['scope']):
        raise ValueError('score matrix/custody changed')
    # Validate ALL fitted artifacts and attempts before reading test inputs.
    fits={};fit_cpu=0.;latest_fit=None
    for c in cells:
        path=out/'fits'/c['case'].replace(':','-')/c['history']/(c['arm']+'-i'+str(c['init']))
        r=read(path/'receipt.json');files={'receipt_sha256':sha(path/'receipt.json'),'model_sha256':sha(path/'models.pt')}
        heads={'flat'} if c['arm']=='simpler' else {n for n in training[c['case']+'/'+c['history']]['order']
            if training[c['case']+'/'+c['history']]['parents'][n]}
        updates=48000 if c['arm']=='online' else 100 if c['arm']=='ablation' else 30000
        if (files!=seal['artifacts'][c['out']] or not r['complete'] or r['arm']!=c['arm'] or r['init']!=c['init'] or
                r['binding']!=c['binding'] or r['model_sha256']!=files['model_sha256'] or set(r['heads'])!=heads or
                r['unique_paid_rows']!=400 or r['calibration_rows']!=50 or r['new_queries_in_fit']!=0 or
                r['evaluation_responses_read']!=0 or r['development_epoch_override'] is not None or r['development_online_tail'] is not None or
                any(v['updates']!=updates or v['eligible_rows']!=400 for v in r['heads'].values()) or
                not math.isfinite(r['cpu_seconds']) or r['cpu_seconds']<=0):
            raise ValueError('fitted artifact/configuration changed')
        fit_cpu+=r['cpu_seconds'];fits[c['index']]=(path,r)
        match(r['normalizers'],calibration_ranges(training[c['case']+'/'+c['history']]),'calibration')
        finished=datetime.fromisoformat(r['finished_at']);latest_fit=max(latest_fit,finished) if latest_fit else finished
    for case in WORLD_IDS:
        path=out/'world_execution'/case.replace(':','-');r=read(path/'complete.json');attempts=read(path/'execution.json')['attempts']
        if (r['n_fits']!=16 or r['protocol_sha256']!=protocol_hash or
                [a['cell_index'] for a in attempts]!=[c['index'] for c in cells if c['case']==case] or
                any(a['status']!='complete' or a['exit_code']!=0 or a['peak_tree_rss_bytes']>p['resources']['rss_bytes'] for a in attempts)):
            raise ValueError('world execution/telemetry incomplete')
    if (out/'stop_new_fits.json').exists():raise ValueError('failure blocks acceptance')
    begun=read(out/'evaluation_started.json')
    if begun['fit_seal_sha256']!=sha(out/'fit_seal.json') or datetime.fromisoformat(begun['at'])<latest_fit:
        raise ValueError('evaluation preceded the full fit seal')
    torch,_,MLP=runtime(source);torch.set_num_threads(1)
    primary=[];all_metrics=[];evaluation_hashes={};replay_deltas={}
    for case in WORLD_IDS:
        spec,actions=descriptors[case];directory=out/'evaluation'/case.replace(':','-')
        evaluation=history_receipt(directory,spec,actions['evaluation'],True)
        first_charge=__import__('json').loads((directory/'queries.ndjson').read_text().splitlines()[0])
        if datetime.fromisoformat(first_charge['at'])<datetime.fromisoformat(begun['at']):
            raise ValueError('heldout response charged before evaluation barrier')
        evaluation_hashes[case]={n:sha(directory/n) for n in ('input.json','queries.ndjson','receipt.json')}
        for c in (c for c in cells if c['case']==case):
            saved=score['cells'][str(c['index'])];path,r=fits[c['index']]
            prediction_file=directory/('cell-'+str(c['index'])+'-predictions.npz')
            if (saved['cell']!=c or saved['evaluation_input_sha256']!=sha(directory/'input.json') or
                    saved['predictions_sha256']!=sha(prediction_file)):raise ValueError('score/prediction binding changed')
            replay,models=replay_predictions(evaluation,torch.load(path/'models.pt',map_location='cpu',weights_only=True),MLP,torch,c['arm'])
            predictions={};maximum_delta=0.
            with np.load(prediction_file,allow_pickle=False) as cached:
                if set(cached.files)!=set(replay):raise ValueError('prediction head set changed')
                for n,values in replay.items():
                    if (cached[n].shape!=values.shape or cached[n].dtype!=values.dtype or
                            not np.allclose(cached[n],values,rtol=REPLAY_RTOL,atol=REPLAY_ATOL,equal_nan=False)):
                        raise ValueError('checkpoint/prediction discrepancy')
                    if n in evaluation['roots'] and not np.array_equal(cached[n],values):
                        raise ValueError('root clamp cannot change through numerical replay')
                    maximum_delta=max(maximum_delta,float(np.max(np.abs(cached[n]-values))))
                    predictions[n]=cached[n].copy()
            replay_deltas[str(c['index'])]=maximum_delta
            for n,model in models.items():
                if sum(parameter.numel() for parameter in model.parameters())!=r['heads'][n]['parameters']:
                    raise ValueError('parameter accounting changed')
                for attribute,key in (('in_lo','lo'),('in_hi','hi')):
                    expected=torch.tensor(r['normalizers'][n][key],dtype=torch.float32)
                    if not torch.equal(getattr(model,attribute).reshape(-1),expected):
                        raise ValueError('checkpoint calibration differs from paid prefix')
            train=training[case+'/'+c['history']];target=evaluation['target']
            var=float(np.var([row['node_values'][target] for row in train['rows']]))
            if not math.isfinite(var) or var<=0:raise ValueError('invalid training-only target variance')
            truth={n:np.array([row['node_values'][n] for row in evaluation['rows']]) for n in evaluation['order']}
            mse=float(np.mean(np.square(predictions[target]-truth[target])))
            metric={'mse':mse,'nmse':mse/var,'training_target_variance':var,'evaluation_rows':400,
                'snapped_error':None,'endpoint':'noise-disabled deterministic target, continuous; no fabricated quantization'}
            if c['arm']!='simpler':
                diagnostic={}
                with torch.no_grad():
                    for n,model in models.items():
                        x=torch.tensor(np.stack([truth[parent] for parent in evaluation['parents'][n]],axis=1),dtype=torch.float32)
                        local=model(x).numpy()
                        diagnostic[n]={'observed_parent_mse':float(np.mean(np.square(local-truth[n]))),
                            'free_running_mse':float(np.mean(np.square(predictions[n]-truth[n]))),
                            'propagated_prediction_shift_mse':float(np.mean(np.square(predictions[n]-local)))}
                metric['secondary_mechanism_diagnostics']=diagnostic
            match(saved['metric'],metric,'cell'+str(c['index']))
            all_metrics.append((c,metric['nmse']))
            if c['init']==0 and c['arm'] in ('delivery','online','simpler'):
                primary.append({'graph_size':c['graph_size'],'system_id':case,'history':c['history'],
                    'arm':c['arm'],'init':0,'nmse':metric['nmse'],'status':'complete'})
    match(score['primary_rows'],primary,'primary_rows');statistics=primary_statistics(primary)
    match(score['primary_analysis'],statistics,'primary_analysis')
    match(complete['fit_cpu_core_hours'],fit_cpu/3600,'fit_CPU')
    result={'at':utc(),'full_acceptance':True,'study_registration_sha256':protocol_hash,
        'scores_sha256':sha(out/'scores.json'),'complete_sha256':sha(out/'complete.json'),
        'fit_seal_sha256':sha(out/'fit_seal.json'),'source_revision':p['source_revision'],
        'auditor_sha256':sha(__file__),'runtime':dependencies(),'training_receipt_hashes':receipt_hashes,
        'phase_execution_hashes':phase_hashes,
        'evaluation_receipt_hashes':evaluation_hashes,'fits_checked':640,'checkpoints_replayed':640,
        'primary_cells_checked':240,'charged_cached_responses':48000,'new_simulator_responses':0,
        'replay_tolerance':{'rtol':REPLAY_RTOL,'atol':REPLAY_ATOL,'roots':'exact'},
        'replay_max_abs_deltas':replay_deltas,'scoring_predictions':'unchanged original cached arrays',
        'audit_cpu_seconds':time.process_time()-cpu0,'audit_wall_seconds':time.monotonic()-start,
        'primary_analysis':statistics,'scope':p['scope']}
    result['secondary_descriptive_analysis']=secondary_summary(all_metrics)
    # Neither old results nor an existing acceptance can be overwritten.
    import json
    with acceptance.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('out','project','source','acceptance'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();r=audit(args.out,args.project,args.source,args.acceptance)
    print({'full_acceptance':r['full_acceptance'],'fits_checked':r['fits_checked'],'new_simulator_responses':0})
