"""Paid-history custody and held-out evaluation for Stage B.

These primitives do not launch a study. Final batch release must bind their
hashes, generator descriptors, acquisition journals, full fit matrix, dependency
versions and resource projection before any confirmation response is collected.
"""
from pathlib import Path
import sys
from delivery_prospective_design import validate_world, structural_values, action_block, reserved
from runner_delivery_confirmation import Journal, journal_count, read, write, sha, utc, no_network


def collect_history(spec, actions, out, evaluation=False):
    """Charge each frozen action BEFORE structural response generation.

    Repeated balanced actions remain separate paid observations. Mechanisms
    are immutable and responses noise-free; no interpolation or new sampling.
    Output directories are exclusive and failed attempts cannot be overwritten.
    """
    roots = validate_world(spec)
    if len(actions) != 400:
        raise ValueError('exactly400 frozen actions required')
    for action in actions:
        if set(action) != set(roots) or reserved(action_block([action[n] for n in roots])) != evaluation:
            raise ValueError('clamps or held-out action partition changed')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    journal = Journal(out/'queries.ndjson',400)
    data = {'order': spec['order'], 'parents': spec['parents'], 'roots': roots,
            'target': 'X3' if spec['size'] == 5 else 'X30', 'rows': []}
    try:
        for action in actions:
            journal.reserve()
            values = structural_values(spec, action)  # supplied noise is zero for every mechanism
            data['rows'].append({'query_index': journal.count, 'clamps': action, 'node_values': values})
        write(out/'input.json',data)
        write(out/'receipt.json',{'at':utc(),'complete':True,'rows':400,'charged_responses':journal_count(out/'queries.ndjson'),
            'input_sha256':sha(out/'input.json'),'journal_sha256':sha(out/'queries.ndjson'),
            'evaluation':evaluation,'noise':'disabled in both collection and evaluation',
            'source_world_coefficients_in_input':False})
    except BaseException as error:
        write(out/'failure.json',{'at':utc(),'charged_responses':journal_count(out/'queries.ndjson'),
                                'error':type(error).__name__+': '+str(error),'rows_completed':len(data['rows'])})
        raise
    return data


def deny_evaluation_reads(event, args):
    no_network(event, args)
    if event != 'open' or not isinstance(args[0],(str,bytes)):
        return
    path = Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
    if (path.name in ('world.json','actions.json','scores.json') or 'evaluation' in path.parts):
        raise PermissionError('fit process cannot read mechanisms or held-out outcomes: '+str(path))


def fit_cell(input_dir, source, out, arm, init, binding):
    """Prospective fixed fits; timing/epoch overrides are deliberately absent."""
    import torch
    from delivery_prospective_models import fit, dependencies
    out, input_dir = Path(out), Path(input_dir)
    if set(binding) != {'protocol_sha256','input_sha256','source_hashes','kernel_sha256','dependencies'}:
        raise ValueError('incomplete frozen fit binding')
    if sha(Path(__file__).with_name('delivery_prospective_models.py')) != binding['kernel_sha256']:
        raise ValueError('model kernel changed')
    if dependencies() != binding['dependencies']:
        raise ValueError('target runtime changed')
    for name,h in binding['source_hashes'].items():
        if sha(Path(source)/name) != h:
            raise ValueError('original learner source changed')
    if sha(input_dir/'input.json') != binding['input_sha256']:
        raise ValueError('paid training input changed')
    receipt = read(input_dir/'receipt.json')
    if (not receipt['complete'] or receipt['evaluation'] or receipt['rows'] != 400 or
            receipt['charged_responses'] != journal_count(input_dir/'queries.ndjson') or
            receipt['journal_sha256'] != sha(input_dir/'queries.ndjson') or
            receipt['input_sha256'] != binding['input_sha256']):
        raise ValueError('training history custody failed')
    out.mkdir(parents=True,exist_ok=False)
    sys.addaudithook(deny_evaluation_reads)
    data = read(input_dir/'input.json')
    try:
        if any(reserved(action_block([row['clamps'][n] for n in data['roots']])) for row in data['rows']):
            raise ValueError('reserved evaluation action block entered fitting')
        states,cost = fit(data,source,arm,init=init)
        if cost['development_epoch_override'] is not None or cost['development_online_tail'] is not None:
            raise ValueError('development override entered prospective fit')
        torch.save(states,out/'models.pt')
        write(out/'receipt.json',{**cost,'binding':binding,'model_sha256':sha(out/'models.pt'),'finished_at':utc()})
    except BaseException as error:
        write(out/'failure.json',{'at':utc(),'binding':binding,'error':type(error).__name__+': '+str(error)})
        raise


def sealed_fits(cells):
    """Verify every cell in the intended full matrix before test collection.

    Caller must first validate the exact registered640cell membership; a list
    selected by outcomes is never a permissible matrix.
    """
    sealed = {}
    if len(cells) != 640 or len({str(Path(c['out']).resolve()) for c in cells}) != 640:
        raise ValueError('full distinct640cell matrix required')
    for cell in cells:
        path = Path(cell['out']);receipt = read(path/'receipt.json')
        if (not receipt['complete'] or receipt['arm'] != cell['arm'] or receipt['init'] != cell['init'] or
                receipt['model_sha256'] != sha(path/'models.pt') or receipt['binding'] != cell['binding'] or
                receipt['development_epoch_override'] is not None or receipt['development_online_tail'] is not None):
            raise ValueError('incomplete, changed or unregistered fit; retain failure, never omit')
        sealed[str(path)] = {'receipt_sha256':sha(path/'receipt.json'),'model_sha256':sha(path/'models.pt')}
    return sealed


def evaluate_states(training, evaluation, states, source, arm):
    """Free-running target metric plus explicitly secondary mechanism diagnostics."""
    import numpy as np
    from delivery_prospective_models import runtime,validate_input
    torch,_,MLP = runtime(source)
    validate_input(training);validate_input(evaluation)
    if any(training[key] != evaluation[key] for key in ('order','parents','roots','target')):
        raise ValueError('training and evaluation graph/target differ')
    roots,order,parents,target = (training[k] for k in ('roots','order','parents','target'))
    tb={action_block([r['clamps'][n] for n in roots]) for r in training['rows']}
    eb={action_block([r['clamps'][n] for n in roots]) for r in evaluation['rows']}
    if tb & eb or any(reserved(b) for b in tb) or any(not reserved(b) for b in eb):
        raise ValueError('joint-action block leakage')
    y_train=np.array([r['node_values'][target] for r in training['rows']])
    variance=float(y_train.var())
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('nonpositive training-only primary normalizer')
    models={node:MLP.from_state_dict(state).eval() for node,state in states.items()}
    expected={'flat'} if arm=='simpler' else {n for n in order if parents[n]}
    if set(models) != expected:
        raise ValueError('unexpected fitted heads')
    truth={n:np.array([r['node_values'][n] for r in evaluation['rows']]) for n in order}
    # Roots are supplied actions. Nonroots can only receive preceding PREDICTIONS.
    predicted={n:np.array([r['clamps'][n] for r in evaluation['rows']]) for n in roots}
    with torch.no_grad():
        if arm=='simpler':
            x=torch.tensor(np.column_stack([predicted[n] for n in roots]),dtype=torch.float32)
            predicted[target]=models['flat'](x).numpy()
        else:
            for node in order:
                if node in predicted:continue
                x=torch.tensor(np.column_stack([predicted[n] for n in parents[node]]),dtype=torch.float32)
                predicted[node]=models[node](x).numpy()
    if any(not np.isfinite(v).all() for v in predicted.values()):
        raise ValueError('nonfinite free-running prediction')
    mse=float(np.mean((predicted[target]-truth[target])**2))
    result={'mse':mse,'nmse':mse/variance,'training_target_variance':variance,
            'evaluation_rows':len(evaluation['rows']),'snapped_error':None,
            'endpoint':'noise-disabled deterministic target, continuous; no fabricated quantization'}
    if arm!='simpler':
        diagnostics={}
        with torch.no_grad():
            for node,model in models.items():
                x=torch.tensor(np.column_stack([truth[n] for n in parents[node]]),dtype=torch.float32)
                local=model(x).numpy()
                diagnostics[node]={'observed_parent_mse':float(np.mean((local-truth[node])**2)),
                    'free_running_mse':float(np.mean((predicted[node]-truth[node])**2)),
                    'propagated_prediction_shift_mse':float(np.mean((predicted[node]-local)**2))}
        result['secondary_mechanism_diagnostics']=diagnostics
    return result,predicted
