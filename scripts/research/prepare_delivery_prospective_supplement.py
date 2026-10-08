"""Prepare complete descriptive B reporting after pinned full supplemental replay.

No fitting, responses, test selection or new significance tests. Original frozen
exporter/workers remain unchanged. Preparation tests use fabricated values only.
The full actual package/runtime/anonymity qualification remains separate.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from replay_delivery_prospective_release import (ARMS, DEPENDENCIES, HISTORIES,
    WORLDS, byte_integrity, gate)
from verify_delivery_release import relative

FLOOR = 1e-12


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def invalid(value): raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def snapshot(path, expected):
    path = Path(path)
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('independent SHA256 pin required')
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('symlink in metadata location')
    with path.open('rb') as stream: raw = stream.read(64*1024*1024+1)
    if len(raw) > 64*1024*1024 or digest(raw) != expected:
        raise ValueError('metadata size or snapshot digest changed')
    return decode(raw)


def number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError('finite nonnegative metric required')
    return value


def close(a, b, label):
    number(a); number(b)
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(label+' differs from accepted cached metrics')


def match(actual, expected):
    """Frozen auditor's declared tolerance, with exact membership/type checks."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):raise ValueError('analysis keys differ')
        for k in expected:match(actual[k],expected[k])
    elif isinstance(expected, list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('analysis length differs')
        for a,b in zip(actual,expected):match(a,b)
    elif type(expected) in (int,float):
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('analysis numeric value differs')
    elif type(actual) != type(expected) or actual != expected:raise ValueError('analysis metadata differs')


def describe(score, primary, secondary):
    """Pure in-memory renderer input; caller owns acceptance before decoding."""
    cells = score['cells']
    if set(cells) != {str(i) for i in range(640)}: raise ValueError('full640 matrix required')
    errors = {}; rows = []
    for i, (case, history, arm, init) in enumerate((c,h,a,k) for c in WORLDS for h in HISTORIES for a,k in ARMS):
        saved = cells[str(i)]; cell, metric = saved['cell'], saved['metric']
        if (type(cell['index']) is not int or cell['index'] != i or
                (cell['case'],cell['history'],cell['arm'],cell['init']) != (case,history,arm,init) or
                type(cell['init']) is not int or type(cell['graph_size']) is not int or
                cell['graph_size'] != int(case.split(':')[0])):
            raise ValueError('cell ordering/identity changed')
        mse, nmse, variance = (number(metric[k], k=='training_target_variance')
                               for k in ('mse','nmse','training_target_variance'))
        if type(metric['evaluation_rows']) is not int or metric['evaluation_rows'] != 400:
            raise ValueError('heldout row count changed')
        close(nmse, mse/variance, 'training-variance normalization')
        errors[case,history,arm,init] = nmse
        rows.append({'system':case,'graph_size':cell['graph_size'],'history':history,'arm':arm,'init':init,
            'mse':mse,'nmse':nmse,'training_target_variance':variance,'floor_active':nmse<FLOOR,
            'receipt':'B/scores.json','selector':f'cells.{i}.metric'})
    # Every arm shares a history and its training-only normalizer.
    for case in WORLDS:
        for history in HISTORIES:
            variances = [r['training_target_variance'] for r in rows if r['system']==case and r['history']==history]
            if len(set(variances)) != 1: raise ValueError('normalizer differs within shared history')
    expected_contrasts = [(size,control) for size in (5,30) for control in ('online','simpler')]
    if (primary['complete'] is not True or type(primary['n_primary_cells']) is not int or
            primary['n_primary_cells'] != 240 or type(primary['primary_init']) is not int or primary['primary_init'] != 0 or primary['floor'] != FLOOR or
            [(c['graph_size'],c['control']) for c in primary['contrasts']] != expected_contrasts):
        raise ValueError('registered primary analysis changed')
    system_rows = []; history_rows = []; ablations = []; floor_rows = []
    for ci, contrast in enumerate(primary['contrasts']):
        size, control = expected_contrasts[ci]; systems = [s for s in WORLDS if s.startswith(str(size)+':')]
        if type(contrast['n_systems']) is not int or contrast['n_systems'] != 20 or set(contrast['system_log_ratios']) != set(systems):
            raise ValueError('complete twenty-system contrast required')
        activations = {'delivery':0,control:0}
        for case in systems:
            logs=[]
            for history in HISTORIES:
                d,c = errors[case,history,'delivery',0],errors[case,history,control,0]
                logs.append(math.log(max(d,FLOOR))-math.log(max(c,FLOOR)))
                activations['delivery'] += int(d<FLOOR); activations[control] += int(c<FLOOR)
            value=sum(logs)/2
            # Log ratios are signed; close() is for nonnegative errors only.
            accepted=contrast['system_log_ratios'][case]
            if type(accepted) not in (int,float) or not math.isfinite(accepted) or not math.isclose(value,accepted,rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('paired system log ratio differs')
            system_rows.append({'system':case,'graph_size':size,'control':control,'log_ratio':value,
                'ratio':math.exp(value),'selector':f'primary_analysis.contrasts.{ci}.system_log_ratios.{case}'})
        if contrast['floor_activations'] != activations: raise ValueError('primary floor counts differ')
        floor_rows.append({'graph_size':size,'control':control,'delivery':activations['delivery'],'control_activations':activations[control]})
        close(math.exp(sum(r['log_ratio'] for r in system_rows if r['graph_size']==size and r['control']==control)/20),contrast['ratio'],'primary ratio')
        for history in HISTORIES:
            ratio=math.exp(sum(math.log(max(errors[s,history,'delivery',0],FLOOR))-math.log(max(errors[s,history,control,0],FLOOR)) for s in systems)/20)
            close(ratio,secondary['strata'][str(size)]['history_specific_primary_ratios'][history][control],'history ratio')
            history_rows.append({'graph_size':size,'history':history,'control':control,'ratio':ratio})
    for size in (5,30):
        systems=[s for s in WORLDS if s.startswith(str(size)+':')]; logs={}
        for case in systems:
            logs[case]=sum(math.log(max(errors[case,h,'delivery',0],FLOOR))-math.log(max(errors[case,h,'ablation',0],FLOOR)) for h in HISTORIES)/2
        expected=secondary['strata'][str(size)]['delivery_vs_short_fit']
        if set(expected['system_log_ratios']) != set(logs) or any(not math.isclose(v,expected['system_log_ratios'][s],rel_tol=1e-9,abs_tol=1e-12) for s,v in logs.items()):
            raise ValueError('ablation system ratios differ')
        ratio=math.exp(sum(logs.values())/20);close(ratio,expected['ratio'],'ablation ratio')
        ablations.append({'graph_size':size,'ratio':ratio,'system_log_ratios':logs})
    return {'scope':'descriptive complete-matrix supplement; original registered tests and init0 rule unchanged',
        'floor':FLOOR,'cells':rows,'fixed_init0': [r for r in rows if r['init']==0],
        'system_log_ratios':system_rows,'history_specific_ratios':history_rows,
        'short_fit_ablation':ablations,'primary_floor_counts':floor_rows,
        'new_fits':0,'new_responses':0,'additional_significance_tests':0}


def export(root, expected_manifest, replay_receipt, expected_replay, destination):
    """No scientific artifact decoding before original and full replay gates."""
    root, destination = Path(root).absolute(),Path(destination).absolute()
    if (any(p.is_symlink() for p in [root,*root.parents,destination,*destination.parents]) or
            destination.resolve().is_relative_to(root.resolve())):
        raise ValueError('exclusive output outside package without symlinks required')
    manifest = snapshot(relative(root,'manifest.json'), expected_manifest)
    entries = {f['path']:f for f in manifest['files']}
    if len(entries) != len(manifest['files']): raise ValueError('duplicate manifest artifact')
    def read(name): return snapshot(relative(root,name),entries[name]['sha256'])
    if 'B/replay_contract.json' not in entries:raise ValueError('full accepted B package required before reporting')
    contract = read('B/replay_contract.json');gate(contract)
    replay = snapshot(replay_receipt, expected_replay)
    if (replay['full_supplemental_replay'] is not True or replay['runtime'] != DEPENDENCIES or
            any(type(replay[k]) is not int or replay[k] != v for k,v in
                {'checkpoints_replayed':640,'primary_cells_recomputed':240,'cached_responses_checked':48000,
                 'new_optimizer_updates':0,'new_responses':0}.items()) or
            replay['integrity']['manifest_sha256'] != expected_manifest or
            replay['integrity']['expected_digest_supplied'] is not True):
        raise ValueError('pinned full target-runtime supplemental replay required')
    byte_integrity(root,manifest)
    # Only now capture and decode scientific values from manifest-bound bytes.
    score, accepted = read('B/scores.json'),read('B/acceptance.json')
    match(score['primary_analysis'], accepted['primary_analysis'])
    match(score['primary_analysis'], replay['primary_analysis'])
    match(accepted['secondary_descriptive_analysis'], replay['secondary_descriptive_analysis'])
    result=describe(score,replay['primary_analysis'],replay['secondary_descriptive_analysis'])
    result['provenance']={'manifest_sha256':expected_manifest,'supplemental_replay_sha256':expected_replay,
        'scores_sha256':entries['B/scores.json']['sha256'],'acceptance_sha256':entries['B/acceptance.json']['sha256'],
        'original_registration_sha256':contract['registration_sha256'],'tool_sha256':digest(Path(__file__).read_bytes())}
    destination.mkdir(parents=True,exist_ok=False)
    with (destination/'supplement.json').open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for name,rows in [('all_cells',result['cells']),('fixed_init0',result['fixed_init0']),
                      ('system_log_ratios',result['system_log_ratios']),('history_specific',result['history_specific_ratios']),('floor_counts',result['primary_floor_counts'])]:
        with (destination/(name+'.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Twenty-row LaTeX sections retain every system in fixed ID order.
    lines=['% Descriptive fixed-init0 absolute errors; all systems/histories retained.']
    for size in (5,30):
        for history in HISTORIES:
            for metric in ('mse','nmse'):
                lines += [r'\begin{table}[ht]',r'\centering\scriptsize',r'\begin{tabular}{lrrrr}',r'\toprule',
                    'System & Delivery & Online & Flat & Short fit'+r'\\',r'\midrule']
                for case in [s for s in WORLDS if s.startswith(str(size)+':')]:
                    values={r['arm']:r[metric] for r in result['fixed_init0'] if r['system']==case and r['history']==history}
                    lines.append(case+' & '+' & '.join(format(values[a],'.6g') for a in ('delivery','online','simpler','ablation'))+r'\\')
                label=history.replace('_',r'\_')
                lines += [r'\bottomrule',r'\end{tabular}',
                    r'\caption{'+f'{size}-node family, {label}, fixed initialization zero: '+metric.upper()+
                    r'. NMSE uses the training target variance of each shared history. Errors here are not floored. The short fit is the registered 100-update SCM ablation. Rows are systems; histories and arms are paired, not additional independent samples.}',r'\end{table}']
    (destination/'delivery_prospective_absolute_table.tex').write_text('\n'.join(lines)+'\n')
    lines=['% Descriptive paired reporting; no additional superiority tests.']
    for size in (5,30):
        for control in ('online','simpler'):
            lines += [r'\begin{table}[ht]',r'\centering\scriptsize',r'\begin{tabular}{lrr}',
                r'\toprule','System & Mean history log ratio & Ratio'+r'\\',r'\midrule']
            for row in result['system_log_ratios']:
                if row['graph_size']==size and row['control']==control:
                    lines.append(row['system']+' & '+format(row['log_ratio'],'.6g')+' & '+format(row['ratio'],'.6g')+r'\\')
            lines += [r'\bottomrule',r'\end{tabular}',r'\caption{'+f'{size}-node delivery/{control}, initialization zero. '+
                r'The two history log NMSE ratios are averaged within each system using the registered floor; the final column exponentiates that mean. Every system is shown in fixed ID order, including worsening systems. These rows are the paired units of the original registered contrast, not extra tests.}',r'\end{table}']
    lines += [r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{rllr}',
        r'\toprule','Nodes & History & Control & Delivery/control'+r'\\',r'\midrule']
    for row in result['history_specific_ratios']:
        lines.append(str(row['graph_size'])+' & '+row['history'].replace('_',r'\_')+' & '+row['control']+' & '+format(row['ratio'],'.6g')+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Descriptive history-specific geometric ratios across twenty systems in each stratum, at initialization zero. No collection-strategy superiority or additional significance test is asserted.}',r'\end{table}',
        r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{rr}',r'\toprule',
        'Nodes & Delivery/short fit'+r'\\',r'\midrule']
    for row in result['short_fit_ablation']:lines.append(str(row['graph_size'])+' & '+format(row['ratio'],'.6g')+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Descriptive delivery versus the registered short-fit ablation. History log ratios are averaged within each system before averaging across twenty systems; both arms use initialization zero. No added significance test.}',r'\end{table}',
        r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{rlrr}',r'\toprule',
        'Nodes & Control & Delivery floor hits & Control floor hits'+r'\\',r'\midrule']
    for row in result['primary_floor_counts']:lines.append(f"{row['graph_size']} & {row['control']} & {row['delivery']} & {row['control_activations']}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Counts of initialization-zero arm/history errors strictly below the registered floor, per primary contrast. The same delivery error appears in two contrasts and is counted there twice, not as extra independent data. All-cell floor flags remain in the machine-readable supplement.}',r'\end{table}']
    (destination/'delivery_prospective_descriptive_table.tex').write_text('\n'.join(lines)+'\n')
    hashes={p.name:digest(p.read_bytes()) for p in destination.iterdir()}
    with (destination/'output_index.json').open('x') as stream: json.dump({'at':datetime.now(timezone.utc).isoformat(),'output_hashes':hashes,'provenance':result['provenance'],'actual_numerical_replay_performed_here':False},stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','replay-receipt','destination'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','replay-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();r=export(args.root,args.manifest_sha256,args.replay_receipt,args.replay_sha256,args.destination)
    print({'cells':len(r['cells']),'fixed_init0':len(r['fixed_init0']),'new_fits':0,'new_responses':0})
