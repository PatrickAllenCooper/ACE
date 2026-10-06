"""Export prospective tables only from a fully accepted complete study.

Never fits models or decides which systems/initializations to include. The
current manuscript stays unchanged until real results pass independent audit.
"""
import argparse
from pathlib import Path
import math

from runner_delivery_confirmation import read,write,sha,utc
from audit_delivery_prospective_results import primary_statistics,secondary_summary,match


def export(results,destination):
    results,destination=map(lambda p:Path(p).resolve(),(results,destination))
    accepted=read(results/'acceptance.json');complete=read(results/'complete.json');scores=read(results/'scores.json')
    reg=read(results/'registration.json')
    execution=read(results/'audit_execution.json')
    if (not accepted['full_acceptance'] or accepted['fits_checked']!=640 or accepted['checkpoints_replayed']!=640 or
            accepted['primary_cells_checked']!=240 or accepted['charged_cached_responses']!=48000 or
            accepted['new_simulator_responses']!=0 or accepted['source_revision']!=reg['source_revision'] or
            accepted['study_registration_sha256']!=sha(results/'registration.json') or
            accepted['scores_sha256']!=sha(results/'scores.json') or accepted['complete_sha256']!=sha(results/'complete.json') or
            accepted['fit_seal_sha256']!=sha(results/'fit_seal.json') or
            complete['n_fits']!=640 or complete['n_primary_cells']!=240 or complete['charged_responses']!=48000 or
            execution['status']!='complete' or execution['exit_code']!=0 or
            execution['registration_sha256']!=sha(results/'registration.json') or
            execution['acceptance_sha256']!=sha(results/'acceptance.json')):
        raise ValueError('fully accepted complete prospective study required')
    match(accepted['scope'],reg['scope'],'scope')
    primary=primary_statistics(scores['primary_rows'])
    match(accepted['primary_analysis'],primary,'accepted primary')
    match(scores['primary_analysis'],primary,'scored primary')
    if set(scores['cells'])!={str(i) for i in range(640)}:raise ValueError('full640score matrix required')
    secondary=secondary_summary([(v['cell'],v['metric']['nmse']) for v in scores['cells'].values()])
    match(accepted['secondary_descriptive_analysis'],secondary,'accepted secondary')
    if primary['n_primary_cells']!=240:raise ValueError('primary matrix incomplete')
    for c in primary['contrasts']:
        if c['n_systems']!=20 or not all(math.isfinite(v) for v in (c['ratio'],c['p_holm_four_tests'],*c['ci95_marginal'])):
            raise ValueError('invalid primary contrast')
    # No overwrite: an interrupted export remains evidence of that attempt.
    destination.mkdir(parents=True,exist_ok=False)
    macros={};provenance={}
    def macro(name,value,selector):
        macros[name]=value;provenance[name]={'receipt':'acceptance.json','selector':selector}
    macro('BSystemsPerStratum','20','primary_analysis.contrasts[*].n_systems')
    macro('BScoredFits','640','fits_checked')
    rows=['% Generated from an independently accepted full640fit receipt.',
        '\\begin{table}[ht]','\\centering\\small',
        '\\begin{tabular}{rlrrrl}','\\toprule',
        'Nodes & Control & Ratio & Marginal95\\% CI & Holm $p$ & Registered gate\\\\','\\midrule']
    for i,c in enumerate(primary['contrasts']):
        prefix='B'+('Five' if c['graph_size']==5 else 'Thirty')+('Online' if c['control']=='online' else 'Flat')
        for suffix,value,key in (('Ratio',c['ratio'],'ratio'),('CILow',c['ci95_marginal'][0],'ci95_marginal[0]'),
            ('CIHigh',c['ci95_marginal'][1],'ci95_marginal[1]'),('HolmP',c['p_holm_four_tests'],'p_holm_four_tests')):
            macro(prefix+suffix,f'{value:.6g}',f'primary_analysis.contrasts[{i}].{key}')
        macro(prefix+'Gate','met' if c['superiority'] else 'not met',f'primary_analysis.contrasts[{i}].superiority')
        rows.append(f"{c['graph_size']} & {c['control']} & {c['ratio']:.6g} & "
            f"[{c['ci95_marginal'][0]:.6g}, {c['ci95_marginal'][1]:.6g}] & "
            f"{c['p_holm_four_tests']:.6g} & {'met' if c['superiority'] else 'not met'}\\\\")
    rows+=['\\bottomrule','\\end{tabular}',
        '\\caption{Fixed-initialization-zero delivery/control ratios across twenty independently parameterized systems per stratum. The two history log ratios are averaged within each system. Intervals are marginal; Holm adjusts all four tests. The registered gate requires ratio $\\leq0.8$, upper interval $<1$, and adjusted $p<0.05$. Every registered system is required; a failed or missing fit blocks export. These are noise-disabled root-action recipe comparisons with differing supervision and compute.}',
        '\\label{tab:prospective-primary}','\\end{table}']
    (destination/'delivery_prospective_primary_table.tex').write_text('\n'.join(rows)+'\n')
    rows=['% Descriptive initialization sensitivity; no scored model selection.',
        '\\begin{table}[ht]','\\centering\\small','\\begin{tabular}{rlrrr}',
        '\\toprule','Nodes & Arm & Init & Geometric NMSE & Ratio/init0\\\\','\\midrule']
    for size,s in secondary['strata'].items():
        for arm,by_init in s['initialization_sensitivity'].items():
            for init,v in by_init.items():
                rows.append(f"{size} & {arm} & {init} & {v['geometric_nmse']:.6g} & {v['ratio_to_fixed_init0']:.6g}\\\\")
    rows+=['\\bottomrule','\\end{tabular}',
        '\\caption{Descriptive sensitivity across all registered initializations. Initialization zero remains the deployment model; no best or median scored model is selected. History log errors are averaged within each system, then across twenty systems per stratum. A fixed $10^{-12}$ floor is used, with activations retained in the receipt. These repeats are not independent systems or additional superiority tests.}',
        '\\label{tab:prospective-init}','\\end{table}']
    (destination/'delivery_prospective_sensitivity_table.tex').write_text('\n'.join(rows)+'\n')
    (destination/'delivery_prospective_claims.tex').write_text('% Generated only after full independent acceptance.\n'+
        '\n'.join('\\newcommand{\\'+n+'}{'+v+'}' for n,v in macros.items())+'\n')
    write(destination/'claim_index.json',{'at':utc(),'acceptance_sha256':sha(results/'acceptance.json'),
        'registration_sha256':sha(results/'registration.json'),'scores_sha256':sha(results/'scores.json'),
        'complete_sha256':sha(results/'complete.json'),'exporter_sha256':sha(__file__),
        'audit_execution_sha256':sha(results/'audit_execution.json'),
        'auditor_sha256':accepted['auditor_sha256'],'source_revision':reg['source_revision'],
        'macros':macros,'macro_provenance':provenance,'primary_analysis':primary,
        'secondary_descriptive_analysis':secondary,'scope':reg['scope'],
        'supervision_and_compute_matched':False,'deployment_initialization':0,
        'output_hashes':{f.name:sha(f) for f in destination.glob('*.tex')}})
    return macros


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    print(export(args.results,args.destination))
