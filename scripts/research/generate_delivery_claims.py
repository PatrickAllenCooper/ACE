"""Generate retained empirical LaTeX claims from the validated sealed receipt."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'results/delivery_final_history_20261006'
raw=(folder/'scores.json').read_bytes();stats=json.loads((folder/'independent_statistics.json').read_text())
assert hashlib.sha256(raw).hexdigest()==stats['score_sha256'] and stats['matches_receipt']
pairs=json.loads(raw)['pairs'];assert len(pairs)==stats['n']==12
w=stats['worsening_histories'];assert len(w)==1 and w[0]['seed']==124753321
macros={'DeliveryN':str(stats['n']),'DeliveryRatio':f"{stats['ratio']:.3f}",
        'DeliveryCILow':f"{stats['ci_lo']:.3f}",'DeliveryCIHigh':f"{stats['ci_hi']:.3f}",
        'DeliveryP':f"{stats['exact_sign_flip_p']:.8f}",'DeliveryWins':str(len(pairs)-len(w)),
        'WorseSeed':str(w[0]['seed']),'WorseOnline':f"{w[0]['online_error']:.6f}",
        'WorseDelivery':f"{w[0]['delivery_median_error']:.6f}",'WorseRatio':f"{w[0]['ratio']:.6f}"}
terminal_path=folder/'terminal_audit.json'
terminal=json.loads(terminal_path.read_text())
assert terminal['status']=='complete' and terminal['all12_seal_verified']
assert terminal['scores_sha256']==hashlib.sha256(raw).hexdigest()
macros['ConfirmationChargedCalls']=f"{terminal['aggregate_calls_including_discarded']:,}"

confirmation_accounting=None
accounting_path=ROOT/'results/delivery_release_preparation_20261007/confirmation_verification.json'
if accounting_path.exists():
    verification=json.loads(accounting_path.read_text())
    counts=verification['replay']['accounting']
    assert verification['replay']['complete_histories']==12 and verification['replay']['model_sets_replayed']==48
    assert counts['aggregate_charged_attempts']==terminal['aggregate_calls_including_discarded']
    assert counts['persisted_complete_responses']+counts['interrupted_charged_reservations']==counts['aggregate_charged_attempts']
    macros['PersistedConfirmationResponses']=f"{counts['persisted_complete_responses']:,}"
    macros['InterruptedConfirmationAttempts']=f"{counts['interrupted_charged_reservations']:,}"
    confirmation_accounting={'verification_sha256':hashlib.sha256(accounting_path.read_bytes()).hexdigest(),
        'scope':'thirteen distinct acquisition attempts; charged reservations differ from persisted responses',
        'interrupted_returned_responses':'unknown; no persisted interrupted response history'}

attribution=ROOT/'results/delivery_attribution_20261006'
gate_path=ROOT/'results/delivery_paper_implementation_20261006/attribution_gate.json'
attribution_index=None
if attribution.exists():
    def digest(file):return hashlib.sha256(file.read_bytes()).hexdigest()
    summary=json.loads((attribution/'summary.json').read_text())
    sealed=json.loads((attribution/'summary_complete.json').read_text())
    gate=json.loads(gate_path.read_text());done=json.loads((attribution/'complete.json').read_text())
    assert digest(attribution/'summary.json')==sealed['summary_sha256']
    assert summary['gate_sha256']==digest(gate_path) and summary['custody']['full_acceptance']
    assert digest(attribution/'complete.json')==gate['complete_receipt_sha256']
    for case,h in done['score_hashes'].items():assert digest(attribution/'scores'/f'{case}.json')==h
    # Independently recompute paired ratios from the retained twelve rows.
    ratios={key:math.exp(sum(math.log(max(c['delivery']['nmse'],1e-12)/max(c[key]['nmse'],1e-12)) for c in gate['histories'])/12)
            for key in gate['ratios_continuous_nmse_init0']}
    for key,v in ratios.items():assert math.isclose(v,gate['ratios_continuous_nmse_init0'][key],rel_tol=1e-12)
    macros.update({'AOnlineRatio':f"{ratios['online']:.3f}",'AFlatRatio':f"{ratios['simpler']:.3f}",
        'AMatchedRatio':f"{ratios['matched_cpu_flat']:.3f}",'ABufferRatio':f"{ratios['data_ablation']:.3f}",
        'AOptimizationRatio':f"{ratios['optimization_ablation']:.3f}",
        'AUnusedRatio':f"{ratios['unused_observations_ablation']:.3f}",
        'AFitCPU':f"{done['fit_cpu_core_hours']:.2f}"})
    diagnosis=summary['worsening_history']['fits']['all_paid-scm-30000-i0']
    head=diagnosis['node_diagnostics']['engagement_rate']
    for key,value in [('AWorseLocalMSE',head['observed_parent_mse']),('AWorseChainMSE',head['free_running_mse']),
                      ('AWorseShiftMSE',head['propagated_prediction_shift_mse'])]:
        mantissa,exponent=f'{value:.2e}'.split('e');macros[key]=mantissa+'\\times10^{'+str(int(exponent))+'}'
    macros['AWorseOutsidePercent']=f"{100*head['outside_training_parent_box_fraction']:.3f}"
    macros['AWorseSnapped']=f"{1-diagnosis['score']['exact']:.3f}"
    # Descriptive additions requested by review; no new outcomes or selection.
    unused=[c['delivery']['nmse']/c['unused_observations_ablation']['nmse'] for c in gate['histories']]
    snapped=[(1-c['delivery']['exact'])/(1-c['unused_observations_ablation']['exact']) for c in gate['histories']]
    macros.update({'AUnusedWins':str(sum(v<1 for v in unused)),
        'AUnusedMin':f'{min(unused):.3f}','AUnusedMax':f'{max(unused):.3f}',
        'AUnusedSnappedRatio':f'{math.exp(sum(map(math.log,snapped))/12):.3f}'})
    for init in (1,2):
        numerator=summary['configurations'][f'all_paid-scm-30000-i{init}']['geomean_nmse']
        denominator=summary['configurations'][f'all_paid-flat-30000-i{init}']['geomean_nmse']
        macros['AFlatInit'+{1:'One',2:'Two'}[init]+'Ratio']=f'{numerator/denominator:.3f}'
    diagnostic_lines=['% Generated from accepted fixed-init0 attribution rows; no selection.',
        '\\begin{table}[ht]','\\centering\\small','\\begin{tabular}{lrrrrr}',
        '\\toprule','History & Online NMSE & SCM NMSE & Flat NMSE & SCM/admitted & SCM snap\\\\','\\midrule']
    for c in gate['histories']:
        diagnostic_lines.append(str(c['seed'])+' & '+
            ' & '.join(f"{c[key]['nmse']:.6f}" for key in ('online','delivery','simpler'))+
            f" & {c['delivery']['nmse']/c['unused_observations_ablation']['nmse']:.3f}"+
            f" & {1-c['delivery']['exact']:.3f}\\\\")
    diagnostic_lines+=['\\bottomrule','\\end{tabular}',
        '\\caption{Every archived history at fixed initialization zero. SCM and flat draw their respective eligible subsets from the same paid history and use 30,000 full-batch updates per SCM head and for the flat model. A mechanism head excludes rows that clamp that mechanism; the root-to-target flat fit excludes rows that clamp any nonroot node. Admitted refers to the long SCM fit on the admitted-row union. NMSE uses the shared exposed-grid target variance. The final column is SCM exact-level error. These exploratory outcomes retain history124753321 and are not prospective tests.}',
        '\\label{tab:attribution-histories}','\\end{table}']
    (ROOT/'paper/aistats_ace_2027/delivery_history_table.tex').write_text('\n'.join(diagnostic_lines)+'\n')
    lines=['% Generated exploratory init0 matrix; no inference from exposed grid.',
           '\\begin{table}[ht]','\\centering\\small',
           '\\begin{tabular}{llrrrr}','\\toprule',
           'Data & Learner & Epochs & NMSE/online & Snap/online & CPU h\\\\','\\midrule']
    for label,values in summary['configurations'].items():
        if not label.endswith('-i0'):continue
        dataset=next(s for s in ('final_buffer','online_admitted','all_paid') if label.startswith(s+'-'))
        # Explicit parsing avoids treating matched-CPU as30,000epochs.
        parts=label[len(dataset)+1:].split('-');learner=parts[0]
        epochs='--' if learner not in ('scm','flat') else 'CPU' if parts[1]=='matched' else f'{int(parts[1]):,}'
        data_label={'final_buffer':'Final50','online_admitted':'Admitted','all_paid':'All paid'}[dataset]
        cpu=f"{values['fit_cpu_core_hours']:.3f}" if values['fit_cpu_core_hours']>=.001 else '$<0.001$'
        lines.append(f"{data_label} & {learner.upper() if learner=='scm' else learner} & {epochs} & "
                     f"{values['geomean_nmse_ratio_online']:.3f} & {values['geomean_snapped_ratio_online']:.3f} & "
                     f"{cpu}\\\\")
    lines+=['\\bottomrule','\\end{tabular}',
            '\\caption{Exploratory primary initialization0 across the twelve archived histories. Ratios are geometric means relative to unchanged online weights; lower is better. CPU hours sum fit process time across the twelve fits in each row, excluding evaluation and imports performed before the fit timer. Classical-library imports inside the timer remain included. SCM cost includes all five heads. CPU denotes the matched-CPU flat fit, whose update count varies. Epoch counts apply only to neural fits. All configurations and sensitivity initializations are retained in the receipt.}',
            '\\label{tab:attribution}','\\end{table}']
    (ROOT/'paper/aistats_ace_2027/delivery_attribution_table.tex').write_text('\n'.join(lines)+'\n')
    attribution_index={'gate_sha256':digest(gate_path),'summary_sha256':digest(attribution/'summary.json'),
        'complete_sha256':digest(attribution/'complete.json'),'score_hashes':done['score_hashes'],
        'scope':'480 fits, one emulator, full exposed grid; exploratory; init0 primary',
        'worsening_diagnostic_source':'scores/124753321.json, all_paid-scm-30000-i0',
        'descriptive_review_additions':'unused-row range/wins, init1/2 flat ratios and all12init0 errors from accepted summary and gate; no new fits'}
    attribution_index['normalizer']='MSE divided by population target variance on the shared exposed grid; not training variance'
    attribution_index['input_protocol_sha256']=digest(ROOT/'results/delivery_paper_implementation_20261006/stage_a_input_protocol.json')
physical_index=None
physical=ROOT/'results/delivery_chambers_20261006'
if physical.exists():
    def digest(file):return hashlib.sha256(file.read_bytes()).hexdigest()
    accepted=json.loads((physical/'acceptance.json').read_text())
    assert accepted['full_acceptance'] and accepted['selected_gate_sha256']==digest(gate_path)
    for file,h in accepted['receipt_hashes'].items():assert digest(physical/file)==h
    scored=json.loads((physical/'scores.json').read_text())
    assert set(scored['conditions'])==set(accepted['conditions']) and len(scored['conditions'])==11
    controls=('rolling_buffer','physics','fourier')
    for control,macro in zip(controls,('CRollingWins','CPhysicsWins','CFourierWins')):
        count=sum(v['nmse']['delivery']<v['nmse'][control] for v in scored['conditions'].values())
        assert count==accepted['descriptive_comparisons'][control]['conditions_with_lower_row_weighted_nmse']
        macros[macro]=str(count)
    macros['CCPU']=f"{accepted['cpu_core_hours']:.3f}"
    def cell(value):
        if value>=.001:return f'{value:.3f}'
        mantissa,exponent=f'{value:.2e}'.split('e')
        return '$'+mantissa+'\\times10^{'+str(int(exponent))+'}$'
    lines=['% Generated from the fully accepted eleven-condition physical receipt.',
           '\\begin{table}[ht]','\\centering\\small','\\begin{tabular}{lrrrr}',
           '\\toprule','Condition & Delivery & Rolling & Physics & Fourier\\\\','\\midrule']
    for condition,v in scored['conditions'].items():
        lines.append(condition.replace('_','\\_')+' & '+' & '.join(cell(v['nmse'][key]) for key in ('delivery',)+controls)+'\\\\')
    lines+=['\\bottomrule','\\end{tabular}',
            '\\caption{Continuous held-out MSE divided by the training variance for all eleven evaluation conditions, excluding development condition \\texttt{white\\_64}. Lower is better. These conditions are readings of one apparatus, not independent systems.}',
            '\\label{tab:physical}','\\end{table}',
            '\\begin{table}[ht]','\\centering\\scriptsize','\\begin{tabular}{lrrr}',
            '\\toprule','Condition & Delivery/Rolling & Delivery/Physics & Delivery/Fourier\\\\','\\midrule']
    for condition,v in scored['conditions'].items():
        cells=[]
        for control in controls:
            q=v['conditional_uncertainty'][control];lo,hi=q['conditional_bootstrap_ci95']
            cells.append(f"{q['block_weighted_ratio']:.2f} [{lo:.2f}, {hi:.2f}]")
        lines.append(condition.replace('_','\\_')+' & '+' & '.join(cells)+'\\\\')
    lines+=['\\bottomrule','\\end{tabular}',
            '\\caption{Equal-action-block error ratios and conditional 95\\% bootstrap intervals. Eight held-out blocks are resampled within each condition. These ratios weight blocks equally, whereas Table~\\ref{tab:physical} weights held-out rows equally. Intervals describe uncertainty conditional on this apparatus and observed blocks; they are not independent-world confidence intervals or multiplicity-adjusted superiority tests.}',
            '\\label{tab:physical-ci}','\\end{table}']
    assert all(q['n_action_blocks']==8 for v in scored['conditions'].values() for q in v['conditional_uncertainty'].values())
    (ROOT/'paper/aistats_ace_2027/delivery_physical_table.tex').write_text('\n'.join(lines)+'\n')
    physical_index={'acceptance_sha256':digest(physical/'acceptance.json'),'receipt_hashes':accepted['receipt_hashes'],
                    'scope':accepted['scope'],'all_conditions_reported':True,'new_queries':0}
tex='% Generated by scripts/research/generate_delivery_claims.py; do not hand edit.\n'
tex+='\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n'
(ROOT/'paper/aistats_ace_2027/delivery_claims.tex').write_text(tex)
index={'scores_sha256':hashlib.sha256(raw).hexdigest(),
       'statistics_sha256':hashlib.sha256((folder/'independent_statistics.json').read_bytes()).hexdigest(),
       'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'terminal_audit_sha256':hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
       'macros':macros,'scope':'12 histories of one emulator, median scored optimization performance, full exposed grid',
       'pending_claims':['causal architecture attribution','equal-compute advantage','prospective generalization','external physical superiority'],
       'removed_claims':['acquisition superiority','foundation-model benefit','DPO optimality','unrestricted causal identification','generic MSE gain equals mutual information']}
if confirmation_accounting:index['confirmation_accounting']=confirmation_accounting
if attribution_index:index['attribution']=attribution_index
if physical_index:
    index['physical']=physical_index
    index['pending_claims'].remove('external physical superiority')
    index['reported_boundaries']=['Physical delivery loses to Fourier in all eleven conditions; no superiority claim over that control.']
(ROOT/'paper/aistats_ace_2027/claim_index.json').write_text(json.dumps(index,indent=2)+'\n')
