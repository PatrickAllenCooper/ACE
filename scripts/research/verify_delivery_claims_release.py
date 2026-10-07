"""Recompute current receipt-generated empirical macros from a private release.

Standard-library metadata analysis only: no checkpoint loading, optimization,
response acquisition, B losses or historical freeze proof. Tables, prose, final
B claims and human anonymity review remain separate gates.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from verify_delivery_release import verify
from replay_delivery_confirmation_release import accounting, read


def macros(root):
    root = Path(root); stats = read(root/'F/statistics.json'); counts = accounting(root)
    scores = read(root/'F/scores.json'); worsening = stats['worsening_histories']
    if len(scores['pairs']) != 12 or len(worsening) != 1 or worsening[0]['seed'] != 124753321:
        raise ValueError('confirmation macro cohort differs')
    w = worsening[0]
    result = {'DeliveryN': str(stats['n']), 'DeliveryRatio': f"{stats['ratio']:.3f}",
              'DeliveryCILow': f"{stats['ci_lo']:.3f}", 'DeliveryCIHigh': f"{stats['ci_hi']:.3f}",
              'DeliveryP': f"{stats['exact_sign_flip_p']:.8f}", 'DeliveryWins': str(len(scores['pairs'])-len(worsening)),
              'WorseSeed': str(w['seed']), 'WorseOnline': f"{w['online_error']:.6f}",
              'WorseDelivery': f"{w['delivery_median_error']:.6f}", 'WorseRatio': f"{w['ratio']:.6f}",
              'ConfirmationChargedCalls': f"{counts['aggregate_charged_attempts']:,}",
              'PersistedConfirmationResponses': f"{counts['persisted_complete_responses']:,}",
              'InterruptedConfirmationAttempts': f"{counts['interrupted_charged_reservations']:,}"}
    gate, summary, complete = (read(root/'A'/n) for n in ('gate.json', 'summary.json', 'complete.json'))
    ratios = {key: math.exp(sum(math.log(max(c['delivery']['nmse'], 1e-12)/max(c[key]['nmse'], 1e-12))
                               for c in gate['histories'])/12) for key in gate['ratios_continuous_nmse_init0']}
    for key, value in ratios.items():
        if not math.isclose(value, gate['ratios_continuous_nmse_init0'][key], rel_tol=1e-12):
            raise ValueError('attribution aggregate differs from histories')
    for name, key in [('AOnlineRatio', 'online'), ('AFlatRatio', 'simpler'), ('AMatchedRatio', 'matched_cpu_flat'),
                      ('ABufferRatio', 'data_ablation'), ('AOptimizationRatio', 'optimization_ablation'),
                      ('AUnusedRatio', 'unused_observations_ablation')]:
        result[name] = f'{ratios[key]:.3f}'
    result['AFitCPU'] = f"{complete['fit_cpu_core_hours']:.2f}"
    diagnosis = summary['worsening_history']['fits']['all_paid-scm-30000-i0']
    head = diagnosis['node_diagnostics']['engagement_rate']
    for name, key in [('AWorseLocalMSE', 'observed_parent_mse'), ('AWorseChainMSE', 'free_running_mse'),
                      ('AWorseShiftMSE', 'propagated_prediction_shift_mse')]:
        mantissa, exponent = f'{head[key]:.2e}'.split('e')
        result[name] = mantissa+'\\times10^{'+str(int(exponent))+'}'
    result['AWorseOutsidePercent'] = f"{100*head['outside_training_parent_box_fraction']:.3f}"
    result['AWorseSnapped'] = f"{1-diagnosis['score']['exact']:.3f}"
    unused = [c['delivery']['nmse']/c['unused_observations_ablation']['nmse'] for c in gate['histories']]
    snapped = [(1-c['delivery']['exact'])/(1-c['unused_observations_ablation']['exact']) for c in gate['histories']]
    result.update({'AUnusedWins': str(sum(v < 1 for v in unused)), 'AUnusedMin': f'{min(unused):.3f}',
                   'AUnusedMax': f'{max(unused):.3f}', 'AUnusedSnappedRatio': f'{math.exp(sum(map(math.log, snapped))/12):.3f}'})
    for init, name in [(1, 'AFlatInitOneRatio'), (2, 'AFlatInitTwoRatio')]:
        numerator = summary['configurations'][f'all_paid-scm-30000-i{init}']['geomean_nmse']
        denominator = summary['configurations'][f'all_paid-flat-30000-i{init}']['geomean_nmse']
        result[name] = f'{numerator/denominator:.3f}'
    physical, accepted = read(root/'C/scores.json'), read(root/'C/acceptance.json')
    if set(physical['conditions']) != set(accepted['conditions']) or len(physical['conditions']) != 11:
        raise ValueError('physical condition membership differs')
    for control, name in [('rolling_buffer', 'CRollingWins'), ('physics', 'CPhysicsWins'), ('fourier', 'CFourierWins')]:
        wins = sum(c['nmse']['delivery'] < c['nmse'][control] for c in physical['conditions'].values())
        if wins != accepted['descriptive_comparisons'][control]['conditions_with_lower_row_weighted_nmse']:
            raise ValueError('physical macro differs from accepted boundary')
        result[name] = str(wins)
    result['CCPU'] = f"{accepted['cpu_core_hours']:.3f}"
    return result


def check(root, expected):
    root = Path(root).resolve(); integrity = verify(root, expected)
    actual = macros(root); index = read(root/'claims/claim_index.json')
    if actual != index['macros']:
        raise ValueError('empirical macros differ from reconstructed receipt values')
    expected_tex = ('% Generated by scripts/research/generate_delivery_claims.py; do not hand edit.\n'+
                    '\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name, value in index['macros'].items())+'\n')
    if (root/'claims/delivery_claims.tex').read_text() != expected_tex:
        raise ValueError('LaTeX empirical macros differ from bound index')
    return {'integrity': integrity, 'empirical_macros_reconstructed': len(actual),
            'latex_macro_bytes_match_index': True, 'accounting': accounting(root),
            'new_fits': 0, 'new_responses': 0,
            'scope': 'current A/C/confirmation empirical macro metadata analysis; not all prose/table verification or B claims'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.expected_manifest_sha256), indent=2))
