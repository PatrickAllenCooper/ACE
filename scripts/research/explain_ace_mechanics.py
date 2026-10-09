"""Exact, illustrative ACE acquisition arithmetic; never queries an experiment.

This uses the covariance score in PropagatedVariancePolicy with a deliberately
small linear ensemble, fixed references and no candidate jitter/exploration.
The update is transparent one-step gradient descent, NOT production Adam.
"""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def mean(xs):
    return sum(xs, F(0)) / len(xs)


def ivr(slopes, x, references, noise):
    predictions = [a*x for a in slopes]
    centered = [y-mean(predictions) for y in predictions]
    variance = mean([a*a for a in centered])
    covariances = []
    for r in references:
        ref = [a*r for a in slopes]
        covariances.append(mean([a*(b-mean(ref)) for a, b in zip(centered, ref)]))
    return mean([c*c for c in covariances]) / (variance+noise)


def demonstration(repo):
    a, b = [F(1), F(2), F(3)], [F(2), F(3), F(4)]
    refs, noise = [F(-1), F(0), F(1)], F(1, 20)
    candidates = []
    for node in ('X', 'M'):
        for x in refs:
            intermediate = mean(a)*x if node == 'X' else x
            score_m = ivr(a, x, refs, noise) if node == 'X' else F(0)
            score_y = ivr(b, intermediate, refs, noise)
            candidates.append(dict(node=node, value=x, predicted_M=intermediate,
                                   predicted_Y=mean(b)*intermediate,
                                   score_M=score_m, score_Y=score_y,
                                   score=score_m+score_y))
    chosen = max(candidates, key=lambda r: r['score'])
    assert (chosen['node'], chosen['value']) == ('X', F(-1))
    assert candidates[0]['score'] == F(160, 387)+F(640, 1467)
    assert candidates[3]['score'] == F(160, 387)
    x = chosen['value']
    m, y = F(5, 2)*x, 3*F(5, 2)*x
    # Gradient descent on 1/2*(prediction-label)^2 with measured inputs.
    next_a = [v-F(1, 2)*(v*x-m)*x for v in a]
    next_b = [v-F(2, 25)*(v*m-y)*m for v in b]
    after_m, after_y = mean(next_a)*x, mean(next_b)*mean(next_a)*x
    assert next_a == [F(7, 4), F(9, 4), F(11, 4)]
    assert next_b == [F(5, 2), F(3), F(7, 2)]
    assert after_y == F(-27, 4)
    trace = repo/'results/research_pev_shift30_mean_confirmation/shift30/pev/seed_5000'
    raw = (trace/'trajectory.csv').read_bytes()
    with (trace/'trajectory.csv').open(newline='') as f:
        row = next(csv.DictReader(f))
    assert row['step'] == '0' and row['target'] == 'X7'
    assert row['value'] == '3.857086181640625' and row['query_samples'] == '50'
    return {
        'schema': 'ace-mechanistic-demonstration-v1',
        'scope': 'Illustrative exact arithmetic, not a fitted study or production replay',
        'graph': [['X', 'M'], ['M', 'Y']],
        'true_rules': {'M': '2.5*X', 'Y': '3*M', 'noise': 0},
        'ensemble_before': {'M_slopes': a, 'Y_slopes': b},
        'score_configuration': {'references': refs, 'noise_proxy': noise,
                                'candidate_jitter': False, 'epsilon_exploration': 0,
                                'tie_rule': 'first maximum in listed order',
                                'uncertainty': 'population ensemble covariance'},
        'candidates': candidates, 'selected': chosen,
        'illustrative_response': {'X': x, 'M': m, 'Y': y},
        'eligible_training_pairs': {'M': [x, m], 'Y': [m, y]},
        'excluded_label': 'X is clamped; do not update its natural distribution from this row',
        'illustrative_update': {'loss': 'half squared error per head/member',
                                'M_step_size': F(1, 2), 'Y_step_size': F(2, 25),
                                'M_slopes_after': next_a, 'Y_slopes_after': next_b,
                                'predicted_M_after': after_m, 'predicted_Y_after': after_y,
                                'production_difference': 'Production uses neural heads, Adam and member-specific batch masks; this update is explanatory only'},
        'recorded_trace': {'path': str(trace.relative_to(repo)),
                           'trajectory_sha256': hashlib.sha256(raw).hexdigest(),
                           'selection_rule': 'first step of smallest study seed 5000',
                           'first_action': {k: row[k] for k in ('step', 'target', 'value', 'query_samples')},
                           'budget': json.loads((trace/'query_budget.json').read_text()),
                           'unavailable': ['candidate scores', 'raw response rows', 'per-step ensemble weights']},
        'production_sources': ['baselines.py:PropagatedVariancePolicy', 'scripts/research/persistent_scm.py:campaign'],
        'environmental_responses_collected': 0,
        'accepted_artifacts_refitted': 0,
    }


def encode(value):
    if isinstance(value, F):
        return {'exact': str(value), 'decimal': float(value)}
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = demonstration(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, default=encode)
        f.write('\n')
    print(args.output)
