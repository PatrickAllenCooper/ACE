#!/usr/bin/env python3
"""Score frozen two-stage partial-ID responses without model access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {item['id']: item for item in items}
    if len(result) != len(items):
        raise ValueError(f'duplicate id in {path}')
    return result


def number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError('Boolean is not a number')
    return float(Fraction(str(value)))


def score(public: dict, reveal: dict, key: dict, answer: dict) -> dict:
    expected_interval = [number(x) for x in key['candidate_set_doX1_mean_interval']]
    expected_posterior = [number(x) for x in key['posterior_A_B']]
    stage1, stage2 = answer.get('stage1'), answer.get('stage2')
    output = {'id': key['id'], 'informative_menu': key['best_action'] != 'abstain_unresolved',
              'stage1_valid': isinstance(stage1, dict), 'stage2_valid': isinstance(stage2, dict),
              'interval_valid': False, 'identification_correct': False,
              'action_correct': False, 'false_certainty': False, 'false_abstention': False,
              'posterior_valid': False, 'posterior_mae': None, 'posterior_log_loss': None}
    output['posterior_support_correct'] = False
    if isinstance(stage1, dict):
        try:
            interval = stage1['candidate_set_doX1_mean_interval']
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError('interval must be a pair')
            endpoints = [number(v) for v in interval]
            output['interval_valid'] = all(math.isfinite(v) for v in endpoints) and all(
                abs(a - b) <= 1e-6 for a, b in zip(endpoints, expected_interval))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        output['identification_correct'] = stage1.get('observations_identify_candidate') is False
        action = stage1.get('action')
        output['action_correct'] = action == key['best_action']
        output['false_certainty'] = stage1.get('observations_identify_candidate') is True
        output['false_abstention'] = action == 'abstain_unresolved' and output['informative_menu']
        output['illegal_action'] = action not in public['legal_actions'] and action != 'abstain_unresolved'
    else:
        output['illegal_action'] = False
    if isinstance(stage2, dict):
        try:
            posterior = stage2['posterior_A_B']
            if not isinstance(posterior, list) or len(posterior) != 2:
                raise ValueError('posterior must be a pair')
            values = [number(v) for v in posterior]
            if all(math.isfinite(v) and 0 <= v <= 1 for v in values) and abs(sum(values) - 1) <= 1e-6:
                output['posterior_valid'] = True
                output['posterior_mae'] = sum(abs(a-b) for a,b in zip(values, expected_posterior))/2
                # Cross entropy against the exact posterior, with a finite clipping convention.
                output['posterior_log_loss'] = -sum(
                    truth*math.log(max(pred, 1e-12)) for truth,pred in zip(expected_posterior,values)
                    if truth > 0)
                expected_support = [name for name,weight in zip(('A','B'),expected_posterior) if weight > 0]
                output['posterior_support_correct'] = stage2.get('posterior_supported_candidates') == expected_support
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--reveals', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    public, reveals, keys, responses = (rows(p) for p in
        (args.public, args.reveals, args.key, args.responses))
    if set(responses) - set(keys):
        raise ValueError('unknown response IDs')
    if set(public) != set(reveals) or set(public) != set(keys):
        raise ValueError('task ID mismatch')
    scored = [score(public[i], reveals[i], keys[i], responses.get(i, {})) for i in sorted(keys)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(v, sort_keys=True)+'\n' for v in scored))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({'n_tasks':len(scored),'n_responses':len(responses),
                      'scored_sha256':digest,'action_correct':sum(v['action_correct'] for v in scored),
                      'posterior_valid':sum(v['posterior_valid'] for v in scored)},sort_keys=True))


if __name__ == '__main__':
    main()
