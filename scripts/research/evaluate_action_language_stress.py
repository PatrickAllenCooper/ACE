#!/usr/bin/env python3
"""Evaluate the unchanged rule parser on a separate text/schema fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from action_language_fixture import legal_actions
from action_language_rule_baseline import parse_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prompts = [json.loads(line) for line in (args.fixture/'prompts.jsonl').read_text().splitlines()]
    answers = {row['id']: row for row in (json.loads(line) for line in
               (args.fixture/'answer_key.jsonl').read_text().splitlines())}
    if len(prompts) != len(answers) or {x['id'] for x in prompts} != set(answers):
        raise ValueError('fixture ID mismatch')
    results = []
    for task in prompts:
        correct = answers[task['id']]
        try:
            parsed = parse_schema(task['description'])
            predicted = legal_actions(parsed)
            results.append({'id': task['id'], 'parse_error': None,
                            'predicted_schema': parsed,
                            'exact_schema': parsed == correct['schema'],
                            'exact_legal_menu': predicted == correct['legal_actions'],
                            'predicted_legal_count': len(predicted),
                            'correct_legal_count': correct['legal_action_count']})
        except (ValueError, KeyError, TypeError) as exc:
            results.append({'id': task['id'], 'parse_error': str(exc),
                            'predicted_schema': None, 'exact_schema': False,
                            'exact_legal_menu': False,
                            'predicted_legal_count': None,
                            'correct_legal_count': correct['legal_action_count']})
    args.output.mkdir(parents=True)
    payload = ''.join(json.dumps(row, sort_keys=True)+'\n' for row in results)
    (args.output/'evaluation.jsonl').write_text(payload)
    (args.output/'complete.json').write_text(json.dumps({
        'tasks': len(results), 'exact_schemas': sum(x['exact_schema'] for x in results),
        'exact_legal_menus': sum(x['exact_legal_menu'] for x in results),
        'parse_errors': sum(x['parse_error'] is not None for x in results),
        'fixture_complete_sha256': hashlib.sha256((args.fixture/'complete.json').read_bytes()).hexdigest(),
        'evaluation_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'simulator_queries': 0, 'model_calls': 0,
    }, indent=2, sort_keys=True)+'\n')
    print(f"exact menus {sum(x['exact_legal_menu'] for x in results)}/{len(results)}")


if __name__ == '__main__':
    main()
