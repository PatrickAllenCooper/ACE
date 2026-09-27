#!/usr/bin/env python3
"""Build a public-only prompt and validate a bounded mechanism proposal.

This module does not invoke any model. The allowed form is a structural
choice, not arbitrary Python or symbolic code from a text generator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ALLOWED = {'coupled_bilinear_ode', 'independent_exponential', 'unsure'}


def build_prompt(public: Path, condition: str) -> str:
    if condition not in ('descriptive', 'anonymous'):
        raise ValueError('unknown condition')
    message = (public / f'{condition}_message.txt').read_text()
    with (public / 'observations.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        raise ValueError('expected eight public observations')
    observations = '\n'.join(f"t={float(r['time']):.6f}: ({float(r['response_1']):.0f}, "
                             f"{float(r['response_2']):.0f})" for r in rows)
    return ("You are proposing a mathematical structure for a two-response time series. "
            "Use only the description and observations below. Return one JSON object "
            "with exactly two keys: family and reason. family must be one of "
            "coupled_bilinear_ode, independent_exponential, unsure. "
            "The coupled form allows each rate of change to depend on the product of both responses; "
            "the independent form gives each response its own exponential rate. "
            "Use unsure when evidence is insufficient. Do not supply parameter values, code, "
            "predictions, or additional keys. Keep reason under 160 characters.\n\n"
            f"Task description:\n{message}\nObserved time-response pairs:\n{observations}\n")


def parse_proposal(raw: str) -> dict:
    if len(raw) > 4000:
        raise ValueError('oversized response')
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        lines = cleaned.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('invalid fenced JSON')
        cleaned = '\n'.join(lines[1:-1])
    value = json.loads(cleaned)
    if not isinstance(value, dict) or set(value) != {'family', 'reason'}:
        raise ValueError('expected only family and reason')
    if value['family'] not in ALLOWED or not isinstance(value['reason'], str):
        raise ValueError('invalid family or reason')
    if len(value['reason']) > 160:
        raise ValueError('reason too long')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--condition', choices=('descriptive', 'anonymous'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prompt = build_prompt(args.public, args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt)
    print(hashlib.sha256(prompt.encode()).hexdigest())


if __name__ == '__main__':
    main()
