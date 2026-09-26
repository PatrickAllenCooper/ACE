#!/usr/bin/env python3
"""Validate one research cell's actual artifacts, independent of Slurm state."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def valid(directory: Path, kind: str, steps: int = 8) -> tuple[bool, str]:
    try:
        if kind == 'motif_reachability':
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] == 1 and receipt['kind'] == kind
            metrics, spec_file = directory / 'metrics.csv', directory / 'system.json'
            assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
            assert hashlib.sha256(spec_file.read_bytes()).hexdigest() == receipt['system_sha256']
            spec = json.loads(spec_file.read_text())
            with metrics.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == receipt['rows'] == 5
            assert {r['method'] for r in rows} == {'random_single', 'coverage_single', 'random_pair', 'coverage_pair', 'risk_pair'}
            assert len(spec['edges']) == 2 * spec['motifs'] and len(spec['coefficients']) == spec['motifs']
            assert spec['nodes'] >= 3 * spec['motifs']
            for row in rows:
                assert (int(row['seed']), int(row['nodes']), int(row['motifs'])) == (spec['seed'], spec['nodes'], spec['motifs'])
                assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                           for k in ('motif_mse', 'interaction_mse', 'posterior_risk'))
                samples, steps, spent, actuators = (int(row[k]) for k in ('samples', 'steps', 'cost_spent', 'actuator_uses'))
                assert samples == 8 * steps and samples > 0
                assert spent == samples + int(row['penalty']) * actuators <= int(row['budget'])
            return True, '5 SCM acquisition arms with exact costs'
        if kind == 'design_b2':
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] == 1 and receipt['kind'] == 'design_b2'
            metrics = directory / 'metrics.csv'
            assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
            with metrics.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == receipt['rows'] == 5
            assert {row['method'] for row in rows} == {'random_single', 'ivr_single', 'random_pair', 'coverage_pair', 'ivr_pair'}
            assert len({(row['seed'], row['background_sd'], row['penalty'], row['budget']) for row in rows}) == 1
            for row in rows:
                assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                           for k in ('mse', 'interaction_error', 'posterior_risk'))
                samples, steps, spent, actuators = (int(row[k]) for k in ('samples', 'steps', 'cost_spent', 'actuator_uses'))
                assert samples == 8 * steps and samples > 0
                assert spent == samples + int(row['penalty']) * actuators <= int(row['budget'])
            return True, '5 matched-menu numerical arms with exact costs'
        if kind == 'learned_transfer':
            receipt = json.loads((directory / 'complete.json').read_text())
            assert receipt['schema_version'] in (1, 2) and receipt['kind'] == 'learned_transfer'
            metrics = directory / 'metrics.csv'
            spec_file = directory / 'system.json'
            assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
            assert hashlib.sha256(spec_file.read_bytes()).hexdigest() == receipt['system_sha256']
            spec = json.loads(spec_file.read_text())
            with metrics.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == receipt['rows'] == (15 if receipt['schema_version'] == 2 else 12)
            assert len(spec['changed_node_ids']) == spec['changed']
            assert len(spec['source_sha256']) == 64
            assert {int(row['target_budget']) for row in rows} == {120, 200, 400}
            methods = {'scratch', 'warm', 'source_retrieval', 'source_mixture'}
            if receipt['schema_version'] == 2:
                assert spec['guard_margin'] is not None
                methods.add('source_guarded')
            assert {row['method'] for row in rows} == methods
            assert all(int(row['source_samples']) == spec['source_samples'] for row in rows)
            assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                       for row in rows for k in ('mse', 'changed_mse', 'unchanged_mse'))
            return True, f'{len(rows)} valid transfer comparisons'
        if kind == 'persistent':
            receipt = json.loads((directory / 'complete.json').read_text())
            metrics = directory / 'trajectory.csv'
            assert receipt['schema_version'] in (1, 2)
            assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
            if receipt['schema_version'] == 2:
                spec_file = directory / 'system.json'
                assert hashlib.sha256(spec_file.read_bytes()).hexdigest() == receipt['system_sha256']
                spec = json.loads(spec_file.read_text())
                assert (spec['family'], spec['seed']) == (receipt['family'], receipt['seed'])
                assert len(spec['nodes']) == len(spec['graph'])
            with metrics.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == receipt['steps'] and rows
            samples = [int(row['query_samples']) for row in rows]
            assert all(int(row['step']) == i for i, row in enumerate(rows))
            assert all(a < b for a, b in zip(samples, samples[1:]))
            assert samples[-1] == receipt['query_samples'] <= receipt['budget']
            assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                       for row in rows for k in ('broad_total_loss', 'observed_total_loss'))
            if receipt['schema_version'] == 2:
                assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                           for row in rows for k in ('broad_nonroot_loss',
                                                    'observed_nonroot_loss',
                                                    'feasible_nonroot_loss'))
            queries = json.loads((directory / 'query_budget.json').read_text())
            assert queries['total']['samples'] == samples[-1]
            assert queries.get('candidate_probe', {}).get('samples', 0) == 0
            return True, f"{len(rows)} steps, {samples[-1]} samples"
        if kind == 'agenda':
            receipt = json.loads((directory / 'complete.json').read_text())
            metrics = directory / 'metrics.csv'
            assert receipt['schema_version'] == 1
            assert receipt['all_finite'] and receipt['rows'] > 0
            assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
            with metrics.open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == receipt['rows']
            assert all(math.isfinite(float(row['mse'])) and float(row['mse']) >= 0 for row in rows)
            return True, f"{len(rows)} valid metrics"
        receipt = json.loads((directory / 'complete.json').read_text())
        assert receipt['kind'] == 'pev_canary' and receipt['steps'] == steps
        file_name = 'node_losses.csv' if (directory / 'node_losses.csv').exists() else 'results.csv'
        metrics = directory / file_name
        assert hashlib.sha256(metrics.read_bytes()).hexdigest() == receipt['metrics_sha256']
        with metrics.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == steps
        assert all(int(row['episode']) == 0 and int(row['step']) == i for i, row in enumerate(rows))
        assert all(math.isfinite(float(row['ace_total_loss'])) for row in rows)
        queries = json.loads((directory / 'query_budget.json').read_text())
        assert queries['executed']['samples'] == steps * 50
        assert queries['total']['samples'] >= steps * 50
        assert queries.get('candidate_probe', {}).get('samples', 0) == 0
        with (directory / 'summary.csv').open() as stream:
            summary = list(csv.DictReader(stream))
        assert len(summary) == 1
        return True, f"{len(rows)} steps, {queries['total']['samples']} samples"
    except (OSError, ValueError, KeyError, AssertionError, csv.Error) as exc:
        return False, str(exc) or 'artifact check failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--kind', choices=('agenda', 'pev', 'persistent', 'learned_transfer', 'design_b2', 'motif_reachability'), required=True)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    ok, detail = valid(args.directory, args.kind, args.steps)
    print(('VALID' if ok else 'INVALID') + ': ' + str(args.directory) + ': ' + detail)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
