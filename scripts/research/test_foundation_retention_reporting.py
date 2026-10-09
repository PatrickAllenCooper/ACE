"""Fabricated saved files only: no worker, worlds, estimator or PFN imports."""
import copy
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import summarize_foundation_retention as report


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def read(path):
    return json.loads(Path(path).read_text())


def tiny_arrays():
    # Repeated hand-authored coordinates, not RNG samples or a scientific world.
    x = np.resize(np.array([-1., 0., 1.]), 32)
    train = np.column_stack((x, x, 2*x+.5))
    clamp = np.array(report.LAYOUT)
    comp = np.tile([.75, .75, 2.], (256, 1))
    lm = np.tile([.25, .25, 1.], (256, 1))
    m = np.resize(np.array([-2., -1., 0., 1., 2.]), 256)
    ly = np.column_stack((np.zeros(256), m, 2*m+.5))
    pred = np.column_stack((np.full(256, .35), ly[:, 2]+.1, np.full(256, 1.9)))
    parents = np.full((256, 1), 2.)
    return train, clamp, (comp, lm, ly), pred, parents


def choice(method):
    names = list(report.ORDER[:-1] if method == 'combined_no_pfn' else report.ORDER)
    mode = 'combined' if method == 'combined_no_pfn' else method
    admitted = [names, names] if mode in ('raw', 'interval') else [['retained'], ['retained']]
    return {'mode': mode, 'names': names,
            'intervals': [{'lower': -1., 'upper': 1.}]*2,
            'local_errors': [[node, name, 1.] for node in ('M', 'Y') for name in names],
            'admitted': admitted,
            'composed_errors': [[list(pair), 4.] for pair in itertools.product(*admitted)],
            'selected': ['retained', 'retained'], 'composed_mse': 4., 'calibration_inside': [5, 8]}


def fabricated_metrics(pred, probes, train):
    targets = (probes[1][:, 1], probes[2][:, 2], probes[0][:, 2])
    variances = (float(np.var(train[:20, 1])), float(np.var(train[:, 2])), float(np.var(train[:, 2])))
    return {ep: {'mse': float(np.mean((pred[:, j]-targets[j])**2)),
                 'nmse': float(np.mean((pred[:, j]-targets[j])**2))/max(v, 1e-12),
                 'training_variance': v, 'floor_active': v < 1e-12}
            for j, (ep, v) in enumerate(zip(report.ENDPOINTS, variances))}


def fabricated_diagnostics(method):
    # Independent oracle for the repeated coordinates in tiny_arrays().
    y_count = sum(-1 <= (-2., -1., 0., 1., 2.)[i % 5] <= 1 for i in range(256))
    return {'intervals': [{'lower': -1., 'upper': 1.}]*2,
            'local': {'M_local': {'inside': {'count': 256, 'mse': .01},
                                  'outside': {'count': 0, 'mse': None}},
                      'Y_local': {'inside': {'count': y_count, 'mse': .01},
                                  'outside': {'count': 256-y_count, 'mse': .01}}},
            'composed_Y_parent_inside': 0, 'composed_Y_parent_outside': 256,
            'composed_Y_parent_inside_rate': 0.,
            'interval_gating_enabled': method in ('interval', 'combined', 'combined_no_pfn')}


def build(root):
    train, clamp, probes, pred, parents = tiny_arrays()
    seed = 223456
    parent = root/str(seed); parent.mkdir()
    np.savez(parent/'prehistory.npz', train=train, clamp=clamp)
    write(parent/'prehistory.reserved.json', {'responses': 32, 'kind': 'training', 'at_unix': 1.})
    write(parent/'prehistory.returned.json', {'responses': 32, 'at_unix': 2., 'sha256': report.sha(parent/'prehistory.npz')})
    cells = []
    for variant in report.VARIANTS:
        d = parent/variant; d.mkdir()
        np.savez(d/'training.npz', train=train, clamp=clamp)
        write(d/'training.reserved.json', {'responses': 32, 'kind': 'training', 'at_unix': 3.})
        write(d/'training.returned.json', {'responses': 32, 'at_unix': 4., 'sha256': report.sha(d/'training.npz')})
        seal = {'at_unix': 5., 'training_sha256': report.sha(d/'training.npz'),
                'fit_indices': list(report.FIT_INDICES), 'calibration_indices': list(report.CALIBRATION_INDICES),
                'choices': {m: choice(m) for m in report.SELECTORS}, 'errors': {}, 'evaluation_generated': False}
        write(d/'selection_seal.json', seal)
        np.savez(d/'private_probes.npz', composed=probes[0], local_m=probes[1], local_y=probes[2])
        write(d/'evaluation.reserved.json', {'responses': 768, 'kind': 'private', 'at_unix': 6.,
                                           'selection_seal_sha256': report.sha(d/'selection_seal.json')})
        write(d/'evaluation.returned.json', {'private_responses': 768, 'at_unix': 7.,
                 'sha256': report.sha(d/'private_probes.npz'), 'selection_seal_sha256': report.sha(d/'selection_seal.json')})
        for method in report.METHODS:
            np.save(d/(method+'_predictions.npy'), pred)
            np.save(d/(method+'_parents.npy'), parents)
            row = {'seed': seed, 'variant': variant, 'method': method, 'status': 'complete',
                   'metrics': fabricated_metrics(pred, probes, train), 'diagnostics': fabricated_diagnostics(method),
                   'prediction_sha256': report.sha(d/(method+'_predictions.npy')),
                   'parent_sha256': report.sha(d/(method+'_parents.npy'))}
            write(d/(method+'.json'), row); cells.append(row)
    write(root/'plan.json', {'mode': 'fixture', 'cells': [{k: r[k] for k in ('seed', 'variant', 'method')} for r in cells]})
    completion = {'mode': 'fixture', 'cells': cells, 'planned_cells': 36, 'completed_cells': 36,
                  'training_responses_total': 160, 'private_responses_total': 3072}
    write(root/'complete.json', completion)
    freeze = {'schema': 'ace-retention-freeze-v1', 'mode': 'fixture',
              'sources': {report.REPORTER_KEY: report.sha(report.__file__)}}
    write(root/'freeze.json', freeze)
    terminal = {'status': 'complete', 'reason': 'exited', 'error': None, 'exit_code': 0,
                'mode': 'fixture', 'planned_cells': 36, 'cells': cells,
                'freeze_sha256': report.sha(root/'freeze.json'),
                **dict.fromkeys(report.RESOURCE_KEYS, 0)}
    write(root/'terminal.json', terminal)
    return freeze, terminal


def bind_seal(d):
    for leaf in ('evaluation.reserved.json', 'evaluation.returned.json'):
        obj = read(d/leaf); obj['selection_seal_sha256'] = report.sha(d/'selection_seal.json'); write(d/leaf, obj)


class SavedReporting(unittest.TestCase):
    def test_full_saved_contract_and_actual_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            result = report.verify(root, freeze, terminal)
            self.assertEqual(len(result['cells']), 36)
            self.assertEqual(len(result['comparisons']), 96)
            self.assertEqual(len(result['direct_comparisons']), 72)
            self.assertEqual(result['response_accounting']['training']['validated_returned'], 160)
            self.assertEqual(result['response_accounting']['private']['validated_returned'], 3072)
            self.assertTrue(all(h['mse_difference'] == 0 for h in result['local_harm']))
            command = [sys.executable, '-B', report.__file__, '--root', str(root),
                       '--freeze', str(root/'freeze.json'), '--terminal', str(root/'terminal.json'),
                       '--freeze-sha256', report.sha(root/'freeze.json'),
                       '--terminal-sha256', report.sha(root/'terminal.json'), '--output', str(root/'summary.json')]
            child = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertEqual(child.returncode, 0, child.stderr)
            saved = read(root/'summary.json')
            self.assertEqual(saved['attempt_status'], 'complete')
            self.assertEqual(saved['lineage']['reporter_sha256'], report.sha(report.__file__))
            self.assertEqual(saved['invalid_numeric_fields'], [])
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertNotEqual(rerun.returncode, 0)  # Exclusive output, no overwrite.

    def test_choice_strict_admission_ties_and_finite_matrix(self):
        intervals = [{'lower': -1., 'upper': 1.}]*2
        for method in report.SELECTORS:
            report.validate_choice(choice(method), method, intervals, [5, 8])
        for label in ('admission', 'tierank', 'candidate_order', 'gate_mode', 'pair_omission', 'nonfinite', 'interval'):
            with self.subTest(label=label):
                c = choice('raw' if label in ('tierank', 'pair_omission') else 'combined')
                method = 'raw' if c['mode'] == 'raw' else 'combined'
                if label == 'admission': c['local_errors'][1][2] = .5
                elif label == 'tierank': c['selected'] = ['grammar', 'retained']
                elif label == 'candidate_order': c['names'][1:3] = ['rbf', 'grammar']
                elif label == 'gate_mode': c['mode'] = 'local'
                elif label == 'pair_omission': c['composed_errors'].pop()
                elif label == 'nonfinite': c['local_errors'][0][2] = float('nan')
                else: c['intervals'][0] = {'lower': -2., 'upper': 1.}
                with self.assertRaises(ValueError): report.validate_choice(c, method, intervals, [5, 8])
        c = choice('raw')
        for p in c['composed_errors']: p[1] = 5.
        for pair in (['retained', 'grammar'], ['grammar', 'retained']):
            next(p for p in c['composed_errors'] if p[0] == pair)[1] = 4.
        c['selected'] = ['grammar', 'retained']; c['composed_mse'] = 4.
        with self.assertRaises(ValueError): report.validate_choice(c, 'raw', intervals, [5, 8])

    def test_direct_ratios_are_paired_before_aggregation_and_failures_retained(self):
        seeds = (1, 2)  # Fabricated ledger identities, not world generation.
        cells = [{'seed': s, 'variant': v, 'method': m, 'status': 'complete',
                  'metrics': {e: {'mse': 1., 'nmse': 1., 'training_variance': 1.} for e in report.ENDPOINTS}}
                 for s, v, m in itertools.product(seeds, report.VARIANTS, report.METHODS)]
        for row in cells:
            if row['method'] in ('pfn24', 'rbf24'):
                error = {('pfn24', 1): 2., ('pfn24', 2): 6., ('rbf24', 1): 1., ('rbf24', 2): 12.}[row['method'], row['seed']]
                for values in row['metrics'].values(): values.update(mse=error, nmse=error)
        output = report.summarize(cells, seeds)
        direct = next(c for c in output['direct_comparisons'] if c['variant'] == 'null' and c['method'] == 'pfn24' and c['endpoint'] == 'Y_composed')
        self.assertEqual([p['ratio'] for p in direct['pairs']], [2., .5])
        self.assertEqual(direct['arithmetic_mean'], 1.25)
        self.assertAlmostEqual(direct['geometric_mean'], 1.)
        self.assertNotEqual(direct['arithmetic_mean'], 4/6.5)
        cells[2]['metrics']['Y_composed']['nmse'] = 0.
        cells[len(report.VARIANTS)*len(report.METHODS)+2]['status'] = 'failed'
        changed = report.summarize(cells, seeds)
        direct = next(c for c in changed['direct_comparisons'] if c['variant'] == 'null' and c['method'] == 'pfn24' and c['endpoint'] == 'Y_composed')
        self.assertFalse(direct['defined']); self.assertEqual(len(direct['pairs']), 2)
        with self.assertRaises(ValueError): report.summarize(cells[:-1], seeds)

    def test_predicted_parent_rate_empty_partition_and_pin_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            row = terminal['cells'][0]
            self.assertIsNone(row['diagnostics']['local']['M_local']['outside']['mse'])
            self.assertEqual(row['diagnostics']['composed_Y_parent_inside_rate'], 0.)
            local_rate = row['diagnostics']['local']['Y_local']['inside']['count']/256
            self.assertGreater(local_rate, 0.)
            row['diagnostics']['composed_Y_parent_inside_rate'] = local_rate
            write(root/'223456/null/grammar32.json', row)
            completion = read(root/'complete.json'); completion['cells'] = terminal['cells']; write(root/'complete.json', completion)
            with self.assertRaisesRegex(ValueError, 'composed_Y_parent_inside_rate'): report.verify(root, freeze, terminal)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            np.save(root/'223456/null/grammar32_parents.npy', np.zeros((256, 1)))
            with self.assertRaisesRegex(ValueError, 'array pin mismatch'): report.verify(root, freeze, terminal)

    def test_partial_journals_unknown_returns_raw_corruption_and_pre_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            (root/'complete.json').unlink()
            for variant in ('missing_M', 'missing_Y'): shutil.rmtree(root/'223456'/variant)
            (root/'223456/coefficient_M/evaluation.returned.json').unlink()
            for row in terminal['cells']:
                if row['variant'] != 'null': row['status'] = 'interrupted'
            terminal.update(status='failed', reason='wall_timeout', error='fabricated timeout')
            result = report.verify(root, freeze, terminal)
            self.assertEqual(len(result['cells']), 36)
            private = result['response_accounting']['private']
            self.assertEqual((private['validated_returned'], private['reserved_unknown_returns'], private['unreserved_planned']), (768, 768, 1536))
            self.assertEqual(sum(r['status'] == 'complete' for r in result['cells']), 9)
            (root/'223456/prehistory.returned.json').unlink()
            terminal['cells'][0] = None
            result = report.verify(root, freeze, terminal)
            self.assertEqual(len(result['cells']), 36)
            self.assertIsNone(result['raw_terminal_cells'][0])
            self.assertEqual(result['cells'][3]['status'], 'invalid_record')
            self.assertTrue(all(h['mse_difference'] is None for h in result['local_harm'] if h['variant'] == 'null'))
            self.assertFalse(result['retry_authorized'])

    def test_malformed_response_arrays_revoke_validated_returns(self):
        for category in ('prehistory', 'training', 'private'):
            for defect in ('shape', 'nonfinite'):
                with self.subTest(category=category, defect=defect), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); freeze, terminal = build(root)
                    terminal.update(status='failed', reason='wall_timeout')
                    d = root/'223456' if category == 'prehistory' else root/'223456/null'
                    leaf = 'prehistory.npz' if category == 'prehistory' else ('training.npz' if category == 'training' else 'private_probes.npz')
                    with np.load(d/leaf) as archive: data = {k: archive[k].copy() for k in archive.files}
                    key = 'composed' if category == 'private' else 'train'
                    if defect == 'shape': data[key] = data[key][:-1]
                    else: data[key][0, 0] = float('nan')
                    np.savez(d/leaf, **data)
                    stem = 'evaluation' if category == 'private' else category
                    returned = read(d/(stem+'.returned.json')); returned['sha256'] = report.sha(d/leaf); write(d/(stem+'.returned.json'), returned)
                    if category == 'training':
                        seal = read(d/'selection_seal.json'); seal['training_sha256'] = report.sha(d/leaf); write(d/'selection_seal.json', seal); bind_seal(d)
                    result = report.verify(root, freeze, terminal)
                    accounting = result['response_accounting']['private' if category == 'private' else 'training']
                    planned, bad = (3072, 768) if category == 'private' else (160, 32)
                    self.assertEqual(accounting['validated_returned'], planned-bad)
                    self.assertEqual(accounting['reserved_unknown_returns'], bad)
                    self.assertTrue(any(b['path'].endswith(leaf) for b in accounting['invalid_blocks']))

    def test_internal_grammar_failure_and_independent_no_pfn_arm(self):
        for expert in ('grammar24', 'pfn24'):
            with self.subTest(expert=expert), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); freeze, terminal = build(root)
                failed = list(report.SELECTORS if expert == 'grammar24' else report.SELECTORS[:-1])
                if expert == 'pfn24': failed.append('pfn24')
                for variant in report.VARIANTS:
                    d = root/'223456'/variant; seal = read(d/'selection_seal.json')
                    seal['errors'] = {m: 'fabricated expert failure' for m in [expert]+failed}
                    for m in failed: seal['choices'].pop(m, None)
                    write(d/'selection_seal.json', seal); bind_seal(d)
                for row in terminal['cells']:
                    if row['method'] in failed:
                        row['status'] = 'failed'; row['error'] = 'fabricated dependent failure'
                        write(root/'223456'/row['variant']/(row['method']+'.json'), row)
                completion = read(root/'complete.json'); completion.update(cells=terminal['cells'], completed_cells=sum(r['status'] == 'complete' for r in terminal['cells'])); write(root/'complete.json', completion)
                result = report.verify(root, freeze, terminal)
                if expert == 'pfn24':
                    self.assertTrue(all(r['status'] == 'complete' for r in result['cells'] if r['method'] == 'combined_no_pfn'))
                    self.assertTrue(all(not c['defined'] for c in result['direct_comparisons'] if c['method'] == 'pfn24'))

    def test_required_expert_failure_cannot_keep_selector_choice(self):
        train, clamp, _, _, _ = tiny_arrays()
        for expert in ('prechange24', 'grammar24', 'rbf24', 'pfn24'):
            with self.subTest(expert=expert):
                seal = {'fit_indices': list(report.FIT_INDICES),
                        'calibration_indices': list(report.CALIBRATION_INDICES),
                        'choices': {m: choice(m) for m in report.SELECTORS},
                        'errors': {expert: 'fabricated required expert failure'}}
                valid, invalid = report.validate_choices(seal, train, clamp)
                affected = set(report.SELECTORS) - ({'combined_no_pfn'} if expert == 'pfn24' else set())
                self.assertEqual(set(invalid), affected)
                self.assertTrue(all('required expert failure' in value for value in invalid.values()))
                self.assertEqual(set(valid), {'combined_no_pfn'} if expert == 'pfn24' else set())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            d = root/'223456/null'; seal = read(d/'selection_seal.json')
            seal['errors']['grammar24'] = 'fabricated required expert failure'
            write(d/'selection_seal.json', seal); bind_seal(d)
            with self.assertRaisesRegex(ValueError, 'required expert failure'):
                report.verify(root, freeze, terminal)
            terminal.update(status='failed', reason='wall_timeout')
            result = report.verify(root, freeze, terminal)
            selectors = [r for r in result['cells'] if r['variant'] == 'null' and r['method'] in report.SELECTORS]
            self.assertEqual(len(selectors), 5)
            self.assertTrue(all(r['status'] == 'invalid_record' for r in selectors))

    def test_reason_schema_source_pins_and_safe_invalid_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze, terminal = build(root)
            terminal['reason'] = 'wall_timeout'
            with self.assertRaisesRegex(ValueError, 'supervision reason'): report.verify(root, freeze, terminal)
            terminal['reason'] = 'exited'
            bad = copy.deepcopy(freeze); bad['schema'] = 'ace-mismatch-freeze-v1'
            with self.assertRaisesRegex(ValueError, 'schema'): report.verify(root, bad, terminal)
            bad = copy.deepcopy(freeze); bad['sources'][report.REPORTER_KEY] = '0'*64
            with self.assertRaisesRegex(ValueError, 'source pin'): report.verify(root, bad, terminal)
            terminal.update(status='failed', reason='cancelled')
            terminal['cells'][0]['status'] = 'invalid_record'
            terminal['cells'][0]['metrics']['M_local']['nmse'] = float('nan')
            output = report.verify(root, freeze, terminal)
            invalid = []; safe = report.json_safe(output, invalid)
            self.assertTrue(invalid)
            self.assertTrue(any(v['original_representation'] == 'nan' for v in invalid))
            json.dumps(safe, allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
