#!/usr/bin/env python3
"""Saved-artifact retention reporting; no models, fitting or world generation.

Derived from summarize_foundation_mismatch.py without modifying that source.
Choice losses are validated as metadata: no saved head/model is executed.
Private errors, support partitions and composed-parent rates are reconstructed.
"""
import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

VARIANTS = ('null', 'coefficient_M', 'missing_M', 'missing_Y')
METHODS = ('grammar32', 'rbf24', 'pfn24', 'prechange24', 'raw', 'local',
           'interval', 'combined', 'combined_no_pfn')
ENDPOINTS = ('M_local', 'Y_local', 'Y_composed')
SELECTORS = METHODS[4:]
ORDER = ('retained', 'grammar', 'rbf', 'pfn')
DIRECT = (('pfn24', 'rbf24'), ('combined', 'raw'), ('combined', 'local'),
          ('combined', 'interval'), ('combined', 'combined_no_pfn'),
          ('combined', 'rbf24'))
FIT_INDICES = tuple(range(6)) + tuple(range(8, 17)) + tuple(range(20, 29))
CALIBRATION_INDICES = (6, 7, 17, 18, 19, 29, 30, 31)
LAYOUT = ('none',)*8 + ('X',)*12 + ('M',)*12
REPORTER_KEY = 'scripts/research/summarize_foundation_retention.py'
RESOURCE_KEYS = ('child_cpu_s', 'supervisor_process_cpu_s', 'elapsed_s',
                 'peak_child_rss_bytes', 'gpu_seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric value required')
    if nonnegative and value < 0:
        raise ValueError('negative loss/count/variance')
    return value


def integer(value, lower=0, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError('bounded integer required')
    return value


class Artifacts:
    """Hash, parse and decode the same captured bytes, including partial records."""
    def __init__(self):
        self.cache = {}

    def raw(self, path):
        path = Path(path)
        if path not in self.cache:
            self.cache[path] = path.read_bytes()
        return self.cache[path]

    def digest(self, path):
        return hashlib.sha256(self.raw(path)).hexdigest()

    def json(self, path, pin=None):
        raw = self.raw(path)
        if pin is not None and hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('pin mismatch: '+str(path))
        return json.loads(raw)

    def array(self, path, pin):
        import numpy as np
        raw = self.raw(path)
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('array pin mismatch: '+str(path))
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def archive(self, path, pin, names):
        with self.array(path, pin) as archive:
            if set(archive.files) != set(names):
                raise ValueError('archive membership: '+str(path))
            return tuple(archive[name].copy() for name in names)


def seeds_for(mode):
    if mode not in ('pilot', 'fixture'):
        raise ValueError('undeclared mode')
    return tuple(range(93000, 93006)) if mode == 'pilot' else (223456,)


def validate_training(train, clamp):
    import numpy as np
    if train.shape != (32, 3) or not np.isfinite(train).all() or tuple(clamp) != LAYOUT:
        raise ValueError('training shape/finiteness/intervention layout')


def fit_metadata(train, clamp):
    import numpy as np
    validate_training(train, clamp)
    fit = train[list(FIT_INDICES)]
    fit_clamp = clamp[list(FIT_INDICES)]
    eligible = fit[fit_clamp != 'M']
    intervals = [{'lower': float(a.min()), 'upper': float(a.max())}
                 for a in (eligible[:, 0], fit[:, 1])]
    cal = train[list(CALIBRATION_INDICES)]
    coords = (cal[:5, 0], cal[:, 1])
    inside = [int(((a >= i['lower']) & (a <= i['upper'])).sum())
              for a, i in zip(coords, intervals)]
    return intervals, inside


def validate_choice(choice, method, intervals, inside):
    """Validate the closed candidate matrix and inequalities, not model outputs."""
    fields = {'mode', 'names', 'intervals', 'local_errors', 'admitted',
              'composed_errors', 'selected', 'composed_mse', 'calibration_inside'}
    if not isinstance(choice, dict) or set(choice) != fields:
        raise ValueError('selection fields')
    mode = 'combined' if method == 'combined_no_pfn' else method
    names = list(ORDER[:-1] if method == 'combined_no_pfn' else ORDER)
    if choice['mode'] != mode or choice['names'] != names:
        raise ValueError('selection mode/candidate order')
    if choice['intervals'] != intervals or choice['calibration_inside'] != inside:
        raise ValueError('fit-only interval/calibration inside metadata')
    for i in choice['intervals']:
        number(i['lower']); number(i['upper'])
    for count, limit in zip(choice['calibration_inside'], (5, 8)):
        integer(count, upper=limit)
    local = choice['local_errors']
    expected = list(itertools.product(('M', 'Y'), names))
    if not isinstance(local, list) or len(local) != len(expected):
        raise ValueError('full local score matrix required')
    scores = {}
    for record, key in zip(local, expected):
        if not isinstance(record, list) or len(record) != 3 or record[:2] != list(key):
            raise ValueError('local score order/membership')
        scores[key] = number(record[2], True)
    admitted = [[n for n in names if n == 'retained' or mode in ('raw', 'interval')
                 or scores[node, n] < scores[node, 'retained']] for node in ('M', 'Y')]
    if choice['admitted'] != admitted:
        raise ValueError('strict local admission/retained feasibility')
    pairs = choice['composed_errors']
    expected_pairs = list(itertools.product(*admitted))
    if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
        raise ValueError('full admitted pair matrix required')
    for record, key in zip(pairs, expected_pairs):
        if not isinstance(record, list) or len(record) != 2 or record[0] != list(key):
            raise ValueError('composed score order/membership')
        number(record[1], True)
    def rank(record):
        a, b = record[0]
        return record[1], int(a != 'retained')+int(b != 'retained'), names.index(a), names.index(b)
    best = min(pairs, key=rank)
    if choice['selected'] != best[0] or number(choice['composed_mse'], True) != best[1]:
        raise ValueError('composed choice/tierank')
    retained_loss = next(p[1] for p in pairs if p[0] == ['retained', 'retained'])
    if best[1] > retained_loss:
        raise ValueError('retained composed constraint')
    if mode in ('local', 'combined'):
        for node, name in zip(('M', 'Y'), choice['selected']):
            if scores[node, name] > scores[node, 'retained']:
                raise ValueError('selected local constraint')


def validate_choices(seal, train, clamp):
    if seal['fit_indices'] != list(FIT_INDICES) or seal['calibration_indices'] != list(CALIBRATION_INDICES):
        raise ValueError('fit/calibration split')
    choices, errors = seal['choices'], seal['errors']
    allowed_errors = set(METHODS) | {'grammar24'}
    if not isinstance(choices, dict) or not isinstance(errors, dict):
        raise ValueError('choice/error dictionaries required')
    if set(choices) - set(SELECTORS) or set(errors) - allowed_errors:
        raise ValueError('undeclared choice/error key')
    if any(not isinstance(v, str) or not v for v in errors.values()):
        raise ValueError('explicit nonempty failure reason required')
    intervals, inside = fit_metadata(train, clamp)
    valid, invalid = {}, {}
    for method in SELECTORS:
        try:
            if method in errors:
                if method in choices:
                    raise ValueError('choice and failure simultaneously present')
                continue
            required = {'prechange24', 'grammar24', 'rbf24'}
            if method != 'combined_no_pfn':
                required.add('pfn24')
            failed_experts = sorted(required.intersection(errors))
            if failed_experts:
                raise ValueError('required expert failure without explicit selector failure: '
                                 + ', '.join(failed_experts))
            if method not in choices:
                raise ValueError('missing selection or explicit failure')
            validate_choice(choices[method], method, intervals, inside)
            valid[method] = choices[method]
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            invalid[method] = str(exc)
    return valid, invalid


def mean_square(pred, target):
    if len(pred) != len(target) or not len(pred):
        raise ValueError('nonempty paired error vectors required')
    losses = [(float(a)-float(b))**2 for a, b in zip(pred, target)]
    for value in losses:
        number(value, True)
    return number(math.fsum(v/len(losses) for v in losses), True)


def recompute_diagnostics(pred, parents, probes, train, clamp, method):
    """The exact worker row['diagnostics'] schema; empty partitions have null MSE."""
    intervals, _ = fit_metadata(train, clamp)
    _, lm, ly = probes
    local = {}
    for j, (key, coords, targets) in enumerate((('M_local', lm[:, 0], lm[:, 1]),
                                               ('Y_local', ly[:, 1], ly[:, 2]))):
        interval = intervals[j]
        mask = (coords >= interval['lower']) & (coords <= interval['upper'])
        local[key] = {}
        for label, part in (('inside', mask), ('outside', ~mask)):
            count = int(part.sum())
            local[key][label] = {'count': count, 'mse': mean_square(pred[part, j], targets[part]) if count else None}
    interval = intervals[1]
    count = int(((parents[:, 0] >= interval['lower']) & (parents[:, 0] <= interval['upper'])).sum())
    return {'intervals': intervals, 'local': local, 'composed_Y_parent_inside': count,
            'composed_Y_parent_outside': 256-count, 'composed_Y_parent_inside_rate': count/256,
            'interval_gating_enabled': method in ('interval', 'combined', 'combined_no_pfn')}


def compare_structure(actual, expected, path='diagnostics'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(path+' fields')
        for k, value in expected.items():
            compare_structure(actual[k], value, path+'.'+k)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(path+' length')
        for i, value in enumerate(expected):
            compare_structure(actual[i], value, path+'['+str(i)+']')
    elif type(expected) is bool or expected is None:
        if actual is not expected:
            raise ValueError(path+' value')
    elif type(expected) is int:
        if integer(actual) != expected:
            raise ValueError(path+' count')
    else:
        number(actual)
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(path+' numeric mismatch')


def metric(row, endpoint, key):
    metrics = row.get('metrics')
    item = metrics.get(endpoint) if isinstance(metrics, dict) else None
    value = item.get(key) if isinstance(item, dict) else None
    return value if type(value) in (int, float) and math.isfinite(value) else None


def aggregate(values):
    if not values or any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        return {'arithmetic_mean': None, 'geometric_mean': None, 'defined': False}
    try:
        ar = math.fsum(v/len(values) for v in values)
        ge = math.exp(math.fsum(math.log(v)/len(values) for v in values))
        if not math.isfinite(ar) or not math.isfinite(ge):
            raise ValueError('nonfinite aggregate')
        return {'arithmetic_mean': ar, 'geometric_mean': ge, 'defined': True}
    except (ValueError, OverflowError):
        return {'arithmetic_mean': None, 'geometric_mean': None, 'defined': False}


def summarize(cells, seeds):
    expected = list(itertools.product(seeds, VARIANTS, METHODS))
    if [(r['seed'], r['variant'], r['method']) for r in cells] != expected:
        raise ValueError('full ordered cell matrix required')
    mapping = dict(zip(expected, cells))
    def comparisons(contrasts):
        result = []
        for variant, (candidate, reference), endpoint in itertools.product(VARIANTS, contrasts, ENDPOINTS):
            pairs = []
            for seed in seeds:
                a, b = mapping[seed, variant, candidate], mapping[seed, variant, reference]
                value, why = None, None
                if a['status'] != 'complete' or b['status'] != 'complete':
                    why = 'missing_failed_or_invalid_cell'
                else:
                    x, y = metric(a, endpoint, 'nmse'), metric(b, endpoint, 'nmse')
                    if x is None or y is None:
                        why = 'missing_nonnumeric_or_nonfinite_error'
                    elif x <= 0 or y <= 0:
                        why = 'zero_or_negative_error'
                    else:
                        value = x/y
                        if not math.isfinite(value) or value <= 0:
                            value, why = None, 'nonfinite_or_underflowed_ratio'
                pairs.append({'seed': seed, 'ratio': value, 'undefined_reason': why,
                              'candidate_status': a['status'], 'reference_status': b['status']})
            result.append({'variant': variant, 'method': candidate, 'reference': reference,
                           'endpoint': endpoint, 'direction': 'candidate_NMSE / reference_NMSE',
                           'planned_worlds': len(seeds), 'defined_worlds': sum(p['ratio'] is not None for p in pairs),
                           'pairs': pairs, **aggregate([p['ratio'] for p in pairs])})
        return result
    harm = []
    for seed, variant, method, endpoint in itertools.product(seeds, VARIANTS,
            tuple(m for m in METHODS if m != 'prechange24'), ENDPOINTS[:2]):
        a, b = mapping[seed, variant, method], mapping[seed, variant, 'prechange24']
        delta, norm = None, None
        if a['status'] == b['status'] == 'complete':
            x, y = metric(a, endpoint, 'mse'), metric(b, endpoint, 'mse')
            v, w = metric(a, endpoint, 'training_variance'), metric(b, endpoint, 'training_variance')
            if x is not None and y is not None and v is not None and v >= 0 and v == w:
                delta, norm = x-y, (x-y)/max(v, 1e-12)
                if not math.isfinite(delta) or not math.isfinite(norm):
                    delta = norm = None
            elif v is not None and w is not None and v != w:
                raise ValueError('different normalization phases')
        unchanged = variant == 'null' or (endpoint == 'Y_local' and variant in ('coefficient_M', 'missing_M')) or (endpoint == 'M_local' and variant == 'missing_Y')
        harm.append({'seed': seed, 'variant': variant, 'method': method, 'endpoint': endpoint,
                     'mechanism_unchanged': unchanged, 'mse_difference': delta, 'normalized_difference': norm})
    return {'cells': cells, 'comparisons': comparisons(tuple((m, 'grammar32') for m in METHODS[1:])),
            'direct_comparisons': comparisons(DIRECT), 'local_harm': harm,
            'interpretation': 'development pipelines; positive local difference means harm; no population safety, equivalence, pretraining attribution or acquisition-efficiency inference'}


def verify(root, freeze, terminal):
    """Strict successful closure; authenticated failed attempts retain all slots."""
    import numpy as np
    root = Path(root)
    if freeze.get('schema') != 'ace-retention-freeze-v1':
        raise ValueError('freeze schema')
    seeds = seeds_for(freeze['mode'])
    if freeze.get('sources', {}).get(REPORTER_KEY) != sha(__file__):
        raise ValueError('reporter source pin')
    if terminal.get('mode') != freeze['mode'] or terminal.get('status') not in ('complete', 'failed'):
        raise ValueError('terminal mode/status')
    strict = terminal['status'] == 'complete'
    reader, issues = Artifacts(), []
    expected = list(itertools.product(seeds, VARIANTS, METHODS))
    def problem(kind, error, **context):
        if strict:
            raise ValueError(kind+': '+str(error))
        issues.append({'kind': kind, 'reason': str(error), **context})
    raw = terminal.get('cells')
    if not isinstance(raw, list):
        problem('invalid_terminal_cells', 'list required'); records = []
    else:
        records = raw
    mapping, duplicates = {}, set()
    for row in records:
        if not isinstance(row, dict) or type(row.get('seed')) is not int or type(row.get('variant')) is not str or type(row.get('method')) is not str:
            problem('invalid_terminal_record', 'typed object identity required', record=row); continue
        key = row['seed'], row['variant'], row['method']
        if key not in expected or key in mapping:
            if key in mapping:
                duplicates.add(key)
            problem('invalid_terminal_identity', 'unexpected or duplicate identity', record=row); continue
        mapping[key] = row
    if strict and list(mapping) != expected:
        raise ValueError('full ordered terminal matrix required')
    if strict:
        plan = reader.json(root/'plan.json')
        if plan['mode'] != freeze['mode'] or [(r['seed'], r['variant'], r['method']) for r in plan['cells']] != expected:
            raise ValueError('plan matrix')
        completion = reader.json(root/'complete.json')
        if completion['mode'] != freeze['mode'] or completion['cells'] != raw or completion['planned_cells'] != len(expected) or terminal['planned_cells'] != len(expected):
            raise ValueError('completion/terminal closure')
        if completion['completed_cells'] != sum(r['status'] == 'complete' for r in records):
            raise ValueError('completed count')
        if any(r['status'] not in ('complete', 'failed') for r in records):
            raise ValueError('successful terminal has unfinished cells')
        if terminal.get('reason') != 'exited' or terminal.get('exit_code') != 0 or terminal.get('error') is not None:
            raise ValueError('successful supervision reason/exit/error')
    accounting = {k: {'planned': len(seeds)*(160 if k == 'training' else 3072),
                     'reserved': 0, 'validated_returned': 0, 'reserved_unknown_returns': 0,
                     'unreserved_planned': 0, 'invalid_reservation_planned': 0, 'invalid_blocks': []}
                  for k in ('training', 'private')}
    def block(directory, stem, category, count, filename, seal_check=None):
        dst = accounting[category]
        rp, qp = directory/(stem+'.reserved.json'), directory/(stem+'.returned.json')
        if not rp.exists():
            dst['unreserved_planned'] += count
            if qp.exists():
                dst['invalid_blocks'].append({'path': str(qp), 'count': count, 'reason': 'return without reservation'})
            if strict:
                raise ValueError('missing reservation: '+str(rp))
            return None
        try:
            r = reader.json(rp)
            if integer(r['responses']) != count or r['kind'] != category:
                raise ValueError('reservation count/kind')
            number(r['at_unix'])
        except Exception as exc:
            dst['invalid_reservation_planned'] += count
            dst['invalid_blocks'].append({'path': str(rp), 'count': count, 'reason': str(exc)})
            problem('invalid_reservation', exc, path=str(rp)); return None
        dst['reserved'] += count
        try:
            q = reader.json(qp)
            if integer(q['private_responses' if category == 'private' else 'responses']) != count or number(q['at_unix']) < r['at_unix'] or q['sha256'] != reader.digest(directory/filename):
                raise ValueError('return count/time/hash')
            if seal_check:
                seal_check(r, q)
            dst['validated_returned'] += count
            return q
        except Exception as exc:
            dst['reserved_unknown_returns'] += count
            if qp.exists():
                dst['invalid_blocks'].append({'path': str(qp), 'count': count, 'reason': str(exc)})
            problem('unvalidated_return', exc, path=str(qp)); return None
    def revoke_return(directory, filename, category, count, error):
        """A matching byte pin does not qualify malformed response-array contents."""
        dst = accounting[category]
        dst['validated_returned'] -= count
        dst['reserved_unknown_returns'] += count
        dst['invalid_blocks'].append({'path': str(directory/filename), 'count': count,
                                      'reason': str(error)})
    cells, selections = [], {}
    for seed in seeds:
        parent = root/str(seed)
        pre = block(parent, 'prehistory', 'training', 32, 'prehistory.npz')
        if pre is not None:
            try:
                validate_training(*reader.archive(parent/'prehistory.npz', pre['sha256'], ('train', 'clamp')))
            except Exception as exc:
                revoke_return(parent, 'prehistory.npz', 'training', 32, exc)
                problem('invalid_prehistory_layout', exc, seed=seed); pre = None
        for variant in VARIANTS:
            d = parent/variant
            td = block(d, 'training', 'training', 32, 'training.npz')
            train = clamp = seal = None
            valid_choices, bad_choices = {}, {}
            if td is not None:
                try:
                    train, clamp = reader.archive(d/'training.npz', td['sha256'], ('train', 'clamp'))
                    validate_training(train, clamp)
                except Exception as exc:
                    revoke_return(d, 'training.npz', 'training', 32, exc)
                    problem('invalid_training_layout', exc, seed=seed, variant=variant); td = None
            if td is not None:
                try:
                    seal = reader.json(d/'selection_seal.json')
                    if seal['training_sha256'] != td['sha256'] or number(seal['at_unix']) < td['at_unix'] or seal['evaluation_generated'] is not False:
                        raise ValueError('selection training binding/order')
                    valid_choices, bad_choices = validate_choices(seal, train, clamp)
                    for method, error in bad_choices.items():
                        problem('invalid_choice', error, seed=seed, variant=variant, method=method)
                    selections[str(seed)+':'+variant] = valid_choices
                except Exception as exc:
                    problem('invalid_selection_or_training', exc, seed=seed, variant=variant); seal = None
            def check_evaluation(r, q):
                if seal is None or td is None:
                    raise ValueError('evaluation without qualified training/selection seal')
                pin = reader.digest(d/'selection_seal.json')
                if r['selection_seal_sha256'] != pin or q['selection_seal_sha256'] != pin or r['at_unix'] < seal['at_unix']:
                    raise ValueError('evaluation selection binding/order')
            ed = block(d, 'evaluation', 'private', 768, 'private_probes.npz', check_evaluation)
            probes = None
            if ed is not None:
                try:
                    probes = reader.archive(d/'private_probes.npz', ed['sha256'], ('composed', 'local_m', 'local_y'))
                    if any(a.shape != (256, 3) or not np.isfinite(a).all() for a in probes):
                        raise ValueError('private probe shape/finiteness')
                    if any((a[:, 0] < -1).any() or (a[:, 0] > 1).any() for a in probes) or (probes[2][:, 1] < -2).any() or (probes[2][:, 1] > 2).any():
                        raise ValueError('private intervention domain')
                except Exception as exc:
                    revoke_return(d, 'private_probes.npz', 'private', 768, exc)
                    problem('invalid_private_probes', exc, seed=seed, variant=variant); probes = None
            for method in METHODS:
                key = seed, variant, method
                original = mapping.get(key)
                row = dict(original) if original is not None else {'seed': seed, 'variant': variant, 'method': method, 'status': 'invalid_record', 'error': 'missing terminal identity'}
                if key in duplicates or row.get('status') not in ('complete', 'failed', 'unattempted', 'interrupted', 'invalid_record'):
                    row.update(status='invalid_record', verification_error='duplicate identity or invalid disposition')
                try:
                    if row['status'] in ('complete', 'failed') and reader.json(d/(method+'.json')) != original:
                        raise ValueError('cell closure mismatch')
                    if row['status'] == 'complete':
                        if td is None or seal is None or probes is None:
                            raise ValueError('unqualified training/selection/private response closure')
                        if method in seal['errors']:
                            raise ValueError('completed method recorded as failed in seal')
                        if (method == 'prechange24' or method in SELECTORS) and pre is None:
                            raise ValueError('retention reference lacks prehistory closure')
                        if method in SELECTORS and method not in valid_choices:
                            raise ValueError('missing or invalid selector choice')
                        pred = reader.array(d/(method+'_predictions.npy'), row['prediction_sha256'])
                        parents = reader.array(d/(method+'_parents.npy'), row['parent_sha256'])
                        if pred.shape != (256, 3) or parents.shape != (256, 1) or not np.isfinite(pred).all() or not np.isfinite(parents).all():
                            raise ValueError('prediction/parent layout or finiteness')
                        targets = (probes[1][:, 1], probes[2][:, 2], probes[0][:, 2])
                        variances = [float(np.var(a)) for a in (train[clamp != 'M', 1], train[:, 2], train[:, 2])]
                        for j, endpoint in enumerate(ENDPOINTS):
                            v, m = number(variances[j], True), row['metrics'][endpoint]
                            loss = mean_square(pred[:, j], targets[j])
                            if number(m['training_variance'], True) != v or type(m['floor_active']) is not bool or m['floor_active'] != (v < 1e-12):
                                raise ValueError('normalizer/floor mismatch')
                            for k, expected_value in (('mse', loss), ('nmse', loss/max(v, 1e-12))):
                                if not math.isclose(number(m[k], True), expected_value, rel_tol=1e-10, abs_tol=1e-12):
                                    raise ValueError('metric mismatch: '+endpoint+'.'+k)
                        compare_structure(row['diagnostics'], recompute_diagnostics(pred, parents, probes, train, clamp, method))
                        if row['diagnostics']['intervals'] != fit_metadata(train, clamp)[0]:
                            raise ValueError('diagnostic interval mismatch')
                except Exception as exc:
                    problem('invalid_cell_artifacts', exc, seed=seed, variant=variant, method=method)
                    row.update(status='invalid_record', verification_error=str(exc))
                cells.append(row)
    if strict:
        if completion['training_responses_total'] != len(seeds)*160 or completion['private_responses_total'] != len(seeds)*3072:
            raise ValueError('completion response totals')
        if any(a['validated_returned'] != a['planned'] for a in accounting.values()):
            raise ValueError('incomplete successful response closure')
    summary = summarize(cells, seeds)
    summary.update(attempt_status=terminal['status'], failure_reason=terminal.get('reason') if not strict else None,
                   retry_authorized=False, raw_terminal=terminal, raw_terminal_cells=raw,
                   verification_issues=issues, response_accounting=accounting, selection=selections,
                   supervised_resources={k: terminal.get(k) for k in RESOURCE_KEYS})
    return summary


def json_safe(value, invalid, path='$'):
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append({'path': path, 'original_representation': repr(value)})
        return None
    if isinstance(value, dict):
        return {k: json_safe(v, invalid, path+'.'+str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, invalid, path+'['+str(i)+']') for i, v in enumerate(value)]
    return value


def main():
    parser = argparse.ArgumentParser()
    for key in ('root', 'freeze', 'terminal', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('freeze-sha256', 'terminal-sha256'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    reader = Artifacts()
    freeze = reader.json(args.freeze, args.freeze_sha256)
    terminal = reader.json(args.terminal, args.terminal_sha256)
    if terminal.get('freeze_sha256') != args.freeze_sha256:
        raise ValueError('terminal/freeze binding')
    summary = verify(args.root, freeze, terminal)
    summary['lineage'] = {'freeze_sha256': args.freeze_sha256, 'terminal_sha256': args.terminal_sha256,
                          'reporter_sha256': sha(__file__)}
    invalid = []
    summary = json_safe(summary, invalid)
    summary['invalid_numeric_fields'] = invalid
    raw = (json.dumps(summary, indent=2, allow_nan=False)+'\n').encode()
    with args.output.open('xb') as output:
        output.write(raw)


if __name__ == '__main__':
    main()
