"""Supplemental read-only Stage B replay from explicitly derived relative artifacts.

Requires an independently pinned manifest, successful unchanged upstream audit,
and exact recorded runtime. This adapter is not a relocated historical worker,
historical freeze authentication, refitting, or permission to acquire responses.
All scores use the original cached predictions after checkpoint comparison.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.abc
import importlib.util
from io import BytesIO
import json
import math
import re
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from verify_delivery_release import relative, verify, sha

REGISTRATION = 'e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5'
REVISION = '45ebeb89d2c76daa97a55f07728239245e0c4f60'
DEPENDENCIES = {'torch': '2.9.1', 'numpy': '2.2.6', 'scipy': '1.15.3',
                'pandas': '2.3.3', 'sympy': '1.14.0', 'PyYAML': '6.0.3'}
CORE_SHA = '5c8caa608a5012571b538bf292faa8f39345a5adfd1241e18e790adcf2e00aa3'
CORE_CONTRACT_SHA = 'ab6ae92cfb36005d44d140faee4ceddd3f782a1581e814c5cbf67cd8f794258d'
AUDITOR_SHA = '400bc49f4832b37231e12f803d240040f6d8a12255d7579cbf38da3cf5b1335a'
KERNEL_SHA = '044c7856c5e838be9e0bcfbf3dc981a61239003ba24a0433b74fa8a5b43e997f'
WORLDS = [f'{size}:{i:02d}' for size in (5, 30) for i in range(20)]
HISTORIES = {'balanced_varied_value': 'balanced', 'matched_random': 'random'}
ARMS = [('delivery', 0), ('online', 0), ('simpler', 0), ('ablation', 0),
        ('delivery', 1), ('simpler', 1), ('delivery', 2), ('simpler', 2)]


def captured(path, expected=None):
    raw = Path(path).read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if expected is not None and h != expected:
        raise ValueError('artifact snapshot digest changed')
    return raw, h


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def matrix_membership(cells):
    expected = [(case, history, arm, init) for case in WORLDS
                for history in HISTORIES for arm, init in ARMS]
    if len(cells) != 640:
        raise ValueError('full 640-cell matrix required')
    for i, (c, key) in enumerate(zip(cells, expected)):
        if (type(c['index']) is not int or c['index'] != i or
                (c['case'], c['history'], c['arm'], c['init']) != key or
                c['graph_size'] != int(c['case'].split(':')[0]) or
                c['folder'] != f"B/fits/{c['case'].replace(':', '-')}/{c['history']}/{c['arm']}-i{c['init']}"):
            raise ValueError('matrix ordering or cell identity changed')


def gate(contract):
    """Derived upstream projection gate; no score/array parsing here.

    Independent manifest authentication binds the derivation, not an original
    signature. The private planner must have revalidated original audit bytes.
    """
    if (contract['schema'] != 'delivery-prospective-replay-v1' or
            contract['registration_sha256'] != REGISTRATION or
            contract['original_revision'] != REVISION or
            contract['dependencies'] != DEPENDENCIES or
            contract['core_sha256'] != CORE_SHA or contract['core_contract_sha256'] != CORE_CONTRACT_SHA or
            contract['matrix_fits'] != 640 or contract['primary_cells'] != 240):
        raise ValueError('frozen prospective contract changed')
    g = contract['upstream_gate']
    projection = g['custody_bound_acceptance_projection']
    a = projection['metadata']
    s, c = contract['supervisor'], contract['complete']
    if (g['outcomes_opened'] is not False or g['registration_sha256'] != REGISTRATION or
            projection['original_acceptance_sha256'] != g['acceptance_sha256'] or
            projection['scientific_outcome_values_decoded'] is not False or
            a['full_acceptance'] is not True or a['source_revision'] != REVISION or
            a['runtime'] != DEPENDENCIES or a['study_registration_sha256'] != REGISTRATION or
            a['auditor_sha256'] != AUDITOR_SHA or
            a['scope'] != contract['scope'] or
            any(type(a[k]) is not int or a[k] != v for k, v in {
                'fits_checked': 640, 'checkpoints_replayed': 640, 'primary_cells_checked': 240,
                'charged_cached_responses': 48000, 'new_simulator_responses': 0}.items()) or
            s['status'] != 'complete' or type(s['exit_code']) is not int or s['exit_code'] != 0 or
            s['registration_sha256'] != REGISTRATION or s['acceptance_sha256'] != g['acceptance_sha256'] or
            type(s['peak_tree_rss_bytes']) is not int or not finite(s['peak_tree_rss_bytes']) or
            not finite(s['elapsed_seconds']) or s['peak_tree_rss_bytes'] > contract['resources']['rss_bytes'] or
            s['elapsed_seconds'] > contract['resources']['audit_wall_seconds'] or
            any(type(c[k]) is not int or c[k] != v for k, v in {
                'n_fits': 640, 'n_primary_cells': 240, 'charged_responses': 48000}.items()) or
            c['registration_sha256'] != REGISTRATION or c['scores_sha256'] != g['scores_sha256'] or
            a['scores_sha256'] != g['scores_sha256'] or
            c['fit_seal_sha256'] != contract['fit_seal_sha256'] or
            a['fit_seal_sha256'] != contract['fit_seal_sha256'] or
            a['complete_sha256'] != contract['original_complete_sha256']):
        raise ValueError('complete successful original audit projection required')
    hashes = [g[k] for k in ('registration_sha256', 'acceptance_sha256', 'audit_execution_sha256', 'scores_sha256')]
    hashes.extend([a[k] for k in ('scores_sha256', 'complete_sha256', 'fit_seal_sha256', 'auditor_sha256')])
    worlds = set(WORLDS)
    histories = {case+'/'+h for case in WORLDS for h in HISTORIES}
    for key, names in [('training_receipt_hashes', histories), ('evaluation_receipt_hashes', worlds)]:
        if set(a[key]) != names or any(set(v) != {'input.json', 'queries.ndjson', 'receipt.json'} for v in a[key].values()):
            raise ValueError('complete response evidence required')
        hashes.extend(h for files in a[key].values() for h in files.values())
    if (set(a['replay_max_abs_deltas']) != {str(i) for i in range(640)} or
            not all(finite(v) for v in a['replay_max_abs_deltas'].values()) or
            a['replay_tolerance'] != {'rtol': 1e-6, 'atol': 1e-7, 'roots': 'exact'} or
            set(a['phase_execution_hashes']) != {'qualification_execution.json', 'collect_execution.json', 'evaluate_execution.json'}):
        raise ValueError('upstream replay/phase evidence incomplete')
    hashes.extend(a['phase_execution_hashes'].values())
    if any(not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None for h in hashes):
        raise ValueError('original digest syntax changed')
    matrix_membership(contract['cells'])
    if (set(contract['training']) != histories or set(contract['evaluation']) != worlds or
            set(contract['descriptors']) != worlds or set(contract['world_execution']) != worlds):
        raise ValueError('complete relative input/attempt membership required')


def load_code(root, name, expected, module_name):
    if module_name in sys.modules:
        raise ValueError('fresh process required for authenticated helpers')
    path = relative(root, name)
    raw, _ = captured(path, expected)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Execute precisely the bytes that were authenticated, rather than reopening.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def byte_integrity(root, manifest):
    """No JSON artifact decoding, semantic screening or binding parsing here."""
    if manifest['schema'] != 'delivery-anonymous-derived-v1':
        raise ValueError('unknown release manifest schema')
    names = [f['path'] for f in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('duplicate artifact paths')
    for f in manifest['files']:
        p = relative(root, f['path'])
        if not p.is_file() or p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
            raise ValueError('artifact bytes changed before replay barrier')
        if (not isinstance(f['original_sha256'], str) or re.fullmatch('[0-9a-f]{64}', f['original_sha256']) is None or
                f['transform']['kind'] not in ('identity', 'project-json') or
                f['transform']['kind'] == 'identity' and f['original_sha256'] != f['sha256']):
            raise ValueError('artifact derivation changed')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('unlisted symlink')
        if p.is_file(): actual.add(p.relative_to(root).as_posix())
    if actual != set(names) | {'manifest.json', 'manifest.sha256'}:
        raise ValueError('unlisted or missing release artifact')


def dependency_modules():
    """Check both distribution metadata and the modules actually imported."""
    mapping = {'PyYAML': 'yaml'}
    for distribution, version in DEPENDENCIES.items():
        name = mapping.get(distribution, distribution)
        dist = importlib.metadata.distribution(distribution)
        module = __import__(name)
        origin = Path(module.__file__).resolve()
        expected = Path(dist.locate_file(name+'/__init__.py')).resolve()
        if (dist.version != version or module.__version__.split('+', 1)[0] != version or origin != expected):
            raise ValueError('imported dependency version/origin differs: '+name)


class LearnerSnapshots(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load only the complete authenticated archived learner closure."""
    def __init__(self, root, learner_root, hashes):
        self.root, self.learner_root, self.hashes = root, learner_root, hashes
        self.contents = {}
        for name, digest in hashes.items():
            self.contents[name] = captured(relative(root, learner_root+'/'+name), digest)[0]

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'ace' and not fullname.startswith('ace.'): return None
        stem = fullname.replace('.', '/')
        name = stem+'/__init__.py' if stem+'/__init__.py' in self.contents else stem+'.py'
        if name not in self.contents:
            raise ValueError('unrecorded learner module import')
        spec = importlib.util.spec_from_loader(fullname, self, is_package=name.endswith('/__init__.py'))
        spec.loader_state = name
        return spec

    def create_module(self, spec): return None

    def exec_module(self, module):
        name = module.__spec__.loader_state
        module.__file__ = str(relative(self.root, self.learner_root+'/'+name))
        if name.endswith('/__init__.py'): module.__path__ = [str(Path(module.__file__).parent)]
        exec(compile(self.contents[name], module.__file__, 'exec'), module.__dict__)


def no_network(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
        raise PermissionError('offline replay prohibits network access')


def snapshot_io(root, entries):
    """Supplemental I/O adapters bind hash and parse to the same captured bytes.

    They retain contiguous charged-attempt semantics. These are explicit new
    adapters around the extracted functions, not edits to historical workers.
    """
    def bytes_at(path):
        p = Path(path)
        name = p.relative_to(root).as_posix()
        return captured(relative(root, name), entries[name]['sha256'])[0]
    def read_at(path):
        return json.loads(bytes_at(path))
    def sha_at(path):
        return hashlib.sha256(bytes_at(path)).hexdigest()
    def journal_at(path):
        values = [json.loads(line) for line in bytes_at(path).decode().splitlines()]
        if (any(type(v['attempt']) is not int for v in values) or
                [v['attempt'] for v in values] != list(range(1, len(values)+1))):
            raise ValueError('charged-attempt journal sequence changed')
        return len(values)
    return bytes_at, read_at, sha_at, journal_at


def metric_from_cached(evaluation, training, predictions, models, arm, torch):
    """Original endpoint plus observed-parent diagnostics, no simulator calls."""
    import numpy as np
    target = evaluation['target']
    var = float(np.var([row['node_values'][target] for row in training['rows']]))
    if not math.isfinite(var) or var <= 0:
        raise ValueError('invalid training-only target variance')
    truth = {n: np.array([row['node_values'][n] for row in evaluation['rows']]) for n in evaluation['order']}
    mse = float(np.mean(np.square(predictions[target]-truth[target])))
    metric = {'mse': mse, 'nmse': mse/var, 'training_target_variance': var, 'evaluation_rows': 400,
              'snapped_error': None, 'endpoint': 'noise-disabled deterministic target, continuous; no fabricated quantization'}
    if arm != 'simpler':
        diagnostic = {}
        with torch.no_grad():
            for n, model in models.items():
                x = torch.tensor(np.stack([truth[p] for p in evaluation['parents'][n]], axis=1), dtype=torch.float32)
                local = model(x).numpy()
                diagnostic[n] = {'observed_parent_mse': float(np.mean(np.square(local-truth[n]))),
                                 'free_running_mse': float(np.mean(np.square(predictions[n]-truth[n]))),
                                 'propagated_prediction_shift_mse': float(np.mean(np.square(predictions[n]-local)))}
        metric['secondary_mechanism_diagnostics'] = diagnostic
    return metric


def replay(root, expected):
    root = Path(root).resolve()
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('independently supplied SHA256 manifest pin required')
    # Pin manifest and the contract before any loss parsing, model imports or inference.
    raw, _ = captured(relative(root, 'manifest.json'), expected)
    manifest = json.loads(raw)
    entries = {f['path']: f for f in manifest['files']}
    contract = json.loads(captured(relative(root, 'B/replay_contract.json'),
                                   entries['B/replay_contract.json']['sha256'])[0])
    gate(contract)
    actual = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    if actual != DEPENDENCIES:
        raise ValueError('exact B runtime required before outcomes; local A/C tests are not qualification')
    byte_integrity(root, manifest)
    dependency_modules()
    sys.addaudithook(no_network)
    cpu0, wall0 = time.process_time(), time.monotonic()
    def read(name):
        return json.loads(captured(relative(root, name), entries[name]['sha256'])[0])
    cc = read(contract['core_contract_path'])
    if (entries[contract['core_contract_path']]['sha256'] != contract['core_contract_sha256'] or
            cc['derived_core_sha256'] != contract['core_sha256'] or
            cc['registration_sha256'] != REGISTRATION or cc['target_dependencies'] != DEPENDENCIES):
        raise ValueError('derived core provenance changed')
    core = load_code(root, contract['core_path'], contract['core_sha256'], 'delivery_prospective_replay_core')
    bytes_at, core.read, core.sha, core.journal_count = snapshot_io(root, entries)
    load_code(root, 'delivery_prospective_design.py', contract['helper_hashes']['delivery_prospective_design.py'],
              'delivery_prospective_design')
    if any(n == 'ace' or n.startswith('ace.') for n in sys.modules):
        raise ValueError('fresh process required for archived learner')
    for name, digest in contract['learner_hashes'].items():
        if entries[contract['learner_root']+'/'+name]['sha256'] != digest:
            raise ValueError('exact learner source binding changed')
    loader = LearnerSnapshots(root, contract['learner_root'], contract['learner_hashes'])
    sys.meta_path.insert(0, loader)
    import numpy as np
    import torch
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(1)
    training, descriptors = {}, {}
    a = contract['upstream_gate']['custody_bound_acceptance_projection']['metadata']
    for case in WORLDS:
        d = contract['descriptors'][case]
        spec, actions = read(d['world']), read(d['actions'])
        if spec['size'] != int(case.split(':')[0]):
            raise ValueError('descriptor graph stratum changed')
        descriptors[case] = spec, actions
        for history, strategy in HISTORIES.items():
            key = case+'/'+history; folder = contract['training'][key]
            for name, digest in a['training_receipt_hashes'][key].items():
                if entries[folder+'/'+name]['sha256'] != digest:
                    raise ValueError('original training response bytes changed')
            training[key] = core.history_receipt(relative(root, folder), spec, actions[strategy], False)
    fits, latest, cpu = {}, None, 0.
    for cell in contract['cells']:
        folder = cell['folder']; r = read(folder+'/receipt.json')
        data = training[cell['case']+'/'+cell['history']]
        heads = {'flat'} if cell['arm'] == 'simpler' else {n for n in data['order'] if data['parents'][n]}
        updates = 48000 if cell['arm'] == 'online' else 100 if cell['arm'] == 'ablation' else 30000
        expected_binding = {'protocol_sha256': REGISTRATION,
                            'input_sha256': a['training_receipt_hashes'][cell['case']+'/'+cell['history']]['input.json'],
                            'kernel_sha256': KERNEL_SHA, 'source_hashes': contract['learner_hashes'], 'dependencies': DEPENDENCIES}
        if (entries[folder+'/receipt.json']['sha256'] != cell['receipt_sha256'] or
                entries[folder+'/models.pt']['sha256'] != cell['model_sha256'] or
                r['complete'] is not True or r['arm'] != cell['arm'] or r['init'] != cell['init'] or
                cell['binding'] != expected_binding or r['binding'] != expected_binding or
                r['model_sha256'] != cell['model_sha256'] or
                set(r['heads']) != heads or r['unique_paid_rows'] != 400 or r['calibration_rows'] != 50 or
                r['new_queries_in_fit'] != 0 or r['evaluation_responses_read'] != 0 or
                r['development_epoch_override'] is not None or r['development_online_tail'] is not None or
                any(v['updates'] != updates or v['eligible_rows'] != 400 for v in r['heads'].values()) or
                not finite(r['cpu_seconds']) or r['cpu_seconds'] <= 0):
            raise ValueError('full fit configuration or custody changed')
        core.match(r['normalizers'], core.calibration_ranges(data), 'calibration')
        fits[cell['index']] = r
        finished = datetime.fromisoformat(r['finished_at']); latest = max(latest, finished) if latest else finished
        cpu += r['cpu_seconds']
    for case in WORLDS:
        folder = contract['world_execution'][case]
        complete, execution = read(folder+'/complete.json'), read(folder+'/execution.json')
        attempts = execution['attempts']
        if (complete['n_fits'] != 16 or complete['protocol_sha256'] != REGISTRATION or
                [v['cell_index'] for v in attempts] != [c['index'] for c in contract['cells'] if c['case'] == case] or
                any(v['status'] != 'complete' or type(v['exit_code']) is not int or v['exit_code'] != 0 or
                    type(v['peak_tree_rss_bytes']) is not int or not finite(v['peak_tree_rss_bytes']) or
                    v['peak_tree_rss_bytes'] > contract['resources']['rss_bytes'] or
                    'elapsed_seconds' in v and not finite(v['elapsed_seconds']) for v in attempts)):
            raise ValueError('all world attempts must be retained and successful')
    barrier = contract['barrier']
    if barrier['fit_seal_sha256'] != contract['fit_seal_sha256'] or datetime.fromisoformat(barrier['at']) < latest:
        raise ValueError('full fit-before-evaluation barrier changed')
    for name, original in [('B/acceptance.json', contract['upstream_gate']['acceptance_sha256']),
                           ('B/scores.json', contract['upstream_gate']['scores_sha256'])]:
        derivation = contract['metadata_projections'][name]
        if (derivation['original_sha256'] != original or derivation['derived_sha256'] != entries[name]['sha256'] or
                derivation['method'] != 'explicit structured projection; original private bytes preserved'):
            raise ValueError('original outcome derivation link changed')
    # Semantic JSON screening/bindings decode outcomes, so run only after barrier.
    integrity = verify(root, expected)
    # All 640 fits and 40 journals checked before heldout arrays are parsed.
    score, acceptance = read('B/scores.json'), read('B/acceptance.json')
    if set(score['cells']) != {str(i) for i in range(640)}:
        raise ValueError('complete score matrix required')
    primary, metrics, deltas = [], [], {}
    for case in WORLDS:
        spec, actions = descriptors[case]; folder = contract['evaluation'][case]
        for name, digest in a['evaluation_receipt_hashes'][case].items():
            if entries[folder+'/'+name]['sha256'] != digest:
                raise ValueError('original heldout response bytes changed')
        evaluation = core.history_receipt(relative(root, folder), spec, actions['evaluation'], True)
        first = json.loads(bytes_at(relative(root, folder+'/queries.ndjson')).decode().splitlines()[0])
        if datetime.fromisoformat(first['at']) < datetime.fromisoformat(barrier['at']):
            raise ValueError('heldout charge preceded full fit seal')
        for cell in (c for c in contract['cells'] if c['case'] == case):
            i, r = cell['index'], fits[cell['index']]; saved = score['cells'][str(i)]
            identity = {k: cell[k] for k in ('index', 'case', 'graph_size', 'history', 'arm', 'init', 'binding')}
            pred = folder+'/cell-'+str(i)+'-predictions.npz'
            if (saved['cell'] != identity or saved['evaluation_input_sha256'] != entries[folder+'/input.json']['sha256'] or
                    saved['predictions_sha256'] != entries[pred]['sha256']):
                raise ValueError('derived score/cell identity or original prediction binding changed')
            model_bytes, _ = captured(relative(root, cell['folder']+'/models.pt'), cell['model_sha256'])
            states = torch.load(BytesIO(model_bytes), map_location='cpu', weights_only=True)
            replayed, models = core.replay_predictions(evaluation, states, MLPSurrogate, torch, cell['arm'])
            predictions, delta = {}, 0.
            prediction_bytes, _ = captured(relative(root, pred), saved['predictions_sha256'])
            with np.load(BytesIO(prediction_bytes), allow_pickle=False) as cached:
                if set(cached.files) != set(replayed):
                    raise ValueError('prediction node membership changed')
                for node, values in replayed.items():
                    original = cached[node]
                    if (original.shape != values.shape or original.dtype != values.dtype or
                            not np.allclose(original, values, rtol=core.REPLAY_RTOL, atol=core.REPLAY_ATOL, equal_nan=False) or
                            node in evaluation['roots'] and not np.array_equal(original, values)):
                        raise ValueError('checkpoint/prediction or root-clamp discrepancy')
                    delta = max(delta, float(np.max(np.abs(original-values))))
                    predictions[node] = original.copy()
            deltas[str(i)] = delta
            for node, model in models.items():
                if sum(p.numel() for p in model.parameters()) != r['heads'][node]['parameters']:
                    raise ValueError('parameter accounting changed')
                for attr, key in [('in_lo', 'lo'), ('in_hi', 'hi')]:
                    if not torch.equal(getattr(model, attr).reshape(-1), torch.tensor(r['normalizers'][node][key], dtype=torch.float32)):
                        raise ValueError('checkpoint normalization changed')
            metric = metric_from_cached(evaluation, training[case+'/'+cell['history']], predictions, models, cell['arm'], torch)
            core.match(saved['metric'], metric, 'cell'+str(i))
            metrics.append((cell, metric['nmse']))
            if cell['init'] == 0 and cell['arm'] in ('delivery', 'online', 'simpler'):
                primary.append({'graph_size': cell['graph_size'], 'system_id': case, 'history': cell['history'],
                                'arm': cell['arm'], 'init': 0, 'nmse': metric['nmse'], 'status': 'complete'})
        print(json.dumps({'event': 'world_replayed', 'world': case, 'fits': 16}), file=sys.stderr, flush=True)
    core.match(score['primary_rows'], primary, 'primary rows')
    stats, secondary = core.primary_statistics(primary), core.secondary_summary(metrics)
    core.match(score['primary_analysis'], stats, 'score primary')
    core.match(acceptance['primary_analysis'], stats, 'original acceptance primary')
    core.match(acceptance['secondary_descriptive_analysis'], secondary, 'original acceptance secondary')
    core.match(contract['complete']['fit_cpu_core_hours'], cpu/3600, 'fit CPU')
    return {'integrity': integrity, 'full_supplemental_replay': True, 'checkpoints_replayed': 640,
            'primary_cells_recomputed': 240, 'cached_responses_checked': 48000,
            'primary_analysis': stats, 'secondary_descriptive_analysis': secondary,
            'replay_max_abs_deltas': deltas, 'replay_tolerance': {'rtol': 1e-6, 'atol': 1e-7, 'roots': 'exact'},
            'runtime': actual, 'replay_cpu_seconds': time.process_time()-cpu0,
            'supplemental_io_adapters': ['hash-bound snapshot JSON', 'hash-bound contiguous journal', 'captured model/prediction bytes'],
            'replay_wall_seconds': time.monotonic()-wall0, 'new_optimizer_updates': 0, 'new_responses': 0,
            'scope': 'supplemental relative-artifact replay; unchanged original acceptance remains separate; no historical freeze or anonymity certification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('exclusive receipt outside verified package required')
    result = replay(args.root, args.expected_manifest_sha256)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('full_supplemental_replay', 'checkpoints_replayed', 'new_responses')}))
