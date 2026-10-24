"""Supervised, source-pinned reporting from accepted replay12 artifacts only.

No model loading, optimization, simulator calls or new inferential rules. The
separate replay acceptance is a prerequisite, not established by this runner.
Original reporters are captured unchanged. Runtime inventory checking reuses
the accepted qualifier; only NumPy/SciPy are imported for statistical reporting.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import types
from datetime import datetime, timezone

WALL_SECONDS = 3600
CHILD_SECONDS = 3550
SLURM_WALL = '01:00:00'

ORIGINAL_INPUTS = {'registration.json', 'acceptance.json', 'scores.json',
                   'complete.json', 'fit_seal.json', 'audit_execution.json'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def capture(path, pin):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'regular pinned input required')
    raw = path.read_bytes()
    require(sha(raw) == pin, 'input changed: '+str(path))
    return raw


def write(root, name, value):
    require(Path(name).name == name, 'output basename required')
    fd = os.open(root/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def module(root, name, sources):
    require(name not in sys.modules, 'cached project module forbidden: '+name)
    obj = types.ModuleType(name); obj.__file__ = str(root/(name+'.py'))
    sys.modules[name] = obj
    exec(compile(sources[name+'.py'], obj.__file__, 'exec', dont_inherit=True), obj.__dict__)
    return obj


def resource_contract(reg):
    require(type(reg['wall_seconds']) is int and reg['wall_seconds'] == WALL_SECONDS and reg['rss_bytes'] == 3*2**30
            and reg['cpus'] == 1 and reg['account'] == 'ucb736_asc1'
            and reg['partition'] == 'acpu' and reg['qos'] == 'cpu-normal'
            and reg['slurm_wall_limit'] == SLURM_WALL, 'bounded resource contract differs')


def context(root, pin):
    root = Path(root).resolve(strict=True)
    reg = json.loads(capture(root/'report_freeze.json', pin))
    info = root.stat()
    require(info.st_ino == reg['output_inode'] and info.st_uid == os.geteuid()
            and info.st_mode & 0o077 == 0, 'output identity or ownership changed')
    require(not sys.flags.optimize and sys.version_info[:2] == (3, 11), 'Python3.11 without optimization required')
    require(sha(Path('/proc/self/exe').read_bytes()) == reg['python_sha256']
            and Path(sys.prefix).resolve() == Path(reg['environment']).resolve(), 'interpreter differs')
    require(reg['new_fits'] == reg['new_responses'] == reg['new_optimizer_updates'] == 0,
            'report-only contract required')
    resource_contract(reg)
    sources = {name: capture(root/name, value) for name, value in reg['source_pins'].items()}
    require(sha(Path(__file__).read_bytes()) == reg['source_pins'][Path(__file__).name], 'runner source changed')
    for name, value in reg['metadata_pins'].items():
        capture(root/name, value)
    accepted = json.loads(capture(root/'replay_acceptance.json', reg['metadata_pins']['replay_acceptance.json']))
    require(accepted['accepted'] is True and accepted['job_id'] == 34863006
            and accepted['replay_receipt_sha256'] == reg['metadata_pins']['replay_receipt.json']
            and accepted['manifest_sha256'] == reg['manifest_sha256'], 'independent acceptance differs')
    require(accepted['runtime_inventory_sha256'] == reg['metadata_pins']['runtime_inventory.json'],
            'inventory is not the accepted replay inventory')
    require(set(reg['original_input_pins']) == ORIGINAL_INPUTS, 'complete original-input pin set required')
    return root, reg, sources


def original_lineage(root, reg, replay_module):
    """Join historical inputs to the independently accepted projected package."""
    require(set(reg['original_input_pins']) == ORIGINAL_INPUTS, 'complete original-input pin set required')
    package = Path(reg['package']).resolve(strict=True)
    manifest = json.loads(capture(package/'manifest.json', reg['manifest_sha256']))
    entries = {e['path']: e for e in manifest['files']}
    require(len(entries) == len(manifest['files']), 'duplicate manifest entry')
    accepted = json.loads(capture(root/'replay_acceptance.json', reg['metadata_pins']['replay_acceptance.json']))
    contract_pin = accepted['contract_sha256']
    require(entries['B/replay_contract.json']['sha256'] == contract_pin, 'accepted contract differs')
    contract = json.loads(capture(package/'B/replay_contract.json', contract_pin))
    replay_module.gate(contract)
    upstream = contract['upstream_gate']
    expected = {'registration.json': contract['registration_sha256'],
                'acceptance.json': upstream['acceptance_sha256'],
                'scores.json': upstream['scores_sha256'],
                'complete.json': contract['original_complete_sha256'],
                'fit_seal.json': contract['fit_seal_sha256'],
                'audit_execution.json': upstream['audit_execution_sha256']}
    require(reg['original_input_pins'] == expected, 'original/projected study lineage differs')
    original = Path(reg['original_results']).resolve(strict=True)
    for name, value in expected.items():
        capture(original/name, value)
    return original


def statistics_runtime(root, prefix, inventory, qualifier):
    """Same inventory/import checks, with phase/owned-stack diagnostics only."""
    fd = os.open(root, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        qualifier.start_diagnostics(fd)
        qualifier.diagnostic_phase(fd, 'report_inventory_started')
        entries = qualifier.verify_environment_inventory(prefix, inventory)
        qualifier.diagnostic_phase(fd, 'report_inventory_verified', objects=len(entries))
        write(root, 'report_inventory_verified.json', {'at': utc(), 'environment_objects': len(entries),
              'diagnostic_only': True})
        dependencies = {}
        for name, version in {'numpy': '2.2.6', 'scipy': '1.15.3'}.items():
            qualifier.diagnostic_phase(fd, 'report_'+name+'_import_started')
            require(name not in sys.modules, 'fresh statistics import required')
            value = importlib.import_module(name); dist = importlib.metadata.distribution(name)
            origin = Path(value.__file__).resolve()
            require(value.__version__ == dist.version == version and origin.is_relative_to(prefix)
                    and origin == Path(dist.locate_file(name+'/__init__.py')).resolve(), 'statistics runtime mismatch')
            dependencies[name] = {'version': version, 'origin': str(origin), 'sha256': sha(origin.read_bytes())}
            qualifier.diagnostic_phase(fd, 'report_'+name+'_import_verified')
        return entries, dependencies
    finally:
        try:
            qualifier.stop_diagnostics()
        finally:
            os.close(fd)


def child(root, pin):
    root, reg, sources = context(root, pin)
    cpu0, wall0 = time.process_time(), time.monotonic()
    require(not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'no models in reporting')
    def no_science_import(event, args):
        if event == 'import' and args and (args[0] == 'torch' or args[0].startswith('torch.')):
            raise RuntimeError('model runtime forbidden in report-only stage')
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('network forbidden in report-only stage')
    sys.addaudithook(no_science_import)
    qualifier = module(root, 'qualify_delivery_replay_runtime', sources)
    prefix, base = Path(reg['environment']).resolve(), Path(sys.base_prefix).resolve()
    qualifier.search_path_check(prefix, base)
    write(root, 'report_runtime_started.json', {'at': utc(), 'freeze_sha256': pin})
    inventory = json.loads(capture(root/'runtime_inventory.json', reg['metadata_pins']['runtime_inventory.json']))
    entries, dependencies = statistics_runtime(root, prefix, inventory, qualifier)
    write(root, 'report_runtime_verified.json', {'at': utc(), 'environment_objects': len(entries),
        'inventory_sha256': reg['metadata_pins']['runtime_inventory.json'], 'dependencies': dependencies,
        'elapsed_seconds': time.monotonic()-wall0, 'process_cpu_seconds': time.process_time()-cpu0})
    modules = {}
    for name in ('runner_delivery_confirmation', 'audit_delivery_prospective_results',
                 'verify_delivery_release', 'replay_delivery_prospective_release',
                 'export_delivery_prospective_claims', 'prepare_delivery_prospective_supplement'):
        modules[name] = module(root, name, sources)
    original = original_lineage(root, reg, modules['replay_delivery_prospective_release'])
    write(root, 'report_exports_started.json', {'at': utc(), 'freeze_sha256': pin})
    modules['export_delivery_prospective_claims'].export(original, root/'primary')
    result = modules['prepare_delivery_prospective_supplement'].export(
        Path(reg['package']).resolve(), reg['manifest_sha256'], root/'replay_receipt.json',
        reg['metadata_pins']['replay_receipt.json'], root/'descriptive')
    counts = {k: len(result[k]) for k in ('cells', 'fixed_init0', 'system_log_ratios',
              'history_specific_ratios', 'short_fit_ablation', 'primary_floor_counts')}
    require(counts == dict(zip(counts, (640, 320, 80, 8, 2, 4))), 'report membership differs')
    # Rebind loaded statistics code to the already authenticated inventory.
    loaded = {}
    for name, obj in list(sys.modules.items()):
        if name.split('.')[0] not in ('numpy', 'scipy'):
            continue
        filename = vars(obj).get('__file__')
        if not filename:
            continue  # synthetic aliases carry no separately loaded file
        path = Path(filename).resolve()
        require(path.is_relative_to(prefix), 'external statistics module')
        key = path.relative_to(prefix).as_posix(); entry = entries.get(key)
        require(entry is not None and entry['kind'] == 'file' and sha(path.read_bytes()) == entry['sha256'],
                'loaded statistics file changed')
        loaded[name] = {'path': key, 'sha256': entry['sha256']}
    qualifier.search_path_check(prefix, base)
    for name, value in reg['source_pins'].items():
        capture(root/name, value)
    for name, value in reg['original_input_pins'].items():
        capture(original/name, value)
    outputs = {p.relative_to(root).as_posix(): sha(p.read_bytes())
               for folder in ('primary', 'descriptive') for p in (root/folder).iterdir()}
    require(len(outputs) == 13, 'both complete report directories required')
    write(root, 'report_receipt.json', {'status': 'complete', 'at': utc(), 'freeze_sha256': pin,
        'source_pins': reg['source_pins'], 'metadata_pins': reg['metadata_pins'],
        'original_input_pins': reg['original_input_pins'], 'outputs': outputs, 'counts': counts,
        'statistics_dependencies': dependencies, 'loaded_statistics_files': loaded,
        'process_cpu_seconds': time.process_time()-cpu0, 'elapsed_seconds': time.monotonic()-wall0,
        'peak_self_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_fits': 0, 'new_optimizer_updates': 0, 'new_responses': 0,
        'scope': 'unchanged original registered exporter and descriptive reporter; no model replay here'})


BOOT = """import hashlib,sys
from pathlib import Path
root,pin,source_pin,mode=sys.argv[1:]
p=Path(root)/'run_delivery_prospective_reporting.py';raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=source_pin:raise ValueError('runner bytes differ')
sys.argv=[str(p),'--root',root,'--freeze-sha256',pin,mode]
exec(compile(raw,str(p),'exec',dont_inherit=True),{'__name__':'__main__','__file__':str(p)})
"""


def parent(root, pin):
    root, reg, sources = context(root, pin)
    started = time.monotonic(); job = os.environ.get('SLURM_JOB_ID', '')
    out = {'status': 'failed', 'job_id': job, 'freeze_sha256': pin, 'new_fits': 0, 'new_responses': 0}
    write(root, 'report_started.json', {'at': utc(), 'job_id': job, 'freeze_sha256': pin})
    fd = os.open(root, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        gate = module(root, 'supervise_delivery_qualified_replay', sources)
        out['allocation'] = gate.allocation(job, reg)
        supervisor = module(root, 'delivery_replay_supervision', sources)
        sampler = module(root, 'runner_delivery_confirmation', sources)
        env = dict(os.environ)
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
            env.pop(key, None)
        env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-I', '-B', '-c', BOOT, str(root), pin,
                   reg['source_pins']['run_delivery_prospective_reporting.py'], '--child']
        out['child'] = supervisor.supervise(command, started+CHILD_SECONDS, reg['rss_bytes'], fd,
                                           'report.log', env, sampler.process_rss)
        require(out['child']['status'] == 'complete' and out['child']['exit_code'] == 0
                and out['child']['cleanup']['complete'] is True, 'report child failed')
        raw = (root/'report_receipt.json').read_bytes(); receipt = json.loads(raw)
        require(receipt['status'] == 'complete' and receipt['freeze_sha256'] == pin, 'report receipt differs')
        for name, value in receipt['outputs'].items():
            capture(root/name, value)
        out.update(status='complete', report_receipt_sha256=sha(raw))
    except BaseException as exc:
        out.update(failure_type=type(exc).__name__, failure_reason=str(exc))
    finally:
        out.update(at=utc(), elapsed_seconds=time.monotonic()-started)
        write(root, 'report_execution.json', out); os.close(fd)
    if out['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--freeze-sha256', required=True); parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    (child if args.child else parent)(args.root, args.freeze_sha256)
