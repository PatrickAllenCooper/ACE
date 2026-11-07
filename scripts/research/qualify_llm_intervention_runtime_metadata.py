"""CPU-only observation of a pinned captured runtime; no service qualification.

Never repairs metadata/caches or calls the old service.authenticate gate. Only
archive stage and the original probe's kill_owned are reused. No model is loaded.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib
import importlib.util
import io
import json
import marshal
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import subprocess
import sys
import time
import traceback
import types

VERSION = 'ace-runtime-metadata-diagnostic-v1'
ARCHIVE_SHA = '6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98'
MANIFEST_SHA = 'b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a'
HELPER_SHA = '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e'
PROBE_SHA = '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951'
RUNTIME_SOURCE = '/projects/paco0228/miniconda3/envs/ace'
ACTUAL = {'python': '3.10.19', 'numpy': '2.2.5', 'scipy': '1.15.3', 'torch': '2.9.1+cu128', 'transformers': '4.57.3'}
RESOURCES = {'account': 'ucb736_asc2', 'partition': 'acpu', 'qos': 'cpu-normal',
             'cpus': 1, 'memory_gib': 3, 'wall_seconds': 600, 'gpus': 0, 'requeue': False}
LIMITS = {'supervisor_seconds': 580, 'child_success_seconds': 180, 'child_outer_seconds': 185,
          'cleanup_seconds': 5, 'stage_peak_rss_bytes': 512*1024**2, 'worker_peak_rss_bytes': 1536*1024**2}
CACHE_PAIRS = (
    ('lib/python3.10/site-packages/numpy/__pycache__/__config__.cpython-310.pyc', 'lib/python3.10/site-packages/numpy/__config__.py'),
    ('lib/python3.10/site-packages/scipy/__pycache__/__config__.cpython-310.pyc', 'lib/python3.10/site-packages/scipy/__config__.py'),
    ('lib/python3.10/__pycache__/_sysconfigdata__linux_x86_64-linux-gnu.cpython-310.pyc', 'lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py'))
FLAGS = {'model_weights_loaded': False, 'explicit_cuda_api_called': False, 'training_updates': 0,
         'environment_responses': 0, 'service_qualified': False, 'project_closure_qualified': False,
         'historical_runtime_certified': False, 'runtime_adopted': False, 'retry_allowed': False}
BOOTSTRAP = '''import hashlib,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
with path.open('rb') as f: data=f.read(1024**2+1)
if len(data)>1024**2 or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('independent bootstrap source pin differs')
sys.argv=[str(path),*sys.argv[3:]]
exec(compile(data,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
'''


def need(ok, message):
    if not ok: raise ValueError(message)


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def write(path, value):
    with Path(path).open('xb') as f:
        f.write(raw(value)); f.flush(); os.fsync(f.fileno())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, maximum):
    with Path(path).open('rb') as f: data = f.read(maximum+1)
    need(len(data) <= maximum, 'input exceeds byte bound: '+str(path))
    return data


def pinned(path, expected, maximum=32*1024**2):
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None, 'invalid independent pin')
    data = read(path, maximum)
    need(sha(data) == expected, 'independent pin differs: '+str(path))
    return data


def decode(data):
    def unique(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    value = json.loads(data, object_pairs_hook=unique)
    raw(value)
    return value


def keys(value, expected):
    need(type(value) is dict and set(value) == set(expected), 'exact packet fields required')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def costs(start, cpu):
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {'utc': utc(), 'elapsed_seconds': time.monotonic()-start, 'process_cpu_seconds': time.process_time()-cpu,
            'peak_self_rss_bytes': r.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)}


def authenticate(args):
    p = decode(pinned(args.packet, args.packet_sha256, 128*1024))
    keys(p, ('version', 'sources', 'archive', 'manifest', 'runtime_source', 'output', 'resources', 'limits', 'actual_versions', 'numpy_metadata_versions'))
    need(p['version'] == VERSION and raw(p['resources']) == raw(RESOURCES) and raw(p['limits']) == raw(LIMITS)
         and p['actual_versions'] == ACTUAL and p['numpy_metadata_versions'] == ['2.2.5', '2.2.6']
         and p['runtime_source'] == RUNTIME_SOURCE, 'fixed diagnostic contract differs')
    keys(p['sources'], ('self', 'helper', 'probe'))
    source_bytes = {}
    for name in ('self', 'helper', 'probe'):
        item = p['sources'][name]; keys(item, ('path', 'sha256'))
        need(Path(item['path']).is_absolute(), 'absolute source path required')
        expected = getattr(args, name+'_sha256')
        need(item['sha256'] == expected, 'independent source inventory differs')
        source_bytes[name] = pinned(item['path'], expected, 1024**2)
    need(Path(p['sources']['self']['path']).resolve() == Path(__file__).resolve(), 'running self path differs')
    need(p['sources']['helper']['sha256'] == HELPER_SHA and p['sources']['probe']['sha256'] == PROBE_SHA,
         'unchanged reviewed helper/probe required')
    for name, expected in (('archive', ARCHIVE_SHA), ('manifest', MANIFEST_SHA)):
        keys(p[name], ('path', 'sha256'))
        need(Path(p[name]['path']).is_absolute() and p[name]['sha256'] == expected, 'captured archive/manifest pin differs')
    need(type(p['output']) is str and Path(p['output']).is_absolute(), 'exclusive absolute output required')
    need(not inside(p['output'], RUNTIME_SOURCE), 'output must not modify original runtime')
    return p, source_bytes


def load_verified(data, path, name):
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def inside(path, parent):
    p, base = Path(path).resolve(), Path(parent).resolve()
    return p == base or base in p.parents


def allocation(out):
    job = os.environ.get('SLURM_JOB_ID', '')
    need(job.isdigit(), 'CPU Slurm job required')
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True, timeout=10)
    write(out/'allocation.json', {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode})
    fields = dict(re.findall(r'(\w+)=(\S+)', result.stdout))
    expected = {'JobId': job, 'JobState': 'RUNNING', 'Account': 'ucb736_asc2', 'Partition': 'acpu', 'QOS': 'cpu-normal',
                'NumCPUs': '1', 'NumTasks': '1', 'NumNodes': '1', 'TimeLimit': '00:10:00', 'Requeue': '0', 'Restarts': '0'}
    tres = fields.get('AllocTRES', '').split(',')
    need(result.returncode == 0 and all(fields.get(k) == v for k, v in expected.items())
         and 'cpu=1' in tres and 'mem=3G' in tres and not any('gres/' in x for x in tres), 'allocation differs from fixed CPU-only envelope')
    need(not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'), 'unexpected GPU environment')
    return job


def stage_runtime(packet, helper, staged, event):
    m = decode(pinned(packet['manifest']['path'], packet['manifest']['sha256']))
    need(m['archive_sha256'] == packet['archive']['sha256'] and m['source'] == packet['runtime_source'],
         'independent archive/source pin differs from manifest')
    # Retain only six small source/cache records; do not hold two parsed manifests
    # while the unchanged helper performs its own full verification/extraction.
    wanted = {x for pair in CACHE_PAIRS for x in pair}
    cache_pins = {r['path']: r['sha256'] for r in m['records'] if r['path'] in wanted and r['kind'] == 'file'}
    need(set(cache_pins) == wanted, 'three exact cache/source pairs required')
    free = os.statvfs(staged.parent)
    need(free.f_bavail*free.f_frsize >= m['file_bytes']+1024**3, 'insufficient node-local space')
    del m; gc.collect()
    stage = helper.stage(packet['archive']['path'], packet['manifest']['path'], packet['manifest']['sha256'], staged,
                         progress=lambda detail: event('stage_progress', detail=detail))
    need(stage['archive_sha256'] == packet['archive']['sha256'], 'staged archive pin differs')
    return {**stage, 'cache_source_sha256': cache_pins}


def code_shape(value):
    """Compare semantics AND debug tables recursively; never execute cached code."""
    if isinstance(value, types.CodeType):
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals', 'co_stacksize', 'co_flags',
                  'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_filename', 'co_name', 'co_qualname',
                  'co_firstlineno', 'co_lnotab', 'co_linetable', 'co_exceptiontable', 'co_freevars', 'co_cellvars')
        return ('code', tuple((k, code_shape(getattr(value, k))) for k in fields if hasattr(value, k)))
    if type(value) is tuple: return ('tuple', tuple(code_shape(x) for x in value))
    if type(value) is frozenset: return ('frozenset', tuple(sorted((code_shape(x) for x in value), key=repr)))
    if type(value) is float: return ('float', struct.pack('!d', value))
    if type(value) is complex: return ('complex', struct.pack('!dd', value.real, value.imag))
    need(value is None or value is Ellipsis or type(value) in (bool, int, str, bytes), 'unexpected cached code constant')
    return (type(value).__name__, value)


def inspect_cache(cache, source):
    b, s = read(cache, 1024**2), read(source, 1024**2)
    stat = Path(source).stat()
    result = {'cache_path': str(cache), 'source_path': str(source), 'cache_sha256': sha(b), 'source_sha256': sha(s),
              'cache_bytes': len(b), 'source_bytes': len(s), 'header_hex': b[:16].hex(), 'code_equivalent': False,
              'source_stat_mtime': stat.st_mtime, 'source_stat_mtime_ns': stat.st_mtime_ns, 'source_stat_size': stat.st_size,
              'admissible_for_diagnosis': False, 'cache_executed_by_comparison': False,
              'historical_hash_difference_explained': False}
    try:
        need(len(b) >= 16 and b[:4] == importlib.util.MAGIC_NUMBER, 'cache magic/header differs from this interpreter')
        flags = int.from_bytes(b[4:8], 'little'); result['flags'] = flags
        need(flags in (0, 1, 3), 'invalid cache header flags')
        if flags & 1:
            result['header_kind'] = 'hash'; result['header_source_hash_hex'] = b[8:16].hex()
            need(b[8:16] == importlib.util.source_hash(s), 'cache header source hash differs')
        else:
            result.update(header_kind='timestamp', header_timestamp=int.from_bytes(b[8:12], 'little'),
                          header_source_bytes=int.from_bytes(b[12:16], 'little'))
            result['timestamp_matches_copied_source'] = result['header_timestamp'] == (int(stat.st_mtime) & 0xffffffff)
            result['size_matches_copied_source'] = result['header_source_bytes'] == (stat.st_size & 0xffffffff)
            result['cpython_timestamp_reject_expected'] = not (result['timestamp_matches_copied_source'] and result['size_matches_copied_source'])
            if result['cpython_timestamp_reject_expected']:
                result.update(admissible_for_diagnosis=True, code_equivalent=None,
                              reconciliation='stale timestamp header: CPython source compilation expected, not traced',
                              code_comparison_performed=False)
                return result
        stream = io.BytesIO(b[16:]); loaded = marshal.load(stream)
        need(type(loaded) is types.CodeType and not stream.read(1), 'one bounded code object required')
        result['compiled_filename'] = loaded.co_filename
        compiled = compile(s, loaded.co_filename, 'exec', dont_inherit=True, optimize=0)
        result['code_equivalent'] = code_shape(loaded) == code_shape(compiled)
        need(result['code_equivalent'], 'captured cache differs from compilation of captured source')
        result.update(admissible_for_diagnosis=True, code_comparison_performed=True,
                      reconciliation='eligible header: exact recursive code-field equality')
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    return result


def capture_caches(prefix, expected, out):
    rows = [inspect_cache(prefix/a, prefix/b) for a, b in CACHE_PAIRS]
    for row, (a, b) in zip(rows, CACHE_PAIRS):
        row['matches_stage_pins'] = row['cache_sha256'] == expected[a] and row['source_sha256'] == expected[b]
    write(out/'cache-reconciliation.json', {'pairs': rows, 'no_cache_modified': True,
        'pyc_reads_allowed': True, 'source_only_execution': False,
        'scope': 'local code-field equivalence only; original Conda cache hash differences remain unexplained'})
    need(all(r['matches_stage_pins'] and r['admissible_for_diagnosis'] and 'error' not in r for r in rows), 'cache/source reconciliation failed before package imports')
    return rows


def metadata_inventory(metadata=None):
    if metadata is None:
        import importlib.metadata as metadata
    rows, names = [], set()
    for d in metadata.distributions():
        need(len(rows) < 4096, 'metadata distribution bound exceeded')
        name, version = d.metadata.get('Name'), d.version
        need(type(name) is str and 0 < len(name) <= 256 and type(version) is str and len(version) <= 128, 'invalid distribution identity')
        rows.append({'name': name, 'version': version, 'metadata_path': str(Path(d._path).resolve()),
                     'distribution_root': str(Path(d.locate_file('')).resolve())})
        names.add(re.sub(r'[-_.]+', '-', name).lower())
    defaults = {}
    for name in sorted(names):
        d = metadata.distribution(name)
        defaults[name] = {'version': metadata.version(name), 'metadata_path': str(Path(d._path).resolve())}
    return {'distributions': rows, 'resolved_defaults': defaults,
            'numpy_entries': [r for r in rows if re.sub(r'[-_.]+', '-', r['name']).lower() == 'numpy'],
            'default_selection_is_not_imported_version': True}


def check_metadata(inventory, prefix):
    need(sorted(r['version'] for r in inventory['numpy_entries']) == ['2.2.5', '2.2.6'], 'both captured NumPy metadata versions must remain visible')
    need(all(inside(r['metadata_path'], prefix) and inside(r['distribution_root'], prefix) for r in inventory['distributions'])
         and all(inside(r['metadata_path'], prefix) for r in inventory['resolved_defaults'].values()), 'mixed distribution metadata paths')


def runtime_snapshot(prefix, original, *, modules=None, maps=None):
    modules = sys.modules if modules is None else modules
    paths = {name: str(Path(vars(m)['__file__']).resolve()) for name, m in list(modules.items())
             if isinstance(m, types.ModuleType) and type(vars(m).get('__file__')) is str and not vars(m)['__file__'].startswith('<')}
    package = {name: {'loaded': name in modules, 'version': vars(modules[name]).get('__version__') if name in modules else None,
                      'origin': paths.get(name)} for name in ('numpy', 'scipy', 'torch', 'transformers')}
    allowed = Path(__file__).resolve()
    mixed = [p for p in paths.values() if inside(p, original) or not inside(p, prefix) and Path(p) != allowed]
    if maps is None: maps = read('/proc/self/maps', 8*1024**2).decode('utf-8', errors='replace')
    mapped = [line.split(None, 5)[5].removesuffix(' (deleted)') for line in maps.splitlines()
              if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')]
    native_mixed = [p for p in mapped if inside(p, original) or not inside(p, prefix)
                    and not any(inside(p, root) for root in ('/lib', '/lib64', '/usr/lib', '/usr/lib64'))]
    return {'python_version': '.'.join(map(str, sys.version_info[:3])), 'executable': sys.executable, 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'sys_path': list(sys.path), 'packages': package, 'module_paths': paths,
            'mixed_module_paths': sorted(set(mixed)), 'mixed_native_paths': sorted(set(native_mixed)),
            'process_maps': maps, 'scope': 'observed loaded paths only; not universal binary dependency closure'}


def check_prefix(prefix, snapshot):
    need(Path(snapshot['prefix']).resolve() == prefix.resolve() and inside(snapshot['executable'], prefix)
         and Path(snapshot['base_prefix']).resolve() == prefix.resolve(), 'copied interpreter prefix differs')
    need(all(type(p) is str and Path(p).is_absolute() and inside(p, prefix) for p in snapshot['sys_path']), 'mixed interpreter search paths')
    need(not snapshot['mixed_module_paths'] and not snapshot['mixed_native_paths'], 'mixed loaded module/native paths')


def original_imports(event):
    event('torch_start')
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    event('torch_complete')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    event('transformers_auto_complete')
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    event('qwen_class_complete')
    import numpy
    import scipy
    return numpy, scipy


def arithmetic(numpy, scipy_linalg):
    a, b = numpy.array([[3., 1.], [1., 2.]]), numpy.array([9., 8.])
    dot = float(numpy.dot(numpy.array([1., 2.]), numpy.array([3., 4.])))
    n = list(map(float, numpy.linalg.solve(a, b)))
    s = list(map(float, scipy_linalg.solve(a, b)))
    return {'dot': dot, 'numpy_solution': n, 'scipy_solution': s, 'expected_solution': [2., 3.],
            'passed': dot == 11. and len(n) == len(s) == 2 and all(math.isclose(x, y, rel_tol=0., abs_tol=1e-12)
                     for values in (n, s) for x, y in zip(values, (2., 3.)))}


def publish(out, value):
    """One candidate/terminal; errors after linking never publish another outcome."""
    expected = raw(value); target = out/'terminal.json'
    try:
        write(out/'terminal.candidate.json', value)
        os.link(out/'terminal.candidate.json', target)
    except BaseException as exc:
        diagnostic = {'type': type(exc).__name__, 'message': str(exc), 'retry_allowed': False}
        try:
            observed = target.read_bytes()
        except FileNotFoundError:
            diagnostic['authority'] = 'no_terminal_published'
        except OSError as err:
            diagnostic.update(authority='requires_outer_reconciliation', read_error=str(err))
        else:
            diagnostic.update(authority='terminal.json' if observed == expected else 'requires_outer_reconciliation',
                              expected_sha256=sha(expected), observed_sha256=sha(observed))
        write(out/'publication-diagnostic.json', diagnostic)
        raise


def events(out, start, cpu, memory_cap):
    serial = 0
    def event(phase, **extra):
        nonlocal serial
        value = dict(phase=phase, **costs(start, cpu), **extra)
        write(out/f'phase-{serial:04d}.json', value); serial += 1
        need(value['peak_self_rss_bytes'] <= memory_cap, 'observed process RSS exceeded diagnostic budget')
    return event


def capture_snapshot(prefix, out, name):
    snapshot = runtime_snapshot(prefix, RUNTIME_SOURCE)
    maps = snapshot.pop('process_maps')
    with (out/(name+'.maps.txt')).open('x') as f:
        f.write(maps); f.flush(); os.fsync(f.fileno())
    write(out/(name+'.json'), snapshot)
    return snapshot


def worker(packet, args, start, cpu):
    out = Path(packet['output'])/'worker'; out.mkdir(mode=0o700, exist_ok=False)
    prefix = Path(args.staged_prefix).resolve()
    event = events(out, start, cpu, LIMITS['worker_peak_rss_bytes'])
    error, snapshot = None, None
    try:
        stage = decode(pinned(Path(packet['output'])/'stage.json', args.stage_sha256, 16384))
        need(stage['staged_prefix'] == str(prefix) and stage['manifest_sha256'] == MANIFEST_SHA
             and stage['archive_sha256'] == ARCHIVE_SHA, 'worker stage binding differs')
        snapshot = capture_snapshot(prefix, out, 'before-imports')
        check_prefix(prefix, snapshot)
        need(snapshot['python_version'] == ACTUAL['python'], 'actual Python version differs')
        event('cache_capture_start')
        capture_caches(prefix, stage['cache_source_sha256'], out)
        event('cache_reconciliation_complete')
        metadata = metadata_inventory()
        write(out/'metadata-before-imports.json', metadata)
        check_metadata(metadata, prefix)
        event('metadata_capture_complete')
        numpy, scipy = original_imports(event)
        snapshot = capture_snapshot(prefix, out, 'after-imports')
        check_prefix(prefix, snapshot)
        need({k: v['version'] for k, v in snapshot['packages'].items()} == {k: v for k, v in ACTUAL.items() if k != 'python'}, 'actual package versions differ from diagnostic contract')
        import scipy.linalg
        controls = arithmetic(numpy, scipy.linalg)
        write(out/'arithmetic.json', controls); need(controls['passed'], 'deterministic arithmetic failed')
        event('arithmetic_complete')
    except BaseException as exc:
        error = exc
        # Failure evidence may use the remainder of the parent's 185s window;
        # it cannot turn a >180s execution into a successful diagnosis.
        signal.setitimer(signal.ITIMER_REAL, max(.001, start+185-time.monotonic()))
        write(out/'failure-detail.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        try:
            snapshot = capture_snapshot(prefix, out, 'final-runtime')
            check_prefix(prefix, snapshot)
        except BaseException as exc:
            if error is None: error = exc
            write(out/'origin-capture-error.json', {'type': type(exc).__name__, 'message': str(exc)})
    measurement = costs(start, cpu)
    if error is None and (measurement['elapsed_seconds'] > 180 or measurement['peak_self_rss_bytes'] > LIMITS['worker_peak_rss_bytes']):
        error = TimeoutError('worker success envelope exceeded')
    evidence = {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
    publish(out, {'version': VERSION, 'status': 'diagnostic_complete' if error is None else 'failed', **FLAGS,
                  **measurement, 'packet_sha256': args.packet_sha256, 'stage_sha256': args.stage_sha256,
                  'actual_packages': snapshot['packages'] if snapshot else None, 'evidence_sha256': evidence,
                  'error': None if error is None else type(error).__name__+': '+str(error),
                  'dependency_capability_checks_instrumented': False, 'source_only_execution': False})
    if error is not None: raise error


def validate_worker(out, packet_pin, stage_pin):
    directory = out/'worker'
    b = read(directory/'terminal.json', 128*1024); value = decode(b)
    need((directory/'terminal.candidate.json').read_bytes() == b and value['status'] == 'diagnostic_complete'
         and value['packet_sha256'] == packet_pin and value['stage_sha256'] == stage_pin
         and all(value.get(k) == v for k, v in FLAGS.items()) and value['elapsed_seconds'] <= 180
         and value['peak_self_rss_bytes'] <= LIMITS['worker_peak_rss_bytes'], 'worker diagnostic did not complete within contract')
    need({k: v['version'] for k, v in value['actual_packages'].items()} == {k: v for k, v in ACTUAL.items() if k != 'python'}, 'worker actual versions differ')
    files = {p.name for p in directory.iterdir() if p.is_file()}
    need(files == set(value['evidence_sha256']) | {'terminal.json', 'terminal.candidate.json'}, 'worker evidence inventory differs')
    for name, expected in value['evidence_sha256'].items(): pinned(directory/name, expected)
    return sha(b)


def run_child(command, env, out, overall_deadline, kill_owned, validate):
    start = time.monotonic(); deadline = start+185
    need(overall_deadline-start >= 200, 'insufficient time for fixed child and cleanup')
    proc = None; cleanup = None; receipt_pin = None
    signal.setitimer(signal.ITIMER_REAL, deadline-start)
    try:
        with (out/'worker.stdout').open('xb') as stdout, (out/'worker.stderr').open('xb') as stderr:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            write(out/'worker.started.json', {'command': command, 'pid': proc.pid, 'pgid': proc.pid, 'utc': utc(), 'deadline_monotonic': deadline})
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('child receipt I/O consumed outer deadline')
            rc = proc.wait(timeout=remaining)
        need(rc == 0, 'diagnostic worker failed')
        receipt_pin = validate()
        write(out/'worker.returned.json', {'returncode': rc, 'terminal_sha256': receipt_pin, 'elapsed_seconds': time.monotonic()-start})
        if time.monotonic()-start > 180: raise TimeoutError('child success including receipt I/O exceeded 180 seconds')
    finally:
        # No receipt I/O precedes owned-group cleanup. Ignore additional external
        # interrupts only during the five-second cleanup window; alarm remains.
        signal.setitimer(signal.ITIMER_REAL, 0)
        old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
        cleanup_start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, 5)
        try:
            if proc is not None:
                cleanup = kill_owned(proc)
                signal.setitimer(signal.ITIMER_REAL, max(.001, cleanup_start+5-time.monotonic()))
                write(out/'worker.cleanup.json', cleanup)
            if time.monotonic()-cleanup_start > 5: raise TimeoutError('owned cleanup including receipt I/O exceeded five seconds')
        finally:
            for s, handler in old.items(): signal.signal(s, handler)
            signal.setitimer(signal.ITIMER_REAL, max(.001, overall_deadline-time.monotonic()))
    need(cleanup is not None and cleanup['group_still_exists'] is False and not cleanup['errors']
         and cleanup['returncode'] == 0, 'owned process group cleanup unconfirmed')
    return {'terminal_sha256': receipt_pin, 'elapsed_including_cleanup_seconds': time.monotonic()-start, 'cleanup': cleanup}


def child_command(packet, args, prefix, stage_pin):
    return [str(prefix/'bin/python'), '-I', '-B', '-X', 'importtime', '-c', BOOTSTRAP,
            packet['sources']['self']['path'], args.self_sha256,
            '--packet', str(Path(args.packet).absolute()), '--packet-sha256', args.packet_sha256,
            '--self-sha256', args.self_sha256, '--helper-sha256', args.helper_sha256, '--probe-sha256', args.probe_sha256,
            '--worker', '--staged-prefix', str(prefix), '--stage-sha256', stage_pin]


def supervise(packet, source_bytes, args, start, cpu):
    out = Path(packet['output']); out.mkdir(mode=0o700, exist_ok=False)
    event = events(out, start, cpu, LIMITS['stage_peak_rss_bytes'])
    error, result = None, None
    try:
        write(out/'packet.json', packet)
        event('authenticated', packet_sha256=args.packet_sha256)
        job = allocation(out)
        scratch = os.environ.get('SLURM_SCRATCH', '')
        need(bool(scratch) and Path(scratch).is_absolute(), 'absolute SLURM_SCRATCH required')
        scratch = Path(scratch).resolve(strict=True)
        need(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), 'owned node scratch required')
        local = scratch/('ace-runtime-metadata-'+job); local.mkdir(mode=0o700, exist_ok=False)
        helper = load_verified(source_bytes['helper'], packet['sources']['helper']['path'], 'metadata_archive_helper')
        probe = load_verified(source_bytes['probe'], packet['sources']['probe']['path'], 'metadata_original_probe')
        event('stage_start', scratch=str(local), prior_stage_seconds=35.303541)
        stage = stage_runtime(packet, helper, local/'runtime', event)
        write(out/'stage.json', stage); stage_pin = sha((out/'stage.json').read_bytes())
        event('stage_complete', staged_prefix=stage['staged_prefix'])
        prefix = Path(stage['staged_prefix'])
        command = child_command(packet, args, prefix, stage_pin)
        env = dict(os.environ)
        for k in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD'): env.pop(k, None)
        env.update(LD_LIBRARY_PATH=str(prefix/'lib'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        event('worker_start', command=command)
        result = run_child(command, env, out, start+580, probe.kill_owned,
                           lambda: validate_worker(out, args.packet_sha256, stage_pin))
        event('worker_complete', **result)
    except BaseException as exc:
        error = exc
        write(out/'failure-detail.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    measurement = costs(start, cpu)
    if error is None and measurement['elapsed_seconds'] >= 580: error = TimeoutError('supervisor deadline exceeded')
    publish(out, {'version': VERSION, 'status': 'diagnostic_complete' if error is None else 'failed', **FLAGS,
                  **measurement, 'packet_sha256': args.packet_sha256, 'archive_sha256': ARCHIVE_SHA, 'manifest_sha256': MANIFEST_SHA,
                  'worker': result, 'error': None if error is None else type(error).__name__+': '+str(error),
                  'memory_budget_is_not_proof_of_cgroup_headroom': True})
    if error is not None: raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'packet-sha256', 'self-sha256', 'helper-sha256', 'probe-sha256'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--staged-prefix'); parser.add_argument('--stage-sha256')
    args = parser.parse_args()
    start, cpu = time.monotonic(), time.process_time()
    def interrupted(signum, frame): raise InterruptedError('diagnostic signal '+str(signum))
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(s, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 180 if args.worker else 580)
    try:
        packet, sources = authenticate(args)
        if args.worker:
            need(args.staged_prefix is not None and args.stage_sha256 is not None, 'worker stage pins required')
            return worker(packet, args, start, cpu)
        need(args.staged_prefix is None and args.stage_sha256 is None, 'supervisor cannot borrow a prior stage')
        return supervise(packet, sources, args, start, cpu)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__': main()
