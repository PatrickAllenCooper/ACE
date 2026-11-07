"""CPU-only observation of a pinned captured runtime; no service qualification.

Never repairs metadata/caches or calls the old service.authenticate gate. Only
archive stage and the original probe's kill_owned are reused. No model is loaded.
"""
from __future__ import annotations

import argparse
import ast
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
import stat
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import types

VERSION = 'ace-runtime-metadata-diagnostic-v2'
ARCHIVE_SHA = '6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98'
MANIFEST_SHA = 'b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a'
HELPER_SHA = '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e'
PROBE_SHA = '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951'
RUNTIME_SOURCE = '/projects/paco0228/miniconda3/envs/ace'
GENERATED_MODULE = '_remote_module_non_scriptable'
GENERATOR = 'torch.distributed.nn.jit.instantiator'
TEMPLATE = 'torch.distributed.nn.jit.templates.remote_module_template'
LAZY_UTILS = 'transformers.utils.import_utils'
# Independent entries from the retained, pinned archive manifest. No installed
# source is executed to reconstruct the generated module's expected bytes.
PROVENANCE_SOURCES = {
    'torch.distributed.nn': ('torch/distributed/nn/__init__.py', '4dae25d09af680e2bb79047e7b7dacb5668328f3a557f2a28b4484c7275793f4'),
    'torch.distributed.nn.api.remote_module': ('torch/distributed/nn/api/remote_module.py', 'd51779fdbafb1ca1d99d002e6f32cf17bc1fef730a6251e3ef0278d8078bd112'),
    GENERATOR: ('torch/distributed/nn/jit/instantiator.py', '567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615'),
    TEMPLATE: ('torch/distributed/nn/jit/templates/remote_module_template.py', '0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc'),
    'transformers': ('transformers/__init__.py', '7eb0743ed843f24c1d4e8b8daf4b18249cb81403b4571a70c00be0a4c0a67bd4'),
    LAZY_UTILS: ('transformers/utils/import_utils.py', '66b2ac40be2c538dbd9409370c1f6bf769934b8b232edac365e16299c5ec1108'),
}
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


def owned_stat(path, *, directory=False, private=False):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve(strict=True), 'noncanonical/symlink provenance path')
    s = path.lstat()
    need((stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode))
         and s.st_uid == os.getuid(), 'provenance object type/ownership differs')
    need(not private or stat.S_IMODE(s.st_mode) == 0o700, 'private directory mode differs')
    if not directory:
        need(s.st_nlink == 1 and not s.st_mode & 0o022, 'provenance file links/permissions differ')
    return {k: getattr(s, 'st_'+k) for k in ('dev', 'ino', 'uid', 'gid', 'mode', 'nlink', 'size', 'mtime_ns', 'ctime_ns')}


def owned_read(path, maximum=256*1024):
    before = owned_stat(path)
    need(before['size'] <= maximum, 'provenance source exceeds bound')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        s = os.fstat(f.fileno())
        need((s.st_dev, s.st_ino) == (before['dev'], before['ino']), 'provenance file replaced before read')
        b = f.read(maximum+1)
    need(len(b) <= maximum and owned_stat(path) == before, 'provenance file changed during read')
    return b, before


def static_generated_source(generator, template):
    """Read only literal AST nodes; never import/exec/eval generator or template."""
    def assignment(body, name):
        found = [n.value for n in body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        need(len(found) == 1, 'one static assignment required: '+name)
        return found[0]
    def literal(node):
        need(isinstance(node, ast.Constant) and type(node.value) is str, 'static string literal required')
        return node.value
    g, t = ast.parse(generator), ast.parse(template)
    prefix = literal(assignment(g.body, '_FILE_PREFIX'))
    functions = [n for n in g.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'instantiate_non_scriptable_remote_module_template']
    need(len(functions) == 1 and prefix+'non_scriptable' == GENERATED_MODULE, 'known generator function/name required')
    call = assignment(functions[0].body, 'str_dict')
    need(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'dict'
         and not call.args and len(call.keywords) == 7, 'literal generator substitutions required')
    substitutions = {k.arg: literal(k.value) for k in call.keywords}
    need(set(substitutions) == {'assign_module_interface_cls', 'args', 'kwargs', 'arg_types',
                               'arrow_and_return_type', 'arrow_and_future_return_type', 'jit_script_decorator'},
         'generator substitution fields differ')
    text = literal(assignment(t.body, '_TEMPLATE_PREFIX')) + literal(assignment(t.body, '_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA'))
    result = text.format(**substitutions).encode('utf-8')
    need(len(result) <= 128*1024, 'generated source exceeds bound')
    return result


def provenance_contract(prefix):
    captured, records = {}, {}
    for name, (relative, expected) in PROVENANCE_SOURCES.items():
        path = prefix/'lib/python3.10/site-packages'/relative
        b, metadata = owned_read(path)
        need(sha(b) == expected, 'pinned provenance source differs: '+name)
        captured[name] = b
        records[name] = {'path': str(path), 'sha256': expected, 'stat': metadata}
    expected = static_generated_source(captured[GENERATOR], captured[TEMPLATE])
    return {'sources': records, 'expected_source': expected.decode('utf-8'), 'expected_source_sha256': sha(expected)}


def source_module(name, modules, contract):
    module = modules.get(name)
    need(isinstance(module, types.ModuleType), 'provenance module missing: '+name)
    d = vars(module); record = contract['sources'][name]
    need(d.get('__file__') == record['path'], 'provenance module origin differs: '+name)
    spec = d.get('__spec__')
    need(type(spec) is importlib.machinery.ModuleSpec and spec.name == name and spec.origin == record['path'],
         'provenance module spec differs: '+name)
    b, metadata = owned_read(record['path'])
    need(sha(b) == record['sha256'] and metadata == record['stat'], 'provenance source changed: '+name)
    return d


def temp_root_for(prefix):
    return prefix.parent/'worker-tmp'


def create_child_temp(prefix):
    root = temp_root_for(prefix)
    need(not inside(root, prefix) and not inside(root, RUNTIME_SOURCE), 'temporary root overlaps runtime')
    root.mkdir(mode=0o700, exist_ok=False)
    return root, owned_stat(root, directory=True, private=True)


def generated_provenance(prefix, root, root_stat, contract, modules):
    result = {'present': GENERATED_MODULE in modules or GENERATOR in modules, 'verified': False}
    if not result['present']: return result
    try:
        need(contract is not None and root_stat is not None, 'generated provenance context absent')
        result['source_pins'] = {n: r['sha256'] for n, r in contract['sources'].items() if n.startswith('torch.')}
        loaded = {n: source_module(n, modules, contract) for n in result['source_pins']}
        root_now = owned_stat(root, directory=True, private=True)
        need(all(root_now[k] == root_stat[k] for k in ('dev', 'ino', 'uid', 'gid', 'mode')), 'temporary root identity changed')
        owner = loaded[GENERATOR].get('_TEMP_DIR')
        need(type(owner) is tempfile.TemporaryDirectory, 'generator temporary owner type differs')
        directory = loaded[GENERATOR].get('INSTANTIATED_TEMPLATE_DIR_PATH')
        need(type(directory) is str and vars(owner).get('name') == directory, 'generator temporary owner path differs')
        folder = Path(directory)
        need(folder.parent == root and folder.name.startswith('tmp'), 'generated directory outside exclusive root')
        result.update(temporary_root=str(root), temporary_root_stat=root_now, directory=directory,
                      directory_stat=owned_stat(folder, directory=True, private=True), owner_type='tempfile.TemporaryDirectory')
        path = folder/(GENERATED_MODULE+'.py')
        need(sorted(p.name for p in root.iterdir()) == [folder.name]
             and sorted(p.name for p in folder.iterdir()) == [path.name], 'unexpected temporary directory contents')
        b, metadata = owned_read(path, 128*1024)
        result.update(path=str(path), file_stat=metadata, source_content=b.decode('utf-8'), source_sha256=sha(b),
                      expected_source_sha256=contract['expected_source_sha256'])
        module = modules.get(GENERATED_MODULE)
        need(type(module) is types.ModuleType, 'generated module type differs')
        d = vars(module); spec = d.get('__spec__')
        result.update(module_file=d.get('__file__'), spec_origin=spec.origin if type(spec) is importlib.machinery.ModuleSpec else None)
        need(d.get('__file__') == str(path) and type(spec) is importlib.machinery.ModuleSpec
             and spec.name == GENERATED_MODULE and spec.origin == str(path) and spec.has_location
             and spec.submodule_search_locations is None, 'generated module file/spec origin differs')
        loader = spec.loader
        need(type(loader) is importlib.machinery.SourceFileLoader and loader.path == str(path)
             and loader.name == GENERATED_MODULE, 'generated module source loader differs')
        need(b == contract['expected_source'].encode('utf-8') and sha(b) == contract['expected_source_sha256'],
             'generated source differs from pinned literal expansion')
        functions = {}
        for name in ('forward', 'forward_async', '_remote_forward'):
            f = d.get(name)
            need(type(f) is types.FunctionType and f.__module__ == GENERATED_MODULE
                 and f.__code__.co_filename == str(path), 'generated function origin differs: '+name)
            functions[name] = {'module': f.__module__, 'code_filename': f.__code__.co_filename}
        result.update(function_origins=functions, verified=True)
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    return result


def package_observation(name, modules, paths, contract):
    d = vars(modules[name]) if name in modules else {}
    direct = d.get('__version__')
    result = {'loaded': name in modules, 'origin': paths.get(name), 'direct_version': direct,
              'direct_version_present': '__version__' in d, 'version': direct,
              'version_resolution': 'module_dict' if '__version__' in d else 'absent'}
    if name != 'transformers' or name not in modules: return result
    try:
        need(contract is not None, 'lazy version provenance context absent')
        utils = source_module(LAZY_UTILS, modules, contract)
        source_module('transformers', modules, contract)
        cls = utils.get('_LazyModule')
        need(isinstance(cls, type) and type(modules[name]) is cls and cls.__module__ == LAZY_UTILS,
             'Transformers lazy module class differs')
        objects = d.get('_objects')
        need(type(objects) is dict and type(objects.get('__version__')) is str
             and len(objects['__version__']) <= 128, 'Transformers lazy version string absent')
        resolved = objects['__version__']; result['lazy_objects_version'] = resolved
        need('__version__' not in d or direct == resolved, 'direct/lazy Transformers versions conflict')
        result.update(version=resolved, version_resolution='transformers_pinned_lazy_objects')
        need(resolved == ACTUAL['transformers'], 'actual Transformers version differs')
    except Exception as exc:
        result.update(version=None, version_resolution='rejected', version_error=type(exc).__name__+': '+str(exc))
    return result


def runtime_snapshot(prefix, original, *, modules=None, maps=None, contract=None, temp_root=None, temp_stat=None):
    modules = sys.modules if modules is None else modules
    paths = {name: str(Path(vars(m)['__file__']).resolve()) for name, m in list(modules.items())
             if isinstance(m, types.ModuleType) and type(vars(m).get('__file__')) is str and not vars(m)['__file__'].startswith('<')}
    package = {name: package_observation(name, modules, paths, contract) for name in ('numpy', 'scipy', 'torch', 'transformers')}
    allowed = Path(__file__).resolve()
    mixed = [p for p in paths.values() if inside(p, original) or not inside(p, prefix) and Path(p) != allowed]
    mixed_modules = {n: p for n, p in paths.items() if p in mixed}
    if maps is None: maps = read('/proc/self/maps', 8*1024**2).decode('utf-8', errors='replace')
    mapped = [line.split(None, 5)[5].removesuffix(' (deleted)') for line in maps.splitlines()
              if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')]
    native_mixed = [p for p in mapped if inside(p, original) or not inside(p, prefix)
                    and not any(inside(p, root) for root in ('/lib', '/lib64', '/usr/lib', '/usr/lib64'))]
    return {'python_version': '.'.join(map(str, sys.version_info[:3])), 'executable': sys.executable, 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'sys_path': list(sys.path), 'packages': package, 'module_paths': paths,
            'mixed_module_paths': sorted(set(mixed)), 'mixed_native_paths': sorted(set(native_mixed)),
            'mixed_modules': mixed_modules,
            'generated_provenance': generated_provenance(prefix, temp_root, temp_stat, contract, modules),
            'process_maps': maps, 'scope': 'observed loaded paths only; not universal binary dependency closure'}


def check_prefix(prefix, snapshot, baseline=None):
    need(Path(snapshot['prefix']).resolve() == prefix.resolve() and inside(snapshot['executable'], prefix)
         and Path(snapshot['base_prefix']).resolve() == prefix.resolve(), 'copied interpreter prefix differs')
    proof = snapshot.get('generated_provenance', {})
    if proof.get('present'):
        need(proof.get('verified') is True, 'generated module provenance rejected: '+proof.get('error', 'unverified'))
        need(baseline is not None and snapshot['sys_path'] == baseline+[proof['directory']], 'unexpected generated search-path delta')
        need(snapshot.get('mixed_modules') == {GENERATED_MODULE: proof['path']}
             and snapshot['mixed_module_paths'] == [proof['path']], 'unexpected generated module exception')
    else:
        need(all(type(p) is str and Path(p).is_absolute() and inside(p, prefix) for p in snapshot['sys_path']), 'mixed interpreter search paths')
        need(baseline is None or snapshot['sys_path'] == baseline, 'unexpected interpreter search-path delta')
        need(not snapshot['mixed_module_paths'], 'mixed loaded module paths')
    need(not snapshot['mixed_native_paths'], 'mixed loaded native paths')
    need(not any(p.get('version_error') for p in snapshot['packages'].values()), 'package version observation rejected')


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


def capture_snapshot(prefix, out, name, **provenance):
    snapshot = runtime_snapshot(prefix, RUNTIME_SOURCE, **provenance)
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
    contract, temp_stat, baseline = None, None, None
    temp_root = temp_root_for(prefix)
    try:
        stage = decode(pinned(Path(packet['output'])/'stage.json', args.stage_sha256, 16384))
        need(stage['staged_prefix'] == str(prefix) and stage['manifest_sha256'] == MANIFEST_SHA
             and stage['archive_sha256'] == ARCHIVE_SHA, 'worker stage binding differs')
        temp_stat = owned_stat(temp_root, directory=True, private=True)
        bound_temp = stage['temporary_root']
        need(bound_temp['path'] == str(temp_root)
             and all(temp_stat[k] == bound_temp['stat'][k] for k in ('dev', 'ino', 'uid', 'gid', 'mode')),
             'temporary root differs from pinned stage receipt')
        need(not inside(temp_root, prefix) and not inside(temp_root, RUNTIME_SOURCE)
             and all(os.environ.get(k) == str(temp_root) for k in ('TMPDIR', 'TMP', 'TEMP'))
             and tempfile.gettempdir() == str(temp_root) and not list(temp_root.iterdir()), 'exclusive child temporary root differs')
        write(out/'temporary-root.json', {'path': str(temp_root), 'stat': temp_stat, 'outside_runtime': True})
        contract = provenance_contract(prefix)
        write(out/'provenance-sources.json', contract)
        snapshot = capture_snapshot(prefix, out, 'before-imports', contract=contract, temp_root=temp_root, temp_stat=temp_stat)
        check_prefix(prefix, snapshot)
        baseline = list(snapshot['sys_path'])
        need(snapshot['python_version'] == ACTUAL['python'], 'actual Python version differs')
        event('cache_capture_start')
        capture_caches(prefix, stage['cache_source_sha256'], out)
        event('cache_reconciliation_complete')
        metadata = metadata_inventory()
        write(out/'metadata-before-imports.json', metadata)
        check_metadata(metadata, prefix)
        event('metadata_capture_complete')
        numpy, scipy = original_imports(event)
        snapshot = capture_snapshot(prefix, out, 'after-imports', contract=contract, temp_root=temp_root, temp_stat=temp_stat)
        check_prefix(prefix, snapshot, baseline)
        need(snapshot['generated_provenance']['verified'], 'known generated Torch module not verified')
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
            snapshot = capture_snapshot(prefix, out, 'final-runtime', contract=contract, temp_root=temp_root, temp_stat=temp_stat)
            check_prefix(prefix, snapshot, baseline)
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
                  'generated_module_verified': bool(snapshot and snapshot.get('generated_provenance', {}).get('verified')),
                  'error': None if error is None else type(error).__name__+': '+str(error),
                  'dependency_capability_checks_instrumented': False, 'source_only_execution': False})
    if error is not None: raise error


def validate_worker(out, packet_pin, stage_pin):
    directory = out/'worker'
    b = read(directory/'terminal.json', 128*1024); value = decode(b)
    need((directory/'terminal.candidate.json').read_bytes() == b and value['status'] == 'diagnostic_complete'
         and value['version'] == VERSION and value['generated_module_verified'] is True
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
        prefix = Path(stage['staged_prefix'])
        temp_root, temp_stat = create_child_temp(prefix)
        stage['temporary_root'] = {'path': str(temp_root), 'stat': temp_stat}
        write(out/'stage.json', stage); stage_pin = sha((out/'stage.json').read_bytes())
        event('stage_complete', staged_prefix=stage['staged_prefix'])
        event('temporary_root_ready', path=str(temp_root), stat=temp_stat)
        command = child_command(packet, args, prefix, stage_pin)
        env = dict(os.environ)
        for k in ('PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD'): env.pop(k, None)
        env.update(LD_LIBRARY_PATH=str(prefix/'lib'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TMPDIR=str(temp_root), TMP=str(temp_root), TEMP=str(temp_root))
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
