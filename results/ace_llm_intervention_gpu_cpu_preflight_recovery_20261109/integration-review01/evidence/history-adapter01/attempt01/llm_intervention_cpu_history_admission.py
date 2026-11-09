"""Pinned CPU history metadata for a controller-scoped campaign barrier.

No inventory, acceptance, runtime, response or world is produced here. The outer
controller must authenticate completed CPU pre-custody and require CPU post-
custody before acceptance. Only unused historical payload reads are deferred.
The unchanged finalizer must run outside this context.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import types
from types import MappingProxyType

SCHEMA = 'ace-scientific-cpu-history-readset-v1'
CAMPAIGN_SOURCE_SHA256 = '9a2717dd68c37ddde6d7fb63c6cc519c067f7b1d457d4f49f919c18342424bbe'
VERSION = 'ace-supplied-plan-campaign-v1'
CPU_IDS = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker')
GPU_IDS = ('frozen_llm', 'sft_llm', 'dpo_llm')
WORLD_IDS = tuple(range(202611070100, 202611070124))
MAX_READSET = 16 * 1024**2
MAX_ACCEPTANCE = 128 * 1024**2
MAX_BUNDLE = 32 * 1024**2
MAX_SOURCE = 1024**2
CAMPAIGN_SUFFIX = 'worker/artifacts/campaign'
BUNDLE_PATH = 'worker/artifacts/cpu-phase/campaign-bundle.json'
HEADER_FILES = ('campaign.json', 'source-plan-pins.json', 'start.json')
MOUNT_LOGICAL = Path('/scratch/alpine')
MOUNT_PHYSICAL = Path('/gpfs/alpine1/scratch')


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _keys(value, keys):
    _need(type(value) is dict and set(value) == set(keys), 'exact metadata fields required')


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _need(key not in out, 'duplicate metadata key')
            out[key] = value
        return out
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    _canonical(value)  # Includes rejection of NaN and overflowing numeric literals.
    return value


def _capture(raw, pin, maximum):
    _need(type(raw) is bytes and len(raw) <= maximum and _pin(pin) and _sha(raw) == pin,
          'independently pinned bounded captured bytes required')
    return raw


def _relative(name):
    _need(type(name) is str and name and '\x00' not in name and '\\' not in name,
          'plain relative member required')
    p = PurePosixPath(name)
    _need(not p.is_absolute() and str(p) == name and all(x not in ('.', '..') for x in p.parts),
          'normalized relative member required')
    return p.parts


def _absolute(value):
    _need(type(value) is str and '\x00' not in value and Path(value).is_absolute()
          and str(Path(value)) == value and '..' not in Path(value).parts,
          'normalized absolute root required')
    return Path(value)


def _source_pins(c):
    return [{'world_id': s['plan']['world']['world_id'], 'sha256': s['plan_sha256']} for s in c['sources']]


def _roster():
    names = set(HEADER_FILES)
    for i in range(5):
        names.update(f'policy-{i:02d}/{n}' for n in ('attempt.json', 'attempt.candidate.json'))
        names.update(f'policy-{i:02d}/world-{j:02d}/arm-receipt.json' for j in range(24))
    return names


def _historical_attempt(c, campaign_pin, pid, expected, files, accepted_evidence):
    """Original _load_attempt predicates, with captured reads and no tree scan.

    CPU-only service absence and the controller's wholly-successful condition
    specialize the original branches. Evidence hashes equal the accepted full
    map; checking those payload bytes is explicitly an external pre/post gate.
    """
    base = f'policy-{CPU_IDS.index(pid):02d}/'
    raw = files[base+'attempt.json']
    _need(_pin(expected) and _sha(raw) == expected, 'independent attempt receipt pin differs')
    r = _decode(raw)
    _keys(r, ('version', 'campaign_sha256', 'source_plan_pins', 'policy_id', 'phase', 'arms',
              'service', 'errors', 'evidence_sha256', 'retry_allowed', 'private_scores_computed'))
    _need(r['version'] == VERSION and r['campaign_sha256'] == campaign_pin and r['policy_id'] == pid
          and r['phase'] == c['policy_phases'][pid] and r['source_plan_pins'] == _source_pins(c)
          and r['retry_allowed'] is False and r['private_scores_computed'] is False, 'attempt identity differs')
    _need(files[base+'attempt.candidate.json'] == raw
          and r['evidence_sha256'] == dict(accepted_evidence), 'accepted attempt custody differs')
    _need(type(r['arms']) is list and len(r['arms']) == len(c['sources']), 'complete campaign arm inventory required')
    for i, (arm, source) in enumerate(zip(r['arms'], c['sources'])):
        _keys(arm, ('world_id', 'status', 'path', 'receipt_sha256', 'reason'))
        _need(type(arm['world_id']) is int and arm['world_id'] == source['plan']['world']['world_id']
              and arm['status'] in ('completed', 'failed', 'unstarted'), 'campaign arm identity/status differs')
        _need(arm['reason'] is None if arm['status'] == 'completed' else
              type(arm['reason']) is str and 0 < len(arm['reason']) <= 256, 'arm disposition reason required')
        if arm['path'] is None:
            _need(arm['receipt_sha256'] is None and arm['status'] != 'completed', 'missing arm has no success/pin')
        else:
            _need(arm['status'] != 'unstarted' and arm['path'] == f'world-{i:02d}'
                  and _pin(arm['receipt_sha256']), 'arm path/pin differs')
            arm_raw = files[base+arm['path']+'/arm-receipt.json']
            _need(_sha(arm_raw) == arm['receipt_sha256'], 'arm receipt pin differs')
            ar = _decode(arm_raw)
            _need(ar['plan_sha256'] == source['plan_sha256'] and ar['policy_id'] == pid, 'arm source identity differs')
    _keys(r['service'], ('status', 'file', 'sha256'))
    _need(pid not in c['services'] and r['service'] == dict(status='not_started', file=None, sha256=None),
          'CPU task_process has unexpected service')
    _need(not r['errors'] and all(a['status'] == 'completed' for a in r['arms']),
          'CPU attempts not wholly successful')
    return r, None


class _Live:
    """Fixed metadata only, opened through no-follow directory descriptors."""
    def __init__(self, root, files):
        self.logical = _absolute(root)
        self.physical = (MOUNT_PHYSICAL/self.logical.relative_to(MOUNT_LOGICAL)
                         if MOUNT_LOGICAL in self.logical.parents else self.logical)
        self.identities = {}
        self.files = files
        self.active = True

    def _identity(self, path, st):
        _need(stat.S_ISDIR(st.st_mode), 'ordinary directory required')
        key = str(path)
        identity = (st.st_dev, st.st_ino)
        _need(self.identities.setdefault(key, identity) == identity, 'directory or parent identity changed')

    def _open_dir(self, path):
        _need(self.active, 'retired history callback')
        if self.physical != self.logical:
            _need(MOUNT_LOGICAL.resolve(strict=True) == MOUNT_PHYSICAL
                  and self.logical.resolve(strict=True) == self.physical, 'fixed CURC mount alias changed')
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open('/', flags)
        current = Path('/')
        try:
            self._identity(current, os.fstat(fd))
            for part in path.parts[1:]:
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
                current = current/part
                self._identity(current, os.fstat(fd))
            return fd
        except BaseException:
            os.close(fd)
            raise

    def read(self, rel):
        parts = _relative(rel)
        expected = self.files[rel]
        parent = self.physical.joinpath(*parts[:-1])
        directory = self._open_dir(parent)
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, 'rb') as handle:
                before = os.fstat(handle.fileno())
                _need(stat.S_ISREG(before.st_mode) and before.st_size == len(expected), 'bounded ordinary metadata file required')
                raw = handle.read(len(expected)+1)
                after = os.fstat(handle.fileno())
                live = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            _need(identity(before) == identity(after) == identity(live) and raw == expected,
                  'live metadata changed or differs from accepted captured bytes')
        finally:
            os.close(directory)
        check = self._open_dir(parent)
        os.close(check)
        return raw

    def guards(self):
        fd = self._open_dir(self.physical)
        try:
            for name in ('finalization-intent.json', 'finalization.json', 'finalization.candidate.json', 'private'):
                try:
                    os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ValueError('history adapter cannot cover finalization')
        finally:
            os.close(fd)
        for i in range(5):
            fd = self._open_dir(self.physical/f'policy-{i:02d}')
            try:
                try:
                    os.stat('service', dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ValueError('CPU service appeared')
            finally:
                os.close(fd)
        for name in HEADER_FILES:
            self.read(name)


def _verify_original(module, source):
    _need(type(module) is types.ModuleType and type(module.__file__) is str, 'captured campaign module required')
    compiled = compile(source, module.__file__, 'exec', dont_inherit=True, optimize=0)
    expected = {c.co_name: c for c in compiled.co_consts if isinstance(c, types.CodeType)}
    for name in ('_load_attempt', '_inventory', '_closure', '_need', '_keys', '_decode', '_source_pins'):
        function = getattr(module, name, None)
        _need(type(function) is types.FunctionType and function.__globals__ is module.__dict__
              and function.__module__ == module.__name__ and function.__name__ == name
              and function.__closure__ is None and function.__defaults__ is None
              and function.__kwdefaults__ is None and function.__code__ == expected.get(name)
              and function.__code__.co_filename == module.__file__, 'original campaign callable differs: '+name)
    _need(module.VERSION == VERSION, 'campaign version differs')
    return module._load_attempt


@dataclass(frozen=True, slots=True)
class _History:
    acceptance_sha256: str
    bundle_sha256: str
    readset_sha256: str
    campaign_source_sha256: str
    campaign_root: str
    campaign_sha256: str
    _campaign: bytes
    _files: object
    _attempt_pins: object
    _evidence: object
    _source: bytes

    @contextmanager
    def install(self, campaign_module):
        """Controller scope only. No global inventory patch; always restore."""
        original = _verify_original(campaign_module, self._source)
        live = _Live(self.campaign_root, self._files)
        try:
            live.guards()
            # Establish fixed parent identities before any policy is started.
            for rel in self._files:
                fd = live._open_dir(live.physical/Path(rel).parent)
                os.close(fd)

            def admitted(c, campaign_pin, path, pid, expected):
                _need(live.active and campaign_module._load_attempt is admitted, 'inactive or replaced history callback')
                _need(type(pid) is str and pid in CPU_IDS+GPU_IDS, 'policy outside fixed roster')
                _need(campaign_pin == self.campaign_sha256 and _canonical(c) == self._campaign, 'campaign identity differs')
                wanted = Path(self.campaign_root)/f'policy-{(CPU_IDS+GPU_IDS).index(pid):02d}'
                _need(isinstance(path, (str, Path)) and str(path) == str(wanted), 'attempt path differs')
                _need(_pin(expected), 'independent attempt pin required')
                if pid in CPU_IDS:
                    _need(expected == self._attempt_pins[pid], 'CPU expected pin differs from acceptance')
                live.guards()
                if pid in GPU_IDS:
                    fd = live._open_dir(live.physical/wanted.name)
                    os.close(fd)
                    return original(c, campaign_pin, path, pid, expected)
                prefix = wanted.name+'/'
                captured = {n: live.read(n) for n in self._files if n.startswith(prefix)}
                result = _historical_attempt(c, campaign_pin, pid, expected, captured, self._evidence[pid])
                live.guards()
                return result

            campaign_module._load_attempt = admitted
            try:
                yield self
            finally:
                campaign_module._load_attempt = original
        finally:
            live.active = False


def capture_history(*, acceptance_raw, acceptance_sha256, bundle_raw, bundle_sha256,
                    readset_raw, readset_sha256, campaign_source_raw, campaign_source_sha256):
    """Authenticate a detached 133-file readset; perform no live reads or writes.

    CPU pre/post-custody and phase-terminal admission are the outer caller's
    gates. Pins are caller inputs, never inferred from live historical receipts.
    """
    a = _decode(_capture(acceptance_raw, acceptance_sha256, MAX_ACCEPTANCE))
    b = _decode(_capture(bundle_raw, bundle_sha256, MAX_BUNDLE))
    r = _decode(_capture(readset_raw, readset_sha256, MAX_READSET))
    source = _capture(campaign_source_raw, campaign_source_sha256, MAX_SOURCE)
    _need(campaign_source_sha256 == CAMPAIGN_SOURCE_SHA256, 'unsupported original campaign source')
    _keys(a, ('schema', 'status', 'cpu_packet_sha256', 'job_id', 'allocation', 'cpu_output', 'campaign_root',
              'cpu_output_files', 'cpu_output_excluded_directory', 'campaign_files', 'bundle_path', 'bundle_sha256',
              'phase_terminal_path', 'phase_terminal_sha256', 'attempt_pins', 'scientific_qualified'))
    _need(a['schema'] == 'ace-scientific-campaign-cpu-acceptance-v1'
          and a['status'] == 'development_cpu_phase_accepted' and a['scientific_qualified'] is False,
          'accepted CPU campaign record required')
    _need(_pin(a['cpu_packet_sha256']) and type(a['job_id']) is str and a['job_id'].isascii()
          and a['job_id'].isdigit() and int(a['job_id']) > 0
          and a['allocation'] == dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal',
              cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'), 'accepted CPU allocation differs')
    root = _absolute(a['campaign_root'])
    _need(root == _absolute(a['cpu_output'])/CAMPAIGN_SUFFIX
          and a['cpu_output_excluded_directory'] == CAMPAIGN_SUFFIX, 'accepted CPU roots differ')
    _need(a['bundle_path'] == BUNDLE_PATH and a['bundle_sha256'] == bundle_sha256
          and a['cpu_output_files'][BUNDLE_PATH] == bundle_sha256, 'accepted bundle pin differs')
    _need(a['phase_terminal_path'] == 'worker/artifacts/cpu-phase/terminal.json'
          and _pin(a['phase_terminal_sha256'])
          and a['cpu_output_files'][a['phase_terminal_path']] == a['phase_terminal_sha256'],
          'accepted phase terminal pin differs')
    _keys(r, ('schema', 'cpu_acceptance_sha256', 'campaign_root', 'files'))
    _need(r['schema'] == SCHEMA and r['cpu_acceptance_sha256'] == acceptance_sha256
          and r['campaign_root'] == str(root), 'readset acceptance/root binding differs')
    _keys(r['files'], _roster())
    _need(type(a['campaign_files']) is dict, 'accepted full campaign map required')
    for name, pin in a['campaign_files'].items():
        _relative(name)
        _need(_pin(pin), 'accepted campaign member pin differs')
        _need(name in HEADER_FILES or name.split('/')[0] in {f'policy-{i:02d}' for i in range(5)},
              'unexpected accepted campaign subtree')
    files = {}
    for name, record in r['files'].items():
        _keys(record, ('sha256', 'bytes', 'utf8'))
        _need(type(record['utf8']) is str and type(record['bytes']) is int
              and 0 <= record['bytes'] <= MAX_READSET, 'bounded UTF8 metadata record required')
        raw = record['utf8'].encode('utf-8')
        _need(len(raw) == record['bytes'] and _pin(record['sha256'])
              and _sha(raw) == record['sha256'] == a['campaign_files'].get(name), 'readset member pin/length differs')
        files[name] = raw
    _need(sum(map(len, files.values())) <= MAX_READSET, 'readset aggregate bound exceeded')
    c = b['campaign']; campaign_pin = b['campaign_sha256']
    _need(_pin(campaign_pin) and _sha(_canonical(c)) == campaign_pin and b['scope'] == c['scope'] == 'development'
          and c['version'] == VERSION and c['policy_ids'] == list(CPU_IDS+GPU_IDS)
          and c['policy_phases'] == {p: 'cpu' if p in CPU_IDS else 'gpu' for p in CPU_IDS+GPU_IDS}
          and set(c['services']) == set(GPU_IDS), 'fixed campaign/policy identity differs')
    _need(type(c['sources']) is list and len(c['sources']) == 24
          and [s['plan']['world']['world_id'] for s in c['sources']] == list(WORLD_IDS)
          and all(type(s['plan']['world']['world_id']) is int and _pin(s['plan_sha256'])
                  and _sha(_canonical(s['plan'])) == s['plan_sha256'] for s in c['sources'])
          and b['source_plan_pins'] == _source_pins(c), 'fixed source-plan identity differs')
    _need(files['campaign.json'] == _canonical(c)+b'\n'
          and files['source-plan-pins.json'] == _canonical(b['source_plan_pins'])+b'\n', 'campaign metadata differs')
    start = _decode(files['start.json'])
    _keys(start, ('utc', 'campaign_sha256', 'retry_allowed'))
    _need(type(start['utc']) is str and 0 < len(start['utc']) <= 128
          and start['campaign_sha256'] == campaign_pin and start['retry_allowed'] is False, 'campaign start differs')
    _keys(a['attempt_pins'], CPU_IDS)
    evidence = {}
    for i, pid in enumerate(CPU_IDS):
        prefix = f'policy-{i:02d}/'
        pin = a['attempt_pins'][pid]
        _need(_pin(pin) and all(a['campaign_files'].get(prefix+n) == pin
              for n in ('attempt.json', 'attempt.candidate.json')), 'accepted attempt/candidate pin differs')
        evidence[pid] = MappingProxyType({n[len(prefix):]: h for n, h in a['campaign_files'].items()
                                        if n.startswith(prefix) and n[len(prefix):] not in ('attempt.json', 'attempt.candidate.json')})
        _historical_attempt(c, campaign_pin, pid, pin, files, evidence[pid])
    return _History(acceptance_sha256, bundle_sha256, readset_sha256, campaign_source_sha256,
                    str(root), campaign_pin, _canonical(c), MappingProxyType(files),
                    MappingProxyType(dict(a['attempt_pins'])), MappingProxyType(evidence), source)
