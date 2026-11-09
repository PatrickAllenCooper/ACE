"""Local argument assembly only. Never imports or runs a builder/backend/project.

Caller supplies independently accepted CPU audit/custody and fresh readiness pins.
This utility checks named metadata joins; it neither audits nor grants acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

KNOWN_SHA256 = '4a62de3c6505a1f07c1ee44f811fe457f1c940212d3cf6f74fa624a595b01d28'
FUTURE = ('cpu-acceptance', 'cpu-baseline', 'cpu-review', 'cpu-phase-terminal',
          'cpu-disposition', 'cpu-inventory', 'site-readiness')
STORAGE = ('receipt', 'seal', 'after', 'reference')
MIB = 1024**2


FULL_MAP_METADATA = frozenset(('cpu-inventory', 'cpu-disposition', 'cpu-acceptance', 'gpu-inventory', 'gpu-inventory-after', 'gpu-custody-inventory', 'gpu-disposition', 'gpu-acceptance'))

def metadata_maximum(name):
    return 128*1024**2 if name in FULL_MAP_METADATA else 32*1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate metadata key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite metadata value')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def capture(path, expected, limit=32*MIB):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'absolute ordinary file required')
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'explicit independent SHA256 required')
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
             'independent regular file exceeds the unchanged builder bound: '+str(path))
        data = stream.read(limit+1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    need(identity(before) == identity(after) == identity(path.stat()), 'input changed during capture')
    need(len(data) <= limit and sha(data) == expected, 'input SHA256 differs: '+str(path))
    return data


def write_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in FUTURE + tuple('storage-'+name for name in STORAGE):
        parser.add_argument('--'+name, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--cpu-remote-custody', required=True)
    parser.add_argument('--cpu-terminal-mirror', required=True)
    parser.add_argument('--python', required=True, help='absolute local builder interpreter; not executed here')
    parser.add_argument('--output-dir', required=True, help='new exclusive directory with existing parent')
    args = parser.parse_args()
    known = decode(capture(Path(__file__).resolve().with_name('known-inputs.json'), KNOWN_SHA256, MIB))
    need(known['schema'] == 'ace-gpu-cli-known-inputs-v1' and set(known['pending_metadata']) == set(FUTURE),
         'known input map differs')
    metadata = {name: {'path': v['path'], 'sha256': v['sha256']} for name, v in known['known_metadata'].items()}
    need(set(known['full_map_metadata']) == set(FULL_MAP_METADATA), 'known role bounds differ')
    for name, entry in metadata.items():
        capture(entry['path'], entry['sha256'], metadata_maximum(name))  # Opaque bytes; no scientific input decode.
    pending = {}
    for name in FUTURE:
        attribute = name.replace('-', '_')
        entry = dict(path=getattr(args, attribute), sha256=getattr(args, attribute+'_sha256'))
        pending[name] = capture(entry['path'], entry['sha256'], metadata_maximum(name))
        metadata[name] = entry
    need(set(metadata) == set(known['metadata_order']) and len(metadata) == 56, 'exact 56 metadata pairs required')
    need(pending['cpu-review'].strip() and pending['site-readiness'].strip(), 'explicit actual audit/readiness records required')

    # Only audit/accounting/custody metadata is decoded, never reports or inputs.
    acceptance, baseline, disposition = (decode(pending[name]) for name in ('cpu-acceptance', 'cpu-baseline', 'cpu-disposition'))
    need(acceptance['schema'] == 'ace-scientific-campaign-cpu-acceptance-v1'
         and acceptance['status'] == 'development_cpu_phase_accepted'
         and acceptance['job_id'] == known['cpu_job_id']
         and acceptance['cpu_packet_sha256'] == metadata['cpu-packet']['sha256']
         and acceptance['phase_terminal_sha256'] == metadata['cpu-phase-terminal']['sha256']
         and acceptance['scientific_qualified'] is False, 'explicit accepted actual CPU record required')
    need(baseline['terminal_jobs'] == 46 and baseline['submission_journals'] == 94
         and baseline['outstanding_reservation'] == 0
         and baseline['input_pins']['predecessor'] == metadata['training-baseline']['sha256']
         and baseline['input_pins']['acceptance'] == metadata['cpu-acceptance']['sha256']
         and baseline['input_pins']['disposition'] == metadata['cpu-disposition']['sha256'],
         'explicit accepted 46/94 accounting join required')
    need(disposition['job_id'] == known['cpu_job_id']
         and disposition['remote_root'] == known['cpu_remote_root']
         and disposition['packet_sha256'] == metadata['cpu-launch-packet']['sha256']
         and disposition['submission_result_sha256'] == metadata['cpu-result']['sha256']
         and disposition['custody_bytes_verified'] is True and disposition['current_lineage_verified'] is True
         and disposition['errors'] == [] and disposition['input_pin_mismatches'] == [],
         'successful final CPU terminal disposition required')

    terminal = Path(known['cpu_terminal_path'])
    remote = Path(args.cpu_remote_custody)
    mirror = Path(args.cpu_terminal_mirror)
    need(remote == terminal/'remote' and remote.is_dir() and not remote.is_symlink()
         and mirror.is_absolute() and mirror.is_dir() and not mirror.is_symlink()
         and terminal.resolve() != mirror.resolve()
         and terminal.resolve() not in mirror.resolve().parents and mirror.resolve() not in terminal.resolve().parents,
         'explicit terminal02 selected root and disjoint mirror required')
    for name, relative in {'cpu-disposition': 'disposition.json', 'cpu-inventory': 'inventory_transport.json',
        'cpu-phase-terminal': 'remote/launch/cpu/worker/artifacts/cpu-phase/terminal.json'}.items():
        need(Path(metadata[name]['path']).resolve() == (terminal/relative).resolve(), 'named terminal02 record differs: '+name)

    members = dict(known['storage_static'])
    for entry in members.values():
        capture(entry['path'], entry['sha256'])  # Backend/review source is NEVER imported.
    storage_data = {}
    filenames = dict(receipt='remote-custody.json', seal='custody-inventory.json',
                     after='inventory_after_transport.json', reference='predecessor-reference.json')
    for name in STORAGE:
        entry = dict(path=getattr(args, 'storage_'+name), sha256=getattr(args, 'storage_'+name+'_sha256'))
        need(Path(entry['path']).resolve() == (terminal/filenames[name]).resolve(), 'named storage record differs: '+name)
        storage_data[name] = capture(entry['path'], entry['sha256'], 128*MIB if name in ('receipt', 'seal', 'after') else 32*MIB)
        members[name] = entry
    receipt, seal = decode(storage_data['receipt']), decode(storage_data['seal'])
    need(receipt['helper_sha256'] == members['backend']['sha256'] and receipt['full_backup_verified'] is True
         and receipt['full_local_copy'] is False, 'successful4161 compact backup receipt required')
    need(set(seal['copies']) == {str(terminal.resolve()), str(mirror.resolve())}, 'explicit completed two-copy seal roots differ')
    config = raw(dict(schema='ace-compact-downstream-inputs-v1', predecessor='cpu', members=members))
    need(len(config) <= 128*1024, 'unchanged storage-config limit exceeded')

    builder_dir = Path(known['builder_directory'])
    for name, expected in known['builder_sources'].items():
        capture(builder_dir/name, expected, MIB)
    python = Path(args.python)
    need(python.is_absolute() and python.is_file() and os.access(python, os.X_OK), 'explicit local interpreter path required')
    output = Path(args.output_dir)
    need(output.is_absolute() and '..' not in output.parts and not os.path.lexists(output)
         and output.parent.is_dir() and output.parent == output.parent.resolve(), 'new canonical output directory required')
    argv = [args.python, '-I', '-B', str(builder_dir/'build_packet.py')]
    for name in known['metadata_order']:
        argv += ['--'+name, metadata[name]['path'], '--'+name+'-sha256', metadata[name]['sha256']]
    argv += ['--cpu-remote-custody', args.cpu_remote_custody, '--cpu-terminal-mirror', args.cpu_terminal_mirror,
             '--storage-config', str(output/'storage-config.json'), '--storage-config-sha256', sha(config)]
    need(argv[4::2] == known['required_builder_flags'], 'literal builder flag roster differs')
    command = raw(dict(schema='ace-gpu-cli-assembly-v1', argv=argv, cwd=known['repository'],
        metadata=metadata, storage_config=dict(path=str(output/'storage-config.json'), sha256=sha(config)),
        known_inputs_sha256=KNOWN_SHA256, builder_sources=known['builder_sources'],
        executed=False, scientific_qualified=False, independent_acceptance_performed_here=False))
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        write_new(output/'storage-config.json', config)
        write_new(output/'command.json', command)
    except BaseException as exc:
        try:
            write_new(output/'assembly-failure.json', raw(dict(type=type(exc).__name__, message=str(exc), retry_allowed=False)))
        except BaseException:
            pass
        raise
    print(raw(dict(command_json=str(output/'command.json'), command_sha256=sha(command),
                   storage_config_sha256=sha(config), executed=False)).decode(), end='')


if __name__ == '__main__':
    main()
