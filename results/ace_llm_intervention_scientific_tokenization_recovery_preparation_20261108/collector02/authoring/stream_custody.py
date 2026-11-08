#!/usr/bin/env python3
"""Prospective transport only. No SSH/site/science policy; caller retains all gates.

emit: read a pinned JSON request from stdin; emit USTAR to stdout, receipt to stderr.
capture: run ONE explicit caller-pinned transport command, retaining every byte.
materialize: authenticate successful capture + pinned before/after inventories,
then create two exclusive independent byte copies. Never extractall or retry.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tarfile
import time

CHUNK = 1024 * 1024
MAX_FILES = 262144
MAX_BYTES = 8 * 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_METADATA = 128 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def decode(data):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('nonfinite JSON')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def name_ok(name):
    need(isinstance(name, str) and name and len(name.encode('utf-8')) <= 255,
         'bounded UTF-8 member name required')
    need(not name.startswith('/') and '\\' not in name
         and all(ord(c) >= 32 and ord(c) != 127 for c in name), 'unsafe member name')
    need(all(p not in ('', '.', '..') for p in name.split('/')), 'unsafe path component')
    return name


def normalized(value):
    need(type(value) is dict and type(value.get('files')) is dict
         and type(value.get('directories')) is list, 'inventory shape')
    files, dirs = value['files'], value['directories']
    need(len(files) + len(dirs) <= MAX_FILES, 'member count bound')
    dir_set = set(dirs)
    need(len(dir_set) == len(dirs), 'duplicate directory')
    need(not set(files).intersection(dirs), 'file/directory collision')
    for n in list(files) + dirs:
        name_ok(n)
        for i in range(1, len(n.split('/'))):
            need('/'.join(n.split('/')[:i]) in dir_set, 'missing parent directory')
    for n, v in files.items():
        need(type(v) is dict and set(v) == {'bytes', 'sha256'}, 'file metadata fields')
        need(type(v['bytes']) is int and 0 <= v['bytes'] <= MAX_FILE_BYTES, 'file byte bound')
        need(type(v['sha256']) is str and re.fullmatch('[0-9a-f]{64}', v['sha256']), 'file pin')
    need(sum(v['bytes'] for v in files.values()) <= MAX_BYTES, 'total byte bound')
    return dict(files=files, directories=sorted(dirs))


def request_check(request):
    need(type(request) is dict and set(request) == {'roots', 'single_files', 'inventory'}, 'request keys')
    need(type(request['roots']) is dict and type(request['single_files']) is dict, 'source maps')
    labels = list(request['roots']) + list(request['single_files'])
    need(len(labels) == len(set(labels)) and labels, 'unique source labels required')
    for label in labels:
        need('/' not in name_ok(label), 'top-level source label required')
    for path in list(request['roots'].values()) + list(request['single_files'].values()):
        need(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path,
             'absolute canonical spelling required')
        need('..' not in Path(path).parts, 'source traversal')
    expected = normalized(request['inventory'])
    need(set(request['roots']) <= set(expected['directories']), 'missing root directories')
    need(set(request['single_files']) <= set(expected['files']), 'missing single files')
    need({n.split('/')[0] for n in list(expected['files']) + expected['directories']} == set(labels),
         'inventory/source label mismatch')
    for n in expected['directories']:
        header(n)
    for n, v in expected['files'].items():
        header(n, v['bytes'])
    return expected


def identity(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


@contextlib.contextmanager
def plain_open(path, directory=False):
    """Reject source/target symlinks at EVERY path component; no root alias guessing."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute safe path required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            final = index == len(path.parts[1:]) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if not final or directory:
                flags |= os.O_DIRECTORY
            nxt = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = nxt
        need(stat.S_ISDIR(os.fstat(fd).st_mode) if directory else stat.S_ISREG(os.fstat(fd).st_mode),
             'ordinary directory/file required')
        with os.fdopen(os.dup(fd), 'rb') if not directory else contextlib.nullcontext(fd) as opened:
            yield opened
    finally:
        os.close(fd)


def fingerprint(path, independent=False, max_bytes=None):
    with plain_open(path) as f:
        before = os.fstat(f.fileno())
        need(max_bytes is None or before.st_size <= max_bytes, 'file exceeds read bound')
        need(not independent or before.st_nlink == 1, 'independent regular bytes required')
        h = hashlib.sha256()
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
        need(identity(before) == identity(os.fstat(f.fileno())) == identity(Path(path).lstat()),
             'file changed during read')
    return dict(bytes=before.st_size, sha256=h.hexdigest()), identity(before)


def read_pinned(path, pin):
    meta, _ = fingerprint(Path(path), independent=True, max_bytes=MAX_METADATA)
    need(meta['bytes'] <= MAX_METADATA and meta['sha256'] == pin, 'pinned metadata mismatch')
    with plain_open(path) as f:
        data = f.read(MAX_METADATA + 1)
    need(sha(data) == pin, 'metadata changed')
    return data


def scan(request):
    expected = request_check(request)
    files, dirs, identities, paths = {}, [], {}, {}
    expected_dirs = set(expected['directories'])
    def walk(path, name):
        with plain_open(path, directory=True) as fd:
            before = identity(os.fstat(fd))
            need(name in expected_dirs, 'unexpected source directory')
            dirs.append(name)
            identities[name] = before
            for child in sorted(os.listdir(fd)):
                n = name_ok(name + '/' + child)
                p = Path(path) / child
                s = os.stat(child, dir_fd=fd, follow_symlinks=False)
                need(not stat.S_ISLNK(s.st_mode), 'source symlink forbidden')
                if stat.S_ISDIR(s.st_mode):
                    walk(p, n)
                else:
                    need(stat.S_ISREG(s.st_mode), 'source special file forbidden')
                    need(n in expected['files'], 'unexpected source file')
                    files[n], identities[n] = fingerprint(p, max_bytes=MAX_FILE_BYTES)
                    paths[n] = p
                need(len(files) + len(dirs) <= MAX_FILES, 'source member bound')
            need(before == identity(os.fstat(fd)) == identity(Path(path).lstat()), 'directory changed')
    for label, path in sorted(request['roots'].items()):
        walk(Path(path), label)
    for label, path in sorted(request['single_files'].items()):
        files[label], identities[label] = fingerprint(Path(path), max_bytes=MAX_FILE_BYTES)
        paths[label] = Path(path)
    actual = normalized(dict(files=files, directories=dirs))
    need(actual == expected, 'source inventory differs')
    return actual, identities, paths


class DigestReader:
    def __init__(self, source):
        self.source, self.digest, self.count = source, hashlib.sha256(), 0
    def read(self, size):
        data = self.source.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def header(name, size=None):
    info = tarfile.TarInfo(name + '/' if size is None else name)
    info.type = tarfile.DIRTYPE if size is None else tarfile.REGTYPE
    info.mode = 0o700 if size is None else 0o600
    info.size = 0 if size is None else size
    # Deliberately USTAR only: no PAX/GNU extension ambiguity or hidden members.
    info.tobuf(format=tarfile.USTAR_FORMAT, encoding='utf-8', errors='strict')
    return info


def emit(request, stream):
    expected, before, paths = scan(request)
    headers = [header(n) for n in expected['directories']]
    headers += [header(n, v['bytes']) for n, v in sorted(expected['files'].items())]
    # Preflight every header before writing any stream bytes (bounded USTAR names).
    with tarfile.open(fileobj=stream, mode='w|', format=tarfile.USTAR_FORMAT,
                      encoding='utf-8', errors='strict') as archive:
        for info in headers:
            if info.isdir():
                archive.addfile(info)
                continue
            with plain_open(paths[info.name]) as f:
                need(identity(os.fstat(f.fileno())) == before[info.name], 'source changed before emit')
                reader = DigestReader(f)
                archive.addfile(info, reader)
                need(dict(bytes=reader.count, sha256=reader.digest.hexdigest()) == expected['files'][info.name],
                     'emitted file differs')
                need(identity(os.fstat(f.fileno())) == before[info.name], 'source changed while emitting')
        after_inventory, after, _ = scan(request)
        need(after_inventory == expected and after == before, 'source changed across stream')
    links = {}
    for n in paths:
        s = before[n]
        if s[-1] > 1:
            links.setdefault((s[0], s[1], s[-1]), []).append(n)
    return dict(source_unchanged=True, inventory_sha256=sha(raw(expected)),
                original_hardlink_groups=sorted([dict(paths=sorted(v), link_count=k[2])
                    for k, v in links.items()], key=lambda x: x['paths']), acceptance=False)


def archive_bound(expected):
    # Include both end blocks before rounding to tarfile's complete output record.
    members = 512 * (len(expected['directories']) + len(expected['files'])) + sum(
        ((v['bytes'] + 511) // 512) * 512 for v in expected['files'].values())
    return ((members + 1024 + 10239) // 10240) * 10240


def close_owned_group(proc):
    """Close the local start_new_session group even if its leader has exited."""
    result = dict(process_group=proc.pid, signal_sent=False, group_absent=False,
                  leader_reaped=False, errors=[], probe_observations=[])
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        result['signal_sent'] = True
    except ProcessLookupError:
        result['group_absent'] = True
    except BaseException as exc:
        result['errors'].append(dict(operation='signal_group', type=type(exc).__name__, message=str(exc)))
    try:
        proc.wait(timeout=10)
        result['leader_reaped'] = True
    except BaseException as exc:
        result['errors'].append(dict(operation='reap_leader', type=type(exc).__name__, message=str(exc)))
    deadline = time.monotonic() + 2
    while not result['group_absent']:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            result['group_absent'] = True
            break
        except PermissionError as exc:
            # Darwin may transiently return EPERM while killed orphans are reaped.
            # Preserve that uncertainty and require a later positive ESRCH result.
            result['probe_observations'].append(dict(type=type(exc).__name__, message=str(exc)))
        except BaseException as exc:
            result['errors'].append(dict(operation='probe_group', type=type(exc).__name__, message=str(exc)))
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    try:
        proc.stdout.close()
    except BaseException as exc:
        result['errors'].append(dict(operation='close_stdout', type=type(exc).__name__, message=str(exc)))
    result['verified'] = result['group_absent'] and result['leader_reaped'] and not result['errors']
    return result


def exclusive_write(path, data):
    with Path(path).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def capture(command, request_bytes, attempt, timeout):
    """One explicit command only. Caller owns SSH auth/locks/full terminal gates."""
    request = decode(request_bytes)
    expected = request_check(request)
    need(type(command) is list and command and all(type(x) is str and x and '\x00' not in x for x in command),
         'explicit argv required; no shell')
    need(type(timeout) is int and 1 <= timeout <= 7200, 'bounded transport deadline required')
    attempt = Path(attempt)
    with plain_open(attempt.parent, directory=True):
        attempt.mkdir(mode=0o700)  # Exclusive; never resume or erase a failed attempt.
    exclusive_write(attempt/'request.json', request_bytes)
    exclusive_write(attempt/'command.json', raw(command))
    proc = None
    started = time.monotonic()
    receipt = dict(status='failed', acceptance=False, request_sha256=sha(request_bytes),
                   command_sha256=sha(raw(command)), returncode=None)
    try:
        with (attempt/'request.json').open('rb') as inp, (attempt/'archive.tar').open('xb') as out, \
                (attempt/'stderr.log').open('xb') as err, selectors.DefaultSelector() as selector:
            proc = subprocess.Popen(command, stdin=inp, stdout=subprocess.PIPE, stderr=err,
                                    start_new_session=True, shell=False)
            selector.register(proc.stdout, selectors.EVENT_READ)
            count = 0
            while selector.get_map():
                need(time.monotonic() - started < timeout, 'transport deadline exceeded')
                need(err.tell() <= MAX_METADATA, 'transport stderr bound exceeded')
                for key, _ in selector.select(0.1):
                    block = os.read(key.fd, CHUNK)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        # Retain the over-bound chunk as failed transport evidence, then stop.
                        out.write(block)
                        count += len(block)
                        need(count <= archive_bound(expected), 'archive byte bound exceeded')
            receipt['returncode'] = proc.wait(timeout=max(0.001, timeout-(time.monotonic()-started)))
            out.flush(); os.fsync(out.fileno())
            err.flush(); os.fsync(err.fileno())
            need(receipt['returncode'] == 0, 'producer/transport returned nonzero')
        receipt['status'] = 'captured_pending_inventory_verification'
    except BaseException as exc:
        receipt['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        original_error = 'error' in receipt
        if proc is not None:
            receipt['cleanup'] = close_owned_group(proc)
            receipt['returncode'] = proc.returncode
            if not receipt['cleanup']['verified']:
                receipt['status'] = 'failed'
                receipt.setdefault('error', dict(type='RuntimeError', message='owned local transport cleanup unverified'))
        receipt['elapsed_seconds'] = time.monotonic() - started
        for name in ('archive.tar', 'stderr.log'):
            if (attempt/name).exists():
                try:
                    receipt[name] = fingerprint(attempt/name, independent=True)[0]
                except BaseException as exc:
                    receipt['status'] = 'failed'
                    receipt.setdefault('evidence_errors', []).append(dict(path=name, type=type(exc).__name__, message=str(exc)))
                    receipt.setdefault('error', dict(type='RuntimeError', message='transport evidence readback failed'))
        exclusive_write(attempt/'capture.json', raw(receipt))
        if not original_error and receipt['status'] == 'failed':
            raise RuntimeError(receipt['error']['message'])
    return receipt


def exact_read(source, size):
    data = source.read(size)
    need(len(data) == size, 'truncated archive')
    return data


def extract_archive(archive_path, expected, destination):
    """Parse raw 512-byte USTAR records; do NOT let tarfile hide extension members."""
    expected = normalized(expected)
    destination = Path(destination)
    with plain_open(destination.parent, directory=True):
        destination.mkdir(mode=0o700)
    expected_dirs = set(expected['directories'])
    seen, actual_files, actual_dirs = set(), {}, []
    with plain_open(archive_path) as source:
        need(os.fstat(source.fileno()).st_size <= archive_bound(expected), 'archive too large')
        while True:
            block = exact_read(source, 512)
            if block == b'\0' * 512:
                need(exact_read(source, 512) == b'\0' * 512, 'missing second end block')
                remainder = source.read(10241)
                need(len(remainder) <= 10240 and len(remainder) % 512 == 0
                     and not remainder.strip(b'\0') and not source.read(1), 'nonzero/overlong trailing archive')
                break
            need(block[257:263] == b'ustar\0' and block[263:265] == b'00', 'USTAR only')
            info = tarfile.TarInfo.frombuf(block, encoding='utf-8', errors='strict')
            raw_name = block[:100].split(b'\0', 1)[0].decode('utf-8', 'strict')
            prefix = block[345:500].split(b'\0', 1)[0].decode('utf-8', 'strict')
            original = prefix + '/' + raw_name if prefix else raw_name
            need(original == info.name + ('/' if info.type == tarfile.DIRTYPE else ''),
                 'noncanonical archive spelling')
            need(info.type in (tarfile.REGTYPE, tarfile.DIRTYPE), 'links/extensions/devices forbidden')
            need(not info.linkname and info.devmajor == info.devminor == 0, 'link/device metadata forbidden')
            # frombuf strips directory trailing slash. Reject every other normalization.
            n = name_ok(info.name)
            need(n not in seen, 'duplicate archive name')
            seen.add(n)
            target = destination / n
            if info.type == tarfile.DIRTYPE:
                need(n in expected_dirs and info.size == 0, 'unexpected directory')
                target.mkdir(mode=0o700)
                actual_dirs.append(n)
            else:
                need(n in expected['files'] and info.size == expected['files'][n]['bytes'], 'unexpected file/size')
                h = hashlib.sha256()
                left = info.size
                with target.open('xb') as out:
                    while left:
                        data = exact_read(source, min(left, CHUNK))
                        out.write(data); h.update(data); left -= len(data)
                    out.flush(); os.fsync(out.fileno())
                need(exact_read(source, (-info.size) % 512) == b'\0' * ((-info.size) % 512), 'nonzero file padding')
                actual_files[n] = dict(bytes=info.size, sha256=h.hexdigest())
                need(actual_files[n] == expected['files'][n], 'file digest differs')
    need(normalized(dict(files=actual_files, directories=actual_dirs)) == expected, 'archive membership differs')
    # Read the actual new tree back; independent inodes, not just in-stream hash.
    return local_readback(destination, expected)


def local_readback(destination, expected):
    destination = Path(destination)
    local_request = dict(roots={n: str(destination/n) for n in expected['directories'] if '/' not in n},
                         single_files={n: str(destination/n) for n in expected['files'] if '/' not in n}, inventory=expected)
    labels = set(local_request['roots']) | set(local_request['single_files'])
    with plain_open(destination, directory=True) as fd:
        before = identity(os.fstat(fd))
        need(set(os.listdir(fd)) == labels, 'unexpected local top-level membership')
        actual, identities, _ = scan(local_request)
        need(set(os.listdir(fd)) == labels, 'local top-level membership changed')
        need(before == identity(os.fstat(fd)) == identity(destination.lstat()), 'local destination changed')
    need(all(identities[n][-1] == 1 for n in expected['files']), 'local hardlink forbidden')
    return actual, identities


def materialize(attempt, result_pin, before_path, before_pin, after_path, after_pin, copy_a, copy_b):
    attempt = Path(attempt)
    result = decode(read_pinned(attempt/'capture.json', result_pin))
    need(result['status'] == 'captured_pending_inventory_verification' and result['returncode'] == 0,
         'successful producer AND transport required')
    request = decode(read_pinned(attempt/'request.json', result['request_sha256']))
    read_pinned(attempt/'command.json', result['command_sha256'])
    expected = request_check(request)
    before = decode(read_pinned(before_path, before_pin))
    after = decode(read_pinned(after_path, after_pin))
    need(expected == normalized(before) == normalized(after), 'pinned before/after inventory differs')
    need(before.get('original_hardlink_groups') == after.get('original_hardlink_groups'), 'source link topology differs')
    archive = attempt/'archive.tar'
    pin_before = fingerprint(archive, independent=True)[0]
    need(pin_before == result['archive.tar'], 'captured archive changed')
    need(fingerprint(attempt/'stderr.log', independent=True)[0] == result['stderr.log'], 'producer stderr changed')
    # The process may be ssh. Require the emitter's own terminal receipt as well as exit zero.
    with plain_open(attempt/'stderr.log') as f:
        lines = f.read(MAX_METADATA + 1).splitlines()
    need(lines, 'missing emitter receipt')
    emitted = decode(lines[-1])
    need(emitted.get('source_unchanged') is True and emitted.get('inventory_sha256') == sha(raw(expected)),
         'missing producer source-unchanged proof')
    if 'original_hardlink_groups' in before:
        need(emitted['original_hardlink_groups'] == before['original_hardlink_groups'], 'producer link observation differs')
    a, b = Path(copy_a), Path(copy_b)
    need(a.is_absolute() and b.is_absolute() and a != b and a not in b.parents and b not in a.parents,
         'separate exclusive copy roots required')
    exclusive_write(attempt/'materialization.intent.json', raw(dict(capture_sha256=result_pin,
        inventory_before_sha256=before_pin, inventory_after_sha256=after_pin, copies=[str(a), str(b)])))
    first, ids_a = extract_archive(archive, expected, a)
    second, ids_b = extract_archive(archive, expected, b)
    final_a, final_ids_a = local_readback(a, expected)
    final_b, final_ids_b = local_readback(b, expected)
    need(final_ids_a == ids_a and final_ids_b == ids_b, 'local copy changed across readbacks')
    need(first == second == final_a == final_b == expected, 'copies differ')
    need(all(ids_a[n][:2] != ids_b[n][:2] for n in expected['files']), 'copies share inode')
    need(fingerprint(archive, independent=True)[0] == pin_before, 'archive changed during extraction')
    receipt = dict(status='transport_verified_pending_collector_gates', acceptance=False,
                   capture_sha256=result_pin, inventory_before_sha256=before_pin, inventory_after_sha256=after_pin,
                   archive=pin_before, copies=[str(a), str(b)], files=len(expected['files']),
                   bytes=sum(v['bytes'] for v in expected['files'].values()), directories=len(expected['directories']))
    exclusive_write(attempt/'materialization.json', raw(receipt))
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    emit_parser = sub.add_parser('emit'); emit_parser.add_argument('--request-sha256', required=True)
    cap = sub.add_parser('capture')
    for n in ('request', 'request-sha256', 'command', 'command-sha256', 'attempt'):
        cap.add_argument('--'+n, required=True)
    cap.add_argument('--timeout', type=int, required=True)
    mat = sub.add_parser('materialize')
    for n in ('attempt', 'capture-sha256', 'before', 'before-sha256', 'after', 'after-sha256', 'copy-a', 'copy-b'):
        mat.add_argument('--'+n, required=True)
    a = ap.parse_args()
    if a.mode == 'emit':
        data = sys.stdin.buffer.read(MAX_METADATA + 1)
        need(len(data) <= MAX_METADATA and sha(data) == a.request_sha256, 'request pin/bound')
        result = emit(decode(data), sys.stdout.buffer)
        sys.stdout.buffer.flush()
        print(json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    elif a.mode == 'capture':
        result = capture(decode(read_pinned(Path(a.command), a.command_sha256)),
                         read_pinned(Path(a.request), a.request_sha256), Path(a.attempt), a.timeout)
        print(raw(result).decode(), end='')
    else:
        result = materialize(a.attempt, a.capture_sha256, Path(a.before), a.before_sha256,
                             Path(a.after), a.after_sha256, a.copy_a, a.copy_b)
        print(raw(result).decode(), end='')


if __name__ == '__main__':
    main()
