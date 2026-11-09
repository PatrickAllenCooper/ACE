"""Capture only independently pinned historical CPU admission metadata.

This creates no acceptance. Full failed-job custody, the CPU preflight binding,
and independently reviewed operational recovery are separate prerequisites.
No project module, model, scientific generator or report is executed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import stat

MAXIMUM = 16 * 1024**2
FILES = tuple(['campaign.json', 'source-plan-pins.json', 'start.json'] +
    [f'policy-{i:02d}/{name}' for i in range(5)
     for name in ('attempt.json', 'attempt.candidate.json')] +
    [f'policy-{i:02d}/world-{j:02d}/arm-receipt.json'
     for i in range(5) for j in range(24)])


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def decode(value):
    def unique(items):
        result = {}
        for key, item in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = item
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(value, object_pairs_hook=unique, parse_constant=bad)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def pinned_read(path, expected, maximum=MAXIMUM):
    """Capture one stable, sole-linked file through no-follow directory FDs."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute plain path required')
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
         'independent digest required')
    descriptors = []
    try:
        current = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(current)
        for name in path.parts[1:-1]:
            current = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=current)
            descriptors.append(current)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=current)
        with os.fdopen(fd, 'rb') as handle:
            before = os.fstat(handle.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and before.st_size <= maximum, 'bounded independent regular file required')
            value = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                              s.st_ctime_ns, s.st_nlink)
        live = os.stat(path.name, dir_fd=current, follow_symlinks=False)
        need(identity(before) == identity(after) == identity(live),
             'input changed during capture')
        # Reopen every absolute ancestor to detect a renamed/replaced parent.
        check = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        verification = [check]
        try:
            for index, name in enumerate(path.parts[1:-1], 1):
                check = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=check)
                verification.append(check)
                a, b = os.fstat(check), os.fstat(descriptors[index])
                need((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino),
                     'ancestor changed during capture')
        finally:
            for item in reversed(verification):
                os.close(item)
        need(len(value) <= maximum and sha(value) == expected, 'input digest differs')
        return value
    finally:
        for item in reversed(descriptors):
            os.close(item)


def capture(acceptance_raw, acceptance_sha256, local_campaign):
    need(sha(acceptance_raw) == acceptance_sha256, 'CPU acceptance digest differs')
    acceptance = decode(acceptance_raw)
    need(acceptance['schema'] == 'ace-scientific-campaign-cpu-acceptance-v1'
         and acceptance['status'] == 'development_cpu_phase_accepted'
         and acceptance['scientific_qualified'] is False,
         'accepted original CPU phase required')
    root = Path(local_campaign)
    need(root.is_absolute() and root == root.resolve(strict=True),
         'ordinary absolute local campaign required')
    retained = {}
    total = 0
    for name in FILES:
        expected = acceptance['campaign_files'][name]
        value = pinned_read(root / name, expected)
        total += len(value)
        need(total <= MAXIMUM, 'bounded aggregate read set required')
        text = value.decode('utf-8')
        need(text.encode('utf-8') == value, 'lossless UTF-8 capture required')
        retained[name] = dict(sha256=expected, bytes=len(value), utf8=text)
    result = dict(schema='ace-cpu-history-readset-v1',
                  cpu_acceptance_sha256=acceptance_sha256,
                  campaign_root=acceptance['campaign_root'], files=retained)
    output = canonical(result)
    need(len(output) <= MAXIMUM, 'bounded encoded read set required')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-acceptance', 'cpu-acceptance-sha256', 'local-campaign', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = pinned_read(Path(args.cpu_acceptance), args.cpu_acceptance_sha256)
    result = capture(source, args.cpu_acceptance_sha256, args.local_campaign)
    out = Path(args.output)
    need(out.is_absolute() and out.parent == out.parent.resolve(strict=True),
         'ordinary output parent required')
    with out.open('xb') as handle:
        handle.write(result)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(output=str(out), sha256=sha(result), bytes=len(result),
                          files=len(FILES), acceptance=False, models_loaded=False,
                          new_responses=0)))


if __name__ == '__main__':
    main()
