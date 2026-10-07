"""Offline, read-only verification of a derived anonymous delivery package.

The manifest is a new provenance object, not a rewritten experimental receipt.
No model deserialization, fitting, network access or simulator calls occur here.
Pin --expected-manifest-sha256 from independent custody for authenticity; a
matching adjacent checksum alone establishes only internal byte consistency.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(root, name):
    """Disallow traversal, aliases and symlinks, including parent symlinks."""
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid relative artifact location')
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(v in ('.', '..') for v in p.parts):
        raise ValueError('noncanonical relative artifact location')
    root = Path(root).resolve()
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink in artifact location')
    if not current.resolve().is_relative_to(root):
        raise ValueError('artifact escaped package')
    return current


def pointer(value, name):
    if not isinstance(name, str) or not name.startswith('/'):
        raise ValueError('JSON pointer required')
    for part in name[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def identifying_bytes(path):
    """Conservative automated screen, not a guarantee of reviewer anonymity.

    Scan decompressed ZIP members as well as outer bytes (NPZ/PT/archive files).
    Metadata/citation/license and binary model review remains a human gate.
    """
    import zipfile
    pattern = re.compile(rb'/Users/|/scratch/|/projects/|paco0228|ucb736_asc1|'
                         rb'PatrickAllenCooper|ACE_Study_Results|defab-curc\.sock', re.I)
    # Raw byte matching alone misses JSON slash/Unicode escape spellings.
    if Path(path).suffix in ('.json', '.ndjson'):
        with Path(path).open() as stream:
            values = ([json.load(stream)] if Path(path).suffix == '.json'
                      else (json.loads(line) for line in stream if line.strip()))
            for value in values:
                if pattern.search(json.dumps(value, ensure_ascii=False).encode()):
                    return True
    def scan(stream):
        tail = b''
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            data = tail + block
            if pattern.search(data):
                return True
            tail = data[-256:]
        return False
    with Path(path).open('rb') as stream:
        if scan(stream):
            return True
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if pattern.search(member.filename.encode()):
                    return True
                with archive.open(member) as stream:
                    if scan(stream):
                        return True
    return False


def verify(root, expected=None):
    root = Path(root).resolve()
    manifest_file = relative(root, 'manifest.json')
    digest = sha(manifest_file)
    if expected is not None and digest != expected:
        raise ValueError('externally pinned manifest changed')
    if relative(root, 'manifest.sha256').read_text().strip() != digest:
        raise ValueError('manifest checksum mismatch')
    manifest = json.loads(manifest_file.read_text())
    if manifest['schema'] != 'delivery-anonymous-derived-v1':
        raise ValueError('unsupported manifest schema')
    files = manifest['files']
    names = [f['path'] for f in files]
    if len(set(names)) != len(names):
        raise ValueError('duplicate artifact location')
    indexed = {f['path']: f for f in files}
    for f in files:
        p = relative(root, f['path'])
        if not p.is_file() or p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
            raise ValueError('artifact bytes changed: ' + f['path'])
        if not re.fullmatch('[0-9a-f]{64}', f['original_sha256']):
            raise ValueError('invalid original digest')
        transform = f['transform']
        if transform['kind'] == 'identity':
            if f['sha256'] != f['original_sha256']:
                raise ValueError('identity transform changed original bytes')
        elif transform['kind'] == 'project-json':
            data = json.loads(p.read_text())
            if set(data) != set(transform['keys']):
                raise ValueError('derived metadata projection mismatch')
        else:
            raise ValueError('unsupported derivation')
        # Apply screening only to research artifacts, not this verifier's own
        # explicit list of blocked patterns. A privileged role is reserved for it.
        if f['role'] == 'verification-tool':
            if f['path'] != 'verify_delivery_release.py' or f['sha256'] != sha(__file__):
                raise ValueError('verification exemption requires the executing trusted tool bytes')
        elif identifying_bytes(p):
            raise ValueError('identifying material requires disposition: ' + f['path'])
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('unlisted symlink')
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != set(names) | {'manifest.json', 'manifest.sha256'}:
        raise ValueError('unlisted or missing package file')
    if identifying_bytes(manifest_file):
        raise ValueError('identifying manifest metadata')
    for b in manifest['bindings']:
        value = pointer(json.loads(relative(root, b['record']).read_text()), b['pointer'])
        target = indexed[b['artifact']]
        if b['digest'] not in ('sha256', 'original_sha256') or value != target[b['digest']]:
            raise ValueError('receipt/artifact digest binding mismatch')
    return {'manifest_sha256': digest, 'files_verified': len(files),
            'bindings_verified': len(manifest['bindings']),
            'expected_digest_supplied': expected is not None,
            'pin_origin_verified': False, 'verifier_sha256': sha(__file__), 'status': manifest['status'],
            'scope': 'byte/digest/relocation verification; no checkpoint replay or historical protocol proof',
            'new_fits': 0, 'new_responses': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.expected_manifest_sha256), indent=2))
