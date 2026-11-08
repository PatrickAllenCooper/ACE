"""Build an exclusive derived package from an explicit private artifact plan.

Original receipts are never edited. Only top-level, explicitly named metadata
fields can be projected; unmodified numerical/model/source bytes retain their
original SHA256. Plans and private derivation records contain custody paths and
must not be included in an anonymous public package. No network/fits/responses.
"""
import argparse
import json
from pathlib import Path
import shutil

from verify_delivery_release import sha, relative, verify, project_toml_bytes


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def reconcile(sources, expected):
    """Every claimed composite copy must agree; never silently pick a winner."""
    if not sources:
        raise ValueError('missing custody source')
    for source in map(Path, sources):
        if source.is_symlink() or not source.is_file() or sha(source) != expected:
            raise ValueError('missing/changed/conflicting custody artifact')
    return Path(sources[0])


def build(plan_file, destination, private_receipt):
    plan = json.loads(Path(plan_file).read_text())
    destination, private_receipt = Path(destination).resolve(), Path(private_receipt).resolve()
    if private_receipt.is_relative_to(destination) or private_receipt.exists():
        raise ValueError('private derivation record requires exclusive external custody')
    if Path(plan_file).resolve().is_relative_to(destination):
        raise ValueError('private artifact plan cannot be included in package')
    destination.mkdir(parents=True, exist_ok=False)
    files, provenance = [], []
    seen = set()
    for entry in plan['files']:
        name = entry['path']
        if name in seen or name in ('manifest.json', 'manifest.sha256'):
            raise ValueError('duplicate/reserved artifact location')
        seen.add(name)
        target = relative(destination, name)
        source = reconcile(entry['sources'], entry['original_sha256'])
        target.parent.mkdir(parents=True, exist_ok=True)
        transform = entry.get('transform', {'kind': 'identity'})
        if transform['kind'] == 'identity':
            shutil.copyfile(source, target)
        elif transform['kind'] == 'project-json':
            original = json.loads(source.read_text())
            keys = transform['keys']
            if len(set(keys)) != len(keys) or not keys:
                raise ValueError('explicit unique projection keys required')
            write(target, {key: original[key] for key in keys})
        elif transform['kind'] == 'project-toml':
            import hashlib
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['original_sha256']:
                raise ValueError('original TOML snapshot changed')
            target.write_bytes(project_toml_bytes(raw, transform))
        else:
            raise ValueError('unsupported derivation')
        files.append({'path': name, 'sha256': sha(target), 'bytes': target.stat().st_size,
                      'original_sha256': entry['original_sha256'], 'transform': transform,
                      'role': entry['role']})
        provenance.append({'path': name, 'sources': entry['sources'],
                           'original_sha256': entry['original_sha256'], 'sha256': sha(target)})
    manifest = {'schema': 'delivery-anonymous-derived-v1', 'status': plan['status'],
                'original_protocol_proof': 'Original receipts remain in private custody; projected metadata is newly hashed.',
                'files': files, 'bindings': plan['bindings']}
    write(destination/'manifest.json', manifest)
    digest = sha(destination/'manifest.json')
    (destination/'manifest.sha256').write_text(digest+'\n')
    verified = verify(destination, digest)
    # Only seal the private derivation after all package integrity gates pass.
    write(private_receipt, {'plan_sha256': sha(plan_file), 'manifest_sha256': digest,
                           'destination': str(destination), 'derivations': provenance,
                           'verification': verified})
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'destination', 'private-receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.destination, args.private_receipt), indent=2))
