"""Preserve hash-bound frozen study workers in exclusive PRIVATE source custody.

This is provenance preparation, not a relocated execution or public release.
Never rewrite historical source to remove identifiers or disable its guards.
No fitting, response acquisition, Slurm changes, dependency installs or B losses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_delivery_release import write
from verify_delivery_release import identifying_bytes, sha


def archive(repo, attribution_protocol, physical_protocol, prospective_registration, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    records = [('A', Path(attribution_protocol)), ('C', Path(physical_protocol)),
               ('B', Path(prospective_registration))]
    metadata = {stage: json.loads(path.read_text()) for stage, path in records}
    specs = []
    for stage, revision, field, name in [('A', 'af7809c5', 'adapter_sha256', 'delivery_attribution.py'),
                                         ('C', '0357cda7', 'worker_sha256', 'chambers_delivery_validation.py')]:
        specs.append((stage, revision, 'scripts/research/'+name, metadata[stage][field], field))
        specs.append((stage, revision, 'scripts/research/runner_delivery_confirmation.py',
                      metadata[stage]['guard_sha256'], 'guard_sha256'))
    b = metadata['B']
    for name, digest in b['worker_hashes'].items():
        specs.append(('B', b['source_revision'], 'scripts/research/'+name, digest, 'worker_hashes'))
    for name, digest in b['generator_hashes'].items():
        specs.append(('B', b['source_revision'], name, digest, 'generator_hashes'))
    # Resolve and validate all immutable Git objects before creating custody.
    verified = []
    for stage, revision, path, expected, binding in specs:
        resolved = subprocess.check_output(['git', 'rev-parse', revision+'^{commit}'], cwd=repo, text=True).strip()
        data = subprocess.check_output(['git', 'show', resolved+':'+path], cwd=repo)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('frozen worker binding differs: '+stage+'/'+path)
        verified.append((stage, resolved, path, expected, binding, data))
    out.mkdir(parents=True, exist_ok=False)
    inventory = []
    for stage, revision, path, expected, binding, data in verified:
        destination = out/stage/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(data)
        if sha(destination) != expected:
            raise ValueError('source copy changed')
        inventory.append({'stage': stage, 'path': stage+'/'+path, 'source_revision': revision,
                          'sha256': expected, 'protocol_field': binding,
                          'known_identifier_screen_positive': identifying_bytes(destination),
                          'transformation': 'identity', 'public_release_approved': False})
    record = {'status': 'private frozen-worker custody only; anonymous relocation adapter still required',
              'protocol_hashes': {stage: sha(path) for stage, path in records},
              'files': inventory, 'worker_bindings': len(inventory),
              'source_checking_bypassed': False, 'historical_receipts_modified': False,
              'new_fits': 0, 'new_responses': 0, 'B_outcomes_read': False}
    write(out/'inventory.json', record)
    return {'inventory_sha256': sha(out/'inventory.json'), **record}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'attribution-protocol', 'physical-protocol', 'prospective-registration', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.repo, args.attribution_protocol, args.physical_protocol,
                             args.prospective_registration, args.out), indent=2))
