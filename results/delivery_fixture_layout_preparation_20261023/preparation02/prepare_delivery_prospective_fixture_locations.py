"""Prepare two distinctly hashed B fixture sources, without executing fixtures.

Only the model fixture's two location declarations change. Scientific tests,
workers, guards and original source receipts are never edited or executed.
This is a private intermediate, not qualified anonymous training reproduction.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

from prepare_delivery_publication_snapshot import (
    exclusive_directory, plain_path, seal_directory, store)

REVISION = '45ebeb89d2c76daa97a55f07728239245e0c4f60'
REGISTRATION = 'e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5'
MODEL = 'scripts/research/test_delivery_prospective_models.py'
IO = 'scripts/research/test_delivery_prospective_io.py'
PINS = {
    MODEL: '913dd23cf0aa34a22447f9c807753a1011e1c6684a70abbe1443eb9f26533f67',
    IO: '9cd49e43934fce0f542ba38e51f992a85533899b6634bc274bdd7e6d6fc63a2c',
}
ORIGINAL = {
    'PROJECT': 'Path(__file__).resolve().parents[2]',
    'SOURCE': "Path(os.environ.get('ACE_DELIVERY_RUNNER_SOURCE', '/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source'))",
}
REPLACEMENTS = {
    'PROJECT': "Path(os.environ['ACE_DELIVERY_FIXTURE_ROOT']).resolve(strict=True) / 'project'",
    'SOURCE': "Path(os.environ['ACE_DELIVERY_FIXTURE_ROOT']).resolve(strict=True) / 'source'",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def read_pinned(path, pin):
    if not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin):
        raise ValueError('independent full SHA256 required')
    path = plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        raw = stream.read()
    if digest(raw) != pin:
        raise ValueError('source/metadata pin differs')
    return raw


def derive_model(raw):
    """Change exact top-level RHS spans; every other original byte survives."""
    text = raw.decode('utf-8')
    tree = ast.parse(text)
    changes = []
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for name, expected in ORIGINAL.items():
        candidates = [node for node in tree.body if isinstance(node, ast.Assign)
                      and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == name]
        if len(candidates) != 1:
            raise ValueError('exact unique original location declaration required')
        value = candidates[0].value
        if ast.dump(value) != ast.dump(ast.parse(expected, mode='eval').body):
            raise ValueError('unreviewed original location expression')
        first = offsets[value.lineno - 1] + value.col_offset
        last = offsets[value.end_lineno - 1] + value.end_col_offset
        changes.append((first, last, REPLACEMENTS[name].encode()))
    result = raw
    for first, last, replacement in sorted(changes, reverse=True):
        result = result[:first] + replacement + result[last:]
    # Location expressions are the sole semantic edits, including at module scope.
    derived = ast.parse(result)
    for parsed in (tree, derived):
        for node in parsed.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ORIGINAL:
                node.value = ast.Constant(value='explicit fixture location')
    if ast.dump(tree) != ast.dump(derived):
        raise ValueError('non-location semantics changed')
    return result


def prepare(closure, expected_closure, destination):
    closure, destination = plain_path(closure), plain_path(destination)
    root = closure.parent
    if destination.exists() or destination.resolve().is_relative_to(root) or destination.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('new exclusive external preparation required')
    raw = read_pinned(closure, expected_closure)
    record = json.loads(raw, object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
    if record['source_revision'] != REVISION or record['registration_sha256'] != REGISTRATION or record['outcomes_accessed'] is not False:
        raise ValueError('original frozen source closure required')
    files = record['project_files']
    learners = record['learner_files']
    if len(files) != 24 or len(learners) != 19 or any(files.get(name) != pin for name, pin in PINS.items()):
        raise ValueError('original fixture and closure membership differs')
    # Verify the complete original closure from one byte snapshot per file.
    # No imported worker, checkpoint, action descriptor or response is opened.
    captured = {}
    for directory, mapping in (('project', files), ('source', learners)):
        for name, pin in mapping.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
                raise ValueError('canonical relative source path required')
            captured[directory+'/'+name] = read_pinned(root/directory/name, pin)
    model = captured['project/'+MODEL]
    io = captured['project/'+IO]
    derived_model = derive_model(model)
    # IO imports the model fixture by its original module name. Its entire bytes
    # remain unchanged; both must later be placed in the same reviewed fixture directory.
    outputs = {
        'originals/test_delivery_prospective_models.py': model,
        'originals/test_delivery_prospective_io.py': io,
        'fixtures/test_delivery_prospective_models.py': derived_model,
        'fixtures/test_delivery_prospective_io.py': io,
    }
    contract = {
        'schema': 'delivery-B-fixture-locations-v1',
        'scope': 'private source derivation only; no fixture or worker execution',
        'original_revision': REVISION, 'registration_sha256': REGISTRATION,
        'original_closure_sha256': expected_closure,
        'location_edits': {name: {'original_expression': ORIGINAL[name],
                                  'derived_expression': REPLACEMENTS[name]}
                           for name in ORIGINAL},
        'required_environment': {'ACE_DELIVERY_FIXTURE_ROOT': 'explicit root of an authenticated relocated project/source layout; no default'},
        'files': [{'path': name, 'sha256': digest(data), 'bytes': len(data)}
                  for name, data in outputs.items()],
        'preserved': ['all model/IO test function and class bytes',
                      'all charging, heldout, calibration, predicted-parent and optimizer assertions',
                      'all original workers, guards, source receipts and scientific inputs'],
        'remaining_gates': [
            'authenticated import closure and cached-module rejection before fixture imports',
            'existing exact target dependency environment and explicit import layout',
            'separate reviewed fixture execution authorization; no accepted fit rerun',
            'new source/notice/manifest dispositions for any anonymous release',
            'remaining batch/audit/collection/training location contracts'],
        'fixtures_executed': False, 'original_training_reproduction_qualified': False,
        'anonymous_execution_qualified': False, 'public_release_approved': False,
        'new_fits': 0, 'new_responses': 0,
    }
    encoded = (json.dumps(contract, sort_keys=True, indent=2) + '\n').encode()
    directory = exclusive_directory(destination)
    try:
        for name, data in outputs.items():
            store(directory, name, data)
        store(directory, 'contract.json', encoded)
        seal_directory(directory)
        a, b = os.stat(plain_path(destination), follow_symlinks=False), os.fstat(directory)
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise ValueError('custody location changed')
    finally:
        os.close(directory)
    return {'contract_sha256': digest(encoded), 'files': len(outputs),
            'verified_original_project_files': len(files),
            'verified_original_learner_files': len(learners),
            'fixtures_executed': False, 'reproduction_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure', type=Path, required=True)
    parser.add_argument('--expected-closure-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.closure, args.expected_closure_sha256, args.destination), indent=2))
