"""Package already-qualified B reports for the manuscript; no science execution.

This checks pinned reporting/custody metadata and byte relationships. Actual
supervised execution, scheduler disposition and custody must be accepted BEFORE
calling prepare; a mutually consistent fabricated packet is not execution proof.
Original exporters, replay inputs and earlier claim indexes are never modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

PRIMARY = ('delivery_prospective_claims.tex', 'delivery_prospective_primary_table.tex',
           'delivery_prospective_sensitivity_table.tex')
DESCRIPTIVE = ('delivery_prospective_absolute_table.tex',
               'delivery_prospective_descriptive_table.tex')
COMPANIONS = PRIMARY + DESCRIPTIVE
DATA = ('supplement.json', 'all_cells.csv', 'fixed_init0.csv',
        'system_log_ratios.csv', 'history_specific.csv', 'floor_counts.csv')
PRIMARY_INDEX = 'delivery_prospective_claim_index.json'
DESCRIPTIVE_INDEX = 'delivery_prospective_output_index.json'
PROOFS = {'freeze': 'delivery_prospective_freeze.json',
          'execution': 'delivery_prospective_execution.json',
          'replay': 'delivery_prospective_replay.json',
          'manifest': 'delivery_prospective_replay_manifest.json'}
BUNDLE = 'delivery_prospective_bundle.json'
FILES = (*COMPANIONS, *('delivery_prospective_'+n for n in DATA),
         PRIMARY_INDEX, DESCRIPTIVE_INDEX, *PROOFS.values(), BUNDLE)
DEPENDENCIES = {'torch': '2.9.1', 'numpy': '2.2.6', 'scipy': '1.15.3',
                'pandas': '2.3.3', 'sympy': '1.14.0', 'PyYAML': '6.0.3'}
REGISTRATION = 'e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5'
REVISION = '45ebeb89d2c76daa97a55f07728239245e0c4f60'
EXPECTED_KEYS = {'primary_index', 'descriptive_index', 'freeze', 'execution',
                 'replay', 'manifest', 'exporter', 'reporter'}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'independent full SHA256 pin required')
    return value


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate metadata key')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('nonfinite metadata constant: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            'symlink in reporting location')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'regular reporting file required')
        raw = stream.read(16*1024*1024+1)
    require(len(raw) <= 16*1024*1024, 'reporting object exceeds size bound')
    return raw


def integer(value, expected, message):
    require(type(value) is int and value == expected, message)


def match(actual, expected):
    """Compare recorded analyses at the frozen auditor's tolerance, no new tests."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), 'analysis keys differ')
        for key in expected:
            match(actual[key], expected[key])
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), 'analysis length differs')
        for a, b in zip(actual, expected):
            match(a, b)
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), 'analysis numeric value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, 'analysis value differs')


def validate(raw, expected_bundle):
    """Validate captured bytes without imports, fits or statistical recomputation."""
    require(set(raw) == set(FILES), 'complete prospective publication membership required')
    require(digest(raw[BUNDLE]) == pin(expected_bundle), 'prospective bundle digest differs')
    bundle = decode(raw[BUNDLE])
    require(bundle['schema'] == 'delivery-prospective-publication-v1', 'unknown reporting schema')
    expected = bundle['expected']
    require(set(expected) == EXPECTED_KEYS, 'exact reporting pin set required')
    for value in expected.values():
        pin(value)
    require(set(bundle['files']) == set(FILES)-{BUNDLE}, 'prospective bundle membership differs')
    for name, expected_file in bundle['files'].items():
        require(digest(raw[name]) == pin(expected_file), 'prospective file changed: '+name)
    for key, name in {**PROOFS, 'primary_index': PRIMARY_INDEX,
                      'descriptive_index': DESCRIPTIVE_INDEX}.items():
        require(digest(raw[name]) == expected[key], 'independent input pin differs: '+key)
    # Metadata acceptance is checked before decoding either scientific index.
    freeze, execution, replay = (decode(raw[PROOFS[n]]) for n in ('freeze', 'execution', 'replay'))
    require(freeze['manifest_sha256'] == execution['manifest_sha256'] == expected['manifest'],
            'replay manifest lineage differs')
    require(execution['freeze_sha256'] == expected['freeze'] and
            execution['replay_receipt_sha256'] == expected['replay'], 'supervision lineage differs')
    require(execution['status'] == 'complete' and execution['stage'] == 'qualified',
            'successful supervised qualification required')
    integer(execution['exit_code'], 0, 'supervisor exit differs')
    child = execution['child_execution']
    require(child['status'] == 'complete' and child['cleanup']['complete'] is True,
            'successful child and cleanup required')
    integer(child['exit_code'], 0, 'child exit differs')
    for value in (freeze, execution):
        for key in ('new_fits', 'new_responses'):
            integer(value[key], 0, 'new scientific work forbidden')
    require(freeze['exact_dependencies'] == replay['runtime'] == DEPENDENCIES,
            'exact replay dependency versions required')
    require(replay['full_supplemental_replay'] is True and
            replay['integrity']['expected_digest_supplied'] is True and
            replay['integrity']['manifest_sha256'] == expected['manifest'],
            'full pinned supplemental replay required')
    for key, value in {'checkpoints_replayed': 640, 'primary_cells_recomputed': 240,
                       'cached_responses_checked': 48000, 'new_optimizer_updates': 0,
                       'new_responses': 0}.items():
        integer(replay[key], value, 'replay counter differs: '+key)
    for key, counter in (('package_files', 'files_verified'), ('package_bindings', 'bindings_verified')):
        require(type(freeze[key]) is int and freeze[key] > 0, 'invalid frozen package count')
        integer(replay['integrity'][counter], freeze[key], 'replayed package membership differs')
    manifest = decode(raw[PROOFS['manifest']])
    require(manifest['schema'] == 'delivery-anonymous-derived-v1', 'unknown replay manifest schema')
    entries = {entry['path']: entry for entry in manifest['files']}
    require(len(entries) == len(manifest['files']) == freeze['package_files'], 'unique full replay membership required')
    runtime = execution['authenticated_runtime']
    for key in ('runtime_qualification_sha256', 'runtime_inventory_sha256', 'pre_checkpoint_runtime_sha256', 'python_sha256'):
        pin(runtime[key])
    require(runtime['python_sha256'] == freeze['base_python_sha256'] and
            type(runtime['environment_objects_verified']) is int and runtime['environment_objects_verified'] > 0,
            'supervised runtime closure missing')
    primary, descriptive = decode(raw[PRIMARY_INDEX]), decode(raw[DESCRIPTIVE_INDEX])
    provenance = descriptive['provenance']
    require(set(primary['output_hashes']) == set(PRIMARY) and
            set(descriptive['output_hashes']) == set(DESCRIPTIVE+DATA), 'complete reporting outputs required')
    for names, index in ((PRIMARY, primary), (DESCRIPTIVE+DATA, descriptive)):
        for name in names:
            target = 'delivery_prospective_'+name if name in DATA else name
            require(digest(raw[target]) == pin(index['output_hashes'][name]), 'report output binding differs: '+name)
    require(primary['exporter_sha256'] == expected['exporter'] and provenance['tool_sha256'] == expected['reporter'],
            'independent reporter source pins differ')
    require(primary['registration_sha256'] == provenance['original_registration_sha256'] == REGISTRATION and
            primary['source_revision'] == REVISION, 'original registered study differs')
    for key in ('acceptance_sha256', 'scores_sha256'):
        name = 'B/'+key.removesuffix('_sha256')+'.json'
        entry = entries[name]
        require(pin(primary[key]) == pin(entry['original_sha256']) and
                pin(provenance[key]) == pin(entry['sha256']), 'original/derived reporting input lineage differs: '+key)
        require(entry['transform']['kind'] in ('identity', 'project-json') and
                (entry['transform']['kind'] != 'identity' or entry['original_sha256'] == entry['sha256']),
                'unsupported reporting input derivation: '+name)
    require(provenance['manifest_sha256'] == expected['manifest'] and
            provenance['supplemental_replay_sha256'] == expected['replay'], 'descriptive replay lineage differs')
    integer(primary['deployment_initialization'], 0, 'fixed init0 reporting required')
    require(primary['supervision_and_compute_matched'] is False and
            descriptive['actual_numerical_replay_performed_here'] is False, 'reporting scope differs')
    match(primary['primary_analysis'], replay['primary_analysis'])
    match(primary['secondary_descriptive_analysis'], replay['secondary_descriptive_analysis'])
    supplement = decode(raw['delivery_prospective_supplement.json'])
    require(supplement['provenance'] == provenance, 'descriptive data provenance differs')
    for key, count in {'cells': 640, 'fixed_init0': 320, 'system_log_ratios': 80,
                       'history_specific_ratios': 8, 'short_fit_ablation': 2,
                       'primary_floor_counts': 4}.items():
        require(type(supplement[key]) is list and len(supplement[key]) == count, 'descriptive matrix length differs: '+key)
    for key in ('new_fits', 'new_responses', 'additional_significance_tests'):
        integer(supplement[key], 0, 'descriptive science scope differs')
    for name in COMPANIONS:
        require('\\end{filecontents*}' not in raw[name].decode(), 'nested companion terminator')
    return bundle


def read_bundle(root, expected_bundle):
    return validate({name: read(Path(root)/name) for name in FILES}, expected_bundle)


def capture_bundle(root, expected_bundle):
    raw = {name: read(Path(root)/name) for name in FILES}
    validate(raw, expected_bundle)
    return raw


def prepare(primary, descriptive, proof_paths, expected, destination):
    """Stage into a NEW directory only after actual upstream acceptance elsewhere."""
    primary, descriptive, destination = map(lambda p: Path(p).absolute(), (primary, descriptive, destination))
    require(set(proof_paths) == set(PROOFS) and set(expected) == EXPECTED_KEYS, 'complete explicit inputs required')
    origins = [primary, descriptive, *(Path(p).absolute().parent for p in proof_paths.values())]
    require(not any(p.is_symlink() for p in (destination, *destination.parents)) and
            not any(destination.resolve().is_relative_to(p.resolve()) for p in origins),
            'new output outside reporting inputs required')
    raw = {n: read(primary/n) for n in PRIMARY}
    raw.update({n: read(descriptive/n) for n in DESCRIPTIVE})
    raw.update({'delivery_prospective_'+n: read(descriptive/n) for n in DATA})
    raw[PRIMARY_INDEX] = read(primary/'claim_index.json')
    raw[DESCRIPTIVE_INDEX] = read(descriptive/'output_index.json')
    raw.update({PROOFS[n]: read(path) for n, path in proof_paths.items()})
    raw[BUNDLE] = (json.dumps({'schema': 'delivery-prospective-publication-v1', 'expected': expected,
        'files': {n: digest(b) for n, b in raw.items()}, 'scope': 'reporting byte/metadata consistency; upstream execution and final publication approval remain separately established'}, indent=2, sort_keys=True)+'\n').encode()
    bundle_pin = digest(raw[BUNDLE]);validate(raw, bundle_pin)
    # Reuse the reviewed macOS descriptor/ancestor/ACL custody writer. This
    # runtime import avoids a top-level cycle; no captured study code executes.
    from prepare_delivery_publication_snapshot import exclusive_directory, store, seal_directory
    directory = exclusive_directory(destination)
    try:
        for name, content in raw.items():
            store(directory, name, content)
        seal_directory(directory)
        final = os.stat(destination, follow_symlinks=False);opened = os.fstat(directory)
        require((final.st_dev, final.st_ino) == (opened.st_dev, opened.st_ino),
                'assembly custody location changed during writes')
    finally:
        os.close(directory)
    return {'bundle_sha256': bundle_pin, 'files': len(raw), 'companions': len(COMPANIONS),
            'actual_execution_validated_here': False, 'destination': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'descriptive', 'freeze', 'execution', 'replay', 'manifest', 'pins', 'destination'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.primary, args.descriptive,
        {name: getattr(args, name) for name in PROOFS}, decode(read(args.pins)), args.destination), indent=2))
