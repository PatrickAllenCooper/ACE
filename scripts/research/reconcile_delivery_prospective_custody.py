"""Reconcile complete split Stage B custody before a private relative plan.

Explicit study snapshots only; learner/project closure follows the original
registration. Every claimed copy is checked and conflicts reject. Original
upstream acceptance must succeed before any inventory is built. Scores and
heldout values are hashed but never decoded. No fitting or response generation.
"""
import argparse
import json
from pathlib import Path

from build_delivery_release import reconcile, write
from prepare_delivery_prospective_release_core import accepted_before_scores, read_snapshot, REGISTRATION_SHA
from reconcile_delivery_fit_snapshots import inventory as fit_inventory
from verify_delivery_release import relative, sha


def inventory(accepted_root, snapshots, project, source):
    gate = accepted_before_scores(accepted_root)
    root = Path(accepted_root).resolve()
    reg, _ = read_snapshot(root/'registration.json', REGISTRATION_SHA)
    snapshots = [Path(p).resolve() for p in snapshots]
    if not snapshots or len(set(snapshots)) != len(snapshots) or root not in snapshots:
        raise ValueError('explicit distinct snapshots must include accepted complete metadata root')
    matrix_path = root/'matrix.json'
    seal, _ = read_snapshot(root/'fit_seal.json', gate['custody_bound_acceptance_projection']['metadata']['fit_seal_sha256'])
    fits = fit_inventory(matrix_path, snapshots, seal['matrix_sha256'])
    if fits['cells_verified'] != 640 or fits['missing_indices'] or fits['registration_sha256'] != REGISTRATION_SHA:
        raise ValueError('complete conflict-rejecting 640 fit custody required')
    files = {}
    def add(path, name, expected=None):
        path = Path(path)
        # resolve roots above, but never resolve away a child symlink.
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('symlink in custody input')
        if not path.is_file(): raise ValueError('missing explicit custody artifact')
        h = sha(path)
        if expected is not None and h != expected: raise ValueError('original closure digest changed')
        if name in files:
            if files[name]['sha256'] != h: raise ValueError('conflicting claimed custody copies')
            files[name]['sources'].append(str(path))
        else:
            files[name] = {'path': name, 'sha256': h, 'sources': [str(path)]}
    for snapshot in snapshots:
        for path in sorted(snapshot.rglob('*')):
            if path.is_symlink(): raise ValueError('symlink in study snapshot')
            if path.is_file():
                name = path.relative_to(snapshot).as_posix()
                # Snapshots must be dedicated copies of the study, not an
                # arbitrary repository/environment. Project/learner roots follow.
                if name.startswith(('project/', 'source/', 'runner/', 'bundle/', '.git/')):
                    raise ValueError('non-study tree in study snapshot')
                add(path, 'B/'+name)
    project, source = Path(project).resolve(), Path(source).resolve()
    committed, _ = read_snapshot(project/'source_commit_receipt.json', reg['source_commit_receipt_sha256'])
    if committed['source_revision'] != reg['source_revision']:
        raise ValueError('committed project revision changed')
    add(project/'source_commit_receipt.json', 'B/project/source_commit_receipt.json', reg['source_commit_receipt_sha256'])
    for name, h in committed['files'].items():
        add(relative(project, name), 'B/project/'+name, h)
    for name, h in reg['source_hashes'].items():
        add(relative(source, name), 'source/runner/'+name, h)
    # "Completed" requires the raw scientific/attempt closure as well as fits.
    accepted = gate['custody_bound_acceptance_projection']['metadata']
    required = {'B/'+name for name in (
        'registration.json', 'matrix.json', 'complete.json', 'acceptance.json', 'audit_execution.json',
        'scores.json', 'fit_seal.json', 'evaluation_started.json', 'collection_complete.json',
        'qualification_complete.json', 'qualification.log', 'descriptor_manifest.json',
        'attribution_gate.json', 'pilot_acceptance.json', 'pilot_projection.json',
        'historical_pilot_failure.json', 'descriptor_runtime_parity.json')}
    pinned = {'B/registration.json': REGISTRATION_SHA, 'B/acceptance.json': gate['acceptance_sha256'],
              'B/audit_execution.json': gate['audit_execution_sha256'], 'B/scores.json': gate['scores_sha256'],
              'B/complete.json': accepted['complete_sha256'], 'B/fit_seal.json': accepted['fit_seal_sha256']}
    pinned.update({'B/'+name: h for name, h in accepted['phase_execution_hashes'].items()})
    for key, maps, prefix in [('training_receipt_hashes', accepted['training_receipt_hashes'], 'training'),
                               ('evaluation_receipt_hashes', accepted['evaluation_receipt_hashes'], 'evaluation')]:
        for logical, hashes in maps.items():
            case, _, history = logical.partition('/')
            folder = 'B/'+prefix+'/'+case.replace(':', '-')+('/'+history if history else '')
            pinned.update({folder+'/'+name: h for name, h in hashes.items()})
    matrix, _ = read_snapshot(matrix_path, seal['matrix_sha256'])
    if (len(matrix['cells']) != 640 or len(seal['artifacts']) != 640 or
            set(seal['artifacts']) != {cell['out'] for cell in matrix['cells']}):
        raise ValueError('exact original 640-cell fit-seal membership required')
    for cell in matrix['cells']:
        world = cell['case'].replace(':', '-')
        folder = 'B/fits/'+world+'/'+cell['history']+'/'+cell['arm']+'-i'+str(cell['init'])
        artifact = seal['artifacts'][cell['out']]
        if set(artifact) != {'receipt_sha256', 'model_sha256'}:
            raise ValueError('original fit-seal binding fields changed')
        pinned[folder+'/receipt.json'] = artifact['receipt_sha256']
        pinned[folder+'/models.pt'] = artifact['model_sha256']
        required.add('B/evaluation/'+world+'/cell-'+str(cell['index'])+'-predictions.npz')
        for name in ('execution.json', 'complete.json'):
            required.add('B/world_execution/'+world+'/'+name)
        for name in ('world.json', 'actions.json'):
            required.add('B/descriptors/'+world+'/'+name)
    required.update(pinned)
    if not required <= set(files):
        raise ValueError('raw response/descriptor/attempt/phase closure incomplete')
    if any(files[name]['sha256'] != h for name, h in pinned.items()):
        raise ValueError('original acceptance raw-file bindings differ')
    # Recheck every copy immediately before emitting an inventory; this does
    # not replace the planner's independent checks/captured private snapshots.
    for entry in files.values():
        reconcile(entry['sources'], entry['sha256'])
    return {'schema': 'delivery-prospective-composite-v1', 'completed': True,
            'conflict_policy': 'reject', 'conflicts': [], 'missing_indices': [],
            'cells_verified': 640, 'registration_sha256': REGISTRATION_SHA,
            'matrix_sha256': seal['matrix_sha256'], 'files': list(files.values()),
            'upstream_acceptance_sha256': gate['acceptance_sha256'],
            'upstream_audit_execution_sha256': gate['audit_execution_sha256'],
            'duplicate_fit_copies_verified': fits['duplicate_copies_verified'],
            'outcomes_decoded': False, 'scope': 'explicit complete composite custody; planner membership and target replay remain separate'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-root', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, nargs='+', required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('exclusive inventory output required')
    result = inventory(args.accepted_root, args.snapshots, args.project, args.source)
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in [*args.snapshots, args.project, args.source]):
        raise ValueError('inventory output must be external to original custody')
    write(args.out, result)
    print(json.dumps({k: result[k] for k in ('cells_verified', 'outcomes_decoded', 'scope')}))
