"""Read-only reconciliation of explicit Stage B custody snapshots, no losses.

Missing cells are retained as missing; conflicting bytes fail. This prepares
relocation inventory and cannot substitute for complete checkpoint replay.
"""
import argparse
import json
from pathlib import Path

from build_delivery_release import reconcile, write
from verify_delivery_release import sha, relative


def inventory(matrix, snapshots, expected=None):
    matrix = Path(matrix)
    if expected is not None and sha(matrix) != expected:
        raise ValueError('pinned matrix changed')
    data = json.loads(matrix.read_text())
    cells = data['cells']
    if len(cells) != 640 or {c['index'] for c in cells} != set(range(640)):
        raise ValueError('full frozen matrix required')
    snapshots = [Path(p).resolve() for p in snapshots]
    if len(set(snapshots)) != len(snapshots):
        raise ValueError('duplicate snapshot root')
    allowed = set()
    result, missing = [], []
    for cell in cells:
        name = 'fits/'+cell['case'].replace(':', '-')+'/'+cell['history']+'/'+cell['arm']+'-i'+str(cell['init'])
        if name+'/receipt.json' in allowed:
            raise ValueError('duplicate logical matrix cell')
        allowed.add(name+'/receipt.json')
        receipts = [relative(s, name+'/receipt.json') for s in snapshots]
        receipts = [p for p in receipts if p.exists()]
        if not receipts:
            missing.append(cell['index'])
            continue
        reconcile(receipts, sha(receipts[0]))
        receipt = json.loads(receipts[0].read_text())
        if (receipt['complete'] is not True or receipt['arm'] != cell['arm'] or receipt['init'] != cell['init']
                or receipt['binding'] != cell['binding'] or receipt['new_queries_in_fit'] != 0
                or receipt['evaluation_responses_read'] != 0):
            raise ValueError('cell/receipt/boundary mismatch')
        models = [p.parent/'models.pt' for p in receipts]
        reconcile(models, receipt['model_sha256'])
        result.append({'index': cell['index'], 'relative_folder': name,
                       'receipt_sha256': sha(receipts[0]), 'model_sha256': receipt['model_sha256'],
                       'receipt_sources': list(map(str, receipts)), 'model_sources': list(map(str, models))})
    for snapshot in snapshots:
        for p in snapshot.glob('fits/*/*/*/receipt.json'):
            if p.relative_to(snapshot).as_posix() not in allowed:
                raise ValueError('unregistered custody receipt')
    return {'matrix_sha256': sha(matrix), 'registration_sha256': data['registration_sha256'],
            'cells_verified': len(result), 'cells_expected': 640, 'missing_indices': missing,
            'duplicate_copies_verified': sum(len(r['receipt_sources'])-1 for r in result),
            'cells': result, 'outcomes_read': False, 'scientific_acceptance': False,
            'scope': 'model/receipt bytes and frozen input-binding reconciliation only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-matrix-sha256')
    args = parser.parse_args()
    result = inventory(args.matrix, args.snapshots, args.expected_matrix_sha256)
    write(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))
