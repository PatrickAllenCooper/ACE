"""Invoke reviewed pure saved-metadata assembly, then the pinned pure checker.

No source from the experiment is imported and no remote path is accessed.
Input copies already exist in sealed custody. This command records their pins.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys

REPO = Path('/Users/pat/code/ACE')
PREP = REPO/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
GPU = REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
INPUTS = {
    'authority': (PREP/'authority01/objects/gpu-recovery-authority.json', 'b3630d400649ecd186e35232b24e9020e5bced7ad616379c7eaf5eef84498603'),
    'failure_acceptance': (GPU/'failure-acceptance01/failure-acceptance.json', 'cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0'),
    'failure_proof': (GPU/'failure-acceptance01/no-policy-work-proof.json', '164ab63397e6dec86f0b811dfde192d11d3983b507fb742bf2d2f6adaba2b349'),
    'baseline': (GPU/'failure-acceptance01/baseline.json', 'cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef'),
    'cpu_acceptance': (GPU/'launch01/cpu-acceptance.json', '3c5ce7375d0c64292553ded6ad3078e634745eb7996d83fd102e344eeab3f063'),
    'before_observation': (GPU/'terminal01/inventory_transport.json', 'f9c7dc2ecf1e0c91990e2c7d787abea37187fa520675ad580a1a7f5359f8d9d3'),
    'after_observation': (GPU/'terminal01/inventory_after_transport.json', '4303d7d4ef775ac133dde12b3d2ac875b7c31ed2382b45abd0c6cc6a55a188ac'),
    'custody': (GPU/'terminal01/custody-failure-inventory.json', '8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0'),
    'remote_custody': (GPU/'terminal01/remote-custody.json', '5224bf51110996d0a72f334dc976db7644f24d99e7500b99730ddf00eff19214'),
    'readset': (PREP/'readset01/cpu-history-readset.json', '9be8223034cadedb0a3ff885236f923519b81f8c98bd46a7ffbcd2dafd9812b0'),
    'original_gpu_packet': (GPU/'launch01/inputs/packet.json', '85d2607d94114109819ca5e8150c5849c13f1bbf4b37765d3f7ee2ff64e77c58'),
    'original_cpu_packet': (GPU/'launch01/inputs/cpu-packet.json', 'ad046a726e3a3a7744e54a860f8cf1a30947a198e8be3bc25ae06ec7b2909344'),
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def raw(o):
    return (json.dumps(o, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def module(path, pin, name):
    b = Path(path).read_bytes()
    if sha(b) != pin:
        raise ValueError('reviewed source digest differs: '+name)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    # Execute exactly the captured pure metadata bytes, not a later reopen.
    exec(compile(b, str(path), 'exec'), result.__dict__)
    return result, b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('assembler', 'assembler-sha256', 'review', 'review-sha256', 'output'):
        p.add_argument('--'+n, required=True)
    a = p.parse_args()
    out = Path(a.output)
    if not out.is_absolute() or out.parent != out.parent.resolve(strict=True):
        raise ValueError('ordinary absolute exclusive output required')
    out.mkdir()
    review = Path(a.review).read_bytes()
    if sha(review) != a.review_sha256:
        raise ValueError('independent assembler review differs')
    api, source = module(a.assembler, a.assembler_sha256, 'reviewed_saved_metadata_assembler')
    checker, checker_source = module(PREP/'orchestration04/llm_intervention_cpu_preflight_receipt.py',
        'b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8', 'reviewed_saved_metadata_checker')
    (out/'assemble_cpu_preflight.py').write_bytes(source)
    (out/'llm_intervention_cpu_preflight_receipt.py').write_bytes(checker_source)
    (out/'source-review.md').write_bytes(review)
    inputs = {}
    pins = {}
    for role, (path, pin) in INPUTS.items():
        b = path.read_bytes()
        if sha(b) != pin:
            raise ValueError('independent input differs: '+role)
        inputs[role] = b
        pins[role] = pin
    (out/'input-pins.json').write_bytes(raw({r: dict(path=str(p), sha256=h, bytes=len(inputs[r])) for r, (p, h) in INPUTS.items()}))
    result = api.assemble(inputs=inputs, pins=pins,
        assembled_utc=datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'))
    expected, receipt = result['expected_raw'], result['receipt_raw']
    scoped = checker.validate_receipt(receipt, sha(receipt), expected, sha(expected))
    for name, b in (('cpu-history-expected.json', expected), ('cpu-history-preflight.json', receipt),
                    ('assembly-record.json', result['assembly_record_raw']), ('checker-result.json', raw(scoped))):
        with (out/name).open('xb') as handle:
            handle.write(b)
    print(json.dumps(dict(expected_sha256=sha(expected), receipt_sha256=sha(receipt),
        assembly_record_sha256=sha(result['assembly_record_raw']), checker_acceptance=scoped['acceptance'],
        scientific_qualified=False, fresh_remote_scan=False)))


if __name__ == '__main__':
    main()
