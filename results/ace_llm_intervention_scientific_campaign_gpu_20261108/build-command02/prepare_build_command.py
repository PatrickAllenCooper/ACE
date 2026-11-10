"""Prepare a concrete GPU02 builder invocation without invoking it."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

REPO = Path('/Users/pat/code/ACE')
PREP = REPO/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
GPU = REPO/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def raw(o):
    return (json.dumps(o, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata-review', required=True)
    p.add_argument('--metadata-review-sha256', required=True)
    a = p.parse_args()
    review = Path(a.metadata_review)
    assert sha(review.read_bytes()) == a.metadata_review_sha256
    old = json.loads((GPU/'build-command01/command.json').read_bytes())
    metadata = dict(old['metadata'])
    additions = {
        'gpu-failure-acceptance': (GPU/'failure-acceptance01/failure-acceptance.json', 'cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0'),
        'gpu-failure-baseline': (GPU/'failure-acceptance01/baseline.json', 'cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef'),
        'gpu-failure-packet': (GPU/'launch01/packet.json', '436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'),
        'gpu-failure-result': (GPU/'launch01/submission.result.json', '7c484f72ecafe9188cf4a71cb7023844c46390eb200ffe7507123d473ba43b86'),
        'gpu-failure-disposition': (GPU/'terminal01/disposition.json', '545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507'),
        'gpu-failure-seal': (GPU/'terminal01/custody-failure-inventory.json', '8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0'),
        'gpu-failure-inventory': (GPU/'terminal01/inventory_transport.json', 'f9c7dc2ecf1e0c91990e2c7d787abea37187fa520675ad580a1a7f5359f8d9d3'),
        'gpu-failure-inventory-after': (GPU/'terminal01/inventory_after_transport.json', '4303d7d4ef775ac133dde12b3d2ac875b7c31ed2382b45abd0c6cc6a55a188ac'),
        'gpu-recovery-authority': (PREP/'authority01/objects/gpu-recovery-authority.json', 'b3630d400649ecd186e35232b24e9020e5bced7ad616379c7eaf5eef84498603'),
        'gpu-recovery-review': (review, a.metadata_review_sha256),
        'operational-source-manifest': (PREP/'authority01/objects/operational-source-manifest.json', '7eceb8f0fb61696d5c42024a981f5e97dde683e58f2e24427b30a9e7703470f8'),
        'operational-source-review': (PREP/'integration-recheck01/review.md', '77ab1d1d58572703711ebe37040613ab8f45448a0db14238b11cdd9ae570335a'),
        'cpu-history-expected': (PREP/'actual-preflight01/objects/cpu-history-expected.json', 'e3c8b4be09558e5f9d182c4e2264d5f51a2497457546020d57a855543b8fa789'),
        'cpu-history-preflight': (PREP/'actual-preflight01/objects/cpu-history-preflight.json', 'ab9d0ac13e2cbb17cc7c32c9ac0ba07a6e6a988e890402c02757807a519f1421'),
        'cpu-history-readset': (PREP/'readset01/cpu-history-readset.json', '9be8223034cadedb0a3ff885236f923519b81f8c98bd46a7ffbcd2dafd9812b0'),
        'preflight-review': (review, a.metadata_review_sha256),
    }
    for name, (path, pin) in additions.items():
        assert name not in metadata and sha(path.read_bytes()) == pin
        metadata[name] = dict(path=str(path), sha256=pin)
    builder = PREP/'coordinator02/authoring/build_packet.py'
    assert sha(builder.read_bytes()) == '85691a9c7d52e40b334d9e42ea2da67762a955953c115de8a7d58be385935c5a'
    argv = [sys.executable, '-I', '-B', str(builder)]
    for name, record in metadata.items():
        assert sha(Path(record['path']).read_bytes()) == record['sha256']
        argv.extend(['--'+name, record['path'], '--'+name+'-sha256', record['sha256']])
    for name in ('cpu-remote-custody', 'cpu-terminal-mirror', 'storage-config', 'storage-config-sha256'):
        pos = old['argv'].index('--'+name)
        argv.extend(old['argv'][pos:pos+2])
    dest = GPU/'build-command02'
    dest.mkdir()
    command = dict(schema='ace-gpu02-cli-assembly-v1', argv=argv, cwd=str(REPO), metadata=metadata,
        builder_sha256=sha(builder.read_bytes()), executed=False, submitted=False,
        independent_acceptance_performed_here=False, scientific_qualified=False)
    (dest/'command.json').write_bytes(raw(command))
    (dest/'prepare_build_command.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(command=str(dest/'command.json'), sha256=sha(raw(command)), argv_count=len(argv),
        metadata_count=len(metadata), submitted=False)))


if __name__ == '__main__':
    main()
