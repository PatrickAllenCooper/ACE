"""Assemble the narrow GPU02 operational authority from independently pinned metadata.

Run only after independent source review. This does not submit, qualify a GPU,
execute project source, or grant new scientific scope. The user authorized the
routine recovery. Original inputs and failed attempt evidence remain unchanged.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess

REPO = Path('/Users/pat/code/ACE')
PREP = 'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
GPU = 'results/ace_llm_intervention_scientific_campaign_gpu_20261108'
OLDROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
ENTRY = 'qualify_llm_intervention_scientific_campaign_gpu.py'
CHECKER = 'llm_intervention_cpu_preflight_receipt.py'
HISTORY = 'llm_intervention_cpu_history_admission.py'
PINS = {
    'cpu_acceptance': (GPU+'/launch01/cpu-acceptance.json', '3c5ce7375d0c64292553ded6ad3078e634745eb7996d83fd102e344eeab3f063'),
    'failed_acceptance': (GPU+'/failure-acceptance01/failure-acceptance.json', 'cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0'),
    'baseline': (GPU+'/failure-acceptance01/baseline.json', 'cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef'),
    'old_packet': (GPU+'/launch01/inputs/packet.json', '85d2607d94114109819ca5e8150c5849c13f1bbf4b37765d3f7ee2ff64e77c58'),
    'namespace': (GPU+'/launch01/namespace-claim.json', '40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a'),
    'custody': (GPU+'/terminal01/custody-failure-inventory.json', '8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0'),
    'before': (GPU+'/terminal01/inventory_transport.json', 'f9c7dc2ecf1e0c91990e2c7d787abea37187fa520675ad580a1a7f5359f8d9d3'),
    'after': (GPU+'/terminal01/inventory_after_transport.json', '4303d7d4ef775ac133dde12b3d2ac875b7c31ed2382b45abd0c6cc6a55a188ac'),
    'readset': (PREP+'/readset01/cpu-history-readset.json', '9be8223034cadedb0a3ff885236f923519b81f8c98bd46a7ffbcd2dafd9812b0'),
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def raw(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source-commit', 'source-directory', 'entry-sha256', 'history-sha256', 'source-review', 'source-review-sha256', 'output'):
        p.add_argument('--'+n, required=True)
    a = p.parse_args()
    need(re.fullmatch('[0-9a-f]{40}', a.source_commit), 'full committed source identity required')
    source_dir = Path(a.source_directory)
    need(not source_dir.is_absolute() and '..' not in source_dir.parts, 'repository-relative source directory required')
    review = Path(a.source_review).read_bytes()
    need(sha(review) == a.source_review_sha256, 'independent source review bytes differ')
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS='1', GIT_NO_LAZY_FETCH='1')
    need(subprocess.check_output(['git', '--no-replace-objects', 'cat-file', '-t', a.source_commit], cwd=REPO, env=env).strip() == b'commit', 'source is not a commit')
    source_files = {}
    for name, pin in ((ENTRY, a.entry_sha256), (CHECKER, 'b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8'), (HISTORY, a.history_sha256)):
        need(re.fullmatch('[0-9a-f]{64}', pin), 'independent source pin required')
        rel = (source_dir/name).as_posix()
        data = subprocess.check_output(['git', '--no-replace-objects', 'show', a.source_commit+':'+rel], cwd=REPO, env=env)
        need(sha(data) == pin, 'committed operational source differs: '+name)
        source_files[name] = dict(repository_path=rel, sha256=pin)
    captured = {}
    for name, (rel, pin) in PINS.items():
        data = (REPO/rel).read_bytes()
        need(sha(data) == pin, 'independent metadata pin differs: '+name)
        captured[name] = data
    prior = json.loads(captured['old_packet'])
    accepted = json.loads(captured['failed_acceptance'])
    base = json.loads(captured['baseline'])
    need(accepted['status'] == 'accepted_failed_prepolicy_attempt' and accepted['required_findings'] == 0 and accepted['original_collector_returncode'] == 1 and accepted['new_policy_attempts'] == accepted['new_scientific_responses'] == 0, 'closed prepolicy failure required')
    need((base['terminal_jobs'], base['submission_journals'], base['closed_cpu_seconds'], base['closed_gpu_slice_seconds'], base['outstanding_reservation']) == (47, 96, 30241, 2750, 0), 'exact closed failed-attempt baseline required')
    sources = {n: prior['files'][n] for n in prior['source_names']}
    sources.update({n: v['sha256'] for n, v in source_files.items()})
    authority = dict(
        schema='ace-gpu-cpu-history-recovery-authority-v1', status='operationally_adopted',
        historical_postcustody_required=True, unchanged_scientific_inputs=True,
        original_entry_sha256='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1',
        original_source_commit='a9346022f145cc29439325b6f291597b2e6ec037',
        original_gpu01_packet_sha256='436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7',
        original_gpu01_job_id='35498218', control_root=ROOT, allowed_new_output=OLDROOT+'/gpu',
        failed_gpu01_root=OLDROOT, baseline=dict(sha256=PINS['baseline'][1], terminal_jobs=47, submission_journals=96),
        failed_gpu01_acceptance_sha256=PINS['failed_acceptance'][1], cpu_acceptance_sha256=PINS['cpu_acceptance'][1],
        namespace_sha256=PINS['namespace'][1], original_namespace_sha256='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29',
        historical_verifier_sha256='0ecf0f979f4df6f8dee07a007e2ce91c52246d3b251b0d9c3ec5cd0464b5a1bc',
        historical_proof_pins=dict(custody_sha256=PINS['custody'][1], before_observation_sha256=PINS['before'][1], after_observation_sha256=PINS['after'][1], read_set_artifact_sha256=PINS['readset'][1]),
        operational_source_files=sources,
        source_review_sha256=a.source_review_sha256,
        receipt_not_after_utc=datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
        scope='Routine CPU-first host verification and exclusive absent output creation only. Original scientific inputs, methods, resources and clocks remain fixed. Complete external historical postcustody and independent GPU acceptance remain mandatory.')
    manifest = dict(schema='ace-gpu02-operational-sources-v1', source_commit=a.source_commit, files=source_files)
    out = Path(a.output)
    need(out.is_absolute() and out.parent == out.parent.resolve(strict=True), 'ordinary absolute exclusive output required')
    out.mkdir()
    for name, obj in (('gpu-recovery-authority.json', authority), ('operational-source-manifest.json', manifest)):
        with (out/name).open('xb') as f:
            f.write(raw(obj))
    (out/'input-pins.json').write_bytes(raw({n: dict(repository_path=p, sha256=h) for n, (p, h) in PINS.items()}))
    print(json.dumps({n: sha((out/n).read_bytes()) for n in ('gpu-recovery-authority.json', 'operational-source-manifest.json')}))


if __name__ == '__main__':
    main()
