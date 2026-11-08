#!/usr/bin/env python3
"""PREPARED ONLY. Future cpu01 packet builder; no execution/staging/submission.

Require the eventual reviewed commit and independent source/review/evidence
pins. After building, obtain and commit the concrete packet review before the
separate launcher. Partial destinations are retained; never rerun blindly.
"""
from pathlib import Path
import argparse, ast, hashlib, json, os, re, shlex, subprocess, types

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL = REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/launch01'
EXT = BASE/'ACE-LLM-intervention-screen-prompt-cpu-20261108-01'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01'
FINAL = REPO/'results/ace_llm_intervention_campaign_finalization_20261108'
FINALROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
CPUROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
FIXTURES = REPO/'results/ace_llm_intervention_screen_prompt_preparation_20261108/preparation01/cli-packet'
ENTRY = 'qualify_llm_intervention_screen_prompt_cpu.py'
FINALIZER = 'finalize_llm_intervention_campaign_cpu.py'
JOB = '35389391'
RESOURCES = dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal', cpus=2,
                 memory_mib=7680, wall_seconds=300, gpus=0, requeue=False)
ADMISSION = dict(closed_cpu_seconds=10368, closed_gpu_slice_seconds=1483,
    cumulative_preparation_cpu_allowance=3600, new_cpu_reservation=600,
    new_gpu_slice_reservation=0, cpu_admission=14568, gpu_slice_admission=1483,
    cpu_ceiling=57600, gpu_slice_ceiling=7200)

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(value): return hashlib.sha256(value).hexdigest()
def raw(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def git(revision, path): return subprocess.check_output(['git', 'show', revision+':'+path], cwd=REPO)
def literal(source, name):
    values = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    need(len(values) == 1, 'one literal contract assignment required: '+name)
    return ast.literal_eval(values[0])
def module(name, value, path):
    result = types.ModuleType(name); result.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), vars(result)); return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-commit', 'source-sha256', 'source-review', 'source-review-sha256',
                 'final-acceptance-sha256', 'final-disposition-sha256', 'final-review-receipt-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--final-acceptance', default=str(FINAL/'acceptance01/acceptance.json'))
    parser.add_argument('--final-disposition', default=str(FINAL/'terminal01/disposition.json'))
    parser.add_argument('--final-review-receipt', default=str(FINAL/'acceptance01/review_receipt.json'))
    parser.add_argument('--launcher', default='/tmp/ace-launch-screen-prompt-cpu01-20261108.py')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', args.source_commit+'^{commit}'], cwd=REPO, text=True).strip()
    need(re.fullmatch('[0-9a-f]{40}', revision), 'resolved committed revision required')
    source = git(revision, 'scripts/research/'+ENTRY)
    need(sha(source) == args.source_sha256 and literal(source, 'ENTRY') == ENTRY, 'reviewed worker source differs')
    helpers = literal(source, 'HELPER_PINS')
    base_raw = git(revision, 'scripts/research/'+FINALIZER)
    need(sha(base_raw) == helpers[FINALIZER], 'accepted supervisor helper differs')
    f = module('screen_builder_bytes', base_raw, REPO/'scripts/research'/FINALIZER)
    need(f.RESOURCES == RESOURCES and (f.STARTUP, f.WORK, f.CLEANUP, f.HARD) == (120, 260, 280, 290),
         'fixed resource/deadline contract differs')
    source_review = f.read(args.source_review, args.source_review_sha256)
    need(source_review.strip() and git(revision, str(Path(args.source_review).resolve().relative_to(REPO))) == source_review,
         'independently pinned source review must be committed with source')

    # Closed predecessor custody and authority: opaque byte/inventory checks,
    # not another run of its finalizer, evaluator, or old tests.
    disposition_raw = f.read(args.final_disposition, args.final_disposition_sha256)
    acceptance_raw = f.read(args.final_acceptance, args.final_acceptance_sha256)
    review_raw = f.read(args.final_review_receipt, args.final_review_receipt_sha256)
    disposition, acceptance, review = map(f.decode, (disposition_raw, acceptance_raw, review_raw))
    need(acceptance['schema'] == 'ace-artificial-campaign-finalization-acceptance-v1'
        and acceptance['status'] == 'artificial_finalization_accepted' and acceptance['job_id'] == JOB
        and acceptance['scientific_qualified'] is False and acceptance['artificial_qualified'] is True
        and all(acceptance['accepted_checks'].get(k) is True for k in ('scheduler', 'runtime', 'cleanup', 'custody', 'source')),
        'independent accepted finalization required')
    need(review['acceptance_sha256'] == sha(acceptance_raw)
        and review['allocation_cost']['terminal_jobs'] == 36 and review['allocation_cost']['submission_journals'] == 74
        and review['allocation_cost']['closed_cpu_seconds'] == 10368
        and review['allocation_cost']['closed_gpu_slice_seconds'] == 1483, 'final review accounting differs')
    need(disposition['job_id'] == JOB and disposition['scheduler_state'] == 'COMPLETED'
        and disposition['scheduler_exit'] == '0:0' and disposition['account'] == 'ucb736_asc2'
        and disposition['resource_contract_matched'] is True and disposition['accounting_errors'] == []
        and disposition['input_pin_mismatches'] == [] and disposition['gpu_count'] == 0
        and (disposition['allocated_cpus'], disposition['elapsed_seconds'], disposition['allocated_cpu_seconds']) == (2, 61, 122)
        and disposition['closed_cpu_seconds'] == 10368 and disposition['closed_gpu_slice_seconds'] == 1483
        and disposition['cpu_admission_after_release'] == 13968, 'terminal resource closure differs')
    remote = FINAL/'terminal01/remote'
    inventory_raw = f.read(remote/'launch/inputs/qualify_llm_intervention_campaign_gpu.py', f.GPU_SHA)
    api = module('screen_custody_inventory', inventory_raw, remote/'launch/inputs/qualify_llm_intervention_campaign_gpu.py')
    need(api.inventory(remote) == {n: v['sha256'] for n, v in disposition['files'].items()}
        and all((remote/n).stat().st_size == v['bytes'] for n, v in disposition['files'].items()), 'complete retained custody differs')
    previous_raw = f.read(remote/'launch/inputs/packet.json', acceptance['finalizer_packet_sha256'])
    previous = f.decode(previous_raw)
    need(api.inventory(remote/'launch/inputs') == dict(previous['files'], **{'packet.json': sha(previous_raw)}), 'final01 input roster differs')
    for name, pin in previous['files'].items(): f.read(remote/'launch/inputs'/name, pin)
    attempt = f.decode(f.read(remote/'launch/finalization/attempt.json'))
    terminal = f.decode(f.read(remote/'launch/finalization/worker/terminal.json'))
    need(terminal['status'] == 'descriptive_finalization_complete'
        and terminal['packet_sha256'] == sha(previous_raw)
        and attempt['status'] == 'completed' and attempt['error'] is None
        and attempt['packet_sha256'] == sha(previous_raw) and attempt['result']['job_id'] == JOB
        and attempt['result']['report'] == terminal['report']
        and terminal['report']['finalization_sha256'] == acceptance['finalization_sha256']
        and terminal['report']['summary_sha256'] == acceptance['summary_sha256'], 'accepted finalizer output differs')
    transport = f.decode(f.read(FINAL/'terminal01/inventory_transport.json'))
    observed = f.decode(transport['stdout']); row = observed['parent_accounting']
    need(transport['returncode'] == 0 and observed['files'] == disposition['files']
        and observed['results']['accounting']['returncode'] == 0
        and row[:9] == [JOB, JOB, 'ace_llm_campaign_final01', 'COMPLETED', '0:0', '2', '61',
                        'ucb736_asc2', disposition['alloc_tres']], 'actual parent scheduler row differs')
    intent_raw, result_raw = (f.read(remote/'launch'/n) for n in ('submission.intent.json', 'submission.result.json'))
    intent, result = f.decode(intent_raw), f.decode(result_raw)
    need(result['returncode'] == 0 and result['job_id'] == JOB and result['intent_sha256'] == sha(intent_raw)
        and intent['remote_root'] == FINALROOT and intent['finalizer_packet_sha256'] == sha(previous_raw)
        and len(intent['prior_accounting']) == 35 and len(intent['prior_journals']) == 72
        and intent['closed_cpu_seconds'] == 10246 and intent['closed_gpu_slice_seconds'] == 1483,
        'finalizer once-only submission lineage differs')
    for name, pin in intent['pins'].items():
        need(f.relative(name), 'unsafe original launch name'); f.read(remote/'launch'/name, pin)
    rows = [r[:8] for r in intent['prior_accounting']] + [[row[0], *row[2:9]]]
    journals = {**intent['prior_journals'], FINALROOT+'/submission.intent.json': sha(intent_raw),
                FINALROOT+'/submission.result.json': sha(result_raw)}
    need(len(rows) == len({r[0] for r in rows}) == 36 and len(journals) == 74
        and sum(int(r[4])*int(r[5]) for r in rows) == 10368, '36-job/74-journal reconciliation differs')

    # Capture every source from the requested committed revision. probe.py is
    # the original captured helper under its committed accepted-packet path.
    data, origins = {}, {}
    def capture(name, path):
        value = git(revision, path); data[name] = value; origins[name] = path; return value
    service_name = literal(source, 'SERVICE')
    service_raw = capture(service_name, 'scripts/research/'+service_name)
    need(sha(service_raw) == helpers[service_name], 'accepted service contract differs')
    service = module('screen_service_contract', service_raw, REPO/'scripts/research'/service_name)
    sources = sorted(service.SOURCE_NAMES | {ENTRY})
    for name in sorted(set(sources) | set(helpers)):
        path = ('results/ace_llm_intervention_campaign_finalization_20261108/launch01/inputs/probe.py'
                if name == 'probe.py' else 'scripts/research/'+name)
        value = capture(name, path)
        if name in helpers: need(sha(value) == helpers[name], 'reviewed helper differs: '+name)
    need(data[ENTRY] == source, 'worker capture changed')
    manifest_raw = f.read(FIXTURES/'manifest.json', literal(source, 'FIXTURE_MANIFEST_SHA'))
    need(git(revision, str((FIXTURES/'manifest.json').relative_to(REPO))) == manifest_raw, 'retained manifest differs from commit')
    manifest = f.decode(manifest_raw)
    need(len(manifest['fixtures']) == len({r['fixture_id'] for r in manifest['fixtures']}) == 120
        and manifest['actual_tokenization_performed'] is False, 'exact untouched fixture roster required')
    data['fixture-manifest.json'] = manifest_raw
    for fixture in manifest['fixtures']:
        for key in ('envelope_file', 'prompt_file'):
            item = fixture[key]; name = item['path']
            need(Path(name).name == name and name not in data, 'unique flat fixture name required')
            value = f.read(FIXTURES/name, item['sha256'])
            need(len(value) == item['utf8_bytes'] and git(revision, str((FIXTURES/name).relative_to(REPO))) == value,
                 'retained committed fixture bytes differ: '+name)
            data[name] = value
    need(len(data) == 264, 'exact 264 worker inputs required')
    old = f.decode(f.read(remote/'launch/inputs/cpu-packet.json', previous['cpu_packet_sha256']))
    campaign = module('screen_model_contract', data[literal(source, 'CAMPAIGN')], REPO/'scripts/research'/literal(source, 'CAMPAIGN'))
    diagnostic = module('screen_bootstrap', data[literal(source, 'DIAGNOSTIC')], REPO/'scripts/research'/literal(source, 'DIAGNOSTIC'))
    need(old['runtime'] == service.RUNTIME and old['runtime_source'] == diagnostic.RUNTIME_SOURCE
        and old['model_files'] == campaign.MODEL_FILES and old['archive']['sha256'] == diagnostic.ARCHIVE_SHA
        and old['manifest']['sha256'] == diagnostic.MANIFEST_SHA, 'accepted model/archive/runtime differs')
    packet = dict(schema=literal(source, 'VERSION'), scope='artificial', request_mode='all_120_frozen_fixtures',
        source_commit=revision, source_names=sources, files={n: sha(v) for n, v in sorted(data.items())},
        fixture_manifest_sha256=sha(manifest_raw), archive=old['archive'], manifest=old['manifest'],
        runtime_source=old['runtime_source'], runtime=old['runtime'], model_directory=old['model_directory'],
        model_files=old['model_files'], resources=RESOURCES, output=ROOT+'/cpu01')
    need(set(packet) == literal(source, 'PACKET_KEYS'), 'reviewed worker packet schema changed')
    data['packet.json'] = raw(packet)
    command = [old['runtime_source']+'/bin/python', '-I', '-B', '-c', diagnostic.BOOTSTRAP,
        ROOT+'/inputs/'+ENTRY, sha(source), '--packet', ROOT+'/inputs/packet.json', '--packet-sha256', sha(data['packet.json']),
        '--source-commit', revision, '--output', packet['output']]
    job = ('#!/bin/bash\n#SBATCH --job-name=ace_llm_screen_prompt_cpu01\n'
        '#SBATCH --account=ucb736_asc2\n#SBATCH --partition=acpu\n#SBATCH --qos=cpu-normal\n'
        '#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=7680M\n'
        '#SBATCH --time=00:05:00\n#SBATCH --no-requeue\n#SBATCH --output='+ROOT+'/logs/%j.out\n'
        '#SBATCH --error='+ROOT+'/logs/%j.err\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=\'\'\n'
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU\n'
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\n'
        'unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL\nexec '+shlex.join(command)+'\n').encode()
    provenance = dict(source_commit=revision, worker_sha256=sha(source), source_origins=origins,
        source_review_sha256=sha(source_review), final_acceptance_sha256=sha(acceptance_raw),
        final_disposition_sha256=sha(disposition_raw), final_review_receipt_sha256=sha(review_raw),
        admission=ADMISSION, deadlines=dict(startup=120, work=260, cleanup=280, hard=290),
        resource_rationale='Bounded CPU-only measurement of 120 fixtures; no extrapolated token or GPU-memory claim. '
            'Same archive final01 took 61 seconds/2 CPUs; its close memory margin is not a tokenization qualification.',
        model_scope='Existing seven-file directory authenticated by worker; weights opaque-hashed, never loaded or copied into inputs.',
        scientific_qualified=False, actual_tokenization_performed=False, retry_allowed=False)
    artifacts = {'job.sbatch': job, 'intent-provenance.json': raw(provenance),
        'source-review.txt': source_review, 'final-acceptance.json': acceptance_raw,
        'final-disposition.json': disposition_raw, 'final-review-receipt.json': review_raw,
        'final-submission.intent.json': intent_raw, 'final-submission.result.json': result_raw,
        'custody_inventory.py': inventory_raw, 'build_packet.py': Path(__file__).read_bytes(),
        'launch_once.py': Path(args.launcher).read_bytes()}
    meta = dict(source_commit=revision, source_origins=origins, worker_sha256=sha(source),
        remote_root=ROOT, input_directory=ROOT+'/inputs', output=packet['output'],
        screen_packet_sha256=sha(data['packet.json']), fixture_manifest_file_sha256=sha(manifest_raw),
        fixture_manifest_canonical_sha256=sha(f.raw(manifest)[:-1]),
        input_files={n: sha(v) for n, v in data.items()}, root_files={n: sha(v) for n, v in artifacts.items()},
        resources=RESOURCES, admission=ADMISSION, deadlines=provenance['deadlines'],
        expected_prior_jobs=36, expected_prior_submission_journals=74,
        prior_accounting=sorted(rows, key=lambda r: int(r[0])), prior_journals=journals,
        predecessor_roots={'launch': FINALROOT, 'common': CPUROOT})
    artifacts['packet-metadata.json'] = raw(meta)
    for dest in (EXT, LOCAL): need(not dest.exists() and not dest.is_symlink(), 'exclusive destination already exists: '+str(dest))
    for dest in (EXT, LOCAL):
        dest.mkdir(parents=True, exist_ok=False); (dest/'inputs').mkdir()
        for name, value in {**artifacts, **{'inputs/'+n: v for n, v in data.items()}}.items():
            with (dest/name).open('xb') as handle: handle.write(value); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(dict(local=str(LOCAL), external=str(EXT), packet_sha256=meta['screen_packet_sha256'],
        worker_inputs=264, admission=ADMISSION, status='unsubmitted; concrete packet review and commit required'), indent=2))

if __name__ == '__main__': main()
