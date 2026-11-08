#!/usr/bin/env python3
"""Future GPU01 packet builder. SOURCE ONLY until explicitly invoked by the parent.

Only the standard library is imported, even when building. Project/model modules
are never imported or executed. This file is also the coordinator's pinned,
read-only admission library. main() is the only local packet-writing entry.

Required independent inputs:
  * CPU acceptance: the EXACT ace-paired-campaign-cpu-acceptance-v1 object;
    never append cost/review fields to it.
  * CPU disposition: the separately pinned collector disposition.json.
  * admission review (ace-campaign-gpu01-admission-review-v1): status
    independently_accepted, required_findings=0, reviewer, reviewed_at,
    cpu_acceptance_sha256, cpu_disposition_sha256, cpu_packet_sha256,
    cpu_source_commit, cpu_packet_commit, job_id, prior_accounting (34 exact
    eight-column rows in ACCOUNTING_FORMAT order), prior_journals (70 absolute
    remote paths -> SHA256), expected_queue (all nonempty squeue lines),
    cpu03_details (Partition,QOS,TimelimitRaw,ReqTRES,User row), and site (exact
    partition/association/qos stdout from an independently reviewed snapshot).
  * source review (ace-campaign-gpu01-source-review-v1): status
    independently_accepted, required_findings=0, reviewer, reviewed_at,
    source_commit, sources (exact EXTRA_SOURCES -> hashes), builder_sha256,
    coordinator_sha256, documents (repo-relative reviewed guidance/reviews ->
    hashes). The source review covers these concrete builder/coordinator bytes.

The caller supplies every receipt SHA separately. Neither this builder nor the
launcher manufactures independent acceptance/review. Snapshot preparation is a
separate CPU/read-only operation. Receipt formats above are integration contracts
to agree with the parent reviewer, not claims that those receipts already exist.

Future launch additionally needs a committed concrete packet and separately
committed packet review; see the coordinator docstring. Failed/partial output is
retained. No overwrite, retry, qualification, remote call or submission here.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LOCAL = REPO / 'results/ace_llm_intervention_campaign_gpu_20261108/launch01'
EXTERNAL = BASE / 'ACE-LLM-intervention-campaign-gpu-20261108-01'
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_gpu_20261108_01'
CPU_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
CPU_INPUT = CPU_ROOT + '/inputs'
CPU_OUTPUT = CPU_ROOT + '/cpu03'
CAMPAIGN = CPU_OUTPUT + '/worker/artifacts/campaign'
GPU_OUTPUT = CPU_ROOT + '/gpu03'
CPU_LOCAL = REPO / 'results/ace_llm_intervention_campaign_artificial_20261108/cpu03'
CPU_PACKET_SHA = '0533b4fad722652ec63ed12850d3a3d55d1f6c391e81f7280d36561aa9e5ac13'
CPU_SOURCE_COMMIT = '183ce3137213dcc1e4f33855bfab7b20dd307bb0'
CPU_PACKET_COMMIT = '70d9ee060d10ed087ebf222e1ac3d5b9836b0fd1'
CPU_JOB = '35387068'
ENTRY = 'qualify_llm_intervention_campaign_gpu.py'
EXTRA_SOURCES = {ENTRY, 'qualify_llm_intervention_model_service.py',
    'supervise_llm_intervention_checkpoint_gpu.py',
    'llm_intervention_campaign_gpu_phase.py', 'llm_intervention_campaign_service_evidence.py'}
GPU_POLICIES = ('frozen_llm', 'sft_llm', 'dpo_llm')
CPU_POLICIES = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker')
RESOURCES = {'account':'ucb736_asc2', 'partition':'aa100', 'qos':'gpu-testing',
    'requested_cpus':4, 'max_allocated_cpus':8, 'memory_mib':15360,
    'wall_seconds':600, 'gpus':1, 'gpu_type':'a100_3g.20gb', 'requeue':False}
LINEAGE_FILES = {'acceptance_raw':'gpu03-acceptance.json', 'prepared_raw':'prepared.json',
    'engine_terminal_raw':'engine-terminal.json', 'sft_checkpoint_raw':'sft.checkpoint.json',
    'sft_adapter_raw':'sft.adapter.pt', 'final_checkpoint_raw':'final.checkpoint.json',
    'final_adapter_raw':'final.adapter.pt'}
PATHS = {'cpu_inputs':CPU_INPUT, 'cpu_output':CPU_OUTPUT, 'campaign':CAMPAIGN, 'gpu_output':GPU_OUTPUT}
ACCOUNTING_FORMAT = 'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180'
DETAIL_FORMAT = 'Partition,QOS,TimelimitRaw,ReqTRES%180,User'
TERMINAL_STATES = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}
GUIDANCE = 'docs/development/guidance/ace_llm_intervention_paired_campaign_integration_2026-11-08.md'
REVIEW_DOCS = {GUIDANCE,
    'docs/development/guidance/reviews/ace_llm_intervention_campaign_gpu_phase_review_2026-11-08.md',
    'docs/development/guidance/reviews/ace_llm_intervention_campaign_gpu_wrapper_review_2026-11-08.md'}
ACCEPTANCE_FIELDS = {'schema','status','cpu_packet_sha256','job_id','allocation',
    'cpu_output','campaign_root','cpu_output_files','cpu_output_excluded_directory','campaign_files',
    'bundle_path','bundle_sha256','phase_terminal_path','phase_terminal_sha256','attempt_pins','scientific_qualified'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def is_pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def decode(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            need(key not in result, 'duplicate JSON key: ' + key)
            result[key] = item
        return result
    result = json.loads(value, object_pairs_hook=unique)
    raw(result)
    return result


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def relative(name):
    need(type(name) is str and name and not Path(name).is_absolute()
         and str(Path(name)) == name and '..' not in Path(name).parts, 'unsafe relative path')
    return name


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        need(not item.is_symlink(), 'symlink path: ' + str(item))


def read(path, expected=None, maximum=32*1024**2):
    path = Path(path)
    no_links(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'bounded regular file required: '+str(path))
        value = handle.read(maximum+1)
        after = os.fstat(handle.fileno())
    need(len(value) <= maximum and (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
         (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'file changed during read')
    if expected is not None:
        need(is_pin(expected) and sha(value) == expected, 'independent pin differs: '+str(path))
    return value


def inventory(root, exclude=None):
    root = Path(root)
    no_links(root)
    need(root.is_dir(), 'custody directory missing: '+str(root))
    result, total, count = {}, 0, 0
    excluded = root/exclude if exclude else None
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            p = Path(directory)/name
            need(not p.is_symlink(), 'symlink custody directory')
            if p == excluded:
                dirs.remove(name)
        for name in files:
            p = Path(directory)/name
            value = read(p)
            count += 1
            total += len(value)
            need(count <= 8192 and total <= 256*1024**2, 'custody inventory exceeds bound')
            result[str(p.relative_to(root))] = sha(value)
    return dict(sorted(result.items()))


def save_bytes(path, value):
    no_links(Path(path).parent)
    with Path(path).open('xb') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def git_bytes(commit, name):
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'full explicit commit required')
    relative(name)
    return subprocess.check_output(['git','show',commit+':'+name], cwd=REPO)


def literal(source, name):
    """Extract a literal constant with AST; NEVER execute captured project code."""
    values = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    need(len(values) == 1, 'one literal source constant required: '+name)
    return ast.literal_eval(values[0])


def tres(value):
    result = {}
    for part in value.split(',') if value else []:
        key, sep, val = part.partition('=')
        need(sep and key not in result and val, 'malformed/duplicate TRES')
        result[key] = val
    return result


def gpu_count(values, cpus, elapsed):
    gpu = {k:v for k,v in values.items() if k.startswith('gres/')}
    need(all(k == 'gres/gpu' or k.startswith('gres/gpu:') for k in gpu), 'unrecognized GRES accounting')
    need(all(v.isdigit() for v in gpu.values()), 'unparseable GPU count')
    typed = sum(int(v) for k,v in gpu.items() if k.startswith('gres/gpu:'))
    generic = int(gpu['gres/gpu']) if 'gres/gpu' in gpu else None
    need(generic is None or not any(k.startswith('gres/gpu:') for k in gpu) or generic == typed,
         'conflicting generic/typed GPU counts')
    need(values or cpus == elapsed == 0, 'missing allocation is uncertainty, not zero GPUs')
    return generic if generic is not None else typed


def table(text, width):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = line.split('|')
        if row[-1] == '':
            row.pop()
        need(len(row) == width, 'scheduler table width differs')
        rows.append(row)
    return rows


def review_shape(value, schema):
    need(value.get('schema') == schema and value.get('status') == 'independently_accepted'
         and type(value.get('required_findings')) is int and value['required_findings'] == 0
         and type(value.get('reviewer')) is str and value['reviewer'].strip()
         and type(value.get('reviewed_at')) is str and value['reviewed_at'].strip(), 'independent review required')


def admission(acceptance, disposition, review, acceptance_pin, disposition_pin):
    """Derive cost from reviewed disposition, not from exit status or acceptance."""
    review_shape(review, 'ace-campaign-gpu01-admission-review-v1')
    need(review['cpu_acceptance_sha256'] == acceptance_pin
         and review['cpu_disposition_sha256'] == disposition_pin
         and review['cpu_packet_sha256'] == CPU_PACKET_SHA
         and review['cpu_source_commit'] == CPU_SOURCE_COMMIT
         and review['cpu_packet_commit'] == CPU_PACKET_COMMIT
         and review['job_id'] == CPU_JOB == acceptance['job_id'] == disposition['job_id'],
         'independently reviewed CPU03 identity differs')
    need(set(acceptance) == ACCEPTANCE_FIELDS
         and acceptance['schema'] == 'ace-paired-campaign-cpu-acceptance-v1'
         and acceptance['status'] == 'artificial_cpu_phase_accepted'
         and acceptance['scientific_qualified'] is False
         and acceptance['cpu_packet_sha256'] == CPU_PACKET_SHA
         and acceptance['allocation'] == {'account':'ucb736_asc2','partition':'acpu','qos':'cpu-normal',
             'cpus':2,'memory_mib':7680,'gpus':0,'state':'COMPLETED','exit_code':'0:0'},
         'separate strict CPU03 acceptance required')
    need(disposition['scheduler_state'] == 'COMPLETED' and disposition['scheduler_exit'] == '0:0'
         and disposition['observed_account'] == 'ucb736_asc2'
         and type(disposition['observed_cpus']) is int and disposition['observed_cpus'] == 2
         and type(disposition['observed_gpu_count']) is int and disposition['observed_gpu_count'] == 0
         and disposition['resource_contract_matched'] is True and disposition['accounting_errors'] == []
         and disposition['input_pin_mismatches'] == [], 'reviewed CPU disposition failed accounting/input admission')
    elapsed = disposition['observed_elapsed_seconds']
    need(type(elapsed) is int and 0 < elapsed <= 600, 'bounded CPU03 duration required')
    cpu_cost = 2*elapsed
    need(disposition['allocated_cpu_seconds'] == cpu_cost and disposition['gpu_slice_seconds'] == 0,
         'CPU03 disposition costs inconsistent')
    rows = review['prior_accounting']
    need(type(rows) is list and len(rows) == 34 and all(type(r) is list and len(r) == 8 for r in rows),
         '34 complete independently reviewed accounting rows required')
    ids, old_cpu, old_gpu, current = set(), 0, 0, []
    for row in rows:
        job, name, state_, exit_, cpus_, elapsed_, account, alloc = row
        need(re.fullmatch('[1-9][0-9]*', job) and job not in ids and name.startswith('ace_llm')
             and state_.split()[0] in TERMINAL_STATES and cpus_.isdigit() and elapsed_.isdigit(),
             'ambiguous/nonterminal predecessor accounting')
        ids.add(job)
        cpus, seconds = int(cpus_), int(elapsed_)
        values = tres(alloc)
        gpus = gpu_count(values, cpus, seconds)
        if job == CPU_JOB:
            current.append(row)
            need(name == 'ace_llm_campaign_cpu03' and state_ == 'COMPLETED' and exit_ == '0:0'
                 and cpus == 2 and seconds == elapsed and account == 'ucb736_asc2'
                 and alloc == disposition['observed_alloc_tres'] and values.get('cpu') == '2'
                 and values.get('mem') in ('7680M','7.5G','7.50G') and values.get('node') == '1'
                 and gpus == 0 and not any(k.startswith('gres/') for k in values), 'CPU03 reviewed row differs')
        else:
            old_cpu += cpus*seconds
            old_gpu += gpus*seconds
    need(len(current) == 1 and old_cpu == 9464 and old_gpu == 1347, 'pre-CPU03 closed baseline changed')
    journals = review['prior_journals']
    need(type(journals) is dict and len(journals) == 70, '70 exact prior journals required')
    for name, pin in journals.items():
        p = Path(name)
        need(str(p) == name and '..' not in p.parts and str(p.parent.parent) == str(Path(ROOT).parent)
             and p.parent.name.startswith('ace_llm_intervention') and p.name.startswith('submission.')
             and p.name.endswith('.json') and is_pin(pin) and str(p.parent) != ROOT, 'invalid prior journal pin')
    need(all(CPU_ROOT+'/'+n in journals for n in ('submission.intent.json','submission.result.json')),
         'CPU03 original submission journals must be in reviewed baseline')
    need(type(review['expected_queue']) is list and all(type(x) is str for x in review['expected_queue'])
         and not any('|ace_llm' in line for line in review['expected_queue']), 'reviewed queue has related active work')
    detail = review['cpu03_details']
    need(type(detail) is list and len(detail) == 5 and detail[:3] == ['acpu','cpu-normal','10']
         and detail[4] == 'paco0228' and tres(detail[3]).get('cpu') == '2'
         and tres(detail[3]).get('mem') in ('7680M','7.5G','7.50G')
         and not any(k.startswith('gres/') for k in tres(detail[3])), 'CPU03 requested scheduler resources differ')
    need(set(review['site']) == {'partition','association','qos'}
         and all(type(v) is str and v.strip() for v in review['site'].values()), 'reviewed site snapshot required')
    closed = old_cpu + cpu_cost
    need(disposition['known_closed_cpu_seconds'] == closed
         and disposition['cpu_admission_after_release'] == closed+3600
         and disposition['gpu_admission_seconds'] == old_gpu, 'disposition cumulative totals differ')
    result = {'cpu03_job_id':CPU_JOB,'cpu03_allocated_cpu_seconds':cpu_cost,'cpu03_gpu_slice_seconds':0,
        'closed_cpu_seconds':closed,'closed_gpu_slice_seconds':old_gpu,'cumulative_preparation_cpu_allowance':3600,
        'new_cpu_reservation':4800,'new_gpu_slice_reservation':600,'cpu_admission':closed+3600+4800,
        'gpu_slice_admission':old_gpu+600,'cpu_ceiling':57600,'gpu_slice_ceiling':7200,
        'expected_prior_jobs':34,'expected_prior_submission_journals':70}
    need(result['cpu_admission'] <= 57600 and result['gpu_slice_admission'] <= 7200, 'admission ceiling exceeded')
    return result


def check_cpu_evidence(old, data, acceptance, input_root, output_root):
    """Byte-only admission; paths may name a retained local copy or live originals."""
    input_root, output_root = Path(input_root), Path(output_root)
    need(sha(data['cpu-packet.json']) == CPU_PACKET_SHA and old['cpu_output'] == CPU_OUTPUT,
         'fixed CPU03 packet/output changed')
    need(acceptance['cpu_output'] == CPU_OUTPUT and acceptance['campaign_root'] == CAMPAIGN
         and acceptance['cpu_output_excluded_directory'] == 'worker/artifacts/campaign', 'CPU evidence layout differs')
    expected_inputs = dict(old['files'], **{'packet.json':CPU_PACKET_SHA})
    need(inventory(input_root) == expected_inputs, 'original CPU inputs changed or gained files')
    for name, pin in old['files'].items():
        need(data[name] == read(input_root/name, pin), 'CPU inputcopy is not original bytes: '+name)
    need(read(input_root/'packet.json', CPU_PACKET_SHA) == data['cpu-packet.json'], 'original CPU packet changed')
    campaign = output_root/'worker/artifacts/campaign'
    need(inventory(output_root, 'worker/artifacts/campaign') == acceptance['cpu_output_files'],
         'accepted CPU output inventory changed')
    need(inventory(campaign) == acceptance['campaign_files'], 'accepted campaign inventory changed')
    need(not any((campaign/('policy-%02d'%i)).exists() for i in (5,6,7))
         and not (campaign/'finalization-intent.json').exists(), 'campaign GPU/finalization already claimed')
    bundle = decode(read(output_root/relative(acceptance['bundle_path']), acceptance['bundle_sha256']))
    terminal = decode(read(output_root/relative(acceptance['phase_terminal_path']), acceptance['phase_terminal_sha256']))
    need(set(acceptance['attempt_pins']) == set(CPU_POLICIES)
         and all(is_pin(p) for p in acceptance['attempt_pins'].values()), 'complete independent CPU attempt pins required')
    need(terminal['status'] == 'cpu_phase_complete' and terminal['failed_policies'] == []
         and terminal['attempt_pins'] == acceptance['attempt_pins']
         and terminal['campaign_sha256'] == bundle['campaign_sha256']
         and terminal['numeric_training_updates'] == 6 and terminal['llm_training_updates'] == 0
         and terminal['gpu_services_started'] == 0 and terminal['private_scores_computed'] is False,
         'independently accepted CPU phase is incomplete')
    # The worker now distinguishes numerical weights from LLM weights. Do not
    # impose the superseded generic model_weights_loaded=False condition.
    need(bundle['campaign']['policy_ids'] == list(CPU_POLICIES+GPU_POLICIES)
         and bundle['campaign']['scope'] == 'artificial' and len(bundle['campaign']['sources']) == 2,
         'fixed artificial campaign differs')
    for i, policy in enumerate(CPU_POLICIES):
        read(campaign/('policy-%02d'%i)/'attempt.json', acceptance['attempt_pins'][policy])
    return bundle


def packet_shape(packet, data, bundle):
    need(set(packet) == {'schema','files','source_names','cpu_packet_sha256','cpu_acceptance_sha256',
        'paths','resources','services','contracts','lineage_files'}
        and packet['schema'] == 'ace-artificial-paired-campaign-gpu-v1'
        and packet['resources'] == RESOURCES and packet['paths'] == PATHS, 'exact outer GPU packet shape differs')
    old = decode(data['cpu-packet.json'])
    need(packet['cpu_packet_sha256'] == CPU_PACKET_SHA
         and packet['cpu_acceptance_sha256'] == sha(data['cpu-acceptance.json']), 'GPU predecessor pins differ')
    expected = set(old['files']) | EXTRA_SOURCES | {'cpu-packet.json','cpu-acceptance.json'}
    need(set(data) == set(packet['files']) == expected and len(expected) <= 128
         and all(Path(n).name == n and n.endswith(('.py','.json','.pt')) for n in expected)
         and packet['files'] == {n:sha(b) for n,b in data.items()}
         and all(packet['files'][n] == p for n,p in old['files'].items())
         and packet['source_names'] == sorted(set(old['source_names']) | EXTRA_SOURCES), 'exact CPU+five-extra closure differs')
    need(literal(data[ENTRY], 'EXTRA_SOURCES') == EXTRA_SOURCES
         and literal(data[ENTRY], 'RESOURCES') == RESOURCES, 'reviewed outer constants changed')
    for name, constant in (('qualify_llm_intervention_model_service.py','GPU_HELPER_SHA'),
                          ('supervise_llm_intervention_checkpoint_gpu.py','PROCESS_HELPER_SHA'),
                          ('qualify_llm_intervention_runtime_metadata.py','DIAGNOSTIC_SHA')):
        need(packet['files'][name] == literal(data[ENTRY], constant), 'outer helper pin differs')
    diagnostic = data['qualify_llm_intervention_runtime_metadata.py']
    bootstrap = literal(diagnostic, 'BOOTSTRAP')
    need(old['runtime_source'] == literal(diagnostic, 'RUNTIME_SOURCE'), 'original runtime differs')
    for field, constant in (('archive','ARCHIVE_SHA'),('manifest','MANIFEST_SHA')):
        need(old[field]['sha256'] == literal(diagnostic, constant), 'accepted runtime pin differs')
    need(packet['lineage_files'] == LINEAGE_FILES and set(LINEAGE_FILES.values()) <= set(old['files']), 'training lineage mapping differs')
    for pid in GPU_POLICIES:
        freeze_name, contract_name = pid+'.freeze.json', pid+'.launch.json'
        need(packet['services'][pid] == freeze_name and packet['contracts'][pid] == contract_name, 'frozen service filenames differ')
        freeze, contract = decode(data[freeze_name]), decode(data[contract_name])
        expected_contract = {'schema':'ace-campaign-staged-launch-v1','policy_id':pid,
            'entry':CPU_INPUT+'/llm_intervention_model_service.py','entry_sha256':packet['files']['llm_intervention_model_service.py'],
            'freeze':CPU_INPUT+'/'+freeze_name,'freeze_sha256':packet['files'][freeze_name],
            'worker_output':GPU_OUTPUT+'/services/'+pid+'/worker','launch_output':GPU_OUTPUT+'/services/'+pid+'/launch',
            'binding_path':GPU_OUTPUT+'/stage-binding.json','stage_path':GPU_OUTPUT+'/stage.json',
            'archive_sha256':old['archive']['sha256'],'manifest_sha256':old['manifest']['sha256'],
            'bootstrap':bootstrap,'bootstrap_sha256':sha(bootstrap.encode())}
        need(contract == expected_contract, 'DO NOT rewrite fixed CPU03 launch contract: '+pid)
        command = [old['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,
            CPU_INPUT+'/llm_intervention_campaign_launch.py',packet['files']['llm_intervention_campaign_launch.py'],
            '--contract',CPU_INPUT+'/'+contract_name,'--contract-sha256',packet['files'][contract_name]]
        need(old['persistent_specs'][pid]['command'] == command
             and freeze['scope'] == 'artificial' and freeze['limits'] == {'startup_seconds':120,'request_seconds':30,
                 'idle_seconds':30,'total_seconds':180,'max_requests':8}, 'frozen command/service limits differ')
        need(bundle['campaign']['services'][pid] == old['services'][pid], 'accepted service identity/settings differ')
        for source in bundle['campaign']['sources']:
            specs = [s for s in source['plan']['policy_specs'] if s['policy_id'] == pid]
            need(len(specs) == 1 and specs[0] == old['persistent_specs'][pid], 'accepted campaign policy command differs')
        if pid == 'frozen_llm':
            need(freeze['adapter'] is None and freeze['treatment'] == 'frozen', 'frozen arm has adapter')
        else:
            stage = 'sft' if pid == 'sft_llm' else 'final'
            adapter = freeze['adapter']
            need(freeze['treatment'] == 'artificial_adapter' and adapter['training_scope'] == 'artificial'
                 and adapter['file'] == stage+'.adapter.pt' and adapter['sha256'] == old['files'][stage+'.adapter.pt']
                 and adapter['training_receipt'] == stage+'.checkpoint.json'
                 and adapter['training_receipt_sha256'] == old['files'][stage+'.checkpoint.json'], 'adapter role changed')
    return old, bootstrap


def job_bytes(packet, packet_pin, bootstrap, runtime_source):
    command = [runtime_source+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/'+ENTRY,
        packet['files'][ENTRY],'--packet',ROOT+'/inputs/packet.json','--packet-sha256',packet_pin,'--output',GPU_OUTPUT]
    return ('''#!/bin/bash
#SBATCH --job-name=ace_llm_campaign_gpu01
#SBATCH --account=ucb736_asc2
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15360M
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output='''+ROOT+'''/logs/%j.out
#SBATCH --error='''+ROOT+'''/logs/%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD MKL_SERVICE_FORCE_INTEL
# Preserve scheduler GPU visibility; the pinned outer verifies the actual allocation.
exec '''+shlex.join(command)+'\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('cpu-inputs','cpu-output-copy','cpu-acceptance','cpu-acceptance-sha256',
                 'cpu-disposition','cpu-disposition-sha256','admission-review','admission-review-sha256',
                 'source-commit','source-review','source-review-sha256','coordinator-source'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    started = utc()
    acceptance_raw = read(args.cpu_acceptance, args.cpu_acceptance_sha256)
    disposition_raw = read(args.cpu_disposition, args.cpu_disposition_sha256)
    review_raw = read(args.admission_review, args.admission_review_sha256)
    source_review_raw = read(args.source_review, args.source_review_sha256)
    acceptance, disposition, review, source_review = map(decode,
        (acceptance_raw, disposition_raw, review_raw, source_review_raw))
    accounting = admission(acceptance, disposition, review, sha(acceptance_raw), sha(disposition_raw))
    builder_raw, coordinator_raw = read(__file__), read(args.coordinator_source)
    review_shape(source_review, 'ace-campaign-gpu01-source-review-v1')
    need(source_review['source_commit'] == args.source_commit
         and source_review['builder_sha256'] == sha(builder_raw)
         and source_review['coordinator_sha256'] == sha(coordinator_raw)
         and set(source_review['sources']) == EXTRA_SOURCES
         and REVIEW_DOCS <= set(source_review['documents']), 'review does not bind current sources')
    old_raw = read(Path(args.cpu_inputs)/'packet.json', CPU_PACKET_SHA)
    old = decode(old_raw)
    committed_input = str((CPU_LOCAL/'inputs').relative_to(REPO))
    need(git_bytes(CPU_PACKET_COMMIT, committed_input+'/packet.json') == old_raw, 'CPU packet commit differs')
    data = {}
    for name, pin in old['files'].items():
        need(Path(name).name == name and is_pin(pin), 'unsafe CPU input member')
        value = read(Path(args.cpu_inputs)/name, pin)
        need(git_bytes(CPU_PACKET_COMMIT, committed_input+'/'+name) == value, 'committed CPU input differs')
        if name in old['source_names']:
            need(git_bytes(CPU_SOURCE_COMMIT, 'scripts/research/'+name) == value, 'accepted CPU source commit differs')
        data[name] = value
    for name in sorted(EXTRA_SOURCES):
        value = git_bytes(args.source_commit, 'scripts/research/'+name)
        need(sha(value) == source_review['sources'][name], 'GPU source review pin differs: '+name)
        need(name not in data or data[name] == value, 'GPU addition would replace accepted CPU bytes')
        data[name] = value
    data['cpu-packet.json'], data['cpu-acceptance.json'] = old_raw, acceptance_raw
    bundle = check_cpu_evidence(old, data, acceptance, args.cpu_inputs, args.cpu_output_copy)
    packet = {'schema':'ace-artificial-paired-campaign-gpu-v1','files':{n:sha(b) for n,b in sorted(data.items())},
        'source_names':sorted(set(old['source_names']) | EXTRA_SOURCES),'cpu_packet_sha256':CPU_PACKET_SHA,
        'cpu_acceptance_sha256':sha(acceptance_raw),'paths':PATHS,'resources':RESOURCES,
        'services':{p:p+'.freeze.json' for p in GPU_POLICIES},'contracts':{p:p+'.launch.json' for p in GPU_POLICIES},
        'lineage_files':LINEAGE_FILES}
    _, bootstrap = packet_shape(packet, data, bundle)
    packet_raw = raw(packet)
    artifacts = {'inputs/'+n:b for n,b in data.items()}
    artifacts.update({'inputs/packet.json':packet_raw,
        'job.sbatch':job_bytes(packet,sha(packet_raw),bootstrap,old['runtime_source']),
        'cpu-disposition.json':disposition_raw,'admission-review.json':review_raw,'source-review.json':source_review_raw,
        'build_packet.py':builder_raw,'launch_once.py':coordinator_raw})
    for i,(name,pin) in enumerate(sorted(source_review['documents'].items())):
        value = git_bytes(args.source_commit, name)
        need(sha(value) == pin, 'committed review/guidance bytes differ: '+name)
        artifacts['source-review-evidence-%02d.md'%i] = value
    artifacts['intent-provenance.json'] = raw({'schema':'ace-campaign-gpu01-build-provenance-v1',
        'started_utc':started,'prepared_utc':utc(),'source_commit':args.source_commit,
        'cpu_source_commit':CPU_SOURCE_COMMIT,'cpu_packet_commit':CPU_PACKET_COMMIT,
        'cpu_acceptance_path':str(Path(args.cpu_acceptance).absolute()),'cpu_acceptance_sha256':sha(acceptance_raw),
        'cpu_disposition_path':str(Path(args.cpu_disposition).absolute()),'cpu_disposition_sha256':sha(disposition_raw),
        'admission_review_sha256':sha(review_raw),'source_review_sha256':sha(source_review_raw),
        'cpu_output_copy':str(Path(args.cpu_output_copy).absolute()),'original_contract_paths':PATHS,
        'source_only_builder_no_project_imports':True,'admission':accounting,'resources':RESOURCES,
        'resource_rationale':{'gpu':'one sequential A100 3g.20gb; prior GPU03 approximately 3.278 GB reserved',
            'host':'15360 MiB: margin above prior 95.05% of 12 GiB; exactly 4 x 3840 MiB',
            'time':'600s bounded campaign; three services, at most eight decisions each',
            'cpu':'four requested, at most eight allocated; reserve 4800 allocated CPU-seconds'},
        'guards':{'first_verified_gpu_seconds':240,'per_service_startup_seconds':120,
            'progress_freshness_seconds':90,'work_deadline_seconds':550,'final_deadline_seconds':580,
            'two_independent_signals':'positive CUDA generation/progress plus owned NVIDIA process memory',
            'missing_telemetry':'uncertain; preserve failure evidence, never infer idle or success'},
        'post_run_obligations':['retain full launch/GPU/campaign custody and original CPU inventories',
            'reconcile requested/observed CPUs, GPU count, memory, utilization, tokens, throughput and elapsed time',
            'separate saved validation, scheduler reconciliation and independent GPU acceptance',
            'separate CPU finalization; never qualify from exit 0 or auto-submit a successor'],
        'scientific_qualified':False,'gpu_executed':False})
    manifest = {'schema':'ace-campaign-gpu01-build-manifest-v1','source_commit':args.source_commit,
        'remote_root':ROOT,'paths':PATHS,'gpu_packet_sha256':sha(packet_raw),
        'cpu_acceptance_sha256':sha(acceptance_raw),'cpu_disposition_sha256':sha(disposition_raw),
        'admission_review_sha256':sha(review_raw),'source_review_sha256':sha(source_review_raw),
        'resources':RESOURCES,'admission':accounting,'artifacts':{n:sha(b) for n,b in sorted(artifacts.items())},
        'integration_assumptions':['independent review producers agree documented receipt schemas',
            'exact prior rows/journals/queue/site snapshot remains current until locked launch',
            'original CPU03 inputs and campaign remain untouched until GPU allocation authenticates them',
            'concrete packet committed and separately reviewed before coordinator invocation',
            'standard-library byte checks do not replace the pinned outer runtime/lineage validators',
            'no terminal puller or finalizer is included; parent owns full custody and independent disposition']}
    artifacts['build-manifest.json'] = raw(manifest)
    # No output mutation occurs before ALL CPU acceptance/source/byte gates pass.
    for dest in (EXTERNAL,LOCAL):
        no_links(dest)
        need(not dest.exists(), 'exclusive destination already exists; do not retry: '+str(dest))
    for dest in (EXTERNAL,LOCAL):
        dest.mkdir(parents=True, exist_ok=False)
        (dest/'inputs').mkdir()
        for name,value in artifacts.items():
            save_bytes(dest/name, value)
    print(json.dumps({'status':'unsubmitted_packet_built','local':str(LOCAL),'external':str(EXTERNAL),
        'packet_sha256':sha(packet_raw),'manifest_sha256':sha(raw(manifest)),'admission':accounting,
        'gpu_qualified':False}, indent=2))


if __name__ == '__main__':
    main()
