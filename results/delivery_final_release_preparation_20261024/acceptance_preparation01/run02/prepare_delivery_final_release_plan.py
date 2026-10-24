"""Prepare a NEW private publication plan; never build or publish a package.

Requires independently trusted request, report acceptance and reviewed accounting
digests. Captures evidence only; never executes a frozen reporter or replays models.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re

import delivery_prospective_publication as reports
import prepare_delivery_publication_snapshot as snapshot
import verify_delivery_publication_release as public
from build_delivery_release import reconcile
from verify_delivery_release import relative, sha, project_toml_bytes, identifying_bytes, verify

require, digest, decode, pin, encoded = public.require, public.digest, public.decode, public.pin, public.encoded
INPUTS = {'plan', 'manifest', 'bundle', 'snapshot_inventory', 'replay_acceptance',
          'report_freeze', 'report_execution', 'report_receipt', 'report_runtime',
          'report_custody', 'report_acceptance', 'report_review', 'report_review_witness', 'execution_verification',
          'runtime_inventory', 'remote_inventory', 'report_input_pins',
          'accounting', 'accounting_source', 'failed_report_inventory', 'accounting_review', 'accounting_review_witness',
          'pdf', 'checker', 'planner'}
REPORT_ACCEPTANCE_SCHEMA = 'ace-delivery-report-acceptance-v1'
REPORT_JOB = 34867191
REPORT_RECORDS = {'report_freeze': 'report_freeze.json', 'report_execution': 'report_execution.json',
    'report_receipt': 'report_receipt.json', 'report_runtime': 'report_runtime_verified.json',
    'runtime_inventory': 'runtime_inventory.json', 'replay_acceptance': 'replay_acceptance.json'}
ACCEPTANCE_EDGES = {'freeze_sha256': 'report_freeze', 'execution_sha256': 'report_execution',
    'report_receipt_sha256': 'report_receipt', 'runtime_verified_sha256': 'report_runtime',
    'remote_inventory_sha256': 'remote_inventory', 'custody_sha256': 'report_custody',
    'execution_verification_sha256': 'execution_verification'}


def capture(path, expected):
    raw = reports.read(path)
    require(digest(raw) == pin(expected), 'independently pinned input changed')
    return raw


def scheduler_row(inventory, job):
    """Parse the preserved collection format; never query a scheduler."""
    command = inventory['commands']['sacct']
    columns = 'JobIDRaw,JobID,State,ExitCode,Start,End,ElapsedRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ReqMem,MaxRSS,AllocTRES,Account,Partition,QOS,NodeList'
    require(command['returncode'] == 0 and '--format='+columns in command['argv'] and
            command['argv'][1:3] == ['-j', str(job)], 'scheduler collection format/job differs')
    rows = [line.split('|') for line in command['stdout'].splitlines()]
    require(rows and all(len(row) == 17 for row in rows), 'scheduler row shape differs')
    parents = [r for r in rows if r[:2] == [str(job), str(job)]]
    require(len(parents) == 1, 'unique scheduler parent required')
    return parents[0], rows


def require_report_acceptance(raw, inputs, originals, expected_acceptance):
    """Join independently accepted bytes to captured custody without executing them.

    expected_acceptance must come from the caller's independent review channel,
    never from the request or a checksum next to the packet being validated.
    """
    require(digest(raw) == pin(expected_acceptance) == inputs['report_acceptance']['sha256'],
            'independent report acceptance pin differs')
    a = decode(raw)
    require(a['schema'] == REPORT_ACCEPTANCE_SCHEMA and a['accepted'] is True and
            a['publication_approved'] is False and a['actual_execution_checks_pass'] is True,
            'independent report acceptance required')
    reports.integer(a['job_id'], REPORT_JOB, 'accepted report02 job required')
    reports.integer(a['review_required_findings'], 0, 'independent review still has required findings')
    require(a['review_sha256'] == digest(originals['report_review']), 'accepted independent review differs')
    require(originals['report_review'].strip(), 'independent review is empty')
    require(a['independent_scientific_output_review_pending'] is False and
            a['review_witness_sha256'] == digest(originals['report_review_witness']), 'independent review not closed/bound')
    witness = decode(originals['report_review_witness'])
    require(witness['schema'] == 'ace-report02-independent-review-witness-v1' and
            type(witness['job_id']) is int and witness['job_id'] == REPORT_JOB and
            witness['required_findings'] == [] and
            encoded(witness['checks']['numerical']['counts']) == encoded(public.COUNTS) and
            witness['checks']['reporting']['outputs'] == 13, 'independent review witness differs')
    verification = decode(originals['execution_verification'])
    for key, value in verification.items():
        # Retain the historical pending field in verification; acceptance may
        # explicitly close it and may carry a later authorship timestamp.
        if key == 'at':
            public.timestamp(a[key])
        elif key == 'independent_scientific_output_review_pending':
            require(type(a[key]) is bool, 'review disposition must be boolean')
        else:
            require(encoded(a[key]) == encoded(value), 'execution verification field differs: '+key)
    for key, name in ACCEPTANCE_EDGES.items():
        require(a[key] == digest(originals[name]), 'accepted evidence digest differs: '+key)
    if 'custody_file' in a:
        require(a['custody_file'] == inputs['report_custody']['path'], 'accepted custody location differs')
    f, e, r, rt, c, remote, inventory, replay = (decode(originals[k]) for k in
        ('report_freeze', 'report_execution', 'report_receipt', 'report_runtime',
         'report_custody', 'remote_inventory', 'runtime_inventory', 'replay_acceptance'))
    for kind in ('report_freeze', 'report_execution', 'report_receipt', 'report_runtime'):
        public.project(kind, originals[kind])
    require(e['job_id'] == str(REPORT_JOB) and c['job_id'] == REPORT_JOB and
            c['parent_state'] == 'COMPLETED', 'custody/execution job differs')
    require(e['freeze_sha256'] == r['freeze_sha256'] == digest(originals['report_freeze']) and
            e['report_receipt_sha256'] == digest(originals['report_receipt']), 'report execution lineage differs')
    for key, kind in (('source_pins', 'sources'), ('metadata_pins', 'metadata'), ('original_input_pins', 'originals')):
        public.field(a[key], kind)
        require(a[key] == f[key] == r[key], 'accepted frozen input map differs: '+key)
    require(f['schema'] == 'delivery-prospective-reporting-execution-v1' and
            f['manifest_sha256'] == public.CANDIDATE_SHA, 'report candidate lineage differs')
    for key in ('new_fits', 'new_optimizer_updates', 'new_responses'):
        for obj in (a, f, r):
            reports.integer(obj[key], 0, 'report-only scope required')
    for key in ('new_fits', 'new_responses'):
        reports.integer(e[key], 0, 'execution scope differs')
    require(e['child']['primary_failure'] is None and e['child']['secondary_failures'] == [],
            'report child failure remains')
    for key in ('counts', 'reviewed_output_counts'):
        require(encoded(a[key]) == encoded(public.COUNTS), 'accepted output counts differ')
    require(encoded(r['counts']) == encoded(public.COUNTS), 'report output counts differ')
    require(c['remote_inventory_sha256'] == digest(originals['remote_inventory']) and
            c['files_verified'] == remote['file_count'] == 50 and
            c['bytes_verified'] == remote['bytes'] == 9147453 and c['frozen_inputs_verified'] == 17 and
            c['scientific_outcomes_decoded_by_collection'] is False, 'report custody summary differs')
    root = Path(inputs['report_freeze']['path']).parent
    entries = public.index(remote['files'])
    require(len(entries) == 50 and sum(row['bytes'] for row in entries.values()) == remote['bytes'] and
            remote['root'] == f['output'], 'remote inventory membership differs')
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths) and
            {p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(entries),
            'local custody membership differs')
    captured = {}
    for name, row in entries.items():
        require(set(row) == {'path', 'sha256', 'bytes'}, 'custody file fields differ')
        public.field(row['bytes'], 'int')
        captured[name] = capture(relative(root, name), row['sha256'])
        require(len(captured[name]) == row['bytes'], 'custody file size differs')
    for key, name in REPORT_RECORDS.items():
        require(originals[key] == captured[name], 'request/custody bytes differ: '+key)
    require('launcher_failure.json' not in captured and 'report_exports_started.json' in captured,
            'report export phase evidence missing or launcher failed')
    frozen = decode(originals['report_input_pins'])
    expected_frozen = {**f['source_pins'], **f['metadata_pins'],
        'report_freeze.json': digest(originals['report_freeze']),
        'authorization.json': f['authorization_sha256'], 'output_claim.json': f['output_claim_sha256'],
        'run.sbatch': frozen.get('run.sbatch')}
    require(frozen == expected_frozen and len(frozen) == 17, '17 frozen launch inputs required')
    require(witness['launch_input_sha256'] == frozen and witness['output_sha256'] == r['outputs'] and
            witness['historical_input_sha256'] == f['original_input_pins'] and
            witness['checks']['execution']['source_revision'] == f['source_revision'],
            'review witness input/output lineage differs')
    for name, value in frozen.items():
        require(digest(captured[name]) == pin(value), 'frozen custody input differs')
    require(replay['accepted'] is True and replay['job_id'] == 34863006 and
            replay['manifest_sha256'] == f['manifest_sha256'] and
            replay['replay_receipt_sha256'] == f['metadata_pins']['replay_receipt.json'] and
            replay['runtime_inventory_sha256'] == rt['inventory_sha256'] ==
            f['metadata_pins']['runtime_inventory.json'] == digest(originals['runtime_inventory']) and
            f['metadata_pins']['replay_acceptance.json'] == public.REPLAY_ACCEPTANCE_SHA,
            'accepted replay/runtime lineage differs')
    objects = public.index(inventory['files'])
    require(len(objects) == rt['environment_objects'] == 28577 and
            len(r['loaded_statistics_files']) == 618, 'runtime membership differs')
    require(rt['dependencies'] == r['statistics_dependencies'] and
            set(rt['dependencies']) == {'numpy', 'scipy'}, 'statistics dependencies differ')
    for name, row in r['loaded_statistics_files'].items():
        item = objects.get(row['path'], {})
        require(item.get('kind') == 'file' and item.get('sha256') == row['sha256'],
                'loaded statistics file absent from accepted inventory')
    for name, dep in rt['dependencies'].items():
        require(dep['version'] == public.reports.DEPENDENCIES[name] and
                dep['sha256'] == r['loaded_statistics_files'][name]['sha256'] and
                dep['origin'] == str(Path(inventory['prefix'])/r['loaded_statistics_files'][name]['path']),
                'statistics dependency origin/version/hash differs')
    require(set(r['outputs']) == {n for n in captured if n.startswith(('primary/', 'descriptive/'))},
            'report output custody membership differs')
    for name, value in r['outputs'].items():
        require(digest(captured[name]) == value, 'report output differs')
    for folder, index_name, members, accepted_key in (
        ('primary', 'claim_index.json', reports.PRIMARY, 'primary_index_sha256'),
        ('descriptive', 'output_index.json', reports.DESCRIPTIVE+reports.DATA, 'descriptive_index_sha256')):
        contents = captured[folder+'/'+index_name]
        require(a[accepted_key] == digest(contents), 'accepted report index differs')
        hashes = decode(contents)['output_hashes']
        require(set(hashes) == set(members), 'report index membership differs')
        require(all(hashes[n] == r['outputs'][folder+'/'+n] for n in members), 'report index edge differs')
    validate_report_resources(f, e, r, a, remote)
    return a


def validate_report_resources(f, e, r, a, remote):
    expected = {'cpus': 1, 'rss_bytes': 3*2**30, 'wall_seconds': 3600,
        'slurm_wall_limit': '01:00:00', 'account': 'ucb736_asc1', 'partition': 'acpu', 'qos': 'cpu-normal'}
    require(all(encoded(f[k]) == encoded(v) for k, v in expected.items()), 'frozen resource contract differs')
    row, rows = scheduler_row(remote, REPORT_JOB)
    require({v[1] for v in rows} == {str(REPORT_JOB), str(REPORT_JOB)+'.batch', str(REPORT_JOB)+'.extern'} and
            all(v[2:4] == ['COMPLETED', '0:0'] for v in rows), 'COMPLETED 0:0 scheduler evidence required')
    require(row[9] == '1' and row[10] == '3G' and row[12:16] ==
            ['cpu=1,mem=3G,node=1', f['account'], f['partition'], f['qos']] and
            int(row[6]) == int(row[7]) == a['allocation_seconds'] == 2513, 'scheduler allocation differs')
    allocation = e['allocation']; raw = allocation['raw_scontrol']
    require(digest(raw.encode()) == allocation['raw_scontrol_sha256'], 'live allocation capture digest differs')
    fields = dict(item.split('=', 1) for item in raw.split() if '=' in item)
    expected_fields = {'JobId': str(REPORT_JOB), 'Account': f['account'], 'Partition': f['partition'],
        'QOS': f['qos'], 'NumNodes': '1', 'NumCPUs': '1', 'NumTasks': '1', 'CPUs/Task': '1',
        'TimeLimit': f['slurm_wall_limit'], 'MinMemoryNode': '3G', 'AllocTRES': 'cpu=1,mem=3G,node=1',
        'TresPerNode': None, 'TresPerTask': 'cpu=1', 'Gres': None}
    require(all(fields.get(k) == v for k, v in expected_fields.items()) and
            allocation['verified_fields'] == expected_fields, 'verified live allocation fields differ')
    policy = e['deadline_policy']
    require(policy['cleanup_reserve_seconds'] == 50 and policy['rounding_reserve_seconds'] == 2 and
            policy['maximum_wrapper_relative_child_seconds'] == 3550 and
            0 < policy['remaining_child_seconds_at_check'] <= 3550, 'cleanup/deadline policy differs')
    require(a['slurm_total_cpu'] == row[8] and a['slurm_max_rss_by_step'] == {v[1]: v[11] for v in rows},
            'accepted scheduler measurements differ')
    for key, value in {'child_process_cpu_seconds': r['process_cpu_seconds'],
        'child_elapsed_seconds': r['elapsed_seconds'], 'child_peak_rss_kib': r['peak_self_rss_kib'],
        'wrapper_elapsed_seconds': e['elapsed_seconds']}.items():
        public.nonnegative(value)
        require(a[key] == value, 'accepted report measurement differs')
    require(e['elapsed_seconds'] < f['wall_seconds'] and r['elapsed_seconds'] <= e['elapsed_seconds'] and
            a['measurements_overlap_not_additive'] is True and a['whole_sprint_cpu_unknown'] is True,
            'report measurement scope differs')


def validate_private_accounting(originals, acceptance):
    value = decode(originals['accounting']); attempts = public.validate_accounting(value)
    require(originals['accounting_review'].strip(), 'independent accounting review required')
    witness = decode(originals['accounting_review_witness'])
    require(witness['schema'] == 'ace-final-accounting-delta-review-witness-v1' and
            witness['failed_checks'] == [] and witness['projection_mismatches'] == [] and
            witness['inherited_rows_checked'] == 57 and witness['unique_allocations'] == 59 and
            witness['allocated_cpu_seconds'] == 132995, 'independent accounting review not complete')
    checks = witness['checks']
    required_checks = {'all57_public_fields_and_private_copies_match', 'report01_status_exit_cleanup',
        'report02_status_exit_cleanup', 'cutoff_matches_report02', '59_unique_public_private_attempts_jobs_rows',
        'four_unknowns_remain_null', 'replay06_zero_explicit_uncertainty', 'nonadditive_scopes_explicit'}
    require(required_checks <= set(checks) and all(v is True for v in checks.values()),
            'required independent accounting check missing or failed')
    for key in ('accounting', 'accounting_source', 'failed_report_inventory', 'remote_inventory'):
        expected = {'sha256': digest(originals[key]), 'bytes': len(originals[key])}
        require(expected in witness['pins'].values(), 'accounting review evidence differs: '+key)
    for name in ('report01', 'report02'):
        report = witness['reports'][name]
        require(report['measurements'] == value['report_measurements'][name] and
                report['row_sha256'] == attempts[name]['raw_scheduler_row_sha256'] and
                report['state'] == attempts[name]['state'] and report['exit_code'] == attempts[name]['exit_code'],
                'reviewed accounting report differs')
    require(public.timestamp(witness['reports']['report02']['end_utc']) == public.timestamp(value['cutoff']['terminal_utc']) and
            witness['caveats'] == value['measurement_caveats'], 'reviewed accounting cutoff/caveats differ')
    require(set(value['source_receipt_sha256s']) == {digest(originals[k]) for k in
            ('accounting_source', 'failed_report_inventory', 'remote_inventory')}, 'accounting source receipts differ')
    source = decode(originals['accounting_source'])
    require(source['schema'] == 'ace-publication-incremental-accounting-review-v1' and
            source['reconciliation']['total_distinct_allocations'] == 57 and
            source['reconciliation']['cumulative_through_replay12_allocated_cpu_seconds'] == 128690,
            'original 57-allocation accounting required')
    old_rows = source['original']['allocation_rows']
    require(len(old_rows) == 46 and len(source['incremental_attempts']) == 11, 'prior allocation membership differs')
    expected = {}
    for i, row in enumerate(old_rows, 1):
        expected['original_'+str(i).zfill(2)] = {
            'kind': row['phase'], 'state': row['state'], 'exit_code': row['exit_code'],
            'allocated_cpus': row['allocated_cpus'], 'allocated_cpu_seconds': row['allocated_cpu_seconds'],
            'elapsed_allocation_seconds': row['elapsed_seconds'], 'requested_wall_seconds': None,
            'slurm_total_cpu_seconds': public.slurm_seconds(row['slurm_total_cpu_raw']),
            'wrapper_elapsed_seconds': None, 'raw_scheduler_row_sha256': digest(row['raw_parent_row'].encode())}
    for row in source['incremental_attempts']:
        expected[row['attempt']] = {'kind': row['kind'], 'state': row['state'], 'exit_code': row['exit_code'],
            'allocated_cpus': row['allocated_cpus'], 'allocated_cpu_seconds': row['allocated_cpu_seconds'],
            'elapsed_allocation_seconds': row['elapsed_allocation_seconds'], 'requested_wall_seconds': row['wall_cap_seconds'],
            'slurm_total_cpu_seconds': row['slurm_total_cpu_seconds_as_recorded'],
            'wrapper_elapsed_seconds': row.get('wrapper_elapsed_seconds'),
            'raw_scheduler_row_sha256': digest(row['raw_parent_row'].encode())}
    require(set(attempts) == set(expected) | {'report01', 'report02'}, 'accounting attempt membership differs')
    for name, row in expected.items():
        require(attempts[name] == {'attempt_id': name, **row}, 'historical allocation changed: '+name)
    for key, job, name in (('failed_report_inventory', 34864950, 'report01'), ('remote_inventory', REPORT_JOB, 'report02')):
        row, rows = scheduler_row(decode(originals[key]), job)
        attempt = attempts[name]; measured = value['report_measurements'][name]
        require(attempt['state'] == row[2] and attempt['exit_code'] == row[3] and
                attempt['allocated_cpus'] == int(row[9]) and attempt['allocated_cpu_seconds'] == int(row[7]) and
                attempt['elapsed_allocation_seconds'] == int(row[6]) and
                attempt['slurm_total_cpu_seconds'] == public.slurm_seconds(row[8]) and
                attempt['raw_scheduler_row_sha256'] == digest('|'.join(row).encode()),
                'accounting scheduler binding differs: '+name)
        batch = [v for v in rows if v[1] == str(job)+'.batch']
        require(len(batch) == 1 and batch[0][11] == str(measured['slurm_batch_max_rss_kib'])+'K',
                'scheduler batch RSS differs')
    measured = value['report_measurements']['report02']
    for public_key, key in (('child_process_cpu_seconds', 'child_process_cpu_seconds'),
        ('child_elapsed_seconds', 'child_elapsed_seconds'), ('child_peak_self_rss_kib', 'child_peak_rss_kib'),
        ('wrapper_elapsed_seconds', 'wrapper_elapsed_seconds')):
        require(measured[public_key] == acceptance[key], 'accepted accounting measurement differs')
    execution = decode(originals['report_execution'])
    require(measured['supervised_peak_tree_rss_bytes'] == max(s['rss_bytes'] for s in execution['child']['samples']),
            'report supervised RSS differs')
    require(acceptance['known_cumulative_allocated_seconds'] == value['cutoff']['allocated_cpu_seconds'] and
            acceptance['known_cumulative_allocated_core_hours'] == 132995/3600 and
            acceptance['reservation_core_hours'] == value['reservation']['reserved'], 'accepted accounting totals differ')


def request_inputs(request_file, request_sha, acceptance_sha, accounting_sha):
    request_raw = capture(request_file, request_sha); request = decode(request_raw)
    require(set(request) == {'schema', 'inputs'} and request['schema'] == 'delivery-final-release-inputs-v1',
            'explicit final-release input schema required')
    require(set(request['inputs']) == INPUTS, 'complete independently pinned input set required')
    for value in request['inputs'].values():
        require(set(value) == {'path', 'sha256'} and Path(value['path']).is_absolute(), 'explicit absolute input and pin required')
        pin(value['sha256'])
    inputs = request['inputs']
    require(inputs['manifest']['sha256'] == public.CANDIDATE_SHA and inputs['plan']['sha256'] == public.PLAN_SHA,
            'candidate03 and recovery plan pins required')
    require(inputs['replay_acceptance']['sha256'] == public.REPLAY_ACCEPTANCE_SHA, 'accepted replay12 pin required')
    # Independent approval pins precede reading outcomes or creating any output.
    require(inputs['report_acceptance']['sha256'] == pin(acceptance_sha) and
            inputs['accounting']['sha256'] == pin(accounting_sha), 'independent acceptance/accounting pin differs')
    originals = {n: capture(v['path'], v['sha256']) for n, v in inputs.items()}
    acceptance = require_report_acceptance(originals['report_acceptance'], inputs, originals, acceptance_sha)
    validate_private_accounting(originals, acceptance)
    return request_raw, inputs, originals


def inherited_plan(plan, manifest):
    """Validate exact planned transformation semantics without executing sources."""
    before, planned = public.index(manifest['files']), public.index(plan['files'])
    require(len(before) == public.OLD_FILES and set(planned) == set(before) and
            len(manifest['bindings']) == public.OLD_BINDINGS and plan['bindings'] == manifest['bindings'],
            'exact 3893/5068 inherited plan required')
    for name, entry in planned.items():
        expected = before[name]
        transform = entry.get('transform', {'kind': 'identity'})
        require(entry['original_sha256'] == expected['original_sha256'] and entry['role'] == expected['role']
                and transform == expected['transform'], 'inherited transformation semantics changed')
        source = reconcile(entry['sources'], expected['original_sha256'])
        if transform['kind'] == 'identity':
            require(sha(source) == expected['sha256'] and source.stat().st_size == expected['bytes'], 'inherited bytes changed')
        else:
            raw = reports.read(source)
            if transform['kind'] == 'project-json':
                value = decode(raw); keys = transform['keys']
                require(keys and len(keys) == len(set(keys)), 'unique projection keys required')
                derived = encoded({k: value[k] for k in keys})
            elif transform['kind'] == 'project-toml':
                derived = project_toml_bytes(raw, transform)
            else:
                raise ValueError('unsupported inherited transform')
            require(digest(derived) == expected['sha256'] and len(derived) == expected['bytes'],
                    'inherited projected bytes changed')


def capture_snapshot(root, inventory_raw, bundle_pin):
    inventory = decode(inventory_raw)
    require(inventory['schema'] == 'delivery-publication-source-snapshot-v1' and
            re.fullmatch('[0-9a-f]{40}', inventory['revision']) and
            inventory['prospective_bundle_sha256'] == bundle_pin and
            inventory['companions'] == list(public.COMPANIONS), 'integrated committed snapshot required')
    entries = public.index(inventory['files'])
    require(set(entries) == set(public.SNAPSHOT_MAP) and len(entries) == 42, 'exact prospective snapshot membership required')
    raw = {}
    for name, entry in entries.items():
        raw[name] = capture(relative(root, 'source/'+name), entry['sha256'])
        require(len(raw[name]) == entry['bytes'], 'snapshot size differs')
    snapshot.check_bundle(raw, bundle_pin)
    return raw


def omitted_selectors(value, selected, prefix=''):
    """Private record of omitted leaves; no private selectors enter the release."""
    if prefix in selected:
        return []
    if type(value) is dict:
        return [p for k, v in value.items() for p in omitted_selectors(v, selected,
                prefix+'/'+k.replace('~', '~0').replace('/', '~1'))]
    if type(value) is list:
        return [p for i, v in enumerate(value) for p in omitted_selectors(v, selected, prefix+'/'+str(i))]
    return [prefix]


def publication_payload(snapshot_raw, old_raw, old_contract_raw, evidence_raw, private_pins, pdf, checker, accounting_raw):
    """Pure construction after the private acceptance gate; never attests execution.

    Kept separate so fabricated boundary tests need no real report/schema or I/O.
    The production entry point alone authenticates private acceptance and custody.
    """
    require(set(snapshot_raw) == set(public.SNAPSHOT_MAP), 'complete source snapshot required')
    require(set(evidence_raw) == set(public.SPECS) and set(private_pins) == set(public.PRIVATE_KEYS),
            'complete evidence inputs required')
    old = decode(old_raw)
    additions, dispositions = {}, {}
    for name, target in public.SNAPSHOT_MAP.items():
        raw = snapshot_raw[name]
        dispositions[name] = {'sha256': digest(raw), 'public_path': target,
                              'disposition': 'identity' if target else 'private-only'}
        if target:
            require(target not in additions or additions[target] == raw, 'public source alias conflict')
            additions[target] = raw
    require(additions[public.PREDECESSOR] == old_raw, 'snapshot candidate manifest differs')
    additions[public.PAPER+'paper.pdf'] = pdf
    additions[public.CHECKER] = checker
    disclosure = public.accounting_disclosure(accounting_raw, private_pins)
    additions[public.ACCOUNTING] = accounting_raw
    projections, omissions = {}, {}
    for kind, raw in evidence_raw.items():
        value = public.project(kind, raw); path = 'publication/evidence/'+kind+'.json'
        additions[path] = encoded(value)
        projections[kind] = {'path': path, 'sha256': digest(additions[path]), 'original_sha256': digest(raw)}
        omissions[kind] = omitted_selectors(decode(raw), set(public.SPECS[kind]))
    old_source = decode(old_contract_raw)
    require(old_source['runner_grant_authority_resolved'] is True and len(old_source['bindings']) == 24,
            'resolved notice and original source relationships required')
    contract = {'schema': 'delivery-publication-source-dispositions-v1', 'authoritative_path': public.CONTRACT,
        'predecessor': {'path': public.PREDECESSOR, 'sha256': digest(old_raw), 'files': public.OLD_FILES,
            'bindings': public.OLD_BINDINGS, 'source_contract': 'reproduction/source_contract.json',
            'source_contract_sha256': digest(old_contract_raw)},
        'private_evidence': {k: {'sha256': pin(v), 'availability': 'private-custody'} for k, v in private_pins.items()},
        'execution_evidence': 'accepted privately; not independently revalidated by this public checker',
        'snapshot_members': dispositions, 'projections': projections,
        'artifacts': {n: digest(raw) for n, raw in additions.items()},
        'original_source_bindings': old_source['bindings'],
        'source_usage': copy.deepcopy(public.SOURCE_USAGE),
        'report_source_dispositions': public.report_source_dispositions(
            public.projection(decode(additions[projections['report_receipt']['path']]),
                              'report_receipt')['/source_pins'], private_pins['report_custody']),
        'accounting_disclosure': disclosure,
        **{k: False for k in public.FALSE_FLAGS}}
    additions[public.CONTRACT] = encoded(contract)
    manifest = copy.deepcopy(old)
    for name, raw in additions.items():
        inherited = next((e for e in old['files'] if e['path'] == name), None)
        if inherited:
            require(inherited['sha256'] == digest(raw) and inherited['bytes'] == len(raw), 'inherited publication collision')
        else:
            manifest['files'].append({'path': name, 'sha256': digest(raw), 'original_sha256': digest(raw),
                'bytes': len(raw), 'role': 'publication-source-contract' if name == public.CONTRACT else 'publication-artifact',
                'transform': {'kind': 'identity'}})
    manifest['bindings'].extend(public.publication_bindings(contract))
    return additions, contract, manifest, omissions


def prepare(request_file, request_sha, destination, *, acceptance_sha, accounting_sha):
    request_raw, inputs, originals = request_inputs(request_file, request_sha, acceptance_sha, accounting_sha)
    require(originals['planner'] == Path(__file__).read_bytes() and
            originals['checker'] == Path(public.__file__).read_bytes(), 'executing preparation sources differ')
    destination = snapshot.plain_path(destination)
    origins = [Path(v['path']).resolve().parent for v in inputs.values()]
    require(not destination.exists() and not any(destination.resolve().is_relative_to(p) for p in origins),
            'new exclusive destination outside all inputs required')
    root = Path(inputs['manifest']['path']).parent
    verify(root, public.CANDIDATE_SHA)
    plan, old = decode(originals['plan']), decode(originals['manifest'])
    inherited_plan(plan, old)
    old_source = capture(relative(root, 'reproduction/source_contract.json'), public.SOURCE_CONTRACT_SHA)
    bundle_pin = inputs['bundle']['sha256']
    bundle_raw = reports.capture_bundle(Path(inputs['bundle']['path']).parent, bundle_pin)
    writing = capture_snapshot(Path(inputs['snapshot_inventory']['path']).parent,
                               originals['snapshot_inventory'], bundle_pin)
    require(all(writing[public.PAPER+n] == raw for n, raw in bundle_raw.items()), 'assembly/captured writing differs')
    evidence_raw = {k: originals[k] for k in public.SPECS if k.startswith('report_')}
    evidence_raw.update({kind: bundle_raw[reports.PROOFS[key]] for key, kind in
                        (('freeze', 'replay_freeze'), ('execution', 'replay_execution'), ('replay', 'replay'))})
    additions, contract, manifest, omissions = publication_payload(writing, originals['manifest'], old_source,
        evidence_raw, {k: request_sha if k == 'request' else inputs[k]['sha256'] for k in public.PRIVATE_KEYS},
        originals['pdf'], originals['checker'], originals['accounting'])
    raw = {**additions, 'reproduction/source_contract.json': old_source,
           'B/acceptance.json': reports.read(relative(root, 'B/acceptance.json'))}
    public.validate_publication(manifest, originals['manifest'], contract, raw)
    final = copy.deepcopy(plan)
    inherited = public.index(old['files'])
    for name, content in additions.items():
        if name not in inherited:
            final['files'].append({'path': name, 'sources': [str(destination/'public'/name)],
                'original_sha256': digest(content), 'transform': {'kind': 'identity'},
                'role': 'publication-source-contract' if name == public.CONTRACT else 'publication-artifact'})
    final['bindings'].extend(public.publication_bindings(contract))
    final['status'] = 'private publication successor preparation; human release/submission approval pending'
    directory = snapshot.exclusive_directory(destination)
    try:
        for name, content in additions.items():
            snapshot.store(directory, 'public/'+name, content)
            require(not identifying_bytes(relative(destination, 'public/'+name)), 'public identifier requires disposition: '+name)
        for name, raw_input in originals.items():
            snapshot.store(directory, 'private/'+name, raw_input)
        for kind, raw_input in evidence_raw.items():
            snapshot.store(directory, 'private/projections/'+kind+'.json', raw_input)
        snapshot.store(directory, 'private/request.json', request_raw)
        snapshot.store(directory, 'private/derivation.json', encoded({'request_sha256': request_sha,
            'independent_report_acceptance_sha256': acceptance_sha, 'independent_accounting_sha256': accounting_sha,
            'input_pins': inputs, 'omitted_selectors': omissions, 'public_contract_sha256': digest(additions[public.CONTRACT]),
            'new_fits': 0, 'new_responses': 0, 'package_built': False}))
        snapshot.store(directory, 'plan.json', encoded(final))
        snapshot.seal_directory(directory)
        info, opened = os.stat(destination, follow_symlinks=False), os.fstat(directory)
        require((info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino), 'preparation custody location changed')
    finally:
        os.close(directory)
    return {'plan_sha256': digest(encoded(final)), 'contract_sha256': digest(additions[public.CONTRACT]),
            'inherited_files': public.OLD_FILES, 'inherited_bindings': public.OLD_BINDINGS,
            'package_built': False, 'public_release_approved': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--report-acceptance-sha256', required=True)
    parser.add_argument('--accounting-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.request, args.request_sha256, args.destination,
        acceptance_sha=args.report_acceptance_sha256, accounting_sha=args.accounting_sha256), indent=2))
