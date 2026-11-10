"""Pure saved-metadata assembly; no filesystem, producer imports or clock reads.

Independent input pins and adopted authority are supplied by the caller. Outputs
are preflight metadata, never GPU qualification or scientific acceptance.
"""
import datetime
import hashlib
import json
from pathlib import PurePosixPath
import re

ROLES = {'authority', 'failure_acceptance', 'failure_proof', 'baseline', 'cpu_acceptance',
         'before_observation', 'after_observation', 'custody', 'remote_custody', 'readset',
         'original_gpu_packet', 'original_cpu_packet'}
FULL_MAP_ROLES = {'before_observation', 'after_observation', 'custody', 'remote_custody', 'cpu_acceptance'}
EXCLUDED = 'worker/artifacts/campaign'
ENTRY = 'qualify_llm_intervention_scientific_campaign_gpu.py'
CHECKER = 'llm_intervention_cpu_preflight_receipt.py'
HISTORY = 'llm_intervention_cpu_history_admission.py'
ORIGINAL_ENTRY = '2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
ORIGINAL_PACKET = '436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'
ORIGINAL_COMMIT = 'a9346022f145cc29439325b6f291597b2e6ec037'
JOB = '35498218'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, names, role):
    need(type(value) is dict and set(value) == set(names), 'exact '+role+' fields required')


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(canonical(value))


def decode(data):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs)
    canonical(value)
    return value


def path(value, absolute=False):
    need(type(value) is str and value and '\\' not in value
         and not any(ord(c) < 32 or ord(c) == 127 for c in value), 'plain metadata path required')
    p = PurePosixPath(value)
    need(str(p) == value and p.is_absolute() is absolute and value != '/'
         and all(s not in ('', '.', '..') for s in value.split('/')[1 if absolute else 0:]),
         'normalized metadata path required')
    return p


def utc(value):
    need(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)', value),
         'explicit saved UTC timestamp required')
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def zulu(value):
    return utc(value).isoformat().replace('+00:00', 'Z')


def inventory(value):
    keys(value, ('files', 'directories'), 'inventory')
    files, dirs = value['files'], value['directories']
    need(type(files) is dict and type(dirs) is list and all(type(n) is str for n in dirs),
         'original file map and directory list required')
    need(len(dirs) == len(set(dirs)) and not set(files).intersection(dirs), 'duplicate/colliding members')
    parents = set(dirs)
    for n in list(files)+dirs:
        p = path(n)
        need(all(str(q) in parents for q in p.parents if str(q) != '.'), 'missing directory parent')
    for info in files.values():
        keys(info, ('bytes', 'sha256'), 'file information')
        need(type(info['bytes']) is int and info['bytes'] >= 0 and pin(info['sha256']), 'invalid file information')
    # Collector traversal order is legitimate. Normalize only AFTER type and uniqueness checks.
    return {'files': files, 'directories': sorted(dirs)}


def observation(value):
    need(type(value) is dict and type(value.get('returncode')) is int and value['returncode'] == 0
         and type(value.get('stdout')) is str and type(value.get('python_source')) is str,
         'successful captured inventory transport required')
    found = decode(value['stdout'].encode('utf-8'))
    need(found.get('root_present') is True and found.get('cpu_present') is True
         and found.get('terminal_gate') is True and found.get('upstream_errors') == {},
         'complete terminal/root inventory observation required')
    need(utc(found['utc']) <= utc(value['utc']), 'observation/transport UTC order differs')
    return found, inventory({k: found[k] for k in ('files', 'directories')})


def read_names():
    result = {'campaign.json', 'source-plan-pins.json', 'start.json'}
    for i in range(5):
        prefix = f'policy-{i:02d}/'
        result.update(prefix+n for n in ('attempt.json', 'attempt.candidate.json'))
        result.update(prefix+f'world-{j:02d}/arm-receipt.json' for j in range(24))
    return result


def sub_inventory(full, prefix):
    need(prefix in full['directories'], 'observed subtree root missing')
    start = prefix+'/'
    return inventory({'files': {n[len(start):]: v for n, v in full['files'].items() if n.startswith(start)},
                      'directories': [n[len(start):] for n in full['directories'] if n.startswith(start)]})


def no_member(full, prefix):
    need(all(n != prefix and not n.startswith(prefix+'/') for n in list(full['files'])+full['directories']),
         'forbidden GPU output or CPU service member')


def summary(value, inventories):
    return {'canonical_sha256': digest(value),
            'file_count': sum(len(i['files']) for i in inventories.values()),
            'directory_count': sum(len(i['directories']) for i in inventories.values()),
            'total_bytes': sum(v['bytes'] for i in inventories.values() for v in i['files'].values())}


def assemble(*, inputs, pins, assembled_utc):
    """Return three immutable JSON byte strings from independently pinned inputs.

    authority is an earlier adopted record; it must not depend on these outputs.
    original_gpu_packet is the original inner packet, not the outer coordinator.
    custody is the ORIGINAL two-copy custody-failure-inventory.json. Its failure
    status is preserved; accepted failure audit is the authority for its use.
    """
    keys(inputs, ROLES, 'input roles'); keys(pins, ROLES, 'pin roles')
    need(sum(len(b) for b in inputs.values() if type(b) is bytes) <= 512*1024**2, 'bounded total metadata required')
    data = {}
    for role in sorted(ROLES):
        maximum = (16 if role == 'readset' else 128 if role in FULL_MAP_ROLES else 32)*1024**2
        b = inputs[role]
        need(type(b) is bytes and 0 < len(b) <= maximum and pin(pins[role]) and sha(b) == pins[role],
             'independently pinned bounded bytes required: '+role)
        data[role] = decode(b)
    a, f, proof, baseline, cpu, seal, backup, reads, old, cp = (
        data[k] for k in ('authority', 'failure_acceptance', 'failure_proof', 'baseline', 'cpu_acceptance',
                         'custody', 'remote_custody', 'readset', 'original_gpu_packet', 'original_cpu_packet'))
    need(a.get('schema') == 'ace-gpu-cpu-history-recovery-authority-v1'
         and a.get('status') == 'operationally_adopted' and a.get('historical_postcustody_required') is True
         and a.get('unchanged_scientific_inputs') is True
         and a.get('original_entry_sha256') == ORIGINAL_ENTRY and a.get('original_source_commit') == ORIGINAL_COMMIT
         and a.get('original_gpu01_packet_sha256') == ORIGINAL_PACKET and a.get('original_gpu01_job_id') == JOB,
         'independent adopted recovery authority required')
    need(not {'execution_context_sha256', 'context_sha256', 'receipt_sha256', 'packet_sha256',
              'expected_sha256', 'future_packet_sha256'}.intersection(a), 'authority cannot depend on future assembly outputs')
    need(f.get('schema') == 'ace-scientific-campaign-gpu-failure-acceptance-v1'
         and f.get('status') == 'accepted_failed_prepolicy_attempt' and f.get('acceptance') is True
         and type(f.get('required_findings')) is int and f['required_findings'] == 0 and f.get('job_id') == JOB
         and f.get('complete_CURC_backup_verified_from_saved_records') is True
         and f.get('all_selected_local_files_verified_in_both_copies') is True
         and f.get('originals_changed') is False and f.get('scientific_qualified') is False,
         'independent complete failed-attempt acceptance required')
    for k in ('new_policy_attempts', 'new_scientific_responses', 'new_training_updates', 'new_world_realizations'):
        need(type(f.get(k)) is int and f[k] == 0 and type(proof.get(k)) is int and proof[k] == 0,
             'accepted prepolicy zero-work boundary required')
    need(proof.get('schema') == 'ace-scientific-campaign-gpu-no-policy-work-proof-v1'
         and proof.get('status') == 'verified_no_new_policy_work' and proof.get('job_id') == JOB
         and type(proof.get('required_findings')) is int and proof['required_findings'] == 0
         and f['no_policy_work_proof_sha256'] == pins['failure_proof'], 'failure proof binding differs')
    for role, name in (('cpu_acceptance', 'predecessor_acceptance'), ('custody', 'original_failure_seal'),
                       ('remote_custody', 'remote_custody')):
        need(f['input_pins'][name]['sha256'] == pins[role], 'accepted failure input pin differs: '+role)
    need(proof['before']['sha256'] == pins['before_observation']
         and proof['after']['sha256'] == pins['after_observation']
         and proof['original_cpu_acceptance']['sha256'] == pins['cpu_acceptance']
         and proof['immutable_outer_packet']['sha256'] == ORIGINAL_PACKET,
         'accepted observation/CPU/original packet binding differs')
    need(baseline.get('schema') == 'ace-scientific-screen-accounting-baseline-v1'
         and baseline['input_pins']['acceptance'] == pins['failure_acceptance']
         and baseline['input_pins']['predecessor'] == f['input_pins']['predecessor_baseline']['sha256']
         and baseline['input_pins']['disposition'] == f['input_pins']['original_disposition']['sha256']
         and type(baseline.get('terminal_jobs')) is int and baseline['terminal_jobs'] == 47
         and type(baseline.get('submission_journals')) is int and baseline['submission_journals'] == 96
         and len(baseline['related_accounting']) == len({row[0] for row in baseline['related_accounting']}) == 47
         and len(baseline['journal_pins']) == 96 and baseline['outstanding_reservation'] == 0,
         'accepted complete47/96 baseline required')
    for k in ('closed_cpu_seconds', 'closed_gpu_slice_seconds', 'cpu_admission_seconds'):
        need(type(baseline[k]) is int and baseline[k] == f[k], 'failure/baseline accounting differs')
    need(a['baseline'] == {'sha256': pins['baseline'], 'terminal_jobs': 47, 'submission_journals': 96}
         and a['failed_gpu01_acceptance_sha256'] == pins['failure_acceptance']
         and a['cpu_acceptance_sha256'] == pins['cpu_acceptance'], 'authority acceptance/baseline differs')
    proof_pins = {'custody_sha256': pins['custody'], 'before_observation_sha256': pins['before_observation'],
                  'after_observation_sha256': pins['after_observation'], 'read_set_artifact_sha256': pins['readset']}
    need(a['historical_proof_pins'] == proof_pins, 'authority historical proof pins differ')
    need(cpu.get('schema') == 'ace-scientific-campaign-cpu-acceptance-v1'
         and cpu.get('status') == 'development_cpu_phase_accepted' and cpu.get('scientific_qualified') is False,
         'independent original CPU acceptance required')
    roots = {k: cpu[k] for k in ('cpu_output', 'campaign_root')}; roots['campaign'] = roots.pop('campaign_root')
    for value in roots.values(): path(value, True)
    need(roots['campaign'] == roots['cpu_output']+'/'+EXCLUDED
         and cpu['cpu_output_excluded_directory'] == EXCLUDED, 'accepted CPU roots differ')
    old_root = path(a['failed_gpu01_root'], True); control = path(a['control_root'], True)
    need(a['allowed_new_output'] == str(old_root/'gpu'), 'only original absent GPU01/gpu may be appended')
    for original in (old_root, path(roots['cpu_output'], True)):
        need(control != original and control not in original.parents and original not in control.parents,
             'new control overlaps immutable original roots')
    need(seal.get('acceptance') is False and seal.get('identical') is True
         and type(seal.get('copies')) is dict and len(seal['copies']) == 2, 'original two-copy failure seal required')
    copies = []
    for root, inv in seal['copies'].items():
        path(root, True); copies.append(inventory(inv))
    need(copies[0] == copies[1], 'original failure seal copies differ')
    for role, member in (('before_observation', 'inventory_transport.json'),
                         ('after_observation', 'inventory_after_transport.json'), ('remote_custody', 'remote-custody.json')):
        need(copies[0]['files'].get(member) == {'bytes': len(inputs[role]), 'sha256': pins[role]},
             'sealed original proof member differs: '+role)
    need(copies[0]['files']['pull_terminal.py']['sha256'] == a['historical_verifier_sha256'], 'historical verifier source differs')
    before, full_before = observation(data['before_observation'])
    after, full_after = observation(data['after_observation'])
    need(data['before_observation']['python_source'] == data['after_observation']['python_source'], 'observation source changed')
    full = inventory(backup['full_inventory'])
    need(backup.get('schema') == 'ace-remote-custody-v1' and backup.get('full_backup_verified') is True
         and backup.get('full_local_copy') is False and backup.get('backup_kind') == 'ustar'
         and backup.get('acceptance') is False and backup.get('retry_allowed') is False
         and full == full_before == full_after, 'sealed before/after/archive full inventories differ')
    # Backend4161 archive inventory pins use its pretty/newline serialization,
    # distinct from the no-newline canonical digest of the preflight schemas.
    backup_digest = sha((json.dumps(full, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
    need(all(backup[k] == backup_digest for k in ('original_before_inventory_sha256',
         'original_after_inventory_sha256', 'project_readback_inventory_sha256')), 'archive readback/full map pins differ')
    need(pin(backup['backup_archive_sha256']) and type(backup['backup_archive_bytes']) is int
         and backup['backup_archive_bytes'] > 0, 'original backup archive identity required')
    path(backup['backup_archive_path'], True)
    need(backup['roots']['launch'] == str(old_root), 'failed launch root label differs')
    cpu_parent = path(backup['roots']['cpu'], True)
    need(cpu_parent in path(roots['cpu_output'], True).parents, 'CPU label root differs')
    prefix = 'cpu/'+str(path(roots['cpu_output'], True).relative_to(cpu_parent))
    output = sub_inventory(full, prefix); campaign = sub_inventory(output, EXCLUDED)
    output = inventory({'files': {n: v for n, v in output['files'].items() if n != EXCLUDED and not n.startswith(EXCLUDED+'/')},
                        'directories': [n for n in output['directories'] if n != EXCLUDED and not n.startswith(EXCLUDED+'/')]})
    inventories = {'cpu_output': output, 'campaign': campaign}
    for role, accepted in (('cpu_output', 'cpu_output_files'), ('campaign', 'campaign_files')):
        need({n: v['sha256'] for n, v in inventories[role]['files'].items()} == cpu[accepted], 'complete accepted CPU map differs')
    need({n.split('/')[0] for n in list(campaign['files'])+campaign['directories']} ==
         {'campaign.json', 'source-plan-pins.json', 'start.json'}|{f'policy-{i:02d}' for i in range(5)}, 'five original CPU policy roots required')
    for observed in (full_before, full_after):
        no_member(observed, 'launch/gpu')
        for i in range(5): no_member(observed, prefix+'/'+EXCLUDED+f'/policy-{i:02d}/service')
    for field, member in (('namespace_sha256', 'namespace.json'), ('original_namespace_sha256', 'original-namespace.json')):
        need(full['files'][member]['sha256'] == a[field], 'original namespace pin differs')
    keys(reads, ('schema', 'cpu_acceptance_sha256', 'campaign_root', 'files'), 'readset')
    need(reads['schema'] == 'ace-cpu-history-readset-v1' and reads['cpu_acceptance_sha256'] == pins['cpu_acceptance']
         and reads['campaign_root'] == roots['campaign'], 'actual readset binding differs')
    keys(reads['files'], read_names(), '133 readset members')
    consumed = {}
    for name, info in reads['files'].items():
        keys(info, ('bytes', 'sha256', 'utf8'), 'readset member')
        need(type(info['utf8']) is str and type(info['bytes']) is int, 'exact lossless readset fields required')
        raw = info['utf8'].encode('utf-8'); retained = {'bytes': len(raw), 'sha256': sha(raw)}
        need(retained == {'bytes': info['bytes'], 'sha256': info['sha256']} == campaign['files'].get(name),
             'readset bytes differ from complete observed accepted map')
        consumed[name] = retained
    need(sum(v['bytes'] for v in consumed.values()) <= 16*1024**2, 'bounded consumed metadata required')
    for role, member in (('original_gpu_packet', 'launch/inputs/packet.json'), ('original_cpu_packet', 'launch/inputs/cpu-packet.json')):
        need(full['files'][member] == {'bytes': len(inputs[role]), 'sha256': pins[role]}, 'original packet custody differs')
    need(old['cpu_packet_sha256'] == pins['original_cpu_packet'] == cpu['cpu_packet_sha256']
         and old['cpu_acceptance_sha256'] == pins['cpu_acceptance']
         and old['paths']['cpu_output'] == cp['cpu_output'] == roots['cpu_output']
         and old['paths']['campaign'] == roots['campaign'] and old['paths']['gpu_output'] == a['allowed_new_output'],
         'original scientific packet/root identity differs')
    old_sources = {n: old['files'][n] for n in old['source_names']}; sources = a['operational_source_files']
    need(len(old['source_names']) == len(old_sources) == 54 and old_sources[ENTRY] == ORIGINAL_ENTRY
         and type(sources) is dict and len(sources) == 56 and set(sources) == set(old_sources)|{CHECKER, HISTORY}
         and all(pin(h) for h in sources.values()) and sources[ENTRY] != ORIGINAL_ENTRY
         and all(sources[n] == h for n, h in old_sources.items() if n != ENTRY), 'exact54-to56 operational source delta required')
    for k in ('resources', 'limits'):
        if k in a: need(canonical(a[k]) == canonical(old[k]), 'authority changed original '+k)
    full_value = {'cpu_roots': roots, 'cpu_output_excluded_directory': EXCLUDED, 'inventories': inventories}
    full_summary = summary(full_value, inventories)
    read_summary = {'canonical_sha256': digest(consumed), 'file_count': 133, 'total_bytes': sum(v['bytes'] for v in consumed.values())}
    context = {'schema': 'ace-gpu-cpu-preflight-context-v1', 'control_root': str(control),
               'gpu_output': a['allowed_new_output'], 'failed_gpu01_root': str(old_root), 'cpu_roots': roots,
               'lineage': {'cpu_acceptance_sha256': pins['cpu_acceptance'], 'failed_gpu01_acceptance_sha256': pins['failure_acceptance'],
                           'recovery_authority_sha256': pins['authority'], 'source_sha256': a['historical_verifier_sha256'],
                           'checker_sha256': sources[CHECKER], 'namespace_sha256': a['namespace_sha256'],
                           'original_namespace_sha256': a['original_namespace_sha256']},
               'baseline': a['baseline'], 'proof_pins': proof_pins, 'source_files_sha256': digest(sources),
               'model_files_sha256': digest(cp['model_files']), 'resources_sha256': digest(old['resources']),
               'limits_sha256': digest(old['limits']), 'cpu_inventory_sha256': full_summary['canonical_sha256'],
               'consumed_read_set_sha256': read_summary['canonical_sha256']}
    started, finished, upper = zulu(before['utc']), zulu(data['after_observation']['utc']), zulu(a['receipt_not_after_utc'])
    need(utc(started) <= utc(data['before_observation']['utc']) <= utc(after['utc']) <= utc(finished)
         <= utc(upper) and utc(finished) <= utc(assembled_utc), 'saved observation/authority/assembly UTC order differs')
    expected = {'schema': 'ace-gpu-cpu-preflight-expected-v1', 'execution_context': context,
                'execution_context_sha256': digest(context), 'cpu_inventories': inventories,
                'consumed_read_set': consumed, 'receipt_not_after_utc': upper}
    receipt = {'schema': 'ace-gpu-cpu-preflight-receipt-v1', 'phase': 'CPU-prelaunch-complete',
               'execution_context_sha256': digest(context), 'inventory_before': full_summary,
               'inventory_after': full_summary, 'consumed_read_set': read_summary,
               'absent_paths': sorted([a['allowed_new_output']]+[roots['campaign']+f'/policy-{i:02d}/service' for i in range(5)]),
               'started_utc': started, 'finished_utc': finished}
    expected_raw, receipt_raw = canonical(expected)+b'\n', canonical(receipt)+b'\n'
    need(len(expected_raw) <= 128*1024**2 and len(receipt_raw) <= 128*1024**2, 'bounded emitted preflight metadata required')
    record = {'schema': 'ace-gpu-cpu-preflight-assembly-v1', 'input_pins': dict(pins), 'assembled_utc': zulu(assembled_utc),
              'expected_sha256': sha(expected_raw), 'receipt_sha256': sha(receipt_raw), 'execution_context_sha256': digest(context),
              'saved_times': {'before_scan': zulu(before['utc']), 'before_transport': zulu(data['before_observation']['utc']),
                              'after_scan': zulu(after['utc']), 'after_transport': finished},
              'archive': {k: backup[k] for k in ('backup_archive_path', 'backup_archive_sha256', 'backup_archive_bytes')},
              'original_collector_returncode': f['original_collector_returncode'],
              'retroactive_collector_success': False, 'full_external_postcustody_required': True,
              'continuous_immutability_proven': False, 'gpu_execution_qualified': False, 'acceptance': False,
              'scientific_qualified': False, 'fresh_remote_scan_performed': False}
    return {'expected_raw': expected_raw, 'receipt_raw': receipt_raw, 'assembly_record_raw': canonical(record)+b'\n'}
