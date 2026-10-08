"""Read-only compute disposition from pinned metadata, never predictions/models.

Keep recorded fit CPU, function CPU, elapsed wall, charged/reserved wall and
Slurm allocation separate. Some rows overlap. Unmeasured costs remain unknown;
the output deliberately has no summed sprint CPU estimate or science verdict.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from verify_delivery_release import relative

CONFIRMATION_WORKER_SHA = '2f93bacbd6c14c66a5aeac5daabcff50d47886eeec38885e6e83ab40d141baee'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'invalid nonnegative measurement: '+label)
    return value


def captured(path, expected):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink in metadata custody')
    require(path.stat().st_size <= 8*1024*1024, 'metadata exceeds bounded size')
    raw = path.read_bytes()
    require(len(raw) <= 8*1024*1024 and hashlib.sha256(raw).hexdigest() == expected,
            'metadata digest differs')
    return raw


def decoded(raw):
    def pairs(items):
        d = {}
        for k,v in items:
            require(k not in d, 'duplicate metadata key')
            d[k] = v
        return d
    def invalid(v):
        raise ValueError('nonfinite JSON constant: '+v)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def audit(root, manifest_sha, accepted_costs, accepted_costs_sha, fit_metadata,
          fit_metadata_sha, scheduler_metadata, scheduler_metadata_sha, confirmation_worker,
          scheduler_identity_mapping, scheduler_identity_mapping_sha):
    root = Path(root)
    manifest = decoded(captured(root/'manifest.json', manifest_sha))
    files = {f['path']:f for f in manifest['files']}
    require(len(files) == len(manifest['files']), 'duplicate packaged location')
    accessed = {}
    def metadata(name):
        require(name.endswith('.json') and name.startswith('F/') and
                name not in ('F/scores.json','F/statistics.json'), 'only confirmation cost metadata allowed')
        f = files[name]; raw = captured(relative(root, name), f['sha256'])
        accessed[name] = f['sha256']
        return decoded(raw)
    protocol = metadata('F/protocol.json'); runtime = metadata('F/runtime.json')
    require(runtime['adapter_sha256'] == CONFIRMATION_WORKER_SHA, 'original confirmation timer source required')
    worker = captured(confirmation_worker, runtime['adapter_sha256'])
    # Source custody establishes what the recorded timer measures. No import or
    # original-worker execution occurs; do not turn monotonic elapsed into CPU.
    require(b"'seconds':time.monotonic()-start" in worker and b"'refit_timings':timings" in worker,
            'confirmation timer provenance differs')
    seeds = protocol['seeds']
    require(len(seeds) == 12 and len(set(seeds)) == 12 and
            all(type(s) is int and s >= 0 for s in seeds) and protocol['delivery']['inits'] == [0,1,2],
            'complete frozen confirmation cohort required')
    expected_cases = {str(s) for s in seeds}
    require({n.split('/')[2] for n in files if n.startswith('F/cases/')} == expected_cases,
            'confirmation case membership differs')
    refits = []
    for seed in seeds:
        name = f'F/cases/{seed}/complete.json'; done = metadata(name)
        require(done['seed'] == seed and done['inits'] == [0,1,2] and
                len(done['refit_timings']) == 3 and
                [r['init'] for r in done['refit_timings']] == [0,1,2], 'refit timing membership differs')
        for r in done['refit_timings']:
            refits.append({'seed':seed,'init':r['init'],'recorded_wall_seconds':number(r['seconds'],'F refit wall')})
    first = metadata('F/original_terminal.json'); prior = metadata('F/prior_execution.json')
    final = metadata('F/execution.json'); terminal = metadata('F/terminal.json')
    require(prior['status'] == 'time_limit' and final['status'] == terminal['status'] == 'complete' and
            final['exit_code'] == 0 and prior['exit_code'] == -9, 'confirmation dispositions differ')
    def close(a,b):
        return math.isclose(number(a,'reserved accounting'), number(b,'reserved accounting'), rel_tol=1e-12, abs_tol=1e-8)
    require(close(prior['aggregate_charged_seconds'], prior['prior_charged_reserved_seconds']+prior['elapsed_seconds']) and
            close(final['prior_charged_reserved_seconds'], prior['aggregate_charged_seconds']) and
            close(final['aggregate_charged_seconds'], final['prior_charged_reserved_seconds']+final['elapsed_seconds']) and
            close(terminal['aggregate_charged_reserved_seconds'], final['aggregate_charged_seconds']) and
            terminal['execution_sha256'] == files['F/execution.json']['sha256'],
            'nested charged wall accounting differs')
    costs = decoded(captured(accepted_costs, accepted_costs_sha))
    fits = decoded(captured(fit_metadata, fit_metadata_sha))
    scheduler = decoded(captured(scheduler_metadata, scheduler_metadata_sha))
    mapping = decoded(captured(scheduler_identity_mapping, scheduler_identity_mapping_sha))
    require(costs['no_sum_across_incompatible_measurement_scopes'] is True and
            costs['total_sprint_cpu_core_hours'] is None, 'prior cost scope differs')
    require(fits['source_revision'] == scheduler['source_revision'] == '45ebeb89d2c76daa97a55f07728239245e0c4f60' and
            fits['registration_sha256'] == scheduler['registration_sha256'] ==
            'e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5' and
            not fits['failed_attempts'] and not scheduler['failed_scheduler_states'] and
            scheduler['evaluation_scores_accessed'] is False and
            fits['charged_training_responses'] == 32000 and fits['histories'] == 80 and fits['matrix_fits'] == 640,
            'original B metadata scope/identity differs')
    require(mapping['schema'] == 'delivery-slurm-identity-map-v1' and
            mapping['registration_sha256'] == scheduler['registration_sha256'] and
            mapping['source_revision'] == scheduler['source_revision'] and mapping['account'] == 'ucb736_asc1' and
            re.fullmatch(r'[0-9a-f]{64}', mapping['captured_scheduler_receipt_sha256']) is not None,
            'authenticated ACE scheduler mapping required')
    pairs = mapping['pairs']
    require(len({p['job_id_raw'] for p in pairs}) == len(pairs), 'duplicate scheduler mapping raw identity')
    authorized = {p['job_id_raw']:p['job_id'] for p in pairs}
    reserved_basis = number(mapping['reserved_CPU_core_hours'], 'original reserved core hours')
    require(reserved_basis <= 150 and mapping['reservation_basis'] == 'original full-registration total_requested_cpu_core_hours',
            'original reservation basis required')
    reserved = scheduler['rounded_total_reserved_CPU_core_hours']
    reservation_unknown = None
    if reserved is None:
        reservation_unknown = scheduler.get('reserved_core_hours_unavailable_reason')
        require(isinstance(reservation_unknown,str) and reservation_unknown.strip(), 'unknown reservation reason required')
    else:
        require(number(reserved,'reserved core hours') == reserved_basis, 'reserved basis differs')
    cells = fits['validated_completed_fits']
    indices = [f['index'] for f in cells]
    require(len(cells) == fits['completed_fit_receipts'] and len(set(indices)) == len(cells) and
            all(type(i) is int and 0 <= i < 640 for i in indices), 'distinct B receipt indices required')
    fit_cpu = sum(number(f['cpu_seconds'],'B fit CPU') for f in cells)
    fit_wall = sum(number(f['wall_seconds'],'B fit wall') for f in cells)
    allocations = scheduler['scheduler']; ids = [a['job_id_raw'] for a in allocations]
    require(len(set(ids)) == len(ids), 'duplicate Slurm raw identity')
    require(len({a['job_id'] for a in allocations}) == len(allocations), 'duplicate Slurm interpreted identity')
    allocation = 0
    coverage = set()
    for a in allocations:
        raw_id = a['job_id_raw']
        require(isinstance(raw_id,str) and re.fullmatch(r'[1-9][0-9]*',raw_id) is not None and
                a['state'] in ('COMPLETED','RUNNING','PENDING') and
                a['exit_code'] == '0:0', 'failed or child-step allocation')
        seconds = number(a['allocated_cpu_seconds_accrued'],'allocated CPU seconds')
        job = a['job_id']
        require(authorized.get(raw_id) == job, 'unmapped or contradictory raw/interpreted Slurm pair')
        known = job in ('33507418','33507419','33507422','33507423')
        if known:
            require(raw_id == job, 'scalar Slurm raw identity differs')
        single = re.fullmatch(r'(3350742[01])_(0|[1-9][0-9]*)', job)
        group = re.fullmatch(r'(3350742[01])_\[(0|[1-9][0-9]*)-(0|[1-9][0-9]*)%2\]', job)
        tasks = set()
        if single is not None and 0 <= int(single[2]) < 20:
            known = True;tasks = {(single[1],int(single[2]))}
        if group is not None and 0 <= int(group[2]) <= int(group[3]) < 20 and a['state'] == 'PENDING' and seconds == 0:
            known = True;tasks = {(group[1],i) for i in range(int(group[2]),int(group[3])+1)}
        require(known, 'unrelated or malformed Slurm identity')
        require(not coverage.intersection(tasks), 'overlapping array task coverage')
        coverage.update(tasks)
        require(type(a['allocated_cpus']) is int and a['allocated_cpus'] in (0,1) and
                seconds == number(a['elapsed_seconds'],'allocation wall')*a['allocated_cpus'],
                'allocation telemetry differs')
        allocation += seconds
    require(allocation == scheduler['current_chain_allocated_CPU_seconds_accrued'], 'allocation sum differs')
    pilots = costs['B_partial']['actual_pilot_allocated_seconds']
    require(pilots == {'failed':126,'successful':119} and
            scheduler['historical_failed_pilot_allocated_CPU_seconds'] == pilots['failed'], 'pilot allocation identity differs')
    return {
        'schema':'delivery-compute-disposition-v1','status':'bounded measured scopes; incomplete total research cost',
        'inputs':{'manifest_sha256':manifest_sha,'accepted_cost_disposition_sha256':accepted_costs_sha,
                  'fit_metadata_sha256':fit_metadata_sha,'scheduler_metadata_sha256':scheduler_metadata_sha,
                  'scheduler_identity_mapping_sha256':scheduler_identity_mapping_sha,
                  'scheduler_mapping_capture_sha256':mapping['captured_scheduler_receipt_sha256'],
                  'confirmation_worker_sha256':runtime['adapter_sha256'],'confirmation_metadata':accessed},
        'A_C_prior_measured_scopes':{k:costs[k] for k in ('A_full','A_development_pilot','C_full','C_development_pilot')},
        'F_confirmation':{'saved_refits':36,'refit_wall_seconds_sum':sum(r['recorded_wall_seconds'] for r in refits),
            'refits':refits,'elapsed_wall_by_distinct_stage':{
                'first_history_timing_gate':number(first['elapsed_seconds'],'first F stage wall'),
                'retained_history_stage_with_interruption':number(prior['elapsed_seconds'],'prior F stage wall'),
                'authorized_final_same_seed_continuation':number(final['elapsed_seconds'],'final F stage wall')},
            'final_nested_charged_reserved_wall_seconds':final['aggregate_charged_seconds'],
            'process_cpu_seconds':None,
            'scope':'monotonic elapsed and reserved accounting, not CPU; refit walls overlap stage walls; stage walls exclude preparation/approval gaps; nested aggregates are not additive'},
        'B_current':{'at':scheduler['at'],'completed_fit_receipts':len(cells),'completed_worlds':len(fits['completed_worlds']),
            'completed_fit_cpu_seconds_sum':fit_cpu,'completed_fit_wall_seconds_sum':fit_wall,
            'original_chain_allocated_cpu_seconds_accrued':allocation,
            'pilot_allocated_cpu_seconds':pilots,'allocated_core_hours_including_pilots':(allocation+sum(pilots.values()))/3600,
            'reserved_CPU_core_hours':reserved,'reservation_unavailable_reason':reservation_unknown,
            'reservation_basis':mapping['reservation_basis'],
            'scope':'current unique fit workers versus distinct Slurm allocations; CPU/wall/allocation rows overlap and are not additive; incomplete tasks/evaluation/audit remain unknown'},
        'no_sum_across_incompatible_or_overlapping_scopes':True,'total_sprint_cpu_core_hours':None,
        'unknown_costs':costs['unknowns']+[
            'F acquisition/refit process CPU was not recorded; elapsed and sampled CPU percent cannot recover it',
            'all historical preparation, tests, review, transfer and metadata-check costs; individual measured subsets remain separate receipts'],
        'new_fits':0,'new_responses':0,'new_optimizer_updates':0,'B_scientific_outcomes_opened':False,
        'limitations':['Metadata pins authenticate captured inputs, not original historical freeze or inventory completeness.',
                       'Prior A/C scope summaries are inherited evidence; their original models are not replayed or cost receipts re-audited here.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('root','accepted-costs','fit-metadata','scheduler-metadata','confirmation-worker','scheduler-identity-mapping','out'):
        parser.add_argument('--'+n,type=Path,required=True)
    for n in ('manifest-sha','accepted-costs-sha','fit-metadata-sha','scheduler-metadata-sha','scheduler-identity-mapping-sha'):
        parser.add_argument('--'+n,required=True)
    args = vars(parser.parse_args()); out=args.pop('out')
    result=audit(**args)
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(out),'scope':result['status']}))
