"""Extend a private A/C plan with immutable confirmation and claim evidence.

Explicit derived approval/accounting metadata omits human/custody identifiers.
Original receipts remain unchanged. No fits, outcomes from B, or simulator calls.
This preparation is not a public release or complete historical attempt audit.
"""
import argparse
import json
from pathlib import Path

from build_delivery_release import write
from verify_delivery_release import sha
from replay_delivery_attribution_release import FROZEN_HISTORIES


def extend(plan_file, repo, confirmation, out):
    plan, repo, confirmation = json.loads(Path(plan_file).read_text()), Path(repo), Path(confirmation)
    regfile = confirmation/'protocol/registration.json'
    reg = json.loads(regfile.read_text())
    if set(map(str, reg['seeds'])) != FROZEN_HISTORIES or reg['delivery']['inits'] != [0, 1, 2]:
        raise ValueError('frozen confirmation cohort or initializations changed')
    from runner_delivery_confirmation import verify_seal
    seal = verify_seal(confirmation, reg)  # original whole-case semantic/byte audit
    scores = repo/'results/delivery_final_history_20261006/scores.json'
    stats = repo/'results/delivery_final_history_20261006/independent_statistics.json'
    terminal = repo/'results/delivery_final_history_20261006/terminal_audit.json'
    indexfile = repo/'paper/aistats_ace_2027/claim_index.json'
    index = json.loads(indexfile.read_text())
    for key, file in [('scores_sha256', scores), ('statistics_sha256', stats), ('terminal_audit_sha256', terminal)]:
        if sha(file) != index[key]:
            raise ValueError('manuscript confirmation evidence changed: '+key)
    if sha(scores) != sha(confirmation/'scores.json'):
        raise ValueError('compact and raw confirmation scores differ')
    indexed = {f['path']: f for f in plan['files']}
    def add(file, name, role, expected=None, keys=None):
        digest = sha(file)
        if expected is not None and digest != expected:
            raise ValueError('historical confirmation artifact differs: '+name)
        if name in indexed:
            if indexed[name]['original_sha256'] != digest:
                raise ValueError('shared archived artifact conflict')
            return
        entry = {'path': name, 'sources': [str(Path(file).resolve())], 'role': role, 'original_sha256': digest}
        if keys is not None:
            entry['transform'] = {'kind': 'project-json', 'keys': keys}
        plan['files'].append(entry); indexed[name] = entry
    def bind(record, pointer, artifact, digest='sha256'):
        plan['bindings'].append({'record': record, 'pointer': pointer, 'artifact': artifact, 'digest': digest})
    add(regfile, 'F/protocol.json', 'confirmation-protocol')
    add(confirmation/'sealed.json', 'F/sealed.json', 'original-confirmation-seal')
    for name, digest in reg['source_hashes'].items():
        add(confirmation/'source'/name, 'source/runner/'+name, 'archived-confirmation-source', digest)
        bind('F/protocol.json', '/source_hashes/'+name.replace('~', '~0').replace('/', '~1'), 'source/runner/'+name)
    for name, digest in seal['case_hashes'].items():
        add(confirmation/'cases'/name, 'F/cases/'+name, 'sealed-confirmation-artifact', digest)
        bind('F/sealed.json', '/case_hashes/'+name.replace('~', '~0').replace('/', '~1'), 'F/cases/'+name)
    # The interrupted acquisition is a distinct charged attempt, never a
    # returned-response history or an extra copy of a retained completed case.
    amendment = json.loads((confirmation/'amendment.json').read_text())
    prior = Path(amendment['prior_output'])
    auditfile = repo/'results/delivery_option_a_20261005/terminal_audit.json'
    audit = json.loads(auditfile.read_text())
    partial = prior/'cases/884825602/calls.jsonl'
    add(partial, 'F/attempts/884825602-interrupted/calls.jsonl', 'interrupted-charged-journal',
        audit['partial_inventory']['calls.jsonl'])
    add(auditfile, 'F/prior_terminal.json', 'derived-prior-attempt-accounting', keys=[
        'at', 'status', 'completed_cases', 'complete_calls', 'partial_seed', 'partial_journal_calls',
        'aggregate_calls', 'call_cap', 'sample_count', 'peak_combined_rss_bytes', 'rss_cap_bytes',
        'source_original_copy_hashes_unchanged', 'seal_present', 'scores_present',
        'case_reports', 'partial_inventory', 'execution_sha256', 'limitations'])
    add(prior/'execution.json', 'F/prior_execution.json', 'prior-interrupted-execution',
        amendment['prior_execution_sha256'])
    bind('F/prior_terminal.json', '/execution_sha256', 'F/prior_execution.json')
    bind('F/prior_terminal.json', '/partial_inventory/calls.jsonl', 'F/attempts/884825602-interrupted/calls.jsonl')
    for i, case in enumerate(audit['completed_cases']):
        for name, digest in case['case_hashes'].items():
            artifact = f"F/cases/{case['seed']}/{name}"
            if indexed[artifact]['original_sha256'] != digest:
                raise ValueError('retained prior case changed')
            bind('F/prior_terminal.json', '/completed_cases/'+str(i)+'/case_hashes/'+name.replace('/', '~1'), artifact)
    originalfile = repo/'results/runner_delivery_confirmation_20261005/audit.json'
    add(originalfile, 'F/original_terminal.json', 'derived-original-timing-gate-audit', keys=[
        'at', 'status', 'elapsed_seconds', 'calls', 'first_case', 'sampled_peak_rss_bytes',
        'rss_samples', 'independent_custody_checks_passed', 'eligible_rows', 'role_counts',
        'source_metadata_derived_counters', 'frozen_adapter_seal_defect', 'case_hashes',
        'primary_verdict', 'remaining_cases_started', 'scores_present', 'seal_present',
        'execution_sha256', 'runtime_sha256', 'approval_sha256', 'no_retry_or_expansion'])
    original = json.loads(originalfile.read_text())
    for name, digest in original['case_hashes'].items():
        artifact = 'F/cases/'+name
        if indexed[artifact]['original_sha256'] != digest:
            raise ValueError('original retained first history changed')
        bind('F/original_terminal.json', '/case_hashes/'+name.replace('/', '~1'), artifact)
    add(scores, 'F/scores.json', 'accepted-confirmation-scores', index['scores_sha256'])
    add(stats, 'F/statistics.json', 'independent-confirmation-statistics', index['statistics_sha256'])
    add(terminal, 'F/terminal.json', 'derived-confirmation-accounting', index['terminal_audit_sha256'], keys=[
        'at', 'status', 'all12_seal_verified', 'retained11_and_prior_partial_unchanged',
        'aggregate_calls_including_discarded', 'aggregate_charged_reserved_seconds',
        'amendment_stage_charged_reserved_seconds', 'peak_combined_rss_bytes', 'telemetry_samples',
        'owned_workers_remaining', 'execution_sha256', 'seal_sha256', 'scores_sha256', 'audit_interpreter'])
    add(confirmation/'approval.json', 'F/runtime.json', 'derived-runtime-approval', keys=[
        'approved', 'registry_reconciled', 'registration_sha256', 'adapter_sha256', 'dependency_versions', 'at'])
    add(confirmation/'execution.json', 'F/execution.json', 'confirmation-final-stage-execution')
    bind('F/runtime.json', '/registration_sha256', 'F/protocol.json')
    bind('F/terminal.json', '/seal_sha256', 'F/sealed.json')
    bind('F/terminal.json', '/scores_sha256', 'F/scores.json')
    bind('F/terminal.json', '/execution_sha256', 'F/execution.json')
    bind('F/statistics.json', '/score_sha256', 'F/scores.json')
    bind('F/protocol.json', '/evaluation/file_sha256', 'A/grid.npz')
    add(indexfile, 'claims/claim_index.json', 'manuscript-claim-index')
    for key, artifact in [('scores_sha256', 'F/scores.json'), ('statistics_sha256', 'F/statistics.json')]:
        bind('claims/claim_index.json', '/'+key, artifact)
    bind('claims/claim_index.json', '/terminal_audit_sha256', 'F/terminal.json', 'original_sha256')
    generator = repo/'scripts/research/generate_delivery_claims.py'
    add(generator, 'source/release/generate_delivery_claims.py', 'original-claim-generator', index['generator_sha256'])
    bind('claims/claim_index.json', '/generator_sha256', 'source/release/generate_delivery_claims.py')
    for name in ['delivery_claims.tex', 'delivery_history_table.tex', 'delivery_attribution_table.tex', 'delivery_physical_table.tex']:
        add(repo/'paper/aistats_ace_2027'/name, 'claims/'+name, 'receipt-generated-manuscript-companion')
    for key, name, file in [('gate_sha256','gate.json',repo/'results/delivery_paper_implementation_20261006/attribution_gate.json'),
                            ('summary_sha256','summary.json',repo/'results/delivery_attribution_20261006/summary.json'),
                            ('complete_sha256','complete.json',repo/'results/delivery_attribution_20261006/complete.json')]:
        add(file, 'A/'+name, 'accepted-attribution-analysis', index['attribution'][key])
        bind('claims/claim_index.json', '/attribution/'+key, 'A/'+name)
    for case, digest in index['attribution']['score_hashes'].items():
        if indexed['A/'+case+'/scores.json']['original_sha256'] != digest:
            raise ValueError('claim index A score mismatch')
        bind('claims/claim_index.json', '/attribution/score_hashes/'+case, 'A/'+case+'/scores.json')
    acceptance = repo/'results/delivery_chambers_20261006/acceptance.json'
    add(acceptance, 'C/acceptance.json', 'derived-physical-acceptance', index['physical']['acceptance_sha256'], keys=[
        'at', 'full_acceptance', 'artifact_hashes_verified', 'archive_rows_splits_normalizers_verified',
        'scores_bootstrap_recomputed', 'selected_gate_sha256', 'projection_sha256', 'receipt_hashes',
        'cpu_core_hours', 'scope', 'new_queries', 'conditions', 'descriptive_comparisons', 'physical_boundary'])
    bind('claims/claim_index.json', '/physical/acceptance_sha256', 'C/acceptance.json', 'original_sha256')
    bind('C/acceptance.json', '/selected_gate_sha256', 'A/gate.json')
    for name in ['protocol.json', 'scores.json', 'fit_seal.json']:
        digest = 'original_sha256' if name == 'protocol.json' else 'sha256'
        bind('claims/claim_index.json', '/physical/receipt_hashes/'+name, 'C/'+name, digest)
        bind('C/acceptance.json', '/receipt_hashes/'+name, 'C/'+name, digest)
    add(Path(__file__).with_name('replay_delivery_confirmation_release.py'),
        'replay_delivery_confirmation_release.py', 'confirmation-replay-adapter')
    add(Path(__file__).with_name('verify_delivery_claims_release.py'),
        'verify_delivery_claims_release.py', 'empirical-macro-analysis-adapter')
    if 'confirmation_accounting' in index:
        add(repo/'results/delivery_release_preparation_20261007/confirmation_verification.json',
            'F/accounting_verification.json', 'supplemental-confirmation-verification',
            index['confirmation_accounting']['verification_sha256'])
        bind('claims/claim_index.json', '/confirmation_accounting/verification_sha256', 'F/accounting_verification.json')
    plan['status'] = 'private A/C/confirmation evidence preparation; B/all-sprint accounting/public release pending'
    write(out, plan)
    return {'plan_sha256': sha(out), 'files': len(plan['files']), 'bindings': len(plan['bindings']),
            'confirmation_sealed_files': len(seal['case_hashes']), 'new_fits': 0, 'new_responses': 0,
            'full_attempt_accounting': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'repo', 'confirmation', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extend(args.plan, args.repo, args.confirmation, args.out), indent=2))
