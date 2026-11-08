"""Prepare a separately hashed B numerical core, without opening B outcomes.

Exact function-body extraction is an explicit derived-source operation, not an
edited frozen worker or bypass of its guards. Full original acceptance and its
supervisor remain mandatory. This tool only prepares supplemental reproduction.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

REGISTRATION_SHA = 'e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5'
REVISION = '45ebeb89d2c76daa97a55f07728239245e0c4f60'
FUNCTIONS = ('match', 'history_receipt', 'replay_predictions', 'calibration_ranges',
             'primary_statistics', 'secondary_summary')
GLOBALS = ('ARMS', 'HISTORIES', 'WORLD_IDS', 'REPLAY_RTOL', 'REPLAY_ATOL')
DEPENDENCIES = {'torch': '2.9.1', 'numpy': '2.2.6', 'scipy': '1.15.3',
                'pandas': '2.3.3', 'sympy': '1.14.0', 'PyYAML': '6.0.3'}
OUTCOME_FIELDS = {'primary_analysis', 'secondary_descriptive_analysis'}
ACCEPTANCE_FIELDS = {'at', 'full_acceptance', 'study_registration_sha256', 'scores_sha256',
    'complete_sha256', 'fit_seal_sha256', 'source_revision', 'auditor_sha256', 'runtime',
    'training_receipt_hashes', 'phase_execution_hashes', 'evaluation_receipt_hashes',
    'fits_checked', 'checkpoints_replayed', 'primary_cells_checked', 'charged_cached_responses',
    'new_simulator_responses', 'replay_tolerance', 'replay_max_abs_deltas', 'scoring_predictions',
    'audit_cpu_seconds', 'audit_wall_seconds', 'primary_analysis', 'scope',
    'secondary_descriptive_analysis'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_snapshot(path, expected=None):
    raw=Path(path).read_bytes(); h=hashlib.sha256(raw).hexdigest()
    if expected is not None and h!=expected: raise ValueError('metadata snapshot digest differs')
    return json.loads(raw),h


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def acceptance_projection(path, expected):
    """Custody-bound metadata projection, without decoding either analysis.

    Hashing original bytes is distinct from parsing scientific outcomes. The
    strict JSON scanner skips analysis values syntactically, rather than loading
    the whole receipt and removing outcomes afterward. No numeric analysis value
    is decoded or returned. Unknown/duplicate top-level fields are rejected.
    """
    raw=Path(path).read_bytes(); snapshot_hash=hashlib.sha256(raw).hexdigest()
    if snapshot_hash!=expected: raise ValueError('acceptance snapshot differs from supervisor binding')
    text=raw.decode('utf-8'); decoder = json.JSONDecoder()
    def ws(i):
        while i < len(text) and text[i] in ' \t\r\n': i += 1
        return i
    def string_end(i):
        if i >= len(text) or text[i]!='"': raise ValueError('invalid JSON string')
        i+=1
        while i < len(text):
            c=text[i]
            if c=='"': return i+1
            if ord(c)<32: raise ValueError('unescaped JSON control character')
            if c=='\\':
                i+=1
                if i >= len(text): raise ValueError('truncated JSON escape')
                if text[i]=='u':
                    if not re.fullmatch('[0-9a-fA-F]{4}',text[i+1:i+5]): raise ValueError('invalid Unicode escape')
                    i+=4
                elif text[i] not in '"\\/bfnrt': raise ValueError('invalid JSON escape')
            i+=1
        raise ValueError('unterminated JSON string')
    def skip(i):
        i=ws(i)
        if i >= len(text): raise ValueError('truncated acceptance JSON')
        c=text[i]
        if c=='"': return string_end(i)
        if c in '[{':
            endchar=']' if c=='[' else '}'; i=ws(i+1)
            if i < len(text) and text[i]==endchar: return i+1
            while True:
                if c=='{':
                    i=string_end(ws(i))
                    i=ws(i)
                    if i >= len(text) or text[i]!=':': raise ValueError('missing JSON colon')
                    i+=1
                i=ws(skip(i))
                if i >= len(text): raise ValueError('unterminated acceptance JSON')
                if text[i]==endchar: return i+1
                if text[i]!=',': raise ValueError('missing JSON comma')
                i=ws(i+1)
        for token in ('true','false','null'):
            if text.startswith(token,i): return i+len(token)
        number=re.match(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',text[i:])
        if number: return i+len(number.group())
        raise ValueError('invalid acceptance JSON value')
    i=ws(0)
    if i >= len(text) or text[i]!='{': raise ValueError('acceptance must be a JSON object')
    i=ws(i+1); keys=set(); metadata={}
    while i < len(text) and text[i]!='}':
        key,i=decoder.raw_decode(text,i)
        if not isinstance(key,str) or key in keys or key not in ACCEPTANCE_FIELDS:
            raise ValueError('unexpected or duplicate acceptance field')
        keys.add(key); i=ws(i)
        if i >= len(text) or text[i]!=':': raise ValueError('missing acceptance colon')
        i=ws(i+1)
        if key in OUTCOME_FIELDS:
            if i >= len(text) or text[i]!='{': raise ValueError('analysis must be an object')
            i=skip(i)
        else:
            metadata[key],i=decoder.raw_decode(text,i)
        i=ws(i)
        if i >= len(text): raise ValueError('truncated acceptance object')
        if text[i]=='}': break
        if text[i]!=',': raise ValueError('missing acceptance comma')
        i=ws(i+1)
        if i >= len(text) or text[i]=='}': raise ValueError('trailing acceptance comma')
    if i >= len(text) or text[i]!='}' or ws(i+1)!=len(text) or keys!=ACCEPTANCE_FIELDS:
        raise ValueError('incomplete acceptance structure or trailing bytes')
    return {'original_acceptance_sha256':snapshot_hash, 'original_fields':sorted(keys),
            'skipped_outcome_fields':sorted(OUTCOME_FIELDS), 'metadata':metadata,
            'scientific_outcome_values_decoded':False}


def evidence_membership(accepted):
    """Require the complete original custody/replay evidence, not only counts."""
    worlds={f'{size}:{i:02d}' for size in (5,30) for i in range(20)}
    if any(type(accepted[k]) is not int for k in ('fits_checked','checkpoints_replayed',
            'primary_cells_checked','charged_cached_responses','new_simulator_responses')): return False
    training={c+'/'+h for c in worlds for h in ('balanced_varied_value','matched_random')}
    for key, expected in [('training_receipt_hashes',training), ('evaluation_receipt_hashes',worlds)]:
        hashes=accepted[key]
        if not isinstance(hashes,dict) or set(hashes)!=expected: return False
        for files in hashes.values():
            if (not isinstance(files,dict) or set(files)!={'input.json','queries.ndjson','receipt.json'} or
                    not all(digest(h) for h in files.values())): return False
    phases=accepted['phase_execution_hashes']
    if (not isinstance(phases,dict) or set(phases)!={'qualification_execution.json','collect_execution.json','evaluate_execution.json'} or
            not all(digest(h) for h in phases.values())): return False
    deltas=accepted['replay_max_abs_deltas']
    if (not isinstance(deltas,dict) or set(deltas)!={str(i) for i in range(640)} or
            not all(finite_nonnegative(d) for d in deltas.values())): return False
    if accepted['replay_tolerance']!={'rtol':1e-6,'atol':1e-7,'roots':'exact'}: return False
    if accepted['scoring_predictions']!='unchanged original cached arrays': return False
    if not all(finite_nonnegative(accepted[k]) for k in ('audit_cpu_seconds','audit_wall_seconds')): return False
    return all(digest(accepted[k]) for k in ('study_registration_sha256','scores_sha256',
        'complete_sha256','fit_seal_sha256','auditor_sha256'))


def accepted_before_scores(root, expected_registration=REGISTRATION_SHA):
    """Validate upstream metadata first; never open scores or predictions.

    Original acceptance is an upstream gate, not a replacement for replay and
    derived-manifest verification. Its hash is checked before parsing it.
    """
    root = Path(root)
    if (root/'stop_new_fits.json').exists():
        raise ValueError('failure flag blocks acceptance before any outcome parsing')
    reg,_ = read_snapshot(root/'registration.json',expected_registration)
    complete,complete_hash = read_snapshot(root/'complete.json')
    supervisor,supervisor_hash = read_snapshot(root/'audit_execution.json')
    if (reg['source_revision'] != REVISION or reg['dependencies'] != DEPENDENCIES or
            reg['matrix_fits'] != 640 or reg['primary_cells'] != 240 or
            complete['n_fits'] != 640 or complete['n_primary_cells'] != 240 or
            complete['charged_responses'] != 48000 or complete['account'] != 'ucb736_asc1' or
            complete['registration_sha256'] != expected_registration or
            complete['fit_seal_sha256'] != sha(root/'fit_seal.json') or
            supervisor['status'] != 'complete' or type(supervisor['exit_code']) is not int or supervisor['exit_code'] != 0 or
            supervisor['account'] != 'ucb736_asc1' or
            supervisor['registration_sha256'] != expected_registration or
            not digest(supervisor['acceptance_sha256']) or
            type(supervisor['peak_tree_rss_bytes']) is not int or not finite_nonnegative(supervisor['peak_tree_rss_bytes']) or
            not finite_nonnegative(supervisor['elapsed_seconds']) or
            supervisor['peak_tree_rss_bytes'] > reg['resources']['rss_bytes'] or
            supervisor['elapsed_seconds'] > reg['resources']['audit_wall_seconds']):
        raise ValueError('full successful upstream audit supervisor required')
    # Scores remain unopened even after this point. The full replay will check
    # their bytes against both hash references before parsing outcome values.
    projection = acceptance_projection(root/'acceptance.json',supervisor['acceptance_sha256'])
    accepted = projection['metadata']
    if (accepted['full_acceptance'] is not True or accepted['fits_checked'] != 640 or
            accepted['checkpoints_replayed'] != 640 or accepted['primary_cells_checked'] != 240 or
            accepted['charged_cached_responses'] != 48000 or accepted['new_simulator_responses'] != 0 or
            accepted['source_revision'] != REVISION or accepted['runtime'] != DEPENDENCIES or
            accepted['study_registration_sha256'] != expected_registration or
            accepted['complete_sha256'] != complete_hash or
            accepted['fit_seal_sha256'] != complete['fit_seal_sha256'] or
            accepted['scores_sha256'] != complete['scores_sha256'] or
            accepted['auditor_sha256'] != reg['worker_hashes']['audit_delivery_prospective_results.py'] or
            accepted['scope'] != reg['scope'] or not evidence_membership(accepted)):
        raise ValueError('full original independent acceptance required')
    return {'registration_sha256': expected_registration,
            'acceptance_sha256': projection['original_acceptance_sha256'],
            'audit_execution_sha256': supervisor_hash,
            'scores_sha256': accepted['scores_sha256'], 'outcomes_opened': False,
            'custody_bound_acceptance_projection':projection}


def runtime_gate(expected=DEPENDENCIES):
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError('exact B runtime required; do not substitute local A/C runtime')
    return actual


def extracted_core(source, expected, guard, guard_expected):
    """Copy exact AST-delimited bodies and constants; never execute a worker."""
    source = Path(source)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('frozen original auditor bytes differ')
    text = raw.decode('utf-8'); tree = ast.parse(text)
    nodes = {}
    for n in tree.body:
        names = [n.name] if isinstance(n, ast.FunctionDef) else [
            t.id for t in n.targets if isinstance(t, ast.Name)] if isinstance(n, ast.Assign) else []
        for name in names:
            if name in FUNCTIONS + GLOBALS:
                if name in nodes:
                    raise ValueError('ambiguous numerical source definition')
                nodes[name] = n
    if set(nodes) != set(FUNCTIONS + GLOBALS):
        raise ValueError('frozen numerical closure incomplete')
    segments = {name: ast.get_source_segment(text, nodes[name]) for name in GLOBALS + FUNCTIONS}
    guard = Path(guard)
    guard_raw = guard.read_bytes()
    if hashlib.sha256(guard_raw).hexdigest() != guard_expected:
        raise ValueError('frozen original utility bytes differ')
    guard_text = guard_raw.decode('utf-8'); guard_tree = ast.parse(guard_text)
    helper_names = ('read', 'sha', 'journal_count')
    helper_nodes = {n.name: n for n in guard_tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in helper_names}
    if set(helper_nodes) != set(helper_names):
        raise ValueError('frozen utility closure incomplete')
    helper_segments = {name: ast.get_source_segment(guard_text, helper_nodes[name]) for name in helper_names}
    header = '''"""Derived exact-body numerical core; original full custody audit is separate.

No fitting, response generation, Slurm execution or historical freeze proof.
This file does not contain the original audit entry point or source guards.
Those guards still run in the mandatory unchanged original upstream audit.
"""
import hashlib
import json
import math
from pathlib import Path

'''
    segments = {**helper_segments, **segments}; nodes.update(helper_nodes)
    result = header + '\n\n'.join(segments.values()) + '\n'
    ast.parse(result)
    # Record both exact text and AST digests. This is a transparent extraction,
    # not a claim that its whole-file digest equals the original worker digest.
    return result, {name: {'source_segment_sha256': hashlib.sha256(s.encode()).hexdigest(),
                          'ast_sha256': hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()}
                    for name, s in segments.items()}


def prepare(registration, inventory, source_root, out):
    registration, inventory, source_root, out = map(Path, (registration, inventory, source_root, out))
    if sha(registration) != REGISTRATION_SHA:
        raise ValueError('not the frozen B registration')
    reg = read(registration); inv = read(inventory)
    if (reg['source_revision'] != REVISION or reg['dependencies'] != DEPENDENCIES or
            reg['matrix_fits'] != 640 or reg['primary_cells'] != 240 or
            inv['protocol_hashes']['B'] != REGISTRATION_SHA):
        raise ValueError('frozen registration/provenance differs')
    b_files = [f for f in inv['files'] if f['stage'] == 'B']
    expected = {'B/scripts/research/'+n: h for n, h in reg['worker_hashes'].items()}
    expected.update({'B/'+n: h for n, h in reg['generator_hashes'].items()})
    if len(b_files) != 20 or {f['path'] for f in b_files} != set(expected):
        raise ValueError('exact frozen worker/generator closure required')
    for f in b_files:
        if (f['source_revision'] != REVISION or f['sha256'] != expected[f['path']] or
                f['transformation'] != 'identity' or sha(source_root/f['path']) != f['sha256']):
            raise ValueError('frozen worker provenance/source differs')
    original = source_root/'B/scripts/research/audit_delivery_prospective_results.py'
    guard = source_root/'B/scripts/research/runner_delivery_confirmation.py'
    code, segments = extracted_core(original, reg['worker_hashes'][original.name],
                                   guard, reg['worker_hashes'][guard.name])
    from verify_delivery_release import identifying_bytes
    core_hash = hashlib.sha256(code.encode()).hexdigest()
    # Private output, exclusive and deliberately distinct from the live study.
    out.mkdir(parents=True, exist_ok=False)
    with (out/'delivery_prospective_replay_core.py').open('x') as f: f.write(code)
    if identifying_bytes(out/'delivery_prospective_replay_core.py'):
        raise ValueError('derived numerical core still contains screened identifiers')
    contract = {'status': 'metadata/source preparation only; B acceptance and runtime replay pending',
        'registration_sha256': REGISTRATION_SHA, 'original_revision': REVISION,
        'original_auditor_sha256': expected['B/scripts/research/'+original.name],
        'original_utility_sha256': reg['worker_hashes'][guard.name],
        'frozen_source_inventory_sha256': sha(inventory), 'frozen_source_bindings_checked': 20,
        'preparation_and_gate_tool_sha256':sha(__file__),
        'derived_core_sha256': core_hash, 'exact_extracted_segments': segments,
        'target_dependencies': DEPENDENCIES, 'matrix_fits': 640, 'primary_cells': 240,
        'primary_init': 0, 'new_fits': 0, 'new_responses': 0, 'B_outcomes_opened': False,
        'source_guards_bypassed': False, 'whole_worker_relocated': False,
        'original_and_derived_source_hashes_distinct': True,
        'required_runtime_inputs': ['complete conflict-rejecting composite inventory',
            'original full independent acceptance and successful bounded audit supervisor',
            'all original 640 fit receipts/models and 40 world attempt journals',
            '80 training and 40 heldout response receipts with exact charge ledgers',
            'scores/prediction arrays, fit seal, evaluation barrier and full phase telemetry',
            '19 original learner modules, retained descriptors and unchanged safe helpers'],
        'next_gate': 'assemble derived B artifacts only after unchanged original acceptance; replay all checkpoints and registered statistics in exact target runtime',
        'not_established': ['B scientific acceptance', 'target-runtime qualification',
            'full anonymous worker relocation', 'historical freeze authentication',
            'redistribution/anonymity approval', 'full sprint accounting']}
    with (out/'contract.json').open('x') as f: json.dump(contract,f,indent=2); f.write('\n')
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'inventory', 'source-root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.registration,args.inventory,args.source_root,args.out),indent=2))
