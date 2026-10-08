"""Check displayed A/C table data against independently pinned saved receipts.

Standard-library, read-only analysis of four existing tables. No checkpoint
inference, refitting, B artifact access, caption/prose review or public release
qualification. This new verifier is not covered by an older package manifest.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HISTORIES = {27424209, 1726880744, 735595885, 983656467, 124753321,
             921441405, 934168586, 546725957, 520888668, 1091699608,
             585254818, 884825602}
CONDITIONS = {f'{color}_{level}' for color in ('blue', 'green', 'red')
              for level in (64, 128, 255)} | {'white_128', 'white_255'}
TABLES = ('delivery_attribution_table.tex', 'delivery_history_table.tex',
          'delivery_physical_table.tex')
HEADERS = {
    TABLES[0]:[['Data','Learner','Epochs','NMSE/online','Snap/online','CPU h']],
    TABLES[1]:[['History','Online NMSE','SCM NMSE','Flat NMSE','SCM/admitted','SCM snap']],
    TABLES[2]:[['Condition','Delivery','Rolling','Physics','Fourier'],
              ['Condition','Delivery/Rolling','Delivery/Physics','Delivery/Fourier']]}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink in artifact location')
    with path.open('rb') as stream:
        raw = stream.read(32*1024*1024+1)
    if len(raw) > 32*1024*1024:
        raise ValueError('artifact exceeds bounded metadata size')
    return raw


def decode(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def numeric(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative saved metric required')
    return value


def tables(raw, expected_headers=None):
    """Accept only the simple generated data-row grammar, never execute TeX."""
    text = raw.decode('utf-8')
    environments = re.findall(r'\\(?:begin|end)\{([^{}]+)\}', text)
    if any(e not in ('table','tabular') for e in environments):
        raise ValueError('unsupported table environment')
    blocks = re.findall(r'\\begin\{tabular\}\{[^{}]+\}(.*?)\\end\{tabular\}', text, re.S)
    if (text.count(r'\begin{tabular}') != len(blocks) or
            text.count(r'\end{tabular}') != len(blocks)):
        raise ValueError('unexamined tabular environment')
    if expected_headers is not None and len(blocks) != len(expected_headers):
        raise ValueError('table/header count changed')
    result = []
    for index,block in enumerate(blocks):
        if (block.count(r'\toprule') != 1 or block.count(r'\midrule') != 1
                or block.count(r'\bottomrule') != 1):
            raise ValueError('table data boundaries changed')
        pre, data = block.split(r'\midrule')
        data, remainder = data.split(r'\bottomrule')
        if remainder.strip():
            raise ValueError('unexamined content after bottom rule')
        before, header = pre.split(r'\toprule')
        if before.strip() or not header.strip().endswith('\\\\'):
            raise ValueError('unsupported table header')
        header = [c.strip() for c in header.strip()[:-2].split('&')]
        if expected_headers is not None and header != expected_headers[index]:
            raise ValueError('column headers or units changed')
        rows = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.endswith('\\\\'):
                raise ValueError('unsupported data row')
            rows.append([c.strip() for c in line[:-2].split('&')])
        result.append(rows)
    return result


def check_data(table_bytes, summary, gate, physical):
    """Caller authenticates receipt bytes and original acceptance first."""
    configs = summary['configurations']
    labels = {f'{dataset}-{arm}-{epochs}-i{init}' for dataset in
              ('final_buffer', 'online_admitted', 'all_paid') for arm in ('scm', 'flat')
              for epochs in (100, 30000) for init in (0, 1, 2)}
    labels |= {f'all_paid-{arm}-100-i0' for arm in ('ridge', 'polynomial', 'tree')}
    labels.add('all_paid-flat-matched-cpu-i0')
    if set(configs) != labels:
        raise ValueError('full forty-configuration summary required')
    expected = []
    for label, saved in configs.items():
        if not label.endswith('-i0'):
            continue
        dataset = next(d for d in ('final_buffer', 'online_admitted', 'all_paid')
                       if label.startswith(d+'-'))
        parts = label[len(dataset)+1:].split('-')
        arm = parts[0]
        if type(saved['n_histories']) is not int or saved['n_histories'] != 12:
            raise ValueError('configuration must retain twelve histories')
        budget = ('--' if arm not in ('scm', 'flat') else
                  'CPU' if parts[1] == 'matched' else f'{int(parts[1]):,}')
        cpu = numeric(saved['fit_cpu_core_hours'])
        expected.append([{'final_buffer':'Final50', 'online_admitted':'Admitted',
                          'all_paid':'All paid'}[dataset], 'SCM' if arm == 'scm' else arm,
                         budget, f"{numeric(saved['geomean_nmse_ratio_online']):.3f}",
                         f"{numeric(saved['geomean_snapped_ratio_online']):.3f}",
                         f'{cpu:.3f}' if cpu >= .001 else '$<0.001$'])
    if tables(table_bytes[TABLES[0]], HEADERS[TABLES[0]]) != [expected]:
        raise ValueError('attribution displayed rows/budgets/ratios/CPU differ')
    config_scalars = sum(len(row)-3 for row in expected)
    epoch_fields = sum(row[2].replace(',','').isdecimal() for row in expected)
    histories = gate['histories']
    if (len(histories) != 12 or any(type(c['seed']) is not int for c in histories)
            or {c['seed'] for c in histories} != HISTORIES):
        raise ValueError('exact twelve-history membership required')
    expected = []
    for c in histories:
        denominator = numeric(c['unused_observations_ablation']['nmse'])
        if denominator == 0:
            raise ValueError('history ratio denominator must be positive')
        exact = numeric(c['delivery']['exact'])
        if exact > 1:
            raise ValueError('invalid exact-level fraction')
        expected.append([str(c['seed']), *[f"{numeric(c[a]['nmse']):.6f}" for a in
                         ('online', 'delivery', 'simpler')],
                         f"{numeric(c['delivery']['nmse'])/denominator:.3f}", f'{1-exact:.3f}'])
    if tables(table_bytes[TABLES[1]], HEADERS[TABLES[1]]) != [expected]:
        raise ValueError('history absolute errors/admitted ratios/snapping differ')
    conditions = physical['conditions']
    if set(conditions) != CONDITIONS:
        raise ValueError('all eleven physical conditions required, development excluded')
    absolute, conditional = [], []
    for condition, saved in conditions.items():
        row = [condition.replace('_', r'\_')]
        for arm in ('delivery', 'rolling_buffer', 'physics', 'fourier'):
            value = numeric(saved['nmse'][arm])
            if value >= .001:
                cell = f'{value:.3f}'
            else:
                mantissa, exponent = f'{value:.2e}'.split('e')
                cell = '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
            row.append(cell)
        absolute.append(row)
        row = [condition.replace('_', r'\_')]
        for arm in ('rolling_buffer', 'physics', 'fourier'):
            stats = saved['conditional_uncertainty'][arm]
            if type(stats['n_action_blocks']) is not int or stats['n_action_blocks'] != 8:
                raise ValueError('conditional eight-block estimand changed')
            lo, hi = [numeric(v) for v in stats['conditional_bootstrap_ci95']]
            if lo > hi:
                raise ValueError('interval endpoints reversed')
            row.append(f"{numeric(stats['block_weighted_ratio']):.2f} [{lo:.2f}, {hi:.2f}]")
        conditional.append(row)
    if tables(table_bytes[TABLES[2]], HEADERS[TABLES[2]]) != [absolute, conditional]:
        raise ValueError('physical row-weighted errors or conditional block summaries differ')
    metric_fields = (config_scalars+sum(len(r)-1 for r in expected)+
                     sum(len(r)-1 for r in absolute)+3*sum(len(r)-1 for r in conditional))
    return {'tables_checked':4, 'attribution_configuration_rows':16,
            'retained_history_rows':12, 'physical_rows_per_table':11,
            'metric_scalar_fields_checked':metric_fields,
            'numeric_epoch_fields_checked':epoch_fields,
            'numeric_display_fields_checked':metric_fields+epoch_fields,
            'scope':'A/C table data and row order only; not captions, prose, F macros, B claims or model inference'}


def audit(root, expected_manifest, table_dir):
    if not isinstance(expected_manifest, str) or not re.fullmatch('[0-9a-f]{64}', expected_manifest):
        raise ValueError('independent manifest SHA256 required')
    root, table_dir = Path(root), Path(table_dir)
    manifest_raw = capture(root/'manifest.json')
    if digest(manifest_raw) != expected_manifest:
        raise ValueError('manifest digest differs')
    manifest = decode(manifest_raw)
    files = {f['path']:f for f in manifest['files']}
    if len(files) != len(manifest['files']):
        raise ValueError('duplicate manifest path')
    input_names = ('A/gate.json','A/summary.json','A/complete.json',
                   'C/acceptance.json','C/scores.json')
    for name in input_names:
        entry = files[name]
        for field in ('sha256','original_sha256'):
            if not isinstance(entry[field], str) or not re.fullmatch('[0-9a-f]{64}',entry[field]):
                raise ValueError('declared original/projected digest required')
        if name != 'C/acceptance.json':
            if entry['transform'] != {'kind':'identity'} or entry['sha256'] != entry['original_sha256']:
                raise ValueError('identity A/C analysis artifact required')
        elif entry['transform']['kind'] not in ('identity','project-json'):
            raise ValueError('unsupported acceptance projection')
        elif (entry['transform']['kind'] == 'identity' and
              entry['sha256'] != entry['original_sha256']):
            raise ValueError('identity acceptance digests differ')
    input_hashes = {}
    def read(name):
        raw = capture(root/name)
        if digest(raw) != files[name]['sha256']:
            raise ValueError('receipt snapshot digest differs: '+name)
        input_hashes[name] = digest(raw)
        return decode(raw)
    gate, summary, complete, accepted = [read(n) for n in
        ('A/gate.json', 'A/summary.json', 'A/complete.json', 'C/acceptance.json')]
    if (gate['custody_audit']['full_acceptance'] is not True or
            summary['custody']['full_acceptance'] is not True or
            type(complete['n_fits']) is not int or complete['n_fits'] != 480 or
            type(complete['n_histories']) is not int or complete['n_histories'] != 12 or
            accepted['full_acceptance'] is not True or len(accepted['conditions']) != 11 or
            set(accepted['conditions']) != CONDITIONS):
        raise ValueError('original accepted A/C metadata required')
    if (gate['complete_receipt_sha256'] != files['A/complete.json']['original_sha256'] or
            summary['complete_sha256'] != files['A/complete.json']['original_sha256'] or
            summary['gate_sha256'] != files['A/gate.json']['original_sha256'] or
            accepted['selected_gate_sha256'] != files['A/gate.json']['original_sha256'] or
            accepted['receipt_hashes']['scores.json'] != files['C/scores.json']['original_sha256']):
        raise ValueError('acceptance-to-input original digest binding differs')
    physical = read('C/scores.json')
    snapshots = {name:capture(table_dir/name) for name in TABLES}
    result = check_data(snapshots, summary, gate, physical)
    result.update({'manifest_sha256':expected_manifest,'input_hashes':input_hashes,
                   'table_hashes':{n:digest(raw) for n,raw in snapshots.items()},
                   'verifier_sha256':digest(Path(__file__).read_bytes()),
                   'new_fits':0,'new_optimizer_updates':0,'new_responses':0,'model_loads':0,
                   'full_package_integrity_verified_here':False,'public_release_ready':False})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--table-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.expected_manifest_sha256, args.table_dir), indent=2))
