"""Seal completed review evidence; do not rerun any arithmetic or display checks."""
from pathlib import Path
import datetime
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parent
LOC = json.loads((ROOT / 'locations.json').read_bytes())
MIRROR = Path(LOC['external'])
REPORT = Path('/Users/pat/code/ACE/docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def pin(path):
    raw = path.read_bytes()
    return {'sha256': digest(raw), 'bytes': len(raw)}

def write(name, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with (ROOT / name).open('xb') as stream:
        stream.write(raw)
    return pin(ROOT / name)

utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
old = (ROOT / 'results-draft.md').read_bytes()
new = REPORT.read_bytes()
assert digest(old) == '0747797fdaea76a2471be7233af12a37476a1afc4a21cfca125040aed2ba0b04'
assert digest(new) == '2b55a679e1ea0eab7dd73a79f1209871a28ba8a70f865998e1ba0d5a9b9c65ac'
replacements = [
    ('The preference-trained LLM required 15.167 units', 'The preference-trained LLM had a mean of 15.167 units'),
    ('former 85 GB local download requirement', 'former 85 GiB Mac free-space planning target'),
]
expected = old.decode()
for before, after in replacements:
    assert expected.count(before) == 1
    expected = expected.replace(before, after)
assert expected.encode() == new
records = {}
for name in ['cells-checks.json', 'aggregate-checks.json', 'presentation-checks.json']:
    saved = json.loads((ROOT / name).read_bytes())
    assert saved['errors'] == []
    records[name] = {**pin(ROOT / name), **{k: saved[k] for k in ['checks', 'source_sha256', 'process_cpu_seconds', 'wall_seconds', 'peak_rss_bytes']}}
presentation = json.loads((ROOT / 'presentation-checks.json').read_bytes())
for path in presentation['canvas_paths']:
    assert pin(Path(path))['sha256'] == presentation['canvas_sha256']
snapshot = write('results-reviewed-final.md', new)
delta = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True), fromfile='results-draft.md', tofile='results-reviewed-final.md'))
delta_pin = write('wording-delta.diff', delta.encode())
wording_pin = write('wording-delta.json', {
    'scope': 'two final wording changes and unchanged canvas source pins only',
    'closed_utc': utc, 'previous_report_sha256': digest(old), 'reviewed_report': snapshot,
    'changes': [{'before': a, 'after': b, 'resolved': True} for a, b in replacements],
    'only_these_two_replacements': True, 'canvas_unchanged_sha256': presentation['canvas_sha256'],
    'arithmetic_or_display_checks_rerun': False, 'diff': delta_pin,
})
review = '''# Independent saved-report numerical and representation review

Required findings: **0**. The completed saved-output arithmetic, extraction and static canvas checks have no errors. Both final prose corrections are closed. This review does not replace the independently accepted execution or custody records.

The exact reviewed report is retained as `results-reviewed-final.md`, SHA256 `2b55a679e1ea0eab7dd73a79f1209871a28ba8a70f865998e1ba0d5a9b9c65ac`. The managed and repository canvas bytes match SHA256 `d9e463a0dd0c698fbe8becce52eaf877c04b70690745db0de0f4cc661b9ed93f`. Subsequent completion-status text and evidence links are outside these reviewed bytes and should be identified as a nonnumerical publication delta.

The review authenticated all 242 saved report members in both retained copies, bound to accepted finalizer `8ae40bd7952694271be0890590972250a5d8777f6856261b983b330fd999cb97` and summary `3f17abf3428411c44095dadcc3d80515267734d3729a0dbaef0989746dd7be0a`. Retrieval anchors are handoff `f95d744d29ad1fa4200b8b654e087cbcf5d7768013580b140c987fe6ebff8db9`, operation receipt `52f9cee616a67dec67a1ed6218b958dc4cf01399e1e8511edd4396a5f8c57e5b` and copy receipt `276213ae040aa5699b47699efe9c289aee23e49d0b60fd16f7ca713acee0e0c5`. Exact member pins remain in `cells-checks.json`.

The three retained checks performed 21,045 assertions, with zero errors: 9,988 cell/curve/custody checks, 9,895 aggregate/extraction checks and 1,162 presentation checks. Their original code, outputs, resource measurements and logs are preserved. Final closure only checked the two wording substitutions and unchanged canvas pins; it did not rerun those checks.

Coverage includes all 192 policy/world reports, 24 world summaries, 4,540 saved curve points, all 28 policy pairs, all nine predeclared strata and the overall group. Counts, ordered pair differences, wins/ties/losses and mean summaries were independently reconstructed for all groups and worlds. All six CSV exports and the complete display dataset agree with those reconstructions. Floating summaries use the documented numerical comparison tolerance; integer counts and vectors agree exactly. Initial cost, action charges, accepted-response counts, first crossing, capped nonattainment and final attainment were checked separately.

All eight policies remain represented. Mean capped total cost is 14.250 for uniform random, 12.458 for menu coverage, 11.958 for local information, 11.625 for target risk, 13.583 for the numerical ranker, 13.625 for the frozen LLM, 14.625 for SFT and 15.167 for DPO. From target risk's perspective against DPO there are nine lower-cost worlds, thirteen ties and two higher-cost worlds. All 192 cells are complete and evaluable. Ten worlds already attain the target at initialization; they contribute the common eight-unit initial cost. The saved reports record 6,097 logical cost units and 5,884 accepted responses.

The descriptive action claims agree: DPO selects `set_x_neg` for all 576 acquisition actions; the frozen LLM selects 574 observations and two `set_x_zero` actions; SFT selects `set_x_neg` for 481 of 523 acquisition actions. SFT stops in 23 worlds, pays 721 total logical cost units, and five stopped worlds remain above the final target. First crossing is not final attainment, and capped nonattainment is not successful attainment. The report's paired and regime claims preserve these distinctions. No significance or universal efficacy conclusion is established by these descriptive comparisons.

The static canvas review checks all ten groups, 192 cells, 4,540 point projections, 28 pair choices with both orientations, and all eight action rows including zeros. Defaults use all worlds and the first declared world. Paired signs, fixed cost/difference domains, selected-world labels and point-only error curves agree with the saved data. There is no pixel, browser-interaction or rendered-layout qualification here; the parent's TypeScript check was not repeated.

The combined prose review covers the previously corrected mechanism, eligibility, structural-noise, label and normalized-error explanations. The stated known-DAG, linear, known-noise, fixed-menu and constant-public-prose scope correctly leaves natural-language-objective conversion, instrument interpretation and proposing new intervention types as separate research questions. Selected accepted training/collection fields support 160 SFT updates, 119 DPO updates, 3,584 retained pairs, 9,145 collection responses and 10,240 collection cost units; these are distinct from development evaluation charges. The saved accounting/storage fields support the report's counts without treating collector CPU time as Slurm allocation or treating logical action costs as physical runtime.

The final two textual changes are recorded in `wording-delta.diff`: “required 15.167 units” becomes “had a mean of 15.167 units,” in the paragraph that explicitly identifies capped costs; the storage description becomes “former 85 GiB Mac free-space planning target.” Neither changes data or calculations. The original pending progress record and draft remain intact.

Limits: saved normalized errors are accepted evaluator outputs used as arithmetic inputs. No ground-truth world, SCM, utility, model, tokenizer or scientific function was reevaluated. The review checks saved prediction/error shapes and arithmetic but does not independently recreate errors from private ground truth. Logical per-arm reservations are not physical cache or resource measurements. No remote operation, scientific response generation, model load, old test repeat or source/packet review repeat was performed. There were no failed arithmetic attempts; two harmless read-path lookup errors remain documented in `lookup-notes.json`.
'''
review_pin = write('review.md', review.encode())
cells = json.loads((ROOT / 'cells-checks.json').read_bytes())
aggregates = json.loads((ROOT / 'aggregate-checks.json').read_bytes())
record_pin = write('review.json', {
    'schema': 'ace-saved-report-numerical-review-v1', 'status': 'complete', 'closed_utc': utc,
    'scope': 'saved_report_arithmetic_and_static_representation', 'required_findings': 0,
    'review': review_pin, 'reviewed_report_path': str(REPORT), 'reviewed_report': snapshot,
    'reviewed_canvas_sha256': presentation['canvas_sha256'], 'wording_closure': wording_pin,
    'retained_completed_checks': records, 'total_completed_assertions': 21045,
    'retrieval_pins': cells['input_pins'], 'finalizer_acceptance_sha256': cells['acceptance_sha256'],
    'summary_sha256': cells['summary_sha256'], 'extraction_source_sha256': aggregates['extraction_source_sha256'],
    'derivation_sha256': aggregates['derivation_sha256'], 'display_export_pins': aggregates['output_pins'],
    'coverage': {'report_members': 242, 'worlds': 24, 'cells': 192, 'curve_points': 4540, 'overall_pairs': 28, 'predeclared_strata': 9, 'all_and_stratum_pair_summaries': 280, 'world_pair_summaries': 672},
    'saved_report_arithmetic_verified': True, 'static_canvas_data_and_logic_verified': True,
    'pixel_or_interaction_review': False, 'ground_truth_errors_recomputed': False,
    'new_scientific_execution_or_acceptance': False, 'old_checks_rerun_for_closure': False,
    'remaining_required_findings': [], 'prior_pending_status_preserved': 'progress-status.json',
})
inventory = {p.name: pin(p) for p in sorted(ROOT.iterdir()) if p.is_file()}
inventory_pin = write('inventory.json', {'schema': 'ace-review-file-inventory-v1', 'files': inventory})
for path in sorted(ROOT.iterdir()):
    assert path.is_file() and not path.is_symlink()
    target = MIRROR / path.name
    if target.exists():
        assert not target.is_symlink() and target.read_bytes() == path.read_bytes()
    else:
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
    assert pin(target) == pin(path)
seal = {
    'schema': 'ace-independent-review-mirror-seal-v1', 'sealed_utc': utc,
    'repo': str(ROOT), 'external': str(MIRROR), 'inventory': inventory_pin,
    'review': review_pin, 'review_record': record_pin, 'required_findings': 0,
    'payload_files': len(inventory), 'payload_bytes': sum(x['bytes'] for x in inventory.values()),
    'both_copies_verified_identical': True,
    'inventory_excludes': ['inventory.json', 'seal.json'], 'seal_is_not_self_hashed': True,
}
seal_pin = write('seal.json', seal)
with (MIRROR / 'seal.json').open('xb') as stream:
    stream.write((ROOT / 'seal.json').read_bytes())
assert pin(MIRROR / 'seal.json') == seal_pin
print(json.dumps({'review': review_pin, 'review_record': record_pin, 'seal': seal_pin, 'inventory': inventory_pin, 'repo': str(ROOT), 'external': str(MIRROR), 'files_per_copy': len(inventory) + 2, 'bytes_per_copy': seal['payload_bytes'] + inventory_pin['bytes'] + seal_pin['bytes'], 'required_findings': 0}, indent=2))
