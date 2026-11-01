"""Assemble the November 1 anonymous writing/report packet, without scientific execution.

This is a scoped submission successor, not a new full reproduction manifest.
The separately attributed October 24 archive and numerical replay remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import zipfile

REPO = Path('/Users/pat/code/ACE')
BASE = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
CAPTURE = BASE / 'delivery-publication-source-snapshot-20261101-01'
RENDER = BASE / 'delivery-paper-submission-layout-20261101-04'
OLD = BASE / 'ACE-submission-review-20261024-02'
OUT = BASE / 'ACE-TMLR-submission-review-20261101-01'
CAPTURE_PIN = 'c910cc5371640566a9f67e87d43fdd69c983255b66d7b80e1de9acf1294bd1c6'
OLD_MEMBERSHIP_PIN = '148357c90b4b789ce6e76373b202dc590b100e6ea859a93f14643e2eb434795b'
OLD_ARCHIVE_PIN = 'eeeca48f7621d9196a843cd5fb3b66e6397040d4c1b3bd85ac90ef9e590f9e9a'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def pinned(path, pin):
    raw = path.read_bytes()
    if digest(raw) != pin:
        raise ValueError('independent pin mismatch: ' + str(path))
    return json.loads(raw)

def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()

def main():
    inv = pinned(CAPTURE / 'inventory.json', CAPTURE_PIN)
    old = pinned(OLD / 'source_membership.json', OLD_MEMBERSHIP_PIN)
    entries = {e['path']: e for e in inv['files']}
    payload = {}
    for item in old['files']:
        name = item['path']
        prefix = 'paper/aistats_ace_2027/'
        if not name.startswith(prefix) or '..' in Path(name).parts:
            raise ValueError('unexpected writing path')
        raw = (CAPTURE / 'source' / name).read_bytes()
        if digest(raw) != entries[name]['sha256'] or len(raw) != entries[name]['bytes']:
            raise ValueError('captured writing changed')
        if raw != (REPO / name).read_bytes():
            raise ValueError('current writing differs from frozen capture')
        payload[name[len(prefix):]] = raw
    if len(payload) != 29:
        raise ValueError('unexpected source/report membership')
    compile_receipt = json.loads((RENDER / 'compile.json').read_text())
    if compile_receipt['exit_code'] != 0 or compile_receipt['source_sha256'] != digest(payload['paper.tex']):
        raise ValueError('PDF export source mismatch')
    if (RENDER / 'paper.tex').read_bytes() != payload['paper.tex']:
        raise ValueError('export input differs from captured manuscript')
    payload['README_REVIEW.md'] = b'''# Anonymous manuscript sources and descriptive reports

This supplement supports the accompanying TMLR manuscript. It contains the
LaTeX source, bibliography, unmodified style files and notices, generated tables,
claim indexes and complete prospective descriptive reports. Compile paper.tex
with a standard LaTeX distribution supporting the included TMLR style.

The CSV files retain all 640 scored cells, 320 fixed-init0 arm/history errors,
80 system log ratios, history summaries and floor counts. Numerical results
are unchanged from the accepted study and separately verified reporting.
SOURCE_MANIFEST.json lists byte hashes of this supplement's payload; these
hashes check internal integrity, not independent authentication of its origin.

This is not the executable reproduction archive: checkpoints, original response
ledgers, worker/runtime closure and private execution/attempt records are not
included. The complete archive is preserved separately with its mandatory named
copyright attribution. It is withheld from the anonymous upload, not relabeled
as anonymous or silently reduced. Later distribution requires an appropriate
nonanonymous or venue-approved route. Required notices in these writing/style
files are preserved. No new fit, response or statistical test was performed to
prepare this supplement.
'''
    public_membership = {n: {'sha256': digest(raw), 'bytes': len(raw)} for n, raw in payload.items()}
    payload['SOURCE_MANIFEST.json'] = encode({'scope': 'writing and descriptive reports only', 'files': public_membership})
    OUT.mkdir(mode=0o700)
    (OUT / 'ACE_manuscript.pdf').write_bytes((RENDER / 'paper.pdf').read_bytes())
    with zipfile.ZipFile(OUT / 'ACE_anonymous_supplement.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    if (OUT / 'ACE_anonymous_supplement.zip').stat().st_size > 100_000_000:
        raise ValueError('supplement exceeds conservative venue size limit')
    with zipfile.ZipFile(OUT / 'ACE_anonymous_supplement.zip') as z:
        if len(z.namelist()) != len(payload) or set(z.namelist()) != set(payload):
            raise ValueError('archive membership mismatch')
        for n, raw in payload.items():
            if z.read(n) != raw:
                raise ValueError('archive readback mismatch')
    author_readme = '''# TMLR packet for final human review

Only ACE_manuscript.pdf and the optional ACE_anonymous_supplement.zip are intended
for anonymous submission. The supplement has 31 members: 29 writing/style/report
files, its scope README and a payload manifest. It is not the full executable archive.

The manuscript now leads with the frozen prospective comparison, gives the absolute
error scale and practical lesson, cites two direct mechanism-modeling precedents,
and accurately describes artifact availability. All scientific outputs, theory,
negative findings, accepted replays and failed attempts remain unchanged.

The complete October24 reproduction archive is preserved at:
/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-submission-review-20261024-02/ACE_reproduction_with_attribution.zip
It is 361,670,105 bytes, contains mandatory named attribution, and must not be
uploaded as the anonymous supplement. Its embedded writing is the October24
version. The new writing is authenticated by this packet's separate submission
manifest, not by relabeling the old full-archive manifest. The underlying reports
and experimental evidence are unchanged. No attribution was removed.

Read the independent scientific and literature recommendations in the repository's
November1 review reports. Actual packet/page checks and their disposition are
recorded separately in the November1 final preparation receipt.

Before submission, confirm author names/order/consent, active OpenReview profiles
and conflicts, funding/competing interests and human-subject applicability, each
author's current submission quota, and absence of overlapping archival submission.
Human authors must verify and take responsibility for the AI-assisted research.
The manuscript already discloses assistance in planning, code, analysis and drafting.
Approve this final paper and the scoped supplement route before any upload.

Nothing has been submitted or published. Both research automations remain paused.
This packet's checks are internal preparation, not a venue acceptance guarantee.
'''
    (OUT / 'README_AUTHOR.md').write_text(author_readme)
    manifest = {
        'schema': 'ace-tmlr-submission-successor-v1',
        'scope': 'new manuscript and anonymous source/report upload packet; not full executable reproduction',
        'writing_revision': inv['revision'], 'writing_inventory_sha256': CAPTURE_PIN,
        'source_membership_predecessor_sha256': OLD_MEMBERSHIP_PIN,
        'full_attributed_archive_predecessor_sha256': OLD_ARCHIVE_PIN,
        'full_archive_not_in_anonymous_upload': True,
        'replay_input_manifest_sha256': 'b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9',
        'replay_acceptance_sha256': '57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232',
        'report_acceptance_sha256': '86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10',
        'prospective_bundle_sha256': '2cca8a3699bcee68adef1a7fe610df87a6518a39513f71b310e990d73183cd84',
        'export_receipt_sha256': digest((RENDER / 'compile.json').read_bytes()),
        'builder_sha256': digest(Path(__file__).read_bytes()),
        'artifacts': {n: {'sha256': digest((OUT / n).read_bytes()), 'bytes': (OUT / n).stat().st_size}
                      for n in ('ACE_manuscript.pdf', 'ACE_anonymous_supplement.zip', 'README_AUTHOR.md')},
        'supplement_members': len(payload),
        'new_scientific_fits': 0, 'new_simulator_responses': 0,
        'numerical_requalification_performed': False,
        'human_author_details_confirmed': False, 'human_submission_approved': False,
        'venue_accepted': False,
    }
    (OUT / 'submission_manifest.json').write_bytes(encode(manifest))
    print(json.dumps({'destination': str(OUT), 'manifest_sha256': digest(encode(manifest)),
                      'artifacts': manifest['artifacts'], 'supplement_members': len(payload)}, indent=2))

if __name__ == '__main__':
    main()
