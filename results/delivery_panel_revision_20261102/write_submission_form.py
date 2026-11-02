"""Prepare a private successor worksheet from the completed packet and confirmed author record."""
from pathlib import Path
import hashlib,json,re
r=Path('/Users/pat/code/ACE'); packet=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-TMLR-submission-review-20261102-01')
m=json.loads((packet/'submission_manifest.json').read_text())
s=(r/'paper/aistats_ace_2027/paper.tex').read_text();a=s.split('\\begin{abstract}',1)[1].split('\\end{abstract}',1)[0];a=' '.join(a.replace('\\%','%').split())
old=r/'results/delivery_submission_audit_20261101/author_metadata_20261101_04.json'
text=f'''# TMLR submission worksheet, November 2 revision

Private author worksheet. This is not an upload attachment or submission approval.

## Title

From Experimental Histories to Causal Surrogates: Data Reuse, Optimization, and Error Propagation

## Abstract

{a}

## Exact review packet

Use only the following two files after both authors approve this revision:

'''
for name in ['ACE_manuscript.pdf','ACE_anonymous_supplement.zip']:
 e=m['artifacts'][name];text+=f'- `{packet/name}`\n  SHA256 `{e["sha256"]}`, {e["bytes"]} bytes.\n'
text+='''
The small supplement is manuscript source and saved scientific reports, not the full executable archive. It preserves the original data and carries an explicitly identified publication projection of the historical claim index. The full archive retains required named attribution and remains withheld from anonymous upload. Keep README_AUTHOR.md, submission_manifest.json and this worksheet private. The old November 1 packet remains unchanged.

## Confirmed author details

1. Patrick Cooper, patrick.cooper@colorado.edu, University of Colorado Boulder.
2. Alvaro Velasquez, alvaro.velasquez@colorado.edu, University of Colorado Boulder.

Patrick has confirmed this order, that Alvaro is the only coauthor and final author, and that Alvaro has agreed to coauthorship. Patrick reports no funding, no competing interests and no overlapping version submitted elsewhere. These declarations are retained from the November 1 author record. They are not requests to repeat those confirmations.

## Remaining human checks

Both authors must verify and approve the final manuscript and disclosed AI-assisted work. Agreement to coauthorship does not establish approval of this exact revision. Finalize the confidential OpenReview profiles and reviewer-conflict records, applicable ethics and eligibility fields, and the chosen supplement-release route before the final submit action. The live authenticated submission form has not been opened or populated.

No paper, data or package has been publicly uploaded. Human submission approval remains pending. The model panel and focused revision reviewers provide internal feedback, not a TMLR decision.

## Provenance

'''
text+=f'Worksheet abstract copied from the current manuscript with whitespace and escaped percent signs normalized. Source SHA256 `{hashlib.sha256(s.encode()).hexdigest()}`. Submission manifest SHA256 `{hashlib.sha256((packet/"submission_manifest.json").read_bytes()).hexdigest()}`. Earlier author-state SHA256 `{hashlib.sha256(old.read_bytes()).hexdigest()}`. No author declaration was inferred from review scores or technical checks.\n'
p=r/'docs/development/guidance/delivery_submission_form_draft_2026-11-02.md'
with p.open('x') as f:f.write(text)
print(p)
