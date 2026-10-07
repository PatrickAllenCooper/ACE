"""Bundle companion LaTeX files into the existing editor document.

The desktop compiler accepts one source file. Authoritative style, claims,
tables, proofs and bibliography remain separate tracked files. Re-run with
--write after regenerating claims or editing those files; --check detects drift.
This changes manuscript packaging only, never experimental protocols.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'paper/aistats_ace_2027'
BEGIN = '% BEGIN EDITOR COMPANION BUNDLE (generated; edit companion files)'
END = '% END EDITOR COMPANION BUNDLE'
FILES = ('tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'delivery_references.bib',
         'delivery_claims.tex', 'delivery_attribution_table.tex',
         'delivery_history_table.tex', 'delivery_physical_table.tex', 'delivery_theory.tex')


def bundled():
    provenance = json.loads((ROOT / 'tmlr_style_provenance.json').read_text())
    chunks = [BEGIN]
    for name in FILES:
        raw = (ROOT / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if name in provenance['files'] and digest != provenance['files'][name]['sha256']:
            raise ValueError('official TMLR file changed: ' + name)
        contents = raw.decode()
        if '\\end{filecontents*}' in contents:
            raise ValueError('nested filecontents terminator: ' + name)
        chunks += [f'% {name}: SHA256 {digest}',
                   f'\\begin{{filecontents*}}[overwrite]{{{name}}}',
                   contents.rstrip('\n'), '\\end{filecontents*}']
    chunks.append(END)
    return '\n'.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'paper.tex'
    text = path.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('exactly one companion bundle is required')
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    refreshed = text[:start] + bundled() + text[stop:]
    if args.write:
        path.write_text(refreshed)
    elif text != refreshed:
        raise ValueError('manuscript companions changed; run --write')
    print(f'Manuscript companion bundle matches all {len(FILES)} files; official style hashes verified.')


if __name__ == '__main__':
    main()
