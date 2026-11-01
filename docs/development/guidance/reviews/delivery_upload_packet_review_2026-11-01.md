**Independent final TMLR upload packet review — November 1, 2026**

**Recommendation: ACCEPT for technical upload readiness of the exact manuscript PDF and scoped source/report ZIP identified below. Required technical package fixes: none.** This is an internal package-review recommendation, not scientific requalification, submission authorization, or venue acceptance. Human author metadata, author order and consent, final manuscript approval, and approval of the scoped supplement route remain explicitly pending. The parent reviewer is inspecting all rendered pages; this review does not certify layout or replace that work.

The exclusive reviewed packet is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-TMLR-submission-review-20261101-01`. Only `ACE_manuscript.pdf` and `ACE_anonymous_supplement.zip` are intended for anonymous upload. `README_AUTHOR.md` and `submission_manifest.json` are local handoff/provenance files, not additional anonymous submission attachments.

I read the actual packet and the complete builder, inspected PDF metadata, all-page extracted text for author identifiers, PDF link targets, ZIP filenames and decoded contents, and the retained lineage evidence. Verification used read-only byte comparisons, SHA256/Git-blob checks, ZIP reads, JSON inspection and CSV row counts. I did not run the builder, compile or render the manuscript, execute project verification/test suites, recompute scores or statistics, import models, acquire responses, allocate resources, or rerun the previous full-archive verification. Only this review file was written.

**1. Exact packet identity and membership pass.** Independently computed hashes and sizes match the user-supplied pins and all three artifact entries in the submission manifest:

- `submission_manifest.json`: 1,868 bytes; SHA256 `369cd061eede65896c54b79776aa75971d764a9bb949b5160f0153bfbf35fdd2`.
- `ACE_manuscript.pdf`: 180,642 bytes; SHA256 `79c56dde1f325e14cf4d29e5b1c52eb136eebaff870d14cafe8199f3d7f04368`.
- `ACE_anonymous_supplement.zip`: 176,883 bytes; SHA256 `e329982ac8b8b37e4d05ba5f3ff3475a446276e775c6424ec34782049a20b6e9`.
- `README_AUTHOR.md`: 2,060 bytes; SHA256 `098cd51452f77be2f07a26b8888c577839a1cf178ca304cc08c75f2791ce43dd`.

The outer directory contains exactly these four files. The ZIP contains exactly 31 unique regular-file members, totaling 842,096 uncompressed bytes. There are no absolute/traversal paths, backslash paths, symlink members, encrypted members, extra fields, member comments or archive comment. Every member was read successfully. Its `SOURCE_MANIFEST.json` covers exactly the other 30 files, with matching byte lengths and hashes; its own omission from its payload list avoids a self-hash claim. The separately supplied whole-ZIP pin authenticates this particular manifest along with the payload.

The exact ZIP member set is:

```text
README_REVIEW.md
SOURCE_MANIFEST.json
TMLR_STYLE_LICENSE
archive/tmlr_official_template.tex
claim_index.json
delivery_attribution_table.tex
delivery_claims.tex
delivery_evaluation_design.tex
delivery_history_table.tex
delivery_method_diagram.tex
delivery_physical_table.tex
delivery_prospective_absolute_table.tex
delivery_prospective_all_cells.csv
delivery_prospective_claim_index.json
delivery_prospective_claims.tex
delivery_prospective_descriptive_table.tex
delivery_prospective_fixed_init0.csv
delivery_prospective_floor_counts.csv
delivery_prospective_history_specific.csv
delivery_prospective_output_index.json
delivery_prospective_primary_table.tex
delivery_prospective_sensitivity_table.tex
delivery_prospective_supplement.json
delivery_prospective_system_log_ratios.csv
delivery_references.bib
delivery_theory.tex
fancyhdr.sty
paper.tex
tmlr.bst
tmlr.sty
tmlr_style_provenance.json
```

**2. Writing, export and independent-review lineage pass.** The builder at `/Users/pat/code/ACE/results/delivery_submission_audit_20261101/prepare_upload_packet.py` hashes to `42913cdf5da041ab63d99994177f3da4872e3392aac1c94d6e2388ebcdf73a58`, matching the submission manifest. Static inspection confirms that it copies writing/report bytes, adds the scope README and payload manifest, copies the saved PDF, and writes the ZIP and handoff metadata. It contains no scientific execution stage. Its exclusive output-directory creation and ZIP creation mode avoid overwriting an existing packet.

The capture at `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-publication-source-snapshot-20261101-01/inventory.json` hashes to `c910cc5371640566a9f67e87d43fdd69c983255b66d7b80e1de9acf1294bd1c6`. Its source tree has exactly the declared 42 files. All 42 lengths, SHA256 values and Git blob IDs match the captured bytes, current repository files and writing commit `a384ff2d6befc0c48c5dcccbbad0554613c97857`. This is a writing/evidence-metadata capture, not a new qualification of empirical execution.

The 29 original ZIP payload paths exactly equal the paper-relative paths in the October 24 `source_membership.json`, SHA256 `148357c90b4b789ce6e76373b202dc590b100e6ea859a93f14643e2eb434795b`; the only added members are the two new review/integrity files. All 29 payloads match the current capture. Comparing both the 29-member predecessor membership and the 42-file October 24 writing inventory, SHA256 `d17693410e124c67286a2b55914b4a404dbd26b6a35d24e6c083752560478238`, identifies changes only to `paper.tex` and `delivery_references.bib`. Theory, generated numerical tables, claim files, reports and captured replay metadata remain byte-identical to their predecessor pins. All 16 embedded `filecontents*` companions in the final manuscript also match the separate ZIP members exactly, including the bibliography and styles.

The final manuscript source SHA256 is `3cbdf644c8d48dda7b05b35b8c2d29c17dffa8988ca431fe53383b9f6d3c833a`. The saved export input under `delivery-paper-submission-layout-20261101-04` is byte-identical to that source; the packet PDF is byte-identical to that directory's `paper.pdf`. Its `compile.json`, SHA256 `0b1c71c9b3525bcfda96c309a674a7694344b1b998be1645cedcc2f0df3a07e6`, records exit zero and the same source hash. I inspected this existing receipt and did not repeat compilation.

I also hashed the saved `delivery-paper-submission-layout-20261101-03/paper.tex` to the independent scientific/literature review pin `86c1e4c3f003e03a9121e14e2fec8a53f9632d3bc53a1519b1a55695bc9fe481`. A direct diff against the final source confirms exactly two comment replacements: the opening manuscript-status comment and the trailing comment after `\deliverydrafttrue`. No TeX command, active prose, bibliography or numerical content changes in that delta. This independently corroborates `results/delivery_submission_audit_20261101/source_comment_delta.json`, SHA256 `58e64e7211c3c12f1bc1ea04851c1bdeb604779a04f5636465c4b8413093ba1b`.

The November 1 scientific review retains ACCEPT with no required scientific fixes, and the literature review closes its required positioning finding. Their reviewed source therefore has a valid comment-only bridge to this packet. The review files read here have SHA256 `6494e9eb274a25d4d504ddfa8dd4baaf2b780867e580d85c0bc8d00f33d89048` and `c945174133bd5a70f80d78810cd934a301f015c2f7e5a822a6301c874b48e4b5`, respectively. Those recommendations did not determine my package verdict.

**3. Report lineage and unchanged empirical reporting pass.** All 13 packaged prospective reporting outputs match the original accepted report witness, `/Users/pat/code/ACE/results/delivery_prospective_reporting_20261024/report02_review/review_witness.json`, SHA256 `6e00103b6a5015a7f52a4884fe5cf51fef16af00100f6e016cf72ccf110cfb64`. Filename relocation preserves the original bytes. Both original report indexes are present, their output hashes resolve to the packaged files, and their hashes match the report acceptance receipt. CSV inspection confirms 640 all-cell rows, 320 fixed-initialization-zero rows, 80 system log-ratio rows, eight history-summary rows and four floor-count rows, excluding headers: 1,052 rows total. I counted saved records without recomputing their scientific values.

The following lineage pins were checked against the retained files and their relevant joins:

- Prospective bundle: `2cca8a3699bcee68adef1a7fe610df87a6518a39513f71b310e990d73183cd84`.
- Frozen candidate03 replay input manifest: `b30e7b5f0f26f1e7f803c47e5cb1cbd22bbcb24ddac999e1f7e07cfc10c454a9`.
- Private replay acceptance, `results/delivery_diagnostic_replay_20261024/terminal12/replay_acceptance.json`: `57da98e2bbce82faad642442daa491b2cf3722df7524bfd4c881a188a36e3232`.
- Private report acceptance, `results/delivery_prospective_reporting_20261024/terminal02/report_acceptance.json`: `86abaf25f19e62ed5088311288b4ad8c6ea854863ce683488f7b333cb5102c10`.
- Primary report index: `fdd5a3ee01708efbaaee7202d03c907cb6aede06dc1752ded7df5278d87f8deb`.
- Descriptive output index: `022e78c2a7e862066265f16eb490d2b0df316d11884b3aa7f8dfac22b2ae9ada`.
- Complete descriptive supplement JSON: `58b8530c8b10f96009266aa838d2ce49b306ce29232efe5aa3e9d29405a1bdaa`.

The primary index preserves the historical original-input hashes; the descriptive index preserves projected-input provenance and the original registration hash. The differing original/projected scores and acceptance digests are intentional lineage distinctions, not unexplained substitutions. Report acceptance binds the primary index's historical inputs and the accepted replay. Saved-checkpoint qualification remains against candidate03, not this new submission manifest. Neither this review nor the new ZIP qualifies anonymous original collection/training, reconstructs historical freeze timing, refits a model, or newly replays any model. Raw empirical payload integrity remains inherited from the prior accepted evidence; it was not re-audited here.

**4. Style notices and plausible anonymity pass.** The packet retains `TMLR_STYLE_LICENSE`, the original notice headers in `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty`, the archived official template and `tmlr_style_provenance.json`. Every style-provenance hash matches its packaged file. The TMLR adaptation credits, bibliography-style copyright/LPPL notice, and fancyhdr copyright/LPPL notice remain intact, including embedded manuscript copies. The provenance explicitly says the existing fancyhdr file differs from the pinned TMLR repository's version; this is not a claim that every bundled style file is identical to that upstream checkout. No notice removal is needed or recommended.

The actual PDF has 25 pages, title metadata “From Experimental Histories to Causal Surrogates,” author metadata “Anonymous authors,” generic LaTeX/hyperref and xdvipdfmx creator/producer fields, no custom metadata or metadata stream, no encryption, no JavaScript and no embedded files. Its first-page text identifies anonymous authors and double-blind review. All-page text and PDF URL-annotation inspection found no apparent submitting-author name, email, institution, personal repository or private filesystem path; external PDF links point to cited literature. The AI-assistance disclosure remains in first-page text.

The same screen covered all ZIP filenames and decoded member contents, including comments, provenance and manifests. Matches for names and contacts were attributable to third-party style maintainers, cited literature, or the unchanged official template's example authors. In particular, Patrick W. Daly in the bibliography style and Patrick Chao/Patrick Blöbaum in a cited paper do not identify Patrick Cooper. The archived template's example names and email addresses are clearly associated with the official formatting instructions, not the submitted manuscript. There is no author README, private path or identifying runner notice inside the ZIP. This is a plausible identifier screen, not a guarantee against inference from prose, public history or hash linkage.

**5. Upload size and supplement description pass.** TMLR's current [author guidelines](https://jmlr.org/tmlr/author-guide.html), consulted November 1, require anonymized submissions and supplementary materials, a manuscript PDF using the TMLR style, and PDF/ZIP supplementary material up to 100 MB. The active source uses the submission-mode TMLR style. The ZIP is 0.176883 decimal MB and passes even a conservative 100,000,000-byte interpretation; the PDF is 0.180642 MB. These measurements provide ample size margin without interpreting the venue's MB notation permissively.

The ZIP README and manuscript Appendix E accurately describe manuscript sources and complete descriptive reports. They expressly exclude executable reproduction, checkpoints, original response ledgers and private execution records. The retained claim indexes may point to evidence outside this supplement; that is consistent with the disclosed limited scope, not a promise of a self-contained replay package. The README correctly distinguishes internal payload integrity from independent authentication of origin.

The omitted full archive is `/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-submission-review-20261024-02/ACE_reproduction_with_attribution.zip`. A filesystem size check confirms 361,670,105 bytes (361.670105 decimal MB). Its preserved archive receipt, SHA256 `fb6be817980327a8a86041b6101c55637df8e3206e29c06a7c0db0405fa418e2`, records archive SHA256 `eeeca48f7621d9196a843cd5fb3b66e6397040d4c1b3bd85ac90ef9e590f9e9a` and 3,942 members. I did not reopen, rehash or rerun verification of that archive.

The existing successor source contract, SHA256 `0ffe94be8bc1dae770346bac6160d5aad134476bfa6fb6e57deafe6723423596`, and the closed October 24 package review establish its retained mandatory named attribution and explicitly nonanonymous status. The new README accurately warns that its embedded writing is the October 24 version. The current packet uses a new submission manifest rather than relabeling that archive or claiming its old source manifest authenticates the November 1 paper. Withholding the full archive from this anonymous upload preserves the notices and avoids both the size and identity problems. Owner authority is not reopened. Later distribution still needs an appropriate nonanonymous or venue-approved route.

**6. Optional clarification only.** A future README revision could explicitly say that `archive/tmlr_official_template.tex` is retained upstream provenance and that its example authors are unrelated to this submission. Its name, formatting-instructions title, provenance and separate `paper.tex` compilation instruction already make that distinction adequate. This is not a required fix and does not justify rebuilding the pinned packet. The independent scientific review's optional anonymous design specification remains optional; I found no package defect that makes it mandatory.

**Human disposition remains pending.** The actual submission manifest records `human_author_details_confirmed: false`, `human_submission_approved: false`, `venue_accepted: false` and `numerical_requalification_performed: false`. Its author README calls for author names/order/consent, active profiles/conflicts and other submission declarations, human responsibility for AI-assisted work, and final approval of these concrete files and the supplement route. Those tasks, the parent's rendered-page disposition and any actual upload remain outside this technical ACCEPT. Nothing was submitted or published by this reviewer.
