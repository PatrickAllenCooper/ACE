# Active Causal Experimentalism — current presentation: v8

Twelve slides with a white background, compressor teaching examples, and the title **Active Causal Experimentalism** with **Every experiment counts**.

- **Editable deck:** [ACE_mechanisms_and_evidence_v8.pptx](ACE_mechanisms_and_evidence_v8.pptx)
- **Viewing PDF:** [ACE_mechanisms_and_evidence_v8.pdf](ACE_mechanisms_and_evidence_v8.pdf), a raster viewing copy
- **Verbatim ten-minute script:** [ACE_10_minute_read_aloud_v8.md](ACE_10_minute_read_aloud_v8.md), 1,341 spoken words, approximately ten minutes at 134 words per minute. Twelve prose blocks follow slide order, without stage directions or asides.
- **Technical notes:** [ACE_mechanisms_speaker_notes_v8.md](ACE_mechanisms_speaker_notes_v8.md)
- **Title preview:** [ACE_title_card_v8.png](ACE_title_card_v8.png)

The cover uses a new white compressor concept with SCMs in the background. Slides2–3 explain a simplified compressor test rig: drive command X affects shaft speed M, which affects pressure rise Y. The same interpretation continues through the six-action worked example. Values are normalized deviations, not physical RPM or calibrated compressor measurements. Holding speed assumes an approved independent test-rig controller.

The proposed foundation-model branch, local/joint-table explanation, and separate joint-table figure on the closing slide remain. Historical acquisition comparisons and all native table values, chart XML, and workbooks are unchanged. Source and limitations remain in technical notes. No scientific computation or new experiment occurred.

Version8 imports and restyles version7. `edit_mechanisms_deck_v8.mjs` writes the draft, `restore_chart_sources_v8.py` retains chart sources, and `finalize_mechanisms_deck_v8.mjs` validates and exports the editable deck. Original and new cover assets plus the exact built-in image-generation prompt are retained. Previous versions below remain historical.

---

# ACE — current presentation: v7

Twelve slides. Two visual explanations have been added to v6, and its closing comparison has been expanded.

- **Editable deck:** [ACE_mechanisms_and_evidence_v7.pptx](ACE_mechanisms_and_evidence_v7.pptx)
- **Viewing PDF:** [ACE_mechanisms_and_evidence_v7.pdf](ACE_mechanisms_and_evidence_v7.pdf), a raster copy of final slide renders
- **Updated ten-minute script:** [ACE_10_minute_read_aloud_v7.md](ACE_10_minute_read_aloud_v7.md)
- **Technical speaker notes:** [ACE_mechanisms_speaker_notes_v7.md](ACE_mechanisms_speaker_notes_v7.md)

Slide4 visualizes local parent-to-child tables versus one joint input-to-output table. Slide6 shows the proposed foundation-model candidate branch connected to the SCM experiment loop, with the worked example's six legal clamp actions, predictions, observations and updates. Slide12 retains the random/ACE intervention bars and adds a separate SCM-free joint-table figure. The latter has different units and is an analytic illustration, not a third measured arm. The foundation-model acquisition extension remains proposed; measured curves use the historical neural SCM learners.

The old slide3 remains3; old4–9 become5,7,8,9,10,11. The old closing slide10 becomes12. The compressor cover and eight other retained slide bodies are unchanged, apart from page numbering. Native editable tables, diagrams and charts are retained. All three original chart XML parts and workbook bytes are preserved exactly. No scientific result or experiment has changed.

The v7 editing source imports v6, inserts the two new slides and recomposes the closing slide. It uses the existing bundled artifact runtime. The accompanying source-workbook restoration helper preserves the original chart data after import/export. Prior versions and their scripts remain historical below.

---

# ACE — current presentation: v6

The 10-slide marketing presentation opens with a new compressor concept rendering and uses larger directional SCM arrows. Slides 1–4 introduce the theory; slides 5–7 show exact inputs, scores, observations and updates; slides 8–10 show the archived graph and recorded comparisons.

- **Editable deck:** [ACE_mechanisms_and_evidence_v6.pptx](ACE_mechanisms_and_evidence_v6.pptx)
- **Viewing PDF:** [ACE_mechanisms_and_evidence_v6.pdf](ACE_mechanisms_and_evidence_v6.pdf) (raster copy of final deck renders)
- **10-minute read-aloud script:** [ACE_10_minute_read_aloud.md](ACE_10_minute_read_aloud.md)
- **Technical notes:** [ACE_mechanisms_speaker_notes.md](ACE_mechanisms_speaker_notes.md)
- **Compressor illustration:** [ACE_compressor_concept.png](ACE_compressor_concept.png) — newly generated conceptual illustration, not the experimental apparatus or recovered original image.

## Comparisons

The empirical comparison retains SCM learners in both arms and changes the intervention policy. It supports the 65% lower final mean mechanism error and 19/20 wins. The 64% fewer intervention batches is a retrospective first-crossing comparison of group mean curves, not a prospectively validated stopping rule or actual campaign resources saved.

Slide 4 separately illustrates SCM-free exhaustive lookup and uniform random joint-grid coverage. Ten five-valued inputs produce 9,765,625 joint entries; uniform draws with replacement take 29,255,197 draws for 95% expected coverage. Nine two-parent five-valued local mechanism tables contain 225 entries. This is an analytic coverage/representation illustration under explicit access assumptions, not a measured ACE advantage against a new associative algorithm. Exact definitions and arithmetic are in [SCM_free_grid_comparison.json](SCM_free_grid_comparison.json).

Prior versions below remain historical.

---

# ACE presentation

Current deck: **ACE_mechanisms_and_evidence_v4.pptx** (ten editable slides).
Viewing copy: **ACE_mechanisms_and_evidence_v4.pdf** (rendered pages).

Slides1–4 introduce SCMs and intervention selection. Slides5–7 give precise numerical inputs, candidate scores, the measured row and an explicitly simplified update. Slide8 uses an actual archived action. Slides9–10 distinguish empirical evidence from the proposed foundation-model synthesis.

Speaker notes, the exact-arithmetic JSON and the deck builder accompany the presentation. Generate the miniature with `scripts/research/explain_ace_mechanics.py --output <new-json-path>` from the repository. It uses only Python's standard library and refuses to overwrite an existing output. The builder uses the pinned Codex artifact runtime and repository paths recorded in its source. It preserves native editable tables. The earlier theory/evidence v2 remains historical.

No claim of generic ACE superiority, causal graph discovery or successful foundation-model integration follows from this deck. Current original Stage B acceptance is separate from its still-unqualified supplemental replay.

## Marketing revision (October 9): version 5

Open `ACE_mechanisms_and_evidence_v5.pptx` or its PDF viewing copy. Ten slides retain theory first, the exact worked example, explicit SCM graphs, and measured random comparisons. Headlines:65%lower final error,19/20wins,64%fewer batches to a retrospective matched-error crossing. The final metric is a post-hoc curve illustration, not a prospective stopping experiment. Source calculations and fullprecision inputs are in `recorded_curves_2026-10-09.json`; context is in speaker notes and `docs/presentations/ACE_marketing_visuals_2026-10-09.md`. The full-grid comparison is separately labeled representation-size arithmetic. Previous versions remain available.
