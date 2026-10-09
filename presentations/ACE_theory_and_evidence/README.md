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
