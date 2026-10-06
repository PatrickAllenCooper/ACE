# Paper versions

## Verified delivery result — 6 October 2026

The twelve-history delivery confirmation is complete. Independent recomputation
matches the sealed receipt: geometric mean paired snapped exact-level error ratio
0.182719, 95% log-t interval [0.101970, 0.327410], two-sided exhaustive sign-flip
p=0.00146484. All registered thresholds pass. Endpoint is misclassification after
mapping predictions to target levels, not MSE. Numerator is each history's median
error across optimization initializations0/1/2; denominator is its online SCM chain
error. The independent sample count is12 histories, not36 fits.

Eleven histories improve; seed124753321 worsens from0.196339 online error to
0.395658 delivery median error (ratio2.015177), retained without exclusion.
Delivery refits all eligible paid rows for30000epochs; online uses a rolling50-step
buffer and100epochs per update. Additional data retention and fitting compute are
part of the treatment. This supports a post-acquisition delivery recipe on one
fixed deterministic emulator. It establishes neither acquisition superiority,
foundation-model benefit, a pure architecture effect, nor generalization across
independent worlds. Flat-head metrics are descriptive; no separately registered
flat or individual-mechanism aggregate acceptance tests were run.

Raw scored pairs and independent computation are in
`results/delivery_final_history_20261006/{scores,independent_statistics}.json`.
All12 custody/seal gates passed;60962charged attempts include3326 interrupted
attempts. No further experiments are implied by this result. Older manuscript
claims require the prior metric-audit evidence; this result does not rehabilitate
them. Preserve historical drafts and registries.


Three venue-specific directories, kept in parallel on purpose. **Do not delete the
older ones** — they are the archival record of the long-form writing, and in one
case they hold the only full-length copy of prose that was compressed to fit a
page limit.

| Directory | Venue | Status |
|---|---|---|
| `aistats_ace_2027/` | AISTATS 2027 (deadline Oct 6, 2026) | **ACTIVE submission target** |
| `iclr_ace_2027/` | ICLR 2027 | Superseded — kept as the full-length reference draft |
| `neurips_ace_2026/` | NeurIPS 2026 | Submitted, rejected. Historical record; do not edit |

## Why the older directories still matter

AISTATS is **two-column with an 8-page main-body limit**; ICLR was single-column
with roughly 10. Fitting the AISTATS limit meant cutting the main body from 15
pages to 8 (done, Sept 10 2026). Almost all of that was **relocation, not
deletion** — the theory section, the 5-node and 30-node schematics, the 30-node
results figure, the scaling figure, the component-ablation table, and the
"scaling principles" paragraph were moved into the AISTATS supplement, which has
no length limit, and are fully present there. Broader Impacts and the
Reproducibility Statement were dropped outright: AISTATS does not mandate them
and its checklist covers reproducibility.

**The one exception is Related Work.** It was genuinely rewritten shorter for
AISTATS (954 → 494 words). Every citation key was preserved and verified, and the
substantive positioning survives, but the longer discursive treatment — in
particular the extended passages on Bayesian active causal discovery and on
in-context sequential decision-making (DPT / ICPE / Krishnamurthy et al.) — exists
in full **only** in `iclr_ace_2027/paper.tex`. If a future venue allows more space,
recover it from there rather than rewriting it.

## Content that is version-specific, not just reformatted

- `iclr_ace_2027/` keeps the theory section (Assumption 4.1, Proposition 4.2, proof
  sketch, and the surrounding motivation) **in the main body**. In AISTATS it is a
  single summary paragraph in the body plus the full formalism in the supplement.
- `neurips_ace_2026/` predates several corrections and should not be used as a
  source of truth for any number. Specifically it contains the retracted reward
  constants (`alpha=0.1`, `gamma=0.05`) that were later found not to match the code
  path actually exercised, and it reports the 30-node result without the
  seed-expansion caveat. It is retained for provenance only.

## Corrections applied to `iclr_ace_2027/` after it was superseded

The ICLR copy was not frozen at the moment AISTATS forked from it — a few
correctness fixes were applied to both so the reference draft does not preserve
known errors:

- de-anonymization leak (a real GitHub URL in the appendix, inherited from the
  de-anonymized NeurIPS tree) replaced with an anonymous mirror;
- Table 3's "No DPO" row relabeled — it removed the LM proposer as well, and read
  as a DPO ablation contradicting the paper's own Contribution 1;
- the 5-node budget-fairness result and the Bayesian-OED-at-30-nodes row added;
- the 30-node seed-expansion non-replication reported rather than the three-seed
  number alone;
- (Sept 10) the canonical ACE-vs-ACE-w/o-DPO pair put on one statistic — the
  printed 1.73 was an episode-level minimum while Table 2's 1.95 is step-level;
  on the consistent step-level basis w/o-DPO is 1.67±0.19 — with the two
  definitions now stated in Table 2's caption;
- (Sept 10) the scaling-sweep sentence "tied at N=15,30" corrected: paired by
  seed, ACE-w/o-DPO is significantly better at N=15 and N=30;
- (Sept 10) "statistically tied" replaced by the actual tests (Welch p=0.59
  canonical; paired t / Wilcoxon / TOST-equivalence for anon30);
- (Sept 9) bibliography verified against primary sources; 5 entries corrected,
  10 fabricated or duplicate unused entries removed. Both copies share the file.

## Building

`aistats_ace_2027/` compiles with `tectonic -X compile paper.tex` (verified). Note
it uses `dsfont` rather than `bbm` for the indicator symbol — the `bbm10` Type1
font is not resolvable by tectonic and aborts PDF output.

The style pack is `AISTATS2026PaperPack`; the 2027 pack was not yet released when
this was set up. Re-check and swap `aistats2026.sty` once AISTATS 2027 posts its
own.
Until then a clearly commented, temporary preamble override
(`\renewcommand{\@conferenceyear}{2027}`) makes the running header read
"AISTATS 2027"; the `.sty` itself is unmodified. Delete the override at the swap.

`iclr_ace_2027/` does **not** build under tectonic: it still loads `bbm`, whose
`bbm10` Type1 font tectonic cannot resolve. That is a toolchain difference, not
a defect — it compiles under pdflatex/MiKTeX, which is how it was always built.
Leave it; it is the archival copy. (If you ever need a tectonic build of it,
the one-line fix is the same `bbm`→`dsfont` swap the AISTATS copy uses.)
