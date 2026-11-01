# ACE literature and positioning readiness review — November 1, 2026

**Final recommendation: literature and positioning are ready for submission for the manuscript's current, bounded claims. The original required positioning finding is closed; zero required findings remain in this review.** This is an independent literature assessment, not a prediction of TMLR acceptance or clearance of the parent's separate abstract, availability, practical-significance, rendering and release work.

The initial review read the active manuscript after `\begin{document}`, `delivery_references.bib`, the included theory text, and the October 24 integrated scientific review including its closing I1 delta. It used a bounded primary-source literature check covering known-graph mechanism surrogates, replay/scratch retraining, interventional prediction and propagation. The final recheck examined only the added Related Work wording and the two bibliography entries. It did not restart foundation-model exploration or a broader literature survey.

## Original finding and disposition

**L1 — Required positioning correction: direct known-graph mechanism-modeling precedents were missing.** The initial Related Work discussion already acknowledged GDumb, experience replay and online-to-batch learning, and appropriately disclaimed a new temporal fitting principle. Its causal citations were broader than the concrete practice of learning nodewise mechanisms on a supplied graph and composing them for downstream queries. I recommended two direct precedents: Sussex, Makarova and Krause (ICLR 2023), and Chao, Blöbaum, Patel and Kasiviswanathan (TMLR 2024). A further sentence connecting the elementary propagation discussion to MCBO was optional. The finding required citation and positioning changes, not a new fitted comparison.

**Disposition: closed.** At [paper.tex:3239](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3239), the revised paragraph accurately acknowledges MCBO's known-graph mechanism models and uncertainty-based intervention selection. It then accurately describes diffusion-based causal models learned from observational data and a supplied graph for interventional and counterfactual queries. The following sentence clearly locates ACE's assessment in fixed-history, post-collection fitting for deterministic structural-map prediction. These changes supply the missing direct context without implying that ACE evaluates either method or establishes superiority over it.

The additional Lipschitz sentence at [paper.tex:3245](/Users/pat/code/ACE/paper/aistats_ace_2027/paper.tex:3245) is also sound. MCBO's Appendix A.2, Lemma 3, propagates discrepancies between true and optimistic SCM dynamics under Lipschitz and calibration assumptions; its Section 4 limits the stated convergence analysis to the soft-intervention setting. Calling these arguments related, while explicitly declining to inherit their guarantees, is appropriately scoped. Locating this attribution in Related Work is sufficient; no theory-companion edit is required. [Primary full text](https://arxiv.org/html/2211.10257).

## Bibliography and primary evidence

- **`sussex2023model`: verified.** Scott Sussex, Anastasiia Makarova, and Andreas Krause, *Model-based Causal Bayesian Optimization*, International Conference on Learning Representations, 2023. The [author record](https://arxiv.org/abs/2211.10257v2) gives the title and author spelling and explicitly records acceptance at ICLR 2023. The manuscript's venue/year and arXiv URL are appropriate; the original 2022 preprint submission date does not change the conference year.
- **`chao2024mechanisms`: verified.** Patrick Chao, Patrick Blöbaum, Sapan Patel, and Shiva Prasad Kasiviswanathan, *Modeling Causal Mechanisms with Diffusion Models for Interventional and Counterfactual Queries*, Transactions on Machine Learning Research, 2024. The [published OpenReview PDF](https://openreview.net/pdf?id=EDHQDsqiSe), first-page text returned by the web search index, identifies the publication date as **11/2024** and gives this revised title and four-author list. A focused retrieval on November 1 reconfirmed that indexed primary-PDF evidence. Direct PDF/forum opening was challenged, so this review does **not** claim a fresh visual inspection of the published PDF. The [October 9, 2024 author manuscript, v3](https://arxiv.org/abs/2302.00860v3), independently supplies the same title and authors, and explicitly notes the title change from the earlier preprint. The TMLR year is supported by the indexed published PDF, not inferred from arXiv's original 2023 submission date. The bibliography correctly uses 2024 and the OpenReview forum URL.

The new entries in [delivery_references.bib:46](/Users/pat/code/ACE/paper/aistats_ace_2027/delivery_references.bib:46) agree with these records, including the four Chao authors and the LaTeX encoding of Blöbaum. No metadata correction is required.

## Remaining required versus optional work

**Required within this review: none.** The two additions close L1. The bounded currency check did not identify another omission that materially changes the current positioning. This is not an exhaustive-literature certification.

**Optional future study:** comparisons against additional GP, flow or diffusion mechanism models could address broader comparative-performance questions. They are not prerequisites for the current claims about the specified delivery recipes and controls. Additional replay citations and a broader foundation-model survey are unnecessary for that scope. Baseline breadth alone does not justify an adverse publication verdict here, and citation completeness does not guarantee acceptance.

The October 24 integrated scientific review's final delta already closes I1; this review does not reopen it or replace the parent's other readiness assessments.

## Reviewed source pins and activity boundary

Read at **2026-11-01T17:51:32Z**. Line references above refer to these bytes:

- `paper/aistats_ace_2027/paper.tex`: SHA256 `86c1e4c3f003e03a9121e14e2fec8a53f9632d3bc53a1519b1a55695bc9fe481`.
- `paper/aistats_ace_2027/delivery_references.bib`: SHA256 `222d370a099bfde7186eb4923c0a27676b86d8c30188a36969068ca5608ff8c4`.
- `paper/aistats_ace_2027/delivery_theory.tex`: SHA256 `0125d36d1ca33e18415ac46b21988cbb8de1a98b4ff2fd4d1b97d9c07aa0679c`, unchanged from the October 24 integrated review's theory pin.

Only this requested report was written. No manuscript, bibliography or theory edits; no experiments, tests, compilation, uploads, remote jobs or GPU allocations were performed. No claim is made here about rendered bibliography output or release-bundle synchronization.
