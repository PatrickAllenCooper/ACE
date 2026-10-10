# ACE: 10-minute presentation script, version 7

Deck: **ACE_mechanisms_and_evidence_v7.pptx**, 12 slides. Timing includes pauses for the diagrams and tables. Bracketed directions are not spoken. The compressor is a conceptual illustration. The foundation-model branch is proposed integration; the recorded acquisition results use neural SCM ensembles.

## Slide 1 — ACE | 0:00–0:35

[Pause on the compressor.]

Imagine trying to understand a complicated machine. You can change its settings, run it, and measure what happens. Every experiment costs time, energy, materials, or access to equipment.

ACE asks: what should we change next to learn the most?

It uses a model of how the system's parts affect one another to choose an informative intervention. The compressor represents that ambition. We will explain the mechanism with a small example, then look at recorded synthetic experiments.

## Slide 2 — The causal model inside ACE | 0:35–1:20

[Trace X, then M, then Y.]

A structural causal model, or SCM, gives each variable its own mechanism. Here X affects M, and M affects Y. Each arrow points from cause to effect.

Think of a control setting, an internal state, and an output. The graph tells us where effects can travel. We still need to learn how strongly they travel.

For this teaching example, the true rules are M equals two and a half times X, and Y equals three times M. These are dimensionless equations, not compressor physics. Knowing the wiring does not mean knowing the response.

## Slide 3 — Choose where to intervene | 1:20–2:00

[Point to the missing incoming arrow in the lower graph.]

An intervention deliberately fixes a variable. If we clamp M to one, M stops following its usual mechanism during that experiment. Its incoming arrow disappears, while Y still responds to M.

That tells us which training labels remain useful. The imposed M cannot teach us how X naturally produces M. The resulting Y can still teach us how M produces Y.

If we change X instead, both downstream mechanisms can respond naturally. Several mechanisms can learn from one paid response.

## Slide 4 — Local tables versus a joint table | 2:00–2:50

[Point to the small parent graph and local table, then the joint table.]

Here is the difference between the two representations.

On the left, the table for M needs only its direct parents, A and B. If each parent has five settings, that table has twenty-five input combinations. Other mechanisms have their own local tables. We connect their predictions through the graph.

On the right, one joint table maps a complete set of ten input settings to the final output. It needs a row for every combination of those ten inputs.

Nine local tables contain two hundred twenty-five entries. The joint table contains nearly ten million. This discrete illustration explains factorization. ACE's implemented learners fit functions rather than literal lookup tables.

## Slide 5 — Exhaustive grids multiply the work | 2:50–3:30

[Pause on the two representation counts.]

Ten inputs with five settings each produce 9.77 million joint configurations.

If we sample that grid uniformly at random and allow repeats, reaching ninety-five percent expected coverage takes about 29.26 million draws. Random draws revisit configurations.

The two hundred twenty-five local entries assume a known graph, observed intermediate variables, and access to the needed parent settings. This is a representation and coverage calculation. It is separate from the measured intervention comparison later. Other associative learners can exploit regularity and need not enumerate the full grid.

## Slide 6 — Overall mechanism | 3:30–4:35

[Follow the coral branch, then the teal loop.]

This is the overall architecture, including our proposed foundation-model extension.

The inputs are a known graph, eligible observations, permitted controls, and a budget. A numerical foundation model proposes candidate mechanism predictors. We compare those candidates with numerical and retained mechanisms before incorporating them into the SCM.

The teal loop shows how experiments work. The SCM predicts responses and represents uncertainty. The action selector compares legal interventions. In this example it can clamp either X or M to minus one, zero, or plus one. Its selected action goes to the environment.

The environment returns measured X, M, and Y. Those values update the mechanisms that remained natural, and the loop repeats while budget remains.

The foundation model supplies alternatives; it does not get unrestricted control of the apparatus. The acquisition results we will show come from the neural SCM loop, without this proposed pretrained branch.

## Slide 7 — Exact inputs | 4:35–5:20

[Point to the two rows of slopes.]

Now the decision becomes numerical. Three candidate models for M have slopes one, two, and three. Three models for Y have slopes two, three, and four.

The average predicts M as twice X, then Y as three times M. Disagreement between models represents uncertainty an experiment might reduce.

We supply the six legal actions, three reference settings, and a noise proxy of point zero five. One paid response remains. The selector compares what each action could teach before buying its outcome.

## Slide 8 — The selected action | 5:20–6:05

[Read the first row of the score table.]

Setting X to minus one earns about point four one for M and point four four for Y, totaling about point eight five.

Setting M directly can teach Y, but M itself is clamped, so its total is about point four one. Zero scores nothing in this particular example because the predictions agree there.

Positive and negative one tie. The declared tie rule chooses the first action: set X to minus one. The current forecast is M equals minus two and Y equals minus six. That is the precise output of planning.

## Slide 9 — Observation and update | 6:05–6:50

[Follow the observed row into the two training pairs.]

We perform the intervention and observe X equals minus one, M equals minus two point five, and Y equals minus seven point five.

For M, the training input is measured X and the label is measured M. For Y, the input is measured M and the label is measured Y.

After one illustrative gradient step, the mean forecast becomes M equals minus two point two five and Y equals minus six point seven five.

This explains the update mechanically. It is not held-out evidence, and the production learner uses a fuller neural training procedure.

## Slide 10 — Thirty linked mechanisms | 6:50–7:30

[Follow the coral arrows from X7.]

This graph comes from an archived experiment: thirty variables across five layers. The coral paths show downstream effects of an intervention at X7.

The first selected action set X7 to approximately three point eight five seven and purchased fifty responses.

One setting can inform several connected mechanisms. The following comparison summarizes twenty systems from this experimental setting, rather than one chosen trajectory.

## Slide 11 — Lower final error | 7:30–8:35

[Point to both curves, then the final values.]

Both methods received the same budget: two thousand responses per system. Random selection used the same SCM learning machinery, so this comparison changes the intervention policy.

ACE's final mean mechanism prediction error was about point zero one five four, compared with point zero four three nine for random selection. That is approximately sixty-five percent lower final prediction error.

ACE finished ahead on nineteen of the twenty systems.

The full curve appears on the left and a closer view on the right. The metric predicts individual mechanisms using observed parents. Within this setting, choosing interventions through the causal model produced better final predictions than choosing them randomly.

## Slide 12 — Closing comparison | 8:35–10:00

[Point to the two bars, then the separate joint-table figure.]

On the left, we ask when the recorded mean learning curves first reached the same error level. The target is random selection's final mean error.

ACE first crossed it after ten batches. Random selection first crossed after twenty-eight. That is about sixty-four percent fewer intervention batches: five hundred intervention responses versus fourteen hundred.

Including observations, those prefixes contain six hundred twenty and seventeen hundred sixty total responses. This is retrospective. Both actual campaigns completed their full budgets.

The figure on the right brings back the SCM-free joint table: 9.77 million entries, or 29.26 million random draws for ninety-five percent expected coverage in the discrete illustration. Its units differ from the measured bars. It is not a third experimental result at the same prediction-error target.

Together, these views explain the opportunity: use structure to organize what we learn, and use uncertainty to decide which experiment to buy next.

ACE: learn what to change next.

## Presenter references

- Slides 2–3 and 7–9: `mechanistic_demonstration_2026-10-09.json` and the exact-arithmetic explanation in `scripts/research/explain_ace_mechanics.py`.
- Slides 4–5 and the right side of 12: `SCM_free_grid_comparison.json`. Local accessibility and deterministic five-valued mechanisms are assumptions, not measured compressor properties.
- Slide 6: `docs/development/guidance/ace_foundation_cycle_closeout_2026-10-09.md` and `ace_foundation_retention_design_2026-10-09.md`. Conditional pretrained candidate benefits have been explored; foundation-model acquisition efficiency remains unestablished.
- Slides 10–12, measured comparisons: `recorded_curves_2026-10-09.json`. Full inferential and retrospective limitations remain in the deck's technical notes.
