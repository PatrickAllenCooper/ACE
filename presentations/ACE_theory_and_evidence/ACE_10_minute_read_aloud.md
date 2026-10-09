# ACE — 10-minute presentation script

**Deck:** ACE_mechanisms_and_evidence_v6.pptx  
**Delivery:** approximately 120 words per minute, with short pauses to follow arrows and read charts. Timing windows total 10 minutes. Bracketed directions are not spoken. The compressor is a conceptual illustration; the reported experiments used synthetic systems.

## Slide 1 — ACE | 0:00–0:50

[Let the compressor image settle.]

Imagine you are trying to understand a complicated machine. You can change its settings, run it, and measure what happens. Every experiment costs something: time, energy, materials, or access to the equipment.

ACE asks a simple question: **what should we change next to learn the most?**

The idea is to give the experiment planner a model of how the system’s parts affect one another. Then use uncertainty in those mechanisms to choose the next intervention.

The compressor is a picture of that ambition. Today I’ll show the mechanism with a small, exact example, then show what happened in our recorded experiments.

Our shorthand is: **learn what to change next.**

## Slide 2 — The causal model inside ACE | 0:50–1:50

[Trace the large arrows from X to M to Y.]

These arrows are the starting point. X affects M, and M affects Y. Each arrow points from cause to effect.

A structural causal model, or SCM, gives each variable its own mechanism. The equation at the top says that a variable depends on its direct causes and any outside influences.

Think of a simplified machine: a control changes an internal state, and that state changes an output. The graph tells us where effects travel. We still have to learn how strongly they travel.

For a transparent numerical example, the true rules are M equals two and a half times X, and Y equals three times M. These are dimensionless teaching equations, not compressor physics.

The important distinction is that knowing the wiring does not mean knowing the response. ACE learns those responses through experiments.

## Slide 3 — Choose where to intervene | 1:50–2:50

[Point to the missing X-to-M arrow in the lower graph.]

An intervention means deliberately setting a variable. If we clamp M to one, M no longer follows its usual equation for that experiment. That is why its incoming arrow disappears.

But its outgoing arrow remains. Y still responds to M.

This tells ACE which observations can teach which mechanisms. A clamped M is not a natural example of how X produces M. The resulting Y is still an example of how M produces Y.

If we instead change X, both downstream mechanisms can respond naturally. One experiment can therefore provide several useful training labels.

It is still one paid experimental response. The causal structure tells us how to use that response without treating an imposed value as evidence about the mechanism we interrupted.

## Slide 4 — Exhaustive grids multiply the work | 2:50–4:00

[Point first to the large number on the left, then to 225.]

Here is why structure can matter so much.

Suppose there are ten independent inputs, each with five possible settings. A naive approach builds one enormous table of input combinations and final responses, without an SCM.

That table has almost **9.77 million configurations**.

Now sample that same grid uniformly at random, allowing repeats. Reaching 95 percent expected coverage takes about **29.26 million draws**. Random sampling revisits configurations it has already seen.

In a deliberately structured example with nine mechanisms, each having two five-valued parents, the local mechanism tables contain just **225 entries** altogether.

That is more than forty-three thousand times fewer table entries than the full joint table.

These are mathematical coverage and representation counts. They assume a known graph, observable variables, and access to the local settings. They illustrate the opportunity; our measured ACE results come later. A more capable associative learner need not enumerate this grid.

## Slide 5 — Exact inputs to the demonstration | 4:00–5:00

[Point to the two rows of model slopes.]

Now let’s make the decision concrete.

ACE begins with three candidate models for each mechanism. For M, their slopes are one, two, and three. For Y, they are two, three, and four.

Their average prediction is M equals twice X, followed by Y equals three times M. Their disagreement represents uncertainty that an experiment might resolve.

We allow six actions: set X or M to minus one, zero, or plus one. There is one experimental response left in the budget.

The scoring calculation also receives three reference settings and a noise proxy of point zero five. Everything needed to reproduce this small decision is specified.

Before spending the response, ACE compares what each candidate action is expected to teach about the downstream mechanisms.

## Slide 6 — The score selects do(X = −1) | 5:00–6:00

[Pause on the score table.]

Here are all six scores. Setting X to minus one gives about point four one units of score for M and point four four for Y. Together, that is about point eight five.

Setting M directly can teach Y, but M itself is clamped. Its total is about point four one.

Zero scores nothing in this particular linear example because the model predictions agree there. Positive and negative one tie. Our declared tie rule selects the first action: set X to minus one.

The current model predicts M will be minus two and Y will be minus six.

The output of this planning step is therefore precise: **perform this intervention next**. The model has scored the possibilities without first purchasing their real outcomes.

## Slide 7 — The response updates eligible mechanisms | 6:00–7:00

[Read the observed row, then follow its two uses.]

We run the chosen experiment and observe X equals minus one, M equals minus two point five, and Y equals minus seven point five.

That single response teaches two natural mechanisms.

For M, the training input is the measured X, and the label is the measured M. For Y, the input is the measured M, and the label is the measured Y.

After one illustrative gradient update, the average model predicts M at minus two point two five and Y at minus six point seven five.

This is a transparent demonstration of the learning loop, not a held-out performance result. The production models use neural networks and a fuller training procedure.

The loop is now complete: propose, intervene, observe, update. With more budget, ACE repeats it.

## Slide 8 — A real scenario: 30 linked mechanisms | 7:00–7:50

[Follow the coral arrows from X7 toward the right.]

This is an actual archived graph from the experiment: thirty variables connected through five layers.

The arrows again show cause to effect. The coral paths show how an intervention at X7 can influence downstream variables.

The first selected action set X7 to approximately three point eight five seven and purchased a batch of fifty responses.

The point of showing the graph is to make the scale visible. A decision at one node can inform multiple connected mechanisms.

The next two slides summarize twenty systems from this experimental setting, rather than selecting a single especially favorable trajectory.

## Slide 9 — 65% lower final prediction error | 7:50–8:55

[Point to the curve detail and the final values.]

Here is the measured comparison.

Both methods received the same total budget: two thousand responses per system. The comparator chose non-leaf interventions randomly while using the same SCM learning machinery. This isolates the experiment-selection comparison.

At the end, ACE’s mean mechanism prediction error was about point zero one five four, compared with point zero four three nine for random selection.

That is approximately **65 percent lower final prediction error**.

ACE also finished ahead on **nineteen of the twenty systems**.

The full curve is shown on the left, with a closer view on the right. This measures prediction of individual mechanisms using their observed parents.

So the evidence supports a specific, useful result: in this setting, choosing interventions through the causal model produced better final predictions than choosing the interventions randomly.

## Slide 10 — 64% fewer intervention batches | 8:55–10:00

[Point to 28, then 10. Pause on the final line.]

The final graph asks when those recorded learning curves first reached the same error level.

We use random selection’s final mean error as the target. ACE first crossed it after ten intervention batches. Random selection first crossed it after twenty-eight.

That is approximately **64 percent fewer intervention batches**: five hundred intervention responses instead of fourteen hundred.

Including the accompanying observations, those prefixes contain six hundred twenty total responses for ACE and seventeen hundred sixty for random selection.

This is a retrospective comparison of the recorded curves. Both original runs completed their full budgets; we did not stop them early and claim those resources were actually saved.

Together, the results give us a clear direction: better final predictions, and earlier arrival at a useful error level.

**ACE: learn what to change next.**

---

## Presenter reference — not read aloud

- Slides 2–7: `mechanistic_demonstration_2026-10-09.json`; exact rational arithmetic in `scripts/research/explain_ace_mechanics.py`.
- Slide 4: `SCM_free_grid_comparison.json`. Uniform independent sampling with replacement: expected covered fraction is `1 − (1 − 1/N)^n`, `N = 5^10`; the first integer reaching 0.95 is **29,255,197**. This is not a 95% probability of complete coverage. No empirical comparison against an SCM-free joint-table learner has been run here.
- Slides 8–10: `recorded_curves_2026-10-09.json`, backed by all 40 archived trajectory files and seed5000 graph. The measured random comparator retains SCM learners. Mean MSE reduction is 64.8499%; batch reduction is 64.2857% at the retrospective group-mean target.
- Full inferential scope and unfavorable comparisons remain in the deck’s technical speaker notes. This script does not claim universal superiority, compressor validation, or verified prospective stopping savings.
