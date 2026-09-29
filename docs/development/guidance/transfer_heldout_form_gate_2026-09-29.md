# C next gate: transfer under a held-out mechanism form

Status: implementation plan; **no outcome generated**. Freeze the protocol and runner in a separate commit before generating systems. This follows the 20-system top-eight allocation study. Its candidate soft mixture failed the unchanged-node and changed-node comparisons in the 16-source coefficient stratum, while adaptive scratch captured most of the changed-node gain. The existing source and target generators use the same six-feature bank, so that study does not test a family outside the fitted mechanism class.

## Question

Does the top-eight acquisition gain survive a change whose conditional mean includes a form absent from the source and fitted target bank? Can a finite-source prior protect unchanged nodes without preventing changed-node repair?

## Specified construction for the next implementation

- Use 20 new systems, seeds 1200–1219, with 30 nodes and one changed node. Draw old/source coefficients and finite source observations exactly as in the top-eight study, using 16 and 64 counted source responses per node. No target outcome enters the source posterior.
- Keep the six fitted features `x1`, `x2`, `x1*x2`, `x1^2`, `sin(1.4*x1)`, and `tanh(1.3*x2)`. For the changed target node, add `0.85*tanh(1.7*x1 + 0.8*x2)` to its old conditional mean. This form is specified before the new systems are generated. Keep the existing 0.15 response noise, action/input support `[-2,2]^2`, and independent 1,024-point test design.
- Run the existing four-response-per-node assay, then allocate 80 more responses either to the top eight evidence-ranked nodes (ten each) or uniformly as in the prior study. Every policy receives exactly 200 target responses. The assay and ranking may use only source posterior and acquired target responses.
- Fit `scratch`, `source_warm`, and the existing fixed-prior `soft_mixture` within the original six-feature bank. Score predictions against the **true conditional mean including the held-out form**. Report irreducible approximation error separately from the noise-free MSE. Do not use a coefficient-vector score that silently omits the new form.
- Include the matched-family changed-node case as a positive control using the same new seed set and source posteriors. The held-out-form result is primary for this gate; positive-control outcomes cannot rescue a failure.

## Decision and custody

For each source size, report paired system-level changed-node and unchanged-node errors, top-eight changed-node nomination rate, action counts, and intervals. The held-out-form candidate is useful only if adaptive `soft_mixture` changed-node MSE is at most 0.8 times uniform `soft_mixture`, at most 1.05 times adaptive `scratch`, and its unchanged-node MSE is at most 1.05 times uniform `soft_mixture` and adaptive `source_warm`. A failure remains a result; do not retune top-eight size, source prior, or function amplitude on these systems.

Freeze the runner and protocol commit before generating any seed. Hash source data/posteriors, acquired target actions and responses, conditional-mean evaluation design, and metrics. Record exact source, target arm, and unique acquired-prefix counts. Replay the full run deterministically. This is a local numerical experiment expected to take seconds or minutes, so no CURC job is justified unless measured runtime changes. No closed-source model API is permitted.

The final runner must independently verify finite conditional means and nonzero out-of-bank projection residual before any system result is interpreted.
